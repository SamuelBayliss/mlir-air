# Example 03: GEMM on NPU
#
# Computes: C = A @ B
# where A is M×K, B is K×N, C is M×N, accumulation in f32.
#
# Concepts introduced:
#   - air.segment with requires=[air.Scratchpad] (maps to NPU MemTile on npu1/npu2)
#   - seg.private(): MemTile SRAM, shared across all AIE cores in the column
#   - Two-level DMA: global → seg.private (MemTile) → h.private (AIE L1)
#   - air.dot: matrix multiply-accumulate
#   - Explicit AffineLayout on a staging buffer
#
# Memory hierarchy used:
#   global          (LPDDR)        A, B, C — host-accessible
#   seg.private()   (MemTile SRAM) A_stage, B_stage — one tile of A/B loaded per K step
#   h.private()     (AIE L1)       per-tile a_buf, b_buf, acc — one sub-tile per AIE core
#
# Tiling structure:
#   Segment grid:  (M // seg_m) × (N // seg_n)  — one NPU column per segment tile
#   Herd grid:     (seg_m // tile_m) × (seg_n // tile_n)  — one AIE core per herd tile
#   K loop:        K // tile_k iterations inside the herd body
#
# See spec/ARCHITECTURES.md §2–3 for the NPU scope → physical resource table.

import argparse

import numpy as np
from itertools import product

import airapi as air
import airapi.ops
from airapi.types import bf16, f32
from airapi.layout import AffineLayout


def build_gemm_npu(M, N, K,
                   seg_m, seg_n,
                   tile_m, tile_n, tile_k):
    """
    Build a tiled GEMM launch for NPU.

    Segment (NPU column) covers a (seg_m × seg_n) output tile.
    Herd within the segment covers (tile_m × tile_n) sub-tiles per AIE core.
    K is tiled into steps of tile_k; the loop runs inside the herd body.
    """
    assert M % seg_m == 0 and N % seg_n == 0
    assert seg_m % tile_m == 0 and seg_n % tile_n == 0
    assert K % tile_k == 0

    A = air.tensor([M, K], bf16)
    B = air.tensor([K, N], bf16)
    C = air.tensor([M, N], f32)

    with air.launch() as launch:
        @launch.body
        def _():
            # Segment grid: one NPU column per (seg_m × seg_n) output patch.
            # requires=[air.Scratchpad] tells the compiler this segment needs a shared
            # low-latency scratch region. On npu1/npu2 this resolves to the MemTile SRAM;
            # on a GPU target it would resolve to LDS. All tiles are co-scheduled.
            with air.segment(
                product(range(0, M, seg_m), range(0, N, seg_n)),
                requires=[air.Scratchpad],
            ) as seg:

                @seg.body
                def _(ss, st):
                    row = ss * seg.tile_sizes[0]   # M offset of this segment tile
                    col = st * seg.tile_sizes[1]   # N offset of this segment tile

                    # MemTile staging buffers (seg.private = MemTile SRAM).
                    # One (seg_m × tile_k) slice of A and one (tile_k × seg_n) slice
                    # of B are loaded here per K step, then broadcast to all AIE cores.
                    # AffineLayout with explicit row stride aligns each row to 64 bytes.
                    A_stage = air.alloc(
                        [seg_m, tile_k], bf16, scope=seg.private(),
                        layout=AffineLayout(strides=[tile_k, 1]),
                    )
                    B_stage = air.alloc(
                        [tile_k, seg_n], bf16, scope=seg.private(),
                        layout=AffineLayout(strides=[seg_n, 1]),
                    )

                    # Herd grid: one AIE core per (tile_m × tile_n) output sub-tile.
                    with air.herd(
                        product(range(0, seg_m, tile_m), range(0, seg_n, tile_n))
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            tm, tn = h.tile_sizes
                            hrow = tx * tm   # row offset within segment tile
                            hcol = ty * tn   # col offset within segment tile

                            # Per-AIE accumulator (f32, kept in L1 across K loop).
                            acc = air.alloc([tm, tn], f32, scope=h.private())
                            acc[:] = 0.0

                            # Per-AIE load buffers for one K slice.
                            a_buf = air.alloc([tm, tile_k], bf16, scope=h.private())
                            b_buf = air.alloc([tile_k, tn], bf16, scope=h.private())

                            for k in range(0, K, tile_k):
                                # L3 → MemTile: load the K-slice for this segment tile.
                                # On NPU, the MemTile DMA engines handle this transfer;
                                # all AIE cores in the column see the updated A_stage/B_stage
                                # after the load completes.
                                air.ops.load(A_stage, A[row : row + seg_m, k : k + tile_k])
                                air.ops.load(B_stage, B[k : k + tile_k, col : col + seg_n])

                                # MemTile → AIE L1: each core loads its sub-tile.
                                air.ops.load(a_buf, A_stage[hrow : hrow + tm, :])
                                air.ops.load(b_buf, B_stage[:, hcol : hcol + tn])

                                # Multiply-accumulate in L1.
                                air.ops.dot(a_buf, b_buf, acc=acc)

                            # AIE L1 → global: write accumulated result.
                            c_buf = air.alloc([tm, tn], f32, scope=h.private())
                            c_buf[:] = acc[:]
                            air.ops.store(
                                c_buf,
                                C[row + hrow : row + hrow + tm,
                                  col + hcol : col + hcol + tn],
                            )

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM on NPU")
    parser.add_argument("--m",       type=int, default=512)
    parser.add_argument("--n",       type=int, default=512)
    parser.add_argument("--k",       type=int, default=512)
    parser.add_argument("--seg-m",   type=int, default=128,
                        help="Output rows per NPU column segment")
    parser.add_argument("--seg-n",   type=int, default=128,
                        help="Output cols per NPU column segment")
    parser.add_argument("--tile-m",  type=int, default=32,
                        help="Output rows per AIE core")
    parser.add_argument("--tile-n",  type=int, default=32,
                        help="Output cols per AIE core")
    parser.add_argument("--tile-k",  type=int, default=32,
                        help="K reduction tile")
    parser.add_argument("--target",  type=str, default="npu2",
                        choices=["npu1", "npu2"])
    parser.add_argument("--print-ir",     action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    a_np = np.random.randn(args.m, args.k).astype(np.float16)
    b_np = np.random.randn(args.k, args.n).astype(np.float16)

    launch = build_gemm_npu(
        M=args.m, N=args.n, K=args.k,
        seg_m=args.seg_m, seg_n=args.seg_n,
        tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
    )

    if args.print_ir:
        print(launch.mlir())

    if args.compile_only:
        launch.compile(target=args.target)
        print("Compiled successfully.")
    else:
        fn = launch.compile(target=args.target)
        c_np = fn(a_np, b_np)

        expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
        passed = np.allclose(c_np, expected, rtol=1e-2, atol=1e-2)
        print(f"GEMM (NPU) {'PASSED' if passed else 'FAILED'}: "
              f"M={args.m}, N={args.n}, K={args.k}")
