# Example 07: GEMM with MXFP8 inputs (BlockType)
#
# Computes: C = dequant(A) @ dequant(B)
# where A and B are stored in MXFP8 block floating-point format and C accumulates in f32.
#
# Concepts introduced:
#   - air.BlockType: hierarchical element type with shared-exponent fields (§2.4)
#   - air.Field with granularity=32: scale field maps 32 K-elements to one e8m0 value
#   - Staging compressed mxfp8 through MemTile (saves scratchpad vs staging in f32)
#   - Per-field allocs in AIE L1: separate fp8 value and e8m0 scale buffers
#   - air.dequant: explicit dequantisation before air.dot
#
# MXFP8 format (MX microscaling, block size 32 along K):
#   Each group of 32 elements along the K axis shares one e8m0 (8-bit exponent) scale.
#
# Two-stage field lookup for element at index (i, k) — §2.4:
#   Stage 1 — field mapping (dtype descriptor):
#     scale: [i, k] → [i, k // 32]   (granularity=32, block_dim=-1)
#     value: [i, k] → [i, k]         (granularity=1)
#   Stage 2 — physical location (alloc layout):
#     Each field's backing alloc maps its logical field index to a byte offset.
#
# Memory hierarchy:
#   global          (LPDDR)        A, B in mxfp8; C in f32
#   seg.private()   (MemTile SRAM) A_stage, B_stage in mxfp8 — two backing allocs each:
#                                    value alloc (fp8)  + scale alloc (e8m0)
#   h.private()     (AIE L1)       a_val, a_scale, b_val, b_scale (per-field buffers)
#                                  a_f32, b_f32 (dequanted fragments); acc in f32

import argparse

import numpy as np
from itertools import product

import airapi as air
import airapi.ops
from airapi.types import f32, fp8, e8m0


# MXFP8 block floating-point dtype.
# One e8m0 scale per 32-element block along the last (K) axis.
mxfp8 = air.BlockType(
    fields=[
        air.Field("scale", e8m0, granularity=32, block_dim=-1),
        air.Field("value", fp8,  granularity=1),
    ],
    dequant=lambda scale, value: value.to(f32) * (2.0 ** scale.to(f32)),
)

M = air.symbol()
K = air.symbol()
N = air.symbol()

A = air.tensor([M, K], mxfp8)
B = air.tensor([K, N], mxfp8)
C = air.tensor([M, N], f32)


def build_gemm_mxfp8(seg_m, seg_n, tile_m, tile_n, tile_k):
    """
    Build a tiled MXFP8 GEMM launch for NPU.

    Compressed mxfp8 data is staged through MemTile SRAM unchanged, reducing
    MemTile traffic vs staging in f32. Dequant to f32 occurs in AIE L1 immediately
    before the multiply-accumulate.

    tile_k must be a multiple of 32 (the mxfp8 block size along K).
    """
    assert seg_m % tile_m == 0 and seg_n % tile_n == 0
    assert tile_k % 32 == 0, "tile_k must be a multiple of the mxfp8 block size (32)"

    with air.launch() as launch:
        @launch.body
        def _():
            with air.segment(
                product(range(0, M, seg_m), range(0, N, seg_n)),
                requires=[air.Scratchpad],
            ) as seg:

                @seg.body
                def _(ss, st):
                    row = ss * seg.tile_sizes[0]
                    col = st * seg.tile_sizes[1]

                    # MemTile staging in mxfp8.
                    # Each mxfp8 alloc has two backing buffers in MemTile SRAM:
                    #   value alloc: seg_m × tile_k fp8 elements
                    #   scale alloc: seg_m × (tile_k // 32) e8m0 elements
                    # Staging compressed saves ~4× MemTile capacity vs f32 staging.
                    A_stage = air.alloc([seg_m, tile_k], mxfp8, scope=seg.private())
                    B_stage = air.alloc([tile_k, seg_n], mxfp8, scope=seg.private())

                    with air.herd(
                        product(range(0, seg_m, tile_m), range(0, seg_n, tile_n))
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            tm, tn = h.tile_sizes
                            hrow = tx * tm
                            hcol = ty * tn

                            # f32 accumulator kept in AIE L1 across the K loop.
                            acc = air.alloc([tm, tn], f32, scope=h.private())
                            acc[:] = 0.0

                            # Per-field allocs for one K-slice of A tile.
                            # scale field: one e8m0 per 32 K-elements → tile_k // 32 cols.
                            a_val   = air.alloc([tm, tile_k],       fp8,  scope=h.private())
                            a_scale = air.alloc([tm, tile_k // 32], e8m0, scope=h.private())

                            # Per-field allocs for one K-slice of B tile.
                            b_val   = air.alloc([tile_k, tn],       fp8,  scope=h.private())
                            b_scale = air.alloc([tile_k // 32, tn], e8m0, scope=h.private())

                            # f32 working buffers after dequant.
                            a_f32 = air.alloc([tm, tile_k], f32, scope=h.private())
                            b_f32 = air.alloc([tile_k, tn], f32, scope=h.private())

                            for k in range(0, K, tile_k):
                                # L3 → MemTile: per-field DMA, compressed format preserved.
                                # Compiler emits separate DMA ops for value and scale fields.
                                air.ops.load(A_stage, A[row : row + seg_m, k : k + tile_k])
                                air.ops.load(B_stage, B[k : k + tile_k, col : col + seg_n])

                                # MemTile → AIE L1: per-field DMA for this core's sub-tile.
                                # .fields[] selects the backing alloc for each field.
                                air.ops.load(a_val,   A_stage.fields["value"][hrow : hrow + tm, :])
                                air.ops.load(a_scale, A_stage.fields["scale"][hrow : hrow + tm, :])
                                air.ops.load(b_val,   B_stage.fields["value"][:, hcol : hcol + tn])
                                air.ops.load(b_scale, B_stage.fields["scale"][:, hcol : hcol + tn])

                                # Dequant: value.to(f32) * 2^scale, broadcast across the
                                # 32-element block dimension.
                                air.ops.dequant(a_val, a_scale, out=a_f32, dtype=mxfp8)
                                air.ops.dequant(b_val, b_scale, out=b_f32, dtype=mxfp8)

                                # f32 multiply-accumulate in AIE L1.
                                air.ops.dot(a_f32, b_f32, acc=acc)

                            # AIE L1 → global: write the f32 output tile.
                            c_buf = air.alloc([tm, tn], f32, scope=h.private())
                            c_buf[:] = acc[:]
                            air.ops.store(
                                c_buf,
                                C[row + hrow : row + hrow + tm,
                                  col + hcol : col + hcol + tn],
                            )

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MXFP8 GEMM on NPU")
    parser.add_argument("--seg-m",  type=int, default=128)
    parser.add_argument("--seg-n",  type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--tile-k", type=int, default=32,
                        help="Must be a multiple of 32 (mxfp8 block size)")
    parser.add_argument("--target", type=str, default="npu2",
                        choices=["npu1", "npu2"])
    parser.add_argument("--print-ir",     action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    m, k, n = 512, 512, 512
    # Reference inputs in f32; caller is responsible for quantising to mxfp8 before dispatch.
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)

    launch = build_gemm_mxfp8(
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

        expected = a_np @ b_np
        # mxfp8 quantisation error: use loose tolerance.
        passed = np.allclose(c_np, expected, rtol=1e-1, atol=1e-1)
        print(f"GEMM (MXFP8) {'PASSED' if passed else 'FAILED'}: "
              f"M={m}, N={n}, K={k}")
