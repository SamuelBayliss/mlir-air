# Example 04: GEMM on GPU
#
# Computes: C = A @ B  (same computation as example 03)
# where A is M×K, B is K×N, C is M×N, accumulation in f32.
#
# Concepts introduced:
#   - air.segment with requires=[air.Scratchpad] (maps to LDS on GPU targets)
#   - h.shared(): LDS — shared across all wavefronts in the workgroup
#   - Single-level staging: global → h.shared (LDS) → h.private (VGPRs)
#   - Contrast with example 03: same algorithmic structure, different scope choices
#
# Memory hierarchy used:
#   global          (HBM)    A, B, C — device-accessible
#   h.shared()      (LDS)    A_lds, B_lds — one K-slice, shared across wavefronts
#   h.private()     (VGPRs)  per-wavefront fragments and accumulator
#
# Tiling structure:
#   Segment grid:   (M // seg_m) × (N // seg_n)  — one CU (workgroup) per tile
#   Herd grid:      (seg_m // wf_m) × (seg_n // wf_n)  — one wavefront per tile
#   K loop:         K // tile_k iterations, LDS reloaded each iteration
#
# Comparison with example 03 (NPU):
#   NPU:  seg.private() = MemTile SRAM, then h.private() = AIE L1  (two hops from global)
#   GPU:  h.shared()    = LDS,          then h.private() = VGPRs   (one hop from global)
#   On NPU the segment staging is shared per-column; on GPU LDS is shared per-workgroup.
#
# See spec/ARCHITECTURES.md §3 for the GPU scope → physical resource table.

import argparse

import numpy as np
from itertools import product

import airapi as air
import airapi.ops
from airapi.types import bf16, f32


def build_gemm_gpu(M, N, K,
                   seg_m, seg_n,
                   wf_m, wf_n,
                   tile_k):
    """
    Build a tiled GEMM launch for GPU.

    Segment (one CU / workgroup) covers a (seg_m × seg_n) output tile.
    Herd within the segment is (seg_m // wf_m) × (seg_n // wf_n) wavefronts.
    K is tiled into steps of tile_k; A/B tiles stage through LDS (h.shared).
    """
    assert M % seg_m == 0 and N % seg_n == 0
    assert seg_m % wf_m == 0 and seg_n % wf_n == 0
    assert K % tile_k == 0

    A = air.tensor([M, K], bf16)
    B = air.tensor([K, N], bf16)
    C = air.tensor([M, N], f32)

    with air.launch() as launch:
        @launch.body
        def _():
            # Segment grid: one CU per (seg_m × seg_n) output tile.
            # requires=[air.Scratchpad] — same requirement as the NPU GEMM (example 03).
            # On GPU targets this resolves to LDS; the kernel source is identical.
            with air.segment(
                product(range(0, M, seg_m), range(0, N, seg_n)),
                requires=[air.Scratchpad],
            ) as seg:

                @seg.body
                def _(ss, st):
                    row = ss * seg.tile_sizes[0]   # M offset of this workgroup tile
                    col = st * seg.tile_sizes[1]   # N offset of this workgroup tile

                    # Herd: wavefront grid within the workgroup.
                    # LDS staging buffers are declared here — they are shared across
                    # all wavefronts in the herd (h.shared = LDS on GPU).
                    with air.herd(
                        product(range(0, seg_m, wf_m), range(0, seg_n, wf_n))
                    ) as h:
                        # LDS tiles for the current K slice (one copy per workgroup).
                        # All wavefronts cooperatively load and then read from these.
                        A_lds = air.alloc([seg_m, tile_k], bf16, scope=h.shared())
                        B_lds = air.alloc([tile_k, seg_n], bf16, scope=h.shared())

                        @h.body
                        def _(tx, ty):
                            wm, wn = h.tile_sizes
                            wrow = tx * wm   # row offset within workgroup tile
                            wcol = ty * wn   # col offset within workgroup tile

                            # Per-wavefront accumulator (VGPRs).
                            acc = air.alloc([wm, wn], f32, scope=h.private())
                            acc[:] = 0.0

                            # Per-wavefront VGPR fragments for the current K slice.
                            a_frag = air.alloc([wm, tile_k], bf16, scope=h.private())
                            b_frag = air.alloc([tile_k, wn], bf16, scope=h.private())

                            for k in range(0, K, tile_k):
                                # Global → LDS: cooperative load by all wavefronts.
                                # Each wavefront loads its share of A_lds and B_lds;
                                # the compiler partitions the load across the herd.
                                air.ops.load(A_lds, A[row : row + seg_m, k : k + tile_k])
                                air.ops.load(B_lds, B[k : k + tile_k, col : col + seg_n])

                                # LDS → VGPRs: each wavefront loads its fragment.
                                air.ops.load(a_frag, A_lds[wrow : wrow + wm, :])
                                air.ops.load(b_frag, B_lds[:, wcol : wcol + wn])

                                # VGPR multiply-accumulate.
                                air.ops.dot(a_frag, b_frag, acc=acc)

                            # VGPRs → global: write wavefront result.
                            c_frag = air.alloc([wm, wn], f32, scope=h.private())
                            c_frag[:] = acc[:]
                            air.ops.store(
                                c_frag,
                                C[row + wrow : row + wrow + wm,
                                  col + wcol : col + wcol + wn],
                            )

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM on GPU")
    parser.add_argument("--m",       type=int, default=4096)
    parser.add_argument("--n",       type=int, default=4096)
    parser.add_argument("--k",       type=int, default=4096)
    parser.add_argument("--seg-m",   type=int, default=128,
                        help="Output rows per workgroup (CU)")
    parser.add_argument("--seg-n",   type=int, default=128,
                        help="Output cols per workgroup (CU)")
    parser.add_argument("--wf-m",    type=int, default=32,
                        help="Output rows per wavefront")
    parser.add_argument("--wf-n",    type=int, default=32,
                        help="Output cols per wavefront")
    parser.add_argument("--tile-k",  type=int, default=32,
                        help="K reduction tile (= LDS staging depth)")
    parser.add_argument("--target",  type=str, default="cdna3",
                        choices=["cdna2", "cdna3", "rdna4"])
    parser.add_argument("--print-ir",     action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    a_np = np.random.randn(args.m, args.k).astype(np.float16)
    b_np = np.random.randn(args.k, args.n).astype(np.float16)

    launch = build_gemm_gpu(
        M=args.m, N=args.n, K=args.k,
        seg_m=args.seg_m, seg_n=args.seg_n,
        wf_m=args.wf_m,   wf_n=args.wf_n,
        tile_k=args.tile_k,
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
        print(f"GEMM (GPU) {'PASSED' if passed else 'FAILED'}: "
              f"M={args.m}, N={args.n}, K={args.k}")
