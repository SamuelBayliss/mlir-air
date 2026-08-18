# Example 05: GEMM with XCD-disjoint placement and fast atomics
#
# Computes: C = A @ B  (same computation as examples 03 and 04)
#
# Concepts introduced:
#   - air.CacheDomain: shared L2 + fast inter-CU atomics within an XCD
#   - requires=[air.CacheDomain] on air.segment: gates the segment on targets providing
#     CacheDomain; compiler selects the matching body variant per target
#   - @seg.body(placement=air.Disjoint(air.CacheDomain)): each segment instance
#     on a distinct cache domain (XCD on CDNA3)
#   - air.ops.atomic_add(..., scope=seg): hardware atomic reduction into shared L2
#   - Fallback body (plain @s.body) for targets without CacheDomain (e.g. npu2, cdna2)
#
# Placement strategy:
#   The output (M × N) is partitioned into seg_count segment instances, one per XCD.
#   Within each XCD, wavefronts compute partial C tiles and atomically reduce to a
#   per-XCD accumulator in the shared L2, avoiding wavefront-level stores to global memory
#   until the full XCD reduction is complete.
#
# Memory hierarchy used (CacheDomain body):
#   global          (HBM)      A, B, C — device-accessible
#   h.shared()      (LDS)      A_lds, B_lds — staged per workgroup
#   h.private()     (VGPRs)    per-wavefront fragments and partial accumulator
#   seg.private()   (shared L2 accumulator, via atomic_add, scope=seg)
#
# Comparison with example 04 (GPU, no CacheDomain):
#   04: each wavefront writes its C fragment directly to global memory (one store per tile)
#   05: wavefronts accumulate into a shared L2 buffer via fast atomics, then one segment-level
#       store to global — fewer global writes when many wavefronts cover the same output tile
#
# See spec/ARCHITECTURES.md §7 for the target × CacheDomain resolution table.

import argparse

import numpy as np
from itertools import product

import airapi as air
import airapi.ops
from airapi.types import bf16, f32
def build_gemm_cachedomain(M, N, K,
                            seg_m, seg_n,
                            wf_m, wf_n,
                            tile_k):
    """
    Build a tiled GEMM launch with XCD-disjoint segment placement.

    The output is partitioned into (M // seg_m) × (N // seg_n) segment instances.
    On CacheDomain-capable targets each instance is placed on a distinct XCD.

    Two segment body variants are registered:
      1. CacheDomain body: uses air.ops.atomic_add(scope=seg) for fast intra-XCD reduction.
         Selected on CDNA3/CDNA4 targets (requires=[air.CacheDomain] on the segment).
         Instances placed disjoint across XCDs via placement=air.Disjoint(air.CacheDomain).
      2. Generic fallback body: standard per-wavefront store to global memory.
         Selected on targets without air.CacheDomain (npu2, cdna2, rdna4).
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
            # Segment grid: seg_count instances, one per (seg_m × seg_n) output tile.
            # The CacheDomain body variant requests disjoint XCD placement for each instance.
            # The generic body variant carries no placement constraint.
            with air.segment(
                product(range(0, M, seg_m), range(0, N, seg_n)),
                requires=[air.Scratchpad, air.CacheDomain],
            ) as s:

                # ── CacheDomain body variant ──────────────────────────────────────────
                # Selected on targets that satisfy requires=[air.CacheDomain] (CDNA3, CDNA4).
                # placement=air.Disjoint(air.CacheDomain): each segment instance is
                # placed on a distinct XCD. The compiler verifies seg_count ≤ XCD count.
                @s.body(placement=air.Disjoint(air.CacheDomain))
                def _(ss, st):
                    row = ss * s.tile_sizes[0]
                    col = st * s.tile_sizes[1]

                    # Shared L2 accumulator for this segment (one per XCD tile).
                    # Wavefronts accumulate partial results here via atomic_add.
                    xcd_acc = air.alloc([seg_m, seg_n], f32, scope=s.private())
                    xcd_acc[:] = 0.0

                    with air.herd(
                        product(range(0, seg_m, wf_m), range(0, seg_n, wf_n)),
                        requires=[air.Scratchpad],
                    ) as h:
                        A_lds = air.alloc([seg_m, tile_k], bf16, scope=h.shared())
                        B_lds = air.alloc([tile_k, seg_n], bf16, scope=h.shared())

                        @h.body
                        def _(tx, ty):
                            wm, wn = h.tile_sizes
                            wrow = tx * wm
                            wcol = ty * wn

                            partial = air.alloc([wm, wn], f32, scope=h.private())
                            partial[:] = 0.0
                            a_frag = air.alloc([wm, tile_k], bf16, scope=h.private())
                            b_frag = air.alloc([tile_k, wn], bf16, scope=h.private())

                            for k in range(0, K, tile_k):
                                # Global → LDS: cooperative load across the herd.
                                air.ops.load(A_lds, A[row : row + seg_m, k : k + tile_k])
                                air.ops.load(B_lds, B[k : k + tile_k, col : col + seg_n])

                                # LDS → VGPRs: per-wavefront fragment.
                                air.ops.load(a_frag, A_lds[wrow : wrow + wm, :])
                                air.ops.load(b_frag, B_lds[:, wcol : wcol + wn])

                                air.ops.dot(a_frag, b_frag, acc=partial)

                            # Atomic add into the shared XCD accumulator (fast L2 atomic).
                            # All wavefronts within this XCD reduce here without touching HBM.
                            air.ops.atomic_add(
                                xcd_acc[wrow : wrow + wm, wcol : wcol + wn],
                                partial,
                                scope=s,
                            )

                    # XCD accumulator → global: one store per segment after all wavefronts done.
                    air.ops.store(xcd_acc, C[row : row + seg_m, col : col + seg_n])

                # ── Generic fallback body ─────────────────────────────────────────────
                # Selected when air.CacheDomain is not available (npu2, cdna2, rdna4, …).
                # Each wavefront stores its partial result directly to global memory.
                @s.body
                def _(ss, st):
                    row = ss * s.tile_sizes[0]
                    col = st * s.tile_sizes[1]

                    with air.herd(
                        product(range(0, seg_m, wf_m), range(0, seg_n, wf_n)),
                        requires=[air.Scratchpad],
                    ) as h:
                        A_lds = air.alloc([seg_m, tile_k], bf16, scope=h.shared())
                        B_lds = air.alloc([tile_k, seg_n], bf16, scope=h.shared())

                        @h.body
                        def _(tx, ty):
                            wm, wn = h.tile_sizes
                            wrow = tx * wm
                            wcol = ty * wn

                            acc = air.alloc([wm, wn], f32, scope=h.private())
                            acc[:] = 0.0
                            a_frag = air.alloc([wm, tile_k], bf16, scope=h.private())
                            b_frag = air.alloc([tile_k, wn], bf16, scope=h.private())

                            for k in range(0, K, tile_k):
                                air.ops.load(A_lds, A[row : row + seg_m, k : k + tile_k])
                                air.ops.load(B_lds, B[k : k + tile_k, col : col + seg_n])
                                air.ops.load(a_frag, A_lds[wrow : wrow + wm, :])
                                air.ops.load(b_frag, B_lds[:, wcol : wcol + wn])
                                air.ops.dot(a_frag, b_frag, acc=acc)

                            air.ops.store(
                                acc,
                                C[row + wrow : row + wrow + wm,
                                  col + wcol : col + wcol + wn],
                            )

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEMM with XCD-disjoint placement and fast atomics"
    )
    parser.add_argument("--m",          type=int, default=4096)
    parser.add_argument("--n",          type=int, default=4096)
    parser.add_argument("--k",          type=int, default=4096)
    parser.add_argument("--seg-m",      type=int, default=512,
                        help="Output rows per segment (XCD tile)")
    parser.add_argument("--seg-n",      type=int, default=512,
                        help="Output cols per segment (XCD tile)")
    parser.add_argument("--wf-m",       type=int, default=32,
                        help="Output rows per wavefront")
    parser.add_argument("--wf-n",       type=int, default=32,
                        help="Output cols per wavefront")
    parser.add_argument("--tile-k",     type=int, default=32,
                        help="K reduction tile (= LDS staging depth)")
    # Number of segment instances is (M // seg_m) * (N // seg_n).
    # On CDNA3 this must not exceed the XCD count (typically 8); adjust seg_m / seg_n.
    parser.add_argument("--target",     type=str, default="cdna3",
                        choices=["cdna2", "cdna3", "rdna4", "npu2"])
    parser.add_argument("--print-ir",      action="store_true")
    parser.add_argument("--compile-only",  action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    a_np = np.random.randn(args.m, args.k).astype(np.float16)
    b_np = np.random.randn(args.k, args.n).astype(np.float16)

    launch = build_gemm_cachedomain(
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
        print(f"GEMM (CacheDomain) {'PASSED' if passed else 'FAILED'}: "
              f"M={args.m}, N={args.n}, K={args.k}, "
              f"target={args.target}")
