# Example 06: Autotuned GEMM
#
# Same computation as example 03 (C = A @ B, NPU target), with two additions:
#
#   - air.symbol() for M, N, K: dispatch-time symbols resolved from input shapes;
#     a single compiled binary handles any problem size.
#
#   - air.symbol(choices, hint) for tile parameters: compile-time symbols searched
#     by the autotuner. Names are inferred from Python variable names at trace time.
#     launch.search_space exposes them; launch.benchmark() finds the best configuration.
#
# Parameter states (§3 of SPEC.md):
#   M, N, K        air.symbol() — dispatch-time, resolved from input array shapes
#   seg_m, seg_n   Fixed — passed by caller; set to NPU-column-sized tiles
#   tile_k         air.symbol(choices=[16,32,64], hint=32) — autotuner searches K-tile
#   tile_m, tile_n air.symbol() — unconstrained compile-time; compiler enumerates divisors
#
# Constraints emitted in launch.constraints after tracing:
#   seg_m % tile_m == 0
#   seg_n % tile_n == 0
#   M % seg_m == 0   (dispatch-time assertion — M is a dispatch-time symbol)
#   N % seg_n == 0   (dispatch-time assertion)
#   K % tile_k == 0  (dispatch-time assertion)
#   tile_m * tile_n * sizeof(f32) <= AIE_L1_capacity
#   (tile_m * tile_k + tile_k * tile_n) * sizeof(bf16) <= AIE_L1_capacity

import argparse

import numpy as np
from itertools import product

import airapi as air
import airapi.ops
from airapi.types import bf16, f32
from airapi.layout import AffineLayout


# Dispatch-time symbols: one binary handles any M × K @ K × N problem.
# Names "M", "K", "N" are inferred from these variable names at trace time.
M = air.symbol()
K = air.symbol()
N = air.symbol()

A = air.tensor([M, K], bf16)
B = air.tensor([K, N], bf16)
C = air.tensor([M, N], f32)


def build_gemm_autotuned(seg_m, seg_n):
    """
    Build an autotunable GEMM launch.

    seg_m and seg_n are fixed (one NPU column per output tile).
    tile_k is a compile-time symbol with three choices.
    tile_m and tile_n are unconstrained compile-time symbols; the compiler
    enumerates valid divisors of seg_m and seg_n.
    """
    # Compile-time symbols: names inferred at trace time.
    tile_k = air.symbol(choices=[16, 32, 64], hint=32)
    tile_m = air.symbol()
    tile_n = air.symbol()

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

                    # MemTile staging buffers.
                    # Shape uses tile_k (compile-time symbol) — each compiled
                    # configuration has a specific value bound by the autotuner.
                    A_stage = air.alloc(
                        [seg_m, tile_k], bf16, scope=seg.private(),
                        layout=AffineLayout(strides=[tile_k, 1]),
                    )
                    B_stage = air.alloc(
                        [tile_k, seg_n], bf16, scope=seg.private(),
                        layout=AffineLayout(strides=[seg_n, 1]),
                    )

                    # tile_m and tile_n: compile-time symbols used as range steps.
                    # Compiler enumerates valid divisors of seg_m / seg_n subject
                    # to the L1-capacity constraint on accumulator and load buffers.
                    with air.herd(
                        product(range(0, seg_m, tile_m), range(0, seg_n, tile_n))
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            tm, tn = h.tile_sizes
                            hrow = tx * tm
                            hcol = ty * tn

                            acc   = air.alloc([tm, tn],     f32,  scope=h.private())
                            a_buf = air.alloc([tm, tile_k], bf16, scope=h.private())
                            b_buf = air.alloc([tile_k, tn], bf16, scope=h.private())
                            acc[:] = 0.0

                            for k in range(0, K, tile_k):
                                air.ops.load(A_stage, A[row : row + seg_m, k : k + tile_k])
                                air.ops.load(B_stage, B[k : k + tile_k, col : col + seg_n])
                                air.ops.load(a_buf, A_stage[hrow : hrow + tm, :])
                                air.ops.load(b_buf, B_stage[:, hcol : hcol + tn])
                                air.ops.dot(a_buf, b_buf, acc=acc)

                            c_buf = air.alloc([tm, tn], f32, scope=h.private())
                            c_buf[:] = acc[:]
                            air.ops.store(
                                c_buf,
                                C[row + hrow : row + hrow + tm,
                                  col + hcol : col + hcol + tn],
                            )

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned GEMM on NPU")
    parser.add_argument("--seg-m",  type=int, default=128)
    parser.add_argument("--seg-n",  type=int, default=128)
    parser.add_argument("--target", type=str, default="npu2",
                        choices=["npu1", "npu2"])
    parser.add_argument("--autotune",    action="store_true",
                        help="Search over compile-time symbols and benchmark each config")
    parser.add_argument("--print-space", action="store_true",
                        help="Print search_space and constraints, then exit")
    args = parser.parse_args()

    launch = build_gemm_autotuned(seg_m=args.seg_m, seg_n=args.seg_n)

    if args.print_space:
        print("Search space (compile-time symbols):")
        for name, sym in launch.search_space.items():
            print(f"  {name}: choices={sym.choices}, hint={sym.hint}")
        print("\nConstraints:")
        for c in launch.constraints:
            print(f"  {c}")
        raise SystemExit(0)

    np.random.seed(0)
    m, k, n = 512, 512, 512
    a_np = np.random.randn(m, k).astype(np.float16)
    b_np = np.random.randn(k, n).astype(np.float16)

    if args.autotune:
        best_flops, best_config = launch.benchmark(
            target=args.target,
            inputs=[a_np, b_np],
        )
        print(f"Best config: {best_config}")
        print(f"Best FLOP/s: {best_flops:.3e}")
    else:
        fn = launch.compile(target=args.target)
        c_np = fn(a_np, b_np)

        print("Compile-time bindings:", fn.compile_bindings)
        print("Dispatch-time symbols:", fn.dispatch_symbols)
        print("Last dispatch values: ", fn.last_dispatch)

        expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
        passed = np.allclose(c_np, expected, rtol=1e-2, atol=1e-2)
        print(f"GEMM (autotuned, hint config) {'PASSED' if passed else 'FAILED'}: "
              f"M={m}, N={n}, K={k}")
