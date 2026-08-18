# Example 02: Element-wise Add
#
# Computes: C = A + B  (element-wise, 2-D)
# where A, B, C are M×N matrices.
#
# Concepts introduced:
#   - 2D herd: product of two ranges
#   - air.symbol() as compile-time tile size: compiler searches for best (tm, tn)
#   - Two-dimensional h.tile_sizes and h.shape
#
# tile_m and tile_n are air.symbol() — compile-time unknowns whose names are inferred
# from their Python variable names at trace time. launch.search_space exposes them.
#
# Compare with example 01 (AXPY): the only structural change is adding a second
# range dimension and promoting scalars to 2-D slices.

import argparse

import numpy as np
from itertools import product
from ml_dtypes import bfloat16

import airapi as air
import airapi.ops
from airapi.types import bf16


def build_eltwise_add(M, N, dtype=bf16):
    A = air.tensor([M, N], dtype)
    B = air.tensor([M, N], dtype)
    C = air.tensor([M, N], dtype)

    # Compile-time symbols: the autotuner searches for the best tile sizes.
    # Names are inferred from the Python variable names at trace time.
    tile_m = air.symbol()
    tile_n = air.symbol()

    with air.launch() as launch:
        @launch.body
        def _():
            # Herd shape is (M // tile_m, N // tile_n) once tile_m/tile_n are resolved.
            with air.herd(product(range(0, M, tile_m), range(0, N, tile_n))) as h:

                @h.body
                def _(tx, ty):
                    tm, tn = h.tile_sizes
                    row = tx * tm
                    col = ty * tn

                    a_buf = air.alloc([tm, tn], dtype, scope=h.private())
                    b_buf = air.alloc([tm, tn], dtype, scope=h.private())
                    c_buf = air.alloc([tm, tn], dtype, scope=h.private())

                    air.ops.load(a_buf, A[row : row + tm, col : col + tn])
                    air.ops.load(b_buf, B[row : row + tm, col : col + tn])

                    c_buf[:] = a_buf[:] + b_buf[:]

                    air.ops.store(c_buf, C[row : row + tm, col : col + tn])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Element-wise add example")
    parser.add_argument("--m",      type=int, default=1024)
    parser.add_argument("--n",      type=int, default=1024)
    parser.add_argument("--target", type=str, default="npu2",
                        choices=["npu1", "npu2"])
    parser.add_argument("--print-ir",     action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)
    a_np = np.random.randn(args.m, args.n).astype(bfloat16)
    b_np = np.random.randn(args.m, args.n).astype(bfloat16)

    launch = build_eltwise_add(M=args.m, N=args.n, dtype=bf16)

    if args.print_ir:
        print(launch.mlir())

    if args.compile_only:
        launch.compile(target=args.target)
        print("Compiled successfully.")
        print("Search space:", launch.search_space)
    else:
        fn = launch.compile(target=args.target)
        c_np = fn(a_np, b_np)

        expected = a_np + b_np
        passed = np.allclose(c_np, expected, rtol=1e-2)
        print(f"Eltwise add {'PASSED' if passed else 'FAILED'}: M={args.m}, N={args.n}")
