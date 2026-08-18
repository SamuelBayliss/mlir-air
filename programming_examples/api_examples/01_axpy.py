# Example 01: AXPY
#
# Computes: out = alpha * x + y
# where x, y, out are 1D vectors of length N and alpha is a scalar.
#
# Concepts introduced:
#   - air.launch: handover from host to accelerator
#   - air.herd with a 1D grid
#   - air.alloc: tile-private scratch buffers (scope=h.private())
#   - air.load / air.store: DMA between global memory and tile-private memory
#   - Elementwise compute over tile buffers
#   - Explicit tile size (no AUTO)
#
# Compare with: mlir-air/programming_examples/axpy/axpy.py  (~252 lines)

import argparse

import numpy as np
from ml_dtypes import bfloat16

import airapi as air
import airapi.ops
from airapi.types import bf16


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

def build_axpy(N, tile_n, alpha, dtype=bf16):
    """
    Build the AXPY launch for a vector of length N tiled into blocks of tile_n.
    Returns a launch object that can be compiled or inspected.

    The herd has shape (N // tile_n,): one tile per block.
    Each tile:
      1. Loads x[col:col+tile_n] and y[col:col+tile_n] from global memory into tile-private buffers.
      2. Computes out_buf = alpha * x_buf + y_buf in tile-private memory.
      3. Stores out_buf back to out[col:col+tile_n] in global memory.
    """
    assert N % tile_n == 0, f"N ({N}) must be divisible by tile_n ({tile_n})"

    # Declare the interface tensors. They are unscoped; capture into the launch
    # body implies global visibility (the compiler enforces this).
    x   = air.tensor([N], dtype)
    y   = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch() as launch:
        @launch.body
        def _():

            # 1D herd: one tile per block of tile_n elements.
            # range(0, N, tile_n) → herd shape = (N // tile_n,), tile_size = tile_n
            with air.herd(range(0, N, tile_n)) as h:

                @h.body
                def _(tx,):
                    # tx: tile grid coordinate, 0 .. N//tile_n - 1
                    tn  = h.tile_sizes[0]    # == tile_n
                    col = tx * tn            # starting offset in L3

                    # Allocate tile-private scratch buffers.
                    # scope=h.private() → one buffer per tile; compiler maps
                    # to core-local memory and may hoist if lifetime permits.
                    x_buf   = air.alloc([tn], dtype, scope=h.private())
                    y_buf   = air.alloc([tn], dtype, scope=h.private())
                    out_buf = air.alloc([tn], dtype, scope=h.private())

                    # DMA: L3 → L1
                    # Offsets, sizes, and strides are derived from the slice
                    # bounds and the default RowMajor layout of x and y.
                    air.ops.load(x_buf, x[col : col + tn])
                    air.ops.load(y_buf, y[col : col + tn])

                    # Elementwise compute in L1.
                    # Lowered to vectorised operations (vector width chosen by
                    # the compiler based on dtype and target SIMD width).
                    out_buf[:] = alpha * x_buf[:] + y_buf[:]

                    # DMA: L1 → L3
                    air.ops.store(out_buf, out[col : col + tn])

    return launch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXPY example")
    parser.add_argument("--n",         type=int,   default=65536)
    parser.add_argument("--tile-n",    type=int,   default=1024)
    parser.add_argument("--alpha",     type=float, default=2.0)
    parser.add_argument("--target",    type=str,   default="npu2",
                        choices=["npu1", "npu2"])
    parser.add_argument("--print-ir",  action="store_true",
                        help="Print the generated AIR MLIR before compiling")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    x_np = np.random.randn(args.n).astype(bfloat16)
    y_np = np.random.randn(args.n).astype(bfloat16)

    launch = build_axpy(
        N=args.n,
        tile_n=args.tile_n,
        alpha=args.alpha,
        dtype=bf16,
    )

    if args.print_ir:
        print(launch.mlir())

    if args.compile_only:
        launch.compile(target=args.target)
        print("Compiled successfully.")
    else:
        fn = launch.compile(target=args.target)

        # fn signature: (x, y) -> out
        # Inputs and outputs are ordered by declaration order of interface tensors.
        # x and y are inputs (read-only); out is the output (written).
        out_np = fn(x_np, y_np)

        expected = args.alpha * x_np + y_np
        passed = np.allclose(out_np, expected, rtol=1e-2)
        print(f"AXPY {'PASSED' if passed else 'FAILED'}: "
              f"N={args.n}, tile_n={args.tile_n}, alpha={args.alpha}")
