# airapi.ops — imperative compute and memory operations
#
# Import as:
#   import airapi.ops
#
# All functions return an airapi.Token carrying temporal (completion) and
# spatial (hardware binding) information. Tokens may be passed to
# dependency= on any subsequent op to enforce ordering.

from airapi._ops import (
    # Memory transfers (lower to air.dma_memcpy_nd / air.execute in MLIR)
    load,    # load(dst, src_slice, pad_before=None, pad_after=None, dependency=None) -> Token
    store,   # store(src, dst_slice, pad_before=None, pad_after=None, dependency=None) -> Token
    copy,    # copy(src_slice, dst_slice, pad_before=None, pad_after=None, dependency=None) -> Token

    # Matrix multiply-accumulate
    # acc += alpha * (a @ b)  [transpose_b transposes b before multiply]
    dot,     # dot(a, b, acc=None, alpha=1.0, transpose_b=False, dependency=None) -> Token

    # Reduction along an axis
    # op in {airapi.ops.sum, airapi.ops.max, airapi.ops.min}
    reduce,  # reduce(a, axis, op, dependency=None) -> Token

    # Reduction sentinels (passed as op= argument to reduce)
    sum,
    max,
    min,

    # Elementwise / unary
    exp,     # exp(a) -> buffer  (same shape and dtype as a)
    relu,    # relu(a) -> buffer

    # Tensor stacking (analogous to numpy.stack)
    stack,   # stack(tensors, axis=0) -> buffer

    # Explicit dequantisation from per-field allocs
    # Applies dtype.dequant(val, scale, ...) and writes result into out.
    # Field arguments must match the BlockType field declaration order.
    dequant, # dequant(val, scale, ..., out, dtype) -> Token

    # Segment-scoped hardware atomic
    # scope=h  → intra-herd atomic (always available)
    # scope=seg → intra-XCD L2 atomic (requires airapi.CacheDomain in segment requires)
    atomic_add,  # atomic_add(dst, src, scope, dependency=None) -> Token

    # Token join (also available as airapi.wait)
    wait,      # wait(*tokens) -> Token        AND-join: blocks until all complete
    wait_any,  # wait_any(*tokens) -> Token    OR-join: blocks until first completes; returns that input token
    cancel,    # cancel(token) -> None         abort pending op; no-op if already complete
)

__all__ = [
    "load", "store", "copy",
    "dot",
    "reduce", "sum", "max", "min",
    "exp", "relu",
    "stack",
    "dequant",
    "atomic_add",
    "wait", "wait_any", "cancel",
]
