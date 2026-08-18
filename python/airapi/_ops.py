"""Stub implementations of AIR compute and memory operations."""
from airapi._core import Token, Buffer, _Stub


# ---------------------------------------------------------------------------
# Reduction sentinels (passed as op= to reduce)
# ---------------------------------------------------------------------------

class _Sentinel:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"airapi.ops.{self.name}"
    def __call__(self, *args, **kwargs):
        return args[0] if args else None

sum = _Sentinel("sum")
max = _Sentinel("max")
min = _Sentinel("min")


# ---------------------------------------------------------------------------
# Memory ops
# ---------------------------------------------------------------------------

def load(dst, src_slice, pad_before=None, pad_after=None, dependency=None):
    return Token()


def store(src, dst_slice, pad_before=None, pad_after=None, dependency=None):
    return Token()


def copy(src_slice, dst_slice, pad_before=None, pad_after=None, dependency=None):
    return Token()


# ---------------------------------------------------------------------------
# Compute ops
# ---------------------------------------------------------------------------

def dot(a, b, acc=None, alpha=1.0, transpose_b=False, dependency=None):
    return Token()


def reduce(a, axis, op, dependency=None):
    # Return a stub Buffer so callers can do further arithmetic on the result
    shape = getattr(a, "shape", (1,))
    dtype = getattr(a, "dtype", None)
    return Buffer(shape, dtype)


def exp(a):
    return a if isinstance(a, _Stub) else Buffer((1,), None)


def relu(a):
    return a if isinstance(a, _Stub) else Buffer((1,), None)


def stack(tensors, axis=0):
    return Buffer((1,), None)


def dequant(*args, out=None, dtype=None):
    return Token()


def atomic_add(dst, src, scope, dependency=None):
    return Token()


def wait(*tokens):
    return Token()


def wait_any(*tokens):
    # Returns the first token to complete; identity (is) comparison tells caller which fired.
    return tokens[0] if tokens else Token()


def cancel(token):
    pass
