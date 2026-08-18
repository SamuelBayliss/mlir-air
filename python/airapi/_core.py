"""Stub tracer for the AIR Python DSL.

All context managers eagerly call their body function once with representative
zero-valued tile coordinates. Buffer arithmetic is absorbed silently so that
complex expressions (online softmax, etc.) run without error.
"""
import inspect
import warnings


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class Token:
    """Dummy async token returned by memory and compute ops."""

    def __and__(self, other):
        return Token()   # AND-combine: fires when both inputs complete

    def __or__(self, other):
        return Token()   # OR-combine: fires when either input completes

    def __repr__(self):
        return "airapi.Token()"


# ---------------------------------------------------------------------------
# Symbol — behaves as an int for range(), arithmetic, and comparisons
# ---------------------------------------------------------------------------

class Symbol:
    """An unknown integer resolved at compile time or dispatch time."""

    def __init__(self, choices=None, hint=None):
        self.choices = choices
        self.hint    = hint
        if hint is not None:
            self._value = int(hint)
        elif choices is not None:
            vals = list(choices)
            self._value = int(vals[0]) if vals else 512
        else:
            self._value = 512   # large default: keeps ranges non-empty vs typical tile sizes

    # Let Python use Symbol as a plain integer where needed (range, indexing, …)
    def __index__(self):  return self._value
    def __int__(self):    return self._value
    def __float__(self):  return float(self._value)

    # Arithmetic — return plain int so that downstream range() calls work
    def _i(self):  return self._value
    def __add__     (self, o): return self._i() + (int(o)  if isinstance(o, Symbol) else o)
    def __radd__    (self, o): return (int(o) if isinstance(o, Symbol) else o) + self._i()
    def __sub__     (self, o): return self._i() - (int(o)  if isinstance(o, Symbol) else o)
    def __rsub__    (self, o): return (int(o) if isinstance(o, Symbol) else o) - self._i()
    def __mul__     (self, o): return self._i() * (int(o)  if isinstance(o, Symbol) else o)
    def __rmul__    (self, o): return (int(o) if isinstance(o, Symbol) else o) * self._i()
    def __floordiv__(self, o): return self._i() // (int(o) if isinstance(o, Symbol) else o)
    def __rfloordiv__(self,o): return (int(o) if isinstance(o, Symbol) else o) // self._i()
    def __mod__     (self, o): return self._i() % (int(o)  if isinstance(o, Symbol) else o)
    def __eq__      (self, o): return self._i() == (int(o) if isinstance(o, Symbol) else o)
    def __lt__      (self, o): return self._i() < (int(o)  if isinstance(o, Symbol) else o)
    def __le__      (self, o): return self._i() <= (int(o) if isinstance(o, Symbol) else o)
    def __gt__      (self, o): return self._i() > (int(o)  if isinstance(o, Symbol) else o)
    def __ge__      (self, o): return self._i() >= (int(o) if isinstance(o, Symbol) else o)
    def __hash__    (self):    return hash(self._i())
    def __repr__    (self):    return f"airapi.symbol({self._value})"


# ---------------------------------------------------------------------------
# Arithmetic mixin — silently absorbs all operations, returns self
# ---------------------------------------------------------------------------

class _Stub:
    """Mixin: absorbs arithmetic, slicing, and assignment without error."""

    def __getitem__(self, key):  return self
    def __setitem__(self, key, value): pass

    def __add__     (self, o): return self
    def __radd__    (self, o): return self
    def __sub__     (self, o): return self
    def __rsub__    (self, o): return self
    def __mul__     (self, o): return self
    def __rmul__    (self, o): return self
    def __truediv__ (self, o): return self
    def __rtruediv__(self, o): return self
    def __matmul__  (self, o): return self
    def __rmatmul__ (self, o): return self
    def __neg__     (self):    return self
    def __pos__     (self):    return self
    def __abs__     (self):    return self
    def __iter__    (self):    return iter([self])
    def __len__     (self):    return 1
    def __bool__    (self):    return True


# ---------------------------------------------------------------------------
# Buffer and Tensor
# ---------------------------------------------------------------------------

class Buffer(_Stub):
    """A physical buffer allocated in a hardware scope."""

    def __init__(self, shape, dtype, scope=None, layout=None):
        self.shape  = shape
        self.dtype  = dtype
        self.scope  = scope
        self.layout = layout
        self._fields_map = None

    @property
    def fields(self):
        if self._fields_map is None:
            if not isinstance(self.dtype, BlockType):
                raise AttributeError("fields is only valid on BlockType buffers")
            self._fields_map = {}
            for f in self.dtype.fields:
                fshape = list(self.shape)
                if f.granularity > 1 and f.block_dim is not None:
                    dim = f.block_dim % len(fshape)
                    fshape[dim] = max(1, fshape[dim] // f.granularity)
                self._fields_map[f.name] = Buffer(tuple(fshape), f.dtype, scope=self.scope)
        return self._fields_map

    def __repr__(self):
        return f"Buffer(shape={self.shape}, dtype={self.dtype})"


class Tensor(_Stub):
    """A global interface tensor (host-accessible, no scope)."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

class Scope:
    def __init__(self, kind, ctx):
        self.kind = kind
        self.ctx  = ctx
    def __repr__(self):
        return f"{self.ctx.__class__.__name__}.{self.kind}()"


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class Channel(_Stub):
    def __init__(self, shape, dtype, kind="unicast", routing="circuit",
                 axis=None, fanout=None, depth=None):
        self.shape   = shape
        self.dtype   = dtype
        self.kind    = kind
        self.routing = routing
        self.axis    = axis
        self.fanout  = fanout
        self.depth   = depth

    def __getitem__(self, key):
        # ch[n] or ch[(n, m)] — selects subchannel (multicast) or tag (packet)
        return self

    def get(self, indices=None, asynchronous=False):
        shape = self.shape if isinstance(self.shape, tuple) else (int(self.shape),)
        buf = Buffer(shape, self.dtype)
        if asynchronous:
            return Token()
        return buf

    def put(self, indices=None, value=None):
        pass


# ---------------------------------------------------------------------------
# Fabric properties
# ---------------------------------------------------------------------------

class _FabricProp:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"airapi.{self.name}"
    def __eq__(self, other):
        return type(self) is type(other)
    def __hash__(self):
        return hash(self.name)

Scratchpad  = _FabricProp("Scratchpad")
Broadcast   = _FabricProp("Broadcast")
CacheDomain = _FabricProp("CacheDomain")


class Cascade(_FabricProp):
    def __init__(self, axis):
        super().__init__("Cascade")
        self.axis = axis
    def __eq__(self, other):
        return isinstance(other, Cascade) and self.axis == other.axis
    def __hash__(self):
        return hash(("Cascade", self.axis))
    def __repr__(self):
        return f"airapi.Cascade(axis={self.axis})"


class Adjacency(_FabricProp):
    def __init__(self, axis):
        super().__init__("Adjacency")
        self.axis = axis
    def __repr__(self):
        return f"airapi.Adjacency(axis={self.axis})"


class Disjoint:
    def __init__(self, prop):
        self.prop = prop
    def __repr__(self):
        return f"airapi.Disjoint({self.prop})"


class Fabric:
    def __init__(self, props=None):
        self.props = set(props or [])
    def __contains__(self, prop):
        return prop in self.props


# ---------------------------------------------------------------------------
# BlockType
# ---------------------------------------------------------------------------

class Field:
    def __init__(self, name, dtype, granularity=1, block_dim=None):
        self.name        = name
        self.dtype       = dtype
        self.granularity = granularity
        self.block_dim   = block_dim


class BlockType:
    def __init__(self, fields, dequant=None):
        self.fields  = fields
        self.dequant = dequant
    def __repr__(self):
        return f"airapi.BlockType(fields={[f.name for f in self.fields]})"


# ---------------------------------------------------------------------------
# Grid parsing helpers
# ---------------------------------------------------------------------------

def _parse_grid(iterable):
    """Return (shape, tile_sizes) by peeking at the iterable.

    For a plain range: exact shape and tile_sizes.
    For itertools.product or other: n_dims from first element, tile_sizes=(64,...).
    Falls back to ((1,), (64,)) if iteration is empty.
    """
    if isinstance(iterable, range):
        n = len(iterable)
        return (n,), (iterable.step,)

    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return (1,), (64,)

    if isinstance(first, tuple):
        n = len(first)
    else:
        n = 1
    # tile_sizes unknown for product; use 64 as a safe default
    return tuple(1 for _ in range(n)), tuple(64 for _ in range(n))


def _n_args(fn):
    """Count positional parameters of fn."""
    try:
        sig = inspect.signature(fn)
        return sum(
            1 for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
    except (ValueError, TypeError):
        return 0


def _call_body(ctx, fn):
    """Call fn with zero tile-coords, resizing ctx.tile_sizes to match fn arity."""
    n = _n_args(fn)
    # If we couldn't determine n_dims from the iterable, fix it now
    if len(ctx.tile_sizes) != n:
        ctx.tile_sizes = tuple(64 for _ in range(n))
        ctx.shape      = tuple(1  for _ in range(n))
    try:
        fn(*([0] * n))
    except Exception as e:
        warnings.warn(f"air stub: body call raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

class LaunchContext:
    def __init__(self, asynchronous=False, dependency=None):
        self.asynchronous = asynchronous
        self.token        = Token() if asynchronous else None
        self.search_space = {}
        self.constraints  = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def body(self):
        def decorator(fn):
            fn()
            return fn
        return decorator

    def compile(self, target="npu2"):
        return _CompiledKernel()

    def mlir(self):
        return "# (stub: MLIR emission not implemented)"

    def benchmark(self, target, inputs):
        return 0.0, {}


class _CompiledKernel:
    def __init__(self):
        self.compile_bindings = {}
        self.dispatch_symbols = set()
        self.last_dispatch    = {}

    def __call__(self, *args):
        import numpy as np
        if args:
            return np.zeros_like(args[-1])
        return None


class SegmentContext:
    def __init__(self, iterable, requires=None, asynchronous=False,
                 dependency=None, affinity=None, concurrency=None):
        self.requires    = requires or []
        self.asynchronous = asynchronous
        self.token       = Token() if asynchronous else None
        self.shape, self.tile_sizes = _parse_grid(iterable)
        self._registered = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def body(self, fn=None, *, placement=None):
        """Supports both @seg.body and @seg.body(placement=...)."""
        def _register(f):
            if not self._registered:
                self._registered = True
                _call_body(self, f)
            return f

        if fn is not None:
            # Used as bare @seg.body decorator
            return _register(fn)
        # Used as @seg.body(placement=...) — return decorator
        return _register

    def private(self):
        return Scope("private", self)


class HerdContext:
    def __init__(self, iterable, requires=None, asynchronous=False,
                 dependency=None, affinity=None, concurrency=None):
        self.requires     = requires or []
        self.asynchronous = asynchronous
        self.token        = Token() if asynchronous else None
        self.shape, self.tile_sizes = _parse_grid(iterable)
        self._registered  = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def body(self):
        def decorator(fn):
            if not self._registered:
                self._registered = True
                _call_body(self, fn)
            return fn
        return decorator

    def private(self):  return Scope("private", self)
    def shared(self):   return Scope("shared",  self)

    def channel(self, shape, dtype, kind="unicast", routing="circuit",
                axis=None, fanout=None, depth=None):
        return Channel(shape, dtype, kind=kind, routing=routing,
                       axis=axis, fanout=fanout, depth=depth)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def launch(ranges=None, capture=True, asynchronous=False, dependency=None):
    return LaunchContext(asynchronous=asynchronous, dependency=dependency)


def segment(iterable, requires=None, asynchronous=False,
            dependency=None, affinity=None, concurrency=None):
    return SegmentContext(iterable, requires=requires, asynchronous=asynchronous,
                          dependency=dependency, affinity=affinity, concurrency=concurrency)


def herd(iterable, requires=None, asynchronous=False,
         dependency=None, affinity=None, concurrency=None):
    return HerdContext(iterable, requires=requires, asynchronous=asynchronous,
                       dependency=dependency, affinity=affinity, concurrency=concurrency)


def alloc(shape, dtype, scope=None, layout=None):
    concrete = tuple(int(s) if isinstance(s, Symbol) else s for s in shape)
    return Buffer(concrete, dtype, scope=scope, layout=layout)


def tensor(shape, dtype):
    return Tensor(tuple(shape), dtype)


def symbol(choices=None, hint=None):
    return Symbol(choices=choices, hint=hint)


def requires(prop):
    """@air.requires(prop) — marks a body variant as requiring a fabric property."""
    def decorator(fn):
        fn._air_requires = prop
        return fn
    return decorator


def wait(*tokens):
    return Token()


# ---------------------------------------------------------------------------
# jit / compile
# ---------------------------------------------------------------------------

class _JitFunction:
    """Wraps a @air.jit-decorated function; compiles on first call."""

    def __init__(self, fn, strategy=None, target=None):
        self.fn       = fn
        self.strategy = strategy
        self.target   = target or "npu2"
        self._cache   = {}

    def __call__(self, *args, **kwargs):
        import numpy as np
        if args:
            return np.zeros_like(args[-1])
        return None

    def compile(self, strategy=None, target=None, tile_sizes=None,
                symbols=None, aot=False):
        return _CompiledKernel()

    def benchmark(self, inputs=None, target=None):
        return 0.0

    def __repr__(self):
        return f"airapi.jit({self.fn.__name__})"


def jit(fn=None, *, strategy=None, target=None):
    """@air.jit — trace an AIR kernel function.

    Usage:
        @air.jit
        def kernel(...): ...

        @air.jit(strategy="megakernel", target="npu2")
        def kernel(...): ...
    """
    if fn is not None:
        # Bare @air.jit — fn is the decorated function
        return _JitFunction(fn)
    # Called with arguments: @air.jit(strategy=...) — return decorator
    def decorator(f):
        return _JitFunction(f, strategy=strategy, target=target)
    return decorator


def compile(fn, strategy=None, target=None, tile_sizes=None,
            symbols=None, aot=False):
    """Compile a @air.jit function with explicit parameters."""
    return _CompiledKernel()
