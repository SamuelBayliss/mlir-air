# AIR Python DSL (`airapi`)

`airapi` is a high-level Python DSL for programming spatial accelerators. It
sits above the `air.dialects` MLIR bindings and is designed to be the primary
user-facing interface for expressing AIR programs.

For the underlying IR semantics that `airapi` compiles to, see
[AIRComputeModel.md](AIRComputeModel.md).

---

## 1. Compute Model

### 1.1 Hierarchy

Programs are expressed as a three-level hierarchy:

```
air.launch          — hands control from host to accelerator
  air.segment       — a resource container: compute tiles + shared memory
    air.herd        — a concurrent grid of processing elements (tiles/wavefronts)
```

Each level is expressed as a Python context manager. The body of a `launch` or
`segment` is a decorator-registered function; the body of a `herd` receives
tile coordinates as arguments.

```python
with air.launch() as launch:
    @launch.body
    def _():
        with air.segment(range(0, M, seg_m), requires=[air.Scratchpad]) as seg:
            @seg.body
            def _(ss,):
                with air.herd(range(0, seg_m, tile_m)) as h:
                    @h.body
                    def _(tx,):
                        ...
```

### 1.2 Segments as resource containers

A `segment` owns a set of hardware resources: compute tiles, on-device memory
(L2), and DMA engines. These resources are reserved for the duration of the
segment and are not available to other segments while it is live.

**Segments within a launch are implicitly concurrent.** All segments declared
in a launch body execute in parallel; the runtime allocates disjoint hardware
resources for each by default.

**Resources are disjoint by default.** Two segments in the same launch occupy
non-overlapping hardware. This is what makes their concurrent execution
possible.

**Affinity tokens relax disjointness.** Segments that share the same
`affinity=` token are bound to the same hardware resources. Because they
share resources they cannot execute simultaneously — the runtime ensures they
are mutually exclusive. No ordering between them is implied; the runtime may
execute them in either order.

**Dependency tokens impose ordering.** A segment with `dependency=t` does not
begin until token `t` is signaled. Dependency implies sequential execution and
allows the compiler to reuse the preceding segment's hardware resources for the
dependent one, since the two segments cannot overlap in time.

These two mechanisms are orthogonal and compose:

| Token kind | Resource | Time |
|---|---|---|
| Neither (default) | Disjoint | Concurrent |
| `affinity` | Shared | Mutually exclusive, order unspecified |
| `dependency` | May be reused | Ordered sequential |
| Both | Shared, may be reused | Ordered sequential |

### 1.3 Segment group size

The number of segments declared in a launch body is fixed at compile time and
defines the **segment group size** — the number of concurrent resource
containers the runtime must provision per launch iteration. In a future
extension, the group size may be parameterised by a launch-time `air.symbol`,
enabling the runtime to dispatch a dynamically determined number of concurrent
segments based on available hardware.

### 1.4 Herds

A `herd` defines an always-parallel grid of processing elements within its
enclosing segment. Every point in the herd's iteration space executes
concurrently on a distinct hardware resource. Herd bodies receive tile
coordinates as positional arguments and use them to address per-tile regions of
L1 or L2 memory.

---

## 2. Capability System

### 2.1 `requires=` — declaring hardware capabilities

`requires=` on a `segment` or `herd` declares the **minimum set of hardware
capabilities** the implementation needs. These are abstract spatial properties,
not hardware tier names:

| Property | Meaning |
|---|---|
| `air.Scratchpad` | A shared low-latency memory region exists within the segment scope. Maps to MemTile SRAM on NPU, LDS on GPU. |
| `air.CacheDomain` | The segment scope has access to a shared L2 cache with fast hardware atomics. Maps to XCD on CDNA3/CDNA4. |
| `air.Cascade(axis)` | Direct neighbour-to-neighbour links exist along the given axis without DMA overhead. Maps to AIE cascade wiring on NPU. |
| `air.Broadcast` | One-to-many data delivery to all tiles in the herd without per-tile DMA. Maps to MemTile broadcast DMA on NPU. |
| `air.Adjacency(axis)` | Efficient local-port transfers between neighbouring tiles along an axis. |

`requires=` is a conjunction: `requires=[A, B]` means the target must provide
both A and B. If no hardware tier on the target satisfies the full set, the
variant is not compiled for that target.

`requires=` also gates scope operations:

- `seg.private()` allocations require `air.Scratchpad` in the segment's `requires`.
- `air.ops.atomic_add(..., scope=seg)` requires `air.CacheDomain` in the segment's `requires`.
- Cascade channels require `air.Cascade` in the segment's `requires`.

### 2.2 Body variants — alternative implementations

Multiple body functions may be registered on the same context manager. Each
variant may carry its own `@air.requires` decoration and `placement=` constraint.
The compiler produces a compiled variant for each body whose requirements can be
satisfied on some target in the compilation set. The runtime selects the
appropriate variant for the detected hardware.

```python
with air.segment(..., requires=[air.Scratchpad, air.CacheDomain]) as seg:

    @seg.body(placement=air.Disjoint(air.CacheDomain))
    @air.requires(air.CacheDomain)
    def _(ss, st):
        # Implementation using XCD-scoped atomics.
        # Selected on CDNA3/CDNA4 where CacheDomain is available.
        xcd_acc = air.alloc([tm, tn], f32, scope=seg.private())
        ...
        air.ops.atomic_add(xcd_acc, partial, scope=seg)

    @seg.body
    def _(ss, st):
        # Fallback: per-wavefront store direct to global.
        # Selected on NPU, CDNA2, RDNA where CacheDomain is unavailable.
        ...
```

Both variants compute the same result. The compiler selects the first variant
whose `requires` are fully satisfied by the target; if none are satisfied the
compilation fails with a diagnostic listing the unsatisfied properties.

`air.Disjoint(prop)` on a `placement=` argument is a placement constraint
asserting that each segment instance must reside on a distinct physical unit
providing `prop` — for example, a distinct XCD. This is the default for
`CacheDomain`-based reduction patterns where per-XCD L2 isolation is required
for correctness.

### 2.3 Capability registry

The set of capability properties forms a **stable, versioned vocabulary**. A
well-defined capability registry is what makes `requires=` a portable
abstraction rather than a target hint:

- Each property has a **semantic contract** — what it guarantees to the
  program, independent of how the hardware implements it.
- Each property has a **query mechanism** — the runtime can determine whether
  the detected hardware satisfies it without knowing the ISA.
- Capabilities are **composable**: `requires=[A, B, C]` requires all three.
- Capabilities may be **quantitative** in future extensions:
  `air.Scratchpad(min_kb=512)` to gate on sufficient buffer size.

---

## 3. Compilation and Dispatch

### 3.1 Multi-variant AOT compilation

The compiler is invoked offline and produces a **dispatch artifact**: a
collection of compiled ISA binaries, one per (target ISA family, satisfied
capability set) combination, plus a manifest indexed by capability fingerprint.
There is no single-target compilation; the output is always a collection of
variants.

At deployment:

1. The runtime queries the detected hardware for its capability fingerprint.
2. The dispatch manifest is consulted to find the variant whose required
   capabilities are a subset of the detected capabilities.
3. The selected ISA binary is loaded and executed.

This is the mechanism that makes `airapi` programs portable across NPU and GPU
targets with identical source code:

```python
# Same source — different variants compiled and dispatched per target
with air.segment(..., requires=[air.Scratchpad]) as seg:
    @seg.body
    def _(ss, st):
        staging = air.alloc([seg_m, tile_k], bf16, scope=seg.private())
        ...
```

On NPU: `Scratchpad` maps to MemTile SRAM; a two-hop DMA variant is compiled.
On GPU: `Scratchpad` maps to LDS; a one-hop variant is compiled.

### 3.2 AIR as a portable intermediate representation

AIR MLIR occupies a position analogous to CUDA PTX: a target-agnostic
intermediate that captures the programmer's intent at the level of spatial
hierarchy, capability requirements, and data movement, without committing to a
specific ISA.

| | CUDA / PTX | AIR |
|---|---|---|
| Portable intermediate | PTX (virtual ISA) | AIR MLIR dialect |
| Capability index | compute capability integer | `requires=` property set |
| Pre-compiled variants | SASS sections per compute capability | ISA binaries per (ISA family, capability set) |
| JIT fallback | PTX → SASS via driver | AIR MLIR → ISA via runtime JIT (where available) |
| Fat artifact | cubin (SASS + PTX) | dispatch manifest (ISA binaries + optional AIR MLIR) |

The key structural difference from CUDA: AMD does not have a single JIT path
equivalent to PTX→SASS. Today, all variants are pre-compiled AOT. Where a
runtime platform provides a JIT backend, the dispatch artifact may bundle AIR
MLIR as a fallback intermediate: if no pre-compiled variant matches the detected
hardware, the JIT backend compiles the bundled IR to ISA at load time, provided
the hardware satisfies the declared `requires=` properties.

This design means the capability vocabulary (`requires=`) is the central
portability surface. A well-chosen set of properties is what allows a single
AIR program to target NPU, CDNA, and RDNA architectures within the same
compilation run.

---

## 4. API Reference

### 4.1 `airapi` — structural and declaration API

```python
import airapi as air
```

#### `air.launch`

```python
air.launch(ranges=None, capture=True, asynchronous=False, dependency=None)
```

Returns a `LaunchContext`. The launch body is registered with `@launch.body`.
When `asynchronous=True` the launch returns a token (`launch.token`) that
signals when all instances complete.

#### `air.segment`

```python
air.segment(iterable, requires=None, asynchronous=False,
            dependency=None, affinity=None, concurrency=None)
```

`iterable` defines the segment iteration space (a `range` or
`itertools.product`). `requires` is a list of fabric property objects.

The body is registered with `@seg.body` or `@seg.body(placement=...)`. Multiple
body functions may be registered; each may carry `@air.requires(prop)`.

`seg.private()` returns a scope object usable in `air.alloc`; valid only when
`air.Scratchpad` is in `requires`.

#### `air.herd`

```python
air.herd(iterable, requires=None, asynchronous=False,
         dependency=None, affinity=None, concurrency=None)
```

`iterable` defines the herd grid. The body is registered with `@h.body` and
receives tile coordinates as positional arguments — one per dimension of the
iteration space.

`h.private()` returns L1 (per-tile) scope. `h.shared()` returns the enclosing
segment's L2 (Scratchpad) scope.

`h.channel(shape, dtype, kind, routing, axis, fanout, depth)` declares an
intra-segment channel.

#### `air.alloc`

```python
air.alloc(shape, dtype, scope=None, layout=None) -> Buffer
```

Allocates a buffer in the given scope. `scope` is one of `h.private()`,
`h.shared()`, or `seg.private()`. Layout may be an `AffineLayout` or
`LinearLayout` from `airapi.layout`.

#### `air.tensor`

```python
air.tensor(shape, dtype) -> Tensor
```

Declares a host-accessible interface tensor. Tensors captured into a `launch`
body are the program's inputs and outputs. They carry no scope or layout.

#### `air.symbol`

```python
air.symbol(choices=None, hint=None) -> Symbol
```

A value known at dispatch time (from input shapes or explicit binding) rather
than compile time. `choices` enumerates the values the autotuner or dispatch
system may select from. `hint` sets the representative value used during
tracing.

#### `air.wait`

```python
air.wait(*tokens) -> Token
```

AND-join: returns a token that signals when all input tokens have signaled.
Also available as `airapi.ops.wait`.

#### `air.requires`

```python
@air.requires(prop)
def body_fn(...): ...
```

Marks a body function as requiring `prop` to be available on the target.
Applied to body functions registered on a segment or herd.

#### `air.jit`

```python
@air.jit
def kernel(...): ...

@air.jit(strategy=None, target=None)
def kernel(...): ...
```

Marks a function for tracing and compilation. Calling the function runs the
stub tracer; `.compile(target, ...)` triggers the full compilation pipeline.

#### `air.compile`

```python
air.compile(fn, strategy=None, target=None, tile_sizes=None,
            symbols=None, aot=False) -> CompiledKernel
```

Compiles a `@air.jit`-decorated function with explicit parameters.

---

### 4.2 Fabric properties

```python
import airapi as air
```

| Object | Type | Description |
|---|---|---|
| `air.Scratchpad` | singleton | Shared low-latency memory within the segment |
| `air.CacheDomain` | singleton | Shared L2 + hardware atomics within the segment |
| `air.Broadcast` | singleton | Hardware-accelerated one-to-many delivery |
| `air.Cascade(axis)` | class | Direct neighbour links along `axis` (0=column, 1=row) |
| `air.Adjacency(axis)` | class | Local-port transfers between neighbours along `axis` |
| `air.Disjoint(prop)` | class | Placement constraint: each segment on a distinct `prop` unit |

---

### 4.3 `airapi.ops` — compute and memory operations

```python
import airapi.ops
```

All operations return a `Token` (or a `Buffer` for compute ops that produce a
result). Tokens carry temporal (completion) and spatial (hardware binding)
information.

#### Memory transfers

```python
airapi.ops.load(dst, src_slice, pad_before=None, pad_after=None,
                dependency=None) -> Token
```
DMA from a global tensor slice into a local buffer. Lowers to
`air.dma_memcpy_nd`.

```python
airapi.ops.store(src, dst_slice, pad_before=None, pad_after=None,
                 dependency=None) -> Token
```
DMA from a local buffer into a global tensor slice.

```python
airapi.ops.copy(src_slice, dst_slice, pad_before=None, pad_after=None,
                dependency=None) -> Token
```
Direct buffer-to-buffer copy (within the same memory level).

#### Compute

```python
airapi.ops.dot(a, b, acc=None, alpha=1.0, transpose_b=False,
               dependency=None) -> Token
```
Matrix multiply-accumulate: `acc += alpha * (a @ b)`.

```python
airapi.ops.reduce(a, axis, op, dependency=None) -> Buffer
```
Reduction along `axis`. `op` is one of `airapi.ops.sum`, `airapi.ops.max`,
`airapi.ops.min`.

```python
airapi.ops.exp(a) -> Buffer
airapi.ops.relu(a) -> Buffer
```
Elementwise unary operations.

```python
airapi.ops.stack(tensors, axis=0) -> Buffer
```
Stacks a sequence of buffers along `axis`.

```python
airapi.ops.dequant(*fields, out, dtype) -> Token
```
Explicit dequantisation from per-field allocations. Field arguments must match
the declaration order of the `BlockType`.

#### Atomics

```python
airapi.ops.atomic_add(dst, src, scope, dependency=None) -> Token
```
Hardware atomic add into `dst`. `scope=h` is intra-herd (always available).
`scope=seg` is intra-segment L2 atomic; requires `air.CacheDomain` in the
segment's `requires`.

#### Token operations

```python
airapi.ops.wait(*tokens) -> Token       # AND-join
airapi.ops.wait_any(*tokens) -> Token   # OR-join; returns the first to complete
airapi.ops.cancel(token)                # abort a pending op (no-op if complete)
```

---

### 4.4 `airapi.types` — element types

```python
from airapi.types import bf16, f32, f16, i8, i4, e8m0, fp8
```

Each is a `DType` singleton with `.name` and `.itemsize` attributes.

---

### 4.5 `airapi.layout` — buffer layouts

```python
from airapi.layout import AffineLayout, LinearLayout, RowMajor, ColMajor
```

`AffineLayout(strides=[...])` — explicit row strides in element units.  
`LinearLayout(matrix)` — a linear map from logical to physical indices.  
`RowMajor()` / `ColMajor()` — convenience constructors for standard layouts.

---

### 4.6 Block floating-point types

```python
from airapi import BlockType, Field
```

`BlockType(fields, dequant=None)` declares a block floating-point format.
`Field(name, dtype, granularity=1, block_dim=None)` declares one field within
the block. A buffer allocated with a `BlockType` dtype exposes a `.fields`
dict mapping field names to sub-buffers with appropriately scaled shapes.

---

## 5. Examples

The `programming_examples/api_examples/` directory contains worked examples
in order of increasing complexity:

| File | Concepts |
|---|---|
| `01_axpy.py` | Baseline: 1D herd, `h.private()` buffers, `load`/`store` |
| `02_eltwise_add.py` | 2D herd, `air.symbol` for compile-time tile size search |
| `03_gemm_npu.py` | `air.Scratchpad`, `seg.private()`, two-hop DMA, `air.dot` |
| `04_gemm_gpu.py` | Same as 03 compiled to GPU; `Scratchpad` maps to LDS |
| `05_gemm_cachedomain.py` | Body variants: `air.CacheDomain` vs fallback, `atomic_add` |
| `06_gemm_autotuned.py` | `air.symbol` with `choices=` for autotuner search space |
| `07_gemm_mxfp8.py` | `BlockType` with per-field staging, MXFP8 dequantisation |
| `08_flash_attention.py` | `air.Cascade`, `air.Broadcast`, cascade channels, two-herd pipeline |

Examples 03 and 04 illustrate the portability mechanism directly: they express
identical algorithms with the same `requires=[air.Scratchpad]` annotation; the
compiler produces different ISA binaries for NPU and GPU without any change to
the source program.

Example 05 illustrates body variants: the same GEMM reduction is expressed as
an XCD-atomic path (selected on CDNA3/CDNA4) and a per-wavefront-store
fallback (selected elsewhere). Both produce identical results; the
`@air.requires(air.CacheDomain)` annotation controls which is compiled for
which target.
