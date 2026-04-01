# GPT-OSS-120B Progressive Test Plan

This document outlines the progressive test suite for building up to the full 8-GPU GPT-OSS-120B implementation.

## Test Strategy

Build complexity incrementally:
1. **Phase 0**: Basic hierarchy without `air.rank` (tests 00-01)
2. **Phase 1**: Simple `air.rank` usage (tests 02-04)
3. **Phase 2**: Nested `air.rank` in `air.launch` (tests 05-07)
4. **Phase 3**: Cross-rank communication (tests 08-10)
5. **Phase 4**: Full implementation features (tests 11-15)

---

## Phase 0: Baseline Without air.rank

### ✅ Test 00: Basic Hierarchy (`test-00-basic-hierarchy.mlir`)
**Status**: Created

**Purpose**: Establish baseline three-level hierarchy

**Features tested**:
- `air.launch` with single device
- `air.segment` with L2 memory allocation
- `air.herd` with 4x4 compute unit grid
- Tiled GEMV computation (representative of QKV projection)
- DMA operations (DDR ↔ L2 ↔ LDS)

**Success criteria**:
- Parses and round-trips correctly
- Verifier accepts the IR
- Can be lowered through existing AIR passes

---

### ✅ Test 01: Basic Channel Communication (`test-01-basic-channel.mlir`)
**Status**: Created

**Purpose**: Demonstrate channel-based segment communication

**Features tested**:
- `air.channel` declaration
- `air.channel.put` / `air.channel.get`
- Producer-consumer pattern between segments
- Async tokens for ordered execution

**Success criteria**:
- Channel data transfer works correctly
- Async dependencies are properly tracked
- Both sync and async variants work

---

## Phase 1: Simple air.rank Usage (Currently Supported)

### Test 02: Top-Level air.rank (1D)
**Status**: TODO

**Purpose**: Simplest `air.rank` usage at module top-level

**Features tested**:
- `air.universe.alloc` with capacity
- `air.rank` with 1D iteration space (e.g., 2 ranks)
- Rank index used to partition work
- Each rank contains `air.launch` → `air.segment`

**Example structure**:
```mlir
%universe = air.universe.alloc(%c2)
air.rank universe(%universe) (%r) in (%n = %c2) args(...) {
  // Rank %r processes its partition
  air.launch (%lx) in (%ls = %c1) args(%rank = %r, ...) {
    air.segment @work { ... }
  }
}
```

**Success criteria**:
- Works with current PR #1479 implementation
- Verifier accepts top-level rank
- Can lower with `air-rank-to-launch` pass

---

### Test 03: Top-Level air.rank (2D)
**Status**: TODO

**Purpose**: Cartesian rank grid (e.g., 2×3 mesh)

**Features tested**:
- 2D rank iteration space
- Rank coordinates used for neighbor communication (future)
- Multi-dimensional indexing

**Example structure**:
```mlir
air.rank (%rx, %ry) in (%nx = %c2, %ny = %c3) args(...) {
  // Each rank (%rx, %ry) in the 2×3 grid
  air.launch (%lx) in (%ls = %c1) args(%x = %rx, %y = %ry, ...) { ... }
}
```

**Success criteria**:
- 2D indexing works correctly
- Serialization to nested loops (`scf.for`) is correct

---

### Test 04: Top-Level air.rank with Async Dependencies
**Status**: TODO

**Purpose**: Multiple ranks with async orchestration

**Features tested**:
- `air.rank async` with dependency tokens
- Sequential vs. parallel rank execution
- `air.wait_all` for synchronization

**Example structure**:
```mlir
%t0 = air.rank async (%r0) in (%n0 = %c2) { ... }
%t1 = air.rank async [%t0] (%r1) in (%n1 = %c2) { ... }
air.wait_all [%t1]
```

**Success criteria**:
- Dependencies prevent premature execution
- Serialization respects async order

---

## Phase 2: Nested air.rank (Requires Verifier Relaxation)

### Test 05: Single air.rank Inside air.launch
**Status**: TODO

**Purpose**: Test basic nested rank (requires PR #1479 verifier change)

**Features tested**:
- `air.rank` nested inside `air.launch`
- Single universe within launch context
- Rank index accessible to segment

**Example structure**:
```mlir
%universe = air.universe.alloc(%c4)
air.launch (%lx) in (%ls = %c1) args(%u = %universe, ...) {
  air.rank universe(%u) (%r) in (%n = %c4) args(...) {
    air.segment @work args(%rank = %r, ...) { ... }
  }
}
```

**Success criteria**:
- Verifier accepts nested rank
- Correct semantic: rank index → device ID mapping
- Each rank instance maps to distinct GPU device

**Blocker**: Requires relaxing verifier restriction

---

### Test 06: Two Sequential air.rank Inside air.launch
**Status**: TODO

**Purpose**: Two ranks executing sequentially within launch

**Features tested**:
- Multiple rank ops in same launch
- Sequential execution (no concurrency tokens)
- Different universes for each rank

**Example structure**:
```mlir
%u1 = air.universe.alloc(%c2)
%u2 = air.universe.alloc(%c3)
air.launch (%lx) in (%ls = %c1) {
  air.rank universe(%u1) (%r1) in (%n1 = %c2) { ... }  // Runs first
  air.rank universe(%u2) (%r2) in (%n2 = %c3) { ... }  // Runs second
}
```

**Success criteria**:
- Ranks execute in order (serial)
- No resource conflicts

**Blocker**: Requires relaxing verifier restriction

---

### Test 07: Two Concurrent air.rank Inside air.launch
**Status**: TODO

**Purpose**: Two ranks executing concurrently (key for GPT-OSS-120B)

**Features tested**:
- Async tokens force concurrent execution
- Two separate universes (e.g., attention + FFN clusters)
- `air.wait_all` ensures both complete

**Example structure**:
```mlir
%u_attn = air.universe.alloc(%c2)
%u_ffn  = air.universe.alloc(%c6)
air.launch (%lx) in (%ls = %c1) {
  %t_attn = air.rank async universe(%u_attn) (%r_attn) in (%n_attn = %c2) { ... }
  %t_ffn  = air.rank async universe(%u_ffn)  (%r_ffn)  in (%n_ffn  = %c6) { ... }
  air.wait_all [%t_attn, %t_ffn]
}
```

**Success criteria**:
- Both ranks execute in parallel (conceptually)
- Tokens properly express concurrency intent

**Blocker**: Requires relaxing verifier restriction

---

## Phase 3: Cross-Rank Communication

### Test 08: Channel Between Ranks (Broadcast)
**Status**: TODO

**Purpose**: One rank broadcasts to multiple ranks

**Features tested**:
- Channel declared under `air.launch`
- 1-to-N communication pattern
- Channel shape `[1, N]` for broadcast

**Example structure**:
```mlir
air.launch (%lx) in (%ls = %c1) {
  air.channel @broadcast [1, 6]  // 1 source → 6 targets

  %t_src = air.rank async universe(%u_src) (%r_src) in (%n_src = %c1) {
    scf.for %i = %c0 to %c6 step %c1 {
      air.channel.put @broadcast[%c0, %i] (%data[...]) : ...
    }
  }

  %t_dst = air.rank async universe(%u_dst) (%r_dst) in (%n_dst = %c6) {
    air.channel.get @broadcast[%c0, %r_dst] (%buf[...]) : ...
  }

  air.wait_all [%t_src, %t_dst]
}
```

**Success criteria**:
- All target ranks receive data
- Channel correctly routes based on index

---

### Test 09: Channel Between Ranks (Gather)
**Status**: TODO

**Purpose**: Multiple ranks send to one rank

**Features tested**:
- N-to-1 communication pattern
- Channel shape `[N, 1]` for gather
- Receiving rank collects from all sources

**Example structure**:
```mlir
air.channel @gather [6, 1]  // 6 sources → 1 target

// 6 FFN ranks each send their result
%t_ffn = air.rank async universe(%u_ffn) (%r_ffn) in (%n_ffn = %c6) {
  air.channel.put @gather[%r_ffn, %c0] (%result[...]) : ...
}

// 1 attention rank receives all
%t_attn = air.rank async universe(%u_attn) (%r_attn) in (%n_attn = %c1) {
  scf.for %i = %c0 to %c6 step %c1 {
    air.channel.get @gather[%i, %c0] (%buf[...]) : ...
  }
}
```

**Success criteria**:
- Source rank can send independently
- Target rank receives from all sources

---

### Test 10: Intra-Rank All-Reduce (Peer-to-Peer)
**Status**: TODO

**Purpose**: Ranks within same universe exchange data

**Features tested**:
- Peer-to-peer communication pattern
- Channel shape `[N, N]` for all-to-all
- Example: attention all-reduce between 2 GPUs

**Example structure**:
```mlir
air.channel @allreduce [2, 1]

air.rank universe(%u_attn) (%r) in (%n = %c2) {
  // Each rank sends its partial
  air.channel.put @allreduce[%r, %c0] (%partial[...]) : ...

  // Each rank receives peer's partial
  %peer = arith.xori %r, %c1 : index  // 0↔1 swap
  air.channel.get @allreduce[%peer, %c0] (%peer_partial[...]) : ...

  // Accumulate: local + peer
  linalg.add ins(%partial, %peer_partial) outs(%result)
}
```

**Success criteria**:
- Each rank gets peer's data
- Reduction is correct

---

## Phase 4: Full Implementation Features

### Test 11: Routed Channel (MoE Dispatch)
**Status**: TODO

**Purpose**: Content-based routing (expert_id → FFN rank)

**Features tested**:
- `routing_key_field` attribute
- `routing_table` for expert mapping
- Dynamic routing based on packet content

**Requires**: Dialect extension for routed channels

---

### Test 12: Gather-Reduce Channel
**Status**: TODO

**Purpose**: Weighted accumulation from multiple sources

**Features tested**:
- `reduce_op = "add"` attribute
- `expected_contributions` for top-k
- Automatic weighted sum on get side

**Requires**: Dialect extension for gather-reduce semantics

---

### Test 13: Multi-Segment Rank Body
**Status**: TODO

**Purpose**: Complex computation within each rank

**Features tested**:
- Multiple segments per rank
- Segment-to-segment channels within rank
- Representative of full attention or FFN pipeline

---

### Test 14: Simplified Two-Cluster GPT
**Status**: TODO

**Purpose**: Minimal attention + FFN without full MoE

**Features tested**:
- 2-rank attention cluster
- 2-rank FFN cluster (fixed experts, no routing)
- Cross-cluster broadcast and gather
- End-to-end layer

---

### Test 15: Full GPT-OSS-120B
**Status**: TODO (already exists as `gpt_oss_120b_refactored.mlir`)

**Purpose**: Complete 8-GPU implementation

**Features tested**:
- 2-rank attention cluster (TP)
- 6-rank FFN cluster (expert parallelism)
- Routed expert dispatch
- Gather-reduce expert results
- All channel patterns

---

## Test Execution Notes

### Running Tests
```bash
# Parse and round-trip
air-opt test-XX-name.mlir | FileCheck test-XX-name.mlir

# Run specific passes
air-opt test-XX-name.mlir -air-rank-to-launch | FileCheck test-XX-name.mlir

# Full lowering pipeline (once implemented)
air-opt test-XX-name.mlir -air-rank-to-launch -air-to-aie | FileCheck test-XX-name.mlir
```

### Dependencies Between Tests

```
Phase 0 (00-01) ← No dependencies
    ↓
Phase 1 (02-04) ← Requires PR #1479 (MERGED)
    ↓
Phase 2 (05-07) ← Requires verifier relaxation
    ↓
Phase 3 (08-10) ← Requires Phase 2
    ↓
Phase 4 (11-15) ← Requires channel extensions
```

### Blockers

1. **Phase 2 blocker**: Verifier restriction in PR #1479
   - **Action**: Comment posted to PR #1479 requesting relaxation
   - **Status**: Awaiting response

2. **Phase 4 blocker**: Missing dialect ops
   - `air.routing_table` op
   - Routed channel attributes
   - Gather-reduce channel attributes
   - **Action**: File issues for each extension
   - **Status**: Not yet filed

---

## Success Metrics

Each test must:
1. ✅ Parse without errors
2. ✅ Round-trip correctly (`air-opt test.mlir | air-opt`)
3. ✅ Pass verifier
4. ✅ Have FileCheck assertions for key IR properties
5. ✅ Document what it tests and why

Optional (for later phases):
6. Lower through AIR pipeline
7. Generate executable code
8. Run on actual hardware
