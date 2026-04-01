# GPT-OSS-120B Refactoring Summary: Using air.rank and air.universe

## Overview

Refactored `gpt-oss-120b.mlir` to properly use the new `air.rank` and `air.universe` primitives from PR #1479 to express the 8-GPU multi-device architecture.

## Key Architectural Changes

### Original Implementation Issues

1. **Single air.launch with manual branching**: Used `air.launch (%rank) in (%universe_size = %c8)` where `%rank` was incorrectly treated as a GPU ID
2. **No explicit multi-device topology**: GPU assignment was handled through control flow rather than IR structure
3. **Missing universe declarations**: No `!air.universe` type or allocation

### New Implementation (Option B)

**Structure:**
```
air.universe.alloc (2 GPUs for attention)
air.universe.alloc (6 GPUs for FFN)
air.launch (single host→accelerator transfer)
  ├─ air.rank async universe(attn) → 2-way parallelism
  │   └─ air.segment @attn_seg → QKV, Flash Attention, Router
  │
  ├─ air.rank async universe(ffn) → 6-way parallelism
  │   └─ air.segment @expert_seg → Weight-stationary FFN/MoE
  │
  └─ air.wait_all [%tok_attn, %tok_ffn]
```

## Key Design Decisions

### 1. Two Separate Universes
```mlir
%attn_universe = air.universe.alloc(%c2)  // 2 GPUs for attention
%ffn_universe  = air.universe.alloc(%c6)  // 6 GPUs for FFN/MoE
```

**Rationale:**
- Explicitly declares resource pools for each cluster
- Enables separate scheduling/placement policies per cluster
- Future: could specify topology attributes (e.g., interconnect bandwidth)

### 2. Nested air.rank Inside air.launch
```mlir
air.launch (%lx) in (%ls = %c1) args(...) {
  %tok_attn = air.rank async universe(%u_attn) (%attn_gpu) in (%na = %c2) {...}
  %tok_ffn  = air.rank async universe(%u_ffn) (%ffn_gpu) in (%nf = %c6) {...}
  air.wait_all [%tok_attn, %tok_ffn]
}
```

**Rationale:**
- Single host→accelerator control transfer (preserves mega-kernel semantics)
- Both `air.rank` ops are **async** with **concurrency tokens** → forces parallel execution
- Channels scoped under `air.launch` enable cross-universe communication
- Clean semantic: `air.launch` establishes context, `air.rank` establishes multi-device parallelism

**Requires:** Relaxing the verifier restriction from PR #1479 to allow `air.rank` inside `air.launch`

### 3. Concurrency Tokens Force Parallelism
```mlir
%tok_attn = air.rank async universe(%u_attn) ...
%tok_ffn  = air.rank async universe(%u_ffn) ...
air.wait_all [%tok_attn, %tok_ffn]
```

**Rationale:**
- Without tokens, serial semantics would execute attention cluster, then FFN cluster
- **Async tokens** + **wait_all** at the end ensures both clusters run **concurrently**
- This is critical for the architecture: attention and FFN must overlap for pipeline efficiency

### 4. Channel Scope Under air.launch
```mlir
air.launch (...) {
  air.channel @hidden_dispatch [2, 6]       // attn→FFN broadcast
  air.channel @expert_result_gather [6, 2]  // FFN→attn gather
  air.channel @attn_allreduce [2, 1]        // intra-attention reduce
  air.channel @expert_exchange [6, 6]       // intra-FFN exchange

  air.rank async universe(%u_attn) ...
  air.rank async universe(%u_ffn) ...
}
```

**Rationale:**
- Channels visible to both rank universes → cross-universe communication
- Intra-universe channels (e.g., `@attn_allreduce`) also declared here
- Clean scoping: channels lifetime matches the launch context

## Index Mapping

### Attention Cluster
- `air.rank` index `%attn_gpu` ∈ {0, 1}
- Maps to absolute GPU IDs 0, 1 in the system
- Each rank handles 12 attention heads (head sharding)

### FFN Cluster
- `air.rank` index `%ffn_gpu` ∈ {0, 1, 2, 3, 4, 5}
- Maps to absolute GPU IDs 2, 3, 4, 5, 6, 7 in the system
- Expert assignment:
  - ffn_gpu 0 → experts [0,1,2]
  - ffn_gpu 1 → experts [3,4,5]
  - ffn_gpu 2 → experts [6,7,8]
  - ffn_gpu 3 → experts [9,10,11]
  - ffn_gpu 4 → experts [12,13]
  - ffn_gpu 5 → experts [14,15]

## Communication Patterns

### Cross-Universe Channels

1. **Attention → FFN dispatch** (`@hidden_dispatch [2, 6]`)
   - Source: attention rank 0 (after all-reduce)
   - Target: all 6 FFN ranks (broadcast)
   - Payload: hidden state [1x2880xf16]

2. **Attention → FFN routing** (`@expert_route_dispatch [2, 6]`)
   - Source: attention rank 0
   - Target: FFN ranks (routed by expert_id)
   - Payload: route packets [token_id, expert_id, gate_weight]
   - **Extended semantic:** routed_put using routing table

3. **FFN → Attention gather** (`@expert_result_gather [6, 2]`)
   - Source: all 6 FFN ranks (those with activated experts)
   - Target: attention ranks
   - Payload: expert outputs [1x2880xf16]
   - **Extended semantic:** gather_reduce (weighted sum of top-k results)

### Intra-Universe Channels

4. **Attention all-reduce** (`@attn_allreduce [2, 1]`)
   - Between the 2 attention ranks
   - Reduces partial O-projection outputs

5. **FFN peer exchange** (`@expert_exchange [6, 6]`)
   - Between FFN ranks
   - Handles tokens whose 2 experts are on different FFN GPUs

## Advantages Over Original Implementation

### 1. **Explicit Multi-Device Topology**
- Compiler can see the 2-cluster + 6-FFN GPU structure in the IR
- Enables topology-aware optimizations (placement, routing, memory allocation)

### 2. **Declarative Parallelism**
- Concurrency tokens replace control-flow branching for cluster selection
- Compiler can reason about parallel execution

### 3. **Clean Semantic Hierarchy**
```
air.universe (resource pool declaration)
  ↓
air.launch (host→accelerator transfer)
  ↓
air.rank (multi-device parallelism)
  ↓
air.segment (tile + L2 memory)
  ↓
air.herd (array of compute units)
```

### 4. **Cross-Universe Communication**
- Channels scoped under `air.launch` enable rank-to-rank communication
- Channel attributes (routed_put, gather_reduce) express complex patterns
- Compiler can analyze cross-device data movement

### 5. **Future-Proof for Phases 3 & 4**
- Phase 3 (multi-device lowering): `air.rank` → actual device contexts
- Phase 4 (collective communication): channels → MPI/RCCL/NCCL
- Routing tables → physical network topology mapping

## Required Dialect Extensions

Beyond `air.rank`/`air.universe`, the implementation uses proposed channel extensions:

1. **Routed channels** (line 98-102): `routing_key_field`, `routing_table`, `num_slots` attrs
2. **Gather-reduce channels** (line 104-108): `reduce_op`, `expected_contributions` attrs
3. **Routing tables** (line 110-118): `air.routing_table` op
4. **Extended put/get** (line 173, 229-231): `routing_key`, `reduce`, `contributions` attrs

These would need to be added to the AIR dialect to support the full MoE routing semantics.

## Migration Path from Current Implementation

1. **Phase 1:** Relax verifier to allow `air.rank` in `air.launch`
2. **Phase 2:** Refactor to nested `air.rank` structure (refactored code)
3. **Phase 3:** Add routing table and channel extension ops to dialect
4. **Phase 4:** Lower to actual multi-device execution (per PR #1479 roadmap)

## Summary

The refactored implementation:
- Uses `air.universe` to declare 2 distinct resource pools (attention, FFN)
- Uses nested `air.rank` inside `air.launch` to express concurrent multi-device parallelism
- Uses concurrency tokens to force parallel execution of both clusters
- Uses channels for clean cross-universe and intra-universe communication
- Provides a clean, compiler-analyzable representation of the 8-GPU architecture
- Aligns with the original `air.rank` design intent from issue #1414
