# GPT-OSS-120B: Multi-GPU Decode Implementation

This directory contains MLIR-AIR implementations of an 8-GPU single-node decode kernel for GPT-OSS-120B, demonstrating multi-device parallelism using `air.rank` and `air.universe`.

## Model Configuration

- **Model**: GPT-OSS-120B (Mixture of Experts)
- **Hidden size**: 2880
- **Num heads**: 24 (head_dim = 120)
- **Num KV heads**: 8 (GQA)
- **Num experts**: 128 (top-2 routing, 16 staged in prototype)
- **FFN intermediate**: 3840 (SwiGLU)
- **Quantization**: MXFP4 weights, f16 activations

## Architecture

### 8-GPU Assignment
- **Attention cluster** (GPUs 0-1): 2-way tensor parallelism, 12 heads each
- **FFN/MoE cluster** (GPUs 2-7): 6-way expert parallelism, weight-stationary

### Expert Distribution (16 staged experts prototype)
```
GPU 2 (ffn_rank 0) → experts [0, 1, 2]
GPU 3 (ffn_rank 1) → experts [3, 4, 5]
GPU 4 (ffn_rank 2) → experts [6, 7, 8]
GPU 5 (ffn_rank 3) → experts [9, 10, 11]
GPU 6 (ffn_rank 4) → experts [12, 13]
GPU 7 (ffn_rank 5) → experts [14, 15]
```

## Files

### Core Implementations

1. **`gpt-oss-120b-original.mlir`** (ORIGINAL)
   - Initial prototype using single `air.launch` with manual GPU branching
   - Does NOT use `air.rank` or `air.universe`
   - Illustrates the problem: multi-device topology is implicit in control flow

2. **`gpt_oss_120b_refactored.mlir`** (REFACTORED)
   - Uses `air.rank` and `air.universe` to express multi-device structure
   - Two concurrent rank universes (attention + FFN) inside `air.launch`
   - Async tokens force parallel execution
   - Channels enable cross-universe communication
   - **Requires**: Relaxed verifier to allow `air.rank` nested in `air.launch`

### Documentation

3. **`gpt_oss_120b_refactoring_summary.md`**
   - Detailed analysis of refactoring decisions
   - Comparison of original vs. refactored approaches
   - Rationale for nested `air.rank` design
   - Communication patterns and index mappings

## Key Design Patterns

### Nested air.rank for Concurrent Clusters

```mlir
%attn_universe = air.universe.alloc(%c2)  // 2 GPUs
%ffn_universe  = air.universe.alloc(%c6)  // 6 GPUs

air.launch (%lx) in (%ls = %c1) {
  // Attention cluster (concurrent)
  %tok_attn = air.rank async universe(%attn_universe) (%attn_gpu) in (%na = %c2) {
    air.segment @attn_seg { /* QKV, Flash Attention, Router */ }
  }

  // FFN cluster (concurrent)
  %tok_ffn = air.rank async universe(%ffn_universe) (%ffn_gpu) in (%nf = %c6) {
    air.segment @expert_seg { /* Weight-stationary MoE */ }
  }

  // Force concurrent execution
  air.wait_all [%tok_attn, %tok_ffn]
}
```

### Cross-Universe Communication via Channels

```mlir
air.launch (...) {
  air.channel @hidden_dispatch [2, 6]       // attn → FFN broadcast
  air.channel @expert_result_gather [6, 2]  // FFN → attn gather-reduce

  air.rank async universe(%attn_universe) ... {
    air.channel.put @hidden_dispatch[%attn_gpu, %ffn_r] (...)
  }

  air.rank async universe(%ffn_universe) ... {
    air.channel.get @hidden_dispatch[%attn_r, %ffn_gpu] (...)
  }
}
```

## Progressive Test Plan

See **`TEST_PLAN.md`** for the complete progressive test strategy.

### Test Files

**Phase 0: Baseline (No air.rank)**
- ✅ `test-00-basic-hierarchy.mlir` - Basic launch → segment → herd hierarchy
- ✅ `test-01-basic-channel.mlir` - Channel communication between segments

**Phase 1: Simple air.rank (Top-level)**
- ⬜ `test-02-toplevel-rank-1d.mlir` - Simple 1D rank at module top
- ⬜ `test-03-toplevel-rank-2d.mlir` - Cartesian 2D rank grid
- ⬜ `test-04-toplevel-rank-async.mlir` - Async dependencies between ranks

**Phase 2: Nested air.rank (Requires verifier relaxation)**
- ⬜ `test-05-nested-rank-single.mlir` - Single rank inside launch
- ⬜ `test-06-nested-rank-sequential.mlir` - Two sequential ranks in launch
- ⬜ `test-07-nested-rank-concurrent.mlir` - Two concurrent ranks (key test!)

**Phase 3: Cross-rank Communication**
- ⬜ `test-08-cross-rank-broadcast.mlir` - 1-to-N broadcast pattern
- ⬜ `test-09-cross-rank-gather.mlir` - N-to-1 gather pattern
- ⬜ `test-10-intra-rank-allreduce.mlir` - Peer-to-peer within universe

**Phase 4: Full Features**
- ⬜ `test-11-routed-channel.mlir` - MoE-style content routing
- ⬜ `test-12-gather-reduce-channel.mlir` - Weighted accumulation
- ⬜ `test-13-multi-segment-rank.mlir` - Complex rank body
- ⬜ `test-14-simplified-two-cluster.mlir` - Minimal attention+FFN
- ⬜ `test-15-full-gpt.mlir` - Complete 8-GPU (same as refactored)

## Required AIR Dialect Features

### Phase 1 (Available in PR #1479)
- ✅ `!air.universe` type
- ✅ `air.universe.alloc` op
- ✅ `air.rank` op with `HierarchyInterface`
- ✅ `air.rank_terminator`
- ⚠️ Verifier restriction (needs relaxation)

### Phase 1+ (Proposed Extensions)
- ❌ Nested `air.rank` inside `air.launch` (requires verifier change)
- ❌ Routed channels: `routing_key_field`, `routing_table` attrs
- ❌ Gather-reduce channels: `reduce_op`, `expected_contributions` attrs
- ❌ `air.routing_table` op
- ❌ Extended put/get: `routing_key`, `reduce`, `contributions` attrs

### Phase 3+ (Future)
- ❌ Multi-device lowering: `air.rank` → actual device contexts
- ❌ Collective communication: channels → MPI/RCCL/NCCL

## References

- Issue #1414: "New op: air.rank — multi-device / multi-host parallelism"
- PR #1479: "Add air.rank hierarchy op for multi-device parallelism" (MERGED)
- Design doc: `docs/AIRRankOp.md` (branch `compute-model-explanation`)
