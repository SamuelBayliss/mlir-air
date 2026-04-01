# Summary: GPT-OSS-120B Progressive Test Suite

Created: 2026-03-31

## What We've Built

A comprehensive test suite directory for developing and validating the 8-GPU GPT-OSS-120B multi-device implementation using MLIR-AIR's new `air.rank` and `air.universe` primitives.

## Directory Structure

```
programming_examples/gpt-oss-120b/
├── README.md                              # Overview and architecture
├── TEST_PLAN.md                           # Detailed test progression (15 tests)
├── SUMMARY.md                             # This file
│
├── gpt-oss-120b-original.mlir             # Original (problematic) implementation
├── gpt_oss_120b_refactored.mlir           # Refactored using air.rank/universe
├── gpt_oss_120b_refactoring_summary.md    # Technical analysis of refactoring
│
├── test-00-basic-hierarchy.mlir           # ✅ Phase 0: Basic hierarchy
└── test-01-basic-channel.mlir             # ✅ Phase 0: Channel communication
```

## Completed Work

### 1. ✅ PR Comment Posted to #1479

Posted a detailed comment to the merged PR #1479 requesting relaxation of the verifier restriction that prevents `air.rank` from being nested inside `air.launch`.

**Key argument**: `air.launch` represents host→accelerator control transfer, not single-device restriction. Nested `air.rank` should create cross-GPU barriers analogous to how `air.segment` creates tile resource requirements.

**Comment URL**: https://github.com/Xilinx/mlir-air/pull/1479#issuecomment-4167467977

### 2. ✅ Original Implementation Analysis

**File**: `gpt-oss-120b-original.mlir` (34 KB)

**Issues identified**:
- Uses single `air.launch (%rank) in (%universe_size = %c8)` incorrectly
- Manual branching via `cf.cond_br` based on rank index
- No explicit multi-device topology in IR
- No use of `air.rank` or `air.universe`

### 3. ✅ Refactored Implementation

**File**: `gpt_oss_120b_refactored.mlir` (16 KB)

**Key design decisions**:
```mlir
// Two separate universes
%attn_universe = air.universe.alloc(%c2)  // 2 GPUs for attention
%ffn_universe  = air.universe.alloc(%c6)  // 6 GPUs for FFN

air.launch (%lx) in (%ls = %c1) {
  // Channels scoped here enable cross-universe communication
  air.channel @hidden_dispatch [2, 6]
  air.channel @expert_result_gather [6, 2]

  // Concurrent execution via async tokens
  %tok_attn = air.rank async universe(%attn_universe) (%attn_gpu) in (%na = %c2) {
    air.segment @attn_seg { /* QKV, Flash Attention, Router */ }
  }

  %tok_ffn = air.rank async universe(%ffn_universe) (%ffn_gpu) in (%nf = %c6) {
    air.segment @expert_seg { /* Weight-stationary MoE */ }
  }

  // Ensure both clusters complete
  air.wait_all [%tok_attn, %tok_ffn]
}
```

**Advantages**:
- Explicit multi-device topology
- Declarative parallelism (async tokens, not control flow)
- Compiler-analyzable structure
- Clean semantic hierarchy
- Cross-universe communication via channels

### 4. ✅ Refactoring Analysis

**File**: `gpt_oss_120b_refactoring_summary.md` (7.4 KB)

Comprehensive technical analysis including:
- Design rationale for nested air.rank
- Index mapping (rank indices → GPU IDs)
- Communication patterns (broadcast, gather, all-reduce)
- Advantages over original implementation
- Required dialect extensions
- Migration path

### 5. ✅ Progressive Test Plan

**File**: `TEST_PLAN.md` (11 KB)

Detailed plan for 15 progressive tests across 5 phases:

**Phase 0** (Baseline): Tests 00-01 ✅ **COMPLETED**
- No `air.rank`, establish basic hierarchy and channels

**Phase 1** (Simple rank): Tests 02-04 ⬜ TODO
- Top-level `air.rank` (currently supported by PR #1479)

**Phase 2** (Nested rank): Tests 05-07 ⬜ TODO (BLOCKED)
- `air.rank` inside `air.launch` (requires verifier relaxation)
- **Test 07** is the key test: concurrent clusters with async tokens

**Phase 3** (Communication): Tests 08-10 ⬜ TODO (BLOCKED)
- Cross-rank channels (broadcast, gather, all-reduce)
- Depends on Phase 2

**Phase 4** (Full features): Tests 11-15 ⬜ TODO (BLOCKED)
- Routed channels (MoE dispatch)
- Gather-reduce channels (weighted accumulation)
- Full 8-GPU implementation

### 6. ✅ Phase 0 Test Implementations

#### Test 00: Basic Hierarchy (`test-00-basic-hierarchy.mlir`) - 5.2 KB

Demonstrates the three-level hierarchy without multi-device:
- `air.launch` → single device context
- `air.segment` → L2 memory tiling
- `air.herd` → 4×4 compute unit grid
- Tiled GEMV operation (representative of QKV projection)
- DMA operations across memory hierarchy (DDR → L2 → LDS)

**Purpose**: Baseline before introducing `air.rank`

#### Test 01: Basic Channel Communication (`test-01-basic-channel.mlir`) - 6.2 KB

Demonstrates channel-based communication:
- Producer segment → channel → consumer segment
- Both synchronous and asynchronous variants
- Establishes channel patterns for later cross-rank communication

**Purpose**: Channel fundamentals before cross-universe usage

## Current Status

### ✅ Completed
1. Refactored implementation using `air.rank`/`air.universe`
2. Technical analysis and documentation
3. PR comment requesting verifier relaxation
4. Complete test plan (15 tests)
5. Phase 0 tests (2/2 completed)

### ⬜ TODO - Phase 1 (Not Blocked)
- Test 02: Top-level 1D rank
- Test 03: Top-level 2D rank
- Test 04: Top-level rank with async deps

These can be implemented immediately using current PR #1479 functionality.

### 🚫 BLOCKED - Phases 2-4
**Blocker**: Verifier restriction in PR #1479

**Tests blocked**: 05-15 (11 tests)

**Required action**: Maintainer approval to relax verifier

**Workaround**: None for nested `air.rank` tests; they fundamentally require the verifier change

## Next Steps

### Immediate (Can do now)
1. Implement Phase 1 tests (02-04) using top-level `air.rank`
2. Validate they work with `air-rank-to-launch` serialization pass
3. Document any issues or edge cases

### Waiting on Verifier Relaxation
1. Monitor PR #1479 for responses to comment
2. If approved, implement Phase 2 tests (05-07)
3. Test 07 (concurrent clusters) is **critical** for GPT-OSS-120B

### Waiting on Dialect Extensions
1. File issues for required channel extensions:
   - Routed channels (`routing_key_field`, `routing_table`)
   - Gather-reduce channels (`reduce_op`, `expected_contributions`)
   - `air.routing_table` op
2. Once implemented, complete Phase 4 tests (11-15)

## Key Architectural Insights

### 1. Nested air.rank is Essential

The original design in issue #1414 intended nested `air.rank` for multi-GPU within a single host context. The PR #1479 implementation simplified to top-level only, but this limits expressiveness.

**Why nested is better**:
- Single host-to-accelerator transition (preserves mega-kernel semantics)
- Concurrent execution via async tokens
- Cross-universe communication via scoped channels
- Clean mapping: `air.launch` = context, `air.rank` = devices

### 2. Async Tokens Force Concurrency

Critical insight: Without async tokens, two `air.rank` ops would execute serially.

```mlir
// WRONG - serial execution
air.rank (%r1) in (%n1 = %c2) { ... }  // Runs first
air.rank (%r2) in (%n2 = %c6) { ... }  // Runs second

// CORRECT - concurrent execution
%t1 = air.rank async (%r1) in (%n1 = %c2) { ... }  // Both run
%t2 = air.rank async (%r2) in (%n2 = %c6) { ... }  // in parallel
air.wait_all [%t1, %t2]  // Join point
```

This is essential for the attention+FFN cluster architecture.

### 3. Channels Enable Cross-Universe Communication

Scoping channels under `air.launch` (rather than module-level) allows them to be visible to both rank universes while maintaining encapsulation.

```mlir
air.launch {
  air.channel @cross_universe [2, 6]  // Visible to both ranks below

  air.rank universe(%u1) ... {
    air.channel.put @cross_universe[...] (...)
  }

  air.rank universe(%u2) ... {
    air.channel.get @cross_universe[...] (...)
  }
}
```

## References

- **Issue #1414**: "New op: air.rank — multi-device / multi-host parallelism"
- **PR #1479**: "Add air.rank hierarchy op for multi-device parallelism" (MERGED)
- **PR #1479 Comment**: https://github.com/Xilinx/mlir-air/pull/1479#issuecomment-4167467977
- **Design Doc**: `docs/AIRRankOp.md` (branch `compute-model-explanation`)

## Contact

For questions about this test suite:
- See `TEST_PLAN.md` for detailed test specifications
- See `gpt_oss_120b_refactoring_summary.md` for technical rationale
- See `README.md` for architecture overview
