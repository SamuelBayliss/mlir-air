// =============================================================================
// GPT-OSS-120B: 8-GPU single-node decode, mlir-air prototype
// REFACTORED to use air.rank and air.universe
//
// Changes from original:
// - Added air.universe.alloc for attention (2 GPUs) and FFN (6 GPUs) clusters
// - Wrapped each cluster in air.rank inside air.launch
// - Used concurrency token to force parallel execution of both clusters
// - Channels declared under air.launch provide cross-universe communication
// =============================================================================

module @gpt_oss_120b_8gpu {

// ---------------------------------------------------------------------------
// Channel declarations
// Channels are now scoped under the outer air.launch, enabling communication
// between the two concurrent rank universes (attention and FFN)
// ---------------------------------------------------------------------------

// NOTE: These channel declarations would move inside the air.launch body
// or be passed as arguments. Showing the semantic intent here.

// Attention → FFN: broadcast hidden state (1 attn rank → 6 FFN ranks)
// air.channel @hidden_dispatch [2, 6]  // [attn_ranks, ffn_ranks]

// Attention → FFN: routing packets (expert_id, routing_weight)
// air.channel @expert_route_dispatch [2, 6] { ... routed_put attrs ... }

// FFN → Attention: expert outputs (gathered, weighted-summed)
// air.channel @expert_result_gather [6, 2] { ... gather_reduce attrs ... }

// Intra-FFN: peer exchange for tokens with experts on different FFN GPUs
// air.channel @expert_exchange [6, 6]

// KV cache channels
// air.channel @kv_write [2, 1]
// air.channel @kv_read  [1, 2]

// Routing table (maps expert_id → absolute FFN rank within FFN universe)
// air.routing_table @expert_group_table { entries = [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4, 5,5] }

// ---------------------------------------------------------------------------
// Main mega-kernel: single persistent air.launch coordinating two concurrent
// air.rank universes
// ---------------------------------------------------------------------------
func.func @gpt_decode_megakernel(
    %input     : memref<1x2880xf16>,
    %output    : memref<1x2880xf16>,
    %wq        : memref<2880x2880xf8>,
    %wk        : memref<2880x960xf8>,
    %wv        : memref<2880x960xf8>,
    %wo        : memref<2880x2880xf8>,
    %kv_cache  : memref<2x8192x1920xf16>,
    %cache_len : index,
    %w_router  : memref<2880x128xf8>,
    %ew1       : memref<16x2880x7680xf4>,
    %ew2       : memref<16x7680x2880xf4>
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index

  // -------------------------------------------------------------------------
  // Allocate universes for the two clusters
  // -------------------------------------------------------------------------
  %attn_universe = air.universe.alloc(%c2)  // 2 GPUs for attention
  %ffn_universe  = air.universe.alloc(%c6)  // 6 GPUs for FFN/MoE

  // -------------------------------------------------------------------------
  // air.launch: single host-to-accelerator transfer establishing the
  // execution context for both clusters. Channels declared here enable
  // cross-universe communication.
  // -------------------------------------------------------------------------
  air.launch (%lx) in (%ls = %c1)
    args(%a_in   = %input,   %a_out  = %output,
         %a_wq   = %wq,      %a_wk   = %wk,
         %a_wv   = %wv,      %a_wo   = %wo,
         %a_kvc  = %kv_cache, %a_cl  = %cache_len,
         %a_wr   = %w_router,
         %a_ew1  = %ew1,     %a_ew2  = %ew2,
         %u_attn = %attn_universe, %u_ffn = %ffn_universe) :
    memref<1x2880xf16>, memref<1x2880xf16>,
    memref<2880x2880xf8>, memref<2880x960xf8>,
    memref<2880x960xf8>,  memref<2880x2880xf8>,
    memref<2x8192x1920xf16>, index,
    memref<2880x128xf8>,
    memref<16x2880x7680xf4>, memref<16x7680x2880xf4>,
    !air.universe, !air.universe
  {
    // -----------------------------------------------------------------------
    // Channel declarations (scoped to this air.launch)
    // These enable communication between the two concurrent rank universes
    // -----------------------------------------------------------------------
    air.channel @hidden_dispatch [2, 6]

    air.channel @expert_route_dispatch [2, 6] {
      routing_key_field = 1 : i32,
      routing_table = @expert_group_table,
      num_slots = 2 : i32
    }

    air.channel @expert_result_gather [6, 2] {
      reduce_op = "add",
      expected_contributions = 2 : i32
    }

    air.channel @expert_exchange [6, 6]

    air.channel @attn_allreduce [2, 1]  // Intra-attention all-reduce

    air.routing_table @expert_group_table {
      entries = [0, 0, 0,   // experts 0,1,2  → ffn_rank 0
                 1, 1, 1,   // experts 3,4,5  → ffn_rank 1
                 2, 2, 2,   // experts 6,7,8  → ffn_rank 2
                 3, 3, 3,   // experts 9,10,11→ ffn_rank 3
                 4, 4,      // experts 12,13  → ffn_rank 4
                 5, 5]      // experts 14,15  → ffn_rank 5
    }

    // -----------------------------------------------------------------------
    // Attention cluster: 2-way parallelism across GPUs 0,1
    // Each rank instance maps to one physical GPU
    // concurrency token forces this to run in parallel with FFN cluster
    // -----------------------------------------------------------------------
    %tok_attn = air.rank async universe(%u_attn) (%attn_gpu) in (%na = %c2)
      args(%in_a   = %a_in,  %out_a  = %a_out,
           %wq_a   = %a_wq,  %wk_a   = %a_wk,
           %wv_a   = %a_wv,  %wo_a   = %a_wo,
           %kvc_a  = %a_kvc, %cl_a   = %a_cl,
           %wr_a   = %a_wr) :
      memref<1x2880xf16>, memref<1x2880xf16>,
      memref<2880x2880xf8>, memref<2880x960xf8>,
      memref<2880x960xf8>,  memref<2880x2880xf8>,
      memref<2x8192x1920xf16>, index,
      memref<2880x128xf8>
    {
      // LDS scratchpad allocation (each attn GPU works on 1440-element shard)
      %c1440 = arith.constant 1440 : index
      %c2880 = arith.constant 2880 : index
      %c4 = arith.constant 4 : index

      %q_lds  = memref.alloc() : memref<1x1440xf16, 3>
      %k_lds  = memref.alloc() : memref<1x480xf16, 3>
      %v_lds  = memref.alloc() : memref<1x480xf16, 3>
      %o_lds  = memref.alloc() : memref<1x1440xf16, 3>
      %h_lds  = memref.alloc() : memref<1x2880xf16, 3>
      %r_lds  = memref.alloc() : memref<1x128xf32,  3>

      // -------------------------------------------------------------------
      // Segment: QKV + Flash Attention + O-projection
      // -------------------------------------------------------------------
      air.segment @attn_seg
        args(%rank_s = %attn_gpu, %in_s = %in_a,
             %wq_s = %wq_a, %wk_s = %wk_a, %wv_s = %wv_a, %wo_s = %wo_a,
             %kvc_s = %kvc_a, %cl_s = %cl_a,
             %q_s = %q_lds, %k_s = %k_lds, %v_s = %v_lds,
             %o_s = %o_lds, %h_s = %h_lds) :
        index, memref<1x2880xf16>,
        memref<2880x2880xf8>, memref<2880x960xf8>,
        memref<2880x960xf8>,  memref<2880x2880xf8>,
        memref<2x8192x1920xf16>, index,
        memref<1x1440xf16, 3>, memref<1x480xf16, 3>, memref<1x480xf16, 3>,
        memref<1x1440xf16, 3>, memref<1x2880xf16, 3>
      {
        %q_col_off = arith.muli %rank_s, %c1440 : index

        // QKV Projection herds (same as original, omitted for brevity)
        air.herd @qkv_herd tile (%tx, %ty) in (%sx = %c4, %sy = %c4)
          args(...) {
          // ... QKV GEMV computation (unchanged from original) ...
        }

        // Flash Attention herd (same as original, omitted for brevity)
        air.herd @flash_attn_herd tile (%tx, %ty) in (%sx = %c4, %sy = %c1)
          args(...) {
          // ... Flash attention computation (unchanged from original) ...
        }

        // Output Projection + all-reduce
        %o_partial = memref.alloc() : memref<1x2880xf32, 2>
        air.execute {
          linalg.matmul
            ins(%o_s, %wo_s : memref<1x1440xf16, 3>, memref<2880x2880xf8>)
            outs(%o_partial : memref<1x2880xf32, 2>)
        }

        // All-reduce between the 2 attention ranks
        air.channel.put @attn_allreduce[%rank_s, %c0]
          (%o_partial[%c0, %c0][%c1, %c2880][%c1, %c1])
          : memref<1x2880xf32, 2>

        %peer_partial = memref.alloc() : memref<1x2880xf32, 2>
        %peer_rank = arith.xori %rank_s, %c1 : index
        air.channel.get @attn_allreduce[%peer_rank, %c0]
          (%peer_partial[%c0, %c0][%c1, %c2880][%c1, %c1])
          : memref<1x2880xf32, 2>

        // Accumulate + LayerNorm → h_s
        air.execute {
          linalg.add ins(%o_partial, %peer_partial : ...) outs(%h_s : ...)
          // layer_norm_inplace(%h_s)
        }

        memref.dealloc %o_partial   : memref<1x2880xf32, 2>
        memref.dealloc %peer_partial : memref<1x2880xf32, 2>

        air.segment_terminator
      } // end @attn_seg

      // -------------------------------------------------------------------
      // Router: compute logits and select top-2 experts
      // -------------------------------------------------------------------
      air.execute {
        linalg.matmul
          ins(%h_lds, %wr_a : memref<1x2880xf16, 3>, memref<2880x128xf8>)
          outs(%r_lds : memref<1x128xf32, 3>)
        // top-2 selection → %expert_ids[2], %gate_weights[2]
      }

      // -------------------------------------------------------------------
      // Dispatch phase (only rank 0 drives the dispatch)
      // -------------------------------------------------------------------
      %is_dispatch_rank = arith.cmpi eq, %attn_gpu, %c0 : index
      cf.cond_br %is_dispatch_rank, ^dispatch, ^wait_result

    ^dispatch:
      // Broadcast hidden to all 6 FFN ranks
      scf.for %ffn_r = %c0 to %c6 step %c1 {
        air.channel.put @hidden_dispatch[%attn_gpu, %ffn_r]
          (%h_lds[%c0, %c0][%c1, %c2880][%c1, %c1])
          : memref<1x2880xf16, 3>
      }

      // Routed put for each top-k slot
      scf.for %k = %c0 to %c2 step %c1 {
        %pkt = memref.alloc() : memref<3xindex, 3>
        // pkt[0] = token_id, pkt[1] = expert_id, pkt[2] = gate_weight_bits
        // ... store routing packet ...

        // Routed put (channel infrastructure routes based on expert_id)
        air.channel.put @expert_route_dispatch[%attn_gpu, #any]
          (%pkt[%c0][%c3][%c1]) {routing_key = 1 : i32}
          : memref<3xindex, 3>

        memref.dealloc %pkt : memref<3xindex, 3>
      }
      cf.br ^wait_result

    ^wait_result:
      // -------------------------------------------------------------------
      // Gather expert results from FFN cluster
      // -------------------------------------------------------------------
      %expert_result = memref.alloc() : memref<1x2880xf16, 3>
      air.channel.get @expert_result_gather[%attn_gpu, %c0]
        (%expert_result[%c0, %c0][%c1, %c2880][%c1, %c1])
        {reduce = "weighted_add", contributions = 2 : i32}
        : memref<1x2880xf16, 3>

      // Final residual add
      air.execute {
        linalg.add
          ins(%in_a, %h_lds, %expert_result : ...)
          outs(%out_a : memref<1x2880xf16>)
      }

      memref.dealloc %q_lds         : memref<1x1440xf16, 3>
      memref.dealloc %k_lds         : memref<1x480xf16, 3>
      memref.dealloc %v_lds         : memref<1x480xf16, 3>
      memref.dealloc %o_lds         : memref<1x1440xf16, 3>
      memref.dealloc %h_lds         : memref<1x2880xf16, 3>
      memref.dealloc %r_lds         : memref<1x128xf32, 3>
      memref.dealloc %expert_result : memref<1x2880xf16, 3>

      air.rank_terminator
    } // end attention cluster air.rank

    // -----------------------------------------------------------------------
    // FFN cluster: 6-way parallelism across GPUs 2-7
    // Each rank instance maps to one physical GPU
    // concurrency token forces this to run in parallel with attention cluster
    // -----------------------------------------------------------------------
    %tok_ffn = air.rank async universe(%u_ffn) (%ffn_gpu) in (%nf = %c6)
      args(%ew1_f = %a_ew1, %ew2_f = %a_ew2) :
      memref<16x2880x7680xf4>, memref<16x7680x2880xf4>
    {
      %c2880 = arith.constant 2880 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index

      // Expert count per GPU: first 4 GPUs hold 3, last 2 hold 2
      %is_dense = arith.cmpi ult, %ffn_gpu, %c4 : index
      %n_my_experts = arith.select %is_dense, %c3, %c2 : index
      %my_exp_start = arith.muli %ffn_gpu, %c3 : index

      // -------------------------------------------------------------------
      // Stage expert weights into L2 (weight stationary)
      // -------------------------------------------------------------------
      %staged_w1 = memref.alloc() : memref<3x2880x7680xf4, 2>
      %staged_w2 = memref.alloc() : memref<3x7680x2880xf4, 2>

      air.execute {
        air.dma_memcpy_nd
          (%staged_w1[] [] [],
           %ew1_f[%my_exp_start, %c0, %c0] [...] [...]) :
          memref<3x2880x7680xf4, 2>, memref<16x2880x7680xf4>
        air.dma_memcpy_nd
          (%staged_w2[] [] [],
           %ew2_f[%my_exp_start, %c0, %c0] [...] [...]) :
          memref<3x7680x2880xf4, 2>, memref<16x7680x2880xf4>
      }

      // -------------------------------------------------------------------
      // Receive hidden state from attention cluster
      // -------------------------------------------------------------------
      %hidden_buf = memref.alloc() : memref<1x2880xf16, 3>

      // Receive from attention cluster (broadcast)
      // Note: In actual implementation, we'd iterate over both attention ranks
      // or use a gather pattern. Simplified here.
      air.channel.get @hidden_dispatch[%c0, %ffn_gpu]
        (%hidden_buf[%c0, %c0][%c1, %c2880][%c1, %c1])
        : memref<1x2880xf16, 3>

      // -------------------------------------------------------------------
      // Receive route packets and process experts
      // -------------------------------------------------------------------
      %result_buf = memref.alloc() : memref<1x2880xf32, 2>
      air.execute {
        linalg.fill ins(arith.constant 0.0 : f32)
                    outs(%result_buf : memref<1x2880xf32, 2>)
      }

      // Process routed packets (same logic as original, omitted for brevity)
      scf.for %slot = %c0 to %c4 step %c1 {
        // ... receive packets, check ownership, compute expert, accumulate ...
        // (Same as original lines 465-641)
      }

      // -------------------------------------------------------------------
      // Send expert results back to attention cluster
      // -------------------------------------------------------------------
      %result_f16 = memref.alloc() : memref<1x2880xf16, 3>
      air.execute {
        linalg.generic { /* f32→f16 */ }
          ins(%result_buf : memref<1x2880xf32, 2>)
          outs(%result_f16 : memref<1x2880xf16, 3>)
      }

      // Send to attention cluster (indexed by attention rank)
      air.channel.put @expert_result_gather[%ffn_gpu, %c0]
        (%result_f16[%c0, %c0][%c1, %c2880][%c1, %c1])
        {contribution_weight_field = 2 : i32}
        : memref<1x2880xf16, 3>

      memref.dealloc %hidden_buf  : memref<1x2880xf16, 3>
      memref.dealloc %result_buf  : memref<1x2880xf32, 2>
      memref.dealloc %result_f16  : memref<1x2880xf16, 3>

      air.rank_terminator
    } // end FFN cluster air.rank

    // -----------------------------------------------------------------------
    // Both clusters complete (tokens ensure both finish)
    // -----------------------------------------------------------------------
    air.wait_all [%tok_attn, %tok_ffn]

    air.launch_terminator
  } // end air.launch

  return
} // end @gpt_decode_megakernel

} // end module
