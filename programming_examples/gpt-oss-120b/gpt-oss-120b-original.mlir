// =============================================================================
// GPT-OSS-120B: 8-GPU single-node decode, mlir-air prototype
//
// Model config (from gpt-oss-120b):
//   hidden_size      = 2880
//   num_heads        = 24 (head_dim = 120)
//   num_kv_heads     = 8  (GQA; kv_head_dim = 120)
//   num_experts      = 128 (top-2 routing)
//   experts_staged   = 16  (prototype; held in FFN cluster L2/LDS)
//   ffn_intermediate = 3840 (SwiGLU gate+up interleaved → w1 is [2880 x 7680])
//   quant            = MXFP4 (weights) / f16 (activations / KV cache)
//
// GPU assignment:
//   rank 0, 1  → Attention cluster  (12 heads each, TP=2)
//   rank 2..7  → FFN/MoE cluster    (6 GPUs, ~2-3 experts each)
//
// Expert layout across FFN cluster (16 experts / 6 GPUs):
//   rank 2 → experts [0, 1, 2]
//   rank 3 → experts [3, 4, 5]
//   rank 4 → experts [6, 7, 8]
//   rank 5 → experts [9,10,11]
//   rank 6 → experts [12,13]
//   rank 7 → experts [14,15]
//
// NOTE: "staged expert id" is expert_id % 16 (mod into the 16-slot prototype).
//       In the real system the FFN cluster is larger and holds all 128 experts.
// =============================================================================

module @gpt_oss_120b_8gpu {

// ---------------------------------------------------------------------------
// Type aliases (for readability)
// ---------------------------------------------------------------------------
// #hidden_f16   = memref<1x2880xf16>
// #router_out   = memref<1x128xf32>        -- full router logits
// #route_pkt    = memref<3xindex>           -- [token_id, expert_id, gate_score_bits]
// #expert_w1    = memref<16x2880x7680xf4>  -- MXFP4, staged in DDR, L2-buffered
// #expert_w2    = memref<16x7680x2880xf4>

// ---------------------------------------------------------------------------
// Channel declarations
//
// Proposed extended semantics (marked with †):
//   air.channel with #routed_put attr:
//     put carries a routing key (expert_id); the channel infrastructure
//     dispatches to the rank in @expert_group whose range contains that key.
//     This models the MoE routing fabric — in real HW this becomes a
//     credit-based network transaction.
//
//   air.channel with #gather_reduce attr:
//     get aggregates contributions from multiple sources with a reduce op.
//     Models the output accumulation across top-k expert results.
// ---------------------------------------------------------------------------

// Attention → FFN: broadcast hidden state (1 source → 6 FFN GPUs)
// Each of the 2 attn GPUs can originate; FFN cluster receives whichever holds
// the majority attention shard (after all-reduce on attn side).
air.channel @hidden_dispatch [1, 6]

// Attention → FFN: routing packet (expert_id, routing_weight) per top-k slot
// † routed_put: the channel inspects payload[1] (expert_id) to select target
// This is the key new semantic — put is not rank-addressed but key-addressed.
air.channel @expert_route_dispatch [1, 6] {
  routing_key_field = 1 : i32,           // field index in route_pkt that holds expert_id
  routing_table = @expert_group_table,   // maps expert_id → ffn_rank
  num_slots = 2 : i32                    // top-k = 2 in-flight route pkts
}

// FFN → Attention: expert outputs (gathered, weighted-summed on attn side)
// † gather_reduce: attention GPU accumulates weighted results from up to k=2 sources
air.channel @expert_result_gather [6, 1] {
  reduce_op = "add",
  expected_contributions = 2 : i32       // top-k = 2 contributions expected
}

// Intra-FFN: peer exchange for tokens whose 2 experts land on different FFN GPUs
// Full [6,6] mesh; only off-diagonal entries are live for any given token
air.channel @expert_exchange [6, 6]

// KV cache: written by attention cluster, read back next iteration
// (Could be a channel or a direct shared memref; using channel to model
// the case where KV cache eventually lives in a dedicated memory partition)
air.channel @kv_write [2, 1]
air.channel @kv_read  [1, 2]

// ---------------------------------------------------------------------------
// Routing table (static for prototype; 16 experts → ffn ranks 2..7)
// expert_group_table[i] = absolute GPU rank that holds expert i
// ---------------------------------------------------------------------------
air.routing_table @expert_group_table {
  entries = [2, 2, 2,   // experts 0,1,2  → rank 2
             3, 3, 3,   // experts 3,4,5  → rank 3
             4, 4, 4,   // experts 6,7,8  → rank 4
             5, 5, 5,   // experts 9,10,11→ rank 5
             6, 6,      // experts 12,13  → rank 6
             7, 7]      // experts 14,15  → rank 7
}

// ---------------------------------------------------------------------------
// Main mega-kernel: single persistent launch across all 8 GPUs.
// No re-launch per layer — channel synchronisation drives the pipeline.
// In a real serving loop this would wrap a scf.while over decode steps.
// ---------------------------------------------------------------------------
func.func @gpt_decode_megakernel(
    // Activations
    %input     : memref<1x2880xf16>,         // current token hidden state
    %output    : memref<1x2880xf16>,
    // Attention weights (sharded; each GPU sees the full tensor, slices by rank)
    %wq        : memref<2880x2880xf8>,       // Q projection
    %wk        : memref<2880x960xf8>,        // K (GQA: 8 kv_heads → 960)
    %wv        : memref<2880x960xf8>,        // V
    %wo        : memref<2880x2880xf8>,       // output projection
    // KV cache (full; attention cluster owns this)
    %kv_cache  : memref<2x8192x1920xf16>,   // [kv, seq, kv_hidden]
    %cache_len : index,                       // current kv sequence length
    // Router weights (on both clusters for low-latency; attention computes logits)
    %w_router  : memref<2880x128xf8>,
    // Expert weights (all 16 staged experts; FFN cluster indexes into this)
    %ew1       : memref<16x2880x7680xf4>,   // gate+up interleaved, MXFP4
    %ew2       : memref<16x7680x2880xf4>    // down projection, MXFP4
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c2880 = arith.constant 2880 : index
  %c1440 = arith.constant 1440 : index   // hidden / 2 (attn TP shard)
  %c3840 = arith.constant 3840 : index   // ffn intermediate per expert

  // -------------------------------------------------------------------------
  // air.launch: one logical kernel across 8 GPUs (the universe)
  // air.rank gives each GPU its identity; universe_size = 8
  // -------------------------------------------------------------------------
  air.launch (%rank) in (%universe_size = %c8)
    args(%a_in   = %input,   %a_out  = %output,
         %a_wq   = %wq,      %a_wk   = %wk,
         %a_wv   = %wv,      %a_wo   = %wo,
         %a_kvc  = %kv_cache, %a_cl  = %cache_len,
         %a_wr   = %w_router,
         %a_ew1  = %ew1,     %a_ew2  = %ew2) :
    memref<1x2880xf16>, memref<1x2880xf16>,
    memref<2880x2880xf8>, memref<2880x960xf8>,
    memref<2880x960xf8>,  memref<2880x2880xf8>,
    memref<2x8192x1920xf16>, index,
    memref<2880x128xf8>,
    memref<16x2880x7680xf4>, memref<16x7680x2880xf4>
  {
    // Branch on partition membership
    %is_attn = arith.cmpi ult, %rank, %c2 : index
    cf.cond_br %is_attn, ^attention_cluster, ^ffn_cluster

    // =========================================================================
    // ^attention_cluster  (ranks 0, 1)
    // Responsibilities:
    //   1. QKV projection (TP across 2 GPUs, each owns 12 heads)
    //   2. Flash attention (decode: single query, attend to KV cache)
    //   3. Output projection + all-reduce with peer attn GPU
    //   4. LayerNorm
    //   5. Router logit computation → top-2 expert selection
    //   6. Dispatch: hidden state + route pkts to FFN cluster
    //   7. Receive + weighted accumulate expert results
    //   8. Final residual add → output
    // =========================================================================
    ^attention_cluster:

      // -----------------------------------------------------------------------
      // LDS scratchpad allocation (address space 3)
      // Each attn GPU works on its 1440-element (12-head) shard
      // -----------------------------------------------------------------------
      %q_lds  = memref.alloc() : memref<1x1440xf16, 3>
      %k_lds  = memref.alloc() : memref<1x480xf16, 3>    // GQA: 4 kv_heads per GPU
      %v_lds  = memref.alloc() : memref<1x480xf16, 3>
      %o_lds  = memref.alloc() : memref<1x1440xf16, 3>   // attn output shard
      %h_lds  = memref.alloc() : memref<1x2880xf16, 3>   // full hidden after O-proj + allreduce
      %r_lds  = memref.alloc() : memref<1x128xf32,  3>   // router logits

      // -----------------------------------------------------------------------
      // Segment: QKV + Flash Attention + O-projection
      // -----------------------------------------------------------------------
      %tok_attn = air.segment @attn_seg
        args(%rank_s = %rank, %in_s = %a_in,
             %wq_s = %a_wq, %wk_s = %a_wk, %wv_s = %a_wv, %wo_s = %a_wo,
             %kvc_s = %a_kvc, %cl_s = %a_cl,
             %q_s = %q_lds, %k_s = %k_lds, %v_s = %v_lds,
             %o_s = %o_lds, %h_s = %h_lds) :
        index, memref<1x2880xf16>,
        memref<2880x2880xf8>, memref<2880x960xf8>,
        memref<2880x960xf8>,  memref<2880x2880xf8>,
        memref<2x8192x1920xf16>, index,
        memref<1x1440xf16, 3>, memref<1x480xf16, 3>, memref<1x480xf16, 3>,
        memref<1x1440xf16, 3>, memref<1x2880xf16, 3>
      {
        // Each GPU's Q shard offset: rank 0 → [0,1440), rank 1 → [1440,2880)
        %q_col_off = arith.muli %rank_s, %c1440 : index

        // --- QKV Projection (GEMV in decode) ---
        // Herd of 4x4 = 16 CUs; each handles a 90-element Q output tile
        // (1440 / 16 CUs = 90 elements each)
        air.herd @qkv_herd tile (%tx, %ty) in (%sx = %c4, %sy = %c4)
          args(%in_h = %in_s, %wq_h = %wq_s, %wk_h = %wk_s, %wv_h = %wv_s,
               %q_h = %q_s, %k_h = %k_s, %v_h = %v_s,
               %qoff = %q_col_off, %rank_h = %rank_s) :
          memref<1x2880xf16>, memref<2880x2880xf8>,
          memref<2880x960xf8>, memref<2880x960xf8>,
          memref<1x1440xf16, 3>, memref<1x480xf16, 3>, memref<1x480xf16, 3>,
          index, index
        {
          // Q tile: CU (%tx,%ty) computes output cols [base, base+90)
          %cu_id   = arith.addi (arith.muli %ty, %c4 : index), %tx : index
          %tile_sz = arith.constant 90 : index
          %q_base  = arith.muli %cu_id, %tile_sz : index

          // L2 accumulation buffer for Q tile (f32 accumulator)
          %acc_q = memref.alloc() : memref<1x90xf32, 2>

          // GEMV tiled: input[1x2880] × wq[2880x2880][qoff:qoff+1440, q_base:q_base+90]
          // (In practice this is a tiled loop over the K=2880 reduction dim
          //  with double-buffered DMA from DDR weight → L2 → LDS)
          air.execute {
            // inner K-loop: load weight tile into LDS, mac into L2 accumulator
            scf.for %ki = %c0 to %c2880 step arith.constant 128 : index {
              %wq_tile = memref.alloc() : memref<128x90xf8, 3>   // LDS weight tile
              // DMA: wq_s[ki:ki+128, qoff+q_base:qoff+q_base+90] → wq_tile
              air.dma_memcpy_nd (%wq_tile[] [] [],
                                  %wq_s[%ki, arith.addi %qoff %q_base : index]
                                       [arith.constant 128 : index, %tile_sz]
                                       [%c2880, %c1]) :
                memref<128x90xf8, 3>, memref<2880x2880xf8>
              // MAC
              linalg.matmul
                ins(%in_h   : memref<1x2880xf16>,    // (uses row subview)
                    %wq_tile : memref<128x90xf8, 3>)
                outs(%acc_q  : memref<1x90xf32, 2>)
              memref.dealloc %wq_tile : memref<128x90xf8, 3>
            }
          }
          // Downcast and store Q tile into LDS
          air.execute {
            linalg.generic { /* f32→f16 element-wise */ }
              ins(%acc_q : memref<1x90xf32, 2>)
              outs(%q_h[%c0, %q_base][%c1, %tile_sz][%c1, %c1] : memref<1x1440xf16, 3>)
          }
          // K, V: similar GEMV (abbreviated — GQA means smaller output)
          // ... (analogous tiled GEMV for wk, wv) ...

          memref.dealloc %acc_q : memref<1x90xf32, 2>
        } // end @qkv_herd

        // --- Flash Attention (decode: single query against full KV cache) ---
        // KV cache lives in DDR; we stream K/V tiles into L2
        air.herd @flash_attn_herd tile (%tx, %ty) in (%sx = %c4, %sy = %c1)
          args(%q_h = %q_s, %kvc_h = %kvc_s, %cl_h = %cl_s,
               %o_h = %o_s, %rank_h = %rank_s) :
          memref<1x1440xf16, 3>, memref<2x8192x1920xf16>, index,
          memref<1x1440xf16, 3>, index
        {
          // Each CU handles a 4-head block of the 12-head shard
          // (12 heads / 4 CUs along x = 3 heads each, head_dim=120 → 360 elements)
          %head_tile_sz = arith.constant 360 : index
          %head_base    = arith.muli %tx, %head_tile_sz : index
          %kv_tile_len  = arith.constant 64 : index   // KV streaming tile

          %acc_out  = memref.alloc() : memref<1x360xf32, 2>  // online softmax acc (o)
          %acc_m    = memref.alloc() : memref<1x1xf32, 2>     // running max (m)
          %acc_l    = memref.alloc() : memref<1x1xf32, 2>     // running sum (l)

          // Online softmax loop over KV cache tiles
          air.execute {
            scf.for %kv_off = %c0 to %cl_h step %kv_tile_len {
              %k_tile = memref.alloc() : memref<64x120xf16, 3>
              %v_tile = memref.alloc() : memref<64x120xf16, 3>
              // DMA K/V tiles (GQA: each kv_head serves 3 q_heads on this GPU)
              air.dma_memcpy_nd (%k_tile[] [] [], %kvc_h[%c0, %kv_off, %head_base] ...)
              air.dma_memcpy_nd (%v_tile[] [] [], %kvc_h[%c1, %kv_off, %head_base] ...)
              // S = Q_tile × K_tile^T,  online-softmax update,  O += P × V_tile
              // (abbreviated; standard flash-attn-2 decode kernel)
              // ...
              memref.dealloc %k_tile : memref<64x120xf16, 3>
              memref.dealloc %v_tile : memref<64x120xf16, 3>
            }
            // Final rescale: o_h[head_base:head_base+360] = acc_out / l
          }
          memref.dealloc %acc_out : memref<1x360xf32, 2>
          memref.dealloc %acc_m   : memref<1x1xf32, 2>
          memref.dealloc %acc_l   : memref<1x1xf32, 2>
        } // end @flash_attn_herd

        // --- Output Projection (O: [1x1440] × [1440x2880]) ---
        // Result is a 2880-wide partial; all-reduce across the 2 attn GPUs
        // This is the one intra-attn collective (could be a peer channel or HW link)
        %o_partial = memref.alloc() : memref<1x2880xf32, 2>
        air.execute {
          linalg.matmul
            ins(%o_s, %wo_s : memref<1x1440xf16, 3>, memref<2880x2880xf8>)
            outs(%o_partial   : memref<1x2880xf32, 2>)
        }
        // All-reduce between rank 0 and rank 1 (peer put/get over xgmi)
        // For simplicity shown as a channel; in practice this is an atomic op
        air.channel @attn_allreduce [2, 1]
        air.channel.put @attn_allreduce[%rank_s, %c0]
          (%o_partial[%c0, %c0][%c1, %c2880][%c1, %c1]) : memref<1x2880xf32, 2>
        %peer_partial = memref.alloc() : memref<1x2880xf32, 2>
        air.channel.get @attn_allreduce[arith.xori %rank_s, %c1 : index, %c0]
          (%peer_partial[%c0, %c0][%c1, %c2880][%c1, %c1]) : memref<1x2880xf32, 2>
        // Accumulate + LayerNorm + downcast → h_s (full hidden in LDS)
        air.execute {
          linalg.add ins(%o_partial, %peer_partial : ...) outs(%h_s : ...)
          // layer_norm_inplace(%h_s) — abbreviated
        }
        memref.dealloc %o_partial   : memref<1x2880xf32, 2>
        memref.dealloc %peer_partial : memref<1x2880xf32, 2>

        air.segment_terminator
      } // end @attn_seg

      // -----------------------------------------------------------------------
      // Router: hidden[1x2880] × w_router[2880x128] → logits[1x128]
      // top-2 argmax → (expert_id_0, expert_id_1, weight_0, weight_1)
      // Mod expert ids into [0,16) for the prototype
      // -----------------------------------------------------------------------
      %tok_router = air.execute [%tok_attn] {
        linalg.matmul
          ins(%h_lds, %a_wr : memref<1x2880xf16, 3>, memref<2880x128xf8>)
          outs(%r_lds        : memref<1x128xf32, 3>)
        // top-2 selection → %expert_ids[2], %gate_weights[2]
        // (abbreviated; argpartition over 128 floats)
      }

      // -----------------------------------------------------------------------
      // Dispatch phase:
      //   1. Broadcast hidden state to all 6 FFN GPUs
      //   2. Send a routed packet per top-k slot
      //      † The @expert_route_dispatch channel uses the expert_id field to
      //        select which FFN rank receives each packet — the put is NOT
      //        rank-addressed, it is key-addressed.
      // -----------------------------------------------------------------------

      // Only rank 0 drives the dispatch (after the all-reduce, both have h_lds)
      %is_dispatch_rank = arith.cmpi eq, %rank, %c0 : index
      cf.cond_br %is_dispatch_rank, ^dispatch, ^wait_result

    ^dispatch:
      // Broadcast hidden to all 6 FFN GPUs
      %dispatch_tokens:6 = scf.for %ffn_r = %c0 to %c6 step %c1
          iter_args() -> () {
        air.channel.put @hidden_dispatch[%c0, %ffn_r]
          (%h_lds[%c0, %c0][%c1, %c2880][%c1, %c1]) : memref<1x2880xf16, 3>
      }

      // Routed put for each top-k slot
      // route_pkt = [token_id=0, expert_id_k, gate_weight_bits]
      scf.for %k = %c0 to arith.constant 2 : index step %c1 {
        %pkt = memref.alloc() : memref<3xindex, 3>
        memref.store %c0,        %pkt[%c0] : memref<3xindex, 3>  // token_id
        memref.store %expert_ids[%k], %pkt[%c1] : ...             // expert_id (routing key)
        memref.store %gate_bits[%k],  %pkt[%c2] : ...             // softmax weight (bitcast)

        // † Routed put: channel infrastructure reads pkt[1], looks up
        //   @expert_group_table, and routes to the corresponding FFN rank.
        //   The sending GPU does not need to know which rank owns the expert.
        air.channel.put @expert_route_dispatch[%c0, #any]
          (%pkt[%c0][%c3][%c1]) {routing_key = 1 : i32} : memref<3xindex, 3>

        memref.dealloc %pkt : memref<3xindex, 3>
      }
      cf.br ^wait_result

    ^wait_result:
      // -----------------------------------------------------------------------
      // Gather expert results:
      // † @expert_result_gather has gather_reduce semantics:
      //   the get blocks until expected_contributions (=2) have arrived,
      //   then returns their weighted sum in the output buffer.
      //   The weight applied to each contribution is carried in the route pkt.
      // -----------------------------------------------------------------------
      %expert_result = memref.alloc() : memref<1x2880xf16, 3>
      %tok_gather = air.channel.get @expert_result_gather[%c0, %c0]
        (%expert_result[%c0, %c0][%c1, %c2880][%c1, %c1])
        {reduce = "weighted_add", contributions = 2 : i32}
        : memref<1x2880xf16, 3>

      // Final residual add: output = input + attn_output + ffn_output
      air.execute [%tok_gather] {
        linalg.add
          ins(%a_in, %h_lds, %expert_result :
              memref<1x2880xf16>, memref<1x2880xf16, 3>, memref<1x2880xf16, 3>)
          outs(%a_out : memref<1x2880xf16>)
      }

      memref.dealloc %q_lds         : memref<1x1440xf16, 3>
      memref.dealloc %k_lds         : memref<1x480xf16, 3>
      memref.dealloc %v_lds         : memref<1x480xf16, 3>
      memref.dealloc %o_lds         : memref<1x1440xf16, 3>
      memref.dealloc %h_lds         : memref<1x2880xf16, 3>
      memref.dealloc %r_lds         : memref<1x128xf32, 3>
      memref.dealloc %expert_result : memref<1x2880xf16, 3>

      cf.br ^end

    // =========================================================================
    // ^ffn_cluster  (ranks 2..7)
    // Responsibilities:
    //   1. Stage expert weights into L2 (weight stationary — done once)
    //   2. Receive hidden state from attention cluster
    //   3. Receive routed packets; execute expert if local, else forward
    //   4. Send weighted output to @expert_result_gather
    // =========================================================================
    ^ffn_cluster:

      %ffn_rank      = arith.subi %rank, %c2 : index   // local [0,6)
      // Expert count per GPU: first 4 GPUs hold 3, last 2 hold 2
      %c3            = arith.constant 3 : index
      %c4            = arith.constant 4 : index
      %is_dense      = arith.cmpi ult, %ffn_rank, %c4 : index
      %n_my_experts  = arith.select %is_dense, %c3, %c2 : index
      %my_exp_start  = arith.muli %ffn_rank, %c3 : index   // simplified; last 2 GPUs offset by -2

      // -----------------------------------------------------------------------
      // Stage expert weights into L2 (weight stationary)
      // This happens once at model load, not per-token.
      // In the mega-kernel the segment persists; the herd below runs at init.
      // -----------------------------------------------------------------------
      // L2 buffer for up to 3 experts (over-allocated; last 2 GPUs use only 2 slots)
      %staged_w1 = memref.alloc() : memref<3x2880x7680xf4, 2>   // L2 (address space 2)
      %staged_w2 = memref.alloc() : memref<3x7680x2880xf4, 2>

      air.execute {
        // DMA expert weight slices from DDR (a_ew1/a_ew2) into L2
        air.dma_memcpy_nd
          (%staged_w1[] [] [],
           %a_ew1[%my_exp_start, %c0, %c0]
                 [%n_my_experts, arith.constant 2880 : index, arith.constant 7680 : index]
                 [arith.constant 22118400 : index,            // stride over expert dim
                  arith.constant 7680 : index, %c1]) :
          memref<3x2880x7680xf4, 2>, memref<16x2880x7680xf4>
        air.dma_memcpy_nd
          (%staged_w2[] [] [],
           %a_ew2[%my_exp_start, %c0, %c0] [...] [...]) :
          memref<3x7680x2880xf4, 2>, memref<16x7680x2880xf4>
      }

      // -----------------------------------------------------------------------
      // Receive hidden state (broadcast from attention)
      // -----------------------------------------------------------------------
      %hidden_buf = memref.alloc() : memref<1x2880xf16, 3>

      // All 6 FFN GPUs receive from the same broadcast channel
      %tok_hidden = air.channel.get @hidden_dispatch[%c0, %ffn_rank]
        (%hidden_buf[%c0, %c0][%c1, %c2880][%c1, %c1]) : memref<1x2880xf16, 3>

      // -----------------------------------------------------------------------
      // Receive route packets (possibly 0, 1 or 2 for this GPU per token)
      // † The @expert_route_dispatch channel delivers here only if routing table
      //   maps an expert_id to this rank.  get is non-blocking if no packets.
      // -----------------------------------------------------------------------
      %result_buf = memref.alloc() : memref<1x2880xf32, 2>   // accumulates top-k partial
      air.execute {
        linalg.fill ins(arith.constant 0.0 : f32) outs(%result_buf : memref<1x2880xf32, 2>)
      }

      // We may receive 0, 1, or 2 route packets addressed to this GPU
      // (plus possibly forwarded packets from peer FFN GPUs for exchange)
      // Use a counted get loop; max_iter = top_k + exchange_in = 2+2 (conservative)
      scf.for %slot = %c0 to arith.constant 4 : index step %c1 {

        // † Conditional get: returns immediately if no packet available for this slot.
        //   The channel carries a "done" sentinel when all top-k have been dispatched.
        %pkt   = memref.alloc() : memref<3xindex, 3>
        %valid = air.channel.get @expert_route_dispatch[%c0, %ffn_rank]
          (%pkt[%c0][%c3][%c1])
          {non_blocking, sentinel_done}
          : memref<3xindex, 3>    // returns i1 validity flag

        cf.cond_br %valid, ^process_packet(%pkt), ^no_packet

      ^no_packet:
        // Also check for forwarded packets from peer FFN GPUs
        // (tokens whose second expert lives here but was received by another FFN GPU first)
        scf.for %src_r = %c0 to %c6 step %c1 {
          %fwd_pkt  = memref.alloc() : memref<3xindex, 3>
          %fwd_valid = air.channel.get @expert_exchange[%src_r, %ffn_rank]
            (%fwd_pkt[%c0][%c3][%c1]) {non_blocking} : memref<3xindex, 3>
          cf.cond_br %fwd_valid, ^process_packet(%fwd_pkt), ^skip_fwd
        ^skip_fwd:
          memref.dealloc %fwd_pkt : memref<3xindex, 3>
        }
        cf.br ^next_slot

      ^process_packet(%active_pkt : memref<3xindex, 3>):
        %expert_id    = memref.load %active_pkt[%c1] : memref<3xindex, 3>
        %gate_bits    = memref.load %active_pkt[%c2] : memref<3xindex, 3>
        %local_eid    = arith.subi %expert_id, %my_exp_start : index
        %gate_weight  = arith.bitcast %gate_bits : index -> f32   // recover softmax weight

        // Check: is this expert actually mine? (It should be if routed correctly.)
        %in_range_lo  = arith.cmpi sge, %expert_id, %my_exp_start : index
        %exp_end      = arith.addi %my_exp_start, %n_my_experts : index
        %in_range_hi  = arith.cmpi slt, %expert_id, %exp_end : index
        %is_mine      = arith.andi %in_range_lo, %in_range_hi : i1

        cf.cond_br %is_mine, ^compute_expert(%local_eid, %gate_weight), ^forward_packet

      ^forward_packet:
        // Routing table mismatch (shouldn't happen in prototype, but modeled for
        // completeness — in a real system this handles cross-chip expert relocation)
        %correct_ffn_r = // lookup @expert_group_table[expert_id] - 2
        air.channel.put @expert_exchange[%ffn_rank, %correct_ffn_r]
          (%active_pkt[%c0][%c3][%c1]) : memref<3xindex, 3>
        cf.br ^next_slot

      ^compute_expert(%l_eid : index, %g_wt : f32):
        // -------------------------------------------------------------------
        // Expert FFN: SwiGLU
        //   gate_out[1x3840] = hidden[1x2880] × w1[2880x3840] (first half of w1)
        //   up_out  [1x3840] = hidden[1x2880] × w1[2880x3840] (second half)
        //   swiglu  [1x3840] = silu(gate_out) * up_out
        //   down    [1x2880] = swiglu[1x3840] × w2[3840x2880]
        //
        // w1 is stored interleaved [2880 x 7680] = [2880 x (gate|up)]
        // Both GEMVs run simultaneously in the herd (two output-tile halves)
        // -------------------------------------------------------------------
        %gate_buf   = memref.alloc() : memref<1x3840xf32, 3>
        %up_buf     = memref.alloc() : memref<1x3840xf32, 3>
        %swiglu_buf = memref.alloc() : memref<1x3840xf16, 3>
        %down_buf   = memref.alloc() : memref<1x2880xf32, 3>

        %tok_expert = air.segment @expert_seg
          args(%hid   = %hidden_buf, %w1_all = %staged_w1, %w2_all = %staged_w2,
               %eid_a = %l_eid,
               %gb_a  = %gate_buf, %ub_a = %up_buf, %sg_a = %swiglu_buf, %db_a = %down_buf) :
          memref<1x2880xf16, 3>, memref<3x2880x7680xf4, 2>, memref<3x7680x2880xf4, 2>,
          index,
          memref<1x3840xf32, 3>, memref<1x3840xf32, 3>, memref<1x3840xf16, 3>, memref<1x2880xf32, 3>
        {
          // Herd: 8 CUs along x, each computes 480 of the 3840 gate (and up) elements
          %c8_h = arith.constant 8 : index
          %tile_expert = arith.constant 480 : index
          air.herd @expert_gemv_herd tile (%tx, %ty) in (%sx = %c8_h, %sy = %c1)
            args(%hid_h = %hid, %w1_h = %w1_all, %eid_h = %eid_a,
                 %gb_h = %gb_a, %ub_h = %ub_a) :
            memref<1x2880xf16, 3>, memref<3x2880x7680xf4, 2>, index,
            memref<1x3840xf32, 3>, memref<1x3840xf32, 3>
          {
            %out_off = arith.muli %tx, %tile_expert : index

            // Gate GEMV tile: w1[eid, :, out_off:out_off+480]
            scf.for %ki = %c0 to %c2880 step arith.constant 64 : index {
              %w_gate_tile = memref.alloc() : memref<64x480xf4, 3>
              %w_up_tile   = memref.alloc() : memref<64x480xf4, 3>
              // DMA gate and up halves of w1 (they're interleaved; stride 7680)
              air.dma_memcpy_nd
                (%w_gate_tile[] [] [],
                 %w1_h[%eid_h, %ki, %out_off]
                      [arith.constant 64 : index, %tile_expert]
                      [arith.constant 7680 : index, %c1]) :
                memref<64x480xf4, 3>, memref<3x2880x7680xf4, 2>
              air.dma_memcpy_nd
                (%w_up_tile[] [] [],
                 %w1_h[%eid_h, %ki, arith.addi %out_off arith.constant 3840 : index]
                      [arith.constant 64 : index, %tile_expert]
                      [arith.constant 7680 : index, %c1]) :
                memref<64x480xf4, 3>, memref<3x2880x7680xf4, 2>
              // Accumulate (with MXFP4 dequant implicit in the matmul lowering)
              linalg.matmul
                ins(%hid_h, %w_gate_tile : memref<1x2880xf16, 3>, memref<64x480xf4, 3>)
                outs(%gb_h[%c0, %out_off][%c1, %tile_expert][%c1, %c1] : memref<1x3840xf32, 3>)
              linalg.matmul
                ins(%hid_h, %w_up_tile   : memref<1x2880xf16, 3>, memref<64x480xf4, 3>)
                outs(%ub_h[%c0, %out_off][%c1, %tile_expert][%c1, %c1] : memref<1x3840xf32, 3>)
              memref.dealloc %w_gate_tile : memref<64x480xf4, 3>
              memref.dealloc %w_up_tile   : memref<64x480xf4, 3>
            }
          } // end @expert_gemv_herd

          // SwiGLU: swiglu[i] = silu(gate[i]) * up[i],  downcast to f16
          air.execute {
            linalg.generic {
              indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>,   // gate
                               affine_map<(d0,d1) -> (d0,d1)>,   // up
                               affine_map<(d0,d1) -> (d0,d1)>],  // out
              iterator_types = ["parallel", "parallel"]
            }
            ins(%gb_a, %ub_a : memref<1x3840xf32, 3>, memref<1x3840xf32, 3>)
            outs(%sg_a : memref<1x3840xf16, 3>)
            {
              ^bb0(%gate_v : f32, %up_v : f32, %out_v : f16):
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                %neg_g  = arith.negf %gate_v : f32
                %exp_g  = math.exp %neg_g : f32
                %sig_g  = arith.divf arith.constant 1.0 : f32,
                                     arith.addf arith.constant 1.0 : f32, %exp_g : f32
                %silu_g = arith.mulf %gate_v, %sig_g : f32
                %prod   = arith.mulf %silu_g, %up_v : f32
                %down   = arith.truncf %prod : f32 to f16
                linalg.yield %down : f16
            }
          }

          // Down projection: swiglu[1x3840] × w2[3840x2880]
          air.herd @down_proj_herd tile (%tx, %ty) in (%sx = %c8_h, %sy = %c1)
            args(%sg_h = %sg_a, %w2_h = %w2_all, %eid_h = %eid_a, %db_h = %db_a) :
            memref<1x3840xf16, 3>, memref<3x7680x2880xf4, 2>, index, memref<1x2880xf32, 3>
          {
            %out_tile = arith.constant 360 : index   // 2880 / 8 CUs
            %d_off    = arith.muli %tx, %out_tile : index
            scf.for %ki = %c0 to %c3840 step arith.constant 64 : index {
              %w2_tile = memref.alloc() : memref<64x360xf4, 3>
              air.dma_memcpy_nd
                (%w2_tile[] [] [],
                 %w2_h[%eid_h, %ki, %d_off]
                      [arith.constant 64 : index, %out_tile]
                      [%c2880, %c1]) :
                memref<64x360xf4, 3>, memref<3x7680x2880xf4, 2>
              linalg.matmul
                ins(%sg_h, %w2_tile : memref<1x3840xf16, 3>, memref<64x360xf4, 3>)
                outs(%db_h[%c0, %d_off][%c1, %out_tile][%c1, %c1] : memref<1x2880xf32, 3>)
              memref.dealloc %w2_tile : memref<64x360xf4, 3>
            }
          } // end @down_proj_herd

          air.segment_terminator
        } // end @expert_seg

        // Accumulate into result_buf (weighted by gate_weight)
        air.execute [%tok_expert] {
          linalg.generic { /* result_buf += gate_weight * down_buf, with f32→f32 */ }
            ins(%down_buf, %gate_weight : memref<1x2880xf32, 3>, f32)
            outs(%result_buf : memref<1x2880xf32, 2>)
        }

        memref.dealloc %gate_buf   : memref<1x3840xf32, 3>
        memref.dealloc %up_buf     : memref<1x3840xf32, 3>
        memref.dealloc %swiglu_buf : memref<1x3840xf16, 3>
        memref.dealloc %down_buf   : memref<1x2880xf32, 3>
        memref.dealloc %active_pkt : memref<3xindex, 3>
        cf.br ^next_slot

      ^next_slot:
        memref.dealloc %pkt : memref<3xindex, 3>
      } // end scf.for over packet slots

      // -----------------------------------------------------------------------
      // Send expert result(s) back to attention cluster
      // † @expert_result_gather: each FFN GPU that contributed puts its partial.
      //   The gather_reduce semantic on the get side accumulates them.
      //   If this GPU had no experts activated this token, it sends a zero vector
      //   with contribution_count=0 so the gather doesn't stall.
      // -----------------------------------------------------------------------
      %result_f16 = memref.alloc() : memref<1x2880xf16, 3>
      air.execute {
        linalg.generic { /* truncf f32→f16 */ }
          ins(%result_buf : memref<1x2880xf32, 2>)
          outs(%result_f16 : memref<1x2880xf16, 3>)
      }
      air.channel.put @expert_result_gather[%ffn_rank, %c0]
        (%result_f16[%c0, %c0][%c1, %c2880][%c1, %c1])
        {contribution_weight_field = 2 : i32}   // † channel reads gate_weight from pkt[2]
        : memref<1x2880xf16, 3>

      memref.dealloc %hidden_buf  : memref<1x2880xf16, 3>
      memref.dealloc %result_buf  : memref<1x2880xf32, 2>
      memref.dealloc %result_f16  : memref<1x2880xf16, 3>
      // Note: staged_w1, staged_w2 are NOT freed — they are weight-stationary
      //       across all tokens in the serve loop.

      cf.br ^end

    ^end:
      air.launch_terminator
  } // end air.launch

  return
} // end @gpt_decode_megakernel

} // end module
