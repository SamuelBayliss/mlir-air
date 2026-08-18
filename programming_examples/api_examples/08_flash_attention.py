# Example 08: Flash Attention on NPU
#
# Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
# using the online flash attention algorithm — no materialisation of the full S×S
# attention matrix. Each AIE core processes one head independently; K and V blocks
# stream through the herd via cascade to avoid redundant loads from global memory.
#
# Concepts introduced:
#   - air.Cascade(axis=0): K/V blocks stream tile-to-tile along the head axis
#   - air.Broadcast: Q block broadcast to all head tiles from the loader tile
#   - Double-buffer pipeline ring (§7.4): depth=AUTO on cascade channel; compiler
#     resolves to 2 so the loader tile runs one K-step ahead of compute tiles
#   - asynchronous=True on herd: token used for explicit join before seg body returns
#   - Online softmax: running (m, l, O) update within each tile — fully local,
#     no cross-tile reduction needed
#   - Multiple herds per segment (§6.4): load herd (Q prefetch) runs before
#     the main attention herd
#
# Tiling structure:
#   Segment grid: (S // seg_s,) — one NPU column per sequence block
#   Herd:         (H,)          — one AIE core per attention head
#   Inner loop:   S // tile_k   — K/V blocks streamed via cascade
#
# Memory hierarchy:
#   global          (LPDDR)        Q, K, V, O — host-accessible
#   seg.private()   (MemTile SRAM) Q_stage — current Q block, broadcast to all heads
#   h.private()     (AIE L1)       q_tile, k_tile, v_tile — per-head working buffers
#                                  scores, m, l, acc — online softmax state
#
# See spec/SPEC.md §7.4 for the double-buffer pipeline ring depth analysis.
# See spec/ARCHITECTURES.md §5 for cascade channel physical mapping.

import argparse

import numpy as np
from itertools import product
import math

import airapi as air
import airapi.ops
from airapi.types import bf16, f32
from airapi.layout import AffineLayout


S = air.symbol()   # sequence length (same for Q, K, V)
H = air.symbol()   # number of attention heads
d = air.symbol()   # head dimension

Q = air.tensor([S, H, d], bf16)
K = air.tensor([S, H, d], bf16)
V = air.tensor([S, H, d], bf16)
O = air.tensor([S, H, d], f32)


def build_flash_attention(seg_s, tile_s, tile_k, num_heads):
    """
    Build a flash attention launch for NPU.

    Each segment processes one seg_s-length block of the query sequence.
    Within the segment, one AIE core per head processes its head independently.
    K and V blocks stream through the herd via cascade to avoid per-core global loads.

    seg_s:     query block size (rows per segment tile)
    tile_s:    query sub-tile per AIE core (must divide seg_s)
    tile_k:    K/V streaming block size (columns consumed per inner iteration)
    num_heads: number of attention heads H (must match actual H at dispatch)
    """
    assert seg_s % tile_s == 0

    with air.launch() as launch:
        @launch.body
        def _():
            with air.segment(
                product(range(0, S, seg_s),),
                requires=[air.Scratchpad, air.Cascade(axis=0), air.Broadcast],
            ) as seg:

                @seg.body
                def _(ss,):
                    q_row = ss * seg.tile_sizes[0]

                    # Q staging buffer in MemTile — broadcast to all head tiles.
                    # One seg_s × d slice of Q is loaded here and broadcast via h.shared().
                    Q_stage = air.alloc(
                        [seg_s, d], bf16, scope=seg.private(),
                        layout=AffineLayout(strides=[d, 1]),
                    )

                    # ── Prefetch herd: load Q block into MemTile ──────────────────────
                    # Sequential (no asynchronous=True): next herd waits for this to
                    # complete so Q_stage is populated before attention begins.
                    with air.herd(range(0, 1, 1)) as h_load:
                        @h_load.body
                        def _(_,):
                            air.ops.load(Q_stage, Q[q_row : q_row + seg_s, :, :])

                    # ── Attention herd: one AIE core per head ─────────────────────────
                    with air.herd(
                        range(0, num_heads, 1),
                        asynchronous=True,
                    ) as h:
                        # Cascade channel: K and V blocks stream head-to-head.
                        # Tile 0 loads from global; tiles 1..H-1 receive via cascade.
                        # pipe_depth: compile-time symbol for cascade buffer depth.
                        # Compiler resolves to 2 (double-buffer, §7.4): tile 0 can
                        # load K[i+1] while K[i] is still being processed downstream.
                        pipe_depth = air.symbol()
                        k_pipe = h.channel(
                            shape=h.shape, dtype=bf16,
                            kind="cascade", axis=0, depth=pipe_depth,
                        )
                        v_pipe = h.channel(
                            shape=h.shape, dtype=bf16,
                            kind="cascade", axis=0, depth=pipe_depth,
                        )

                        # Broadcast channel: Q tile distributed from MemTile to all cores.
                        q_chan = h.channel(
                            shape=(1,), dtype=bf16,
                            kind="broadcast",
                        )

                        @h.body
                        def _(tx,):
                            head = tx   # this core handles head `tx`

                            # Per-head working buffers in AIE L1.
                            q_tile = air.alloc([tile_s, d], bf16, scope=h.private())
                            k_tile = air.alloc([tile_k, d], bf16, scope=h.private())
                            v_tile = air.alloc([tile_k, d], bf16, scope=h.private())

                            # Online softmax state (Dao et al. 2022).
                            scores  = air.alloc([tile_s, tile_k], f32, scope=h.private())
                            m_i     = air.alloc([tile_s],         f32, scope=h.private())
                            l_i     = air.alloc([tile_s],         f32, scope=h.private())
                            acc     = air.alloc([tile_s, d],      f32, scope=h.private())
                            m_i[:]  = float("-inf")
                            l_i[:]  = 0.0
                            acc[:]  = 0.0

                            # Receive Q tile from MemTile broadcast.
                            q_full = q_chan.get(indices=[0])
                            q_tile[:] = q_full[:tile_s, head, :]

                            # Inner loop: stream K and V blocks through the herd.
                            for k in range(0, S, tile_k):
                                if tx == 0:
                                    # Loader tile: fetch K[k] and V[k] from global.
                                    air.ops.load(k_tile, K[k : k + tile_k, head, :])
                                    air.ops.load(v_tile, V[k : k + tile_k, head, :])
                                    k_pipe.put(indices=[0], value=k_tile)
                                    v_pipe.put(indices=[0], value=v_tile)
                                else:
                                    # Consumer tile: receive from predecessor.
                                    k_recv = k_pipe.get(indices=[tx - 1])
                                    v_recv = v_pipe.get(indices=[tx - 1])
                                    k_tile[:] = k_recv[:]
                                    v_tile[:] = v_recv[:]
                                    # Forward to successor (all tiles except the last).
                                    if tx < num_heads - 1:
                                        k_pipe.put(indices=[tx], value=k_tile)
                                        v_pipe.put(indices=[tx], value=v_tile)

                                # Attention scores: q_tile @ k_tile^T / sqrt(d)
                                # Lowers to air.dot with transposed B.
                                scale = 1.0 / math.sqrt(d)
                                air.ops.dot(q_tile, k_tile, acc=scores, alpha=scale,
                                        transpose_b=True)

                                # Online softmax update (flash attention Algorithm 1).
                                m_new  = air.ops.reduce(scores, axis=1, op=air.ops.max)
                                m_new  = air.ops.reduce(
                                    air.ops.stack([m_i, m_new]), axis=0, op=air.ops.max
                                )
                                l_i[:] = (air.ops.exp(m_i - m_new) * l_i
                                          + air.ops.reduce(
                                              air.ops.exp(scores - m_new[:, None]),
                                              axis=1, op=air.ops.sum,
                                          ))
                                acc[:] = (air.ops.exp(m_i - m_new)[:, None] * acc
                                          + air.ops.exp(scores - m_new[:, None]) @ v_tile)
                                m_i[:] = m_new[:]

                            # Normalise and write output.
                            o_tile = air.alloc([tile_s, d], f32, scope=h.private())
                            o_tile[:] = acc[:] / l_i[:, None]
                            air.ops.store(
                                o_tile,
                                O[q_row : q_row + tile_s, head, :],
                            )

                        # Wait for all head tiles to complete before segment body returns.
                        air.ops.wait(h.token)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention on NPU")
    parser.add_argument("--seq-len",   type=int, default=1024,
                        help="Sequence length S")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads H")
    parser.add_argument("--head-dim",  type=int, default=64,
                        help="Head dimension d")
    parser.add_argument("--seg-s",     type=int, default=64,
                        help="Query sequence block per NPU column")
    parser.add_argument("--tile-s",    type=int, default=64,
                        help="Query sub-tile per AIE core (must equal seg_s for single-tile)")
    parser.add_argument("--tile-k",    type=int, default=64,
                        help="K/V streaming block size")
    parser.add_argument("--target",    type=str, default="npu2",
                        choices=["npu1", "npu2"])
    parser.add_argument("--print-ir",      action="store_true")
    parser.add_argument("--compile-only",  action="store_true")
    args = parser.parse_args()

    s, h_n, d_n = args.seq_len, args.num_heads, args.head_dim
    np.random.seed(0)
    q_np = np.random.randn(s, h_n, d_n).astype(np.float16)
    k_np = np.random.randn(s, h_n, d_n).astype(np.float16)
    v_np = np.random.randn(s, h_n, d_n).astype(np.float16)

    launch = build_flash_attention(
        seg_s=args.seg_s,
        tile_s=args.tile_s,
        tile_k=args.tile_k,
        num_heads=args.num_heads,
    )

    if args.print_ir:
        print(launch.mlir())

    if args.compile_only:
        launch.compile(target=args.target)
        print("Compiled successfully.")
    else:
        fn = launch.compile(target=args.target)
        o_np = fn(q_np, k_np, v_np)

        # Reference: standard scaled dot-product attention in f32.
        q_f = q_np.astype(np.float32)
        k_f = k_np.astype(np.float32)
        v_f = v_np.astype(np.float32)
        scale = 1.0 / math.sqrt(d_n)
        # (S, H, S) attention weights per head
        attn = np.einsum("shd,khd->shk", q_f, k_f) * scale
        attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("shk,khd->shd", attn, v_f)

        passed = np.allclose(o_np, expected, rtol=1e-2, atol=1e-2)
        print(f"Flash Attention {'PASSED' if passed else 'FAILED'}: "
              f"S={s}, H={h_n}, d={d_n}")
