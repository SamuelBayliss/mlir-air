#!/usr/bin/env python3
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Visualization of Blocked Matrix–Vector Multiplication on an AMD AIE Spatial Array.

Illustrates:
  - How the matrix A[K, N] and input vector y[K] are blocked and distributed
    across a 2D array of AMD AIE tiles.
  - The cascade dataflow pattern: partial sums propagate along rows of tiles,
    accumulating toward the final result vector x[N].
  - The memory hierarchy: DDR (L3) → Segment memory (L2) → Tile memory (L1).

Computation:  x[N] = y[K] × A[K, N]   (vector × matrix → vector)
Array layout: ROWS_AIE × COLS_AIE tiles; columns handle cascade reduction over K,
              rows handle independent slices of the N output dimension.

Usage:
  python aie_matvec_dataflow.py [--output PATH] [--no-show]

Dependencies: matplotlib, numpy
"""

import argparse
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ─── Configurable dimensions ─────────────────────────────────────────────────
K = 512           # reduction / inner dimension
N = 256           # output dimension
ROWS_AIE = 2      # rows of AIE tiles  (parallel N segments)
COLS_AIE = 4      # cols of AIE tiles  (cascade K reduction stages)
TILE_K = K // COLS_AIE   # 128 per column
TILE_N = N // ROWS_AIE   # 128 per row

# ─── Colour palette ──────────────────────────────────────────────────────────
BG       = "#0D1117"
PANEL    = "#161B22"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
DIM      = "#8B949E"
WHITE    = "#FFFFFF"

# Vivid colours for K-slices (one per cascade column)
K_COLS = ["#FF6B35", "#FFD700", "#39FF14", "#00BFFF"]   # orange, gold, neon-green, sky-blue
# Vivid colours for N-slices (one per AIE row / output segment)
N_COLS = ["#FF1493", "#9B59FF"]                          # deep-pink, violet
# Cascade partial-sum colours (glow effect)
CASCADE_COL = "#FFFFFF"
DMA_COL     = "#FFA500"
OUTPUT_COL  = "#7CFF7C"

# ─── Geometry constants (normalised axes units) ───────────────────────────────
TILE_W  = 1.6
TILE_H  = 1.3
HGAP    = 0.55    # horizontal gap between tiles
VGAP    = 0.65    # vertical gap between rows


# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════

def _blend(c1, c2, t=0.50):
    """Linear-blend two hex colours; t=0 → c1, t=1 → c2."""
    import matplotlib.colors as mc
    r1, g1, b1 = mc.to_rgb(c1)
    r2, g2, b2 = mc.to_rgb(c2)
    return (r1 + t * (r2 - r1), g1 + t * (g2 - g1), b1 + t * (b2 - b1))


def tile_fill(row, col, alpha=0.90):
    """Return an RGBA tuple for the AIE tile at (row, col)."""
    return (*_blend(K_COLS[col], N_COLS[row], t=0.40), alpha)


def dark_ax(ax, facecolor=PANEL):
    ax.set_facecolor(facecolor)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=DIM, labelsize=7)
    return ax


def add_label(ax, x, y, text, color=TEXT, fs=8, fw="bold", ha="center", va="center",
              bbox_color=None):
    kw = dict(ha=ha, va=va, color=color, fontsize=fs, fontweight=fw,
              transform=ax.transData)
    if bbox_color:
        kw["bbox"] = dict(boxstyle="round,pad=0.25", facecolor=bbox_color,
                          edgecolor="none", alpha=0.80)
    ax.text(x, y, text, **kw)


def rounded_rect(ax, x, y, w, h, color, edge="#FFFFFF44", lw=1.2, zorder=2,
                 corner_r=0.06):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={corner_r}",
                       facecolor=color, edgecolor=edge,
                       linewidth=lw, zorder=zorder,
                       transform=ax.transData)
    ax.add_patch(p)
    return p


def draw_arrow(ax, x0, y0, x1, y1, color=WHITE, lw=1.8, style="->",
               rad=0.0, zorder=5, shrink=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="data", textcoords="data",
                arrowprops=dict(
                    arrowstyle=style,
                    color=color,
                    lw=lw,
                    shrinkA=shrink, shrinkB=shrink,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=zorder)


# ═════════════════════════════════════════════════════════════════════════════
# Section drawers
# ═════════════════════════════════════════════════════════════════════════════

def draw_memory_hierarchy(ax):
    """Top panel: DDR ➜ L2 ➜ L1 memory hierarchy overview."""
    dark_ax(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    # ════════════════════════════════════════════════════════════════════════
    # Three horizontal bands: DDR (L3) at top, L2 in middle, L1 hint at bottom
    # ════════════════════════════════════════════════════════════════════════
    band_h = 0.80
    y_ddr = 2.40
    y_l2  = 1.40
    y_l1  = 0.40

    # Background bands
    for y_band, label, col_bg, col_edge in [
        (y_ddr, "DDR  (L3)  –  External memory",   "#1C2433", "#3A4A5E"),
        (y_l2,  "Segment Memory  (L2)  –  On-chip shared buffer", "#1A2B1A", "#2E5E2E"),
        (y_l1,  "AIE Tile Memory  (L1)  –  Per-core SRAM",        "#2A1A2A", "#5E2E5E"),
    ]:
        rounded_rect(ax, 0.0, y_band, 10.0, band_h,
                     color=col_bg, edge=col_edge, lw=1.2, corner_r=0.04)
        ax.text(0.12, y_band + band_h - 0.10, label,
                color=DIM, fontsize=7.5, va="top", style="italic")

    # ── DDR contents: y-vector and A-matrix blocks ────────────────────────────
    # y-vector (left portion)
    yw   = 1.70 / COLS_AIE
    y_off = y_ddr + 0.12
    for col in range(COLS_AIE):
        x0 = 0.12 + col * (yw + 0.03)
        rounded_rect(ax, x0, y_off, yw, 0.55,
                     color=K_COLS[col], edge="#FFFFFF66", lw=1, corner_r=0.04)
        lo, hi = col * TILE_K, (col + 1) * TILE_K
        add_label(ax, x0 + yw / 2, y_off + 0.275,
                  f"y[{lo}:{hi}]", color=BG, fs=6.5)
    ax.text(0.12 + COLS_AIE * (yw + 0.03) / 2, y_ddr + 0.06,
            "y[K]", color=DIM, fontsize=6.5, ha="center")

    # A matrix blocks (centre)
    am_x0  = 2.10
    am_bw  = 0.60
    am_bh  = 0.27
    for row in range(ROWS_AIE):
        for col in range(COLS_AIE):
            bx = am_x0 + col * (am_bw + 0.03)
            by = y_off + row * (am_bh + 0.02)
            rounded_rect(ax, bx, by, am_bw, am_bh,
                         color=tile_fill(row, col, alpha=0.90),
                         edge="#FFFFFF44", lw=0.7, corner_r=0.03)
            add_label(ax, bx + am_bw / 2, by + am_bh / 2,
                      f"A[{col*TILE_K}:{(col+1)*TILE_K},\n {row*TILE_N}:{(row+1)*TILE_N}]",
                      color=TEXT, fs=5)
    ax.text(am_x0 + COLS_AIE * (am_bw + 0.03) / 2, y_ddr + 0.06,
            "A[K×N]", color=DIM, fontsize=6.5, ha="center")

    # x output vector (right)
    xv_x0 = 8.60
    xv_h  = (band_h - 0.24) / ROWS_AIE
    for row in range(ROWS_AIE):
        by = y_off + row * (xv_h + 0.04)
        rounded_rect(ax, xv_x0, by, 1.20, xv_h,
                     color=N_COLS[row], edge="#FFFFFF55", lw=0.8, corner_r=0.04)
        add_label(ax, xv_x0 + 0.60, by + xv_h / 2,
                  f"x[{row*TILE_N}:{(row+1)*TILE_N}]", color=BG, fs=6.5)
    ax.text(xv_x0 + 0.60, y_ddr + 0.06, "x[N]", color=DIM,
            fontsize=6.5, ha="center")

    # ── L2 contents: one buffer per (row, col) tile ───────────────────────────
    l2_bw  = 0.85
    l2_bh  = 0.52
    l2_off = y_l2 + (band_h - l2_bh) / 2
    spacing = (10.0 - COLS_AIE * ROWS_AIE * l2_bw) / (COLS_AIE * ROWS_AIE + 1)
    idx = 0
    for col in range(COLS_AIE):
        for row in range(ROWS_AIE):
            bx = spacing + idx * (l2_bw + spacing)
            rounded_rect(ax, bx, l2_off, l2_bw, l2_bh,
                         color=tile_fill(row, col, alpha=0.70),
                         edge="#FFFFFF44", lw=0.8, corner_r=0.04)
            add_label(ax, bx + l2_bw / 2, l2_off + l2_bh / 2,
                      f"L2\ny[{col*TILE_K}:{(col+1)*TILE_K}]\n"
                      f"A[:,{row*TILE_N}:{(row+1)*TILE_N}]",
                      color=TEXT, fs=5.5)
            idx += 1

    # ── L1 hint: one coloured square per tile ─────────────────────────────────
    l1_bw  = 0.70
    l1_bh  = 0.44
    l1_off = y_l1 + (band_h - l1_bh) / 2
    for col in range(COLS_AIE):
        for row in range(ROWS_AIE):
            bx = spacing + (col * ROWS_AIE + row) * (l2_bw + spacing)
            rounded_rect(ax, bx + (l2_bw - l1_bw) / 2, l1_off, l1_bw, l1_bh,
                         color=tile_fill(row, col, alpha=0.85),
                         edge=WHITE + "66", lw=1.0, corner_r=0.04)
            add_label(ax, bx + l2_bw / 2, l1_off + l1_bh / 2,
                      f"[{row},{col}]", color=TEXT, fs=6.5)

    # ── DMA arrows: DDR → L2, L2 → L1 ───────────────────────────────────────
    for col in range(COLS_AIE):
        for row in range(ROWS_AIE):
            buf_idx = col * ROWS_AIE + row
            bx_center = spacing + buf_idx * (l2_bw + spacing) + l2_bw / 2
            # DDR → L2
            y_src_ddr = y_ddr
            y_dst_l2  = y_l2 + band_h
            draw_arrow(ax, bx_center, y_src_ddr,
                       bx_center, y_dst_l2,
                       color=DMA_COL, lw=1.3, shrink=2)
            # L2 → L1
            y_src_l2 = y_l2
            y_dst_l1 = y_l1 + band_h
            draw_arrow(ax, bx_center, y_src_l2,
                       bx_center, y_dst_l1,
                       color=DMA_COL, lw=1.3, shrink=2)

    # DMA label (once)
    ax.text(9.85, y_l2 + band_h / 2, "DMA", color=DMA_COL,
            fontsize=7, ha="right", va="center", style="italic")


def draw_aie_array(ax):
    """Central panel: 2D AIE array with tiles, cascade arrows, I/O."""
    dark_ax(ax, facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)

    total_w = COLS_AIE * TILE_W + (COLS_AIE - 1) * HGAP
    total_h = ROWS_AIE * TILE_H + (ROWS_AIE - 1) * VGAP
    margin_x = 1.80
    margin_y = 0.50

    ax.set_xlim(-margin_x, total_w + 2.8)
    ax.set_ylim(-0.80, total_h + margin_y + 1.00)
    ax.axis("off")

    # ── Column headers (K slices) ─────────────────────────────────────────────
    for col in range(COLS_AIE):
        cx = col * (TILE_W + HGAP) + TILE_W / 2
        cy = total_h + margin_y + 0.60
        lo, hi = col * TILE_K, (col + 1) * TILE_K
        rounded_rect(ax, cx - 0.68, cy - 0.22, 1.36, 0.44,
                     color=K_COLS[col], edge="none", corner_r=0.08)
        add_label(ax, cx, cy, f"y[{lo}:{hi}]\nK-stage {col}", color=BG, fs=7.5)
        # Small arrow pointing down into col
        draw_arrow(ax, cx, cy - 0.22, cx, total_h + margin_y,
                   color=K_COLS[col], lw=1.5, shrink=2)

    # ── Row labels (N segments) ───────────────────────────────────────────────
    for row in range(ROWS_AIE):
        ry = (ROWS_AIE - 1 - row) * (TILE_H + VGAP) + TILE_H / 2
        lo, hi = row * TILE_N, (row + 1) * TILE_N
        ax.text(-margin_x + 0.10, ry,
                f"N-row {row}\nx[{lo}:{hi}]",
                color=N_COLS[row], fontsize=8, fontweight="bold",
                ha="left", va="center")

    # ── Tiles ─────────────────────────────────────────────────────────────────
    for row in range(ROWS_AIE):
        for col in range(COLS_AIE):
            tx = col * (TILE_W + HGAP)
            ty = (ROWS_AIE - 1 - row) * (TILE_H + VGAP)

            fc = tile_fill(row, col)
            rounded_rect(ax, tx, ty, TILE_W, TILE_H,
                         color=fc, edge=WHITE + "88", lw=1.5, corner_r=0.10)

            # Tile ID
            add_label(ax, tx + TILE_W / 2, ty + TILE_H - 0.22,
                      f"AIE [{row},{col}]", color=TEXT, fs=7.5)

            # Matrix block label
            kl, kh = col * TILE_K, (col + 1) * TILE_K
            nl, nh = row * TILE_N, (row + 1) * TILE_N
            add_label(ax, tx + TILE_W / 2, ty + TILE_H / 2 + 0.02,
                      f"L1: A[{kl}:{kh},\n       {nl}:{nh}]",
                      color=TEXT, fs=7)

            # Partial-sum indicator (bottom strip)
            n_accumulated = col + 1
            strip_w = TILE_W * (n_accumulated / COLS_AIE)
            rounded_rect(ax, tx + 0.08, ty + 0.08,
                         strip_w - 0.08, 0.22,
                         color=OUTPUT_COL, edge="none", corner_r=0.05)
            add_label(ax, tx + 0.08 + (strip_w - 0.08) / 2, ty + 0.08 + 0.11,
                      f"Σ ×{n_accumulated}", color=BG, fs=6.5)

    # ── Cascade arrows (partial sums flow right along each row) ───────────────
    for row in range(ROWS_AIE):
        ry = (ROWS_AIE - 1 - row) * (TILE_H + VGAP) + TILE_H * 0.18
        for col in range(COLS_AIE - 1):
            x0 = col * (TILE_W + HGAP) + TILE_W
            x1 = (col + 1) * (TILE_W + HGAP)
            draw_arrow(ax, x0, ry, x1, ry,
                       color=CASCADE_COL, lw=2.5,
                       style="-|>", shrink=4)
            add_label(ax, (x0 + x1) / 2, ry + 0.22,
                      "partial\nΣ", color=CASCADE_COL, fs=5.5)

    # ── Input DMA arrows (top of column → tile) ───────────────────────────────
    for col in range(COLS_AIE):
        cx = col * (TILE_W + HGAP) + TILE_W / 2
        y_top = total_h + margin_y

        for row in range(ROWS_AIE):
            ty = (ROWS_AIE - 1 - row) * (TILE_H + VGAP) + TILE_H
            draw_arrow(ax, cx - 0.25 + row * 0.50, y_top,
                       cx - 0.25 + row * 0.50, ty,
                       color=K_COLS[col], lw=1.2, rad=0.0, shrink=3)

    # ── Output arrows (final column → output vector) ─────────────────────────
    out_x0 = (COLS_AIE - 1) * (TILE_W + HGAP) + TILE_W
    out_vec_x = out_x0 + 0.45
    for row in range(ROWS_AIE):
        ry = (ROWS_AIE - 1 - row) * (TILE_H + VGAP) + TILE_H * 0.18
        lo, hi = row * TILE_N, (row + 1) * TILE_N
        draw_arrow(ax, out_x0, ry, out_vec_x + 0.05, ry,
                   color=OUTPUT_COL, lw=2.5, style="-|>", shrink=3)
        rounded_rect(ax, out_vec_x + 0.06, ry - 0.30, 1.35, 0.60,
                     color=N_COLS[row], edge=WHITE + "66", lw=1.2, corner_r=0.08)
        add_label(ax, out_vec_x + 0.06 + 0.675, ry,
                  f"x[{lo}:{hi}]\n(output)", color=BG, fs=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=K_COLS[0],    label=f"K-block 0  (y[0:{TILE_K}])"),
        mpatches.Patch(color=K_COLS[1],    label=f"K-block 1  (y[{TILE_K}:{2*TILE_K}])"),
        mpatches.Patch(color=K_COLS[2],    label=f"K-block 2  (y[{2*TILE_K}:{3*TILE_K}])"),
        mpatches.Patch(color=K_COLS[3],    label=f"K-block 3  (y[{3*TILE_K}:{K}])"),
        mpatches.Patch(color=OUTPUT_COL,   label="Partial / full Σ"),
        mpatches.Patch(color=CASCADE_COL,  label="Cascade link  (partial sum)"),
        mpatches.Patch(color=DMA_COL,      label="DMA transfer"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=7, framealpha=0.25,
              facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT, ncol=2,
              bbox_to_anchor=(0.0, -0.18))

    # ── Section title ─────────────────────────────────────────────────────────
    ax.text(total_w / 2, total_h + margin_y + 1.15,
            f"AIE Spatial Array  ·  {ROWS_AIE} rows × {COLS_AIE} cols  "
            f"·  TILE_K={TILE_K}  TILE_N={TILE_N}",
            color=TEXT, fontsize=10, fontweight="bold", ha="center")


def draw_dataflow_timeline(ax):
    """Bottom panel: horizontal execution timeline showing cascade pipeline.

    Each row represents one AIE row.  Time flows right.
    Each coloured bar segment shows which K-block a tile is processing
    at that point in time.  Tiles in different columns work simultaneously
    on different K-blocks, and their partial sums cascade rightward.
    """
    dark_ax(ax)

    # Time axis: simulate COLS_AIE pipeline stages plus a combine phase
    N_PHASES = COLS_AIE + 1   # compute phases + final output
    ax.set_xlim(-0.05, N_PHASES + 0.5)
    ax.set_ylim(-0.70, ROWS_AIE + 0.55)
    ax.set_xlabel("Pipeline phase  (time →)", color=DIM, fontsize=8)
    ax.set_title("Cascade pipeline: all tiles compute in parallel "
                 "— partial sums flow rightward each phase",
                 color=TEXT, fontsize=9, pad=6)
    ax.tick_params(bottom=True, left=False, labelleft=False)
    ax.set_xticks(range(N_PHASES + 1))
    ax.set_xticklabels(
        [f"t{i}" for i in range(N_PHASES)] + ["done"],
        color=DIM, fontsize=7)

    bar_h   = 0.45      # height of each tile bar
    bar_gap = 0.10      # gap between bars within an AIE-row group
    grp_h   = COLS_AIE * (bar_h + bar_gap)   # total height of one AIE-row group
    grp_gap = 0.45      # gap between AIE-row groups

    total_h = ROWS_AIE * grp_h + (ROWS_AIE - 1) * grp_gap
    ax.set_ylim(-0.55, total_h + 0.35)

    for aie_row in range(ROWS_AIE):
        grp_y0 = (ROWS_AIE - 1 - aie_row) * (grp_h + grp_gap)

        # Row label (left)
        ax.text(-0.04, grp_y0 + grp_h / 2,
                f"AIE row {aie_row}\nx[{aie_row*TILE_N}:{(aie_row+1)*TILE_N}]",
                color=N_COLS[aie_row], fontsize=7.5, fontweight="bold",
                ha="right", va="center")

        for col in range(COLS_AIE):
            by = grp_y0 + col * (bar_h + bar_gap)

            # Background bar (full timeline width)
            rounded_rect(ax, 0.0, by, N_PHASES, bar_h,
                         color=BORDER, edge="none", corner_r=0.04)

            for phase in range(N_PHASES):
                if phase == col:
                    # ── Active compute phase ──────────────────────────────────
                    c = tile_fill(aie_row, col, alpha=0.95)
                    rounded_rect(ax, phase, by, 1.0, bar_h,
                                 color=c, edge=WHITE + "66", lw=1.0, corner_r=0.04)
                    lbl = (f"compute: y[{col*TILE_K}:{(col+1)*TILE_K}] × A"
                           f"  →  Σ{col+1}")
                    add_label(ax, phase + 0.50, by + bar_h / 2,
                              lbl, color=BG, fs=6)
                elif phase > col:
                    # ── Cascade carry (partial sum in transit) ────────────────
                    c = (*_blend(K_COLS[col], CASCADE_COL, t=0.20), 0.32)
                    rounded_rect(ax, phase, by, 1.0, bar_h,
                                 color=c, edge="none", corner_r=0.04)
                    if phase == COLS_AIE:
                        add_label(ax, phase + 0.50, by + bar_h / 2,
                                  "→ output", color=OUTPUT_COL, fs=6)

            # Tile row label (right of bg)
            ax.text(N_PHASES + 0.06, by + bar_h / 2,
                    f"tile [{aie_row},{col}]",
                    color=DIM, fontsize=6.5, ha="left", va="center")

        # Cascade arrows between phase slots (drawn below the group)
        arrow_y = grp_y0 - 0.18
        for phase in range(COLS_AIE):
            draw_arrow(ax, phase + 0.97, arrow_y,
                       phase + 1.03, arrow_y,
                       color=CASCADE_COL, lw=2.0, style="-|>", shrink=1)
        ax.text(COLS_AIE / 2, arrow_y - 0.12,
                f"← partial Σ cascades right across columns →",
                color=CASCADE_COL, fontsize=6.5, ha="center")


def draw_blocking_diagram(ax):
    """Right panel: schematic of the blocked A matrix and y/x vectors."""
    dark_ax(ax)
    ax.set_xlim(-0.3, COLS_AIE + 0.4)
    ax.set_ylim(-0.5, ROWS_AIE + 1.4)
    ax.axis("off")
    ax.set_title("Memory blocking\nschematic", color=TEXT, fontsize=8, pad=4)

    cell_w = 0.88
    cell_h = 0.78

    # ── y vector (top) ────────────────────────────────────────────────────────
    for col in range(COLS_AIE):
        rounded_rect(ax, col * cell_w, ROWS_AIE + 0.60, cell_w - 0.06, 0.55,
                     color=K_COLS[col], edge=WHITE + "55", corner_r=0.05)
        lo, hi = col * TILE_K, (col + 1) * TILE_K
        add_label(ax, col * cell_w + cell_w / 2, ROWS_AIE + 0.875,
                  f"y[{lo}:{hi}]", color=BG, fs=6)
    ax.text(COLS_AIE * cell_w / 2, ROWS_AIE + 1.25,
            "Input vector y[K]", color=DIM, fontsize=7.5, ha="center")

    # ── A matrix blocks ───────────────────────────────────────────────────────
    for row in range(ROWS_AIE):
        for col in range(COLS_AIE):
            bx = col * cell_w
            by = (ROWS_AIE - 1 - row) * cell_h
            rounded_rect(ax, bx, by, cell_w - 0.06, cell_h - 0.06,
                         color=tile_fill(row, col), edge=WHITE + "55", corner_r=0.05)
            kl, kh = col * TILE_K, (col + 1) * TILE_K
            nl, nh = row * TILE_N, (row + 1) * TILE_N
            add_label(ax, bx + cell_w / 2, by + cell_h / 2,
                      f"A[{kl}:{kh},\n {nl}:{nh}]", color=TEXT, fs=5.5)

    # Row bracket labels
    for row in range(ROWS_AIE):
        by = (ROWS_AIE - 1 - row) * cell_h + cell_h / 2
        ax.text(COLS_AIE * cell_w + 0.10, by,
                f"x[{row*TILE_N}:{(row+1)*TILE_N}]",
                color=N_COLS[row], fontsize=6.5, fontweight="bold",
                ha="left", va="center")

    # Dimension labels
    ax.text(COLS_AIE * cell_w / 2, -0.30,
            f"K = {K}  (split into {COLS_AIE} blocks of {TILE_K})",
            color=DIM, fontsize=6.5, ha="center")
    ax.text(-0.25, ROWS_AIE * cell_h / 2,
            f"N={N}\n({ROWS_AIE}×{TILE_N})",
            color=DIM, fontsize=6.5, ha="center", va="center", rotation=90)

    ax.text(COLS_AIE * cell_w / 2, ROWS_AIE * cell_h + 0.45,
            "Matrix A[K × N]", color=DIM, fontsize=7.5, ha="center")


# ═════════════════════════════════════════════════════════════════════════════
# Main figure assembly
# ═════════════════════════════════════════════════════════════════════════════

def build_figure():
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "Blocked Matrix–Vector Multiply on AMD AIE Spatial Array  ·  "
        f"y[{K}] × A[{K}×{N}] → x[{N}]  ·  "
        f"{ROWS_AIE}×{COLS_AIE} tile cascade",
        color=TEXT, fontsize=14, fontweight="bold", y=0.985,
    )

    # Layout:  row 0 = memory hierarchy (full width)
    #          row 1 = AIE array (left 3/4) + blocking diagram (right 1/4)
    #          row 2 = timeline   (full width)
    gs = GridSpec(3, 4, figure=fig,
                  left=0.05, right=0.97,
                  top=0.96, bottom=0.04,
                  hspace=0.42, wspace=0.28,
                  height_ratios=[1.1, 2.8, 1.5])

    ax_mem  = fig.add_subplot(gs[0, :])        # memory hierarchy (top, full width)
    ax_aie  = fig.add_subplot(gs[1, :3])       # AIE array (middle-left)
    ax_blk  = fig.add_subplot(gs[1,  3])       # blocking schematic (middle-right)
    ax_time = fig.add_subplot(gs[2, :])        # timeline (bottom, full width)

    for ax in [ax_mem, ax_aie, ax_blk, ax_time]:
        ax.set_facecolor(PANEL)

    draw_memory_hierarchy(ax_mem)
    draw_aie_array(ax_aie)
    draw_blocking_diagram(ax_blk)
    draw_dataflow_timeline(ax_time)

    # ── Annotation: key insight ───────────────────────────────────────────────
    insight = (
        "Key insight: Each AIE column holds one K-slice of A in fast L1 SRAM "
        "and a matching slice of y — all columns operate in parallel, "
        "then partial sums cascade right-to-right, hiding memory latency "
        "behind arithmetic throughput."
    )
    fig.text(0.50, 0.008, "\n".join(textwrap.wrap(insight, width=130)),
             color=DIM, fontsize=7.5, ha="center", va="bottom", style="italic")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", default="aie_matvec_dataflow.png",
                        help="Output file (PNG, SVG, or PDF). Default: %(default)s")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open an interactive window; just save.")
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
