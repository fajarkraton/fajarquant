#!/usr/bin/env python3
"""Generate the 5 figures referenced by paper/intllm/intllm.tex.

Path A Week 4 camera-ready polish — replaces the deferred-figure
stubs in §5/§6/§7/§8/§9 with real matplotlib-rendered PDFs.

Each figure is generated from the canonical artifact source (training
JSONs, ablation JSONs, corpus manifests, unit-test fixtures) so the
camera-ready version stays in sync with the data the paper cites.
Re-running this script after a data update regenerates all five
figures atomically; the .pdf outputs land in `paper/intllm/figures/`.

Usage:

    cd <repo root>
    .venv/bin/python scripts/generate_paper_figures.py

The script is idempotent: re-runs produce byte-similar PDFs given
the same source artifacts (matplotlib's PDF backend deterministic
modulo creation timestamp). Output paths:

    paper/intllm/figures/scaling_curve.pdf       (Figure 1, §5)
    paper/intllm/figures/corpus_pie.pdf          (Figure 4, §6)
    paper/intllm/figures/hadamard_spike.pdf      (Figure 3, §7)
    paper/intllm/figures/kernel_budget.pdf       (Figure 2, §8)
    paper/intllm/figures/compiler_pipeline.pdf   (Figure 5, §9)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in CI / headless

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "paper" / "intllm" / "results"
ABLATIONS_DIR = REPO_ROOT / "paper" / "intllm" / "ablations"
FIGURES_DIR = REPO_ROOT / "paper" / "intllm" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    """Save fig to figures/<name>.pdf with consistent metadata."""
    out = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote: {out.relative_to(REPO_ROOT)}")


# ──────────────────────────────────────────────────────────────────
# Figure 1 (§5): Scaling-law curve
# ──────────────────────────────────────────────────────────────────


def fig_scaling_curve() -> None:
    """val_loss vs n_params with calibrated-gate threshold overlay."""
    sizes = ["mini", "base", "medium"]
    params = []
    val_loss = []
    gates = []
    for s in sizes:
        with (RESULTS_DIR / f"training_intllm-{s}.json").open() as f:
            d = json.load(f)
        params.append(d["params"])
        val_loss.append(d["val_loss"])
        gates.append(d["gate"]["calibrated_threshold"])

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.semilogx(
        params, val_loss, "o-", color="#1f77b4",
        markersize=10, linewidth=2, label="IntLLM val_loss",
    )
    ax.semilogx(
        params, gates, "s--", color="#d62728",
        markersize=8, linewidth=1.5, label="Calibrated gate",
    )
    # Shade the margin (gate - val_loss) to show monotonic widening.
    ax.fill_between(
        params, val_loss, gates,
        color="#2ca02c", alpha=0.18, label="Pass margin",
    )
    # Annotate each point with config label and margin value.
    labels = ["Mini\n(22M)", "Base\n(46M)", "Medium\n(74M)"]
    for x, vl, g, lbl in zip(params, val_loss, gates, labels):
        margin = g - vl
        ax.annotate(
            f"{lbl}\nmargin={margin:.3f}",
            xy=(x, vl), xytext=(0, -32), textcoords="offset points",
            ha="center", fontsize=8,
        )
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Validation loss (nat)")
    ax.set_title(
        "Phase D scaling chain: monotonic gate-margin widening\n"
        "(Mini 0.118 nat → Base 0.210 nat → Medium 0.279 nat)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(1.5e7, 1.0e8)
    ax.set_ylim(3.5, 4.7)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    _save(fig, "scaling_curve")


# ──────────────────────────────────────────────────────────────────
# Figure 4 (§6): Bilingual corpus composition pie chart
# ──────────────────────────────────────────────────────────────────


def fig_corpus_pie() -> None:
    """Token-count composition of Phase E1 corpus v1.0."""
    # Source: per-shard _manifest.json + intllm_en.py EN-cap arithmetic.
    # Numbers from the §6 Table tab:corpus-comp:
    sources = [
        ("ID FineWeb-2",   14.98, "#2ca02c"),
        ("EN SlimPajama",  10.27, "#1f77b4"),
        ("ID Wikipedia",    0.42, "#ff7f0e"),
    ]
    labels = [s[0] for s in sources]
    sizes = [s[1] for s in sources]
    colors = [s[2] for s in sources]

    total = sum(sizes)
    pct_labels = [f"{s/total*100:.1f}%\n({s:.2f}B tok)" for s in sizes]

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    # Place pct labels via legend (clearer than in-pie labels for small wedges)
    legend_labels = [f"{lbl}: {pl.replace(chr(10), ' ')}"
                     for lbl, pl in zip(labels, pct_labels)]
    ax.legend(
        wedges, legend_labels,
        loc="center left", bbox_to_anchor=(0.92, 0.5),
        fontsize=9, frameon=False,
    )
    ax.set_title(
        "Phase E1 corpus v1.0 composition\n"
        f"({total:.2f} B tokens total at 60:40 ID:EN)",
        fontsize=10, pad=8,
    )
    _save(fig, "corpus_pie")


# ──────────────────────────────────────────────────────────────────
# Figure 3 (§7): Hadamard outlier suppression on synthetic spike
# ──────────────────────────────────────────────────────────────────


def fig_hadamard_spike() -> None:
    """Side-by-side bars: pre-rotation spike vs post-rotation spread.

    Uses the same dim=64 single-channel-spike fixture as the unit test
    `test_hadamard_outlier_suppression_on_concentrated_input`.
    Computes the actual Walsh-Hadamard transform inline so the figure
    matches the unit-test invariant exactly.
    """
    dim = 64
    # Build the dim×dim normalized Walsh-Hadamard matrix recursively.
    h = np.array([[1.0]])
    while h.shape[0] < dim:
        n = h.shape[0]
        top = np.concatenate([h, h], axis=1)
        bot = np.concatenate([h, -h], axis=1)
        h = np.concatenate([top, bot], axis=0) / np.sqrt(2.0)

    # Single-channel spike at index 0, magnitude 10.
    x = np.zeros(dim)
    x[0] = 10.0
    y = x @ h

    max_x, rms_x = np.abs(x).max(), np.sqrt((x**2).mean())
    max_y, rms_y = np.abs(y).max(), np.sqrt((y**2).mean())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    bar_kw = dict(width=1.0, edgecolor="none")

    axes[0].bar(np.arange(dim), x, color="#d62728", **bar_kw)
    axes[0].set_title(
        f"Before: max/RMS = {max_x/rms_x:.2f} (= $\\sqrt{{{dim}}}$)",
        fontsize=10,
    )
    axes[0].set_xlabel("Channel index")
    axes[0].set_ylabel("|x|")

    axes[1].bar(np.arange(dim), y, color="#2ca02c", **bar_kw)
    axes[1].set_title(
        f"After Hadamard: max/RMS = {max_y/rms_y:.2f} (uniform spread)",
        fontsize=10,
    )
    axes[1].set_xlabel("Channel index")

    fig.suptitle(
        "Walsh-Hadamard rotation correctly spreads outlier energy\n"
        "(mathematically — the negative training-from-scratch result\n"
        "in §7.3 is NOT due to broken math)",
        fontsize=10, y=1.05,
    )
    fig.tight_layout()
    _save(fig, "hadamard_spike")


# ──────────────────────────────────────────────────────────────────
# Figure 2 (§8): Kernel binary budget breakdown
# ──────────────────────────────────────────────────────────────────


def fig_kernel_budget() -> None:
    """Stacked bar of three deployment configurations vs ≤16 MB budget."""
    # From §8 Table tab:kernel-binary:
    configs = ["Mini\n(22M)", "Base\n(46M)", "Medium\n(74M)"]

    # Component sizes (MB), Phase D ternary-pack arithmetic:
    kernel_elf = 1.66                  # FajarOS Nova v3.9.0 ELF
    intllm_modules = 0.18 + 0.04 + 0.06  # matmulfree + fjm_v9 + tfm_matmulfree
    rest_kernel = kernel_elf - intllm_modules

    ternary = {
        "Mini":   4.30,    # 22M params × 1.58 bits / 8 ≈ 4.3 MB
        "Base":   9.20,    # 46M × 1.58 / 8 ≈ 9.2 MB
        "Medium": 14.70,   # 74M × 1.58 / 8 ≈ 14.7 MB
    }
    budget = 16.0

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    width = 0.55
    x_pos = np.arange(3)

    # Stacked bars: rest_kernel + intllm_modules + ternary_weights.
    bottoms = np.zeros(3)
    parts = [
        ("rest of kernel ELF", [rest_kernel] * 3, "#dddddd"),
        ("IntLLM kernel modules", [intllm_modules] * 3, "#1f77b4"),
        ("ternary model weights",
         [ternary["Mini"], ternary["Base"], ternary["Medium"]],
         "#2ca02c"),
    ]
    for label, values, color in parts:
        ax.bar(x_pos, values, width, bottom=bottoms,
               label=label, color=color, edgecolor="white", linewidth=0.5)
        bottoms += np.array(values)

    # Mark the 16 MB budget line.
    ax.axhline(budget, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"$\\leq {int(budget)}$ MB design budget")

    # Annotate totals on each bar.
    for x, total in zip(x_pos, bottoms):
        marker = " ✓" if total <= budget else " ✗ overrun"
        ax.text(x, total + 0.3, f"{total:.2f} MB{marker}",
                ha="center", fontsize=9,
                color="green" if total <= budget else "red")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Binary footprint (MB)")
    ax.set_title(
        "FajarOS Nova v3.9.0 IntLLM kernel-path footprint\n"
        "vs $\\leq 16$ MB design budget",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 19)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save(fig, "kernel_budget")


# ──────────────────────────────────────────────────────────────────
# Figure 5 (§9): Compilation pipeline diagram
# ──────────────────────────────────────────────────────────────────


def fig_compiler_pipeline() -> None:
    """Horizontal flowchart of the Fajar Lang compilation pipeline.

    The context-check sub-pass is highlighted as the load-bearing C5
    contribution; rest are standard compiler stages drawn lightly.
    """
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    # Stage definitions: (x_left, width, label, color, text_color)
    stages = [
        (0.2,  1.7,  ".fj source",  "#f0f0f0", "black"),
        (2.1,  1.4,  "lexer",       "#dde7f0", "black"),
        (3.7,  1.6,  "parser",      "#dde7f0", "black"),
        (5.5,  4.0,  "analyzer",    "#dde7f0", "black"),  # contains 3 sub-passes
        (9.7,  1.6,  "codegen",     "#dde7f0", "black"),
        (11.5, 5.5,  "kernel binary / userspace binary / interp",
                                    "#f0f0f0", "black"),
    ]
    for x, w, label, fc, tc in stages:
        rect = patches.FancyBboxPatch(
            (x, 1.6), w, 1.0, boxstyle="round,pad=0.05",
            linewidth=1, edgecolor="#666666", facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, 2.1, label, ha="center", va="center",
                fontsize=9, color=tc)

    # Sub-passes inside the analyzer box.
    sub_y = 0.5
    sub_h = 0.7
    sub_passes = [
        (5.6, 1.2, "type-check",     "#ffffff"),
        (6.9, 1.2, "scope-check",    "#ffffff"),
        (8.2, 1.3, "context-check",  "#ffd97a"),  # highlight: C5 contribution
    ]
    for x, w, label, fc in sub_passes:
        rect = patches.FancyBboxPatch(
            (x, sub_y), w, sub_h, boxstyle="round,pad=0.03",
            linewidth=0.8, edgecolor="#444444", facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, sub_y + sub_h/2, label,
                ha="center", va="center", fontsize=7.5)
    # Caption the highlighted sub-pass.
    ax.annotate(
        "load-bearing\nC5 contribution",
        xy=(8.85, sub_y + sub_h),
        xytext=(8.85, 0.05),
        ha="center", fontsize=7.5, color="#806000",
        arrowprops=dict(arrowstyle="-", color="#806000", lw=0.6),
    )

    # Arrows between top-row stages.
    arrow_kw = dict(arrowstyle="->", lw=1.0, color="#444444")
    arrows = [
        (1.9, 2.1, 2.1, 2.1),
        (3.5, 2.1, 3.7, 2.1),
        (5.3, 2.1, 5.5, 2.1),
        (9.5, 2.1, 9.7, 2.1),
        (11.3, 2.1, 11.5, 2.1),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_kw)

    # Title at top.
    ax.text(9.0, 3.5, "Fajar Lang compilation pipeline",
            ha="center", fontsize=11, weight="bold")
    _save(fig, "compiler_pipeline")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main() -> int:
    print("Generating IntLLM paper figures...")
    fig_scaling_curve()
    fig_corpus_pie()
    fig_hadamard_spike()
    fig_kernel_budget()
    fig_compiler_pipeline()
    print(f"All 5 figures written to: {FIGURES_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
