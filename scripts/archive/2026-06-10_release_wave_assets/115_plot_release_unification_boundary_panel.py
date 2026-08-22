"""
115 - Plot the unification boundary panel for the current release-intelligence stack.

Purpose:
    Summarize, in one support figure, which layers of "unified drug release"
    are currently evidence-backed, which are only partially established, and
    which are still off-limits.

Consumes:
    outputs/82_release_capability_snapshot/summary.txt
    outputs/88_release_uncertainty_snapshot/summary.txt
    outputs/89_release_uq_timescale_gap_snapshot/summary.txt
    outputs/106_release_target_registry/summary.txt
    outputs/108_release_stack_manifest/stack_manifest_summary.csv

Produces:
    outputs/92_release_strategy_figures/unification_boundary/unification_boundary_panel.png
    outputs/92_release_strategy_figures/unification_boundary/unification_boundary_panel.pdf
    outputs/92_release_strategy_figures/unification_boundary/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


IN_STACK = Path("outputs/108_release_stack_manifest/stack_manifest_summary.csv")
OUTDIR = Path("outputs/92_release_strategy_figures/unification_boundary")


ROWS = [
    ("Shared schema + benchmark contract", 1.00, "evidence-backed"),
    ("First non-PLGA liposome bridge", 0.88, "bridge-backed"),
    ("Shared posterior-family object", 1.00, "evidence-backed"),
    ("9-cell calibrated-family panel", 0.90, "evidence-backed"),
    ("Shape-aware target layer", 0.82, "ahead"),
    ("Locked prospective discipline", 0.68, "locked"),
    ("Deep-threshold timing layer", 0.20, "weak"),
    ("L5 decision / reporting layer", 0.25, "weak"),
    ("Route-consistent benchmark-side duration UQ", 0.05, "missing"),
    ("Completed prospective cross-mechanism validation", 0.00, "not yet"),
]

STATUS_COLOR = {
    "evidence-backed": "#2E8B57",
    "bridge-backed": "#4C78A8",
    "ahead": "#1F7A8C",
    "locked": "#B07D2D",
    "weak": "#D97706",
    "missing": "#C03A2B",
    "not yet": "#7A7A7A",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_boundary(ax: plt.Axes) -> None:
    labels = [row[0] for row in ROWS]
    values = np.array([row[1] for row in ROWS], dtype=float)
    colors = [STATUS_COLOR[row[2]] for row in ROWS]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.0, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Current evidence-backed unification strength")
    ax.set_title("A. What is actually unified now, and what is still off-limits", loc="left", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for yi, val, row in zip(y, values, ROWS):
        ax.text(min(val + 0.02, 0.98), yi, row[2], va="center", ha="left", fontsize=8.2, color="#333333")


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    facecolor: str,
    *,
    line_font_size: float = 8.6,
    line_step: float = 0.073,
) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(x + 0.03, y + h - 0.06, title, ha="left", va="top", fontsize=11, fontweight="bold")
    cursor = y + h - 0.14
    for line in lines:
        ax.text(x + 0.03, cursor, line, ha="left", va="top", fontsize=line_font_size)
        cursor -= line_step


def plot_claim_boxes(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B. What we can say now vs what still needs another evidence jump", loc="left", fontweight="bold")

    can_say = [
        "unified drug-release intelligence",
        "shared release-intelligence stack",
        "shared posterior-family object",
        "first non-PLGA bridge",
    ]
    cannot_say = [
        "universal solved drug-release model",
        "universal t80",
        "benchmark-wide route-consistent duration UQ",
    ]
    next_unlock = [
        "route-consistent uncertainty-aware duration / shape",
        "completed chitosan reveal",
        "more non-PLGA bridge evidence beyond liposome",
    ]

    add_box(ax, 0.02, 0.12, 0.30, 0.72, "Can say now", can_say, "#E8F4EC", line_font_size=8.4, line_step=0.115)
    add_box(ax, 0.35, 0.12, 0.30, 0.72, "Cannot say yet", cannot_say, "#FBE9E7", line_font_size=8.4, line_step=0.135)
    add_box(ax, 0.68, 0.12, 0.30, 0.72, "Next real unlocks", next_unlock, "#EAF1FB", line_font_size=8.4, line_step=0.135)


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 115 -- unification boundary panel ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Bar chart of current evidence-backed unification strength across stack layers",
        "  B. Can-say vs cannot-say vs next-unlock claim boxes",
        "",
        "Topline",
        "  We can credibly claim a shared release-intelligence stack, but not a universal solved model.",
        "  Shape-aware target unification is materially ahead of deep-threshold timing.",
        "  The biggest remaining unlock is route-consistent uncertainty-aware duration / shape.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    _ = pd.read_csv(IN_STACK)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [1.22, 0.95], "hspace": 0.22})
    plot_boundary(axes[0])
    plot_claim_boxes(axes[1])
    fig.suptitle(
        "Unification boundary of the current drug-release program",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    out_png = OUTDIR / "unification_boundary_panel.png"
    out_pdf = OUTDIR / "unification_boundary_panel.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
