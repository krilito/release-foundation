"""
95 - Plot Figure 1 for the current drug-release paper draft.

Purpose:
    Build the concept / framing figure that distinguishes the field default
    from the current release-intelligence formulation.

Consumes:
    docs/drug_release_paper_spine_and_figure_map_2026-05-29.md
    docs/release_posterior_family_schema_2026-05-29.md
    ARCHITECTURE.md

Produces:
    outputs/91_release_paper_figures/figure1/figure1_problem_framing.png
    outputs/91_release_paper_figures/figure1/figure1_problem_framing.pdf
    outputs/91_release_paper_figures/figure1/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTDIR = Path("outputs/91_release_paper_figures/figure1")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: int = 10,
    weight: str = "normal",
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#444444",
    text: str | None = None,
    text_offset: tuple[float, float] = (0.0, 0.03),
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=9, ha="center", va="center", color=color)


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A. Field default", loc="left", fontweight="bold", fontsize=13)

    add_box(ax, (0.10, 0.62), (0.28, 0.17), "Formulation /\ndrug descriptors", "#E8F1FA", fontsize=11, weight="bold")
    add_box(ax, (0.62, 0.62), (0.24, 0.17), "Direct release\nregressor", "#FBE6E6", fontsize=11, weight="bold")
    add_box(ax, (0.38, 0.22), (0.28, 0.17), "Pointwise or full\nrelease output", "#F6F2E8", fontsize=11, weight="bold")

    add_arrow(ax, (0.38, 0.70), (0.62, 0.70), text="train predictor")
    add_arrow(ax, (0.74, 0.62), (0.54, 0.39), text="predict Q(t)")

    ax.text(
        0.5,
        0.08,
        "Dominant external pattern:\ndescriptors -> release",
        ha="center",
        va="center",
        fontsize=10,
        color="#5A5A5A",
    )


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B. Our core object", loc="left", fontweight="bold", fontsize=13)

    add_box(ax, (0.05, 0.64), (0.28, 0.16), "Formulation /\nmaterial context", "#E8F1FA", fontsize=10, weight="bold")
    add_box(ax, (0.05, 0.34), (0.28, 0.16), "Sparse early release\nprefix", "#E6F5EA", fontsize=10, weight="bold")
    add_box(ax, (0.40, 0.49), (0.25, 0.18), "Feasible release-state\nfamily", "#FFF0C9", fontsize=10, weight="bold")
    add_box(ax, (0.72, 0.64), (0.21, 0.13), "Mechanism-specific\ndecoder", "#EDE5FA", fontsize=10, weight="bold")
    add_box(ax, (0.72, 0.44), (0.21, 0.13), "Future release\ncurve", "#F6F2E8", fontsize=10, weight="bold")
    add_box(ax, (0.72, 0.24), (0.21, 0.13), "Timing / shape /\nuncertainty", "#F6F2E8", fontsize=10, weight="bold")

    add_arrow(ax, (0.33, 0.72), (0.40, 0.61))
    add_arrow(ax, (0.33, 0.42), (0.40, 0.55))
    add_arrow(ax, (0.65, 0.58), (0.72, 0.70), text="decode")
    add_arrow(ax, (0.82, 0.64), (0.82, 0.57))
    add_arrow(ax, (0.82, 0.44), (0.82, 0.37))

    ax.text(
        0.52,
        0.24,
        "Not one universal theta.\nA shared feasible-state object under\npartial observation.",
        ha="center",
        va="center",
        fontsize=10,
        color="#6B5A00",
    )


def draw_panel_c(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("C. What is actually unified", loc="left", fontweight="bold", fontsize=13)

    y_positions = [0.77, 0.61, 0.45, 0.29, 0.13]
    labels = [
        "Shared curve /\nmetadata schema",
        "Shared partially observed\nbenchmark contract",
        "Shared posterior-family\nobject",
        "Mechanism-specific\ndecoders",
        "Shared decision-support /\nreporting layer",
    ]
    colors = ["#E8F1FA", "#E6F5EA", "#FFF0C9", "#EDE5FA", "#F6F2E8"]

    for y, label, color in zip(y_positions, labels, colors):
        add_box(ax, (0.17, y), (0.46, 0.11), label, color, fontsize=10, weight="bold")

    for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
        add_arrow(ax, (0.40, y1), (0.40, y2 + 0.11), color="#666666")

    add_box(ax, (0.72, 0.66), (0.18, 0.10), "PLGA", "#F4F4F4", fontsize=10, weight="bold")
    add_box(ax, (0.72, 0.49), (0.18, 0.10), "Liposome", "#F4F4F4", fontsize=10, weight="bold")
    add_box(ax, (0.72, 0.32), (0.18, 0.10), "Chitosan", "#F4F4F4", fontsize=10, weight="bold")
    add_arrow(ax, (0.63, 0.345), (0.72, 0.71), color="#7A78C9")
    add_arrow(ax, (0.63, 0.345), (0.72, 0.54), color="#7A78C9")
    add_arrow(ax, (0.63, 0.345), (0.72, 0.37), color="#7A78C9")

    ax.text(
        0.5,
        0.03,
        "Bold but legal now: unified drug-release intelligence.\nNot legal yet: universal solved release model.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#7A1F1F",
    )


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 95 -- release paper Figure 1 ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Field default: descriptors -> release",
        "  B. Core object: formulation/material + sparse prefix -> feasible state family -> decoder -> release outputs",
        "  C. Unification stack: shared schema, shared benchmark, shared posterior-family object, mechanism-specific decoders, shared reporting layer",
        "",
        "Interpretation",
        "  1. The paper introduces a different problem formulation, not just another regressor.",
        "  2. What is shared across mechanisms is the interface and posterior-family contract, not one literal mechanism parameterization.",
        "  3. The figure should not be used to claim empirical superiority or solved cross-mechanism validation by itself.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    plt.rcParams.update({"font.size": 10})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    fig.suptitle(
        "Figure 1 | Controlled release can be reframed as a shared partially observed inference problem",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    out_png = OUTDIR / "figure1_problem_framing.png"
    out_pdf = OUTDIR / "figure1_problem_framing.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
