"""
93 - Plot Figure 3 for the current drug-release paper draft.

Purpose:
    Build the timing/shape honesty figure from the assembled Figure 3 panel
    assets produced by `90_release_paper_figure_assets.py`.

Consumes:
    outputs/90_release_paper_figure_assets/figure3/panelA_plga_timescale_tallies.csv
    outputs/90_release_paper_figure_assets/figure3/panelB_plga_shape_tallies.csv
    outputs/90_release_paper_figure_assets/figure3/panelC_representative_divergence.csv

Produces:
    outputs/91_release_paper_figures/figure3/figure3_timing_shape_honesty.png
    outputs/91_release_paper_figures/figure3/figure3_timing_shape_honesty.pdf
    outputs/91_release_paper_figures/figure3/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INROOT = Path("outputs/90_release_paper_figure_assets/figure3")
OUTDIR = Path("outputs/91_release_paper_figures/figure3")

PANEL_A = INROOT / "panelA_plga_timescale_tallies.csv"
PANEL_B = INROOT / "panelB_plga_shape_tallies.csv"
PANEL_C = INROOT / "panelC_representative_divergence.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_tally_panel(ax: plt.Axes, df: pd.DataFrame, title: str, color: str) -> None:
    labels = list(df["metric"])
    wins = df["mechanism_wins"].to_numpy()
    totals = df["n_cells"].to_numpy()
    losses = totals - wins
    y = np.arange(len(labels))

    ax.barh(y, totals, color="#E8E8E8", edgecolor="black", linewidth=0.6)
    ax.barh(y, wins, color=color, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, int(totals.max()))
    ax.set_xticks(np.arange(0, int(totals.max()) + 1, 1))
    ax.set_xlabel("Mechanism wins out of 4 cells")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    for yi, win, total, loss in zip(y, wins, totals, losses):
        ax.text(
            total + 0.05,
            yi,
            f"{int(win)}/{int(total)}",
            va="center",
            fontsize=9,
            color=color if win >= loss else "#7A1F1F",
        )


def plot_panel_c(ax_left: plt.Axes, ax_right: plt.Axes, row: pd.Series) -> None:
    curve_labels = ["Pooled $R^2$", "Median curve $R^2$"]
    curve_mech = [row["mechanism_pooled_r2"], row["mechanism_median_curve_r2"]]
    curve_direct = [row["direct_pooled_r2"], row["direct_median_curve_r2"]]
    x_curve = np.arange(len(curve_labels))
    width = 0.34

    ax_left.bar(x_curve - width / 2, curve_mech, width=width, color="#2C7FB8", edgecolor="black", linewidth=0.6, label="Mechanism route")
    ax_left.bar(x_curve + width / 2, curve_direct, width=width, color="#B34D4D", edgecolor="black", linewidth=0.6, label="Direct route")
    ax_left.set_xticks(x_curve)
    ax_left.set_xticklabels(curve_labels, fontsize=10)
    ax_left.set_ylim(0.0, 1.02)
    ax_left.set_ylabel("$R^2$")
    ax_left.set_title("C. Representative PLGA cell: curve space", loc="left", fontweight="bold")
    ax_left.legend(frameon=False, fontsize=9, loc="upper left")

    timing_labels = ["Future t10", "Future t50", "Future t80"]
    timing_mech = [
        row["future_t10_mae_mechanism_d"],
        row["future_t50_mae_mechanism_d"],
        row["future_t80_mae_mechanism_d"],
    ]
    timing_direct = [
        row["future_t10_mae_direct_d"],
        row["future_t50_mae_direct_d"],
        row["future_t80_mae_direct_d"],
    ]
    x_timing = np.arange(len(timing_labels))

    ax_right.bar(x_timing - width / 2, timing_mech, width=width, color="#2C7FB8", edgecolor="black", linewidth=0.6)
    ax_right.bar(x_timing + width / 2, timing_direct, width=width, color="#B34D4D", edgecolor="black", linewidth=0.6)
    ax_right.set_xticks(x_timing)
    ax_right.set_xticklabels(timing_labels, fontsize=10)
    ax_right.set_ylabel("MAE (days)")
    ax_right.set_title("Same cell: future timing error", loc="left", fontweight="bold")
    ax_right.text(
        0.01,
        0.97,
        "Cell: cross321 / group-by-polymer / formulation+early",
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 93 -- release paper Figure 3 ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. PLGA timing tallies across four canonical OOD cells",
        "  B. PLGA shape tallies across the same four cells",
        "  C. One representative divergence cell split into curve-space and future timing views",
        "",
        "Interpretation",
        "  1. Mechanism route wins the audited PLGA curve-space center more consistently than it wins duration metrics.",
        "  2. Post-window and residual-tail shape objects favor the mechanism route more cleanly than t50/t80 timing.",
        "  3. The representative cell makes the core honesty point explicit: curve fit and release-duration accuracy are not the same target.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    timescale = pd.read_csv(PANEL_A)
    shape = pd.read_csv(PANEL_B)
    divergence = pd.read_csv(PANEL_C).iloc[0]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.05], hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    sub = gs[1, :].subgridspec(1, 2, wspace=0.28)
    ax_c1 = fig.add_subplot(sub[0, 0])
    ax_c2 = fig.add_subplot(sub[0, 1])

    plot_tally_panel(ax_a, timescale, "A. Timing tallies across PLGA OOD cells", "#E3872D")
    plot_tally_panel(ax_b, shape, "B. Shape tallies across PLGA OOD cells", "#2E8B57")
    plot_panel_c(ax_c1, ax_c2, divergence)

    fig.suptitle(
        "Figure 3 | Curve fit, duration, and shape do not share a universal winning route",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_png = OUTDIR / "figure3_timing_shape_honesty.png"
    out_pdf = OUTDIR / "figure3_timing_shape_honesty.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
