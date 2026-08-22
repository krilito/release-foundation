"""
92 - Plot Figure 4 for the current drug-release paper draft.

Purpose:
    Build the first non-PLGA bridge figure from the assembled Figure 4 panel
    assets produced by `90_release_paper_figure_assets.py`.

Consumes:
    outputs/90_release_paper_figure_assets/figure4/panelAB_liposome_bridge_metrics.csv
    outputs/90_release_paper_figure_assets/figure4/panelC_liposome_shape_metrics.csv

Produces:
    outputs/91_release_paper_figures/figure4/figure4_liposome_bridge.png
    outputs/91_release_paper_figures/figure4/figure4_liposome_bridge.pdf
    outputs/91_release_paper_figures/figure4/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


INROOT = Path("outputs/90_release_paper_figure_assets/figure4")
OUTDIR = Path("outputs/91_release_paper_figures/figure4")

PANEL_AB = INROOT / "panelAB_liposome_bridge_metrics.csv"
PANEL_C = INROOT / "panelC_liposome_shape_metrics.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    groups = [
        ("Group by API", "pooled_r2", "API\nPooled"),
        ("Group by API", "median_curve_r2", "API\nMedian"),
        ("Group by release method", "pooled_r2", "Method\nPooled"),
        ("Group by release method", "median_curve_r2", "Method\nMedian"),
    ]
    x = np.arange(len(groups))
    width = 0.34
    mech_vals = []
    dir_vals = []
    labels = []

    for scheme, metric, label in groups:
        row = df[df["scheme_label"] == scheme].iloc[0]
        mech_vals.append(row[f"mechanism_{metric}"])
        dir_vals.append(row[f"direct_{metric}"])
        labels.append(label)

    ax.bar(x - width / 2, mech_vals, width=width, color="#2C7FB8", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, dir_vals, width=width, color="#B34D4D", edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("$R^2$")
    ax.set_title("A. Liposome bridge in curve space", loc="left", fontweight="bold")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.text(0.5, -0.18, "Group by API", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.text(0.82, -0.18, "Group by release method", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2C7FB8", ec="black", lw=0.6, label="Mechanism route"),
            plt.Rectangle((0, 0), 1, 1, color="#B34D4D", ec="black", lw=0.6, label="Direct route"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    thresholds = [("t10", "t10 MAE"), ("t50", "t50 MAE"), ("t80", "t80 MAE")]
    schemes = list(df["scheme_label"])
    x = np.arange(len(thresholds))
    width = 0.18
    colors = {"Group by API": "#5AA469", "Group by release method": "#7A78C9"}

    for j, scheme in enumerate(schemes):
        row = df[df["scheme_label"] == scheme].iloc[0]
        mech = [row[f"{thr}_mae_mechanism_h"] for thr, _ in thresholds]
        direct = [row[f"{thr}_mae_direct_h"] for thr, _ in thresholds]
        base = x + (j - 0.5) * (width * 2.6)
        ax.bar(base, mech, width=width, color=colors[scheme], edgecolor="black", linewidth=0.6, alpha=0.95)
        ax.bar(
            base + width,
            direct,
            width=width,
            color=colors[scheme],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.35,
        )
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels([lab for _, lab in thresholds], fontsize=10)
    ax.set_ylabel("MAE (hours)")
    ax.set_title("B. Timing metrics remain target-dependent", loc="left", fontweight="bold")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#5AA469", ec="black", lw=0.6, label="Group by API (mechanism)"),
            plt.Rectangle((0, 0), 1, 1, color="#5AA469", ec="black", lw=0.6, alpha=0.35, label="Group by API (direct)"),
            plt.Rectangle((0, 0), 1, 1, color="#7A78C9", ec="black", lw=0.6, label="Group by release method (mechanism)"),
            plt.Rectangle((0, 0), 1, 1, color="#7A78C9", ec="black", lw=0.6, alpha=0.35, label="Group by release method (direct)"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )


def plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    main_metrics = [
        ("burst_mae", "Burst"),
        ("post_window_release_mae", "Post-window"),
        ("residual_tail_mae", "Residual tail"),
    ]
    tail_metric = ("tail_auc_mae", "Tail AUC")
    x = np.arange(len(main_metrics))
    width = 0.18
    rows = {row["scheme_label"]: row for _, row in df.iterrows()}
    colors = {"Group by API": "#5AA469", "Group by release method": "#7A78C9"}

    for j, scheme in enumerate(["Group by API", "Group by release method"]):
        row = rows[scheme]
        mech = [row[f"{m}_mechanism"] for m, _ in main_metrics]
        direct = [row[f"{m}_direct"] for m, _ in main_metrics]
        base = x + (j - 0.5) * (width * 2.6)
        ax.bar(base, mech, width=width, color=colors[scheme], edgecolor="black", linewidth=0.6, alpha=0.95)
        ax.bar(
            base + width,
            direct,
            width=width,
            color=colors[scheme],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.35,
        )
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels([lab for _, lab in main_metrics], fontsize=9)
    ax.set_ylabel("Error")
    ax.set_title("C. Shape metrics split by scheme", loc="left", fontweight="bold")
    ax.set_ylim(0.0, 0.24)

    tail_ax = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=1.1)
    tail_x = np.arange(1)
    for j, scheme in enumerate(["Group by API", "Group by release method"]):
        row = rows[scheme]
        mech = row[f"{tail_metric[0]}_mechanism"]
        direct = row[f"{tail_metric[0]}_direct"]
        base = tail_x + (j - 0.5) * (width * 2.6)
        tail_ax.bar(base, [mech], width=width, color=colors[scheme], edgecolor="black", linewidth=0.6, alpha=0.95)
        tail_ax.bar(
            base + width,
            [direct],
            width=width,
            color=colors[scheme],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.35,
        )
    tail_ax.set_xticks(tail_x + width * 0.5)
    tail_ax.set_xticklabels([tail_metric[1]], fontsize=8)
    tail_ax.set_ylabel("Error", fontsize=8)
    tail_ax.tick_params(axis="both", labelsize=8)
    tail_ax.set_title("Tail AUC inset", fontsize=8)
    tail_ax.spines["top"].set_visible(False)
    tail_ax.spines["right"].set_visible(False)


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 92 -- release paper Figure 4 ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Liposome bridge in curve space for two split families",
        "  B. t10/t50/t80 timing MAE under the same two split families",
        "  C. Shape descriptors under the same two split families",
        "",
        "Interpretation",
        "  1. The shared interface survives a first non-PLGA retrospective bridge.",
        "  2. The mechanism route is much more stable in curve space under group-by-release-method.",
        "  3. t50 and some shape metrics remain route-dependent, so the figure should not be used to claim universal superiority.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    bridge = pd.read_csv(PANEL_AB)
    shape = pd.read_csv(PANEL_C)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.05, 1.0], hspace=0.35, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    plot_panel_a(ax_a, bridge)
    plot_panel_b(ax_b, bridge)
    plot_panel_c(ax_c, shape)

    fig.suptitle(
        "Figure 4 | A first non-PLGA bridge supports a shared release-intelligence interface",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_png = OUTDIR / "figure4_liposome_bridge.png"
    out_pdf = OUTDIR / "figure4_liposome_bridge.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
