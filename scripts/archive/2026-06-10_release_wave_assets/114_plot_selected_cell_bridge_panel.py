"""
114 - Plot the selected-cell bridge summary panel.

Purpose:
    Turn the selected-cell family-duration/shape bridge summaries into a
    compact strategy/support figure that makes the main pattern visually
    obvious: shape-aware uncertainty is materially more mature than
    deep-threshold timing uncertainty.

Consumes:
    outputs/101_selected_cell_bridge_summary_v2/pattern_snapshot.csv
    outputs/101_selected_cell_bridge_summary_v2/threshold_cross_cell.csv
    outputs/101_selected_cell_bridge_summary_v2/descriptor_cross_cell.csv

Produces:
    outputs/92_release_strategy_figures/selected_cell_bridge/selected_cell_bridge_panel.png
    outputs/92_release_strategy_figures/selected_cell_bridge/selected_cell_bridge_panel.pdf
    outputs/92_release_strategy_figures/selected_cell_bridge/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IN_PATTERN = Path("outputs/101_selected_cell_bridge_summary_v2/pattern_snapshot.csv")
IN_THRESH = Path("outputs/101_selected_cell_bridge_summary_v2/threshold_cross_cell.csv")
IN_DESC = Path("outputs/101_selected_cell_bridge_summary_v2/descriptor_cross_cell.csv")
OUTDIR = Path("outputs/92_release_strategy_figures/selected_cell_bridge")

METHOD_ORDER = ["raw", "local", "global"]
METHOD_COLORS = {"raw": "#C97A38", "local": "#4C78A8", "global": "#2E8B57"}
COMPONENT_ORDER = ["t10", "t50", "t80", "burst", "post_window", "residual_tail", "tail_auc"]
COMPONENT_LABELS = ["t10", "t50", "t80", "burst", "post-window", "residual-tail", "tail-AUC"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_heatmap_frame() -> pd.DataFrame:
    thresh = pd.read_csv(IN_THRESH)
    desc = pd.read_csv(IN_DESC)
    thresh = thresh.assign(component=thresh["threshold"].map({0.1: "t10", 0.5: "t50", 0.8: "t80"}))
    thresh = thresh[["cell_label", "method", "component", "coverage"]]
    desc = desc.rename(columns={"descriptor": "component"})[["cell_label", "method", "component", "coverage"]]
    both = pd.concat([thresh, desc], ignore_index=True)
    global_df = both[both["method"] == "global"].copy()
    global_df["component"] = pd.Categorical(global_df["component"], categories=COMPONENT_ORDER, ordered=True)
    return global_df.sort_values(["cell_label", "component"])


def plot_bar_panel(ax: plt.Axes, pattern: pd.DataFrame) -> None:
    x = np.arange(len(COMPONENT_ORDER))
    width = 0.23
    for idx, method in enumerate(METHOD_ORDER):
        sub = pattern[pattern["method"] == method].copy()
        sub["component"] = pd.Categorical(sub["component"], categories=COMPONENT_ORDER, ordered=True)
        sub = sub.sort_values("component")
        vals = sub["mean_coverage_across_cells"].to_numpy(dtype=float)
        offset = (idx - 1) * width
        ax.bar(x + offset, vals, width=width, color=METHOD_COLORS[method], label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(COMPONENT_LABELS, rotation=0, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean coverage across selected cells")
    ax.set_title("A. Selected-cell bridge: shape coverage stays far above deep-threshold timing", loc="left", fontweight="bold")
    ax.axhline(0.9, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(-0.45, 0.915, "0.9", fontsize=8, color="#666666")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_heatmap(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    cell_order = list(dict.fromkeys(global_df["cell_label"]))
    mat = (
        global_df.pivot(index="cell_label", columns="component", values="coverage")
        .reindex(index=cell_order, columns=COMPONENT_ORDER)
        .to_numpy(dtype=float)
    )
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(COMPONENT_ORDER)))
    ax.set_xticklabels(COMPONENT_LABELS, fontsize=9)
    ax.set_yticks(np.arange(len(cell_order)))
    ax.set_yticklabels(cell_order, fontsize=9)
    ax.set_title("B. Global-method coverage by selected cell: timing collapses first, shape survives longer", loc="left", fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            txt_color = "white" if value >= 0.55 else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1.0"])


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 114 -- selected-cell bridge panel ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Mean coverage across selected cells for timing thresholds and shape descriptors",
        "  B. Global-method per-cell heatmap across selected PLGA and liposome bridge cells",
        "",
        "Topline",
        "  The selected-cell bridge already supports shape-aware uncertainty much more strongly than deep-threshold timing uncertainty.",
        "  This strengthens the case for a shared shape-aware target layer while keeping universal deep-threshold timing claims off-limits.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    pattern = pd.read_csv(IN_PATTERN)
    global_df = load_heatmap_frame()

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.8), gridspec_kw={"height_ratios": [1.0, 1.1], "hspace": 0.28})
    plot_bar_panel(axes[0], pattern)
    plot_heatmap(axes[1], global_df)
    fig.suptitle(
        "Selected-cell family bridge | shape-aware uncertainty is materially ahead of deep-threshold timing",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    out_png = OUTDIR / "selected_cell_bridge_panel.png"
    out_pdf = OUTDIR / "selected_cell_bridge_panel.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
