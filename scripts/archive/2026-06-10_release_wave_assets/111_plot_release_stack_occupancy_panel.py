"""
111 - Plot the paper-facing release stack occupancy panel.

Purpose:
    Turn the stack occupancy paper asset into a compact supplementary /
    strategy figure that compares our current executable stack with a small
    set of representative external families on the same five-layer surface.

Consumes:
    outputs/110_release_stack_occupancy_paper_asset/panel_stack_occupancy.csv
    outputs/110_release_stack_occupancy_paper_asset/panel_layer_advantage.csv

Produces:
    outputs/92_release_strategy_figures/stack_occupancy/stack_occupancy_panel.png
    outputs/92_release_strategy_figures/stack_occupancy/stack_occupancy_panel.pdf
    outputs/92_release_strategy_figures/stack_occupancy/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


IN_OCC = Path("outputs/110_release_stack_occupancy_paper_asset/panel_stack_occupancy.csv")
IN_ADV = Path("outputs/110_release_stack_occupancy_paper_asset/panel_layer_advantage.csv")
OUTDIR = Path("outputs/92_release_strategy_figures/stack_occupancy")


LEVEL_MAP = {
    "no": 0.0,
    "weak": 0.25,
    "partial": 0.5,
    "strong": 1.0,
    "partial_strong": 0.75,
}

LAYER_COLS = [
    "l1_schema",
    "l2_partial_observation_benchmark",
    "l3_shared_posterior_object",
    "l4_mechanism_decoder",
    "l5_reporting_decision_layer",
]

LAYER_LABELS = ["L1\nSchema", "L2\nPO\nbenchmark", "L3\nPosterior\nobject", "L4\nDecoder", "L5\nReport /\ndecision"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def shorten(name: str) -> str:
    mapping = {
        "Our current executable stack": "Our stack",
        "Explainable LAI forecaster": "Explainable LAI",
        "NC/Bannigan few-shot and zero-shot ML": "NC / Bannigan",
        "Liposome IVR workflow": "Liposome IVR",
        "FormulationLAI": "FormulationLAI",
        "FormulationAI platform": "FormulationAI",
        "Scientific Data PLGA dataset": "Scientific Data PLGA",
    }
    return mapping.get(name, name)


def plot_heatmap(ax: plt.Axes, occ: pd.DataFrame) -> None:
    values = occ[LAYER_COLS].apply(lambda col: col.map(LEVEL_MAP)).to_numpy(dtype=float)
    labels = [shorten(x) for x in occ["family"]]
    cmap = LinearSegmentedColormap.from_list("occ", ["#F3F4F6", "#B6D7EA", "#6BA3CE", "#134E7A"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS, fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("A. Stack occupancy across representative external families", loc="left", fontweight="bold")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = str(occ.iloc[i][LAYER_COLS[j]]).replace("_", " ")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if values[i, j] >= 0.66 else "#222")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["no", "weak", "partial", "part-strong", "strong"])


def plot_advantage(ax: plt.Axes, adv: pd.DataFrame) -> None:
    x = np.arange(len(adv))
    vals = adv["advantage_vs_best_external"].to_numpy(dtype=float)
    colors = ["#2E8B57" if v >= 0 else "#C03A2B" for v in vals]
    ax.bar(x, vals, color=colors, width=0.66)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"], fontsize=9)
    ax.set_ylabel("Advantage vs best external")
    ax.set_title("B. Where the current stack is structurally ahead or still qualified", loc="left", fontweight="bold")
    for xi, yi in zip(x, vals):
        ax.text(xi, yi + (0.03 if yi >= 0 else -0.05), f"{yi:+.2f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 111 -- release stack occupancy panel ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Occupancy heatmap comparing our executable stack to representative external families",
        "  B. Layer-level advantage vs best external family",
        "",
        "Topline",
        "  The strongest current edge is at L2-L3; L5 remains the most qualified layer.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    occ = pd.read_csv(IN_OCC)
    adv = pd.read_csv(IN_ADV)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.5), gridspec_kw={"height_ratios": [1.5, 0.9], "hspace": 0.30})
    plot_heatmap(axes[0], occ)
    plot_advantage(axes[1], adv)
    fig.suptitle(
        "Executable release-intelligence stack occupancy vs representative external families",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    out_png = OUTDIR / "stack_occupancy_panel.png"
    out_pdf = OUTDIR / "stack_occupancy_panel.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
