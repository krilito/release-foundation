"""
96 - Plot the external ML + drug-release competitive landscape.

Purpose:
    Turn the stack-level competitor scorecard into a visual landscape that
    separates benchmark pressure, platform pressure, and infrastructure
    pressure while showing who covers which layers of the unified
    release-intelligence stack.

Consumes:
    docs/drug_release_stack_competitor_scorecard_2026-05-29.csv

Produces:
    outputs/92_release_strategy_figures/competitor_landscape/competitor_landscape.png
    outputs/92_release_strategy_figures/competitor_landscape/competitor_landscape.pdf
    outputs/92_release_strategy_figures/competitor_landscape/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


INFILE = Path("docs/drug_release_stack_competitor_scorecard_2026-05-29.csv")
OUTDIR = Path("outputs/92_release_strategy_figures/competitor_landscape")


LEVEL_MAP = {"no": 0.0, "weak": 0.33, "partial": 0.66, "strong": 1.0}
THREAT_X = {
    "direct benchmark": 0.0,
    "bridge benchmark": 0.35,
    "adjacent hybrid": 0.55,
    "platform threat": 0.8,
    "field narrative": 0.9,
    "infrastructure": 1.15,
}
ACTION_Y = {"fight": 0.15, "borrow": 0.55, "defend and borrow": 0.9, "borrow and defend": 0.9}
ACTION_COLOR = {
    "fight": "#C03A2B",
    "borrow": "#2E8B57",
    "defend and borrow": "#7A78C9",
    "borrow and defend": "#7A78C9",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def shorten_family(name: str) -> str:
    mapping = {
        "Direct supervised PLGA predictor": "PLGA predictor",
        "Explainable LAI forecaster": "Explainable LAI",
        "NC/Bannigan few-shot and zero-shot ML": "NC / Bannigan",
        "Liposome IVR workflow": "Liposome IVR",
        "FormulationLAI": "FormulationLAI",
        "Hydrogel active learning": "Hydrogel AL",
        "Physics-informed nanocarrier design": "Physics-informed nano",
        "AI/ML optimization review": "Optimization review",
        "FormulationAI platform": "FormulationAI",
        "Scientific Data PLGA dataset": "Scientific Data PLGA",
        "AI-ready IVR/formulation data": "AI-ready IVR data",
    }
    return mapping.get(name, name)


def plot_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    cols = [
        "l1_schema",
        "l2_partial_observation_benchmark",
        "l3_shared_posterior_object",
        "l4_mechanism_decoder",
        "l5_reporting_decision_layer",
    ]
    col_labels = ["L1\nSchema", "L2\nPO benchmark", "L3\nPosterior object", "L4\nDecoder", "L5\nDecision/report"]
    values = df[cols].apply(lambda col: col.map(LEVEL_MAP)).to_numpy(dtype=float)
    row_labels = [shorten_family(x) for x in df["family"]]

    cmap = LinearSegmentedColormap.from_list("stack", ["#F2F2F2", "#B9D9F3", "#6FAAD8", "#1F5F99"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("A. External families across the unified-release stack", loc="left", fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                df.iloc[i][cols[j]].replace("_", " "),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if values[i, j] >= 0.66 else "#222222",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0.0, 0.33, 0.66, 1.0])
    cbar.set_ticklabels(["no", "weak", "partial", "strong"])


def plot_pressure_map(ax: plt.Axes, df: pd.DataFrame) -> None:
    counts: dict[str, int] = {}
    offsets = [-0.12, -0.06, 0.0, 0.06, 0.12, 0.18]

    for _, row in df.iterrows():
        key = row["threat_type"]
        idx = counts.get(key, 0)
        counts[key] = idx + 1
        x = THREAT_X[key] + offsets[min(idx, len(offsets) - 1)]
        y = ACTION_Y[row["borrow_or_fight"]]
        color = ACTION_COLOR[row["borrow_or_fight"]]
        ax.scatter(x, y, s=170, color=color, edgecolor="black", linewidth=0.7, alpha=0.9)
        ax.text(x + 0.02, y + 0.01, shorten_family(row["family"]), fontsize=8.2, ha="left", va="bottom")

    ax.axvspan(-0.08, 0.25, color="#F9E0DD", alpha=0.6)
    ax.axvspan(0.25, 0.95, color="#ECE8FA", alpha=0.45)
    ax.axvspan(0.95, 1.28, color="#E3F3E7", alpha=0.6)

    ax.text(0.08, 1.02, "Benchmark pressure", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.60, 1.02, "Workflow / platform pressure", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(1.12, 1.02, "Infrastructure pressure", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.08, 1.28)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([])
    ax.set_yticks([0.15, 0.55, 0.9])
    ax.set_yticklabels(["Fight", "Borrow", "Defend + borrow"])
    ax.set_title("B. What to fight, borrow, or defend against", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 96 -- release competitor landscape ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Heatmap of external families across the five-layer unified-release stack",
        "  B. Pressure map separating benchmark, platform/workflow, and infrastructure forces",
        "",
        "Interpretation",
        "  1. Benchmark rivals cluster around direct release prediction and only weakly cover the shared-object layers.",
        "  2. Platform and optimization systems threaten the translational narrative more than the core release-inference contract.",
        "  3. Infrastructure works are mostly borrowing targets that can strengthen intake and benchmark discipline.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    df = pd.read_csv(INFILE)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.32})

    plot_heatmap(axes[0], df)
    plot_pressure_map(axes[1], df)

    fig.suptitle(
        "External ML + drug-release landscape | The open slot is a unified release-intelligence stack",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    out_png = OUTDIR / "competitor_landscape.png"
    out_pdf = OUTDIR / "competitor_landscape.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
