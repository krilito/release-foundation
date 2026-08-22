"""
116 - Plot a competitor-response panel for the release-intelligence stack.

Purpose:
    Turn the current competitor scorecard and Chinese battlecard into a single
    strategy-facing panel that shows:
    1. where representative external objects occupy the stack, and
    2. how we should respond to them: fight, borrow, or defend.

Consumes:
    docs/drug_release_stack_competitor_scorecard_2026-05-29.csv
    docs/drug_release_battlecard_zh_2026-05-29.csv

Produces:
    outputs/92_release_strategy_figures/competitor_response/competitor_response_panel.png
    outputs/92_release_strategy_figures/competitor_response/competitor_response_panel.pdf
    outputs/92_release_strategy_figures/competitor_response/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle


IN_SCORECARD = Path("docs/drug_release_stack_competitor_scorecard_2026-05-29.csv")
IN_BATTLECARD = Path("docs/drug_release_battlecard_zh_2026-05-29.csv")
OUTDIR = Path("outputs/92_release_strategy_figures/competitor_response")

LAYER_COLS = [
    "l1_schema",
    "l2_partial_observation_benchmark",
    "l3_shared_posterior_object",
    "l4_mechanism_decoder",
    "l5_reporting_decision_layer",
]

LAYER_LABELS = [
    "L1\nSchema",
    "L2\nPO\nbenchmark",
    "L3\nPosterior\nobject",
    "L4\nDecoder",
    "L5\nReport /\ndecision",
]

LEVEL_MAP = {
    "no": 0.0,
    "weak": 0.25,
    "partial": 0.5,
    "strong": 1.0,
    "partial_strong": 0.75,
}

ACTION_COLORS = {
    "self": "#4F46E5",
    "fight": "#C65A32",
    "borrow": "#2E8B57",
    "defend": "#0F766E",
}

DISPLAY_ORDER = [
    "NC/Bannigan few-shot and zero-shot ML",
    "Explainable LAI forecaster",
    "Scientific Data PLGA dataset",
    "Liposome IVR workflow",
    "FormulationLAI",
    "FormulationAI platform",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_name(name: str) -> str:
    mapping = {
        "Our current executable stack": "Our stack",
        "NC/Bannigan few-shot and zero-shot ML": "NC / Bannigan",
        "Explainable LAI forecaster": "Explainable LAI",
        "Scientific Data PLGA dataset": "Scientific Data",
        "Liposome IVR workflow": "Liposome IVR",
        "FormulationLAI": "FormulationLAI",
        "FormulationAI platform": "FormulationAI",
    }
    return mapping.get(name, name)


def family_to_action() -> dict[str, str]:
    return {
        "NC/Bannigan few-shot and zero-shot ML": "fight",
        "Explainable LAI forecaster": "fight",
        "Scientific Data PLGA dataset": "borrow",
        "Liposome IVR workflow": "borrow",
        "FormulationLAI": "defend",
        "FormulationAI platform": "defend",
    }


def load_heatmap_frame() -> pd.DataFrame:
    scorecard = pd.read_csv(IN_SCORECARD)
    scorecard = scorecard[scorecard["family"].isin(DISPLAY_ORDER)].copy()
    scorecard["family"] = pd.Categorical(scorecard["family"], DISPLAY_ORDER, ordered=True)
    scorecard = scorecard.sort_values("family").reset_index(drop=True)
    action_map = family_to_action()
    scorecard["action"] = scorecard["family"].map(action_map)
    return scorecard


def plot_heatmap(ax: plt.Axes, frame: pd.DataFrame) -> None:
    values = frame[LAYER_COLS].apply(lambda col: col.map(LEVEL_MAP)).to_numpy(dtype=float)
    labels = [short_name(x) for x in frame["family"]]
    cmap = LinearSegmentedColormap.from_list("occ", ["#F3F4F6", "#B6D7EA", "#6BA3CE", "#134E7A"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS, fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("A. Which external objects pressure which stack layers", loc="left", fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = str(frame.iloc[i][LAYER_COLS[j]]).replace("_", " ")
            color = "white" if values[i, j] >= 0.66 else "#1F2937"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

    for i, action in enumerate(frame["action"]):
        patch = Rectangle(
            (-1.95, i - 0.36),
            0.28,
            0.72,
            facecolor=ACTION_COLORS[action],
            edgecolor="none",
            transform=ax.transData,
            clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(
            -1.45,
            i,
            action.upper(),
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color=ACTION_COLORS[action],
            clip_on=False,
        )

    for boundary in [0.5, 2.5, 4.5]:
        ax.axhline(boundary, color="#D1D5DB", linewidth=1.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["no", "weak", "partial", "part-strong", "strong"])


def add_action_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    color: str,
    lines: list[str],
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="white",
    )
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y + h - 0.10), w, 0.10, facecolor=color, edgecolor="none"))
    ax.text(x + 0.02, y + h - 0.15, title, fontsize=12, fontweight="bold", color="#111827", va="top")
    cursor = y + h - 0.26
    for line in lines:
        ax.text(x + 0.025, cursor, line, fontsize=9.6, color="#374151", va="top")
        line_count = line.count("\n") + 1
        cursor -= 0.055 * line_count + 0.05


def plot_response_cards(ax: plt.Axes) -> None:
    ax.set_title("B. How we should respond", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_action_card(
        ax,
        0.02,
        0.10,
        0.30,
        0.78,
        "FIGHT",
        ACTION_COLORS["fight"],
        [
            "Objects:\nNC / Bannigan, Explainable LAI,\nbigger PLGA predictor",
            "Pressure:\ndirect benchmark identity and\nreviewer-friendly forecasting story",
            "Response:\nwin at L2-L3 with sparse-prefix OOD\nand family-object framing",
        ],
    )
    add_action_card(
        ax,
        0.35,
        0.10,
        0.30,
        0.78,
        "BORROW",
        ACTION_COLORS["borrow"],
        [
            "Objects:\nScientific Data, AI-ready release data,\nliposome IVR workflow",
            "Value:\nschema discipline, assay packaging,\nmechanism-local workflow habits",
            "Response:\nborrow L1/L4 discipline without\nshrinking back to direct regression",
        ],
    )
    add_action_card(
        ax,
        0.68,
        0.10,
        0.30,
        0.78,
        "DEFEND",
        ACTION_COLORS["defend"],
        [
            "Objects:\nFormulationLAI, FormulationAI,\noptimization systems",
            "Pressure:\nplatform usefulness and\ndecision-support narrative at L5",
            "Response:\nmove L2-L3 advantage toward\nrelease decisions and assay value",
        ],
    )


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 116 -- release competitor response panel ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Occupancy heatmap for representative external objects",
        "  B. Action cards for fight / borrow / defend",
        "",
        "Topline",
        "  Fight benchmark rivals at L2-L3, borrow schema/workflow discipline at L1/L4, and defend against platform narratives at L5.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    frame = load_heatmap_frame()

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.2, 11.4),
        gridspec_kw={"height_ratios": [1.35, 0.95], "hspace": 0.26},
    )
    plot_heatmap(axes[0], frame)
    plot_response_cards(axes[1])
    fig.subplots_adjust(left=0.15, right=0.95)
    fig.suptitle(
        "Competitor pressure, stack occupancy, and strategic response",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    out_png = OUTDIR / "competitor_response_panel.png"
    out_pdf = OUTDIR / "competitor_response_panel.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
