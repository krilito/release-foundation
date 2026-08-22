"""
94 - Plot Figure 5 for the current drug-release paper draft.

Purpose:
    Build the calibrated-family + prospective-discipline figure from the
    assembled Figure 5 panel assets produced by `90_release_paper_figure_assets.py`.

Consumes:
    outputs/90_release_paper_figure_assets/figure5/panelA_calibrated_family_panel.csv
    outputs/90_release_paper_figure_assets/figure5/panelB_representative_conformal_examples.csv
    outputs/90_release_paper_figure_assets/figure5/panelC_chitosan_lock_summary.csv

Produces:
    outputs/91_release_paper_figures/figure5/figure5_uq_and_prospective.png
    outputs/91_release_paper_figures/figure5/figure5_uq_and_prospective.pdf
    outputs/91_release_paper_figures/figure5/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INROOT = Path("outputs/90_release_paper_figure_assets/figure5")
OUTDIR = Path("outputs/91_release_paper_figures/figure5")

PANEL_A = INROOT / "panelA_calibrated_family_panel.csv"
PANEL_B = INROOT / "panelB_representative_conformal_examples.csv"
PANEL_C = INROOT / "panelC_chitosan_lock_summary.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cell_label(dataset: str, scheme: str) -> str:
    dataset_short = {"cross321": "C321", "internal181": "I181", "liposome": "Lipo"}[dataset]
    scheme_short = {
        "group_by_drug": "D",
        "group_by_polymer": "P",
        "random_5fold": "R",
    }[scheme]
    return f"{dataset_short}-{scheme_short}"


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df.copy()
    df["cell_label"] = [cell_label(d, s) for d, s in zip(df["dataset"], df["scheme"])]
    x = np.arange(len(df))
    width = 0.24

    ax.bar(x - width, df["cov90_raw_mean"], width=width, color="#D9D9D9", edgecolor="black", linewidth=0.5, label="Raw cov90")
    ax.bar(x, df["cov90_global_mean"], width=width, color="#2C7FB8", edgecolor="black", linewidth=0.5, label="Global conformal cov90")
    ax.bar(x + width, df["cov90_local_mean"], width=width, color="#7FCDBB", edgecolor="black", linewidth=0.5, label="Local conformal cov90")

    ax.axhline(0.90, color="#7A1F1F", linestyle="--", linewidth=1.2)
    ax.text(len(df) - 0.2, 0.905, "Nominal 90%", ha="right", va="bottom", fontsize=9, color="#7A1F1F")

    ax.axvline(2.5, color="#AAAAAA", linewidth=0.8)
    ax.axvline(5.5, color="#AAAAAA", linewidth=0.8)
    ax.text(1.0, 0.995, "cross321", ha="center", va="top", fontsize=9)
    ax.text(4.0, 0.995, "internal181", ha="center", va="top", fontsize=9)
    ax.text(7.0, 0.995, "liposome", ha="center", va="top", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["cell_label"], fontsize=9)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel("Coverage")
    ax.set_title("A. Nine-cell calibrated-family panel", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))


def plot_example_axis(ax: plt.Axes, row: pd.Series, title: str) -> None:
    labels = ["Raw", "Global", "Local"]
    coverage = [row["cov90_raw_mean"], row["cov90_global_mean"], row["cov90_local_mean"]]
    width = [row["width90_raw_median"], row["width90_global_median"], row["width90_local_median"]]
    x = np.arange(len(labels))
    bw = 0.34

    ax.bar(x - bw / 2, coverage, width=bw, color="#2C7FB8", edgecolor="black", linewidth=0.5, label="Coverage")
    ax2 = ax.twinx()
    ax2.bar(x + bw / 2, width, width=bw, color="#E3872D", edgecolor="black", linewidth=0.5, alpha=0.9, label="Median width")

    ax.axhline(0.90, color="#7A1F1F", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.65, 0.96)
    ax2.set_ylim(0.0, max(width) * 1.25)
    ax.set_ylabel("Coverage", fontsize=9)
    ax2.set_ylabel("Median width", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax2.tick_params(axis="y", labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")


def plot_panel_c(ax: plt.Axes, row: pd.Series) -> None:
    ax.axis("off")
    lines = [
        "C. Locked chitosan prospective stress test",
        "",
        f"Status: {row['status']}",
        f"Curves locked: {int(row['curve_count'])}",
        f"Formulations: {int(row['formulation_count'])}",
        f"Drugs: {int(row['drug_count'])}",
        f"Primary endpoint: {row['primary_endpoint']}",
        "",
        "Interpretation",
        "- this is a preregistered locked prediction package",
        "- it upgrades evidence discipline before reveal",
        "- it does not count as completed prospective success yet",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F7F7F7", "edgecolor": "#BBBBBB"},
    )


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 94 -- release paper Figure 5 ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. Nine-cell calibrated-family panel across PLGA and liposome",
        "  B. Representative conformal examples showing coverage-width tradeoff",
        "  C. Locked chitosan prospective discipline summary",
        "",
        "Interpretation",
        "  1. Shared calibrated-family artifacts now exist beyond a single PLGA split.",
        "  2. Conformal repair improves coverage but widens intervals, so the figure should not be read as solved UQ.",
        "  3. The chitosan panel documents prospective discipline, not revealed prospective success.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    panel = pd.read_csv(PANEL_A)
    examples = pd.read_csv(PANEL_B)
    chitosan = pd.read_csv(PANEL_C).iloc[0]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.0], hspace=0.42, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, :])
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.45)
    ax_b1 = fig.add_subplot(sub[0, 0])
    ax_b2 = fig.add_subplot(sub[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, panel)

    ex_plga = examples[(examples["dataset"] == "cross321") & (examples["scheme"] == "group_by_drug")].iloc[0]
    ex_lipo = examples[(examples["dataset"] == "liposome") & (examples["scheme"] == "group_by_drug")].iloc[0]
    plot_example_axis(ax_b1, ex_plga, "B1. cross321 / group-by-drug")
    plot_example_axis(ax_b2, ex_lipo, "B2. liposome / group-by-drug")
    ax_b1.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2C7FB8", ec="black", lw=0.5, label="Coverage"),
            plt.Rectangle((0, 0), 1, 1, color="#E3872D", ec="black", lw=0.5, label="Median width"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.16),
        ncol=2,
    )
    plot_panel_c(ax_c, chitosan)

    fig.suptitle(
        "Figure 5 | Shared calibrated-family artifacts and prospective discipline upgrade the evidence layer",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_png = OUTDIR / "figure5_uq_and_prospective.png"
    out_pdf = OUTDIR / "figure5_uq_and_prospective.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
