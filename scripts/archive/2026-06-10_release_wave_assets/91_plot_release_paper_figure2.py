"""
91 - Plot Figure 2 for the current drug-release paper draft.

Purpose:
    Build the first main empirical figure from the assembled Figure 2 panel
    assets produced by `90_release_paper_figure_assets.py`.

Consumes:
    outputs/90_release_paper_figure_assets/figure2/panelA_input_source_ablation.csv
    outputs/90_release_paper_figure_assets/figure2/panelB_theta_vs_direct.csv
    outputs/90_release_paper_figure_assets/figure2/panelC_middle_layer_gain.csv

Produces:
    outputs/91_release_paper_figures/figure2/figure2_plga_core.png
    outputs/91_release_paper_figures/figure2/figure2_plga_core.pdf
    outputs/91_release_paper_figures/figure2/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INROOT = Path("outputs/90_release_paper_figure_assets/figure2")
OUTDIR = Path("outputs/91_release_paper_figures/figure2")

PANEL_A = INROOT / "panelA_input_source_ablation.csv"
PANEL_B = INROOT / "panelB_theta_vs_direct.csv"
PANEL_C = INROOT / "panelC_middle_layer_gain.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_group(dataset: str, scheme_label: str) -> str:
    return f"{dataset}\n{scheme_label}"


def _short_scheme_label(value: str) -> str:
    return {
        "Group by drug": "Drug OOD",
        "Group by polymer": "Polymer OOD",
        "Random 5-fold": "Random",
    }.get(value, value)


def _short_input_label(value: str) -> str:
    return {
        "Formulation only": "Formulation",
        "Early only": "Early",
        "Formulation + early": "Form.+Early",
    }.get(value, value)


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    order_inputs = ["Formulation only", "Early only", "Formulation + early"]
    ood = df[df["scheme"].isin(["group_by_drug", "group_by_polymer"])].copy()
    ood["group_label"] = ood.apply(
        lambda r: _format_group(r["dataset"], _short_scheme_label(r["scheme_label"])),
        axis=1,
    )
    group_order = [
        _format_group("cross321", "Drug OOD"),
        _format_group("cross321", "Polymer OOD"),
        _format_group("internal181", "Drug OOD"),
        _format_group("internal181", "Polymer OOD"),
    ]
    x = np.arange(len(group_order))
    width = 0.24
    colors = {
        "Formulation only": "#B34D4D",
        "Early only": "#2C7FB8",
        "Formulation + early": "#3AA76D",
    }
    for offset_idx, mode in enumerate(order_inputs):
        sub = ood[ood["input_mode_label"] == mode].set_index("group_label").reindex(group_order)
        ax.bar(
            x + (offset_idx - 1) * width,
            sub["median"].values,
            width=width,
            label=_short_input_label(mode),
            color=colors[mode],
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=9)
    ax.set_ylabel("Best median $R^2$")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("A. OOD input-source ablation", loc="left", fontweight="bold")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    colors = {
        "Formulation only": "#B34D4D",
        "Early only": "#2C7FB8",
        "Formulation + early": "#3AA76D",
    }
    plot_df = df.copy()
    plot_df["row_label"] = plot_df.apply(
        lambda r: (
            f"{'c321' if r['dataset'] == 'cross321' else 'i181'} | "
            f"{_short_scheme_label(r['scheme_label']).replace(' OOD', '')} | "
            f"{_short_input_label(r['input_mode_label'])}"
        ),
        axis=1,
    )
    plot_df = plot_df.sort_values("theta_minus_direct", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    ax.barh(
        y,
        plot_df["theta_minus_direct"].values,
        color=[colors[m] for m in plot_df["input_mode_label"]],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["row_label"], fontsize=7)
    ax.set_xlabel(r"$\Delta$ median $R^2$ (theta - direct)")
    ax.set_title("B. Theta route minus direct route", loc="left", fontweight="bold")
    ax.axvline(0.0, color="black", linewidth=0.8)


def plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    keep = [
        "Best route / all",
        "Best route / OOD formulation + early",
        "Family matched / all",
        "Family matched / OOD formulation + early",
    ]
    plot_df = df[df["group_label"].isin(keep)].copy()
    plot_df["group_label"] = pd.Categorical(plot_df["group_label"], categories=keep, ordered=True)
    plot_df = plot_df.sort_values("group_label").reset_index(drop=True)
    y = np.arange(len(plot_df))
    means = plot_df["mean_gain"].values
    lower = means - plot_df["min_gain"].values
    upper = plot_df["max_gain"].values - means
    ax.barh(y, means, color="#7A78C9", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
    )
    ax.scatter(plot_df["median_gain"].values, y, color="#D95F02", s=30, zorder=3, label="Median gain")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["group_label"], fontsize=9)
    ax.set_xlabel("Gain over direct route")
    ax.set_title("C. Aggregate middle-layer gain audit", loc="left", fontweight="bold")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def write_summary(out_png: Path, out_pdf: Path) -> None:
    lines = [
        "=== 91 -- release paper Figure 2 ===",
        "",
        f"PNG: {out_png.as_posix()}",
        f"PDF: {out_pdf.as_posix()}",
        "",
        "Panels",
        "  A. OOD input-source ablation from panelA_input_source_ablation.csv",
        "  B. Theta-minus-direct gains across 18 audited PLGA cells",
        "  C. Aggregate middle-layer gain audit summary",
        "",
        "Interpretation",
        "  1. Formulation-only routes remain much weaker than early-observation routes under OOD splits.",
        "  2. Theta-route gains stay positive across the audited PLGA panel.",
        "  3. Aggregate middle-layer gains support the mechanism route without claiming universal duration superiority.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    panel_a = pd.read_csv(PANEL_A)
    panel_b = pd.read_csv(PANEL_B)
    panel_c = pd.read_csv(PANEL_C)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.1, 1.0], hspace=0.35, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    plot_panel_a(ax_a, panel_a)
    plot_panel_b(ax_b, panel_b)
    plot_panel_c(ax_c, panel_c)

    fig.suptitle(
        "Figure 2 | Sparse early release dominates OOD forecasting and activates a mechanism-routed middle layer",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_png = OUTDIR / "figure2_plga_core.png"
    out_pdf = OUTDIR / "figure2_plga_core.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    write_summary(out_png, out_pdf)


if __name__ == "__main__":
    main()
