"""
74 - Draft manuscript figures for the current GREEN claims.

Figures:
    Fig. 2 draft: low-dimensional, regime-dependent kinetic states.
    Fig. 3 draft: middle-layer gain and sparse-classical baseline stress test.

Inputs:
    outputs/33_active_set_regimes/regime_summary.csv
    outputs/33_active_set_regimes/regime_assignments.csv
    outputs/36_r5_regime_prototype/per_curve_fits.csv
    outputs/36_r6_regime_prototype/per_curve_fits.csv
    outputs/71_nature_gate_diagnostics/g1_middle_layer_bootstrap.csv
    outputs/71_nature_gate_diagnostics/g1_lgbm_bootstrap.csv
    outputs/71_nature_gate_diagnostics/g2_classical_sparse_summary.csv

Outputs:
    outputs/74_manuscript_green_figures/fig2_regime_states.{svg,pdf,png,tiff}
    outputs/74_manuscript_green_figures/fig3_middle_layer_gain.{svg,pdf,png,tiff}
    outputs/74_manuscript_green_figures/source_data_*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARAM_ORDER = [
    "log_kw",
    "log_kh",
    "log_alpha",
    "log_kd",
    "log_ke",
    "m_crit",
    "q_burst",
    "log_tau_burst",
    "Q_max",
]

PARAM_LABELS = {
    "log_kw": "hydration\nkw",
    "log_kh": "hydrolysis\nkh",
    "log_alpha": "autocat.\nalpha",
    "log_kd": "diffusion\nkd",
    "log_ke": "erosion\nke",
    "m_crit": "erosion\nthreshold",
    "q_burst": "burst\nQ0",
    "log_tau_burst": "burst\ntau",
    "Q_max": "capacity\nQmax",
}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def _parse_active_set(cell: object) -> list[str]:
    if not isinstance(cell, str):
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def active_frequency(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime, sub in assignments.groupby("regime", sort=True):
        n = len(sub)
        counts = {p: 0 for p in PARAM_ORDER}
        for cell in sub["active_set_k4"]:
            for p in _parse_active_set(cell):
                if p in counts:
                    counts[p] += 1
        for p in PARAM_ORDER:
            rows.append({
                "regime": int(regime),
                "parameter": p,
                "frequency": counts[p] / max(n, 1),
                "n_curves": n,
            })
    return pd.DataFrame(rows)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_fig2(args: argparse.Namespace) -> None:
    regime_summary = pd.read_csv(args.regime_summary)
    assignments = pd.read_csv(args.regime_assignments)
    r5 = pd.read_csv(args.r5_fits)
    r6 = pd.read_csv(args.r6_fits)
    freq = active_frequency(assignments)

    args.out.mkdir(parents=True, exist_ok=True)
    freq.to_csv(args.out / "source_data_fig2_active_frequency.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[1.0, 1.1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A: regime composition by dataset.
    regime_summary = regime_summary.sort_values("regime")
    x = np.arange(len(regime_summary))
    internal = regime_summary["size_internal181"].to_numpy()
    cross = regime_summary["size_cross321"].to_numpy()
    ax_a.bar(x, internal, color="#6BAED6", label="internal181")
    ax_a.bar(x, cross, bottom=internal, color="#BDBDBD", label="cross321")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"R{r}" for r in regime_summary["regime"]])
    ax_a.set_ylabel("Curves")
    ax_a.set_title("Regime occupancy")
    ax_a.legend(loc="upper left", handlelength=1.0)
    _add_panel_label(ax_a, "a")

    # Panel B: active-set heatmap.
    mat = freq.pivot(index="regime", columns="parameter", values="frequency").loc[
        sorted(freq["regime"].unique()), PARAM_ORDER
    ]
    im = ax_b.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax_b.set_yticks(np.arange(len(mat.index)))
    ax_b.set_yticklabels([f"R{r}" for r in mat.index])
    ax_b.set_xticks(np.arange(len(PARAM_ORDER)))
    ax_b.set_xticklabels([PARAM_LABELS[p] for p in PARAM_ORDER], rotation=45, ha="right")
    ax_b.set_title("Active-parameter frequency")
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.035, pad=0.02)
    cbar.set_label("Frequency in k=4 active set")
    _add_panel_label(ax_b, "b")

    # Panel C: R5/R6 conditional prototype R2 distributions.
    fit_cols = [
        ("r2_oracle9", "oracle 9"),
        ("r2_greedy4_29", "greedy 4"),
        ("r2_r6cond4", "regime 4"),
    ]
    data = []
    positions = []
    labels = []
    colors = []
    pos = 1
    for regime_label, df in [("R5", r5), ("R6", r6)]:
        for col, label in fit_cols:
            data.append(df[col].dropna().to_numpy(dtype=float))
            positions.append(pos)
            labels.append(f"{regime_label}\n{label}")
            colors.append("#9ECAE1" if regime_label == "R5" else "#FDD0A2")
            pos += 1
        pos += 1
    bp = ax_c.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#525252")
    for median in bp["medians"]:
        median.set_color("#111111")
    ax_c.axhline(0.95, color="#737373", ls=":", lw=0.8)
    ax_c.set_ylim(0.85, 1.02)
    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(labels, rotation=45, ha="right")
    ax_c.set_ylabel("Curve R$^2$")
    ax_c.set_title("Regime-conditioned 4-parameter fits")
    _add_panel_label(ax_c, "c")

    # Panel D: loss relative to oracle.
    delta_frames = []
    for regime_label, df in [("R5", r5), ("R6", r6)]:
        tmp = pd.DataFrame({
            "regime": regime_label,
            "delta_cond_minus_oracle": df["delta_r6cond_minus_oracle"].astype(float),
            "delta_cond_minus_greedy": df["delta_r6cond_minus_greedy"].astype(float),
        })
        delta_frames.append(tmp)
    delta = pd.concat(delta_frames, ignore_index=True)
    delta.to_csv(args.out / "source_data_fig2_delta.csv", index=False)
    for i, (regime_label, color) in enumerate([("R5", "#3182BD"), ("R6", "#E6550D")]):
        vals = delta[delta["regime"] == regime_label]["delta_cond_minus_oracle"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(vals))
        rng = np.random.default_rng(0)
        rng.shuffle(jitter)
        ax_d.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, s=9, alpha=0.45, color=color)
        med = np.median(vals)
        lo, hi = np.percentile(vals, [25, 75])
        ax_d.plot([i - 0.22, i + 0.22], [med, med], color="#111111", lw=1.2)
        ax_d.plot([i, i], [lo, hi], color="#111111", lw=0.8)
    ax_d.axhline(0, color="#737373", lw=0.8)
    ax_d.axhline(-0.01, color="#737373", ls=":", lw=0.8)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["R5", "R6"])
    ax_d.set_ylabel("Regime 4 R$^2$ − oracle 9 R$^2$")
    ax_d.set_title("Expressivity loss is small for dominant regimes")
    _add_panel_label(ax_d, "d")

    save_figure(fig, args.out / "fig2_regime_states")
    plt.close(fig)


def _plot_forest(ax: plt.Axes, df: pd.DataFrame, title: str, color: str) -> None:
    df = df.sort_values(["dataset", "scheme"]).reset_index(drop=True)
    y = np.arange(len(df))
    ax.axvline(0, color="#737373", lw=0.8)
    ax.errorbar(
        df["gain"],
        y,
        xerr=[
            df["gain"] - df["ci_low"],
            df["ci_high"] - df["gain"],
        ],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.0,
        markersize=4.0,
    )
    labels = [f"{r.dataset} / {r.scheme.replace('group_by_', 'by_')}" for r in df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Median R$^2$ gain (theta→ODE − direct Q)")
    ax.set_title(title)
    ax.set_xlim(min(-0.04, float(df["ci_low"].min()) - 0.01), float(df["ci_high"].max()) + 0.02)


def make_fig3(args: argparse.Namespace) -> None:
    rf = pd.read_csv(args.g1_rf_et)
    lgbm = pd.read_csv(args.g1_lgbm)
    classical = pd.read_csv(args.g2_classical)

    rf_focus = rf[rf["input_mode"] == "formulation_plus_early"].copy()
    lgbm_focus = lgbm[lgbm["input_mode"] == "formulation_plus_early"].copy()
    rf_focus.to_csv(args.out / "source_data_fig3_rf_et_gain.csv", index=False)
    lgbm_focus.to_csv(args.out / "source_data_fig3_lgbm_gain.csv", index=False)

    two = classical[classical["window"].str.contains("two_point", regex=False)].copy()
    best_two = (
        two.sort_values("future_r2_median", ascending=False)
        .groupby(["dataset", "window"], as_index=False)
        .head(1)
        .sort_values(["dataset", "window"])
    )
    best_two.to_csv(args.out / "source_data_fig3_best_classical_two_point.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.9])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _plot_forest(ax_a, rf_focus, "RF/ET best-route middle-layer gain", "#3182BD")
    _add_panel_label(ax_a, "a")
    _plot_forest(ax_b, lgbm_focus, "LightGBM middle-layer gain", "#756BB1")
    _add_panel_label(ax_b, "b")

    x = np.arange(len(best_two))
    colors = ["#9ECAE1" if d == "cross321" else "#FDD0A2" for d in best_two["dataset"]]
    y = best_two["future_r2_median"].to_numpy(dtype=float)
    yerr = np.vstack([
        y - best_two["future_r2_ci_low"].to_numpy(dtype=float),
        best_two["future_r2_ci_high"].to_numpy(dtype=float) - y,
    ])
    ax_c.bar(x, y, color=colors, edgecolor="#525252", linewidth=0.6)
    ax_c.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#252525", elinewidth=0.8, capsize=2)
    ax_c.axhline(0.85, color="#B2182B", ls="--", lw=1.0)
    ax_c.axhline(0, color="#737373", lw=0.7)
    labels = [
        f"{r.dataset}\n{r.window.replace('two_point_', '').replace('_', ' ')}\n{r.method}"
        for r in best_two.itertuples()
    ]
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Future-only R$^2$ median")
    ax_c.set_title("Best classical two-point extrapolator remains below the 0.85 threshold")
    ax_c.set_ylim(min(-0.35, float(np.nanmin(y - yerr[0])) - 0.1), 0.95)
    ax_c.text(
        len(best_two) - 0.15,
        0.87,
        "0.85 threshold",
        color="#B2182B",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    _add_panel_label(ax_c, "c")

    save_figure(fig, args.out / "fig3_middle_layer_gain")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/74_manuscript_green_figures"))
    ap.add_argument("--regime-summary", type=Path, default=Path("outputs/33_active_set_regimes/regime_summary.csv"))
    ap.add_argument("--regime-assignments", type=Path, default=Path("outputs/33_active_set_regimes/regime_assignments.csv"))
    ap.add_argument("--r5-fits", type=Path, default=Path("outputs/36_r5_regime_prototype/per_curve_fits.csv"))
    ap.add_argument("--r6-fits", type=Path, default=Path("outputs/36_r6_regime_prototype/per_curve_fits.csv"))
    ap.add_argument("--g1-rf-et", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g1_middle_layer_bootstrap.csv"))
    ap.add_argument("--g1-lgbm", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g1_lgbm_bootstrap.csv"))
    ap.add_argument("--g2-classical", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g2_classical_sparse_summary.csv"))
    args = ap.parse_args()

    set_style()
    make_fig2(args)
    make_fig3(args)
    print(f"Wrote draft figures to {args.out}")


if __name__ == "__main__":
    main()
