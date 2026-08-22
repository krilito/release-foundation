"""
Visualize active_kinetic_observer results.

Reads outputs_active_observer/ and produces:
    1. Strategy comparison bar chart (RMSE, CRPS, coverage)
    2. Active timepoint selection heatmap
    3. Example prediction curves (prior vs active vs truth)
    4. Posterior uncertainty shrinkage (width_90 before/after)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs_active_observer")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def plot_strategy_comparison():
    s = pd.read_csv(OUT / "metrics_summary.csv")
    # Filter out zero_early for main comparison
    strategies = [x for x in s["strategy"] if "zero_early" not in x]
    sub = s[s["strategy"].isin(strategies)].copy()
    sub = sub.sort_values("rmse_mean")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # RMSE
    ax = axes[0]
    colors = ["#d62728" if "directQ" in x else "#1f77b4" if "active" in x else "#aec7e8" for x in sub["strategy"]]
    ax.barh(sub["strategy"], sub["rmse_mean"], color=colors)
    ax.set_xlabel("Future RMSE (mean)")
    ax.set_title("RMSE by Strategy")

    # CRPS (only for posterior strategies)
    ax = axes[1]
    post = sub[sub["crps_mean"].notna()]
    colors2 = ["#1f77b4" if "active" in x else "#aec7e8" for x in post["strategy"]]
    ax.barh(post["strategy"], post["crps_mean"], color=colors2)
    ax.set_xlabel("CRPS (mean)")
    ax.set_title("CRPS by Strategy")

    # Coverage
    ax = axes[2]
    post2 = sub[sub["coverage90_mean"].notna()]
    colors3 = ["#1f77b4" if "active" in x else "#aec7e8" for x in post2["strategy"]]
    ax.barh(post2["strategy"], post2["coverage90_mean"], color=colors3)
    ax.axvline(0.9, color="red", linestyle="--", label="target 0.9")
    ax.set_xlabel("Coverage 90%")
    ax.set_title("90% Interval Coverage")

    plt.tight_layout()
    plt.savefig(FIG / "strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {FIG / 'strategy_comparison.png'}")


def plot_active_timepoint_selection():
    u = pd.read_csv(OUT / "active_utilities.csv")
    chosen = u[u["chosen"] == 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of chosen times
    ax = axes[0]
    times = chosen["candidate_time"]
    ax.hist(times, bins=np.arange(0.5, 8.5, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Chosen observation time (days)")
    ax.set_ylabel("Count")
    ax.set_title("Active Timepoint Selection Distribution")

    # Utility vs time for each curve (spaghetti)
    ax = axes[1]
    for cid in u["curve_id"].unique():
        sub = u[u["curve_id"] == cid]
        ax.plot(sub["candidate_time"], sub["utility"], alpha=0.3, color="gray", linewidth=0.5)
    mean_util = u.groupby("candidate_time")["utility"].mean()
    ax.plot(mean_util.index, mean_util.values, color="red", linewidth=3, label="mean")
    ax.set_xlabel("Candidate time (days)")
    ax.set_ylabel("Utility")
    ax.set_title("Utility vs Candidate Time")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG / "active_selection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {FIG / 'active_selection.png'}")


def plot_example_curves(n_examples: int = 6):
    p = pd.read_csv(OUT / "prediction_curves_active.csv")
    cids = p["curve_id"].unique()
    rng = np.random.default_rng(42)
    examples = rng.choice(cids, size=min(n_examples, len(cids)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i, cid in enumerate(examples):
        ax = axes[i]
        sub = p[p["curve_id"] == cid].sort_values("time")
        obs_t = sub["obs_time"].iloc[0]
        obs_q = sub["obs_q"].iloc[0]

        ax.fill_between(sub["time"], sub["pred_q05"], sub["pred_q95"], alpha=0.2, color="blue", label="90% CI")
        ax.plot(sub["time"], sub["pred_mean"], "b-", label="posterior mean")
        ax.plot(sub["time"], sub["true_release"], "ko-", markersize=4, label="truth")
        ax.axvline(obs_t, color="red", linestyle="--", alpha=0.5, label=f"obs t={obs_t:.1f}d")
        ax.plot(obs_t, obs_q, "r*", markersize=12)
        ax.set_title(f"Curve {int(cid)}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Release")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG / "example_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {FIG / 'example_curves.png'}")


def plot_uncertainty_shrinkage():
    m = pd.read_csv(OUT / "metrics_by_curve.csv")
    prior = m[m["strategy"] == "zero_early_prior_only"][["curve_id", "width_90", "crps"]].rename(
        columns={"width_90": "width_90_prior", "crps": "crps_prior"}
    )
    active = m[m["strategy"] == "active_one_point_posterior"][["curve_id", "width_90", "crps"]].rename(
        columns={"width_90": "width_90_active", "crps": "crps_active"}
    )
    merged = prior.merge(active, on="curve_id")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(merged["width_90_prior"], merged["width_90_active"], alpha=0.5, s=20)
    lim = [0, max(merged["width_90_prior"].max(), merged["width_90_active"].max()) * 1.1]
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlabel("Prior width_90")
    ax.set_ylabel("Posterior width_90")
    ax.set_title("90% Interval Width: Prior vs Posterior")

    ax = axes[1]
    ax.scatter(merged["crps_prior"], merged["crps_active"], alpha=0.5, s=20)
    lim = [0, max(merged["crps_prior"].max(), merged["crps_active"].max()) * 1.1]
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlabel("Prior CRPS")
    ax.set_ylabel("Posterior CRPS")
    ax.set_title("CRPS: Prior vs Posterior")

    plt.tight_layout()
    plt.savefig(FIG / "uncertainty_shrinkage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {FIG / 'uncertainty_shrinkage.png'}")


def plot_prior_coverage_by_curve():
    m = pd.read_csv(OUT / "metrics_by_curve.csv")
    prior = m[m["strategy"] == "zero_early_prior_only"].sort_values("coverage_90")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(prior)), prior["coverage_90"], color="#1f77b4", alpha=0.7)
    ax.axhline(0.9, color="red", linestyle="--", label="target 0.9")
    ax.set_xlabel("Curve index (sorted)")
    ax.set_ylabel("Coverage 90%")
    ax.set_title(f"Zero-Early Prior Coverage by Curve (mean={prior['coverage_90'].mean():.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "prior_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {FIG / 'prior_coverage.png'}")


if __name__ == "__main__":
    plot_strategy_comparison()
    plot_active_timepoint_selection()
    plot_example_curves()
    plot_uncertainty_shrinkage()
    plot_prior_coverage_by_curve()
    print("\n[done] all figures saved to", FIG)
