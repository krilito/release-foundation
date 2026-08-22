"""
07 — Synthetic-data audit.

What this does:
    1. Samples N synthetic curves from prior + simulator on the canonical
       time grid.
    2. Per-curve summary statistics: q_burst observed, Q_max observed,
       Q_final, dynamic range, t_50, max slope.
    3. Identifies degenerate curves:
         - flat: dynamic range < 0.05
         - immediate saturation: Q hits 95 % of Q_max within 5 days
         - never rises: max slope < 1e-3 per day
    4. Plots feature histograms + a sample of degenerate vs normal curves.
    5. Reports degenerate fraction. Drives ADR-012 reversal decisions.

Why this exists. After three rounds of SBC where q_burst and Q_max keep
failing with systematic bias, the leading hypothesis is no longer
architectural but data-level: the synthetic distribution contains a
fraction of curves where these end-point parameters are not identifiable
at all. The flow learns to default to prior mean for those, polluting
calibration globally.

Run:
    python scripts/07_synthetic_audit.py

Outputs:
    outputs/07_synthetic_audit/feature_histograms.png
    outputs/07_synthetic_audit/example_curves.png
    outputs/07_synthetic_audit/audit_summary.txt
    outputs/07_synthetic_audit/per_curve_stats.csv

Expected runtime: 2-5 min (no training, just simulator).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402


def compute_per_curve_stats(
    theta: torch.Tensor,
    Q: torch.Tensor,
    t_grid: torch.Tensor,
    sim: PLGABiphasic,
) -> dict[str, np.ndarray]:
    """Per-curve summary statistics."""
    n = Q.shape[0]
    Q_np = Q.numpy()
    t_np = t_grid.numpy()

    stats: dict[str, np.ndarray] = {}
    stats["q_burst_obs"] = Q_np[:, 0]
    stats["Q_max_obs"] = Q_np.max(axis=1)
    stats["Q_final"] = Q_np[:, -1]
    stats["dynamic_range"] = stats["Q_max_obs"] - stats["q_burst_obs"]

    # t_50: time to reach half of (Q_max - q_burst) above q_burst.
    halfway = stats["q_burst_obs"] + 0.5 * stats["dynamic_range"]
    t_50 = np.full(n, np.nan)
    for i in range(n):
        reached = np.where(Q_np[i] >= halfway[i])[0]
        if len(reached) > 0:
            t_50[i] = t_np[reached[0]]
    stats["t_50"] = t_50

    # Max instantaneous slope (per day)
    dt = np.diff(t_np)
    dQ = np.diff(Q_np, axis=1)
    slopes = dQ / dt[np.newaxis, :]
    stats["max_slope"] = slopes.max(axis=1)

    # True params from the prior (for corner analysis later)
    theta_np = theta.numpy()
    for k, name in enumerate(sim.param_names):
        stats[f"theta_{name}"] = theta_np[:, k]

    # Degeneracy flags
    stats["is_flat"] = stats["dynamic_range"] < 0.05
    stats["is_immediate_saturation"] = (
        (Q_np[:, np.searchsorted(t_np, 5.0)] >= 0.95 * stats["Q_max_obs"])
        & ~stats["is_flat"]
    )
    stats["is_never_rises"] = stats["max_slope"] < 1e-3
    stats["is_degenerate"] = (
        stats["is_flat"] | stats["is_immediate_saturation"] | stats["is_never_rises"]
    )

    return stats


def plot_histograms(stats: dict[str, np.ndarray], out_path: Path, n: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    axes[0].hist(stats["q_burst_obs"], bins=50, edgecolor="black", alpha=0.85)
    axes[0].set_title("q_burst observed (Q at t=0)")
    axes[0].set_xlabel("Q")

    axes[1].hist(stats["Q_max_obs"], bins=50, edgecolor="black", alpha=0.85)
    axes[1].set_title("Q_max observed (max Q over curve)")
    axes[1].set_xlabel("Q")

    axes[2].hist(stats["Q_final"], bins=50, edgecolor="black", alpha=0.85)
    axes[2].set_title("Q_final (Q at last time point)")
    axes[2].set_xlabel("Q")

    axes[3].hist(stats["dynamic_range"], bins=50, edgecolor="black", alpha=0.85)
    axes[3].axvline(0.05, color="red", linestyle="--", label="flat threshold")
    axes[3].set_title("Dynamic range  (Q_max − q_burst)")
    axes[3].set_xlabel("range")
    axes[3].legend()

    valid_t50 = stats["t_50"][~np.isnan(stats["t_50"])]
    if len(valid_t50) > 0:
        axes[4].hist(valid_t50, bins=50, edgecolor="black", alpha=0.85)
    axes[4].set_title(f"t_50 (days)  — n={len(valid_t50)} valid")
    axes[4].set_xlabel("days")

    log_slope = np.log10(np.clip(stats["max_slope"], 1e-9, None))
    axes[5].hist(log_slope, bins=50, edgecolor="black", alpha=0.85)
    axes[5].axvline(np.log10(1e-3), color="red", linestyle="--", label="rises threshold")
    axes[5].set_title("log10(max slope per day)")
    axes[5].set_xlabel("log10(dQ/dt) max")
    axes[5].legend()

    fig.suptitle(f"Synthetic data audit  (n={n})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_examples(
    Q: torch.Tensor,
    t_grid: torch.Tensor,
    stats: dict[str, np.ndarray],
    out_path: Path,
    n_each: int = 5,
) -> None:
    """Pick a few curves from each category for visual inspection."""
    Q_np = Q.numpy()
    t_np = t_grid.numpy()

    flat_idx = np.where(stats["is_flat"])[0][:n_each]
    sat_idx = np.where(stats["is_immediate_saturation"])[0][:n_each]
    nev_idx = np.where(stats["is_never_rises"] & ~stats["is_flat"])[0][:n_each]
    norm_idx = np.where(~stats["is_degenerate"])[0][:n_each]

    categories = [
        ("flat", flat_idx),
        ("immediate saturation", sat_idx),
        ("never rises", nev_idx),
        ("normal", norm_idx),
    ]

    fig, axes = plt.subplots(4, n_each, figsize=(3 * n_each, 9))
    if n_each == 1:
        axes = axes.reshape(-1, 1)
    for row, (label, indices) in enumerate(categories):
        for col in range(n_each):
            ax = axes[row, col]
            if col < len(indices):
                ax.plot(t_np, Q_np[indices[col]], linewidth=1.5)
                ax.set_title(f"{label} #{indices[col]}", fontsize=8)
            else:
                ax.text(0.5, 0.5, f"(no {label})", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylim(-0.02, 1.02)
            if col == 0:
                ax.set_ylabel("Q")
            if row == len(categories) - 1:
                ax.set_xlabel("t (days)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/plga_phase1.yaml"))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/07_synthetic_audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    print(f"[07] sampling {args.n} synthetic curves on t_grid "
          f"({t_cfg['start']}, {t_cfg['end']}, {t_cfg['n']})...")
    theta = sim.sample_prior(args.n)
    with torch.no_grad():
        Q = sim.simulate(theta, t_grid)

    has_nan = bool(torch.isnan(Q).any().item())
    has_inf = bool(torch.isinf(Q).any().item())
    if has_nan or has_inf:
        print(f"[warn] simulator produced NaN={has_nan} Inf={has_inf}")

    print(f"[07] computing per-curve statistics...")
    stats = compute_per_curve_stats(theta, Q, t_grid, sim)

    plot_histograms(stats, args.out / "feature_histograms.png", args.n)
    plot_examples(Q, t_grid, stats, args.out / "example_curves.png")
    print(f"[07] wrote histograms and example curves to {args.out}")

    # CSV
    csv_path = args.out / "per_curve_stats.csv"
    keys = [k for k in stats if not k.startswith("theta_")] + \
           [f"theta_{name}" for name in sim.param_names]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(args.n):
            writer.writerow([stats[k][i] for k in keys])
    print(f"[07] wrote {csv_path}")

    # Summary
    n_flat = int(stats["is_flat"].sum())
    n_sat = int(stats["is_immediate_saturation"].sum())
    n_nev = int(stats["is_never_rises"].sum())
    n_degen = int(stats["is_degenerate"].sum())
    pct = lambda x: f"{x / args.n * 100:.1f}%"  # noqa: E731

    summary_lines = [
        f"n_samples                : {args.n}",
        f"t_grid                   : ({t_cfg['start']}, {t_cfg['end']}, {t_cfg['n']})",
        f"NaN in Q                 : {has_nan}",
        f"Inf in Q                 : {has_inf}",
        "",
        f"flat (range < 0.05)      : {n_flat}  ({pct(n_flat)})",
        f"immediate saturation     : {n_sat}  ({pct(n_sat)})",
        f"never rises              : {n_nev}  ({pct(n_nev)})",
        f"any degenerate           : {n_degen}  ({pct(n_degen)})",
        "",
        f"q_burst obs   median     : {np.median(stats['q_burst_obs']):.4f}",
        f"Q_max obs     median     : {np.median(stats['Q_max_obs']):.4f}",
        f"dynamic range median     : {np.median(stats['dynamic_range']):.4f}",
        f"t_50 median (days)       : {np.nanmedian(stats['t_50']):.2f}",
        f"max slope median /day    : {np.median(stats['max_slope']):.4f}",
        "",
        "ADR-012 decision rule:",
        "  degen < 10%  -> prior is fine; go back to architecture investigation",
        "  10-30%       -> narrow prior corners that produce degeneracy",
        "  > 30%        -> prior fundamentally misspecified; rederive bounds",
    ]
    summary = "\n".join(summary_lines)
    (args.out / "audit_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
