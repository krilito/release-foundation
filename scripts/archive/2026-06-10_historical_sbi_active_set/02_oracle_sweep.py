"""
02 — Oracle sweep over all PLGA curves (Task C + E).

What this does:
    1. Per-curve NLS oracle fit using PLGABiphasic, every formulation.
    2. Writes outputs/02_oracle_sweep/per_curve.csv with theta_hat + R² + n_points.
    3. Plots R² histogram, R² vs n_points scatter, SA/OLA boxplot, and a
       residual plot for one specified curve (default: max n_points).

Why it exists. Day 1 sampled 4 curves and produced a mean R² of 0.94. With
all curves we see the shape of the R² distribution, the SA/OLA gap, and
whether the residuals on the most-densely-sampled curve are structured
(systematic ODE deficiency) or white noise (acceptable).

Run:
    .\.venv\Scripts\python.exe scripts/02_oracle_sweep.py --data data/Dataset_17_feat_augmented.csv
    .\.venv\Scripts\python.exe scripts/02_oracle_sweep.py --data ... --residual-id 142

Outputs:
    outputs/02_oracle_sweep/per_curve.csv
    outputs/02_oracle_sweep/r2_histogram.png
    outputs/02_oracle_sweep/r2_vs_n_points.png
    outputs/02_oracle_sweep/sa_vs_ola.png
    outputs/02_oracle_sweep/residual_<id>.png

Expected runtime: ~15–30 min on a single core for 181 curves.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_plga_181  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


def fit_one(
    sim: PLGABiphasic,
    t_obs: np.ndarray,
    Q_obs: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    x0: np.ndarray,
) -> tuple[np.ndarray, float]:
    def residual(theta_flat: np.ndarray) -> np.ndarray:
        return sim.simulate_numpy(theta_flat, t_obs) - Q_obs

    result = least_squares(residual, x0=x0, bounds=(lows, highs), method="trf", max_nfev=300)
    res = residual(result.x)
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return result.x, r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--residual-id", type=str, default=None,
                        help="formulation_id for residual plot; default = max n_points")
    parser.add_argument("--out", type=Path, default=Path("outputs/02_oracle_sweep"))
    parser.add_argument("--la-ga-col", type=str, default="LA/GA",
                        help="descriptor column used for SA/OLA split")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[..] loading data from {args.data}")
    curves = load_plga_181(args.data)
    print(f"[ok] loaded {len(curves)} curves")

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy()
    highs = prior.high.numpy()
    x0 = 0.5 * (lows + highs)

    rows: list[dict] = []
    t0 = time.time()
    for i, curve in enumerate(curves):
        theta_hat, r2 = fit_one(sim, curve.t.numpy(), curve.Q.numpy(), lows, highs, x0)
        rows.append({
            "formulation_id": curve.formulation_id,
            "n_points": len(curve.t),
            "r2": r2,
            **{name: float(theta_hat[k]) for k, name in enumerate(sim.param_names)},
            "la_ga": curve.descriptors.get(args.la_ga_col, float("nan")),
        })
        if (i + 1) % 20 == 0:
            print(f"[..] {i+1}/{len(curves)}  elapsed {time.time()-t0:.1f}s")
    print(f"[ok] swept {len(curves)} curves in {time.time()-t0:.1f}s")

    csv_path = args.out / "per_curve.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    r2s = np.array([r["r2"] for r in rows])
    ns = np.array([r["n_points"] for r in rows])
    la_ga = np.array([r["la_ga"] for r in rows])

    # R² histogram ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2s, bins=30, edgecolor="black")
    ax.axvline(float(np.median(r2s)), color="red", linestyle="--",
               label=f"median {np.median(r2s):.3f}")
    ax.set_xlabel("oracle R²")
    ax.set_ylabel("count")
    ax.set_title(f"Oracle R² distribution (n={len(rows)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "r2_histogram.png", dpi=150)
    plt.close(fig)

    # R² vs n_points --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ns, r2s, alpha=0.6)
    ax.set_xlabel("n_points per curve")
    ax.set_ylabel("oracle R²")
    ax.set_title("Oracle R² vs sampling density")
    ax.set_ylim(min(0.0, float(r2s.min()) - 0.05), 1.02)
    fig.tight_layout()
    fig.savefig(args.out / "r2_vs_n_points.png", dpi=150)
    plt.close(fig)

    # SA vs OLA boxplot -----------------------------------------------------
    if not np.all(np.isnan(la_ga)):
        is_ola = la_ga > 0
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot([r2s[~is_ola], r2s[is_ola]],
                   labels=[f"SA (n={(~is_ola).sum()})", f"OLA (n={is_ola.sum()})"])
        ax.set_ylabel("oracle R²")
        ax.set_title("SA vs OLA oracle fit quality")
        fig.tight_layout()
        fig.savefig(args.out / "sa_vs_ola.png", dpi=150)
        plt.close(fig)

    # Residual plot for one curve ------------------------------------------
    if args.residual_id is None:
        target = max(curves, key=lambda c: len(c.t))
    else:
        target = next((c for c in curves if c.formulation_id == args.residual_id), None)
        if target is None:
            raise SystemExit(f"formulation_id {args.residual_id} not found")

    row = next(r for r in rows if r["formulation_id"] == target.formulation_id)
    theta_hat = np.array([row[n] for n in sim.param_names])
    Q_obs = target.Q.numpy()
    t_obs = target.t.numpy()
    Q_fit = sim.simulate_numpy(theta_hat, t_obs)
    res = Q_obs - Q_fit

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(t_obs, Q_obs, "o", label="observed")
    axes[0].plot(t_obs, Q_fit, "-", label=f"fit R²={row['r2']:.3f}", linewidth=2)
    axes[0].set_ylabel("Q(t)")
    axes[0].set_title(f"{target.formulation_id} (n={len(t_obs)})")
    axes[0].legend()
    axes[1].plot(t_obs, res, "o-", color="tab:red")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("time (days)")
    axes[1].set_ylabel("residual (obs - fit)")
    fig.tight_layout()
    fig.savefig(args.out / f"residual_{target.formulation_id}.png", dpi=150)
    plt.close(fig)
    print(f"[ok] wrote residual plot for {target.formulation_id}")

    print("\n=== sweep summary ===")
    print(f"  R2 median:     {np.median(r2s):.4f}")
    print(f"  R2 mean:       {np.mean(r2s):.4f}")
    print(f"  R2 10th pct:   {np.percentile(r2s, 10):.4f}")
    print(f"  R2 < 0.80:     {(r2s < 0.80).sum()} curves")
    print(f"  R2 < 0.60:     {(r2s < 0.60).sum()} curves")
    if not np.all(np.isnan(la_ga)):
        is_ola = la_ga > 0
        print(f"  SA  median:    {np.median(r2s[~is_ola]):.4f}")
        print(f"  OLA median:    {np.median(r2s[is_ola]):.4f}")


if __name__ == "__main__":
    main()
