"""
01 — Oracle-baseline sanity check.

What this does:
    1. Samples 8 parameter vectors from the PLGABiphasic prior.
    2. Simulates the corresponding release curves on a dense time grid.
    3. Plots them, so the user can visually confirm biphasic shape.
    4. If a PLGA dataset CSV is provided, fits the simulator per-curve to
       four representative real curves via scipy NLS, then plots overlays
       (real vs. simulated) and reports per-curve R².

Why it exists. Before pointing an amortized SBI network at this simulator,
the simulator itself must be able to reproduce real PLGA shapes. If the
oracle (per-curve NLS) cannot reach R² ≥ ~0.9 on a few well-behaved real
curves, the ODE form or the prior bounds are wrong, and SBI will not save
us.

Run:
    uv run python scripts/01_oracle_baseline.py
    uv run python scripts/01_oracle_baseline.py --data data/Dataset_17.csv

Outputs:
    outputs/01_oracle_baseline/prior_curves.png
    outputs/01_oracle_baseline/real_vs_fit.png   (only if --data given)
    outputs/01_oracle_baseline/oracle_r2.csv     (only if --data given)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares

# Repo root is one level up from scripts/. Insert it on sys.path so this
# script can be invoked directly with `uv run python scripts/01_*.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402
from data import load_plga_181       # noqa: E402


def plot_prior_curves(out_dir: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    sim = PLGABiphasic()
    theta = sim.sample_prior(8)
    t = torch.linspace(0.0, 60.0, 200)
    Q = sim.simulate(theta, t).detach().numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(Q.shape[0]):
        ax.plot(t.numpy(), Q[i], label=f"sample {i}", alpha=0.85)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("cumulative release Q(t)")
    ax.set_title("PLGABiphasic — 8 prior-sampled release curves")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "prior_curves.png", dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_dir / 'prior_curves.png'}")


def fit_curve_nls(
    sim: PLGABiphasic,
    t_obs: np.ndarray,
    Q_obs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-curve nonlinear least squares.

    Bounds come from the prior. Initial guess is the prior midpoint.
    Returns (theta_hat, R²).
    """
    prior = sim.prior().base_dist  # Independent wraps Uniform
    lows = prior.low.numpy()
    highs = prior.high.numpy()
    x0 = 0.5 * (lows + highs)

    t_t = torch.tensor(t_obs, dtype=torch.float32)

    def residual(theta_flat: np.ndarray) -> np.ndarray:
        theta = torch.tensor(theta_flat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Q_pred = sim.simulate(theta, t_t).squeeze(0).numpy()
        return Q_pred - Q_obs

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lows, highs),
        method="trf",
        max_nfev=300,
    )
    theta_hat = result.x

    # R²
    ss_res = float(np.sum((residual(theta_hat)) ** 2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return theta_hat, r2


def fit_and_plot_real_curves(data_path: Path, out_dir: Path, n_curves: int = 4) -> None:
    print(f"[..] loading data from {data_path}")
    curves = load_plga_181(data_path)
    print(f"[ok] loaded {len(curves)} formulations")

    # Pick curves with diverse shapes: longest, shortest, smallest area,
    # largest area. Crude but adequate for a sanity probe.
    by_npts = sorted(curves, key=lambda c: len(c.t))
    by_area = sorted(curves, key=lambda c: float(c.Q.mean()))
    selected = [
        by_npts[-1],     # most densely sampled
        by_npts[0],      # most sparsely sampled
        by_area[0],      # slowest-release shape
        by_area[-1],     # fastest-release shape
    ]
    selected = selected[:n_curves]

    sim = PLGABiphasic()
    rows = []
    fig, axes = plt.subplots(1, n_curves, figsize=(4.0 * n_curves, 4.0), sharey=True)
    if n_curves == 1:
        axes = [axes]

    for ax, curve in zip(axes, selected, strict=False):
        t_obs = curve.t.numpy()
        Q_obs = curve.Q.numpy()
        theta_hat, r2 = fit_curve_nls(sim, t_obs, Q_obs)

        t_dense = np.linspace(0.0, float(t_obs.max()) * 1.05, 200)
        with torch.no_grad():
            Q_dense = sim.simulate(
                torch.tensor(theta_hat, dtype=torch.float32).unsqueeze(0),
                torch.tensor(t_dense, dtype=torch.float32),
            ).squeeze(0).numpy()

        ax.plot(t_obs, Q_obs, "o", label="observed", markersize=4)
        ax.plot(t_dense, Q_dense, "-", label=f"fit (R²={r2:.3f})", linewidth=2)
        ax.set_xlabel("time (days)")
        ax.set_title(f"{curve.formulation_id}  (n={len(t_obs)})", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        rows.append({"formulation_id": curve.formulation_id, "n_points": len(t_obs), "r2": r2})

    axes[0].set_ylabel("cumulative release")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_fit.png", dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_dir / 'real_vs_fit.png'}")

    import csv

    with (out_dir / "oracle_r2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["formulation_id", "n_points", "r2"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_dir / 'oracle_r2.csv'}")

    mean_r2 = float(np.mean([r["r2"] for r in rows]))
    print(f"\n=== oracle mean R2 across {n_curves} curves: {mean_r2:.4f} ===")
    if mean_r2 < 0.85:
        print(
            "[warn] mean R² < 0.85. Either the ODE is missing a term or the prior\n"
            "       bounds clamp the fit. Inspect real_vs_fit.png and adjust\n"
            "       PLGABiphasic.prior() or the ODE form."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None, help="path to PLGA CSV")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/01_oracle_baseline"),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plot_prior_curves(args.out, seed=args.seed)

    if args.data is not None:
        fit_and_plot_real_curves(args.data, args.out)
    else:
        print("[info] --data not provided; skipping real-curve fits.")
        print("       Re-run with --data <path-to-Dataset_17_feat_augmented.csv>")


if __name__ == "__main__":
    main()
