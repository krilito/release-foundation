"""
05 — Simulation-Based Calibration (SBC) for CurvePosterior.

What this does:
    1. Load the trained CurvePosterior from script 04.
    2. For N_sbc held-out (θ*, Q*) pairs sampled from prior + simulator:
       a. Draw n posterior samples θ ~ q_φ(θ | Q*).
       b. Compute the rank of θ* in each parameter's marginal posterior.
    3. Plot rank histograms (one per parameter) and KS statistics.

Why it exists. SBC is the gold-standard *necessary* check for amortized
posteriors: if q_φ(θ | Q) is well-calibrated, the rank statistics will be
uniform on [0, n]. Systematic deviations reveal posterior mis-calibration
(too narrow / too wide / biased) per parameter dimension. ADR-006 requires
SBC to pass before Phase 1 is considered successful.

Run:
    .\.venv\Scripts\python.exe scripts/05_sbc_curve_posterior.py

Outputs:
    outputs/05_sbc/rank_histograms.png
    outputs/05_sbc/sbc_summary.txt

Expected runtime: 5–15 min (driven by posterior.sample on N_sbc curves).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sbi.diagnostics import check_sbc, run_sbc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import CurvePosterior  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/plga_phase1.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/05_sbc"),
    )
    parser.add_argument("--n-sbc", type=int, default=300,
                        help="number of (θ*, Q*) replicates for SBC")
    parser.add_argument("--n-posterior-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    print(f"[05] loading posterior from {args.posterior}")
    cp = CurvePosterior.load(args.posterior, simulator=sim)
    print(f"[05] training log: {cp._training_log}")

    # Seed AFTER CurvePosterior.load: load() calls generate_synthetic_pairs
    # with seed=0 internally (to build the dummy batch needed before
    # load_state_dict), which would otherwise clobber any seed set here.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Draw SBC replicates from prior + simulator (held-out).
    # Apply the SAME observation noise the posterior was trained on
    # (see ADR-014). Without this, SBC would test a different observation
    # distribution than training and the result is uninterpretable.
    print(f"[05] drawing {args.n_sbc} SBC replicates"
          f" (noise_sigma={cp.noise_sigma})...")
    theta_star = sim.sample_prior(args.n_sbc)
    with torch.no_grad():
        Q_star = sim.simulate(theta_star, t_grid)
    Q_star = cp.add_obs_noise(Q_star)

    # sbi.run_sbc handles the per-replicate posterior sampling and ranking.
    print(f"[05] running SBC ({args.n_posterior_samples} posterior samples per replicate)...")
    ranks, dap_samples = run_sbc(
        thetas=theta_star,
        xs=Q_star,
        posterior=cp._posterior,
        num_posterior_samples=args.n_posterior_samples,
    )

    check_stats = check_sbc(
        ranks=ranks,
        prior_samples=sim.sample_prior(args.n_sbc),
        dap_samples=dap_samples,
        num_posterior_samples=args.n_posterior_samples,
    )

    # Plot rank histograms per parameter dimension.
    n_params = sim.n_params
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    axes = axes.ravel()
    expected_count = args.n_sbc / 20  # 20 bins
    for k in range(n_params):
        ax = axes[k]
        ax.hist(ranks[:, k].numpy(), bins=20, edgecolor="black", alpha=0.85)
        ax.axhline(expected_count, color="red", linestyle="--", linewidth=1,
                   label=f"uniform exp = {expected_count:.0f}")
        ks_p = float(check_stats["ks_pvals"][k])
        flag = "PASS" if ks_p > 0.05 else "FAIL"
        ax.set_title(f"{sim.param_names[k]}  KS p={ks_p:.3f}  [{flag}]",
                     fontsize=9)
        ax.set_xlabel("rank")
        if k % 4 == 0:
            ax.set_ylabel("count")
    # Hide the 8th subplot if n_params == 8 (we have exactly 8, so all used).
    for k in range(n_params, len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        f"SBC rank histograms ({args.n_sbc} replicates × {args.n_posterior_samples} samples)"
    )
    fig.tight_layout()
    fig.savefig(args.out / "rank_histograms.png", dpi=150)
    plt.close(fig)
    print(f"[05] wrote {args.out / 'rank_histograms.png'}")

    # Write summary.
    summary_path = args.out / "sbc_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"n_sbc                  : {args.n_sbc}\n")
        f.write(f"n_posterior_samples    : {args.n_posterior_samples}\n")
        f.write(f"seed                   : {args.seed}\n\n")
        f.write("per-parameter KS p-values (uniform-rank test):\n")
        f.write("  >0.05 = consistent with uniform = calibrated\n\n")
        any_fail = False
        for k, name in enumerate(sim.param_names):
            ks_p = float(check_stats["ks_pvals"][k])
            flag = "PASS" if ks_p > 0.05 else "FAIL"
            if ks_p <= 0.05:
                any_fail = True
            f.write(f"  {name:14s}  KS p = {ks_p:.4f}   {flag}\n")
        f.write(f"\nOverall: {'PASS' if not any_fail else 'AT LEAST ONE FAIL'}\n")
        f.write("\ncheck_sbc raw stats:\n")
        for k, v in check_stats.items():
            f.write(f"  {k}: {v}\n")
    print(f"[05] wrote {summary_path}")

    print("\n=== SBC summary ===")
    for k, name in enumerate(sim.param_names):
        ks_p = float(check_stats["ks_pvals"][k])
        flag = "PASS" if ks_p > 0.05 else "FAIL"
        print(f"  {name:14s}  KS p = {ks_p:.4f}   {flag}")


if __name__ == "__main__":
    main()
