"""
06 — Single-replicate posterior diagnostic for CurvePosterior.

What this does:
    1. Load trained CurvePosterior.
    2. Sample N held-out (θ*, Q*) from prior + simulator.
    3. For each, draw 5000 posterior samples θ ~ q_φ(θ | Q*).
    4. Compute per-parameter: true value, posterior median, 95 % CI, and
       whether θ* falls inside the 95 % CI.
    5. Plot the worst (most mis-located) example as an 8-panel marginal
       histogram with θ* marked.

Why it exists. SBC tells us *whether* the posterior is calibrated, but
not *how* it fails. If 7/8 params fail KS uniformity, the failure mode
is one of:
  - too narrow (posterior overconfident, true θ outside support)
  - too wide (posterior under-confident, ranks pile in the middle)
  - biased (posterior mean systematically off, ranks skew left or right)
This script directly visualizes each mode on individual replicates.

Run:
    python scripts/06_diagnose_curve_posterior.py
    python scripts/06_diagnose_curve_posterior.py --n-replicates 20

Outputs:
    outputs/06_diagnose/coverage_table.csv     per-replicate, per-param coverage
    outputs/06_diagnose/worst_case_marginals.png  marginal hists for worst replicate
    outputs/06_diagnose/coverage_summary.txt
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

from posterior import CurvePosterior  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path,
                        default=Path("outputs/04_curve_posterior/posterior.pt"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/plga_phase1.yaml"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/06_diagnose"))
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-posterior-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
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
    cp = CurvePosterior.load(args.posterior, simulator=sim)

    theta_star = sim.sample_prior(args.n_replicates)
    with torch.no_grad():
        Q_star = sim.simulate(theta_star, t_grid)

    rows: list[dict] = []
    posterior_samples_all = []  # for plotting the worst case
    distances = []  # mean abs (median - true) / prior_range for ranking

    prior_base = sim.prior().base_dist
    prior_range = (prior_base.high - prior_base.low).numpy()

    for i in range(args.n_replicates):
        samples = cp.sample(Q_star[i], n_samples=args.n_posterior_samples).cpu().numpy()
        posterior_samples_all.append(samples)
        theta_true_i = theta_star[i].cpu().numpy()
        per_param_row = {"replicate": i}
        per_param_dist = []
        for k, name in enumerate(sim.param_names):
            s = samples[:, k]
            true = float(theta_true_i[k])
            median = float(np.median(s))
            lo, hi = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
            inside = lo <= true <= hi
            per_param_row[f"{name}_true"] = true
            per_param_row[f"{name}_median"] = median
            per_param_row[f"{name}_lo95"] = lo
            per_param_row[f"{name}_hi95"] = hi
            per_param_row[f"{name}_inside95"] = inside
            per_param_dist.append(abs(median - true) / prior_range[k])
        per_param_row["mean_normalized_distance"] = float(np.mean(per_param_dist))
        rows.append(per_param_row)
        distances.append(per_param_row["mean_normalized_distance"])

    # Write CSV
    csv_path = args.out / "coverage_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[06] wrote {csv_path}")

    # Coverage summary
    coverage_by_param = {}
    for k, name in enumerate(sim.param_names):
        inside_count = sum(r[f"{name}_inside95"] for r in rows)
        coverage_by_param[name] = inside_count / args.n_replicates

    summary_path = args.out / "coverage_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"n_replicates       : {args.n_replicates}\n")
        f.write(f"n_posterior_samples: {args.n_posterior_samples}\n\n")
        f.write("95 % posterior credible interval coverage per parameter:\n")
        f.write("  (well-calibrated -> ~0.95)\n\n")
        for name, cov in coverage_by_param.items():
            flag = "OK" if 0.90 <= cov <= 0.99 else ("NARROW" if cov < 0.90 else "WIDE")
            f.write(f"  {name:14s}  coverage = {cov:.2f}   {flag}\n")
        f.write("\n")
        f.write(f"mean coverage across params: {np.mean(list(coverage_by_param.values())):.3f}\n")
    print(f"[06] wrote {summary_path}")

    # Plot the worst case (largest median-to-true distance)
    worst_idx = int(np.argmax(distances))
    worst_samples = posterior_samples_all[worst_idx]
    worst_true = theta_star[worst_idx].cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for k, name in enumerate(sim.param_names):
        ax = axes[k]
        ax.hist(worst_samples[:, k], bins=50, alpha=0.75, edgecolor="black")
        ax.axvline(worst_true[k], color="red", linewidth=2, label="true θ*")
        lo, hi = np.percentile(worst_samples[:, k], [2.5, 97.5])
        ax.axvspan(lo, hi, color="orange", alpha=0.2, label="95 % CI")
        ax.set_title(f"{name}", fontsize=10)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Worst replicate (idx={worst_idx}, "
        f"mean norm dist={distances[worst_idx]:.3f}) "
        f"— posterior marginals with true θ*"
    )
    fig.tight_layout()
    fig.savefig(args.out / "worst_case_marginals.png", dpi=150)
    plt.close(fig)
    print(f"[06] wrote worst case plot")

    print("\n=== coverage summary ===")
    for name, cov in coverage_by_param.items():
        flag = "OK" if 0.90 <= cov <= 0.99 else ("NARROW" if cov < 0.90 else "WIDE")
        print(f"  {name:14s}  coverage = {cov:.2f}   {flag}")
    print(f"\nworst replicate idx: {worst_idx}, mean norm dist {distances[worst_idx]:.3f}")
    print(f"best  replicate idx: {int(np.argmin(distances))}, mean norm dist {min(distances):.3f}")


if __name__ == "__main__":
    main()
