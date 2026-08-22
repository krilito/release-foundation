"""
05b — Multi-seed SBC for CurvePosterior.

Why this exists. Script 05 reports per-parameter KS p-values at a single
seed with n_sbc=300. The KS test on 300 replicates has a non-trivial false-
fail rate at the p>0.05 threshold; the verdict for any one parameter can
flip across seeds. See ADR-017 for the seed-clobber bug that originally
masked this (load() reseeded torch to 0, making script 05 effectively
single-seed regardless of --seed). With that fixed, --seed actually
varies the SBC sample, and per-seed verdicts oscillate noticeably for
borderline parameters.

This script loads the posterior once, then loops over K seeds drawing
fresh (θ*, Q*) replicates each time, and reports the *distribution* of
KS p-values and c2st_ranks per parameter. The c2st_ranks metric (uniform
vs. observed ranks via a binary classifier) is more stable than KS at
n_sbc=300 and serves as the primary calibration verdict; KS p is
secondary diagnostic.

Run:
    .\.venv\Scripts\python.exe scripts/05b_sbc_multi_seed.py

Outputs:
    outputs/05_sbc_multiseed/per_seed.csv
    outputs/05_sbc_multiseed/summary.txt

Expected runtime: ~90 s/seed × K seeds.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
        default=Path("outputs/05_sbc_multiseed"),
    )
    parser.add_argument("--n-sbc", type=int, default=300)
    parser.add_argument("--n-posterior-samples", type=int, default=1000)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 7, 13, 42, 99, 101, 200, 314, 777],
    )
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

    print(f"[05b] loading posterior from {args.posterior}")
    cp = CurvePosterior.load(args.posterior, simulator=sim)
    print(f"[05b] training log: {cp._training_log}")

    n_params = sim.n_params
    param_names = sim.param_names

    # rows[seed] = dict(seed=..., ks_pvals=[...], c2st_ranks=[...], c2st_dap=[...])
    rows: list[dict] = []
    for s_idx, seed in enumerate(args.seeds):
        print(f"\n[05b] seed {seed} ({s_idx + 1}/{len(args.seeds)})")
        # Reseed AFTER load (see ADR-017 — load reseeds to 0 internally).
        torch.manual_seed(seed)
        np.random.seed(seed)

        theta_star = sim.sample_prior(args.n_sbc)
        with torch.no_grad():
            Q_star = sim.simulate(theta_star, t_grid)
        Q_star = cp.add_obs_noise(Q_star)

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

        ks_pvals = [float(p) for p in check_stats["ks_pvals"]]
        c2st_ranks = [float(c) for c in check_stats["c2st_ranks"]]
        c2st_dap = [float(c) for c in check_stats["c2st_dap"]]

        rows.append({
            "seed": seed,
            "ks_pvals": ks_pvals,
            "c2st_ranks": c2st_ranks,
            "c2st_dap": c2st_dap,
        })

        print(f"[05b] seed {seed} KS p: " + "  ".join(
            f"{n}={p:.3f}" for n, p in zip(param_names, ks_pvals)
        ))

    # ---- Aggregation ----
    K = len(args.seeds)
    ks_arr = np.array([r["ks_pvals"] for r in rows])          # (K, n_params)
    c2st_arr = np.array([r["c2st_ranks"] for r in rows])      # (K, n_params)
    c2st_dap_arr = np.array([r["c2st_dap"] for r in rows])    # (K, n_params)

    # Per-param: median KS p, fraction PASS (p>0.05), c2st_ranks mean/std.
    median_ks = np.median(ks_arr, axis=0)
    frac_pass = (ks_arr > 0.05).mean(axis=0)
    mean_c2st_ranks = c2st_arr.mean(axis=0)
    std_c2st_ranks = c2st_arr.std(axis=0)
    max_c2st_ranks = c2st_arr.max(axis=0)
    mean_c2st_dap = c2st_dap_arr.mean(axis=0)

    # ---- per_seed.csv ----
    csv_path = args.out / "per_seed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["seed"]
        for name in param_names:
            header += [f"ks_p_{name}"]
        for name in param_names:
            header += [f"c2st_ranks_{name}"]
        w.writerow(header)
        for r in rows:
            w.writerow([r["seed"], *r["ks_pvals"], *r["c2st_ranks"]])
    print(f"\n[05b] wrote {csv_path}")

    # ---- summary.txt ----
    summary_path = args.out / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Multi-seed SBC summary\n")
        f.write(f"n_sbc                  : {args.n_sbc}\n")
        f.write(f"n_posterior_samples    : {args.n_posterior_samples}\n")
        f.write(f"K seeds                : {K}\n")
        f.write(f"seeds                  : {args.seeds}\n\n")

        f.write("Per-parameter aggregation across seeds:\n")
        f.write("  PASS = c2st_ranks max <= 0.60 (primary, stable across seeds)\n")
        f.write("  KS p median > 0.05 = consistent with uniform (secondary)\n\n")

        # Header
        f.write(
            f"  {'param':14s}  {'med KS p':>9s}  {'frac>0.05':>9s}"
            f"  {'c2st_r mean':>11s}  {'c2st_r max':>10s}"
            f"  {'verdict':>9s}\n"
        )
        any_c2st_fail = False
        for k, name in enumerate(param_names):
            verdict = "PASS" if max_c2st_ranks[k] <= 0.60 else "FAIL"
            if max_c2st_ranks[k] > 0.60:
                any_c2st_fail = True
            f.write(
                f"  {name:14s}  {median_ks[k]:>9.4f}  {frac_pass[k]:>9.2f}"
                f"  {mean_c2st_ranks[k]:>11.4f}  {max_c2st_ranks[k]:>10.4f}"
                f"  {verdict:>9s}\n"
            )
        f.write(
            f"\nOverall (c2st_ranks-based): "
            f"{'PASS' if not any_c2st_fail else 'AT LEAST ONE FAIL'}\n"
        )

        # Raw per-seed table
        f.write("\nPer-seed KS p-values:\n")
        f.write("  " + " ".join(f"{n:>10s}" for n in ["seed"] + list(param_names)) + "\n")
        for r in rows:
            f.write(
                "  "
                + " ".join(
                    [f"{r['seed']:>10d}"]
                    + [f"{p:>10.4f}" for p in r["ks_pvals"]]
                )
                + "\n"
            )
        f.write("\nPer-seed c2st_ranks:\n")
        f.write("  " + " ".join(f"{n:>10s}" for n in ["seed"] + list(param_names)) + "\n")
        for r in rows:
            f.write(
                "  "
                + " ".join(
                    [f"{r['seed']:>10d}"]
                    + [f"{c:>10.4f}" for c in r["c2st_ranks"]]
                )
                + "\n"
            )

    print(f"[05b] wrote {summary_path}")

    print("\n=== Multi-seed SBC verdict (c2st_ranks-based) ===")
    for k, name in enumerate(param_names):
        verdict = "PASS" if max_c2st_ranks[k] <= 0.60 else "FAIL"
        print(
            f"  {name:14s}  med KS p={median_ks[k]:.3f}"
            f"  frac>0.05={frac_pass[k]:.2f}"
            f"  c2st_r max={max_c2st_ranks[k]:.3f}  [{verdict}]"
        )


if __name__ == "__main__":
    main()
