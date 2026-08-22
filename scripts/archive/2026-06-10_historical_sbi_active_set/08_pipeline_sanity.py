"""
08 — Minimal SBI pipeline sanity test.

What this does:
    Trains `sbi.NPE` directly (no CurvePosterior wrapper, no FCEmbedding)
    on a small, intentionally easy 2-parameter sub-problem:

      Free params: q_burst and Q_max
      Other 7 params: fixed at the prior midpoint, except log_tau_burst
                      fixed to -3 so the burst is concentrated in the
                      first 0.5 day.
      Observation:  3-D handcrafted features [Q(0.5d), Q(t_end), max_slope]

    ADR-026 changed Q(0) from q_burst to 0. q_burst is therefore no longer
    directly observable at t=0. This smoke test fixes tau_burst small enough
    that Q(0.5d) carries the burst signal, while Q(t_end) carries the
    asymptote signal. Both KS p > 0.05 in SBC is the bare-minimum signal
    that the pipeline works.

Why this exists. After repeated full-param SBC failures, we have
to rule out upstream pipeline bugs (save/load, z-scoring, sbi-prior
format interaction, device handling) before continuing to attribute the
failure to prior / embedding / flow. If even this trivial setup fails
SBC, no architectural fix downstream will help.

Decision rule on output:
    both PASS   -> pipeline is healthy; proceed to prior narrowing
                   based on 07 audit (30.3 % immediate saturation).
    one FAILS   -> isolate which parameter's feature mapping is broken.
    both FAIL   -> pipeline bug (sbi interaction, z-scoring, or version
                   incompatibility). Debug sbi.NPE on a toy problem.

Run:
    .\.venv\Scripts\python.exe scripts/08_pipeline_sanity.py
    .\.venv\Scripts\python.exe scripts/08_pipeline_sanity.py --device cpu

Outputs:
    outputs/08_pipeline_sanity/sanity_results.txt
    outputs/08_pipeline_sanity/rank_histograms.png

Expected runtime: ~2 min on CUDA, ~10 min on CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi.diagnostics import check_sbc, run_sbc
from sbi.inference import NPE
from torch.distributions import Independent, Uniform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402


def build_minimal_simulator(
    full_sim: PLGABiphasic,
    t_grid: torch.Tensor,
    fixed_theta: torch.Tensor,
):
    """Return a closure: theta_2d -> 3-D handcrafted observation."""

    def minimal_sim(theta_2d: torch.Tensor) -> torch.Tensor:
        # theta_2d: (n, 2) = [q_burst, Q_max]
        n = theta_2d.shape[0]
        theta_full = fixed_theta.to(theta_2d.device).repeat(n, 1)
        theta_full[:, 6] = theta_2d[:, 0]   # q_burst
        theta_full[:, 8] = theta_2d[:, 1]   # Q_max
        with torch.no_grad():
            Q = full_sim.simulate(theta_full, t_grid.to(theta_2d.device))
        # Handcrafted 3-D feature
        # t_grid includes 0.5d explicitly. With log_tau_burst fixed at -3,
        # this is a near-direct burst readout without reverting to Q(0)>0.
        Q_early = Q[:, 1]
        Q_end = Q[:, -1]
        dQ = Q[:, 1:] - Q[:, :-1]
        dt = (t_grid[1:] - t_grid[:-1]).to(Q.device)
        slopes = dQ / dt
        max_slope = slopes.max(dim=1).values
        return torch.stack([Q_early, Q_end, max_slope], dim=-1).cpu()

    return minimal_sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-sbc", type=int, default=300)
    parser.add_argument("--n-posterior-samples", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-sigma", type=float, default=0.03,
        help="observation noise stddev on x (default 0.03 per ADR-014). "
             "Set to 0 to reproduce the noiseless failure mode.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/08_pipeline_sanity"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[08] device: {args.device}")
    print(f"[08] n_train: {args.n_train}, n_sbc: {args.n_sbc}")

    full_sim = PLGABiphasic()
    # Include an early probe point for the continuous-burst model. The
    # canonical 64-point grid's first positive point is ~1.43d, too late
    # to isolate the small-tau burst term cleanly.
    t_grid = torch.cat([torch.tensor([0.0, 0.5]), torch.linspace(90.0 / 63.0, 90.0, 63)])

    # Fix non-target params at the midpoint of the full prior. For this
    # sanity test, fix log_tau_burst at -3 so q_burst has a clean early-time
    # signature instead of being intentionally partially non-identifiable.
    base = full_sim.prior().base_dist
    midpoint = 0.5 * (base.low + base.high)
    fixed_theta = midpoint.clone()
    fixed_theta[7] = -3.0  # log_tau_burst; fast but still continuous.
    print("[08] fixed parameter values:")
    for k, name in enumerate(full_sim.param_names):
        if name not in {"q_burst", "Q_max"}:
            print(f"       {name:14s} = {fixed_theta[k].item():+.4f}")

    # 2-D prior over (q_burst, Q_max), matching the full prior bounds.
    prior_device = torch.device(args.device)
    minimal_prior = Independent(
        Uniform(
            low=torch.tensor([base.low[6].item(), base.low[8].item()], device=prior_device),
            high=torch.tensor([base.high[6].item(), base.high[8].item()], device=prior_device),
        ),
        1,
    )
    print("[08] minimal prior bounds:")
    print(f"       q_burst in [{base.low[6].item():.2f}, {base.high[6].item():.2f}]")
    print(f"       Q_max   in [{base.low[8].item():.2f}, {base.high[8].item():.2f}]")

    minimal_sim = build_minimal_simulator(full_sim, t_grid, fixed_theta)

    # Generate synthetic (theta, x) pairs.
    print(f"[08] generating {args.n_train} synthetic pairs...")
    t0 = time.time()
    theta_train = minimal_prior.sample((args.n_train,))
    x_train = minimal_sim(theta_train).to(prior_device)
    # ADR-014: noiseless simulator -> delta posterior -> MAF cannot fit ->
    # SBC catastrophically fails. Inject Gaussian observation noise.
    if args.noise_sigma > 0:
        x_train = (
            x_train + args.noise_sigma * torch.randn_like(x_train)
        ).clamp(0.0, 1.0)
        print(f"[08] applied observation noise sigma={args.noise_sigma}")
    print(f"[08] sim done in {time.time() - t0:.1f}s. "
          f"x_train mean: {x_train.mean(dim=0).tolist()}")
    print(f"     x_train std:  {x_train.std(dim=0).tolist()}")

    # Train sbi.NPE directly — no CurvePosterior, no FCEmbedding.
    print("[08] training NPE-MAF (no embedding net)...")
    t0 = time.time()
    inference = NPE(
        prior=minimal_prior,
        density_estimator="maf",
        device=args.device,
        show_progress_bars=False,
    )
    inference.append_simulations(theta_train, x_train)
    estimator = inference.train(
        training_batch_size=128,
        max_num_epochs=args.max_epochs,
        stop_after_epochs=20,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(estimator)
    print(f"[08] training done in {time.time() - t0:.1f}s")

    # SBC — apply the SAME noise model as training (ADR-014).
    print(f"[08] running SBC with {args.n_sbc} replicates...")
    theta_sbc = minimal_prior.sample((args.n_sbc,))
    x_sbc = minimal_sim(theta_sbc).to(prior_device)
    if args.noise_sigma > 0:
        x_sbc = (
            x_sbc + args.noise_sigma * torch.randn_like(x_sbc)
        ).clamp(0.0, 1.0)
    ranks, dap = run_sbc(
        thetas=theta_sbc,
        xs=x_sbc,
        posterior=posterior,
        num_posterior_samples=args.n_posterior_samples,
    )
    check_stats = check_sbc(
        ranks=ranks,
        prior_samples=minimal_prior.sample((args.n_sbc,)),
        dap_samples=dap,
        num_posterior_samples=args.n_posterior_samples,
    )

    # Plot 2-panel rank histogram.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    expected = args.n_sbc / 20
    param_names = ["q_burst", "Q_max"]
    summary_rows = []
    for k, name in enumerate(param_names):
        ax = axes[k]
        ax.hist(ranks[:, k].numpy(), bins=20, edgecolor="black", alpha=0.85)
        ax.axhline(expected, color="red", linestyle="--", linewidth=1)
        ks_p = float(check_stats["ks_pvals"][k])
        flag = "PASS" if ks_p > 0.05 else "FAIL"
        ax.set_title(f"{name}  KS p={ks_p:.4f}  [{flag}]", fontsize=10)
        ax.set_xlabel("rank")
        summary_rows.append((name, ks_p, flag))
    axes[0].set_ylabel("count")
    fig.suptitle(f"Pipeline sanity SBC  (n_train={args.n_train}, n_sbc={args.n_sbc})")
    fig.tight_layout()
    fig.savefig(args.out / "rank_histograms.png", dpi=150)
    plt.close(fig)

    # Write summary
    pass_count = sum(1 for _, _, flag in summary_rows if flag == "PASS")
    verdict = (
        "PIPELINE HEALTHY"  if pass_count == 2 else
        "PARTIAL — one feature mapping suspect" if pass_count == 1 else
        "PIPELINE BUG — debug sbi/save-load/zscore"
    )
    summary_lines = [
        f"n_train               : {args.n_train}",
        f"n_sbc                 : {args.n_sbc}",
        f"n_posterior_samples   : {args.n_posterior_samples}",
        f"device                : {args.device}",
        f"seed                  : {args.seed}",
        "",
        "observation features  : [Q(0.5d), Q(t_end), max_slope] (3-D handcrafted)",
        "free params           : q_burst, Q_max (others fixed; log_tau_burst=-3)",
        "",
        "Per-parameter KS p-values:",
    ]
    for name, ks_p, flag in summary_rows:
        summary_lines.append(f"  {name:10s}  KS p = {ks_p:.4f}   {flag}")
    summary_lines.extend(["", f"Verdict: {verdict}"])
    summary = "\n".join(summary_lines)
    (args.out / "sanity_results.txt").write_text(summary + "\n", encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":
    main()
