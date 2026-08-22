"""
27b -- SBC diagnostic for widened-prior proposal (post-27a audit).

What this does:
    1. Define PLGABiphasicWidened: PLGABiphasic with upper bounds widened on
       log_kw, log_alpha, q_burst (the three params 27a flagged at 39-44%
       boundary-hit on 321 full-curve fits).
    2. Train a fresh CurvePosterior (q_phi_widened) on synthetic pairs drawn
       from the widened prior. Same flow / embedding / noise as production
       q_phi from script 04.
    3. Run SBC on n_sbc held-out widened-prior replicates. Report KS p-values
       per parameter (same convention as scripts/05).
    4. Verdict: does widened prior pass SBC? If yes -> safe to feed into 27.
       If no -> revert to tight prior and accept ~40% boundary pinning.

Why this exists:
    27a found that the 9-D PLGA ODE has high expressivity on 321 (median
    full-curve R^2 = 0.998, 95% well-fit), so STOP is overturned. BUT 35-44%
    of curves have theta_hat pinned at upper bounds of log_kw / log_alpha /
    q_burst. The current bounds were tightened by ADR-015/016 to fix
    immediate-saturation SBC failures. So widening might re-introduce those
    failures.

    AGENTS.md Hard YES #1 requires diagnostic-before-prior-change. 27a is
    the first piece of evidence (data demands wider bounds). 27b is the
    second (SBC under widened bounds still calibrated). Both passing = green
    light to update ADR and use widened prior in 27.

Widening (small-scale, audit-justified):
    log_kw_hi      : 1.0  -> 1.5  (kw_max: 2.7 -> 4.5 /day)
    log_alpha_hi   : 1.0  -> 1.5  (alpha_max: 2.7 -> 4.5)
    q_burst_hi     : 0.30 -> 0.40 (max burst: 30% -> 40%)
    [other 6 dims unchanged; lower bounds unchanged]

    These move halfway back toward the pre-ADR-015 bounds. If 27b SBC
    passes, we have evidence the calibration failure ADR-015 fixed was
    driven by the OUTER half of the widening (1.0 -> 2.0), not the inner
    half (1.0 -> 1.5). If it fails, we revert.

Pre-declared PASS criterion:
    All 9 parameter KS p-values > 0.05 on n_sbc=300 held-out replicates.
    Same threshold as scripts/05_sbc_curve_posterior.py and ADR-006.

Outputs:
    outputs/27b_widened_prior_sbc/posterior_widened.pt
    outputs/27b_widened_prior_sbc/rank_histograms.png
    outputs/27b_widened_prior_sbc/sbc_summary.txt
    outputs/27b_widened_prior_sbc/training_log.txt

Expected runtime:
    ~45-75 min on a 4060 (30-60 min training + 5-15 min SBC).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sbi.diagnostics import check_sbc, run_sbc
from torch.distributions import Distribution, Independent, Uniform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import CurvePosterior  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


class PLGABiphasicWidened(PLGABiphasic):
    """PLGABiphasic with the 3 upper bounds 27a flagged on 321 widened.

    Lower bounds and the other 6 dims are unchanged from the base class.
    Only `prior()` is overridden; the ODE / simulate / simulate_numpy paths
    are inherited verbatim, so this is a pure prior change.

    See module docstring for the audit -> widening mapping. The intent is
    to be the *smallest* widening that has any chance of relaxing the
    boundary-pinning seen in 27a, so the SBC test is informative even if
    larger widenings would also pass.
    """

    def prior(self) -> Distribution:
        lows = torch.tensor([
            -3.0,    # log_kw         (unchanged)
            -4.0,    # log_kh         (unchanged)
            -2.0,    # log_alpha      (unchanged)
            -5.0,    # log_kd         (unchanged)
            -3.0,    # log_ke         (unchanged)
             0.05,   # m_crit         (unchanged)
             0.0,    # q_burst        (unchanged)
            -3.0,    # log_tau_burst  (unchanged)
             0.50,   # Q_max          (unchanged)
        ])
        highs = torch.tensor([
             1.5,    # log_kw         WIDENED  1.0 -> 1.5  (kw 2.7 -> 4.5 /d)
             -1.0,   # log_kh         (unchanged)
             1.5,    # log_alpha      WIDENED  1.0 -> 1.5  (alpha 2.7 -> 4.5)
             0.0,    # log_kd         (unchanged)
             2.0,    # log_ke         (unchanged)
             0.50,   # m_crit         (unchanged)
             0.40,   # q_burst        WIDENED  0.30 -> 0.40 (max burst 30% -> 40%)
             0.0,    # log_tau_burst  (unchanged)
             1.00,   # Q_max          (already at physical limit; not widened)
        ])
        return Independent(Uniform(lows, highs), 1)


def _report_widening(sim_widened: PLGABiphasicWidened, sim_base: PLGABiphasic) -> str:
    """Side-by-side dump of widened vs base bounds for the summary log."""
    w_prior = sim_widened.prior().base_dist
    b_prior = sim_base.prior().base_dist
    lines = ["param            base_lo  base_hi  new_lo  new_hi  diff"]
    for k, name in enumerate(sim_widened.param_names):
        b_lo, b_hi = float(b_prior.low[k]), float(b_prior.high[k])
        w_lo, w_hi = float(w_prior.low[k]), float(w_prior.high[k])
        diff = []
        if w_lo != b_lo:
            diff.append(f"lo {b_lo:+.2f}->{w_lo:+.2f}")
        if w_hi != b_hi:
            diff.append(f"hi {b_hi:+.2f}->{w_hi:+.2f}")
        diff_str = " ".join(diff) if diff else "-"
        lines.append(
            f"  {name:14s} {b_lo:+6.2f}   {b_hi:+6.2f}   {w_lo:+6.2f}  {w_hi:+6.2f}   {diff_str}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/plga_phase1.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/27b_widened_prior_sbc"),
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=None, help="overrides config")
    parser.add_argument("--n-sim", type=int, default=None, help="overrides config")
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="overrides config; if omitted, 27b may cap the config "
                             "batch size on small CUDA cards to avoid WDDM OOM")
    parser.add_argument("--n-sbc", type=int, default=300,
                        help="number of (theta*, Q*) replicates for SBC")
    parser.add_argument("--n-posterior-samples", type=int, default=1000)
    parser.add_argument("--sbc-seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n_sim = args.n_sim or cfg["synthetic"]["n_pairs"]
    seed = args.seed if args.seed is not None else cfg["synthetic"]["seed"]
    train_batch_size = args.train_batch_size or cfg["train"]["batch_size"]
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    if args.device.startswith("cuda") and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
        if total_mem <= 9 * 1024**3 and train_batch_size > 64:
            print(
                "[27b] capping train batch size "
                f"{train_batch_size} -> 64 for <=9 GiB CUDA device"
            )
            train_batch_size = 64

    sim_widened = PLGABiphasicWidened(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    sim_base = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    widening_table = _report_widening(sim_widened, sim_base)
    print("[27b] widening:")
    print(widening_table)

    cp = CurvePosterior(
        simulator=sim_widened,
        t_grid=t_grid,
        flow=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        embedding_output_dim=cfg["posterior"].get("embedding_output_dim", 16),
        embedding_hidden=cfg["posterior"].get("embedding_hidden", 64),
        embedding_layers=cfg["posterior"].get("embedding_layers", 3),
        noise_sigma=cfg["posterior"].get("noise_sigma", 0.03),
        device=args.device,
    )
    ckpt_path = args.out / "posterior_widened.pt"

    print(f"[27b] device         : {args.device}")
    print(f"[27b] n_simulations  : {n_sim}")
    print(f"[27b] train_batch    : {train_batch_size}")
    print(f"[27b] t_grid         : {t_cfg['start']} to {t_cfg['end']} days, {t_cfg['n']} pts")
    print(f"[27b] flow           : {cfg['posterior']['flow_type']}, "
          f"hidden={cfg['posterior']['hidden_features']}, "
          f"transforms={cfg['posterior']['num_transforms']}")
    print(f"[27b] noise_sigma    : {cp.noise_sigma}")

    train_elapsed = 0.0
    if ckpt_path.exists():
        print(f"[27b] reusing existing checkpoint at {ckpt_path}")
        cp = CurvePosterior.load(ckpt_path, simulator=sim_widened, device=args.device)
        train_log = cp._training_log or {}
    else:
        t0 = time.time()
        train_log = cp.train(
            n_simulations=n_sim,
            training_batch_size=train_batch_size,
            max_num_epochs=cfg["train"]["max_epochs"],
            seed=seed,
            stop_after_epochs=cfg["train"]["early_stopping_patience"],
            verbose=True,
        )
        train_elapsed = time.time() - t0
        print(f"\n[27b] training finished in {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)")

        cp.save(ckpt_path)
        print(f"[27b] saved checkpoint to {ckpt_path}")

    # SBC: draw held-out replicates from the WIDENED prior, then check
    # whether q_phi_widened is calibrated under its own prior. This is the
    # right comparison: we are not asking "does q_phi_widened recover the
    # tight-prior truth", we are asking "is q_phi_widened a valid posterior
    # for the widened prior".
    torch.manual_seed(args.sbc_seed)
    np.random.seed(args.sbc_seed)
    est_device = cp._estimator_device()
    print(f"\n[27b] drawing {args.n_sbc} SBC replicates (noise_sigma={cp.noise_sigma})...")
    theta_star = sim_widened.sample_prior(args.n_sbc).to(est_device)
    with torch.no_grad():
        Q_star = sim_widened.simulate(theta_star, t_grid.to(est_device))
    Q_star = cp.add_obs_noise(Q_star)

    print(f"[27b] running SBC ({args.n_posterior_samples} posterior samples per replicate)...")
    sbc_t0 = time.time()
    ranks, dap_samples = run_sbc(
        thetas=theta_star,
        xs=Q_star,
        posterior=cp._posterior,
        num_posterior_samples=args.n_posterior_samples,
    )
    check_stats = check_sbc(
        ranks=ranks,
        prior_samples=sim_widened.sample_prior(args.n_sbc).to(est_device),
        dap_samples=dap_samples,
        num_posterior_samples=args.n_posterior_samples,
    )
    sbc_elapsed = time.time() - sbc_t0
    print(f"[27b] SBC done in {sbc_elapsed:.1f}s")

    # Rank histograms -- 3x3 grid for 9 params (NOT 2x4 like script 05,
    # which silently drops the 9th param after ADR-026 added q_burst,
    # log_tau_burst, Q_max).
    n_params = sim_widened.n_params
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
    axes = axes.ravel()
    expected_count = args.n_sbc / 20  # 20 bins
    for k in range(n_params):
        ax = axes[k]
        ax.hist(ranks[:, k].cpu().numpy(), bins=20, edgecolor="black", alpha=0.85)
        ax.axhline(expected_count, color="red", linestyle="--", linewidth=1,
                   label=f"uniform = {expected_count:.0f}")
        ks_p = float(check_stats["ks_pvals"][k])
        flag = "PASS" if ks_p > 0.05 else "FAIL"
        ax.set_title(f"{sim_widened.param_names[k]}  KS p={ks_p:.3f}  [{flag}]",
                     fontsize=9)
        ax.set_xlabel("rank")
        if k % 3 == 0:
            ax.set_ylabel("count")
    for k in range(n_params, len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        f"27b SBC rank histograms -- widened prior "
        f"({args.n_sbc} reps x {args.n_posterior_samples} samples)"
    )
    fig.tight_layout()
    fig.savefig(args.out / "rank_histograms.png", dpi=150)
    plt.close(fig)
    print(f"[27b] wrote {args.out / 'rank_histograms.png'}")

    # Verdict.
    ks_per_param = [float(check_stats["ks_pvals"][k]) for k in range(n_params)]
    fails = [
        name for name, p in zip(sim_widened.param_names, ks_per_param, strict=True)
        if p <= 0.05
    ]
    overall_pass = len(fails) == 0

    summary_path = args.out / "sbc_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== 27b -- widened-prior SBC diagnostic ===\n\n")
        f.write("Widening proposal (relative to current ADR-015/016 bounds):\n")
        f.write(widening_table + "\n\n")
        f.write(f"n_simulations          : {n_sim}\n")
        f.write(f"training_batch_size    : {train_batch_size}\n")
        f.write(f"training_seed          : {seed}\n")
        f.write(f"training_elapsed_min   : {train_elapsed/60:.1f}\n")
        f.write(f"sbc_seed               : {args.sbc_seed}\n")
        f.write(f"n_sbc                  : {args.n_sbc}\n")
        f.write(f"n_posterior_samples    : {args.n_posterior_samples}\n")
        f.write(f"sbc_elapsed_min        : {sbc_elapsed/60:.1f}\n\n")
        f.write("Per-parameter KS p-values (uniform-rank test):\n")
        f.write("  >0.05 = consistent with uniform = calibrated\n\n")
        for k, name in enumerate(sim_widened.param_names):
            p = ks_per_param[k]
            f.write(f"  {name:14s}  KS p = {p:.4f}   {'PASS' if p > 0.05 else 'FAIL'}\n")
        f.write("\n=== VERDICT ===\n")
        if overall_pass:
            f.write(
                "PASS. All 9 KS p > 0.05 under widened prior.\n\n"
                "Decision: it is safe to use the widened bounds in script 27. The\n"
                "calibration failure ADR-015/016 fixed was driven by the OUTER half\n"
                "of the widening (1.0 -> 2.0), not the inner half (1.0 -> 1.5).\n"
                "Recommended follow-up:\n"
                "  1. Open ADR-027 (this script + 27a per_curve.csv as evidence).\n"
                "  2. Update PLGABiphasic.prior() in simulator.py to the widened\n"
                "     bounds (one source of truth, AGENTS.md #6).\n"
                "  3. Rerun script 04 with the new bounds to refresh production q_phi.\n"
                "  4. Proceed with script 27 using the widened prior.\n"
            )
        else:
            f.write(
                f"FAIL. {len(fails)} param(s) below KS p=0.05: {', '.join(fails)}\n\n"
                "Decision: do NOT use the widened bounds. The 27a boundary-pinning\n"
                "signal is real, but the underlying calibration brittleness ADR-015\n"
                "diagnosed is also real. Options:\n"
                "  A. Try a narrower widening (e.g. log_kw_hi=1.25 not 1.5) and\n"
                "     rerun 27b.\n"
                "  B. Keep tight bounds; accept ~40%% boundary pinning on 321 in\n"
                "     script 27. Document explicitly that these point estimates\n"
                "     are mode estimates pinned to a prior edge; report posterior\n"
                "     credible intervals instead of point predictions.\n"
                "  C. Treat the boundary-pinned subset as out-of-model-support;\n"
                "     evaluate 27 on the interior subset only (be transparent about\n"
                "     coverage).\n"
            )
        f.write("\ncheck_sbc raw stats:\n")
        for k, v in check_stats.items():
            f.write(f"  {k}: {v}\n")
    print(f"[27b] wrote {summary_path}")

    log_path = args.out / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"seed                : {seed}\n")
        f.write(f"n_simulations       : {n_sim}\n")
        f.write(f"training_batch_size : {train_batch_size}\n")
        f.write(f"device              : {args.device}\n")
        f.write(f"training_elapsed_s  : {train_elapsed:.1f}\n")
        for k, v in train_log.items():
            f.write(f"{k:20s}: {v}\n")
    print(f"[27b] wrote {log_path}")

    print("\n=== 27b SBC summary ===")
    print(widening_table)
    print()
    for k, name in enumerate(sim_widened.param_names):
        p = ks_per_param[k]
        print(f"  {name:14s}  KS p = {p:.4f}   {'PASS' if p > 0.05 else 'FAIL'}")
    print(f"\nVERDICT: {'PASS -- widened prior is safe for 27' if overall_pass else 'FAIL -- see ' + str(summary_path)}")


if __name__ == "__main__":
    main()
