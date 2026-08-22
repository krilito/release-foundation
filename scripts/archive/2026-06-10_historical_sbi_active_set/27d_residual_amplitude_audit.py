"""
27d M1.5 -- residual amplitude audit (forward-only, no SBI).

What this does:
    Sample N (theta_base, c) pairs from the joint prior. For each:
      - simulate Q_base (c = 0)
      - simulate Q_res  (with sampled c)
      - compute amplitude statistics of the residual:
          * sup_t |Q_res - Q_base|       (worst pointwise deviation)
          * AUC(|Q_res - Q_base|)        (trapezoidal area under |ΔQ| over t)
          * argmax_t |Q_res - Q_base|    (timing of peak deviation)

Why this exists (M1.5 between M1 and M2):
    M1 proves the residual preserves physical invariants (Q monotone, in
    [0, Q_max], Q(0)=0). It does NOT prove the residual is small. A small
    instantaneous rate `residual(t) * (Q_max - Q)` can integrate over
    90 days to a large cumulative Q deviation. If the prior puts
    significant mass on c that rewrites release timing rather than
    nudging it, the residual is functionally a second mechanism model,
    not a misspecification correction. That would risk identifiability
    collapse in M2 (R1 in plan_27d).

Pre-declared thresholds (these are interpretive bins, not pass/fail):
    "small"    : sup_t |ΔQ| <  5% of Q-range  (good: small correction prior)
    "moderate" : sup_t |ΔQ| < 15%             (acceptable; residual visible)
    "large"    : sup_t |ΔQ| > 15%             (red flag; lower sigma_c)
    "huge"     : sup_t |ΔQ| > 30%             (definitely lower sigma_c)

Decision rule (informal):
    If >= 80% of samples are "small" or "moderate"
        AND median sup_t |ΔQ| < 0.08
        -> proceed to full M2 at current sigma_c.
    Else
        -> reduce sigma_c and rerun this audit, then full M2.

Outputs:
    outputs/27d_residual_amplitude_audit/amplitude_summary.txt
    outputs/27d_residual_amplitude_audit/distributions.png
    outputs/27d_residual_amplitude_audit/sample_curves.png

Expected runtime: < 2 min on CPU.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic, PLGABiphasicResidual  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/27d_residual_amplitude_audit"))
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="number of (theta_base, c) samples")
    ap.add_argument("--n-modes", type=int, default=5)
    ap.add_argument("--sigma-c", type=float, default=0.05)
    ap.add_argument("--t-max", type=float, default=90.0)
    ap.add_argument("--n-grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sim_base = PLGABiphasic()
    sim_res = PLGABiphasicResidual(
        n_modes=args.n_modes, sigma_c=args.sigma_c, t_max=args.t_max,
    )
    t_grid = torch.linspace(0.0, args.t_max, args.n_grid)
    t_np = t_grid.numpy().astype(float)

    print(f"[M1.5] n_samples={args.n_samples} n_modes={args.n_modes} sigma_c={args.sigma_c}")

    # Sample joint (theta_base, c)
    theta_aug = sim_res.sample_prior(args.n_samples)        # (N, 14)
    theta_base = theta_aug[:, :9]
    c = theta_aug[:, 9:]
    theta_zero = torch.cat([theta_base, torch.zeros(args.n_samples, args.n_modes)], dim=-1)

    # Forward simulate
    print("[M1.5] simulating base and residual curves...")
    with torch.no_grad():
        Q_base = sim_res.simulate(theta_zero, t_grid).numpy()  # (N, T)
        Q_res = sim_res.simulate(theta_aug, t_grid).numpy()    # (N, T)
    delta = Q_res - Q_base                                       # (N, T)

    # Per-sample stats
    sup_dq = np.max(np.abs(delta), axis=1)                       # (N,)
    auc_dq = np.trapezoid(np.abs(delta), t_np, axis=1) / args.t_max  # (N,) avg-magnitude norm
    argmax_dq_t = t_np[np.argmax(np.abs(delta), axis=1)]         # (N,)
    c_norm = np.linalg.norm(c.numpy(), axis=1)                   # (N,)
    qmax = theta_base[:, 8].numpy()                              # (N,)
    rel_sup = sup_dq / np.maximum(qmax, 1e-6)                    # (N,) sup as fraction of Q_max

    # Bins
    bins = {
        "small (<5%)":     int(np.sum(rel_sup < 0.05)),
        "moderate (5-15%)": int(np.sum((rel_sup >= 0.05) & (rel_sup < 0.15))),
        "large (15-30%)":   int(np.sum((rel_sup >= 0.15) & (rel_sup < 0.30))),
        "huge (>=30%)":     int(np.sum(rel_sup >= 0.30)),
    }

    pct = {k: v / args.n_samples * 100 for k, v in bins.items()}
    frac_small_or_moderate = (bins["small (<5%)"] + bins["moderate (5-15%)"]) / args.n_samples

    # Quantiles
    qs = [0.50, 0.75, 0.90, 0.95, 0.99]
    sup_qs = {q: float(np.quantile(sup_dq, q)) for q in qs}
    rel_qs = {q: float(np.quantile(rel_sup, q)) for q in qs}
    auc_qs = {q: float(np.quantile(auc_dq, q)) for q in qs}

    # ===== plots ===========================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(sup_dq, bins=50, edgecolor="black")
    for thr, lbl, col in [(0.05, "5%", "green"), (0.15, "15%", "orange"), (0.30, "30%", "red")]:
        ax.axvline(thr, color=col, linestyle="--", label=lbl)
    ax.set_xlabel("sup_t |Q_res - Q_base|  (absolute units)")
    ax.set_ylabel("count")
    ax.set_title("Worst pointwise deviation per sample")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(rel_sup, bins=50, edgecolor="black")
    for thr, lbl, col in [(0.05, "5%", "green"), (0.15, "15%", "orange"), (0.30, "30%", "red")]:
        ax.axvline(thr, color=col, linestyle="--", label=lbl)
    ax.set_xlabel("sup_t |ΔQ| / Q_max  (relative)")
    ax.set_ylabel("count")
    ax.set_title("Relative sup-deviation (sub-divides the prior into regimes)")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(auc_dq, bins=50, edgecolor="black")
    ax.set_xlabel("AUC(|ΔQ|) / t_max  (time-averaged magnitude)")
    ax.set_ylabel("count")
    ax.set_title("Integrated deviation over time")

    ax = axes[1, 1]
    ax.hist(argmax_dq_t, bins=40, edgecolor="black")
    ax.set_xlabel("argmax_t |ΔQ|  (days)")
    ax.set_ylabel("count")
    ax.set_title("Timing of peak deviation")
    fig.suptitle(
        f"M1.5 amplitude audit (n={args.n_samples}, n_modes={args.n_modes}, sigma_c={args.sigma_c})"
    )
    fig.tight_layout()
    fig.savefig(args.out / "distributions.png", dpi=150)
    plt.close(fig)

    # Sample curves at three relative-sup percentiles for visual reference
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey=True)
    target_pcts = {"low (10th pct rel sup)": 0.10,
                   "med (50th pct rel sup)": 0.50,
                   "high (90th pct rel sup)": 0.90}
    for row, (label, p) in enumerate(target_pcts.items()):
        target_val = np.quantile(rel_sup, p)
        idx_sorted = np.argsort(np.abs(rel_sup - target_val))[:5]
        for col, idx in enumerate(idx_sorted):
            ax = axes[row, col]
            ax.plot(t_np, Q_base[idx], "-", color="tab:blue", linewidth=2,
                    label="base" if col == 0 else None)
            ax.plot(t_np, Q_res[idx], "--", color="tab:red", linewidth=1.5,
                    label="+residual" if col == 0 else None)
            ax.set_title(
                f"rel_sup={rel_sup[idx]:.3f}, ||c||={c_norm[idx]:.3f}",
                fontsize=8,
            )
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("t (days)")
        axes[row, 0].set_ylabel(f"{label}\nQ(t)")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Sample curves at low / median / high relative-sup percentiles")
    fig.tight_layout()
    fig.savefig(args.out / "sample_curves.png", dpi=150)
    plt.close(fig)

    # ===== decision ========================================================
    median_rel_sup = rel_qs[0.50]
    decision_pass = (frac_small_or_moderate >= 0.80) and (median_rel_sup < 0.08)

    summary_lines = [
        "=== 27d M1.5 -- residual amplitude audit ===",
        "",
        f"  n_samples   : {args.n_samples}",
        f"  n_modes     : {args.n_modes}",
        f"  sigma_c     : {args.sigma_c}",
        f"  t_max       : {args.t_max}",
        f"  wallclock   : {time.time()-t0:.1f}s",
        "",
        "Bin counts (sup_t |ΔQ| as fraction of Q_max per curve):",
        *[f"  {name:18s} : {n:5d}  ({pct[name]:5.1f}%)"
          for name, n in bins.items()],
        "",
        "Quantiles of sup_t |Q_res - Q_base| (absolute):",
        *[f"  q{int(q*100):02d} = {sup_qs[q]:.4f}" for q in qs],
        "",
        "Quantiles of sup_t |ΔQ| / Q_max (relative):",
        *[f"  q{int(q*100):02d} = {rel_qs[q]:.4f}" for q in qs],
        "",
        "Quantiles of AUC(|ΔQ|) / t_max (time-averaged magnitude):",
        *[f"  q{int(q*100):02d} = {auc_qs[q]:.4f}" for q in qs],
        "",
        f"||c||_2 quantiles: q50={np.quantile(c_norm, 0.50):.4f}, "
        f"q95={np.quantile(c_norm, 0.95):.4f}, "
        f"max={float(np.max(c_norm)):.4f}",
        f"  (prior std sigma_c * sqrt(n_modes) = {args.sigma_c * np.sqrt(args.n_modes):.4f})",
        "",
        "=== DECISION ===",
        f"  frac small+moderate (sup<15% Q_max)  : {frac_small_or_moderate:.3f}",
        f"  median sup_t |ΔQ|/Q_max              : {median_rel_sup:.4f}",
        f"  threshold (frac >= 0.80 AND median < 0.08)",
        f"",
        f"  -> {'PROCEED with current sigma_c=' + str(args.sigma_c) if decision_pass else 'REDUCE sigma_c (try halving and rerun this audit)'}",
        "",
        "Plots:",
        "  distributions.png : sup, rel_sup, AUC, peak-time histograms",
        "  sample_curves.png : 15 curves at low/median/high rel-sup percentiles",
    ]
    (args.out / "amplitude_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8",
    )
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
