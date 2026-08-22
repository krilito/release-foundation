"""
07b — Burst-term prior audit (ADR-026 Step 3).

Consumes: simulator prior (9-param, ADR-026).
Produces: outputs/07b_burst_prior_audit/audit_summary.txt + per_sample_stats.csv
Expected runtime: ~1 min for 5000 samples on CPU.

What this checks (ADR-026 Step 3 failure-mode thresholds, all must be < 10%):

  (A) burst_vs_kd_collapse [REPLACES ADR-026 step-3 (A) — session 6 rev].
      For each prior sample with q_burst > 0.05, sweep an alternative
      θ' with q_burst = 0 and a swept log_kd over a fine grid; record
      the best max|Q(θ) - Q(θ')| over the canonical t_grid. If best
      match < observation noise σ=0.03 (ADR-014), the burst-active θ
      is indistinguishable from a kd-only alternative at noise scale =
      identifiability collapse candidate for (q_burst, log_kd).
      Threshold: < 10% of burst-active samples collapse.

      Why this replaces the original (A). ADR-026's original step-3 (A)
      was an internally contradictory pass/fail on a descriptive stat
      ("burst is fast enough"). User reviewed session-6 audit code and
      rejected unilateral downgrade to descriptive; this is the rewrite:
      a real noise-floor identifiability test on the #1 ADR-026 concern.

  (B) burst_then_stop: fraction of curves with Q(1 d) > 0.5 · Q_max but the
      slope on [1, 90] d is < 0.01 · Q_max / day (= burst delivers most of
      the release, then nothing). If high, log_tau_burst lower bound is too
      aggressive OR q_burst upper bound interacts pathologically with
      slow-kd/ke samples.

  (C) monotone_violations: fraction of curves where Q is not monotone
      non-decreasing within 1e-3 tolerance. ADR-026 worried the stiff
      exp(-t/tau) near t=0 might require atol/rtol tuning.

If any fraction > 10%, narrow prior bounds and rerun (ADR-016 pattern).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402


def run_audit(n_samples: int, seed: int, out_dir: Path) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    sim = PLGABiphasic()
    theta = sim.sample_prior(n_samples)

    # Use the canonical t_grid (ADR-012) augmented with the early-time
    # probe points the audit needs.
    t_extra = torch.tensor([0.1, 1.0])
    t_grid = torch.linspace(0.0, 90.0, 64)
    t_combined, sort_idx = torch.sort(torch.cat([t_grid, t_extra]))
    # Indices of the two probes after sorting.
    idx_01 = int((t_combined == 0.1).nonzero(as_tuple=False)[0])
    idx_1d = int((t_combined == 1.0).nonzero(as_tuple=False)[0])

    with torch.no_grad():
        Q = sim.simulate(theta, t_combined)  # (n_samples, T)

    Q_max = theta[:, 8]  # 9-param layout
    q_burst = theta[:, 6]
    log_tau_burst = theta[:, 7]

    Q01 = Q[:, idx_01]
    Q1d = Q[:, idx_1d]
    Q_last = Q[:, -1]

    # (A) identifiability collapse: burst-active θ vs kd-swept twin.
    # See docstring rationale. Operates on the q_burst>0.05 subsample.
    burst_active_mask = q_burst > 0.05
    theta_ba = theta[burst_active_mask]
    # We want Q over the canonical t_grid only for the L_inf metric.
    # Re-simulate on t_grid for clarity (cost is small for n_ba samples).
    with torch.no_grad():
        Q_ba_grid = sim.simulate(theta_ba, t_grid)
    # Sweep log_kd alternatives.
    log_kd_grid = torch.linspace(-5.0, 0.0, 51)
    best_max_err = torch.full((len(theta_ba),), float("inf"))
    for log_kd_cand in log_kd_grid:
        theta_alt = theta_ba.clone()
        theta_alt[:, 6] = 0.0           # q_burst = 0
        theta_alt[:, 3] = log_kd_cand   # try this log_kd
        with torch.no_grad():
            Q_alt = sim.simulate(theta_alt, t_grid)
        max_err = (Q_ba_grid - Q_alt).abs().max(dim=1).values
        best_max_err = torch.minimum(best_max_err, max_err)
    NOISE_SIGMA = 0.03  # ADR-014
    collapse_mask = best_max_err < NOISE_SIGMA
    n_ba = int(burst_active_mask.sum().item())
    frac_A = float(collapse_mask.float().mean().item()) if n_ba > 0 else 0.0

    # (B) burst_then_stop — Q at 1 d already > 50% Q_max AND late slope ~ 0.
    late_slope = (Q_last - Q1d) / (t_combined[-1] - 1.0)
    b_mask = (Q1d > 0.5 * Q_max) & (late_slope < 0.01 * Q_max)
    frac_B = b_mask.float().mean().item()

    # (C) monotone violations beyond 1e-3 tolerance.
    diffs = Q[:, 1:] - Q[:, :-1]
    c_mask = (diffs < -1e-3).any(dim=1)
    frac_C = c_mask.float().mean().item()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "audit_summary.txt"
    csv_path = out_dir / "per_sample_stats.csv"

    with open(summary_path, "w") as f:
        f.write("# ADR-026 burst-term prior audit\n")
        f.write(f"# n_samples = {n_samples}, seed = {seed}\n")
        f.write("# 9-param prior, log_tau_burst in [-3, 0]\n\n")
        f.write("check                                                       fraction   threshold\n")
        f.write(f"(A) burst_vs_kd_collapse  L_inf<{NOISE_SIGMA} on q_b>0.05 sub  {frac_A:>8.3%}   < 10%\n")
        f.write(f"    (n_burst_active = {n_ba} of {n_samples})\n")
        f.write(f"(B) burst_then_stop  Q(1d)>0.5Qmax & flat-late              {frac_B:>8.3%}   < 10%\n")
        f.write(f"(C) monotone_violations  (diff<-1e-3)                       {frac_C:>8.3%}   < 10%\n\n")
        verdict = "PASS" if max(frac_A, frac_B, frac_C) < 0.10 else "FAIL"
        f.write(f"verdict: {verdict}\n")

    # Expand collapse_mask (defined only on burst_active subsample) back to
    # the full sample index space, so per-sample CSV stays 1-row-per-sample.
    is_A_full = torch.zeros(n_samples, dtype=torch.bool)
    ba_indices = burst_active_mask.nonzero(as_tuple=False).flatten()
    is_A_full[ba_indices] = collapse_mask

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_burst", "log_tau_burst", "Q_max",
                         "Q_at_0.1d", "Q_at_1d", "Q_last",
                         "burst_active", "is_A_collapse", "is_B", "is_C"])
        for i in range(n_samples):
            writer.writerow([
                float(q_burst[i]), float(log_tau_burst[i]), float(Q_max[i]),
                float(Q01[i]), float(Q1d[i]), float(Q_last[i]),
                bool(burst_active_mask[i].item()),
                bool(is_A_full[i].item()),
                bool(b_mask[i].item()), bool(c_mask[i].item()),
            ])

    with open(summary_path) as f:
        print(f.read())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/07b_burst_prior_audit"))
    args = p.parse_args()
    run_audit(args.n_samples, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
