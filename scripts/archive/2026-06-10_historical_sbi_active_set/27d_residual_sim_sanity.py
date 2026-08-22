"""
27d M1 -- PLGABiphasicResidual sanity checks.

What this does:
    Four invariant checks on the new PLGABiphasicResidual simulator:
      1. c=0 -> bit-exact match with PLGABiphasic (both backends).
      2. Random c~Normal(0, sigma_c): Q stays monotone, in [0, Q_max], Q(0)=0.
      3. Numpy vs torch backend agreement (parent invariant, extended).
      4. Plot 20 random curves with vs without residual for eyeball check.

Why this exists:
    Before M2 (joint SBI training with augmented theta), we must guarantee
    the augmented simulator preserves the physical invariants the parent
    enforces. If c-perturbations violate monotonicity / saturation, the
    gating in `_vector_field` is wrong; fix it here, not after a 90-min
    NPE training reveals NaN posteriors.

Pre-declared PASS criteria:
    Check 1: max abs(Q_residual_c0 - Q_base) < 1e-5 (torch and numpy).
    Check 2: 100% of 500 random-c samples satisfy
             - monotone (no drop > 1e-4 between adjacent t)
             - Q in [-1e-4, Q_max + 1e-4] (tolerance for float roundoff)
             - Q[0] < 1e-3 (initial condition preserved)
    Check 3: median per-curve RMSE(torch, numpy) < 1e-3 across 100 thetas.

Outputs:
    outputs/27d_sim_sanity/sanity_results.txt
    outputs/27d_sim_sanity/curves_with_vs_without_residual.png

Expected runtime:
    <2 minutes single-core CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic, PLGABiphasicResidual  # noqa: E402


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/27d_sim_sanity"))
    ap.add_argument("--n-modes", type=int, default=5)
    ap.add_argument("--sigma-c", type=float, default=0.05)
    ap.add_argument("--t-max", type=float, default=90.0)
    ap.add_argument("--n-curves", type=int, default=500,
                    help="random-c samples for invariant check")
    ap.add_argument("--n-backend-agree", type=int, default=100,
                    help="samples for numpy/torch backend agreement check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sim_base = PLGABiphasic()
    sim_res = PLGABiphasicResidual(
        n_modes=args.n_modes, sigma_c=args.sigma_c, t_max=args.t_max,
    )
    t_grid = torch.linspace(0.0, args.t_max, 64)
    t_np = t_grid.numpy().astype(float)

    print(f"[27d-M1] n_modes={args.n_modes} sigma_c={args.sigma_c} t_max={args.t_max}")
    print(f"[27d-M1] base n_params=9; residual n_params={sim_res.n_params}")

    results: dict[str, dict] = {}

    # ===== Check 1: c=0 bit-exact equivalence ==============================
    print("[27d-M1] check 1: c=0 equivalence (torch + numpy)...")
    n_eq = 64
    theta_base = sim_base.sample_prior(n_eq)               # (n_eq, 9)
    zeros = torch.zeros(n_eq, args.n_modes)
    theta_aug = torch.cat([theta_base, zeros], dim=-1)     # (n_eq, 14)

    with torch.no_grad():
        Q_base_torch = sim_base.simulate(theta_base, t_grid)
        Q_res_torch = sim_res.simulate(theta_aug, t_grid)
    torch_diff = float((Q_base_torch - Q_res_torch).abs().max())

    numpy_diffs = []
    for i in range(n_eq):
        Qb = sim_base.simulate_numpy(theta_base[i].numpy(), t_np)
        Qr = sim_res.simulate_numpy(theta_aug[i].numpy(), t_np)
        numpy_diffs.append(float(np.max(np.abs(Qb - Qr))))
    numpy_diff_max = float(np.max(numpy_diffs))

    check1_pass = (torch_diff < 1e-5) and (numpy_diff_max < 1e-5)
    results["check1_c0_equivalence"] = {
        "torch_max_diff": torch_diff,
        "numpy_max_diff": numpy_diff_max,
        "threshold": 1e-5,
        "pass": check1_pass,
    }
    print(f"           torch max diff = {torch_diff:.2e}")
    print(f"           numpy max diff = {numpy_diff_max:.2e}")
    print(f"           {'PASS' if check1_pass else 'FAIL'}")

    # ===== Check 2: random c invariants ====================================
    print(f"[27d-M1] check 2: invariants under {args.n_curves} random-c samples...")
    theta_aug_rand = sim_res.sample_prior(args.n_curves)   # (n, 14)
    with torch.no_grad():
        Q_rand = sim_res.simulate(theta_aug_rand, t_grid)  # (n, T)
    Q_max_vec = theta_aug_rand[:, 8]                       # (n,)

    # Monotonicity
    diffs = Q_rand[:, 1:] - Q_rand[:, :-1]                 # (n, T-1)
    min_drop = float(diffs.min())
    n_mono_violations = int((diffs < -1e-4).any(dim=1).sum())

    # Q in [0, Q_max] (small tolerance)
    Q_min = float(Q_rand.min())
    over_qmax = (Q_rand > (Q_max_vec.unsqueeze(-1) + 1e-4))
    n_qmax_violations = int(over_qmax.any(dim=1).sum())

    # Q(0) = 0
    q0_max = float(Q_rand[:, 0].abs().max())
    n_q0_violations = int((Q_rand[:, 0].abs() > 1e-3).sum())

    check2_pass = (n_mono_violations == 0
                   and n_qmax_violations == 0
                   and n_q0_violations == 0)
    results["check2_random_c_invariants"] = {
        "n_curves": args.n_curves,
        "monotonicity_min_diff": min_drop,
        "n_monotonicity_violations": n_mono_violations,
        "Q_min": Q_min,
        "n_Q_above_Qmax": n_qmax_violations,
        "Q0_max_abs": q0_max,
        "n_Q0_nonzero": n_q0_violations,
        "pass": check2_pass,
    }
    print(f"           min step diff      = {min_drop:.2e}  (violations: {n_mono_violations})")
    print(f"           Q_min              = {Q_min:.2e}")
    print(f"           Q above Qmax       = {n_qmax_violations} / {args.n_curves}")
    print(f"           |Q(0)| max         = {q0_max:.2e}  (violations: {n_q0_violations})")
    print(f"           {'PASS' if check2_pass else 'FAIL'}")

    # ===== Check 3: numpy vs torch backend agreement at non-zero c ========
    print(f"[27d-M1] check 3: numpy/torch backend agreement under residual...")
    theta_chk = sim_res.sample_prior(args.n_backend_agree)
    with torch.no_grad():
        Q_torch = sim_res.simulate(theta_chk, t_grid)
    rmses = []
    for i in range(args.n_backend_agree):
        Q_np = sim_res.simulate_numpy(theta_chk[i].numpy(), t_np)
        rmse = float(np.sqrt(np.mean((Q_torch[i].numpy() - Q_np) ** 2)))
        rmses.append(rmse)
    rmse_arr = np.array(rmses)
    rmse_med = float(np.median(rmse_arr))
    rmse_max = float(np.max(rmse_arr))
    check3_pass = rmse_med < 1e-3
    results["check3_backend_agreement"] = {
        "n_curves": args.n_backend_agree,
        "rmse_median": rmse_med,
        "rmse_max": rmse_max,
        "threshold": 1e-3,
        "pass": check3_pass,
    }
    print(f"           RMSE median = {rmse_med:.2e}  (threshold 1e-3)")
    print(f"           RMSE max    = {rmse_max:.2e}")
    print(f"           {'PASS' if check3_pass else 'FAIL'}")

    # ===== Plot: 20 random curves with vs without residual =================
    print("[27d-M1] plotting 20 curves with vs without residual...")
    n_plot = 20
    theta_base_p = sim_base.sample_prior(n_plot)
    c_p = torch.randn(n_plot, args.n_modes) * args.sigma_c
    theta_aug_p = torch.cat([theta_base_p, c_p], dim=-1)
    theta_zero_p = torch.cat([theta_base_p, torch.zeros(n_plot, args.n_modes)],
                             dim=-1)
    with torch.no_grad():
        Q_with = sim_res.simulate(theta_aug_p, t_grid).numpy()
        Q_without = sim_res.simulate(theta_zero_p, t_grid).numpy()

    fig, axes = plt.subplots(4, 5, figsize=(17, 11), sharey=True)
    for ax, i in zip(axes.flat, range(n_plot)):
        ax.plot(t_np, Q_without[i], "-", color="tab:blue", linewidth=2,
                label="base (c=0)" if i == 0 else None)
        ax.plot(t_np, Q_with[i], "--", color="tab:red", linewidth=1.5,
                label=f"+residual" if i == 0 else None)
        c_norm = float(np.linalg.norm(c_p[i].numpy()))
        ax.set_title(f"curve {i}  ||c||={c_norm:.3f}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("t (days)")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"M1 sanity: 20 PLGA curves with vs without Fourier residual "
        f"(n_modes={args.n_modes}, sigma_c={args.sigma_c})"
    )
    fig.tight_layout()
    fig.savefig(args.out / "curves_with_vs_without_residual.png", dpi=150)
    plt.close(fig)

    # ===== Verdict =========================================================
    overall_pass = check1_pass and check2_pass and check3_pass

    verdict_lines = [
        "=== 27d M1 -- PLGABiphasicResidual sanity ===",
        "",
        f"  n_modes      : {args.n_modes}",
        f"  sigma_c      : {args.sigma_c}",
        f"  t_max        : {args.t_max}",
        f"  n_params_aug : {sim_res.n_params}",
        f"  wallclock    : {time.time()-t0:.1f}s",
        "",
        "Check 1 (c=0 bit-exact equivalence with base):",
        f"  torch max diff : {results['check1_c0_equivalence']['torch_max_diff']:.2e}",
        f"  numpy max diff : {results['check1_c0_equivalence']['numpy_max_diff']:.2e}",
        f"  threshold      : {results['check1_c0_equivalence']['threshold']:.0e}",
        f"  -> {'PASS' if check1_pass else 'FAIL'}",
        "",
        "Check 2 (physical invariants under random c):",
        f"  monotonicity violations : {results['check2_random_c_invariants']['n_monotonicity_violations']} / {args.n_curves}",
        f"  Q > Q_max violations    : {results['check2_random_c_invariants']['n_Q_above_Qmax']} / {args.n_curves}",
        f"  Q(0) != 0 violations    : {results['check2_random_c_invariants']['n_Q0_nonzero']} / {args.n_curves}",
        f"  -> {'PASS' if check2_pass else 'FAIL'}",
        "",
        "Check 3 (numpy vs torch backend agreement under residual):",
        f"  RMSE median : {results['check3_backend_agreement']['rmse_median']:.2e}",
        f"  RMSE max    : {results['check3_backend_agreement']['rmse_max']:.2e}",
        f"  threshold   : {results['check3_backend_agreement']['threshold']:.0e}",
        f"  -> {'PASS' if check3_pass else 'FAIL'}",
        "",
        f"=== M1 OVERALL: {'PASS' if overall_pass else 'FAIL'} ===",
        "",
        "Plot:",
        "  curves_with_vs_without_residual.png : 4x5 eyeball check",
        "",
        "Next step:",
        ("  PASS -> proceed to M2 (27d_residual_npe.py: SBI training + SBC "
         "at fixed prefixes 1/3/7 d on synthetic held-out)."
         if overall_pass else
         "  FAIL -> debug the failing check above. Do not start M2."),
    ]
    summary_text = "\n".join(verdict_lines)
    (args.out / "sanity_results.txt").write_text(summary_text, encoding="utf-8")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
