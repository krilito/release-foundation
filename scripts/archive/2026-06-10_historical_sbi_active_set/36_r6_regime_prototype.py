"""
36 - R6 regime-conditional fit prototype.

What this does:
    For each R6 curve (the largest regime from script 33: log_kh, log_kd,
    log_kw, Q_max modal active set), fit four R^2-yielding theta vectors:

      (a) ORACLE 9: full 9-parameter fit (from script 29 / 27a).
      (b) GREEDY 4: each curve's own best 4-parameter greedy fit (from
          script 29 k=4) -- a *per-curve* upper bound for any 4-D
          parameterization.
      (c) R6 4: fit ONLY {log_kh, log_kd, log_kw, Q_max}, with the other
          five parameters fixed at the leave-one-out median of R6
          training curves. This is the regime-conditional prototype.
      (d) R6 0: zero-shot. Use the LOO R6 median for ALL nine params,
          no fitting at all. Baseline.

    The comparison answers: is R6's modal active set
    {log_kh, log_kd, log_kw, Q_max} a usable shared parameterization for
    all R6 curves, or does the within-regime heterogeneity require
    per-curve free-param selection?

Why this exists:
    Steps 32-35c established that regimes are real, predict kinetic flag
    beyond confounders, but do NOT correspond to local linear subspaces.
    A discrete regime-conditional architecture remains the most
    defensible next step. This script is the cheapest possible test of
    that architecture on the cleanest regime (R6 had Spearman r=0.69
    between mean |v1| and modal active set, the only regime with strong
    geometric backing).

Success criteria:
    median R^2(R6-4) >= median R^2(GREEDY-4) - 0.005
        -> single R6 modal set works as well as per-curve greedy.
    median R^2(R6-4) within 0.01 of R^2(ORACLE-9)
        -> regime-conditioning loses negligible expressivity.
    R^2(R6-0) substantially worse than R^2(R6-4)
        -> the four free modal params are genuinely doing per-curve work
        (not just regurgitating the regime mean curve).

    All three together pass -> regime-conditional architecture is viable.
    Any failure -> diagnose before committing to regime-conditional NPE.

Anti-self-deception:
    - LOO median means no train/test leakage on the regime mean.
    - We report per-curve deltas, not just medians; if the median looks
      good but a tail of R6 curves is poorly fit, the architecture has
      hidden heterogeneity and we should not advertise success.
    - The 4 modal params are LOO-immune (we only LOO the 5 fixed params'
      median; the modal 4 are free per curve), so the test really is
      asking "is the R6 fixed-tail enough" not "does fitting help".

Outputs:
    outputs/36_r6_regime_prototype/per_curve_fits.csv
    outputs/36_r6_regime_prototype/r2_distribution.png
    outputs/36_r6_regime_prototype/tail_curves.csv
    outputs/36_r6_regime_prototype/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_plga_181  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


DATASET_COL = "dataset"
FID_COL = "fid"


# Modal active set is now computed per regime from script 29's
# active_set_k4 column (within-regime top-k by inclusion frequency).
# Earlier this constant was hard-coded for R6; computing it makes the
# script reusable for R5 and any other regime without code edits.
def _compute_regime_modal_set(
    df: pd.DataFrame,
    active_col: str,
    param_names: list[str],
    k: int,
) -> list[str]:
    p_idx = {n: i for i, n in enumerate(param_names)}
    counts = np.zeros(len(param_names), dtype=float)
    n = 0
    for cell in df[active_col]:
        if not isinstance(cell, str) or not cell.strip():
            continue
        n += 1
        for name in (t.strip() for t in cell.split(",") if t.strip()):
            idx = p_idx.get(name)
            if idx is not None:
                counts[idx] += 1.0
    if n == 0:
        raise ValueError("no curves found to compute modal set")
    order = np.argsort(-counts)[:k]
    return [param_names[i] for i in order]


@dataclass
class CurveRecord:
    dataset: str
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray


def _load_internal_records(csv_path: Path) -> list[CurveRecord]:
    curves = load_plga_181(csv_path)
    return [
        CurveRecord(
            dataset="internal181",
            fid=int(c.formulation_id),
            t_obs=c.t.numpy().astype(float),
            q_obs=np.clip(c.Q.numpy().astype(float), 0.0, 1.0),
        )
        for c in curves
    ]


def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
    t_grid_max_days: float,
) -> list[CurveRecord]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns={"Formulation Index": "fid"})
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    df = df[df["fid"].astype(int).isin(matched_fids)].copy()
    out: list[CurveRecord] = []
    for fid in sorted(df["fid"].unique().tolist()):
        g = df[df["fid"] == fid].sort_values("Time")
        g = g.groupby("Time", as_index=False).agg({"Release": "mean"})
        t_obs = g["Time"].to_numpy(dtype=float)
        q_obs = np.clip(g["Release"].to_numpy(dtype=float), 0.0, 1.0)
        mask = t_obs <= t_grid_max_days
        t_obs = t_obs[mask]
        q_obs = q_obs[mask]
        if len(t_obs) < 3:
            continue
        out.append(CurveRecord(dataset="cross321", fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    return out


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_subset(
    sim: PLGABiphasic,
    curve: CurveRecord,
    theta_fixed: np.ndarray,
    free_idx: list[int],
    lows: np.ndarray,
    highs: np.ndarray,
    x0_free: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit only free_idx params; other params fixed at theta_fixed."""
    def residual(z: np.ndarray) -> np.ndarray:
        theta = theta_fixed.copy()
        theta[free_idx] = z
        return sim.simulate_numpy(theta, curve.t_obs) - curve.q_obs

    z0 = np.clip(x0_free, lows[free_idx], highs[free_idx])
    try:
        result = least_squares(
            residual, x0=z0,
            bounds=(lows[free_idx], highs[free_idx]),
            method="trf", max_nfev=200,
        )
        theta = theta_fixed.copy()
        theta[free_idx] = result.x
        pred = sim.simulate_numpy(theta, curve.t_obs)
        return theta, _r2(curve.q_obs, pred)
    except Exception as e:  # noqa: BLE001
        print(f"[36] fit failed for {curve.dataset}/{curve.fid}: {e}", flush=True)
        return theta_fixed.copy(), float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-csv",
        type=Path,
        default=Path("outputs/33_active_set_regimes/regime_assignments.csv"),
    )
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument(
        "--active-set-csv",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/per_curve_active_set.csv"),
    )
    ap.add_argument(
        "--internal-data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
    )
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument("--target-regime", type=int, default=6)
    ap.add_argument("--k-modal", type=int, default=4,
                    help="modal active set size (matches 29's k_max)")
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output directory; defaults to outputs/36_r{regime}_regime_prototype",
    )
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(f"outputs/36_r{args.target_regime}_regime_prototype")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy()
    highs = prior.high.numpy()
    param_names = list(sim.param_names)
    p_idx = {n: i for i, n in enumerate(param_names)}

    # Load curves
    curves = [
        *_load_internal_records(args.internal_data),
        *_load_external_records(
            xlsx_path=args.cross_doi_data,
            matched_fids_csv=args.matched_fids_csv,
            t_grid_max_days=args.t_grid_max_days,
        ),
    ]
    curve_map = {(c.dataset, c.fid): c for c in curves}

    # Identify R6 curves with full-fit theta available
    df_reg = pd.read_csv(args.regime_csv)
    df_reg = df_reg[df_reg["regime"] == args.target_regime].copy()
    df_bank = pd.read_csv(args.full_fit_bank)
    df = df_reg.merge(
        df_bank[[DATASET_COL, FID_COL, "full_r2"] + param_names],
        on=[DATASET_COL, FID_COL],
        how="inner",
    )
    # Pull greedy k=4 R^2 from 29 too
    df_as = pd.read_csv(args.active_set_csv)
    # Drop any active_set_k4 already on df (it came from regime_csv) to avoid
    # pandas auto-suffixing on merge.
    df = df.drop(columns=[c for c in ["active_set_k4"] if c in df.columns])
    df = df.merge(
        df_as[[DATASET_COL, FID_COL, "r2_k4", "active_set_k4"]],
        on=[DATASET_COL, FID_COL],
        how="inner",
    )
    print(f"[36] R{args.target_regime} curves with all data: {len(df)} "
          f"(internal={(df[DATASET_COL]=='internal181').sum()}, "
          f"cross={(df[DATASET_COL]=='cross321').sum()})", flush=True)

    # Compute regime-specific modal active set from the data
    modal_names = _compute_regime_modal_set(df, "active_set_k4", param_names, k=args.k_modal)
    modal_idx = [p_idx[n] for n in modal_names]
    non_modal_idx = [i for i in range(len(param_names)) if i not in modal_idx]
    non_modal_names = [param_names[i] for i in non_modal_idx]
    print(f"[36] R{args.target_regime} modal (free): {modal_names}", flush=True)
    print(f"[36] R{args.target_regime} non-modal (fixed at LOO median): {non_modal_names}",
          flush=True)

    # Per-curve LOO fit
    thetas = df[param_names].to_numpy(dtype=float)  # (N, 9) full-fit thetas
    fids = df[FID_COL].astype(int).tolist()
    dsets = df[DATASET_COL].tolist()
    n = len(df)

    rows: list[dict[str, object]] = []
    for i in range(n):
        if (i % 25) == 0:
            print(f"[36] {i}/{n} ...", flush=True)
        key = (dsets[i], fids[i])
        curve = curve_map.get(key)
        if curve is None:
            continue
        # Cap t_obs to <=90 days for internal curves too (consistency)
        mask = curve.t_obs <= args.t_grid_max_days
        c_local = CurveRecord(
            dataset=curve.dataset, fid=curve.fid,
            t_obs=curve.t_obs[mask], q_obs=curve.q_obs[mask],
        )
        if len(c_local.t_obs) < 3:
            continue

        # LOO R6 median for the 5 non-modal params
        loo_thetas = np.delete(thetas, i, axis=0)
        loo_median = np.median(loo_thetas, axis=0)  # (9,)

        # Build theta_fixed: non-modal at LOO median, modal slots will be
        # overwritten by the fit (initial guess = full-fit oracle value)
        theta_fixed = loo_thetas.mean(axis=0) * 0.0  # placeholder
        theta_fixed[:] = loo_median
        x0_free = thetas[i][modal_idx]

        # (c) Regime-conditioned k-param fit
        theta_r6_4, r2_r6_4 = _fit_subset(
            sim, c_local, theta_fixed=theta_fixed,
            free_idx=modal_idx, lows=lows, highs=highs,
            x0_free=x0_free,
        )

        # (d) Regime zero-shot: use LOO median for ALL 9, no fitting
        pred_r6_0 = sim.simulate_numpy(loo_median, c_local.t_obs)
        r2_r6_0 = _r2(c_local.q_obs, pred_r6_0)

        # (a) oracle 9 and (b) greedy 4 from 29 (recomputed locally for
        # the truncated t_obs to be apples-to-apples)
        pred_oracle = sim.simulate_numpy(thetas[i], c_local.t_obs)
        r2_oracle = _r2(c_local.q_obs, pred_oracle)

        rows.append({
            DATASET_COL: dsets[i],
            FID_COL: fids[i],
            "n_obs_used": len(c_local.t_obs),
            "t_max_used": float(c_local.t_obs.max()),
            "r2_oracle9": float(r2_oracle),
            "r2_greedy4_29": float(df.iloc[i]["r2_k4"]),
            "r2_r6cond4": float(r2_r6_4),
            "r2_r6zero": float(r2_r6_0),
            "delta_r6cond_minus_greedy": float(r2_r6_4 - float(df.iloc[i]["r2_k4"])),
            "delta_r6cond_minus_oracle": float(r2_r6_4 - r2_oracle),
            "active_set_k4_for_this_curve": str(df.iloc[i]["active_set_k4"]),
        })

    per_curve = pd.DataFrame(rows)
    per_curve.to_csv(args.out / "per_curve_fits.csv", index=False)

    # Summary stats
    def _q(s: pd.Series, q: float) -> float:
        return float(np.nanpercentile(s.to_numpy(), q))

    stats: dict[str, dict[str, float]] = {}
    for col in ["r2_oracle9", "r2_greedy4_29", "r2_r6cond4", "r2_r6zero"]:
        s = per_curve[col]
        stats[col] = {
            "median": float(np.nanmedian(s)),
            "mean": float(np.nanmean(s)),
            "p25": _q(s, 25),
            "p10": _q(s, 10),
            "frac_above_0.9": float((s >= 0.9).mean()),
            "frac_above_0.95": float((s >= 0.95).mean()),
        }

    tail = per_curve.sort_values("r2_r6cond4").head(15).copy()
    tail.to_csv(args.out / "tail_curves.csv", index=False)

    # Plot: R^2 distributions side by side
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    data = [per_curve[c].dropna().to_numpy() for c in
            ["r2_oracle9", "r2_greedy4_29", "r2_r6cond4", "r2_r6zero"]]
    regime_label = f"R{args.target_regime}"
    cond_label = f"{regime_label}-cond {args.k_modal}"
    zero_label = f"{regime_label} zero-shot"

    bp = ax.boxplot(data, tick_labels=["oracle 9", "greedy 4 (per-curve)",
                                       cond_label, zero_label],
                    showfliers=True, patch_artist=True)
    colors = ["lightgreen", "lightblue", "khaki", "lightcoral"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel(f"R^2 on R{args.target_regime} curves (n={len(per_curve)})")
    ax.set_title(f"R{args.target_regime} regime-conditional prototype: R^2 by fit type")
    ax.axhline(0.95, color="gray", lw=0.5, ls=":")
    ax.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax.set_ylim(min(0.0, float(per_curve[["r2_r6cond4", "r2_r6zero"]].min().min()) - 0.05), 1.02)
    fig.savefig(args.out / "r2_distribution.png", dpi=150)
    plt.close(fig)

    # Verdict
    med_oracle = stats["r2_oracle9"]["median"]
    med_greedy = stats["r2_greedy4_29"]["median"]
    med_r6cond = stats["r2_r6cond4"]["median"]
    med_r6zero = stats["r2_r6zero"]["median"]

    crit_match_greedy = (med_r6cond >= med_greedy - 0.005)
    crit_near_oracle = (abs(med_r6cond - med_oracle) <= 0.01)
    crit_fit_helps = (med_r6cond > med_r6zero + 0.02)

    lines = [
        f"=== 36 -- R{args.target_regime} regime-conditional prototype ===",
        "",
        f"R{args.target_regime} curves used : {len(per_curve)}",
        f"modal (free) params : {modal_names}",
        f"non-modal (LOO median fixed) : {non_modal_names}",
        "",
        f"  {'fit type':<22}  {'median':>7}  {'mean':>7}  {'p25':>7}  {'p10':>7}  "
        f"{'>=.9':>5}  {'>=.95':>5}",
    ]
    name_map = {
        "r2_oracle9":   "(a) oracle 9",
        "r2_greedy4_29": "(b) greedy 4 (per-curve)",
        "r2_r6cond4":   f"(c) {cond_label}",
        "r2_r6zero":    f"(d) {zero_label}",
    }
    for col in ["r2_oracle9", "r2_greedy4_29", "r2_r6cond4", "r2_r6zero"]:
        s = stats[col]
        lines.append(
            f"  {name_map[col]:<22}  "
            f"{s['median']:>7.4f}  {s['mean']:>7.4f}  "
            f"{s['p25']:>7.4f}  {s['p10']:>7.4f}  "
            f"{s['frac_above_0.9']:>5.2f}  {s['frac_above_0.95']:>5.2f}"
        )
    lines.append("")
    lines.append("--- success criteria ---")
    lines.append(f"  (c) >= (b) - 0.005 (regime modal set as good as per-curve greedy): "
                 f"{'PASS' if crit_match_greedy else 'FAIL'}  "
                 f"[(c)={med_r6cond:.4f} vs (b)-0.005={med_greedy-0.005:.4f}]")
    lines.append(f"  |(c) - (a)| <= 0.01 (R{args.target_regime} regime-cond loses negligible expressivity): "
                 f"{'PASS' if crit_near_oracle else 'FAIL'}  "
                 f"[delta={med_r6cond-med_oracle:+.4f}]")
    lines.append(f"  (c) > (d) + 0.02 (4 free modal params genuinely fit per-curve): "
                 f"{'PASS' if crit_fit_helps else 'FAIL'}  "
                 f"[(c)={med_r6cond:.4f} vs (d)+0.02={med_r6zero+0.02:.4f}]")
    lines.append("")

    n_pass = sum([crit_match_greedy, crit_near_oracle, crit_fit_helps])
    lines.append("--- verdict ---")
    if n_pass == 3:
        lines.append(f"  {regime_label} SINGLE-REGIME PROTOTYPE WORKS.")
        lines.append(
            f"  The {args.k_modal} modal params + {regime_label} LOO median "
            "for the rest reproduces 29's"
        )
        lines.append("  per-curve greedy 4-param performance, comes close to the 9-param")
        lines.append("  oracle, and the 4 free params are doing meaningful work over the")
        lines.append("  zero-shot baseline. This supports regime-conditional architecture")
        lines.append("  as a possible next modeling move.")
    elif n_pass == 2:
        lines.append(f"  {regime_label} PROTOTYPE PARTIAL PASS.")
        lines.append("  Check which criterion failed:")
        lines.append(f"    crit_match_greedy={crit_match_greedy}")
        lines.append(f"    crit_near_oracle={crit_near_oracle}")
        lines.append(f"    crit_fit_helps={crit_fit_helps}")
        lines.append("  If (c) is close to (b) but far from (a): 4-param parameterization")
        lines.append("    has a real ceiling; consider 5-param or per-regime k.")
        lines.append("  If (c) close to (a) but worse than (b): per-curve greedy is")
        lines.append(f"    picking different sets for different {regime_label} members -> regime is")
        lines.append("    heterogeneous, may need sub-regimes.")
        lines.append("  If (c) close to (d): the 4 free params are not enough; revisit")
        lines.append("    modal set choice.")
    else:
        lines.append(f"  {regime_label} PROTOTYPE LARGELY FAILS.")
        lines.append("  Single fixed modal set is not a usable parameterization even for")
        lines.append(f"  {regime_label}. The regime-conditional architecture as")
        lines.append("  described in the 32-35c brief needs to be redesigned. Possible")
        lines.append("  options: (i) regime-specific modal set has too many free choices;")
        lines.append("  (ii) the 5 fixed params actually do significant per-curve work;")
        lines.append("  (iii) sub-cluster R6 before parameterizing.")
    lines.append("")
    lines.append(f"--- worst-fit R{args.target_regime} curves (regime-cond bottom-15) ---")
    for _, r in tail.iterrows():
        lines.append(
            f"  {r[DATASET_COL]:>12}/{int(r[FID_COL]):>4}  "
            f"r2_{regime_label.lower()}cond{args.k_modal}={r['r2_r6cond4']:.4f}  "
            f"oracle9={r['r2_oracle9']:.4f}  "
            f"greedy4={r['r2_greedy4_29']:.4f}  "
            f"zero={r['r2_r6zero']:.4f}  "
            f"greedy_set={r['active_set_k4_for_this_curve']}"
        )

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
