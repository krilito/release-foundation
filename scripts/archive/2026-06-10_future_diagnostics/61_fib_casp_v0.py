"""61 - FIB-CASP v0: Fisher-Information posterior on top of RF -> theta.

Phase 1 sanity (per docs/plan_61_fib_casp.md). Scope:
  - cross321 random_5fold, fold 0 only
  - Use sklearn RandomForestRegressor as the theta point predictor
    (same recipe as script 38d B2 RF -> theta route)
  - For each test curve, compute Fisher info at theta_RF, build (mu, U, sigma),
    sample S=128 draws, simulate, compute PI band.

Outputs:
  outputs/61_fib_casp_v0/per_curve.csv
  outputs/61_fib_casp_v0/summary.txt

Success bar (Phase 1):
  median R^2 in range of RF's 0.957 (we sample around RF's point estimate),
  90% PI coverage in [0.80, 0.95],
  PI width median in [0.05, 0.30].
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import SimulatorDecoder, fib_posterior, sample_theta_fib
from casp.calibration import per_curve_r2, coverage
from simulator import PLGABiphasic


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader = _load_script(
    Path(__file__).resolve().parent / "38d_baselines_groupkfold.py",
    "_loader_38d_for_61",
)
FORMULATION_COL_MAP = _loader.FORMULATION_COL_MAP
FID_COL = _loader.FID_COL
load_external_records = _loader._load_external_records
interp_at = _loader._interp_at

CROSS_DOI_DATA = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)
MATCHED_FIDS_CSV = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
FULL_FIT_BANK = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")
EARLY_TIMES = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)
LATE_GRID = np.array([10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0], dtype=float)
ALL_TIMES = np.concatenate([EARLY_TIMES, LATE_GRID])


def build_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[object]
]:
    sim = PLGABiphasic()
    param_names = list(sim.param_names)

    df_meta = pd.read_excel(CROSS_DOI_DATA, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())
    df_bank = pd.read_csv(FULL_FIT_BANK)
    df_bank_cross = df_bank[df_bank["dataset"] == "cross321"].copy()
    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )
    curves = load_external_records(
        xlsx_path=CROSS_DOI_DATA, matched_fids_csv=MATCHED_FIDS_CSV,
        t_grid_max_days=90.0,
    )
    curve_map = {c.fid: c for c in curves}
    early_Q_list, full_Q_list, keep_rows = [], [], []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        early_Q_list.append(interp_at(c.t_obs, c.q_obs, EARLY_TIMES))
        full_Q_list.append(interp_at(c.t_obs, c.q_obs, ALL_TIMES))
        keep_rows.append(i)
    df = df.loc[keep_rows].reset_index(drop=True)
    early_Q = np.stack(early_Q_list, axis=0).astype(np.float32)
    full_Q = np.stack(full_Q_list, axis=0).astype(np.float32)
    X_form = df[feature_cols].to_numpy(dtype=np.float32)
    X = np.concatenate([X_form, early_Q], axis=1)
    theta_oracle = df[param_names].to_numpy(dtype=np.float32)
    fids = df[FID_COL].to_numpy()
    # Also return the raw CurveRecords (irregular t_obs, q_obs) in same
    # order as the arrays so per-curve native-time R^2 can be computed
    # (matches 38d's evaluation protocol — fixed-grid interpolation
    # adds flat-extrapolation artifacts that artificially deflate R^2).
    curves_aligned = [curve_map[int(df[FID_COL].iloc[i])] for i in range(len(df))]
    return X, full_Q, theta_oracle, fids, feature_cols, curves_aligned


def run_fold(
    X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray,
    theta_tr_oracle: np.ndarray,
    curves_te: list[object],
    n_form: int,
    n_samples: int = 128,
    fib_rank: int = 4,
    fib_alpha: float = 1e-2,
    fib_sigma0_frac: float = 0.05,
    sigma_obs: float = 0.05,
) -> tuple[dict, np.ndarray, list[dict]]:
    sim_eval = PLGABiphasic()
    sim_train = PLGABiphasic(rtol=1e-3, atol=1e-4)   # for Jacobian, looser tol
    prior = sim_eval.prior().base_dist
    prior_low = prior.low.float(); prior_high = prior.high.float()

    # Standardize formulation features (same convention as 38d).
    mu_f = X_tr[:, :n_form].mean(axis=0, keepdims=True)
    sd_f = X_tr[:, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
    X_tr_s = X_tr.copy(); X_te_s = X_te.copy()
    X_tr_s[:, :n_form] = (X_tr_s[:, :n_form] - mu_f) / sd_f
    X_te_s[:, :n_form] = (X_te_s[:, :n_form] - mu_f) / sd_f

    # RF -> theta (matches 38d B2 baseline recipe exactly).
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=0,
    )
    rf.fit(X_tr_s, theta_tr_oracle)
    theta_pred_raw = rf.predict(X_te_s).astype(np.float32)
    # Clamp to prior box (matches 38d line 402). Without this, RF can
    # produce theta slightly outside the box; simulator integrates whatever
    # it gets and returns nonsense curves.
    eps_box = 1e-4
    lows_np = prior_low.numpy(); highs_np = prior_high.numpy()
    theta_pred = np.clip(theta_pred_raw, lows_np + eps_box, highs_np - eps_box)
    n_clipped = int((theta_pred != theta_pred_raw).any(axis=1).sum())
    print(f"[61] RF predictions clipped to prior box: {n_clipped} / {len(theta_pred)}",
          flush=True)

    # Per-curve FIB construction and PI evaluation.
    decoder = SimulatorDecoder(sim_eval)
    t_t = torch.tensor(ALL_TIMES, dtype=torch.float32)

    per_curve_rows: list[dict] = []
    Q_samples_all = []                    # (j, t_eval_j, Q_samples_at_t_eval)
    Q_point_all = []                      # (j, t_eval_j, Q_point_at_t_eval)
    diagnostics = {"singular_failures": 0, "ode_failures": 0}

    # Per-curve simulation at c.t_obs (matches 38d evaluation exactly).
    # No fixed-grid interpolation -> no precision loss.
    for j in range(len(X_te_s)):
        theta_hat = torch.tensor(theta_pred[j], dtype=torch.float32)
        c = curves_te[j]
        # Restrict t_obs to (0, 90] — simulator integrates from 0; t=0
        # itself is by definition Q=0 so we don't compare there.
        t_obs_np = c.t_obs.astype(np.float64)
        mask = (t_obs_np > 0.0) & (t_obs_np <= 90.0)
        if mask.sum() < 2:
            continue
        t_eval_np = t_obs_np[mask]
        t_eval_t = torch.tensor(t_eval_np, dtype=torch.float32)

        # Point prediction at c.t_obs (deterministic theta_RF simulation).
        with torch.no_grad():
            try:
                Q_point = decoder(theta_hat.unsqueeze(0), t_eval_t).cpu().numpy()[0]
                Q_point_all.append((j, t_eval_np, Q_point))
            except Exception:
                pass

        # Fisher posterior at c.t_obs (Jacobian d Q(c.t_obs) / d theta).
        try:
            q = fib_posterior(
                simulator=sim_train,
                theta_hat=theta_hat, t=t_eval_t,
                sigma_obs=sigma_obs, rank=fib_rank,
                alpha=fib_alpha, sigma_0_frac=fib_sigma0_frac,
                prior_low=prior_low, prior_high=prior_high,
            )
        except Exception:
            diagnostics["singular_failures"] += 1
            continue

        try:
            theta_samples = sample_theta_fib(
                q, n_samples=n_samples,
                prior_low=prior_low, prior_high=prior_high,
            )                                                       # (S, P)
            with torch.no_grad():
                Q = decoder(theta_samples, t_eval_t).cpu().numpy()  # (S, T_j)
        except Exception:
            diagnostics["ode_failures"] += 1
            continue

        Q_samples_all.append((j, t_eval_np, Q))

    # Aggregate per-curve. Eval is now ON THE NATIVE c.t_obs GRID with no
    # interpolation: simulator was called per-curve at c.t_obs[j] directly.
    # This matches 38d's evaluation exactly.
    #
    # Two design decisions kept from prior version:
    # (1) Point prediction is the DETERMINISTIC simulation at theta_RF
    #     (Q_point), NOT the sample mean (simulator non-linearity biases
    #     the sample mean).
    # (2) PI band is recentered onto Q_point so the percentile band's
    #     center matches the point prediction.
    cov90s, cov50s, pi_widths, r2s = [], [], [], []
    point_dict = {j: (t, Q_p) for (j, t, Q_p) in Q_point_all}
    for j, t_eval, Q in Q_samples_all:
        if j not in point_dict:
            continue
        t_point, Q_point = point_dict[j]
        # Sanity: t_eval should equal t_point (both built from same c.t_obs mask).
        if not np.array_equal(t_eval, t_point):
            # Should never happen — skip defensively.
            continue
        c = curves_te[j]
        # Re-build y_eval from c.t_obs / c.q_obs at the same mask.
        t_obs_np = c.t_obs.astype(np.float64); q_obs_np = c.q_obs.astype(np.float64)
        mask = (t_obs_np > 0.0) & (t_obs_np <= 90.0)
        y_eval = q_obs_np[mask]
        if y_eval.shape[0] != t_eval.shape[0]:
            continue

        sample_mean = Q.mean(axis=0)                                   # (T_j,)
        recenter = Q_point - sample_mean                               # (T_j,)
        lo90 = np.percentile(Q, 5.0, axis=0)  + recenter
        hi90 = np.percentile(Q, 95.0, axis=0) + recenter
        lo50 = np.percentile(Q, 25.0, axis=0) + recenter
        hi50 = np.percentile(Q, 75.0, axis=0) + recenter
        cov90 = float(((y_eval >= lo90) & (y_eval <= hi90)).mean())
        cov50 = float(((y_eval >= lo50) & (y_eval <= hi50)).mean())
        pi_w = float((hi90 - lo90).mean())
        r2 = float(per_curve_r2(y_eval[None, :], Q_point[None, :])[0])
        cov90s.append(cov90); cov50s.append(cov50)
        pi_widths.append(pi_w); r2s.append(r2)
        per_curve_rows.append({
            "test_idx": j,
            "r2_point": r2,
            "cov90": cov90, "cov50": cov50,
            "pi_width_90": pi_w,
            "n_t_eval": int(len(t_eval)),
        })

    r2_arr = np.array(r2s); cov90_arr = np.array(cov90s)

    # Diagnostic R^2 at theta_RF directly (now also on native c.t_obs).
    r2_point = []
    for j, t_p, Q_p in Q_point_all:
        c = curves_te[j]
        t_obs_np = c.t_obs.astype(np.float64); q_obs_np = c.q_obs.astype(np.float64)
        mask = (t_obs_np > 0.0) & (t_obs_np <= 90.0)
        y_p = q_obs_np[mask]
        if y_p.shape[0] != Q_p.shape[0]:
            continue
        r2_point.append(float(per_curve_r2(y_p[None, :], Q_p[None, :])[0]))
    r2_point_arr = np.array(r2_point)

    summary = {
        "n_eval": len(r2_arr),
        "n_skipped_singular": diagnostics["singular_failures"],
        "n_skipped_ode": diagnostics["ode_failures"],
        # r2_point uses theta_RF deterministic simulation = the official
        # point prediction. r2_arr is the same now (we changed to point-
        # centered evaluation). Keep both for compatibility.
        "r2_point_median": float(np.nanmedian(r2_point_arr)),
        "r2_point_mean": float(np.nanmean(r2_point_arr)),
        "r2_mean_median": float(np.nanmedian(r2_arr)),
        "r2_mean_mean": float(np.nanmean(r2_arr)),
        "frac_R2_gte_0": float((r2_arr >= 0).mean()),
        "frac_R2_gte_0p9": float((r2_arr >= 0.9).mean()),
        "cov90_mean": float(cov90_arr.mean()),
        "cov90_median": float(np.median(cov90_arr)),
        "cov50_mean": float(np.mean(cov50s)),
        "pi_width_90_median": float(np.median(pi_widths)),
    }
    return summary, np.array([]), per_curve_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/61_fib_casp_v0"))
    ap.add_argument("--single-fold", action="store_true")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--sigma0-frac", type=float, default=0.05)
    ap.add_argument("--sigma-obs", type=float, default=0.05)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    X, y, theta_oracle, fids, feature_cols, curves_aligned = build_arrays()
    n = len(X); n_form = len(feature_cols)
    print(f"[61] n_curves={n}, n_features={X.shape[1]} ({n_form} form + 4 early Q)",
          flush=True)
    print(f"[61] FIB config: rank={args.rank} alpha={args.alpha} "
          f"sigma0_frac={args.sigma0_frac} sigma_obs={args.sigma_obs}", flush=True)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(n)))
    if args.single_fold:
        folds = folds[:1]

    all_per_curve: list[dict] = []
    fold_rows: list[dict] = []
    for fi, (tr, te) in enumerate(folds):
        print(f"\n[61] === fold {fi+1}/{len(folds)} "
              f"n_train={len(tr)} n_test={len(te)} ===", flush=True)
        summary, _, per_curve = run_fold(
            X_tr=X[tr], X_te=X[te], y_tr=y[tr], y_te=y[te],
            theta_tr_oracle=theta_oracle[tr],
            curves_te=[curves_aligned[k] for k in te],
            n_form=n_form,
            n_samples=args.n_samples, fib_rank=args.rank,
            fib_alpha=args.alpha, fib_sigma0_frac=args.sigma0_frac,
            sigma_obs=args.sigma_obs,
        )
        fold_rows.append({"fold": fi, **summary})
        print(f"[61] fold {fi}: "
              f"R^2 @theta_RF median={summary['r2_point_median']:+.4f}  "
              f"R^2 sample-mean median={summary['r2_mean_median']:+.4f}  "
              f"cov90={summary['cov90_mean']:.3f}  "
              f"PI90 width={summary['pi_width_90_median']:.3f}  "
              f"(singular_fail={summary['n_skipped_singular']}, "
              f"ode_fail={summary['n_skipped_ode']})", flush=True)
        for row in per_curve:
            row = {**row, "fold": fi, "fid": int(fids[te[row["test_idx"]]])}
            all_per_curve.append(row)

    pd.DataFrame(all_per_curve).to_csv(args.out / "per_curve.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out / "fold_summary.csv", index=False)

    df_pc = pd.DataFrame(all_per_curve)
    # r2_point column = R^2 at theta_RF deterministic simulation
    # (the official point prediction in the recentered-PI evaluation).
    r2_col = "r2_point" if "r2_point" in df_pc.columns else "r2_mean"
    r2_med = float(df_pc[r2_col].median())
    cov90_overall = float(df_pc["cov90"].mean())
    pi_w_med = float(df_pc["pi_width_90"].median())
    frac_pos = float((df_pc[r2_col] >= 0).mean())

    with open(args.out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== FIB-CASP v0 — cross321 random 5-fold CV ===\n\n")
        f.write(f"n_curves        : {n}\n")
        f.write(f"n_folds_run     : {len(folds)}\n")
        f.write(f"FIB rank        : {args.rank}\n")
        f.write(f"FIB alpha (ridge): {args.alpha}\n")
        f.write(f"FIB sigma_0 frac: {args.sigma0_frac}\n")
        f.write(f"sigma_obs       : {args.sigma_obs}\n")
        f.write(f"n_samples       : {args.n_samples}\n\n")
        f.write("Per-curve aggregates:\n")
        f.write(f"  R^2 median           : {r2_med:+.4f}\n")
        f.write(f"  R^2 mean             : {float(df_pc[r2_col].mean()):+.4f}\n")
        f.write(f"  frac R^2 >= 0        : {frac_pos:.3f}\n")
        f.write(f"  90% PI coverage      : {cov90_overall:.3f}  (target [0.80, 0.95])\n")
        f.write(f"  90% PI width (median): {pi_w_med:.3f}\n\n")
        f.write("Phase 1 success bar (per plan_61):\n")
        bar_r2 = "PASS" if r2_med >= 0.90 else "FAIL"
        bar_cov = "PASS" if 0.80 <= cov90_overall <= 0.95 else "FAIL"
        bar_width = "PASS" if 0.05 <= pi_w_med <= 0.30 else "FAIL"
        f.write(f"  median R^2 >= 0.90        : {bar_r2}  ({r2_med:+.4f})\n")
        f.write(f"  cov90 in [0.80, 0.95]     : {bar_cov}  ({cov90_overall:.3f})\n")
        f.write(f"  PI90 width in [0.05, 0.30]: {bar_width}  ({pi_w_med:.3f})\n\n")
        f.write("Reference: RF -> theta (38d) median R^2 = 0.957 (point estimate only).\n")

    print(f"\n[61] wrote {args.out / 'summary.txt'}", flush=True)
    with open(args.out / "summary.txt", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
