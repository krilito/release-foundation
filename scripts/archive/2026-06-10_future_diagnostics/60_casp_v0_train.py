"""60 - CASP v0 training: cross321 random_5fold CV.

Phase 1 deliverable (per docs/plan_60_...). Goals:

  - confirm differentiable ODE training is stable (no NaN, no mode collapse on U),
  - report median R^2 of the posterior-mean curve vs RF's 0.957,
  - report 90% PI coverage (the new metric RF cannot produce),
  - report per-curve learned U so we can spot-check active subspace sanity.

Inputs:
  cross321 cleaned curves (218 matched)
  outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv (warm-start)

Outputs:
  outputs/60_casp_v0/per_curve.csv         per-curve R^2 + PI coverage
  outputs/60_casp_v0/fold_summary.csv      per-fold aggregates
  outputs/60_casp_v0/training_history.csv  per-epoch loss components
  outputs/60_casp_v0/summary.txt           paper-facing summary

Comparator: RF -> theta from 38d (cross321 random 5-fold median R^2 = 0.957).
v0 success bar: median R^2 >= 0.92, 90% PI coverage in [0.85, 0.95].
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
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import CASPEncoder, CASPLoss, SimulatorDecoder
from casp.calibration import predict_with_uncertainty, summarize_calibration, per_curve_r2, coverage
from casp.train import TrainConfig, train_casp
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
    "_loader_38d_for_60",
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


def build_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
        xlsx_path=CROSS_DOI_DATA,
        matched_fids_csv=MATCHED_FIDS_CSV,
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

    return X, full_Q, theta_oracle, fids, feature_cols


def run_fold(
    X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray,
    theta_tr: np.ndarray,
    n_form: int,
    epochs: int, seed: int,
) -> tuple[dict, dict[str, np.ndarray], dict[str, list[float]]]:
    # Loose-tol simulator for training (rtol=1e-3 keeps gradients clean
    # and avoids dopri5 step-size underflow on boundary theta samples).
    # Eval-time uses default tight tolerance for accurate Q_hat.
    sim_train = PLGABiphasic(rtol=1e-3, atol=1e-4)
    sim_eval = PLGABiphasic()
    prior = sim_eval.prior().base_dist
    prior_low = prior.low.float(); prior_high = prior.high.float()

    # Standardize formulation features using train-fold statistics.
    mu_f = X_tr[:, :n_form].mean(axis=0, keepdims=True)
    sd_f = X_tr[:, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
    X_tr_s = X_tr.copy(); X_te_s = X_te.copy()
    X_tr_s[:, :n_form] = (X_tr_s[:, :n_form] - mu_f) / sd_f
    X_te_s[:, :n_form] = (X_te_s[:, :n_form] - mu_f) / sd_f

    encoder = CASPEncoder(
        n_features=X_tr_s.shape[1],
        n_params=sim_eval.n_params,
        prior_low=prior_low, prior_high=prior_high,
        rank=4, hidden=64, depth=3, dropout=0.3,
    )
    decoder_train = SimulatorDecoder(sim_train)
    decoder_eval = SimulatorDecoder(sim_eval)
    loss_fn = CASPLoss(
        prior_low=prior_low, prior_high=prior_high,
        sigma_obs=0.08,
        recon_w=1.0, kl_w=1.0, orth_w=1e-3, warm_w_init=2.0,
    )
    cfg = TrainConfig(
        epochs=epochs, batch_size=32, lr=1e-3,
        weight_decay=1e-3, grad_clip=1.0, n_samples=4,
        # Warm-start ON for the full run (no decay). v0b reframe: CASP
        # supplies identifiability-aware UQ on top of a point predictor
        # (oracle theta from script 29). The model learns U/sigma; it does
        # NOT relearn mu from scratch. This converts the task from full
        # amortized VI (data-hungry) into local UQ around a known anchor
        # (tractable on 200 curves).
        warm_decay_epochs=0,
        oob_w=1e-2,
        learn_sigma_obs=False, log_sigma_obs_init=float(np.log(0.08)),
        log_every=max(1, epochs // 10), seed=seed,
    )

    hist = train_casp(
        encoder=encoder, decoder=decoder_train, loss_fn=loss_fn,
        X_train=torch.tensor(X_tr_s, dtype=torch.float32),
        y_train=torch.tensor(y_tr, dtype=torch.float32),
        t_grid=torch.tensor(ALL_TIMES, dtype=torch.float32),
        oracle_mu_train=torch.tensor(theta_tr, dtype=torch.float32),
        prior_low=prior_low, prior_high=prior_high,
        cfg=cfg, device="cpu",
    )

    # Evaluate with many samples for stable PI quantiles; use tight-tol sim.
    pred = predict_with_uncertainty(
        encoder=encoder, decoder=decoder_eval,
        X=torch.tensor(X_te_s, dtype=torch.float32),
        t_grid=torch.tensor(ALL_TIMES, dtype=torch.float32),
        n_samples=128, device="cpu",
    )
    summary = summarize_calibration(y_te, pred)

    return summary, pred, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/60_casp_v0"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-fold", action="store_true",
                    help="only run fold 0, for a quick sanity check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[60] config: epochs={args.epochs} n_folds={args.n_folds} "
          f"single_fold={args.single_fold}", flush=True)

    X, y, theta_oracle, fids, feature_cols = build_arrays()
    n = len(X)
    n_form = len(feature_cols)
    print(f"[60] n_curves={n}, n_features={X.shape[1]} "
          f"({n_form} form + {X.shape[1]-n_form} early Q)", flush=True)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(n)))
    if args.single_fold:
        folds = folds[:1]

    per_curve_rows: list[dict] = []
    fold_rows: list[dict] = []
    histories: list[dict] = []

    for fi, (tr, te) in enumerate(folds):
        print(f"\n[60] === fold {fi+1}/{len(folds)} "
              f"(n_train={len(tr)}, n_test={len(te)}) ===", flush=True)
        summary, pred, hist = run_fold(
            X_tr=X[tr], X_te=X[te],
            y_tr=y[tr], y_te=y[te],
            theta_tr=theta_oracle[tr],
            n_form=n_form,
            epochs=args.epochs, seed=args.seed + fi,
        )

        fold_row = {"fold": fi, "n_train": int(len(tr)), "n_test": int(len(te)), **summary}
        fold_rows.append(fold_row)
        print(f"[60] fold {fi} summary: "
              f"R^2 median={summary['r2_mean_median']:+.4f}, "
              f"mean={summary['r2_mean_mean']:+.4f}, "
              f"frac>=0={summary['frac_R2_gte_0']:.3f}, "
              f"cov90={summary['cov90_mean']:.3f}, "
              f"PI90 width={summary['pi_width_90_median']:.3f}",
              flush=True)

        r2 = per_curve_r2(y[te], pred["mean"])
        cov90 = coverage(y[te], pred["lo90"], pred["hi90"])
        cov50 = coverage(y[te], pred["lo50"], pred["hi50"])
        for j, fid in enumerate(fids[te]):
            per_curve_rows.append({
                "fold": fi,
                "fid": int(fid),
                "r2": float(r2[j]),
                "cov90": float(cov90[j]),
                "cov50": float(cov50[j]),
                "pi_width_90": float((pred["hi90"][j] - pred["lo90"][j]).mean()),
            })

        for ep in range(args.epochs):
            row = {"fold": fi, "epoch": ep}
            for k, v in hist.items():
                row[k] = float(v[ep])
            histories.append(row)

    pd.DataFrame(per_curve_rows).to_csv(args.out / "per_curve.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out / "fold_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(args.out / "training_history.csv", index=False)

    # Aggregate summary
    df_pc = pd.DataFrame(per_curve_rows)
    r2_overall_med = float(df_pc["r2"].median())
    r2_overall_mean = float(df_pc["r2"].mean())
    cov90_overall = float(df_pc["cov90"].mean())
    cov50_overall = float(df_pc["cov50"].mean())
    pi_w_med = float(df_pc["pi_width_90"].median())
    frac_pos = float((df_pc["r2"] >= 0).mean())
    frac_90 = float((df_pc["r2"] >= 0.9).mean())

    with open(args.out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== CASP v0 — cross321 random 5-fold CV ===\n\n")
        f.write(f"n_curves        : {n}\n")
        f.write(f"n_folds         : {len(folds)}\n")
        f.write(f"epochs/fold     : {args.epochs}\n")
        f.write(f"rank (active r) : 4\n\n")
        f.write("Per-curve aggregates:\n")
        f.write(f"  R^2 median           : {r2_overall_med:+.4f}\n")
        f.write(f"  R^2 mean             : {r2_overall_mean:+.4f}\n")
        f.write(f"  frac R^2 >= 0        : {frac_pos:.3f}\n")
        f.write(f"  frac R^2 >= 0.9      : {frac_90:.3f}\n")
        f.write(f"  90% PI coverage      : {cov90_overall:.3f}  (target [0.85, 0.95])\n")
        f.write(f"  50% PI coverage      : {cov50_overall:.3f}  (target [0.40, 0.60])\n")
        f.write(f"  90% PI width (median): {pi_w_med:.3f}\n\n")
        f.write("Phase 1 success bar (per plan_60):\n")
        bar_r2 = "PASS" if r2_overall_med >= 0.92 else "FAIL"
        bar_cov = "PASS" if 0.85 <= cov90_overall <= 0.95 else "FAIL"
        f.write(f"  median R^2 >= 0.92     : {bar_r2}  ({r2_overall_med:+.4f})\n")
        f.write(f"  cov90 in [0.85, 0.95]  : {bar_cov}  ({cov90_overall:.3f})\n\n")
        f.write("Reference: RF -> theta (38d) cross321 random 5-fold median R^2 = 0.957.\n")

    print(f"\n[60] wrote {args.out / 'summary.txt'}", flush=True)
    with open(args.out / "summary.txt", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
