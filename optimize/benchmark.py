"""Comprehensive benchmark: FIB distillation vs baselines.

Runs three methods head-to-head on cross321 PLGA with group-by-drug CV:

    1. Direct Q regression (ExtraTrees, script 38d recipe)
    2. NPE with FCEmbedding (current baseline, FIB teacher)
    3. NPE with TemporalConvEmbedding (proposed, FIB teacher)

Also compares the FIB teacher quality itself: what R² do you get
if you just use the oracle theta (upper bound) vs the RF-predicted
theta (current production) vs the FIB posterior mean?

Usage:
    python optimize/benchmark.py --device cuda
    python optimize/benchmark.py --device cpu --quick   # 1 fold, 50 epochs
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from casp import fib_posterior, sample_theta_fib
from casp.calibration import per_curve_r2
from optimize.calibration import effective_sample_size, coverage_at_level
from optimize.distill_fib import (
    load_cross321,
    generate_fib_teacher,
    train_amortized_posterior,
    evaluate_posterior,
)
from simulator import PLGABiphasic


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def direct_q_baseline(
    X_train: np.ndarray,
    Q_train: np.ndarray,
    X_test: np.ndarray,
    Q_test: np.ndarray,
    t_grid: np.ndarray,
) -> dict:
    """Direct Q regression: ExtraTrees on (X, t) -> Q(t).

    This is the strongest non-SBI baseline (script 38d showed
    ExtraTrees > RandomForest > Ridge for this task).
    """
    # Build (X_i, t_j) -> Q_ij pairs
    T = len(t_grid)
    X_rep = np.repeat(X_train, T, axis=0)
    t_rep = np.tile(t_grid, len(X_train))
    X_aug = np.column_stack([X_rep, t_rep])
    y_flat = Q_train.flatten()

    et = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1)
    et.fit(X_aug, y_flat)

    # Predict
    X_test_rep = np.repeat(X_test, T, axis=0)
    t_test_rep = np.tile(t_grid, len(X_test))
    X_test_aug = np.column_stack([X_test_rep, t_test_rep])
    Q_pred_flat = et.predict(X_test_aug)
    Q_pred = Q_pred_flat.reshape(-1, T)

    r2s = [per_curve_r2(Q_test[i], Q_pred[i]) for i in range(len(Q_test))]
    return {
        "r2_median": float(np.median(r2s)),
        "r2_mean": float(np.mean(r2s)),
        "frac_r2_gte_0": float(np.mean(np.array(r2s) >= 0)),
        "per_curve_r2": r2s,
    }


def rf_theta_baseline(
    X_train: np.ndarray,
    theta_train: np.ndarray,
    X_test: np.ndarray,
    theta_test: np.ndarray,
    Q_test: np.ndarray,
    simulator: PLGABiphasic,
    t_grid: torch.Tensor,
) -> dict:
    """RF -> theta -> simulate: the current production pipeline."""
    rf = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1)
    rf.fit(X_train, theta_train)
    theta_pred = rf.predict(X_test)

    Q_preds = []
    with torch.no_grad():
        for i in range(len(theta_pred)):
            th = torch.tensor(theta_pred[i:i+1], dtype=torch.float32)
            q = simulator.simulate(th, t_grid).squeeze(0).numpy()
            Q_preds.append(q)
    Q_preds = np.array(Q_preds)

    r2s = [per_curve_r2(Q_test[i], Q_preds[i]) for i in range(len(Q_test))]
    return {
        "r2_median": float(np.median(r2s)),
        "r2_mean": float(np.mean(r2s)),
        "frac_r2_gte_0": float(np.mean(np.array(r2s) >= 0)),
        "per_curve_r2": r2s,
    }


def main():
    parser = argparse.ArgumentParser(description="FIB Distillation Benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="1 fold, 50 epochs")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="optimize/outputs/benchmark")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_folds = 1 if args.quick else 5
    epochs = min(args.epochs, 50) if args.quick else args.epochs

    print(f"{'='*70}")
    print(f"FIB Distillation Benchmark — {n_folds} folds, {epochs} epochs")
    print(f"{'='*70}")

    # Load data
    print("\n[1] Loading cross321...")
    data = load_cross321()
    X, fids, curves = data["X"], data["fids"], data["curves"]
    theta_oracle, drug_groups = data["theta_oracle"], data["drug_groups"]

    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 90.0, 64)

    # FIB teacher
    print(f"\n[2] Generating FIB teacher (K={args.n_samples})...")
    theta_teacher, Q_curves, valid_mask = generate_fib_teacher(
        X, curves, fids, theta_oracle, sim,
        n_samples=args.n_samples, seed=args.seed,
    )
    n_valid = valid_mask.sum()
    print(f"  {n_valid}/{len(fids)} valid")

    X_v, Q_v = X[valid_mask], Q_curves[valid_mask]
    theta_v = theta_teacher[valid_mask]
    theta_oracle_v = theta_oracle[valid_mask]
    groups_v = drug_groups[valid_mask]

    # Cross-validation
    print(f"\n[3] Running {n_folds}-fold group-by-drug CV...")
    gkf = GroupKFold(n_splits=n_folds)

    results = {
        "direct_Q": [],
        "rf_theta": [],
        "npe_fc": [],
        "npe_temporal": [],
    }

    for fold_idx, (tr, te) in enumerate(gkf.split(X_v, groups=groups_v)):
        print(f"\n--- Fold {fold_idx} (train={len(tr)}, test={len(te)}) ---")

        X_tr, X_te = X_v[tr], X_v[te]
        Q_tr, Q_te = Q_v[tr], Q_v[te]
        th_tr, th_te = theta_v[tr], theta_oracle_v[te]

        # A. Direct Q regression
        r = direct_q_baseline(X_tr, Q_tr, X_te, Q_te, t_grid.numpy())
        results["direct_Q"].append(r)
        print(f"  direct_Q:     R²={r['r2_median']:.4f}")

        # B. RF -> theta
        r = rf_theta_baseline(X_tr, theta_oracle_v[tr], X_te, th_te, Q_te, sim, t_grid)
        results["rf_theta"].append(r)
        print(f"  rf_theta:     R²={r['r2_median']:.4f}")

        # C. NPE with FCEmbedding (baseline)
        t0 = time.time()
        res_fc = train_amortized_posterior(
            Q_tr, th_tr, sim,
            use_temporal_embed=False,
            max_epochs=epochs, device=args.device,
            seed=args.seed + fold_idx,
        )
        ev_fc = evaluate_posterior(
            res_fc["posterior"], Q_te, th_te, sim,
            device=args.device,
        )
        ev_fc["wall_seconds"] = time.time() - t0
        results["npe_fc"].append(ev_fc)
        print(f"  npe_fc:       R²={ev_fc['r2_median']:.4f}, "
              f"cov90={ev_fc['coverage_90_mean']:.2f}, "
              f"ESS={ev_fc['ess_median']:.1f} ({ev_fc['wall_seconds']:.0f}s)")

        # D. NPE with TemporalConvEmbedding
        t0 = time.time()
        res_tc = train_amortized_posterior(
            Q_tr, th_tr, sim,
            use_temporal_embed=True,
            max_epochs=epochs, device=args.device,
            seed=args.seed + fold_idx,
        )
        ev_tc = evaluate_posterior(
            res_tc["posterior"], Q_te, th_te, sim,
            device=args.device,
        )
        ev_tc["wall_seconds"] = time.time() - t0
        results["npe_temporal"].append(ev_tc)
        print(f"  npe_temporal: R²={ev_tc['r2_median']:.4f}, "
              f"cov90={ev_tc['coverage_90_mean']:.2f}, "
              f"ESS={ev_tc['ess_median']:.1f} ({ev_tc['wall_seconds']:.0f}s)")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*70}")

    summary_rows = []
    for method, folds in results.items():
        if method in ("direct_Q", "rf_theta"):
            r2s = [f["r2_median"] for f in folds]
            row = {
                "method": method,
                "r2_median": float(np.median(r2s)),
                "r2_mean": float(np.mean(r2s)),
                "frac_r2_gte_0": float(np.mean([f["frac_r2_gte_0"] for f in folds])),
            }
        else:
            r2s = [f["r2_median"] for f in folds if not np.isnan(f["r2_median"])]
            covs = [f["coverage_90_mean"] for f in folds]
            esses = [f["ess_median"] for f in folds if f["ess_median"] > 0]
            walls = [f["wall_seconds"] for f in folds]
            row = {
                "method": method,
                "r2_median": float(np.median(r2s)) if r2s else float("nan"),
                "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
                "frac_r2_gte_0": float(np.mean([f["frac_r2_gte_0"] for f in folds])),
                "coverage_90": float(np.mean(covs)),
                "ess_median": float(np.median(esses)) if esses else float("nan"),
                "wall_s": float(np.sum(walls)),
            }
        summary_rows.append(row)
        print(f"\n{method}:")
        for k, v in row.items():
            if k != "method":
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
    with open(out / "summary.txt", "w") as f:
        f.write("=== FIB Distillation Benchmark ===\n\n")
        f.write(f"n_folds : {n_folds}\n")
        f.write(f"device  : {args.device}\n")
        f.write(f"K       : {args.n_samples}\n")
        f.write(f"epochs  : {epochs}\n")
        f.write(f"n_valid : {n_valid}/{len(fids)}\n\n")
        for row in summary_rows:
            f.write(f"--- {row['method']} ---\n")
            for k, v in row.items():
                if k != "method":
                    f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
            f.write("\n")

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
