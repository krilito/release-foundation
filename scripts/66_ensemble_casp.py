"""66 - Ensemble-CASP zero-shot: formulation-only RF tree-ensemble posterior.

For each test curve:
  1. Train RandomForestRegressor on (formulation, theta_oracle) — formulation ONLY,
     no early Q (true zero-shot setup).
  2. At inference, get theta predictions from all 400 individual trees.
  3. Fit a low-rank Gaussian over those tree-level predictions = empirical
     formulation-conditional posterior.
  4. Sample, simulate at native c.t_obs, recenter PI band onto the forest-mean
     simulation. Same downstream pipeline as FIB-CASP / pull-back.

Why this should work where FIB fails in zero-shot:
  FIB Fisher info gives identifiability of theta given the CURVE, but the curve
  isn't observed at inference in zero-shot. Tree ensemble variance gives
  uncertainty given the FORMULATION, which IS the zero-shot uncertainty
  source. Where formulation poorly constrains theta, trees disagree, PI widens.

Outputs:
  outputs/66_ensemble_casp/{dataset}/{scheme}/per_curve.csv
  outputs/66_ensemble_casp/aggregate_table.csv
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
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import SimulatorDecoder, ensemble_posterior, per_tree_predictions
from casp.calibration import per_curve_r2
from simulator import PLGABiphasic


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m62 = _load_script(Path(__file__).resolve().parent / "62_fib_casp_benchmark.py",
                   "_m62_for_66")


def run_dataset_scheme_ensemble(
    bundle, scheme: str, out_dir: Path,
    n_estimators: int = 400, rank: int = 4,
    n_samples: int = 64, n_folds: int = 5, seed: int = 0,
) -> dict:
    """Like m62.run_dataset_scheme but uses tree-ensemble posterior instead of FIB."""
    n = len(bundle.X)
    splits = m62.make_splits(scheme, n, bundle.groups_drug, bundle.groups_polymer,
                              n_folds=n_folds, seed=seed)
    sim = bundle.simulator
    sim_eval = sim
    prior = sim.prior().base_dist
    prior_low = prior.low.float(); prior_high = prior.high.float()
    lo_np = prior_low.numpy(); hi_np = prior_high.numpy()
    decoder = SimulatorDecoder(sim_eval)

    # Note: bundle.X may include early-Q features. For zero-shot, strip them.
    # The caller passes a formulation-only bundle (see main()).
    n_form = bundle.n_form

    all_rows: list[dict] = []
    for fi, (tr, te) in enumerate(splits):
        t0 = time.time()
        # standardize features by train statistics
        mu_f = bundle.X[tr, :n_form].mean(axis=0, keepdims=True)
        sd_f = bundle.X[tr, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
        X_tr = (bundle.X[tr, :n_form] - mu_f) / sd_f
        X_te = (bundle.X[te, :n_form] - mu_f) / sd_f

        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=None, n_jobs=-1,
            random_state=seed + fi,
        )
        rf.fit(X_tr, bundle.theta_oracle[tr])

        # Per-tree predictions: (n_te, n_trees, P)
        tree_preds = per_tree_predictions(rf, X_te)
        eps_box = 1e-4 * (hi_np - lo_np)
        tree_preds = np.clip(tree_preds, lo_np + eps_box, hi_np - eps_box)

        for j in range(len(te)):
            c = bundle.curves[te[j]]
            t_obs = c.t_obs.astype(np.float64)
            q_obs = c.q_obs.astype(np.float64)
            mask = (t_obs > 0.0) & (t_obs <= bundle.t_max)
            if mask.sum() < 2:
                continue
            t_eval = t_obs[mask]
            y_eval = q_obs[mask]
            t_eval_t = torch.tensor(t_eval, dtype=torch.float32)

            # Build ensemble posterior from this curve's 400 tree theta predictions
            theta_trees = tree_preds[j]                        # (n_trees, P)
            tgt = ensemble_posterior(
                theta_per_tree=theta_trees, rank=rank,
                prior_low=lo_np, prior_high=hi_np,
            )
            mu_hat = torch.tensor(tgt.mu, dtype=torch.float32)
            U_hat = torch.tensor(tgt.U, dtype=torch.float32)
            sigma_hat = torch.tensor(tgt.sigma_active, dtype=torch.float32)

            # Sample theta from N(mu, U diag(sigma^2) U^T).  No soft-box
            # because mu is already in-box (averaged tree predictions) and
            # ensemble sigma is generally much smaller than prior_std.
            S = n_samples
            eps = torch.randn(S, sigma_hat.shape[0])
            delta = (U_hat @ (eps * sigma_hat).T).T                # (S, P)
            theta_samples = mu_hat.unsqueeze(0) + delta            # (S, P)
            # Clip to prior box (rare excursions only).
            theta_samples = torch.clamp(theta_samples, prior_low, prior_high)

            # Simulate point + samples
            with torch.no_grad():
                try:
                    Q_point = decoder(mu_hat.unsqueeze(0), t_eval_t).cpu().numpy()[0]
                    Q_samples = decoder(theta_samples, t_eval_t).cpu().numpy()
                except Exception:
                    continue

            sample_mean = Q_samples.mean(axis=0)
            recenter = Q_point - sample_mean
            lo90 = np.percentile(Q_samples, 5.0, axis=0)  + recenter
            hi90 = np.percentile(Q_samples, 95.0, axis=0) + recenter
            lo50 = np.percentile(Q_samples, 25.0, axis=0) + recenter
            hi50 = np.percentile(Q_samples, 75.0, axis=0) + recenter
            cov90 = float(((y_eval >= lo90) & (y_eval <= hi90)).mean())
            cov50 = float(((y_eval >= lo50) & (y_eval <= hi50)).mean())
            pi_w = float((hi90 - lo90).mean())
            r2 = float(per_curve_r2(y_eval[None, :], Q_point[None, :])[0])
            all_rows.append({
                "fold": fi, "fid": int(bundle.fids[te[j]]),
                "r2_point": r2, "cov90": cov90, "cov50": cov50,
                "pi_width_90": pi_w, "n_t_eval": int(len(t_eval)),
                "sigma_mean": float(tgt.sigma_active.mean()),
            })
        elapsed = time.time() - t0
        if all_rows:
            recent = [r for r in all_rows if r["fold"] == fi]
            if recent:
                r2s = np.array([r["r2_point"] for r in recent])
                cov90s = np.array([r["cov90"] for r in recent])
                print(f"  [{bundle.name}/{scheme}] fold {fi+1}/{len(splits)}  "
                      f"R^2 med={np.median(r2s):+.4f}  cov90={cov90s.mean():.3f}  "
                      f"n={len(recent)}  ({elapsed:.0f}s)", flush=True)

    cell_dir = out_dir / bundle.name / scheme
    cell_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(cell_dir / "per_curve.csv", index=False)
    df = pd.DataFrame(all_rows)
    if df.empty:
        return {"dataset": bundle.name, "scheme": scheme, "n_curves": 0}
    return {
        "dataset": bundle.name, "scheme": scheme, "n_curves": len(df),
        "r2_median": float(df["r2_point"].median()),
        "r2_mean": float(df["r2_point"].mean()),
        "frac_R2_gte_0": float((df["r2_point"] >= 0).mean()),
        "frac_R2_gte_0p5": float((df["r2_point"] >= 0.5).mean()),
        "frac_R2_gte_0p9": float((df["r2_point"] >= 0.9).mean()),
        "cov90_mean": float(df["cov90"].mean()),
        "cov90_median": float(df["cov90"].median()),
        "cov50_mean": float(df["cov50"].mean()),
        "pi_width_90_median": float(df["pi_width_90"].median()),
        "sigma_mean_median": float(df["sigma_mean"].median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/66_ensemble_casp"))
    ap.add_argument("--datasets", nargs="+", default=["cross321", "internal181"])
    ap.add_argument("--schemes", nargs="+",
                    default=["random_5fold", "group_by_drug", "group_by_polymer"])
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loaders = {
        "cross321": m62.load_cross321,
        "internal181": m62.load_internal181,
    }

    summary_rows: list[dict] = []
    for ds_name in args.datasets:
        print(f"\n[66] === dataset: {ds_name} (Ensemble-CASP zero-shot) ===", flush=True)
        bundle = loaders[ds_name]()
        # Strip early-Q features for true zero-shot.
        bundle.X = bundle.X[:, :bundle.n_form].copy()
        print(f"  n_curves={len(bundle.X)}  features={bundle.X.shape[1]} (form only)",
              flush=True)
        for scheme in args.schemes:
            print(f"  scheme: {scheme}", flush=True)
            t0 = time.time()
            row = run_dataset_scheme_ensemble(
                bundle, scheme=scheme, out_dir=args.out,
                n_estimators=args.n_estimators, rank=args.rank,
                n_samples=args.n_samples, n_folds=args.n_folds, seed=args.seed,
            )
            row["wallclock_s"] = round(time.time() - t0, 1)
            summary_rows.append(row)
            print(f"  -> cell done in {row['wallclock_s']}s  "
                  f"r2_med={row.get('r2_median', float('nan')):+.4f}  "
                  f"cov90={row.get('cov90_mean', float('nan')):.3f}", flush=True)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.out / "aggregate_table.csv", index=False)
    print("\n[66] === Aggregate (Ensemble-CASP zero-shot) ===", flush=True)
    print(df_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
