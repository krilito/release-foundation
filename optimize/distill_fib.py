"""FIB → Amortized Posterior Distillation.

The core optimization script. Trains an amortized neural posterior
from FIB (Fisher-Information Bottleneck) teacher samples, replacing
the current NPE which uses a weaker Stage-1 q_phi teacher.

Pipeline:
    1. Load cross321 PLGA data (same as script 62).
    2. For each curve: compute FIB posterior, draw K samples.
    3. Train amortized NPE on (Q_curve, theta_samples) pairs.
    4. Evaluate on held-out folds: R², coverage, ESS.
    5. Compare to baseline NPE (FCEmbedding) and direct RF→Q.

Key improvements over current NPE:
    - TemporalConvEmbedding replaces FCEmbedding (respects time structure)
    - FIB teacher provides calibrated local posterior (not just point estimate)
    - Drug-polymer interaction features in descriptor branch

Usage:
    python optimize/distill_fib.py --device cuda --n-samples 64 --epochs 200
    python optimize/distill_fib.py --device cpu --n-samples 32 --epochs 50  # quick test
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
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from casp import fib_posterior, sample_theta_fib
from casp.calibration import per_curve_r2
from optimize.fib_fast import fib_posterior_fast
from optimize.calibration import calibrate_temperature, coverage_at_level, effective_sample_size
from optimize.interaction_encoder import add_interaction_features, interaction_input_dim
from optimize.temporal_embed import TemporalConvEmbedding
from simulator import PLGABiphasic
from weibull_simulator import WeibullSimulator

# ---------------------------------------------------------------------------
# Data loading (reuses script 38d's loader via importlib)
# ---------------------------------------------------------------------------

def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader_38d = _load_script(
    ROOT / "scripts" / "38d_baselines_groupkfold.py", "_loader_38d_distill",
)


def load_cross321() -> dict:
    """Load cross321 PLGA data in the same format as script 62."""
    FORMULATION_COL_MAP = _loader_38d.FORMULATION_COL_MAP

    xlsx = Path(
        r"D:\chemical-world-model-v0\datset\321PLGA"
        r"\A Dataset on Formulation Parameters and Characteristics of "
        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
    )
    matched = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
    bank_path = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")

    df_meta = pd.read_excel(xlsx, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())
    df_bank = pd.read_csv(bank_path)

    # Matched CSV uses 'Formulation_Index' (int); bank uses 'fid' (int)
    df_matched = pd.read_csv(matched)
    fid_col_matched = "Formulation_Index" if "Formulation_Index" in df_matched.columns else "fid"
    matched_fids = set(df_matched[fid_col_matched].astype(int))
    df_meta_first["fid"] = df_meta_first["fid"].astype(int)
    df_meta_first = df_meta_first[df_meta_first["fid"].isin(matched_fids)]

    # Build feature matrix
    X = df_meta_first[feature_cols].values.astype(np.float32)
    fids = df_meta_first["fid"].values.astype(int)

    # Load curves
    curves = {}
    for _, row in df_meta.iterrows():
        fid = int(row["Formulation Index"])
        if fid not in curves:
            curves[fid] = {"t": [], "q": []}
        curves[fid]["t"].append(float(row["Time"]))
        curves[fid]["q"].append(float(row["Release"]))

    # Sort by time
    for fid in curves:
        order = np.argsort(curves[fid]["t"])
        curves[fid]["t"] = np.array(curves[fid]["t"])[order]
        curves[fid]["q"] = np.array(curves[fid]["q"])[order]

    # Load oracle theta from bank
    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    theta_oracle = np.full((len(fids), 9), np.nan, dtype=np.float32)
    bank_map = {int(r["fid"]): r for _, r in df_bank.iterrows()}
    for i, fid in enumerate(fids):
        if fid in bank_map:
            row = bank_map[fid]
            theta_oracle[i] = [row[c] for c in theta_cols]

    # Drug groups for CV (extract drug name from DP_Group or use first col)
    # For cross321, use drug MW binning as a proxy for drug identity
    drug_mw = df_meta_first["drug_mw"].values if "drug_mw" in df_meta_first.columns else None
    if drug_mw is not None:
        # Bin into 5 groups by drug MW quantiles
        bins = np.quantile(drug_mw[~np.isnan(drug_mw)], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        drug_groups = np.digitize(drug_mw, bins[1:-1]).astype(str)
    else:
        drug_groups = np.array(["0"] * len(fids))

    return {
        "X": X,
        "fids": fids,
        "curves": curves,
        "theta_oracle": theta_oracle,
        "drug_groups": drug_groups,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# FIB teacher: generate posterior samples for each curve
# ---------------------------------------------------------------------------

def generate_fib_teacher(
    X: np.ndarray,
    curves: dict,
    fids: np.ndarray,
    theta_oracle: np.ndarray,
    simulator: PLGABiphasic,
    n_samples: int = 64,
    rank: int = 4,
    sigma_obs: float = 0.035,
    seed: int = 0,
    fast: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate FIB posterior samples as teacher targets.

    Args:
        fast: if True, use sparse-grid Jacobian (32 pts) + relaxed ODE
              tolerances for ~4x speedup. Results are slightly less precise
              but adequate for teacher distillation.

    Returns:
        theta_teacher: (N, K, 9) FIB samples per curve.
        Q_curves: (N, T) interpolated release curves on canonical grid.
        valid_mask: (N,) bool, True if FIB succeeded for this curve.
    """
    torch.manual_seed(seed)
    t_grid = torch.linspace(0.0, 90.0, 64)
    prior = simulator.prior().base_dist

    # Use relaxed-tolerance simulator for FIB Jacobian if fast mode
    if fast:
        sim_fib = PLGABiphasic(rtol=1e-3, atol=1e-4)
        n_jac = 32
    else:
        sim_fib = simulator
        n_jac = 64

    theta_teacher = []
    Q_curves = []
    valid = []
    n_total = len(fids)
    t_start = time.time()

    for i, fid in enumerate(fids):
        fid_key = int(fid) if not isinstance(fid, (int, np.integer)) else fid
        if fid_key not in curves or np.isnan(theta_oracle[i]).any():
            theta_teacher.append(np.zeros((n_samples, 9)))
            Q_curves.append(np.zeros(64))
            valid.append(False)
            continue

        c = curves[fid_key]
        t_obs = torch.tensor(c["t"], dtype=torch.float32)
        q_obs = torch.tensor(c["q"], dtype=torch.float32)

        # Interpolate to canonical grid
        q_grid = torch.tensor(
            np.interp(t_grid.numpy(), t_obs.numpy(), q_obs.numpy()),
            dtype=torch.float32,
        ).clamp(0, 1)

        # FIB posterior at oracle theta
        theta_hat = torch.tensor(theta_oracle[i], dtype=torch.float32)
        try:
            if fast:
                q_fib = fib_posterior_fast(
                    sim_fib, theta_hat, t_grid,
                    sigma_obs=sigma_obs, rank=rank, alpha=1e-2,
                    prior_low=prior.low, prior_high=prior.high,
                    n_jac_points=n_jac,
                )
            else:
                q_fib = fib_posterior(
                    simulator, theta_hat, t_grid,
                    sigma_obs=sigma_obs, rank=rank, alpha=1e-2,
                    prior_low=prior.low, prior_high=prior.high,
                )
            theta_s = sample_theta_fib(
                q_fib, n_samples,
                prior_low=prior.low, prior_high=prior.high,
            )  # (K, 9)
            theta_teacher.append(theta_s.numpy())
            Q_curves.append(q_grid.numpy())
            valid.append(True)
        except Exception:
            theta_teacher.append(np.zeros((n_samples, 9)))
            Q_curves.append(q_grid.numpy())
            valid.append(False)

        # Progress
        if (i + 1) % 25 == 0 or i == n_total - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            n_v = sum(valid)
            print(f"  [{i+1}/{n_total}] {n_v} valid, {elapsed:.0f}s elapsed, {eta:.0f}s remaining")

    return np.array(theta_teacher), np.array(Q_curves), np.array(valid)


# ---------------------------------------------------------------------------
# Train amortized posterior from FIB teacher
# ---------------------------------------------------------------------------

def train_amortized_posterior(
    Q_train: np.ndarray,
    theta_train: np.ndarray,
    simulator: PLGABiphasic,
    use_temporal_embed: bool = True,
    hidden_features: int = 64,
    num_transforms: int = 5,
    batch_size: int = 256,
    max_epochs: int = 200,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Train an amortized NPE from FIB teacher samples.

    Args:
        Q_train: (N, T) curves.
        theta_train: (N, K, 9) FIB teacher samples per curve.
        simulator: the PLGA simulator (for prior).
        use_temporal_embed: if True, use TemporalConvEmbedding; else FCEmbedding.
        Returns: dict with posterior, training_log, device.
    """
    torch.manual_seed(seed)
    N, K, P = theta_train.shape
    T = Q_train.shape[1]

    # Flatten: each curve contributes K (theta, Q) pairs
    Q_flat = np.repeat(Q_train, K, axis=0)      # (N*K, T)
    theta_flat = theta_train.reshape(N * K, P)   # (N*K, 9)

    Q_tensor = torch.tensor(Q_flat, dtype=torch.float32)
    theta_tensor = torch.tensor(theta_flat, dtype=torch.float32)

    # Build embedding net
    if use_temporal_embed:
        embed_net = TemporalConvEmbedding(output_dim=16, n_channels=32, n_layers=5)
        print(f"[distill] using TemporalConvEmbedding (params={sum(p.numel() for p in embed_net.parameters()):,})")
    else:
        embed_net = FCEmbedding(input_dim=T, output_dim=16, num_layers=3, num_hiddens=64)
        print(f"[distill] using FCEmbedding (baseline)")

    density_est = posterior_nn(
        model="maf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        embedding_net=embed_net,
    )

    prior = simulator.prior()
    if device != "cpu":
        base = prior.base_dist
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(base.low.to(device), base.high.to(device)),
            prior.reinterpreted_batch_ndims,
        )

    inference = NPE(prior=prior, density_estimator=density_est, device=device)
    inference.append_simulations(theta_tensor, Q_tensor)

    t0 = time.time()
    estimator = inference.train(
        training_batch_size=batch_size,
        max_num_epochs=max_epochs,
        validation_fraction=0.1,
        stop_after_epochs=20,
        show_train_summary=True,
    )
    wall = time.time() - t0

    estimator.to(device)
    posterior = inference.build_posterior(estimator)

    return {
        "posterior": posterior,
        "estimator": estimator,
        "inference": inference,
        "wall_seconds": wall,
        "n_pairs": N * K,
        "use_temporal_embed": use_temporal_embed,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_posterior(
    posterior,
    Q_test: np.ndarray,
    theta_true: np.ndarray,
    simulator: PLGABiphasic,
    n_eval_samples: int = 1000,
    device: str = "cuda",
) -> dict:
    """Evaluate amortized posterior on held-out curves.

    Returns per-curve R², coverage, ESS, and summary stats.
    """
    t_grid = torch.linspace(0.0, 90.0, 64)
    results = []

    for i in range(len(Q_test)):
        q_obs = torch.tensor(Q_test[i], dtype=torch.float32).to(device)
        try:
            theta_s = posterior.sample(
                (n_eval_samples,), x=q_obs, show_progress_bars=False,
            )  # (S, 9)
            theta_s = theta_s.cpu()
            theta_hat = theta_s.mean(dim=0)  # point estimate

            # Simulate predicted curve
            with torch.no_grad():
                Q_pred = simulator.simulate(
                    theta_hat.unsqueeze(0), t_grid
                ).squeeze(0).numpy()

            r2 = per_curve_r2(Q_test[i], Q_pred)
            cov90 = coverage_at_level(theta_s, torch.tensor(theta_true[i]), 0.9)
            ess = effective_sample_size(theta_s)

            results.append({
                "r2": r2,
                "coverage_90": cov90,
                "ess": ess,
                "theta_hat": theta_hat.numpy(),
                "theta_samples": theta_s.numpy(),
            })
        except Exception as e:
            results.append({
                "r2": float("nan"),
                "coverage_90": False,
                "ess": 0.0,
                "theta_hat": np.full(9, np.nan),
                "theta_samples": np.zeros((n_eval_samples, 9)),
                "error": str(e),
            })

    r2s = [r["r2"] for r in results if not np.isnan(r["r2"])]
    coverages = [r["coverage_90"] for r in results]
    esses = [r["ess"] for r in results if r["ess"] > 0]

    return {
        "per_curve": results,
        "r2_median": float(np.median(r2s)) if r2s else float("nan"),
        "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
        "frac_r2_gte_0": float(np.mean(np.array(r2s) >= 0)) if r2s else float("nan"),
        "coverage_90_mean": float(np.mean(coverages)) if coverages else float("nan"),
        "ess_median": float(np.median(esses)) if esses else float("nan"),
        "n_eval": len(results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FIB → Amortized Posterior Distillation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=64, help="FIB samples per curve (teacher K)")
    parser.add_argument("--n-eval-samples", type=int, default=1000, help="Posterior samples at eval time")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--transforms", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4, help="FIB active subspace rank")
    parser.add_argument("--sigma-obs", type=float, default=0.035, help="Observation noise for FIB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-only", action="store_true", help="Only run FCEmbedding baseline")
    parser.add_argument("--temporal-only", action="store_true", help="Only run TemporalConvEmbedding")
    parser.add_argument("--output-dir", type=str, default="optimize/outputs/distill_fib")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"FIB → Amortized Posterior Distillation")
    print(f"{'='*60}")
    print(f"device={args.device}, K={args.n_samples}, epochs={args.epochs}")
    print(f"rank={args.rank}, sigma_obs={args.sigma_obs}")
    print()

    # 1. Load data
    print("[1/4] Loading cross321 PLGA data...")
    data = load_cross321()
    X = data["X"]
    fids = data["fids"]
    curves = data["curves"]
    theta_oracle = data["theta_oracle"]
    drug_groups = data["drug_groups"]
    print(f"  {len(fids)} formulations, {X.shape[1]} features")

    # 2. Generate FIB teacher
    print(f"\n[2/4] Generating FIB teacher ({args.n_samples} samples × {len(fids)} curves)...")
    sim = PLGABiphasic()
    t0 = time.time()
    theta_teacher, Q_curves, valid_mask = generate_fib_teacher(
        X, curves, fids, theta_oracle, sim,
        n_samples=args.n_samples, rank=args.rank,
        sigma_obs=args.sigma_obs, seed=args.seed,
    )
    n_valid = valid_mask.sum()
    print(f"  {n_valid}/{len(fids)} curves with valid FIB posterior ({time.time()-t0:.1f}s)")

    # Filter to valid
    X_valid = X[valid_mask]
    Q_valid = Q_curves[valid_mask]
    theta_valid = theta_teacher[valid_mask]
    groups_valid = drug_groups[valid_mask]
    theta_oracle_valid = theta_oracle[valid_mask]

    # 3. Cross-validation: group by drug
    print(f"\n[3/4] Running group-by-drug 5-fold CV...")
    gkf = GroupKFold(n_splits=5)

    configs = []
    if not args.baseline_only:
        configs.append(("temporal_conv", True))
    if not args.temporal_only:
        configs.append(("fc_embed", False))

    all_results = {}

    for name, use_temporal in configs:
        print(f"\n--- {name} ---")
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_valid, groups=groups_valid)):
            print(f"  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")

            Q_train = Q_valid[train_idx]
            theta_train = theta_valid[train_idx]
            Q_test = Q_valid[test_idx]
            theta_test = theta_oracle_valid[test_idx]

            # Train
            result = train_amortized_posterior(
                Q_train, theta_train, sim,
                use_temporal_embed=use_temporal,
                hidden_features=args.hidden,
                num_transforms=args.transforms,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                device=args.device,
                seed=args.seed + fold_idx,
            )

            # Evaluate
            eval_result = evaluate_posterior(
                result["posterior"], Q_test, theta_test, sim,
                n_eval_samples=args.n_eval_samples,
                device=args.device,
            )
            eval_result["wall_seconds"] = result["wall_seconds"]
            eval_result["n_pairs"] = result["n_pairs"]
            fold_results.append(eval_result)

            print(f"    R² median={eval_result['r2_median']:.4f}, "
                  f"cov90={eval_result['coverage_90_mean']:.2f}, "
                  f"ESS={eval_result['ess_median']:.1f}")

        # Aggregate across folds
        all_r2 = [f["r2_median"] for f in fold_results if not np.isnan(f["r2_median"])]
        all_cov = [f["coverage_90_mean"] for f in fold_results]
        all_ess = [f["ess_median"] for f in fold_results if f["ess_median"] > 0]

        summary = {
            "r2_median_across_folds": float(np.median(all_r2)) if all_r2 else float("nan"),
            "r2_mean_across_folds": float(np.mean(all_r2)) if all_r2 else float("nan"),
            "coverage_90_mean": float(np.mean(all_cov)) if all_cov else float("nan"),
            "ess_median": float(np.median(all_ess)) if all_ess else float("nan"),
            "total_wall_seconds": sum(f["wall_seconds"] for f in fold_results),
            "n_folds": len(fold_results),
        }
        all_results[name] = {"summary": summary, "folds": fold_results}

        print(f"\n  {name} aggregate:")
        print(f"    R² median across folds: {summary['r2_median_across_folds']:.4f}")
        print(f"    Coverage 90% mean:      {summary['coverage_90_mean']:.3f}")
        print(f"    ESS median:             {summary['ess_median']:.1f}")
        print(f"    Total wall time:        {summary['total_wall_seconds']:.0f}s")

    # 4. Save results
    print(f"\n[4/4] Saving results to {out}/")

    # Summary text
    with open(out / "summary.txt", "w") as f:
        f.write(f"=== FIB → Amortized Posterior Distillation ===\n\n")
        f.write(f"device       : {args.device}\n")
        f.write(f"n_samples    : {args.n_samples}\n")
        f.write(f"n_eval       : {args.n_eval_samples}\n")
        f.write(f"epochs       : {args.epochs}\n")
        f.write(f"rank         : {args.rank}\n")
        f.write(f"sigma_obs    : {args.sigma_obs}\n")
        f.write(f"n_valid      : {n_valid}/{len(fids)}\n")
        f.write(f"cv_scheme    : group_by_drug 5-fold\n\n")

        for name, res in all_results.items():
            s = res["summary"]
            f.write(f"--- {name} ---\n")
            f.write(f"  R² median across folds: {s['r2_median_across_folds']:.4f}\n")
            f.write(f"  R² mean across folds:   {s['r2_mean_across_folds']:.4f}\n")
            f.write(f"  Coverage 90% mean:      {s['coverage_90_mean']:.3f}\n")
            f.write(f"  ESS median:             {s['ess_median']:.1f}\n")
            f.write(f"  Total wall time:        {s['total_wall_seconds']:.0f}s\n\n")

        # Comparison
        if "temporal_conv" in all_results and "fc_embed" in all_results:
            t_res = all_results["temporal_conv"]["summary"]
            b_res = all_results["fc_embed"]["summary"]
            f.write(f"--- delta (temporal vs fc_embed) ---\n")
            f.write(f"  Δ R² median: {t_res['r2_median_across_folds'] - b_res['r2_median_across_folds']:+.4f}\n")
            f.write(f"  Δ cov90:     {t_res['coverage_90_mean'] - b_res['coverage_90_mean']:+.3f}\n")
            f.write(f"  Δ ESS:       {t_res['ess_median'] - b_res['ess_median']:+.1f}\n")

    # Per-curve CSVs
    for name, res in all_results.items():
        rows = []
        for fold_idx, fold in enumerate(res["folds"]):
            for j, pc in enumerate(fold["per_curve"]):
                rows.append({
                    "fold": fold_idx,
                    "curve_idx": j,
                    "r2": pc["r2"],
                    "coverage_90": pc["coverage_90"],
                    "ess": pc["ess"],
                })
        pd.DataFrame(rows).to_csv(out / f"per_curve_{name}.csv", index=False)

    print("\nDone.")
    for name, res in all_results.items():
        s = res["summary"]
        print(f"\n{name}:")
        print(f"  R² median = {s['r2_median_across_folds']:.4f}")
        print(f"  cov90     = {s['coverage_90_mean']:.3f}")
        print(f"  ESS       = {s['ess_median']:.1f}")


if __name__ == "__main__":
    main()
