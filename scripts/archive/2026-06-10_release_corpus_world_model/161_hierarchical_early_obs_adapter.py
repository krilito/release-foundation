from __future__ import annotations

"""
161_hierarchical_early_obs_adapter.py

Consume:
- outputs/158_freeze_theta_targets/theta_table_v1.parquet
- outputs/148_release_main_cumulative_v1/curves_long.csv
- outputs/148_release_main_cumulative_v1/formulations.csv

Produce:
- outputs/161_hierarchical_early_obs_adapter/per_curve_metrics.csv
- outputs/161_hierarchical_early_obs_adapter/system_summary.csv
- outputs/161_hierarchical_early_obs_adapter/manifest.json
- outputs/161_hierarchical_early_obs_adapter/summary.md

Expected runtime:
- ~1-3 min on the current 751-curve theta table
"""

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_baseline_utils import (
    FAMILY_BOUNDS,
    FAMILY_FUNCS,
    evaluate_prediction,
    inverse_targets,
    json_list,
    load_theta_and_pool,
    median_param_fallback,
    prepare_design,
    ridge_predict,
    transform_targets,
)


EARLY_OBS_LIST = (0, 3, 5)


def parse_args() -> argparse.Namespace:
    repo_root = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theta-dir",
        type=Path,
        default=repo_root / "outputs" / "158_freeze_theta_targets",
        help="frozen theta target directory",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=repo_root / "outputs" / "148_release_main_cumulative_v1",
        help="main cumulative pool directory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "161_hierarchical_early_obs_adapter",
        help="output directory",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--alpha", type=float, default=1e-2, help="ridge penalty for pooled prior")
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=0.25,
        help="regularization strength that pulls posterior parameters back toward the pooled prior",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def predict_prior_params(heldout: pd.Series, train_df: pd.DataFrame, alpha: float) -> tuple[list[float], str, str, int]:
    family = str(heldout["theta_family"])
    fam_train = train_df[train_df["theta_family"] == family].copy()
    if len(fam_train) == 0:
        raise ValueError(f"No pooled prior rows for family: {family}")
    if len(fam_train) >= 2:
        x_train, x_test, design_info = prepare_design(fam_train, heldout.to_frame().T)
        y_train = transform_targets(family, fam_train)
        y_pred = ridge_predict(x_train=x_train, y_train=y_train, x_test=x_test, alpha=alpha)
        params = inverse_targets(family, y_pred)[0].tolist()
        return params, "ridge", "", int(len(design_info["design_cols"]))
    params = median_param_fallback(fam_train, family)
    return params, "median_fallback", "single_train_example", 0


def posterior_update(
    family: str,
    prior_params: list[float],
    obs_t: np.ndarray,
    obs_y: np.ndarray,
    prior_strength: float,
) -> list[float]:
    if len(obs_t) == 0:
        return prior_params
    lo, hi = FAMILY_BOUNDS[family]
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    prior = np.clip(np.asarray(prior_params, dtype=float), lo_arr, hi_arr)
    scale = np.maximum(hi_arr - lo_arr, 1e-8)
    fn = FAMILY_FUNCS[family]

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = fn(obs_t, params)
        data_resid = pred - obs_y
        prior_resid = np.sqrt(prior_strength) * ((params - prior) / scale)
        return np.concatenate([data_resid, prior_resid])

    result = least_squares(
        residuals,
        x0=prior,
        bounds=(lo_arr, hi_arr),
        max_nfev=2000,
    )
    return np.clip(result.x, lo_arr, hi_arr).tolist()


def write_summary(out_path: Path, system_summary: pd.DataFrame) -> None:
    lines = [
        "# Hierarchical Early Observation Adapter",
        "",
        "LOSO pooled prior followed by early-observation parameter updates.",
        "",
    ]
    for n_obs, sub in system_summary.groupby("n_obs", sort=True):
        lines.extend([f"## early_obs = {int(n_obs)}", ""])
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['system_id']}`: n_eval={int(row['n_eval_curves'])}, "
                f"median_R2={row['median_curve_r2']:.4f}, "
                f"median_RMSE={row['median_curve_rmse']:.4f}, "
                f"median_param_MAE={row['median_param_mae_mean']:.4f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    loaded = load_theta_and_pool(args.theta_dir, args.pool_dir)
    theta = loaded["theta"]
    curve_groups = loaded["curve_groups"]

    rows: list[dict[str, object]] = []
    for heldout_system in sorted(theta["system_id"].dropna().unique().tolist()):
        test_df = theta[theta["system_id"] == heldout_system].copy().reset_index(drop=True)
        train_df = theta[theta["system_id"] != heldout_system].copy().reset_index(drop=True)
        for _, heldout in test_df.iterrows():
            prior_params, prior_mode, prior_fallback_reason, design_n_features = predict_prior_params(
                heldout=heldout,
                train_df=train_df,
                alpha=args.alpha,
            )
            curve_df = curve_groups[str(heldout["unified_curve_id"])].sort_values("time_days").reset_index(drop=True)
            t_all = curve_df["time_days"].to_numpy(dtype=float)
            y_all = curve_df["release_fraction"].to_numpy(dtype=float)
            target_params = json.loads(str(heldout["theta_params_json"]))

            for n_obs in EARLY_OBS_LIST:
                obs_t = t_all[:n_obs]
                obs_y = y_all[:n_obs]
                posterior_params = posterior_update(
                    family=str(heldout["theta_family"]),
                    prior_params=prior_params,
                    obs_t=obs_t,
                    obs_y=obs_y,
                    prior_strength=args.prior_strength,
                )
                metrics = evaluate_prediction(curve_df, str(heldout["theta_family"]), posterior_params)
                rows.append(
                    {
                        "mode": "loso_pooled_oracle_family_early_obs",
                        "system_id": heldout_system,
                        "unified_curve_id": heldout["unified_curve_id"],
                        "theta_family": heldout["theta_family"],
                        "n_obs": int(n_obs),
                        "prior_mode": prior_mode,
                        "prior_fallback_reason": prior_fallback_reason,
                        "design_n_features": int(design_n_features),
                        "curve_r2": metrics["curve_r2"],
                        "curve_rmse": metrics["curve_rmse"],
                        "curve_mae": metrics["curve_mae"],
                        "param_mae_mean": float(
                            np.mean(np.abs(np.asarray(posterior_params) - np.asarray(target_params, dtype=float)))
                        ),
                        "prior_params_json": json_list(prior_params),
                        "posterior_params_json": json_list(posterior_params),
                        "target_params_json": json.dumps(target_params),
                    }
                )

    per_curve = pd.DataFrame(rows)
    system_summary = (
        per_curve.groupby(["system_id", "n_obs"], dropna=False)
        .agg(
            n_eval_curves=("unified_curve_id", "count"),
            median_curve_r2=("curve_r2", "median"),
            mean_curve_r2=("curve_r2", "mean"),
            median_curve_rmse=("curve_rmse", "median"),
            mean_curve_rmse=("curve_rmse", "mean"),
            median_param_mae_mean=("param_mae_mean", "median"),
        )
        .reset_index()
        .sort_values(["n_obs", "system_id"])
        .reset_index(drop=True)
    )

    manifest = {
        "artifact_name": "hierarchical_early_obs_adapter",
        "seed": int(args.seed),
        "theta_dir": str(args.theta_dir),
        "pool_dir": str(args.pool_dir),
        "alpha": float(args.alpha),
        "prior_strength": float(args.prior_strength),
        "n_obs_grid": list(EARLY_OBS_LIST),
        "mode": "loso_pooled_oracle_family_early_obs",
        "n_rows": int(len(per_curve)),
        "systems": sorted(theta["system_id"].dropna().unique().tolist()),
    }

    per_curve.to_csv(args.outdir / "per_curve_metrics.csv", index=False)
    system_summary.to_csv(args.outdir / "system_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(args.outdir / "summary.md", system_summary)

    print(f"[hierarchical-early-obs-adapter] wrote outputs to {args.outdir}")
    print(f"[hierarchical-early-obs-adapter] rows={manifest['n_rows']} systems={len(manifest['systems'])}")


if __name__ == "__main__":
    main()
