from __future__ import annotations

"""
160_baseline_shared_hierarchical.py

Consume:
- outputs/158_freeze_theta_targets/theta_table_v1.parquet
- outputs/148_release_main_cumulative_v1/curves_long.csv
- outputs/148_release_main_cumulative_v1/formulations.csv

Produce:
- outputs/160_baseline_shared_hierarchical/per_curve_metrics.csv
- outputs/160_baseline_shared_hierarchical/system_summary.csv
- outputs/160_baseline_shared_hierarchical/manifest.json
- outputs/160_baseline_shared_hierarchical/summary.md

Expected runtime:
- < 30 s on the current 751-curve theta table
"""

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_baseline_utils import (
    evaluate_prediction,
    inverse_targets,
    json_list,
    load_theta_and_pool,
    median_param_fallback,
    prepare_design,
    ridge_predict,
    transform_targets,
)


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
        default=repo_root / "outputs" / "160_baseline_shared_hierarchical",
        help="output directory",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--alpha", type=float, default=1e-2, help="ridge penalty")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def predict_one_curve(
    heldout: pd.Series,
    train_df: pd.DataFrame,
    curve_groups: dict[str, pd.DataFrame],
    alpha: float,
) -> dict[str, object]:
    family = str(heldout["theta_family"])
    fam_train = train_df[train_df["theta_family"] == family].copy()
    if len(fam_train) == 0:
        return {
            "available": False,
            "fit_mode": "unavailable",
            "fallback_reason": "no_same_family_train_rows",
        }

    if len(fam_train) >= 2:
        x_train, x_test, design_info = prepare_design(fam_train, heldout.to_frame().T)
        y_train = transform_targets(family, fam_train)
        y_pred = ridge_predict(x_train=x_train, y_train=y_train, x_test=x_test, alpha=alpha)
        params = inverse_targets(family, y_pred)[0].tolist()
        fit_mode = "ridge"
        fallback_reason = ""
    else:
        params = median_param_fallback(fam_train, family)
        fit_mode = "median_fallback"
        fallback_reason = "single_train_example"
        design_info = {"design_cols": []}

    curve_df = curve_groups[str(heldout["unified_curve_id"])]
    metrics = evaluate_prediction(curve_df, family, params)
    target_params = json.loads(str(heldout["theta_params_json"]))
    return {
        "available": True,
        "fit_mode": fit_mode,
        "fallback_reason": fallback_reason,
        "design_n_features": int(len(design_info.get("design_cols", []))),
        "predicted_params_json": json_list(params),
        "target_params_json": json.dumps(target_params),
        "param_mae_mean": float(np.mean(np.abs(np.asarray(params) - np.asarray(target_params)))),
        **metrics,
    }


def write_summary(out_path: Path, system_summary: pd.DataFrame) -> None:
    lines = [
        "# Baseline Shared Hierarchical",
        "",
        "LOSO pooled prior baseline using oracle routed theta families.",
        "",
    ]
    for _, row in system_summary.iterrows():
        lines.extend(
            [
                f"## {row['system_id']}",
                "",
                f"- n_total_curves: `{int(row['n_total_curves'])}`",
                f"- n_eval_curves: `{int(row['n_eval_curves'])}`",
                f"- median_curve_r2: `{row['median_curve_r2']:.4f}`",
                f"- mean_curve_r2: `{row['mean_curve_r2']:.4f}`",
                f"- median_curve_rmse: `{row['median_curve_rmse']:.4f}`",
                f"- median_param_mae_mean: `{row['median_param_mae_mean']:.4f}`",
                f"- ridge_fraction: `{row['ridge_fraction']:.4f}`",
                "",
            ]
        )
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
            result = predict_one_curve(heldout=heldout, train_df=train_df, curve_groups=curve_groups, alpha=args.alpha)
            rows.append(
                {
                    "mode": "loso_pooled_oracle_family",
                    "system_id": heldout_system,
                    "unified_curve_id": heldout["unified_curve_id"],
                    "theta_family": heldout["theta_family"],
                    "available": bool(result["available"]),
                    "fit_mode": result["fit_mode"],
                    "fallback_reason": result.get("fallback_reason", ""),
                    "design_n_features": result.get("design_n_features", 0),
                    "curve_r2": result.get("curve_r2", np.nan),
                    "curve_rmse": result.get("curve_rmse", np.nan),
                    "curve_mae": result.get("curve_mae", np.nan),
                    "param_mae_mean": result.get("param_mae_mean", np.nan),
                    "predicted_params_json": result.get("predicted_params_json", ""),
                    "target_params_json": result.get("target_params_json", ""),
                }
            )

    per_curve = pd.DataFrame(rows)
    eval_df = per_curve[per_curve["available"]].copy()
    system_summary = (
        eval_df.groupby("system_id", dropna=False)
        .agg(
            n_eval_curves=("unified_curve_id", "count"),
            median_curve_r2=("curve_r2", "median"),
            mean_curve_r2=("curve_r2", "mean"),
            median_curve_rmse=("curve_rmse", "median"),
            mean_curve_rmse=("curve_rmse", "mean"),
            median_param_mae_mean=("param_mae_mean", "median"),
            ridge_fraction=("fit_mode", lambda x: float(np.mean(pd.Series(x) == "ridge"))),
        )
        .reset_index()
    )
    total_counts = theta.groupby("system_id")["unified_curve_id"].nunique().rename("n_total_curves").reset_index()
    system_summary = total_counts.merge(system_summary, on="system_id", how="left").sort_values("system_id").reset_index(
        drop=True
    )

    manifest = {
        "artifact_name": "baseline_shared_hierarchical",
        "seed": int(args.seed),
        "theta_dir": str(args.theta_dir),
        "pool_dir": str(args.pool_dir),
        "alpha": float(args.alpha),
        "mode": "loso_pooled_oracle_family",
        "n_total_rows": int(len(per_curve)),
        "n_eval_rows": int(len(eval_df)),
        "systems": sorted(theta["system_id"].dropna().unique().tolist()),
    }

    per_curve.to_csv(args.outdir / "per_curve_metrics.csv", index=False)
    system_summary.to_csv(args.outdir / "system_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(args.outdir / "summary.md", system_summary)

    print(f"[baseline-shared-hierarchical] wrote outputs to {args.outdir}")
    print(f"[baseline-shared-hierarchical] eval_rows={manifest['n_eval_rows']} systems={len(manifest['systems'])}")


if __name__ == "__main__":
    main()
