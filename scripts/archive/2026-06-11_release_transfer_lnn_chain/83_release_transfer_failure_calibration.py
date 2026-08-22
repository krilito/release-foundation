"""83 - Release transfer failure and interval calibration diagnostics.

Purpose:
    Post-process a completed release-corpus transfer run and ask two narrow
    questions:

    1. Which held-out systems/budgets lose to which baseline families?
    2. How under-calibrated are the raw ensemble intervals under a
       leave-one-heldout-system conformal scaling diagnostic?

Consumes:
    outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored/

Produces:
    outputs/83_release_transfer_failure_calibration/
      failure_per_curve.csv
      failure_by_system_budget.csv
      named_best_by_system_budget.csv
      conformal_by_system_budget.csv
      conformal_by_budget.csv
      summary.md

Scope:
    This is diagnostic post-processing. The conformal scaling uses other LOSO
    held-out systems as calibration data and must not be reported as a proper
    deployment calibration guarantee.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN = Path("outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored")
DEFAULT_OUT = Path("outputs/83_release_transfer_failure_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Failure and calibration diagnostics for release transfer probe.")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budget-kind", default="time_days", choices=["time_days", "point"])
    parser.add_argument("--split-kind", default="loso-system")
    parser.add_argument("--near-tie-eps", type=float, default=0.01)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(values.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def baseline_family(name: str) -> str:
    if name.startswith("ShapePriorIncrementAnchored"):
        return "shape_increment_anchored"
    if name.startswith("ShapePrior"):
        return "shape_prior"
    if name == "LastValueCarryForward":
        return "carry_forward"
    if "Anchored" in name:
        return "mean_anchored"
    if name.endswith("Mean"):
        return "mean_prior"
    if name == "Reference161ThetaAdapter":
        return "theta_adapter_reference"
    return "other"


def classify_failure(row: pd.Series, near_tie_eps: float) -> str:
    delta = float(row["model_minus_best_baseline"])
    if delta <= -near_tie_eps:
        return "model_wins"
    if abs(delta) <= near_tie_eps:
        return "near_tie"

    family = str(row["best_baseline_family"])
    true_span = float(row["target_true_span"])
    mean_bias = float(row["target_mean_bias"])
    rmse = float(row["model_rmse"])

    if family == "carry_forward" and true_span <= 0.03:
        return "flat_or_saturated_carry_forward_advantage"
    if family == "carry_forward":
        return "carry_forward_advantage"
    if family in {"shape_increment_anchored", "shape_prior"}:
        return "shape_family_advantage"
    if family == "mean_anchored":
        return "anchored_mean_advantage"
    if rmse >= 0.10 and abs(mean_bias) >= 0.08:
        return "large_bias_forecast_miss"
    return "baseline_advantage_other"


def build_failure_tables(
    per_curve: pd.DataFrame,
    baselines: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    split_kind: str,
    budget_kind: str,
    near_tie_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["split_kind", "heldout_system", "unified_curve_id", "budget_kind", "budget_label", "budget_value"]
    model = per_curve[(per_curve["split_kind"] == split_kind) & (per_curve["budget_kind"] == budget_kind)].copy()
    base = baselines[(baselines["split_kind"] == split_kind) & (baselines["budget_kind"] == budget_kind)].copy()
    pred = predictions[(predictions["split_kind"] == split_kind) & (predictions["budget_kind"] == budget_kind)].copy()

    best_baseline = (
        base.sort_values(key_cols + ["rmse", "baseline"])
        .groupby(key_cols, as_index=False)
        .first()
        .rename(columns={"rmse": "best_baseline_rmse", "baseline": "best_baseline"})
    )
    pred_summary = (
        pred.groupby(key_cols, dropna=False)
        .agg(
            n_target_points=("y_true", "size"),
            target_time_min=("time_days", "min"),
            target_time_max=("time_days", "max"),
            target_true_start=("y_true", "first"),
            target_true_end=("y_true", "last"),
            target_true_min=("y_true", "min"),
            target_true_max=("y_true", "max"),
            target_pred_end=("y_pred_mean", "last"),
            target_mean_bias=("y_pred_mean", lambda s: float(np.mean(s.to_numpy(dtype=float)))),
        )
        .reset_index()
    )
    true_mean = pred.groupby(key_cols, dropna=False)["y_true"].mean().reset_index(name="target_true_mean")
    pred_summary = pred_summary.merge(true_mean, on=key_cols, how="left")
    pred_summary["target_mean_bias"] = pred_summary["target_mean_bias"] - pred_summary["target_true_mean"]
    pred_summary["target_true_span"] = pred_summary["target_true_max"] - pred_summary["target_true_min"]

    failure = model[key_cols + ["n_obs", "n_target", "rmse", "mae", "cov90", "width90", "crps"]].rename(
        columns={"rmse": "model_rmse", "mae": "model_mae"}
    )
    failure = failure.merge(best_baseline[key_cols + ["best_baseline", "best_baseline_rmse"]], on=key_cols, how="left")
    failure = failure.merge(pred_summary, on=key_cols, how="left")
    failure["best_baseline_family"] = failure["best_baseline"].astype(str).map(baseline_family)
    failure["model_minus_best_baseline"] = failure["model_rmse"] - failure["best_baseline_rmse"]
    failure["model_beats_best_baseline"] = failure["model_minus_best_baseline"] < -near_tie_eps
    failure["failure_mode"] = failure.apply(lambda row: classify_failure(row, near_tie_eps), axis=1)
    failure = failure.sort_values(["budget_value", "heldout_system", "model_minus_best_baseline"], ascending=[True, True, False])

    by_system = (
        failure.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_n_obs=("n_obs", "median"),
            median_model_rmse=("model_rmse", "median"),
            median_best_baseline_rmse=("best_baseline_rmse", "median"),
            median_delta_model_minus_best=("model_minus_best_baseline", "median"),
            mean_delta_model_minus_best=("model_minus_best_baseline", "mean"),
            fraction_model_wins=("model_beats_best_baseline", "mean"),
            median_target_true_span=("target_true_span", "median"),
            median_target_time_max=("target_time_max", "median"),
        )
        .reset_index()
    )
    mode_counts = (
        failure.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="n_mode")
        .sort_values(["split_kind", "heldout_system", "budget_value", "n_mode"], ascending=[True, True, True, False])
    )
    top_modes = mode_counts.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], as_index=False).first()
    top_modes = top_modes.rename(columns={"failure_mode": "top_failure_mode"})
    by_system = by_system.merge(
        top_modes[["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value", "top_failure_mode", "n_mode"]],
        on=["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"],
        how="left",
    )
    return failure, by_system.sort_values(["budget_value", "median_delta_model_minus_best"], ascending=[True, False])


def build_named_best_table(
    per_curve: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    split_kind: str,
    budget_kind: str,
) -> pd.DataFrame:
    model = per_curve[(per_curve["split_kind"] == split_kind) & (per_curve["budget_kind"] == budget_kind)].copy()
    base = baselines[(baselines["split_kind"] == split_kind) & (baselines["budget_kind"] == budget_kind)].copy()
    model_summary = (
        model.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_model_rmse=("rmse", "median"),
            mean_model_rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    baseline_summary = (
        base.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value", "baseline"], dropna=False)
        .agg(
            n_pairs=("unified_curve_id", "nunique"),
            median_baseline_rmse=("rmse", "median"),
            mean_baseline_rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    named_best = (
        baseline_summary.sort_values(["split_kind", "heldout_system", "budget_value", "median_baseline_rmse", "baseline"])
        .groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], as_index=False)
        .first()
        .rename(columns={"baseline": "best_named_baseline"})
    )
    out = model_summary.merge(
        named_best,
        on=["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"],
        how="left",
    )
    out["delta_model_minus_best_named"] = out["median_model_rmse"] - out["median_baseline_rmse"]
    out["best_named_baseline_family"] = out["best_named_baseline"].astype(str).map(baseline_family)
    return out.sort_values(["budget_value", "delta_model_minus_best_named"], ascending=[True, False])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.nan
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


def interval_scores(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    eps = 1e-8
    lower_half = np.maximum(out["y_pred_mean"].to_numpy(dtype=float) - out["y_pred_p05"].to_numpy(dtype=float), eps)
    upper_half = np.maximum(out["y_pred_p95"].to_numpy(dtype=float) - out["y_pred_mean"].to_numpy(dtype=float), eps)
    y = out["y_true"].to_numpy(dtype=float)
    mean = out["y_pred_mean"].to_numpy(dtype=float)
    score_low = (mean - y) / lower_half
    score_high = (y - mean) / upper_half
    out["score90"] = np.maximum(0.0, np.where(y < mean, score_low, score_high))
    out["raw_width90"] = out["y_pred_p95"] - out["y_pred_p05"]
    out["raw_cov90_point"] = ((out["y_true"] >= out["y_pred_p05"]) & (out["y_true"] <= out["y_pred_p95"])).astype(float)
    return out


def scaled_coverage_width(pred: pd.DataFrame, scale: float) -> tuple[float, float]:
    lo = pred["y_pred_mean"].to_numpy(dtype=float) + scale * (
        pred["y_pred_p05"].to_numpy(dtype=float) - pred["y_pred_mean"].to_numpy(dtype=float)
    )
    hi = pred["y_pred_mean"].to_numpy(dtype=float) + scale * (
        pred["y_pred_p95"].to_numpy(dtype=float) - pred["y_pred_mean"].to_numpy(dtype=float)
    )
    y = pred["y_true"].to_numpy(dtype=float)
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def build_conformal_tables(
    predictions: pd.DataFrame,
    *,
    split_kind: str,
    budget_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = predictions[(predictions["split_kind"] == split_kind) & (predictions["budget_kind"] == budget_kind)].copy()
    pred = interval_scores(pred)
    rows: list[dict[str, Any]] = []
    for (budget_label, budget_value), budget_df in pred.groupby(["budget_label", "budget_value"], dropna=False):
        for heldout_system, test_df in budget_df.groupby("heldout_system", dropna=False):
            cal_df = budget_df[budget_df["heldout_system"] != heldout_system]
            scale90 = conformal_quantile(cal_df["score90"].to_numpy(dtype=float), alpha=0.10)
            if not np.isfinite(scale90):
                continue
            raw_cov = float(test_df["raw_cov90_point"].mean())
            raw_width = float(test_df["raw_width90"].mean())
            cal_cov, cal_width = scaled_coverage_width(test_df, scale90)
            rows.append(
                {
                    "split_kind": split_kind,
                    "heldout_system": heldout_system,
                    "budget_kind": budget_kind,
                    "budget_label": budget_label,
                    "budget_value": float(budget_value),
                    "n_points": int(len(test_df)),
                    "n_cal_points": int(len(cal_df)),
                    "raw_cov90": raw_cov,
                    "raw_width90": raw_width,
                    "loso_conformal_scale90": float(scale90),
                    "loso_conformal_cov90": cal_cov,
                    "loso_conformal_width90": cal_width,
                    "width90_inflation": float(cal_width / max(raw_width, 1e-12)),
                }
            )
    by_system_cols = [
        "split_kind",
        "heldout_system",
        "budget_kind",
        "budget_label",
        "budget_value",
        "n_points",
        "n_cal_points",
        "raw_cov90",
        "raw_width90",
        "loso_conformal_scale90",
        "loso_conformal_cov90",
        "loso_conformal_width90",
        "width90_inflation",
    ]
    by_budget_cols = [
        "split_kind",
        "budget_kind",
        "budget_label",
        "budget_value",
        "n_systems",
        "total_points",
        "median_raw_cov90",
        "median_conformal_cov90",
        "median_scale90",
        "median_width90_inflation",
        "mean_raw_width90",
        "mean_conformal_width90",
    ]
    if not rows:
        return pd.DataFrame(columns=by_system_cols), pd.DataFrame(columns=by_budget_cols)

    by_system = pd.DataFrame(rows, columns=by_system_cols).sort_values(["budget_value", "heldout_system"])
    by_budget = (
        by_system.groupby(["split_kind", "budget_kind", "budget_label", "budget_value"], dropna=False)
        .agg(
            n_systems=("heldout_system", "nunique"),
            total_points=("n_points", "sum"),
            median_raw_cov90=("raw_cov90", "median"),
            median_conformal_cov90=("loso_conformal_cov90", "median"),
            median_scale90=("loso_conformal_scale90", "median"),
            median_width90_inflation=("width90_inflation", "median"),
            mean_raw_width90=("raw_width90", "mean"),
            mean_conformal_width90=("loso_conformal_width90", "mean"),
        )
        .reset_index()
        .sort_values("budget_value")
    )
    return by_system, by_budget


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def write_summary(
    out: Path,
    run: Path,
    checks: list[dict[str, Any]],
    named_best: pd.DataFrame,
    failure_by_system: pd.DataFrame,
    conformal_by_budget: pd.DataFrame,
) -> None:
    latest_named = named_best.sort_values("budget_value").groupby("heldout_system", as_index=False).tail(1)
    latest_named = latest_named.sort_values("delta_model_minus_best_named", ascending=False)
    latest_oracle = failure_by_system.sort_values("budget_value").groupby("heldout_system", as_index=False).tail(1)
    latest_oracle = latest_oracle.sort_values("median_delta_model_minus_best", ascending=False)
    lines = [
        "# Release Transfer Failure + Calibration Diagnostics",
        "",
        f"- Source run: `{run}`",
        "- Scope: diagnostic only; no new model training.",
        "- `named_best` compares against the single best named baseline per system/budget.",
        "- `oracle_best` compares against the best baseline chosen separately for each curve; it is a harsh envelope, not a deployable competitor.",
        "- Conformal scaling uses other LOSO held-out systems as calibration data, so report it as a stress test rather than deployment-valid uncertainty.",
        "",
        "## Named-Baseline Systems At Latest Budget",
        "",
        markdown_table(
            latest_named[
                [
                    "heldout_system",
                    "budget_label",
                    "n_curves",
                    "median_model_rmse",
                    "best_named_baseline",
                    "median_baseline_rmse",
                    "delta_model_minus_best_named",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Oracle-Envelope Failure Modes At Latest Budget",
        "",
        markdown_table(
            latest_oracle[
                [
                    "heldout_system",
                    "budget_label",
                    "n_curves",
                    "median_model_rmse",
                    "median_best_baseline_rmse",
                    "median_delta_model_minus_best",
                    "fraction_model_wins",
                    "top_failure_mode",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## LOSO Conformal Interval Diagnostic",
        "",
        markdown_table(conformal_by_budget, max_rows=20),
        "",
        "## Integrity Checks",
        "",
        markdown_table(pd.DataFrame(checks), max_rows=20),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_curve = read_csv(args.run / "per_curve_metrics.csv")
    baselines = read_csv(args.run / "baseline_per_curve_metrics.csv")
    predictions = read_csv(args.run / "predictions_long.csv")

    failure, failure_by_system = build_failure_tables(
        per_curve,
        baselines,
        predictions,
        split_kind=args.split_kind,
        budget_kind=args.budget_kind,
        near_tie_eps=float(args.near_tie_eps),
    )
    named_best = build_named_best_table(
        per_curve,
        baselines,
        split_kind=args.split_kind,
        budget_kind=args.budget_kind,
    )
    conformal_by_system, conformal_by_budget = build_conformal_tables(
        predictions,
        split_kind=args.split_kind,
        budget_kind=args.budget_kind,
    )

    checks = [
        finite_numeric_check("failure_per_curve", failure),
        finite_numeric_check("failure_by_system_budget", failure_by_system),
        finite_numeric_check("named_best_by_system_budget", named_best),
        finite_numeric_check("conformal_by_system_budget", conformal_by_system),
        finite_numeric_check("conformal_by_budget", conformal_by_budget),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")

    failure.to_csv(args.out / "failure_per_curve.csv", index=False)
    failure_by_system.to_csv(args.out / "failure_by_system_budget.csv", index=False)
    named_best.to_csv(args.out / "named_best_by_system_budget.csv", index=False)
    conformal_by_system.to_csv(args.out / "conformal_by_system_budget.csv", index=False)
    conformal_by_budget.to_csv(args.out / "conformal_by_budget.csv", index=False)
    metadata = {
        "script": "scripts/83_release_transfer_failure_calibration.py",
        "run": str(args.run),
        "out": str(args.out),
        "split_kind": str(args.split_kind),
        "budget_kind": str(args.budget_kind),
        "near_tie_eps": float(args.near_tie_eps),
        "note": "Diagnostic only. LOSO conformal scaling uses other held-out systems as calibration data.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out, args.run, checks, named_best, failure_by_system, conformal_by_budget)

    print(f"[83] wrote {args.out}")
    print(failure_by_system.sort_values(['budget_value', 'median_delta_model_minus_best'], ascending=[True, False]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
