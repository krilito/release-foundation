"""84 - Data-regime audit for release transfer negative cases.

Purpose:
    Explain when the sparse-observation transfer probe is being judged on
    genuine future dynamics versus short, flat, or saturated residual windows.

Consumes:
    outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored/
    outputs/83_release_transfer_failure_calibration_shapeanchored/

Produces:
    outputs/84_release_transfer_data_regime_audit/
      data_regime_per_curve.csv
      data_regime_by_system_budget.csv
      informative_budget_review.csv
      negative_case_review.csv
      summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN = Path("outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored")
DEFAULT_FAILURE = Path("outputs/83_release_transfer_failure_calibration_shapeanchored")
DEFAULT_OUT = Path("outputs/84_release_transfer_data_regime_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data regimes behind transfer negative cases.")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--failure-dir", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-kind", default="loso-system")
    parser.add_argument("--budget-kind", default="time_days", choices=["time_days", "point"])
    parser.add_argument("--flat-span-threshold", type=float, default=0.03)
    parser.add_argument("--saturation-threshold", type=float, default=0.90)
    parser.add_argument("--short-target-max-points", type=int, default=2)
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


def build_per_curve(
    predictions: pd.DataFrame,
    per_curve_metrics: pd.DataFrame,
    named_best: pd.DataFrame,
    *,
    split_kind: str,
    budget_kind: str,
    flat_span_threshold: float,
    saturation_threshold: float,
    short_target_max_points: int,
) -> pd.DataFrame:
    key_cols = ["split_kind", "heldout_system", "unified_curve_id", "budget_kind", "budget_label", "budget_value"]
    pred = predictions[(predictions["split_kind"] == split_kind) & (predictions["budget_kind"] == budget_kind)].copy()
    model = per_curve_metrics[(per_curve_metrics["split_kind"] == split_kind) & (per_curve_metrics["budget_kind"] == budget_kind)].copy()
    named = named_best[(named_best["split_kind"] == split_kind) & (named_best["budget_kind"] == budget_kind)].copy()

    pred = pred.sort_values(key_cols + ["time_days", "target_point_idx"])
    agg = (
        pred.groupby(key_cols, dropna=False)
        .agg(
            n_target_points=("y_true", "size"),
            target_time_min=("time_days", "min"),
            target_time_max=("time_days", "max"),
            target_duration_days=("time_days", lambda s: float(np.max(s) - np.min(s))),
            target_true_start=("y_true", "first"),
            target_true_end=("y_true", "last"),
            target_true_min=("y_true", "min"),
            target_true_max=("y_true", "max"),
            target_pred_start=("y_pred_mean", "first"),
            target_pred_end=("y_pred_mean", "last"),
            target_abs_error_end=("y_true", "size"),
        )
        .reset_index()
    )
    end_rows = pred.groupby(key_cols, as_index=False).tail(1)
    agg = agg.drop(columns=["target_abs_error_end"]).merge(
        end_rows[key_cols + ["y_true", "y_pred_mean"]].rename(
            columns={"y_true": "target_true_last_point", "y_pred_mean": "target_pred_last_point"}
        ),
        on=key_cols,
        how="left",
    )
    agg["target_abs_error_end"] = (agg["target_pred_last_point"] - agg["target_true_last_point"]).abs()
    agg["target_true_span"] = agg["target_true_max"] - agg["target_true_min"]
    agg["target_release_gain"] = agg["target_true_end"] - agg["target_true_start"]
    agg["flat_future"] = agg["target_true_span"] <= flat_span_threshold
    agg["single_or_short_target"] = agg["n_target_points"] <= short_target_max_points
    agg["future_starts_saturated"] = agg["target_true_start"] >= saturation_threshold
    agg["future_ends_saturated"] = agg["target_true_end"] >= saturation_threshold
    agg["future_any_saturated"] = agg["future_ends_saturated"] | agg["future_starts_saturated"]
    agg["hard_dynamic_future"] = (~agg["flat_future"]) & (~agg["single_or_short_target"])

    model_cols = key_cols + ["n_obs", "n_target", "rmse", "mae", "cov90", "width90"]
    out = agg.merge(model[model_cols].rename(columns={"rmse": "model_rmse", "mae": "model_mae"}), on=key_cols, how="left")
    named_cols = [
        "split_kind",
        "heldout_system",
        "budget_kind",
        "budget_label",
        "budget_value",
        "best_named_baseline",
        "median_baseline_rmse",
        "delta_model_minus_best_named",
        "best_named_baseline_family",
    ]
    out = out.merge(named[named_cols], on=["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], how="left")
    out["curve_regime"] = np.select(
        [
            out["single_or_short_target"],
            out["flat_future"] & out["future_any_saturated"],
            out["flat_future"],
            out["future_any_saturated"],
            out["hard_dynamic_future"],
        ],
        [
            "short_target_window",
            "flat_saturated_future",
            "flat_future",
            "saturated_but_dynamic_future",
            "dynamic_future",
        ],
        default="other",
    )
    return out.sort_values(["budget_value", "heldout_system", "model_rmse"], ascending=[True, True, False])


def build_by_system(per_curve: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_curve.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_n_obs=("n_obs", "median"),
            median_n_target=("n_target_points", "median"),
            frac_short_target=("single_or_short_target", "mean"),
            frac_flat_future=("flat_future", "mean"),
            frac_saturated_future=("future_any_saturated", "mean"),
            frac_dynamic_future=("hard_dynamic_future", "mean"),
            median_target_span=("target_true_span", "median"),
            median_target_gain=("target_release_gain", "median"),
            median_target_duration_days=("target_duration_days", "median"),
            median_model_rmse=("model_rmse", "median"),
            best_named_baseline=("best_named_baseline", "first"),
            median_named_baseline_rmse=("median_baseline_rmse", "first"),
            delta_model_minus_best_named=("delta_model_minus_best_named", "first"),
        )
        .reset_index()
    )
    regime_counts = (
        per_curve.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value", "curve_regime"], dropna=False)
        .size()
        .reset_index(name="n_regime")
        .sort_values(["split_kind", "heldout_system", "budget_value", "n_regime"], ascending=[True, True, True, False])
    )
    top = regime_counts.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"], as_index=False).first()
    top = top.rename(columns={"curve_regime": "dominant_curve_regime"})
    grouped = grouped.merge(
        top[["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value", "dominant_curve_regime", "n_regime"]],
        on=["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"],
        how="left",
    )
    return grouped.sort_values(["budget_value", "delta_model_minus_best_named"], ascending=[True, False])


def build_negative_review(by_system: pd.DataFrame) -> pd.DataFrame:
    latest = by_system.sort_values("budget_value").groupby("heldout_system", as_index=False).tail(1).copy()
    latest = latest.sort_values("delta_model_minus_best_named", ascending=False)
    rows = []
    for row in latest.itertuples(index=False):
        if float(row.delta_model_minus_best_named) <= 0:
            verdict = "model_wins_or_ties_named_best"
        elif float(row.frac_short_target) >= 0.5:
            verdict = "negative_case_mostly_short_target"
        elif float(row.frac_flat_future) >= 0.5:
            verdict = "negative_case_mostly_flat_future"
        elif float(row.frac_saturated_future) >= 0.5:
            verdict = "negative_case_saturated_future"
        elif float(row.frac_dynamic_future) >= 0.5:
            verdict = "negative_case_dynamic_forecasting_failure"
        else:
            verdict = "negative_case_mixed_regime"
        item = row._asdict()
        item["review_verdict"] = verdict
        rows.append(item)
    return pd.DataFrame(rows)


def build_informative_budget_review(by_system: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, sub in by_system.groupby("heldout_system", sort=True):
        sub = sub.sort_values("budget_value")
        informative = sub[(sub["median_n_target"] >= 3) & (sub["frac_dynamic_future"] >= 0.5)].copy()
        if len(informative):
            chosen = informative.iloc[-1].copy()
            selection_rule = "latest_budget_with_dynamic_future"
        else:
            ranked = sub.sort_values(["frac_dynamic_future", "median_n_target", "budget_value"], ascending=[False, False, False])
            chosen = ranked.iloc[0].copy()
            selection_rule = "no_budget_has_majority_dynamic_future"
        item = chosen.to_dict()
        item["selection_rule"] = selection_rule
        rows.append(item)
    return pd.DataFrame(rows).sort_values("delta_model_minus_best_named", ascending=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def write_summary(
    out: Path,
    run: Path,
    failure_dir: Path,
    by_system: pd.DataFrame,
    negative: pd.DataFrame,
    informative: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    latest = by_system.sort_values("budget_value").groupby("heldout_system", as_index=False).tail(1)
    latest = latest.sort_values("delta_model_minus_best_named", ascending=False)
    lines = [
        "# Release Transfer Data-Regime Audit",
        "",
        f"- Source run: `{run}`",
        f"- Failure diagnostics: `{failure_dir}`",
        "- Purpose: distinguish real dynamic forecasting failures from short, flat, or saturated target windows.",
        "",
        "## Latest-Budget Regimes",
        "",
        markdown_table(
            latest[
                [
                    "heldout_system",
                    "budget_label",
                    "n_curves",
                    "frac_short_target",
                    "frac_flat_future",
                    "frac_saturated_future",
                    "frac_dynamic_future",
                    "dominant_curve_regime",
                    "delta_model_minus_best_named",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Negative-Case Review",
        "",
        markdown_table(
            negative[
                [
                    "heldout_system",
                    "budget_label",
                    "best_named_baseline",
                    "delta_model_minus_best_named",
                    "dominant_curve_regime",
                    "review_verdict",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Informative Budget Review",
        "",
        "For each system, this chooses the latest budget with at least three median target points and a majority of dynamic futures. If none exists, it chooses the least bad budget by dynamic-future fraction.",
        "",
        markdown_table(
            informative[
                [
                    "heldout_system",
                    "budget_label",
                    "selection_rule",
                    "median_n_target",
                    "frac_dynamic_future",
                    "dominant_curve_regime",
                    "best_named_baseline",
                    "delta_model_minus_best_named",
                ]
            ],
            max_rows=12,
        ),
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
    predictions = read_csv(args.run / "predictions_long.csv")
    per_curve_metrics = read_csv(args.run / "per_curve_metrics.csv")
    named_best = read_csv(args.failure_dir / "named_best_by_system_budget.csv")

    per_curve = build_per_curve(
        predictions,
        per_curve_metrics,
        named_best,
        split_kind=str(args.split_kind),
        budget_kind=str(args.budget_kind),
        flat_span_threshold=float(args.flat_span_threshold),
        saturation_threshold=float(args.saturation_threshold),
        short_target_max_points=int(args.short_target_max_points),
    )
    by_system = build_by_system(per_curve)
    negative = build_negative_review(by_system)
    informative = build_informative_budget_review(by_system)

    checks = [
        finite_numeric_check("data_regime_per_curve", per_curve),
        finite_numeric_check("data_regime_by_system_budget", by_system),
        finite_numeric_check("negative_case_review", negative),
        finite_numeric_check("informative_budget_review", informative),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")

    per_curve.to_csv(args.out / "data_regime_per_curve.csv", index=False)
    by_system.to_csv(args.out / "data_regime_by_system_budget.csv", index=False)
    informative.to_csv(args.out / "informative_budget_review.csv", index=False)
    negative.to_csv(args.out / "negative_case_review.csv", index=False)
    metadata = {
        "script": "scripts/84_release_transfer_data_regime_audit.py",
        "run": str(args.run),
        "failure_dir": str(args.failure_dir),
        "out": str(args.out),
        "split_kind": str(args.split_kind),
        "budget_kind": str(args.budget_kind),
        "flat_span_threshold": float(args.flat_span_threshold),
        "saturation_threshold": float(args.saturation_threshold),
        "short_target_max_points": int(args.short_target_max_points),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out, args.run, args.failure_dir, by_system, negative, informative, checks)
    print(f"[84] wrote {args.out}")
    print(negative.to_string(index=False))


if __name__ == "__main__":
    main()
