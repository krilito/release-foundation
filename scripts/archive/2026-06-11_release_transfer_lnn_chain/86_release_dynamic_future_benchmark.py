"""86 - Dynamic-future masked benchmark for release transfer.

Purpose:
    Recompute the transfer-vs-baseline comparison only on curve/budget rows
    where the future target window still contains meaningful dynamics.

Consumes:
    outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored/
    outputs/84_release_transfer_data_regime_audit_shapeanchored/

Produces:
    outputs/86_release_dynamic_future_benchmark/
      dynamic_curve_rows.csv
      dynamic_by_system_budget.csv
      dynamic_by_budget.csv
      informative_dynamic_review.csv
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
DEFAULT_REGIME = Path("outputs/84_release_transfer_data_regime_audit_shapeanchored")
DEFAULT_OUT = Path("outputs/86_release_dynamic_future_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic-future masked release-transfer benchmark.")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--regime-dir", type=Path, default=DEFAULT_REGIME)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-kind", default="loso-system")
    parser.add_argument("--budget-kind", default="time_days", choices=["time_days", "point"])
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(values.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def build_dynamic_curve_rows(
    regime: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    split_kind: str,
    budget_kind: str,
) -> pd.DataFrame:
    key_cols = ["split_kind", "heldout_system", "unified_curve_id", "budget_kind", "budget_label", "budget_value"]
    reg = regime[(regime["split_kind"] == split_kind) & (regime["budget_kind"] == budget_kind)].copy()
    reg["hard_dynamic_future"] = bool_series(reg["hard_dynamic_future"])
    dyn = reg[reg["hard_dynamic_future"]].copy()
    if dyn.empty:
        return pd.DataFrame()

    base = baselines[(baselines["split_kind"] == split_kind) & (baselines["budget_kind"] == budget_kind)].copy()
    joined = dyn[key_cols + ["model_rmse", "model_mae", "n_obs", "n_target", "target_true_span", "target_release_gain", "curve_regime"]].merge(
        base[key_cols + ["baseline", "rmse"]],
        on=key_cols,
        how="inner",
    )
    joined = joined.rename(columns={"rmse": "baseline_rmse"})
    joined["delta_model_minus_baseline"] = joined["model_rmse"] - joined["baseline_rmse"]
    return joined.sort_values(["budget_value", "heldout_system", "unified_curve_id", "baseline"])


def summarize_named_best(dynamic_rows: pd.DataFrame, total_regime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dynamic_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"]
    model = (
        dynamic_rows.groupby(group_cols, dropna=False)
        .agg(
            n_dynamic_curves=("unified_curve_id", "nunique"),
            median_n_obs=("n_obs", "median"),
            median_n_target=("n_target", "median"),
            median_model_rmse=("model_rmse", "median"),
            mean_model_rmse=("model_rmse", "mean"),
            median_target_span=("target_true_span", "median"),
        )
        .reset_index()
    )
    base = (
        dynamic_rows.groupby(group_cols + ["baseline"], dropna=False)
        .agg(
            n_baseline_curves=("unified_curve_id", "nunique"),
            median_baseline_rmse=("baseline_rmse", "median"),
            mean_baseline_rmse=("baseline_rmse", "mean"),
        )
        .reset_index()
    )
    best = (
        base.sort_values(group_cols + ["median_baseline_rmse", "baseline"])
        .groupby(group_cols, as_index=False)
        .first()
        .rename(columns={"baseline": "best_dynamic_baseline"})
    )
    total = (
        total_regime.groupby(group_cols, dropna=False)
        .agg(n_total_curves=("unified_curve_id", "nunique"))
        .reset_index()
    )
    by_system = model.merge(best, on=group_cols, how="left").merge(total, on=group_cols, how="left")
    by_system["frac_curves_dynamic"] = by_system["n_dynamic_curves"] / by_system["n_total_curves"].clip(lower=1)
    by_system["delta_model_minus_best_dynamic"] = by_system["median_model_rmse"] - by_system["median_baseline_rmse"]
    by_system = by_system.sort_values(["budget_value", "delta_model_minus_best_dynamic"], ascending=[True, False])

    agg_group_cols = ["split_kind", "budget_kind", "budget_label", "budget_value"]
    agg_model = (
        dynamic_rows.groupby(agg_group_cols, dropna=False)
        .agg(
            n_dynamic_curves=("unified_curve_id", "nunique"),
            n_systems=("heldout_system", "nunique"),
            median_model_rmse=("model_rmse", "median"),
            mean_model_rmse=("model_rmse", "mean"),
            median_target_span=("target_true_span", "median"),
        )
        .reset_index()
    )
    agg_base = (
        dynamic_rows.groupby(agg_group_cols + ["baseline"], dropna=False)
        .agg(
            n_baseline_curves=("unified_curve_id", "nunique"),
            median_baseline_rmse=("baseline_rmse", "median"),
            mean_baseline_rmse=("baseline_rmse", "mean"),
        )
        .reset_index()
    )
    agg_best = (
        agg_base.sort_values(agg_group_cols + ["median_baseline_rmse", "baseline"])
        .groupby(agg_group_cols, as_index=False)
        .first()
        .rename(columns={"baseline": "best_dynamic_baseline"})
    )
    by_budget = agg_model.merge(agg_best, on=agg_group_cols, how="left")
    by_budget["delta_model_minus_best_dynamic"] = by_budget["median_model_rmse"] - by_budget["median_baseline_rmse"]
    by_budget = by_budget.sort_values("budget_value")
    return by_system, by_budget


def build_informative_review(by_system: pd.DataFrame, informative: pd.DataFrame) -> pd.DataFrame:
    if by_system.empty:
        return pd.DataFrame()
    keys = ["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value"]
    info = informative[keys + ["selection_rule"]].copy()
    review = info.merge(by_system, on=keys, how="left")
    review["has_dynamic_rows_at_informative_budget"] = review["n_dynamic_curves"].fillna(0).gt(0)
    numeric_fill = [
        "n_dynamic_curves",
        "median_n_obs",
        "median_n_target",
        "median_model_rmse",
        "mean_model_rmse",
        "median_target_span",
        "n_baseline_curves",
        "median_baseline_rmse",
        "mean_baseline_rmse",
        "n_total_curves",
        "frac_curves_dynamic",
        "delta_model_minus_best_dynamic",
    ]
    for col in numeric_fill:
        if col in review.columns:
            review[col] = review[col].fillna(0.0)
    for col in ["best_dynamic_baseline"]:
        if col in review.columns:
            review[col] = review[col].fillna("NO_DYNAMIC_ROWS")
    return review.sort_values("delta_model_minus_best_dynamic", ascending=False, na_position="last")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def write_summary(
    out: Path,
    run: Path,
    regime_dir: Path,
    by_system: pd.DataFrame,
    by_budget: pd.DataFrame,
    informative: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    latest = by_system.sort_values("budget_value").groupby("heldout_system", as_index=False).tail(1)
    latest = latest.sort_values("delta_model_minus_best_dynamic", ascending=False)
    lines = [
        "# Dynamic-Future Masked Release Transfer Benchmark",
        "",
        f"- Source run: `{run}`",
        f"- Regime audit: `{regime_dir}`",
        "- Includes only curve/budget rows where `hard_dynamic_future == True`.",
        "",
        "## Aggregate Dynamic-Future Curve",
        "",
        markdown_table(by_budget, max_rows=20),
        "",
        "## Latest Dynamic Budget Per System",
        "",
        markdown_table(
            latest[
                [
                    "heldout_system",
                    "budget_label",
                    "n_dynamic_curves",
                    "frac_curves_dynamic",
                    "median_model_rmse",
                    "best_dynamic_baseline",
                    "median_baseline_rmse",
                    "delta_model_minus_best_dynamic",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Informative-Budget Dynamic Review",
        "",
        markdown_table(
            informative[
                [
                    "heldout_system",
                    "budget_label",
                    "selection_rule",
                    "has_dynamic_rows_at_informative_budget",
                    "n_dynamic_curves",
                    "median_model_rmse",
                    "best_dynamic_baseline",
                    "median_baseline_rmse",
                    "delta_model_minus_best_dynamic",
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
    baselines = read_csv(args.run / "baseline_per_curve_metrics.csv")
    regime = read_csv(args.regime_dir / "data_regime_per_curve.csv")
    informative = read_csv(args.regime_dir / "informative_budget_review.csv")

    dynamic_rows = build_dynamic_curve_rows(regime, baselines, split_kind=args.split_kind, budget_kind=args.budget_kind)
    by_system, by_budget = summarize_named_best(dynamic_rows, regime)
    informative_review = build_informative_review(by_system, informative)

    checks = [
        finite_numeric_check("dynamic_curve_rows", dynamic_rows),
        finite_numeric_check("dynamic_by_system_budget", by_system),
        finite_numeric_check("dynamic_by_budget", by_budget),
        finite_numeric_check("informative_dynamic_review", informative_review),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")

    dynamic_rows.to_csv(args.out / "dynamic_curve_rows.csv", index=False)
    by_system.to_csv(args.out / "dynamic_by_system_budget.csv", index=False)
    by_budget.to_csv(args.out / "dynamic_by_budget.csv", index=False)
    informative_review.to_csv(args.out / "informative_dynamic_review.csv", index=False)
    metadata = {
        "script": "scripts/86_release_dynamic_future_benchmark.py",
        "run": str(args.run),
        "regime_dir": str(args.regime_dir),
        "out": str(args.out),
        "split_kind": str(args.split_kind),
        "budget_kind": str(args.budget_kind),
        "n_dynamic_rows": int(len(dynamic_rows)),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out, args.run, args.regime_dir, by_system, by_budget, informative_review, checks)
    print(f"[86] wrote {args.out}")
    print(by_budget.to_string(index=False))


if __name__ == "__main__":
    main()
