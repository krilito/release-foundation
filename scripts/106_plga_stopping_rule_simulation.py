"""106 - PLGA stopping-rule simulation.

Experiment:
    E4. Can PLGA release curves stop early with bounded accuracy loss?

Consumes:
    outputs/105_plga_timepoint_value/per_curve_metrics.csv

Produces:
    outputs/106_plga_stopping_rule_simulation/
      stopping_policy_results.csv
      stopping_policy_by_split.csv
      stopping_tradeoff_table.csv
      calibration_by_fold.csv
      marginal_calibration_by_fold.csv
      data_checks.csv
      lock_metadata.json
      stopping_rule_report.md

Interpretation:
    Rules are calibrated from other folds and then applied to heldout curves.
    The stopping decision never uses the heldout curve's future RMSE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PER_CURVE = Path("outputs/105_plga_timepoint_value/per_curve_metrics.csv")
DEFAULT_OUT = Path("outputs/106_plga_stopping_rule_simulation")

STRICT_SPLITS = ["source-group-kfold", "source-dataset-lodo"]
MAX_K = 5
ACCEPT_MAX_MEDIAN_RMSE_DELTA = 0.025
ACCEPT_MAX_COV90_DROP = 0.05
ACCEPT_MIN_MEAN_SAVED_OBS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLGA E4 stopping-rule simulation.")
    parser.add_argument("--per-curve", type=Path, default=DEFAULT_PER_CURVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--accept-max-median-rmse-delta", type=float, default=ACCEPT_MAX_MEDIAN_RMSE_DELTA)
    parser.add_argument("--accept-max-cov90-drop", type=float, default=ACCEPT_MAX_COV90_DROP)
    parser.add_argument("--accept-min-mean-saved-obs", type=float, default=ACCEPT_MIN_MEAN_SAVED_OBS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def args_to_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def read_inputs(path: Path, max_k: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = [
        "split_kind",
        "fold",
        "method",
        "policy_id",
        "policy_kind",
        "unified_curve_id",
        "observed_index",
        "last_context_time",
        "future_rmse",
        "future_mae",
        "future_r2",
        "n_future",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    out = df[
        (df["split_kind"].astype(str).isin(STRICT_SPLITS))
        & (df["method"].astype(str) == "direct_static_context_time")
        & (df["policy_kind"].astype(str).isin(["cumulative", "marginal_ablation"]))
        & (pd.to_numeric(df["observed_index"], errors="coerce") <= max_k)
    ].copy()
    out["split_kind"] = out["split_kind"].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    out["observed_index"] = pd.to_numeric(out["observed_index"], errors="raise").astype(int)
    out["policy_id"] = out["policy_id"].astype(str)
    out["policy_kind"] = out["policy_kind"].astype(str)
    out["unified_curve_id"] = out["unified_curve_id"].astype(str)
    for col in ["last_context_time", "future_rmse", "future_mae", "future_r2", "n_future"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def quantile(values: pd.Series, q: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    return float(np.quantile(vals.to_numpy(dtype=float), q))


def build_calibration(per_curve: pd.DataFrame, max_k: int) -> pd.DataFrame:
    cumulative = per_curve[per_curve["policy_kind"] == "cumulative"].copy()
    rows: list[dict[str, Any]] = []
    for (split_kind, fold), _heldout in cumulative.groupby(["split_kind", "fold"], dropna=False):
        train = cumulative[(cumulative["split_kind"] == split_kind) & (cumulative["fold"] != int(fold))]
        for k in range(1, max_k + 1):
            policy = f"first{k}"
            sub = train[train["policy_id"] == policy]
            if sub.empty:
                continue
            q90 = quantile(sub["future_rmse"], 0.90)
            rows.append(
                {
                    "split_kind": split_kind,
                    "fold": int(fold),
                    "policy_id": policy,
                    "stop_k": k,
                    "n_calibration": int(len(sub)),
                    "median_future_rmse_cal": float(sub["future_rmse"].median()),
                    "mean_future_rmse_cal": float(sub["future_rmse"].mean()),
                    "q90_future_rmse_cal": q90,
                    "width90_cal": 2.0 * q90 if np.isfinite(q90) else np.nan,
                    "median_stop_time_cal": float(sub["last_context_time"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split_kind", "fold", "stop_k"]).reset_index(drop=True)


def build_marginal_calibration(per_curve: pd.DataFrame, max_k: int) -> pd.DataFrame:
    cumulative = per_curve[per_curve["policy_kind"] == "cumulative"].copy()
    ablation = per_curve[per_curve["policy_kind"] == "marginal_ablation"].copy()
    rows: list[dict[str, Any]] = []
    for split_kind in STRICT_SPLITS:
        folds = sorted(cumulative.loc[cumulative["split_kind"] == split_kind, "fold"].dropna().astype(int).unique())
        for fold in folds:
            for added_k in range(2, max_k + 1):
                full = cumulative[
                    (cumulative["split_kind"] == split_kind)
                    & (cumulative["fold"] != fold)
                    & (cumulative["policy_id"] == f"first{added_k}")
                ].copy()
                without = ablation[
                    (ablation["split_kind"] == split_kind)
                    & (ablation["fold"] != fold)
                    & (ablation["policy_id"] == f"first{added_k - 1}_cut{added_k}")
                ].copy()
                if full.empty or without.empty:
                    continue
                key = ["split_kind", "fold", "unified_curve_id"]
                paired = without[key + ["future_rmse", "future_mae"]].merge(
                    full[key + ["future_rmse", "future_mae"]],
                    on=key,
                    suffixes=("_without_added", "_with_added"),
                )
                if paired.empty:
                    continue
                paired["rmse_gain"] = paired["future_rmse_without_added"] - paired["future_rmse_with_added"]
                paired["mae_gain"] = paired["future_mae_without_added"] - paired["future_mae_with_added"]
                rows.append(
                    {
                        "split_kind": split_kind,
                        "fold": int(fold),
                        "added_point_index": int(added_k),
                        "n_calibration_pairs": int(len(paired)),
                        "median_rmse_gain_cal": float(paired["rmse_gain"].median()),
                        "mean_rmse_gain_cal": float(paired["rmse_gain"].mean()),
                        "median_mae_gain_cal": float(paired["mae_gain"].median()),
                        "fraction_improved_cal": float((paired["rmse_gain"] > 0).mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["split_kind", "fold", "added_point_index"]).reset_index(drop=True)


def candidate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "rule_id": "never_stop_k5",
            "rule_family": "never_stop",
            "description": "Always collect the first five observed points.",
        }
    ]
    for threshold in [0.70, 0.65, 0.60]:
        rules.append(
            {
                "rule_id": f"width90_abs_le_{threshold:.2f}",
                "rule_family": "width_abs",
                "width90_threshold": threshold,
                "description": f"Stop at earliest k with calibrated width90 <= {threshold:.2f}.",
            }
        )
    for ratio in [1.15, 1.10, 1.05]:
        rules.append(
            {
                "rule_id": f"width90_ratio_to_k5_le_{ratio:.2f}",
                "rule_family": "width_ratio_to_k5",
                "width90_ratio_threshold": ratio,
                "description": f"Stop at earliest k with calibrated width90 <= {ratio:.2f} * calibrated k5 width90.",
            }
        )
    for epsilon in [0.005, 0.010, 0.015]:
        rules.append(
            {
                "rule_id": f"next_marginal_gain_le_{epsilon:.3f}",
                "rule_family": "next_marginal_gain",
                "marginal_gain_threshold": epsilon,
                "description": f"Stop if calibrated next-point RMSE gain <= {epsilon:.3f}.",
            }
        )
    for epsilon in [0.010, 0.015]:
        rules.append(
            {
                "rule_id": f"last_marginal_gain_le_{epsilon:.3f}",
                "rule_family": "last_marginal_gain",
                "marginal_gain_threshold": epsilon,
                "description": f"After measuring k, stop if calibrated gain from point k <= {epsilon:.3f}.",
            }
        )
    return rules


def stop_from_rule(
    rule: dict[str, Any],
    cal_fold: pd.DataFrame,
    marg_fold: pd.DataFrame,
    max_k: int,
) -> tuple[int, str]:
    if rule["rule_family"] == "never_stop":
        return max_k, "always_collect_max_k"

    width_by_k = {int(row.stop_k): float(row.width90_cal) for row in cal_fold.itertuples(index=False)}
    if len(width_by_k) < max_k:
        return max_k, "missing_width_calibration"

    if rule["rule_family"] == "width_abs":
        threshold = float(rule["width90_threshold"])
        for k in range(1, max_k + 1):
            if width_by_k.get(k, np.inf) <= threshold:
                return k, f"width90_{width_by_k[k]:.3f}_le_{threshold:.3f}"
        return max_k, "no_width_threshold_hit"

    if rule["rule_family"] == "width_ratio_to_k5":
        k5_width = width_by_k.get(max_k, np.nan)
        threshold = float(rule["width90_ratio_threshold"])
        if not np.isfinite(k5_width) or k5_width <= 0:
            return max_k, "missing_k5_width"
        for k in range(1, max_k + 1):
            ratio = width_by_k.get(k, np.inf) / k5_width
            if ratio <= threshold:
                return k, f"width_ratio_{ratio:.3f}_le_{threshold:.3f}"
        return max_k, "no_width_ratio_threshold_hit"

    gain_by_added_k = {
        int(row.added_point_index): float(row.median_rmse_gain_cal) for row in marg_fold.itertuples(index=False)
    }

    if rule["rule_family"] == "next_marginal_gain":
        epsilon = float(rule["marginal_gain_threshold"])
        for k in range(1, max_k):
            next_gain = gain_by_added_k.get(k + 1, np.nan)
            if np.isfinite(next_gain) and next_gain <= epsilon:
                return k, f"next_gain_{next_gain:.3f}_le_{epsilon:.3f}"
        return max_k, "no_next_gain_threshold_hit"

    if rule["rule_family"] == "last_marginal_gain":
        epsilon = float(rule["marginal_gain_threshold"])
        for k in range(2, max_k + 1):
            last_gain = gain_by_added_k.get(k, np.nan)
            if np.isfinite(last_gain) and last_gain <= epsilon:
                return k, f"last_gain_{last_gain:.3f}_le_{epsilon:.3f}"
        return max_k, "no_last_gain_threshold_hit"

    raise KeyError(rule["rule_family"])


def simulate_stopping(
    per_curve: pd.DataFrame,
    calibration: pd.DataFrame,
    marginal_cal: pd.DataFrame,
    max_k: int,
) -> pd.DataFrame:
    cumulative = per_curve[per_curve["policy_kind"] == "cumulative"].copy()
    baseline = cumulative[cumulative["policy_id"] == f"first{max_k}"].copy()
    baseline_key = ["split_kind", "fold", "unified_curve_id"]
    baseline_cols = baseline_key + [
        "future_rmse",
        "future_mae",
        "future_r2",
        "n_future",
        "last_context_time",
    ]
    baseline = baseline[baseline_cols].rename(
        columns={
            "future_rmse": "baseline_k5_future_rmse",
            "future_mae": "baseline_k5_future_mae",
            "future_r2": "baseline_k5_future_r2",
            "n_future": "baseline_k5_n_future",
            "last_context_time": "baseline_k5_stop_time",
        }
    )
    eligible = set(map(tuple, baseline[baseline_key].to_numpy()))
    rows: list[pd.DataFrame] = []
    for (split_kind, fold), heldout in cumulative.groupby(["split_kind", "fold"], dropna=False):
        fold = int(fold)
        cal_fold = calibration[(calibration["split_kind"] == split_kind) & (calibration["fold"] == fold)]
        marg_fold = marginal_cal[(marginal_cal["split_kind"] == split_kind) & (marginal_cal["fold"] == fold)]
        for rule in candidate_rules():
            stop_k, reason = stop_from_rule(rule, cal_fold, marg_fold, max_k)
            stopped = heldout[heldout["policy_id"] == f"first{stop_k}"].copy()
            stopped = stopped[
                stopped.apply(lambda r: (r["split_kind"], int(r["fold"]), r["unified_curve_id"]) in eligible, axis=1)
            ].copy()
            if stopped.empty:
                continue
            cal_row = cal_fold[cal_fold["stop_k"] == stop_k]
            if cal_row.empty:
                continue
            q90 = float(cal_row["q90_future_rmse_cal"].iloc[0])
            width90 = float(cal_row["width90_cal"].iloc[0])
            stopped = stopped.merge(baseline, on=baseline_key, how="inner")
            stopped.insert(0, "rule_id", rule["rule_id"])
            stopped.insert(1, "rule_family", rule["rule_family"])
            stopped["stop_k"] = int(stop_k)
            stopped["stop_reason"] = reason
            stopped["calibrated_q90_future_rmse"] = q90
            stopped["calibrated_width90"] = width90
            stopped["covered90"] = stopped["future_rmse"] <= q90
            stopped["baseline_k5_covered90"] = stopped["baseline_k5_future_rmse"] <= float(
                cal_fold.loc[cal_fold["stop_k"] == max_k, "q90_future_rmse_cal"].iloc[0]
            )
            stopped["saved_observations_vs_k5"] = max_k - int(stop_k)
            stopped["saved_observation_fraction_vs_k5"] = stopped["saved_observations_vs_k5"] / max_k
            stopped["rmse_delta_vs_k5"] = stopped["future_rmse"] - stopped["baseline_k5_future_rmse"]
            stopped["mae_delta_vs_k5"] = stopped["future_mae"] - stopped["baseline_k5_future_mae"]
            stopped["same_future_n_as_k5"] = stopped["n_future"] == stopped["baseline_k5_n_future"]
            rows.append(stopped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_results(
    results: pd.DataFrame,
    accept_max_median_rmse_delta: float,
    accept_max_cov90_drop: float,
    accept_min_mean_saved_obs: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = (
        results.groupby(["split_kind", "rule_id", "rule_family"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            n_rows=("unified_curve_id", "size"),
            mean_stop_k=("stop_k", "mean"),
            median_stop_k=("stop_k", "median"),
            median_stop_time=("last_context_time", "median"),
            mean_saved_observations=("saved_observations_vs_k5", "mean"),
            median_saved_observations=("saved_observations_vs_k5", "median"),
            saved_observation_fraction=("saved_observation_fraction_vs_k5", "mean"),
            stopped_median_rmse=("future_rmse", "median"),
            stopped_mean_rmse=("future_rmse", "mean"),
            baseline_k5_median_rmse=("baseline_k5_future_rmse", "median"),
            median_rmse_delta_vs_k5=("rmse_delta_vs_k5", "median"),
            mean_rmse_delta_vs_k5=("rmse_delta_vs_k5", "mean"),
            stopped_median_mae=("future_mae", "median"),
            baseline_k5_median_mae=("baseline_k5_future_mae", "median"),
            median_mae_delta_vs_k5=("mae_delta_vs_k5", "median"),
            cov90_after_stop=("covered90", "mean"),
            cov90_k5=("baseline_k5_covered90", "mean"),
            median_width90_after_stop=("calibrated_width90", "median"),
            fraction_same_future_n_as_k5=("same_future_n_as_k5", "mean"),
        )
        .reset_index()
    )
    grouped["cov90_delta_vs_k5"] = grouped["cov90_after_stop"] - grouped["cov90_k5"]
    grouped["passes_acceptance"] = (
        (grouped["mean_saved_observations"] >= accept_min_mean_saved_obs)
        & (grouped["median_rmse_delta_vs_k5"] <= accept_max_median_rmse_delta)
        & (grouped["cov90_delta_vs_k5"] >= -accept_max_cov90_drop)
    )
    grouped = grouped.sort_values(
        ["split_kind", "passes_acceptance", "mean_saved_observations", "median_rmse_delta_vs_k5"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    tradeoff = (
        grouped.groupby(["rule_id", "rule_family"], dropna=False)
        .agg(
            n_splits=("split_kind", "nunique"),
            all_strict_splits_pass=("passes_acceptance", "all"),
            any_strict_split_pass=("passes_acceptance", "any"),
            mean_saved_observations=("mean_saved_observations", "mean"),
            max_median_rmse_delta_vs_k5=("median_rmse_delta_vs_k5", "max"),
            min_cov90_delta_vs_k5=("cov90_delta_vs_k5", "min"),
            mean_stop_k=("mean_stop_k", "mean"),
            median_stop_time=("median_stop_time", "median"),
        )
        .reset_index()
        .sort_values(
            ["all_strict_splits_pass", "mean_saved_observations", "max_median_rmse_delta_vs_k5"],
            ascending=[False, False, True],
        )
    )
    return grouped, tradeoff


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    allowed_cols = {
        "future_r2",
        "baseline_k5_future_r2",
        "median_stop_time",
        "median_stop_time_cal",
    }
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        bad = int((~np.isfinite(values)).sum())
        if bad == 0:
            continue
        if col in allowed_cols or "_r2" in col or col.endswith("_time"):
            allowed[col] = bad
        else:
            unexpected[col] = bad
    return {
        "name": name,
        "rows": int(len(df)),
        "unexpected_nonfinite": json.dumps(unexpected, sort_keys=True),
        "allowed_nonfinite": json.dumps(allowed, sort_keys=True),
    }


def format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: format_float(x, 3))
    return view.to_markdown(index=False)


def write_report(
    out: Path,
    by_split: pd.DataFrame,
    tradeoff: pd.DataFrame,
    checks: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    accepted = tradeoff[tradeoff["all_strict_splits_pass"]].copy()
    lines = [
        "# PLGA E4 Stopping-Rule Simulation",
        "",
        "Date: 2026-06-12",
        "",
        "## Scope",
        "",
        "This is E4 from `docs/plga_observation_budget_goal_design_2026-06-11.md`.",
        "It simulates stopping rules on the strict PLGA splits using E3 per-curve outputs.",
        "",
        "Stopping decisions are calibrated from other folds; heldout future RMSE is used only for evaluation.",
        "",
        "## Acceptance Criteria",
        "",
        f"- mean saved observations >= `{args.accept_min_mean_saved_obs:.3f}`",
        f"- median RMSE delta vs k5 <= `{args.accept_max_median_rmse_delta:.3f}`",
        f"- cov90 delta vs k5 >= `-{args.accept_max_cov90_drop:.3f}`",
        "",
        "## Accepted Tradeoffs",
        "",
        markdown_table(
            accepted,
            [
                "rule_id",
                "rule_family",
                "mean_saved_observations",
                "max_median_rmse_delta_vs_k5",
                "min_cov90_delta_vs_k5",
                "mean_stop_k",
                "median_stop_time",
            ],
        ),
        "",
        "## Tradeoff Table",
        "",
        markdown_table(
            tradeoff,
            [
                "rule_id",
                "rule_family",
                "all_strict_splits_pass",
                "mean_saved_observations",
                "max_median_rmse_delta_vs_k5",
                "min_cov90_delta_vs_k5",
                "mean_stop_k",
                "median_stop_time",
            ],
        ),
        "",
        "## By Split",
        "",
        markdown_table(
            by_split,
            [
                "split_kind",
                "rule_id",
                "passes_acceptance",
                "mean_stop_k",
                "mean_saved_observations",
                "stopped_median_rmse",
                "baseline_k5_median_rmse",
                "median_rmse_delta_vs_k5",
                "cov90_after_stop",
                "cov90_k5",
                "cov90_delta_vs_k5",
            ],
        ),
        "",
        "## Data Checks",
        "",
        markdown_table(checks, ["name", "rows", "unexpected_nonfinite", "allowed_nonfinite"]),
        "",
        "## Reading Rule",
        "",
        "- `never_stop_k5` is the conservative reference.",
        "- Width rules stop when other-fold calibrated empirical width is small enough.",
        "- Marginal rules stop when other-fold calibrated added-point gain is below threshold.",
        "- A passing rule is not a clinical protocol; it is evidence that a conservative early-stop policy is feasible.",
        "",
    ]
    (out / "stopping_rule_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_k != MAX_K:
        raise ValueError("This first E4 implementation expects --max-k 5 to match E3 first5 baseline")
    args.out.mkdir(parents=True, exist_ok=True)

    per_curve = read_inputs(args.per_curve, args.max_k)
    calibration = build_calibration(per_curve, args.max_k)
    marginal_cal = build_marginal_calibration(per_curve, args.max_k)
    results = simulate_stopping(per_curve, calibration, marginal_cal, args.max_k)
    by_split, tradeoff = summarize_results(
        results,
        args.accept_max_median_rmse_delta,
        args.accept_max_cov90_drop,
        args.accept_min_mean_saved_obs,
    )
    checks = pd.DataFrame(
        [
            finite_check("input_per_curve_subset", per_curve),
            finite_check("calibration_by_fold", calibration),
            finite_check("marginal_calibration_by_fold", marginal_cal),
            finite_check("stopping_policy_results", results),
            finite_check("stopping_policy_by_split", by_split),
            finite_check("stopping_tradeoff_table", tradeoff),
        ]
    )
    if checks["unexpected_nonfinite"].ne("{}").any():
        raise ValueError(f"Unexpected non-finite values:\n{checks.to_string(index=False)}")

    results.to_csv(args.out / "stopping_policy_results.csv", index=False)
    by_split.to_csv(args.out / "stopping_policy_by_split.csv", index=False)
    tradeoff.to_csv(args.out / "stopping_tradeoff_table.csv", index=False)
    calibration.to_csv(args.out / "calibration_by_fold.csv", index=False)
    marginal_cal.to_csv(args.out / "marginal_calibration_by_fold.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "cli_args": args_to_metadata(args),
        "strict_splits": STRICT_SPLITS,
        "max_k": args.max_k,
        "acceptance_criteria": {
            "mean_saved_observations_min": args.accept_min_mean_saved_obs,
            "median_rmse_delta_vs_k5_max": args.accept_max_median_rmse_delta,
            "cov90_delta_vs_k5_min": -args.accept_max_cov90_drop,
        },
        "decision_leakage_guard": "rules calibrated from other folds; heldout future RMSE used only for evaluation",
        "stable_prediction_rule_status": "not implemented because E3 does not store future prediction trajectories",
        "outputs": [
            "stopping_policy_results.csv",
            "stopping_policy_by_split.csv",
            "stopping_tradeoff_table.csv",
            "calibration_by_fold.csv",
            "marginal_calibration_by_fold.csv",
            "data_checks.csv",
            "lock_metadata.json",
            "stopping_rule_report.md",
        ],
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.out, by_split, tradeoff, checks, args)

    print((args.out / "stopping_rule_report.md").resolve())
    print(
        tradeoff[
            [
                "rule_id",
                "all_strict_splits_pass",
                "mean_saved_observations",
                "max_median_rmse_delta_vs_k5",
                "min_cov90_delta_vs_k5",
                "mean_stop_k",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
