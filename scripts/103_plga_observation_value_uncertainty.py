"""103 - PLGA observation-budget value package.

First implemented experiment:
    E1. Window-matched observation-budget curve.

Consumes:
    outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv
    outputs/99_plga_static_to_early_proxy_probe/per_curve_metrics.csv

Produces:
    outputs/103_plga_observation_value_uncertainty/
      budget_curve.csv
      budget_contrasts.csv
      budget_contrast_bootstrap_ci.csv
      data_checks.csv
      lock_metadata.json
      budget_curve.md

Interpretation:
    This script does not introduce a new model. It converts the committed
    direct-control/proxy outputs into paired, window-matched observation-budget
    evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DIRECT = Path("outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv")
DEFAULT_PROXY = Path("outputs/99_plga_static_to_early_proxy_probe/per_curve_metrics.csv")
DEFAULT_OUT = Path("outputs/103_plga_observation_value_uncertainty")

STRICT_SPLITS = ["source-group-kfold", "source-dataset-lodo"]
ALL_SPLITS = ["random-kfold", *STRICT_SPLITS]
DEFAULT_BUDGETS = [0, 1, 2, 3, 5]

DIRECT_METHOD_MAP = {
    "direct_static_time": "static_time",
    "direct_early_time": "measured_early_time",
    "direct_static_early_time": "measured_static_early_time",
}

PROXY_METHOD_MAP = {
    "static_time": "proxy_run_static_time",
    "true_early_time": "proxy_run_measured_early_time",
    "pred_early_time": "proxy_run_predicted_early_time",
    "true_static_early_time": "proxy_run_measured_static_early_time",
    "pred_static_early_time": "proxy_run_predicted_static_early_time",
}

CONTRASTS = [
    {
        "contrast_id": "static_to_measured_static_early",
        "source_run": "97_direct",
        "baseline_raw_method": "direct_static_time",
        "comparison_raw_method": "direct_static_early_time",
        "baseline_method": "static_time",
        "comparison_method": "measured_static_early_time",
        "claim_role": "value of measured early Q beyond static descriptors",
    },
    {
        "contrast_id": "static_to_measured_early_only",
        "source_run": "97_direct",
        "baseline_raw_method": "direct_static_time",
        "comparison_raw_method": "direct_early_time",
        "baseline_method": "static_time",
        "comparison_method": "measured_early_time",
        "claim_role": "value of measured early Q without static descriptors",
    },
    {
        "contrast_id": "static_predicted_to_measured_static_early",
        "source_run": "99_proxy",
        "baseline_raw_method": "pred_static_early_time",
        "comparison_raw_method": "true_static_early_time",
        "baseline_method": "proxy_run_predicted_static_early_time",
        "comparison_method": "proxy_run_measured_static_early_time",
        "claim_role": "measured early state vs static-synthesized early state",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PLGA E1 observation-budget curves.")
    parser.add_argument("--direct-per-curve", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--proxy-per-curve", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
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


def parse_budgets(text: str) -> list[int]:
    budgets = sorted({int(part.strip()) for part in str(text).split(",") if part.strip()})
    if not budgets:
        raise ValueError("--budgets produced an empty list")
    return budgets


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, path: Path, cols: list[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def safe_r2(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def load_direct(path: Path, budgets: list[int]) -> pd.DataFrame:
    df = read_csv(path)
    require_columns(
        df,
        path,
        [
            "split_kind",
            "fold",
            "budget",
            "method",
            "unified_curve_id",
            "future_rmse",
            "future_mae",
            "future_r2",
            "n_future",
        ],
    )
    out = df[df["method"].astype(str).isin(DIRECT_METHOD_MAP)].copy()
    out = out[out["budget"].astype(int).isin(budgets)].copy()
    out["source_run"] = "97_direct"
    out["normalized_method"] = out["method"].map(DIRECT_METHOD_MAP)
    return normalize_numeric(out)


def load_proxy(path: Path, budgets: list[int]) -> pd.DataFrame:
    df = read_csv(path)
    require_columns(
        df,
        path,
        [
            "split_kind",
            "fold",
            "budget",
            "method",
            "unified_curve_id",
            "future_rmse",
            "future_mae",
            "future_r2",
            "n_future",
        ],
    )
    out = df[df["method"].astype(str).isin(PROXY_METHOD_MAP)].copy()
    out = out[out["budget"].astype(int).isin([b for b in budgets if b > 0])].copy()
    out["source_run"] = "99_proxy"
    out["normalized_method"] = out["method"].map(PROXY_METHOD_MAP)
    return normalize_numeric(out)


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["budget"] = pd.to_numeric(out["budget"], errors="raise").astype(int)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    for col in ["future_rmse", "future_mae", "future_r2", "n_future", "last_context_time"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["unified_curve_id"] = out["unified_curve_id"].astype(str)
    out["split_kind"] = out["split_kind"].astype(str)
    out["method"] = out["method"].astype(str)
    return out


def build_budget_curve(per_curve: pd.DataFrame) -> pd.DataFrame:
    if per_curve.empty:
        return pd.DataFrame()
    grouped = (
        per_curve.groupby(["source_run", "split_kind", "budget", "normalized_method", "method"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            n_rows=("unified_curve_id", "size"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_mae=("future_mae", "median"),
            mean_future_mae=("future_mae", "mean"),
            median_future_r2=("future_r2", safe_r2),
            frac_r2_positive=("future_r2", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.0))),
            median_context_time=("last_context_time", "median"),
            median_n_future=("n_future", "median"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget", "source_run", "median_future_rmse", "normalized_method"])
    )
    grouped["headline_split"] = grouped["split_kind"].isin(STRICT_SPLITS)
    return grouped


def pair_methods(per_curve: pd.DataFrame, contrast: dict[str, str], split: str, budget: int) -> pd.DataFrame:
    sub = per_curve[
        (per_curve["source_run"] == contrast["source_run"])
        & (per_curve["split_kind"] == split)
        & (per_curve["budget"] == budget)
        & (per_curve["method"].isin([contrast["baseline_raw_method"], contrast["comparison_raw_method"]]))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    key_cols = ["split_kind", "fold", "budget", "unified_curve_id"]
    metric_cols = ["future_rmse", "future_mae", "future_r2", "n_future", "last_context_time"]
    wide_parts = []
    for raw_method, suffix in [
        (contrast["baseline_raw_method"], "baseline"),
        (contrast["comparison_raw_method"], "comparison"),
    ]:
        part = sub[sub["method"] == raw_method][key_cols + [c for c in metric_cols if c in sub.columns]].copy()
        rename = {col: f"{col}_{suffix}" for col in metric_cols if col in part.columns}
        wide_parts.append(part.rename(columns=rename))
    if len(wide_parts) != 2:
        return pd.DataFrame()
    paired = wide_parts[0].merge(wide_parts[1], on=key_cols, how="inner")
    paired = paired.dropna(subset=["future_rmse_baseline", "future_rmse_comparison"])
    return paired


def summarize_pair(paired: pd.DataFrame, contrast: dict[str, str]) -> dict[str, Any]:
    b_rmse = paired["future_rmse_baseline"].to_numpy(dtype=float)
    c_rmse = paired["future_rmse_comparison"].to_numpy(dtype=float)
    b_mae = paired["future_mae_baseline"].to_numpy(dtype=float)
    c_mae = paired["future_mae_comparison"].to_numpy(dtype=float)
    rmse_delta = b_rmse - c_rmse
    mae_delta = b_mae - c_mae
    return {
        "contrast_id": contrast["contrast_id"],
        "source_run": contrast["source_run"],
        "split_kind": str(paired["split_kind"].iloc[0]),
        "budget": int(paired["budget"].iloc[0]),
        "baseline_method": contrast["baseline_method"],
        "comparison_method": contrast["comparison_method"],
        "claim_role": contrast["claim_role"],
        "n_pairs": int(len(paired)),
        "baseline_median_rmse": float(np.median(b_rmse)),
        "comparison_median_rmse": float(np.median(c_rmse)),
        "median_rmse_gain": float(np.median(b_rmse) - np.median(c_rmse)),
        "mean_rmse_gain": float(np.mean(rmse_delta)),
        "median_paired_rmse_delta": float(np.median(rmse_delta)),
        "fraction_curves_improved": float(np.mean(rmse_delta > 0.0)),
        "baseline_median_mae": float(np.median(b_mae)),
        "comparison_median_mae": float(np.median(c_mae)),
        "median_mae_gain": float(np.median(b_mae) - np.median(c_mae)),
        "mean_mae_gain": float(np.mean(mae_delta)),
        "median_context_time": float(np.nanmedian(paired.get("last_context_time_comparison", pd.Series(np.nan)))),
        "headline_split": str(paired["split_kind"].iloc[0]) in STRICT_SPLITS,
    }


def bootstrap_pair(
    paired: pd.DataFrame,
    contrast: dict[str, str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    b_rmse = paired["future_rmse_baseline"].to_numpy(dtype=float)
    c_rmse = paired["future_rmse_comparison"].to_numpy(dtype=float)
    b_mae = paired["future_mae_baseline"].to_numpy(dtype=float)
    c_mae = paired["future_mae_comparison"].to_numpy(dtype=float)
    n = len(paired)
    if n < 2:
        raise ValueError("paired bootstrap requires at least two pairs")
    rmse_gains = np.empty(n_bootstrap, dtype=float)
    mean_rmse_gains = np.empty(n_bootstrap, dtype=float)
    mae_gains = np.empty(n_bootstrap, dtype=float)
    improved = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rb = b_rmse[idx]
        rc = c_rmse[idx]
        mb = b_mae[idx]
        mc = c_mae[idx]
        delta = rb - rc
        rmse_gains[i] = np.median(rb) - np.median(rc)
        mean_rmse_gains[i] = np.mean(delta)
        mae_gains[i] = np.median(mb) - np.median(mc)
        improved[i] = np.mean(delta > 0.0)
    return {
        "contrast_id": contrast["contrast_id"],
        "source_run": contrast["source_run"],
        "split_kind": str(paired["split_kind"].iloc[0]),
        "budget": int(paired["budget"].iloc[0]),
        "baseline_method": contrast["baseline_method"],
        "comparison_method": contrast["comparison_method"],
        "n_pairs": int(n),
        "n_bootstrap": int(n_bootstrap),
        "median_rmse_gain_ci_low": float(np.quantile(rmse_gains, 0.025)),
        "median_rmse_gain_ci_high": float(np.quantile(rmse_gains, 0.975)),
        "median_rmse_gain_bootstrap_median": float(np.median(rmse_gains)),
        "mean_rmse_gain_ci_low": float(np.quantile(mean_rmse_gains, 0.025)),
        "mean_rmse_gain_ci_high": float(np.quantile(mean_rmse_gains, 0.975)),
        "median_mae_gain_ci_low": float(np.quantile(mae_gains, 0.025)),
        "median_mae_gain_ci_high": float(np.quantile(mae_gains, 0.975)),
        "fraction_improved_ci_low": float(np.quantile(improved, 0.025)),
        "fraction_improved_ci_high": float(np.quantile(improved, 0.975)),
        "headline_split": str(paired["split_kind"].iloc[0]) in STRICT_SPLITS,
    }


def build_contrasts(
    per_curve: pd.DataFrame,
    budgets: list[int],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    paired_rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for contrast in CONTRASTS:
        contrast_budgets = [b for b in budgets if b > 0 or contrast["source_run"] == "97_direct"]
        for split in ALL_SPLITS:
            for budget in contrast_budgets:
                paired = pair_methods(per_curve, contrast, split, budget)
                if paired.empty:
                    continue
                if len(paired) < 2:
                    continue
                summary = summarize_pair(paired, contrast)
                rows.append(summary)
                boot_rows.append(bootstrap_pair(paired, contrast, n_bootstrap, rng))
                keep = paired.copy()
                keep["contrast_id"] = contrast["contrast_id"]
                keep["source_run"] = contrast["source_run"]
                keep["rmse_delta_baseline_minus_comparison"] = (
                    keep["future_rmse_baseline"] - keep["future_rmse_comparison"]
                )
                keep["mae_delta_baseline_minus_comparison"] = (
                    keep["future_mae_baseline"] - keep["future_mae_comparison"]
                )
                paired_rows.append(keep)
    contrasts = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    paired_all = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    if not contrasts.empty:
        contrasts = contrasts.sort_values(["split_kind", "budget", "contrast_id"]).reset_index(drop=True)
    if not boot.empty:
        boot = boot.sort_values(["split_kind", "budget", "contrast_id"]).reset_index(drop=True)
    return contrasts, boot, paired_all


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    allowed_cols = {"median_context_time", "future_r2", "median_future_r2", "last_context_time"}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].to_numpy(dtype=float)
        bad = ~np.isfinite(vals)
        if not bool(bad.any()):
            continue
        if col in allowed_cols or "_r2" in col or col.endswith("context_time"):
            allowed[col] = int(bad.sum())
        else:
            unexpected[col] = int(bad.sum())
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": unexpected, "allowed_nonfinite": allowed}


def write_report(
    out: Path,
    curve: pd.DataFrame,
    contrasts: pd.DataFrame,
    boot: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    strict_contrasts = contrasts[contrasts["headline_split"]].copy() if not contrasts.empty else pd.DataFrame()
    strict_boot = boot[boot["headline_split"]].copy() if not boot.empty else pd.DataFrame()
    headline_cols = [
        "split_kind",
        "budget",
        "contrast_id",
        "baseline_median_rmse",
        "comparison_median_rmse",
        "median_rmse_gain",
        "fraction_curves_improved",
        "n_pairs",
    ]
    boot_cols = [
        "split_kind",
        "budget",
        "contrast_id",
        "median_rmse_gain_ci_low",
        "median_rmse_gain_ci_high",
        "fraction_improved_ci_low",
        "fraction_improved_ci_high",
    ]
    curve_head = curve[
        curve["split_kind"].isin(STRICT_SPLITS)
        & curve["normalized_method"].isin(
            [
                "static_time",
                "measured_static_early_time",
                "measured_early_time",
                "proxy_run_predicted_static_early_time",
                "proxy_run_measured_static_early_time",
            ]
        )
    ].copy()
    lines = [
        "# PLGA E1 Window-Matched Observation-Budget Curve",
        "",
        "This is the first experiment from `docs/plga_observation_budget_goal_design_2026-06-11.md`.",
        "It converts existing script 97 and 99 outputs into paired observation-budget contrasts.",
        "",
        "## Interpretation Rule",
        "",
        "- Headline evidence is restricted to `source-group-kfold` and `source-dataset-lodo`.",
        "- Random-kfold is retained only as a sanity / upper-bound view.",
        "- Positive RMSE gain means the comparison method is better than the baseline.",
        "- Static-predicted early proxy is evaluated within script 99's matched protocol.",
        "",
        "## Strict-Split Budget Contrasts",
        "",
        strict_contrasts[headline_cols].to_markdown(index=False) if not strict_contrasts.empty else "_No strict contrasts._",
        "",
        "## Bootstrap 95% CIs",
        "",
        strict_boot[boot_cols].to_markdown(index=False) if not strict_boot.empty else "_No bootstrap rows._",
        "",
        "## Strict-Split Budget Curve Rows",
        "",
        curve_head[
            [
                "source_run",
                "split_kind",
                "budget",
                "normalized_method",
                "n_curves",
                "median_future_rmse",
                "mean_future_rmse",
                "median_future_mae",
                "median_context_time",
            ]
        ]
        .sort_values(["split_kind", "budget", "source_run", "normalized_method"])
        .to_markdown(index=False),
        "",
        "## Verification",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Current Limitations",
        "",
        "- This script only implements E1. Uncertainty contraction, timepoint value, and stopping rules remain future steps.",
        "- It summarizes per-curve errors from existing outputs; it does not retrain predictors.",
        "- Bootstrap CIs are paired by heldout curve IDs and do not replace an external prospective validation set.",
        "",
    ]
    (out / "budget_curve.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    args.out.mkdir(parents=True, exist_ok=True)
    direct = load_direct(args.direct_per_curve, budgets)
    proxy = load_proxy(args.proxy_per_curve, budgets)
    per_curve = pd.concat([direct, proxy], ignore_index=True)
    budget_curve = build_budget_curve(per_curve)
    contrasts, boot, paired = build_contrasts(per_curve, budgets, args.n_bootstrap, args.seed)
    checks = [
        finite_check("normalized_per_curve", per_curve),
        finite_check("budget_curve", budget_curve),
        finite_check("budget_contrasts", contrasts),
        finite_check("budget_contrast_bootstrap_ci", boot),
        finite_check("paired_curve_deltas", paired),
    ]
    bad = {check["name"]: check["unexpected_nonfinite"] for check in checks if check["unexpected_nonfinite"]}
    if bad:
        raise RuntimeError(f"Unexpected non-finite values: {bad}")
    budget_curve.to_csv(args.out / "budget_curve.csv", index=False)
    contrasts.to_csv(args.out / "budget_contrasts.csv", index=False)
    boot.to_csv(args.out / "budget_contrast_bootstrap_ci.csv", index=False)
    paired.to_csv(args.out / "paired_curve_deltas.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out / "data_checks.csv", index=False)
    metadata = {
        "script": "scripts/103_plga_observation_value_uncertainty.py",
        "git_hash": git_hash(),
        "args": args_to_metadata(args),
        "budgets": budgets,
        "experiment_status": "E1_window_matched_observation_budget_only",
        "strict_splits": STRICT_SPLITS,
        "inputs": {
            "direct_per_curve": str(args.direct_per_curve),
            "proxy_per_curve": str(args.proxy_per_curve),
        },
        "outputs": [
            "budget_curve.csv",
            "budget_contrasts.csv",
            "budget_contrast_bootstrap_ci.csv",
            "paired_curve_deltas.csv",
            "data_checks.csv",
            "budget_curve.md",
        ],
        "checks": checks,
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out, budget_curve, contrasts, boot, checks)
    print((args.out / "budget_curve.md").resolve())
    strict = contrasts[contrasts["headline_split"]]
    print(strict[["split_kind", "budget", "contrast_id", "median_rmse_gain", "n_pairs"]].to_string(index=False))


if __name__ == "__main__":
    main()
