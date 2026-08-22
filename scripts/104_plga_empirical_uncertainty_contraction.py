"""104 - PLGA empirical uncertainty contraction.

Experiment:
    E2. Does measured early Q reduce empirical uncertainty envelopes?

Consumes:
    outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv
    outputs/99_plga_static_to_early_proxy_probe/per_curve_metrics.csv

Produces:
    outputs/104_plga_empirical_uncertainty_contraction/
      uncertainty_by_budget.csv
      uncertainty_contraction.csv
      calibration_summary.csv
      data_checks.csv
      lock_metadata.json
      uncertainty_report.md

Interpretation:
    This script does not claim Bayesian posterior uncertainty. It builds
    fold-jackknife empirical error envelopes from out-of-fold per-curve future
    RMSE. Width is therefore an error-tolerance envelope around point forecasts,
    not a full predictive distribution over future release points.
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
DEFAULT_OUT = Path("outputs/104_plga_empirical_uncertainty_contraction")

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

CONTRACTION_CONTRASTS = [
    {
        "contrast_id": "prior_to_window_matched_static",
        "source_run": "97_direct",
        "baseline_budget": 0,
        "baseline_method": "static_time",
        "comparison_method": "static_time",
        "claim_role": "target-window restriction without measured early Q",
    },
    {
        "contrast_id": "prior_to_measured_static_early",
        "source_run": "97_direct",
        "baseline_budget": 0,
        "baseline_method": "static_time",
        "comparison_method": "measured_static_early_time",
        "claim_role": "total contraction after measured early Q",
    },
    {
        "contrast_id": "window_static_to_measured_static_early",
        "source_run": "97_direct",
        "baseline_budget": None,
        "baseline_method": "static_time",
        "comparison_method": "measured_static_early_time",
        "claim_role": "observation-only contraction at equal future window",
    },
    {
        "contrast_id": "proxy_predicted_to_measured_static_early",
        "source_run": "99_proxy",
        "baseline_budget": None,
        "baseline_method": "proxy_run_predicted_static_early_time",
        "comparison_method": "proxy_run_measured_static_early_time",
        "claim_role": "measured early state vs static-synthesized early state",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PLGA E2 empirical uncertainty contraction.")
    parser.add_argument("--direct-per-curve", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--proxy-per-curve", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
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
    out["normalized_method"] = out["normalized_method"].astype(str)
    out["source_run"] = out["source_run"].astype(str)
    return out


def quantile(values: pd.Series, q: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    return float(np.quantile(vals.to_numpy(dtype=float), q))


def build_fold_jackknife_envelopes(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["source_run", "split_kind", "budget", "normalized_method", "method"]
    keep = per_curve.dropna(subset=["future_rmse"]).copy()
    for keys, group in keep.groupby(group_cols, dropna=False):
        source_run, split_kind, budget, normalized_method, method = keys
        folds = sorted(group["fold"].dropna().astype(int).unique().tolist())
        for fold, fold_df in group.groupby("fold", dropna=False):
            if len(folds) > 1:
                calibration = group[group["fold"].astype(int) != int(fold)]
                calibration_scope = "other_folds"
            else:
                calibration = group[~group["unified_curve_id"].isin(fold_df["unified_curve_id"])]
                calibration_scope = "other_curves_same_fold"
            q50 = quantile(calibration["future_rmse"], 0.50)
            q90 = quantile(calibration["future_rmse"], 0.90)
            q95 = quantile(calibration["future_rmse"], 0.95)
            for row in fold_df.itertuples(index=False):
                rmse = float(row.future_rmse)
                rows.append(
                    {
                        "source_run": source_run,
                        "split_kind": split_kind,
                        "budget": int(budget),
                        "normalized_method": normalized_method,
                        "method": method,
                        "fold": int(fold),
                        "unified_curve_id": row.unified_curve_id,
                        "future_rmse": rmse,
                        "future_mae": float(row.future_mae) if pd.notna(row.future_mae) else np.nan,
                        "future_r2": float(row.future_r2) if pd.notna(row.future_r2) else np.nan,
                        "n_future": float(row.n_future) if pd.notna(row.n_future) else np.nan,
                        "last_context_time": (
                            float(row.last_context_time)
                            if hasattr(row, "last_context_time") and pd.notna(row.last_context_time)
                            else np.nan
                        ),
                        "calibration_scope": calibration_scope,
                        "n_calibration": int(len(calibration)),
                        "q50_error": q50,
                        "q90_error": q90,
                        "q95_error": q95,
                        "width50": 2.0 * q50 if np.isfinite(q50) else np.nan,
                        "width90": 2.0 * q90 if np.isfinite(q90) else np.nan,
                        "width95": 2.0 * q95 if np.isfinite(q95) else np.nan,
                        "covered50": bool(rmse <= q50) if np.isfinite(q50) else False,
                        "covered90": bool(rmse <= q90) if np.isfinite(q90) else False,
                        "covered95": bool(rmse <= q95) if np.isfinite(q95) else False,
                    }
                )
    return pd.DataFrame(rows)


def build_uncertainty_by_budget(envelopes: pd.DataFrame) -> pd.DataFrame:
    if envelopes.empty:
        return pd.DataFrame()
    grouped = (
        envelopes.groupby(["source_run", "split_kind", "budget", "normalized_method", "method"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            n_rows=("unified_curve_id", "size"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_mae=("future_mae", "median"),
            cov50=("covered50", "mean"),
            cov90=("covered90", "mean"),
            cov95=("covered95", "mean"),
            median_width50=("width50", "median"),
            median_width90=("width90", "median"),
            median_width95=("width95", "median"),
            mean_width90=("width90", "mean"),
            median_q90_error=("q90_error", "median"),
            median_context_time=("last_context_time", "median"),
            median_n_future=("n_future", "median"),
            min_n_calibration=("n_calibration", "min"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget", "source_run", "median_width90", "normalized_method"])
    )
    grouped["headline_split"] = grouped["split_kind"].isin(STRICT_SPLITS)
    return grouped


def lookup_summary(summary: pd.DataFrame, source_run: str, split_kind: str, budget: int, method: str) -> pd.Series | None:
    match = summary[
        (summary["source_run"] == source_run)
        & (summary["split_kind"] == split_kind)
        & (summary["budget"] == budget)
        & (summary["normalized_method"] == method)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def build_contraction(summary: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast in CONTRACTION_CONTRASTS:
        source_run = contrast["source_run"]
        allowed_budgets = [b for b in budgets if b > 0]
        for split_kind in ALL_SPLITS:
            for budget in allowed_budgets:
                baseline_budget = contrast["baseline_budget"]
                if baseline_budget is None:
                    baseline_budget = budget
                baseline = lookup_summary(
                    summary,
                    source_run,
                    split_kind,
                    int(baseline_budget),
                    contrast["baseline_method"],
                )
                comparison = lookup_summary(
                    summary,
                    source_run,
                    split_kind,
                    budget,
                    contrast["comparison_method"],
                )
                if baseline is None or comparison is None:
                    continue
                base_width = float(baseline["median_width90"])
                comp_width = float(comparison["median_width90"])
                width_ratio = comp_width / base_width if base_width > 0 else np.nan
                rows.append(
                    {
                        "contrast_id": contrast["contrast_id"],
                        "source_run": source_run,
                        "split_kind": split_kind,
                        "budget": budget,
                        "baseline_budget": int(baseline_budget),
                        "baseline_method": contrast["baseline_method"],
                        "comparison_method": contrast["comparison_method"],
                        "claim_role": contrast["claim_role"],
                        "baseline_median_width90": base_width,
                        "comparison_median_width90": comp_width,
                        "width90_ratio": width_ratio,
                        "width90_contraction": 1.0 - width_ratio if np.isfinite(width_ratio) else np.nan,
                        "baseline_cov90": float(baseline["cov90"]),
                        "comparison_cov90": float(comparison["cov90"]),
                        "cov90_delta": float(comparison["cov90"]) - float(baseline["cov90"]),
                        "baseline_median_rmse": float(baseline["median_future_rmse"]),
                        "comparison_median_rmse": float(comparison["median_future_rmse"]),
                        "median_rmse_gain": float(baseline["median_future_rmse"])
                        - float(comparison["median_future_rmse"]),
                        "n_curves_comparison": int(comparison["n_curves"]),
                        "headline_split": split_kind in STRICT_SPLITS,
                    }
                )
    return pd.DataFrame(rows).sort_values(["split_kind", "budget", "contrast_id"]).reset_index(drop=True)


def build_calibration_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["abs_cov50_error"] = (out["cov50"] - 0.50).abs()
    out["abs_cov90_error"] = (out["cov90"] - 0.90).abs()
    out["cov90_status"] = np.where(
        out["cov90"] >= 0.85,
        "acceptable",
        np.where(out["cov90"] >= 0.75, "watch", "poor"),
    )
    cols = [
        "source_run",
        "split_kind",
        "budget",
        "normalized_method",
        "n_curves",
        "cov50",
        "cov90",
        "cov95",
        "abs_cov50_error",
        "abs_cov90_error",
        "cov90_status",
        "median_width90",
        "median_future_rmse",
        "headline_split",
    ]
    return out[cols].sort_values(["split_kind", "budget", "source_run", "normalized_method"]).reset_index(drop=True)


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    allowed_cols = {"future_r2", "median_context_time", "last_context_time"}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
        if bad == 0:
            continue
        if col in allowed_cols or "_r2" in col or col.endswith("context_time"):
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
    summary: pd.DataFrame,
    contraction: pd.DataFrame,
    calibration: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    headline_contraction = contraction[
        (contraction["headline_split"])
        & (
            contraction["contrast_id"].isin(
                [
                    "window_static_to_measured_static_early",
                    "prior_to_measured_static_early",
                    "proxy_predicted_to_measured_static_early",
                ]
            )
        )
    ].copy()
    headline_calibration = calibration[
        (calibration["headline_split"])
        & (
            calibration["normalized_method"].isin(
                [
                    "static_time",
                    "measured_static_early_time",
                    "proxy_run_predicted_static_early_time",
                    "proxy_run_measured_static_early_time",
                ]
            )
        )
    ].copy()

    lines = [
        "# PLGA E2 Empirical Uncertainty Contraction",
        "",
        "Date: 2026-06-11",
        "",
        "## Scope",
        "",
        "This is E2 from `docs/plga_observation_budget_goal_design_2026-06-11.md`.",
        "It uses fold-jackknife empirical envelopes over out-of-fold future RMSE.",
        "",
        "Important limitation: these are empirical error-tolerance envelopes, not Bayesian posterior intervals.",
        "",
        "## Output Files",
        "",
        "- `uncertainty_by_budget.csv`",
        "- `uncertainty_contraction.csv`",
        "- `calibration_summary.csv`",
        "- `data_checks.csv`",
        "- `lock_metadata.json`",
        "",
        "## Headline Width Contraction",
        "",
        markdown_table(
            headline_contraction,
            [
                "split_kind",
                "contrast_id",
                "budget",
                "baseline_median_width90",
                "comparison_median_width90",
                "width90_ratio",
                "width90_contraction",
                "baseline_cov90",
                "comparison_cov90",
                "median_rmse_gain",
            ],
        ),
        "",
        "## Headline Calibration",
        "",
        markdown_table(
            headline_calibration,
            [
                "split_kind",
                "source_run",
                "budget",
                "normalized_method",
                "cov50",
                "cov90",
                "cov90_status",
                "median_width90",
                "median_future_rmse",
            ],
        ),
        "",
        "## Data Checks",
        "",
        markdown_table(
            checks,
            ["name", "rows", "unexpected_nonfinite", "allowed_nonfinite"],
        ),
        "",
        "## Reading Rule",
        "",
        "- A lower `median_width90` means a narrower empirical future-error envelope.",
        "- `width90_ratio < 1` means the comparison condition contracts relative to the baseline.",
        "- Coverage is checked on heldout rows against envelopes calibrated from other folds.",
        "- Treat this as uncertainty evidence only if width contracts without severe cov90 collapse.",
        "",
    ]
    (out / "uncertainty_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    args.out.mkdir(parents=True, exist_ok=True)

    direct = load_direct(args.direct_per_curve, budgets)
    proxy = load_proxy(args.proxy_per_curve, budgets)
    per_curve = pd.concat([direct, proxy], ignore_index=True)

    envelopes = build_fold_jackknife_envelopes(per_curve)
    summary = build_uncertainty_by_budget(envelopes)
    contraction = build_contraction(summary, budgets)
    calibration = build_calibration_summary(summary)
    checks = pd.DataFrame(
        [
            finite_check("normalized_per_curve", per_curve),
            finite_check("fold_jackknife_envelopes", envelopes),
            finite_check("uncertainty_by_budget", summary),
            finite_check("uncertainty_contraction", contraction),
            finite_check("calibration_summary", calibration),
        ]
    )

    if checks["unexpected_nonfinite"].ne("{}").any():
        raise ValueError(f"Unexpected non-finite values:\n{checks.to_string(index=False)}")

    summary.to_csv(args.out / "uncertainty_by_budget.csv", index=False)
    contraction.to_csv(args.out / "uncertainty_contraction.csv", index=False)
    calibration.to_csv(args.out / "calibration_summary.csv", index=False)
    envelopes.to_csv(args.out / "fold_jackknife_envelopes.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "cli_args": args_to_metadata(args),
        "budgets": budgets,
        "strict_splits": STRICT_SPLITS,
        "experiment_status": "E2_empirical_uncertainty_contraction",
        "uncertainty_kind": "fold_jackknife_empirical_future_rmse_envelope",
        "width_definition": "2 * empirical future_rmse quantile",
        "outputs": [
            "uncertainty_by_budget.csv",
            "uncertainty_contraction.csv",
            "calibration_summary.csv",
            "fold_jackknife_envelopes.csv",
            "data_checks.csv",
            "lock_metadata.json",
            "uncertainty_report.md",
        ],
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.out, summary, contraction, calibration, checks)

    print((args.out / "uncertainty_report.md").resolve())
    headline = contraction[
        (contraction["headline_split"])
        & (contraction["contrast_id"] == "window_static_to_measured_static_early")
    ]
    print(headline[["split_kind", "budget", "width90_ratio", "width90_contraction", "comparison_cov90"]].to_string(index=False))


if __name__ == "__main__":
    main()
