"""100 - Audit original long-acting-injectables author models.

Purpose:
    Inspect and summarize the original author's tabular ML baselines from a
    local checkout supplied with --author-root or LAI_AUTHOR_ROOT.

    The original repository has two regimes:
      - zero-shot: static descriptors + Time -> Release
      - few-shot: static descriptors + Time + T=0.25/T=0.5/T=1.0 -> Release

    This script does not retrain their models. It reads the released nested-CV
    result pickles and compares their evaluation protocol against our current
    PLGA probes.

Produces:
    outputs/100_lai_author_model_audit/
      dataset_summary.csv
      author_cv_iteration_metrics.csv
      author_cv_summary_by_model.csv
      our_anchor_summary.csv
      report.md
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("outputs/100_lai_author_model_audit")
DEFAULT_OUR_DIRECT = Path("outputs/97_plga_direct_early_blackbox_control/summary_by_method.csv")
DEFAULT_OUR_PROXY = Path("outputs/99_plga_static_to_early_proxy_probe/decision_table.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit original LAI author model results.")
    parser.add_argument(
        "--author-root",
        type=Path,
        default=None,
        help="Path to the original long-acting-injectables checkout. Can also use LAI_AUTHOR_ROOT.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--our-direct", type=Path, default=DEFAULT_OUR_DIRECT)
    parser.add_argument("--our-proxy", type=Path, default=DEFAULT_OUR_PROXY)
    return parser.parse_args()


def resolve_author_root(author_root: Path | None) -> Path:
    if author_root is not None:
        return author_root
    env_root = os.environ.get("LAI_AUTHOR_ROOT")
    if env_root:
        return Path(env_root)
    raise SystemExit("--author-root is required, or set LAI_AUTHOR_ROOT")


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


def safe_numeric_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def safe_string_array(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def dataset_summary(author_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime, rel in [
        ("few-shot", Path("few_shot_models/Dataset_17_feat.xlsx")),
        ("zero-shot", Path("zero_shot_models/Dataset_14_feat.xlsx")),
    ]:
        path = author_root / rel
        df = pd.read_excel(path)
        feature_cols = [c for c in df.columns if c not in {"Experimental_index", "DP_Group", "Release"}]
        early_cols = [c for c in feature_cols if str(c).startswith("T=")]
        rows.append(
            {
                "regime": regime,
                "path": str(path),
                "rows": int(len(df)),
                "experimental_curves": int(df["Experimental_index"].nunique()),
                "dp_groups": int(df["DP_Group"].nunique()),
                "feature_count": int(len(feature_cols)),
                "feature_cols": ", ".join(feature_cols),
                "early_observation_cols": ", ".join(early_cols),
                "time_min": float(pd.to_numeric(df["Time"], errors="coerce").min()),
                "time_max": float(pd.to_numeric(df["Time"], errors="coerce").max()),
                "release_min": float(pd.to_numeric(df["Release"], errors="coerce").min()),
                "release_max": float(pd.to_numeric(df["Release"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def flatten_cv_row(row: pd.Series) -> pd.DataFrame:
    exp = safe_string_array(row["Experimental Index"])
    dp = safe_string_array(row["DP_Groups"])
    time = safe_numeric_array(row["Time"])
    y = safe_numeric_array(row["Experimental_Release"])
    pred = safe_numeric_array(row["Predicted_Release"])
    n = min(len(exp), len(dp), len(time), len(y), len(pred))
    return pd.DataFrame(
        {
            "experimental_index": exp[:n].astype(str),
            "dp_group": dp[:n].astype(str),
            "time": time[:n],
            "y_true": y[:n],
            "y_pred": pred[:n],
        }
    )


def summarize_cv_file(path: Path, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = path.stem
    prefix = "17_feat_" if regime == "few-shot" else "14_feat_"
    if model.startswith(prefix):
        model = model[len(prefix) :]
    cv = pd.read_pickle(path)
    iter_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for _, row in cv.iterrows():
        flat = flatten_cv_row(row)
        if flat.empty:
            continue
        err = flat["y_pred"].to_numpy(dtype=float) - flat["y_true"].to_numpy(dtype=float)
        iter_id = int(row["Iter"])
        iter_rows.append(
            {
                "regime": regime,
                "model": model,
                "iter": iter_id,
                "valid_mae": float(row["Valid Score"]),
                "test_mae": float(row["Test Score"]),
                "point_mae_recomputed": float(np.mean(np.abs(err))),
                "point_rmse": float(np.sqrt(np.mean(err**2))),
                "point_r2": safe_r2(flat["y_true"].to_numpy(dtype=float), flat["y_pred"].to_numpy(dtype=float)),
                "n_points": int(len(flat)),
                "n_curves": int(flat["experimental_index"].nunique()),
                "n_dp_groups": int(flat["dp_group"].nunique()),
            }
        )
        for cid, sub in flat.groupby("experimental_index", sort=True):
            yt = sub["y_true"].to_numpy(dtype=float)
            yp = sub["y_pred"].to_numpy(dtype=float)
            curve_rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "iter": iter_id,
                    "experimental_index": cid,
                    "dp_group": str(sub["dp_group"].iloc[0]),
                    "curve_mae": float(np.mean(np.abs(yp - yt))),
                    "curve_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
                    "curve_r2": safe_r2(yt, yp),
                    "n_points": int(len(sub)),
                }
            )
    return pd.DataFrame(iter_rows), pd.DataFrame(curve_rows)


def author_cv_metrics(author_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    iter_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for regime, rel in [
        ("few-shot", Path("few_shot_models/NESTED_CV_RESULTS")),
        ("zero-shot", Path("zero_shot_models/NESTED_CV_RESULTS")),
    ]:
        for path in sorted((author_root / rel).glob("*.pkl")):
            iter_df, curve_df = summarize_cv_file(path, regime)
            if not iter_df.empty:
                iter_frames.append(iter_df)
            if not curve_df.empty:
                curve_frames.append(curve_df)
    return (
        pd.concat(iter_frames, ignore_index=True) if iter_frames else pd.DataFrame(),
        pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(),
    )


def summarize_author(iter_metrics: pd.DataFrame, curve_metrics: pd.DataFrame) -> pd.DataFrame:
    iter_summary = (
        iter_metrics.groupby(["regime", "model"], dropna=False)
        .agg(
            n_iters=("iter", "nunique"),
            median_test_mae=("test_mae", "median"),
            mean_test_mae=("test_mae", "mean"),
            min_test_mae=("test_mae", "min"),
            max_test_mae=("test_mae", "max"),
            median_point_rmse=("point_rmse", "median"),
            median_point_r2=("point_r2", "median"),
            median_n_points=("n_points", "median"),
            median_n_curves=("n_curves", "median"),
            median_n_dp_groups=("n_dp_groups", "median"),
        )
        .reset_index()
    )
    curve_summary = (
        curve_metrics.groupby(["regime", "model"], dropna=False)
        .agg(
            median_curve_mae=("curve_mae", "median"),
            median_curve_rmse=("curve_rmse", "median"),
            median_curve_r2=("curve_r2", "median"),
        )
        .reset_index()
    )
    return (
        iter_summary.merge(curve_summary, on=["regime", "model"], how="left")
        .sort_values(["regime", "median_test_mae", "median_curve_rmse"])
        .reset_index(drop=True)
    )


def load_our_anchors(direct_path: Path, proxy_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if direct_path.exists():
        direct = pd.read_csv(direct_path)
        methods = ["direct_static_time", "direct_early_time", "direct_static_early_time"]
        sub = direct[direct["method"].isin(methods)].copy()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "source": "ours-97",
                    "split_kind": row["split_kind"],
                    "budget": int(row["budget"]),
                    "method": row["method"],
                    "metric": "median_future_rmse",
                    "value": float(row["median_future_rmse"]),
                    "note": "future-only per-curve RMSE; not point-MAE comparable to author CV",
                }
            )
    if proxy_path.exists():
        proxy = pd.read_csv(proxy_path)
        for _, row in proxy.iterrows():
            rows.append(
                {
                    "source": "ours-99",
                    "split_kind": row["split_kind"],
                    "budget": int(row["budget"]),
                    "method": "static_to_early_proxy",
                    "metric": "predicted_early_rmse",
                    "value": float(row["predicted_early_rmse"]),
                    "note": f"static={float(row['static_rmse']):.3f}; true_early={float(row['true_early_rmse']):.3f}; recovery={row.get('recovery_fraction', np.nan)}",
                }
            )
    return pd.DataFrame(rows)


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].to_numpy(dtype=float)
        nonfinite = ~np.isfinite(vals)
        if not bool(nonfinite.any()):
            continue
        allowed_mask = np.zeros(len(df), dtype=bool)
        if col in {"point_r2", "curve_r2", "median_point_r2", "median_curve_r2", "value"}:
            allowed_mask |= nonfinite
        n_allowed = int((nonfinite & allowed_mask).sum())
        n_bad = int((nonfinite & ~allowed_mask).sum())
        if n_allowed:
            allowed[col] = n_allowed
        if n_bad:
            unexpected[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": unexpected, "allowed_nonfinite": allowed}


def write_report(
    out: Path,
    ds: pd.DataFrame,
    summary: pd.DataFrame,
    anchors: pd.DataFrame,
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    best = summary.groupby("regime", as_index=False).head(5)
    author_key = summary[
        ((summary["regime"] == "few-shot") & (summary["model"].isin(["LGBM", "RF", "NGB"])))
        | ((summary["regime"] == "zero-shot") & (summary["model"].isin(["RF", "LGBM", "NGB"])))
    ].copy()
    anchor_focus = anchors[
        (anchors["source"] == "ours-97")
        & (anchors["method"].isin(["direct_static_time", "direct_static_early_time"]))
        & (anchors["budget"].isin([0, 1, 3, 5]))
    ].copy()
    lines = [
        "# Original LAI Author Model Audit",
        "",
        "The original repository is a tabular point-regression setup, not a continuous curve model.",
        "",
        "## Dataset Regimes",
        "",
        ds.to_markdown(index=False),
        "",
        "## Best Author Nested-CV Models",
        "",
        best.to_markdown(index=False),
        "",
        "## Author Key Models",
        "",
        author_key.to_markdown(index=False),
        "",
        "## Our Current Anchors",
        "",
        anchor_focus.to_markdown(index=False) if not anchor_focus.empty else "_No anchors found._",
        "",
        "## Interpretation",
        "",
        "- Author `zero-shot`: static descriptors plus `Time` predict pointwise release.",
        "- Author `few-shot`: same features plus fixed early releases `T=0.25`, `T=0.5`, `T=1.0`.",
        "- Their outer split holds out `DP_Group`, not source dataset. It is stricter than point-random but looser than our source-dataset LODO.",
        "- Their published saved RF model pickle is not portable under the current sklearn version; the nested-CV result pickles load cleanly because they are pandas frames.",
        "- The fair next comparison is to reproduce author-style zero/few-shot feature sets on our exact 91/97 split assignments.",
        "",
        "## Verification",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Run Metadata",
        "",
        f"- author_root: `{args.author_root}`",
        f"- git_hash: `{git_hash()}`",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.author_root = resolve_author_root(args.author_root)
    args.out.mkdir(parents=True, exist_ok=True)

    ds = dataset_summary(args.author_root)
    iter_metrics, curve_metrics = author_cv_metrics(args.author_root)
    summary = summarize_author(iter_metrics, curve_metrics)
    anchors = load_our_anchors(args.our_direct, args.our_proxy)
    checks = [
        finite_check("dataset_summary", ds),
        finite_check("author_cv_iteration_metrics", iter_metrics),
        finite_check("author_cv_curve_metrics", curve_metrics),
        finite_check("author_cv_summary_by_model", summary),
        finite_check("our_anchor_summary", anchors),
    ]

    ds.to_csv(args.out / "dataset_summary.csv", index=False)
    iter_metrics.to_csv(args.out / "author_cv_iteration_metrics.csv", index=False)
    curve_metrics.to_csv(args.out / "author_cv_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "author_cv_summary_by_model.csv", index=False)
    anchors.to_csv(args.out / "our_anchor_summary.csv", index=False)
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "author_root": str(args.author_root),
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "checks": checks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out, ds, summary, anchors, checks, args)
    print((args.out / "report.md").resolve())
    print(summary.groupby("regime", as_index=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
