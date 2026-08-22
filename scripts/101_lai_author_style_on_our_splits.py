"""101 - Reproduce author-style LAI baselines on our PLGA splits.

Purpose:
    Port the original LAI paper's tabular feature design onto our current PLGA
    split assignments.

    Author-style regimes:
      - zero-shot: static descriptors + query Time -> Release
      - few-shot: static descriptors + query Time + Q(0.25), Q(0.5), Q(1.0)

    Unlike the original pointwise benchmark, this script defaults to scoring
    only target points later than 1.0 day. That prevents fixed early features
    from trivially predicting the same early timepoints.

Consumes:
    outputs/148_release_main_cumulative_v1/{curves_long.csv,formulations.csv}
    outputs/91_plga_static_feature_ceiling_*/per_curve_metrics.csv

Produces:
    outputs/101_lai_author_style_on_our_splits/
      per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      lock_metadata.json
      report.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - reported in lock metadata
    LGBMRegressor = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT97 = ROOT / "scripts" / "97_plga_direct_early_blackbox_control.py"
DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/101_lai_author_style_on_our_splits")

AUTHOR_STATIC_COLS = [
    "LA/GA",
    "Polymer_MW",
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Initial D/M ratio",
    "DLC",
    "SA-V",
    "SE",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_NHA",
    "Drug_LogP",
]
EARLY_TIMES = [0.25, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Author-style LAI baselines on our splits.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-target-time", type=float, default=1.0)
    parser.add_argument("--max-curves", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_script97():
    spec = importlib.util.spec_from_file_location("script97", SCRIPT97)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT97}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def make_lgbm(regime: str, seed: int):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not importable")
    if regime == "author_few_lgbm_fixed_early":
        return LGBMRegressor(
            boosting_type="dart",
            learning_rate=0.1,
            min_child_samples=40,
            min_child_weight=0.01,
            n_estimators=250,
            n_jobs=-1,
            num_leaves=16,
            random_state=seed,
            reg_alpha=0,
            reg_lambda=0,
            subsample=0.8,
            verbose=-1,
        )
    return LGBMRegressor(
        boosting_type="goss",
        learning_rate=0.1,
        min_child_samples=2,
        min_child_weight=10.0,
        n_estimators=400,
        n_jobs=-1,
        num_leaves=64,
        random_state=seed,
        reg_alpha=0.015,
        reg_lambda=0.005,
        subsample=0.8,
        verbose=-1,
    )


def static_matrix(train_forms: pd.DataFrame, test_forms: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in AUTHOR_STATIC_COLS if c in train_forms.columns]
    tr = train_forms[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    te = test_forms[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = tr.median(axis=0).fillna(0.0)
    tr = tr.fillna(med)
    te = te.fillna(med)
    scaler = StandardScaler().fit(tr)
    return scaler.transform(tr), scaler.transform(te), cols


def interp_release(curve: pd.DataFrame, time: float) -> float:
    t = curve["time_days"].to_numpy(dtype=float)
    q = curve["release_fraction"].to_numpy(dtype=float)
    order = np.argsort(t)
    t = t[order]
    q = q[order]
    if len(t) < 2:
        return np.nan
    return float(np.interp(time, t, q, left=q[0], right=q[-1]))


def early_fixed_matrix(curve_groups: dict[str, pd.DataFrame], curve_ids: list[str]) -> np.ndarray:
    out = np.full((len(curve_ids), len(EARLY_TIMES)), np.nan, dtype=float)
    for i, cid in enumerate(curve_ids):
        curve = curve_groups.get(cid)
        if curve is None or curve.empty:
            continue
        out[i] = [interp_release(curve, t) for t in EARLY_TIMES]
    return out


def build_point_table(
    *,
    forms: pd.DataFrame,
    static_x: np.ndarray,
    early_x: np.ndarray,
    curve_groups: dict[str, pd.DataFrame],
    method: str,
    min_target_time: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    curve_ids = forms["unified_curve_id"].astype(str).tolist()
    for i, cid in enumerate(curve_ids):
        curve = curve_groups[cid]
        if method == "author_few_lgbm_fixed_early" and not np.isfinite(early_x[i]).all():
            continue
        static_part = static_x[i]
        early_part = early_x[i] if method == "author_few_lgbm_fixed_early" else np.zeros(0, dtype=float)
        for _, row in curve.iterrows():
            t = float(row["time_days"])
            y = float(row["release_fraction"])
            if not np.isfinite(t) or not np.isfinite(y) or t <= min_target_time + 1e-12:
                continue
            parts = [static_part, np.asarray([t], dtype=float)]
            if method == "author_few_lgbm_fixed_early":
                parts.append(early_part)
            x_rows.append(np.concatenate(parts))
            y_rows.append(y)
            meta_rows.append(
                {
                    "unified_curve_id": cid,
                    "time_days": t,
                    "source_dataset": str(forms.iloc[i].get("source_dataset", "")),
                    "source_group": str(forms.iloc[i].get("source_group", "")),
                }
            )
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), pd.DataFrame(meta_rows)


def evaluate_points(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    work = meta.copy()
    work["y_true"] = y_true
    work["y_pred"] = np.clip(y_pred, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    for cid, sub in work.groupby("unified_curve_id", sort=True):
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "unified_curve_id": cid,
                "source_dataset": str(sub["source_dataset"].iloc[0]),
                "source_group": str(sub["source_group"].iloc[0]),
                "future_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
                "future_mae": float(np.mean(np.abs(yp - yt))),
                "future_r2": safe_r2(yt, yp),
                "n_future": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def run_probe(args: argparse.Namespace) -> pd.DataFrame:
    s97 = load_script97()
    s97.seed_all(args.seed)
    forms, curve_groups = s97.load_plga_data(args)
    splits = s97.load_split_assignments(args)
    forms = forms[forms["unified_curve_id"].isin(set(splits["unified_curve_id"]))].copy()
    forms = forms[forms["unified_curve_id"].isin(set(curve_groups))].copy()
    splits = splits[splits["unified_curve_id"].isin(set(forms["unified_curve_id"]))].copy()
    forms_indexed = forms.set_index("unified_curve_id", drop=False)

    rows: list[pd.DataFrame] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        split_df = splits[splits["split_kind"] == split_kind].copy()
        for fold in sorted(split_df["fold"].unique()):
            test_ids = sorted(split_df.loc[split_df["fold"] == fold, "unified_curve_id"].astype(str).unique())
            train_ids = sorted(set(split_df["unified_curve_id"].astype(str).unique()) - set(test_ids))
            train_forms = forms_indexed.loc[[cid for cid in train_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            test_forms = forms_indexed.loc[[cid for cid in test_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            if len(train_forms) < 20 or test_forms.empty:
                continue
            x_train_static, x_test_static, used_cols = static_matrix(train_forms, test_forms)
            train_early = early_fixed_matrix(curve_groups, train_forms["unified_curve_id"].astype(str).tolist())
            test_early = early_fixed_matrix(curve_groups, test_forms["unified_curve_id"].astype(str).tolist())
            for method in ["author_zero_lgbm", "author_few_lgbm_fixed_early"]:
                train_points = build_point_table(
                    forms=train_forms,
                    static_x=x_train_static,
                    early_x=train_early,
                    curve_groups=curve_groups,
                    method=method,
                    min_target_time=args.min_target_time,
                )
                test_points = build_point_table(
                    forms=test_forms,
                    static_x=x_test_static,
                    early_x=test_early,
                    curve_groups=curve_groups,
                    method=method,
                    min_target_time=args.min_target_time,
                )
                x_train, y_train, _ = train_points
                x_test, y_test, meta_test = test_points
                if len(x_train) < 50 or len(x_test) < 1:
                    continue
                model = make_lgbm(method, args.seed + int(fold) * 17 + len(method))
                model.fit(x_train, y_train)
                pred = np.asarray(model.predict(x_test), dtype=float)
                eval_df = evaluate_points(meta_test, y_test, pred)
                eval_df.insert(0, "split_kind", split_kind)
                eval_df.insert(1, "fold", int(fold))
                eval_df.insert(2, "method", method)
                eval_df["min_target_time"] = float(args.min_target_time)
                eval_df["n_train_points"] = int(len(x_train))
                eval_df["n_test_points"] = int(len(x_test))
                eval_df["author_static_cols"] = ", ".join(used_cols)
                rows.append(eval_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    if per_curve.empty:
        return pd.DataFrame()
    return (
        per_curve.groupby(["split_kind", "method"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_mae=("future_mae", "median"),
            median_future_r2=("future_r2", "median"),
            frac_r2_positive=("future_r2", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.0))),
            median_n_future=("n_future", "median"),
        )
        .reset_index()
        .sort_values(["split_kind", "median_future_rmse"])
    )


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        sub = summary[summary["split_kind"] == split_kind].copy()
        vals = {row["method"]: float(row["median_future_rmse"]) for _, row in sub.iterrows()}
        if {"author_zero_lgbm", "author_few_lgbm_fixed_early"}.issubset(vals):
            gain = vals["author_zero_lgbm"] - vals["author_few_lgbm_fixed_early"]
            rows.append(
                {
                    "split_kind": split_kind,
                    "question": "Do fixed early observations improve author-style LGBM on our splits?",
                    "answer": "yes" if gain > 0 else "no",
                    "zero_rmse": vals["author_zero_lgbm"],
                    "few_rmse": vals["author_few_lgbm_fixed_early"],
                    "rmse_gain": gain,
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
        if col in {"future_r2", "median_future_r2"}:
            allowed_mask |= nonfinite
        n_allowed = int((nonfinite & allowed_mask).sum())
        n_bad = int((nonfinite & ~allowed_mask).sum())
        if n_allowed:
            allowed[col] = n_allowed
        if n_bad:
            unexpected[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": unexpected, "allowed_nonfinite": allowed}


def write_report(out: Path, summary: pd.DataFrame, decisions: pd.DataFrame, checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Author-Style LAI Baselines On Our Splits",
        "",
        "This reproduces the original paper's feature design on our PLGA split assignments.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "_No decisions generated._",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No summary generated._",
        "",
        "## Interpretation Guard",
        "",
        "- `author_zero_lgbm`: static descriptors plus query time.",
        "- `author_few_lgbm_fixed_early`: static descriptors plus query time plus interpolated Q(0.25), Q(0.5), Q(1.0).",
        f"- Scoring uses only points with `time_days > {args.min_target_time}`.",
        "- This is not nested hyperparameter tuning; it uses released best LGBM-style hyperparameters inside our folds.",
        "",
        "## Verification",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    per_curve = run_probe(args)
    summary = summarize(per_curve)
    decisions = build_decision_table(summary)
    checks = [
        finite_check("per_curve_metrics", per_curve),
        finite_check("summary_by_method", summary),
        finite_check("decision_table", decisions),
    ]
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "pool_dir": str(args.pool_dir),
                "min_target_time": args.min_target_time,
                "early_times": EARLY_TIMES,
                "checks": checks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out, summary, decisions, checks, args)
    print((args.out / "report.md").resolve())
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
