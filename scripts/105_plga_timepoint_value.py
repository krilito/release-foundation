"""105 - PLGA observed timepoint value.

Experiment:
    E3. Which early observed release points are worth taking?

Consumes:
    outputs/148_release_main_cumulative_v1/curves_long.csv
    outputs/148_release_main_cumulative_v1/formulations.csv
    outputs/91_plga_static_feature_ceiling_*/per_curve_metrics.csv

Produces:
    outputs/105_plga_timepoint_value/
      per_curve_metrics.csv
      summary_by_policy.csv
      policy_contrasts.csv
      timepoint_rank_by_split.csv
      marginal_value.csv
      data_checks.csv
      lock_metadata.json
      timepoint_value_report.md

Interpretation:
    This is a direct black-box experimental-design probe. It does not introduce
    a new model class. It reuses script 97's PLGA data, splits, static feature
    design, and ExtraTrees control, then changes only the observed context
    policy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT97 = ROOT / "scripts" / "97_plga_direct_early_blackbox_control.py"
DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/105_plga_timepoint_value")

STRICT_SPLITS = ["source-group-kfold", "source-dataset-lodo"]
ALL_SPLITS = ["random-kfold", *STRICT_SPLITS]
DEFAULT_SPLITS = STRICT_SPLITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLGA observed timepoint value probe.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument("--model", choices=["extra_trees", "random_forest", "ridge"], default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--include-context-only", action="store_true")
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def args_to_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def parse_splits(text: str) -> list[str]:
    splits = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = sorted(set(splits) - set(ALL_SPLITS))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}")
    if not splits:
        raise ValueError("--splits produced an empty list")
    return splits


def clean_category(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "__MISSING__"


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def build_policies(max_index: int) -> list[dict[str, Any]]:
    if max_index < 2:
        raise ValueError("--max-index must be at least 2")
    policies: list[dict[str, Any]] = []
    for k in range(1, max_index + 1):
        policies.append(
            {
                "policy_id": f"first{k}",
                "policy_kind": "cumulative",
                "context_indices": list(range(k)),
                "cutoff_index": k - 1,
                "n_context": k,
                "observed_index": k,
                "headline": k in {1, 2, 3, 5},
            }
        )
    for k in range(1, max_index + 1):
        policies.append(
            {
                "policy_id": f"point{k}",
                "policy_kind": "single",
                "context_indices": [k - 1],
                "cutoff_index": k - 1,
                "n_context": 1,
                "observed_index": k,
                "headline": k in {1, 2, 3, 5},
            }
        )
    for k in range(2, max_index + 1):
        policies.append(
            {
                "policy_id": f"first{k - 1}_cut{k}",
                "policy_kind": "marginal_ablation",
                "context_indices": list(range(k - 1)),
                "cutoff_index": k - 1,
                "n_context": k - 1,
                "observed_index": k,
                "headline": False,
            }
        )
    return policies


def context_features(
    curve_groups: dict[str, pd.DataFrame],
    curve_ids: list[str],
    context_indices: list[int],
    cutoff_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_context = len(context_indices)
    feats = np.full((len(curve_ids), n_context * 3), np.nan, dtype=float)
    first_t = np.full(len(curve_ids), np.nan, dtype=float)
    last_t = np.full(len(curve_ids), np.nan, dtype=float)
    span = np.full(len(curve_ids), np.nan, dtype=float)
    for i, cid in enumerate(curve_ids):
        curve = curve_groups[cid]
        if len(curve) <= cutoff_index or any(idx >= len(curve) for idx in context_indices):
            continue
        selected = curve.iloc[context_indices].sort_values("time_days")
        cutoff_t = float(curve.iloc[cutoff_index]["time_days"])
        if not np.isfinite(cutoff_t):
            continue
        t = selected["time_days"].to_numpy(dtype=float)
        q = selected["release_fraction"].to_numpy(dtype=float)
        if not (np.isfinite(t).all() and np.isfinite(q).all()):
            continue
        dt = np.diff(np.r_[0.0, t])
        dq = np.diff(np.r_[0.0, q])
        slope = dq / np.clip(dt, 1e-8, None)
        feats[i, 0:n_context] = np.log1p(np.clip(t, 0.0, None))
        feats[i, n_context : 2 * n_context] = q
        feats[i, 2 * n_context : 3 * n_context] = slope
        first_t[i] = float(t[0])
        last_t[i] = cutoff_t
        span[i] = max(cutoff_t - float(t[0]), 0.0)
    return feats, first_t, last_t, span


def query_features(t: float, last_t: float) -> np.ndarray:
    if np.isfinite(last_t):
        delta = max(float(t) - float(last_t), 0.0)
        ratio = float(t) / max(float(last_t), 1e-8)
    else:
        delta = float(t)
        ratio = 1.0
    return np.asarray([float(t), np.log1p(max(float(t), 0.0)), delta, np.log1p(delta), ratio], dtype=float)


def build_point_table(
    *,
    forms: pd.DataFrame,
    static_design: pd.DataFrame,
    context_matrix: np.ndarray,
    first_times: np.ndarray,
    cutoff_times: np.ndarray,
    spans: np.ndarray,
    curve_groups: dict[str, pd.DataFrame],
    policy: dict[str, Any],
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    curve_ids = forms["unified_curve_id"].astype(str).tolist()

    for i, cid in enumerate(curve_ids):
        if not np.isfinite(cutoff_times[i]):
            continue
        if "context" in feature_mode and not np.isfinite(context_matrix[i]).all():
            continue
        curve = curve_groups[cid]
        cutoff_t = float(cutoff_times[i])
        t_all = curve["time_days"].to_numpy(dtype=float)
        y_all = curve["release_fraction"].to_numpy(dtype=float)
        mask = t_all > cutoff_t + 1e-12
        if int(mask.sum()) < 1:
            continue

        static_part = static_design.iloc[i].to_numpy(dtype=float)
        context_part = context_matrix[i] if "context" in feature_mode else np.zeros(0, dtype=float)
        for t, y in zip(t_all[mask], y_all[mask]):
            parts = []
            if "static" in feature_mode:
                parts.append(static_part)
            if "context" in feature_mode:
                parts.append(context_part)
            parts.append(query_features(float(t), cutoff_t))
            x_rows.append(np.concatenate(parts))
            y_rows.append(float(y))
            meta_rows.append(
                {
                    "unified_curve_id": cid,
                    "time_days": float(t),
                    "source_dataset": str(forms.iloc[i].get("source_dataset", "")),
                    "source_group": clean_category(forms.iloc[i].get("source_group", "")),
                    "first_context_time": float(first_times[i]) if np.isfinite(first_times[i]) else np.nan,
                    "last_context_time": cutoff_t,
                    "context_time_span": float(spans[i]) if np.isfinite(spans[i]) else np.nan,
                    "policy_id": policy["policy_id"],
                    "policy_kind": policy["policy_kind"],
                    "n_context": int(policy["n_context"]),
                    "observed_index": int(policy["observed_index"]),
                    "cutoff_index": int(policy["cutoff_index"]) + 1,
                    "context_indices": ",".join(str(int(idx) + 1) for idx in policy["context_indices"]),
                    "headline_policy": bool(policy["headline"]),
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
                "policy_id": str(sub["policy_id"].iloc[0]),
                "policy_kind": str(sub["policy_kind"].iloc[0]),
                "n_context": int(sub["n_context"].iloc[0]),
                "observed_index": int(sub["observed_index"].iloc[0]),
                "cutoff_index": int(sub["cutoff_index"].iloc[0]),
                "context_indices": str(sub["context_indices"].iloc[0]),
                "headline_policy": bool(sub["headline_policy"].iloc[0]),
                "unified_curve_id": cid,
                "source_dataset": str(sub["source_dataset"].iloc[0]),
                "source_group": str(sub["source_group"].iloc[0]),
                "first_context_time": float(sub["first_context_time"].iloc[0]),
                "last_context_time": float(sub["last_context_time"].iloc[0]),
                "context_time_span": float(sub["context_time_span"].iloc[0]),
                "min_future_time": float(sub["time_days"].min()),
                "future_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
                "future_mae": float(np.mean(np.abs(yp - yt))),
                "future_r2": safe_r2(yt, yp),
                "n_future": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def run_policies(args: argparse.Namespace, splits_to_run: list[str]) -> pd.DataFrame:
    script97 = load_script97()
    forms, curve_groups = script97.load_plga_data(args)
    splits = script97.load_split_assignments(args)
    forms = forms[forms["unified_curve_id"].isin(set(splits["unified_curve_id"]))].copy()
    forms = forms[forms["unified_curve_id"].isin(set(curve_groups))].copy()
    split_ids = set(forms["unified_curve_id"])
    splits = splits[splits["unified_curve_id"].isin(split_ids)].copy()
    forms_indexed = forms.set_index("unified_curve_id", drop=False)
    policies = build_policies(args.max_index)
    feature_modes = ["static", "static_context"]
    if args.include_context_only:
        feature_modes.insert(1, "context")
    rows: list[pd.DataFrame] = []

    for split_kind in splits_to_run:
        split_df = splits[splits["split_kind"] == split_kind].copy()
        for fold in sorted(split_df["fold"].unique()):
            test_ids = sorted(split_df.loc[split_df["fold"] == fold, "unified_curve_id"].astype(str).unique())
            train_ids = sorted(set(split_df["unified_curve_id"].astype(str).unique()) - set(test_ids))
            train_forms_base = forms_indexed.loc[[cid for cid in train_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            test_forms_base = forms_indexed.loc[[cid for cid in test_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            if len(train_forms_base) < 20 or test_forms_base.empty:
                continue

            for policy in policies:
                print(
                    f"[105] split={split_kind} fold={int(fold)} policy={policy['policy_id']} "
                    f"kind={policy['policy_kind']}",
                    flush=True,
                )
                train_context, train_first, train_cutoff, train_span = context_features(
                    curve_groups,
                    train_forms_base["unified_curve_id"].astype(str).tolist(),
                    policy["context_indices"],
                    policy["cutoff_index"],
                )
                test_context, test_first, test_cutoff, test_span = context_features(
                    curve_groups,
                    test_forms_base["unified_curve_id"].astype(str).tolist(),
                    policy["context_indices"],
                    policy["cutoff_index"],
                )
                valid_train = np.isfinite(train_cutoff)
                valid_test = np.isfinite(test_cutoff)
                train_forms = train_forms_base.loc[valid_train].copy().reset_index(drop=True)
                test_forms = test_forms_base.loc[valid_test].copy().reset_index(drop=True)
                train_context_v = train_context[valid_train]
                test_context_v = test_context[valid_test]
                train_first_v = train_first[valid_train]
                test_first_v = test_first[valid_test]
                train_cutoff_v = train_cutoff[valid_train]
                test_cutoff_v = test_cutoff[valid_test]
                train_span_v = train_span[valid_train]
                test_span_v = test_span[valid_test]
                if len(train_forms) < 20 or test_forms.empty:
                    continue

                x_train_static, x_test_static = script97.build_static_design(train_forms, test_forms)
                for feature_mode in feature_modes:
                    x_train, y_train, _ = build_point_table(
                        forms=train_forms,
                        static_design=x_train_static,
                        context_matrix=train_context_v,
                        first_times=train_first_v,
                        cutoff_times=train_cutoff_v,
                        spans=train_span_v,
                        curve_groups=curve_groups,
                        policy=policy,
                        feature_mode=feature_mode,
                    )
                    x_test, y_test, meta_test = build_point_table(
                        forms=test_forms,
                        static_design=x_test_static,
                        context_matrix=test_context_v,
                        first_times=test_first_v,
                        cutoff_times=test_cutoff_v,
                        spans=test_span_v,
                        curve_groups=curve_groups,
                        policy=policy,
                        feature_mode=feature_mode,
                    )
                    if len(x_train) < 50 or len(x_test) < 1:
                        continue
                    model_seed = (
                        args.seed
                        + int(fold) * 31
                        + int(policy["observed_index"]) * 101
                        + len(policy["policy_id"]) * 17
                        + len(feature_mode)
                    )
                    model = script97.make_model(args.model, model_seed, args.n_estimators)
                    model.fit(x_train, y_train)
                    pred = np.asarray(model.predict(x_test), dtype=float)
                    eval_df = evaluate_points(meta_test, y_test, pred)
                    eval_df.insert(0, "split_kind", split_kind)
                    eval_df.insert(1, "fold", int(fold))
                    eval_df.insert(2, "method", f"direct_{feature_mode}_time")
                    eval_df.insert(3, "model", args.model)
                    eval_df["n_train_points"] = int(len(x_train))
                    eval_df["n_test_points"] = int(len(x_test))
                    rows.append(eval_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    if per_curve.empty:
        return pd.DataFrame()
    out = (
        per_curve.groupby(
            [
                "split_kind",
                "policy_id",
                "policy_kind",
                "n_context",
                "observed_index",
                "cutoff_index",
                "context_indices",
                "headline_policy",
                "method",
            ],
            dropna=False,
        )
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            n_rows=("unified_curve_id", "size"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_mae=("future_mae", "median"),
            mean_future_mae=("future_mae", "mean"),
            median_future_r2=("future_r2", "median"),
            frac_r2_positive=("future_r2", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.0))),
            median_last_context_time=("last_context_time", "median"),
            median_context_time_span=("context_time_span", "median"),
            median_n_future=("n_future", "median"),
        )
        .reset_index()
        .sort_values(["split_kind", "policy_kind", "observed_index", "method"])
    )
    out["headline_split"] = out["split_kind"].isin(STRICT_SPLITS)
    return out


def pair_methods(per_curve: pd.DataFrame, baseline_method: str, comparison_method: str) -> pd.DataFrame:
    key_cols = ["split_kind", "fold", "policy_id", "unified_curve_id"]
    metric_cols = [
        "future_rmse",
        "future_mae",
        "future_r2",
        "n_future",
        "last_context_time",
        "context_time_span",
        "min_future_time",
    ]
    base = per_curve[per_curve["method"] == baseline_method][
        key_cols
        + [
            "policy_kind",
            "n_context",
            "observed_index",
            "cutoff_index",
            "context_indices",
            "headline_policy",
            *metric_cols,
        ]
    ].copy()
    comp = per_curve[per_curve["method"] == comparison_method][key_cols + metric_cols].copy()
    paired = base.merge(comp, on=key_cols, suffixes=("_baseline", "_comparison"))
    paired["rmse_delta_baseline_minus_comparison"] = paired["future_rmse_baseline"] - paired["future_rmse_comparison"]
    paired["mae_delta_baseline_minus_comparison"] = paired["future_mae_baseline"] - paired["future_mae_comparison"]
    paired["future_window_matched"] = (
        np.isclose(paired["last_context_time_baseline"], paired["last_context_time_comparison"])
        & (paired["n_future_baseline"] == paired["n_future_comparison"])
    )
    return paired


def contrast_summary(paired: pd.DataFrame, contrast_id: str, baseline_method: str, comparison_method: str) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in paired.groupby(
        [
            "split_kind",
            "policy_id",
            "policy_kind",
            "n_context",
            "observed_index",
            "cutoff_index",
            "context_indices",
            "headline_policy",
        ],
        dropna=False,
    ):
        (
            split_kind,
            policy_id,
            policy_kind,
            n_context,
            observed_index,
            cutoff_index,
            context_indices,
            headline_policy,
        ) = keys
        median_gain = float(group["rmse_delta_baseline_minus_comparison"].median())
        median_time = float(group["last_context_time_comparison"].median())
        rows.append(
            {
                "contrast_id": contrast_id,
                "split_kind": split_kind,
                "policy_id": policy_id,
                "policy_kind": policy_kind,
                "n_context": int(n_context),
                "observed_index": int(observed_index),
                "cutoff_index": int(cutoff_index),
                "context_indices": context_indices,
                "headline_policy": bool(headline_policy),
                "baseline_method": baseline_method,
                "comparison_method": comparison_method,
                "n_pairs": int(len(group)),
                "baseline_median_rmse": float(group["future_rmse_baseline"].median()),
                "comparison_median_rmse": float(group["future_rmse_comparison"].median()),
                "median_rmse_gain": median_gain,
                "mean_rmse_gain": float(group["rmse_delta_baseline_minus_comparison"].mean()),
                "fraction_curves_improved": float((group["rmse_delta_baseline_minus_comparison"] > 0).mean()),
                "median_mae_gain": float(group["mae_delta_baseline_minus_comparison"].median()),
                "median_last_context_time": median_time,
                "gain_per_context_point": median_gain / max(int(n_context), 1),
                "gain_per_elapsed_day": median_gain / median_time if median_time > 0 else np.nan,
                "all_future_windows_matched": bool(group["future_window_matched"].all()),
                "headline_split": split_kind in STRICT_SPLITS,
            }
        )
    return pd.DataFrame(rows).sort_values(["split_kind", "policy_kind", "observed_index", "contrast_id"]).reset_index(drop=True)


def build_policy_contrasts(per_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = []
    summaries = []
    for contrast_id, baseline, comparison in [
        ("static_to_measured_static_context", "direct_static_time", "direct_static_context_time"),
        ("static_to_measured_context_only", "direct_static_time", "direct_context_time"),
    ]:
        paired = pair_methods(per_curve, baseline, comparison)
        paired.insert(0, "contrast_id", contrast_id)
        paired.insert(1, "baseline_method", baseline)
        paired.insert(2, "comparison_method", comparison)
        pairs.append(paired)
        summaries.append(contrast_summary(paired, contrast_id, baseline, comparison))
    return pd.concat(summaries, ignore_index=True), pd.concat(pairs, ignore_index=True)


def build_rank_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    head = contrasts[
        (contrasts["contrast_id"] == "static_to_measured_static_context")
        & (contrasts["headline_split"])
        & (contrasts["policy_kind"].isin(["cumulative", "single"]))
        & (contrasts["headline_policy"])
    ].copy()
    if head.empty:
        return pd.DataFrame()
    head["rank_by_gain"] = (
        head.groupby(["split_kind", "policy_kind"])["median_rmse_gain"].rank(method="dense", ascending=False).astype(int)
    )
    head["rank_by_gain_per_day"] = (
        head.groupby(["split_kind", "policy_kind"])["gain_per_elapsed_day"].rank(method="dense", ascending=False).astype("Int64")
    )
    return head.sort_values(["split_kind", "policy_kind", "rank_by_gain"]).reset_index(drop=True)


def build_marginal_value(per_curve: pd.DataFrame) -> pd.DataFrame:
    full = per_curve[
        (per_curve["method"] == "direct_static_context_time") & (per_curve["policy_kind"] == "cumulative")
    ].copy()
    ablated = per_curve[
        (per_curve["method"] == "direct_static_context_time") & (per_curve["policy_kind"] == "marginal_ablation")
    ].copy()
    rows: list[dict[str, Any]] = []
    for k in sorted(full["observed_index"].dropna().astype(int).unique()):
        if k <= 1:
            continue
        full_k = full[full["policy_id"] == f"first{k}"].copy()
        ablate_k = ablated[ablated["policy_id"] == f"first{k - 1}_cut{k}"].copy()
        if full_k.empty or ablate_k.empty:
            continue
        key_cols = ["split_kind", "fold", "unified_curve_id"]
        paired = ablate_k[
            key_cols + ["future_rmse", "future_mae", "last_context_time", "n_future"]
        ].merge(
            full_k[key_cols + ["future_rmse", "future_mae", "last_context_time", "n_future"]],
            on=key_cols,
            suffixes=("_without_new_point", "_with_new_point"),
        )
        paired["rmse_gain_from_new_point"] = paired["future_rmse_without_new_point"] - paired["future_rmse_with_new_point"]
        paired["mae_gain_from_new_point"] = paired["future_mae_without_new_point"] - paired["future_mae_with_new_point"]
        paired["future_window_matched"] = (
            np.isclose(paired["last_context_time_without_new_point"], paired["last_context_time_with_new_point"])
            & (paired["n_future_without_new_point"] == paired["n_future_with_new_point"])
        )
        for split_kind, group in paired.groupby("split_kind", dropna=False):
            rows.append(
                {
                    "split_kind": split_kind,
                    "added_point_index": int(k),
                    "baseline_policy_id": f"first{k - 1}_cut{k}",
                    "comparison_policy_id": f"first{k}",
                    "n_pairs": int(len(group)),
                    "median_rmse_gain_from_new_point": float(group["rmse_gain_from_new_point"].median()),
                    "mean_rmse_gain_from_new_point": float(group["rmse_gain_from_new_point"].mean()),
                    "median_mae_gain_from_new_point": float(group["mae_gain_from_new_point"].median()),
                    "fraction_curves_improved": float((group["rmse_gain_from_new_point"] > 0).mean()),
                    "all_future_windows_matched": bool(group["future_window_matched"].all()),
                    "headline_split": split_kind in STRICT_SPLITS,
                }
            )
    return pd.DataFrame(rows).sort_values(["split_kind", "added_point_index"]).reset_index(drop=True)


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    allowed_cols = {
        "future_r2",
        "median_future_r2",
        "gain_per_elapsed_day",
        "rank_by_gain_per_day",
    }
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        bad_mask = ~np.isfinite(values)
        bad = int(bad_mask.sum())
        if bad == 0:
            continue
        if col in allowed_cols or "_r2" in col:
            allowed[col] = bad
        else:
            unexpected[col] = bad
    return {
        "name": name,
        "rows": int(len(df)),
        "unexpected_nonfinite": json.dumps(unexpected, sort_keys=True),
        "allowed_nonfinite": json.dumps(allowed, sort_keys=True),
    }


def target_window_check(per_curve: pd.DataFrame) -> pd.DataFrame:
    work = per_curve.copy()
    violations = int((work["min_future_time"] <= work["last_context_time"] + 1e-12).sum())
    return pd.DataFrame(
        [
            {
                "name": "target_times_after_context",
                "rows": int(len(work)),
                "unexpected_nonfinite": json.dumps({"violations": violations} if violations else {}, sort_keys=True),
                "allowed_nonfinite": json.dumps({}, sort_keys=True),
            }
        ]
    )


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
    contrasts: pd.DataFrame,
    ranks: pd.DataFrame,
    marginal: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    headline_contrasts = contrasts[
        (contrasts["contrast_id"] == "static_to_measured_static_context")
        & (contrasts["headline_split"])
        & (contrasts["policy_kind"].isin(["cumulative", "single"]))
        & (contrasts["headline_policy"])
    ].copy()
    lines = [
        "# PLGA E3 Observed Timepoint Value",
        "",
        "Date: 2026-06-11",
        "",
        "## Scope",
        "",
        "This is E3 from `docs/plga_observation_budget_goal_design_2026-06-11.md`.",
        "It re-runs script 97's direct control with alternative observed-context policies.",
        "",
        "Default run uses strict splits only: source-group-kfold and source-dataset-lodo.",
        "",
        "## Headline Static To Measured Context Contrast",
        "",
        markdown_table(
            headline_contrasts,
            [
                "split_kind",
                "policy_kind",
                "policy_id",
                "n_context",
                "observed_index",
                "baseline_median_rmse",
                "comparison_median_rmse",
                "median_rmse_gain",
                "fraction_curves_improved",
                "median_last_context_time",
            ],
        ),
        "",
        "## Rank By Split",
        "",
        markdown_table(
            ranks,
            [
                "split_kind",
                "policy_kind",
                "policy_id",
                "rank_by_gain",
                "median_rmse_gain",
                "gain_per_context_point",
                "gain_per_elapsed_day",
            ],
        ),
        "",
        "## Marginal Value Of Added Cumulative Point",
        "",
        markdown_table(
            marginal[marginal["headline_split"]],
            [
                "split_kind",
                "added_point_index",
                "baseline_policy_id",
                "comparison_policy_id",
                "median_rmse_gain_from_new_point",
                "fraction_curves_improved",
                "all_future_windows_matched",
            ],
        ),
        "",
        "## Data Checks",
        "",
        markdown_table(checks, ["name", "rows", "unexpected_nonfinite", "allowed_nonfinite"]),
        "",
        "## Reading Rule",
        "",
        "- `firstK` means the first K observed release points are used as context.",
        "- `pointK` means only the K-th observed release point is used as context.",
        "- Static baselines are window-matched: target points are after the same context cutoff.",
        "- Marginal rows compare `firstK` against `first(K-1)_cutK`, so the future window is identical.",
        "",
    ]
    (out / "timepoint_value_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    splits_to_run = parse_splits(args.splits)
    args.out.mkdir(parents=True, exist_ok=True)

    per_curve = run_policies(args, splits_to_run)
    if per_curve.empty:
        raise RuntimeError("No per-curve metrics were produced")

    summary = summarize(per_curve)
    contrasts, paired = build_policy_contrasts(per_curve)
    ranks = build_rank_table(contrasts)
    marginal = build_marginal_value(per_curve)
    checks = pd.DataFrame(
        [
            finite_check("per_curve_metrics", per_curve),
            finite_check("summary_by_policy", summary),
            finite_check("policy_contrasts", contrasts),
            finite_check("paired_policy_deltas", paired),
            finite_check("timepoint_rank_by_split", ranks),
            finite_check("marginal_value", marginal),
        ]
    )
    checks = pd.concat([checks, target_window_check(per_curve)], ignore_index=True)
    if checks["unexpected_nonfinite"].ne("{}").any():
        raise ValueError(f"Unexpected data check issue:\n{checks.to_string(index=False)}")

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_policy.csv", index=False)
    contrasts.to_csv(args.out / "policy_contrasts.csv", index=False)
    paired.to_csv(args.out / "paired_policy_deltas.csv", index=False)
    ranks.to_csv(args.out / "timepoint_rank_by_split.csv", index=False)
    marginal.to_csv(args.out / "marginal_value.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "cli_args": args_to_metadata(args),
        "splits": splits_to_run,
        "strict_splits": STRICT_SPLITS,
        "max_index": args.max_index,
        "experiment_status": "E3_observed_timepoint_value",
        "policy_families": ["cumulative_firstK", "single_pointK", "marginal_ablation_firstK_minus_last"],
        "outputs": [
            "per_curve_metrics.csv",
            "summary_by_policy.csv",
            "policy_contrasts.csv",
            "paired_policy_deltas.csv",
            "timepoint_rank_by_split.csv",
            "marginal_value.csv",
            "data_checks.csv",
            "lock_metadata.json",
            "timepoint_value_report.md",
        ],
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.out, summary, contrasts, ranks, marginal, checks)

    print((args.out / "timepoint_value_report.md").resolve())
    headline = ranks[
        (ranks["contrast_id"] == "static_to_measured_static_context")
        & (ranks["policy_kind"].isin(["cumulative", "single"]))
    ]
    print(
        headline[
            [
                "split_kind",
                "policy_kind",
                "policy_id",
                "rank_by_gain",
                "median_rmse_gain",
                "gain_per_context_point",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
