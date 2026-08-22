"""109 - Liposome static whole-curve prediction E1.

Purpose:
    Test whether liposome IVR release curves can be predicted from static
    formulation / assay descriptors alone. This is the E1 gate after the
    candidate audit in script 108.

Consumes:
    outputs/149_release_caveat_augmented_cumulative_v1/curves_long.csv
    outputs/149_release_caveat_augmented_cumulative_v1/formulations.csv

Produces:
    outputs/109_liposome_static_whole_curve_prediction/
      static_summary_by_split.csv
      static_per_curve_metrics.csv
      static_feature_set_ablation.csv
      static_baseline_contrasts.csv
      data_checks.csv
      lock_metadata.json
      static_prediction_report.md

Interpretation:
    This script uses no early release observations as inputs. It is a
    static-descriptor ceiling probe for liposome, not a transfer model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_POOL = Path("outputs/149_release_caveat_augmented_cumulative_v1")
DEFAULT_OUT = Path("outputs/109_liposome_static_whole_curve_prediction")

SPLIT_AXES = ["API_name", "structure_type", "media_pH", "media_temp_oC", "release_method"]

FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "time_only": {"numeric": [], "categorical": []},
    "drug_api": {"numeric": ["Drug_Mw"], "categorical": ["API_name"]},
    "structure_medium": {
        "numeric": ["media_pH", "media_temp_oC"],
        "categorical": ["structure_type", "release_method"],
    },
    "colloid": {"numeric": ["Particle_Size", "PDI", "zeta_potential"], "categorical": []},
    "all_no_source": {
        "numeric": [
            "Drug_Mw",
            "Particle_Size",
            "PDI",
            "zeta_potential",
            "media_pH",
            "media_temp_oC",
        ],
        "categorical": ["API_name", "structure_type", "release_method"],
    },
}

PRIMARY_FEATURE_SET = "all_no_source"
EARLY_FORBIDDEN_COLUMNS = {
    "release_fraction",
    "release_percent",
    "release_raw",
    "sd_release_percent",
    "release_sd_percent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liposome E1 static whole-curve prediction.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--min-group-curves", type=int, default=5)
    parser.add_argument("--min-train-curves", type=int, default=30)
    parser.add_argument("--max-shape-nfev", type=int, default=5000)
    return parser.parse_args()


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def as_nonempty_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isna() | (out == "") | (out.str.lower() == "nan"))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def stable_seed_offset(text: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100_000)


def read_inputs(pool: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves_path = pool / "curves_long.csv"
    formulations_path = pool / "formulations.csv"
    if not curves_path.exists():
        raise FileNotFoundError(curves_path)
    if not formulations_path.exists():
        raise FileNotFoundError(formulations_path)
    curves = pd.read_csv(curves_path)
    formulations = pd.read_csv(formulations_path)
    required_curves = {"unified_curve_id", "time_days", "release_fraction"}
    required_formulations = {"unified_curve_id", "polymer_family"}
    missing_curves = sorted(required_curves - set(curves.columns))
    missing_formulations = sorted(required_formulations - set(formulations.columns))
    if missing_curves:
        raise ValueError(f"curves_long.csv missing required columns: {missing_curves}")
    if missing_formulations:
        raise ValueError(f"formulations.csv missing required columns: {missing_formulations}")
    return curves, formulations


def select_liposome(formulations: pd.DataFrame) -> pd.DataFrame:
    family = as_nonempty_string(formulations["polymer_family"])
    liposome = formulations.loc[family.str.contains("liposome", case=False, na=False)].copy()
    if liposome.empty:
        raise ValueError("No liposome rows found in formulations.csv")
    return liposome


def build_point_table(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    ids = set(formulations["unified_curve_id"].astype(str))
    points = curves[curves["unified_curve_id"].astype(str).isin(ids)].copy()
    points["time_days"] = pd.to_numeric(points["time_days"], errors="coerce")
    points["release_fraction_raw"] = pd.to_numeric(points["release_fraction"], errors="coerce")
    points = points[np.isfinite(points["time_days"]) & np.isfinite(points["release_fraction_raw"])].copy()
    points = points[points["time_days"] >= 0].copy()
    points["target_release_fraction"] = points["release_fraction_raw"].clip(0.0, 1.2)
    points["log1p_time_days"] = np.log1p(points["time_days"].to_numpy(dtype=float))
    meta_cols = sorted(set().union(*(set(v["numeric"]) | set(v["categorical"]) for v in FEATURE_SETS.values())))
    meta_cols += [axis for axis in SPLIT_AXES if axis not in meta_cols]
    meta_cols = ["unified_curve_id", *[col for col in meta_cols if col in formulations.columns]]
    merged = points.merge(formulations[meta_cols], on="unified_curve_id", how="left")
    for col in merged.columns:
        if col in {"unified_curve_id", "source_dataset"} or col in SPLIT_AXES:
            merged[col] = as_nonempty_string(merged[col])
    return merged


def group_values(curve_meta: pd.DataFrame, axis: str) -> pd.Series:
    if axis not in curve_meta.columns:
        return pd.Series(["__MISSING__"] * len(curve_meta), index=curve_meta.index, dtype="string")
    return as_nonempty_string(curve_meta[axis]).fillna("__MISSING__")


def build_splits(
    curve_meta: pd.DataFrame,
    axes: list[str],
    n_splits: int,
    min_group_curves: int,
    min_train_curves: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_defs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    curve_ids = curve_meta["unified_curve_id"].astype(str).to_numpy()
    n_curves = len(curve_ids)
    for axis in axes:
        groups = group_values(curve_meta, axis).astype(str).to_numpy()
        counts = pd.Series(groups).value_counts()
        eligible_groups = set(counts[counts >= min_group_curves].index.astype(str))
        keep_mask = np.asarray([group in eligible_groups for group in groups], dtype=bool)
        n_groups = len(eligible_groups)
        n_kept = int(keep_mask.sum())
        axis_ok = n_groups >= 2 and n_kept >= min_train_curves + min_group_curves
        audit_rows.append(
            {
                "split_kind": f"groupkfold_{axis}",
                "split_axis": axis,
                "n_total_curves": n_curves,
                "n_eligible_curves": n_kept,
                "n_total_groups": int(len(counts)),
                "n_eligible_groups": int(n_groups),
                "axis_feasible": bool(axis_ok),
                "notes": "" if axis_ok else "insufficient eligible groups or curves",
            }
        )
        if not axis_ok:
            continue
        kept_idx = np.flatnonzero(keep_mask)
        kept_groups = groups[kept_idx]
        n_eff = min(n_splits, n_groups)
        splitter = GroupKFold(n_splits=n_eff)
        for fold, (tr_local, te_local) in enumerate(splitter.split(kept_idx, groups=kept_groups)):
            train_idx = kept_idx[tr_local]
            test_idx = kept_idx[te_local]
            if len(train_idx) < min_train_curves:
                continue
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = sorted(train_groups & test_groups)
            split_defs.append(
                {
                    "split_kind": f"groupkfold_{axis}",
                    "split_axis": axis,
                    "fold": int(fold),
                    "train_curve_ids": set(curve_ids[train_idx]),
                    "test_curve_ids": set(curve_ids[test_idx]),
                    "train_groups": sorted(train_groups),
                    "test_groups": sorted(test_groups),
                    "group_overlap": overlap,
                }
            )
    return split_defs, audit_rows


def make_encoder(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", one_hot),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def feature_columns(feature_set: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    spec = FEATURE_SETS[feature_set]
    numeric = ["time_days", "log1p_time_days", *spec["numeric"]]
    categorical = list(spec["categorical"])
    numeric = [col for col in numeric if col in df.columns]
    categorical = [col for col in categorical if col in df.columns]
    return numeric, categorical


def prepare_feature_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in numeric:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical:
        out[col] = as_nonempty_string(df[col]).fillna("__MISSING__").astype(str)
    return out


def fit_direct_et(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    n_estimators: int,
    seed: int,
) -> np.ndarray:
    numeric, categorical = feature_columns(feature_set, train)
    model = Pipeline(
        [
            ("features", make_encoder(numeric, categorical)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    x_train = prepare_feature_frame(train, numeric, categorical)
    x_test = prepare_feature_frame(test, numeric, categorical)
    y_train = train["target_release_fraction"].to_numpy(dtype=float)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def median_curve_predict(train: pd.DataFrame, test: pd.DataFrame, group_col: str | None = None) -> np.ndarray:
    def curve_from(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        med = (
            sub.groupby("time_days", dropna=False)["target_release_fraction"]
            .median()
            .reset_index()
            .sort_values("time_days")
        )
        t = med["time_days"].to_numpy(dtype=float)
        q = med["target_release_fraction"].to_numpy(dtype=float)
        if len(t) == 0:
            return np.asarray([0.0]), np.asarray([0.0])
        unique_t, inverse = np.unique(t, return_inverse=True)
        if len(unique_t) != len(t):
            q_unique = np.asarray([np.median(q[inverse == i]) for i in range(len(unique_t))], dtype=float)
            return unique_t, q_unique
        return t, q

    global_t, global_q = curve_from(train)
    group_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if group_col is not None and group_col in train.columns and group_col in test.columns:
        for group, sub in train.groupby(as_nonempty_string(train[group_col]).fillna("__MISSING__")):
            group_curves[str(group)] = curve_from(sub)

    preds = []
    for row in test.itertuples(index=False):
        t_value = float(getattr(row, "time_days"))
        if group_col is None or group_col not in test.columns:
            t_grid, q_grid = global_t, global_q
        else:
            group_value = getattr(row, group_col)
            if pd.isna(group_value) or str(group_value).strip() == "":
                group_value = "__MISSING__"
            t_grid, q_grid = group_curves.get(str(group_value), (global_t, global_q))
        preds.append(np.interp(t_value, t_grid, q_grid, left=q_grid[0], right=q_grid[-1]))
    return np.asarray(preds, dtype=float)


def weibull_release(t: np.ndarray, qmax: float, tau: float, beta: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return qmax * (1.0 - np.exp(-((t / tau) ** beta)))


def fit_weibull_theta(sub: pd.DataFrame, max_nfev: int) -> np.ndarray | None:
    t = sub["time_days"].to_numpy(dtype=float)
    q = sub["target_release_fraction"].to_numpy(dtype=float)
    order = np.argsort(t)
    t = t[order]
    q = q[order]
    if len(np.unique(t)) < 4 or np.nanmax(q) <= 1e-6:
        return None
    p0 = [float(np.clip(np.nanmax(q), 0.05, 1.2)), float(np.clip(np.median(t[t > 0]), 1e-3, 10.0)), 1.0]
    try:
        theta, _ = curve_fit(
            weibull_release,
            t,
            q,
            p0=p0,
            bounds=([0.0, 1e-5, 0.1], [1.3, 50.0, 6.0]),
            max_nfev=max_nfev,
        )
    except Exception:
        return None
    if not np.all(np.isfinite(theta)):
        return None
    qmax, tau, beta = theta
    return np.asarray([qmax, np.log(tau), np.log(beta)], dtype=float)


def fit_shape_static_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    n_estimators: int,
    seed: int,
    max_nfev: int,
) -> np.ndarray:
    theta_rows = []
    train_curve_meta = train.drop_duplicates("unified_curve_id").set_index("unified_curve_id")
    for curve_id, sub in train.groupby("unified_curve_id"):
        theta = fit_weibull_theta(sub, max_nfev)
        if theta is None:
            continue
        row = train_curve_meta.loc[curve_id].copy()
        row["theta_qmax"] = theta[0]
        row["theta_log_tau"] = theta[1]
        row["theta_log_beta"] = theta[2]
        theta_rows.append(row)
    if len(theta_rows) < 10:
        return median_curve_predict(train, test)

    theta_df = pd.DataFrame(theta_rows).reset_index(drop=True)
    test_meta = test.drop_duplicates("unified_curve_id").copy()
    numeric, categorical = feature_columns(feature_set, theta_df)
    numeric = [col for col in numeric if col not in {"time_days", "log1p_time_days"}]
    model = Pipeline(
        [
            ("features", make_encoder(numeric, categorical)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    x_train = prepare_feature_frame(theta_df, numeric, categorical)
    y_train = theta_df[["theta_qmax", "theta_log_tau", "theta_log_beta"]].to_numpy(dtype=float)
    model.fit(x_train, y_train)
    x_test = prepare_feature_frame(test_meta, numeric, categorical)
    theta_pred = np.asarray(model.predict(x_test), dtype=float)
    theta_by_curve = {
        str(curve_id): theta_pred[i]
        for i, curve_id in enumerate(test_meta["unified_curve_id"].astype(str).to_numpy())
    }
    preds = []
    for row in test.itertuples(index=False):
        theta = theta_by_curve[str(getattr(row, "unified_curve_id"))]
        qmax = float(np.clip(theta[0], 0.0, 1.3))
        tau = float(np.exp(np.clip(theta[1], np.log(1e-5), np.log(50.0))))
        beta = float(np.exp(np.clip(theta[2], np.log(0.1), np.log(6.0))))
        preds.append(weibull_release(np.asarray([float(getattr(row, "time_days"))]), qmax, tau, beta)[0])
    return np.asarray(preds, dtype=float)


def enforce_curve_shape(df: pd.DataFrame, pred_col: str = "y_pred") -> pd.DataFrame:
    out = df.copy()
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce").clip(0.0, 1.2)
    shaped_parts = []
    for _, sub in out.groupby("unified_curve_id", sort=False):
        sub = sub.sort_values("time_days").copy()
        sub[pred_col] = np.maximum.accumulate(sub[pred_col].to_numpy(dtype=float))
        shaped_parts.append(sub)
    return pd.concat(shaped_parts, ignore_index=True)


def metric_row(df: pd.DataFrame) -> dict[str, float]:
    y = df["y_true"].to_numpy(dtype=float)
    pred = df["y_pred"].to_numpy(dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": safe_r2(y, pred),
        "n_points": int(len(df)),
    }


def evaluate_predictions(pred: pd.DataFrame, split_info: dict[str, Any], method: str, feature_set: str) -> pd.DataFrame:
    pred = enforce_curve_shape(pred)
    rows = []
    for curve_id, sub in pred.groupby("unified_curve_id", sort=False):
        metrics = metric_row(sub)
        meta = sub.iloc[0]
        rows.append(
            {
                "split_kind": split_info["split_kind"],
                "split_axis": split_info["split_axis"],
                "fold": split_info["fold"],
                "method": method,
                "feature_set": feature_set,
                "unified_curve_id": curve_id,
                "n_points": metrics["n_points"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "API_name": meta.get("API_name", np.nan),
                "structure_type": meta.get("structure_type", np.nan),
                "media_pH": meta.get("media_pH", np.nan),
                "media_temp_oC": meta.get("media_temp_oC", np.nan),
                "release_method": meta.get("release_method", np.nan),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(test: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unified_curve_id": test["unified_curve_id"].astype(str).to_numpy(),
            "time_days": test["time_days"].to_numpy(dtype=float),
            "y_true": test["target_release_fraction"].to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "API_name": test["API_name"].to_numpy() if "API_name" in test.columns else np.nan,
            "structure_type": test["structure_type"].to_numpy() if "structure_type" in test.columns else np.nan,
            "media_pH": test["media_pH"].to_numpy() if "media_pH" in test.columns else np.nan,
            "media_temp_oC": test["media_temp_oC"].to_numpy() if "media_temp_oC" in test.columns else np.nan,
            "release_method": test["release_method"].to_numpy() if "release_method" in test.columns else np.nan,
        }
    )


def summarize_per_curve(per_curve: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    point_meta = points[["unified_curve_id", "target_release_fraction"]].rename(
        columns={"target_release_fraction": "y_true"}
    )
    for keys, sub in per_curve.groupby(["split_kind", "split_axis", "method", "feature_set"], dropna=False):
        split_kind, split_axis, method, feature_set = keys
        curve_ids = set(sub["unified_curve_id"].astype(str))
        pooled = points[points["unified_curve_id"].astype(str).isin(curve_ids)].copy()
        # Pooled values are recomputed from curve-level RMSE approximately only
        # where point predictions are not retained; point-level summaries are
        # represented by weighted curve metrics.
        rows.append(
            {
                "split_kind": split_kind,
                "split_axis": split_axis,
                "method": method,
                "feature_set": feature_set,
                "n_curves": int(sub["unified_curve_id"].nunique()),
                "n_points": int(sub["n_points"].sum()),
                "mean_curve_rmse": float(sub["rmse"].mean()),
                "median_curve_rmse": float(sub["rmse"].median()),
                "mean_curve_mae": float(sub["mae"].mean()),
                "median_curve_mae": float(sub["mae"].median()),
                "weighted_curve_rmse": float(
                    np.sqrt(np.average(sub["rmse"].to_numpy(dtype=float) ** 2, weights=sub["n_points"]))
                ),
                "weighted_curve_mae": float(
                    np.average(sub["mae"].to_numpy(dtype=float), weights=sub["n_points"])
                ),
                "mean_curve_r2": float(sub["r2"].replace([np.inf, -np.inf], np.nan).mean()),
                "pooled_target_sd": float(pooled["target_release_fraction"].std()) if len(pooled) else np.nan,
                "point_metric_note": "weighted curve metrics; point predictions are not stored",
            }
        )
    return pd.DataFrame(rows)


def feature_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    model_rows = summary[summary["method"] == "direct_et_point"].copy()
    base = model_rows[model_rows["feature_set"] == "time_only"][
        ["split_kind", "median_curve_rmse", "weighted_curve_rmse"]
    ].rename(
        columns={
            "median_curve_rmse": "time_only_median_curve_rmse",
            "weighted_curve_rmse": "time_only_weighted_curve_rmse",
        }
    )
    merged = model_rows.merge(base, on="split_kind", how="left")
    merged["median_rmse_gain_vs_time_only"] = (
        merged["time_only_median_curve_rmse"] - merged["median_curve_rmse"]
    )
    merged["weighted_rmse_gain_vs_time_only"] = (
        merged["time_only_weighted_curve_rmse"] - merged["weighted_curve_rmse"]
    )
    return merged.sort_values(["split_kind", "median_rmse_gain_vs_time_only"], ascending=[True, False])


def baseline_contrasts(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = per_curve[
        (per_curve["method"] == "global_median_curve") & (per_curve["feature_set"] == "none")
    ].copy()
    base_cols = ["split_kind", "fold", "unified_curve_id", "rmse", "mae"]
    baseline = baseline[base_cols].rename(columns={"rmse": "baseline_rmse", "mae": "baseline_mae"})
    for keys, sub in per_curve.groupby(["split_kind", "split_axis", "method", "feature_set"], dropna=False):
        split_kind, split_axis, method, feature_set = keys
        if method == "global_median_curve" and feature_set == "none":
            continue
        paired = sub.merge(baseline, on=["split_kind", "fold", "unified_curve_id"], how="inner")
        if paired.empty:
            continue
        rows.append(
            {
                "split_kind": split_kind,
                "split_axis": split_axis,
                "method": method,
                "feature_set": feature_set,
                "baseline_method": "global_median_curve",
                "n_paired_curves": int(len(paired)),
                "baseline_median_rmse": float(paired["baseline_rmse"].median()),
                "comparison_median_rmse": float(paired["rmse"].median()),
                "median_rmse_gain": float(paired["baseline_rmse"].median() - paired["rmse"].median()),
                "baseline_mean_rmse": float(paired["baseline_rmse"].mean()),
                "comparison_mean_rmse": float(paired["rmse"].mean()),
                "mean_rmse_gain": float(paired["baseline_rmse"].mean() - paired["rmse"].mean()),
                "median_mae_gain": float(paired["baseline_mae"].median() - paired["mae"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split_kind", "median_rmse_gain"], ascending=[True, False])


def data_checks(
    points: pd.DataFrame,
    split_defs: list[dict[str, Any]],
    per_curve: pd.DataFrame,
    used_feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    forbidden_used = sorted(
        col
        for cols in used_feature_columns.values()
        for col in cols
        if col in EARLY_FORBIDDEN_COLUMNS
    )
    overlap_count = sum(1 for split in split_defs if split["group_overlap"])
    nonfinite_metrics = int(
        per_curve[["rmse", "mae"]].replace([np.inf, -np.inf], np.nan).isna().sum().sum()
    )
    checks = [
        {
            "check": "finite_point_rows",
            "status": "pass" if len(points) > 0 else "fail",
            "value": f"n_points={len(points)}; n_curves={points['unified_curve_id'].nunique()}",
            "details": "{}",
        },
        {
            "check": "no_heldout_group_in_training_fold",
            "status": "pass" if overlap_count == 0 else "fail",
            "value": f"overlap_folds={overlap_count}",
            "details": json.dumps(
                [
                    {
                        "split_kind": split["split_kind"],
                        "fold": split["fold"],
                        "overlap": split["group_overlap"],
                    }
                    for split in split_defs
                    if split["group_overlap"]
                ]
            ),
        },
        {
            "check": "no_release_observation_inputs",
            "status": "pass" if not forbidden_used else "fail",
            "value": ",".join(forbidden_used) if forbidden_used else "none",
            "details": json.dumps(used_feature_columns),
        },
        {
            "check": "no_nonfinite_metrics",
            "status": "pass" if nonfinite_metrics == 0 else "fail",
            "value": f"nonfinite_metric_cells={nonfinite_metrics}",
            "details": "{}",
        },
        {
            "check": "outputs_have_all_required_methods",
            "status": "pass"
            if {"global_median_curve", "api_local_median_curve", "direct_et_point", "weibull_static_theta"}
            <= set(per_curve["method"])
            else "fail",
            "value": ",".join(sorted(per_curve["method"].unique())),
            "details": "{}",
        },
    ]
    return pd.DataFrame(checks)


def write_report(
    out: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    checks: pd.DataFrame,
    split_audit: pd.DataFrame,
) -> None:
    primary = summary[
        (summary["method"] == "direct_et_point") & (summary["feature_set"] == PRIMARY_FEATURE_SET)
    ].copy()
    base = summary[summary["method"] == "global_median_curve"].copy()
    shape = summary[summary["method"] == "weibull_static_theta"].copy()
    best_rows = (
        summary.sort_values("median_curve_rmse")
        .groupby("split_kind", as_index=False)
        .first()[["split_kind", "method", "feature_set", "median_curve_rmse", "weighted_curve_rmse"]]
    )
    static_success = bool((primary["median_curve_rmse"] <= 0.12).all()) if not primary.empty else False
    partial_success = bool((primary["median_curve_rmse"] <= 0.18).any()) if not primary.empty else False
    decision = "static_success" if static_success else "static_partial" if partial_success else "static_failure"
    next_action = (
        "Treat liposome as a static-descriptor contrast case before early-observation work."
        if decision == "static_success"
        else "Run E2 bottleneck decomposition before any early-observation or neural-model work."
    )

    lines = [
        "# Liposome Static Whole-Curve Prediction E1",
        "",
        "Date: 2026-06-12",
        "",
        "## Decision",
        "",
        f"E1 decision: **{decision}**.",
        "",
        next_action,
        "",
        "Threshold note: this first E1 report uses median per-curve RMSE <= 0.12",
        "across strict splits as a strong static-success heuristic, and <= 0.18 on",
        "at least one strict split as partial success.",
        "",
        "## Primary Static Model",
        "",
    ]
    if primary.empty:
        lines.append("No primary static model rows were produced.")
    else:
        lines.extend(
            [
                "| Split | Median RMSE | Weighted RMSE | Median MAE | n curves |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in primary.itertuples(index=False):
            lines.append(
                f"| {row.split_kind} | {row.median_curve_rmse:.3f} | "
                f"{row.weighted_curve_rmse:.3f} | {row.median_curve_mae:.3f} | {int(row.n_curves)} |"
            )
    lines.extend(["", "## Baseline View", ""])
    for label, table in [("Global median", base), ("Weibull static theta", shape)]:
        lines.append(f"### {label}")
        if table.empty:
            lines.append("")
            lines.append("No rows.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Split | Median RMSE | Weighted RMSE | n curves |")
        lines.append("|---|---:|---:|---:|")
        for row in table.itertuples(index=False):
            lines.append(
                f"| {row.split_kind} | {row.median_curve_rmse:.3f} | "
                f"{row.weighted_curve_rmse:.3f} | {int(row.n_curves)} |"
            )
        lines.append("")
    lines.extend(["## Best Rows By Split", "", "| Split | Method | Feature set | Median RMSE | Weighted RMSE |", "|---|---|---|---:|---:|"])
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"| {row.split_kind} | {row.method} | {row.feature_set} | "
            f"{row.median_curve_rmse:.3f} | {row.weighted_curve_rmse:.3f} |"
        )
    lines.extend(["", "## Verification", ""])
    for row in checks.itertuples(index=False):
        lines.append(f"- `{row.check}`: {row.status} ({row.value})")
    feasible = split_audit[split_audit["axis_feasible"]]
    lines.extend(
        [
            "",
            "## Split Audit",
            "",
            f"Feasible axes used: `{', '.join(feasible['split_axis'].astype(str))}`.",
            "",
            "Source split is still not a headline split for this corpus.",
            "",
            "## Next Step",
            "",
            "If E1 is not a strong static success, run E2 bottleneck decomposition:",
            "group residuals by API, structure, pH, temperature, and release method; then",
            "decide whether early release observations are justified.",
            "",
        ]
    )
    (out / "static_prediction_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    curves, formulations = read_inputs(args.pool)
    liposome_formulations = select_liposome(formulations)
    points = build_point_table(curves, liposome_formulations)
    curve_meta = points.drop_duplicates("unified_curve_id").copy()
    split_defs, split_audit_rows = build_splits(
        curve_meta,
        SPLIT_AXES,
        args.n_splits,
        args.min_group_curves,
        args.min_train_curves,
    )
    if not split_defs:
        raise RuntimeError("No feasible liposome E1 splits were built.")

    used_feature_columns: dict[str, list[str]] = {}
    per_curve_parts: list[pd.DataFrame] = []
    for split in split_defs:
        train = points[points["unified_curve_id"].astype(str).isin(split["train_curve_ids"])].copy()
        test = points[points["unified_curve_id"].astype(str).isin(split["test_curve_ids"])].copy()
        fold_seed = args.seed + 1000 * int(split["fold"]) + stable_seed_offset(split["split_kind"])

        for method, feature_set, pred in [
            ("global_median_curve", "none", median_curve_predict(train, test)),
            ("api_local_median_curve", "API_name", median_curve_predict(train, test, "API_name")),
        ]:
            frame = prediction_frame(test, pred)
            per_curve_parts.append(evaluate_predictions(frame, split, method, feature_set))

        for feature_set in FEATURE_SETS:
            numeric, categorical = feature_columns(feature_set, points)
            used_feature_columns[f"direct_et_point:{feature_set}"] = numeric + categorical
            pred = fit_direct_et(
                train,
                test,
                feature_set,
                args.n_estimators,
                fold_seed + len(feature_set),
            )
            frame = prediction_frame(test, pred)
            per_curve_parts.append(evaluate_predictions(frame, split, "direct_et_point", feature_set))

        shape_feature_set = PRIMARY_FEATURE_SET
        numeric, categorical = feature_columns(shape_feature_set, points)
        used_feature_columns[f"weibull_static_theta:{shape_feature_set}"] = [
            col for col in numeric + categorical if col not in {"time_days", "log1p_time_days"}
        ]
        pred = fit_shape_static_prior(
            train,
            test,
            shape_feature_set,
            args.n_estimators,
            fold_seed + 101,
            args.max_shape_nfev,
        )
        frame = prediction_frame(test, pred)
        per_curve_parts.append(evaluate_predictions(frame, split, "weibull_static_theta", shape_feature_set))

    per_curve = pd.concat(per_curve_parts, ignore_index=True)
    summary = summarize_per_curve(per_curve, points)
    ablation = feature_ablation(summary)
    contrasts = baseline_contrasts(per_curve)
    split_audit = pd.DataFrame(split_audit_rows)
    checks = data_checks(points, split_defs, per_curve, used_feature_columns)

    summary.to_csv(out / "static_summary_by_split.csv", index=False)
    per_curve.to_csv(out / "static_per_curve_metrics.csv", index=False)
    ablation.to_csv(out / "static_feature_set_ablation.csv", index=False)
    contrasts.to_csv(out / "static_baseline_contrasts.csv", index=False)
    checks.to_csv(out / "data_checks.csv", index=False)
    split_audit.to_csv(out / "split_audit.csv", index=False)
    write_report(out, summary, contrasts, checks, split_audit)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {
            "pool": str(args.pool),
            "out": str(args.out),
            "seed": args.seed,
            "n_splits": args.n_splits,
            "n_estimators": args.n_estimators,
            "min_group_curves": args.min_group_curves,
            "min_train_curves": args.min_train_curves,
            "max_shape_nfev": args.max_shape_nfev,
        },
        "inputs": {
            "curves_long": file_meta(args.pool / "curves_long.csv"),
            "formulations": file_meta(args.pool / "formulations.csv"),
        },
        "feature_sets": FEATURE_SETS,
        "split_axes": SPLIT_AXES,
        "row_counts": {
            "point_rows": int(len(points)),
            "curve_ids": int(points["unified_curve_id"].nunique()),
            "split_folds": int(len(split_defs)),
            "per_curve_metric_rows": int(len(per_curve)),
        },
        "data_check_status": checks[["check", "status", "value"]].to_dict(orient="records"),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
