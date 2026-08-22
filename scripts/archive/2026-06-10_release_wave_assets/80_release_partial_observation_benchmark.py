"""
80 - Shared benchmark API for partially observed drug-release forecasting.

Purpose:
    Define a mechanism-agnostic benchmark contract for release datasets and
    prediction tables. This is the common evaluation layer that lets PLGA,
    chitosan, liposome, and future mechanisms report into one schema.

Consumes:
    curves_long.csv
        Long-form observed curves. Required columns, with aliases accepted:
            curve_id / (formulation + drug fallback)
            t / t_hours / time_h / t_days / time_days
            Q_obs / Q_observed / release / q
        Extra columns are preserved as per-curve metadata when constant within
        a curve.

    predictions.csv (optional)
        Long-form forecast table. Required columns, with aliases accepted:
            curve_id / (formulation + drug fallback)
            t / t_hours / time_h / t_days / time_days
            Q_pred / Q_predicted_point / prediction
        Optional uncertainty columns:
            Q_lo90, Q_hi90, Q_lo50, Q_hi50

    metadata.csv (optional)
        One row per curve_id with benchmark grouping fields such as drug,
        polymer, material, or assay method.

    mechanism_metadata.json (optional)
        Free-form benchmark context copied into split_metadata.json.

Produces:
    outputs/.../curves_standardized.csv
    outputs/.../prediction_trajectories.csv
    outputs/.../metrics_by_curve.csv
    outputs/.../metrics_summary.csv
    outputs/.../timescale_by_curve.csv
    outputs/.../timescale_summary.csv
    outputs/.../uq_summary.csv
    outputs/.../fold_assignments.csv
    outputs/.../split_metadata.json

Expected runtime:
    Seconds for table-level evaluation; no model fitting happens here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


CURVE_ID_ALIASES = ("curve_id", "fid", "id", "sample_id", "curve", "sample")
FORMULATION_ALIASES = ("formulation", "formulation_id", "formula")
DRUG_ALIASES = ("drug", "drug_name", "api")
TIME_ALIASES = ("t", "time", "t_hours", "time_h", "time_hours", "t_days", "time_days")
OBS_ALIASES = ("q_obs", "q_observed", "q", "q_release", "release", "release_pct", "observed")
PRED_ALIASES = ("q_pred", "q_predicted_point", "prediction", "pred", "yhat")
LO90_ALIASES = ("q_lo90", "lo90", "pred_lo90", "q_lower_90")
HI90_ALIASES = ("q_hi90", "hi90", "pred_hi90", "q_upper_90")
LO50_ALIASES = ("q_lo50", "lo50", "pred_lo50", "q_lower_50")
HI50_ALIASES = ("q_hi50", "hi50", "pred_hi50", "q_upper_50")


@dataclass(frozen=True)
class StandardizeConfig:
    time_unit: str
    dataset_name: str
    mechanism_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves-long", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--mechanism-metadata-json", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="release_benchmark")
    parser.add_argument("--mechanism-id", type=str, default="unknown")
    parser.add_argument(
        "--time-unit",
        choices=("auto", "hours", "days", "as_is"),
        default="auto",
        help="Convert detected day/hour columns to a common unit if requested.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("none", "random_kfold", "group_kfold"),
        default="none",
    )
    parser.add_argument("--group-col", type=str, default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.5,0.8",
        help="Comma-separated cumulative-release thresholds for timescale metrics.",
    )
    parser.add_argument(
        "--q-floor-mape",
        type=float,
        default=0.05,
        help="Ignore near-zero observed release values below this floor for MAPE.",
    )
    return parser.parse_args()


def _lower_map(columns: list[str]) -> dict[str, str]:
    return {col.lower().strip(): col for col in columns}


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lower_map = _lower_map(columns)
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _normalize_release(values: pd.Series) -> pd.Series:
    out = values.astype(float)
    finite = np.isfinite(out.to_numpy(dtype=float))
    if finite.any() and float(out[finite].max()) > 2.0:
        out = out / 100.0
    return out.clip(0.0, 1.2)


def _convert_time(values: pd.Series, source_col: str, time_unit: str) -> tuple[pd.Series, str]:
    source = source_col.lower().strip()
    out = values.astype(float)
    if time_unit == "as_is":
        return out, "as_is"
    if source.endswith("_days") or source == "t_days" or source == "time_days":
        if time_unit in ("auto", "hours"):
            return out * 24.0, "hours"
        return out, "days"
    if source.endswith("_hours") or source == "t_hours" or source == "time_h" or source == "time_hours":
        if time_unit in ("auto", "days"):
            return out / 24.0, "days"
        return out, "hours"
    return out, ("hours" if time_unit == "hours" else "days" if time_unit == "days" else "as_is")


def _build_curve_id(df: pd.DataFrame) -> pd.Series:
    curve_id_col = _pick_column(list(df.columns), CURVE_ID_ALIASES)
    if curve_id_col is not None:
        return df[curve_id_col].astype(str).str.strip()

    formulation_col = _pick_column(list(df.columns), FORMULATION_ALIASES)
    drug_col = _pick_column(list(df.columns), DRUG_ALIASES)
    if formulation_col is None or drug_col is None:
        raise KeyError(
            "Need either curve_id-like column or both formulation and drug columns "
            f"to define a curve key. Found columns: {list(df.columns)}"
        )
    return (
        df[formulation_col].astype(str).str.strip()
        + "::"
        + df[drug_col].astype(str).str.strip()
    )


def _collect_constant_metadata(raw: pd.DataFrame, curve_id: pd.Series, exclude: set[str]) -> pd.DataFrame:
    meta = raw.copy()
    meta["curve_id"] = curve_id
    if meta.columns[0] != "curve_id":
        ordered = ["curve_id"] + [col for col in meta.columns if col != "curve_id"]
        meta = meta[ordered]
    keep_cols: list[str] = []
    for col in meta.columns:
        if col in exclude or col == "curve_id":
            continue
        nunique = meta.groupby("curve_id")[col].nunique(dropna=False)
        if int(nunique.max()) <= 1:
            keep_cols.append(col)
    if not keep_cols:
        return meta[["curve_id"]].drop_duplicates().reset_index(drop=True)
    return (
        meta[["curve_id"] + keep_cols]
        .drop_duplicates(subset=["curve_id"])
        .reset_index(drop=True)
    )


def standardize_curves(raw: pd.DataFrame, cfg: StandardizeConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    curve_id = _build_curve_id(raw)
    time_col = _pick_column(list(raw.columns), TIME_ALIASES)
    obs_col = _pick_column(list(raw.columns), OBS_ALIASES)
    if time_col is None or obs_col is None:
        raise KeyError(
            "Observed curves need time and release columns. "
            f"Found columns: {list(raw.columns)}"
        )

    t_std, resolved_time_unit = _convert_time(raw[time_col], time_col, cfg.time_unit)
    out = pd.DataFrame(
        {
            "curve_id": curve_id,
            "t": t_std,
            "Q_obs": _normalize_release(raw[obs_col]),
            "dataset_name": cfg.dataset_name,
            "mechanism_id": cfg.mechanism_id,
        }
    )
    out = (
        out.dropna(subset=["curve_id", "t", "Q_obs"])
        .groupby(["curve_id", "t", "dataset_name", "mechanism_id"], as_index=False)
        .agg({"Q_obs": "mean"})
        .sort_values(["curve_id", "t"])
        .reset_index(drop=True)
    )

    metadata = _collect_constant_metadata(
        raw=raw,
        curve_id=curve_id,
        exclude={time_col, obs_col},
    )
    return out, metadata, resolved_time_unit


def standardize_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    curve_id = _build_curve_id(raw)
    meta = raw.copy()
    meta["curve_id"] = curve_id
    if meta.columns[0] != "curve_id":
        ordered = ["curve_id"] + [col for col in meta.columns if col != "curve_id"]
        meta = meta[ordered]
    meta = meta.drop_duplicates(subset=["curve_id"]).reset_index(drop=True)
    return meta


def standardize_predictions(raw: pd.DataFrame, cfg: StandardizeConfig) -> tuple[pd.DataFrame, str]:
    curve_id = _build_curve_id(raw)
    time_col = _pick_column(list(raw.columns), TIME_ALIASES)
    pred_col = _pick_column(list(raw.columns), PRED_ALIASES)
    if time_col is None or pred_col is None:
        raise KeyError(
            "Prediction table needs time and point-prediction columns. "
            f"Found columns: {list(raw.columns)}"
        )

    t_std, resolved_time_unit = _convert_time(raw[time_col], time_col, cfg.time_unit)
    out = pd.DataFrame(
        {
            "curve_id": curve_id,
            "t": t_std,
            "Q_pred": _normalize_release(raw[pred_col]),
            "dataset_name": cfg.dataset_name,
            "mechanism_id": cfg.mechanism_id,
        }
    )

    alias_map = {
        "Q_lo90": LO90_ALIASES,
        "Q_hi90": HI90_ALIASES,
        "Q_lo50": LO50_ALIASES,
        "Q_hi50": HI50_ALIASES,
    }
    for out_col, aliases in alias_map.items():
        src = _pick_column(list(raw.columns), aliases)
        if src is not None:
            out[out_col] = _normalize_release(raw[src])

    out = (
        out.dropna(subset=["curve_id", "t", "Q_pred"])
        .groupby(["curve_id", "t", "dataset_name", "mechanism_id"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["curve_id", "t"])
        .reset_index(drop=True)
    )
    return out, resolved_time_unit


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y = y_true[mask]
    yh = y_pred[mask]
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, q_floor: float) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= q_floor)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    return float(np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])))


def _interp_column(t_src: np.ndarray, y_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    return np.interp(t_target, t_src, y_src, left=y_src[0], right=y_src[-1])


def merge_predictions(curves: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pred_cols = [col for col in preds.columns if col.startswith("Q_")]
    curve_ids = sorted(curves["curve_id"].unique())
    pred_ids = set(preds["curve_id"].unique())
    missing = [curve_id for curve_id in curve_ids if curve_id not in pred_ids]
    if missing:
        raise KeyError(f"Missing predictions for curves: {missing[:10]}")

    for curve_id, curve_sub in curves.groupby("curve_id", sort=True):
        pred_sub = preds[preds["curve_id"] == curve_id].sort_values("t")
        curve_sub = curve_sub.sort_values("t").copy()
        t_curve = curve_sub["t"].to_numpy(dtype=float)
        t_pred = pred_sub["t"].to_numpy(dtype=float)
        if len(t_pred) == 0:
            raise ValueError(f"No prediction rows for curve_id={curve_id}")
        for col in pred_cols:
            curve_sub[col] = _interp_column(
                t_src=t_pred,
                y_src=pred_sub[col].to_numpy(dtype=float),
                t_target=t_curve,
            )
        rows.append(curve_sub)

    merged = pd.concat(rows, ignore_index=True)
    if {"Q_lo90", "Q_hi90"}.issubset(merged.columns):
        merged["inside90"] = (
            (merged["Q_obs"] >= merged["Q_lo90"]) & (merged["Q_obs"] <= merged["Q_hi90"])
        )
        merged["pi_width90"] = merged["Q_hi90"] - merged["Q_lo90"]
    if {"Q_lo50", "Q_hi50"}.issubset(merged.columns):
        merged["inside50"] = (
            (merged["Q_obs"] >= merged["Q_lo50"]) & (merged["Q_obs"] <= merged["Q_hi50"])
        )
        merged["pi_width50"] = merged["Q_hi50"] - merged["Q_lo50"]
    merged["abs_error"] = np.abs(merged["Q_obs"] - merged["Q_pred"])
    return merged.sort_values(["curve_id", "t"]).reset_index(drop=True)


def _time_to_threshold(t: np.ndarray, q: np.ndarray, target: float) -> float:
    mask = np.isfinite(t) & np.isfinite(q)
    if mask.sum() < 2:
        return float("nan")
    t_use = t[mask]
    q_use = np.maximum.accumulate(np.clip(q[mask], 0.0, 1.2))
    if q_use.max() < target:
        return float("nan")
    hit = int(np.argmax(q_use >= target))
    if q_use[hit] < target:
        return float("nan")
    if hit == 0:
        return float(t_use[0])
    q0, q1 = q_use[hit - 1], q_use[hit]
    t0, t1 = t_use[hit - 1], t_use[hit]
    if q1 <= q0:
        return float(t1)
    frac = float((target - q0) / (q1 - q0))
    return float(t0 + frac * (t1 - t0))


def build_timescale_tables(merged: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for curve_id, sub in merged.groupby("curve_id", sort=True):
        t = sub["t"].to_numpy(dtype=float)
        q_obs = sub["Q_obs"].to_numpy(dtype=float)
        q_pred = sub["Q_pred"].to_numpy(dtype=float)
        for target in thresholds:
            t_obs = _time_to_threshold(t, q_obs, target)
            t_pred = _time_to_threshold(t, q_pred, target)
            curve_rows.append(
                {
                    "curve_id": curve_id,
                    "threshold": target,
                    "t_obs": t_obs,
                    "t_pred": t_pred,
                    "signed_error": t_pred - t_obs if np.isfinite(t_obs) and np.isfinite(t_pred) else np.nan,
                    "abs_error": abs(t_pred - t_obs) if np.isfinite(t_obs) and np.isfinite(t_pred) else np.nan,
                }
            )

    curve_table = pd.DataFrame(curve_rows)
    if curve_table.empty:
        return curve_table, pd.DataFrame()

    for target, sub in curve_table.groupby("threshold", sort=True):
        err = sub["signed_error"].to_numpy(dtype=float)
        abs_err = sub["abs_error"].to_numpy(dtype=float)
        mask = np.isfinite(err)
        summary_rows.append(
            {
                "threshold": target,
                "n_valid": int(mask.sum()),
                "mae_t": float(np.nanmean(abs_err)) if mask.any() else np.nan,
                "rmse_t": float(np.sqrt(np.nanmean(err[mask] ** 2))) if mask.any() else np.nan,
                "median_abs_error_t": float(np.nanmedian(abs_err)) if mask.any() else np.nan,
                "bias_t": float(np.nanmean(err)) if mask.any() else np.nan,
            }
        )
    return curve_table, pd.DataFrame(summary_rows)


def build_curve_metrics(merged: pd.DataFrame, q_floor_mape: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for curve_id, sub in merged.groupby("curve_id", sort=True):
        q_obs = sub["Q_obs"].to_numpy(dtype=float)
        q_pred = sub["Q_pred"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "curve_id": curve_id,
            "n_points": int(len(sub)),
            "t_max": float(sub["t"].max()),
            "Q_obs_final": float(sub["Q_obs"].iloc[-1]),
            "Q_pred_final": float(sub["Q_pred"].iloc[-1]),
            "r2": _r2(q_obs, q_pred),
            "rmse": _rmse(q_obs, q_pred),
            "mae": _mae(q_obs, q_pred),
            "mape": _mape(q_obs, q_pred, q_floor=q_floor_mape),
        }
        if "inside90" in sub.columns:
            row["cov90"] = float(np.nanmean(sub["inside90"].to_numpy(dtype=float)))
            row["pi_width90_mean"] = float(np.nanmean(sub["pi_width90"].to_numpy(dtype=float)))
        if "inside50" in sub.columns:
            row["cov50"] = float(np.nanmean(sub["inside50"].to_numpy(dtype=float)))
            row["pi_width50_mean"] = float(np.nanmean(sub["pi_width50"].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(
    merged: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    cfg: StandardizeConfig,
    time_unit: str,
    thresholds: list[float],
    q_floor_mape: float,
) -> pd.DataFrame:
    q_obs = merged["Q_obs"].to_numpy(dtype=float)
    q_pred = merged["Q_pred"].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "dataset_name": cfg.dataset_name,
        "mechanism_id": cfg.mechanism_id,
        "time_unit": time_unit,
        "n_curves": int(merged["curve_id"].nunique()),
        "n_points": int(len(merged)),
        "pooled_r2": _r2(q_obs, q_pred),
        "pooled_rmse": _rmse(q_obs, q_pred),
        "pooled_mae": _mae(q_obs, q_pred),
        "pooled_mape": _mape(q_obs, q_pred, q_floor=q_floor_mape),
        "median_curve_r2": float(curve_metrics["r2"].median()) if not curve_metrics.empty else np.nan,
        "frac_curve_r2_ge0": float((curve_metrics["r2"] >= 0.0).mean()) if not curve_metrics.empty else np.nan,
        "timescale_thresholds": ",".join(str(x) for x in thresholds),
    }
    if {"Q_lo90", "Q_hi90"}.issubset(merged.columns):
        summary["cov90"] = _coverage(
            y_true=q_obs,
            lo=merged["Q_lo90"].to_numpy(dtype=float),
            hi=merged["Q_hi90"].to_numpy(dtype=float),
        )
        summary["pi_width90_mean"] = float(np.nanmean(merged["pi_width90"].to_numpy(dtype=float)))
    if {"Q_lo50", "Q_hi50"}.issubset(merged.columns):
        summary["cov50"] = _coverage(
            y_true=q_obs,
            lo=merged["Q_lo50"].to_numpy(dtype=float),
            hi=merged["Q_hi50"].to_numpy(dtype=float),
        )
        summary["pi_width50_mean"] = float(np.nanmean(merged["pi_width50"].to_numpy(dtype=float)))
    return pd.DataFrame([summary])


def build_uq_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if {"Q_lo90", "Q_hi90"}.issubset(merged.columns):
        rows.append(
            {
                "level": 0.90,
                "coverage": _coverage(
                    merged["Q_obs"].to_numpy(dtype=float),
                    merged["Q_lo90"].to_numpy(dtype=float),
                    merged["Q_hi90"].to_numpy(dtype=float),
                ),
                "mean_width": float(np.nanmean(merged["pi_width90"].to_numpy(dtype=float))),
            }
        )
    if {"Q_lo50", "Q_hi50"}.issubset(merged.columns):
        rows.append(
            {
                "level": 0.50,
                "coverage": _coverage(
                    merged["Q_obs"].to_numpy(dtype=float),
                    merged["Q_lo50"].to_numpy(dtype=float),
                    merged["Q_hi50"].to_numpy(dtype=float),
                ),
                "mean_width": float(np.nanmean(merged["pi_width50"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def build_fold_assignments(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
    split_mode: str,
    group_col: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    curve_ids = np.array(sorted(curves["curve_id"].unique()))
    fold_df = pd.DataFrame({"curve_id": curve_ids, "fold": -1})
    if split_mode == "none":
        return fold_df
    if split_mode == "random_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, test_idx) in enumerate(splitter.split(curve_ids)):
            fold_df.loc[test_idx, "fold"] = fold
        return fold_df
    if split_mode == "group_kfold":
        if not group_col:
            raise ValueError("--group-col is required for group_kfold")
        if group_col not in metadata.columns:
            raise KeyError(f"group_col={group_col} not found in metadata columns {list(metadata.columns)}")
        meta = metadata[["curve_id", group_col]].drop_duplicates(subset=["curve_id"]).copy()
        meta = fold_df.merge(meta, on="curve_id", how="left")
        if meta[group_col].isna().any():
            missing = meta.loc[meta[group_col].isna(), "curve_id"].tolist()
            raise ValueError(f"Missing group labels for curves: {missing[:10]}")
        splitter = GroupKFold(n_splits=n_splits)
        X_dummy = np.zeros((len(meta), 1), dtype=float)
        for fold, (_, test_idx) in enumerate(splitter.split(X_dummy, groups=meta[group_col].to_numpy())):
            meta.loc[test_idx, "fold"] = fold
        return meta.drop(columns=[group_col])
    raise ValueError(f"Unsupported split_mode={split_mode}")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg = StandardizeConfig(
        time_unit=args.time_unit,
        dataset_name=args.dataset_name,
        mechanism_id=args.mechanism_id,
    )
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    curves_raw = pd.read_csv(args.curves_long)
    curves_std, curves_meta_from_long, curves_time_unit = standardize_curves(curves_raw, cfg)

    metadata = curves_meta_from_long
    if args.metadata_csv is not None:
        metadata_raw = pd.read_csv(args.metadata_csv)
        metadata_file = standardize_metadata(metadata_raw)
        metadata = metadata.merge(metadata_file, on="curve_id", how="outer", suffixes=("", "_meta"))

    curves_with_meta = curves_std.merge(metadata, on="curve_id", how="left")
    curves_with_meta.to_csv(args.outdir / "curves_standardized.csv", index=False)

    fold_assignments = build_fold_assignments(
        curves=curves_std,
        metadata=metadata,
        split_mode=args.split_mode,
        group_col=args.group_col,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    fold_assignments.to_csv(args.outdir / "fold_assignments.csv", index=False)

    split_metadata: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "mechanism_id": args.mechanism_id,
        "time_unit": curves_time_unit,
        "curves_long": str(args.curves_long),
        "predictions_csv": str(args.predictions_csv) if args.predictions_csv is not None else None,
        "metadata_csv": str(args.metadata_csv) if args.metadata_csv is not None else None,
        "mechanism_metadata_json": (
            str(args.mechanism_metadata_json) if args.mechanism_metadata_json is not None else None
        ),
        "split_mode": args.split_mode,
        "group_col": args.group_col,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "thresholds": thresholds,
    }
    if args.mechanism_metadata_json is not None and args.mechanism_metadata_json.exists():
        split_metadata["mechanism_metadata"] = json.loads(
            args.mechanism_metadata_json.read_text(encoding="utf-8")
        )

    if args.predictions_csv is None:
        split_metadata["status"] = "validated_inputs_only"
        (args.outdir / "split_metadata.json").write_text(
            json.dumps(split_metadata, indent=2),
            encoding="utf-8",
        )
        return

    preds_raw = pd.read_csv(args.predictions_csv)
    preds_std, preds_time_unit = standardize_predictions(preds_raw, cfg)
    if preds_time_unit != curves_time_unit and args.time_unit != "as_is":
        raise ValueError(
            "Observed and prediction tables resolved to different time units: "
            f"{curves_time_unit} vs {preds_time_unit}"
        )
    merged = merge_predictions(curves=curves_with_meta, preds=preds_std)
    merged.to_csv(args.outdir / "prediction_trajectories.csv", index=False)

    curve_metrics = build_curve_metrics(merged=merged, q_floor_mape=args.q_floor_mape)
    curve_metrics.to_csv(args.outdir / "metrics_by_curve.csv", index=False)

    summary = build_summary(
        merged=merged,
        curve_metrics=curve_metrics,
        cfg=cfg,
        time_unit=curves_time_unit,
        thresholds=thresholds,
        q_floor_mape=args.q_floor_mape,
    )
    summary.to_csv(args.outdir / "metrics_summary.csv", index=False)

    timescale_by_curve, timescale_summary = build_timescale_tables(merged=merged, thresholds=thresholds)
    timescale_by_curve.to_csv(args.outdir / "timescale_by_curve.csv", index=False)
    timescale_summary.to_csv(args.outdir / "timescale_summary.csv", index=False)

    uq_summary = build_uq_summary(merged=merged)
    uq_summary.to_csv(args.outdir / "uq_summary.csv", index=False)

    split_metadata["status"] = "evaluated_predictions"
    split_metadata["n_curves"] = int(curves_std["curve_id"].nunique())
    split_metadata["n_points"] = int(len(curves_std))
    split_metadata["prediction_rows"] = int(len(preds_std))
    (args.outdir / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
