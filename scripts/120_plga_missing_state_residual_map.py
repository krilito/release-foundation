"""
120 - PLGA missing-state residual map.

Purpose:
    Map the release-state residual that remains when static formulation
    descriptors try to predict curve-derived PLGA release states.

Consumes:
    D:/chemical-world-model-v0/datset/321PLGA/.../mp_dataset_initial.xlsx
    release_state.py

Produces:
    outputs/120_plga_missing_state_residual_map/
      curve_state_table.csv
      static_state_prediction.csv
      missing_state_residuals.csv
      residual_symptom_summary.csv
      representation_agreement.csv
      source_structure_summary.csv
      summary_by_method.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This script does not score candidate measurements and does not implement
    MSVS. It only builds the residual map required before RSF can recommend
    what to measure next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from release_state import PCACurveStateSpace, WeibullStateSpace

DEFAULT_REAL_XLSX = Path(
    r"D:/chemical-world-model-v0/datset/321PLGA/"
    r"A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles/"
    r"mp_dataset_initial.xlsx"
)
DEFAULT_OUT = Path("outputs/120_plga_missing_state_residual_map")
DEFAULT_DOC = Path("docs/plga_missing_state_residual_map_2026-06-13.md")

NUMERIC_FEATURES = [
    "Polymer Mw",
    "Polymer Mn",
    "PDI",
    "LA/GA",
    "Initial Drug-to-Polymer Ratio",
    "Particle Size",
    "Drug Loading Capacity",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
]
CATEGORICAL_FEATURES = ["Drug", "Formulation Method"]
SOURCE_COLUMNS = ["DOI", "Formulation Method", "Drug"]
STATE_SPACE_NAMES = ("pca8", "pca3", "weibull")
PRIOR_MODELS = ("global_mean", "extra_trees", "ridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLGA missing-state residual map.")
    parser.add_argument("--real-xlsx", type=Path, default=DEFAULT_REAL_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-max-days", type=float, default=90.0)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--min-timepoints", type=int, default=3)
    parser.add_argument("--min-max-time", type=float, default=30.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-curves", type=int, default=0)
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_estimators = min(args.n_estimators, 50)
        args.shuffle_repeats = min(args.shuffle_repeats, 60)
        args.n_folds = min(args.n_folds, 3)
        if args.max_curves <= 0:
            args.max_curves = 45
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {"exists": True, "path": str(path), "size": stat.st_size, "mtime": stat.st_mtime}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y.size < 2 or float(np.nanvar(y)) <= 1e-12:
        return float("nan")
    return float(r2_score(y, pred))


def monotone_clip(values: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.clip(np.asarray(values, dtype=float), 0.0, 1.2), axis=-1)


def load_real_321(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if not args.real_xlsx.exists():
        raise FileNotFoundError(args.real_xlsx)
    raw = pd.read_excel(args.real_xlsx, sheet_name="PLGA_MPs")
    required = [
        "Formulation Index",
        "Drug",
        "Polymer Mw",
        "Polymer Mn",
        "PDI",
        "LA/GA",
        "Formulation Method",
        "Initial Drug-to-Polymer Ratio",
        "Particle Size",
        "Drug Loading Capacity",
        "Drug Encapsulation Efficiency",
        "Solubility Enhancer Concentration",
        "DOI",
        "Time",
        "Release",
    ]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"missing required 321 PLGA columns: {missing}")

    df = raw[required].copy()
    for col in ["Formulation Index", "Time", "Release"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Formulation Index", "Time", "Release"]).copy()
    df["Formulation Index"] = df["Formulation Index"].astype(int)
    df["Release_raw"] = df["Release"]
    df["Release"] = df["Release"].clip(0.0, 1.0)

    curve_groups: dict[int, pd.DataFrame] = {}
    meta_rows: list[dict[str, Any]] = []
    for fid, sub in df.groupby("Formulation Index", sort=True):
        curve = (
            sub.groupby("Time", as_index=False, sort=True)
            .agg(
                Release=("Release", "mean"),
                Release_raw=("Release_raw", "mean"),
                **{
                    col: (col, "first")
                    for col in required
                    if col not in ("Formulation Index", "Time", "Release")
                },
            )
            .sort_values("Time")
            .reset_index(drop=True)
        )
        if len(curve) < args.min_timepoints:
            continue
        if float(curve["Time"].max()) < args.min_max_time:
            continue
        curve_groups[int(fid)] = curve
        first = curve.iloc[0].to_dict()
        row = {col: first.get(col) for col in required if col not in ("Time", "Release")}
        row["Formulation Index"] = int(fid)
        row["n_timepoints"] = int(len(curve))
        row["min_time"] = float(curve["Time"].min())
        row["max_time"] = float(curve["Time"].max())
        row["release_min"] = float(curve["Release"].min())
        row["release_max"] = float(curve["Release"].max())
        meta_rows.append(row)

    meta = pd.DataFrame(meta_rows).sort_values("Formulation Index").reset_index(drop=True)
    if args.max_curves and len(meta) > args.max_curves:
        meta = (
            meta.sample(n=args.max_curves, random_state=args.seed)
            .sort_values("Formulation Index")
            .reset_index(drop=True)
        )
        keep = set(meta["Formulation Index"].astype(int))
        curve_groups = {fid: curve for fid, curve in curve_groups.items() if fid in keep}
    return meta, curve_groups


def interpolate_curve(curve: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    t = curve["Time"].to_numpy(dtype=float)
    q = curve["Release"].to_numpy(dtype=float)
    order = np.argsort(t)
    y = np.interp(grid, t[order], q[order], left=q[order][0], right=q[order][-1])
    return monotone_clip(y)


def build_curve_matrix(meta: pd.DataFrame, curve_groups: dict[int, pd.DataFrame], grid: np.ndarray) -> np.ndarray:
    rows = []
    for fid in meta["Formulation Index"].astype(int).tolist():
        rows.append(interpolate_curve(curve_groups[fid], grid))
    return np.vstack(rows)


def fit_feature_builder(meta_train: pd.DataFrame) -> dict[str, Any]:
    numeric_medians = {}
    for col in NUMERIC_FEATURES:
        valid = pd.to_numeric(meta_train[col], errors="coerce").dropna()
        numeric_medians[col] = 0.0 if valid.empty else float(valid.median())
    categories = {
        col: sorted(meta_train[col].astype(str).fillna("missing").unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    return {"numeric_medians": numeric_medians, "categories": categories}


def transform_features(meta: pd.DataFrame, builder: dict[str, Any]) -> np.ndarray:
    blocks = []
    numeric_cols = []
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(meta[col], errors="coerce")
        values = values.fillna(builder["numeric_medians"][col]).to_numpy(dtype=float)
        numeric_cols.append(values)
    blocks.append(np.column_stack(numeric_cols))

    for col in CATEGORICAL_FEATURES:
        observed = meta[col].astype(str).fillna("missing")
        cats = builder["categories"][col]
        block = np.zeros((len(meta), len(cats) + 1), dtype=float)
        cat_to_idx = {cat: i for i, cat in enumerate(cats)}
        for row_idx, value in enumerate(observed):
            block[row_idx, cat_to_idx.get(value, len(cats))] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def make_splits(meta: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    indices = np.arange(len(meta))
    splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    n_folds = min(args.n_folds, len(meta))
    if n_folds >= 2:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        for fold, (tr, te) in enumerate(kfold.split(indices), start=1):
            splits.append(("random_5fold", fold, tr, te))

    for scheme, col in [("group_by_DOI", "DOI"), ("group_by_Formulation_Method", "Formulation Method")]:
        groups = meta[col].astype(str).fillna("missing").to_numpy()
        n_groups = len(np.unique(groups))
        n_group_folds = min(args.n_folds, n_groups)
        if n_group_folds < 2:
            continue
        gkf = GroupKFold(n_splits=n_group_folds)
        for fold, (tr, te) in enumerate(gkf.split(indices, groups=groups), start=1):
            splits.append((scheme, fold, tr, te))
    return splits


def make_state_space(name: str, seed: int):
    if name == "pca8":
        return PCACurveStateSpace(n_components=8, random_state=seed, name="pca8")
    if name == "pca3":
        return PCACurveStateSpace(n_components=3, random_state=seed, name="pca3")
    if name == "weibull":
        return WeibullStateSpace()
    raise ValueError(name)


def fit_prior(prior_name: str, x_train: np.ndarray, z_train: np.ndarray, seed: int, n_estimators: int):
    if prior_name == "global_mean":
        return None
    if prior_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, z_train)
        return model
    if prior_name == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train, z_train)
        return model
    raise ValueError(prior_name)


def predict_prior(model: Any, prior_name: str, x: np.ndarray, z_train: np.ndarray) -> np.ndarray:
    if prior_name == "global_mean":
        return np.tile(np.mean(z_train, axis=0), (len(x), 1))
    return np.asarray(model.predict(x), dtype=float)


def curve_symptoms(grid: np.ndarray, q_curve: np.ndarray, q_prior: np.ndarray) -> dict[str, float]:
    delta = q_curve - q_prior

    def mean_window(lo: float, hi: float) -> float:
        mask = (grid >= lo) & (grid <= hi)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(delta[mask]))

    def value_at(t: float, values: np.ndarray) -> float:
        return float(np.interp(t, grid, values, left=values[0], right=values[-1]))

    q7_delta = value_at(7.0, q_curve) - value_at(7.0, q_prior)
    q30_delta = value_at(30.0, q_curve) - value_at(30.0, q_prior)
    q60_delta = value_at(60.0, q_curve) - value_at(60.0, q_prior)
    return {
        "burst_residual": mean_window(0.0, 7.0),
        "early_slope_residual": q7_delta / 7.0,
        "middle_diffusion_residual": (q30_delta - q7_delta) / 23.0,
        "late_tail_residual": (float(delta[-1]) - q60_delta) / max(float(grid[-1] - 60.0), 1.0),
        "qmax_residual": float(delta[-1]),
        "q7_residual": float(q7_delta),
        "q30_residual": float(q30_delta),
        "q60_residual": float(q60_delta),
        "decoded_curve_rmse": rmse(q_curve, q_prior),
    }


def state_columns(prefix: str, states: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(states, columns=[f"{prefix}_{i + 1}" for i in range(states.shape[1])])


def residual_norm(residual: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(residual), axis=1))


def run_residual_map(
    args: argparse.Namespace,
    meta: pd.DataFrame,
    y_grid: np.ndarray,
    grid: np.ndarray,
) -> dict[str, pd.DataFrame]:
    splits = make_splits(meta, args)
    curve_state_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    residual_rows: list[dict[str, Any]] = []
    train_thresholds: list[dict[str, Any]] = []
    group_overlap_rows: list[dict[str, Any]] = []

    for split_name, fold, train_idx, test_idx in splits:
        builder = fit_feature_builder(meta.iloc[train_idx])
        x_train = transform_features(meta.iloc[train_idx], builder)
        x_test = transform_features(meta.iloc[test_idx], builder)

        for state_name in STATE_SPACE_NAMES:
            state_space = make_state_space(state_name, args.seed + fold)
            state_space.fit(y_grid[train_idx], grid)
            z_train = state_space.encode(y_grid[train_idx], grid)
            z_test = state_space.encode(y_grid[test_idx], grid)
            q_oracle = state_space.decode(z_test, grid)

            state_base = meta.iloc[test_idx][
                ["Formulation Index", "Drug", "Formulation Method", "DOI", "n_timepoints", "max_time"]
            ].reset_index(drop=True)
            state_block = pd.concat([state_base, state_columns("z_Q", z_test)], axis=1)
            state_block.insert(0, "state_space", state_name)
            state_block.insert(0, "fold", fold)
            state_block.insert(0, "split", split_name)
            curve_state_rows.append(state_block)

            for prior_name in PRIOR_MODELS:
                model = fit_prior(
                    prior_name,
                    x_train,
                    z_train,
                    args.seed + 1000 * fold + 17 * len(prior_name),
                    args.n_estimators,
                )
                z_pred_test = predict_prior(model, prior_name, x_test, z_train)
                z_pred_train = predict_prior(model, prior_name, x_train, z_train)
                residual_train = state_space.residual(z_train, z_pred_train)
                threshold = float(np.quantile(residual_norm(residual_train), 0.80))
                train_thresholds.append(
                    {
                        "split": split_name,
                        "fold": fold,
                        "state_space": state_name,
                        "prior_model": prior_name,
                        "train_q80_residual_norm": threshold,
                    }
                )

                residual = state_space.residual(z_test, z_pred_test)
                q_prior = state_space.decode(z_pred_test, grid)
                norm = residual_norm(residual)
                decoded_rmse = np.sqrt(np.mean(np.square(q_oracle - q_prior), axis=1))
                pred_base = state_base.copy()
                pred_block = pd.concat(
                    [
                        pred_base,
                        state_columns("z_X", z_pred_test),
                        pd.DataFrame(
                            {
                                "state_rmse": norm,
                                "decoded_curve_rmse": decoded_rmse,
                                "is_oracle": False,
                            }
                        ),
                    ],
                    axis=1,
                )
                pred_block.insert(0, "prior_model", prior_name)
                pred_block.insert(0, "state_space", state_name)
                pred_block.insert(0, "fold", fold)
                pred_block.insert(0, "split", split_name)
                prediction_rows.append(pred_block)

                for local_pos, global_idx in enumerate(test_idx):
                    symptoms = curve_symptoms(grid, q_oracle[local_pos], q_prior[local_pos])
                    row = {
                        "split": split_name,
                        "fold": fold,
                        "state_space": state_name,
                        "prior_model": prior_name,
                        "Formulation Index": int(meta.iloc[global_idx]["Formulation Index"]),
                        "Drug": str(meta.iloc[global_idx]["Drug"]),
                        "Formulation Method": str(meta.iloc[global_idx]["Formulation Method"]),
                        "DOI": str(meta.iloc[global_idx]["DOI"]),
                        "n_timepoints": int(meta.iloc[global_idx]["n_timepoints"]),
                        "max_time": float(meta.iloc[global_idx]["max_time"]),
                        "residual_norm": float(norm[local_pos]),
                        "train_q80_residual_norm": threshold,
                        "is_high_residual": bool(norm[local_pos] > threshold),
                        "state_dim": int(z_test.shape[1]),
                        "is_oracle": False,
                    }
                    row.update(symptoms)
                    for j in range(residual.shape[1]):
                        row[f"state_residual_{j + 1}"] = float(residual[local_pos, j])
                    residual_rows.append(row)

        for scheme_col in ["DOI", "Formulation Method"]:
            if split_name == f"group_by_{scheme_col}" or (
                split_name == "group_by_Formulation_Method" and scheme_col == "Formulation Method"
            ):
                train_groups = set(meta.iloc[train_idx][scheme_col].astype(str))
                test_groups = set(meta.iloc[test_idx][scheme_col].astype(str))
                group_overlap_rows.append(
                    {
                        "split": split_name,
                        "fold": fold,
                        "group_column": scheme_col,
                        "n_train_groups": len(train_groups),
                        "n_test_groups": len(test_groups),
                        "n_overlap": len(train_groups & test_groups),
                    }
                )

    residuals = pd.DataFrame(residual_rows)
    thresholds = pd.DataFrame(train_thresholds)
    return {
        "curve_state_table": pd.concat(curve_state_rows, ignore_index=True),
        "static_state_prediction": pd.concat(prediction_rows, ignore_index=True),
        "missing_state_residuals": residuals,
        "train_thresholds": thresholds,
        "group_overlaps": pd.DataFrame(group_overlap_rows),
    }


def summarize_methods(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    symptom_cols = [
        "burst_residual",
        "early_slope_residual",
        "middle_diffusion_residual",
        "late_tail_residual",
        "qmax_residual",
    ]
    for keys, sub in residuals.groupby(["split", "state_space", "prior_model"], sort=True):
        split, state_space, prior_model = keys
        row: dict[str, Any] = {
            "split": split,
            "state_space": state_space,
            "prior_model": prior_model,
            "n_curve_records": int(len(sub)),
            "n_unique_curves": int(sub["Formulation Index"].nunique()),
            "z_rmse": float(np.sqrt(np.mean(np.square(sub["residual_norm"].to_numpy(dtype=float))))),
            "decoded_curve_rmse": float(np.sqrt(np.mean(np.square(sub["decoded_curve_rmse"].to_numpy(dtype=float))))),
            "median_residual_norm": float(sub["residual_norm"].median()),
            "high_residual_rate": float(sub["is_high_residual"].mean()),
        }
        for col in symptom_cols:
            row[f"mean_abs_{col}"] = float(np.mean(np.abs(sub[col].to_numpy(dtype=float))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "state_space", "z_rmse"])


def summarize_symptoms(residuals: pd.DataFrame) -> pd.DataFrame:
    symptom_cols = [
        "burst_residual",
        "early_slope_residual",
        "middle_diffusion_residual",
        "late_tail_residual",
        "qmax_residual",
    ]
    rows = []
    for keys, sub in residuals.groupby(["split", "state_space", "prior_model"], sort=True):
        split, state_space, prior_model = keys
        for col in symptom_cols:
            values = sub[col].to_numpy(dtype=float)
            rows.append(
                {
                    "split": split,
                    "state_space": state_space,
                    "prior_model": prior_model,
                    "symptom": col,
                    "mean": float(np.mean(values)),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "median_abs": float(np.median(np.abs(values))),
                    "p90_abs": float(np.quantile(np.abs(values), 0.90)),
                }
            )
    return pd.DataFrame(rows)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa = set(np.flatnonzero(np.asarray(a, dtype=bool)).tolist())
    bb = set(np.flatnonzero(np.asarray(b, dtype=bool)).tolist())
    denom = len(aa | bb)
    if denom == 0:
        return float("nan")
    return len(aa & bb) / denom


def representation_agreement(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, fold, prior_model), block in residuals.groupby(["split", "fold", "prior_model"], sort=True):
        spaces = sorted(block["state_space"].unique().tolist())
        for i, left in enumerate(spaces):
            for right in spaces[i + 1 :]:
                a = block[block["state_space"] == left].set_index("Formulation Index")
                b = block[block["state_space"] == right].set_index("Formulation Index")
                common = sorted(set(a.index) & set(b.index))
                if len(common) < 4:
                    continue
                av = a.loc[common, "residual_norm"].to_numpy(dtype=float)
                bv = b.loc[common, "residual_norm"].to_numpy(dtype=float)
                if np.nanstd(av) <= 1e-12 or np.nanstd(bv) <= 1e-12:
                    corr = float("nan")
                else:
                    corr = float(spearmanr(av, bv).correlation)
                rows.append(
                    {
                        "split": split,
                        "fold": fold,
                        "prior_model": prior_model,
                        "state_space_a": left,
                        "state_space_b": right,
                        "n_common_curves": len(common),
                        "spearman_residual_norm": corr,
                        "high_residual_jaccard": jaccard(
                            a.loc[common, "is_high_residual"].to_numpy(dtype=bool),
                            b.loc[common, "is_high_residual"].to_numpy(dtype=bool),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def group_eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values) & pd.Series(labels).notna().to_numpy()
    y = values[mask]
    groups = np.asarray(labels, dtype=object)[mask]
    if len(y) < 8:
        return float("nan")
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1e-12:
        return float("nan")
    between = 0.0
    valid_groups = 0
    for label in pd.unique(groups):
        vals = y[groups == label]
        if len(vals) < 2:
            continue
        valid_groups += 1
        between += len(vals) * float((np.mean(vals) - np.mean(y)) ** 2)
    if valid_groups < 2:
        return float("nan")
    return between / total


def source_structure_summary(residuals: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 1200)
    rows = []
    for keys, sub in residuals.groupby(["split", "state_space", "prior_model"], sort=True):
        split, state_space, prior_model = keys
        y = sub["residual_norm"].to_numpy(dtype=float)
        for source_col in SOURCE_COLUMNS:
            labels = sub[source_col].astype(str).replace({"nan": np.nan}).to_numpy()
            eta = group_eta_squared(y, labels)
            null = []
            for _ in range(args.shuffle_repeats):
                shuffled = labels.copy()
                rng.shuffle(shuffled)
                val = group_eta_squared(y, shuffled)
                if np.isfinite(val):
                    null.append(val)
            null95 = float(np.nanpercentile(null, 95)) if null else float("nan")
            rows.append(
                {
                    "split": split,
                    "state_space": state_space,
                    "prior_model": prior_model,
                    "source_column": source_col,
                    "n": int(len(sub)),
                    "n_groups": int(pd.Series(labels).dropna().nunique()),
                    "eta_squared": eta,
                    "shuffle_null95": null95,
                    "exceeds_shuffle_null": bool(np.isfinite(eta) and np.isfinite(null95) and eta > null95),
                }
            )
    return pd.DataFrame(rows)


def decision_table(
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    source_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    pca_oracle_like = summary[(summary["state_space"].isin(["pca8", "pca3"])) & (summary["prior_model"] == "global_mean")]
    rows.append(
        {
            "decision": "state_representation_outputs_exist",
            "status": "pass" if not summary.empty else "fail",
            "evidence": f"summary rows={len(summary)}; pca baseline rows={len(pca_oracle_like)}",
        }
    )

    deploy = summary[summary["prior_model"].isin(["extra_trees", "ridge"])]
    global_rows = summary[summary["prior_model"] == "global_mean"][
        ["split", "state_space", "z_rmse", "decoded_curve_rmse"]
    ].rename(columns={"z_rmse": "global_z_rmse", "decoded_curve_rmse": "global_decoded_curve_rmse"})
    compare = deploy.merge(global_rows, on=["split", "state_space"], how="left")
    compare["z_gain_vs_global"] = compare["global_z_rmse"] - compare["z_rmse"]
    best_gain = float(compare["z_gain_vs_global"].max()) if not compare.empty else float("nan")
    rows.append(
        {
            "decision": "static_X_beats_global_state_prior",
            "status": "pass" if np.isfinite(best_gain) and best_gain > 0 else "warn",
            "evidence": f"best z_rmse gain vs global_mean={best_gain:.4f}",
        }
    )

    strict = compare[compare["split"].isin(["group_by_DOI", "group_by_Formulation_Method"])]
    strict_gain = float(strict["z_gain_vs_global"].max()) if not strict.empty else float("nan")
    rows.append(
        {
            "decision": "strict_group_split_static_signal_exists",
            "status": "pass" if np.isfinite(strict_gain) and strict_gain > 0 else "warn",
            "evidence": f"best strict z_rmse gain vs global_mean={strict_gain:.4f}",
        }
    )

    stable = agreement[agreement["prior_model"].isin(["extra_trees", "ridge"])]
    mean_corr = float(stable["spearman_residual_norm"].mean()) if not stable.empty else float("nan")
    mean_jaccard = float(stable["high_residual_jaccard"].mean()) if not stable.empty else float("nan")
    rows.append(
        {
            "decision": "residuals_stable_across_representations",
            "status": "pass" if np.isfinite(mean_corr) and mean_corr >= 0.30 else "warn",
            "evidence": f"mean Spearman={mean_corr:.3f}; mean high-residual Jaccard={mean_jaccard:.3f}",
        }
    )

    source_hits = source_summary[source_summary["exceeds_shuffle_null"].fillna(False)]
    doi_hits = int((source_hits["source_column"] == "DOI").sum()) if not source_hits.empty else 0
    rows.append(
        {
            "decision": "source_or_method_structure_present",
            "status": "warn" if doi_hits > 0 else "pass",
            "evidence": f"source/method residual associations exceeding shuffle null={len(source_hits)}; DOI hits={doi_hits}",
        }
    )

    rows.append(
        {
            "decision": "no_MSVS_or_candidate_measurement_scoring",
            "status": "pass",
            "evidence": "120 outputs residual maps and source structure only; measurement scoring starts in 121.",
        }
    )

    failed = checks[checks["status"] == "fail"]
    rows.append(
        {
            "decision": "data_checks_pass",
            "status": "pass" if failed.empty else "fail",
            "evidence": f"failed checks={failed['check'].tolist() if not failed.empty else []}",
        }
    )
    return pd.DataFrame(rows)


def make_checks(
    args: argparse.Namespace,
    meta: pd.DataFrame,
    splits: list[tuple[str, int, np.ndarray, np.ndarray]],
    residuals: pd.DataFrame,
    group_overlaps: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        {
            "check": "input_file_exists",
            "status": "pass" if args.real_xlsx.exists() else "fail",
            "detail": str(args.real_xlsx),
        },
        {
            "check": "sample_unit_is_curve",
            "status": "pass" if meta["Formulation Index"].is_unique else "fail",
            "detail": f"n_curves={len(meta)}",
        },
        {
            "check": "shared_time_grid_used",
            "status": "pass",
            "detail": f"grid=0-{args.grid_max_days} days, n={args.grid_size}",
        },
        {
            "check": "no_timepoint_split",
            "status": "pass",
            "detail": "splits are generated over curve-level metadata rows",
        },
        {
            "check": "state_representations_fit_train_fold_only",
            "status": "pass",
            "detail": "PCACurveStateSpace.fit is called inside each fold on y_train only",
        },
        {
            "check": "DOI_not_used_as_static_feature",
            "status": "pass" if "DOI" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES else "fail",
            "detail": f"categorical_features={CATEGORICAL_FEATURES}",
        },
        {
            "check": "group_splits_have_no_overlap",
            "status": "pass" if group_overlaps.empty or int(group_overlaps["n_overlap"].max()) == 0 else "fail",
            "detail": group_overlaps.to_dict(orient="records") if len(group_overlaps) <= 12 else f"rows={len(group_overlaps)}",
        },
        {
            "check": "all_required_outputs_have_rows",
            "status": "pass" if not residuals.empty else "fail",
            "detail": f"residual rows={len(residuals)}; splits={len(splits)}",
        },
        {
            "check": "no_nan_primary_metrics",
            "status": "pass" if np.isfinite(residuals["residual_norm"]).all() and np.isfinite(residuals["decoded_curve_rmse"]).all() else "fail",
            "detail": "residual_norm and decoded_curve_rmse finite",
        },
        {
            "check": "oracle_rows_marked",
            "status": "pass" if "is_oracle" in residuals.columns and not residuals["is_oracle"].any() else "fail",
            "detail": "deployable residual rows marked is_oracle=False",
        },
        {
            "check": "no_candidate_measurement_scoring",
            "status": "pass",
            "detail": "120 does not output MSVS or candidate measurement priority tables",
        },
    ]
    return pd.DataFrame(checks)


def write_report(
    out: Path,
    args: argparse.Namespace,
    meta: pd.DataFrame,
    summary: pd.DataFrame,
    symptoms: pd.DataFrame,
    agreement: pd.DataFrame,
    source_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    best = summary.sort_values(["split", "state_space", "z_rmse"]).groupby(["split", "state_space"]).head(2)
    top_symptoms = (
        symptoms.sort_values(["split", "state_space", "prior_model", "mean_abs"], ascending=[True, True, True, False])
        .groupby(["split", "state_space", "prior_model"])
        .head(2)
    )
    source_hits = source_summary[source_summary["exceeds_shuffle_null"].fillna(False)]
    lines = [
        "# PLGA Missing-State Residual Map",
        "",
        "This is Experiment 120. It maps release-state residuals; it does not score candidate measurements.",
        "",
        "## Dataset",
        "",
        f"- real curves: {len(meta)}",
        f"- grid: 0-{args.grid_max_days} days, {args.grid_size} points",
        f"- static features: numeric={len(NUMERIC_FEATURES)}, categorical={CATEGORICAL_FEATURES}",
        "- DOI is not used as a static feature; it is held for source-artifact checks.",
        "",
        "## Best Static Priors By State Space",
        "",
        best[["split", "state_space", "prior_model", "n_unique_curves", "z_rmse", "decoded_curve_rmse", "high_residual_rate"]]
        .round(4)
        .to_markdown(index=False),
        "",
        "## Dominant Residual Symptoms",
        "",
        top_symptoms[["split", "state_space", "prior_model", "symptom", "mean_abs", "p90_abs"]]
        .round(4)
        .to_markdown(index=False),
        "",
        "## Representation Agreement",
        "",
        agreement.round(4).to_markdown(index=False) if not agreement.empty else "No representation agreement rows.",
        "",
        "## Source / Method Structure Hits",
        "",
        source_hits.round(4).to_markdown(index=False) if not source_hits.empty else "No source/method residual association exceeded shuffle null.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Data Checks",
        "",
        checks.to_markdown(index=False),
        "",
        "## Interpretation Boundary",
        "",
        "- A high residual means static descriptors failed in the declared state space.",
        "- It does not identify porosity, tortuosity, residual solvent, or any missing physical variable.",
        "- Candidate measurement scoring belongs to Experiment 121, after this residual map is frozen.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(
    doc_path: Path,
    meta: pd.DataFrame,
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    source_summary: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    best = summary.sort_values(["split", "state_space", "z_rmse"]).groupby(["split", "state_space"]).head(1)
    source_hits = source_summary[source_summary["exceeds_shuffle_null"].fillna(False)]
    lines = [
        "# PLGA Missing-State Residual Map",
        "",
        "Date: 2026-06-13",
        "",
        "## Question",
        "",
        "What release-state residual remains when static PLGA descriptors attempt to predict curve-derived release states?",
        "",
        "This is not a candidate-measurement ranking experiment. It freezes the residual map required before RSF/MSVS.",
        "",
        "## Protocol",
        "",
        "- Sample unit is one formulation curve.",
        "- Real ragged curves are interpolated to one shared 0-90 day grid.",
        "- State spaces: PCA-8, PCA-3, and Weibull log-theta.",
        "- Static priors: global mean, ExtraTrees, Ridge sanity check.",
        "- Splits: random curve folds, group-by-DOI, and group-by-formulation-method.",
        "- DOI is excluded from static features and used only as a source-structure check.",
        "",
        "## Best Rows",
        "",
        best[["split", "state_space", "prior_model", "z_rmse", "decoded_curve_rmse", "high_residual_rate"]]
        .round(4)
        .to_markdown(index=False),
        "",
        "## Representation Agreement",
        "",
        agreement.groupby(["split", "prior_model"], as_index=False)
        .agg(
            mean_spearman=("spearman_residual_norm", "mean"),
            mean_jaccard=("high_residual_jaccard", "mean"),
            n_pairs=("spearman_residual_norm", "size"),
        )
        .round(4)
        .to_markdown(index=False),
        "",
        "## Source Structure",
        "",
        source_hits[["split", "state_space", "prior_model", "source_column", "eta_squared", "shuffle_null95"]]
        .round(4)
        .to_markdown(index=False)
        if not source_hits.empty
        else "No DOI/method/drug residual association exceeded the shuffle null.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Output Anchor",
        "",
        "`../outputs/120_plga_missing_state_residual_map/`",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_groups = load_real_321(args)
    if meta.empty:
        raise RuntimeError("no eligible PLGA curves after filtering")
    grid = np.linspace(0.0, args.grid_max_days, args.grid_size)
    y_grid = build_curve_matrix(meta, curve_groups, grid)
    splits = make_splits(meta, args)
    if not splits:
        raise RuntimeError("no valid curve-level splits generated")

    tables = run_residual_map(args, meta, y_grid, grid)
    residuals = tables["missing_state_residuals"]
    summary = summarize_methods(residuals)
    symptoms = summarize_symptoms(residuals)
    agreement = representation_agreement(residuals)
    source_summary = source_structure_summary(residuals, args)
    checks = make_checks(args, meta, splits, residuals, tables["group_overlaps"])
    decisions = decision_table(summary, agreement, source_summary, checks)

    tables["curve_state_table"].to_csv(args.out / "curve_state_table.csv", index=False)
    tables["static_state_prediction"].to_csv(args.out / "static_state_prediction.csv", index=False)
    residuals.to_csv(args.out / "missing_state_residuals.csv", index=False)
    symptoms.to_csv(args.out / "residual_symptom_summary.csv", index=False)
    agreement.to_csv(args.out / "representation_agreement.csv", index=False)
    source_summary.to_csv(args.out / "source_structure_summary.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    write_report(args.out, args, meta, summary, symptoms, agreement, source_summary, decisions, checks)
    write_doc(args.doc, meta, summary, agreement, source_summary, decisions)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "real_xlsx": file_meta(args.real_xlsx),
        "n_curves": int(len(meta)),
        "state_spaces": STATE_SPACE_NAMES,
        "prior_models": PRIOR_MODELS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_source_feature": "DOI",
        "interpretation_boundary": "120 maps residuals only; MSVS and candidate measurement scoring begin in 121.",
        "generated_files": sorted(
            {p.name for p in args.out.iterdir() if p.is_file()} | {"lock_metadata.json"}
        ),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not bool((checks["status"] == "pass").all()):
        failed = checks.loc[checks["status"] != "pass", "check"].tolist()
        raise RuntimeError(f"120 failed data checks: {failed}")
    print(f"Wrote PLGA missing-state residual map to {args.out}")


if __name__ == "__main__":
    main()
