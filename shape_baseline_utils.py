from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAFE_NUMERIC_COLS = [
    "Polymer_MW",
    "LA/GA",
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Initial D/M ratio",
    "DLC",
    "DLC_percent",
    "EE",
    "Particle_Size",
    "SA-V",
    "SE",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_NHA",
    "Drug_LogP",
    "media_pH",
    "media_temp_oC",
    "PDI",
    "zeta_potential",
    "weighted_Mw",
    "weighted_Tm",
]

SAFE_CATEGORICAL_COLS = [
    "polymer_family",
    "payload_name",
    "release_medium_condition",
    "release_method",
    "measurement_assay",
    "structure_type",
    "light_condition",
    "source_group",
]


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def weibull_curve(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    qmax, tau, beta = params
    t_pos = np.clip(t, 0.0, None)
    return qmax * (1.0 - np.exp(-np.power(t_pos / max(tau, 1e-8), beta)))


def biexponential_curve(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    qmax, w_fast, k_fast, k_slow = params
    t_pos = np.clip(t, 0.0, None)
    fast = 1.0 - np.exp(-k_fast * t_pos)
    slow = 1.0 - np.exp(-k_slow * t_pos)
    return qmax * (w_fast * fast + (1.0 - w_fast) * slow)


def hill_curve(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    qmax, t50, alpha = params
    t_pos = np.clip(t, 0.0, None)
    num = np.power(t_pos, alpha)
    den = np.power(max(t50, 1e-8), alpha) + num + 1e-12
    return qmax * (num / den)


FAMILY_FUNCS = {
    "weibull": weibull_curve,
    "biexponential": biexponential_curve,
    "hill": hill_curve,
}

FAMILY_BOUNDS = {
    "weibull": ([0.2, 1e-4, 0.1], [2.0, 1e3, 8.0]),
    "biexponential": ([0.2, 0.0, 1e-5, 1e-5], [2.0, 1.0, 50.0, 10.0]),
    "hill": ([0.2, 1e-4, 0.1], [2.0, 1e3, 8.0]),
}


def load_theta_and_pool(theta_dir: Path, pool_dir: Path) -> dict[str, Any]:
    theta = pd.read_parquet(theta_dir / "theta_table_v1.parquet")
    curves = pd.read_csv(pool_dir / "curves_long.csv")
    formulations = pd.read_csv(pool_dir / "formulations.csv")
    feature_cols = [col for col in SAFE_NUMERIC_COLS + SAFE_CATEGORICAL_COLS if col in formulations.columns]
    feature_table = formulations[["unified_curve_id"] + feature_cols].copy()
    theta = theta.merge(feature_table, on="unified_curve_id", how="left")
    curves["time_days"] = pd.to_numeric(curves["time_days"], errors="coerce")
    curves["release_fraction"] = pd.to_numeric(curves["release_fraction"], errors="coerce")
    curves = curves.dropna(subset=["unified_curve_id", "time_days", "release_fraction"]).copy()
    curve_groups = {
        str(curve_id): sub.sort_values("time_days").reset_index(drop=True)
        for curve_id, sub in curves.groupby("unified_curve_id", sort=True)
    }
    return {"theta": theta, "curves": curves, "curve_groups": curve_groups}


def transform_targets(family: str, params_frame: pd.DataFrame) -> np.ndarray:
    if family == "weibull":
        qmax = np.log(np.clip(params_frame["theta_param_1"].to_numpy(dtype=float), 1e-8, None))
        tau = np.log(np.clip(params_frame["theta_param_2"].to_numpy(dtype=float), 1e-8, None))
        beta = np.log(np.clip(params_frame["theta_param_3"].to_numpy(dtype=float), 1e-8, None))
        return np.column_stack([qmax, tau, beta])
    if family == "biexponential":
        qmax = np.log(np.clip(params_frame["theta_param_1"].to_numpy(dtype=float), 1e-8, None))
        w_fast = np.clip(params_frame["theta_param_2"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        logit_w = np.log(w_fast / (1.0 - w_fast))
        k_fast = np.log(np.clip(params_frame["theta_param_3"].to_numpy(dtype=float), 1e-8, None))
        k_slow = np.log(np.clip(params_frame["theta_param_4"].to_numpy(dtype=float), 1e-8, None))
        return np.column_stack([qmax, logit_w, k_fast, k_slow])
    if family == "hill":
        qmax = np.log(np.clip(params_frame["theta_param_1"].to_numpy(dtype=float), 1e-8, None))
        t50 = np.log(np.clip(params_frame["theta_param_2"].to_numpy(dtype=float), 1e-8, None))
        alpha = np.log(np.clip(params_frame["theta_param_3"].to_numpy(dtype=float), 1e-8, None))
        return np.column_stack([qmax, t50, alpha])
    raise KeyError(family)


def inverse_targets(family: str, target_array: np.ndarray) -> np.ndarray:
    target_array = np.asarray(target_array, dtype=float)
    if family == "weibull":
        params = np.column_stack(
            [
                np.exp(target_array[:, 0]),
                np.exp(target_array[:, 1]),
                np.exp(target_array[:, 2]),
            ]
        )
        lo, hi = FAMILY_BOUNDS[family]
        return np.clip(params, lo, hi)
    if family == "biexponential":
        qmax = np.exp(target_array[:, 0])
        w_fast = 1.0 / (1.0 + np.exp(-target_array[:, 1]))
        k_fast = np.exp(target_array[:, 2])
        k_slow = np.exp(target_array[:, 3])
        params = np.column_stack([qmax, w_fast, k_fast, k_slow])
        lo, hi = FAMILY_BOUNDS[family]
        return np.clip(params, lo, hi)
    if family == "hill":
        params = np.column_stack(
            [
                np.exp(target_array[:, 0]),
                np.exp(target_array[:, 1]),
                np.exp(target_array[:, 2]),
            ]
        )
        lo, hi = FAMILY_BOUNDS[family]
        return np.clip(params, lo, hi)
    raise KeyError(family)


def prepare_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    numeric_cols = numeric_cols or [col for col in SAFE_NUMERIC_COLS if col in train_df.columns]
    categorical_cols = categorical_cols or [col for col in SAFE_CATEGORICAL_COLS if col in train_df.columns]

    train_num = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy() if numeric_cols else pd.DataFrame()
    test_num = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy() if numeric_cols else pd.DataFrame()
    if numeric_cols:
        train_num = train_num.replace([np.inf, -np.inf], np.nan)
        test_num = test_num.replace([np.inf, -np.inf], np.nan)
    medians = train_num.median(axis=0).fillna(0.0) if numeric_cols else pd.Series(dtype=float)
    if numeric_cols:
        train_num = train_num.fillna(medians)
        test_num = test_num.fillna(medians)

    train_cat = train_df[categorical_cols].copy() if categorical_cols else pd.DataFrame()
    test_cat = test_df[categorical_cols].copy() if categorical_cols else pd.DataFrame()
    if categorical_cols:
        for col in categorical_cols:
            train_cat[col] = train_cat[col].astype(str).fillna("__MISSING__")
            test_cat[col] = test_cat[col].astype(str).fillna("__MISSING__")
        train_dummy = pd.get_dummies(train_cat, prefix=categorical_cols, dummy_na=False)
        test_dummy = pd.get_dummies(test_cat, prefix=categorical_cols, dummy_na=False)
        test_dummy = test_dummy.reindex(columns=train_dummy.columns, fill_value=0)
    else:
        train_dummy = pd.DataFrame(index=train_df.index)
        test_dummy = pd.DataFrame(index=test_df.index)

    train_arrays = [np.ones((len(train_df), 1), dtype=float)]
    test_arrays = [np.ones((len(test_df), 1), dtype=float)]
    design_cols = ["bias"]
    if numeric_cols:
        train_arrays.append(train_num.to_numpy(dtype=float))
        test_arrays.append(test_num.to_numpy(dtype=float))
        design_cols.extend(train_num.columns.tolist())
    if categorical_cols:
        train_arrays.append(train_dummy.to_numpy(dtype=float))
        test_arrays.append(test_dummy.to_numpy(dtype=float))
        design_cols.extend(train_dummy.columns.tolist())

    train_matrix = np.concatenate(train_arrays, axis=1)
    test_matrix = np.concatenate(test_arrays, axis=1)
    train_matrix = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    test_matrix = np.nan_to_num(test_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        train_matrix,
        test_matrix,
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "design_cols": design_cols,
            "numeric_medians": medians.to_dict() if numeric_cols else {},
        },
    )


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float = 1e-2,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    mean[:, 0] = 0.0
    std[:, 0] = 1.0
    x_train_std = (x_train - mean) / np.where(std > 0, std, 1.0)
    x_test_std = (x_test - mean) / np.where(std > 0, std, 1.0)

    penalty = alpha * np.eye(x_train_std.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train_std.T @ x_train_std + penalty) @ x_train_std.T @ y_train
    return x_test_std @ beta


def family_param_vector(row: pd.Series) -> list[float]:
    if row["theta_family"] == "weibull":
        return [float(row["theta_param_1"]), float(row["theta_param_2"]), float(row["theta_param_3"])]
    if row["theta_family"] == "biexponential":
        return [
            float(row["theta_param_1"]),
            float(row["theta_param_2"]),
            float(row["theta_param_3"]),
            float(row["theta_param_4"]),
        ]
    if row["theta_family"] == "hill":
        return [float(row["theta_param_1"]), float(row["theta_param_2"]), float(row["theta_param_3"])]
    raise KeyError(str(row["theta_family"]))


def evaluate_prediction(curve_df: pd.DataFrame, family: str, params: list[float]) -> dict[str, float]:
    t = curve_df["time_days"].to_numpy(dtype=float)
    y = curve_df["release_fraction"].to_numpy(dtype=float)
    pred = FAMILY_FUNCS[family](t, np.asarray(params, dtype=float))
    return {
        "curve_r2": safe_r2(y, pred),
        "curve_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "curve_mae": float(np.mean(np.abs(pred - y))),
    }


def median_param_fallback(train_df: pd.DataFrame, family: str) -> list[float]:
    sub = train_df[train_df["theta_family"] == family].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows available for family fallback: {family}")
    if family == "biexponential":
        cols = ["theta_param_1", "theta_param_2", "theta_param_3", "theta_param_4"]
    else:
        cols = ["theta_param_1", "theta_param_2", "theta_param_3"]
    return [float(sub[col].median()) for col in cols]


def json_list(values: list[float]) -> str:
    return json.dumps([float(v) for v in values])
