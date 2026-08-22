from __future__ import annotations

"""
157_curve_world_hard_gate.py

Consume:
- outputs/148_release_main_cumulative_v1/curves_long.csv
- outputs/148_release_main_cumulative_v1/formulations.csv

Produce:
- outputs/157_curve_world_hard_gate/q1_per_curve_family_fits.csv
- outputs/157_curve_world_hard_gate/q1_family_summary.csv
- outputs/157_curve_world_hard_gate/q1_best_family_summary.csv
- outputs/157_curve_world_hard_gate/q1_fpca_support.json
- outputs/157_curve_world_hard_gate/q2_cluster_assignments.csv
- outputs/157_curve_world_hard_gate/q2_cluster_summary.csv
- outputs/157_curve_world_hard_gate/q2_cluster_selection.csv
- outputs/157_curve_world_hard_gate/q3_transfer_metrics.csv
- outputs/157_curve_world_hard_gate/q3_transfer_summary.csv
- outputs/157_curve_world_hard_gate/verdict.json
- outputs/157_curve_world_hard_gate/summary.md

Expected runtime:
- ~2-8 min on the current 148 main cumulative pool
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class FamilySpec:
    name: str
    n_params: int
    bounds_lo: tuple[float, ...]
    bounds_hi: tuple[float, ...]


FAMILY_SPECS = {
    "weibull": FamilySpec("weibull", 3, (0.2, 1e-4, 0.1), (2.0, 1e3, 8.0)),
    "biexponential": FamilySpec("biexponential", 4, (0.2, 0.0, 1e-5, 1e-5), (2.0, 1.0, 50.0, 10.0)),
    "hill": FamilySpec("hill", 3, (0.2, 1e-4, 0.1), (2.0, 1e3, 8.0)),
}
CANONICAL_FAMILY = "biexponential"
Q3_EARLY_OBS = (0, 3, 5)
RIDGE_ALPHA = 1e-2


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=repo_root / "outputs" / "148_release_main_cumulative_v1",
        help="main cumulative release pool directory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "157_curve_world_hard_gate",
        help="output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def clamp_positive(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), eps)


def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def weibull_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, tau, beta = p
    t_pos = np.clip(t, 0.0, None)
    return qmax * (1.0 - np.exp(-np.power(t_pos / max(tau, 1e-8), beta)))


def biexponential_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, w_fast, k_fast, k_slow = p
    t_pos = np.clip(t, 0.0, None)
    fast = 1.0 - np.exp(-k_fast * t_pos)
    slow = 1.0 - np.exp(-k_slow * t_pos)
    return qmax * (w_fast * fast + (1.0 - w_fast) * slow)


def hill_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, t50, alpha = p
    t_pos = np.clip(t, 0.0, None)
    num = np.power(t_pos, alpha)
    den = np.power(max(t50, 1e-8), alpha) + num + 1e-12
    return qmax * (num / den)


FAMILY_FUNCS = {
    "weibull": weibull_curve,
    "biexponential": biexponential_curve,
    "hill": hill_curve,
}


def initial_guesses(family: str, t: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    y_max = float(np.clip(np.nanmax(y), 0.3, 1.5))
    t_end = float(max(np.nanmax(t), 1e-3))
    if family == "weibull":
        return [
            np.array([y_max, max(t_end / 3.0, 1e-3), 1.0]),
            np.array([min(1.8, y_max + 0.1), max(t_end / 2.0, 1e-3), 0.7]),
            np.array([min(1.8, y_max + 0.2), max(t_end / 6.0, 1e-3), 1.5]),
        ]
    if family == "biexponential":
        return [
            np.array([y_max, 0.6, 1.0 / max(t_end / 5.0, 1e-3), 1.0 / max(t_end, 1e-3)]),
            np.array([min(1.8, y_max + 0.1), 0.3, 1.0 / max(t_end / 3.0, 1e-3), 1.0 / max(t_end * 2.0, 1e-3)]),
            np.array([min(1.8, y_max + 0.2), 0.8, 1.0 / max(t_end / 8.0, 1e-3), 1.0 / max(t_end / 2.0, 1e-3)]),
        ]
    if family == "hill":
        return [
            np.array([y_max, max(t_end / 3.0, 1e-3), 1.2]),
            np.array([min(1.8, y_max + 0.1), max(t_end / 2.0, 1e-3), 0.9]),
            np.array([min(1.8, y_max + 0.2), max(t_end / 5.0, 1e-3), 1.8]),
        ]
    raise KeyError(family)


def fit_family(family: str, t: np.ndarray, y: np.ndarray) -> dict[str, object]:
    spec = FAMILY_SPECS[family]
    fn = FAMILY_FUNCS[family]
    best: dict[str, object] | None = None

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = fn(t, p)
        weights = 1.0 / np.sqrt(max(len(y), 1))
        return (pred - y) * weights

    for x0 in initial_guesses(family, t, y):
        try:
            result = least_squares(
                residuals,
                x0=np.clip(x0, spec.bounds_lo, spec.bounds_hi),
                bounds=(np.asarray(spec.bounds_lo), np.asarray(spec.bounds_hi)),
                max_nfev=4000,
            )
        except Exception:
            continue
        pred = fn(t, result.x)
        row = {
            "family": family,
            "params": result.x.tolist(),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))),
            "r2": safe_r2(y, pred),
            "pred": pred,
        }
        if best is None or float(row["rmse"]) < float(best["rmse"]):
            best = row

    if best is None:
        mean_pred = np.full_like(y, np.mean(y))
        return {
            "family": family,
            "params": [np.nan] * spec.n_params,
            "rmse": float(np.sqrt(np.mean((mean_pred - y) ** 2))),
            "mae": float(np.mean(np.abs(mean_pred - y))),
            "r2": safe_r2(y, mean_pred),
            "pred": mean_pred,
        }
    return best


def map_system_id(source_dataset: str, polymer_family: str) -> str:
    if source_dataset in {"internal181", "cross321"}:
        return "PLGA"
    return polymer_family


def load_pool(pool_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(pool_dir / "curves_long.csv")
    formulations = pd.read_csv(pool_dir / "formulations.csv")
    quality = pd.read_csv(pool_dir / "curve_quality_summary.csv")
    formulations["system_id"] = [
        map_system_id(str(ds), str(fam))
        for ds, fam in zip(formulations["source_dataset"], formulations["polymer_family"])
    ]
    curve_index = formulations[
        ["unified_curve_id", "source_dataset", "polymer_family", "system_id", "payload_name", "source_group"]
    ].copy()
    curves = curves.merge(curve_index, on=["unified_curve_id", "source_dataset"], how="left")
    curves["time_days"] = pd.to_numeric(curves["time_days"], errors="coerce")
    curves["release_fraction"] = pd.to_numeric(curves["release_fraction"], errors="coerce")
    curves = curves.dropna(subset=["unified_curve_id", "time_days", "release_fraction", "system_id"]).copy()
    curves = curves.sort_values(["unified_curve_id", "time_days"], kind="stable").reset_index(drop=True)
    quality = quality.merge(
        formulations[["unified_curve_id", "system_id", "polymer_family", "payload_name", "source_group"]],
        on="unified_curve_id",
        how="left",
    )
    return curves, formulations, quality


def group_curves(curves: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for curve_id, sub in curves.groupby("unified_curve_id", sort=True):
        grouped[str(curve_id)] = sub.sort_values("time_days").reset_index(drop=True)
    return grouped


def run_q1(curve_groups: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    fpca_rows: list[np.ndarray] = []

    for curve_id, sub in curve_groups.items():
        t = sub["time_days"].to_numpy(dtype=float)
        y = sub["release_fraction"].to_numpy(dtype=float)
        meta = sub.iloc[0]
        best_fit: dict[str, object] | None = None
        best_pred: np.ndarray | None = None
        for family in FAMILY_SPECS:
            fit = fit_family(family, t, y)
            rows.append(
                {
                    "unified_curve_id": curve_id,
                    "system_id": meta["system_id"],
                    "source_dataset": meta["source_dataset"],
                    "polymer_family": meta["polymer_family"],
                    "family": family,
                    "n_points": int(len(sub)),
                    "duration_days": float(t[-1] - t[0]),
                    "release_final": float(y[-1]),
                    "release_max": float(np.max(y)),
                    "rmse": float(fit["rmse"]),
                    "mae": float(fit["mae"]),
                    "r2": float(fit["r2"]),
                    "params_json": json.dumps(fit["params"]),
                }
            )
            if best_fit is None or float(fit["rmse"]) < float(best_fit["rmse"]):
                best_fit = fit
                best_pred = np.asarray(fit["pred"], dtype=float)

        if best_fit is not None and best_pred is not None:
            t_rel = t / max(t[-1], 1e-8)
            fpca_grid = np.linspace(0.0, 1.0, 64)
            fpca_rows.append(np.interp(fpca_grid, t_rel, best_pred, left=best_pred[0], right=best_pred[-1]))

    per_curve = pd.DataFrame(rows)
    family_summary = (
        per_curve.groupby(["system_id", "family"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "count"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            p25_r2=("r2", lambda x: float(np.nanpercentile(x, 25))),
            p75_r2=("r2", lambda x: float(np.nanpercentile(x, 75))),
            frac_r2_ge_095=("r2", lambda x: float(np.nanmean(np.asarray(x, dtype=float) >= 0.95))),
            median_rmse=("rmse", "median"),
        )
        .reset_index()
        .sort_values(["system_id", "median_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    idx = per_curve.groupby("unified_curve_id")["rmse"].idxmin()
    best_family = per_curve.loc[idx].copy().reset_index(drop=True)
    best_family_summary = (
        best_family.groupby(["system_id", "family"], dropna=False)
        .agg(
            n_best=("unified_curve_id", "count"),
            median_r2=("r2", "median"),
            median_rmse=("rmse", "median"),
        )
        .reset_index()
        .sort_values(["system_id", "n_best"], ascending=[True, False])
        .reset_index(drop=True)
    )

    fpca_matrix = np.vstack(fpca_rows)
    fpca_centered = fpca_matrix - fpca_matrix.mean(axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(fpca_centered, full_matrices=False)
    var = svals**2
    explained = (var / max(np.sum(var), 1e-12)).tolist()
    fpca_support = {
        "n_curves": int(fpca_matrix.shape[0]),
        "grid_points": int(fpca_matrix.shape[1]),
        "pc1_explained": float(explained[0]),
        "pc2_explained": float(explained[1]) if len(explained) > 1 else np.nan,
        "pc3_explained": float(explained[2]) if len(explained) > 2 else np.nan,
        "pc12_explained": float(np.sum(explained[:2])),
        "pc123_explained": float(np.sum(explained[:3])),
        "note": (
            "FPCA is computed on best-family fitted curves over normalized time and is supporting evidence only; "
            "the gate should lean on per-curve residual structure first."
        ),
    }
    return per_curve, family_summary, best_family_summary, fpca_support


def canonical_param_frame(per_curve: pd.DataFrame) -> pd.DataFrame:
    sub = per_curve[per_curve["family"] == CANONICAL_FAMILY].copy().reset_index(drop=True)
    params = sub["params_json"].apply(json.loads)
    sub["qmax"] = [float(p[0]) for p in params]
    sub["w_fast"] = [float(p[1]) for p in params]
    sub["k_fast"] = [float(p[2]) for p in params]
    sub["k_slow"] = [float(p[3]) for p in params]
    return sub


def canonical_param_matrix(canonical: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log(clamp_positive(canonical["qmax"].to_numpy(dtype=float))),
            logit(canonical["w_fast"].to_numpy(dtype=float)),
            np.log(clamp_positive(canonical["k_fast"].to_numpy(dtype=float))),
            np.log(clamp_positive(canonical["k_slow"].to_numpy(dtype=float))),
        ]
    )


def inverse_canonical_params(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    qmax = np.exp(theta[..., 0])
    w_fast = sigmoid(theta[..., 1])
    k_fast = np.exp(theta[..., 2])
    k_slow = np.exp(theta[..., 3])
    return np.stack([qmax, w_fast, k_fast, k_slow], axis=-1)


def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    dist = cdist(data, data)
    scores: list[float] = []
    for i in range(len(data)):
        own = labels[i]
        same_mask = labels == own
        same_mask[i] = False
        a = float(np.mean(dist[i, same_mask])) if np.any(same_mask) else 0.0
        b = np.inf
        for other in unique:
            if other == own:
                continue
            other_mask = labels == other
            if not np.any(other_mask):
                continue
            b = min(b, float(np.mean(dist[i, other_mask])))
        if not np.isfinite(b):
            continue
        denom = max(a, b, 1e-12)
        scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else np.nan


def run_q2(canonical: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = canonical_param_matrix(canonical)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    standardized = (raw - mean) / np.where(std > 0, std, 1.0)

    selection_rows: list[dict[str, object]] = []
    best_labels: np.ndarray | None = None
    best_score = -np.inf
    best_k = 2
    max_k = min(6, len(canonical) - 1)
    rng = np.random.default_rng(seed)

    for k in range(2, max_k + 1):
        centroids, labels = kmeans2(standardized, k=k, minit="points", iter=50, seed=rng)
        score = silhouette_score(standardized, labels)
        selection_rows.append(
            {
                "k": k,
                "silhouette": score,
                "cluster_sizes": json.dumps(pd.Series(labels).value_counts().sort_index().tolist()),
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        _, best_labels = kmeans2(standardized, k=2, minit="points", iter=50, seed=rng)
        best_k = 2

    assignments = canonical[
        ["unified_curve_id", "system_id", "source_dataset", "polymer_family", "qmax", "w_fast", "k_fast", "k_slow", "r2", "rmse"]
    ].copy()
    assignments["cluster_id"] = best_labels

    cluster_rows: list[dict[str, object]] = []
    for cluster_id, sub in assignments.groupby("cluster_id", sort=True):
        counts = sub["system_id"].value_counts()
        top = counts.index[0]
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_curves": int(len(sub)),
                "n_systems": int(sub["system_id"].nunique()),
                "dominant_system": str(top),
                "dominant_system_fraction": float(counts.iloc[0] / len(sub)),
                "median_r2": float(np.median(sub["r2"])),
                "system_counts_json": json.dumps(counts.to_dict(), ensure_ascii=True),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)
    selection = pd.DataFrame(selection_rows).sort_values("k").reset_index(drop=True)
    selection["selected"] = selection["k"] == best_k
    return assignments, cluster_summary, selection


def early_feature_vector(sub: pd.DataFrame, n_obs: int) -> np.ndarray | None:
    ordered = sub.sort_values("time_days").reset_index(drop=True)
    if len(ordered) < max(n_obs, 1):
        return None
    if n_obs == 0:
        return np.array([1.0], dtype=float)
    part = ordered.iloc[:n_obs]
    t = part["time_days"].to_numpy(dtype=float)
    y = part["release_fraction"].to_numpy(dtype=float)
    t_norm = t / max(float(ordered["time_days"].iloc[-1]), 1e-8)
    return np.concatenate([np.array([1.0], dtype=float), t_norm, y])


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    # Keep the intercept column untouched.
    std[:, 0] = 1.0
    mean[:, 0] = 0.0
    x_train_std = (x_train - mean) / np.where(std > 0, std, 1.0)
    x_test_std = (x_test - mean) / np.where(std > 0, std, 1.0)
    penalty = alpha * np.eye(x_train_std.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train_std.T @ x_train_std + penalty, x_train_std.T @ y_train)
    return x_test_std @ beta


def predict_curve_from_theta(t: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    params = inverse_canonical_params(theta_hat.reshape(1, -1))[0]
    return biexponential_curve(t, params)


def run_q3(
    curve_groups: dict[str, pd.DataFrame],
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = canonical.copy()
    theta = canonical_param_matrix(canonical)
    canonical["theta_0"] = theta[:, 0]
    canonical["theta_1"] = theta[:, 1]
    canonical["theta_2"] = theta[:, 2]
    canonical["theta_3"] = theta[:, 3]
    target_cols = ["theta_0", "theta_1", "theta_2", "theta_3"]

    rows: list[dict[str, object]] = []
    for n_obs in Q3_EARLY_OBS:
        feature_map: dict[str, np.ndarray] = {}
        for curve_id, sub in curve_groups.items():
            vec = early_feature_vector(sub, n_obs)
            if vec is not None:
                feature_map[curve_id] = vec

        eligible = canonical[canonical["unified_curve_id"].isin(feature_map)].copy().reset_index(drop=True)
        for system_id, system_sub in eligible.groupby("system_id", sort=True):
            system_curve_ids = system_sub["unified_curve_id"].tolist()
            for curve_id in system_curve_ids:
                test_meta = eligible[eligible["unified_curve_id"] == curve_id].iloc[0]
                test_curve = curve_groups[curve_id]
                t_test = test_curve["time_days"].to_numpy(dtype=float)
                y_test = test_curve["release_fraction"].to_numpy(dtype=float)
                x_test = feature_map[curve_id].reshape(1, -1)

                within_train = eligible[
                    (eligible["system_id"] == system_id) & (eligible["unified_curve_id"] != curve_id)
                ].copy()
                transfer_train = eligible[eligible["unified_curve_id"] != curve_id].copy()
                for mode, train_df in (("within_system", within_train), ("pooled_transfer", transfer_train)):
                    if len(train_df) < 2:
                        continue
                    x_train = np.vstack([feature_map[cid] for cid in train_df["unified_curve_id"]])
                    y_train = train_df[target_cols].to_numpy(dtype=float)
                    theta_hat = ridge_predict(x_train=x_train, y_train=y_train, x_test=x_test, alpha=RIDGE_ALPHA)[0]
                    pred = predict_curve_from_theta(t_test, theta_hat)
                    rows.append(
                        {
                            "unified_curve_id": curve_id,
                            "system_id": system_id,
                            "mode": mode,
                            "n_obs": int(n_obs),
                            "n_train_curves": int(len(train_df)),
                            "r2": safe_r2(y_test, pred),
                            "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
                            "mae": float(np.mean(np.abs(pred - y_test))),
                            "release_final": float(y_test[-1]),
                            "duration_days": float(t_test[-1] - t_test[0]),
                            "canonical_fit_r2": float(test_meta["r2"]),
                        }
                    )

    metrics = pd.DataFrame(rows)
    within = metrics[metrics["mode"] == "within_system"].rename(
        columns={"r2": "within_r2", "rmse": "within_rmse", "mae": "within_mae"}
    )
    transfer = metrics[metrics["mode"] == "pooled_transfer"].rename(
        columns={"r2": "transfer_r2", "rmse": "transfer_rmse", "mae": "transfer_mae"}
    )
    paired = within.merge(
        transfer[
            ["unified_curve_id", "system_id", "n_obs", "transfer_r2", "transfer_rmse", "transfer_mae", "n_train_curves"]
        ],
        on=["unified_curve_id", "system_id", "n_obs"],
        how="inner",
        suffixes=("", "_transfer"),
    )
    paired["delta_r2"] = paired["transfer_r2"] - paired["within_r2"]
    paired["delta_rmse"] = paired["transfer_rmse"] - paired["within_rmse"]

    summary = (
        paired.groupby(["system_id", "n_obs"], dropna=False)
        .agg(
            n_eval_curves=("unified_curve_id", "count"),
            within_median_r2=("within_r2", "median"),
            transfer_median_r2=("transfer_r2", "median"),
            median_delta_r2=("delta_r2", "median"),
            mean_delta_r2=("delta_r2", "mean"),
            transfer_win_fraction=("delta_r2", lambda x: float(np.mean(np.asarray(x, dtype=float) > 0.0))),
        )
        .reset_index()
        .sort_values(["n_obs", "system_id"])
        .reset_index(drop=True)
    )
    return paired, summary


def build_verdict(
    best_family: pd.DataFrame,
    fpca_support: dict[str, object],
    cluster_summary: pd.DataFrame,
    q3_summary: pd.DataFrame,
) -> dict[str, object]:
    valid_q1_r2 = best_family["r2"].dropna().to_numpy(dtype=float)
    q1_frac = float(np.mean(valid_q1_r2 >= 0.95))
    q1_median = float(np.median(valid_q1_r2))
    q2_cross = float(np.mean(cluster_summary["n_systems"].to_numpy(dtype=float) > 1.0))
    q2_dom = float(np.median(cluster_summary["dominant_system_fraction"].to_numpy(dtype=float)))
    q3_by_obs = {}
    for n_obs, sub in q3_summary.groupby("n_obs", sort=True):
        q3_by_obs[int(n_obs)] = {
            "systems_with_positive_median_delta": int(np.sum(sub["median_delta_r2"].to_numpy(dtype=float) > 0.0)),
            "n_systems": int(len(sub)),
            "median_of_system_median_delta_r2": float(np.median(sub["median_delta_r2"].to_numpy(dtype=float))),
            "median_transfer_win_fraction": float(np.median(sub["transfer_win_fraction"].to_numpy(dtype=float))),
        }

    if q1_median >= 0.95 and q1_frac >= 0.70:
        grammar_verdict = "thin"
    elif q1_median >= 0.90:
        grammar_verdict = "moderately_thin"
    else:
        grammar_verdict = "thick_or_unresolved"

    if q2_cross >= 0.6 and q2_dom <= 0.85:
        structure_verdict = "shared_cross_system_families_present"
    elif q2_cross <= 0.3 and q2_dom >= 0.9:
        structure_verdict = "mostly_system_islands"
    else:
        structure_verdict = "mixed_partial_sharing"

    obs0 = q3_by_obs.get(0, {})
    obs5 = q3_by_obs.get(5, {})
    if (
        obs0.get("median_of_system_median_delta_r2", -np.inf) <= 0.0
        and obs5.get("median_of_system_median_delta_r2", -np.inf) > 0.0
    ):
        q3_verdict = "observations_unlock_transfer"
    elif obs5.get("median_of_system_median_delta_r2", -np.inf) <= 0.0:
        q3_verdict = "pooled_transfer_not_helping_even_with_observations"
    else:
        q3_verdict = "pooled_transfer_helps_without_strong_observation_dependence"

    return {
        "q1": {
            "best_family_median_r2": q1_median,
            "best_family_fraction_r2_ge_095": q1_frac,
            "fpca_pc123_explained": float(fpca_support["pc123_explained"]),
            "verdict": grammar_verdict,
        },
        "q2": {
            "cluster_cross_system_fraction": q2_cross,
            "cluster_median_dominant_system_fraction": q2_dom,
            "verdict": structure_verdict,
        },
        "q3": {
            "by_n_obs": q3_by_obs,
            "verdict": q3_verdict,
        },
    }


def write_summary(
    out_path: Path,
    verdict: dict[str, object],
    family_summary: pd.DataFrame,
    best_family_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    q3_summary: pd.DataFrame,
) -> None:
    q1 = verdict["q1"]
    q2 = verdict["q2"]
    q3 = verdict["q3"]
    lines = [
        "# Curve World Hard Gate",
        "",
        "This is the gate for the current `148_release_main_cumulative_v1` pool.",
        "",
        "## Topline",
        "",
        f"- Q1 grammar verdict: `{q1['verdict']}`",
        f"- Q2 cross-system structure verdict: `{q2['verdict']}`",
        f"- Q3 transfer verdict: `{q3['verdict']}`",
        "",
        "The safe interpretation is:",
        "",
        "- 751 curves are enough for diagnosis and first-pass modeling.",
        "- They are not enough to pre-announce a strong unseen-system transfer claim.",
        "- The real gate is whether pooled structure survives the system split and whether early observations unlock it.",
        "",
        "## Q1 Residual Gate",
        "",
        f"- best-family median R2: `{q1['best_family_median_r2']:.4f}`",
        f"- fraction of curves with best-family R2 >= 0.95: `{q1['best_family_fraction_r2_ge_095']:.4f}`",
        f"- supporting FPCA cumulative explained variance (PC1-3): `{q1['fpca_pc123_explained']:.4f}`",
        "",
    ]
    top_family = (
        family_summary.sort_values(["system_id", "median_r2"], ascending=[True, False])
        .groupby("system_id", sort=False)
        .head(1)
    )
    lines.append("Best family by system:")
    lines.append("")
    for _, row in top_family.iterrows():
        lines.append(
            f"- `{row['system_id']}`: `{row['family']}` "
            f"(median R2={row['median_r2']:.4f}, frac>=0.95={row['frac_r2_ge_095']:.3f})"
        )
    lines.extend(["", "## Q2 Cluster Gate", ""])
    for _, row in cluster_summary.iterrows():
        lines.append(
            f"- cluster `{int(row['cluster_id'])}`: n={int(row['n_curves'])}, "
            f"systems={int(row['n_systems'])}, dominant=`{row['dominant_system']}`, "
            f"dominant_fraction={row['dominant_system_fraction']:.3f}"
        )
    lines.extend(["", "## Q3 Transfer Gate", ""])
    for n_obs, sub in q3_summary.groupby("n_obs", sort=True):
        lines.append(f"### early observations = {int(n_obs)}")
        lines.append("")
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['system_id']}`: within_med_R2={row['within_median_r2']:.4f}, "
                f"transfer_med_R2={row['transfer_median_r2']:.4f}, "
                f"delta={row['median_delta_r2']:.4f}, "
                f"transfer_win_frac={row['transfer_win_fraction']:.3f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Readout",
            "",
            "- If Q1 is thin, the object shrinks toward low-dimensional shape prediction.",
            "- If Q2 is mostly islands, cross-system syntax is weak and system-local routes dominate.",
            "- If Q3 only turns positive at 3-5 observations, the observation-conditioned route is stronger than static-only transfer.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    curves, _, _ = load_pool(args.pool_dir)
    curve_groups = group_curves(curves)

    per_curve, family_summary, best_family_summary, fpca_support = run_q1(curve_groups)
    canonical = canonical_param_frame(per_curve)
    cluster_assignments, cluster_summary, cluster_selection = run_q2(canonical, seed=args.seed)
    q3_metrics, q3_summary = run_q3(curve_groups, canonical)
    best_idx = per_curve.groupby("unified_curve_id")["rmse"].idxmin()
    best_family = per_curve.loc[best_idx].copy().reset_index(drop=True)
    verdict = build_verdict(best_family, fpca_support, cluster_summary, q3_summary)

    per_curve.to_csv(args.outdir / "q1_per_curve_family_fits.csv", index=False)
    family_summary.to_csv(args.outdir / "q1_family_summary.csv", index=False)
    best_family_summary.to_csv(args.outdir / "q1_best_family_summary.csv", index=False)
    (args.outdir / "q1_fpca_support.json").write_text(json.dumps(fpca_support, indent=2), encoding="utf-8")
    cluster_assignments.to_csv(args.outdir / "q2_cluster_assignments.csv", index=False)
    cluster_summary.to_csv(args.outdir / "q2_cluster_summary.csv", index=False)
    cluster_selection.to_csv(args.outdir / "q2_cluster_selection.csv", index=False)
    q3_metrics.to_csv(args.outdir / "q3_transfer_metrics.csv", index=False)
    q3_summary.to_csv(args.outdir / "q3_transfer_summary.csv", index=False)
    (args.outdir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_summary(
        out_path=args.outdir / "summary.md",
        verdict=verdict,
        family_summary=family_summary,
        best_family_summary=best_family_summary,
        cluster_summary=cluster_summary,
        q3_summary=q3_summary,
    )

    print(f"[curve-world-hard-gate] wrote outputs to {args.outdir}")
    print(
        "[curve-world-hard-gate] "
        f"q1={verdict['q1']['verdict']} "
        f"q2={verdict['q2']['verdict']} "
        f"q3={verdict['q3']['verdict']}"
    )


if __name__ == "__main__":
    main()
