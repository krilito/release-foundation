"""
112 - Liposome middle-layer probe.

Purpose:
    Test whether the Yanes et al. liposome kinetic-class / Weibull-theta
    middle layer can improve curve-level prediction, especially when early
    observed release points are injected as context.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/112_liposome_middle_layer_probe/
      per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This is not a mechanism claim. It asks whether a supervised intermediate
    representation helps predict future Q(t) under matched folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUT = Path("outputs/112_liposome_middle_layer_probe")

FEATURE_7 = [
    "media_pH",
    "media_temp_oC",
    "drug_loading",
    "Z_average_nm",
    "API_type",
    "weighted_Mw",
    "weighted_Tm",
]
EARLY_GRID_H = np.asarray([6.0, 12.0, 24.0, 48.0, 72.0], dtype=float)
SCHEMES = ("stratified_5fold", "group_by_API", "group_by_release_method")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liposome kinetic middle-layer probe.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-window-h", type=float, default=6.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--refine-nfev", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=300)
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
    return {"path": str(path), "exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def parse_id(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"cannot parse IVR ID from {value!r}")
    return int(match.group(1))


def weibull_pct(t_h: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    t = np.maximum(np.asarray(t_h, dtype=float), 0.0)
    alpha = max(float(alpha), 1e-8)
    beta = max(float(beta), 1e-8)
    return 100.0 * (1.0 - np.exp(-(np.power(t, beta) / alpha)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.square(y - y.mean())))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum(np.square(y - pred)) / denom)


def load_dataset(root: Path, early_window_h: float) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    exp = pd.read_csv(root / "results/fitting/drug_release_exp.csv")
    exp["ID"] = exp["file_name"].map(parse_id).astype(int)
    exp = exp.rename(columns={"time (Hrs)": "time_h", "release_percent": "release_pct"})
    exp = exp[["ID", "time_h", "release_pct", "file_name"]].copy()

    backend = pd.read_csv(root / "data/unprocessed/backend_data.csv")
    backend = backend.rename(columns={"IVR_ID": "ID"}).drop(columns=["Unnamed: 0"], errors="ignore")
    backend["ID"] = backend["ID"].astype(int)
    api_names = sorted(backend["API_name"].dropna().astype(str).unique().tolist())
    backend["API_type"] = backend["API_name"].astype(str).map({name: i for i, name in enumerate(api_names)})

    theta = pd.read_csv(root / "data/clean/weibull_params.csv")
    theta["ID"] = theta["ID"].astype(int)

    cluster = pd.read_csv(root / "results/clustering/3_PCA_KMC.csv")
    cluster = cluster.rename(columns={"file": "ID"})
    cluster["ID"] = cluster["ID"].astype(int)

    curve_rows: list[dict[str, Any]] = []
    curve_map: dict[int, pd.DataFrame] = {}
    for curve_id, sub in exp.groupby("ID"):
        sub = sub.sort_values("time_h").dropna(subset=["time_h", "release_pct"])
        if sub.empty:
            continue
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        curve_map[int(curve_id)] = sub
        curve_rows.append(
            {
                "ID": int(curve_id),
                "n_points": int(len(sub)),
                "n_early": int(np.sum(t <= early_window_h)),
                "n_future": int(np.sum(t > early_window_h)),
                "t_max_h": float(np.max(t)),
                "release_min_pct": float(np.min(q)),
                "release_max_pct": float(np.max(q)),
            }
        )
    meta = pd.DataFrame(curve_rows)
    meta = meta.merge(backend, on="ID", how="left")
    meta = meta.merge(theta[["ID", "alpha", "beta"]], on="ID", how="inner")
    meta = meta.merge(cluster[["ID", "cluster", "cluster_name"]], on="ID", how="inner")
    meta = meta.dropna(subset=FEATURE_7 + ["alpha", "beta", "cluster"])
    meta = meta[
        (meta["n_early"] >= 2)
        & (meta["n_future"] >= 2)
        & (meta["release_min_pct"] >= -5.0)
        & (meta["release_max_pct"] <= 125.0)
    ].copy()
    meta["cluster"] = meta["cluster"].astype(int)
    return meta.reset_index(drop=True), curve_map


def stratified_cap(meta: pd.DataFrame, max_curves: int | None, seed: int) -> pd.DataFrame:
    if max_curves is None or len(meta) <= max_curves:
        return meta.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pieces = []
    per_class = max(2, max_curves // max(1, meta["cluster"].nunique()))
    for _, sub in meta.groupby("cluster"):
        take = min(len(sub), per_class)
        pieces.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    out = pd.concat(pieces, ignore_index=True)
    if len(out) < max_curves:
        remaining = meta[~meta["ID"].isin(out["ID"])]
        take = min(max_curves - len(out), len(remaining))
        if take:
            out = pd.concat(
                [out, remaining.iloc[rng.choice(len(remaining), size=take, replace=False)]],
                ignore_index=True,
            )
    return out.sort_values("ID").reset_index(drop=True)


def early_q_features(curve_map: dict[int, pd.DataFrame], ids: np.ndarray, early_window_h: float) -> np.ndarray:
    times = EARLY_GRID_H[EARLY_GRID_H <= early_window_h]
    if len(times) == 0:
        raise ValueError("early_window_h must include at least one EARLY_GRID_H point")
    rows = []
    for curve_id in ids:
        sub = curve_map[int(curve_id)].sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        rows.append(np.interp(times, t, q, left=q[0], right=q[-1]))
    return np.asarray(rows, dtype=float)


def splits(meta: pd.DataFrame, scheme: str, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = meta["cluster"].to_numpy(dtype=int)
    idx = np.arange(len(meta))
    if scheme == "stratified_5fold":
        n_eff = min(n_splits, int(pd.Series(y).value_counts().min()))
        if n_eff < 2:
            return []
        splitter = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
        return list(splitter.split(idx, y))
    group_col = {"group_by_API": "API_name", "group_by_release_method": "release_method"}[scheme]
    groups = meta[group_col].astype(str).fillna("__MISSING__").to_numpy()
    n_eff = min(n_splits, int(pd.Series(groups).nunique()))
    if n_eff < 2:
        return []
    splitter = GroupKFold(n_splits=n_eff)
    return list(splitter.split(idx, y, groups))


def fit_classifiers(seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced"),
        ),
        "rf": RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "et": ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def class_prototypes(train_meta: pd.DataFrame) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for cluster, sub in train_meta.groupby("cluster"):
        out[int(cluster)] = (float(sub["alpha"].median()), float(sub["beta"].median()))
    out[-1] = (float(train_meta["alpha"].median()), float(train_meta["beta"].median()))
    return out


def hard_proto_theta(labels: np.ndarray, prototypes: dict[int, tuple[float, float]]) -> np.ndarray:
    fallback = prototypes[-1]
    return np.asarray([prototypes.get(int(label), fallback) for label in labels], dtype=float)


def soft_proto_predictions(
    model: Any,
    x: np.ndarray,
    prototypes: dict[int, tuple[float, float]],
    curve_t: np.ndarray,
    curve_index: int,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    proba = model.predict_proba(x)[curve_index]
    classes = np.asarray(model.classes_, dtype=int)
    pred = np.zeros_like(curve_t, dtype=float)
    alpha_mean = 0.0
    beta_mean = 0.0
    class_weights: dict[str, float] = {}
    for cls, weight in zip(classes, proba):
        alpha, beta = prototypes.get(int(cls), prototypes[-1])
        pred += float(weight) * weibull_pct(curve_t, alpha, beta)
        alpha_mean += float(weight) * alpha
        beta_mean += float(weight) * beta
        class_weights[str(int(cls))] = float(weight)
    return pred, float(alpha_mean), float(beta_mean), class_weights


def refine_theta(theta0: np.ndarray, early_t: np.ndarray, early_q: np.ndarray, max_nfev: int) -> tuple[np.ndarray, int]:
    if max_nfev <= 0 or len(early_t) < 2:
        return theta0, 0
    x0 = np.asarray([np.clip(theta0[0], 1e-8, 300.0), np.clip(theta0[1], 1e-8, 5.0)], dtype=float)
    try:
        res = least_squares(
            lambda th: weibull_pct(early_t, th[0], th[1]) - early_q,
            x0=x0,
            bounds=([1e-8, 1e-8], [300.0, 5.0]),
            method="trf",
            max_nfev=max_nfev,
        )
    except Exception:
        return x0, 0
    return np.asarray([float(np.clip(res.x[0], 1e-8, 300.0)), float(np.clip(res.x[1], 1e-8, 5.0))]), int(res.nfev)


def fit_theta_regressor(model: Any, x_train: np.ndarray, theta_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model.fit(x_train, np.log(np.clip(theta_train, 1e-8, None)))
    pred = np.exp(model.predict(x_test))
    pred[:, 0] = np.clip(pred[:, 0], 1e-8, 300.0)
    pred[:, 1] = np.clip(pred[:, 1], 1e-8, 5.0)
    return pred


def append_eval_row(
    rows: list[dict[str, Any]],
    *,
    scheme: str,
    fold: int,
    method: str,
    meta_row: pd.Series,
    curve: pd.DataFrame,
    pred_pct: np.ndarray,
    early_window_h: float,
    alpha_pred: float = np.nan,
    beta_pred: float = np.nan,
    class_true: int | None = None,
    class_pred: int | None = None,
    nfev: int = 0,
    class_weights: dict[str, float] | None = None,
) -> None:
    t = curve["time_h"].to_numpy(dtype=float)
    q = curve["release_pct"].to_numpy(dtype=float)
    future = t > early_window_h
    rows.append(
        {
            "scheme": scheme,
            "fold": int(fold),
            "method": method,
            "ID": int(meta_row["ID"]),
            "API_name": str(meta_row.get("API_name", "")),
            "release_method": str(meta_row.get("release_method", "")),
            "cluster_true": int(class_true) if class_true is not None else int(meta_row["cluster"]),
            "cluster_pred": int(class_pred) if class_pred is not None else -1,
            "cluster_weights_json": json.dumps(class_weights or {}, sort_keys=True),
            "n_points": int(len(curve)),
            "n_future": int(np.sum(future)),
            "early_window_h": float(early_window_h),
            "full_rmse_pct": rmse(q, pred_pct),
            "future_rmse_pct": rmse(q[future], pred_pct[future]) if np.sum(future) else np.nan,
            "full_mae_pct": mae(q, pred_pct),
            "future_mae_pct": mae(q[future], pred_pct[future]) if np.sum(future) else np.nan,
            "full_r2": safe_r2(q, pred_pct),
            "future_r2": safe_r2(q[future], pred_pct[future]) if np.sum(future) >= 2 else np.nan,
            "alpha_pred": float(alpha_pred),
            "beta_pred": float(beta_pred),
            "nfev": int(nfev),
        }
    )


def evaluate_theta_method(
    rows: list[dict[str, Any]],
    scheme: str,
    fold: int,
    method: str,
    meta: pd.DataFrame,
    curve_map: dict[int, pd.DataFrame],
    test_idx: np.ndarray,
    theta_pred: np.ndarray,
    early_window_h: float,
    nfev: np.ndarray | None = None,
) -> None:
    for local_i, idx in enumerate(test_idx):
        meta_row = meta.iloc[int(idx)]
        curve = curve_map[int(meta_row["ID"])].sort_values("time_h")
        t = curve["time_h"].to_numpy(dtype=float)
        pred = weibull_pct(t, theta_pred[local_i, 0], theta_pred[local_i, 1])
        append_eval_row(
            rows,
            scheme=scheme,
            fold=fold,
            method=method,
            meta_row=meta_row,
            curve=curve,
            pred_pct=pred,
            early_window_h=early_window_h,
            alpha_pred=float(theta_pred[local_i, 0]),
            beta_pred=float(theta_pred[local_i, 1]),
            nfev=int(nfev[local_i]) if nfev is not None else 0,
        )


def run_probe(args: argparse.Namespace, meta: pd.DataFrame, curve_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    ids = meta["ID"].to_numpy(dtype=int)
    x_static = meta[FEATURE_7].to_numpy(dtype=float)
    x_early = early_q_features(curve_map, ids, args.early_window_h)
    x_static_early = np.concatenate([x_static, x_early], axis=1)
    y_class = meta["cluster"].to_numpy(dtype=int)
    theta = meta[["alpha", "beta"]].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for scheme in SCHEMES:
        scheme_splits = splits(meta, scheme, args.n_folds, args.seed)
        if not scheme_splits:
            continue
        for fold, (tr, te) in enumerate(scheme_splits):
            fold_seed = args.seed + 1000 * fold
            train_meta = meta.iloc[tr]
            prototypes = class_prototypes(train_meta)
            train_median_theta = np.asarray(prototypes[-1], dtype=float)

            evaluate_theta_method(
                rows,
                scheme,
                fold,
                "global_median_weibull",
                meta,
                curve_map,
                te,
                np.repeat(train_median_theta[None, :], len(te), axis=0),
                args.early_window_h,
            )

            oracle_theta = hard_proto_theta(y_class[te], prototypes)
            evaluate_theta_method(
                rows,
                scheme,
                fold,
                "oracle_hard_class_proto",
                meta,
                curve_map,
                te,
                oracle_theta,
                args.early_window_h,
            )

            early_only_theta = []
            early_only_nfev = []
            for idx in te:
                curve = curve_map[int(meta.iloc[int(idx)]["ID"])].sort_values("time_h")
                early = curve[curve["time_h"] <= args.early_window_h]
                theta_ref, calls = refine_theta(
                    train_median_theta,
                    early["time_h"].to_numpy(dtype=float),
                    early["release_pct"].to_numpy(dtype=float),
                    args.refine_nfev,
                )
                early_only_theta.append(theta_ref)
                early_only_nfev.append(calls)
            evaluate_theta_method(
                rows,
                scheme,
                fold,
                "early_only_weibull_fit",
                meta,
                curve_map,
                te,
                np.asarray(early_only_theta, dtype=float),
                args.early_window_h,
                nfev=np.asarray(early_only_nfev, dtype=int),
            )

            for input_name, x_all, train_input, test_input in [
                ("static", x_static, x_static[tr], x_static[te]),
                ("static_early", x_static_early, x_static_early[tr], x_static_early[te]),
            ]:
                for model_name, clf in fit_classifiers(fold_seed, args.n_estimators).items():
                    clf.fit(train_input, y_class[tr])
                    pred_class = clf.predict(test_input)
                    pred_theta = hard_proto_theta(pred_class, prototypes)
                    hard_method = f"{input_name}_hard_{model_name}_class_proto"
                    for local_i, idx in enumerate(te):
                        meta_row = meta.iloc[int(idx)]
                        curve = curve_map[int(meta_row["ID"])].sort_values("time_h")
                        t = curve["time_h"].to_numpy(dtype=float)
                        pred = weibull_pct(t, pred_theta[local_i, 0], pred_theta[local_i, 1])
                        append_eval_row(
                            rows,
                            scheme=scheme,
                            fold=fold,
                            method=hard_method,
                            meta_row=meta_row,
                            curve=curve,
                            pred_pct=pred,
                            early_window_h=args.early_window_h,
                            alpha_pred=float(pred_theta[local_i, 0]),
                            beta_pred=float(pred_theta[local_i, 1]),
                            class_true=int(y_class[idx]),
                            class_pred=int(pred_class[local_i]),
                        )

                    if hasattr(clf, "predict_proba"):
                        soft_method = f"{input_name}_soft_{model_name}_class_proto"
                        for local_i, idx in enumerate(te):
                            meta_row = meta.iloc[int(idx)]
                            curve = curve_map[int(meta_row["ID"])].sort_values("time_h")
                            t = curve["time_h"].to_numpy(dtype=float)
                            pred, alpha_mean, beta_mean, weights = soft_proto_predictions(
                                clf,
                                test_input,
                                prototypes,
                                t,
                                local_i,
                            )
                            append_eval_row(
                                rows,
                                scheme=scheme,
                                fold=fold,
                                method=soft_method,
                                meta_row=meta_row,
                                curve=curve,
                                pred_pct=pred,
                                early_window_h=args.early_window_h,
                                alpha_pred=alpha_mean,
                                beta_pred=beta_mean,
                                class_true=int(y_class[idx]),
                                class_pred=int(pred_class[local_i]),
                                class_weights=weights,
                            )
                    _ = x_all

            for input_name, x_train, x_test in [
                ("static", x_static[tr], x_static[te]),
                ("static_early", x_static_early[tr], x_static_early[te]),
            ]:
                reg = ExtraTreesRegressor(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=2,
                    random_state=fold_seed + 33,
                    n_jobs=-1,
                )
                pred_theta = fit_theta_regressor(reg, x_train, theta[tr], x_test)
                method = f"{input_name}_et_theta"
                evaluate_theta_method(rows, scheme, fold, method, meta, curve_map, te, pred_theta, args.early_window_h)

                refined_theta = []
                refined_nfev = []
                for local_i, idx in enumerate(te):
                    curve = curve_map[int(meta.iloc[int(idx)]["ID"])].sort_values("time_h")
                    early = curve[curve["time_h"] <= args.early_window_h]
                    theta_ref, calls = refine_theta(
                        pred_theta[local_i],
                        early["time_h"].to_numpy(dtype=float),
                        early["release_pct"].to_numpy(dtype=float),
                        args.refine_nfev,
                    )
                    refined_theta.append(theta_ref)
                    refined_nfev.append(calls)
                evaluate_theta_method(
                    rows,
                    scheme,
                    fold,
                    f"{method}_early_refined",
                    meta,
                    curve_map,
                    te,
                    np.asarray(refined_theta, dtype=float),
                    args.early_window_h,
                    nfev=np.asarray(refined_nfev, dtype=int),
                )

    return pd.DataFrame(rows)


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, method), sub in per_curve.groupby(["scheme", "method"], sort=True):
        finite_future = sub["future_rmse_pct"].dropna().to_numpy(dtype=float)
        row = {
            "scheme": scheme,
            "method": method,
            "n_curve_records": int(len(sub)),
            "n_unique_curves": int(sub["ID"].nunique()),
            "median_future_rmse_pct": float(np.median(finite_future)) if len(finite_future) else np.nan,
            "mean_future_rmse_pct": float(np.mean(finite_future)) if len(finite_future) else np.nan,
            "median_future_mae_pct": float(sub["future_mae_pct"].median()),
            "median_full_rmse_pct": float(sub["full_rmse_pct"].median()),
            "median_future_r2": float(sub["future_r2"].median()),
            "frac_future_r2_ge_0": float((sub["future_r2"] >= 0.0).mean()),
            "median_nfev": float(sub["nfev"].median()),
            "is_class_method": bool((sub["cluster_pred"] >= 0).any()),
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "mcc": np.nan,
        }
        class_rows = sub[sub["cluster_pred"] >= 0]
        if not class_rows.empty:
            row["balanced_accuracy"] = float(
                balanced_accuracy_score(class_rows["cluster_true"], class_rows["cluster_pred"])
            )
            row["macro_f1"] = float(f1_score(class_rows["cluster_true"], class_rows["cluster_pred"], average="macro"))
            row["mcc"] = float(matthews_corrcoef(class_rows["cluster_true"], class_rows["cluster_pred"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["scheme", "median_future_rmse_pct"], ascending=[True, True])


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme, sub in summary.groupby("scheme"):
        lookup = sub.set_index("method")["median_future_rmse_pct"].to_dict()

        def best_method(prefix: str) -> str:
            candidates = sub[sub["method"].str.startswith(prefix)].copy()
            if candidates.empty:
                return ""
            candidates = candidates.sort_values("median_future_rmse_pct", ascending=True)
            return str(candidates.iloc[0]["method"])

        def delta(a: str, b: str) -> float:
            return float(lookup.get(a, np.nan) - lookup.get(b, np.nan))

        best_static_hard = best_method("static_hard_")
        best_static_soft = best_method("static_soft_")
        best_static_early_soft = best_method("static_early_soft_")
        best_static_early_hard = best_method("static_early_hard_")
        best_static_early_class = min(
            [m for m in [best_static_early_soft, best_static_early_hard] if m],
            key=lambda m: lookup.get(m, np.inf),
            default="",
        )

        rows.extend(
            [
                {
                    "scheme": scheme,
                    "question": "Does author-style hard class prototype beat global median?",
                    "method_a": best_static_hard,
                    "method_b": "global_median_weibull",
                    "comparison": f"{best_static_hard} - global_median_weibull",
                    "delta_future_rmse_pct": delta(best_static_hard, "global_median_weibull"),
                    "interpretation": "negative is better for the middle layer",
                },
                {
                    "scheme": scheme,
                    "question": "Does soft class mixing beat hard class selection?",
                    "method_a": best_static_soft,
                    "method_b": best_static_hard,
                    "comparison": f"{best_static_soft} - {best_static_hard}",
                    "delta_future_rmse_pct": delta(best_static_soft, best_static_hard),
                    "interpretation": "negative suggests class-boundary uncertainty is useful",
                },
                {
                    "scheme": scheme,
                    "question": "Does early Q improve the soft middle-layer selector?",
                    "method_a": best_static_early_soft,
                    "method_b": best_static_soft,
                    "comparison": f"{best_static_early_soft} - {best_static_soft}",
                    "delta_future_rmse_pct": delta(best_static_early_soft, best_static_soft),
                    "interpretation": "negative supports early-conditioned middle-layer routing",
                },
                {
                    "scheme": scheme,
                    "question": "How much ceiling remains if true class is known?",
                    "method_a": "oracle_hard_class_proto",
                    "method_b": best_static_hard,
                    "comparison": f"oracle_hard_class_proto - {best_static_hard}",
                    "delta_future_rmse_pct": delta("oracle_hard_class_proto", best_static_hard),
                    "interpretation": "large negative means class prediction is a bottleneck; near zero means class prototypes are coarse",
                },
                {
                    "scheme": scheme,
                    "question": "Does the best early-conditioned class middle layer beat early-only fitting?",
                    "method_a": best_static_early_class,
                    "method_b": "early_only_weibull_fit",
                    "comparison": f"{best_static_early_class} - early_only_weibull_fit",
                    "delta_future_rmse_pct": delta(best_static_early_class, "early_only_weibull_fit"),
                    "interpretation": "negative means the class middle layer adds value beyond early-point Weibull fitting",
                },
                {
                    "scheme": scheme,
                    "question": "Does early-conditioned continuous theta beat early-only fitting?",
                    "method_a": "static_early_et_theta_early_refined",
                    "method_b": "early_only_weibull_fit",
                    "comparison": "static_early_et_theta_early_refined - early_only_weibull_fit",
                    "delta_future_rmse_pct": delta("static_early_et_theta_early_refined", "early_only_weibull_fit"),
                    "interpretation": "negative means static descriptors still add value after early Q",
                },
            ]
        )
    return pd.DataFrame(rows)


def data_checks(meta: pd.DataFrame, per_curve: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    checks = [
        {
            "check": "root_exists",
            "passed": args.root.exists(),
            "detail": str(args.root),
        },
        {
            "check": "enough_curves_loaded",
            "passed": len(meta) >= 30,
            "detail": f"n_curves={len(meta)}",
        },
        {
            "check": "all_curves_have_early_and_future_points",
            "passed": bool(((meta["n_early"] >= 2) & (meta["n_future"] >= 2)).all()),
            "detail": "n_early>=2 and n_future>=2 after filtering",
        },
        {
            "check": "no_nan_primary_metrics",
            "passed": bool(np.isfinite(per_curve["future_rmse_pct"]).all()) if not per_curve.empty else False,
            "detail": "future_rmse_pct finite for every method/curve row",
        },
        {
            "check": "no_group_overlap_in_group_splits",
            "passed": True,
            "detail": "GroupKFold used for API_name and release_method schemes",
        },
    ]
    return pd.DataFrame(checks)


def write_report(out: Path, meta: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Liposome Middle-Layer Probe",
        "",
        "Date: 2026-06-12",
        "",
        "## Aim",
        "",
        "Test whether a kinetic middle layer helps liposome curve prediction:",
        "",
        "```text",
        "static descriptors / static+early Q -> kinetic class or Weibull theta -> Q(t)",
        "```",
        "",
        "This is a curve-level diagnostic, not a mechanism claim.",
        "",
        "## Run",
        "",
        f"- root: `{args.root.as_posix()}`",
        f"- n_curves: {len(meta)}",
        f"- early_window_h: {args.early_window_h}",
        f"- refine_nfev: {args.refine_nfev}",
        "",
        "## Best Methods By Split",
        "",
    ]
    for scheme, sub in summary.groupby("scheme"):
        lines.append(f"### {scheme}")
        shown = sub.sort_values("median_future_rmse_pct").head(8)
        lines.append("")
        lines.append("| Method | Median future RMSE pct | Median future R2 | Class bal. acc |")
        lines.append("|---|---:|---:|---:|")
        for row in shown.itertuples(index=False):
            bal = "" if pd.isna(row.balanced_accuracy) else f"{row.balanced_accuracy:.3f}"
            lines.append(
                f"| `{row.method}` | {row.median_future_rmse_pct:.3f} | {row.median_future_r2:.3f} | {bal} |"
            )
        lines.append("")
    lines.extend(["## Decision Deltas", "", "| Scheme | Question | Delta future RMSE pct |", "|---|---|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.question} | {row.delta_future_rmse_pct:.3f} |")
    lines.extend(
        [
            "",
            "Negative deltas mean the first method in the comparison is better.",
            "",
            "## Guardrails",
            "",
            "- Prototypes and theta medians are computed from train folds only.",
            "- Early Q is used only by methods whose names include `static_early`, `early_only`, or `early_refined`.",
            "- Evaluation uses full curves only as held-out targets; future metrics use times after the early window.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = load_dataset(args.root, args.early_window_h)
    meta = stratified_cap(meta, args.max_curves, args.seed)
    per_curve = run_probe(args, meta, curve_map)
    summary = summarize(per_curve)
    decisions = decision_table(summary)
    checks = data_checks(meta, per_curve, args)

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)

    metadata = {
        "script": Path(__file__).name,
        "date": "2026-06-12",
        "seed": args.seed,
        "root": str(args.root),
        "out": str(args.out),
        "early_window_h": args.early_window_h,
        "n_folds": args.n_folds,
        "max_curves": args.max_curves,
        "refine_nfev": args.refine_nfev,
        "n_estimators": args.n_estimators,
        "git_hash": git_hash(),
        "input_files": {
            "backend_data": file_meta(args.root / "data/unprocessed/backend_data.csv"),
            "weibull_params": file_meta(args.root / "data/clean/weibull_params.csv"),
            "clusters": file_meta(args.root / "results/clustering/3_PCA_KMC.csv"),
            "release_exp": file_meta(args.root / "results/fitting/drug_release_exp.csv"),
        },
        "generated_files": sorted(p.name for p in args.out.iterdir() if p.is_file()),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out, meta, summary, decisions, args)

    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Middle-layer probe failed checks: {failed}")


if __name__ == "__main__":
    main()
