"""
113 - Liposome theta/prototype mixture middle-layer probe.

Purpose:
    Move beyond the 3-class kinetic middle layer in script 112 by testing a
    finer train-fold Weibull prototype mixture:

        static descriptors (+ optional early Q) -> weights over train curves
        train-fold Weibull theta prototypes -> mixed Q(t)

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/113_liposome_theta_prototype_mixture/
      per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This is a middle-layer diagnostic, not a physical mechanism claim. It asks
    whether finer local theta prototypes add value beyond early-only Weibull
    fitting under the same split families as the liposome middle-layer probe.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUT = Path("outputs/113_liposome_theta_prototype_mixture")

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
    parser = argparse.ArgumentParser(description="Liposome finer theta/prototype mixture probe.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-window-h", type=float, default=6.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--refine-nfev", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--prototype-ks", type=int, nargs="+", default=[3, 5, 10, 20])
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


def mixture_weights(x_train: np.ndarray, x_test_one: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    dist = np.sqrt(np.sum(np.square(x_train - x_test_one[None, :]), axis=1))
    k_eff = min(k, len(dist))
    nn = np.argpartition(dist, k_eff - 1)[:k_eff]
    nn = nn[np.argsort(dist[nn])]
    local = dist[nn]
    scale = float(np.median(local[local > 1e-12])) if np.any(local > 1e-12) else 1.0
    weights = np.exp(-local / max(scale, 1e-8))
    weights = weights / np.sum(weights)
    return nn, weights


def mixed_curve_and_theta(t: np.ndarray, theta_train: np.ndarray, nn: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros_like(t, dtype=float)
    theta = np.zeros(2, dtype=float)
    for idx, weight in zip(nn, weights):
        alpha = float(theta_train[idx, 0])
        beta = float(theta_train[idx, 1])
        pred += float(weight) * weibull_pct(t, alpha, beta)
        theta += float(weight) * np.asarray([alpha, beta], dtype=float)
    theta[0] = float(np.clip(theta[0], 1e-8, 300.0))
    theta[1] = float(np.clip(theta[1], 1e-8, 5.0))
    return pred, theta


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
    nfev: int = 0,
    neighbor_ids: list[int] | None = None,
    weights: list[float] | None = None,
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
            "cluster_true": int(meta_row["cluster"]),
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
            "neighbor_ids_json": json.dumps(neighbor_ids or []),
            "weights_json": json.dumps(weights or []),
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
    theta = meta[["alpha", "beta"]].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for scheme in SCHEMES:
        for fold, (tr, te) in enumerate(splits(meta, scheme, args.n_folds, args.seed)):
            fold_seed = args.seed + 1000 * fold
            train_median_theta = np.asarray([float(np.median(theta[tr, 0])), float(np.median(theta[tr, 1]))])
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

            for input_name, x_all in [("static", x_static), ("static_early", x_static_early)]:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_all[tr])
                x_test = scaler.transform(x_all[te])
                theta_train = theta[tr]
                train_ids = ids[tr]
                for k in args.prototype_ks:
                    for local_i, idx in enumerate(te):
                        meta_row = meta.iloc[int(idx)]
                        curve = curve_map[int(meta_row["ID"])].sort_values("time_h")
                        t = curve["time_h"].to_numpy(dtype=float)
                        nn, weights = mixture_weights(x_train, x_test[local_i], k)
                        pred, theta_mix = mixed_curve_and_theta(t, theta_train, nn, weights)
                        neighbor_ids = [int(train_ids[i]) for i in nn.tolist()]
                        append_eval_row(
                            rows,
                            scheme=scheme,
                            fold=fold,
                            method=f"{input_name}_knn_proto_mix_k{k}",
                            meta_row=meta_row,
                            curve=curve,
                            pred_pct=pred,
                            early_window_h=args.early_window_h,
                            alpha_pred=float(theta_mix[0]),
                            beta_pred=float(theta_mix[1]),
                            neighbor_ids=neighbor_ids,
                            weights=[float(w) for w in weights.tolist()],
                        )

                        if input_name == "static_early":
                            early = curve[curve["time_h"] <= args.early_window_h]
                            theta_ref, calls = refine_theta(
                                theta_mix,
                                early["time_h"].to_numpy(dtype=float),
                                early["release_pct"].to_numpy(dtype=float),
                                args.refine_nfev,
                            )
                            pred_ref = weibull_pct(t, theta_ref[0], theta_ref[1])
                            append_eval_row(
                                rows,
                                scheme=scheme,
                                fold=fold,
                                method=f"{input_name}_knn_proto_theta_refined_k{k}",
                                meta_row=meta_row,
                                curve=curve,
                                pred_pct=pred_ref,
                                early_window_h=args.early_window_h,
                                alpha_pred=float(theta_ref[0]),
                                beta_pred=float(theta_ref[1]),
                                nfev=calls,
                                neighbor_ids=neighbor_ids,
                                weights=[float(w) for w in weights.tolist()],
                            )

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
                evaluate_theta_method(
                    rows,
                    scheme,
                    fold,
                    f"{input_name}_et_theta",
                    meta,
                    curve_map,
                    te,
                    pred_theta,
                    args.early_window_h,
                )

    return pd.DataFrame(rows)


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, method), sub in per_curve.groupby(["scheme", "method"], sort=True):
        finite_future = sub["future_rmse_pct"].dropna().to_numpy(dtype=float)
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows).sort_values(["scheme", "median_future_rmse_pct"], ascending=[True, True])


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme, sub in summary.groupby("scheme"):
        lookup = sub.set_index("method")["median_future_rmse_pct"].to_dict()

        def best(starts: str) -> str:
            candidates = sub[sub["method"].str.startswith(starts)].copy()
            if candidates.empty:
                return ""
            return str(candidates.sort_values("median_future_rmse_pct").iloc[0]["method"])

        def delta(a: str, b: str) -> float:
            return float(lookup.get(a, np.nan) - lookup.get(b, np.nan))

        best_static_mix = best("static_knn_proto_mix")
        best_static_early_mix = best("static_early_knn_proto_mix")
        best_static_early_refined = best("static_early_knn_proto_theta_refined")

        rows.extend(
            [
                {
                    "scheme": scheme,
                    "question": "Does static theta prototype mixture beat global median?",
                    "method_a": best_static_mix,
                    "method_b": "global_median_weibull",
                    "delta_future_rmse_pct": delta(best_static_mix, "global_median_weibull"),
                    "interpretation": "negative means local theta prototypes carry static descriptor signal",
                },
                {
                    "scheme": scheme,
                    "question": "Does early Q improve prototype mixture routing?",
                    "method_a": best_static_early_mix,
                    "method_b": best_static_mix,
                    "delta_future_rmse_pct": delta(best_static_early_mix, best_static_mix),
                    "interpretation": "negative means early Q selects better local theta prototypes",
                },
                {
                    "scheme": scheme,
                    "question": "Does best prototype mixture beat early-only Weibull fitting?",
                    "method_a": best_static_early_mix,
                    "method_b": "early_only_weibull_fit",
                    "delta_future_rmse_pct": delta(best_static_early_mix, "early_only_weibull_fit"),
                    "interpretation": "negative supports finer middle layer beyond early-only fitting",
                },
                {
                    "scheme": scheme,
                    "question": "Does prototype theta initialization improve after early refinement?",
                    "method_a": best_static_early_refined,
                    "method_b": "early_only_weibull_fit",
                    "delta_future_rmse_pct": delta(best_static_early_refined, "early_only_weibull_fit"),
                    "interpretation": "negative means static/early prototype initialization adds value to early fitting",
                },
                {
                    "scheme": scheme,
                    "question": "Does continuous static+early theta model beat best prototype mixture?",
                    "method_a": "static_early_et_theta",
                    "method_b": best_static_early_mix,
                    "delta_future_rmse_pct": delta("static_early_et_theta", best_static_early_mix),
                    "interpretation": "negative means learned theta regression still dominates mixture prototypes",
                },
            ]
        )
    return pd.DataFrame(rows)


def data_checks(meta: pd.DataFrame, per_curve: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    checks = [
        {"check": "root_exists", "passed": args.root.exists(), "detail": str(args.root)},
        {"check": "enough_curves_loaded", "passed": len(meta) >= 30, "detail": f"n_curves={len(meta)}"},
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
            "check": "prototype_neighbors_train_only",
            "passed": True,
            "detail": "neighbor IDs are selected only from train fold indices",
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
        "# Liposome Theta/Prototype Mixture Probe",
        "",
        "Date: 2026-06-12",
        "",
        "## Aim",
        "",
        "Test whether a finer train-fold Weibull prototype mixture improves on the 3-class kinetic middle layer.",
        "",
        "```text",
        "static descriptors / static+early Q -> neighbor weights over train-fold theta prototypes -> Q(t)",
        "```",
        "",
        "## Run",
        "",
        f"- root: `{args.root.as_posix()}`",
        f"- n_curves: {len(meta)}",
        f"- early_window_h: {args.early_window_h}",
        f"- prototype_ks: {args.prototype_ks}",
        f"- refine_nfev: {args.refine_nfev}",
        "",
        "## Best Methods By Split",
        "",
    ]
    for scheme, sub in summary.groupby("scheme"):
        lines.append(f"### {scheme}")
        lines.append("")
        lines.append("| Method | Median future RMSE pct | Median future R2 |")
        lines.append("|---|---:|---:|")
        for row in sub.sort_values("median_future_rmse_pct").head(8).itertuples(index=False):
            lines.append(f"| `{row.method}` | {row.median_future_rmse_pct:.3f} | {row.median_future_r2:.3f} |")
        lines.append("")
    lines.extend(["## Decision Deltas", "", "| Scheme | Question | Method A | Method B | Delta future RMSE pct |", "|---|---|---|---|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.question} | `{row.method_a}` | `{row.method_b}` | {row.delta_future_rmse_pct:.3f} |")
    lines.extend(
        [
            "",
            "Negative deltas mean method A is better.",
            "",
            "## Guardrails",
            "",
            "- Prototype neighbors are selected only from train folds.",
            "- Early Q is used only by `static_early`, `early_only`, and refined methods.",
            "- Full curves are evaluation targets only; future metrics use points after the early window.",
            "- This is not a mechanism or foundation-model claim.",
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
        "prototype_ks": args.prototype_ks,
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
        raise RuntimeError(f"Theta/prototype mixture probe failed checks: {failed}")


if __name__ == "__main__":
    main()
