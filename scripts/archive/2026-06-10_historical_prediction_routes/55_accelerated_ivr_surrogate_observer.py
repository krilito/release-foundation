"""
55 - Accelerated IVR surrogate-observer audit.

Question:
    Early IVR observations calibrate Weibull alpha/beta very well. Can
    cheaper pre-release or rapid-QC features replace some of that
    information?

This script compares feature sets as "observers" for the same kinetic
state:

    observer features -> Weibull alpha/beta -> full curve

The early-only Weibull fit is the gold observer. Formulation/QC feature
sets are surrogate observers.

Outputs:
    outputs/55_accelerated_ivr_surrogate_observer/per_curve.csv
    outputs/55_accelerated_ivr_surrogate_observer/scheme_summary.csv
    outputs/55_accelerated_ivr_surrogate_observer/summary.txt
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline


EARLY_GRID_H = np.asarray([6.0, 12.0, 24.0, 48.0, 72.0], dtype=float)

FEATURE_SETS = {
    "formulation_prior": [
        "media_pH",
        "media_temp_oC",
        "drug_loading",
        "API_type",
        "weighted_Mw",
        "weighted_Tm",
    ],
    "plus_size": [
        "media_pH",
        "media_temp_oC",
        "drug_loading",
        "API_type",
        "weighted_Mw",
        "weighted_Tm",
        "Z_average_nm",
    ],
    "plus_qc": [
        "media_pH",
        "media_temp_oC",
        "drug_loading",
        "API_type",
        "weighted_Mw",
        "weighted_Tm",
        "Z_average_nm",
        "PDI",
        "zeta_potential",
    ],
    "plus_method_structure": [
        "media_pH",
        "media_temp_oC",
        "drug_loading",
        "API_type",
        "weighted_Mw",
        "weighted_Tm",
        "Z_average_nm",
        "PDI",
        "zeta_potential",
        "release_method_code",
        "structure_type_code",
    ],
}


def _parse_id(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"cannot parse IVR ID from {value!r}")
    return int(match.group(1))


def _weibull(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    return 100.0 * (1.0 - np.exp(-np.power(t, max(float(beta), 1e-8)) / max(float(alpha), 1e-8)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _load_dataset(root: Path, early_window_h: float) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    exp = pd.read_csv(root / "results/fitting/drug_release_exp.csv")
    exp["ID"] = exp["file_name"].map(_parse_id).astype(int)
    exp = exp.rename(columns={"time (Hrs)": "time_h", "release_percent": "release_pct"})

    backend = pd.read_csv(root / "data/unprocessed/backend_data.csv")
    backend = backend.rename(columns={"IVR_ID": "ID"}).drop(columns=["Unnamed: 0"], errors="ignore")
    backend["ID"] = backend["ID"].astype(int)

    weibull = pd.read_csv(root / "data/clean/weibull_params.csv")
    weibull["ID"] = weibull["ID"].astype(int)

    cluster = pd.read_csv(root / "results/clustering/3_PCA_KMC.csv")
    cluster = cluster.rename(columns={"file": "ID"})
    cluster["ID"] = cluster["ID"].astype(int)

    backend["API_type"] = backend["API_name"].astype("category").cat.codes.replace(-1, np.nan)
    backend["release_method_code"] = backend["release_method"].astype("category").cat.codes.replace(-1, np.nan)
    backend["structure_type_code"] = backend["structure_type"].astype("category").cat.codes.replace(-1, np.nan)

    rows: list[dict[str, object]] = []
    curve_map: dict[int, pd.DataFrame] = {}
    for curve_id, sub in exp.groupby("ID"):
        sub = sub.sort_values("time_h").dropna(subset=["time_h", "release_pct"])
        if len(sub) == 0:
            continue
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        curve_map[int(curve_id)] = sub[["ID", "time_h", "release_pct", "file_name"]].copy()
        rows.append({
            "ID": int(curve_id),
            "n_points": int(len(sub)),
            "n_early": int(np.sum(t <= early_window_h)),
            "n_future": int(np.sum(t > early_window_h)),
            "release_min_pct": float(np.min(q)),
            "release_max_pct": float(np.max(q)),
        })

    meta = pd.DataFrame(rows)
    meta = meta.merge(backend, on="ID", how="left")
    meta = meta.merge(weibull[["ID", "alpha", "beta"]], on="ID", how="inner")
    meta = meta.merge(cluster[["ID", "cluster"]], on="ID", how="inner")
    required = ["media_pH", "media_temp_oC", "drug_loading", "API_type", "weighted_Mw", "weighted_Tm"]
    meta = meta.dropna(subset=required + ["alpha", "beta", "cluster"])
    meta = meta[
        (meta["n_early"] >= 2)
        & (meta["n_future"] >= 2)
        & (meta["release_min_pct"] >= -5.0)
        & (meta["release_max_pct"] <= 125.0)
    ].reset_index(drop=True)
    return meta, curve_map


def _early_q_features(curve_map: dict[int, pd.DataFrame], ids: np.ndarray, early_window_h: float) -> np.ndarray:
    times = EARLY_GRID_H[EARLY_GRID_H <= early_window_h]
    values = []
    for curve_id in ids:
        sub = curve_map[int(curve_id)].sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        values.append(np.interp(times, t, q, left=q[0], right=q[-1]))
    return np.asarray(values, dtype=float)


def _splits(meta: pd.DataFrame, scheme: str, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(meta))
    y = meta["cluster"].to_numpy(dtype=int)
    if scheme == "stratified_5fold":
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, y))
    group_col = {"group_by_API": "API_name", "group_by_release_method": "release_method"}[scheme]
    groups = meta[group_col].astype(str).to_numpy()
    n_eff = min(n_splits, int(pd.Series(groups).nunique()))
    if n_eff < 2:
        return []
    return list(GroupKFold(n_splits=n_eff).split(idx, y, groups))


def _fit_theta(model: object, x_train: np.ndarray, theta_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    y_train = np.log(np.clip(theta_train, 1e-8, None))
    model.fit(x_train, y_train)
    pred = np.exp(model.predict(x_test))
    pred[:, 0] = np.clip(pred[:, 0], 1e-8, 300.0)
    pred[:, 1] = np.clip(pred[:, 1], 1e-8, 5.0)
    return pred


def _refine_theta(theta0: np.ndarray, early_t: np.ndarray, early_q: np.ndarray, max_nfev: int) -> tuple[np.ndarray, int]:
    x0 = np.asarray([np.clip(theta0[0], 1e-8, 300.0), np.clip(theta0[1], 1e-8, 5.0)], dtype=float)
    try:
        res = least_squares(
            lambda th: _weibull(early_t, th[0], th[1]) - early_q,
            x0=x0,
            bounds=([1e-8, 1e-8], [300.0, 5.0]),
            method="trf",
            max_nfev=max_nfev,
        )
    except Exception:
        return x0, 0
    return np.asarray([np.clip(res.x[0], 1e-8, 300.0), np.clip(res.x[1], 1e-8, 5.0)]), int(res.nfev)


def _eval_rows(
    meta: pd.DataFrame,
    curve_map: dict[int, pd.DataFrame],
    te: np.ndarray,
    method: str,
    observer: str,
    theta_pred: np.ndarray,
    early_window_h: float,
    nfev: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ids = meta.iloc[te]["ID"].to_numpy(dtype=int)
    for i, curve_id in enumerate(ids):
        sub = curve_map[int(curve_id)].sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        pred = _weibull(t, theta_pred[i, 0], theta_pred[i, 1])
        future = t > early_window_h
        rows.append({
            "ID": int(curve_id),
            "observer": observer,
            "method": method,
            "full_r2": _r2(q, pred),
            "future_r2": _r2(q[future], pred[future]) if np.sum(future) >= 2 else float("nan"),
            "full_rmse": _rmse(q, pred),
            "future_rmse": _rmse(q[future], pred[future]) if np.sum(future) >= 2 else float("nan"),
            "alpha_pred": float(theta_pred[i, 0]),
            "beta_pred": float(theta_pred[i, 1]),
            "nfev": int(nfev[i]) if nfev is not None else 0,
        })
    return rows


def _run_scheme(args: argparse.Namespace, meta: pd.DataFrame, curve_map: dict[int, pd.DataFrame], scheme: str) -> list[dict[str, object]]:
    ids = meta["ID"].to_numpy(dtype=int)
    theta = meta[["alpha", "beta"]].to_numpy(dtype=float)
    early_q = _early_q_features(curve_map, ids, args.early_window_h)
    rows: list[dict[str, object]] = []

    for fold, (tr, te) in enumerate(_splits(meta, scheme, args.n_folds, args.seed)):
        seed = args.seed + 1000 * fold
        train_median_theta = np.asarray([float(np.median(theta[tr, 0])), float(np.median(theta[tr, 1]))])

        early_only = []
        early_nfev = []
        for curve_id in ids[te]:
            sub = curve_map[int(curve_id)].sort_values("time_h")
            early_sub = sub[sub["time_h"] <= args.early_window_h]
            refined, nfev = _refine_theta(
                train_median_theta,
                early_sub["time_h"].to_numpy(dtype=float),
                early_sub["release_pct"].to_numpy(dtype=float),
                args.refine_nfev,
            )
            early_only.append(refined)
            early_nfev.append(nfev)
        fold_rows = _eval_rows(
            meta,
            curve_map,
            te,
            "early_only_weibull_fit",
            "early_release_gold",
            np.asarray(early_only, dtype=float),
            args.early_window_h,
            np.asarray(early_nfev, dtype=int),
        )
        for row in fold_rows:
            row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
        rows.extend(fold_rows)

        for observer, cols in FEATURE_SETS.items():
            x_train = meta.iloc[tr][cols].to_numpy(dtype=float)
            x_test = meta.iloc[te][cols].to_numpy(dtype=float)
            for model_name, model in [
                ("RF_theta", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed + 10, n_jobs=-1)),
                ("ET_theta", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed + 20, n_jobs=-1)),
            ]:
                pipe = make_pipeline(SimpleImputer(strategy="median"), model)
                pred_theta = _fit_theta(pipe, x_train, theta[tr], x_test)
                fold_rows = _eval_rows(meta, curve_map, te, model_name, observer, pred_theta, args.early_window_h)
                for row in fold_rows:
                    row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
                rows.extend(fold_rows)

        x_train = np.concatenate([meta.iloc[tr][FEATURE_SETS["plus_method_structure"]].to_numpy(dtype=float), early_q[tr]], axis=1)
        x_test = np.concatenate([meta.iloc[te][FEATURE_SETS["plus_method_structure"]].to_numpy(dtype=float), early_q[te]], axis=1)
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed + 30, n_jobs=-1),
        )
        pred_theta = _fit_theta(pipe, x_train, theta[tr], x_test)
        fold_rows = _eval_rows(meta, curve_map, te, "ET_theta", "all_proxy_plus_earlyQ", pred_theta, args.early_window_h)
        for row in fold_rows:
            row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
        rows.extend(fold_rows)

    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, observer, method), sub in per_curve.groupby(["scheme", "observer", "method"], sort=False):
        full = sub["full_r2"].dropna().to_numpy(dtype=float)
        fut = sub["future_r2"].dropna().to_numpy(dtype=float)
        rows.append({
            "scheme": scheme,
            "observer": observer,
            "method": method,
            "n": int(len(sub)),
            "median_full_r2": float(np.median(full)) if len(full) else float("nan"),
            "median_future_r2": float(np.median(fut)) if len(fut) else float("nan"),
            "p10_full_r2": float(np.percentile(full, 10)) if len(full) else float("nan"),
            "frac_full_r2_ge_0": float(np.mean(full >= 0.0)) if len(full) else float("nan"),
            "median_full_rmse": float(np.median(sub["full_rmse"])),
            "median_nfev": float(np.median(sub["nfev"])),
        })
    return pd.DataFrame(rows).sort_values(["scheme", "median_full_r2"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/external/accelerated_IVR/repo/accelerated_IVR-main"))
    ap.add_argument("--early-window-h", type=float, default=6.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--refine-nfev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/55_accelerated_ivr_surrogate_observer"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = _load_dataset(args.root, args.early_window_h)
    if args.max_curves is not None:
        meta = meta.iloc[: args.max_curves].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for scheme in ["stratified_5fold", "group_by_API", "group_by_release_method"]:
        if not _splits(meta, scheme, args.n_folds, args.seed):
            continue
        print(f"[55] scheme={scheme} n={len(meta)}", flush=True)
        rows.extend(_run_scheme(args, meta, curve_map, scheme))

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_summary.csv", index=False)

    lines = [
        "=== 55 -- accelerated IVR surrogate observer audit ===",
        "",
        f"n_curves       : {len(meta)}",
        f"early_window_h : {args.early_window_h}",
        "",
        "--- best observers by scheme ---",
    ]
    for scheme in summary["scheme"].drop_duplicates():
        lines.append(f"  {scheme}:")
        for _, row in summary[summary["scheme"] == scheme].head(10).iterrows():
            lines.append(
                f"    {row['observer']:<24} {row['method']:<22} "
                f"full_R2={row['median_full_r2']:+.4f} future_R2={row['median_future_r2']:+.4f} "
                f"p10_full={row['p10_full_r2']:+.4f} frac_full>=0={row['frac_full_r2_ge_0']:.3f}"
            )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  early_release_gold is the information ceiling for cheap observers.",
        "  Surrogate observers are useful only if they approach that curve-R2 without early Q.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
