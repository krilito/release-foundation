"""
54 - Accelerated IVR curve forecast benchmark.

This script turns the Yanes accelerated_IVR kinetic-class pipeline into a
curve-level comparison.

Their natural curve baseline:
    features -> predicted kinetic class -> train-fold class prototype curve

Our extension:
    features + early release -> Weibull alpha/beta -> full curve
    features + early release -> alpha/beta -> early-only local refinement

The comparison is within the same folds and reports curve R^2 on the
experimental observations. Full curves are evaluation-only.

Outputs:
    outputs/54_accelerated_ivr_weibull_forecast/per_curve.csv
    outputs/54_accelerated_ivr_weibull_forecast/scheme_method_summary.csv
    outputs/54_accelerated_ivr_weibull_forecast/summary.txt
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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


def _parse_id(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"cannot parse IVR ID from {value!r}")
    return int(match.group(1))


def _weibull(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    alpha = max(float(alpha), 1e-8)
    beta = max(float(beta), 1e-8)
    return 100.0 * (1.0 - np.exp(-np.power(t, beta) / alpha))


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
    exp = exp[["ID", "time_h", "release_pct", "file_name"]].copy()

    backend = pd.read_csv(root / "data/unprocessed/backend_data.csv")
    backend = backend.rename(columns={"IVR_ID": "ID"}).drop(columns=["Unnamed: 0"], errors="ignore")
    backend["ID"] = backend["ID"].astype(int)

    weibull = pd.read_csv(root / "data/clean/weibull_params.csv")
    weibull["ID"] = weibull["ID"].astype(int)

    cluster = pd.read_csv(root / "results/clustering/3_PCA_KMC.csv")
    cluster = cluster.rename(columns={"file": "ID"})
    cluster["ID"] = cluster["ID"].astype(int)

    api_names = sorted(backend["API_name"].dropna().unique().tolist())
    api_map = {name: i for i, name in enumerate(api_names)}
    backend["API_type"] = backend["API_name"].map(api_map)

    rows: list[dict[str, object]] = []
    curve_map: dict[int, pd.DataFrame] = {}
    for curve_id, sub in exp.groupby("ID"):
        sub = sub.sort_values("time_h").dropna(subset=["time_h", "release_pct"])
        if len(sub) == 0:
            continue
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        curve_map[int(curve_id)] = sub
        rows.append({
            "ID": int(curve_id),
            "n_points": int(len(sub)),
            "t_max_h": float(np.max(t)),
            "n_early": int(np.sum(t <= early_window_h)),
            "n_future": int(np.sum(t > early_window_h)),
            "release_min_pct": float(np.min(q)),
            "release_max_pct": float(np.max(q)),
        })

    meta = pd.DataFrame(rows)
    meta = meta.merge(backend, on="ID", how="left")
    meta = meta.merge(weibull[["ID", "alpha", "beta"]], on="ID", how="inner")
    meta = meta.merge(cluster[["ID", "cluster", "cluster_name"]], on="ID", how="inner")
    meta = meta.dropna(subset=FEATURE_7 + ["alpha", "beta", "cluster"])
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


def _curve_grid_targets(curve_map: dict[int, pd.DataFrame], ids: np.ndarray, t_grid_max_h: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, t_grid_max_h, 50)
    y = []
    for curve_id in ids:
        sub = curve_map[int(curve_id)].sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        y.append(np.interp(grid, t, q, left=q[0], right=q[-1]))
    return grid, np.asarray(y, dtype=float)


def _splits(meta: pd.DataFrame, scheme: str, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = meta["cluster"].to_numpy(dtype=int)
    idx = np.arange(len(meta))
    if scheme == "stratified_5fold":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(idx, y))
    group_col = {
        "group_by_API": "API_name",
        "group_by_release_method": "release_method",
    }[scheme]
    groups = meta[group_col].astype(str).to_numpy()
    n_group = pd.Series(groups).nunique()
    n_eff = min(n_splits, int(n_group))
    if n_eff < 2:
        return []
    splitter = GroupKFold(n_splits=n_eff)
    return list(splitter.split(idx, y, groups))


def _fit_classifiers(seed: int) -> dict[str, object]:
    return {
        "their_LogReg_class_proto": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
        ),
        "their_KNN_class_proto": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "their_RF_class_proto": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "their_ET_class_proto": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def _class_prototypes(theta_train: pd.DataFrame) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    global_alpha = float(np.median(theta_train["alpha"]))
    global_beta = float(np.median(theta_train["beta"]))
    for cluster, sub in theta_train.groupby("cluster"):
        out[int(cluster)] = (float(np.median(sub["alpha"])), float(np.median(sub["beta"])))
    out[-1] = (global_alpha, global_beta)
    return out


def _predict_prototype(labels: np.ndarray, prototypes: dict[int, tuple[float, float]]) -> np.ndarray:
    fallback = prototypes[-1]
    return np.asarray([prototypes.get(int(label), fallback) for label in labels], dtype=float)


def _fit_regressor_targets(model: object, x_train: np.ndarray, theta_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    y_train = np.log(np.clip(theta_train, 1e-8, None))
    model.fit(x_train, y_train)
    pred = np.exp(model.predict(x_test))
    pred[:, 0] = np.clip(pred[:, 0], 1e-8, 300.0)
    pred[:, 1] = np.clip(pred[:, 1], 1e-8, 5.0)
    return pred


def _refine_theta(theta0: np.ndarray, early_t: np.ndarray, early_q: np.ndarray, max_nfev: int) -> tuple[np.ndarray, int]:
    if max_nfev <= 0:
        return theta0, 0
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


def _evaluate_method(
    meta: pd.DataFrame,
    curve_map: dict[int, pd.DataFrame],
    test_local: np.ndarray,
    method: str,
    theta_pred: np.ndarray,
    early_window_h: float,
    nfev: np.ndarray | None = None,
    class_true: np.ndarray | None = None,
    class_pred: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ids = meta.iloc[test_local]["ID"].to_numpy(dtype=int)
    for k, curve_id in enumerate(ids):
        sub = curve_map[int(curve_id)].sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        pred = _weibull(t, theta_pred[k, 0], theta_pred[k, 1])
        future = t > early_window_h
        rows.append({
            "ID": int(curve_id),
            "method": method,
            "full_r2": _r2(q, pred),
            "future_r2": _r2(q[future], pred[future]) if np.sum(future) >= 2 else float("nan"),
            "full_rmse": _rmse(q, pred),
            "future_rmse": _rmse(q[future], pred[future]) if np.sum(future) >= 2 else float("nan"),
            "alpha_pred": float(theta_pred[k, 0]),
            "beta_pred": float(theta_pred[k, 1]),
            "nfev": int(nfev[k]) if nfev is not None else 0,
            "class_true": int(class_true[k]) if class_true is not None else -1,
            "class_pred": int(class_pred[k]) if class_pred is not None else -1,
        })
    return rows


def _run_scheme(args: argparse.Namespace, meta: pd.DataFrame, curve_map: dict[int, pd.DataFrame], scheme: str) -> list[dict[str, object]]:
    base_x = meta[FEATURE_7].to_numpy(dtype=float)
    ids = meta["ID"].to_numpy(dtype=int)
    early_q_all = _early_q_features(curve_map, ids, args.early_window_h)
    x_plus_early = np.concatenate([base_x, early_q_all], axis=1)
    theta = meta[["alpha", "beta"]].to_numpy(dtype=float)
    y_class = meta["cluster"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []

    for fold, (tr, te) in enumerate(_splits(meta, scheme, args.n_folds, args.seed)):
        seed = args.seed + 1000 * fold
        prototypes = _class_prototypes(meta.iloc[tr][["cluster", "alpha", "beta"]])

        for method, clf in _fit_classifiers(seed).items():
            clf.fit(base_x[tr], y_class[tr])
            pred_class = clf.predict(base_x[te])
            pred_theta = _predict_prototype(pred_class, prototypes)
            eval_rows = _evaluate_method(
                meta, curve_map, te, method, pred_theta, args.early_window_h,
                class_true=y_class[te], class_pred=pred_class,
            )
            for row in eval_rows:
                row.update({
                    "scheme": scheme,
                    "fold": fold,
                    "early_window_h": args.early_window_h,
                })
            rows.extend(eval_rows)

        oracle_class_theta = _predict_prototype(y_class[te], prototypes)
        eval_rows = _evaluate_method(
            meta, curve_map, te, "their_oracle_class_proto", oracle_class_theta, args.early_window_h,
            class_true=y_class[te], class_pred=y_class[te],
        )
        for row in eval_rows:
            row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
        rows.extend(eval_rows)

        train_median_theta = np.asarray([
            float(np.median(theta[tr, 0])),
            float(np.median(theta[tr, 1])),
        ])
        early_only = []
        early_only_nfev = []
        for curve_id in ids[te]:
            sub = curve_map[int(curve_id)].sort_values("time_h")
            early_sub = sub[sub["time_h"] <= args.early_window_h]
            early_t = early_sub["time_h"].to_numpy(dtype=float)
            early_q = early_sub["release_pct"].to_numpy(dtype=float)
            theta_ref, calls = _refine_theta(train_median_theta, early_t, early_q, args.refine_nfev)
            early_only.append(theta_ref)
            early_only_nfev.append(calls)
        eval_rows = _evaluate_method(
            meta,
            curve_map,
            te,
            "early_only_weibull_fit",
            np.asarray(early_only, dtype=float),
            args.early_window_h,
            nfev=np.asarray(early_only_nfev, dtype=int),
        )
        for row in eval_rows:
            row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
        rows.extend(eval_rows)

        for name, model in [
            ("our_RF_weibull_theta", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed + 10, n_jobs=-1)),
            ("our_ET_weibull_theta", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed + 20, n_jobs=-1)),
        ]:
            pred_theta = _fit_regressor_targets(model, x_plus_early[tr], theta[tr], x_plus_early[te])
            eval_rows = _evaluate_method(meta, curve_map, te, name, pred_theta, args.early_window_h)
            for row in eval_rows:
                row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
            rows.extend(eval_rows)

            refined = []
            nfev = []
            for local_i, curve_id in enumerate(ids[te]):
                sub = curve_map[int(curve_id)].sort_values("time_h")
                early_sub = sub[sub["time_h"] <= args.early_window_h]
                early_t = early_sub["time_h"].to_numpy(dtype=float)
                early_q = early_sub["release_pct"].to_numpy(dtype=float)
                theta_ref, calls = _refine_theta(pred_theta[local_i], early_t, early_q, args.refine_nfev)
                refined.append(theta_ref)
                nfev.append(calls)
            eval_rows = _evaluate_method(
                meta,
                curve_map,
                te,
                f"{name}_early_refined",
                np.asarray(refined, dtype=float),
                args.early_window_h,
                nfev=np.asarray(nfev, dtype=int),
            )
            for row in eval_rows:
                row.update({"scheme": scheme, "fold": fold, "early_window_h": args.early_window_h})
            rows.extend(eval_rows)

        grid, y_grid_train = _curve_grid_targets(curve_map, ids[tr], args.t_grid_max_h)
        direct = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed + 30, n_jobs=-1)
        direct.fit(x_plus_early[tr], y_grid_train)
        y_grid_pred = direct.predict(x_plus_early[te])
        pred_theta_placeholder = np.zeros((len(te), 2), dtype=float)
        direct_rows: list[dict[str, object]] = []
        for k, curve_id in enumerate(ids[te]):
            sub = curve_map[int(curve_id)].sort_values("time_h")
            t = sub["time_h"].to_numpy(dtype=float)
            q = sub["release_pct"].to_numpy(dtype=float)
            pred = np.interp(t, grid, y_grid_pred[k], left=y_grid_pred[k, 0], right=y_grid_pred[k, -1])
            future = t > args.early_window_h
            direct_rows.append({
                "ID": int(curve_id),
                "method": "direct_ET_Q_grid",
                "full_r2": _r2(q, pred),
                "future_r2": _r2(q[future], pred[future]) if np.sum(future) >= 2 else float("nan"),
                "full_rmse": _rmse(q, pred),
                "future_rmse": _rmse(q[future], pred[future]) if np.sum(future) >= 2 else float("nan"),
                "alpha_pred": float(pred_theta_placeholder[k, 0]),
                "beta_pred": float(pred_theta_placeholder[k, 1]),
                "nfev": 0,
                "class_true": int(y_class[te][k]),
                "class_pred": -1,
                "scheme": scheme,
                "fold": fold,
                "early_window_h": args.early_window_h,
            })
        rows.extend(direct_rows)
    return rows


def _summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, method), sub in per_curve.groupby(["scheme", "method"], sort=False):
        full = sub["full_r2"].dropna().to_numpy(dtype=float)
        fut = sub["future_r2"].dropna().to_numpy(dtype=float)
        row = {
            "scheme": scheme,
            "method": method,
            "n": int(len(sub)),
            "median_full_r2": float(np.median(full)) if len(full) else float("nan"),
            "median_future_r2": float(np.median(fut)) if len(fut) else float("nan"),
            "p10_future_r2": float(np.percentile(fut, 10)) if len(fut) else float("nan"),
            "frac_future_r2_ge_0": float(np.mean(fut >= 0.0)) if len(fut) else float("nan"),
            "median_full_rmse": float(np.median(sub["full_rmse"])),
            "median_future_rmse": float(np.median(sub["future_rmse"].dropna())),
            "median_nfev": float(np.median(sub["nfev"])),
        }
        if method.startswith("their_") and method != "their_oracle_class_proto":
            row["balanced_accuracy"] = float(balanced_accuracy_score(sub["class_true"], sub["class_pred"]))
            row["macro_f1"] = float(f1_score(sub["class_true"], sub["class_pred"], average="macro"))
            row["mcc"] = float(matthews_corrcoef(sub["class_true"], sub["class_pred"]))
        else:
            row["balanced_accuracy"] = float("nan")
            row["macro_f1"] = float("nan")
            row["mcc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scheme", "median_future_r2"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/accelerated_IVR/repo/accelerated_IVR-main"),
    )
    ap.add_argument("--early-window-h", type=float, default=24.0)
    ap.add_argument("--t-grid-max-h", type=float, default=168.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--refine-nfev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/54_accelerated_ivr_weibull_forecast"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = _load_dataset(args.root, args.early_window_h)
    if args.max_curves is not None:
        meta = meta.iloc[: args.max_curves].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for scheme in ["stratified_5fold", "group_by_API", "group_by_release_method"]:
        splits = _splits(meta, scheme, args.n_folds, args.seed)
        if not splits:
            continue
        print(f"[54] scheme={scheme} n={len(meta)} folds={len(splits)}", flush=True)
        rows.extend(_run_scheme(args, meta, curve_map, scheme))

    per_curve = pd.DataFrame(rows)
    summary = _summarize(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)

    lines = [
        "=== 54 -- accelerated IVR Weibull forecast ===",
        "",
        f"root            : {args.root}",
        f"n_curves        : {len(meta)}",
        f"early_window_h  : {args.early_window_h}",
        f"refine_nfev     : {args.refine_nfev}",
        "",
        "--- best methods by scheme (future R2) ---",
    ]
    for scheme in summary["scheme"].drop_duplicates():
        lines.append(f"  {scheme}:")
        sub = summary[summary["scheme"] == scheme].head(8)
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['method']:<34} future_R2={row['median_future_r2']:+.4f} "
                f"full_R2={row['median_full_r2']:+.4f} p10_future={row['p10_future_r2']:+.4f} "
                f"frac_future>=0={row['frac_future_r2_ge_0']:.3f} "
                f"bal_acc={row['balanced_accuracy'] if not np.isnan(row['balanced_accuracy']) else np.nan:.3f}"
            )
    lines.extend([
        "",
        "--- comparison guardrail ---",
        "  their_*_class_proto predicts Fast/Medium/Slow, then decodes a train-fold",
        "  class prototype curve. our_*_weibull_theta predicts continuous alpha/beta.",
        "  Full experimental curves are used only for final R2 evaluation.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
