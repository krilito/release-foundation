"""
79 - Export liposome fold-wise predictions in the shared benchmark format.

Purpose:
    Reuse the accelerated_IVR benchmarking logic from script 54, but write
    per-timepoint predictions into the common release-benchmark schema so the
    shared API can score liposome methods the same way as PLGA or chitosan.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/79_liposome_prediction_export/<scheme>/<method>/predictions.csv
    outputs/79_liposome_prediction_export/<scheme>/<method>/fold_curve_metadata.csv
    outputs/79_liposome_prediction_export/<scheme>/<method>/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUTROOT = Path("outputs/79_liposome_prediction_export")
METHODS = (
    "their_RF_class_proto",
    "their_ET_class_proto",
    "their_LogReg_class_proto",
    "their_KNN_class_proto",
    "their_oracle_class_proto",
    "early_only_weibull_fit",
    "our_RF_weibull_theta",
    "our_RF_weibull_theta_early_refined",
    "our_ET_weibull_theta",
    "our_ET_weibull_theta_early_refined",
    "direct_ET_Q_grid",
)
SCHEMES = ("stratified_5fold", "group_by_API", "group_by_release_method")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scheme", choices=SCHEMES, default="group_by_API")
    parser.add_argument("--method", choices=METHODS, default="our_ET_weibull_theta_early_refined")
    parser.add_argument("--early-window-h", type=float, default=24.0)
    parser.add_argument("--t-grid-max-h", type=float, default=168.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--refine-nfev", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outroot", type=Path, default=DEFAULT_OUTROOT)
    return parser.parse_args()


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _curve_id(source_id: int) -> str:
    return f"liposome_{source_id:03d}"


def _time_unit_map(root: Path) -> dict[int, str]:
    df = pd.read_csv(root / "data/time_units.csv")
    return {int(row.ID): str(row.Time_units).strip().lower() for row in df.itertuples(index=False)}


def _time_multiplier(unit: str) -> float:
    mapping = {"seconds": 1.0 / 3600.0, "mins": 1.0 / 60.0, "minutes": 1.0 / 60.0, "hours": 1.0}
    if unit not in mapping:
        raise ValueError(f"Unsupported time unit: {unit}")
    return float(mapping[unit])


def _fit_weibull_method(
    method: str,
    seed: int,
    tr: np.ndarray,
    te: np.ndarray,
    meta: pd.DataFrame,
    curve_map: dict[int, pd.DataFrame],
    mod54: ModuleType,
    early_window_h: float,
    t_grid_max_h: float,
    refine_nfev: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    base_x = meta[list(mod54.FEATURE_7)].to_numpy(dtype=float)
    ids = meta["ID"].to_numpy(dtype=int)
    theta = meta[["alpha", "beta"]].to_numpy(dtype=float)
    y_class = meta["cluster"].to_numpy(dtype=int)
    early_q_all = mod54._early_q_features(curve_map, ids, early_window_h)
    x_plus_early = np.concatenate([base_x, early_q_all], axis=1)

    if method.startswith("their_") and method != "their_oracle_class_proto":
        clf = mod54._fit_classifiers(seed)[method]
        clf.fit(base_x[tr], y_class[tr])
        pred_class = clf.predict(base_x[te])
        prototypes = mod54._class_prototypes(meta.iloc[tr][["cluster", "alpha", "beta"]])
        theta_pred = mod54._predict_prototype(pred_class, prototypes)
        return theta_pred, pred_class.astype(int), None

    if method == "their_oracle_class_proto":
        prototypes = mod54._class_prototypes(meta.iloc[tr][["cluster", "alpha", "beta"]])
        theta_pred = mod54._predict_prototype(y_class[te], prototypes)
        return theta_pred, y_class[te].astype(int), None

    if method == "early_only_weibull_fit":
        train_median_theta = np.asarray(
            [
                float(np.median(theta[tr, 0])),
                float(np.median(theta[tr, 1])),
            ],
            dtype=float,
        )
        preds: list[np.ndarray] = []
        nfev: list[int] = []
        for curve_id in ids[te]:
            sub = curve_map[int(curve_id)].sort_values("time_h")
            early_sub = sub[sub["time_h"] <= early_window_h]
            early_t = early_sub["time_h"].to_numpy(dtype=float)
            early_q = early_sub["release_pct"].to_numpy(dtype=float)
            theta_ref, calls = mod54._refine_theta(train_median_theta, early_t, early_q, refine_nfev)
            preds.append(theta_ref)
            nfev.append(calls)
        return np.asarray(preds, dtype=float), None, np.asarray(nfev, dtype=int)

    if method in ("our_RF_weibull_theta", "our_RF_weibull_theta_early_refined"):
        reg = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed + 10,
            n_jobs=-1,
        )
        theta_pred = mod54._fit_regressor_targets(reg, x_plus_early[tr], theta[tr], x_plus_early[te])
        if method.endswith("_early_refined"):
            preds: list[np.ndarray] = []
            nfev: list[int] = []
            for local_i, curve_id in enumerate(ids[te]):
                sub = curve_map[int(curve_id)].sort_values("time_h")
                early_sub = sub[sub["time_h"] <= early_window_h]
                early_t = early_sub["time_h"].to_numpy(dtype=float)
                early_q = early_sub["release_pct"].to_numpy(dtype=float)
                theta_ref, calls = mod54._refine_theta(theta_pred[local_i], early_t, early_q, refine_nfev)
                preds.append(theta_ref)
                nfev.append(calls)
            return np.asarray(preds, dtype=float), None, np.asarray(nfev, dtype=int)
        return theta_pred, None, None

    if method in ("our_ET_weibull_theta", "our_ET_weibull_theta_early_refined"):
        reg = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed + 20,
            n_jobs=-1,
        )
        theta_pred = mod54._fit_regressor_targets(reg, x_plus_early[tr], theta[tr], x_plus_early[te])
        if method.endswith("_early_refined"):
            preds: list[np.ndarray] = []
            nfev: list[int] = []
            for local_i, curve_id in enumerate(ids[te]):
                sub = curve_map[int(curve_id)].sort_values("time_h")
                early_sub = sub[sub["time_h"] <= early_window_h]
                early_t = early_sub["time_h"].to_numpy(dtype=float)
                early_q = early_sub["release_pct"].to_numpy(dtype=float)
                theta_ref, calls = mod54._refine_theta(theta_pred[local_i], early_t, early_q, refine_nfev)
                preds.append(theta_ref)
                nfev.append(calls)
            return np.asarray(preds, dtype=float), None, np.asarray(nfev, dtype=int)
        return theta_pred, None, None

    if method == "direct_ET_Q_grid":
        grid, y_grid_train = mod54._curve_grid_targets(curve_map, ids[tr], t_grid_max_h)
        direct = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed + 30,
            n_jobs=-1,
        )
        direct.fit(x_plus_early[tr], y_grid_train)
        y_grid_pred = direct.predict(x_plus_early[te])
        return y_grid_pred, grid, None

    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    mod54 = _load_script(
        Path(__file__).resolve().parent / "54_accelerated_ivr_weibull_forecast.py",
        "_loader_54_79",
    )
    meta, curve_map = mod54._load_dataset(args.root, args.early_window_h)
    time_units = _time_unit_map(args.root)

    splits = mod54._splits(meta, args.scheme, args.n_folds, args.seed)
    if not splits:
        raise RuntimeError(f"No splits available for scheme={args.scheme}")

    outdir = args.outroot / args.scheme / args.method
    outdir.mkdir(parents=True, exist_ok=True)

    pred_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    ids = meta["ID"].to_numpy(dtype=int)
    y_class = meta["cluster"].to_numpy(dtype=int)

    for fold, (tr, te) in enumerate(splits):
        seed = args.seed + 1000 * fold
        pred_obj, aux, nfev = _fit_weibull_method(
            method=args.method,
            seed=seed,
            tr=tr,
            te=te,
            meta=meta,
            curve_map=curve_map,
            mod54=mod54,
            early_window_h=args.early_window_h,
            t_grid_max_h=args.t_grid_max_h,
            refine_nfev=args.refine_nfev,
        )

        for local_i, curve_id_num in enumerate(ids[te]):
            curve_id_num = int(curve_id_num)
            curve_id = _curve_id(curve_id_num)
            sub = curve_map[curve_id_num].sort_values("time_h")
            raw_t_h = sub["time_h"].to_numpy(dtype=float)
            unit = time_units.get(curve_id_num, "hours")
            t_hours = raw_t_h * _time_multiplier(unit)
            q_obs = np.clip(sub["release_pct"].to_numpy(dtype=float) / 100.0, 0.0, 1.2)

            if args.method == "direct_ET_Q_grid":
                grid = aux
                q_pred_pct = np.interp(
                    raw_t_h,
                    grid,
                    pred_obj[local_i],
                    left=pred_obj[local_i, 0],
                    right=pred_obj[local_i, -1],
                )
                alpha_pred = np.nan
                beta_pred = np.nan
            else:
                alpha_pred = float(pred_obj[local_i, 0])
                beta_pred = float(pred_obj[local_i, 1])
                q_pred_pct = mod54._weibull(raw_t_h, alpha_pred, beta_pred)

            q_pred = np.clip(q_pred_pct / 100.0, 0.0, 1.2)
            future_mask = t_hours > args.early_window_h
            future_r2 = mod54._r2(q_obs[future_mask], q_pred[future_mask]) if int(np.sum(future_mask)) >= 2 else float("nan")

            fold_rows.append(
                {
                    "curve_id": curve_id,
                    "source_id": curve_id_num,
                    "fold": fold,
                    "scheme": args.scheme,
                    "method": args.method,
                    "class_true": int(y_class[te][local_i]),
                    "class_pred": int(aux[local_i]) if aux is not None and args.method.startswith("their_") else -1,
                    "alpha_pred": alpha_pred,
                    "beta_pred": beta_pred,
                    "nfev": int(nfev[local_i]) if nfev is not None else 0,
                    "full_r2": float(mod54._r2(q_obs, q_pred)),
                    "future_r2": float(future_r2),
                    "full_rmse": float(mod54._rmse(q_obs, q_pred)),
                    "future_rmse": float(mod54._rmse(q_obs[future_mask], q_pred[future_mask])) if int(np.sum(future_mask)) >= 1 else float("nan"),
                }
            )

            for t_h, q_p, q_o in zip(t_hours.tolist(), q_pred.tolist(), q_obs.tolist()):
                pred_rows.append(
                    {
                        "curve_id": curve_id,
                        "source_id": curve_id_num,
                        "fold": fold,
                        "scheme": args.scheme,
                        "method": args.method,
                        "formulation": f"formulation_{int(meta.iloc[te[local_i]]['formulation_ID'])}",
                        "drug": str(meta.iloc[te[local_i]]["API_name"]),
                        "t_hours": float(t_h),
                        "Q_predicted_point": float(q_p),
                        "Q_observed_reference": float(q_o),
                        "alpha_pred": alpha_pred,
                        "beta_pred": beta_pred,
                    }
                )

    predictions = pd.DataFrame(pred_rows).sort_values(["curve_id", "t_hours"]).reset_index(drop=True)
    fold_meta = pd.DataFrame(fold_rows).sort_values(["fold", "curve_id"]).reset_index(drop=True)

    predictions.to_csv(outdir / "predictions.csv", index=False)
    fold_meta.to_csv(outdir / "fold_curve_metadata.csv", index=False)

    summary_lines = [
        "Liposome shared prediction export complete.",
        f"root: {args.root}",
        f"scheme: {args.scheme}",
        f"method: {args.method}",
        f"curves: {int(fold_meta['curve_id'].nunique())}",
        f"points: {int(len(predictions))}",
        f"median_full_r2: {float(fold_meta['full_r2'].median()):+.4f}",
        f"median_future_r2: {float(fold_meta['future_r2'].median()):+.4f}",
        f"frac_future_r2_ge0: {float((fold_meta['future_r2'] >= 0.0).mean()):.3f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
