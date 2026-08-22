"""
48 - External wet-experiment prediction on NC Prediction.csv.

What this does:
    Evaluates the internal181-trained theta bottleneck on the wet
    prediction curves stored in:

        D:\\最终版框架\\dataset\\Prediction.csv

    The wet file has two deployment curves (`Prediction_1/2`) with
    measured release and early-release columns `T=0.25` and `T=1.0`.
    It is not mixed into CV. Models are trained on the cleaned internal181
    oracle-theta bank, then evaluated on the wet curves only.

    Prediction routes:
        - tree ensemble -> theta -> ODE
        - direct tree ensemble -> Q grid

    Because the wet file lacks `Drug_NHA`, this script trains on the
    descriptor intersection shared by internal181 and the wet file.

Outputs:
    outputs/48_external_wet_prediction_eval/per_curve_method.csv
    outputs/48_external_wet_prediction_eval/selected_summary.csv
    outputs/48_external_wet_prediction_eval/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import PLGA_CONTINUOUS_COLS  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


@dataclass
class WetCurve:
    fid: str
    dp_group: str
    t_obs: np.ndarray
    q_obs: np.ndarray
    early_q: np.ndarray


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, sub in df.groupby("method", sort=False):
        vals = sub["r2"].dropna().to_numpy(dtype=float)
        rows.append({
            "method": method,
            "n_curves": int(len(vals)),
            "median_r2": float(np.median(vals)),
            "mean_r2": float(np.mean(vals)),
            "median_rmse": float(np.median(sub["rmse"].to_numpy(dtype=float))),
            "median_mae": float(np.median(sub["mae"].to_numpy(dtype=float))),
        })
    return pd.DataFrame(rows).sort_values("median_r2", ascending=False)


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(float), lows + eps, highs - eps)


def _fit_predict_raw(model: object, x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model.fit(x, y)
    return model.predict(x_test)


def _fit_predict_ztheta(model: object, x: np.ndarray, theta: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    mu = theta.mean(axis=0, keepdims=True)
    sd = theta.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x, (theta - mu) / sd)
    return model.predict(x_test) * sd + mu


def _make_rf(args: argparse.Namespace, seed: int, **kwargs: object) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        **kwargs,
    )


def _make_et(args: argparse.Namespace, seed: int, **kwargs: object) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        **kwargs,
    )


def _early_col_name(t: float) -> str:
    return f"T={t:.1f}" if float(t).is_integer() else f"T={t:g}"


def _load_wet(
    wet_csv: Path,
    feature_cols: list[str],
    early_times: np.ndarray,
) -> tuple[pd.DataFrame, list[WetCurve]]:
    raw = pd.read_csv(wet_csv)
    missing = [c for c in ["Experimental_index", "DP_Group", "Time", "Release", *feature_cols] if c not in raw.columns]
    if missing:
        raise KeyError(f"wet file missing columns: {missing}")

    raw = raw.copy()
    raw["Release"] = raw["Release"].astype(float).clip(0.0, 1.0)
    raw = (
        raw.groupby(["Experimental_index", "Time"], as_index=False, sort=False)
        .agg({
            **{
                c: "first"
                for c in raw.columns
                if c not in {"Time", "Release", "STD"}
            },
            "Release": "mean",
            "STD": "mean" if "STD" in raw.columns else "first",
        })
        .sort_values(["Experimental_index", "Time"])
        .reset_index(drop=True)
    )

    first = raw.drop_duplicates("Experimental_index").reset_index(drop=True)
    curves: list[WetCurve] = []
    for fid, g in raw.groupby("Experimental_index", sort=False):
        t_obs = g["Time"].to_numpy(dtype=float)
        q_obs = g["Release"].to_numpy(dtype=float)
        early_vals: list[float] = []
        first_row = g.iloc[0]
        for t in early_times:
            col = _early_col_name(float(t))
            if col in g.columns and np.isfinite(first_row[col]):
                early_vals.append(float(first_row[col]))
            else:
                early_vals.append(float(np.interp(float(t), t_obs, q_obs, left=q_obs[0], right=q_obs[-1])))
        curves.append(WetCurve(
            fid=str(fid),
            dp_group=str(first_row["DP_Group"]),
            t_obs=t_obs,
            q_obs=q_obs,
            early_q=np.asarray(early_vals, dtype=float),
        ))
    return first, curves


def _reconstruct_direct(
    pred_grid_q: np.ndarray,
    late_times: np.ndarray,
    curve: WetCurve,
    early_times: np.ndarray,
) -> np.ndarray:
    t_combined = np.concatenate([early_times, late_times])
    q_combined = np.concatenate([curve.early_q, pred_grid_q])
    order = np.argsort(t_combined)
    q_hat = np.interp(
        curve.t_obs,
        t_combined[order],
        q_combined[order],
        left=q_combined[order][0],
        right=q_combined[order][-1],
    )
    return np.clip(q_hat, 0.0, 1.0)


def _quick_oracle_fit(
    sim: PLGABiphasic,
    curve: WetCurve,
    lows: np.ndarray,
    highs: np.ndarray,
    seed: int,
    n_restarts: int,
    max_nfev: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lows + highs)
    best_pred: np.ndarray | None = None
    best_ss = np.inf
    best_nfev = 0
    for k in range(n_restarts):
        x0 = mid if k == 0 else rng.uniform(lows, highs)
        try:
            res = least_squares(
                lambda th: sim.simulate_numpy(th, curve.t_obs) - curve.q_obs,
                x0=x0,
                bounds=(lows, highs),
                method="trf",
                max_nfev=max_nfev,
            )
        except Exception:
            continue
        pred = sim.simulate_numpy(res.x, curve.t_obs)
        ss = float(np.sum((pred - curve.q_obs) ** 2))
        if np.isfinite(ss) and ss < best_ss:
            best_ss = ss
            best_pred = pred
            best_nfev = int(res.nfev)
    if best_pred is None:
        return {
            "fid": curve.fid,
            "dp_group": curve.dp_group,
            "oracle_quick_r2": np.nan,
            "oracle_quick_rmse": np.nan,
            "oracle_quick_mae": np.nan,
            "oracle_nfev": 0,
        }
    return {
        "fid": curve.fid,
        "dp_group": curve.dp_group,
        "oracle_quick_r2": _r2(curve.q_obs, best_pred),
        "oracle_quick_rmse": _rmse(curve.q_obs, best_pred),
        "oracle_quick_mae": _mae(curve.q_obs, best_pred),
        "oracle_nfev": best_nfev,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument("--wet-csv", type=Path, default=Path(r"D:\最终版框架\dataset\Prediction.csv"))
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--early-times", nargs="+", type=float, default=[0.25, 1.0])
    ap.add_argument(
        "--late-grid",
        nargs="+",
        type=float,
        default=[2.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 45.0, 60.0, 90.0],
    )
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--oracle-restarts", type=int, default=3)
    ap.add_argument("--oracle-max-nfev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/48_external_wet_prediction_eval"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mod42 = _load_script(Path(__file__).resolve().parent / "42_internal181_tree_theta_audit.py", "script42_internal")
    desc, curve_map, inventory = mod42._load_clean_internal(
        data_csv=args.data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )

    wet_raw = pd.read_csv(args.wet_csv, nrows=1)
    feature_cols = [c for c in PLGA_CONTINUOUS_COLS if c in wet_raw.columns]
    missing_wet_features = [c for c in PLGA_CONTINUOUS_COLS if c not in wet_raw.columns]
    wet_meta, wet_curves = _load_wet(args.wet_csv, feature_cols, np.asarray(args.early_times, dtype=float))

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    param_names = list(sim.param_names)

    fids = desc[mod42.INTERNAL_FID_COL].to_numpy(dtype=int)
    early_times = np.asarray(args.early_times, dtype=float)
    late_times = np.asarray(args.late_grid, dtype=float)
    train_early_q = np.stack([
        mod42._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, early_times)
        for fid in fids
    ]).astype(np.float32)
    x_train = np.concatenate([
        desc[feature_cols].to_numpy(dtype=np.float32),
        train_early_q,
    ], axis=1)
    theta_train = desc[param_names].to_numpy(dtype=np.float32)
    direct_q_train = np.stack([
        mod42._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, late_times)
        for fid in fids
    ]).astype(np.float32)

    x_wet = np.concatenate([
        wet_meta[feature_cols].to_numpy(dtype=np.float32),
        np.stack([c.early_q for c in wet_curves], axis=0).astype(np.float32),
    ], axis=1)

    theta_pred: dict[str, np.ndarray] = {}
    theta_pred["RF_raw"] = _fit_predict_raw(
        _make_rf(args, args.seed), x_train, theta_train, x_wet
    )
    theta_pred["RF_ztheta"] = _fit_predict_ztheta(
        _make_rf(args, args.seed + 100), x_train, theta_train, x_wet
    )
    theta_pred["RF_ztheta_leaf2"] = _fit_predict_ztheta(
        _make_rf(args, args.seed + 200, min_samples_leaf=2), x_train, theta_train, x_wet
    )
    theta_pred["ET_raw"] = _fit_predict_raw(
        _make_et(args, args.seed + 300), x_train, theta_train, x_wet
    )
    theta_pred["ET_ztheta"] = _fit_predict_ztheta(
        _make_et(args, args.seed + 400), x_train, theta_train, x_wet
    )
    theta_pred["ET_ztheta_leaf2"] = _fit_predict_ztheta(
        _make_et(args, args.seed + 500, min_samples_leaf=2), x_train, theta_train, x_wet
    )
    theta_pred["RF_ET_zavg"] = 0.5 * (theta_pred["RF_ztheta"] + theta_pred["ET_ztheta"])
    theta_pred["RF_raw_zavg"] = 0.5 * (theta_pred["RF_raw"] + theta_pred["RF_ztheta"])

    direct_pred: dict[str, np.ndarray] = {}
    for name, model in {
        "RF_direct_Q": _make_rf(args, args.seed + 600),
        "RF_direct_Q_leaf2": _make_rf(args, args.seed + 700, min_samples_leaf=2),
        "ET_direct_Q": _make_et(args, args.seed + 800),
    }.items():
        model.fit(x_train, direct_q_train)
        direct_pred[name] = model.predict(x_wet)

    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for i, curve in enumerate(wet_curves):
        early_rmse_by_theta: dict[str, float] = {}
        for method, pred in theta_pred.items():
            theta = _clip_theta(pred[i], lows, highs)
            q_hat = sim.simulate_numpy(theta, curve.t_obs)
            early_hat = sim.simulate_numpy(theta, early_times)
            early_rmse_by_theta[method] = _rmse(curve.early_q, early_hat)
            rows.append({
                "fid": curve.fid,
                "dp_group": curve.dp_group,
                "route": "theta_to_ode",
                "method": method,
                "r2": _r2(curve.q_obs, q_hat),
                "rmse": _rmse(curve.q_obs, q_hat),
                "mae": _mae(curve.q_obs, q_hat),
                "early_rmse": early_rmse_by_theta[method],
                "n_obs": int(len(curve.t_obs)),
                "t_max": float(curve.t_obs.max()),
            })

        select_pool = ["RF_ztheta", "ET_ztheta", "RF_ET_zavg"]
        selected = min(select_pool, key=lambda m: early_rmse_by_theta[m])
        theta = _clip_theta(theta_pred[selected][i], lows, highs)
        q_hat = sim.simulate_numpy(theta, curve.t_obs)
        rows.append({
            "fid": curve.fid,
            "dp_group": curve.dp_group,
            "route": "theta_to_ode",
            "method": "early_select",
            "r2": _r2(curve.q_obs, q_hat),
            "rmse": _rmse(curve.q_obs, q_hat),
            "mae": _mae(curve.q_obs, q_hat),
            "early_rmse": early_rmse_by_theta[selected],
            "n_obs": int(len(curve.t_obs)),
            "t_max": float(curve.t_obs.max()),
        })
        selected_rows.append({
            "fid": curve.fid,
            "dp_group": curve.dp_group,
            "route": "theta_to_ode",
            "selected_method": selected,
            "r2": _r2(curve.q_obs, q_hat),
            "rmse": _rmse(curve.q_obs, q_hat),
            "mae": _mae(curve.q_obs, q_hat),
            "early_rmse": early_rmse_by_theta[selected],
        })

        for method, pred in direct_pred.items():
            q_direct = _reconstruct_direct(pred[i], late_times, curve, early_times)
            rows.append({
                "fid": curve.fid,
                "dp_group": curve.dp_group,
                "route": "direct_q",
                "method": method,
                "r2": _r2(curve.q_obs, q_direct),
                "rmse": _rmse(curve.q_obs, q_direct),
                "mae": _mae(curve.q_obs, q_direct),
                "early_rmse": 0.0,
                "n_obs": int(len(curve.t_obs)),
                "t_max": float(curve.t_obs.max()),
            })

    per_curve = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    method_summary = _summary_rows(per_curve)
    oracle = pd.DataFrame([
        _quick_oracle_fit(
            sim=sim,
            curve=curve,
            lows=lows,
            highs=highs,
            seed=args.seed + i,
            n_restarts=args.oracle_restarts,
            max_nfev=args.oracle_max_nfev,
        )
        for i, curve in enumerate(wet_curves)
    ])
    per_curve.to_csv(args.out / "per_curve_method.csv", index=False)
    selected.to_csv(args.out / "selected_summary.csv", index=False)
    method_summary.to_csv(args.out / "method_summary.csv", index=False)
    oracle.to_csv(args.out / "quick_oracle_fit.csv", index=False)
    inventory.to_csv(args.out / "train_curve_inventory.csv", index=False)

    lines = [
        "=== 48 -- external NC wet prediction eval ===",
        "",
        f"train data          : {args.data}",
        f"wet csv             : {args.wet_csv}",
        f"train curves        : {len(desc)}",
        f"wet curves          : {len(wet_curves)}",
        f"early_times         : {list(early_times)}",
        f"feature_cols        : {feature_cols}",
        f"missing wet features: {missing_wet_features}",
        "",
        "--- quick full-curve oracle fit (diagnostic only; uses wet full curve) ---",
    ]
    for _, row in oracle.iterrows():
        lines.append(
            f"  {row['fid']:<12} {row['dp_group']:<10} "
            f"oracle_R2={row['oracle_quick_r2']:+.4f} "
            f"RMSE={row['oracle_quick_rmse']:.4f} nfev={int(row['oracle_nfev'])}"
        )
    lines.extend([
        "",
        "--- selected theta->ODE on wet curves ---",
    ])
    for _, row in selected.iterrows():
        lines.append(
            f"  {row['fid']:<12} {row['dp_group']:<10} method={row['selected_method']:<12} "
            f"R2={row['r2']:+.4f} RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
            f"early_RMSE={row['early_rmse']:.4f}"
        )
    lines.extend(["", "--- best methods by median R2 across wet curves ---"])
    for _, row in method_summary.head(12).iterrows():
        lines.append(
            f"  {row['method']:<18} median_R2={row['median_r2']:+.4f} "
            f"median_RMSE={row['median_rmse']:.4f} median_MAE={row['median_mae']:.4f}"
        )
    lines.extend([
        "",
        "--- caveats ---",
        "  This is an external two-curve wet check, not a CV benchmark.",
        "  Drug_NHA is absent from Prediction.csv, so the model is trained on",
        "  the 12 shared descriptor columns plus the requested early release",
        "  observations. The quick oracle fit is diagnostic only and uses the",
        "  full wet curve, so it is not a deployable prediction result.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
