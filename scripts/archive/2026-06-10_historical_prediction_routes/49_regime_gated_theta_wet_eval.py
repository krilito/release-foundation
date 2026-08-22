"""
49 - Regime-gated theta wet check inspired by ME LGBM.

What this does:
    Script 48 showed that the global internal181 theta mapper struggles
    on the two NC wet deployment curves. The older `ME LGBM.py` script
    performed better on those curves because it explicitly used a
    fast/slow gate and branch-specific curve formulas.

    This script tests the transferable idea without importing that whole
    legacy stack:

        train curves -> t50 fast/slow labels
        formulation + early Q -> fast probability
        branch-specific tree ensemble -> theta
        theta -> PLGABiphasic ODE -> wet curve

    Wet full curves are evaluation-only. Optional local refinement uses
    only the requested wet early observations.

Outputs:
    outputs/49_regime_gated_theta_wet_eval/per_curve_method.csv
    outputs/49_regime_gated_theta_wet_eval/method_summary.csv
    outputs/49_regime_gated_theta_wet_eval/summary.txt
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor

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


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(float), lows + eps, highs - eps)


def _interp_at(t_obs: np.ndarray, q_obs: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.interp(times, t_obs, q_obs, left=q_obs[0], right=q_obs[-1])


def _estimate_t_at_release(t_obs: np.ndarray, q_obs: np.ndarray, target: float = 0.5) -> float:
    order = np.argsort(t_obs)
    t = np.asarray(t_obs, dtype=float)[order]
    q = np.asarray(q_obs, dtype=float)[order]
    if len(t) == 0:
        return float("nan")
    if np.nanmax(q) < target:
        return float(np.nanmax(t) + 30.0 * (target - np.nanmax(q)))
    hit = np.where(q >= target)[0]
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(t[0])
    q0, q1 = float(q[i - 1]), float(q[i])
    t0, t1 = float(t[i - 1]), float(t[i])
    if abs(q1 - q0) < 1e-9:
        return t1
    return float(t0 + (target - q0) * (t1 - t0) / (q1 - q0))


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
        first_row = g.iloc[0]
        early_vals: list[float] = []
        for t in early_times:
            col = _early_col_name(float(t))
            if col in g.columns and np.isfinite(first_row[col]):
                early_vals.append(float(first_row[col]))
            else:
                early_vals.append(float(_interp_at(t_obs, q_obs, np.array([float(t)]))[0]))
        curves.append(WetCurve(
            fid=str(fid),
            dp_group=str(first_row["DP_Group"]),
            t_obs=t_obs,
            q_obs=q_obs,
            early_q=np.asarray(early_vals, dtype=float),
        ))
    return first, curves


def _fit_predict_ztheta(
    model: object,
    x: np.ndarray,
    theta: np.ndarray,
    x_test: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    mu = theta.mean(axis=0, keepdims=True)
    sd = theta.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x, (theta - mu) / sd, sample_weight=sample_weight)
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


def _local_refine_early(
    sim: PLGABiphasic,
    theta0: np.ndarray,
    curve: WetCurve,
    early_times: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, int, float]:
    x0 = _clip_theta(theta0, lows, highs)
    try:
        res = least_squares(
            lambda th: sim.simulate_numpy(th, early_times) - curve.early_q,
            x0=x0,
            bounds=(lows, highs),
            method="trf",
            max_nfev=max_nfev,
        )
    except Exception:
        pred = sim.simulate_numpy(x0, early_times)
        return x0, 0, _rmse(curve.early_q, pred)
    theta = _clip_theta(res.x, lows, highs)
    pred = sim.simulate_numpy(theta, early_times)
    return theta, int(res.nfev), _rmse(curve.early_q, pred)


def _add_eval_row(
    rows: list[dict[str, object]],
    sim: PLGABiphasic,
    curve: WetCurve,
    method: str,
    theta: np.ndarray,
    gate_p_fast: float,
    early_times: np.ndarray,
    nfev: int = 0,
) -> None:
    pred = sim.simulate_numpy(theta, curve.t_obs)
    early_pred = sim.simulate_numpy(theta, early_times)
    rows.append({
        "fid": curve.fid,
        "dp_group": curve.dp_group,
        "method": method,
        "r2": _r2(curve.q_obs, pred),
        "rmse": _rmse(curve.q_obs, pred),
        "mae": _mae(curve.q_obs, pred),
        "early_rmse": _rmse(curve.early_q, early_pred),
        "gate_p_fast": gate_p_fast,
        "nfev": int(nfev),
        "n_obs": int(len(curve.t_obs)),
        "t_max": float(curve.t_obs.max()),
    })


def _method_summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, sub in per_curve.groupby("method", sort=False):
        vals = sub["r2"].to_numpy(dtype=float)
        rows.append({
            "method": method,
            "n_curves": int(len(vals)),
            "median_r2": float(np.median(vals)),
            "mean_r2": float(np.mean(vals)),
            "median_rmse": float(np.median(sub["rmse"].to_numpy(dtype=float))),
            "median_mae": float(np.median(sub["mae"].to_numpy(dtype=float))),
            "median_early_rmse": float(np.median(sub["early_rmse"].to_numpy(dtype=float))),
            "median_nfev": float(np.median(sub["nfev"].to_numpy(dtype=float))),
        })
    return pd.DataFrame(rows).sort_values("median_r2", ascending=False)


def _fit_branch_theta_models(
    args: argparse.Namespace,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    y_fast: np.ndarray,
    x_wet: np.ndarray,
    p_fast: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for model_name, maker in [
        ("RF", lambda seed: _make_rf(args, seed, min_samples_leaf=2)),
        ("ET", lambda seed: _make_et(args, seed, min_samples_leaf=2)),
    ]:
        pred_fast = _fit_predict_ztheta(
            maker(args.seed + 100),
            x_train[y_fast == 1],
            theta_train[y_fast == 1],
            x_wet,
        )
        pred_slow = _fit_predict_ztheta(
            maker(args.seed + 200),
            x_train[y_fast == 0],
            theta_train[y_fast == 0],
            x_wet,
        )
        p = p_fast.reshape(-1, 1)
        out[f"{model_name}_regime_soft"] = p * pred_fast + (1.0 - p) * pred_slow
        out[f"{model_name}_regime_hard"] = np.where((p_fast >= 0.5).reshape(-1, 1), pred_fast, pred_slow)
    out["RF_ET_regime_soft_avg"] = 0.5 * (out["RF_regime_soft"] + out["ET_regime_soft"])
    out["RF_ET_regime_hard_avg"] = 0.5 * (out["RF_regime_hard"] + out["ET_regime_hard"])
    return out


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
    ap.add_argument("--fast-threshold-days", type=float, default=np.nan)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--refine-nfev", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/49_regime_gated_theta_wet_eval"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mod42 = _load_script(Path(__file__).resolve().parent / "42_internal181_tree_theta_audit.py", "script42_internal")
    desc, curve_map, _inventory = mod42._load_clean_internal(
        data_csv=args.data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )
    wet_raw = pd.read_csv(args.wet_csv, nrows=1)
    feature_cols = [c for c in PLGA_CONTINUOUS_COLS if c in wet_raw.columns]
    wet_meta, wet_curves = _load_wet(args.wet_csv, feature_cols, np.asarray(args.early_times, dtype=float))

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    param_names = list(sim.param_names)

    fids = desc[mod42.INTERNAL_FID_COL].to_numpy(dtype=int)
    early_times = np.asarray(args.early_times, dtype=float)
    train_early_q = np.stack([
        _interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, early_times)
        for fid in fids
    ]).astype(np.float32)
    x_train = np.concatenate([
        desc[feature_cols].to_numpy(dtype=np.float32),
        train_early_q,
    ], axis=1)
    x_wet = np.concatenate([
        wet_meta[feature_cols].to_numpy(dtype=np.float32),
        np.stack([c.early_q for c in wet_curves], axis=0).astype(np.float32),
    ], axis=1)
    theta_train = desc[param_names].to_numpy(dtype=np.float32)

    t50_train = np.asarray([
        _estimate_t_at_release(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, target=0.5)
        for fid in fids
    ], dtype=float)
    finite = np.isfinite(t50_train)
    threshold = float(np.nanmedian(t50_train[finite])) if not np.isfinite(args.fast_threshold_days) else float(args.fast_threshold_days)
    y_fast = (t50_train <= threshold).astype(int)

    gate = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        random_state=args.seed + 3000,
        n_jobs=-1,
    )
    gate.fit(x_train, y_fast)
    p_fast_wet = gate.predict_proba(x_wet)[:, list(gate.classes_).index(1)]

    global_theta: dict[str, np.ndarray] = {
        "RF_global_ztheta": _fit_predict_ztheta(
            _make_rf(args, args.seed + 10, min_samples_leaf=2), x_train, theta_train, x_wet
        ),
        "ET_global_ztheta": _fit_predict_ztheta(
            _make_et(args, args.seed + 20, min_samples_leaf=2), x_train, theta_train, x_wet
        ),
    }
    global_theta["RF_ET_global_zavg"] = 0.5 * (
        global_theta["RF_global_ztheta"] + global_theta["ET_global_ztheta"]
    )
    regime_theta = _fit_branch_theta_models(args, x_train, theta_train, y_fast, x_wet, p_fast_wet)

    rows: list[dict[str, object]] = []
    for i, curve in enumerate(wet_curves):
        for method, pred in {**global_theta, **regime_theta}.items():
            theta = _clip_theta(pred[i], lows, highs)
            _add_eval_row(rows, sim, curve, method, theta, float(p_fast_wet[i]), early_times)
            if method in {"RF_ET_regime_soft_avg", "RF_ET_regime_hard_avg", "RF_ET_global_zavg"}:
                refined, nfev, _early_rmse = _local_refine_early(
                    sim=sim,
                    theta0=theta,
                    curve=curve,
                    early_times=early_times,
                    lows=lows,
                    highs=highs,
                    max_nfev=args.refine_nfev,
                )
                _add_eval_row(
                    rows,
                    sim,
                    curve,
                    method=f"{method}_early_refined",
                    theta=refined,
                    gate_p_fast=float(p_fast_wet[i]),
                    early_times=early_times,
                    nfev=nfev,
                )

    per_curve = pd.DataFrame(rows)
    summary = _method_summary(per_curve)
    per_curve.to_csv(args.out / "per_curve_method.csv", index=False)
    summary.to_csv(args.out / "method_summary.csv", index=False)
    pd.DataFrame({
        "fid": [c.fid for c in wet_curves],
        "dp_group": [c.dp_group for c in wet_curves],
        "gate_p_fast": p_fast_wet,
        "early_q": [list(c.early_q) for c in wet_curves],
    }).to_csv(args.out / "wet_gate_scores.csv", index=False)

    lines = [
        "=== 49 -- regime-gated theta wet eval ===",
        "",
        f"train curves         : {len(desc)}",
        f"wet curves           : {len(wet_curves)}",
        f"early_times          : {list(early_times)}",
        f"feature_cols         : {feature_cols}",
        f"fast threshold t50   : {threshold:.3f} days",
        f"fast train fraction  : {float(np.mean(y_fast)):.3f}",
        "",
        "--- wet gate scores ---",
    ]
    for curve, p in zip(wet_curves, p_fast_wet):
        lines.append(f"  {curve.fid:<12} {curve.dp_group:<10} p_fast={p:.3f} early_q={list(np.round(curve.early_q, 4))}")
    lines.extend(["", "--- best methods by median R2 across wet curves ---"])
    for _, row in summary.head(14).iterrows():
        lines.append(
            f"  {row['method']:<34} median_R2={row['median_r2']:+.4f} "
            f"median_RMSE={row['median_rmse']:.4f} early_RMSE={row['median_early_rmse']:.4f} "
            f"nfev={row['median_nfev']:.1f}"
        )
    lines.extend(["", "--- per-curve best method ---"])
    idx = per_curve.groupby("fid")["r2"].idxmax()
    for _, row in per_curve.loc[idx].sort_values("fid").iterrows():
        lines.append(
            f"  {row['fid']:<12} {row['dp_group']:<10} {row['method']:<34} "
            f"R2={row['r2']:+.4f} RMSE={row['rmse']:.4f} p_fast={row['gate_p_fast']:.3f}"
        )
    lines.extend([
        "",
        "--- caveat ---",
        "  This is a two-curve external stress test. It borrows the fast/slow",
        "  gating idea from ME LGBM, but keeps the PLGABiphasic theta->ODE",
        "  decoder. Wet full curves are never used for model selection.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
