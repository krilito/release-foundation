"""
62 - LightGBM middle-layer gain audit.

What this does:
    Script 59 computed middle-layer gain (MLG) from existing RF/ET outputs.
    This script runs the same matched comparison for LightGBM:

        direct-Q route:
            x / early Q -> Q grid

        kinetic-state route:
            x / early Q -> theta -> PLGABiphasic -> Q(t)

    MLG is defined per matched dataset/input/split cell:

        MLG = median R^2(best LGBM theta route) - median R^2(LGBM direct-Q route)

Outputs:
    outputs/62_lgbm_middle_layer_gain/per_curve.csv
    outputs/62_lgbm_middle_layer_gain/method_summary.csv
    outputs/62_lgbm_middle_layer_gain/middle_layer_gain.csv
    outputs/62_lgbm_middle_layer_gain/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import PLGA_CONTINUOUS_COLS  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

DATASETS = ("cross321", "internal181")
INPUT_MODES = ("formulation_only", "early_only", "formulation_plus_early")


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


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(float), lows + eps, highs - eps)


def _x_for_mode(x_form: np.ndarray, early_q: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "formulation_only":
        return x_form
    if input_mode == "early_only":
        return early_q
    return np.concatenate([x_form, early_q], axis=1)


def _target_grid(early_times: np.ndarray, late_times: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "formulation_only":
        return np.unique(np.concatenate([early_times, late_times])).astype(float)
    return late_times.astype(float)


def _reconstruct_curve(
    pred_grid_q: np.ndarray,
    target_times: np.ndarray,
    early_q: np.ndarray,
    early_times: np.ndarray,
    input_mode: str,
    t_obs: np.ndarray,
) -> np.ndarray:
    if input_mode == "formulation_only":
        t_combined = target_times
        q_combined = pred_grid_q
    else:
        t_combined = np.concatenate([early_times, target_times])
        q_combined = np.concatenate([early_q, pred_grid_q])
    order = np.argsort(t_combined)
    q_hat = np.interp(
        t_obs,
        t_combined[order],
        q_combined[order],
        left=q_combined[order][0],
        right=q_combined[order][-1],
    )
    return np.clip(q_hat, 0.0, 1.0)


def _split_schemes(
    n: int,
    drug_groups: np.ndarray,
    polymer_groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    for scheme_name, groups in [
        ("random_5fold", None),
        ("group_by_drug", drug_groups),
        ("group_by_polymer", polymer_groups),
    ]:
        if groups is None:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.arange(n)))
        else:
            n_unique = pd.Series(groups).nunique()
            n_splits = min(n_folds, int(n_unique))
            if n_splits < 2:
                continue
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.arange(n), groups=groups))
        out.append((scheme_name, splits))
    return out


def _make_lgbm(args: argparse.Namespace, seed: int) -> MultiOutputRegressor:
    base = LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def _fit_predict_raw(
    model: MultiOutputRegressor,
    x: np.ndarray,
    theta: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> np.ndarray:
    model.fit(x[tr], theta[tr])
    return np.asarray(model.predict(x[te]), dtype=float)


def _fit_predict_ztheta(
    model: MultiOutputRegressor,
    x: np.ndarray,
    theta: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> np.ndarray:
    mu = theta[tr].mean(axis=0, keepdims=True)
    sd = theta[tr].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x[tr], (theta[tr] - mu) / sd)
    return np.asarray(model.predict(x[te]), dtype=float) * sd + mu


def _load_cross321(args: argparse.Namespace, mod41: ModuleType) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, object],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    df, curve_map, x, theta, _param_names, drug_groups, polymer_groups = mod41._load_dataset(args)
    if args.max_curves is not None:
        n_keep = min(int(args.max_curves), len(df))
        df = df.iloc[:n_keep].reset_index(drop=True)
        x = x[:n_keep]
        theta = theta[:n_keep]
        drug_groups = drug_groups[:n_keep]
        polymer_groups = polymer_groups[:n_keep]
    early_times = np.array(args.early_times, dtype=float)
    early_q = x[:, -len(early_times):]
    x_form = x[:, :-len(early_times)]
    fids = df[mod41.FID_COL].to_numpy(dtype=int)
    return fids, x_form, early_q, theta, curve_map, drug_groups, polymer_groups, early_times


def _load_internal181(args: argparse.Namespace, mod42: ModuleType) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, object],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    desc, curve_map, _inventory = mod42._load_clean_internal(
        data_csv=args.data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )
    if args.max_curves is not None:
        desc = desc.iloc[: int(args.max_curves)].reset_index(drop=True)
    fids = desc[mod42.INTERNAL_FID_COL].to_numpy(dtype=int)
    early_times = np.array(args.early_times, dtype=float)
    early_q = np.stack([
        mod42._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, early_times)
        for fid in fids
    ]).astype(np.float32)
    x_form = desc[list(PLGA_CONTINUOUS_COLS)].to_numpy(dtype=np.float32)
    theta = desc[list(PLGABiphasic().param_names)].to_numpy(dtype=np.float32)
    drug_groups = np.array([curve_map[int(fid)].drug_id for fid in fids], dtype=object)
    polymer_groups = np.array([curve_map[int(fid)].polymer_family for fid in fids], dtype=object)
    return fids, x_form, early_q, theta, curve_map, drug_groups, polymer_groups, early_times


def _run_dataset_mode(
    args: argparse.Namespace,
    dataset: str,
    input_mode: str,
    mod41: ModuleType,
    mod42: ModuleType,
) -> list[dict[str, object]]:
    pack = _load_cross321(args, mod41) if dataset == "cross321" else _load_internal181(args, mod42)
    fids, x_form, early_q, theta, curve_map, drug_groups, polymer_groups, early_times = pack

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)

    x_model = _x_for_mode(x_form, early_q, input_mode).astype(np.float32)
    late_times = np.array(args.late_grid, dtype=float)
    target_times = _target_grid(early_times, late_times, input_mode)
    target_q = np.stack([
        mod41._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, target_times)
        for fid in fids
    ]).astype(np.float32)

    print(
        f"[62] dataset={dataset} input_mode={input_mode} n={len(fids)} "
        f"x_dim={x_model.shape[1]}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for scheme_name, splits in _split_schemes(
        n=len(fids),
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        n_folds=args.n_folds,
        seed=args.seed,
    ):
        print(f"[62]   scheme={scheme_name}", flush=True)
        for fold_idx, (tr, te) in enumerate(splits):
            direct = _make_lgbm(args, args.seed + fold_idx)
            direct.fit(x_model[tr], target_q[tr])
            pred_q = np.asarray(direct.predict(x_model[te]), dtype=float)

            theta_raw = _fit_predict_raw(
                _make_lgbm(args, args.seed + 100 + fold_idx), x_model, theta, tr, te
            )
            theta_z = _fit_predict_ztheta(
                _make_lgbm(args, args.seed + 200 + fold_idx), x_model, theta, tr, te
            )

            for local_i, j in enumerate(te):
                fid = int(fids[j])
                curve = curve_map[fid]
                q_direct = _reconstruct_curve(
                    pred_grid_q=pred_q[local_i],
                    target_times=target_times,
                    early_q=early_q[j].astype(float),
                    early_times=early_times,
                    input_mode=input_mode,
                    t_obs=curve.t_obs,
                )

                row: dict[str, object] = {
                    "dataset": dataset,
                    "input_mode": input_mode,
                    "scheme": scheme_name,
                    "fold": fold_idx,
                    "fid": fid,
                    "n_obs": int(len(curve.t_obs)),
                    "r2_LGBM_direct_Q": _r2(curve.q_obs, q_direct),
                }

                early_rmse: dict[str, float] = {}
                for method, pred_theta in {
                    "LGBM_theta_raw": theta_raw,
                    "LGBM_theta_z": theta_z,
                }.items():
                    clipped = _clip_theta(pred_theta[local_i], lows, highs)
                    pred = sim.simulate_numpy(clipped, curve.t_obs)
                    row[f"r2_{method}"] = _r2(curve.q_obs, pred)
                    pred_early = sim.simulate_numpy(clipped, early_times)
                    early_rmse[method] = _rmse(early_q[j].astype(float), pred_early)

                selected = min(early_rmse, key=early_rmse.get)
                row["LGBM_early_select_method"] = selected
                row["r2_LGBM_theta_early_select"] = row[f"r2_{selected}"]
                row["LGBM_early_select_rmse"] = early_rmse[selected]
                rows.append(row)
    return rows


def _method_summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, input_mode, scheme), sub in per_curve.groupby(
        ["dataset", "input_mode", "scheme"], sort=False
    ):
        for col in [c for c in per_curve.columns if c.startswith("r2_")]:
            vals = sub[col].dropna().to_numpy(dtype=float)
            rows.append({
                "dataset": dataset,
                "input_mode": input_mode,
                "scheme": scheme,
                "method": col.removeprefix("r2_"),
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p10": float(np.percentile(vals, 10)),
                "frac_above_0.9": float(np.mean(vals >= 0.9)),
                "frac_above_0.5": float(np.mean(vals >= 0.5)),
                "frac_above_0": float(np.mean(vals >= 0.0)),
            })
    return pd.DataFrame(rows)


def _middle_layer_gain(summary: pd.DataFrame) -> pd.DataFrame:
    direct = summary[summary["method"] == "LGBM_direct_Q"].rename(columns={
        "median": "direct_median",
        "frac_above_0": "direct_frac_above_0",
    })
    theta = summary[summary["method"].isin({
        "LGBM_theta_raw",
        "LGBM_theta_z",
        "LGBM_theta_early_select",
    })].copy()
    idx = theta.groupby(["dataset", "input_mode", "scheme"])["median"].idxmax()
    theta_best = theta.loc[idx].rename(columns={
        "method": "theta_best_method",
        "median": "theta_median",
        "frac_above_0": "theta_frac_above_0",
    })
    keep = ["dataset", "input_mode", "scheme"]
    out = direct[keep + ["direct_median", "direct_frac_above_0"]].merge(
        theta_best[keep + ["theta_best_method", "theta_median", "theta_frac_above_0"]],
        on=keep,
        how="inner",
    )
    out["middle_layer_gain"] = out["theta_median"] - out["direct_median"]
    return out.sort_values(keep).reset_index(drop=True)


def _aggregate(gain: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [
        ("all", gain),
        ("formulation_plus_early", gain[gain["input_mode"] == "formulation_plus_early"]),
        ("early_only", gain[gain["input_mode"] == "early_only"]),
        (
            "OOD_formulation_plus_early",
            gain[
                (gain["input_mode"] == "formulation_plus_early")
                & (gain["scheme"] != "random_5fold")
            ],
        ),
    ]
    for name, sub in groups:
        vals = sub["middle_layer_gain"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        rows.append({
            "group": name,
            "n_cells": int(len(vals)),
            "mean_gain": float(np.mean(vals)),
            "median_gain": float(np.median(vals)),
            "min_gain": float(np.min(vals)),
            "max_gain": float(np.max(vals)),
            "frac_positive": float(np.mean(vals > 0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-csv",
        type=Path,
        default=Path("outputs/33_active_set_regimes/regime_assignments.csv"),
    )
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument(
        "--late-grid",
        nargs="+",
        type=float,
        default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0],
    )
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--input-mode", choices=(*INPUT_MODES, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=250)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--min-child-samples", type=int, default=10)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/62_lgbm_middle_layer_gain"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta_for62")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal_for62")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    input_modes = INPUT_MODES if args.input_mode == "all" else (args.input_mode,)

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for input_mode in input_modes:
            rows.extend(_run_dataset_mode(args, dataset, input_mode, mod41, mod42))

    per_curve = pd.DataFrame(rows)
    summary = _method_summary(per_curve)
    gain = _middle_layer_gain(summary)
    aggregate = _aggregate(gain)

    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "method_summary.csv", index=False)
    gain.to_csv(args.out / "middle_layer_gain.csv", index=False)
    aggregate.to_csv(args.out / "aggregate_middle_layer_gain.csv", index=False)

    lines = [
        "=== 62 -- LightGBM middle-layer gain audit ===",
        "",
        f"datasets      : {', '.join(datasets)}",
        f"input_modes   : {', '.join(input_modes)}",
        f"n_estimators  : {args.n_estimators}",
        f"learning_rate : {args.learning_rate}",
        "",
        "--- aggregate MLG ---",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"  {row['group']:<30} n={int(row['n_cells']):2d} "
            f"mean={row['mean_gain']:+.4f} median={row['median_gain']:+.4f} "
            f"min={row['min_gain']:+.4f} max={row['max_gain']:+.4f} "
            f"frac>0={row['frac_positive']:.2f}"
        )
    lines.extend(["", "--- matched cells ---"])
    for _, row in gain.sort_values(["dataset", "scheme", "input_mode"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['input_mode']:<22} "
            f"direct={row['direct_median']:+.4f}  "
            f"theta={row['theta_median']:+.4f} ({row['theta_best_method']})  "
            f"MLG={row['middle_layer_gain']:+.4f}"
        )
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
