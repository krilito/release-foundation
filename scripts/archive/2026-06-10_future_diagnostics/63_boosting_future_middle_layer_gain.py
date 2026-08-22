"""
63 - Boosting-family future-only middle-layer gain audit.

What this does:
    Script 62 showed that LightGBM also gets a positive full-curve
    middle-layer gain (MLG). This script asks the harder question:

        does the kinetic-state layer help on the future segment only?

    It compares boosting backbones under matched inputs/splits:

        direct-Q:
            x / early Q -> Q grid

        kinetic-state:
            x / early Q -> theta -> PLGABiphasic -> Q(t)

    Metrics:
        full_mlg   = median full-curve R^2(theta) - median full-curve R^2(direct)
        future_mlg = median future-only R^2(theta) - median future-only R^2(direct)
        future_rmse_gain = median future RMSE(direct) - median future RMSE(theta)

    Positive MLG and positive RMSE gain favor the kinetic-state layer.

Outputs:
    outputs/63_boosting_future_middle_layer_gain/per_curve.csv
    outputs/63_boosting_future_middle_layer_gain/method_summary.csv
    outputs/63_boosting_future_middle_layer_gain/middle_layer_gain.csv
    outputs/63_boosting_future_middle_layer_gain/aggregate_middle_layer_gain.csv
    outputs/63_boosting_future_middle_layer_gain/summary.txt
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402

DATASETS = ("cross321", "internal181")
INPUT_MODES = ("formulation_only", "early_only", "formulation_plus_early")
MODEL_FAMILIES = ("lgbm", "hgb", "xgb")


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


def _xgb_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def _make_model(args: argparse.Namespace, family: str, seed: int) -> MultiOutputRegressor:
    if family == "lgbm":
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

    if family == "hgb":
        base = HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.num_leaves,
            min_samples_leaf=args.min_child_samples,
            l2_regularization=args.reg_lambda,
            random_state=seed,
        )
        return MultiOutputRegressor(base, n_jobs=1)

    if family == "xgb":
        if not _xgb_available():
            raise ImportError("xgboost is not installed")
        from xgboost import XGBRegressor  # type: ignore[import-not-found]

        base = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            random_state=seed,
            n_jobs=args.n_jobs,
            objective="reg:squarederror",
            verbosity=0,
        )
        return MultiOutputRegressor(base, n_jobs=1)

    raise ValueError(f"unknown model family: {family}")


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


def _future_metrics(
    t_obs: np.ndarray,
    q_obs: np.ndarray,
    q_hat: np.ndarray,
    early_max: float,
    min_future_obs: int,
) -> tuple[float, float]:
    mask = t_obs > early_max
    if int(np.sum(mask)) < min_future_obs:
        return float("nan"), float("nan")
    return _r2(q_obs[mask], q_hat[mask]), _rmse(q_obs[mask], q_hat[mask])


def _run_dataset_mode_family(
    args: argparse.Namespace,
    dataset: str,
    input_mode: str,
    family: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod62: ModuleType,
) -> list[dict[str, object]]:
    if dataset == "cross321":
        pack = mod62._load_cross321(args, mod41)
    else:
        pack = mod62._load_internal181(args, mod42)
    fids, x_form, early_q, theta, curve_map, drug_groups, polymer_groups, early_times = pack

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)

    x_model = mod62._x_for_mode(x_form, early_q, input_mode).astype(np.float32)
    late_times = np.array(args.late_grid, dtype=float)
    target_times = mod62._target_grid(early_times, late_times, input_mode)
    target_q = np.stack([
        mod41._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, target_times)
        for fid in fids
    ]).astype(np.float32)

    print(
        f"[63] family={family} dataset={dataset} input_mode={input_mode} "
        f"n={len(fids)} x_dim={x_model.shape[1]}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for scheme_name, splits in mod62._split_schemes(
        n=len(fids),
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        n_folds=args.n_folds,
        seed=args.seed,
    ):
        print(f"[63]   scheme={scheme_name}", flush=True)
        for fold_idx, (tr, te) in enumerate(splits):
            direct = _make_model(args, family, args.seed + fold_idx)
            direct.fit(x_model[tr], target_q[tr])
            pred_q = np.asarray(direct.predict(x_model[te]), dtype=float)

            theta_raw = _fit_predict_raw(
                _make_model(args, family, args.seed + 100 + fold_idx), x_model, theta, tr, te
            )
            theta_z = _fit_predict_ztheta(
                _make_model(args, family, args.seed + 200 + fold_idx), x_model, theta, tr, te
            )

            for local_i, j in enumerate(te):
                fid = int(fids[j])
                curve = curve_map[fid]
                q_direct = mod62._reconstruct_curve(
                    pred_grid_q=pred_q[local_i],
                    target_times=target_times,
                    early_q=early_q[j].astype(float),
                    early_times=early_times,
                    input_mode=input_mode,
                    t_obs=curve.t_obs,
                )
                direct_future_r2, direct_future_rmse = _future_metrics(
                    curve.t_obs,
                    curve.q_obs,
                    q_direct,
                    early_max=float(np.max(early_times)),
                    min_future_obs=args.min_future_obs,
                )

                row: dict[str, object] = {
                    "family": family,
                    "dataset": dataset,
                    "input_mode": input_mode,
                    "scheme": scheme_name,
                    "fold": fold_idx,
                    "fid": fid,
                    "n_obs": int(len(curve.t_obs)),
                    "n_future_obs": int(np.sum(curve.t_obs > float(np.max(early_times)))),
                    "full_r2_direct_Q": _r2(curve.q_obs, q_direct),
                    "future_r2_direct_Q": direct_future_r2,
                    "future_rmse_direct_Q": direct_future_rmse,
                }

                early_rmse: dict[str, float] = {}
                for method, pred_theta in {
                    "theta_raw": theta_raw,
                    "theta_z": theta_z,
                }.items():
                    clipped = _clip_theta(pred_theta[local_i], lows, highs)
                    pred = sim.simulate_numpy(clipped, curve.t_obs)
                    future_r2, future_rmse = _future_metrics(
                        curve.t_obs,
                        curve.q_obs,
                        pred,
                        early_max=float(np.max(early_times)),
                        min_future_obs=args.min_future_obs,
                    )
                    row[f"full_r2_{method}"] = _r2(curve.q_obs, pred)
                    row[f"future_r2_{method}"] = future_r2
                    row[f"future_rmse_{method}"] = future_rmse
                    pred_early = sim.simulate_numpy(clipped, early_times)
                    early_rmse[method] = _rmse(early_q[j].astype(float), pred_early)

                selected = min(early_rmse, key=early_rmse.get)
                row["theta_early_select_method"] = selected
                for metric in ("full_r2", "future_r2", "future_rmse"):
                    row[f"{metric}_theta_early_select"] = row[f"{metric}_{selected}"]
                rows.append(row)
    return rows


def _method_summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = [
        c for c in per_curve.columns
        if c.startswith(("full_r2_", "future_r2_", "future_rmse_"))
    ]
    for (family, dataset, input_mode, scheme), sub in per_curve.groupby(
        ["family", "dataset", "input_mode", "scheme"], sort=False
    ):
        for col in metric_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append({
                "family": family,
                "dataset": dataset,
                "input_mode": input_mode,
                "scheme": scheme,
                "metric_method": col,
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p10": float(np.percentile(vals, 10)),
                "frac_above_0": float(np.mean(vals >= 0.0)),
            })
    return pd.DataFrame(rows)


def _best_theta(summary: pd.DataFrame, metric_prefix: str) -> pd.DataFrame:
    theta = summary[
        summary["metric_method"].isin({
            f"{metric_prefix}_theta_raw",
            f"{metric_prefix}_theta_z",
            f"{metric_prefix}_theta_early_select",
        })
    ].copy()
    idx = theta.groupby(["family", "dataset", "input_mode", "scheme"])["median"].idxmax()
    return theta.loc[idx].rename(columns={
        "metric_method": f"{metric_prefix}_theta_best_method",
        "median": f"{metric_prefix}_theta_median",
    })


def _middle_layer_gain(summary: pd.DataFrame) -> pd.DataFrame:
    keep = ["family", "dataset", "input_mode", "scheme"]
    direct_full = summary[summary["metric_method"] == "full_r2_direct_Q"].rename(
        columns={"median": "full_direct_median"}
    )
    direct_future = summary[summary["metric_method"] == "future_r2_direct_Q"].rename(
        columns={"median": "future_direct_median"}
    )
    direct_future_rmse = summary[summary["metric_method"] == "future_rmse_direct_Q"].rename(
        columns={"median": "future_direct_rmse_median"}
    )
    full_theta = _best_theta(summary, "full_r2")
    future_theta = _best_theta(summary, "future_r2")

    theta_rmse = summary[
        summary["metric_method"].isin({
            "future_rmse_theta_raw",
            "future_rmse_theta_z",
            "future_rmse_theta_early_select",
        })
    ].copy()
    idx_rmse = theta_rmse.groupby(keep)["median"].idxmin()
    theta_rmse_best = theta_rmse.loc[idx_rmse].rename(columns={
        "metric_method": "future_rmse_theta_best_method",
        "median": "future_theta_rmse_median",
    })

    out = direct_full[keep + ["full_direct_median"]]
    out = out.merge(
        full_theta[keep + ["full_r2_theta_best_method", "full_r2_theta_median"]],
        on=keep,
        how="inner",
    )
    out = out.merge(direct_future[keep + ["future_direct_median"]], on=keep, how="left")
    out = out.merge(
        future_theta[keep + ["future_r2_theta_best_method", "future_r2_theta_median"]],
        on=keep,
        how="left",
    )
    out = out.merge(
        direct_future_rmse[keep + ["future_direct_rmse_median"]],
        on=keep,
        how="left",
    )
    out = out.merge(
        theta_rmse_best[keep + ["future_rmse_theta_best_method", "future_theta_rmse_median"]],
        on=keep,
        how="left",
    )
    out["full_mlg"] = out["full_r2_theta_median"] - out["full_direct_median"]
    out["future_mlg"] = out["future_r2_theta_median"] - out["future_direct_median"]
    out["future_rmse_gain"] = (
        out["future_direct_rmse_median"] - out["future_theta_rmse_median"]
    )
    return out.sort_values(keep).reset_index(drop=True)


def _aggregate(gain: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("all", gain)]
    for family in sorted(gain["family"].unique()):
        groups.append((f"family/{family}", gain[gain["family"] == family]))
    groups.extend([
        (
            "formulation_plus_early",
            gain[gain["input_mode"] == "formulation_plus_early"],
        ),
        (
            "OOD_formulation_plus_early",
            gain[
                (gain["input_mode"] == "formulation_plus_early")
                & (gain["scheme"] != "random_5fold")
            ],
        ),
    ])

    for name, sub in groups:
        for metric in ("full_mlg", "future_mlg", "future_rmse_gain"):
            vals = sub[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append({
                "group": name,
                "metric": metric,
                "n_cells": int(len(vals)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
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
    ap.add_argument("--min-future-obs", type=int, default=3)
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument(
        "--late-grid",
        nargs="+",
        type=float,
        default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0],
    )
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--input-mode", choices=(*INPUT_MODES, "all"), default="all")
    ap.add_argument("--families", nargs="+", choices=MODEL_FAMILIES, default=["lgbm", "hgb", "xgb"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=120)
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
    ap.add_argument("--out", type=Path, default=Path("outputs/63_boosting_future_middle_layer_gain"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta_for63")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal_for63")
    mod62 = _load_script(scripts_dir / "62_lgbm_middle_layer_gain.py", "script62_for63")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    input_modes = INPUT_MODES if args.input_mode == "all" else (args.input_mode,)
    families = list(args.families)
    skipped: list[str] = []
    if "xgb" in families and not _xgb_available():
        families.remove("xgb")
        skipped.append("xgb:not_installed")

    rows: list[dict[str, object]] = []
    for family in families:
        for dataset in datasets:
            for input_mode in input_modes:
                rows.extend(
                    _run_dataset_mode_family(args, dataset, input_mode, family, mod41, mod42, mod62)
                )

    per_curve = pd.DataFrame(rows)
    summary = _method_summary(per_curve)
    gain = _middle_layer_gain(summary)
    aggregate = _aggregate(gain)

    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "method_summary.csv", index=False)
    gain.to_csv(args.out / "middle_layer_gain.csv", index=False)
    aggregate.to_csv(args.out / "aggregate_middle_layer_gain.csv", index=False)

    lines = [
        "=== 63 -- boosting-family future-only MLG audit ===",
        "",
        f"families run : {', '.join(families) if families else 'none'}",
        f"skipped      : {', '.join(skipped) if skipped else 'none'}",
        f"datasets     : {', '.join(datasets)}",
        f"input_modes  : {', '.join(input_modes)}",
        f"early_max    : {max(args.early_times):.3f} d",
        f"min_future_obs: {args.min_future_obs}",
        "",
        "--- aggregate ---",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"  {row['group']:<32} {row['metric']:<16} n={int(row['n_cells']):2d} "
            f"mean={row['mean']:+.4f} median={row['median']:+.4f} "
            f"min={row['min']:+.4f} max={row['max']:+.4f} frac>0={row['frac_positive']:.2f}"
        )

    lines.extend(["", "--- OOD formulation+early cells ---"])
    ood = gain[
        (gain["input_mode"] == "formulation_plus_early")
        & (gain["scheme"] != "random_5fold")
    ].sort_values(["family", "dataset", "scheme"])
    for _, row in ood.iterrows():
        lines.append(
            f"  {row['family']:<4} {row['dataset']:<11} {row['scheme']:<16} "
            f"fullMLG={row['full_mlg']:+.4f} futureMLG={row['future_mlg']:+.4f} "
            f"futureRMSEgain={row['future_rmse_gain']:+.4f}"
        )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  Future-only R^2 can be unstable on plateaued curves with low future",
        "  variance, so future_rmse_gain is reported alongside future_mlg.",
    ])

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
