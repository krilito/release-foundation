"""
83 - Export PLGA fold-wise predictions in the shared benchmark format.

Purpose:
    Reuse the audited PLGA tree routes and write shared-format trajectory
    predictions so the dedicated timescale layer can evaluate PLGA in the
    same way as liposome.

Scope:
    This script is intentionally narrow. It exports one dataset / split /
    input-mode / route choice at a time, with `auto` resolving to the
    best audited method from existing summary tables.

Consumes:
    scripts/41_theta_target_scaling_audit.py
    scripts/42_internal181_tree_theta_audit.py
    scripts/46_direct_curve_rf_input_ablation.py
    outputs/45_input_source_ablation/input_source_summary.csv
    outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv

Produces:
    outputs/83_plga_prediction_export/<dataset>/<scheme>/<input_mode>/<route_family>/
        curves_long.csv
        metadata.csv
        predictions.csv
        fold_curve_metadata.csv
        summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import PLGA_CONTINUOUS_COLS
from simulator import PLGABiphasic


DATASETS = ("cross321", "internal181")
SCHEMES = ("random_5fold", "group_by_drug", "group_by_polymer")
INPUT_MODES = ("formulation_only", "early_only", "formulation_plus_early")
ROUTE_FAMILIES = ("theta", "direct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--route-family", choices=ROUTE_FAMILIES, required=True)
    parser.add_argument("--method", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    parser.add_argument("--late-grid", nargs="+", type=float, default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0])
    parser.add_argument("--t-grid-max-days", type=float, default=90.0)
    parser.add_argument("--fit-filter-r2", type=float, default=0.95)
    parser.add_argument("--min-obs", type=int, default=3)
    parser.add_argument("--regime-csv", type=Path, default=Path("outputs/33_active_set_regimes/regime_assignments.csv"))
    parser.add_argument("--full-fit-bank", type=Path, default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"))
    parser.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    parser.add_argument("--matched-fids-csv", type=Path, default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    parser.add_argument("--theta-summary", type=Path, default=Path("outputs/45_input_source_ablation/input_source_summary.csv"))
    parser.add_argument("--direct-summary", type=Path, default=Path("outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv"))
    parser.add_argument("--outroot", type=Path, default=Path("outputs/83_plga_prediction_export"))
    return parser.parse_args()


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _curve_id(dataset: str, fid: int) -> str:
    return f"{dataset}_{fid}"


def _resolve_auto_method(args: argparse.Namespace) -> str:
    if args.method != "auto":
        return args.method
    if args.route_family == "theta":
        df = pd.read_csv(args.theta_summary)
        row = df[
            (df["dataset"] == args.dataset)
            & (df["scheme"] == args.scheme)
            & (df["input_mode"] == args.input_mode)
        ]
        if len(row) != 1:
            raise ValueError("Could not resolve unique theta auto method")
        return str(row.iloc[0]["method"])
    df = pd.read_csv(args.direct_summary)
    row = df[
        (df["dataset"] == args.dataset)
        & (df["scheme"] == args.scheme)
        & (df["input_mode"] == args.input_mode)
    ]
    if len(row) != 1:
        raise ValueError("Could not resolve unique direct auto method")
    return str(row.iloc[0]["direct_best_method"])


def _build_cross321(args: argparse.Namespace, mod41: ModuleType) -> tuple[pd.DataFrame, dict[int, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df, curve_map, x, theta, _param_names, drug_groups, polymer_groups = mod41._load_dataset(args)
    early_times = np.array(args.early_times, dtype=float)
    early_q = x[:, -len(early_times):]
    x_form = x[:, :-len(early_times)]
    meta = df.copy()
    meta["drug_group"] = pd.Series(drug_groups).astype(str)
    meta["polymer_group"] = pd.Series(polymer_groups).astype(str)
    meta["curve_id"] = meta[mod41.FID_COL].astype(int).map(lambda fid: _curve_id("cross321", fid))
    return meta, curve_map, x_form, early_q, theta, drug_groups, polymer_groups


def _build_internal181(args: argparse.Namespace, mod42: ModuleType) -> tuple[pd.DataFrame, dict[int, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    desc, curve_map, _inventory = mod42._load_clean_internal(
        data_csv=args.data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )
    early_times = np.array(args.early_times, dtype=float)
    fids = desc[mod42.INTERNAL_FID_COL].to_numpy(dtype=int)
    early_q = np.stack(
        [mod42._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, early_times) for fid in fids],
        axis=0,
    ).astype(np.float32)
    x_form = desc[list(PLGA_CONTINUOUS_COLS)].to_numpy(dtype=np.float32)
    sim = PLGABiphasic()
    theta = desc[list(sim.param_names)].to_numpy(dtype=np.float32)
    desc = desc.copy()
    desc["drug_group"] = [curve_map[int(fid)].drug_id for fid in fids]
    desc["polymer_group"] = [curve_map[int(fid)].polymer_family for fid in fids]
    desc["curve_id"] = desc[mod42.INTERNAL_FID_COL].astype(int).map(lambda fid: _curve_id("internal181", fid))
    return desc, curve_map, x_form, early_q, theta, np.array(desc["drug_group"], dtype=object), np.array(desc["polymer_group"], dtype=object)


def _x_for_mode(x_form: np.ndarray, early_q: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "formulation_only":
        return x_form
    if input_mode == "early_only":
        return early_q
    return np.concatenate([x_form, early_q], axis=1)


def _split_indices(args: argparse.Namespace, n: int, drug_groups: np.ndarray, polymer_groups: np.ndarray, mod46: ModuleType) -> list[tuple[int, np.ndarray, np.ndarray]]:
    scheme_list = mod46._split_schemes(
        n=n,
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    for scheme_name, splits in scheme_list:
        if scheme_name == args.scheme:
            return [(fold_idx, tr, te) for fold_idx, (tr, te) in enumerate(splits)]
    raise ValueError(f"scheme {args.scheme} not available")


def _fit_theta_method(
    method: str,
    x_model: np.ndarray,
    theta: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    early_times: np.ndarray,
    early_q: np.ndarray,
    fids: np.ndarray,
    curve_map: dict[int, object],
    lows: np.ndarray,
    highs: np.ndarray,
    seed: int,
    sim: PLGABiphasic,
    mod41: ModuleType,
) -> tuple[np.ndarray, np.ndarray]:
    def rf(s: int, **kwargs: object) -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=400, max_depth=None, random_state=s, n_jobs=-1, **kwargs)

    def et(s: int, **kwargs: object) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(n_estimators=400, max_depth=None, random_state=s, n_jobs=-1, **kwargs)

    pred_cache: dict[str, np.ndarray] = {}

    def get_pred(name: str) -> np.ndarray:
        if name in pred_cache:
            return pred_cache[name]
        if name == "RF_raw":
            pred = mod41._fit_predict_raw(rf(seed), x_model, theta, tr, te)
        elif name == "RF_ztheta":
            pred = mod41._fit_predict_ztheta(rf(seed + 100), x_model, theta, tr, te)
        elif name == "RF_ztheta_leaf2":
            pred = mod41._fit_predict_ztheta(rf(seed + 200, min_samples_leaf=2), x_model, theta, tr, te)
        elif name == "ET_raw":
            pred = mod41._fit_predict_raw(et(seed + 300), x_model, theta, tr, te)
        elif name == "ET_ztheta":
            pred = mod41._fit_predict_ztheta(et(seed + 400), x_model, theta, tr, te)
        elif name == "ET_ztheta_leaf2":
            pred = mod41._fit_predict_ztheta(et(seed + 500, min_samples_leaf=2), x_model, theta, tr, te)
        elif name == "RF_ET_zavg":
            pred = 0.5 * (get_pred("RF_ztheta") + get_pred("ET_ztheta"))
        elif name == "RF_raw_zavg":
            pred = 0.5 * (get_pred("RF_raw") + get_pred("RF_ztheta"))
        else:
            raise ValueError(f"Unsupported theta method {name}")
        pred_cache[name] = pred
        return pred

    if method != "early_select":
        theta_pred = get_pred(method)
        selected_method = np.array([method] * len(te), dtype=object)
        return theta_pred, selected_method

    select_pool = ["RF_ztheta", "ET_ztheta", "RF_ET_zavg"]
    pool_preds = {name: get_pred(name) for name in select_pool}
    chosen = []
    selected_thetas = []
    for local_i, j in enumerate(te):
        fid = int(fids[j])
        rmses: dict[str, float] = {}
        for name in select_pool:
            th = mod41._clip_theta(pool_preds[name][local_i], lows, highs)
            pred_early = sim.simulate_numpy(th, early_times)
            rmses[name] = mod41._rmse(early_q[j].astype(float), pred_early)
        best = min(select_pool, key=lambda name: rmses[name])
        chosen.append(best)
        selected_thetas.append(pool_preds[best][local_i])
    return np.asarray(selected_thetas, dtype=float), np.asarray(chosen, dtype=object)


def _fit_direct_method(
    method: str,
    x_model: np.ndarray,
    target_q: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    mod46: ModuleType,
    seed: int,
) -> np.ndarray:
    methods = mod46._make_methods(argparse.Namespace(n_estimators=400), seed)
    if method not in methods:
        raise ValueError(f"Unsupported direct method {method}")
    model = methods[method]
    model.fit(x_model[tr], target_q[tr])
    return model.predict(x_model[te])


def main() -> None:
    args = parse_args()
    resolved_method = _resolve_auto_method(args)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "_script41_83")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "_script42_83")
    mod46 = _load_script(scripts_dir / "46_direct_curve_rf_input_ablation.py", "_script46_83")

    if args.dataset == "cross321":
        meta, curve_map, x_form, early_q, theta, drug_groups, polymer_groups = _build_cross321(args, mod41)
        fid_col = mod41.FID_COL
        curve_id_prefix = "cross321"
        formulation_cols = list(mod41.FORMULATION_COL_MAP.values())
    else:
        meta, curve_map, x_form, early_q, theta, drug_groups, polymer_groups = _build_internal181(args, mod42)
        fid_col = mod42.INTERNAL_FID_COL
        curve_id_prefix = "internal181"
        formulation_cols = list(PLGA_CONTINUOUS_COLS)

    fids = meta[fid_col].to_numpy(dtype=int)
    x_model = _x_for_mode(x_form, early_q, args.input_mode)
    early_times = np.array(args.early_times, dtype=float)
    late_times = np.array(args.late_grid, dtype=float)
    early_window_days = float(np.max(early_times)) if args.input_mode != "formulation_only" else 0.0

    curves_long_rows: list[dict[str, object]] = []
    for _, row in meta.iterrows():
        fid = int(row[fid_col])
        curve = curve_map[fid]
        curve_id = str(row["curve_id"])
        for t_day, q in zip(curve.t_obs.tolist(), curve.q_obs.tolist()):
            curves_long_rows.append(
                {
                    "curve_id": curve_id,
                    "fid": fid,
                    "formulation": f"{curve_id_prefix}_{fid}",
                    "drug": str(row["drug_group"]),
                    "t_days": float(t_day),
                    "Q_observed": float(q),
                    "dataset_name": args.dataset,
                    "mechanism_id": "plga_biphasic",
                    "early_window_days": early_window_days,
                }
            )
    curves_long = pd.DataFrame(curves_long_rows).sort_values(["curve_id", "t_days"]).reset_index(drop=True)

    metadata_cols = ["curve_id", fid_col, "drug_group", "polymer_group"] + [c for c in formulation_cols if c in meta.columns]
    metadata = meta[metadata_cols].copy()
    metadata = metadata.rename(columns={fid_col: "fid", "drug_group": "drug_id", "polymer_group": "polymer_family"})
    metadata["dataset_name"] = args.dataset
    metadata["mechanism_id"] = "plga_biphasic"
    metadata["time_unit_standard"] = "days"
    metadata["early_window_days"] = early_window_days

    outdir = args.outroot / args.dataset / args.scheme / args.input_mode / f"{args.route_family}_{resolved_method}"
    outdir.mkdir(parents=True, exist_ok=True)
    curves_long.to_csv(outdir / "curves_long.csv", index=False)
    metadata.to_csv(outdir / "metadata.csv", index=False)

    splits = _split_indices(args, len(meta), drug_groups, polymer_groups, mod46)
    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)

    predictions_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    target_times = mod46._target_grid(early_times, late_times, args.input_mode)
    target_q = np.stack(
        [mod41._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, target_times) for fid in fids],
        axis=0,
    ).astype(np.float32)

    for fold_idx, tr, te in splits:
        seed = args.seed + fold_idx
        if args.route_family == "theta":
            theta_pred, selected = _fit_theta_method(
                method=resolved_method,
                x_model=x_model,
                theta=theta,
                tr=tr,
                te=te,
                early_times=early_times,
                early_q=early_q,
                fids=fids,
                curve_map=curve_map,
                lows=lows,
                highs=highs,
                seed=seed,
                sim=sim,
                mod41=mod41,
            )
        else:
            pred_grid = _fit_direct_method(
                method=resolved_method,
                x_model=x_model,
                target_q=target_q,
                tr=tr,
                te=te,
                mod46=mod46,
                seed=seed,
            )
            selected = np.array([resolved_method] * len(te), dtype=object)

        for local_i, j in enumerate(te):
            fid = int(fids[j])
            curve = curve_map[fid]
            curve_id = _curve_id(args.dataset, fid)
            if args.route_family == "theta":
                theta_use = mod41._clip_theta(theta_pred[local_i], lows, highs)
                q_hat = sim.simulate_numpy(theta_use, curve.t_obs)
                theta_named = {
                    f"theta_{k}": float(theta_use[k]) for k in range(len(theta_use))
                }
            else:
                q_hat = mod46._reconstruct_curve(
                    pred_grid_q=pred_grid[local_i],
                    target_times=target_times,
                    early_q=early_q[j].astype(float),
                    early_times=early_times,
                    input_mode=args.input_mode,
                    t_obs=curve.t_obs,
                )
                theta_named = {
                    f"theta_{k}": float("nan") for k in range(sim.n_params)
                }

            row = {
                "curve_id": curve_id,
                "fid": fid,
                "fold": fold_idx,
                "dataset": args.dataset,
                "scheme": args.scheme,
                "input_mode": args.input_mode,
                "route_family": args.route_family,
                "resolved_method": resolved_method,
                "selected_method": str(selected[local_i]),
                "r2": float(mod41._r2(curve.q_obs, q_hat)),
                "rmse": float(mod41._rmse(curve.q_obs, q_hat)),
            }
            row.update(theta_named)
            fold_rows.append(row)
            for t_day, q_pred in zip(curve.t_obs.tolist(), q_hat.tolist()):
                predictions_rows.append(
                    {
                        "curve_id": curve_id,
                        "fid": fid,
                        "fold": fold_idx,
                        "scheme": args.scheme,
                        "input_mode": args.input_mode,
                        "route_family": args.route_family,
                        "resolved_method": resolved_method,
                        "selected_method": str(selected[local_i]),
                        "formulation": f"{curve_id_prefix}_{fid}",
                        "drug": str(meta.iloc[j]["drug_group"]),
                        "t_days": float(t_day),
                        "Q_predicted_point": float(q_pred),
                    }
                )

    predictions = pd.DataFrame(predictions_rows).sort_values(["curve_id", "t_days"]).reset_index(drop=True)
    fold_meta = pd.DataFrame(fold_rows).sort_values(["fold", "curve_id"]).reset_index(drop=True)

    predictions.to_csv(outdir / "predictions.csv", index=False)
    fold_meta.to_csv(outdir / "fold_curve_metadata.csv", index=False)

    summary_lines = [
        "PLGA shared prediction export complete.",
        f"dataset: {args.dataset}",
        f"scheme: {args.scheme}",
        f"input_mode: {args.input_mode}",
        f"route_family: {args.route_family}",
        f"resolved_method: {resolved_method}",
        f"curves: {int(fold_meta['curve_id'].nunique())}",
        f"points: {int(len(predictions))}",
        f"median_r2: {float(fold_meta['r2'].median()):+.4f}",
        f"frac_r2_ge0: {float((fold_meta['r2'] >= 0.0).mean()):.3f}",
        f"early_window_days: {early_window_days:.1f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
