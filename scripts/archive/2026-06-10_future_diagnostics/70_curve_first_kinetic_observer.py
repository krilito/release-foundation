"""
70 - Curve-first kinetic observer.

What this does:
    Test the route the project is now converging toward:

        early release curve only -> kinetic state/posterior -> simulator rollout

    Formulation descriptors are deliberately not used by the main methods.
    They appear only in an explicit reference upper baseline
    (`reference_formulation_plus_early_theta`), so we can ask whether the
    curve itself already carries enough kinetic evidence.

Outputs:
    outputs/70_curve_first_kinetic_observer/per_curve.csv
    outputs/70_curve_first_kinetic_observer/scheme_method_summary.csv
    outputs/70_curve_first_kinetic_observer/summary.txt
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
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import PLGABiphasic  # noqa: E402

DATASETS = ("cross321", "internal181")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fit_predict_ztheta(
    model: object,
    x: np.ndarray,
    theta: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> np.ndarray:
    mu = theta[tr].mean(axis=0, keepdims=True)
    sd = theta[tr].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x[tr], (theta[tr] - mu) / sd)
    return model.predict(x[te]) * sd + mu


def _curve_first_fold(
    args: argparse.Namespace,
    sim: PLGABiphasic,
    mod65: ModuleType,
    pack: object,
    grid_times: np.ndarray,
    early_idx: np.ndarray,
    late_mask: np.ndarray,
    y_grid: np.ndarray,
    dataset: str,
    scheme: str,
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> list[dict[str, object]]:
    fold_seed = args.seed + 10_000 * fold + len(dataset)
    rng = np.random.default_rng(fold_seed)
    x_early = pack.early_q.astype(np.float32)
    x_form_early = np.concatenate([pack.x_form, pack.early_q], axis=1).astype(np.float32)

    train_median = np.median(y_grid[tr], axis=0)

    # Direct curve baselines from the exact same early-curve evidence.
    knn = KNeighborsRegressor(n_neighbors=min(args.n_neighbors, len(tr)), weights="distance")
    knn.fit(x_early[tr], y_grid[tr][:, late_mask])
    knn_pred = np.tile(train_median, (len(te), 1))
    knn_pred[:, late_mask] = np.clip(knn.predict(x_early[te]), 0.0, 1.0)

    direct_et = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=2,
        random_state=fold_seed + 11,
        n_jobs=-1,
    )
    direct_et.fit(x_early[tr], y_grid[tr][:, late_mask])
    direct_et_pred = np.tile(train_median, (len(te), 1))
    direct_et_pred[:, late_mask] = np.clip(direct_et.predict(x_early[te]), 0.0, 1.0)

    # Curve-first kinetic state: early Q -> theta -> ODE.
    theta_rf = mod65._clip_theta(
        _fit_predict_ztheta(
            RandomForestRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=fold_seed + 21,
                n_jobs=-1,
            ),
            x_early,
            pack.theta,
            tr,
            te,
        ),
        lows,
        highs,
    )
    theta_et = mod65._clip_theta(
        _fit_predict_ztheta(
            ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=fold_seed + 22,
                n_jobs=-1,
            ),
            x_early,
            pack.theta,
            tr,
            te,
        ),
        lows,
        highs,
    )
    theta_avg = mod65._clip_theta(0.5 * (theta_rf + theta_et), lows, highs)
    theta_rf_pred = mod65._simulate_grid(sim, theta_rf, grid_times, args.device, args.sim_batch_size)
    theta_et_pred = mod65._simulate_grid(sim, theta_et, grid_times, args.device, args.sim_batch_size)
    theta_avg_pred = mod65._simulate_grid(sim, theta_avg, grid_times, args.device, args.sim_batch_size)

    # Explicit upper reference: formulation + early Q, not a curve-first method.
    theta_ref = mod65._clip_theta(
        _fit_predict_ztheta(
            ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=fold_seed + 33,
                n_jobs=-1,
            ),
            x_form_early,
            pack.theta,
            tr,
            te,
        ),
        lows,
        highs,
    )
    theta_ref_pred = mod65._simulate_grid(sim, theta_ref, grid_times, args.device, args.sim_batch_size)

    theta_sd = np.where(pack.theta[tr].std(axis=0) < 1e-6, 1.0, pack.theta[tr].std(axis=0)).astype(np.float32)
    particle_mean: list[np.ndarray] = []
    particle_map: list[np.ndarray] = []
    refined_theta: list[np.ndarray] = []
    extra: dict[str, list[dict[str, float]]] = {
        "curve_first_particle_mean": [],
        "curve_first_particle_map": [],
        "curve_first_map_refined": [],
    }
    for local_i, j in enumerate(te):
        d = np.sum((x_early[tr] - x_early[j].reshape(1, -1)) ** 2, axis=1)
        near = tr[np.argsort(d)[: min(args.max_train_bank_particles, len(tr))]]
        bank_candidates = mod65._clip_theta(pack.theta[near], lows, highs)

        centers = np.stack([theta_rf[local_i], theta_et[local_i], theta_avg[local_i]]).astype(np.float32)
        jitter_parts: list[np.ndarray] = []
        per_center = max(1, args.jitter_particles // len(centers))
        for center in centers:
            jitter = rng.normal(0.0, args.jitter_scale, size=(per_center, center.shape[0])).astype(np.float32)
            jitter_parts.append(center.reshape(1, -1) + jitter * theta_sd.reshape(1, -1))
        candidates = mod65._clip_theta(np.concatenate([bank_candidates, *jitter_parts], axis=0), lows, highs)

        q_mean, q_map, stats = mod65._particle_predict_one(
            sim,
            candidates,
            grid_times,
            early_idx,
            x_early[j],
            args,
        )
        particle_mean.append(q_mean)
        particle_map.append(q_map)
        extra["curve_first_particle_mean"].append(stats)
        extra["curve_first_particle_map"].append(stats)

        theta_refined, refine_rmse = mod65._refine_theta_map(
            sim=sim,
            theta0=theta_avg[local_i],
            theta_sd=theta_sd,
            early_times=pack.early_times,
            early_q=x_early[j],
            lows=lows,
            highs=highs,
            args=args,
        )
        refined_theta.append(theta_refined)
        extra["curve_first_map_refined"].append({"map_refined_early_rmse": refine_rmse})

    refined_pred = mod65._simulate_grid(
        sim,
        np.asarray(refined_theta, dtype=np.float32),
        grid_times,
        args.device,
        args.sim_batch_size,
    )

    pred_by_method = {
        "direct_Q_KNN_early_curve": knn_pred,
        "direct_Q_ET_early_curve": direct_et_pred,
        "curve_first_RF_theta": theta_rf_pred,
        "curve_first_ET_theta": theta_et_pred,
        "curve_first_RF_ET_theta_avg": theta_avg_pred,
        "curve_first_particle_mean": np.asarray(particle_mean, dtype=np.float32),
        "curve_first_particle_map": np.asarray(particle_map, dtype=np.float32),
        "curve_first_map_refined": refined_pred,
        "reference_formulation_plus_early_theta": theta_ref_pred,
    }
    return mod65._eval_predictions(
        dataset=dataset,
        scheme=scheme,
        fold=fold,
        pack=pack,
        te=te,
        grid_times=grid_times,
        pred_by_method=pred_by_method,
        extra_by_method=extra,
        min_future_obs=args.min_future_obs,
    )


def _run_dataset(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
    mod65: ModuleType,
) -> list[dict[str, object]]:
    sim = PLGABiphasic()
    lows, highs = mod65._prior_bounds(sim)
    pack = mod65._load_pack(args, dataset, mod41, mod42, mod46, sim)
    grid_times = np.unique(np.concatenate([
        np.asarray(args.early_times, dtype=float),
        np.asarray(args.grid_times, dtype=float),
    ]))
    grid_times = grid_times[grid_times <= args.t_grid_max_days]
    early_idx = np.array([int(np.where(np.isclose(grid_times, t))[0][0]) for t in pack.early_times], dtype=int)
    late_mask = grid_times > float(np.max(pack.early_times))
    y_grid = mod65._grid_targets(pack, grid_times, mod41._interp_at)
    print(
        f"[70] dataset={dataset} n={len(pack.fids)} "
        f"early_dim={pack.early_q.shape[1]} grid={len(grid_times)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for scheme, splits in mod65._split_schemes(
        len(pack.fids), pack.drug_groups, pack.polymer_groups, args.n_folds, args.seed
    ):
        print(f"[70]   scheme={scheme}", flush=True)
        for fold, (tr, te) in enumerate(splits):
            rows.extend(_curve_first_fold(
                args=args,
                sim=sim,
                mod65=mod65,
                pack=pack,
                grid_times=grid_times,
                early_idx=early_idx,
                late_mask=late_mask,
                y_grid=y_grid,
                dataset=dataset,
                scheme=scheme,
                fold=fold,
                tr=tr,
                te=te,
                lows=lows,
                highs=highs,
            ))
    return rows


def _write_summary(args: argparse.Namespace, datasets: tuple[str, ...], summary: pd.DataFrame) -> None:
    best = summary.loc[summary.groupby(["dataset", "scheme"])["median_future_rmse"].idxmin()].copy()
    lines = [
        "=== 70 -- curve-first kinetic observer ===",
        "",
        f"datasets       : {', '.join(datasets)}",
        f"early_times    : {args.early_times}",
        f"obs_sigma      : {args.obs_sigma}",
        f"map_prior_w    : {args.map_prior_weight}",
        "",
        "--- best method by future RMSE per dataset/scheme ---",
    ]
    for _, row in best.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['method']:<42} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"future_RMSE={row['median_future_rmse']:.4f}"
        )
    lines.extend(["", "--- curve-first kinetic rows ---"])
    focus = summary[summary["method"].str.startswith("curve_first")].sort_values(["dataset", "scheme", "method"])
    for _, row in focus.iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['method']:<30} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"future_RMSE={row['median_future_rmse']:.4f}"
        )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  Main methods use early release Q only; formulation appears only in the explicit reference row.",
        "  A win here supports curve-first kinetic inference rather than formulation-first prediction.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime-csv", type=Path, default=Path("outputs/33_active_set_regimes/regime_assignments.csv"))
    ap.add_argument("--full-fit-bank", type=Path, default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"))
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument("--matched-fids-csv", type=Path, default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"))
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--min-future-obs", type=int, default=3)
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument("--grid-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0])
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=220)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--sim-batch-size", type=int, default=256)
    ap.add_argument("--obs-sigma", type=float, default=0.035)
    ap.add_argument("--max-train-bank-particles", type=int, default=64)
    ap.add_argument("--jitter-particles", type=int, default=96)
    ap.add_argument("--jitter-scale", type=float, default=0.18)
    ap.add_argument("--map-prior-weight", type=float, default=0.25)
    ap.add_argument("--map-refine-nfev", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("outputs/70_curve_first_kinetic_observer"))
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_for70")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_for70")
    mod46 = _load_script(scripts_dir / "46_direct_curve_rf_input_ablation.py", "script46_for70")
    mod65 = _load_script(scripts_dir / "65_mechanistic_particle_world_model.py", "script65_for70")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        rows.extend(_run_dataset(args, dataset, mod41, mod42, mod46, mod65))

    per_curve = pd.DataFrame(rows)
    summary = mod65._summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)
    _write_summary(args, datasets, summary)


if __name__ == "__main__":
    main()
