"""
65 - Mechanistic particle world model.

What this does:
    Test a stage-1 world-model variant that is more mechanistic than script 64:

        formulation + early Q -> theta proposal bank
        early Q likelihood    -> posterior particle weights
        PLGA ODE simulator    -> future release rollout

    This is not a symbolic-regression loop and not a stitched direct-Q model.
    The middle layer is the 9-D PLGA kinetic state theta, and the world model is
    the simulator-conditioned posterior over theta after observing an early
    partial release curve.

Outputs:
    outputs/65_mechanistic_particle_world_model/per_curve.csv
    outputs/65_mechanistic_particle_world_model/scheme_method_summary.csv
    outputs/65_mechanistic_particle_world_model/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import PLGABiphasic  # noqa: E402

DATASETS = ("cross321", "internal181")


@dataclass
class DataPack:
    fids: np.ndarray
    x_form: np.ndarray
    early_q: np.ndarray
    theta: np.ndarray
    curve_map: dict[int, object]
    drug_groups: np.ndarray
    polymer_groups: np.ndarray
    early_times: np.ndarray


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


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _split_schemes(
    n: int,
    drug_groups: np.ndarray,
    polymer_groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    out.append(("random_5fold", list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(n)))))
    for name, groups in [("group_by_drug", drug_groups), ("group_by_polymer", polymer_groups)]:
        n_splits = min(n_folds, int(pd.Series(groups).nunique()))
        if n_splits >= 2:
            out.append((name, list(GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups))))
    return out


def _prior_bounds(sim: PLGABiphasic) -> tuple[np.ndarray, np.ndarray]:
    prior = sim.prior().base_dist
    return prior.low.detach().cpu().numpy(), prior.high.detach().cpu().numpy()


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(np.float32), lows + eps, highs - eps)


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


def _load_pack(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
    sim: PLGABiphasic,
) -> DataPack:
    if dataset == "cross321":
        fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = (
            mod46._load_cross321(args, mod41)
        )
    else:
        fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = (
            mod46._load_internal181(args, mod42)
        )

    bank = pd.read_csv(args.full_fit_bank)
    bank = bank[bank["dataset"] == dataset].copy()
    if dataset == "internal181":
        bank = bank[np.isfinite(bank["full_r2"]) & (bank["full_r2"] >= args.fit_filter_r2)].copy()
    theta_cols = list(sim.param_names)
    theta_by_fid = {
        int(row["fid"]): row[theta_cols].to_numpy(dtype=np.float32)
        for _, row in bank.dropna(subset=theta_cols).iterrows()
    }
    keep = np.array([int(fid) in theta_by_fid for fid in fids], dtype=bool)
    fids = fids[keep]
    x_form = x_form[keep].astype(np.float32)
    early_q = early_q[keep].astype(np.float32)
    drug_groups = drug_groups[keep]
    polymer_groups = polymer_groups[keep]
    theta = np.stack([theta_by_fid[int(fid)] for fid in fids]).astype(np.float32)
    return DataPack(
        fids=fids,
        x_form=x_form,
        early_q=early_q,
        theta=theta,
        curve_map=curve_map,
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        early_times=early_times,
    )


def _grid_targets(
    pack: DataPack,
    grid_times: np.ndarray,
    interp_fn: object,
) -> np.ndarray:
    return np.stack([
        interp_fn(pack.curve_map[int(fid)].t_obs, pack.curve_map[int(fid)].q_obs, grid_times)
        for fid in pack.fids
    ]).astype(np.float32)


def _reconstruct_at_obs(
    pred_grid_q: np.ndarray,
    grid_times: np.ndarray,
    early_q: np.ndarray,
    early_times: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    late_mask = grid_times > float(np.max(early_times))
    t = np.concatenate([early_times, grid_times[late_mask]])
    q = np.concatenate([early_q, pred_grid_q[late_mask]])
    order = np.argsort(t)
    return np.clip(np.interp(t_obs, t[order], q[order], left=q[order][0], right=q[order][-1]), 0.0, 1.0)


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


def _simulate_grid(
    sim: PLGABiphasic,
    theta: np.ndarray,
    grid_times: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    t = torch.tensor(grid_times, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            th = torch.tensor(theta[start:start + batch_size], dtype=torch.float32, device=device)
            q = sim.simulate(th, t).detach().cpu().numpy().astype(np.float32)
            chunks.append(q)
    return np.concatenate(chunks, axis=0)


def _particle_predict_one(
    sim: PLGABiphasic,
    candidates: np.ndarray,
    grid_times: np.ndarray,
    early_idx: np.ndarray,
    early_q: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    q_grid = _simulate_grid(sim, candidates, grid_times, args.device, args.sim_batch_size)
    resid = q_grid[:, early_idx] - early_q.reshape(1, -1)
    sse = np.sum(resid * resid, axis=1)
    sigma2 = max(args.obs_sigma, 1e-4) ** 2
    logw = -0.5 * sse / sigma2
    logw = logw - float(np.max(logw))
    w = np.exp(logw)
    w = w / max(float(np.sum(w)), 1e-12)
    ess = 1.0 / max(float(np.sum(w * w)), 1e-12)
    q_mean = (w.reshape(1, -1) @ q_grid).reshape(-1)
    q_map = q_grid[int(np.argmin(sse))]
    stats = {
        "particle_ess": ess,
        "particle_best_early_rmse": float(np.sqrt(np.min(sse) / len(early_idx))),
        "particle_mean_early_rmse": _rmse(early_q, q_mean[early_idx]),
    }
    return q_mean, q_map, stats


def _refine_theta_map(
    sim: PLGABiphasic,
    theta0: np.ndarray,
    theta_sd: np.ndarray,
    early_times: np.ndarray,
    early_q: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    sigma = max(args.obs_sigma, 1e-4)
    reg = float(np.sqrt(max(args.map_prior_weight, 0.0)))
    scale = np.where(theta_sd < 1e-6, 1.0, theta_sd)

    def residual(theta: np.ndarray) -> np.ndarray:
        q = sim.simulate_numpy(theta.astype(float), early_times)
        data_resid = (q - early_q.astype(float)) / sigma
        if reg <= 0:
            return data_resid
        prior_resid = reg * (theta - theta0) / scale
        return np.concatenate([data_resid, prior_resid])

    result = least_squares(
        residual,
        x0=theta0.astype(float),
        bounds=(lows.astype(float), highs.astype(float)),
        max_nfev=args.map_refine_nfev,
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
    )
    theta = _clip_theta(result.x.astype(np.float32), lows, highs)
    early_rmse = _rmse(early_q.astype(float), sim.simulate_numpy(theta.astype(float), early_times))
    return theta, early_rmse


def _proposal_candidates(
    x_train: np.ndarray,
    theta_train: np.ndarray,
    x_test: np.ndarray,
    centers: np.ndarray,
    theta_sd: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    x_mu = x_train.mean(axis=0, keepdims=True)
    x_sd = np.where(x_train.std(axis=0, keepdims=True) < 1e-6, 1.0, x_train.std(axis=0, keepdims=True))
    d = np.sum(((x_train - x_mu) / x_sd - (x_test.reshape(1, -1) - x_mu) / x_sd) ** 2, axis=1)
    bank_idx = np.argsort(d)[: min(args.max_train_bank_particles, len(theta_train))]
    parts = [theta_train[bank_idx]]
    n_centers = len(centers)
    per_center = max(1, args.jitter_particles // max(n_centers, 1))
    for center in centers:
        jitter = rng.normal(0.0, args.jitter_scale, size=(per_center, center.shape[0])).astype(np.float32)
        parts.append(center.reshape(1, -1) + jitter * theta_sd.reshape(1, -1))
    return _clip_theta(np.concatenate(parts, axis=0), lows, highs)


def _eval_predictions(
    dataset: str,
    scheme: str,
    fold: int,
    pack: DataPack,
    te: np.ndarray,
    grid_times: np.ndarray,
    pred_by_method: dict[str, np.ndarray],
    extra_by_method: dict[str, list[dict[str, float]]],
    min_future_obs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    early_max = float(np.max(pack.early_times))
    for local_i, j in enumerate(te):
        fid = int(pack.fids[j])
        curve = pack.curve_map[fid]
        for method, pred_grid in pred_by_method.items():
            q_hat = _reconstruct_at_obs(
                pred_grid_q=pred_grid[local_i],
                grid_times=grid_times,
                early_q=pack.early_q[j].astype(float),
                early_times=pack.early_times,
                t_obs=curve.t_obs,
            )
            future_r2, future_rmse = _future_metrics(
                curve.t_obs,
                curve.q_obs,
                q_hat,
                early_max=early_max,
                min_future_obs=min_future_obs,
            )
            row: dict[str, object] = {
                "dataset": dataset,
                "scheme": scheme,
                "fold": fold,
                "fid": fid,
                "method": method,
                "n_obs": int(len(curve.t_obs)),
                "n_future_obs": int(np.sum(curve.t_obs > early_max)),
                "full_r2": _r2(curve.q_obs, q_hat),
                "future_r2": future_r2,
                "future_rmse": future_rmse,
            }
            if method in extra_by_method:
                row.update(extra_by_method[method][local_i])
            rows.append(row)
    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, scheme, method), sub in per_curve.groupby(["dataset", "scheme", "method"], sort=False):
        full = sub["full_r2"].dropna().to_numpy(dtype=float)
        future = sub["future_r2"].dropna().to_numpy(dtype=float)
        frmse = sub["future_rmse"].dropna().to_numpy(dtype=float)
        ess = sub["particle_ess"].dropna().to_numpy(dtype=float) if "particle_ess" in sub else np.array([])
        rows.append({
            "dataset": dataset,
            "scheme": scheme,
            "method": method,
            "n": int(len(full)),
            "median_full_r2": float(np.median(full)),
            "p10_full_r2": float(np.percentile(full, 10)),
            "frac_full_r2_ge_0": float(np.mean(full >= 0.0)),
            "median_future_r2": float(np.median(future)) if len(future) else np.nan,
            "median_future_rmse": float(np.median(frmse)) if len(frmse) else np.nan,
            "frac_future_r2_ge_0": float(np.mean(future >= 0.0)) if len(future) else np.nan,
            "median_particle_ess": float(np.median(ess)) if len(ess) else np.nan,
        })
    return pd.DataFrame(rows)


def _run_dataset(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
) -> list[dict[str, object]]:
    sim = PLGABiphasic()
    lows, highs = _prior_bounds(sim)
    pack = _load_pack(args, dataset, mod41, mod42, mod46, sim)
    x_model = np.concatenate([pack.x_form, pack.early_q], axis=1).astype(np.float32)
    grid_times = np.unique(np.concatenate([
        np.asarray(args.early_times, dtype=float),
        np.asarray(args.grid_times, dtype=float),
    ]))
    grid_times = grid_times[grid_times <= args.t_grid_max_days]
    early_idx = np.array([int(np.where(np.isclose(grid_times, t))[0][0]) for t in pack.early_times], dtype=int)
    late_mask = grid_times > float(np.max(pack.early_times))
    y_grid = _grid_targets(pack, grid_times, mod41._interp_at)
    print(f"[65] dataset={dataset} n={len(pack.fids)} grid={len(grid_times)}", flush=True)

    rows: list[dict[str, object]] = []
    for scheme, splits in _split_schemes(len(pack.fids), pack.drug_groups, pack.polymer_groups, args.n_folds, args.seed):
        print(f"[65]   scheme={scheme}", flush=True)
        for fold, (tr, te) in enumerate(splits):
            fold_seed = args.seed + 1000 * fold + len(dataset)
            rng = np.random.default_rng(fold_seed)

            direct_early = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=fold_seed + 11,
                n_jobs=-1,
            )
            direct_early.fit(pack.early_q[tr], y_grid[tr][:, late_mask])
            direct_early_pred = np.tile(np.median(y_grid[tr], axis=0), (len(te), 1))
            direct_early_pred[:, late_mask] = np.clip(direct_early.predict(pack.early_q[te]), 0.0, 1.0)

            direct_x = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=fold_seed + 22,
                n_jobs=-1,
            )
            direct_x.fit(x_model[tr], y_grid[tr][:, late_mask])
            direct_x_pred = np.tile(np.median(y_grid[tr], axis=0), (len(te), 1))
            direct_x_pred[:, late_mask] = np.clip(direct_x.predict(x_model[te]), 0.0, 1.0)

            knn = KNeighborsRegressor(n_neighbors=min(args.n_neighbors, len(tr)), weights="distance")
            knn.fit(pack.early_q[tr], y_grid[tr][:, late_mask])
            knn_pred = np.tile(np.median(y_grid[tr], axis=0), (len(te), 1))
            knn_pred[:, late_mask] = np.clip(knn.predict(pack.early_q[te]), 0.0, 1.0)

            theta_rf = _clip_theta(
                _fit_predict_ztheta(
                    RandomForestRegressor(args.n_estimators, min_samples_leaf=2, random_state=fold_seed + 33, n_jobs=-1),
                    x_model,
                    pack.theta,
                    tr,
                    te,
                ),
                lows,
                highs,
            )
            theta_et = _clip_theta(
                _fit_predict_ztheta(
                    ExtraTreesRegressor(args.n_estimators, min_samples_leaf=2, random_state=fold_seed + 44, n_jobs=-1),
                    x_model,
                    pack.theta,
                    tr,
                    te,
                ),
                lows,
                highs,
            )
            theta_point = _clip_theta(0.5 * (theta_rf + theta_et), lows, highs)
            theta_point_pred = _simulate_grid(sim, theta_point, grid_times, args.device, args.sim_batch_size)

            trainbank_mean: list[np.ndarray] = []
            trainbank_map: list[np.ndarray] = []
            proposal_mean: list[np.ndarray] = []
            proposal_map: list[np.ndarray] = []
            refined_theta: list[np.ndarray] = []
            refined_extra: list[dict[str, float]] = []
            extra: dict[str, list[dict[str, float]]] = {"posterior_map_refined": refined_extra}
            if not args.map_only:
                extra.update({
                    "particle_trainbank_mean": [],
                    "particle_trainbank_map": [],
                    "particle_proposal_mean": [],
                    "particle_proposal_map": [],
                })
            theta_sd = np.where(pack.theta[tr].std(axis=0) < 1e-6, 1.0, pack.theta[tr].std(axis=0)).astype(np.float32)
            for local_i, j in enumerate(te):
                if not args.map_only:
                    bank_candidates = pack.theta[tr]
                    if len(bank_candidates) > args.max_train_bank_particles:
                        d = np.sum((pack.early_q[tr] - pack.early_q[j].reshape(1, -1)) ** 2, axis=1)
                        bank_candidates = bank_candidates[np.argsort(d)[: args.max_train_bank_particles]]
                    bank_candidates = _clip_theta(bank_candidates, lows, highs)
                    q_mean, q_map, stats = _particle_predict_one(
                        sim,
                        bank_candidates,
                        grid_times,
                        early_idx,
                        pack.early_q[j],
                        args,
                    )
                    trainbank_mean.append(q_mean)
                    trainbank_map.append(q_map)
                    extra["particle_trainbank_mean"].append(stats)
                    extra["particle_trainbank_map"].append(stats)

                    centers = np.stack([theta_rf[local_i], theta_et[local_i], theta_point[local_i]]).astype(np.float32)
                    candidates = _proposal_candidates(
                        x_train=x_model[tr],
                        theta_train=pack.theta[tr],
                        x_test=x_model[j],
                        centers=centers,
                        theta_sd=theta_sd,
                        lows=lows,
                        highs=highs,
                        args=args,
                        rng=rng,
                    )
                    q_mean, q_map, stats = _particle_predict_one(
                        sim,
                        candidates,
                        grid_times,
                        early_idx,
                        pack.early_q[j],
                        args,
                    )
                    proposal_mean.append(q_mean)
                    proposal_map.append(q_map)
                    extra["particle_proposal_mean"].append(stats)
                    extra["particle_proposal_map"].append(stats)

                theta_refined, refine_rmse = _refine_theta_map(
                    sim=sim,
                    theta0=theta_point[local_i],
                    theta_sd=theta_sd,
                    early_times=pack.early_times,
                    early_q=pack.early_q[j],
                    lows=lows,
                    highs=highs,
                    args=args,
                )
                refined_theta.append(theta_refined)
                refined_extra.append({"map_refined_early_rmse": refine_rmse})

            refined_pred = _simulate_grid(
                sim,
                np.asarray(refined_theta, dtype=np.float32),
                grid_times,
                args.device,
                args.sim_batch_size,
            )

            pred_by_method = {
                "KNN_direct_Q_early": knn_pred,
                "ET_direct_Q_early": direct_early_pred,
                "ET_direct_Q_formulation_plus_early": direct_x_pred,
                "ET_RF_theta_point": theta_point_pred,
                "posterior_map_refined": refined_pred,
            }
            if not args.map_only:
                pred_by_method.update({
                    "particle_trainbank_mean": np.asarray(trainbank_mean, dtype=np.float32),
                    "particle_trainbank_map": np.asarray(trainbank_map, dtype=np.float32),
                    "particle_proposal_mean": np.asarray(proposal_mean, dtype=np.float32),
                    "particle_proposal_map": np.asarray(proposal_map, dtype=np.float32),
                })
            rows.extend(_eval_predictions(
                dataset=dataset,
                scheme=scheme,
                fold=fold,
                pack=pack,
                te=te,
                grid_times=grid_times,
                pred_by_method=pred_by_method,
                extra_by_method=extra,
                min_future_obs=args.min_future_obs,
            ))
    return rows


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
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--sim-batch-size", type=int, default=256)
    ap.add_argument("--obs-sigma", type=float, default=0.035)
    ap.add_argument("--max-train-bank-particles", type=int, default=96)
    ap.add_argument("--jitter-particles", type=int, default=192)
    ap.add_argument("--jitter-scale", type=float, default=0.18)
    ap.add_argument("--map-prior-weight", type=float, default=0.25)
    ap.add_argument("--map-refine-nfev", type=int, default=60)
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/65_mechanistic_particle_world_model"))
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_for65")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_for65")
    mod46 = _load_script(scripts_dir / "46_direct_curve_rf_input_ablation.py", "script46_for65")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        rows.extend(_run_dataset(args, dataset, mod41, mod42, mod46))

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)

    best = summary.loc[summary.groupby(["dataset", "scheme"])["median_future_rmse"].idxmin()].copy()
    lines = [
        "=== 65 -- mechanistic particle world model ===",
        "",
        f"datasets        : {', '.join(datasets)}",
        f"obs_sigma       : {args.obs_sigma}",
        f"train_particles : {args.max_train_bank_particles}",
        f"jitter_particles: {args.jitter_particles}",
        "",
        "--- best method by future RMSE per dataset/scheme ---",
    ]
    for _, row in best.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['method']:<36} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"future_RMSE={row['median_future_rmse']:.4f}"
        )
    lines.extend(["", "--- mechanism posterior rows ---"])
    particle = summary[
        summary["method"].str.startswith("particle") | summary["method"].eq("posterior_map_refined")
    ].sort_values(["dataset", "scheme", "method"])
    for _, row in particle.iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['method']:<24} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"future_RMSE={row['median_future_rmse']:.4f} "
            f"ESS={row['median_particle_ess']:.1f}"
        )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  This is a mechanism-conditioned posterior particle rollout, not direct-Q curve regression.",
        "  A positive result supports the middle-layer/world-model framing; a negative result diagnoses theta-posterior weakness.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
