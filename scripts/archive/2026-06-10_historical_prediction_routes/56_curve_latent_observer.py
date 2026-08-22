"""
56 - Curve-latent observer diagnostic.

Question:
    Can a model "learn release curves themselves" before seeing
    formulation descriptors or theta?

This is the smallest non-neural version of that idea:

    train full curves -> low-dimensional PCA curve manifold
    test early Q      -> fit latent coordinates on early points only
    latent            -> reconstruct the future curve

Full test curves are evaluation-only. The PCA basis, KNN analog bank,
and direct ET baseline are fit inside each train fold.

Outputs:
    outputs/56_curve_latent_observer/per_curve.csv
    outputs/56_curve_latent_observer/scheme_method_summary.csv
    outputs/56_curve_latent_observer/summary.txt
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
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors


DATASETS = ("cross321", "internal181")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    schemes: list[tuple[str, np.ndarray | None]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_groups),
        ("group_by_polymer", polymer_groups),
    ]
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    for name, groups in schemes:
        if groups is None:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            out.append((name, list(splitter.split(np.arange(n)))))
            continue
        n_unique = pd.Series(groups).nunique()
        n_splits = min(n_folds, int(n_unique))
        if n_splits < 2:
            continue
        splitter = GroupKFold(n_splits=n_splits)
        out.append((name, list(splitter.split(np.arange(n), groups=groups))))
    return out


def _load_dataset(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
) -> tuple[np.ndarray, np.ndarray, dict[int, object], np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "cross321":
        fids, _x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = (
            mod46._load_cross321(args, mod41)
        )
    else:
        fids, _x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = (
            mod46._load_internal181(args, mod42)
        )
    return fids, early_q, curve_map, drug_groups, polymer_groups, early_times


def _grid_targets(
    fids: np.ndarray,
    curve_map: dict[int, object],
    grid_times: np.ndarray,
    interp_fn: object,
) -> np.ndarray:
    return np.stack([
        interp_fn(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, grid_times)
        for fid in fids
    ]).astype(np.float32)


def _late_grid(grid_times: np.ndarray, early_times: np.ndarray) -> np.ndarray:
    return grid_times[grid_times > float(np.max(early_times))]


def _early_indices(grid_times: np.ndarray, early_times: np.ndarray) -> np.ndarray:
    idx: list[int] = []
    for t in early_times:
        hit = np.where(np.isclose(grid_times, t))[0]
        if len(hit) != 1:
            raise ValueError(f"early time {t} must appear once in grid_times")
        idx.append(int(hit[0]))
    return np.asarray(idx, dtype=int)


def _reconstruct_at_obs(
    pred_grid_q: np.ndarray,
    grid_times: np.ndarray,
    early_q: np.ndarray,
    early_times: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    late_mask = grid_times > float(np.max(early_times))
    t_combined = np.concatenate([early_times, grid_times[late_mask]])
    q_combined = np.concatenate([early_q, pred_grid_q[late_mask]])
    order = np.argsort(t_combined)
    q_hat = np.interp(
        t_obs,
        t_combined[order],
        q_combined[order],
        left=q_combined[order][0],
        right=q_combined[order][-1],
    )
    return np.clip(q_hat, 0.0, 1.0)


def _fit_pca_from_early(
    pca: PCA,
    early_q: np.ndarray,
    early_idx: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    mean_early = pca.mean_[early_idx]
    basis_early = pca.components_[:, early_idx].T
    lhs = basis_early.T @ basis_early
    lhs = lhs + ridge_alpha * np.eye(lhs.shape[0])
    rhs = basis_early.T @ (early_q - mean_early)
    try:
        z = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return np.clip(pca.mean_ + z @ pca.components_, 0.0, 1.0)


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, scheme, method), sub in per_curve.groupby(["dataset", "scheme", "method"], sort=False):
        full = sub["full_r2"].dropna().to_numpy(dtype=float)
        future = sub["future_r2"].dropna().to_numpy(dtype=float)
        rows.append({
            "dataset": dataset,
            "scheme": scheme,
            "method": method,
            "n": int(len(full)),
            "median_full_r2": float(np.median(full)),
            "mean_full_r2": float(np.mean(full)),
            "p10_full_r2": float(np.percentile(full, 10)),
            "frac_full_r2_ge_0": float(np.mean(full >= 0.0)),
            "median_future_r2": float(np.median(future)) if len(future) else np.nan,
            "mean_early_rmse": float(sub["early_rmse"].mean()),
        })
    return pd.DataFrame(rows)


def _run_dataset(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
) -> list[dict[str, object]]:
    fids, early_q, curve_map, drug_groups, polymer_groups, early_times = _load_dataset(
        args, dataset, mod41, mod42, mod46
    )
    grid_times = np.unique(np.concatenate([np.asarray(args.early_times, dtype=float), np.asarray(args.grid_times, dtype=float)]))
    grid_times = grid_times[grid_times <= args.t_grid_max_days]
    late_times = _late_grid(grid_times, early_times)
    early_idx = _early_indices(grid_times, early_times)
    y_grid = _grid_targets(fids, curve_map, grid_times, mod41._interp_at)

    print(
        f"[56] dataset={dataset} n={len(fids)} grid={len(grid_times)} "
        f"early={list(early_times)} late={len(late_times)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for scheme, splits in _split_schemes(len(fids), drug_groups, polymer_groups, args.n_folds, args.seed):
        print(f"[56]   scheme={scheme}", flush=True)
        for fold, (tr, te) in enumerate(splits):
            train_median = np.median(y_grid[tr], axis=0)

            knn = NearestNeighbors(n_neighbors=min(args.n_neighbors, len(tr)))
            knn.fit(early_q[tr])
            knn_dist, knn_idx = knn.kneighbors(early_q[te], return_distance=True)
            knn_weights = 1.0 / np.maximum(knn_dist, 1e-6)
            knn_weights = knn_weights / knn_weights.sum(axis=1, keepdims=True)
            knn_pred = np.einsum("ij,ijk->ik", knn_weights, y_grid[tr][knn_idx])

            et = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=args.seed + 1000 + fold,
                n_jobs=-1,
            )
            et.fit(early_q[tr], y_grid[tr][:, grid_times > float(np.max(early_times))])
            et_late = np.clip(et.predict(early_q[te]), 0.0, 1.0)
            et_pred = np.tile(train_median, (len(te), 1))
            et_pred[:, grid_times > float(np.max(early_times))] = et_late

            method_pred: dict[str, np.ndarray] = {
                "train_median_curve": np.tile(train_median, (len(te), 1)),
                "knn_curve_analog": knn_pred,
                "ET_earlyQ_to_lateQ": et_pred,
            }

            for n_comp in args.pca_components:
                k = min(int(n_comp), len(tr) - 1, y_grid.shape[1] - 1)
                if k < 1:
                    continue
                pca = PCA(n_components=k, random_state=args.seed + fold)
                pca.fit(y_grid[tr])
                pred = np.stack([
                    _fit_pca_from_early(pca, early_q[j], early_idx, args.pca_ridge)
                    for j in te
                ]).astype(np.float32)
                method_pred[f"pca_curve_latent_k{k}"] = pred

            for local_i, j in enumerate(te):
                fid = int(fids[j])
                curve = curve_map[fid]
                future = curve.t_obs > float(np.max(early_times))
                for method, pred_grid_q in method_pred.items():
                    q_hat = _reconstruct_at_obs(
                        pred_grid_q=pred_grid_q[local_i],
                        grid_times=grid_times,
                        early_q=early_q[j].astype(float),
                        early_times=early_times,
                        t_obs=curve.t_obs,
                    )
                    rows.append({
                        "dataset": dataset,
                        "scheme": scheme,
                        "fold": fold,
                        "fid": fid,
                        "method": method,
                        "n_obs": int(len(curve.t_obs)),
                        "full_r2": _r2(curve.q_obs, q_hat),
                        "future_r2": _r2(curve.q_obs[future], q_hat[future]) if np.sum(future) >= 2 else np.nan,
                        "early_rmse": _rmse(early_q[j].astype(float), np.interp(early_times, curve.t_obs, q_hat)),
                    })
    return rows


def main() -> None:
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
        "--grid-times",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0],
    )
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--pca-components", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--pca-ridge", type=float, default=1e-3)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/56_curve_latent_observer"))
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta_for56")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal_for56")
    mod46 = _load_script(scripts_dir / "46_direct_curve_rf_input_ablation.py", "script46_direct_for56")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        rows.extend(_run_dataset(args, dataset, mod41, mod42, mod46))

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)

    best = summary.loc[summary.groupby(["dataset", "scheme"])["median_full_r2"].idxmax()].copy()
    lines = [
        "=== 56 -- curve-latent observer diagnostic ===",
        "",
        f"datasets       : {', '.join(datasets)}",
        f"early_times    : {args.early_times}",
        f"pca_components : {args.pca_components}",
        "",
        "--- best method per dataset/scheme ---",
    ]
    for _, row in best.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} "
            f"{row['method']:<24} full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} frac>=0={row['frac_full_r2_ge_0']:.2f}"
        )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  PCA/KNN methods learn only from train-fold full curves.",
        "  Test full curves are used only for final evaluation.",
        "  This is a curve-prior diagnostic, not yet a trained neural world model.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
