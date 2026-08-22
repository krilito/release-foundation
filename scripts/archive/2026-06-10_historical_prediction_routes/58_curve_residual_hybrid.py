"""
58 - Tree + MLP residual hybrid for early-Q -> future-Q.

Question:
    Can a tree model provide the stable coarse curve forecast while a
    small MLP learns only the residual correction?

Leakage guardrail:
    The residual MLP target is built from inner-fold out-of-fold base
    predictions inside each outer train fold. Full outer-test curves are
    evaluation-only.

Outputs:
    outputs/58_curve_residual_hybrid/per_curve.csv
    outputs/58_curve_residual_hybrid/scheme_method_summary.csv
    outputs/58_curve_residual_hybrid/summary.txt
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
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def _split_schemes(
    n: int,
    drug_groups: np.ndarray,
    polymer_groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out.append(("random_5fold", list(splitter.split(np.arange(n)))))
    for name, groups in [("group_by_drug", drug_groups), ("group_by_polymer", polymer_groups)]:
        n_splits = min(n_folds, int(pd.Series(groups).nunique()))
        if n_splits < 2:
            continue
        splitter = GroupKFold(n_splits=n_splits)
        out.append((name, list(splitter.split(np.arange(n), groups=groups))))
    return out


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


def _reconstruct_at_obs(
    late_q: np.ndarray,
    late_times: np.ndarray,
    early_q: np.ndarray,
    early_times: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    t = np.concatenate([early_times, late_times])
    q = np.concatenate([early_q, late_q])
    order = np.argsort(t)
    q_hat = np.interp(t_obs, t[order], q[order], left=q[order][0], right=q[order][-1])
    return np.clip(q_hat, 0.0, 1.0)


def _make_mlp(seed: int, hidden: tuple[int, ...], alpha: float, max_iter: int) -> object:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=seed,
        ),
    )


def _oof_predict(model: object, x: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    n_splits = min(n_folds, len(x))
    if n_splits < 2:
        fitted = clone(model)
        fitted.fit(x, y)
        return np.clip(fitted.predict(x), 0.0, 1.0)
    pred = np.zeros_like(y, dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in splitter.split(np.arange(len(x))):
        fitted = clone(model)
        fitted.fit(x[tr], y[tr])
        pred[va] = fitted.predict(x[va])
    return np.clip(pred, 0.0, 1.0)


def _fit_residual_stack(
    base_model: object,
    residual_model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof_base = _oof_predict(base_model, x_train, y_train, inner_folds, seed)
    residual_x = np.concatenate([x_train, oof_base], axis=1)
    residual_y = y_train - oof_base
    residual_model.fit(residual_x, residual_y)

    final_base = clone(base_model)
    final_base.fit(x_train, y_train)
    base_test = np.clip(final_base.predict(x_test), 0.0, 1.0)
    residual_test = residual_model.predict(np.concatenate([x_test, base_test], axis=1))
    hybrid_test = np.clip(base_test + residual_test, 0.0, 1.0)
    return base_test, hybrid_test


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
            "p10_full_r2": float(np.percentile(full, 10)),
            "frac_full_r2_ge_0": float(np.mean(full >= 0.0)),
            "median_future_r2": float(np.median(future)) if len(future) else np.nan,
            "mean_future_r2": float(np.mean(future)) if len(future) else np.nan,
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
    late_times = np.asarray(args.late_grid, dtype=float)
    target_late = _grid_targets(fids, curve_map, late_times, mod41._interp_at)
    print(
        f"[58] dataset={dataset} n={len(fids)} early={list(early_times)} late={list(late_times)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for scheme, splits in _split_schemes(len(fids), drug_groups, polymer_groups, args.n_folds, args.seed):
        print(f"[58]   scheme={scheme}", flush=True)
        for fold, (tr, te) in enumerate(splits):
            base_models: dict[str, object] = {
                "RF_base": RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=2,
                    random_state=args.seed + 1000 + fold,
                    n_jobs=-1,
                ),
                "ET_base": ExtraTreesRegressor(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=2,
                    random_state=args.seed + 2000 + fold,
                    n_jobs=-1,
                ),
            }
            pred_by_method: dict[str, np.ndarray] = {}
            for base_name, base_model in base_models.items():
                residual_model = _make_mlp(
                    seed=args.seed + 3000 + fold,
                    hidden=(64, 64),
                    alpha=args.alpha,
                    max_iter=args.max_iter,
                )
                base_pred, hybrid_pred = _fit_residual_stack(
                    base_model=base_model,
                    residual_model=residual_model,
                    x_train=early_q[tr],
                    y_train=target_late[tr],
                    x_test=early_q[te],
                    inner_folds=args.inner_folds,
                    seed=args.seed + 4000 + fold,
                )
                pred_by_method[base_name] = base_pred
                pred_by_method[f"{base_name}_plus_MLP_residual"] = hybrid_pred

            avg_base = 0.5 * (pred_by_method["RF_base"] + pred_by_method["ET_base"])
            avg_hybrid = 0.5 * (
                pred_by_method["RF_base_plus_MLP_residual"]
                + pred_by_method["ET_base_plus_MLP_residual"]
            )
            pred_by_method["RF_ET_base_avg"] = avg_base
            pred_by_method["RF_ET_residual_avg"] = avg_hybrid

            for local_i, j in enumerate(te):
                fid = int(fids[j])
                curve = curve_map[fid]
                future = curve.t_obs > float(np.max(early_times))
                for method, pred_late in pred_by_method.items():
                    q_hat = _reconstruct_at_obs(
                        late_q=pred_late[local_i],
                        late_times=late_times,
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
        "--late-grid",
        nargs="+",
        type=float,
        default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0],
    )
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/58_curve_residual_hybrid"))
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta_for58")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal_for58")
    mod46 = _load_script(scripts_dir / "46_direct_curve_rf_input_ablation.py", "script46_direct_for58")

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
        "=== 58 -- tree + MLP residual hybrid ===",
        "",
        f"datasets    : {', '.join(datasets)}",
        f"early_times : {args.early_times}",
        f"late_grid   : {args.late_grid}",
        f"inner_folds : {args.inner_folds}",
        "",
        "--- best method per dataset/scheme ---",
    ]
    for _, row in best.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['method']:<28} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"frac>=0={row['frac_full_r2_ge_0']:.2f}"
        )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  Residual targets are created by inner train-fold OOF base predictions.",
        "  Test full curves are evaluation-only.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
