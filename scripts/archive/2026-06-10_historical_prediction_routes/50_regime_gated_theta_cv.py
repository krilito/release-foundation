"""
50 - Cross-validated regime-gated theta audit.

What this does:
    Script 49 showed that a fast/slow gate can rescue the two NC wet
    stress-test curves when paired with branch-specific theta mappers
    and early-only local refinement. This script asks the obvious
    Karpathy question before we promote that idea:

        Does regime gating help under normal CV, or was 49 just a wet
        two-curve patch?

    For each fold:
        1. Estimate t50 labels on train curves only.
        2. Train a fast/slow classifier from formulation + early Q.
        3. Train global and branch-specific theta mappers.
        4. Decode all theta predictions through PLGABiphasic.
        5. Optionally refine selected theta using test early Q only.

    Full test curves are evaluation-only.

Outputs:
    outputs/50_regime_gated_theta_cv/per_curve.csv
    outputs/50_regime_gated_theta_cv/scheme_method_summary.csv
    outputs/50_regime_gated_theta_cv/summary.txt
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
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import PLGA_CONTINUOUS_COLS  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


DATASETS = ("cross321", "internal181")
FID_COL = "fid"
INTERNAL_FID_COL = "Experimental_index"


@dataclass
class DatasetPack:
    dataset: str
    fids: np.ndarray
    x_form: np.ndarray
    early_q: np.ndarray
    theta: np.ndarray
    curve_map: dict[int, object]
    drug_groups: np.ndarray
    polymer_groups: np.ndarray
    early_times: np.ndarray


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


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(float), lows + eps, highs - eps)


def _fit_predict_ztheta(
    model: object,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    mu = theta_train.mean(axis=0, keepdims=True)
    sd = theta_train.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x_train, (theta_train - mu) / sd)
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
    early_times: np.ndarray,
    early_q: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, int]:
    if max_nfev <= 0:
        return theta0, 0
    x0 = _clip_theta(theta0, lows, highs)
    try:
        res = least_squares(
            lambda th: sim.simulate_numpy(th, early_times) - early_q,
            x0=x0,
            bounds=(lows, highs),
            method="trf",
            max_nfev=max_nfev,
        )
    except Exception:
        return x0, 0
    return _clip_theta(res.x, lows, highs), int(res.nfev)


def _load_cross321(args: argparse.Namespace, mod41: ModuleType) -> DatasetPack:
    df, curve_map, x, theta, _param_names, drug_groups, polymer_groups = mod41._load_dataset(args)
    if args.max_curves is not None:
        n_keep = min(int(args.max_curves), len(df))
        df = df.iloc[:n_keep].reset_index(drop=True)
        x = x[:n_keep]
        theta = theta[:n_keep]
        drug_groups = drug_groups[:n_keep]
        polymer_groups = polymer_groups[:n_keep]
    early_times = np.asarray(args.early_times, dtype=float)
    early_q = x[:, -len(early_times):]
    x_form = x[:, :-len(early_times)]
    fids = df[FID_COL].to_numpy(dtype=int)
    return DatasetPack(
        dataset="cross321",
        fids=fids,
        x_form=x_form.astype(np.float32),
        early_q=early_q.astype(np.float32),
        theta=theta.astype(np.float32),
        curve_map=curve_map,
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        early_times=early_times,
    )


def _load_internal181(args: argparse.Namespace, mod42: ModuleType) -> DatasetPack:
    desc, curve_map, _inventory = mod42._load_clean_internal(
        data_csv=args.data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )
    if args.max_curves is not None:
        desc = desc.iloc[: int(args.max_curves)].reset_index(drop=True)
    fids = desc[INTERNAL_FID_COL].to_numpy(dtype=int)
    early_times = np.asarray(args.early_times, dtype=float)
    early_q = np.stack([
        _interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, early_times)
        for fid in fids
    ]).astype(np.float32)
    x_form = desc[list(PLGA_CONTINUOUS_COLS)].to_numpy(dtype=np.float32)
    theta = desc[list(PLGABiphasic().param_names)].to_numpy(dtype=np.float32)
    drug_groups = np.array([curve_map[int(fid)].drug_id for fid in fids], dtype=object)
    polymer_groups = np.array([curve_map[int(fid)].polymer_family for fid in fids], dtype=object)
    return DatasetPack(
        dataset="internal181",
        fids=fids,
        x_form=x_form,
        early_q=early_q,
        theta=theta,
        curve_map=curve_map,
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        early_times=early_times,
    )


def _splits_for_dataset(pack: DatasetPack, args: argparse.Namespace) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    schemes: list[tuple[str, np.ndarray | None]] = [
        ("random_5fold", None),
        ("group_by_drug", pack.drug_groups),
        ("group_by_polymer", pack.polymer_groups),
    ]
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    n = len(pack.fids)
    for scheme_name, groups in schemes:
        if groups is None:
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(n)))
        else:
            n_unique = pd.Series(groups).nunique()
            n_splits = min(args.n_folds, int(n_unique))
            if n_splits < 2:
                continue
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.arange(n), groups=groups))
        out.append((scheme_name, splits))
    return out


def _branch_predictions(
    args: argparse.Namespace,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    y_fast: np.ndarray,
    p_fast_test: np.ndarray,
    x_test: np.ndarray,
    global_preds: dict[str, np.ndarray],
    fold_seed: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for prefix, maker in [
        ("RF", lambda seed: _make_rf(args, seed, min_samples_leaf=2)),
        ("ET", lambda seed: _make_et(args, seed, min_samples_leaf=2)),
    ]:
        if int(np.sum(y_fast == 1)) < args.min_branch_samples or int(np.sum(y_fast == 0)) < args.min_branch_samples:
            pred_fast = global_preds[f"{prefix}_global_ztheta"]
            pred_slow = global_preds[f"{prefix}_global_ztheta"]
        else:
            pred_fast = _fit_predict_ztheta(
                maker(fold_seed + 100),
                x_train[y_fast == 1],
                theta_train[y_fast == 1],
                x_test,
            )
            pred_slow = _fit_predict_ztheta(
                maker(fold_seed + 200),
                x_train[y_fast == 0],
                theta_train[y_fast == 0],
                x_test,
            )
        p = p_fast_test.reshape(-1, 1)
        out[f"{prefix}_regime_soft"] = p * pred_fast + (1.0 - p) * pred_slow
        out[f"{prefix}_regime_hard"] = np.where((p_fast_test >= 0.5).reshape(-1, 1), pred_fast, pred_slow)
    out["RF_ET_regime_soft_avg"] = 0.5 * (out["RF_regime_soft"] + out["ET_regime_soft"])
    out["RF_ET_regime_hard_avg"] = 0.5 * (out["RF_regime_hard"] + out["ET_regime_hard"])
    return out


def _decode_rows(
    args: argparse.Namespace,
    sim: PLGABiphasic,
    pack: DatasetPack,
    scheme: str,
    fold: int,
    te: np.ndarray,
    theta_pred: dict[str, np.ndarray],
    p_fast: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    method_names = list(theta_pred.keys())
    for local_i, j in enumerate(te):
        fid = int(pack.fids[j])
        curve = pack.curve_map[fid]
        early_rmse: dict[str, float] = {}
        decoded: dict[str, np.ndarray] = {}
        for method in method_names:
            theta = _clip_theta(theta_pred[method][local_i], lows, highs)
            decoded[method] = theta
            pred_early = sim.simulate_numpy(theta, pack.early_times)
            early_rmse[method] = _rmse(pack.early_q[j].astype(float), pred_early)
            pred = sim.simulate_numpy(theta, curve.t_obs)
            rows.append({
                "dataset": pack.dataset,
                "scheme": scheme,
                "fold": fold,
                "fid": fid,
                "method": method,
                "r2": _r2(curve.q_obs, pred),
                "early_rmse": early_rmse[method],
                "gate_p_fast": float(p_fast[local_i]),
                "nfev": 0,
                "n_obs": int(len(curve.t_obs)),
            })

        select_pool = [m for m in ["RF_ET_global_zavg", "RF_ET_regime_soft_avg", "RF_ET_regime_hard_avg"] if m in decoded]
        selected = min(select_pool, key=lambda m: early_rmse[m])
        theta = decoded[selected]
        pred = sim.simulate_numpy(theta, curve.t_obs)
        rows.append({
            "dataset": pack.dataset,
            "scheme": scheme,
            "fold": fold,
            "fid": fid,
            "method": "early_select",
            "r2": _r2(curve.q_obs, pred),
            "early_rmse": early_rmse[selected],
            "gate_p_fast": float(p_fast[local_i]),
            "nfev": 0,
            "n_obs": int(len(curve.t_obs)),
        })

        if args.refine_nfev > 0:
            for method in ["RF_ET_global_zavg", "RF_ET_regime_hard_avg", "early_select"]:
                theta0 = theta if method == "early_select" else decoded[method]
                refined, nfev = _local_refine_early(
                    sim=sim,
                    theta0=theta0,
                    early_times=pack.early_times,
                    early_q=pack.early_q[j].astype(float),
                    lows=lows,
                    highs=highs,
                    max_nfev=args.refine_nfev,
                )
                pred = sim.simulate_numpy(refined, curve.t_obs)
                pred_early = sim.simulate_numpy(refined, pack.early_times)
                rows.append({
                    "dataset": pack.dataset,
                    "scheme": scheme,
                    "fold": fold,
                    "fid": fid,
                    "method": f"{method}_early_refined",
                    "r2": _r2(curve.q_obs, pred),
                    "early_rmse": _rmse(pack.early_q[j].astype(float), pred_early),
                    "gate_p_fast": float(p_fast[local_i]),
                    "nfev": nfev,
                    "n_obs": int(len(curve.t_obs)),
                })
    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, scheme, method), sub in per_curve.groupby(["dataset", "scheme", "method"], sort=False):
        vals = sub["r2"].dropna().to_numpy(dtype=float)
        rows.append({
            "dataset": dataset,
            "scheme": scheme,
            "method": method,
            "n": int(len(vals)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p10": float(np.percentile(vals, 10)),
            "frac_above_0.9": float(np.mean(vals >= 0.9)),
            "frac_above_0.5": float(np.mean(vals >= 0.5)),
            "frac_above_0": float(np.mean(vals >= 0.0)),
            "median_early_rmse": float(np.median(sub["early_rmse"].to_numpy(dtype=float))),
            "median_nfev": float(np.median(sub["nfev"].to_numpy(dtype=float))),
            "median_gate_p_fast": float(np.median(sub["gate_p_fast"].to_numpy(dtype=float))),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "scheme", "median"], ascending=[True, True, False])


def _run_pack(args: argparse.Namespace, sim: PLGABiphasic, pack: DatasetPack, lows: np.ndarray, highs: np.ndarray) -> list[dict[str, object]]:
    x = np.concatenate([pack.x_form, pack.early_q], axis=1).astype(np.float32)
    t50_all = np.asarray([
        _estimate_t_at_release(pack.curve_map[int(fid)].t_obs, pack.curve_map[int(fid)].q_obs, target=0.5)
        for fid in pack.fids
    ], dtype=float)
    rows: list[dict[str, object]] = []
    print(f"[50] dataset={pack.dataset} n={len(pack.fids)} x_dim={x.shape[1]}", flush=True)
    for scheme_name, splits in _splits_for_dataset(pack, args):
        print(f"[50]   scheme={scheme_name}", flush=True)
        for fold_idx, (tr, te) in enumerate(splits):
            fold_seed = args.seed + 1000 * fold_idx
            threshold = float(np.nanmedian(t50_all[tr]))
            y_fast = (t50_all[tr] <= threshold).astype(int)
            if len(np.unique(y_fast)) < 2:
                p_fast = np.full(len(te), float(y_fast[0]), dtype=float)
            else:
                gate = ExtraTreesClassifier(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=2,
                    random_state=fold_seed + 300,
                    n_jobs=-1,
                )
                gate.fit(x[tr], y_fast)
                p_fast = gate.predict_proba(x[te])[:, list(gate.classes_).index(1)]

            global_preds: dict[str, np.ndarray] = {}
            global_preds["RF_global_ztheta"] = _fit_predict_ztheta(
                _make_rf(args, fold_seed + 10, min_samples_leaf=2), x[tr], pack.theta[tr], x[te]
            )
            global_preds["ET_global_ztheta"] = _fit_predict_ztheta(
                _make_et(args, fold_seed + 20, min_samples_leaf=2), x[tr], pack.theta[tr], x[te]
            )
            global_preds["RF_ET_global_zavg"] = 0.5 * (
                global_preds["RF_global_ztheta"] + global_preds["ET_global_ztheta"]
            )
            regime_preds = _branch_predictions(
                args=args,
                x_train=x[tr],
                theta_train=pack.theta[tr],
                y_fast=y_fast,
                p_fast_test=p_fast,
                x_test=x[te],
                global_preds=global_preds,
                fold_seed=fold_seed,
            )
            rows.extend(
                _decode_rows(
                    args=args,
                    sim=sim,
                    pack=pack,
                    scheme=scheme_name,
                    fold=fold_idx,
                    te=te,
                    theta_pred={**global_preds, **regime_preds},
                    p_fast=p_fast,
                    lows=lows,
                    highs=highs,
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-csv",
        type=Path,
        default=Path("outputs/33_active_set_regimes/regime_assignments.csv"),
        help="Compatibility argument consumed by script 41's dataset loader.",
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
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-branch-samples", type=int, default=12)
    ap.add_argument("--refine-nfev", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/50_regime_gated_theta_cv"))
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal")

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)

    packs: list[DatasetPack] = []
    if args.dataset in ("cross321", "all"):
        packs.append(_load_cross321(args, mod41))
    if args.dataset in ("internal181", "all"):
        packs.append(_load_internal181(args, mod42))

    rows: list[dict[str, object]] = []
    for pack in packs:
        rows.extend(_run_pack(args, sim, pack, lows, highs))

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)

    lines = [
        "=== 50 -- regime-gated theta CV audit ===",
        "",
        f"datasets       : {', '.join([p.dataset for p in packs])}",
        f"early_times    : {list(np.asarray(args.early_times, dtype=float))}",
        f"n_estimators   : {args.n_estimators}",
        f"refine_nfev    : {args.refine_nfev}",
        "",
        "--- best methods by dataset x scheme ---",
    ]
    for dataset in [p.dataset for p in packs]:
        for scheme in ["random_5fold", "group_by_drug", "group_by_polymer"]:
            sub = summary[(summary["dataset"] == dataset) & (summary["scheme"] == scheme)]
            if sub.empty:
                continue
            lines.append(f"  {dataset} / {scheme}:")
            for _, row in sub.head(8).iterrows():
                lines.append(
                    f"    {row['method']:<34} median={row['median']:+.4f} "
                    f"p10={row['p10']:+.4f} frac_R2>=0={row['frac_above_0']:.3f} "
                    f"early_RMSE={row['median_early_rmse']:.4f} nfev={row['median_nfev']:.1f}"
                )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  Regime labels are estimated from train-fold t50 only. Test full curves",
        "  are used only for final R2. Early refinement, when enabled, uses only",
        "  the requested early release observations.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
