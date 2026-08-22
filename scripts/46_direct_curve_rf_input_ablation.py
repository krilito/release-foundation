"""
46 - Direct RF/ET curve prediction without the theta bottleneck.

What this does:
    Scripts 41/42 test the tree-ensemble -> theta -> ODE route.
    Script 45 showed that early release observations carry most of the
    predictive signal. This audit asks the direct question:

        do we still need the theta middle layer, or can pure tabular ML
        map the same inputs directly to the release curve?

    For a clean comparison, the split schemes and input modes match 41/42:

        - random_5fold
        - group_by_drug
        - group_by_polymer

        - formulation_only
        - early_only
        - formulation_plus_early

    Direct curve target rule:
        - If early Q is an input, RF/ET predicts late Q only and the
          observed early Q points are used as interpolation anchors.
        - If formulation_only is used, RF/ET predicts early + late Q.
          No observed early Q is leaked into reconstruction.

Outputs:
    outputs/46_direct_curve_rf_input_ablation/per_curve.csv
    outputs/46_direct_curve_rf_input_ablation/scheme_method_summary.csv
    outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv
    outputs/46_direct_curve_rf_input_ablation/summary.txt
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
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import PLGA_CONTINUOUS_COLS  # noqa: E402


INPUT_MODES = ("formulation_only", "early_only", "formulation_plus_early")
DATASETS = ("cross321", "internal181")


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


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, input_mode, scheme), sub in per_curve.groupby(
        ["dataset", "input_mode", "scheme"], sort=False
    ):
        for col in [c for c in per_curve.columns if c.startswith("r2_")]:
            vals = sub[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
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


def _make_methods(args: argparse.Namespace, seed: int) -> dict[str, object]:
    return {
        "RF_direct_Q": RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=None,
            random_state=seed,
            n_jobs=-1,
        ),
        "RF_direct_Q_leaf2": RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed + 1000,
            n_jobs=-1,
        ),
        "ET_direct_Q": ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=None,
            random_state=seed + 2000,
            n_jobs=-1,
        ),
    }


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
    schemes: list[tuple[str, np.ndarray | None]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_groups),
        ("group_by_polymer", polymer_groups),
    ]
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    for scheme_name, groups in schemes:
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


def _load_cross321(args: argparse.Namespace, mod41: ModuleType) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, object],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    df, curve_map, x, _theta, _param_names, drug_groups, polymer_groups = mod41._load_dataset(args)
    if args.max_curves is not None:
        n_keep = min(int(args.max_curves), len(df))
        df = df.iloc[:n_keep].reset_index(drop=True)
        x = x[:n_keep]
        drug_groups = drug_groups[:n_keep]
        polymer_groups = polymer_groups[:n_keep]
    early_times = np.array(args.early_times, dtype=float)
    early_q = x[:, -len(early_times):]
    x_form = x[:, :-len(early_times)]
    fids = df[mod41.FID_COL].to_numpy(dtype=int)
    return fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times


def _load_internal181(args: argparse.Namespace, mod42: ModuleType) -> tuple[
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
    drug_groups = np.array([curve_map[int(fid)].drug_id for fid in fids], dtype=object)
    polymer_groups = np.array([curve_map[int(fid)].polymer_family for fid in fids], dtype=object)
    return fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times


def _run_dataset_mode(
    args: argparse.Namespace,
    dataset: str,
    input_mode: str,
    mod41: ModuleType,
    mod42: ModuleType,
) -> list[dict[str, object]]:
    if dataset == "cross321":
        fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = _load_cross321(args, mod41)
    else:
        fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = _load_internal181(args, mod42)

    x_model = _x_for_mode(x_form, early_q, input_mode)
    late_times = np.array(args.late_grid, dtype=float)
    target_times = _target_grid(early_times, late_times, input_mode)
    target_q = np.stack([
        mod41._interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, target_times)
        for fid in fids
    ]).astype(np.float32)

    print(
        f"[46] dataset={dataset} input_mode={input_mode} n={len(fids)} "
        f"x_dim={x_model.shape[1]} q_targets={len(target_times)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    schemes = _split_schemes(
        n=len(fids),
        drug_groups=drug_groups,
        polymer_groups=polymer_groups,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    for scheme_name, splits in schemes:
        print(f"[46]   scheme={scheme_name}", flush=True)
        for fold_idx, (tr, te) in enumerate(splits):
            pred_by_method: dict[str, np.ndarray] = {}
            for method_name, model in _make_methods(args, args.seed + fold_idx).items():
                model.fit(x_model[tr], target_q[tr])
                pred_by_method[method_name] = model.predict(x_model[te])

            for local_i, j in enumerate(te):
                fid = int(fids[j])
                curve = curve_map[fid]
                row: dict[str, object] = {
                    "dataset": dataset,
                    "input_mode": input_mode,
                    "scheme": scheme_name,
                    "fold": fold_idx,
                    "fid": fid,
                    "n_obs": int(len(curve.t_obs)),
                }
                for method_name, pred_q in pred_by_method.items():
                    q_hat = _reconstruct_curve(
                        pred_grid_q=pred_q[local_i],
                        target_times=target_times,
                        early_q=early_q[j].astype(float),
                        early_times=early_times,
                        input_mode=input_mode,
                        t_obs=curve.t_obs,
                    )
                    row[f"r2_{method_name}"] = _r2(curve.q_obs, q_hat)
                rows.append(row)
    return rows


def _compare_to_theta(summary: pd.DataFrame, theta_summary_path: Path) -> pd.DataFrame:
    best_direct_idx = summary.groupby(["dataset", "input_mode", "scheme"])["median"].idxmax()
    direct = summary.loc[best_direct_idx].copy()
    direct = direct.rename(columns={
        "method": "direct_best_method",
        "median": "direct_best_median",
        "frac_above_0": "direct_frac_above_0",
    })
    direct = direct[[
        "dataset",
        "input_mode",
        "scheme",
        "direct_best_method",
        "direct_best_median",
        "direct_frac_above_0",
    ]]
    if not theta_summary_path.exists():
        direct["theta_best_method"] = ""
        direct["theta_best_median"] = np.nan
        direct["theta_minus_direct"] = np.nan
        return direct

    theta = pd.read_csv(theta_summary_path).rename(columns={
        "method": "theta_best_method",
        "median": "theta_best_median",
        "frac_above_0": "theta_frac_above_0",
    })
    merged = direct.merge(
        theta[[
            "dataset",
            "input_mode",
            "scheme",
            "theta_best_method",
            "theta_best_median",
            "theta_frac_above_0",
        ]],
        on=["dataset", "input_mode", "scheme"],
        how="left",
    )
    merged["theta_minus_direct"] = merged["theta_best_median"] - merged["direct_best_median"]
    return merged


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
    ap.add_argument("--input-mode", choices=(*INPUT_MODES, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument(
        "--theta-summary",
        type=Path,
        default=Path("outputs/45_input_source_ablation/input_source_summary.csv"),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/46_direct_curve_rf_input_ablation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    input_modes = INPUT_MODES if args.input_mode == "all" else (args.input_mode,)

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for input_mode in input_modes:
            rows.extend(_run_dataset_mode(args, dataset, input_mode, mod41, mod42))

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    comparison = _compare_to_theta(summary, args.theta_summary)

    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)
    comparison.to_csv(args.out / "theta_vs_direct_summary.csv", index=False)

    lines = [
        "=== 46 -- direct curve RF/ET input-source ablation ===",
        "",
        f"n_estimators : {args.n_estimators}",
        f"datasets     : {', '.join(datasets)}",
        f"input_modes  : {', '.join(input_modes)}",
        "",
        "--- best direct curve median R^2 vs best theta-route median R^2 ---",
    ]
    for _, row in comparison.sort_values(["dataset", "scheme", "input_mode"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['input_mode']:<22} "
            f"direct={row['direct_best_median']:+.4f} ({row['direct_best_method']})  "
            f"theta={row['theta_best_median']:+.4f} ({row['theta_best_method']})  "
            f"theta-direct={row['theta_minus_direct']:+.4f}"
        )
    lines.extend([
        "",
        "--- reconstruction rule ---",
        "  If early Q is available as input, direct ML predicts late-grid Q",
        "  and uses the observed early points as anchors. If formulation_only",
        "  is used, direct ML predicts both early and late grid points.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
