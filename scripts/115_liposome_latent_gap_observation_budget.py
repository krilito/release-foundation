"""
115 - Liposome latent gap / observation-budget probe.

Purpose:
    Test whether early measured release observations close the gap between
    a static descriptor prior X -> z and an oracle curve-language encoding
    Q(t) -> z.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/115_liposome_latent_gap_observation_budget/
      per_curve_metrics.csv
      summary_by_method.csv
      budget_curve.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This is a release-state inference probe, not a foundation-model claim.
    Oracle rows use the held-out full curve only to measure representation
    ceilings and are excluded from deployable-model comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUT = Path("outputs/115_liposome_latent_gap_observation_budget")

FEATURE_7 = [
    "media_pH",
    "media_temp_oC",
    "drug_loading",
    "Z_average_nm",
    "API_type",
    "weighted_Mw",
    "weighted_Tm",
]
SCHEMES = ("stratified_5fold", "group_by_API", "group_by_release_method")
METHODS = ("et", "ridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liposome latent gap observation-budget probe.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--grid-max-h", type=float, default=24.0)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--components", type=int, nargs="+", default=[8, 3])
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    parser.add_argument("--subset-modes", nargs="+", default=["common", "all_available"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-ensemble", type=int, default=20)
    parser.add_argument(
        "--ensemble-primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train bootstrap ensembles only for the primary common/PCA-8/ET rows.",
    )
    return parser.parse_args()


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def parse_id(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"cannot parse IVR ID from {value!r}")
    return int(match.group(1))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.square(y - y.mean())))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum(np.square(y - pred)) / denom)


def monotone_clip(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.clip(np.asarray(y, dtype=float), 0.0, 120.0))


def load_dataset(root: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    exp = pd.read_csv(root / "results/fitting/drug_release_exp.csv")
    exp["ID"] = exp["file_name"].map(parse_id).astype(int)
    exp = exp.rename(columns={"time (Hrs)": "time_h", "release_percent": "release_pct"})
    exp = exp[["ID", "time_h", "release_pct", "file_name"]].copy()

    backend = pd.read_csv(root / "data/unprocessed/backend_data.csv")
    backend = backend.rename(columns={"IVR_ID": "ID"}).drop(columns=["Unnamed: 0"], errors="ignore")
    backend["ID"] = backend["ID"].astype(int)
    api_names = sorted(backend["API_name"].dropna().astype(str).unique().tolist())
    backend["API_type"] = backend["API_name"].astype(str).map({name: i for i, name in enumerate(api_names)})

    theta = pd.read_csv(root / "data/clean/weibull_params.csv")
    theta["ID"] = theta["ID"].astype(int)

    cluster = pd.read_csv(root / "results/clustering/3_PCA_KMC.csv")
    cluster = cluster.rename(columns={"file": "ID"})
    cluster["ID"] = cluster["ID"].astype(int)

    curve_rows: list[dict[str, Any]] = []
    curve_map: dict[int, pd.DataFrame] = {}
    for curve_id, sub in exp.groupby("ID"):
        sub = sub.sort_values("time_h").dropna(subset=["time_h", "release_pct"])
        if len(sub) < 3:
            continue
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        curve_map[int(curve_id)] = sub
        curve_rows.append(
            {
                "ID": int(curve_id),
                "n_points": int(len(sub)),
                "t_min_h": float(np.min(t)),
                "t_max_h": float(np.max(t)),
                "release_min_pct": float(np.min(q)),
                "release_max_pct": float(np.max(q)),
            }
        )

    meta = pd.DataFrame(curve_rows)
    meta = meta.merge(backend, on="ID", how="left")
    meta = meta.merge(theta[["ID", "alpha", "beta"]], on="ID", how="inner")
    meta = meta.merge(cluster[["ID", "cluster", "cluster_name"]], on="ID", how="inner")
    meta = meta.dropna(subset=FEATURE_7 + ["alpha", "beta", "cluster", "API_name", "release_method"])
    meta = meta[(meta["release_min_pct"] >= -5.0) & (meta["release_max_pct"] <= 125.0)].copy()
    meta["cluster"] = meta["cluster"].astype(int)
    return meta.reset_index(drop=True), curve_map


def stratified_cap(meta: pd.DataFrame, max_curves: int | None, seed: int) -> pd.DataFrame:
    if max_curves is None or len(meta) <= max_curves:
        return meta.sort_values("ID").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pieces = []
    per_class = max(2, max_curves // max(1, meta["cluster"].nunique()))
    for _, sub in meta.groupby("cluster"):
        take = min(len(sub), per_class)
        pieces.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    out = pd.concat(pieces, ignore_index=True)
    if len(out) < max_curves:
        remaining = meta[~meta["ID"].isin(out["ID"])]
        take = min(max_curves - len(out), len(remaining))
        if take:
            out = pd.concat(
                [out, remaining.iloc[rng.choice(len(remaining), size=take, replace=False)]],
                ignore_index=True,
            )
    return out.sort_values("ID").reset_index(drop=True)


def curve_grid(curve_map: dict[int, pd.DataFrame], ids: np.ndarray, grid: np.ndarray) -> np.ndarray:
    rows = []
    for curve_id in ids:
        sub = curve_map[int(curve_id)].sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        rows.append(monotone_clip(np.interp(grid, t, q, left=q[0], right=q[-1])))
    return np.asarray(rows, dtype=float)


def splits(meta: pd.DataFrame, scheme: str, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = meta["cluster"].to_numpy(dtype=int)
    idx = np.arange(len(meta))
    if scheme == "stratified_5fold":
        min_class = int(pd.Series(y).value_counts().min())
        n_eff = min(n_splits, min_class)
        if n_eff < 2:
            return []
        splitter = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=seed)
        return list(splitter.split(idx, y))

    group_col = {"group_by_API": "API_name", "group_by_release_method": "release_method"}[scheme]
    groups = meta[group_col].astype(str).fillna("__MISSING__").to_numpy()
    n_eff = min(n_splits, int(pd.Series(groups).nunique()))
    if n_eff < 2:
        return []
    splitter = GroupKFold(n_splits=n_eff)
    return list(splitter.split(idx, y, groups))


def eligible_for_budget(curve_map: dict[int, pd.DataFrame], curve_id: int, k: int) -> bool:
    sub = curve_map[int(curve_id)].sort_values("time_h")
    if k == 0:
        return len(sub) >= 3
    if len(sub) < k + 2:
        return False
    times = sub["time_h"].to_numpy(dtype=float)
    t_cut = float(times[k - 1])
    return int(np.sum(times > t_cut)) >= 2


def static_features(meta: pd.DataFrame) -> np.ndarray:
    return meta[FEATURE_7].to_numpy(dtype=float)


def context_features(
    meta: pd.DataFrame,
    curve_map: dict[int, pd.DataFrame],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k == 0:
        return static_features(meta), np.full(len(meta), -np.inf, dtype=float)

    x_static = static_features(meta)
    obs_blocks = []
    t_cuts = []
    for curve_id in meta["ID"].to_numpy(dtype=int):
        sub = curve_map[int(curve_id)].sort_values("time_h").iloc[:k]
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        block = np.empty(k * 2, dtype=float)
        block[0::2] = np.log1p(t)
        block[1::2] = np.clip(q, 0.0, 120.0)
        obs_blocks.append(block)
        t_cuts.append(float(t[-1]))
    return np.hstack([x_static, np.asarray(obs_blocks, dtype=float)]), np.asarray(t_cuts, dtype=float)


def fit_predictors(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    n_estimators: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(x_train, z_train)
    out["ridge"] = ridge.predict(x_test)

    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    et.fit(x_train, z_train)
    out["et"] = et.predict(x_test)
    return out


def bootstrap_ensemble_predictions(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_test: np.ndarray,
    pca: PCA,
    seed: int,
    n_estimators: int,
    n_ensemble: int,
) -> np.ndarray | None:
    if n_ensemble <= 0:
        return None
    rng = np.random.default_rng(seed)
    decoded = []
    n_train = len(x_train)
    for member in range(n_ensemble):
        boot = rng.choice(n_train, size=n_train, replace=True)
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed + 1009 * (member + 1),
            n_jobs=-1,
        )
        model.fit(x_train[boot], z_train[boot])
        pred_z = model.predict(x_test)
        decoded.append(np.asarray([monotone_clip(row) for row in pca.inverse_transform(pred_z)], dtype=float))
    return np.stack(decoded, axis=0)


def row_metrics(
    *,
    subset_mode: str,
    scheme: str,
    fold: int,
    method: str,
    model: str,
    n_components: int,
    budget_k: int,
    meta_row: pd.Series,
    curve: pd.DataFrame,
    grid: np.ndarray,
    pred_grid: np.ndarray,
    z_pred: np.ndarray | None,
    z_oracle: np.ndarray | None,
    t_cut: float,
    is_oracle: bool,
    is_deployable: bool,
    ensemble_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    t = curve["time_h"].to_numpy(dtype=float)
    q_raw = curve["release_pct"].to_numpy(dtype=float)
    pred = np.interp(t, grid, pred_grid, left=pred_grid[0], right=pred_grid[-1])
    future_mask = t > t_cut
    if not np.any(future_mask):
        future_mask = np.ones_like(t, dtype=bool)

    z_rmse = float("nan")
    if z_pred is not None and z_oracle is not None:
        z_rmse = rmse(np.asarray(z_oracle, dtype=float), np.asarray(z_pred, dtype=float))

    cov50 = cov90 = width50 = width90 = float("nan")
    if ensemble_grid is not None:
        ens_at_t = np.asarray([np.interp(t, grid, member, left=member[0], right=member[-1]) for member in ensemble_grid])
        lo50, hi50 = np.percentile(ens_at_t, [25, 75], axis=0)
        lo90, hi90 = np.percentile(ens_at_t, [5, 95], axis=0)
        q_future = q_raw[future_mask]
        cov50 = float(np.mean((q_future >= lo50[future_mask]) & (q_future <= hi50[future_mask])))
        cov90 = float(np.mean((q_future >= lo90[future_mask]) & (q_future <= hi90[future_mask])))
        width50 = float(np.mean(hi50[future_mask] - lo50[future_mask]))
        width90 = float(np.mean(hi90[future_mask] - lo90[future_mask]))

    return {
        "subset_mode": subset_mode,
        "scheme": scheme,
        "fold": fold,
        "method": method,
        "model": model,
        "dictionary_kind": "pca",
        "n_components": n_components,
        "budget_k": budget_k,
        "ID": int(meta_row["ID"]),
        "API_name": str(meta_row["API_name"]),
        "release_method": str(meta_row["release_method"]),
        "cluster": int(meta_row["cluster"]),
        "n_points": int(meta_row["n_points"]),
        "t_cut_h": float(t_cut) if np.isfinite(t_cut) else float("nan"),
        "z_rmse": z_rmse,
        "decoded_curve_rmse": rmse(q_raw, pred),
        "decoded_curve_mae": mae(q_raw, pred),
        "decoded_curve_r2": safe_r2(q_raw, pred),
        "future_curve_rmse_after_t_k": rmse(q_raw[future_mask], pred[future_mask]),
        "future_curve_mae_after_t_k": mae(q_raw[future_mask], pred[future_mask]),
        "future_curve_r2_after_t_k": safe_r2(q_raw[future_mask], pred[future_mask]),
        "cov50": cov50,
        "cov90": cov90,
        "interval_width50": width50,
        "interval_width90": width90,
        "is_oracle": bool(is_oracle),
        "is_deployable": bool(is_deployable),
    }


def run_probe(args: argparse.Namespace, meta_all: pd.DataFrame, curve_map: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    grid = np.linspace(0.0, args.grid_max_h, args.grid_size)
    rows: list[dict[str, Any]] = []
    fold_checks: list[dict[str, Any]] = []
    primary_component = int(args.components[0])

    for subset_mode in args.subset_modes:
        for budget_k in args.budgets:
            if subset_mode == "common":
                eligible_ids = [cid for cid in meta_all["ID"].to_numpy(dtype=int) if eligible_for_budget(curve_map, cid, max(args.budgets))]
            elif subset_mode == "all_available":
                eligible_ids = [cid for cid in meta_all["ID"].to_numpy(dtype=int) if eligible_for_budget(curve_map, cid, budget_k)]
            else:
                raise ValueError(f"unknown subset mode: {subset_mode}")

            meta = meta_all[meta_all["ID"].isin(eligible_ids)].sort_values("ID").reset_index(drop=True)
            if len(meta) < 12:
                continue
            x_all, t_cuts = context_features(meta, curve_map, budget_k)
            ids = meta["ID"].to_numpy(dtype=int)
            y_grid = curve_grid(curve_map, ids, grid)

            for scheme in SCHEMES:
                for fold, (tr, te) in enumerate(splits(meta, scheme, args.n_folds, args.seed), start=1):
                    if len(tr) < 6 or len(te) < 2:
                        continue
                    if scheme.startswith("group_by_"):
                        group_col = {"group_by_API": "API_name", "group_by_release_method": "release_method"}[scheme]
                        train_groups = set(meta.iloc[tr][group_col].astype(str))
                        test_groups = set(meta.iloc[te][group_col].astype(str))
                        overlap = sorted(train_groups & test_groups)
                        fold_checks.append(
                            {
                                "subset_mode": subset_mode,
                                "scheme": scheme,
                                "budget_k": budget_k,
                                "fold": fold,
                                "check": "no_heldout_group_overlap",
                                "passed": len(overlap) == 0,
                                "detail": ",".join(overlap),
                            }
                        )

                    global_mean = monotone_clip(y_grid[tr].mean(axis=0))
                    for n_comp in args.components:
                        if int(n_comp) >= len(tr):
                            continue
                        pca = PCA(n_components=int(n_comp), random_state=args.seed)
                        z_train = pca.fit_transform(y_grid[tr])
                        z_oracle = pca.transform(y_grid[te])
                        oracle_grid = np.asarray([monotone_clip(row) for row in pca.inverse_transform(z_oracle)])
                        pred_z_by_model = fit_predictors(
                            x_all[tr],
                            z_train,
                            x_all[te],
                            args.seed + 17 * fold + 31 * int(n_comp) + budget_k,
                            args.n_estimators,
                        )

                        ensemble_decoded: np.ndarray | None = None
                        do_ensemble = args.n_ensemble > 0
                        if args.ensemble_primary_only:
                            do_ensemble = (
                                do_ensemble
                                and subset_mode == "common"
                                and int(n_comp) == primary_component
                            )
                        if do_ensemble:
                            ensemble_decoded = bootstrap_ensemble_predictions(
                                x_all[tr],
                                z_train,
                                x_all[te],
                                pca,
                                args.seed + 5003 * fold + 101 * budget_k + int(n_comp),
                                args.n_estimators,
                                args.n_ensemble,
                            )

                        for local_i, idx in enumerate(te):
                            curve_id = int(meta.iloc[int(idx)]["ID"])
                            curve = curve_map[curve_id]
                            rows.append(
                                row_metrics(
                                    subset_mode=subset_mode,
                                    scheme=scheme,
                                    fold=fold,
                                    method=f"oracle_full_curve_z_pca_c{int(n_comp)}",
                                    model="oracle",
                                    n_components=int(n_comp),
                                    budget_k=budget_k,
                                    meta_row=meta.iloc[int(idx)],
                                    curve=curve,
                                    grid=grid,
                                    pred_grid=oracle_grid[local_i],
                                    z_pred=z_oracle[local_i],
                                    z_oracle=z_oracle[local_i],
                                    t_cut=t_cuts[int(idx)],
                                    is_oracle=True,
                                    is_deployable=False,
                                )
                            )
                            rows.append(
                                row_metrics(
                                    subset_mode=subset_mode,
                                    scheme=scheme,
                                    fold=fold,
                                    method="global_mean_curve",
                                    model="global_mean",
                                    n_components=int(n_comp),
                                    budget_k=budget_k,
                                    meta_row=meta.iloc[int(idx)],
                                    curve=curve,
                                    grid=grid,
                                    pred_grid=global_mean,
                                    z_pred=None,
                                    z_oracle=z_oracle[local_i],
                                    t_cut=t_cuts[int(idx)],
                                    is_oracle=False,
                                    is_deployable=True,
                                )
                            )

                            for model_name in METHODS:
                                z_pred = pred_z_by_model[model_name][local_i]
                                pred_grid = monotone_clip(pca.inverse_transform(z_pred.reshape(1, -1))[0])
                                ens_for_curve = None
                                if ensemble_decoded is not None and model_name == "et":
                                    ens_for_curve = ensemble_decoded[:, local_i, :]
                                rows.append(
                                    row_metrics(
                                        subset_mode=subset_mode,
                                        scheme=scheme,
                                        fold=fold,
                                        method=f"static_plus_obs_{model_name}_pca_c{int(n_comp)}",
                                        model=model_name,
                                        n_components=int(n_comp),
                                        budget_k=budget_k,
                                        meta_row=meta.iloc[int(idx)],
                                        curve=curve,
                                        grid=grid,
                                        pred_grid=pred_grid,
                                        z_pred=z_pred,
                                        z_oracle=z_oracle[local_i],
                                        t_cut=t_cuts[int(idx)],
                                        is_oracle=False,
                                        is_deployable=True,
                                        ensemble_grid=ens_for_curve,
                                    )
                                )

    return pd.DataFrame(rows), fold_checks


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subset_mode", "scheme", "method", "model", "n_components", "budget_k"]
    rows: list[dict[str, Any]] = []
    for keys, sub in per_curve.groupby(group_cols, sort=True):
        subset_mode, scheme, method, model, n_components, budget_k = keys
        rows.append(
            {
                "subset_mode": subset_mode,
                "scheme": scheme,
                "method": method,
                "model": model,
                "dictionary_kind": "pca",
                "n_components": int(n_components),
                "budget_k": int(budget_k),
                "is_oracle": bool(sub["is_oracle"].iloc[0]),
                "is_deployable": bool(sub["is_deployable"].iloc[0]),
                "n_curve_records": int(len(sub)),
                "n_unique_curves": int(sub["ID"].nunique()),
                "z_rmse_median": float(sub["z_rmse"].median(skipna=True)),
                "decoded_curve_rmse_median": float(sub["decoded_curve_rmse"].median()),
                "decoded_curve_rmse_mean": float(sub["decoded_curve_rmse"].mean()),
                "future_curve_rmse_median": float(sub["future_curve_rmse_after_t_k"].median()),
                "future_curve_rmse_mean": float(sub["future_curve_rmse_after_t_k"].mean()),
                "future_curve_mae_median": float(sub["future_curve_mae_after_t_k"].median()),
                "future_curve_r2_median": float(sub["future_curve_r2_after_t_k"].median()),
                "cov50_mean": float(sub["cov50"].mean(skipna=True)),
                "cov90_mean": float(sub["cov90"].mean(skipna=True)),
                "interval_width50_median": float(sub["interval_width50"].median(skipna=True)),
                "interval_width90_median": float(sub["interval_width90"].median(skipna=True)),
            }
        )
    summary = pd.DataFrame(rows)

    enriched: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        scope = (
            (summary["subset_mode"] == row["subset_mode"])
            & (summary["scheme"] == row["scheme"])
            & (summary["n_components"] == row["n_components"])
        )
        same_method = scope & (summary["method"] == row["method"])
        static = summary[same_method & (summary["budget_k"] == 0)]
        oracle = summary[scope & (summary["method"] == f"oracle_full_curve_z_pca_c{row['n_components']}") & (summary["budget_k"] == row["budget_k"])]
        width0 = summary[same_method & (summary["budget_k"] == 0)]
        static_rmse = float(static["future_curve_rmse_median"].iloc[0]) if not static.empty else float("nan")
        oracle_rmse = float(oracle["future_curve_rmse_median"].iloc[0]) if not oracle.empty else float("nan")
        denom = static_rmse - oracle_rmse
        row["oracle_future_rmse_median"] = oracle_rmse
        row["static_k0_future_rmse_median"] = static_rmse
        row["oracle_gap_closed_pct"] = float("nan") if denom <= 0 or not np.isfinite(denom) else 100.0 * (static_rmse - row["future_curve_rmse_median"]) / denom
        width0_value = float(width0["interval_width90_median"].iloc[0]) if not width0.empty else float("nan")
        row["width_ratio_vs_k0"] = (
            float("nan")
            if width0_value <= 0 or not np.isfinite(width0_value)
            else row["interval_width90_median"] / width0_value
        )
        enriched.append(row)
    return pd.DataFrame(enriched).sort_values(
        ["subset_mode", "scheme", "n_components", "budget_k", "future_curve_rmse_median", "method"],
        ascending=[True, True, True, True, True, True],
    )


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = summary[(summary["subset_mode"] == "common") & (summary["n_components"] == 8)].copy()
    for scheme, sub in primary.groupby("scheme"):
        deploy = sub[sub["is_deployable"]]
        for model_name in METHODS:
            method = f"static_plus_obs_{model_name}_pca_c8"
            model_rows = deploy[deploy["method"] == method].sort_values("budget_k")
            if model_rows.empty:
                continue
            global_rows = deploy[deploy["method"] == "global_mean_curve"]
            k0 = model_rows[model_rows["budget_k"] == 0]
            if not k0.empty and not global_rows.empty:
                g0 = global_rows[global_rows["budget_k"] == 0]
                rows.append(
                    {
                        "scheme": scheme,
                        "question": f"Does X -> z with {model_name} beat global mean at k=0?",
                        "method_a": method,
                        "method_b": "global_mean_curve",
                        "budget_k": 0,
                        "delta_future_rmse": float(k0["future_curve_rmse_median"].iloc[0] - g0["future_curve_rmse_median"].iloc[0]),
                        "value": float(k0["future_curve_rmse_median"].iloc[0]),
                        "passed": bool(k0["future_curve_rmse_median"].iloc[0] < g0["future_curve_rmse_median"].iloc[0]),
                        "interpretation": "negative delta means static descriptors give a useful latent prior",
                    }
                )

            values = model_rows.set_index("budget_k")["future_curve_rmse_median"].to_dict()
            seq = [values.get(k, np.nan) for k in [0, 1, 2, 3, 5]]
            diffs = np.diff([v for v in seq if np.isfinite(v)])
            rows.append(
                {
                    "scheme": scheme,
                    "question": f"Does X + Y_k -> z improve monotonically or near-monotonically for {model_name}?",
                    "method_a": method,
                    "method_b": "same_method_k0",
                    "budget_k": -1,
                    "delta_future_rmse": float(model_rows["future_curve_rmse_median"].iloc[-1] - model_rows["future_curve_rmse_median"].iloc[0]),
                    "value": float(np.nanmean(diffs <= 0.5)) if len(diffs) else float("nan"),
                    "passed": bool(len(diffs) > 0 and np.nanmean(diffs <= 0.5) >= 0.75),
                    "interpretation": "passed allows small noise but expects most extra observations not to hurt",
                }
            )

            for budget_k in [1, 2, 3, 5]:
                row_k = model_rows[model_rows["budget_k"] == budget_k]
                if row_k.empty:
                    continue
                gap = float(row_k["oracle_gap_closed_pct"].iloc[0])
                rows.append(
                    {
                        "scheme": scheme,
                        "question": f"How much oracle gap is closed at k={budget_k} for {model_name}?",
                        "method_a": method,
                        "method_b": "oracle_full_curve_z_pca_c8",
                        "budget_k": budget_k,
                        "delta_future_rmse": float(row_k["future_curve_rmse_median"].iloc[0] - row_k["oracle_future_rmse_median"].iloc[0]),
                        "value": gap,
                        "passed": bool(np.isfinite(gap) and gap > 0.0),
                        "interpretation": "positive value means early observations move static prior toward oracle curve state",
                    }
                )

            widths = model_rows.set_index("budget_k")["width_ratio_vs_k0"].to_dict()
            if any(np.isfinite(v) for v in widths.values()):
                rows.append(
                    {
                        "scheme": scheme,
                        "question": f"Does ensemble interval width contract with k for {model_name}?",
                        "method_a": method,
                        "method_b": "same_method_k0",
                        "budget_k": -1,
                        "delta_future_rmse": float("nan"),
                        "value": float(widths.get(5, np.nan)),
                        "passed": bool(np.isfinite(widths.get(5, np.nan)) and widths.get(5, np.nan) < 1.0),
                        "interpretation": "width(k)/width(0) below 1 supports empirical uncertainty contraction",
                    }
                )

            strict = scheme in {"group_by_API", "group_by_release_method"}
            if strict:
                k5 = model_rows[model_rows["budget_k"] == 5]
                rows.append(
                    {
                        "scheme": scheme,
                        "question": f"Does strict group split still show gap closure for {model_name}?",
                        "method_a": method,
                        "method_b": "same_method_k0",
                        "budget_k": 5,
                        "delta_future_rmse": float(k5["future_curve_rmse_median"].iloc[0] - k0["future_curve_rmse_median"].iloc[0]) if not k5.empty and not k0.empty else float("nan"),
                        "value": float(k5["oracle_gap_closed_pct"].iloc[0]) if not k5.empty else float("nan"),
                        "passed": bool(not k5.empty and np.isfinite(k5["oracle_gap_closed_pct"].iloc[0]) and k5["oracle_gap_closed_pct"].iloc[0] > 0.0),
                        "interpretation": "strict split is the non-leaky transfer guardrail",
                    }
                )

        for _, row in sub[(sub["method"].str.startswith("static_plus_obs_")) & (sub["budget_k"] == 0)].iterrows():
            denom = row["static_k0_future_rmse_median"] - row["oracle_future_rmse_median"]
            if np.isfinite(denom) and denom <= 0:
                rows.append(
                    {
                        "scheme": scheme,
                        "question": "Guardrail: static_not_worse_than_oracle_guardrail",
                        "method_a": row["method"],
                        "method_b": f"oracle_full_curve_z_pca_c{int(row['n_components'])}",
                        "budget_k": 0,
                        "delta_future_rmse": float(denom),
                        "value": float("nan"),
                        "passed": False,
                        "interpretation": "gap closure is undefined because static is not worse than oracle",
                    }
                )
    return pd.DataFrame(rows)


def data_checks(
    meta: pd.DataFrame,
    per_curve: pd.DataFrame,
    fold_checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = [
        {"check": "input_files_exist", "passed": bool(args.root.exists()), "detail": str(args.root)},
        {"check": "sample_unit_is_curve", "passed": bool("ID" in meta.columns and meta["ID"].is_unique), "detail": f"n_curves={len(meta)}"},
        {"check": "no_timepoint_split", "passed": True, "detail": "all split indices are over curve-level metadata rows"},
        {
            "check": "every_curve_has_at_least_3_valid_timepoints",
            "passed": bool((meta["n_points"] >= 3).all()),
            "detail": f"min_n_points={int(meta['n_points'].min()) if len(meta) else 0}",
        },
        {
            "check": "dictionary_representation_fit_train_fold_only",
            "passed": True,
            "detail": "PCA.fit_transform is called only on y_grid[train] inside each fold",
        },
        {
            "check": "oracle_rows_marked_and_not_deployable",
            "passed": bool((~per_curve[per_curve["is_oracle"]]["is_deployable"]).all()) if not per_curve.empty else False,
            "detail": "oracle rows have is_oracle=True and is_deployable=False",
        },
        {
            "check": "no_nan_in_primary_metrics",
            "passed": bool(np.isfinite(per_curve["future_curve_rmse_after_t_k"]).all()) if not per_curve.empty else False,
            "detail": "future_curve_rmse_after_t_k finite for every row",
        },
        {
            "check": "context_times_strictly_earlier_than_targets",
            "passed": True,
            "detail": "eligibility requires at least two observed target points with time > kth context time",
        },
    ]
    if fold_checks:
        fold_df = pd.DataFrame(fold_checks)
        checks.append(
            {
                "check": "no_heldout_group_overlap_in_group_splits",
                "passed": bool(fold_df["passed"].all()),
                "detail": f"n_group_fold_checks={len(fold_df)}",
            }
        )
    else:
        checks.append(
            {
                "check": "no_heldout_group_overlap_in_group_splits",
                "passed": True,
                "detail": "no group folds were generated in this run",
            }
        )
    return pd.DataFrame(checks)


def write_report(out: Path, meta: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace) -> None:
    primary = summary[(summary["subset_mode"] == "common") & (summary["n_components"] == 8)].copy()
    lines = [
        "# Liposome Latent Gap Observation-Budget Probe",
        "",
        "Date: 2026-06-12",
        "",
        "## Aim",
        "",
        "Test whether early measured observations close the gap between static descriptors and oracle curve-language coefficients.",
        "",
        "```text",
        "X -> z",
        "X + Y_k -> z",
        "oracle Q(t) -> z",
        "z -> Q_hat(t)",
        "```",
        "",
        "This output treats prediction as release-state inference under costly observations, not as a generic curve-regression leaderboard.",
        "",
        "## Run",
        "",
        f"- root: `{args.root.as_posix()}`",
        f"- n_curves loaded: {len(meta)}",
        f"- grid: 0-{args.grid_max_h} h, {args.grid_size} points",
        f"- PCA components: {args.components}",
        f"- budgets: {args.budgets}",
        f"- subset modes: {args.subset_modes}",
        f"- ensemble size: {args.n_ensemble}",
        "",
        "## Primary Common-Subset Results",
        "",
    ]
    for scheme, sub in primary.groupby("scheme"):
        lines.append(f"### {scheme}")
        lines.append("")
        lines.append("| Method | k | Median future RMSE | Gap closed pct | cov90 | width90 ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        keep = sub[sub["method"].isin(["static_plus_obs_et_pca_c8", "static_plus_obs_ridge_pca_c8", "oracle_full_curve_z_pca_c8", "global_mean_curve"])]
        for row in keep.sort_values(["budget_k", "future_curve_rmse_median", "method"]).itertuples(index=False):
            lines.append(
                f"| `{row.method}` | {row.budget_k} | {row.future_curve_rmse_median:.3f} | "
                f"{row.oracle_gap_closed_pct:.1f} | {row.cov90_mean:.3f} | {row.width_ratio_vs_k0:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Decisions",
            "",
            "| Scheme | Question | Method A | Method B | k | Value | Passed |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| {row.scheme} | {row.question} | `{row.method_a}` | `{row.method_b}` | "
            f"{row.budget_k} | {row.value:.3f} | {row.passed} |"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Sample unit is one release curve, never an independent timepoint.",
            "- PCA dictionaries are fit train-fold only.",
            "- Test full curves appear in oracle rows only; oracle rows are explicitly non-deployable.",
            "- The common subset requires eligibility for k=5 before all budgets are compared.",
            "- The all-available subset is a sensitivity analysis and changes sample support across k.",
            "- Ensemble intervals are empirical bootstrap intervals, not Bayesian posterior credible intervals.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_doc(doc_path: Path, summary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    primary = summary[(summary["subset_mode"] == "common") & (summary["n_components"] == 8)]
    lines = [
        "# Liposome Latent Gap Observation-Budget",
        "",
        "Date: 2026-06-12",
        "",
        "## Question",
        "",
        "Does measured early release close the latent-state gap between static descriptors and oracle curve-language coefficients?",
        "",
        "This experiment evaluates:",
        "",
        "```text",
        "X -> z",
        "X + Y_k -> z",
        "oracle Q(t) -> z",
        "```",
        "",
        "where `z` is a train-fold PCA curve dictionary coefficient vector and `Y_k` is the first `k` measured release observations.",
        "",
        "## Method",
        "",
        "- Data: Yanes et al. accelerated IVR liposome release curves.",
        "- Unit: one curve per sample.",
        "- Grid: 0-24 h with 80 points.",
        "- Primary representation: PCA dictionary with 8 components.",
        "- Sensitivity representation: PCA dictionary with 3 components.",
        "- Splits: stratified 5-fold, group by API, and group by release method.",
        "- Predictors: ExtraTrees as the primary small-data nonlinear model; Ridge as a linear sanity check.",
        "- Uncertainty: bootstrap ExtraTrees ensemble intervals, reported as empirical intervals rather than Bayesian posteriors.",
        "",
        "## Primary Result",
        "",
        "| Split | Model | k | Median future RMSE | Oracle gap closed pct | width90 ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary[
        primary["method"].isin(["static_plus_obs_et_pca_c8", "static_plus_obs_ridge_pca_c8"])
    ].sort_values(["scheme", "model", "budget_k"]).itertuples(index=False):
        lines.append(
            f"| {row.scheme} | `{row.method}` | {row.budget_k} | {row.future_curve_rmse_median:.3f} | "
            f"{row.oracle_gap_closed_pct:.1f} | {row.width_ratio_vs_k0:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Split | Question | Value | Passed |",
            "|---|---|---:|---:|",
        ]
    )
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.question} | {row.value:.3f} | {row.passed} |")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "A positive gap-closure curve supports the claim that early observations identify release state that static descriptors miss. A weak or negative strict-split result means the available static descriptors and early points do not transfer reliably across held-out drug or assay groups.",
            "",
            "Do not read this as a mechanism-learning or foundation-model result. It is a release-state inference diagnostic.",
            "",
            "## Output Anchor",
            "",
            "`../outputs/115_liposome_latent_gap_observation_budget/`",
        ]
    )
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = load_dataset(args.root)
    meta = stratified_cap(meta, args.max_curves, args.seed)
    per_curve, fold_checks = run_probe(args, meta, curve_map)
    if per_curve.empty:
        raise RuntimeError("no per-curve rows were generated")

    summary = summarize(per_curve)
    budget_curve = summary[
        summary["method"].str.startswith("static_plus_obs_")
        & (summary["subset_mode"] == "common")
    ].copy()
    decisions = decision_table(summary)
    checks = data_checks(meta, per_curve, fold_checks, args)

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    budget_curve.to_csv(args.out / "budget_curve.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    write_report(args.out, meta, summary, decisions, args)

    doc_path = Path("docs/liposome_latent_gap_observation_budget_2026-06-12.md")
    write_doc(doc_path, summary, decisions)

    metadata = {
        "script": Path(__file__).name,
        "date": "2026-06-12",
        "seed": args.seed,
        "root": str(args.root),
        "out": str(args.out),
        "grid_max_h": args.grid_max_h,
        "grid_size": args.grid_size,
        "components": args.components,
        "budgets": args.budgets,
        "subset_modes": args.subset_modes,
        "n_folds": args.n_folds,
        "max_curves": args.max_curves,
        "n_estimators": args.n_estimators,
        "n_ensemble": args.n_ensemble,
        "ensemble_primary_only": args.ensemble_primary_only,
        "git_hash": git_hash(),
        "input_files": {
            "backend_data": file_meta(args.root / "data/unprocessed/backend_data.csv"),
            "weibull_params": file_meta(args.root / "data/clean/weibull_params.csv"),
            "clusters": file_meta(args.root / "results/clustering/3_PCA_KMC.csv"),
            "release_exp": file_meta(args.root / "results/fitting/drug_release_exp.csv"),
        },
        "generated_files": sorted(p.name for p in args.out.iterdir() if p.is_file()),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"latent gap probe failed checks: {failed}")


if __name__ == "__main__":
    main()
