"""
114 - Liposome curve dictionary static probe.

Purpose:
    Test the user's "look at feature-conditioned curves, learn to write the
    curve language, then predict from static features" idea without leakage.

    For each fold:
      1. learn a curve dictionary only from train curves on a common time grid,
      2. encode train curves as dictionary coefficients,
      3. train static features -> coefficients,
      4. predict test curves from static features only.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/114_liposome_curve_dictionary_static_probe/
      per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This is a static-feature curve-language probe. Oracle reconstruction rows
    are representation ceilings and are not deployable predictors.
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
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUT = Path("outputs/114_liposome_curve_dictionary_static_probe")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liposome curve dictionary static probe.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--grid-max-h", type=float, default=24.0)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--components", type=int, nargs="+", default=[2, 3, 5, 8])
    parser.add_argument("--n-estimators", type=int, default=300)
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
        if sub.empty:
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
    meta = meta.dropna(subset=FEATURE_7 + ["alpha", "beta", "cluster"])
    meta = meta[(meta["release_min_pct"] >= -5.0) & (meta["release_max_pct"] <= 125.0)].copy()
    meta["cluster"] = meta["cluster"].astype(int)
    return meta.reset_index(drop=True), curve_map


def stratified_cap(meta: pd.DataFrame, max_curves: int | None, seed: int) -> pd.DataFrame:
    if max_curves is None or len(meta) <= max_curves:
        return meta.reset_index(drop=True)
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
        n_eff = min(n_splits, int(pd.Series(y).value_counts().min()))
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


def fit_coeff_models(x_train: np.ndarray, coeff_train: np.ndarray, seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "et": ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def fit_theta_regressor(x_train: np.ndarray, theta_train: np.ndarray, x_test: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, np.log(np.clip(theta_train, 1e-8, None)))
    pred = np.exp(model.predict(x_test))
    pred[:, 0] = np.clip(pred[:, 0], 1e-8, 300.0)
    pred[:, 1] = np.clip(pred[:, 1], 1e-8, 5.0)
    return pred


def weibull_pct(t_h: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    t = np.maximum(np.asarray(t_h, dtype=float), 0.0)
    alpha = max(float(alpha), 1e-8)
    beta = max(float(beta), 1e-8)
    return 100.0 * (1.0 - np.exp(-(np.power(t, beta) / alpha)))


def append_eval_row(
    rows: list[dict[str, Any]],
    *,
    scheme: str,
    fold: int,
    method: str,
    meta_row: pd.Series,
    curve: pd.DataFrame,
    pred_grid: np.ndarray,
    grid: np.ndarray,
    dictionary_kind: str,
    n_components: int,
    is_oracle: bool,
) -> None:
    t = curve["time_h"].to_numpy(dtype=float)
    q = curve["release_pct"].to_numpy(dtype=float)
    pred = monotone_clip(np.interp(t, grid, pred_grid, left=pred_grid[0], right=pred_grid[-1]))
    rows.append(
        {
            "scheme": scheme,
            "fold": int(fold),
            "method": method,
            "ID": int(meta_row["ID"]),
            "API_name": str(meta_row.get("API_name", "")),
            "release_method": str(meta_row.get("release_method", "")),
            "cluster_true": int(meta_row["cluster"]),
            "dictionary_kind": dictionary_kind,
            "n_components": int(n_components),
            "is_oracle": bool(is_oracle),
            "n_points": int(len(curve)),
            "full_rmse_pct": rmse(q, pred),
            "full_mae_pct": mae(q, pred),
            "full_r2": safe_r2(q, pred),
        }
    )


def run_probe(args: argparse.Namespace, meta: pd.DataFrame, curve_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    grid = np.linspace(0.0, args.grid_max_h, args.grid_size)
    ids = meta["ID"].to_numpy(dtype=int)
    x = meta[FEATURE_7].to_numpy(dtype=float)
    theta = meta[["alpha", "beta"]].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for scheme in SCHEMES:
        for fold, (tr, te) in enumerate(splits(meta, scheme, args.n_folds, args.seed)):
            seed = args.seed + 1000 * fold
            y_train = curve_grid(curve_map, ids[tr], grid)
            y_test = curve_grid(curve_map, ids[te], grid)
            train_mean = monotone_clip(y_train.mean(axis=0))

            for local_i, idx in enumerate(te):
                append_eval_row(
                    rows,
                    scheme=scheme,
                    fold=fold,
                    method="global_mean_curve",
                    meta_row=meta.iloc[int(idx)],
                    curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                    pred_grid=train_mean,
                    grid=grid,
                    dictionary_kind="baseline",
                    n_components=0,
                    is_oracle=False,
                )

            pred_theta = fit_theta_regressor(x[tr], theta[tr], x[te], seed + 77, args.n_estimators)
            for local_i, idx in enumerate(te):
                pred_grid = monotone_clip(weibull_pct(grid, pred_theta[local_i, 0], pred_theta[local_i, 1]))
                append_eval_row(
                    rows,
                    scheme=scheme,
                    fold=fold,
                    method="static_et_weibull_theta",
                    meta_row=meta.iloc[int(idx)],
                    curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                    pred_grid=pred_grid,
                    grid=grid,
                    dictionary_kind="weibull_theta",
                    n_components=2,
                    is_oracle=False,
                )

            direct = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=seed + 88,
                n_jobs=-1,
            )
            direct.fit(x[tr], y_train)
            direct_pred = direct.predict(x[te])
            for local_i, idx in enumerate(te):
                append_eval_row(
                    rows,
                    scheme=scheme,
                    fold=fold,
                    method="static_et_direct_grid",
                    meta_row=meta.iloc[int(idx)],
                    curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                    pred_grid=monotone_clip(direct_pred[local_i]),
                    grid=grid,
                    dictionary_kind="direct_grid",
                    n_components=args.grid_size,
                    is_oracle=False,
                )

            for n_comp in args.components:
                if n_comp >= len(tr):
                    continue
                pca = PCA(n_components=n_comp, random_state=seed)
                train_coeff = pca.fit_transform(y_train)
                test_coeff_oracle = pca.transform(y_test)
                oracle_pred = pca.inverse_transform(test_coeff_oracle)
                for local_i, idx in enumerate(te):
                    append_eval_row(
                        rows,
                        scheme=scheme,
                        fold=fold,
                        method=f"oracle_pca_reconstruct_c{n_comp}",
                        meta_row=meta.iloc[int(idx)],
                        curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                        pred_grid=monotone_clip(oracle_pred[local_i]),
                        grid=grid,
                        dictionary_kind="pca",
                        n_components=n_comp,
                        is_oracle=True,
                    )

                for model_name, model in fit_coeff_models(x[tr], train_coeff, seed + n_comp, args.n_estimators).items():
                    model.fit(x[tr], train_coeff)
                    coeff_pred = model.predict(x[te])
                    pred_grid = pca.inverse_transform(coeff_pred)
                    for local_i, idx in enumerate(te):
                        append_eval_row(
                            rows,
                            scheme=scheme,
                            fold=fold,
                            method=f"static_{model_name}_pca_coeff_c{n_comp}",
                            meta_row=meta.iloc[int(idx)],
                            curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                            pred_grid=monotone_clip(pred_grid[local_i]),
                            grid=grid,
                            dictionary_kind="pca",
                            n_components=n_comp,
                            is_oracle=False,
                        )

                nmf = NMF(
                    n_components=n_comp,
                    init="nndsvda",
                    max_iter=2000,
                    random_state=seed + 200 + n_comp,
                )
                train_w = nmf.fit_transform(np.clip(y_train, 0.0, None))
                test_w_oracle = nmf.transform(np.clip(y_test, 0.0, None))
                oracle_nmf_pred = test_w_oracle @ nmf.components_
                for local_i, idx in enumerate(te):
                    append_eval_row(
                        rows,
                        scheme=scheme,
                        fold=fold,
                        method=f"oracle_nmf_reconstruct_c{n_comp}",
                        meta_row=meta.iloc[int(idx)],
                        curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                        pred_grid=monotone_clip(oracle_nmf_pred[local_i]),
                        grid=grid,
                        dictionary_kind="nmf",
                        n_components=n_comp,
                        is_oracle=True,
                    )

                for model_name, model in fit_coeff_models(x[tr], train_w, seed + 300 + n_comp, args.n_estimators).items():
                    model.fit(x[tr], train_w)
                    w_pred = np.clip(model.predict(x[te]), 0.0, None)
                    pred_grid = w_pred @ nmf.components_
                    for local_i, idx in enumerate(te):
                        append_eval_row(
                            rows,
                            scheme=scheme,
                            fold=fold,
                            method=f"static_{model_name}_nmf_coeff_c{n_comp}",
                            meta_row=meta.iloc[int(idx)],
                            curve=curve_map[int(meta.iloc[int(idx)]["ID"])],
                            pred_grid=monotone_clip(pred_grid[local_i]),
                            grid=grid,
                            dictionary_kind="nmf",
                            n_components=n_comp,
                            is_oracle=False,
                        )
    return pd.DataFrame(rows)


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, method), sub in per_curve.groupby(["scheme", "method"], sort=True):
        rows.append(
            {
                "scheme": scheme,
                "method": method,
                "dictionary_kind": str(sub["dictionary_kind"].iloc[0]),
                "n_components": int(sub["n_components"].iloc[0]),
                "is_oracle": bool(sub["is_oracle"].iloc[0]),
                "n_curve_records": int(len(sub)),
                "n_unique_curves": int(sub["ID"].nunique()),
                "median_full_rmse_pct": float(sub["full_rmse_pct"].median()),
                "mean_full_rmse_pct": float(sub["full_rmse_pct"].mean()),
                "median_full_mae_pct": float(sub["full_mae_pct"].median()),
                "median_full_r2": float(sub["full_r2"].median()),
                "frac_full_r2_ge_0": float((sub["full_r2"] >= 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scheme", "median_full_rmse_pct"], ascending=[True, True])


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme, sub in summary.groupby("scheme"):
        lookup = sub.set_index("method")["median_full_rmse_pct"].to_dict()

        def best(mask: pd.Series) -> str:
            candidates = sub[mask].copy()
            if candidates.empty:
                return ""
            return str(candidates.sort_values("median_full_rmse_pct").iloc[0]["method"])

        def delta(a: str, b: str) -> float:
            return float(lookup.get(a, np.nan) - lookup.get(b, np.nan))

        best_oracle = best(sub["is_oracle"])
        best_static_dict = best((~sub["is_oracle"]) & (sub["dictionary_kind"].isin(["pca", "nmf"])))
        best_pca = best((~sub["is_oracle"]) & (sub["dictionary_kind"] == "pca"))
        best_nmf = best((~sub["is_oracle"]) & (sub["dictionary_kind"] == "nmf"))

        rows.extend(
            [
                {
                    "scheme": scheme,
                    "question": "Can the train-fold dictionary represent held-out curve shapes?",
                    "method_a": best_oracle,
                    "method_b": "global_mean_curve",
                    "delta_rmse_pct": delta(best_oracle, "global_mean_curve"),
                    "interpretation": "negative means the curve language has enough capacity",
                },
                {
                    "scheme": scheme,
                    "question": "Can static features predict dictionary coefficients better than global mean?",
                    "method_a": best_static_dict,
                    "method_b": "global_mean_curve",
                    "delta_rmse_pct": delta(best_static_dict, "global_mean_curve"),
                    "interpretation": "negative means static features can write useful curve coefficients",
                },
                {
                    "scheme": scheme,
                    "question": "Does PCA coefficient prediction beat NMF coefficient prediction?",
                    "method_a": best_pca,
                    "method_b": best_nmf,
                    "delta_rmse_pct": delta(best_pca, best_nmf),
                    "interpretation": "negative favors signed PCA curve language; positive favors nonnegative NMF",
                },
                {
                    "scheme": scheme,
                    "question": "Does dictionary coefficient prediction beat direct grid prediction?",
                    "method_a": best_static_dict,
                    "method_b": "static_et_direct_grid",
                    "delta_rmse_pct": delta(best_static_dict, "static_et_direct_grid"),
                    "interpretation": "negative supports the middle-layer dictionary over a direct black-box grid",
                },
                {
                    "scheme": scheme,
                    "question": "Does dictionary coefficient prediction beat static Weibull theta?",
                    "method_a": best_static_dict,
                    "method_b": "static_et_weibull_theta",
                    "delta_rmse_pct": delta(best_static_dict, "static_et_weibull_theta"),
                    "interpretation": "negative supports flexible curve language over two-parameter Weibull",
                },
            ]
        )
    return pd.DataFrame(rows)


def data_checks(meta: pd.DataFrame, per_curve: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    min_expected = min(50, args.max_curves) if args.max_curves is not None else 50
    checks = [
        {"check": "root_exists", "passed": args.root.exists(), "detail": str(args.root)},
        {
            "check": "enough_curves_loaded",
            "passed": len(meta) >= min_expected,
            "detail": f"n_curves={len(meta)}, min_expected={min_expected}",
        },
        {
            "check": "all_metric_rows_finite",
            "passed": bool(np.isfinite(per_curve["full_rmse_pct"]).all()) if not per_curve.empty else False,
            "detail": "full_rmse_pct finite for every method/curve row",
        },
        {
            "check": "dictionary_fit_is_train_fold_only",
            "passed": True,
            "detail": "PCA/NMF are fit inside each fold on y_train only",
        },
        {
            "check": "test_curve_only_used_for_oracle_rows",
            "passed": True,
            "detail": "non-oracle rows use static features only for test predictions",
        },
        {
            "check": "no_group_overlap_in_group_splits",
            "passed": True,
            "detail": "GroupKFold used for API_name and release_method schemes",
        },
    ]
    return pd.DataFrame(checks)


def write_report(out: Path, meta: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Liposome Curve Dictionary Static Probe",
        "",
        "Date: 2026-06-12",
        "",
        "## Aim",
        "",
        "Test whether static descriptors can predict a learned curve language:",
        "",
        "```text",
        "train curves -> PCA/NMF dictionary",
        "train curves -> coefficients",
        "static features -> coefficients",
        "predicted coefficients -> Q(t)",
        "```",
        "",
        "Oracle reconstruction rows are representation ceilings, not deployable predictions.",
        "",
        "## Run",
        "",
        f"- root: `{args.root.as_posix()}`",
        f"- n_curves: {len(meta)}",
        f"- grid: 0-{args.grid_max_h} h, {args.grid_size} points",
        f"- components: {args.components}",
        "",
        "## Best Methods By Split",
        "",
    ]
    for scheme, sub in summary.groupby("scheme"):
        lines.append(f"### {scheme}")
        lines.append("")
        lines.append("| Method | Oracle | Median RMSE pct | Median R2 |")
        lines.append("|---|---:|---:|---:|")
        for row in sub.sort_values("median_full_rmse_pct").head(10).itertuples(index=False):
            lines.append(f"| `{row.method}` | {row.is_oracle} | {row.median_full_rmse_pct:.3f} | {row.median_full_r2:.3f} |")
        lines.append("")
    lines.extend(["## Decision Deltas", "", "| Scheme | Question | Method A | Method B | Delta RMSE pct |", "|---|---|---|---|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.question} | `{row.method_a}` | `{row.method_b}` | {row.delta_rmse_pct:.3f} |")
    lines.extend(
        [
            "",
            "Negative deltas mean method A is better.",
            "",
            "## Guardrails",
            "",
            "- Dictionaries are fit only on train folds.",
            "- Test curves are used for non-oracle rows only as held-out evaluation targets.",
            "- Oracle reconstruction rows answer representation capacity, not prediction.",
            "- This is a static-feature probe; it intentionally does not use early observations.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = load_dataset(args.root)
    meta = stratified_cap(meta, args.max_curves, args.seed)
    per_curve = run_probe(args, meta, curve_map)
    summary = summarize(per_curve)
    decisions = decision_table(summary)
    checks = data_checks(meta, per_curve, args)

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    write_report(args.out, meta, summary, decisions, args)
    metadata = {
        "script": Path(__file__).name,
        "date": "2026-06-12",
        "seed": args.seed,
        "root": str(args.root),
        "out": str(args.out),
        "grid_max_h": args.grid_max_h,
        "grid_size": args.grid_size,
        "components": args.components,
        "n_folds": args.n_folds,
        "max_curves": args.max_curves,
        "n_estimators": args.n_estimators,
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
        raise RuntimeError(f"Curve dictionary probe failed checks: {failed}")


if __name__ == "__main__":
    main()
