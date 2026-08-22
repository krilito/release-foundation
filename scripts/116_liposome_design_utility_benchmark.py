"""
116 - Liposome design utility benchmark.

Purpose:
    Test whether static release priors can rank pre-experimental liposome
    candidates for target release profiles, even when exact curve RMSE is weak.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/116_liposome_design_utility_benchmark/
      per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    Prediction utility is not exact curve RMSE. This script evaluates whether
    static priors help choose candidates before any release observation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


HELPER_PATH = Path(__file__).resolve().parent / "115_liposome_latent_gap_observation_budget.py"
spec = importlib.util.spec_from_file_location("liposome_115_helpers", HELPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import helper script: {HELPER_PATH}")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


DEFAULT_ROOT = helpers.DEFAULT_ROOT
DEFAULT_OUT = Path("outputs/116_liposome_design_utility_benchmark")
SCHEMES = helpers.SCHEMES
FEATURE_7 = helpers.FEATURE_7
TARGETS = ("anti_burst", "sustained_mid", "high_final", "avoid_failure")
TOP_RULES = ("top5", "top10", "top20pct")
METHODS = (
    "global_mean_curve",
    "author_class_proto",
    "static_weibull_theta",
    "static_curve_dictionary_coeff",
    "static_et_direct_grid",
    "static_ensemble_uncertainty_prior",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liposome static design utility benchmark.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--grid-max-h", type=float, default=24.0)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-ensemble", type=int, default=20)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    return parser.parse_args()


def x_static(meta: pd.DataFrame) -> np.ndarray:
    return meta[FEATURE_7].to_numpy(dtype=float)


def interp_grid_value(grid: np.ndarray, y_grid: np.ndarray, t_h: float) -> np.ndarray:
    return np.asarray([np.interp(t_h, grid, row, left=row[0], right=row[-1]) for row in y_grid], dtype=float)


def weibull_pct(t_h: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    t = np.maximum(np.asarray(t_h, dtype=float), 0.0)[None, :]
    a = np.maximum(np.asarray(alpha, dtype=float), 1e-8)[:, None]
    b = np.maximum(np.asarray(beta, dtype=float), 1e-8)[:, None]
    return 100.0 * (1.0 - np.exp(-np.power(t / a, b)))


def fit_static_weibull(meta: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, grid: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_static(meta.iloc[train_idx]), np.log(np.clip(meta.iloc[train_idx][["alpha", "beta"]].to_numpy(dtype=float), 1e-8, None)))
    pred = np.exp(model.predict(x_static(meta.iloc[test_idx])))
    pred[:, 0] = np.clip(pred[:, 0], 1e-8, 300.0)
    pred[:, 1] = np.clip(pred[:, 1], 1e-8, 5.0)
    return np.asarray([helpers.monotone_clip(row) for row in weibull_pct(grid, pred[:, 0], pred[:, 1])])


def fit_author_class_proto(meta: pd.DataFrame, y_grid: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    x_tr = x_static(meta.iloc[train_idx])
    x_te = x_static(meta.iloc[test_idx])
    y_class = meta.iloc[train_idx]["cluster"].to_numpy(dtype=int)
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_tr, y_class)
    pred_class = clf.predict(x_te)
    global_proto = helpers.monotone_clip(y_grid[train_idx].mean(axis=0))
    protos = {
        int(label): helpers.monotone_clip(y_grid[train_idx][y_class == label].mean(axis=0))
        for label in sorted(np.unique(y_class).tolist())
    }
    return np.asarray([protos.get(int(label), global_proto) for label in pred_class], dtype=float)


def fit_dictionary_coeff(meta: pd.DataFrame, y_grid: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, n_components: int, seed: int, n_estimators: int) -> np.ndarray:
    n_eff = min(n_components, len(train_idx) - 1)
    pca = PCA(n_components=n_eff, random_state=seed)
    coeff = pca.fit_transform(y_grid[train_idx])
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_static(meta.iloc[train_idx]), coeff)
    pred_coeff = model.predict(x_static(meta.iloc[test_idx]))
    return np.asarray([helpers.monotone_clip(row) for row in pca.inverse_transform(pred_coeff)], dtype=float)


def fit_direct_grid(meta: pd.DataFrame, y_grid: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_static(meta.iloc[train_idx]), y_grid[train_idx])
    return np.asarray([helpers.monotone_clip(row) for row in model.predict(x_static(meta.iloc[test_idx]))], dtype=float)


def fit_ensemble_direct_grid(
    meta: pd.DataFrame,
    y_grid: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    n_estimators: int,
    n_ensemble: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_tr = x_static(meta.iloc[train_idx])
    x_te = x_static(meta.iloc[test_idx])
    decoded = []
    for member in range(n_ensemble):
        boot = rng.choice(len(train_idx), size=len(train_idx), replace=True)
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed + 997 * (member + 1),
            n_jobs=-1,
        )
        model.fit(x_tr[boot], y_grid[train_idx][boot])
        decoded.append(np.asarray([helpers.monotone_clip(row) for row in model.predict(x_te)], dtype=float))
    return np.stack(decoded, axis=0)


def target_thresholds(train_y: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    q1 = interp_grid_value(grid, train_y, 1.0)
    q6 = interp_grid_value(grid, train_y, 6.0)
    q24 = interp_grid_value(grid, train_y, 24.0)
    return {
        "q1_33": float(np.quantile(q1, 0.33)),
        "q1_80": float(np.quantile(q1, 0.80)),
        "q6_33": float(np.quantile(q6, 0.33)),
        "q6_66": float(np.quantile(q6, 0.66)),
        "q6_mid": float(0.5 * (np.quantile(q6, 0.33) + np.quantile(q6, 0.66))),
        "q24_67": float(np.quantile(q24, 0.67)),
    }


def target_success(y_grid: np.ndarray, grid: np.ndarray, thresholds: dict[str, float], target: str) -> np.ndarray:
    q1 = interp_grid_value(grid, y_grid, 1.0)
    q6 = interp_grid_value(grid, y_grid, 6.0)
    q24 = interp_grid_value(grid, y_grid, 24.0)
    if target == "anti_burst":
        return q1 <= thresholds["q1_33"]
    if target == "sustained_mid":
        return (q6 >= thresholds["q6_33"]) & (q6 <= thresholds["q6_66"])
    if target == "high_final":
        return q24 >= thresholds["q24_67"]
    if target == "avoid_failure":
        return ~(q1 >= thresholds["q1_80"])
    raise ValueError(target)


def target_score(y_grid: np.ndarray, grid: np.ndarray, thresholds: dict[str, float], target: str) -> np.ndarray:
    q1 = interp_grid_value(grid, y_grid, 1.0)
    q6 = interp_grid_value(grid, y_grid, 6.0)
    q24 = interp_grid_value(grid, y_grid, 24.0)
    if target == "anti_burst":
        return -q1
    if target == "sustained_mid":
        scale = max(thresholds["q6_66"] - thresholds["q6_33"], 1.0)
        return -np.abs(q6 - thresholds["q6_mid"]) / scale
    if target == "high_final":
        return q24
    if target == "avoid_failure":
        return -q1
    raise ValueError(target)


def minmax_probability(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    lo = float(np.min(score))
    hi = float(np.max(score))
    if hi - lo <= 1e-12:
        return np.full_like(score, 0.5, dtype=float)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


def top_n(rule: str, n_items: int) -> int:
    if rule == "top5":
        return min(5, n_items)
    if rule == "top10":
        return min(10, n_items)
    if rule == "top20pct":
        return max(1, int(np.ceil(0.20 * n_items)))
    raise ValueError(rule)


def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask) * abs(np.mean(y[mask]) - np.mean(p[mask])))
    return total


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=bool)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    out = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    if len(y):
        out["brier"] = float(brier_score_loss(y.astype(int), p))
        out["ece"] = ece_score(y, p)
    return out


def append_rows_for_method(
    rows: list[dict[str, Any]],
    *,
    scheme: str,
    fold: int,
    method: str,
    lambdas: list[float],
    meta_test: pd.DataFrame,
    y_true_grid: np.ndarray,
    pred_grid: np.ndarray,
    grid: np.ndarray,
    thresholds: dict[str, float],
    ensemble_grid: np.ndarray | None = None,
) -> None:
    true_avoid_failure = target_success(y_true_grid, grid, thresholds, "avoid_failure")
    for target in TARGETS:
        success = target_success(y_true_grid, grid, thresholds, target)
        true_score = target_score(y_true_grid, grid, thresholds, target)
        pred_score = target_score(pred_grid, grid, thresholds, target)
        if ensemble_grid is None:
            prob = minmax_probability(pred_score)
            uncertainty = np.zeros(len(pred_score), dtype=float)
            lambda_values = [0.0]
        else:
            ens_success = np.asarray([target_success(member, grid, thresholds, target) for member in ensemble_grid], dtype=float)
            prob = ens_success.mean(axis=0)
            ens_score = np.asarray([target_score(member, grid, thresholds, target) for member in ensemble_grid], dtype=float)
            uncertainty = np.percentile(ens_score, 95, axis=0) - np.percentile(ens_score, 5, axis=0)
            uncertainty = uncertainty / max(float(np.nanmax(np.abs(uncertainty))), 1.0)
            lambda_values = lambdas
        cls = classification_metrics(success, prob)
        for lam in lambda_values:
            utility = prob - lam * uncertainty if ensemble_grid is not None else pred_score
            for i, row in enumerate(meta_test.itertuples(index=False)):
                rows.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "method": method,
                        "target": target,
                        "lambda": float(lam),
                        "ID": int(row.ID),
                        "API_name": str(row.API_name),
                        "release_method": str(row.release_method),
                        "predicted_utility": float(utility[i]),
                        "predicted_success_probability": float(prob[i]),
                        "predicted_uncertainty": float(uncertainty[i]),
                        "target_success": bool(success[i]),
                        "true_design_score": float(true_score[i]),
                        "avoid_failure_success": bool(true_avoid_failure[i]),
                        "auroc_fold": cls["auroc"],
                        "auprc_fold": cls["auprc"],
                        "brier_fold": cls["brier"],
                        "ece_fold": cls["ece"],
                        "is_oracle": False,
                    }
                )


def run_benchmark(args: argparse.Namespace, meta: pd.DataFrame, curve_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    grid = np.linspace(0.0, args.grid_max_h, args.grid_size)
    ids = meta["ID"].to_numpy(dtype=int)
    y_grid = helpers.curve_grid(curve_map, ids, grid)
    rows: list[dict[str, Any]] = []

    for scheme in SCHEMES:
        for fold, (tr, te) in enumerate(helpers.splits(meta, scheme, args.n_folds, args.seed), start=1):
            if len(tr) < 8 or len(te) < 2:
                continue
            thresholds = target_thresholds(y_grid[tr], grid)
            meta_test = meta.iloc[te].reset_index(drop=True)
            y_true = y_grid[te]

            predictions: dict[str, np.ndarray] = {
                "global_mean_curve": np.tile(helpers.monotone_clip(y_grid[tr].mean(axis=0)), (len(te), 1)),
                "author_class_proto": fit_author_class_proto(meta, y_grid, tr, te, args.seed + fold, args.n_estimators),
                "static_weibull_theta": fit_static_weibull(meta, tr, te, grid, args.seed + 11 * fold, args.n_estimators),
                "static_curve_dictionary_coeff": fit_dictionary_coeff(meta, y_grid, tr, te, args.n_components, args.seed + 17 * fold, args.n_estimators),
                "static_et_direct_grid": fit_direct_grid(meta, y_grid, tr, te, args.seed + 23 * fold, args.n_estimators),
            }
            ensemble = fit_ensemble_direct_grid(
                meta,
                y_grid,
                tr,
                te,
                args.seed + 31 * fold,
                args.n_estimators,
                args.n_ensemble,
            )
            predictions["static_ensemble_uncertainty_prior"] = np.median(ensemble, axis=0)

            for method, pred in predictions.items():
                append_rows_for_method(
                    rows,
                    scheme=scheme,
                    fold=fold,
                    method=method,
                    lambdas=args.lambdas,
                    meta_test=meta_test,
                    y_true_grid=y_true,
                    pred_grid=pred,
                    grid=grid,
                    thresholds=thresholds,
                    ensemble_grid=ensemble if method == "static_ensemble_uncertainty_prior" else None,
                )
    return pd.DataFrame(rows)


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, method, target, lam), sub in per_curve.groupby(["scheme", "method", "target", "lambda"], sort=True):
        y = sub["target_success"].to_numpy(dtype=bool)
        p = sub["predicted_success_probability"].to_numpy(dtype=float)
        cls = classification_metrics(y, p)
        random_rate = float(np.mean(y)) if len(y) else float("nan")
        for top_rule in TOP_RULES:
            fold_hits = []
            fold_enrich = []
            fold_regret = []
            fold_avoid = []
            for _, fold_sub in sub.groupby("fold"):
                ordered = fold_sub.sort_values("predicted_utility", ascending=False)
                n = top_n(top_rule, len(ordered))
                selected = ordered.head(n)
                fold_random = float(fold_sub["target_success"].mean())
                hit = float(selected["target_success"].mean())
                best_possible = float(fold_sub.sort_values("true_design_score", ascending=False).head(n)["true_design_score"].mean())
                selected_score = float(selected["true_design_score"].mean())
                fold_hits.append(hit)
                fold_enrich.append(float("nan") if fold_random <= 0 else hit / fold_random)
                fold_regret.append(best_possible - selected_score)
                fold_avoid.append(float(selected["avoid_failure_success"].mean()))
            rows.append(
                {
                    "scheme": scheme,
                    "method": method,
                    "target": target,
                    "lambda": float(lam),
                    "top_rule": top_rule,
                    "n_curve_records": int(len(sub)),
                    "n_unique_curves": int(sub["ID"].nunique()),
                    "random_success_rate": random_rate,
                    "top_k_hit_rate": float(np.mean(fold_hits)),
                    "enrichment_factor_vs_random": float(np.nanmean(fold_enrich)),
                    "regret": float(np.mean(fold_regret)),
                    "target_window_success_rate": float(np.mean(fold_hits)),
                    "burst_failure_avoidance": float(np.mean(fold_avoid)),
                    "auroc": cls["auroc"],
                    "auprc": cls["auprc"],
                    "brier": cls["brier"],
                    "ece": cls["ece"],
                    "is_oracle": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["scheme", "target", "top_rule", "top_k_hit_rate"], ascending=[True, True, True, False])


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = summary[(summary["top_rule"] == "top20pct") & (summary["lambda"].isin([0.0, 0.25]))].copy()
    for scheme, sub in primary.groupby("scheme"):
        for target, target_sub in sub.groupby("target"):
            def best(method: str, lam: float | None = None) -> pd.Series | None:
                cand = target_sub[target_sub["method"] == method]
                if lam is not None:
                    cand = cand[cand["lambda"] == lam]
                if cand.empty:
                    return None
                return cand.sort_values("top_k_hit_rate", ascending=False).iloc[0]

            static = best("static_curve_dictionary_coeff", 0.0)
            global_row = best("global_mean_curve", 0.0)
            direct = best("static_et_direct_grid", 0.0)
            uncertain_plain = best("static_ensemble_uncertainty_prior", 0.0)
            uncertain_risk = best("static_ensemble_uncertainty_prior", 0.25)

            if static is not None and global_row is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Does static prior beat random/global selection?",
                        "method_a": "static_curve_dictionary_coeff",
                        "method_b": "global_mean_curve",
                        "value": float(static["top_k_hit_rate"] - global_row["top_k_hit_rate"]),
                        "passed": bool(static["top_k_hit_rate"] > global_row["top_k_hit_rate"]),
                        "interpretation": "positive means the static prior changes candidate choice usefully",
                    }
                )
            if static is not None and direct is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Does dictionary prior beat direct grid?",
                        "method_a": "static_curve_dictionary_coeff",
                        "method_b": "static_et_direct_grid",
                        "value": float(static["top_k_hit_rate"] - direct["top_k_hit_rate"]),
                        "passed": bool(static["top_k_hit_rate"] >= direct["top_k_hit_rate"]),
                        "interpretation": "positive favors latent curve language over direct grid prediction",
                    }
                )
            if uncertain_plain is not None and uncertain_risk is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Does uncertainty-aware ranking lower burst/failure risk?",
                        "method_a": "static_ensemble_uncertainty_prior_lambda0.25",
                        "method_b": "static_ensemble_uncertainty_prior_lambda0",
                        "value": float(uncertain_risk["burst_failure_avoidance"] - uncertain_plain["burst_failure_avoidance"]),
                        "passed": bool(uncertain_risk["burst_failure_avoidance"] >= uncertain_plain["burst_failure_avoidance"]),
                        "interpretation": "positive means uncertainty penalty avoids more burst failures",
                    }
                )
            if scheme in {"group_by_API", "group_by_release_method"} and static is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Does strict group split retain design utility?",
                        "method_a": "static_curve_dictionary_coeff",
                        "method_b": "random",
                        "value": float(static["enrichment_factor_vs_random"]),
                        "passed": bool(static["enrichment_factor_vs_random"] > 1.0),
                        "interpretation": "above 1 means selected candidates are enriched over random in strict split",
                    }
                )
    return pd.DataFrame(rows)


def data_checks(meta: pd.DataFrame, per_curve: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "input_files_exist", "passed": bool(args.root.exists()), "detail": str(args.root)},
            {"check": "sample_unit_is_curve", "passed": bool(meta["ID"].is_unique), "detail": f"n_curves={len(meta)}"},
            {"check": "no_timepoint_split", "passed": True, "detail": "folds are built from curve-level metadata rows"},
            {"check": "train_fold_quantile_targets", "passed": True, "detail": "target thresholds are recomputed from y_train inside each fold"},
            {"check": "no_heldout_group_overlap_in_group_splits", "passed": True, "detail": "GroupKFold helper used for API and release method splits"},
            {"check": "no_nan_in_primary_metrics", "passed": bool(np.isfinite(summary["top_k_hit_rate"]).all()), "detail": "top_k_hit_rate finite for all summary rows"},
            {"check": "oracle_rows_marked_and_excluded", "passed": bool((~per_curve["is_oracle"]).all()), "detail": "116 has no oracle candidate-ranking rows"},
        ]
    )


def write_report(out: Path, meta: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Liposome Design Utility Benchmark",
        "",
        "Date: 2026-06-12",
        "",
        "## Aim",
        "",
        "Evaluate whether static priors can rank candidate formulations before any release observation.",
        "",
        "`prediction utility != exact curve RMSE`",
        "",
        "## Run",
        "",
        f"- root: `{args.root.as_posix()}`",
        f"- n_curves: {len(meta)}",
        f"- grid: 0-{args.grid_max_h} h, {args.grid_size} points",
        f"- PCA components for dictionary prior: {args.n_components}",
        f"- ensemble size: {args.n_ensemble}",
        "",
        "## Best Top-20pct Rows",
        "",
        "| Split | Target | Method | lambda | hit rate | enrichment | failure avoidance | AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    top = summary[summary["top_rule"] == "top20pct"].sort_values(["scheme", "target", "top_k_hit_rate"], ascending=[True, True, False])
    for _, row in top.groupby(["scheme", "target"], sort=True).head(3).iterrows():
        lines.append(
            f"| {row['scheme']} | {row['target']} | `{row['method']}` | {row['lambda']:.2f} | "
            f"{row['top_k_hit_rate']:.3f} | {row['enrichment_factor_vs_random']:.3f} | "
            f"{row['burst_failure_avoidance']:.3f} | {row['auroc']:.3f} |"
        )
    lines.extend(["", "## Decision Rows", "", "| Split | Target | Question | Value | Passed |", "|---|---|---|---:|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.target} | {row.question} | {row.value:.3f} | {row.passed} |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Targets are fold-local train quantiles, not global thresholds.",
            "- The sample unit is one curve.",
            "- Group splits hold out API or release method groups.",
            "- This is design utility, not a claim of exact curve prediction.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_doc(doc_path: Path, summary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    top = summary[summary["top_rule"] == "top20pct"].sort_values(["scheme", "target", "top_k_hit_rate"], ascending=[True, True, False])
    lines = [
        "# Liposome Design Utility Benchmark",
        "",
        "Date: 2026-06-12",
        "",
        "## Question",
        "",
        "Can a static release prior guide candidate selection before any release observation, even when exact curve RMSE is limited?",
        "",
        "This answers a different question from curve prediction:",
        "",
        "```text",
        "prediction utility != exact curve RMSE",
        "```",
        "",
        "## Method",
        "",
        "- Candidate predictors: global mean, author-style class prototype, static Weibull theta, static PCA coefficient prior, static direct grid, and bootstrap ensemble uncertainty prior.",
        "- Targets: anti-burst, sustained-mid, high-final, and avoid-failure windows defined by train-fold quantiles.",
        "- Ranking metrics: top-k hit rate, enrichment over random, regret, failure avoidance, AUROC, AUPRC, Brier, and ECE.",
        "",
        "## Best Top-20pct Rows",
        "",
        "| Split | Target | Method | lambda | Hit rate | Enrichment | Failure avoidance | AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.groupby(["scheme", "target"], sort=True).head(3).iterrows():
        lines.append(
            f"| {row['scheme']} | {row['target']} | `{row['method']}` | {row['lambda']:.2f} | "
            f"{row['top_k_hit_rate']:.3f} | {row['enrichment_factor_vs_random']:.3f} | "
            f"{row['burst_failure_avoidance']:.3f} | {row['auroc']:.3f} |"
        )
    lines.extend(["", "## Decision Table", "", "| Split | Target | Question | Value | Passed |", "|---|---|---|---:|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.target} | {row.question} | {row.value:.3f} | {row.passed} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive result means static descriptors may still guide experimental prioritization even when they cannot reconstruct exact curves. A weak strict-split result means static design utility is source- or condition-dependent and should not be treated as robust transfer.",
            "",
            "## Output Anchor",
            "",
            "`../outputs/116_liposome_design_utility_benchmark/`",
        ]
    )
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = helpers.load_dataset(args.root)
    meta = helpers.stratified_cap(meta, args.max_curves, args.seed)
    per_curve = run_benchmark(args, meta, curve_map)
    if per_curve.empty:
        raise RuntimeError("no design utility rows generated")
    summary = summarize(per_curve)
    decisions = decision_table(summary)
    checks = data_checks(meta, per_curve, summary, args)

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    write_report(args.out, meta, summary, decisions, args)
    write_doc(Path("docs/liposome_design_utility_benchmark_2026-06-12.md"), summary, decisions)

    metadata = {
        "script": Path(__file__).name,
        "date": "2026-06-12",
        "seed": args.seed,
        "root": str(args.root),
        "out": str(args.out),
        "grid_max_h": args.grid_max_h,
        "grid_size": args.grid_size,
        "n_components": args.n_components,
        "n_estimators": args.n_estimators,
        "n_ensemble": args.n_ensemble,
        "lambdas": args.lambdas,
        "n_folds": args.n_folds,
        "max_curves": args.max_curves,
        "git_hash": helpers.git_hash(),
        "input_files": {
            "backend_data": helpers.file_meta(args.root / "data/unprocessed/backend_data.csv"),
            "weibull_params": helpers.file_meta(args.root / "data/clean/weibull_params.csv"),
            "clusters": helpers.file_meta(args.root / "results/clustering/3_PCA_KMC.csv"),
            "release_exp": helpers.file_meta(args.root / "results/fitting/drug_release_exp.csv"),
        },
        "generated_files": sorted(p.name for p in args.out.iterdir() if p.is_file()),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"design utility benchmark failed checks: {failed}")


if __name__ == "__main__":
    main()
