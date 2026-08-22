#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.decomposition import NMF
from sklearn.model_selection import GroupKFold


THIS_DIR = Path(__file__).resolve().parent
ALT_PATH = THIS_DIR / "run_alternative_curve_routes.py"
spec = importlib.util.spec_from_file_location("alt_routes", str(ALT_PATH))
alt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(alt)  # type: ignore[attr-defined]

FEATURES = alt.FEATURES
GROUP_COL = alt.GROUP_COL
TIME_COL = alt.TIME_COL
Y_COL = alt.Y_COL


def _train_lgbm(x: pd.DataFrame, y: np.ndarray, seed: int) -> LGBMRegressor:
    params = dict(alt.LGBM_PARAMS)
    params["random_state"] = int(seed)
    model = LGBMRegressor(**params)
    model.fit(x, y)
    return model


def _groupwise_rmse(pred_df: pd.DataFrame, pred_col: str) -> float:
    rows = []
    for _, g in pred_df.groupby(GROUP_COL, sort=False):
        rows.append(alt.metric_row(g[Y_COL].to_numpy(dtype=float), g[pred_col].to_numpy(dtype=float))["rmse"])
    return float(np.mean(rows))


def _predict_fpca(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_groups: list[str],
    *,
    k: int,
    n_grid: int,
    seed: int,
    monotone: bool = False,
) -> np.ndarray:
    grid, pca, models, _ = alt.fit_functional_pca_models(
        train_df=train_df,
        feature_df=feature_df,
        train_groups=train_groups,
        seed=seed,
        n_grid=n_grid,
        n_components=k,
    )
    return alt.predict_functional_pca_for_points(valid_df, feature_df, grid, pca, models, monotone=monotone).to_numpy(dtype=float)


def nested_select_k(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_groups: list[str],
    *,
    candidates: list[int],
    n_grid: int,
    seed: int,
    inner_splits: int,
) -> tuple[int, pd.DataFrame]:
    groups = np.array(sorted([str(g) for g in train_groups]), dtype=object)
    inner_splits = min(int(inner_splits), len(groups))
    gkf = GroupKFold(n_splits=inner_splits)
    rows = []
    for k in candidates:
        for inner_fold, (tr_idx, va_idx) in enumerate(gkf.split(groups, groups=groups), start=1):
            tr_groups = groups[tr_idx].tolist()
            va_groups = groups[va_idx].tolist()
            inner_train = train_df[train_df[GROUP_COL].isin(tr_groups)].copy()
            inner_valid = train_df[train_df[GROUP_COL].isin(va_groups)].copy()
            pred = _predict_fpca(
                inner_train,
                inner_valid,
                feature_df,
                tr_groups,
                k=k,
                n_grid=n_grid,
                seed=seed + 1000 * inner_fold + k,
                monotone=False,
            )
            tmp = inner_valid[[GROUP_COL, Y_COL]].copy()
            tmp["pred"] = pred
            rows.append({"k": int(k), "inner_fold": inner_fold, "groupwise_rmse": _groupwise_rmse(tmp, "pred")})
    score_df = pd.DataFrame(rows)
    mean_scores = score_df.groupby("k", as_index=False)["groupwise_rmse"].mean().sort_values(["groupwise_rmse", "k"])
    return int(mean_scores.iloc[0]["k"]), score_df


def _curve_matrix_monotone_increments(df: pd.DataFrame, groups: list[str], grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    curves, ok_groups = alt.build_curve_matrix(df, groups, grid)
    curves = np.vstack([alt.monotone_clip(row) for row in curves])
    increments = np.diff(np.column_stack([np.zeros(len(curves)), curves]), axis=1)
    increments = np.clip(increments, 0.0, None)
    return increments, ok_groups


def fit_nmf_increment_basis(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_groups: list[str],
    *,
    k: int,
    n_grid: int,
    seed: int,
) -> tuple[np.ndarray, NMF, dict[int, LGBMRegressor], list[str]]:
    grid = alt.make_log_grid(train_df, n_grid)
    inc, ok_groups = _curve_matrix_monotone_increments(train_df, train_groups, grid)
    n_comp = min(int(k), inc.shape[0] - 1, inc.shape[1])
    nmf = NMF(n_components=n_comp, init="nndsvda", random_state=seed, max_iter=2000, tol=1e-5)
    weights = nmf.fit_transform(inc)
    train_meta = feature_df[feature_df[GROUP_COL].isin(ok_groups)].set_index(GROUP_COL).loc[ok_groups].reset_index()
    models = {}
    for j in range(n_comp):
        models[j] = _train_lgbm(train_meta[FEATURES], weights[:, j], seed + 200 + j)
    return grid, nmf, models, ok_groups


def predict_nmf_increment_basis(
    point_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    grid: np.ndarray,
    nmf: NMF,
    models: dict[int, LGBMRegressor],
) -> pd.Series:
    rows = []
    meta = feature_df.set_index(GROUP_COL)
    for gid, g in point_df.groupby(GROUP_COL, sort=False):
        x = meta.loc[str(gid), FEATURES].to_frame().T.astype(float)
        weights = np.array([[max(0.0, float(models[j].predict(x)[0])) for j in sorted(models)]], dtype=float)
        inc = np.clip(nmf.inverse_transform(weights).reshape(-1), 0.0, None)
        grid_pred = np.clip(np.cumsum(inc), 0.0, 1.2)
        t = g[TIME_COL].to_numpy(dtype=float)
        y = np.interp(t, grid, grid_pred, left=grid_pred[0], right=grid_pred[-1])
        rows.append(pd.Series(y, index=g.index))
    return pd.concat(rows).sort_index()


def paired(group_metrics: pd.DataFrame, locked_group_path: Path) -> pd.DataFrame:
    wide = group_metrics.pivot(index=GROUP_COL, columns="method", values="rmse")
    locked = pd.read_csv(locked_group_path)
    locked[GROUP_COL] = locked[GROUP_COL].astype(str)
    rows = []
    for baseline_key, baseline_name in [("direct_lgbm", "Locked Direct LGBM"), ("prior_residual_bridge", "Locked prior-plus-residual MEP")]:
        base = locked[locked["method"].astype(str) == baseline_key].set_index(GROUP_COL)["rmse"].astype(float)
        for challenger in wide.columns:
            joined = pd.concat([base.rename("baseline"), wide[challenger].rename("challenger")], axis=1).dropna()
            diff = joined["baseline"] - joined["challenger"]
            try:
                _, p = wilcoxon(diff)
            except ValueError:
                p = np.nan
            rows.append(
                {
                    "baseline": baseline_name,
                    "challenger": challenger,
                    "n_groups": int(len(joined)),
                    "mean_baseline_rmse": float(joined["baseline"].mean()),
                    "mean_challenger_rmse": float(joined["challenger"].mean()),
                    "mean_diff_baseline_minus_challenger": float(diff.mean()),
                    "median_diff_baseline_minus_challenger": float(diff.median()),
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "better_lower_rmse": baseline_name if joined["baseline"].mean() < joined["challenger"].mean() else challenger,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=r"D:\最终版框架\dataset\Dataset_17_feat_augmented.csv")
    ap.add_argument("--out_dir", default=r"D:\诊断论文\results\fpca_nested_monotone_20260428")
    ap.add_argument("--n_grid", type=int, default=96)
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_candidates", default="4,5,7,10,12,16,18,20")
    ap.add_argument("--nmf_k", default="5,8,10,12,16")
    ap.add_argument("--locked_summary", default=r"D:\最终版框架\project\results\mep_v1_direct10_unified_baseline_20260413_run1\metrics_summary.csv")
    ap.add_argument("--locked_pointwise", default=r"D:\最终版框架\project\results\mep_v1_direct10_unified_baseline_20260413_run1\metrics_summary_pointwise.csv")
    ap.add_argument("--locked_group_metrics", default=r"D:\最终版框架\project\results\mep_v1_direct10_unified_baseline_20260413_run1\metrics_by_group.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = alt.prepare_dataset(Path(args.input_csv))
    feature_df = alt.curve_features(df)
    groups = np.array(sorted(df[GROUP_COL].astype(str).unique()), dtype=object)
    k_candidates = [int(x.strip()) for x in args.k_candidates.split(",") if x.strip()]
    nmf_ks = [int(x.strip()) for x in args.nmf_k.split(",") if x.strip()]

    fold_frames = []
    nested_rows = []
    gkf = GroupKFold(n_splits=int(args.outer_splits))
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(groups, groups=groups), start=1):
        train_groups = groups[tr_idx].tolist()
        test_groups = groups[te_idx].tolist()
        train_df = df[df[GROUP_COL].isin(train_groups)].copy()
        test_df = df[df[GROUP_COL].isin(test_groups)].copy()
        out = test_df.copy()
        out["fold"] = fold

        selected_k, nested = nested_select_k(
            train_df,
            feature_df,
            train_groups,
            candidates=k_candidates,
            n_grid=int(args.n_grid),
            seed=int(args.seed) + 5000 * fold,
            inner_splits=int(args.inner_splits),
        )
        nested["outer_fold"] = fold
        nested_rows.append(nested)
        out["fpca_nested_selected_k"] = selected_k
        out["pred_fpca_nested_k"] = _predict_fpca(
            train_df,
            test_df,
            feature_df,
            train_groups,
            k=selected_k,
            n_grid=int(args.n_grid),
            seed=int(args.seed) + 7000 * fold + selected_k,
            monotone=False,
        )

        for k in nmf_ks:
            grid, nmf, models, _ = fit_nmf_increment_basis(
                train_df,
                feature_df,
                train_groups,
                k=k,
                n_grid=int(args.n_grid),
                seed=int(args.seed) + 9000 * fold + k,
            )
            out[f"pred_monotone_nmf_k{k}"] = predict_nmf_increment_basis(test_df, feature_df, grid, nmf, models).to_numpy(dtype=float)
        fold_frames.append(out)

    pred_df = pd.concat(fold_frames, ignore_index=True)
    method_cols = {"nested fPCA LGBM": "pred_fpca_nested_k"}
    for k in nmf_ks:
        method_cols[f"monotone NMF-increment LGBM k={k}"] = f"pred_monotone_nmf_k{k}"
    group_metrics = alt.per_group_metrics(pred_df, method_cols)
    summary = alt.summarize(group_metrics, pred_df, method_cols)
    locked_rows = []
    for key, label in [("direct_lgbm", "Locked Direct LGBM"), ("prior_residual_bridge", "Locked prior-plus-residual MEP")]:
        row = alt.load_locked_summary(Path(args.locked_summary), Path(args.locked_pointwise), key, label)
        if row is not None:
            locked_rows.append(row)
    summary_with_locked = pd.concat([pd.DataFrame(locked_rows), summary], ignore_index=True).sort_values("groupwise_rmse")
    tests = paired(group_metrics, Path(args.locked_group_metrics))
    nested_scores = pd.concat(nested_rows, ignore_index=True)
    selected = pred_df.groupby("fold", as_index=False)["fpca_nested_selected_k"].first()

    pred_df.to_csv(out_dir / "predictions_wide.csv", index=False, encoding="utf-8-sig")
    group_metrics.to_csv(out_dir / "metrics_by_group.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary_experimental_methods.csv", index=False, encoding="utf-8-sig")
    summary_with_locked.to_csv(out_dir / "summary_with_locked_baselines.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out_dir / "paired_tests.csv", index=False, encoding="utf-8-sig")
    nested_scores.to_csv(out_dir / "nested_k_inner_scores.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "nested_k_selected_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(Path(args.input_csv)),
                "outer_split": f"GroupKFold-{args.outer_splits}",
                "inner_split": f"GroupKFold-{args.inner_splits}",
                "group_key": GROUP_COL,
                "n_grid": int(args.n_grid),
                "k_candidates": k_candidates,
                "nmf_k": nmf_ks,
                "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] wrote outputs to {out_dir}")
    print(summary_with_locked.to_string(index=False))
    print("\nSelected k by outer fold")
    print(selected.to_string(index=False))
    print("\nPaired tests")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
