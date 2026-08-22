#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold


FEATURES = [
    "Polymer_MW",
    "Drug_Mw",
    "Drug_LogP",
    "SA_V",
    "LA/GA",
    "Drug_Pka",
    "DLC",
    "Initial D/M ratio",
    "Drug_TPSA",
]
GROUP_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"

LGBM_PARAMS = dict(
    n_estimators=400,
    max_depth=-1,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
    n_jobs=1,
    force_row_wise=True,
)


def metric_row(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
    yt = yt[mask]
    yp = yp[mask]
    resid = yt - yp
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / denom) if denom > 0 else np.nan
    return {"r2": r2, "rmse": rmse, "mae": mae}


def prepare_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "SA-V" in df.columns and "SA_V" not in df.columns:
        df["SA_V"] = df["SA-V"]
    required = [GROUP_COL, TIME_COL, Y_COL, *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    for col in [TIME_COL, Y_COL, *FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[GROUP_COL, TIME_COL, Y_COL, *FEATURES]).copy()


def curve_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in df.groupby(GROUP_COL, sort=False):
        row0 = g.sort_values(TIME_COL).iloc[0]
        rec = {GROUP_COL: str(gid), "subgroup": "SA" if float(row0["LA/GA"]) == 0.0 else "OLA"}
        for col in FEATURES:
            rec[col] = float(row0[col])
        rows.append(rec)
    return pd.DataFrame(rows)


def make_log_grid(train_df: pd.DataFrame, n_grid: int) -> np.ndarray:
    max_t = float(np.nanmax(train_df[TIME_COL].to_numpy(dtype=float)))
    return np.expm1(np.linspace(0.0, np.log1p(max(max_t, 1.0)), int(n_grid)))


def interpolate_curve(g: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    gs = g.sort_values(TIME_COL)
    t = gs[TIME_COL].to_numpy(dtype=float)
    y = np.clip(gs[Y_COL].to_numpy(dtype=float), 0.0, 1.2)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    unique_t, unique_idx = np.unique(t, return_index=True)
    unique_y = y[unique_idx]
    return np.interp(grid, unique_t, unique_y, left=unique_y[0], right=unique_y[-1])


def build_curve_matrix(df: pd.DataFrame, groups: list[str], grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rows = []
    ok_groups = []
    grouped = {str(k): g for k, g in df.groupby(GROUP_COL, sort=False)}
    for gid in groups:
        g = grouped[str(gid)]
        if len(g) < 2:
            continue
        rows.append(interpolate_curve(g, grid))
        ok_groups.append(str(gid))
    return np.vstack(rows), ok_groups


def train_lgbm(x: pd.DataFrame, y: np.ndarray, seed: int) -> LGBMRegressor:
    params = dict(LGBM_PARAMS)
    params["random_state"] = int(seed)
    model = LGBMRegressor(**params)
    model.fit(x, y)
    return model


def monotone_clip(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.clip(np.asarray(y, dtype=float), 0.0, 1.2))


def run_direct(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    model = train_lgbm(train_df[[*FEATURES, TIME_COL]], train_df[Y_COL].to_numpy(dtype=float), seed)
    return np.clip(model.predict(test_df[[*FEATURES, TIME_COL]]), 0.0, 1.2)


def run_mean_curve_residual(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_groups: list[str],
    seed: int,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = make_log_grid(train_df, n_grid)
    train_matrix, _ = build_curve_matrix(train_df, train_groups, grid)
    mean_curve = np.mean(train_matrix, axis=0)

    train_prior = np.interp(train_df[TIME_COL].to_numpy(dtype=float), grid, mean_curve, left=mean_curve[0], right=mean_curve[-1])
    x_train = train_df[[*FEATURES, TIME_COL]].copy()
    x_train["mean_curve_prior"] = train_prior
    residual = train_df[Y_COL].to_numpy(dtype=float) - train_prior
    model = train_lgbm(x_train, residual, seed)

    test_prior = np.interp(test_df[TIME_COL].to_numpy(dtype=float), grid, mean_curve, left=mean_curve[0], right=mean_curve[-1])
    x_test = test_df[[*FEATURES, TIME_COL]].copy()
    x_test["mean_curve_prior"] = test_prior
    pred = test_prior + model.predict(x_test)
    return np.clip(pred, 0.0, 1.2), np.clip(test_prior, 0.0, 1.2)


def fit_functional_pca_models(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_groups: list[str],
    seed: int,
    n_grid: int,
    n_components: int,
) -> tuple[np.ndarray, PCA, dict[int, LGBMRegressor], list[str]]:
    grid = make_log_grid(train_df, n_grid)
    matrix, ok_groups = build_curve_matrix(train_df, train_groups, grid)
    n_comp = min(int(n_components), matrix.shape[0] - 1, matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    scores = pca.fit_transform(matrix)
    train_meta = feature_df[feature_df[GROUP_COL].isin(ok_groups)].set_index(GROUP_COL).loc[ok_groups].reset_index()
    models = {}
    for j in range(n_comp):
        models[j] = train_lgbm(train_meta[FEATURES], scores[:, j], seed + 100 + j)
    return grid, pca, models, ok_groups


def predict_functional_pca_for_points(
    point_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    grid: np.ndarray,
    pca: PCA,
    models: dict[int, LGBMRegressor],
    monotone: bool,
) -> pd.Series:
    rows = []
    meta = feature_df.set_index(GROUP_COL)
    for gid, g in point_df.groupby(GROUP_COL, sort=False):
        x = meta.loc[str(gid), FEATURES].to_frame().T.astype(float)
        scores = np.array([[models[j].predict(x)[0] for j in sorted(models)]], dtype=float)
        grid_pred = pca.inverse_transform(scores).reshape(-1)
        if monotone:
            grid_pred = monotone_clip(grid_pred)
        else:
            grid_pred = np.clip(grid_pred, 0.0, 1.2)
        t = g[TIME_COL].to_numpy(dtype=float)
        y = np.interp(t, grid, grid_pred, left=grid_pred[0], right=grid_pred[-1])
        rows.append(pd.Series(y, index=g.index))
    return pd.concat(rows).sort_index()


def per_group_metrics(pred_df: pd.DataFrame, method_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for gid, g in pred_df.groupby(GROUP_COL, sort=False):
        subgroup = "SA" if float(g["LA/GA"].iloc[0]) == 0.0 else "OLA"
        y = g[Y_COL].to_numpy(dtype=float)
        for method, col in method_cols.items():
            rows.append(
                {
                    GROUP_COL: str(gid),
                    "fold": int(g["fold"].iloc[0]),
                    "subgroup": subgroup,
                    "method": method,
                    "n_points": int(len(g)),
                    **metric_row(y, g[col].to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(rows)


def summarize(group_metrics: pd.DataFrame, pred_df: pd.DataFrame, method_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for method, col in method_cols.items():
        g = group_metrics[group_metrics["method"] == method]
        micro = metric_row(pred_df[Y_COL].to_numpy(dtype=float), pred_df[col].to_numpy(dtype=float))
        rows.append(
            {
                "method": method,
                "groupwise_macro_r2": float(g["r2"].mean()),
                "groupwise_rmse": float(g["rmse"].mean()),
                "groupwise_mae": float(g["mae"].mean()),
                "median_per_formulation_rmse": float(g["rmse"].median()),
                "pointwise_micro_r2": micro["r2"],
                "pointwise_micro_rmse": micro["rmse"],
                "pointwise_micro_mae": micro["mae"],
                "n_groups": int(g[GROUP_COL].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("groupwise_rmse")


def load_locked_summary(path: Path, pointwise_path: Path | None, method_key: str, label: str) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[(df["method"].astype(str) == method_key) & (df["subgroup"].astype(str) == "ALL")]
    if row.empty:
        return None
    r = row.iloc[0]
    out = {
        "method": label,
        "groupwise_macro_r2": float(r["r2"]),
        "groupwise_rmse": float(r["rmse"]),
        "groupwise_mae": float(r["mae"]),
        "median_per_formulation_rmse": np.nan,
        "pointwise_micro_r2": np.nan,
        "pointwise_micro_rmse": np.nan,
        "pointwise_micro_mae": np.nan,
        "n_groups": 181,
    }
    if pointwise_path is not None and pointwise_path.exists():
        pw = pd.read_csv(pointwise_path)
        prow = pw[(pw["method"].astype(str) == method_key) & (pw["subgroup"].astype(str) == "ALL")]
        if not prow.empty:
            pr = prow.iloc[0]
            out.update(
                {
                    "pointwise_micro_r2": float(pr["r2"]),
                    "pointwise_micro_rmse": float(pr["rmse"]),
                    "pointwise_micro_mae": float(pr["mae"]),
                }
            )
    return out


def load_group_rmse(path: Path, method_key: str) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sub = df[df["method"].astype(str) == method_key].copy()
    if sub.empty:
        return None
    sub[GROUP_COL] = sub[GROUP_COL].astype(str)
    return sub.set_index(GROUP_COL)["rmse"].astype(float)


def paired_tests(group_metrics: pd.DataFrame, external: dict[str, pd.Series], challenger: str) -> pd.DataFrame:
    wide = group_metrics.pivot(index=GROUP_COL, columns="method", values="rmse")
    rows = []
    for baseline in ["Locked Direct LGBM", "Locked prior-plus-residual MEP", "OOF Direct LGBM"]:
        if baseline in external:
            a = external[baseline]
        elif baseline in wide.columns:
            a = wide[baseline]
        else:
            continue
        b = wide[challenger]
        joined = pd.concat([a.rename("baseline"), b.rename("challenger")], axis=1).dropna()
        if joined.empty:
            continue
        diff = joined["baseline"] - joined["challenger"]
        try:
            stat, p = wilcoxon(diff)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append(
            {
                "baseline": baseline,
                "challenger": challenger,
                "n_groups": int(len(joined)),
                "mean_baseline_rmse": float(joined["baseline"].mean()),
                "mean_challenger_rmse": float(joined["challenger"].mean()),
                "mean_diff_baseline_minus_challenger": float(diff.mean()),
                "median_diff_baseline_minus_challenger": float(diff.median()),
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "better_lower_rmse": baseline if joined["baseline"].mean() < joined["challenger"].mean() else challenger,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=r"D:\最终版框架\dataset\Dataset_17_feat_augmented.csv")
    ap.add_argument("--out_dir", default=r"D:\诊断论文\results\alternative_curve_routes_20260428")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_grid", type=int, default=64)
    ap.add_argument("--pca_components", default="2,3,5,8,12,16")
    ap.add_argument("--locked_metrics_summary", default=r"D:\最终版框架\project\results\mep_v1_direct10_unified_baseline_20260413_run1\metrics_summary.csv")
    ap.add_argument("--locked_pointwise_summary", default=r"D:\最终版框架\project\results\mep_v1_direct10_unified_baseline_20260413_run1\metrics_summary_pointwise.csv")
    ap.add_argument("--locked_group_metrics", default=r"D:\最终版框架\project\results\mep_v1_direct10_unified_baseline_20260413_run1\metrics_by_group.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_dataset(Path(args.input_csv))
    feature_df = curve_features(df)
    groups = np.array(sorted(df[GROUP_COL].astype(str).unique()), dtype=object)
    pca_components = [int(x.strip()) for x in str(args.pca_components).split(",") if x.strip()]

    gkf = GroupKFold(n_splits=int(args.n_splits))
    fold_frames = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(groups, groups=groups), start=1):
        train_groups = groups[tr_idx].tolist()
        test_groups = groups[te_idx].tolist()
        train_df = df[df[GROUP_COL].isin(train_groups)].copy()
        test_df = df[df[GROUP_COL].isin(test_groups)].copy()
        out = test_df.copy()
        out["fold"] = fold
        out["pred_oof_direct_lgbm"] = run_direct(train_df, test_df, args.seed + fold)
        residual_pred, mean_prior = run_mean_curve_residual(train_df, test_df, train_groups, args.seed + 10 * fold, args.n_grid)
        out["pred_training_mean_curve"] = mean_prior
        out["pred_mean_curve_residual_lgbm"] = residual_pred

        for k in pca_components:
            grid, pca, models, _ = fit_functional_pca_models(
                train_df,
                feature_df,
                train_groups,
                args.seed + 100 * fold + k,
                args.n_grid,
                k,
            )
            out[f"pred_fpca_lgbm_k{k}"] = predict_functional_pca_for_points(
                out, feature_df, grid, pca, models, monotone=False
            )
            out[f"pred_fpca_lgbm_k{k}_monotone"] = predict_functional_pca_for_points(
                out, feature_df, grid, pca, models, monotone=True
            )
        fold_frames.append(out)

    pred_df = pd.concat(fold_frames, ignore_index=True)
    method_cols = {
        "OOF Direct LGBM": "pred_oof_direct_lgbm",
        "training mean curve baseline": "pred_training_mean_curve",
        "mean-curve residual LGBM": "pred_mean_curve_residual_lgbm",
    }
    for k in pca_components:
        method_cols[f"functional PCA LGBM k={k}"] = f"pred_fpca_lgbm_k{k}"
        method_cols[f"functional PCA LGBM k={k} monotone"] = f"pred_fpca_lgbm_k{k}_monotone"

    group_metrics = per_group_metrics(pred_df, method_cols)
    summary = summarize(group_metrics, pred_df, method_cols)

    locked_rows = []
    locked_direct = load_locked_summary(
        Path(args.locked_metrics_summary), Path(args.locked_pointwise_summary), "direct_lgbm", "Locked Direct LGBM"
    )
    locked_prior = load_locked_summary(
        Path(args.locked_metrics_summary),
        Path(args.locked_pointwise_summary),
        "prior_residual_bridge",
        "Locked prior-plus-residual MEP",
    )
    for row in [locked_direct, locked_prior]:
        if row is not None:
            locked_rows.append(row)
    if locked_rows:
        summary_with_locked = pd.concat([pd.DataFrame(locked_rows), summary], ignore_index=True)
    else:
        summary_with_locked = summary.copy()
    summary_with_locked = summary_with_locked.sort_values("groupwise_rmse").reset_index(drop=True)

    best_challenger = summary.loc[summary["method"] != "OOF Direct LGBM"].sort_values("groupwise_rmse").iloc[0]["method"]
    external = {}
    direct_rmse = load_group_rmse(Path(args.locked_group_metrics), "direct_lgbm")
    prior_rmse = load_group_rmse(Path(args.locked_group_metrics), "prior_residual_bridge")
    if direct_rmse is not None:
        external["Locked Direct LGBM"] = direct_rmse
    if prior_rmse is not None:
        external["Locked prior-plus-residual MEP"] = prior_rmse
    tests = paired_tests(group_metrics, external, str(best_challenger))

    pred_df.to_csv(out_dir / "predictions_wide.csv", index=False, encoding="utf-8-sig")
    group_metrics.to_csv(out_dir / "metrics_by_group.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary_experimental_methods.csv", index=False, encoding="utf-8-sig")
    summary_with_locked.to_csv(out_dir / "summary_with_locked_baselines.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out_dir / "paired_tests_best_challenger.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "input_csv": str(Path(args.input_csv)),
        "split": f"GroupKFold-{args.n_splits}",
        "group_key": GROUP_COL,
        "features": FEATURES,
        "n_grid": int(args.n_grid),
        "pca_components": pca_components,
        "best_non_direct_method": str(best_challenger),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote outputs to {out_dir}")
    print(summary_with_locked.to_string(index=False))
    print("\nBest non-direct:", best_challenger)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
