"""
12 — fPCA + Direct LGBM comparators (ADR-023).

Reimplements two non-bridge baselines from the user's diagnostic paper so
release-foundation has self-contained comparator numbers for the bridge
collapse hypothesis:

    fPCA LGBM    : x -> fPCA scores -> inverse PCA -> curve
    Direct LGBM  : (x, t) -> Q(t)   (no curve representation)

Two run modes:

  --target internal-181 (default)
      GroupKFold-5 by Experimental_index on data/Dataset_17_feat_augmented.csv,
      9 features. Outer 5-fold; inner GroupKFold-3 for fPCA k selection
      from k in {4,5,7,10,12,16,18,20}.
      With --restrict-fids-csv outputs/10_eval_deployment/by_curve/per_curve_metrics.csv
      the comparator restricts to SBI's 133 high-quality curves AND uses
      SBI's exact fold assignment from the `fold_label` column — this is
      the ADR-023 matched-protocol mode that gives apples-to-apples R²
      vs script 10 `--fold-by curve` SBI output.

  --target cross-doi-321
      Train on internal 181, predict on 321 PLGA xlsx. Drops two
      features absent from 321 (`SA_V`, `Drug_Pka`) so only the 7
      cross-DOI-shared descriptors are used on BOTH ends. fPCA `k`
      re-selected via nested GroupKFold-3 on the training set (not
      hardcoded to the diagnostic paper's internal-9-feature optimum).
      With --sbi-matched-csv outputs/10_eval_deployment/cross_doi_per_curve_metrics.csv
      reports two subsets: all-321-filtered and SBI-matched-259.

Multi-seed: all runs sweep --seeds (default 42,43,44,45,46) and report
per-curve metric medians ACROSS seeds plus per-seed median R² distribution
(min, max, std) so ADR-024 condition (3) does not depend on a single
LightGBM random seed.

Run:
    .\\.venv\\Scripts\\python.exe scripts/12_baseline_comparators.py
    .\\.venv\\Scripts\\python.exe scripts/12_baseline_comparators.py \\
        --target internal-181 \\
        --restrict-fids-csv outputs/10_eval_deployment/by_curve/per_curve_metrics.csv
    .\\.venv\\Scripts\\python.exe scripts/12_baseline_comparators.py \\
        --target cross-doi-321 \\
        --sbi-matched-csv outputs/10_eval_deployment/cross_doi_per_curve_metrics.csv

Outputs:
    outputs/12_baseline_comparators/internal-181-{standalone,matched}/{summary,per_curve_medianseed,per_curve_per_seed,selected_k}.csv
    outputs/12_baseline_comparators/cross-doi-321/{summary,per_curve_medianseed,per_curve_per_seed}.csv

Expected runtime (5 seeds):
    internal-181  ~ 15-25 min
    cross-doi-321 ~ 2-3 min
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

# ----------------------------------------------------------------------
# Protocol constants — kept identical to the diagnostic paper for
# numerical reproducibility. Do not edit without writing an ADR.
# ----------------------------------------------------------------------
FEATURES_FULL_9 = (
    "Polymer_MW", "Drug_Mw", "Drug_LogP", "SA_V", "LA/GA",
    "Drug_Pka", "DLC", "Initial D/M ratio", "Drug_TPSA",
)
# Subset present in BOTH internal 181 and the 321 xlsx (after rename +
# unit conversion). Dropped: SA_V, Drug_Pka.
FEATURES_CROSS_DOI_7 = (
    "Polymer_MW", "Drug_Mw", "Drug_LogP", "LA/GA",
    "DLC", "Initial D/M ratio", "Drug_TPSA",
)

GROUP_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"

LGBM_PARAMS = dict(
    n_estimators=400, max_depth=-1, learning_rate=0.02, num_leaves=15,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=-1, n_jobs=1, force_row_wise=True,
)

N_GRID = 96
OUTER_SPLITS = 5
INNER_SPLITS = 3
K_CANDIDATES = (4, 5, 7, 10, 12, 16, 18, 20)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)  # ADR-023 robustness: median-of-seeds
T_MAX_DAYS_CROSS_DOI = 90.0  # match script 10's cross-DOI horizon


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def _load_internal_181(path: Path) -> pd.DataFrame:
    """Load internal 181 CSV; alias SA-V -> SA_V to match diagnostic paper."""
    df = pd.read_csv(path)
    if "SA-V" in df.columns and "SA_V" not in df.columns:
        df["SA_V"] = df["SA-V"]
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    keep = [GROUP_COL, TIME_COL, Y_COL, *FEATURES_FULL_9]
    return df.dropna(subset=keep)[keep].copy()


def _load_cross_doi_321(path: Path) -> pd.DataFrame:
    """Load 321 xlsx, map to 181 schema (7 shared cols only).

    Mirrors script 10's _load_cross_doi_xlsx but without imputation; the
    comparator only consumes the 7 cross-DOI-shared features so the 6
    absent columns are simply not part of the feature set here.
    """
    df = pd.read_excel(path).rename(columns={
        "Formulation Index": GROUP_COL,
        "Drug MW": "Drug_Mw",
        "Drug TPSA": "Drug_TPSA",
        "Drug LogP": "Drug_LogP",
        "Polymer MW": "Polymer_MW",
        "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
        "Drug Loading Capacity": "DLC",
    })
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    # Dedupe (fid, Time) — four 321 formulations have duplicate Time rows.
    df = (
        df.groupby([GROUP_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in (TIME_COL, Y_COL)},
            Y_COL: "mean",
        })
    )
    # Filter to t<=T_MAX horizon (matches q_phi/SBI training grid).
    df = df[df[TIME_COL] <= T_MAX_DAYS_CROSS_DOI].copy()
    # Drop formulations left with <2 observations after filtering.
    counts = df.groupby(GROUP_COL).size()
    df = df[df[GROUP_COL].isin(counts[counts >= 2].index)].copy()
    keep = [GROUP_COL, TIME_COL, Y_COL, *FEATURES_CROSS_DOI_7]
    return df[keep].copy()


def _feature_table(df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """One row per formulation: descriptors picked from the curve's first row."""
    rows = []
    for gid, g in df.groupby(GROUP_COL, sort=False):
        row0 = g.sort_values(TIME_COL).iloc[0]
        rows.append({GROUP_COL: str(gid), **{c: float(row0[c]) for c in features}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Core methods
# ----------------------------------------------------------------------
def _log_grid(train_df: pd.DataFrame, n_grid: int) -> np.ndarray:
    """Log-spaced time grid up to the training set's max time."""
    max_t = float(np.nanmax(train_df[TIME_COL].to_numpy(dtype=float)))
    return np.expm1(np.linspace(0.0, np.log1p(max(max_t, 1.0)), int(n_grid)))


def _interp_to_grid(g: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Per-formulation: sort by time, linear interp onto grid (last-value extend)."""
    gs = g.sort_values(TIME_COL)
    t = gs[TIME_COL].to_numpy(dtype=float)
    y = np.clip(gs[Y_COL].to_numpy(dtype=float), 0.0, 1.2)
    t_u, idx = np.unique(t, return_index=True)
    return np.interp(grid, t_u, y[idx], left=y[idx[0]], right=y[idx[-1]])


def _curve_matrix(df: pd.DataFrame, groups: list[str], grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Stack per-formulation grid-projected curves into a matrix."""
    grouped = {str(k): g for k, g in df.groupby(GROUP_COL, sort=False)}
    rows, ok = [], []
    for gid in groups:
        g = grouped.get(str(gid))
        if g is None or len(g) < 2:
            continue
        rows.append(_interp_to_grid(g, grid))
        ok.append(str(gid))
    return np.vstack(rows), ok


def _fit_lgbm(x: pd.DataFrame, y: np.ndarray, seed: int) -> LGBMRegressor:
    p = dict(LGBM_PARAMS, random_state=int(seed))
    m = LGBMRegressor(**p)
    m.fit(x, y)
    return m


def _fit_fpca(
    train_df: pd.DataFrame, feat_df: pd.DataFrame, train_groups: list[str],
    features: tuple[str, ...], k: int, seed: int,
) -> tuple[np.ndarray, PCA, dict[int, LGBMRegressor]]:
    grid = _log_grid(train_df, N_GRID)
    matrix, ok = _curve_matrix(train_df, train_groups, grid)
    n_comp = min(int(k), matrix.shape[0] - 1, matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    scores = pca.fit_transform(matrix)
    meta = feat_df[feat_df[GROUP_COL].isin(ok)].set_index(GROUP_COL).loc[ok].reset_index()
    models = {
        j: _fit_lgbm(meta[list(features)], scores[:, j], seed + 100 + j)
        for j in range(n_comp)
    }
    return grid, pca, models


def _predict_fpca(
    point_df: pd.DataFrame, feat_df: pd.DataFrame,
    grid: np.ndarray, pca: PCA, models: dict[int, LGBMRegressor],
    features: tuple[str, ...],
) -> pd.Series:
    meta = feat_df.set_index(GROUP_COL)
    out = []
    for gid, g in point_df.groupby(GROUP_COL, sort=False):
        x = meta.loc[str(gid), list(features)].to_frame().T.astype(float)
        scores = np.array([[models[j].predict(x)[0] for j in sorted(models)]], dtype=float)
        grid_pred = np.clip(pca.inverse_transform(scores).reshape(-1), 0.0, 1.2)
        t = g[TIME_COL].to_numpy(dtype=float)
        out.append(pd.Series(
            np.interp(t, grid, grid_pred, left=grid_pred[0], right=grid_pred[-1]),
            index=g.index,
        ))
    return pd.concat(out).sort_index()


def _direct_lgbm(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    features: tuple[str, ...], seed: int,
) -> np.ndarray:
    """(x, t) -> Q(t). No curve representation."""
    cols = [*features, TIME_COL]
    m = _fit_lgbm(train_df[cols], train_df[Y_COL].to_numpy(dtype=float), seed)
    return np.clip(m.predict(test_df[cols]), 0.0, 1.2)


# ----------------------------------------------------------------------
# Metrics — matched to the diagnostic paper's summarize()
# ----------------------------------------------------------------------
def _per_group(pred_df: pd.DataFrame, method_cols: dict[str, str]) -> pd.DataFrame:
    """Per-formulation R²/RMSE/MAE for each method column.

    Mirrors `metric_row` from the diagnostic paper:
    `vendored/diagnostic_paper/scripts/run_alternative_curve_routes.py:49-62`.
    Non-finite y/y_pred entries are masked before computing residuals so
    a single bad row in a long curve does not poison the per-curve number.
    """
    rows = []
    for gid, g in pred_df.groupby(GROUP_COL, sort=False):
        y_all = g[Y_COL].to_numpy(dtype=float)
        for method, col in method_cols.items():
            yp_all = g[col].to_numpy(dtype=float)
            mask = np.isfinite(y_all) & np.isfinite(yp_all)
            if int(mask.sum()) == 0:
                rows.append({
                    GROUP_COL: str(gid), "method": method, "n_points": 0,
                    "r2": np.nan, "rmse": np.nan, "mae": np.nan,
                })
                continue
            y = y_all[mask]
            yp = yp_all[mask]
            resid = y - yp
            denom = float(np.sum((y - y.mean()) ** 2))
            rows.append({
                GROUP_COL: str(gid), "method": method, "n_points": int(mask.sum()),
                "r2": float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else np.nan,
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "mae": float(np.mean(np.abs(resid))),
            })
    return pd.DataFrame(rows)


def _summarize(per_curve: pd.DataFrame, pred_df: pd.DataFrame, method_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for method, col in method_cols.items():
        g = per_curve[per_curve["method"] == method]
        y = pred_df[Y_COL].to_numpy(dtype=float)
        yp = pred_df[col].to_numpy(dtype=float)
        resid = y - yp
        denom = float(np.sum((y - y.mean()) ** 2))
        rows.append({
            "method": method,
            "groupwise_macro_r2": float(g["r2"].mean()),
            "groupwise_rmse": float(g["rmse"].mean()),
            "groupwise_mae": float(g["mae"].mean()),
            "median_per_formulation_rmse": float(g["rmse"].median()),
            "pointwise_micro_r2": float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else np.nan,
            "pointwise_micro_rmse": float(np.sqrt(np.mean(resid ** 2))),
            "n_groups": int(g[GROUP_COL].nunique()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def _nested_k_select(
    train_df: pd.DataFrame, feat_df: pd.DataFrame, train_groups: list[str],
    features: tuple[str, ...], seed: int,
) -> int:
    """Inner GroupKFold-3 over training groups; pick k minimising groupwise RMSE."""
    groups = np.array(sorted(train_groups), dtype=object)
    gkf = GroupKFold(n_splits=min(INNER_SPLITS, len(groups)))
    scores: dict[int, list[float]] = {k: [] for k in K_CANDIDATES}
    for inner, (tr, va) in enumerate(gkf.split(groups, groups=groups), start=1):
        tr_g, va_g = groups[tr].tolist(), groups[va].tolist()
        tr_df = train_df[train_df[GROUP_COL].isin(tr_g)]
        va_df = train_df[train_df[GROUP_COL].isin(va_g)]
        for k in K_CANDIDATES:
            grid, pca, models = _fit_fpca(tr_df, feat_df, tr_g, features,
                                          k=k, seed=seed + 1000 * inner + k)
            pred = _predict_fpca(va_df, feat_df, grid, pca, models, features)
            tmp = va_df[[GROUP_COL, Y_COL]].copy()
            tmp["pred"] = pred.to_numpy(dtype=float)
            rmses = [
                float(np.sqrt(np.mean((g[Y_COL].to_numpy(dtype=float) - g["pred"].to_numpy(dtype=float)) ** 2)))
                for _, g in tmp.groupby(GROUP_COL, sort=False)
            ]
            scores[k].append(float(np.mean(rmses)))
    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    return min(mean_scores, key=lambda k: mean_scores[k])


def _build_fold_iter(
    df: pd.DataFrame, restrict_fids_csv: Path | None,
) -> tuple[pd.DataFrame, list[tuple[str, set[str]]]]:
    """Return (possibly filtered) df + fold_iter as list of (label, test_fids).

    Two modes:
    - restrict_fids_csv = None: standalone GroupKFold-5 over all fids in df.
    - restrict_fids_csv = path: restrict df to fids in CSV and use the CSV's
      `fold_label` column as the partition. ADR-023's matched-protocol mode
      — pass `outputs/10_eval_deployment/by_curve/per_curve_metrics.csv` to
      replicate SBI's exact 5-fold assignment on its 133 high-quality
      subset.
    """
    if restrict_fids_csv is None:
        groups = np.array(sorted(df[GROUP_COL].unique()), dtype=object)
        gkf = GroupKFold(n_splits=OUTER_SPLITS)
        fold_iter: list[tuple[str, set[str]]] = []
        for fold_no, (_, te) in enumerate(gkf.split(groups, groups=groups), start=1):
            fold_iter.append((f"fold{fold_no}", {str(x) for x in groups[te]}))
        return df, fold_iter

    rdf = pd.read_csv(restrict_fids_csv)
    rdf[GROUP_COL] = rdf[GROUP_COL].astype(str)
    if "fold_label" not in rdf.columns:
        raise ValueError(
            f"--restrict-fids-csv {restrict_fids_csv} must contain a "
            f"`fold_label` column; this file does not."
        )
    fid_to_fold = dict(zip(rdf[GROUP_COL], rdf["fold_label"], strict=False))
    df = df[df[GROUP_COL].isin(fid_to_fold)].copy()
    fold_iter = []
    for fold_label in sorted(set(fid_to_fold.values())):
        te = {f for f, fl in fid_to_fold.items() if fl == fold_label}
        fold_iter.append((str(fold_label), te))
    return df, fold_iter


def _multi_seed_aggregate(per_curve_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: per (fid, method) take median R²/RMSE/MAE across seeds.

    Input: long-format with columns (fid, method, seed, r2, rmse, mae, n_points).
    Output: one row per (fid, method) with the median-of-seeds values.
    """
    agg = (
        per_curve_long
        .groupby([GROUP_COL, "method"], as_index=False, sort=False)
        .agg({
            "r2": "median",
            "rmse": "median",
            "mae": "median",
            "n_points": "first",
        })
    )
    return agg


def _run_internal_181(args: argparse.Namespace) -> None:
    print(f"[12] loading internal 181 from {args.internal_data}")
    df = _load_internal_181(args.internal_data)
    df, fold_iter = _build_fold_iter(df, args.restrict_fids_csv)
    feat_df = _feature_table(df, FEATURES_FULL_9)
    print(f"[12] {df[GROUP_COL].nunique()} formulations after restrict, "
          f"{len(df)} (fid,t) rows; features={FEATURES_FULL_9}")
    print(f"[12] {len(fold_iter)} folds; seeds={args.seeds}")

    long_rows = []  # per-seed per-curve per-method metrics
    selected_ks: list[tuple[int, str, int]] = []
    sub_label = "matched" if args.restrict_fids_csv else "standalone"
    for seed in args.seeds:
        folds_pred = []
        for fold_no, (fold_label, te_fids) in enumerate(fold_iter, start=1):
            tr_df = df[~df[GROUP_COL].isin(te_fids)].copy()
            te_df = df[df[GROUP_COL].isin(te_fids)].copy()
            tr_groups = sorted(tr_df[GROUP_COL].unique())
            out = te_df.copy()
            out["fold_label"] = fold_label

            k = _nested_k_select(tr_df, feat_df, tr_groups, FEATURES_FULL_9,
                                 seed=seed + 5000 * fold_no)
            selected_ks.append((seed, fold_label, k))
            print(f"[12]   seed={seed} fold={fold_label}  n_train={len(tr_groups)}  "
                  f"n_test={len(te_fids)}  k={k}")

            grid, pca, models = _fit_fpca(tr_df, feat_df, tr_groups,
                                          FEATURES_FULL_9, k=k,
                                          seed=seed + 7000 * fold_no + k)
            out["pred_fpca"] = _predict_fpca(te_df, feat_df, grid, pca, models,
                                             FEATURES_FULL_9).to_numpy(dtype=float)
            out["pred_direct"] = _direct_lgbm(tr_df, te_df, FEATURES_FULL_9,
                                              seed=seed + 3000 * fold_no)
            folds_pred.append(out)

        pred_df_seed = pd.concat(folds_pred, ignore_index=True)
        method_cols = {"nested fPCA LGBM": "pred_fpca", "Direct LGBM": "pred_direct"}
        per_curve_seed = _per_group(pred_df_seed, method_cols)
        per_curve_seed["seed"] = seed
        long_rows.append(per_curve_seed)

    long_df = pd.concat(long_rows, ignore_index=True)
    per_curve = _multi_seed_aggregate(long_df)

    # Summary from the median-over-seeds per_curve table.
    summary_rows = []
    for method, sub in per_curve.groupby("method", sort=False):
        # Per-seed group-medians for variance reporting
        seed_medians = [
            float(long_df[(long_df["method"] == method) & (long_df["seed"] == s)]["r2"].median())
            for s in args.seeds
        ]
        summary_rows.append({
            "method": method,
            "median_per_curve_R2_medianseed": float(sub["r2"].median()),
            "mean_per_curve_R2_medianseed":   float(sub["r2"].mean()),
            "median_per_curve_RMSE_medianseed": float(sub["rmse"].median()),
            "median_per_curve_MAE_medianseed":  float(sub["mae"].median()),
            "across_seed_median_R2_min": float(np.min(seed_medians)),
            "across_seed_median_R2_max": float(np.max(seed_medians)),
            "across_seed_median_R2_std": float(np.std(seed_medians, ddof=0)),
            "n_groups": int(sub[GROUP_COL].nunique()),
            "n_seeds":  len(args.seeds),
        })
    summary = pd.DataFrame(summary_rows)

    out_dir = args.out / f"internal-181-{sub_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_curve.to_csv(out_dir / "per_curve_medianseed.csv", index=False)
    long_df.to_csv(out_dir / "per_curve_per_seed.csv", index=False)
    pd.DataFrame(selected_ks, columns=["seed", "fold_label", "k"]).to_csv(
        out_dir / "selected_k.csv", index=False,
    )
    print(f"\n[12] wrote {out_dir}/summary.csv")
    print(summary.to_string(index=False))


def _run_cross_doi_321(args: argparse.Namespace) -> None:
    print(f"[12] loading internal 181 from {args.internal_data}")
    train_df = _load_internal_181(args.internal_data)
    train_df = train_df[[GROUP_COL, TIME_COL, Y_COL, *FEATURES_CROSS_DOI_7]].copy()
    train_feat = _feature_table(train_df, FEATURES_CROSS_DOI_7)
    train_groups = sorted(train_df[GROUP_COL].unique())
    print(f"[12]   train: {len(train_groups)} formulations, "
          f"features={FEATURES_CROSS_DOI_7}")

    print(f"[12] loading 321 xlsx from {args.cross_doi_data}")
    test_df = _load_cross_doi_321(args.cross_doi_data)
    test_df[GROUP_COL] = "321_" + test_df[GROUP_COL].astype(str)
    test_feat = _feature_table(test_df, FEATURES_CROSS_DOI_7)
    print(f"[12]   test: {test_df[GROUP_COL].nunique()} formulations, "
          f"{len(test_df)} (fid,t) rows after t<={T_MAX_DAYS_CROSS_DOI}d filter")

    feat_df = pd.concat([train_feat, test_feat], ignore_index=True)

    # Re-select k for cross-DOI via nested CV on the training set (the
    # diagnostic paper picked k=18 on internal full-9-feature data, but
    # with 7-feature cross-DOI the optimum may differ). Selection runs
    # once at seed[0] since it's the same training data across seeds.
    k_seed0 = _nested_k_select(
        train_df, feat_df, train_groups, FEATURES_CROSS_DOI_7,
        seed=args.seeds[0] + 5000,
    )
    print(f"[12]   nested k selection on train set -> k={k_seed0}")

    long_rows = []
    for seed in args.seeds:
        grid, pca, models = _fit_fpca(train_df, feat_df, train_groups,
                                      FEATURES_CROSS_DOI_7, k=k_seed0, seed=seed)
        out = test_df.copy()
        out["pred_fpca"] = _predict_fpca(test_df, feat_df, grid, pca, models,
                                         FEATURES_CROSS_DOI_7).to_numpy(dtype=float)
        out["pred_direct"] = _direct_lgbm(train_df, test_df,
                                          FEATURES_CROSS_DOI_7, seed=seed)
        method_cols = {"nested fPCA LGBM": "pred_fpca", "Direct LGBM": "pred_direct"}
        per_curve_seed = _per_group(out, method_cols)
        per_curve_seed["seed"] = seed
        long_rows.append(per_curve_seed)
        print(f"[12]   seed={seed}  fpca={per_curve_seed[per_curve_seed['method']=='nested fPCA LGBM']['r2'].median():+.4f}  "
              f"direct={per_curve_seed[per_curve_seed['method']=='Direct LGBM']['r2'].median():+.4f}")

    long_df = pd.concat(long_rows, ignore_index=True)
    per_curve = _multi_seed_aggregate(long_df)

    def _summary_block(pc_df: pd.DataFrame, label: str) -> list[dict]:
        rows = []
        for method, sub in pc_df.groupby("method", sort=False):
            seed_medians = [
                float(long_df[
                    (long_df["method"] == method)
                    & (long_df["seed"] == s)
                    & (long_df[GROUP_COL].isin(sub[GROUP_COL]))
                ]["r2"].median())
                for s in args.seeds
            ]
            rows.append({
                "subset": label,
                "method": method,
                "n_groups": int(sub[GROUP_COL].nunique()),
                "n_seeds":  len(args.seeds),
                "median_per_curve_R2_medianseed": float(sub["r2"].median()),
                "mean_per_curve_R2_medianseed":   float(sub["r2"].mean()),
                "median_per_curve_RMSE_medianseed": float(sub["rmse"].median()),
                "median_per_curve_MAE_medianseed":  float(sub["mae"].median()),
                "across_seed_median_R2_min": float(np.min(seed_medians)),
                "across_seed_median_R2_max": float(np.max(seed_medians)),
                "across_seed_median_R2_std": float(np.std(seed_medians, ddof=0)),
            })
        return rows

    summary_rows = _summary_block(per_curve, "all_321_filtered")
    if args.sbi_matched_csv is not None:
        sbi = pd.read_csv(args.sbi_matched_csv)
        sbi_fids = {"321_" + str(f) for f in sbi["Formulation_Index"]}
        matched = per_curve[per_curve[GROUP_COL].isin(sbi_fids)].copy()
        summary_rows.extend(_summary_block(matched, "sbi_matched"))
        print(f"[12]   sbi-matched subset: {matched[GROUP_COL].nunique()} curves "
              f"(SBI ADR-021 evaluated {len(sbi_fids)} high-quality)")

    summary = pd.DataFrame(summary_rows)
    out_dir = args.out / "cross-doi-321"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_curve.to_csv(out_dir / "per_curve_medianseed.csv", index=False)
    long_df.to_csv(out_dir / "per_curve_per_seed.csv", index=False)
    with (out_dir / "selected_k.txt").open("w", encoding="utf-8") as f:
        f.write(f"nested k selected via inner GroupKFold-3 on train 181: k={k_seed0}\n")
    print(f"\n[12] wrote {out_dir}/summary.csv")
    print(summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("internal-181", "cross-doi-321"),
                    default="internal-181")
    ap.add_argument("--internal-data", type=Path,
                    default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument("--cross-doi-data", type=Path,
                    default=Path(
                        r"D:\chemical-world-model-v0\datset\321PLGA"
                        r"\A Dataset on Formulation Parameters and Characteristics of "
                        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
                    ))
    ap.add_argument("--out", type=Path, default=Path("outputs/12_baseline_comparators"))
    ap.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated seeds for multi-seed robustness (ADR-023 fix)",
    )
    ap.add_argument(
        "--restrict-fids-csv", type=Path, default=None,
        help="internal-181 only: restrict to fids + fold assignments from this "
             "CSV (must have Experimental_index + fold_label columns). Used to "
             "replicate SBI's exact 5-fold partition for matched-protocol "
             "comparison.",
    )
    ap.add_argument(
        "--sbi-matched-csv", type=Path, default=None,
        help="cross-doi-321 only: CSV with Formulation_Index column listing the "
             "high-quality 259-curve subset SBI evaluated. If provided, summary "
             "additionally reports the sbi_matched subset.",
    )
    args = ap.parse_args()
    args.seeds = tuple(int(s.strip()) for s in str(args.seeds).split(",") if s.strip())

    if args.target == "internal-181":
        _run_internal_181(args)
    else:
        _run_cross_doi_321(args)


if __name__ == "__main__":
    main()
