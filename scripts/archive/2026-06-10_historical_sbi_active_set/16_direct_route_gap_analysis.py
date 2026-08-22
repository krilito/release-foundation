"""
16 - Diagnose why Direct LGBM beats Toronto zero-shot RF on cross-DOI 321.

What this does:
    1. Hold the task fixed as direct pointwise prediction:
           (descriptors, Time) -> Release
    2. Sweep four variants on the 181 -> 321 protocol:
           - LGBM with 7 truly shared cross-DOI features
           - LGBM with Toronto-style 13 features, imputing the 6 missing
             321 descriptors to internal means
           - RF with the same 7 shared features
           - RF with the same 13 imputed features
    3. Aggregate per-curve metrics across seeds and compare subgroup
       behavior on the already-identified fast / short failure tail.
    4. Dump LightGBM feature importances for the 7-feature and 13-feature
       direct models to expose whether the model leans on descriptors that
       disappear at cross-DOI test time.

Why this exists:
    We already know Direct LGBM beats both SBI and Toronto zero-shot RF on
    the 321 external set. This script isolates whether that comes mainly
    from:
        - the clean 7-feature contract vs. imputed-missing 13-feature input,
        - the model family (boosted trees vs. random forest),
        - or the direct-Q task itself.

Outputs:
    outputs/16_direct_route_gap_analysis/variant_summary.csv
    outputs/16_direct_route_gap_analysis/subgroup_summary.csv
    outputs/16_direct_route_gap_analysis/lgbm_feature_importance.csv
    outputs/16_direct_route_gap_analysis/summary.txt

Expected runtime:
    ~30-60 s on CPU.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

TIME_COL = "Time"
Y_COL = "Release"
FID_COL = "Experimental_index"
T_MAX_DAYS_CROSS_DOI = 90.0
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

FEATURES_7 = (
    "Polymer_MW",
    "Drug_Mw",
    "Drug_LogP",
    "LA/GA",
    "DLC",
    "Initial D/M ratio",
    "Drug_TPSA",
)
FEATURES_13 = (
    "LA/GA",
    "Polymer_MW",
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Initial D/M ratio",
    "DLC",
    "SA-V",
    "SE",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_NHA",
    "Drug_LogP",
)
IMPUTED_321_COLS = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
)

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
    verbosity=-1,
    n_jobs=1,
    force_row_wise=True,
)
RF_PARAMS = dict(
    bootstrap=True,
    ccp_alpha=0.0,
    criterion="absolute_error",
    max_depth=None,
    max_features=1.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=4,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=400,
    n_jobs=-1,
)

_321_RENAME = {
    "Formulation Index": FID_COL,
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",
}
_321_DROP = (
    "Particle Size",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def _load_internal(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_cross_321(path: Path, mean_map: dict[str, float]) -> pd.DataFrame:
    df = pd.read_excel(path).rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df = (
        df.groupby([FID_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in (TIME_COL, Y_COL)},
            Y_COL: "mean",
        })
    )
    df = df[df[TIME_COL] <= T_MAX_DAYS_CROSS_DOI].copy()
    for col in IMPUTED_321_COLS:
        df[col] = float(mean_map[col])
    counts = df.groupby(FID_COL).size()
    return df[df[FID_COL].isin(counts[counts >= 2].index)].copy()


def _per_curve(pred_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for fid, g in pred_df.groupby(FID_COL, sort=False):
        y = g[Y_COL].to_numpy(dtype=float)
        yp = g[pred_col].to_numpy(dtype=float)
        rows.append({
            "Formulation_Index": int(fid),
            "R2": _r2(y, yp),
            "MAE": float(np.mean(np.abs(y - yp))),
        })
    return pd.DataFrame(rows)


def _eval_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: tuple[str, ...],
    model_name: str,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    x_train = train_df[[*features, TIME_COL]]
    y_train = train_df[Y_COL].to_numpy(dtype=float)
    x_test = test_df[[*features, TIME_COL]]
    for seed in seeds:
        if model_name == "lgbm":
            model = LGBMRegressor(**dict(LGBM_PARAMS, random_state=seed))
        elif model_name == "rf":
            model = RandomForestRegressor(**dict(RF_PARAMS, random_state=seed))
        else:
            raise ValueError(f"Unknown model_name={model_name}")
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_test), 0.0, 1.2)
        tmp = test_df[[FID_COL, Y_COL]].copy()
        tmp["pred"] = pred
        per_curve = _per_curve(tmp, pred_col="pred")
        per_curve["seed"] = int(seed)
        per_curve["variant"] = f"{model_name}_{len(features)}f"
        rows.append(per_curve)
    return pd.concat(rows, ignore_index=True)


def _feature_importance(
    train_df: pd.DataFrame,
    features: tuple[str, ...],
    seeds: tuple[int, ...],
    label: str,
) -> pd.DataFrame:
    rows = []
    x_train = train_df[[*features, TIME_COL]]
    y_train = train_df[Y_COL].to_numpy(dtype=float)
    for seed in seeds:
        model = LGBMRegressor(**dict(LGBM_PARAMS, random_state=seed))
        model.fit(x_train, y_train)
        for feat, imp in zip([*features, TIME_COL], model.feature_importances_, strict=False):
            rows.append({
                "variant": label,
                "seed": int(seed),
                "feature": feat,
                "importance": float(imp),
            })
    imp_df = pd.DataFrame(rows)
    out = (
        imp_df.groupby(["variant", "feature"], as_index=False)
        .agg(mean_importance=("importance", "mean"))
    )
    out["importance_fraction"] = out.groupby("variant")["mean_importance"].transform(
        lambda s: s / s.sum()
    )
    return out.sort_values(["variant", "mean_importance"], ascending=[True, False])


def _subgroup_summary(
    agg_df: pd.DataFrame,
    flags_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = agg_df.merge(flags_df, on="Formulation_Index", how="left")
    rows = []
    for variant, vdf in merged.groupby("variant", sort=False):
        subsets = {
            "overall": vdf,
            "fast": vdf[vdf["fast_regime"]],
            "short": vdf[vdf["short_window"]],
            "fast_short": vdf[vdf["fast_regime"] & vdf["short_window"]],
            "neither": vdf[~vdf["fast_regime"] & ~vdf["short_window"]],
        }
        for subset, sdf in subsets.items():
            rows.append({
                "variant": variant,
                "subset": subset,
                "n_curves": int(len(sdf)),
                "median_R2": float(sdf["R2"].median()),
                "mean_R2": float(sdf["R2"].mean()),
                "median_MAE": float(sdf["MAE"].median()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated evaluation seeds for tree-model variability",
    )
    ap.add_argument(
        "--internal-data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
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
    ap.add_argument(
        "--tail-flags-csv",
        type=Path,
        default=Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/16_direct_route_gap_analysis"),
    )
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    eval_seeds = tuple(int(s.strip()) for s in str(args.seeds).split(",") if s.strip())

    train_df = _load_internal(args.internal_data)
    mean_map = {c: float(train_df[c].astype(float).mean()) for c in FEATURES_13}
    test_df = _load_cross_321(args.cross_doi_data, mean_map)

    matched_fids = set(pd.read_csv(args.matched_fids_csv)["Formulation_Index"].astype(int))
    flags_df = pd.read_csv(args.tail_flags_csv)[
        ["Formulation_Index", "fast_regime", "short_window"]
    ].copy()

    per_curve_long = pd.concat(
        [
            _eval_variant(train_df, test_df, FEATURES_7, "lgbm", eval_seeds),
            _eval_variant(train_df, test_df, FEATURES_13, "lgbm", eval_seeds),
            _eval_variant(train_df, test_df, FEATURES_7, "rf", eval_seeds),
            _eval_variant(train_df, test_df, FEATURES_13, "rf", eval_seeds),
        ],
        ignore_index=True,
    )
    per_curve_long = per_curve_long[per_curve_long["Formulation_Index"].isin(matched_fids)].copy()

    per_curve_agg = (
        per_curve_long.groupby(["variant", "Formulation_Index"], as_index=False)
        .agg(R2=("R2", "median"), MAE=("MAE", "median"))
    )
    variant_summary = (
        per_curve_agg.groupby("variant", as_index=False)
        .agg(
            n_curves=("Formulation_Index", "nunique"),
            median_R2=("R2", "median"),
            mean_R2=("R2", "mean"),
            median_MAE=("MAE", "median"),
        )
    )
    subgroup_summary = _subgroup_summary(per_curve_agg, flags_df)
    feature_importance = pd.concat(
        [
            _feature_importance(train_df, FEATURES_7, eval_seeds, "lgbm_7f"),
            _feature_importance(train_df, FEATURES_13, eval_seeds, "lgbm_13f"),
        ],
        ignore_index=True,
    )

    variant_summary.to_csv(args.out / "variant_summary.csv", index=False)
    subgroup_summary.to_csv(args.out / "subgroup_summary.csv", index=False)
    feature_importance.to_csv(args.out / "lgbm_feature_importance.csv", index=False)

    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Direct route gap analysis ===\n\n")
        f.write(f"script_seed: {args.seed}\n")
        f.write(f"eval_seeds: {eval_seeds}\n")
        f.write(f"matched_curves: {len(matched_fids)}\n\n")

        f.write("Variant summary on sbi-matched 259:\n")
        f.write(variant_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Subgroup summary:\n")
        f.write(subgroup_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Top LGBM importances:\n")
        for variant in ("lgbm_7f", "lgbm_13f"):
            f.write(f"\n{variant}\n")
            top = feature_importance[feature_importance["variant"] == variant]
            f.write(top.to_string(index=False))
            f.write("\n")

    print(f"[16] wrote {args.out / 'summary.txt'}")
    print(variant_summary.to_string(index=False))


if __name__ == "__main__":
    main()
