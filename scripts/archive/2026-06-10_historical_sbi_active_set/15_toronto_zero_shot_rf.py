"""
15 - Reproduce Toronto zero-shot RF and evaluate on cross-DOI 321.

What this does:
    1. Lock the public "zero-shot" task definition from the Toronto
       long-acting-injectables repo:
           X = all columns except Experimental_index / DP_Group / Release
             = 13 formulation descriptors + Time
           y = Release
           groups = DP_Group
    2. Lock the public best RF recipe from
       zero_shot_models/NESTED_CV_RESULTS/14_feat_RF.pkl and use it as
       a deterministic baseline.
    3. Run a fixed-parameter internal sanity check with the same outer
       GroupShuffleSplit-by-DP_Group protocol used in the Toronto code.
    4. Train that RF on the full internal 181 dataset and evaluate on the
       321 PLGA xlsx under the current repo's cross-DOI mapping:
           - t <= 90 days
           - duplicate (fid, t) rows averaged
           - six descriptors absent from 321 imputed to the internal
             training mean, matching script 10's contract
    5. Report both:
           - all_321_filtered
           - sbi_matched (the 259 high-quality curves used by script 10)

Why this exists:
    Before retraining any SBI component, we need the exact Toronto
    zero-shot RF baseline on our 181 -> 321 protocol. The Toronto repo
    separates zero-shot from few-shot; zero-shot is RF-favored and does
    NOT include the T=0.25 / 0.5 / 1.0 warm-start features.

Outputs:
    outputs/15_toronto_zero_shot_rf/internal_cv_scores.csv
    outputs/15_toronto_zero_shot_rf/cross_doi_per_curve_all.csv
    outputs/15_toronto_zero_shot_rf/cross_doi_per_curve_sbi_matched.csv
    outputs/15_toronto_zero_shot_rf/summary.txt

Expected runtime:
    ~10-20 s on CPU.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

GROUP_COL = "DP_Group"
FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
T_MAX_DAYS_CROSS_DOI = 90.0

# Exact zero-shot feature definition from the Toronto repo:
# X = df.drop(["Experimental_index", "DP_Group", "Release"], axis="columns")
ZERO_SHOT_FEATURES = (
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
    TIME_COL,
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
_321_IMPUTED_COLS = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
)

# Locked from the Toronto repo's
# zero_shot_models/NESTED_CV_RESULTS/14_feat_RF.pkl best row.
RF_PARAMS = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "criterion": "absolute_error",
    "max_depth": None,
    "max_features": 1.0,  # Toronto used "auto"; sklearn now maps this to all features.
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 400,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 4,
}


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
    df = pd.read_csv(path)
    missing = [c for c in (FID_COL, GROUP_COL, Y_COL, *ZERO_SHOT_FEATURES) if c not in df.columns]
    if missing:
        raise KeyError(f"Internal dataset missing columns: {missing}")
    return df[[FID_COL, GROUP_COL, *ZERO_SHOT_FEATURES, Y_COL]].copy()


def _load_cross_doi_xlsx(
    path: Path,
    mean_by_col: dict[str, float],
) -> pd.DataFrame:
    """Map 321 xlsx to the Toronto zero-shot feature space.

    Columns absent from 321 are imputed to the internal training mean,
    matching this repo's script-10 contract. This is the only coherent
    way to test the Toronto zero-shot recipe on 321 without inventing
    surrogate measurements.
    """
    df = pd.read_excel(path)
    df = df.rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])

    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df[GROUP_COL] = "UNK-PLGA"

    # Average duplicate (fid, t) rows exactly as in script 10.
    df = (
        df.groupby([FID_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in (TIME_COL, Y_COL)},
            Y_COL: "mean",
        })
    )
    df = df[df[TIME_COL] <= T_MAX_DAYS_CROSS_DOI].copy()

    for col in _321_IMPUTED_COLS:
        df[col] = float(mean_by_col[col])

    counts = df.groupby(FID_COL).size()
    df = df[df[FID_COL].isin(counts[counts >= 2].index)].copy()
    return df[[FID_COL, GROUP_COL, *ZERO_SHOT_FEATURES, Y_COL]].copy()


def _fit_scaler_and_rf(train_df: pd.DataFrame) -> tuple[StandardScaler, RandomForestRegressor]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[list(ZERO_SHOT_FEATURES)].to_numpy(dtype=float))
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(x_train, train_df[Y_COL].to_numpy(dtype=float))
    return scaler, model


def _predict(df: pd.DataFrame, scaler: StandardScaler, model: RandomForestRegressor) -> np.ndarray:
    x = scaler.transform(df[list(ZERO_SHOT_FEATURES)].to_numpy(dtype=float))
    return np.clip(model.predict(x), 0.0, 1.2)


def _per_curve_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fid, g in df.groupby(FID_COL, sort=False):
        y = g[Y_COL].to_numpy(dtype=float)
        yp = g[pred_col].to_numpy(dtype=float)
        rows.append({
            "Formulation_Index": int(fid),
            "n_obs": int(len(g)),
            "tmax_obs_d": float(g[TIME_COL].max()),
            "R2": _r2(y, yp),
            "RMSE": float(np.sqrt(np.mean((y - yp) ** 2))),
            "MAE": float(np.mean(np.abs(y - yp))),
        })
    return pd.DataFrame(rows)


def _internal_cv_sanity(df: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    """Fixed-parameter sanity check under Toronto's outer split protocol."""
    rows: list[dict[str, float | int]] = []
    groups = df[GROUP_COL].to_numpy()
    x = df[list(ZERO_SHOT_FEATURES)].to_numpy(dtype=float)
    y = df[Y_COL].to_numpy(dtype=float)

    # Toronto fits StandardScaler on the whole df before splitting. RF is
    # nearly scale-invariant, but we mirror the contract for fidelity.
    scaler = StandardScaler().fit(x)
    x_all = scaler.transform(x)

    for i in range(n_trials):
        cv_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        (train_idx, test_idx), = cv_outer.split(x_all, y, groups=groups)
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(x_all[train_idx], y[train_idx])
        pred = np.clip(model.predict(x_all[test_idx]), 0.0, 1.2)
        rows.append({
            "trial": int(i + 1),
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
            "n_test_groups": int(pd.Series(groups[test_idx]).nunique()),
            "test_mae": float(mean_absolute_error(y[test_idx], pred)),
            "test_rmse": float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))),
            "test_r2_pointwise": _r2(y[test_idx], pred),
        })
    return pd.DataFrame(rows)


def _summary_block(label: str, per_curve: pd.DataFrame) -> str:
    return (
        f"{label}:\n"
        f"  n_curves          : {len(per_curve)}\n"
        f"  median R^2        : {per_curve['R2'].median():+.4f}\n"
        f"  mean R^2          : {per_curve['R2'].mean():+.4f}\n"
        f"  median RMSE       : {per_curve['RMSE'].median():.4f}\n"
        f"  median MAE        : {per_curve['MAE'].median():.4f}\n"
        f"  n curves R^2 > 0  : {int((per_curve['R2'] > 0).sum())} / {len(per_curve)}\n"
        f"  n curves R^2 > .3 : {int((per_curve['R2'] > 0.30).sum())} / {len(per_curve)}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
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
        "--sbi-matched-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
        help="CSV listing the 259 high-quality cross-DOI curves from script 10.",
    )
    ap.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of outer GroupShuffleSplit trials for the internal sanity check.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/15_toronto_zero_shot_rf"),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    internal_df = _load_internal(args.internal_data)
    internal_means = {
        col: float(internal_df[col].astype(float).mean())
        for col in ZERO_SHOT_FEATURES
        if col != TIME_COL
    }

    sanity = _internal_cv_sanity(internal_df, n_trials=args.n_trials)
    sanity.to_csv(args.out / "internal_cv_scores.csv", index=False)

    scaler, model = _fit_scaler_and_rf(internal_df)

    cross_df = _load_cross_doi_xlsx(args.cross_doi_data, mean_by_col=internal_means)
    cross_df["pred_rf"] = _predict(cross_df, scaler, model)

    per_curve_all = _per_curve_metrics(cross_df, pred_col="pred_rf")
    per_curve_all.to_csv(args.out / "cross_doi_per_curve_all.csv", index=False)

    matched_fids = set(pd.read_csv(args.sbi_matched_csv)["Formulation_Index"].astype(int))
    per_curve_matched = per_curve_all[per_curve_all["Formulation_Index"].isin(matched_fids)].copy()
    per_curve_matched.to_csv(args.out / "cross_doi_per_curve_sbi_matched.csv", index=False)

    summary_path = args.out / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== Toronto zero-shot RF reproduction ===\n\n")
        f.write("Locked recipe source:\n")
        f.write("  aspuru-guzik-group/long-acting-injectables\n")
        f.write("  zero_shot_models/NESTED_CV_.py\n")
        f.write("  zero_shot_models/NESTED_CV_RESULTS/14_feat_RF.pkl\n\n")
        f.write("Task definition:\n")
        f.write("  X = 13 descriptors + Time\n")
        f.write("  y = Release\n")
        f.write("  groups = DP_Group\n")
        f.write("  missing 321 descriptors imputed to internal training mean\n\n")
        f.write("Locked RF params:\n")
        for key in sorted(RF_PARAMS):
            f.write(f"  {key}: {RF_PARAMS[key]}\n")
        f.write(f"  script_seed: {args.seed}\n")
        f.write("\n")
        f.write("Internal sanity (fixed-parameter, Toronto outer split):\n")
        f.write(f"  n_trials            : {len(sanity)}\n")
        f.write(f"  mean test MAE       : {sanity['test_mae'].mean():.4f}\n")
        f.write(f"  median test MAE     : {sanity['test_mae'].median():.4f}\n")
        f.write(f"  mean pointwise R^2  : {sanity['test_r2_pointwise'].mean():+.4f}\n\n")
        f.write(_summary_block("all_321_filtered", per_curve_all))
        f.write("\n")
        f.write(_summary_block("sbi_matched_259", per_curve_matched))
        f.write("\n")

    print(f"[15] wrote {summary_path}")
    print(f"[15] internal fixed-param mean test MAE = {sanity['test_mae'].mean():.4f}")
    print(f"[15] cross-DOI all_321_filtered median R^2 = {per_curve_all['R2'].median():+.4f}")
    print(f"[15] cross-DOI sbi_matched_259 median R^2 = {per_curve_matched['R2'].median():+.4f}")


if __name__ == "__main__":
    main()
