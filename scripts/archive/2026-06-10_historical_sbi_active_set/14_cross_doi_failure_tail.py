"""
14 - 321 cross-DOI failure-tail audit.

What this does:
    1. Load the sprint1 cross-DOI per-curve metrics and saved predictions.
    2. Re-link each kept 321 curve to its raw xlsx descriptors using the same
       rename + unit-conversion contract as script 10.
    3. Derive observed-curve shape statistics (q7, q14, t50, t80, final under-
       prediction) from the saved npz files, so we audit failure modes without
       re-running inference.
    4. Compare the 321 kept set against the internal 133-curve training support
       using the sprint1 internal eval outputs and the internal descriptor CSV.
    5. Emit enriched per-curve CSVs plus grouped summaries that answer whether
       the negative mean R^2 is a few extreme outliers or a structured regime.

What this consumes:
    outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv
    outputs/sprint1_10_eval_deployment/cross_doi_predictions/*.npz
    outputs/sprint1_10_eval_deployment/per_curve_metrics.csv
    outputs/sprint1_10_eval_deployment/predictions/*.npz
    data/Dataset_17_feat_augmented.csv
    D:\\chemical-world-model-v0\\datset\\321PLGA\\...\\mp_dataset_processed.xlsx

What this produces:
    outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv
    outputs/sprint1_14_cross_doi_failure_tail/regime_summary.csv
    outputs/sprint1_14_cross_doi_failure_tail/distribution_shift_summary.csv
    outputs/sprint1_14_cross_doi_failure_tail/shared_descriptor_ranges.csv
    outputs/sprint1_14_cross_doi_failure_tail/drug_signature_summary.csv
    outputs/sprint1_14_cross_doi_failure_tail/polymer_signature_summary.csv
    outputs/sprint1_14_cross_doi_failure_tail/feature_bin_summary.csv
    outputs/sprint1_14_cross_doi_failure_tail/bottom20.csv
    outputs/sprint1_14_cross_doi_failure_tail/summary.txt

Expected runtime:
    < 10 s.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_SHARED_COLS = (
    "LA/GA",
    "Polymer_MW",
    "Initial D/M ratio",
    "DLC",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_LogP",
)
_IMPUTED_COLS = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
)
_321_RENAME = {
    "Formulation Index": "Experimental_index",
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",
    "Drug Encapsulation Efficiency": "DEE_pct",
    "Solubility Enhancer Concentration": "SE_conc",
    "Particle Size": "Particle_Size",
}
_BIN_FEATURES = (
    "q7",
    "q14",
    "t50",
    "t80",
    "Polymer_MW_kDa",
    "Particle_Size",
    "SE_conc",
    "DLC_pct",
    "DEE_pct",
)


def _interp_or_nan(t: np.ndarray, q: np.ndarray, day: float) -> float:
    if t[0] <= day <= t[-1]:
        return float(np.interp(day, t, q))
    return float("nan")


def _t_at_level(t: np.ndarray, q: np.ndarray, level: float) -> float:
    if float(np.max(q)) < level:
        return float("nan")
    idx = int(np.where(q >= level)[0][0])
    if idx == 0:
        return float(t[0])
    t0 = float(t[idx - 1])
    t1 = float(t[idx])
    q0 = float(q[idx - 1])
    q1 = float(q[idx])
    if q1 == q0:
        return t1
    frac = (level - q0) / (q1 - q0)
    return t0 + frac * (t1 - t0)


def _shape_from_npz(pred_dir: Path, fids: np.ndarray, id_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fid_value in fids:
        fid = int(fid_value)
        arr = np.load(pred_dir / f"{fid}.npz")
        t = arr["t"].astype(float)
        q_obs = arr["Q_obs"].astype(float)
        q_pred = arr["Q_pred_median"].astype(float)
        q7 = _interp_or_nan(t, q_obs, 7.0)
        q14 = _interp_or_nan(t, q_obs, 14.0)
        final_q = float(q_obs[-1])
        final_pred = float(q_pred[-1])
        rows.append({
            id_col: fid,
            "tmax_obs_d": float(t[-1]),
            "obs_var": float(np.var(q_obs)),
            "obs_range": float(np.max(q_obs) - np.min(q_obs)),
            "q7": q7,
            "q14": q14,
            "t50": _t_at_level(t, q_obs, 0.50),
            "t80": _t_at_level(t, q_obs, 0.80),
            "final_q": final_q,
            "final_pred": final_pred,
            "final_underpred": final_q - final_pred,
            "early_frac_7d": (
                float(q7 / final_q) if np.isfinite(q7) and final_q > 0 else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _load_cross_doi_descriptors(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns=_321_RENAME).copy()
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["Polymer_MW_kDa"] = df["Polymer_MW"] / 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df["DLC_pct"] = df["DLC"] * 100.0
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    keep_cols = [
        "Experimental_index",
        "Drug_Mw",
        "Drug_TPSA",
        "Drug_LogP",
        "Polymer_MW",
        "Polymer_MW_kDa",
        "LA/GA",
        "Initial D/M ratio",
        "DLC",
        "DLC_pct",
        "Particle_Size",
        "DEE_pct",
        "SE_conc",
    ]
    return df.drop_duplicates("Experimental_index")[keep_cols].reset_index(drop=True)


def _distribution_summary(
    dataset: str,
    df: pd.DataFrame,
    feature_cols: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for col in feature_cols:
        s = df[col].dropna().astype(float)
        rows.append({
            "dataset": dataset,
            "feature": col,
            "n": int(s.size),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def _group_summary(df: pd.DataFrame, dataset: str, group_name: str, col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for key, sub in df.groupby(col, dropna=False):
        rows.append({
            "dataset": dataset,
            "group_name": group_name,
            "group_value": key,
            "n": int(len(sub)),
            "median_R2": float(sub["R2"].median()),
            "mean_R2": float(sub["R2"].mean()),
            "frac_neg": float((sub["R2"] < 0).mean()),
            "median_MAE": float(sub["MAE"].median()),
        })
    return pd.DataFrame(rows)


def _feature_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature in _BIN_FEATURES:
        s = df[feature].dropna()
        if s.nunique() < 4:
            continue
        bins = pd.qcut(s, q=4, duplicates="drop")
        work = df.loc[s.index, ["R2", "MAE"]].copy()
        work["bin"] = bins.astype(str)
        for bin_label, sub in work.groupby("bin", sort=False):
            rows.append({
                "feature": feature,
                "bin": bin_label,
                "n": int(len(sub)),
                "median_R2": float(sub["R2"].median()),
                "mean_R2": float(sub["R2"].mean()),
                "frac_neg": float((sub["R2"] < 0).mean()),
                "median_MAE": float(sub["MAE"].median()),
            })
    return pd.DataFrame(rows)


def _signature_summary(df: pd.DataFrame, col: str, min_n: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for key, sub in df.groupby(col):
        if len(sub) < min_n:
            continue
        rows.append({
            col: key,
            "n": int(len(sub)),
            "median_R2": float(sub["R2"].median()),
            "mean_R2": float(sub["R2"].mean()),
            "frac_neg": float((sub["R2"] < 0).mean()),
            "median_tmax": float(sub["tmax_obs_d"].median()),
            "median_MAE": float(sub["MAE"].median()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["median_R2", "mean_R2", "n"], ascending=[True, True, False])


def _shared_range_summary(
    train_desc: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for col in _SHARED_COLS:
        train_min = float(train_desc[col].min())
        train_max = float(train_desc[col].max())
        cross_min = float(cross_df[col].min())
        cross_max = float(cross_df[col].max())
        oor = (cross_df[col] < train_min) | (cross_df[col] > train_max)
        inr = ~oor
        rows.append({
            "feature": col,
            "train_min": train_min,
            "train_max": train_max,
            "cross_min": cross_min,
            "cross_max": cross_max,
            "n_out_of_range": int(oor.sum()),
            "frac_out_of_range": float(oor.mean()),
            "oor_median_R2": float(cross_df.loc[oor, "R2"].median()) if oor.any() else float("nan"),
            "oor_mean_R2": float(cross_df.loc[oor, "R2"].mean()) if oor.any() else float("nan"),
            "in_range_median_R2": float(cross_df.loc[inr, "R2"].median()) if inr.any() else float("nan"),
            "in_range_mean_R2": float(cross_df.loc[inr, "R2"].mean()) if inr.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def _removal_count_for_positive_mean(r2: np.ndarray) -> int | None:
    ordered = np.sort(r2.astype(float))
    for k in range(len(ordered)):
        if float(ordered[k:].mean()) > 0:
            return k
    return None


def _trimmed_means(r2: np.ndarray, fracs: tuple[float, ...]) -> dict[float, float]:
    ordered = np.sort(r2.astype(float))
    out: dict[float, float] = {}
    for frac in fracs:
        k = int(len(ordered) * frac)
        out[frac] = float(ordered[k:].mean())
    return out


def _attach_regimes(
    df: pd.DataFrame,
    fast_t50_days: float,
    fast_q7: float,
    short_tmax_days: float,
) -> pd.DataFrame:
    out = df.copy()
    out["fast_regime"] = ((out["t50"] <= fast_t50_days) | (out["q7"] >= fast_q7)).fillna(False)
    out["short_window"] = out["tmax_obs_d"] < short_tmax_days
    out["fast_short_group"] = np.select(
        [
            out["fast_regime"] & out["short_window"],
            out["fast_regime"] & ~out["short_window"],
            ~out["fast_regime"] & out["short_window"],
        ],
        ["fast&short", "fast_only", "short_only"],
        default="neither",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cross-metrics",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument(
        "--cross-predictions",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_predictions"),
    )
    ap.add_argument(
        "--internal-metrics",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/per_curve_metrics.csv"),
    )
    ap.add_argument(
        "--internal-predictions",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/predictions"),
    )
    ap.add_argument(
        "--internal-data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
    )
    ap.add_argument(
        "--cross-xlsx",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--fast-t50-days",
        type=float,
        default=2.4,
        help="audit threshold for the ultrafast-release regime",
    )
    ap.add_argument(
        "--fast-q7",
        type=float,
        default=0.75,
        help="audit threshold for high day-7 release",
    )
    ap.add_argument(
        "--short-tmax-days",
        type=float,
        default=14.0,
        help="audit threshold for short observation windows",
    )
    ap.add_argument(
        "--min-signature-n",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sprint1_14_cross_doi_failure_tail"),
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cross_metrics = pd.read_csv(args.cross_metrics)
    internal_metrics = pd.read_csv(args.internal_metrics)
    cross_desc = _load_cross_doi_descriptors(args.cross_xlsx)
    internal_desc = pd.read_csv(args.internal_data).drop_duplicates("Experimental_index")

    cross_shape = _shape_from_npz(
        pred_dir=args.cross_predictions,
        fids=cross_metrics["Formulation_Index"].astype(int).to_numpy(),
        id_col="Formulation_Index",
    )
    internal_shape = _shape_from_npz(
        pred_dir=args.internal_predictions,
        fids=internal_metrics["Experimental_index"].astype(int).unique(),
        id_col="Experimental_index",
    )

    # The sprint1 internal per-curve metrics enumerate the 133 high-quality
    # curves used to train the deployed r_psi checkpoint, so they define the
    # descriptor support we should compare 321 against.
    train_ids = internal_metrics["Experimental_index"].astype(int).unique()
    train_desc = internal_desc[internal_desc["Experimental_index"].astype(int).isin(train_ids)].copy()

    cross = cross_metrics.merge(cross_desc, left_on="Formulation_Index", right_on="Experimental_index", how="left")
    cross = cross.merge(cross_shape.drop(columns=["tmax_obs_d"]), on="Formulation_Index", how="left")
    cross = cross.drop(columns=["Experimental_index"])
    cross["drug_signature"] = cross.apply(
        lambda row: (
            f"Mw={row['Drug_Mw']:.3f}|TPSA={row['Drug_TPSA']:.2f}|"
            f"LogP={row['Drug_LogP']:.3f}"
        ),
        axis=1,
    )
    cross["polymer_signature"] = cross.apply(
        lambda row: f"LA/GA={row['LA/GA']:.2f}|MWkDa={row['Polymer_MW_kDa']:.1f}",
        axis=1,
    )
    cross = _attach_regimes(
        df=cross,
        fast_t50_days=args.fast_t50_days,
        fast_q7=args.fast_q7,
        short_tmax_days=args.short_tmax_days,
    )

    for col in _SHARED_COLS:
        lo = float(train_desc[col].min())
        hi = float(train_desc[col].max())
        cross[f"{col}__oor"] = (cross[col] < lo) | (cross[col] > hi)
    cross["n_oor_shared"] = cross[[f"{col}__oor" for col in _SHARED_COLS]].sum(axis=1)

    internal = internal_metrics.merge(internal_shape, on="Experimental_index", how="left")
    internal = _attach_regimes(
        df=internal,
        fast_t50_days=args.fast_t50_days,
        fast_q7=args.fast_q7,
        short_tmax_days=args.short_tmax_days,
    )

    regime_summary = pd.concat(
        [
            _group_summary(cross, "cross_321", "all", "fast_short_group").assign(group_name="fast_short_group"),
            _group_summary(cross, "cross_321", "fast_regime", "fast_regime"),
            _group_summary(cross, "cross_321", "short_window", "short_window"),
            _group_summary(internal, "internal_133", "fast_regime", "fast_regime"),
            _group_summary(internal, "internal_133", "short_window", "short_window"),
        ],
        ignore_index=True,
    )

    shift_features = (
        "LA/GA",
        "Polymer_MW",
        "Initial D/M ratio",
        "DLC",
        "Drug_Mw",
        "Drug_TPSA",
        "Drug_LogP",
        "q7",
        "t50",
        "t80",
        "tmax_obs_d",
        "obs_range",
        "final_q",
    )
    distribution_shift = pd.concat(
        [
            _distribution_summary("internal_133", train_desc.merge(internal_shape, on="Experimental_index"), shift_features),
            _distribution_summary("cross_321_kept", cross, shift_features),
        ],
        ignore_index=True,
    )
    shared_ranges = _shared_range_summary(train_desc, cross)
    feature_bins = _feature_bin_summary(cross)
    drug_summary = _signature_summary(cross, "drug_signature", args.min_signature_n)
    polymer_summary = _signature_summary(cross, "polymer_signature", args.min_signature_n)
    bottom20 = cross.sort_values("R2").head(20).copy()

    cross.to_csv(args.out / "per_curve_enriched.csv", index=False)
    regime_summary.to_csv(args.out / "regime_summary.csv", index=False)
    distribution_shift.to_csv(args.out / "distribution_shift_summary.csv", index=False)
    shared_ranges.to_csv(args.out / "shared_descriptor_ranges.csv", index=False)
    feature_bins.to_csv(args.out / "feature_bin_summary.csv", index=False)
    drug_summary.to_csv(args.out / "drug_signature_summary.csv", index=False)
    polymer_summary.to_csv(args.out / "polymer_signature_summary.csv", index=False)
    bottom20.to_csv(args.out / "bottom20.csv", index=False)

    neg = cross[cross["R2"] < 0].copy()
    removal_count = _removal_count_for_positive_mean(cross["R2"].to_numpy())
    trimmed = _trimmed_means(cross["R2"].to_numpy(), fracs=(0.01, 0.05, 0.10, 0.20))
    neg_breakdown = (
        neg["fast_short_group"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="n")
    )

    summary_path = args.out / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== Sprint 1 cross-DOI 321 failure-tail audit (script 14) ===\n\n")
        f.write("This is a pure error analysis of the current sprint1 outputs.\n")
        f.write("No retraining. No model edits. No new inference.\n\n")
        f.write("1. Headline\n")
        f.write(f"   n curves kept      : {len(cross)}\n")
        f.write(f"   median R^2         : {cross['R2'].median():+.4f}\n")
        f.write(f"   mean R^2           : {cross['R2'].mean():+.4f}\n")
        f.write(f"   curves with R^2<0  : {int((cross['R2'] < 0).sum())} / {len(cross)}\n")
        if removal_count is not None:
            f.write(f"   remove-worst-k for mean>0 : {removal_count}\n")
        f.write(f"   trimmed mean (1%)  : {trimmed[0.01]:+.4f}\n")
        f.write(f"   trimmed mean (5%)  : {trimmed[0.05]:+.4f}\n")
        f.write(f"   trimmed mean (10%) : {trimmed[0.10]:+.4f}\n")
        f.write(f"   trimmed mean (20%) : {trimmed[0.20]:+.4f}\n\n")

        f.write("2. Missing descriptors caveat\n")
        f.write("   The six descriptor axes imputed from the internal-181 training mean are:\n")
        f.write(f"   {_IMPUTED_COLS}\n")
        f.write("   They are constants across the 321 audit after script-10 mapping, so they\n")
        f.write("   cannot explain within-321 rank ordering directly. They can only create a\n")
        f.write("   dataset-wide blind spot, especially where fast-release chemistry needs them.\n\n")

        f.write("3. Negative-tail composition\n")
        f.write(neg_breakdown.to_string(index=False))
        f.write("\n\n")

        f.write("4. Regime summaries (cross 321 vs internal 133)\n")
        f.write(regime_summary.to_string(index=False))
        f.write("\n\n")

        f.write("5. Shared-descriptor range audit\n")
        f.write(shared_ranges.to_string(index=False))
        f.write("\n\n")

        f.write("6. Worst recurring drug signatures (n >= ")
        f.write(str(args.min_signature_n))
        f.write(")\n")
        if drug_summary.empty:
            f.write("   none\n")
        else:
            f.write(drug_summary.head(12).to_string(index=False))
        f.write("\n\n")

        f.write("7. Worst recurring polymer signatures (n >= ")
        f.write(str(args.min_signature_n))
        f.write(")\n")
        if polymer_summary.empty:
            f.write("   none\n")
        else:
            f.write(polymer_summary.head(12).to_string(index=False))
        f.write("\n\n")

        f.write("8. Worst 20 curves\n")
        f.write(
            bottom20[
                [
                    "Formulation_Index",
                    "R2",
                    "MAE",
                    "tmax_obs_d",
                    "n_obs",
                    "q7",
                    "q14",
                    "t50",
                    "t80",
                    "final_underpred",
                    "LA/GA",
                    "Polymer_MW_kDa",
                    "Initial D/M ratio",
                    "DLC_pct",
                    "Particle_Size",
                    "DEE_pct",
                    "SE_conc",
                    "drug_signature",
                    "polymer_signature",
                ]
            ].to_string(index=False)
        )
        f.write("\n")

    print(f"[14] wrote {summary_path}")
    print(f"[14] bottom-20 written to {args.out / 'bottom20.csv'}")
    print("[14] headline:")
    print(f"  median R^2 = {cross['R2'].median():+.4f}")
    print(f"  mean   R^2 = {cross['R2'].mean():+.4f}")
    print(f"  negative curves = {int((cross['R2'] < 0).sum())} / {len(cross)}")


if __name__ == "__main__":
    main()
