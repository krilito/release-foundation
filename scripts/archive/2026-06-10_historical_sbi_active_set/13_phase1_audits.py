"""
13 — Phase 1 paper paper-readiness audits (Sprint 0).

Three one-shot slices of existing per-curve CSVs, produced for the
Phase 1 method paper's supplementary tables. No new training, no
re-inference; pure analysis of:

    outputs/10_eval_deployment/per_curve_metrics.csv           (internal LODO)
    outputs/10_eval_deployment/cross_doi_per_curve_metrics.csv (cross-DOI 321)

Audits:

  1. tmax-stratified R² on cross-DOI. R² has a small-Var(Q_obs)
     pathology on short windows; binning by observation horizon
     isolates the genuine signal from the binning artifact.

  2. LA/GA extrapolation audit. Internal 181 has LA/GA ∈ [0, 3];
     321 includes LA/GA up to 5.67. Split cross-DOI metrics by
     in-range vs extrapolation to quantify the cost.

  3. GEF deep-dive (internal LODO). ADR-019 flagged GEF as the only
     negative-R² fold; dump the 5 GEF curves' per-curve numbers so
     the paper supplementary can show them individually.

Outputs:
    outputs/13_phase1_audits/cross_doi_by_tmax.csv
    outputs/13_phase1_audits/cross_doi_by_laga_range.csv
    outputs/13_phase1_audits/gef_lodo_curves.csv
    outputs/13_phase1_audits/summary.txt

Expected runtime: < 5 s.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _bin_label(left: float, right: float) -> str:
    if np.isinf(right):
        return f">={left:g}d"
    return f"[{left:g},{right:g})d"


def _tmax_stratified(cd: pd.DataFrame) -> pd.DataFrame:
    bins = [(0, 14), (14, 30), (30, 60), (60, float("inf"))]
    rows = []
    for left, right in bins:
        m = (cd["tmax_obs_d"] >= left) & (cd["tmax_obs_d"] < right)
        sub = cd[m]
        rows.append({
            "tmax_bin": _bin_label(left, right),
            "n": int(len(sub)),
            "median_R2": float(sub["R2"].median()) if len(sub) else np.nan,
            "mean_R2": float(sub["R2"].mean()) if len(sub) else np.nan,
            "median_MAE": float(sub["MAE"].median()) if len(sub) else np.nan,
            "frac_R2_pos": float((sub["R2"] > 0).mean()) if len(sub) else np.nan,
            "frac_R2_above_030": float((sub["R2"] > 0.30).mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def _laga_extrapolation(cd: pd.DataFrame, xlsx_path: Path,
                        training_laga_max: float) -> pd.DataFrame:
    """Split cross-DOI 321 metrics by whether LA/GA is in or out of training range."""
    src = pd.read_excel(xlsx_path)
    src = src.rename(columns={"Formulation Index": "Formulation_Index"})
    src = src.drop_duplicates("Formulation_Index")[["Formulation_Index", "LA/GA"]]
    src["Formulation_Index"] = src["Formulation_Index"].astype(int)
    merged = cd.merge(src, on="Formulation_Index", how="left")
    merged["laga_range"] = np.where(
        merged["LA/GA"] <= training_laga_max,
        f"in_range (<={training_laga_max:g})",
        f"extrapolation (>{training_laga_max:g})",
    )
    rows = []
    for label, sub in merged.groupby("laga_range", sort=False):
        rows.append({
            "laga_range": label,
            "n": int(len(sub)),
            "median_LA_GA": float(sub["LA/GA"].median()),
            "median_R2": float(sub["R2"].median()),
            "mean_R2": float(sub["R2"].mean()),
            "median_MAE": float(sub["MAE"].median()),
            "frac_R2_pos": float((sub["R2"] > 0).mean()),
            "frac_R2_above_030": float((sub["R2"] > 0.30).mean()),
        })
    return pd.DataFrame(rows)


def _gef_curves(lodo: pd.DataFrame) -> pd.DataFrame:
    """ADR-019 flagged GEF as the only negative-R² fold; dump its 5 curves."""
    gef = lodo[lodo["drug_id"] == "GEF"].copy()
    return gef[[
        "Experimental_index", "DP_Group", "polymer_family", "n_obs",
        "R2", "MAE", "Q0_obs", "Q0_pred_median", "Q0_residual", "q_burst_mean",
    ]].sort_values("R2", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lodo-csv", type=Path,
                    default=Path("outputs/10_eval_deployment/per_curve_metrics.csv"))
    ap.add_argument("--cross-doi-csv", type=Path,
                    default=Path("outputs/10_eval_deployment/cross_doi_per_curve_metrics.csv"))
    ap.add_argument("--cross-doi-xlsx", type=Path,
                    default=Path(
                        r"D:\chemical-world-model-v0\datset\321PLGA"
                        r"\A Dataset on Formulation Parameters and Characteristics of "
                        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
                    ))
    ap.add_argument("--training-laga-max", type=float, default=3.0,
                    help="upper end of LA/GA seen in internal 181 training (excludes 5.67 extrapolation)")
    ap.add_argument("--out", type=Path, default=Path("outputs/13_phase1_audits"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lodo = pd.read_csv(args.lodo_csv)
    cd = pd.read_csv(args.cross_doi_csv)
    print(f"[13] LODO rows: {len(lodo)}; cross-DOI rows: {len(cd)}")

    tmax_df = _tmax_stratified(cd)
    laga_df = _laga_extrapolation(cd, args.cross_doi_xlsx, args.training_laga_max)
    gef_df = _gef_curves(lodo)

    tmax_df.to_csv(args.out / "cross_doi_by_tmax.csv", index=False)
    laga_df.to_csv(args.out / "cross_doi_by_laga_range.csv", index=False)
    gef_df.to_csv(args.out / "gef_lodo_curves.csv", index=False)

    summary_path = args.out / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== Phase 1 paper-readiness audits (script 13) ===\n\n")
        f.write("1. Cross-DOI 321 R^2 by observation-window length\n")
        f.write("   (small Var(Q_obs) on short windows blows up R^2 even with small MAE)\n")
        f.write(tmax_df.to_string(index=False))
        f.write("\n\n")
        f.write("2. Cross-DOI 321 R^2 split by LA/GA training-range vs extrapolation\n")
        f.write(f"   training_laga_max = {args.training_laga_max}\n")
        f.write(laga_df.to_string(index=False))
        f.write("\n\n")
        f.write("3. GEF leave-one-drug-out per-curve metrics\n")
        f.write("   (ADR-019 deep-dive item: only negative-R^2 drug fold)\n")
        f.write(gef_df.to_string(index=False))
        f.write("\n")

    print(f"\n[13] wrote {summary_path}")
    print("\n--- 1. cross-DOI by tmax ---")
    print(tmax_df.to_string(index=False))
    print("\n--- 2. cross-DOI by LA/GA range ---")
    print(laga_df.to_string(index=False))
    print("\n--- 3. GEF per-curve ---")
    print(gef_df.to_string(index=False))


if __name__ == "__main__":
    main()
