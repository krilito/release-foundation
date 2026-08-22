"""
Prepare three input CSVs for active_kinetic_observer.py from existing data.

Reads:
    data/Dataset_17_feat_augmented.csv  (181 PLGA curves, long format)
    outputs/27a_ode_expressivity_audit/per_curve.csv  (oracle-fitted theta)

Writes:
    data/formulations.csv   — one row per curve_id with formulation features
    data/curves_long.csv    — long format: curve_id, time, release
    data/theta_bank.csv     — curve_id + 9 theta columns
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_CSV = Path("data/Dataset_17_feat_augmented.csv")
THETA_CSV = Path("outputs/27a_ode_expressivity_audit/per_curve.csv")
OUT_DIR = Path("data")

ID_COL = "Experimental_index"
TIME_COL = "Time"
RELEASE_COL = "Release"
DP_GROUP_COL = "DP_Group"

FEATURE_COLS = [
    "LA/GA", "Polymer_MW", "CL Ratio", "Drug_Tm", "Drug_Pka",
    "Initial D/M ratio", "DLC", "SA-V", "SE",
    "Drug_Mw", "Drug_TPSA", "Drug_NHA", "Drug_LogP",
]

THETA_COLS = [
    "log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
    "m_crit", "q_burst", "log_tau_burst", "Q_max",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load raw data ---
    raw = pd.read_csv(DATA_CSV)
    theta_bank = pd.read_csv(THETA_CSV)

    # --- formulations.csv ---
    form = raw.groupby(ID_COL).first().reset_index()
    form_out = form[[ID_COL, DP_GROUP_COL, *FEATURE_COLS]].copy()
    form_out = form_out.rename(columns={ID_COL: "curve_id"})
    form_out.to_csv(OUT_DIR / "formulations.csv", index=False)
    print(f"[ok] formulations.csv: {form_out.shape[0]} rows, {form_out.shape[1]} cols")

    # --- curves_long.csv ---
    curves = raw[[ID_COL, TIME_COL, RELEASE_COL]].copy()
    curves = curves.rename(columns={ID_COL: "curve_id", TIME_COL: "time", RELEASE_COL: "release"})
    curves = curves.sort_values(["curve_id", "time"]).reset_index(drop=True)
    curves.to_csv(OUT_DIR / "curves_long.csv", index=False)
    print(f"[ok] curves_long.csv: {curves.shape[0]} rows")

    # --- theta_bank.csv ---
    # Match fid (in theta_bank) to curve_id (Experimental_index in 1..181)
    theta_sub = theta_bank[theta_bank["fid"].isin(form_out["curve_id"])].copy()
    theta_out = theta_sub[["fid", *THETA_COLS]].copy()
    theta_out = theta_out.rename(columns={"fid": "curve_id"})
    theta_out.to_csv(OUT_DIR / "theta_bank.csv", index=False)
    print(f"[ok] theta_bank.csv: {theta_out.shape[0]} rows")

    # --- Quick sanity ---
    c1 = set(form_out["curve_id"])
    c2 = set(curves["curve_id"])
    c3 = set(theta_out["curve_id"])
    common = c1 & c2 & c3
    print(f"[info] common curve_ids across all three files: {len(common)}")
    print(f"       formulations only: {len(c1 - c2 - c3)}")
    print(f"       curves only:       {len(c2 - c1 - c3)}")
    print(f"       theta only:        {len(c3 - c1 - c2)}")


if __name__ == "__main__":
    main()
