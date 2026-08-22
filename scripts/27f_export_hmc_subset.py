"""27f - Export locked HMC subset inputs for the B5 ground-truth benchmark.

Purpose:
    B5 requires a from-zero HMC / NUTS posterior benchmark, but the sampler
    implementation should not decide the benchmark mouth. This script locks a
    reproducible subset and exports the exact tables a future HMC runner should
    consume.

Primary claim subset:
    - `canonical_test`: all 38 curves from `data/canonical_split_v1.csv`
      whose `split_name == "test"`.

Extended development subset:
    - `canonical_test_plus_cal12`: the 38 canonical test curves plus 12
      calibration curves. The extension is deterministic:
        1. prefer calibration groups not already present in the test split
        2. then fill the remaining slots in canonical `fold_index` order

Outputs:
    outputs/27f_hmc_subset_export/
        subset_manifest.csv
        formulations.csv
        curves_long.csv
        curve_grid_14pt.csv
        fixed_four_mouth.csv
        theta_reference.csv
        lock_metadata.json
        summary.txt

Notes:
    - `theta_reference.csv` is exported only as a simulator-side sanity anchor.
      It is not part of the real-data HMC fitting target.
    - The primary paper-facing comparator should still report on the
      canonical test subset. The 50-curve extension is for development /
      runtime amortization only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")
FORMULATIONS_PATH = Path("data/formulations.csv")
CURVES_PATH = Path("data/curves_long.csv")
THETA_BANK_PATH = Path("data/theta_bank.csv")
DEFAULT_OUT = Path("outputs/27f_hmc_subset_export")

CANONICAL_TIME_GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
FIXED_FOUR_TIMES = np.array([1.0, 3.0, 5.0, 7.0])
FUTURE_JUDGE_TIMES = np.array([21.0, 28.0, 42.0, 56.0, 84.0])


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_df = pd.read_csv(CANONICAL_SPLIT_PATH)
    formulations = pd.read_csv(FORMULATIONS_PATH)
    curves = pd.read_csv(CURVES_PATH)
    theta_df = pd.read_csv(THETA_BANK_PATH)

    required = {"curve_id", "split_name", "fold_index"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Canonical split missing columns: {sorted(missing)}")

    split_df = split_df.copy()
    split_df["curve_id"] = split_df["curve_id"].astype(int)
    split_df["split_name"] = split_df["split_name"].astype(str)
    split_df["fold_index"] = split_df["fold_index"].astype(int)
    split_df = split_df.sort_values("fold_index").reset_index(drop=True)

    curves = curves.copy()
    curves["curve_id"] = curves["curve_id"].astype(int)
    curves["time"] = curves["time"].astype(float)
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    formulations["curve_id"] = formulations["curve_id"].astype(int)
    theta_df["curve_id"] = theta_df["curve_id"].astype(int)
    return split_df, formulations, curves, theta_df


def choose_extension_rows(cal_rows: pd.DataFrame, n_extension: int) -> pd.DataFrame:
    if n_extension <= 0:
        return cal_rows.iloc[0:0].copy()

    test_groups = set()
    if "test_group_seen" in cal_rows.columns:
        test_groups = set(cal_rows.loc[cal_rows["test_group_seen"], "DP_Group"])

    novelty = cal_rows.loc[~cal_rows["DP_Group"].isin(test_groups)].copy()
    novelty = novelty.sort_values("fold_index")
    chosen = novelty.head(n_extension)
    if len(chosen) == n_extension:
        return chosen.reset_index(drop=True)

    remaining = cal_rows.loc[~cal_rows["curve_id"].isin(chosen["curve_id"])].sort_values("fold_index")
    fill = remaining.head(n_extension - len(chosen))
    return pd.concat([chosen, fill], ignore_index=True)


def build_subset_manifest(
    split_df: pd.DataFrame,
    formulations: pd.DataFrame,
    mode: str,
    extension_count: int,
) -> pd.DataFrame:
    manifest = split_df.merge(formulations[["curve_id", "DP_Group"]], on="curve_id", how="left")
    missing_groups = manifest["DP_Group"].isna().sum()
    if missing_groups:
        raise ValueError(f"{missing_groups} split rows are missing DP_Group in formulations.csv")

    test_rows = manifest.loc[manifest["split_name"] == "test"].copy()
    test_rows["subset_role"] = "claim_test"

    if mode == "canonical_test":
        subset = test_rows
    elif mode == "canonical_test_plus_cal12":
        cal_rows = manifest.loc[manifest["split_name"] == "cal"].copy()
        test_groups = set(test_rows["DP_Group"])
        cal_rows["test_group_seen"] = cal_rows["DP_Group"].isin(test_groups)
        extension = choose_extension_rows(cal_rows, extension_count)
        extension["subset_role"] = "dev_extension"
        subset = pd.concat([test_rows, extension], ignore_index=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    subset = subset.sort_values(["subset_role", "fold_index", "curve_id"]).reset_index(drop=True)
    return subset


def interpolate_curve(curves: pd.DataFrame, curve_id: int, times: np.ndarray) -> np.ndarray:
    cdf = curves.loc[curves["curve_id"] == curve_id].sort_values("time")
    if cdf.empty:
        raise ValueError(f"curve_id {curve_id} missing from curves_long.csv")
    t = cdf["time"].to_numpy()
    y = cdf["release"].to_numpy()
    return np.interp(times, t, y, left=y[0], right=y[-1])


def export_subset(
    subset: pd.DataFrame,
    formulations: pd.DataFrame,
    curves: pd.DataFrame,
    theta_df: pd.DataFrame,
    out_dir: Path,
    mode: str,
    extension_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_ids = subset["curve_id"].tolist()

    subset_formulations = formulations.loc[formulations["curve_id"].isin(subset_ids)].copy()
    subset_formulations = subset_formulations.set_index("curve_id").loc[subset_ids].reset_index()

    subset_curves = curves.loc[curves["curve_id"].isin(subset_ids)].copy()
    subset_theta = theta_df.loc[theta_df["curve_id"].isin(subset_ids)].copy()
    subset_theta = subset_theta.set_index("curve_id").loc[subset_ids].reset_index()

    curve_rows = []
    mouth_rows = []
    raw_counts = []
    for cid in subset_ids:
        grid_values = interpolate_curve(curves, cid, CANONICAL_TIME_GRID)
        mouth_values = interpolate_curve(curves, cid, FIXED_FOUR_TIMES)
        future_values = interpolate_curve(curves, cid, FUTURE_JUDGE_TIMES)
        curve_row = {"curve_id": cid}
        curve_row.update({f"release_t{t:g}": float(v) for t, v in zip(CANONICAL_TIME_GRID, grid_values)})
        curve_rows.append(curve_row)

        mouth_row = {"curve_id": cid}
        mouth_row.update({f"obs_release_t{t:g}": float(v) for t, v in zip(FIXED_FOUR_TIMES, mouth_values)})
        mouth_row.update({f"target_release_t{t:g}": float(v) for t, v in zip(FUTURE_JUDGE_TIMES, future_values)})
        mouth_rows.append(mouth_row)

        raw_counts.append(
            {
                "curve_id": cid,
                "n_raw_points": int((subset_curves["curve_id"] == cid).sum()),
                "t_max": float(subset_curves.loc[subset_curves["curve_id"] == cid, "time"].max()),
            }
        )

    curve_grid_df = pd.DataFrame(curve_rows)
    mouth_df = pd.DataFrame(mouth_rows)
    counts_df = pd.DataFrame(raw_counts)
    manifest_out = subset.merge(counts_df, on="curve_id", how="left")

    manifest_out.to_csv(out_dir / "subset_manifest.csv", index=False)
    subset_formulations.to_csv(out_dir / "formulations.csv", index=False)
    subset_curves.to_csv(out_dir / "curves_long.csv", index=False)
    curve_grid_df.to_csv(out_dir / "curve_grid_14pt.csv", index=False)
    mouth_df.to_csv(out_dir / "fixed_four_mouth.csv", index=False)
    subset_theta.to_csv(out_dir / "theta_reference.csv", index=False)

    role_counts = manifest_out["subset_role"].value_counts().to_dict()
    split_counts = manifest_out["split_name"].value_counts().to_dict()
    group_counts = (
        manifest_out.groupby(["subset_role", "DP_Group"]).size().reset_index(name="count").to_dict(orient="records")
    )
    metadata = {
        "mode": mode,
        "canonical_split_path": str(CANONICAL_SPLIT_PATH),
        "formulations_path": str(FORMULATIONS_PATH),
        "curves_path": str(CURVES_PATH),
        "theta_bank_path": str(THETA_BANK_PATH),
        "extension_count": int(extension_count),
        "n_curves": int(len(manifest_out)),
        "role_counts": role_counts,
        "split_counts": split_counts,
        "time_grid_days": CANONICAL_TIME_GRID.tolist(),
        "fixed_four_times_days": FIXED_FOUR_TIMES.tolist(),
        "future_judge_times_days": FUTURE_JUDGE_TIMES.tolist(),
        "group_counts": group_counts,
        "notes": [
            "claim-facing benchmark subset is subset_role=claim_test",
            "dev_extension rows are deterministic calibration additions for runtime amortization only",
            "theta_reference.csv is a simulator-side sanity anchor, not a real-data HMC fitting target",
        ],
    }
    (out_dir / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        f"mode: {mode}",
        f"n_curves: {len(manifest_out)}",
        f"role_counts: {role_counts}",
        f"split_counts: {split_counts}",
        f"dp_groups: {manifest_out['DP_Group'].nunique()}",
        "",
        "Claim-facing rows:",
        f"  claim_test = {role_counts.get('claim_test', 0)}",
        f"  dev_extension = {role_counts.get('dev_extension', 0)}",
        "",
        "Files:",
        "  subset_manifest.csv",
        "  formulations.csv",
        "  curves_long.csv",
        "  curve_grid_14pt.csv",
        "  fixed_four_mouth.csv",
        "  theta_reference.csv",
        "  lock_metadata.json",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canonical_test", "canonical_test_plus_cal12"), default="canonical_test")
    parser.add_argument("--extension-count", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    split_df, formulations, curves, theta_df = load_tables()
    subset = build_subset_manifest(split_df, formulations, args.mode, args.extension_count)
    export_subset(subset, formulations, curves, theta_df, args.out, args.mode, args.extension_count)

    role_counts = subset["subset_role"].value_counts().to_dict()
    split_counts = subset["split_name"].value_counts().to_dict()
    print(f"Exported {len(subset)} curves to {args.out}")
    print(f"role_counts={role_counts}")
    print(f"split_counts={split_counts}")


if __name__ == "__main__":
    main()
