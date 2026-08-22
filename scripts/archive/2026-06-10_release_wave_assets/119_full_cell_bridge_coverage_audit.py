"""
119 - Audit matched-curve coverage for the current 9-cell family bridge panel.

Purpose:
    Quantify whether the current 9-cell route-consistent duration/shape panel
    covers every matched benchmark curve in each cell, or whether some cells
    still use within-cell subsampling.

Consumes:
    - outputs/87_release_conformal_family_export/<dataset>/<scheme>/summary.csv
    - outputs/80_.../prediction_trajectories.csv
    - outputs/99_release_family_duration_shape_cell_bridge/*/threshold_summary.csv

Produces:
    outputs/119_full_cell_bridge_coverage_audit/
        coverage_audit.csv
        summary.txt
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path("outputs")
OUTDIR = ROOT / "119_full_cell_bridge_coverage_audit"


CELL_META = [
    {
        "dataset": "cross321",
        "scheme": "group_by_drug",
        "cell_label": "C321-D",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/cross321/group_by_drug/summary.csv",
        "trajectory_csv": ROOT / "80_cross321_group_by_drug_theta_rf_ztheta_leaf2/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/cross321_group_by_drug_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "cross321",
        "scheme": "group_by_polymer",
        "cell_label": "C321-P",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/cross321/group_by_polymer/summary.csv",
        "trajectory_csv": ROOT / "80_cross321_group_by_polymer_theta_early_select/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/cross321_group_by_polymer_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "cross321",
        "scheme": "random_5fold",
        "cell_label": "C321-R",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/cross321/random_5fold/summary.csv",
        "trajectory_csv": ROOT / "80_cross321_group_by_drug_theta_rf_ztheta_leaf2/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/cross321_random_5fold_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "internal181",
        "scheme": "group_by_drug",
        "cell_label": "I181-D",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/internal181/group_by_drug/summary.csv",
        "trajectory_csv": ROOT / "80_internal181_group_by_drug_theta_early_select/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/internal181_group_by_drug_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "internal181",
        "scheme": "group_by_polymer",
        "cell_label": "I181-P",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/internal181/group_by_polymer/summary.csv",
        "trajectory_csv": ROOT / "80_internal181_group_by_polymer_theta_early_select/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/internal181_group_by_polymer_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "internal181",
        "scheme": "random_5fold",
        "cell_label": "I181-R",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/internal181/random_5fold/summary.csv",
        "trajectory_csv": ROOT / "80_internal181_group_by_drug_theta_early_select/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/internal181_random_5fold_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "liposome",
        "scheme": "group_by_drug",
        "cell_label": "Lipo-D",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/liposome/group_by_drug/summary.csv",
        "trajectory_csv": ROOT / "80_liposome_group_by_api_our_et_refined/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/liposome_group_by_drug_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "liposome",
        "scheme": "group_by_polymer",
        "cell_label": "Lipo-P",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/liposome/group_by_polymer/summary.csv",
        "trajectory_csv": ROOT / "80_liposome_group_by_method_our_et_refined/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/liposome_group_by_method_cell_bridge/threshold_summary.csv",
    },
    {
        "dataset": "liposome",
        "scheme": "random_5fold",
        "cell_label": "Lipo-R",
        "family_summary_csv": ROOT / "87_release_conformal_family_export/liposome/random_5fold/summary.csv",
        "trajectory_csv": ROOT / "80_liposome_group_by_api_our_et_refined/prediction_trajectories.csv",
        "bridge_threshold_csv": ROOT / "99_release_family_duration_shape_cell_bridge/liposome_random_5fold_cell_bridge/threshold_summary.csv",
    },
]


def _read_curve_ids(csv_path: Path, curve_col: str = "curve_id") -> set[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row[curve_col] for row in csv.DictReader(f)}


def _read_used_curve_count(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0
    return int(df["n_curves"].iloc[0])


def _classify(matched_count: int, used_count: int) -> str:
    if matched_count <= 0:
        return "no_matched_curves"
    if used_count >= matched_count:
        return "full_matched_curve_coverage"
    return "within_cell_curve_subsample"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for meta in CELL_META:
        family_ids = _read_curve_ids(meta["family_summary_csv"])
        trajectory_ids = _read_curve_ids(meta["trajectory_csv"])
        matched_ids = family_ids & trajectory_ids
        used_count = _read_used_curve_count(meta["bridge_threshold_csv"])
        matched_count = len(matched_ids)
        rows.append(
            {
                "dataset": meta["dataset"],
                "scheme": meta["scheme"],
                "cell_label": meta["cell_label"],
                "family_unique_curves": len(family_ids),
                "trajectory_unique_curves": len(trajectory_ids),
                "matched_curve_count": matched_count,
                "bridge_used_curve_count": used_count,
                "matched_curve_fraction_used": float(used_count / matched_count) if matched_count else float("nan"),
                "coverage_scope": _classify(matched_count, used_count),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "coverage_audit.csv", index=False)

    full_count = int((df["coverage_scope"] == "full_matched_curve_coverage").sum())
    sampled_count = int((df["coverage_scope"] == "within_cell_curve_subsample").sum())
    mean_fraction = float(df["matched_curve_fraction_used"].mean())
    min_fraction = float(df["matched_curve_fraction_used"].min())
    max_fraction = float(df["matched_curve_fraction_used"].max())

    lines = [
        "=== 119 -- full cell bridge coverage audit ===",
        "",
        "Interpretation",
        f"  1. Cells audited: {len(df)}",
        f"  2. Full matched-curve coverage cells: {full_count}",
        f"  3. Within-cell subsampled cells: {sampled_count}",
        f"  4. Mean matched-curve fraction used: {mean_fraction:.3f}",
        f"  5. Min / max matched-curve fraction used: {min_fraction:.3f} / {max_fraction:.3f}",
        "",
        "Current boundary",
        "  The present 9-cell bridge is cell-panel complete, but several cells still use within-cell curve subsampling rather than full matched-curve coverage.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
