"""
102 - Audit timing failure modes in the selected-cell family bridge layer.

Purpose:
    Classify why route-consistent timing uncertainty fails on the expanded
    chosen-cell panel, focusing on threshold timing rather than shape.

Consumes:
    outputs/99_release_family_duration_shape_cell_bridge/*/threshold_curve_table.csv

Produces:
    outputs/102_release_timing_failure_audit/classified_rows.csv
    outputs/102_release_timing_failure_audit/failure_mode_by_threshold_method.csv
    outputs/102_release_timing_failure_audit/failure_mode_by_cell.csv
    outputs/102_release_timing_failure_audit/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDIR = Path("outputs/99_release_family_duration_shape_cell_bridge")
OUTDIR = Path("outputs/102_release_timing_failure_audit")


CELL_META = {
    "cross321_group_by_drug_cell_bridge": {"dataset": "cross321", "scheme": "group_by_drug", "cell_label": "C321-D"},
    "cross321_group_by_polymer_cell_bridge": {"dataset": "cross321", "scheme": "group_by_polymer", "cell_label": "C321-P"},
    "internal181_group_by_drug_cell_bridge": {"dataset": "internal181", "scheme": "group_by_drug", "cell_label": "I181-D"},
    "internal181_group_by_polymer_cell_bridge": {"dataset": "internal181", "scheme": "group_by_polymer", "cell_label": "I181-P"},
    "liposome_group_by_drug_cell_bridge": {"dataset": "liposome", "scheme": "group_by_drug", "cell_label": "Lipo-D"},
    "liposome_group_by_method_cell_bridge": {"dataset": "liposome", "scheme": "group_by_polymer", "cell_label": "Lipo-P"},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_row(row: pd.Series) -> str:
    t_true = float(row["t_true"]) if pd.notna(row["t_true"]) else np.nan
    t_lo = float(row["t_interval_low"]) if pd.notna(row["t_interval_low"]) else np.nan
    t_hi = float(row["t_interval_high"]) if pd.notna(row["t_interval_high"]) else np.nan

    if np.isnan(t_true):
        if np.isnan(t_lo) and np.isnan(t_hi):
            return "truth_not_reached_and_interval_missing"
        if np.isnan(t_hi):
            return "truth_not_reached_but_interval_open_upper"
        if np.isnan(t_lo):
            return "truth_not_reached_but_interval_open_lower"
        return "truth_not_reached_but_closed_interval_predicted"
    if np.isnan(t_lo) and np.isnan(t_hi):
        return "truth_reached_but_interval_missing"
    if np.isnan(t_hi):
        return "truth_reached_but_interval_open_upper"
    if np.isnan(t_lo):
        return "truth_reached_but_interval_open_lower"
    if t_true < t_lo:
        return "closed_interval_too_late"
    if t_true > t_hi:
        return "closed_interval_too_early"
    return "covered"


def main() -> None:
    ensure_dir(OUTDIR)
    rows = []
    for name, meta in CELL_META.items():
        df = pd.read_csv(INDIR / name / "threshold_curve_table.csv")
        for _, row in df.iterrows():
            item = row.to_dict()
            item.update(meta)
            item["failure_mode"] = classify_row(row)
            item["threshold_label"] = f"t{int(float(row['threshold']) * 100)}"
            rows.append(item)

    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUTDIR / "classified_rows.csv", index=False)

    focus = all_df[all_df["threshold"].isin([0.5, 0.8])].copy()

    agg_tm = (
        focus.groupby(["threshold_label", "method", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    total_tm = focus.groupby(["threshold_label", "method"], dropna=False).size().reset_index(name="n_total")
    agg_tm = agg_tm.merge(total_tm, on=["threshold_label", "method"], how="left")
    agg_tm["fraction"] = agg_tm["n_rows"] / agg_tm["n_total"]
    agg_tm = agg_tm.sort_values(["threshold_label", "method", "fraction"], ascending=[True, True, False])
    agg_tm.to_csv(OUTDIR / "failure_mode_by_threshold_method.csv", index=False)

    agg_cell = (
        focus.groupby(["cell_label", "threshold_label", "method", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    total_cell = focus.groupby(["cell_label", "threshold_label", "method"], dropna=False).size().reset_index(name="n_total")
    agg_cell = agg_cell.merge(total_cell, on=["cell_label", "threshold_label", "method"], how="left")
    agg_cell["fraction"] = agg_cell["n_rows"] / agg_cell["n_total"]
    agg_cell = agg_cell.sort_values(["cell_label", "threshold_label", "method", "fraction"], ascending=[True, True, True, False])
    agg_cell.to_csv(OUTDIR / "failure_mode_by_cell.csv", index=False)

    def top_modes(threshold_label: str, method: str) -> list[str]:
        sub = agg_tm[(agg_tm["threshold_label"] == threshold_label) & (agg_tm["method"] == method)]
        return [f"{r.failure_mode}={r.fraction:.3f}" for r in sub.head(3).itertuples()]

    lines = [
        "=== 102 -- timing failure audit ===",
        "",
        "Panel:",
        "  C321-D, C321-P, I181-D, I181-P, Lipo-D, Lipo-P",
        "",
        "Top failure modes by threshold/method:",
        f"  t50 raw   : {', '.join(top_modes('t50', 'raw'))}",
        f"  t50 global: {', '.join(top_modes('t50', 'global'))}",
        f"  t50 local : {', '.join(top_modes('t50', 'local'))}",
        f"  t80 raw   : {', '.join(top_modes('t80', 'raw'))}",
        f"  t80 global: {', '.join(top_modes('t80', 'global'))}",
        f"  t80 local : {', '.join(top_modes('t80', 'local'))}",
        "",
        "Interpretation:",
        "  1. t80 failures are expected to be dominated by open-upper intervals and truth-not-reached cases rather than small closed-interval miscalibration.",
        "  2. This would indicate a structural release-band reachability bottleneck, not just a conformal scaling bottleneck.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
