"""
117 - Summarize the full current 9-cell family duration/shape cell bridge.

Purpose:
    Extend the 6-cell selected bridge summary to cover the full current
    calibrated-family cell panel:
    - cross321: group_by_drug, group_by_polymer, random_5fold
    - internal181: group_by_drug, group_by_polymer, random_5fold
    - liposome: group_by_drug, group_by_polymer, random_5fold

Consumes:
    outputs/99_release_family_duration_shape_cell_bridge/*/threshold_summary.csv
    outputs/99_release_family_duration_shape_cell_bridge/*/descriptor_summary.csv

Produces:
    outputs/117_full_cell_bridge_summary/
        threshold_cross_cell.csv
        descriptor_cross_cell.csv
        pattern_snapshot.csv
        summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


INDIR = Path("outputs/99_release_family_duration_shape_cell_bridge")
OUTDIR = Path("outputs/117_full_cell_bridge_summary")


CELL_META = {
    "cross321_group_by_drug_cell_bridge": {"dataset": "cross321", "scheme": "group_by_drug", "cell_label": "C321-D"},
    "cross321_group_by_polymer_cell_bridge": {"dataset": "cross321", "scheme": "group_by_polymer", "cell_label": "C321-P"},
    "cross321_random_5fold_cell_bridge": {"dataset": "cross321", "scheme": "random_5fold", "cell_label": "C321-R"},
    "internal181_group_by_drug_cell_bridge": {"dataset": "internal181", "scheme": "group_by_drug", "cell_label": "I181-D"},
    "internal181_group_by_polymer_cell_bridge": {"dataset": "internal181", "scheme": "group_by_polymer", "cell_label": "I181-P"},
    "internal181_random_5fold_cell_bridge": {"dataset": "internal181", "scheme": "random_5fold", "cell_label": "I181-R"},
    "liposome_group_by_drug_cell_bridge": {"dataset": "liposome", "scheme": "group_by_drug", "cell_label": "Lipo-D"},
    "liposome_group_by_method_cell_bridge": {"dataset": "liposome", "scheme": "group_by_polymer", "cell_label": "Lipo-P"},
    "liposome_random_5fold_cell_bridge": {"dataset": "liposome", "scheme": "random_5fold", "cell_label": "Lipo-R"},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTDIR)
    thr_rows = []
    desc_rows = []

    for name, meta in CELL_META.items():
        base = INDIR / name
        thr = pd.read_csv(base / "threshold_summary.csv")
        desc = pd.read_csv(base / "descriptor_summary.csv")
        for _, row in thr.iterrows():
            item = row.to_dict()
            item.update(meta)
            thr_rows.append(item)
        for _, row in desc.iterrows():
            item = row.to_dict()
            item.update(meta)
            desc_rows.append(item)

    thr_df = pd.DataFrame(thr_rows)
    desc_df = pd.DataFrame(desc_rows)
    thr_df.to_csv(OUTDIR / "threshold_cross_cell.csv", index=False)
    desc_df.to_csv(OUTDIR / "descriptor_cross_cell.csv", index=False)

    pattern_rows = []
    for method in sorted(thr_df["method"].unique()):
        for threshold in [0.1, 0.5, 0.8]:
            sub = thr_df[(thr_df["method"] == method) & (thr_df["threshold"] == threshold)]
            pattern_rows.append(
                {
                    "component_type": "threshold",
                    "method": method,
                    "component": f"t{int(threshold * 100)}",
                    "mean_coverage_across_cells": float(sub["coverage"].mean()),
                    "min_coverage_across_cells": float(sub["coverage"].min()),
                    "max_coverage_across_cells": float(sub["coverage"].max()),
                }
            )
        for descriptor in ["burst", "post_window", "residual_tail", "tail_auc"]:
            sub = desc_df[(desc_df["method"] == method) & (desc_df["descriptor"] == descriptor)]
            pattern_rows.append(
                {
                    "component_type": "descriptor",
                    "method": method,
                    "component": descriptor,
                    "mean_coverage_across_cells": float(sub["coverage"].mean()),
                    "min_coverage_across_cells": float(sub["coverage"].min()),
                    "max_coverage_across_cells": float(sub["coverage"].max()),
                }
            )
    pattern_df = pd.DataFrame(pattern_rows)
    pattern_df.to_csv(OUTDIR / "pattern_snapshot.csv", index=False)

    lines = [
        "=== 117 -- full 9-cell bridge summary ===",
        "",
        "Cells included:",
        "  C321-D, C321-P, C321-R, I181-D, I181-P, I181-R, Lipo-D, Lipo-P, Lipo-R",
        "",
        "Interpretation",
        "  1. Route-consistent duration/shape uncertainty is now summarized across the full current 9-cell calibrated-family cell panel.",
        "  2. Coverage is mixed: the three liposome cells plus all three internal181 cells now have full matched-curve coverage, while the remaining three cross321 cells still use within-cell curve subsampling.",
        "  3. Shape-aware uncertainty still remains much more stable than deep-threshold timing uncertainty.",
        "  4. The random_5fold additions sharpen the story rather than simplifying it: C321-R is relatively strong, Lipo-R keeps timing alive, and I181-R remains a harsh timing failure case.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
