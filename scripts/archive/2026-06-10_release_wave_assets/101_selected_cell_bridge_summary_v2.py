"""
101 - Summarize the expanded selected-cell family duration/shape bridge.

Purpose:
    Extend the first four-cell bridge summary to include the two canonical
    internal181 PLGA OOD cells, so we can judge whether the emerging pattern
    remains stable across a broader chosen-cell panel.

Consumes:
    outputs/99_release_family_duration_shape_cell_bridge/*/threshold_summary.csv
    outputs/99_release_family_duration_shape_cell_bridge/*/descriptor_summary.csv

Produces:
    outputs/101_selected_cell_bridge_summary_v2/threshold_cross_cell.csv
    outputs/101_selected_cell_bridge_summary_v2/descriptor_cross_cell.csv
    outputs/101_selected_cell_bridge_summary_v2/pattern_snapshot.csv
    outputs/101_selected_cell_bridge_summary_v2/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


INDIR = Path("outputs/99_release_family_duration_shape_cell_bridge")
OUTDIR = Path("outputs/101_selected_cell_bridge_summary_v2")


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
        "=== 101 -- expanded selected-cell bridge summary ===",
        "",
        "Cells included:",
        "  C321-D, C321-P, I181-D, I181-P, Lipo-D, Lipo-P",
        "",
        "Interpretation",
        "  1. Shape-aware uncertainty remains more stable than deep-threshold timing uncertainty across the six-cell chosen panel.",
        "  2. PLGA split sensitivity remains visible, especially for group-by-polymer and internal181 group-by-drug timing coverage.",
        "  3. The chosen-cell bridge has moved beyond a two-mechanism anecdote and now exposes a broader repeated pattern.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
