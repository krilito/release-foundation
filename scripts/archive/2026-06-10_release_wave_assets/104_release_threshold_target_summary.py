"""
104 - Summarize candidate deep-threshold targets across selected cells.

Purpose:
    Compare t50/t60/t70/t80 as candidate release-duration targets across the
    expanded chosen-cell bridge panel, focusing on truth reachability,
    interval closure, and eventual coverage.

Consumes:
    outputs/103_release_threshold_target_audit/cell_runs/*/threshold_curve_table.csv

Produces:
    outputs/104_release_threshold_target_summary/cell_threshold_metrics.csv
    outputs/104_release_threshold_target_summary/pattern_by_threshold.csv
    outputs/104_release_threshold_target_summary/threshold_target_scorecard.csv
    outputs/104_release_threshold_target_summary/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDIR = Path("outputs/103_release_threshold_target_audit/cell_runs")
OUTDIR = Path("outputs/104_release_threshold_target_summary")


CELL_META = {
    "cross321_group_by_drug_threshold_audit": {"dataset": "cross321", "scheme": "group_by_drug", "cell_label": "C321-D"},
    "cross321_group_by_polymer_threshold_audit": {"dataset": "cross321", "scheme": "group_by_polymer", "cell_label": "C321-P"},
    "internal181_group_by_drug_threshold_audit": {"dataset": "internal181", "scheme": "group_by_drug", "cell_label": "I181-D"},
    "internal181_group_by_polymer_threshold_audit": {"dataset": "internal181", "scheme": "group_by_polymer", "cell_label": "I181-P"},
    "liposome_group_by_drug_threshold_audit": {"dataset": "liposome", "scheme": "group_by_drug", "cell_label": "Lipo-D"},
    "liposome_group_by_method_threshold_audit": {"dataset": "liposome", "scheme": "group_by_polymer", "cell_label": "Lipo-P"},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTDIR)
    rows = []

    for dirname, meta in CELL_META.items():
        df = pd.read_csv(INDIR / dirname / "threshold_curve_table.csv")
        df["truth_reached"] = df["t_true"].notna()
        df["interval_closed"] = df["t_interval_low"].notna() & df["t_interval_high"].notna()
        for (threshold, method), sub in df.groupby(["threshold", "method"], dropna=False):
            rows.append(
                {
                    **meta,
                    "threshold": float(threshold),
                    "threshold_label": f"t{int(float(threshold) * 100)}",
                    "method": str(method),
                    "n_curves": int(sub["curve_id"].nunique()),
                    "truth_reached_fraction": float(sub["truth_reached"].mean()),
                    "interval_closed_fraction": float(sub["interval_closed"].mean()),
                    "coverage_fraction": float(sub["covered"].mean()),
                }
            )

    cell_df = pd.DataFrame(rows).sort_values(["cell_label", "threshold", "method"])
    cell_df.to_csv(OUTDIR / "cell_threshold_metrics.csv", index=False)

    pattern_rows = []
    for (threshold_label, method), sub in cell_df.groupby(["threshold_label", "method"], dropna=False):
        pattern_rows.append(
            {
                "threshold_label": threshold_label,
                "method": method,
                "mean_truth_reached_fraction": float(sub["truth_reached_fraction"].mean()),
                "min_truth_reached_fraction": float(sub["truth_reached_fraction"].min()),
                "max_truth_reached_fraction": float(sub["truth_reached_fraction"].max()),
                "mean_interval_closed_fraction": float(sub["interval_closed_fraction"].mean()),
                "min_interval_closed_fraction": float(sub["interval_closed_fraction"].min()),
                "max_interval_closed_fraction": float(sub["interval_closed_fraction"].max()),
                "mean_coverage_fraction": float(sub["coverage_fraction"].mean()),
                "min_coverage_fraction": float(sub["coverage_fraction"].min()),
                "max_coverage_fraction": float(sub["coverage_fraction"].max()),
            }
        )
    pattern_df = pd.DataFrame(pattern_rows).sort_values(["threshold_label", "method"])
    pattern_df.to_csv(OUTDIR / "pattern_by_threshold.csv", index=False)

    score_rows = []
    for threshold_label, sub in cell_df[cell_df["method"] == "global"].groupby("threshold_label", dropna=False):
        score_rows.append(
            {
                "threshold_label": threshold_label,
                "mean_truth_reached_fraction": float(sub["truth_reached_fraction"].mean()),
                "mean_interval_closed_fraction": float(sub["interval_closed_fraction"].mean()),
                "mean_coverage_fraction": float(sub["coverage_fraction"].mean()),
                "viability_rank_hint": float(sub["truth_reached_fraction"].mean() + sub["interval_closed_fraction"].mean() + sub["coverage_fraction"].mean()),
            }
        )
    score_df = pd.DataFrame(score_rows).sort_values("viability_rank_hint", ascending=False)
    score_df.to_csv(OUTDIR / "threshold_target_scorecard.csv", index=False)

    def line_for(thr: str) -> str:
        sub = pattern_df[(pattern_df["threshold_label"] == thr) & (pattern_df["method"] == "global")]
        if sub.empty:
            return f"  {thr}: missing"
        row = sub.iloc[0]
        return (
            f"  {thr}: truth={row['mean_truth_reached_fraction']:.3f}, "
            f"closed={row['mean_interval_closed_fraction']:.3f}, "
            f"covered={row['mean_coverage_fraction']:.3f}"
        )

    lines = [
        "=== 104 -- threshold target summary ===",
        "",
        "Global-method cross-cell means:",
        line_for("t50"),
        line_for("t60"),
        line_for("t70"),
        line_for("t80"),
        "",
        "Interpretation:",
        "  1. A viable unified timing target should maintain both reasonable truth reachability and reasonable closed-interval rate across cells.",
        "  2. If coverage drops mainly because intervals fail to close, the threshold is structurally too deep for a generic shared target layer.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
