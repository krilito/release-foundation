"""
105 - Build a mechanism-aware release target-layer contract.

Purpose:
    Convert the selected-cell threshold audit into a concrete target-layer
    scorecard for PLGA, liposome, and prospective chitosan placeholders.

Consumes:
    outputs/104_release_threshold_target_summary/cell_threshold_metrics.csv

Produces:
    outputs/105_release_target_layer_contract/mechanism_target_scorecard.csv
    outputs/105_release_target_layer_contract/mechanism_target_aggregate.csv
    outputs/105_release_target_layer_contract/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


INDIR = Path("outputs/104_release_threshold_target_summary")
OUTDIR = Path("outputs/105_release_target_layer_contract")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_target(truth: float, closed: float, covered: float) -> str:
    if truth >= 0.8 and closed >= 0.35 and covered >= 0.2:
        return "candidate_default"
    if truth >= 0.5 and closed >= 0.15:
        return "secondary_only"
    return "not_default"


def mechanism_from_dataset(dataset: str) -> str:
    if dataset in {"cross321", "internal181"}:
        return "plga"
    return dataset


def main() -> None:
    ensure_dir(OUTDIR)
    df = pd.read_csv(INDIR / "cell_threshold_metrics.csv")
    df = df[df["method"] == "global"].copy()
    df["mechanism"] = df["dataset"].map(mechanism_from_dataset)

    agg = (
        df.groupby(["mechanism", "threshold_label"], as_index=False)
        .agg(
            mean_truth_reached_fraction=("truth_reached_fraction", "mean"),
            min_truth_reached_fraction=("truth_reached_fraction", "min"),
            mean_interval_closed_fraction=("interval_closed_fraction", "mean"),
            min_interval_closed_fraction=("interval_closed_fraction", "min"),
            mean_coverage_fraction=("coverage_fraction", "mean"),
            min_coverage_fraction=("coverage_fraction", "min"),
        )
    )
    agg["recommendation"] = agg.apply(
        lambda r: classify_target(
            float(r["mean_truth_reached_fraction"]),
            float(r["mean_interval_closed_fraction"]),
            float(r["mean_coverage_fraction"]),
        ),
        axis=1,
    )
    agg.to_csv(OUTDIR / "mechanism_target_aggregate.csv", index=False)

    rows = []
    for _, row in agg.iterrows():
        caveat = ""
        if row["mechanism"] == "plga" and row["threshold_label"] == "t50":
            caveat = "least_bad_plga_anchor_but_not_stable_across_all_ood_families"
        elif row["mechanism"] == "plga" and row["threshold_label"] in {"t60", "t70"}:
            caveat = "exploratory_only_due_to_low_interval_closure"
        elif row["mechanism"] == "plga" and row["threshold_label"] == "t80":
            caveat = "do_not_use_as_shared_default"
        elif row["mechanism"] == "liposome" and row["threshold_label"] == "t50":
            caveat = "pilot_default_only_under_current_assay_horizon"
        elif row["mechanism"] == "liposome" and row["threshold_label"] in {"t60", "t70", "t80"}:
            caveat = "horizon_limited_and_not_a_good_shared_default"
        rows.append(
            {
                "mechanism": row["mechanism"],
                "threshold_label": row["threshold_label"],
                "recommendation": row["recommendation"],
                "mean_truth_reached_fraction": row["mean_truth_reached_fraction"],
                "mean_interval_closed_fraction": row["mean_interval_closed_fraction"],
                "mean_coverage_fraction": row["mean_coverage_fraction"],
                "caveat": caveat,
            }
        )

    rows.extend(
        [
            {
                "mechanism": "chitosan",
                "threshold_label": "t50",
                "recommendation": "unverified_prospective_placeholder",
                "mean_truth_reached_fraction": None,
                "mean_interval_closed_fraction": None,
                "mean_coverage_fraction": None,
                "caveat": "no_reveal_yet_curve_coverage_only_preregistered",
            },
            {
                "mechanism": "chitosan",
                "threshold_label": "t60",
                "recommendation": "unverified_prospective_placeholder",
                "mean_truth_reached_fraction": None,
                "mean_interval_closed_fraction": None,
                "mean_coverage_fraction": None,
                "caveat": "no_reveal_yet_curve_coverage_only_preregistered",
            },
            {
                "mechanism": "chitosan",
                "threshold_label": "t70",
                "recommendation": "unverified_prospective_placeholder",
                "mean_truth_reached_fraction": None,
                "mean_interval_closed_fraction": None,
                "mean_coverage_fraction": None,
                "caveat": "no_reveal_yet_curve_coverage_only_preregistered",
            },
            {
                "mechanism": "chitosan",
                "threshold_label": "t80",
                "recommendation": "unverified_prospective_placeholder",
                "mean_truth_reached_fraction": None,
                "mean_interval_closed_fraction": None,
                "mean_coverage_fraction": None,
                "caveat": "no_reveal_yet_curve_coverage_only_preregistered",
            },
        ]
    )

    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(OUTDIR / "mechanism_target_scorecard.csv", index=False)

    lines = [
        "=== 105 -- release target-layer contract ===",
        "",
        "Mechanism-level summary:",
        "  PLGA: no timing target is fully stable; t50 is the least-bad shared anchor.",
        "  Liposome: t50 is the best pilot default under the current assay horizon.",
        "  Chitosan: no timing target is verified yet; only curve-space prospective coverage is preregistered.",
        "",
        "Interpretation:",
        "  1. A unified timing layer should be mechanism-aware and threshold-aware.",
        "  2. The stack can share target classes, but not force the same default deep-threshold target across all mechanisms.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
