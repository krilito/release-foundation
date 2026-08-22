"""
107 - Build a mechanism-aware shape-target contract.

Purpose:
    Convert the selected-cell shape bridge evidence into a concrete
    mechanism-aware contract for shared shape targets.

Consumes:
    outputs/101_selected_cell_bridge_summary_v2/descriptor_cross_cell.csv

Produces:
    outputs/107_release_shape_target_contract/mechanism_shape_aggregate.csv
    outputs/107_release_shape_target_contract/mechanism_shape_scorecard.csv
    outputs/107_release_shape_target_contract/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


INDIR = Path("outputs/101_selected_cell_bridge_summary_v2")
OUTDIR = Path("outputs/107_release_shape_target_contract")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_descriptor(mean_cov: float, min_cov: float) -> str:
    if mean_cov >= 0.80 and min_cov >= 0.65:
        return "candidate_default"
    if mean_cov >= 0.70 and min_cov >= 0.65:
        return "secondary_only"
    return "not_default"


def mechanism_from_dataset(dataset: str) -> str:
    if dataset in {"cross321", "internal181"}:
        return "plga"
    return dataset


def main() -> None:
    ensure_dir(OUTDIR)
    df = pd.read_csv(INDIR / "descriptor_cross_cell.csv")
    df = df[df["method"] == "global"].copy()
    df["mechanism"] = df["dataset"].map(mechanism_from_dataset)

    agg = (
        df.groupby(["mechanism", "descriptor"], as_index=False)
        .agg(
            mean_coverage_fraction=("coverage", "mean"),
            min_coverage_fraction=("coverage", "min"),
        )
    )
    agg["recommendation"] = agg.apply(
        lambda r: classify_descriptor(float(r["mean_coverage_fraction"]), float(r["min_coverage_fraction"])),
        axis=1,
    )
    agg.to_csv(OUTDIR / "mechanism_shape_aggregate.csv", index=False)

    rows = []
    for _, row in agg.iterrows():
        caveat = ""
        mechanism = str(row["mechanism"])
        descriptor = str(row["descriptor"])
        rec = str(row["recommendation"])
        if mechanism == "plga" and descriptor == "burst":
            caveat = "usable_but_less_stable_than_other_shape_targets"
        elif mechanism == "plga" and descriptor == "residual_tail":
            caveat = "strongest_plga_shape_default_under_global_family_bridge"
        elif mechanism == "liposome":
            caveat = "pilot-strong_under_current_selected_cell_bridge"
        rows.append(
            {
                "mechanism": mechanism,
                "descriptor": descriptor,
                "recommendation": rec,
                "mean_coverage_fraction": row["mean_coverage_fraction"],
                "min_coverage_fraction": row["min_coverage_fraction"],
                "caveat": caveat,
            }
        )

    rows.extend(
        [
            {
                "mechanism": "chitosan",
                "descriptor": "burst",
                "recommendation": "unverified_prospective_placeholder",
                "mean_coverage_fraction": None,
                "min_coverage_fraction": None,
                "caveat": "no_reveal_yet_no_shape_bridge_audit",
            },
            {
                "mechanism": "chitosan",
                "descriptor": "post_window",
                "recommendation": "unverified_prospective_placeholder",
                "mean_coverage_fraction": None,
                "min_coverage_fraction": None,
                "caveat": "no_reveal_yet_no_shape_bridge_audit",
            },
            {
                "mechanism": "chitosan",
                "descriptor": "residual_tail",
                "recommendation": "unverified_prospective_placeholder",
                "mean_coverage_fraction": None,
                "min_coverage_fraction": None,
                "caveat": "no_reveal_yet_no_shape_bridge_audit",
            },
            {
                "mechanism": "chitosan",
                "descriptor": "tail_auc",
                "recommendation": "unverified_prospective_placeholder",
                "mean_coverage_fraction": None,
                "min_coverage_fraction": None,
                "caveat": "no_reveal_yet_no_shape_bridge_audit",
            },
        ]
    )

    pd.DataFrame(rows).to_csv(OUTDIR / "mechanism_shape_scorecard.csv", index=False)

    lines = [
        "=== 107 -- release shape-target contract ===",
        "",
        "Mechanism-level summary:",
        "  PLGA: residual_tail is the strongest shape default; post_window and tail_auc are also plausible defaults; burst is weaker.",
        "  Liposome: all four audited shape targets behave like strong pilot defaults in the current bridge panel.",
        "  Chitosan: no shape target is verified before reveal.",
        "",
        "Interpretation:",
        "  1. Shape targets are currently more mature than timing targets in the shared release-intelligence stack.",
        "  2. This strengthens the case for a unified reporting layer even before deep-threshold timing is solved.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
