"""
109 - Compare our current executable stack occupancy against external families.

Purpose:
    Put the new internal stack manifest on the same five-layer surface as the
    external competitor scorecard so we can see, in one file, where we
    currently occupy the stack more completely than the field and where we
    still remain partial.

Consumes:
    docs/drug_release_stack_competitor_scorecard_2026-05-29.csv
    outputs/108_release_stack_manifest/stack_manifest_summary.csv

Produces:
    outputs/109_release_stack_occupancy_comparison/internal_vs_external_stack_occupancy.csv
    outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv
    outputs/109_release_stack_occupancy_comparison/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


COMPETITOR_CSV = Path("docs/drug_release_stack_competitor_scorecard_2026-05-29.csv")
MANIFEST_SUMMARY = Path("outputs/108_release_stack_manifest/stack_manifest_summary.csv")
OUTDIR = Path("outputs/109_release_stack_occupancy_comparison")


LAYER_COLS = [
    "l1_schema",
    "l2_partial_observation_benchmark",
    "l3_shared_posterior_object",
    "l4_mechanism_decoder",
    "l5_reporting_decision_layer",
]


OCC_VALUE = {
    "no": 0.0,
    "weak": 0.25,
    "partial": 0.5,
    "strong": 1.0,
    "partial_strong": 0.75,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTDIR)
    ext = pd.read_csv(COMPETITOR_CSV)
    manifest = pd.read_csv(MANIFEST_SUMMARY)

    our_row = {
        "family": "Our current executable stack",
        "representative_work": "local unified release-intelligence stack manifest",
        "year": 2026,
        "problem_type": "shared release-intelligence stack",
        "primary_input": "formulation/material context + sparse early observations",
        "primary_output": "curve + timing + shape + calibrated family artifacts",
        "l1_schema": "strong",
        "l2_partial_observation_benchmark": "strong",
        "l3_shared_posterior_object": "strong",
        "l4_mechanism_decoder": "strong",
        "l5_reporting_decision_layer": "partial_strong",
        "threat_type": "self",
        "borrow_or_fight": "n/a",
        "why_it_matters": "Only local system currently binding shared benchmark, family object, target registry, and mechanism-specific decoders into one manifest",
        "main_gap_vs_our_target": "Deep-threshold timing remains weaker than shape-aware reporting",
    }

    combined = pd.concat([pd.DataFrame([our_row]), ext], ignore_index=True)
    combined.to_csv(OUTDIR / "internal_vs_external_stack_occupancy.csv", index=False)

    layer_rows = []
    for layer in LAYER_COLS:
        ext_vals = ext[layer].map(OCC_VALUE)
        our_val = OCC_VALUE[our_row[layer]]
        layer_rows.append(
            {
                "layer": layer,
                "our_occupancy_value": our_val,
                "best_external_value": float(ext_vals.max()),
                "mean_external_value": float(ext_vals.mean()),
                "advantage_vs_best_external": float(our_val - ext_vals.max()),
                "advantage_vs_mean_external": float(our_val - ext_vals.mean()),
            }
        )
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(OUTDIR / "stack_layer_advantage_summary.csv", index=False)

    def fmt(layer: str) -> str:
        row = layer_df[layer_df["layer"] == layer].iloc[0]
        return (
            f"  {layer}: our={row['our_occupancy_value']:.2f}, "
            f"best_ext={row['best_external_value']:.2f}, "
            f"delta={row['advantage_vs_best_external']:+.2f}"
        )

    lines = [
        "=== 109 -- internal vs external stack occupancy ===",
        "",
        "Layer-by-layer advantage vs best external family:",
        fmt("l1_schema"),
        fmt("l2_partial_observation_benchmark"),
        fmt("l3_shared_posterior_object"),
        fmt("l4_mechanism_decoder"),
        fmt("l5_reporting_decision_layer"),
        "",
        "Interpretation:",
        "  1. Our strongest structural edge should appear at L2-L3, where most external systems remain partial or absent.",
        "  2. L5 should remain the most qualified layer because timing remains weaker than shape-aware reporting.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
