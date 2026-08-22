"""
106 - Export the mechanism-aware release target registry.

Purpose:
    Convert the timing and shape target-layer contracts into a concrete JSON
    artifact that downstream benchmark, reporting, and deployment code can
    read.

Consumes:
    outputs/105_release_target_layer_contract/mechanism_target_scorecard.csv
    outputs/107_release_shape_target_contract/mechanism_shape_scorecard.csv

Produces:
    outputs/106_release_target_registry/release_target_registry.json
    outputs/106_release_target_registry/registry_summary.csv
    outputs/106_release_target_registry/summary.txt
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_target_registry import (
    ReleaseMechanismTargetProfile,
    ReleaseTargetRegistry,
    ReleaseTargetSpec,
)


INDIR = Path("outputs/105_release_target_layer_contract")
SHAPE_INDIR = Path("outputs/107_release_shape_target_contract")
OUTDIR = Path("outputs/106_release_target_registry")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_defaults(mechanism: str, score_df: pd.DataFrame) -> list[str]:
    # score_df here is timing-only and retained for backward-compatible
    # anchor selection. Shape defaults are added separately later.
    if mechanism == "plga":
        return ["t50"]
    if mechanism == "liposome":
        return ["t50"]
    return []


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)

def _timing_specs(mechanism: str, score_df: pd.DataFrame) -> list[ReleaseTargetSpec]:
    sub = score_df[score_df["mechanism"] == mechanism].copy()
    target_specs = []
    for _, row in sub.iterrows():
        evidence = {}
        if pd.notna(row.get("mean_truth_reached_fraction")):
            evidence["mean_truth_reached_fraction"] = float(row["mean_truth_reached_fraction"])
        if pd.notna(row.get("mean_interval_closed_fraction")):
            evidence["mean_interval_closed_fraction"] = float(row["mean_interval_closed_fraction"])
        if pd.notna(row.get("mean_coverage_fraction")):
            evidence["mean_coverage_fraction"] = float(row["mean_coverage_fraction"])
        target_specs.append(
            ReleaseTargetSpec(
                target_id=str(row["threshold_label"]),
                target_class="timing_threshold",
                unit="mechanism_local_time",
                recommendation=_clean_text(row["recommendation"]),
                caveat=_clean_text(row["caveat"]),
                evidence=evidence,
                metadata={"source": "selected_cell_bridge_threshold_audit"},
            )
        )
    return target_specs


def _shape_unit(descriptor: str) -> str:
    return "fraction" if descriptor in {"burst", "post_window", "residual_tail"} else "time_release_area"


def _shape_specs(mechanism: str, shape_df: pd.DataFrame) -> list[ReleaseTargetSpec]:
    sub = shape_df[shape_df["mechanism"] == mechanism].copy()
    target_specs = []
    for _, row in sub.iterrows():
        evidence = {}
        if pd.notna(row.get("mean_coverage_fraction")):
            evidence["mean_coverage_fraction"] = float(row["mean_coverage_fraction"])
        if pd.notna(row.get("min_coverage_fraction")):
            evidence["min_coverage_fraction"] = float(row["min_coverage_fraction"])
        descriptor = str(row["descriptor"])
        target_specs.append(
            ReleaseTargetSpec(
                target_id=descriptor,
                target_class="shape_descriptor",
                unit=_shape_unit(descriptor),
                recommendation=_clean_text(row["recommendation"]),
                caveat=_clean_text(row["caveat"]),
                evidence=evidence,
                metadata={"source": "selected_cell_bridge_shape_audit"},
            )
        )
    return target_specs


def _default_shape_targets(mechanism: str, shape_df: pd.DataFrame) -> list[str]:
    sub = shape_df[(shape_df["mechanism"] == mechanism) & (shape_df["recommendation"] == "candidate_default")]
    return [str(x) for x in sub["descriptor"].tolist()]


def build_profile(mechanism: str, timing_df: pd.DataFrame, shape_df: pd.DataFrame) -> ReleaseMechanismTargetProfile:
    target_specs = _timing_specs(mechanism, timing_df) + _shape_specs(mechanism, shape_df)
    default_targets = pick_defaults(mechanism, timing_df) + _default_shape_targets(mechanism, shape_df)

    notes = []
    if mechanism == "plga":
        notes.append("t50 is the least-bad PLGA anchor, not a solved default.")
        notes.append("residual_tail is the strongest PLGA shape default under the current bridge panel.")
    elif mechanism == "liposome":
        notes.append("t50 is a pilot anchor under the current assay horizon.")
        notes.append("all four current liposome shape targets behave like pilot-strong defaults.")
    elif mechanism == "chitosan":
        notes.append("No timing or shape target is verified before reveal; registry entry is placeholder-only.")

    return ReleaseMechanismTargetProfile(
        mechanism_id=mechanism,
        default_targets=default_targets,
        target_specs=target_specs,
        notes=notes,
        metadata={"target_layer_status": "evidence_backed" if mechanism != "chitosan" else "prospective_placeholder"},
    )


def main() -> None:
    ensure_dir(OUTDIR)
    timing_df = pd.read_csv(INDIR / "mechanism_target_scorecard.csv")
    shape_df = pd.read_csv(SHAPE_INDIR / "mechanism_shape_scorecard.csv")
    mechanisms = ["plga", "liposome", "chitosan"]
    profiles = [build_profile(mech, timing_df, shape_df) for mech in mechanisms]

    registry = ReleaseTargetRegistry(
        version="2026-05-29",
        profiles=profiles,
        shared_target_classes=["timing_threshold", "shape_descriptor", "curve_coverage"],
        metadata={
            "purpose": "mechanism-aware target defaults for unified release-intelligence reporting",
            "source_contract_timing": "outputs/105_release_target_layer_contract/mechanism_target_scorecard.csv",
            "source_contract_shape": "outputs/107_release_shape_target_contract/mechanism_shape_scorecard.csv",
        },
    )
    registry.to_json(OUTDIR / "release_target_registry.json")

    summary_rows = []
    for profile in registry.profiles:
        for spec in profile.target_specs:
            summary_rows.append(
                {
                    "mechanism_id": profile.mechanism_id,
                    "target_id": spec.target_id,
                    "recommendation": spec.recommendation,
                    "is_default": spec.target_id in profile.default_targets,
                    "caveat": spec.caveat,
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "registry_summary.csv", index=False)

    lines = [
        "=== 106 -- release target registry export ===",
        "",
        "Registry version: 2026-05-29",
        "Mechanisms: plga, liposome, chitosan",
        "Shared target classes: timing_threshold, shape_descriptor, curve_coverage",
        "",
        "Default anchors / descriptors:",
        "  plga: t50 + post_window + residual_tail + tail_auc",
        "  liposome: t50 + burst + post_window + residual_tail + tail_auc",
        "  chitosan: none before reveal",
        "",
        "Interpretation:",
        "  1. The target layer is now executable as a registry artifact.",
        "  2. Shared classes are unified, but defaults remain mechanism-aware.",
        "  3. The current registry is strongest on shape targets, not deep-threshold timing.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
