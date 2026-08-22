"""
108 - Export a combined unified release stack manifest.

Purpose:
    Bind together the mechanism example family objects and the target-layer
    registry into a single manifest describing the current executable
    release-intelligence stack.

Consumes:
    outputs/84_release_posterior_family_examples/summary.csv
    outputs/106_release_target_registry/release_target_registry.json

Produces:
    outputs/108_release_stack_manifest/release_stack_manifest.json
    outputs/108_release_stack_manifest/stack_manifest_summary.csv
    outputs/108_release_stack_manifest/summary.txt
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_target_registry import ReleaseTargetRegistry


FAMILY_SUMMARY = Path("outputs/84_release_posterior_family_examples/summary.csv")
TARGET_REGISTRY = Path("outputs/106_release_target_registry/release_target_registry.json")
OUTDIR = Path("outputs/108_release_stack_manifest")


MECHANISM_MAP = {
    "plga": "plga_biphasic",
    "liposome": "liposome_weibull",
    "chitosan": "chitosan_ritger_peppas",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTDIR)
    fam = pd.read_csv(FAMILY_SUMMARY)
    registry = ReleaseTargetRegistry.from_json(TARGET_REGISTRY)

    profiles = []
    summary_rows = []
    for mech_key, family_mechanism_id in MECHANISM_MAP.items():
        fam_row = fam[fam["mechanism_id"] == family_mechanism_id].iloc[0].to_dict()
        profile = registry.profile(mech_key)
        timing_defaults = [spec.target_id for spec in profile.target_specs if spec.target_class == "timing_threshold" and spec.target_id in profile.default_targets]
        shape_defaults = [spec.target_id for spec in profile.target_specs if spec.target_class == "shape_descriptor" and spec.target_id in profile.default_targets]
        payload = {
            "mechanism_key": mech_key,
            "family_example": {
                "mechanism_id": fam_row["mechanism_id"],
                "decoder_handle": fam_row["decoder_handle"],
                "latent_dim": int(fam_row["latent_dim"]),
                "rank": int(fam_row["rank"]),
                "assay_keys": fam_row["assay_keys"],
                "observation_keys": fam_row["observation_keys"],
            },
            "target_layer": {
                "default_targets": profile.default_targets,
                "timing_defaults": timing_defaults,
                "shape_defaults": shape_defaults,
                "target_specs": [spec.as_dict() for spec in profile.target_specs],
                "notes": profile.notes,
                "metadata": profile.metadata,
            },
        }
        profiles.append(payload)
        summary_rows.append(
            {
                "mechanism_key": mech_key,
                "family_mechanism_id": fam_row["mechanism_id"],
                "decoder_handle": fam_row["decoder_handle"],
                "latent_dim": int(fam_row["latent_dim"]),
                "rank": int(fam_row["rank"]),
                "timing_defaults": "|".join(timing_defaults),
                "shape_defaults": "|".join(shape_defaults),
                "n_target_specs": len(profile.target_specs),
                "target_layer_status": profile.metadata.get("target_layer_status", ""),
            }
        )

    manifest = {
        "version": "2026-05-29",
        "stack_claim": "shared release-intelligence stack with mechanism-specific decoders",
        "profiles": profiles,
        "metadata": {
            "family_summary_source": FAMILY_SUMMARY.as_posix(),
            "target_registry_source": TARGET_REGISTRY.as_posix(),
        },
    }
    (OUTDIR / "release_stack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "stack_manifest_summary.csv", index=False)

    lines = [
        "=== 108 -- unified release stack manifest ===",
        "",
        "Mechanisms bound into one executable manifest:",
        "  plga -> plga_biphasic family + target-layer defaults",
        "  liposome -> liposome_weibull family + target-layer defaults",
        "  chitosan -> chitosan_ritger_peppas family + target-layer placeholder targets",
        "",
        "Interpretation:",
        "  1. The stack now binds latent-family objects and reporting-target defaults in one manifest.",
        "  2. This is the strongest current object-level evidence for the unified release-intelligence claim.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
