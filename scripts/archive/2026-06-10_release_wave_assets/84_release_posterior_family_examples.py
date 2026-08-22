"""
84 - Emit example posterior-family objects for PLGA, liposome, and chitosan.

Purpose:
    Turn the shared release posterior-family schema into concrete mechanism
    examples tied to current project assets. This is not a training script.
    It creates executable example objects that downstream code or docs can
    reference when discussing unification at the object level.

Consumes:
    outputs/67_chitosan_prospective/lock_metadata.json

Produces:
    outputs/84_release_posterior_family_examples/
        plga_biphasic_family.json
        liposome_weibull_family.json
        chitosan_ritger_peppas_family.json
        summary.csv
        summary.txt
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_posterior_family import ReleasePosteriorFamily


OUTDIR = Path("outputs/84_release_posterior_family_examples")
CHITOSAN_LOCK = Path("outputs/67_chitosan_prospective/lock_metadata.json")


def build_examples() -> list[tuple[str, ReleasePosteriorFamily]]:
    chitosan_lock = json.loads(CHITOSAN_LOCK.read_text(encoding="utf-8"))

    plga = ReleasePosteriorFamily.low_rank_family(
        mechanism_id="plga_biphasic",
        decoder_handle="PLGABiphasic.simulate_numpy",
        z_center=[-1.2, -2.0, -0.3, -2.4, -0.7, 0.18, 0.09, -1.2, 0.93],
        z_basis=[
            [0.25, 0.11, 0.08, 0.19, 0.05, -0.03, 0.02, 0.09, 0.01],
            [0.06, -0.18, 0.12, -0.04, 0.16, 0.05, -0.02, 0.07, 0.00],
        ],
        z_scale=[0.35, 0.18],
        assay_context={
            "time_unit": "days",
            "canonical_early_window_days": 7.0,
            "release_object": "cumulative_fraction",
        },
        observation_context={
            "observation_mode": "sparse early release + formulation",
            "forecast_targets": ["curve", "threshold timing", "post-window release"],
        },
        metadata={
            "mechanism_family": "plga_microparticle",
            "latent_coordinates": "theta-like kinetic state",
            "example_source": "audited PLGA benchmark schema example",
        },
    )

    liposome = ReleasePosteriorFamily.low_rank_family(
        mechanism_id="liposome_weibull",
        decoder_handle="WeibullSimulator.simulate_numpy",
        z_center=[1.95, -0.85],
        z_basis=[
            [0.22, 0.04],
            [-0.05, 0.18],
        ],
        z_scale=[0.28, 0.10],
        assay_context={
            "time_unit": "hours",
            "canonical_early_window_hours": 24.0,
            "release_object": "cumulative_fraction",
        },
        observation_context={
            "observation_mode": "sparse early release + formulation",
            "forecast_targets": ["curve", "threshold timing", "tail shape"],
        },
        metadata={
            "mechanism_family": "liposome_weibull_adapter",
            "latent_coordinates": "log_alpha_log_beta",
            "example_source": "accelerated IVR liposome schema example",
        },
    )

    chitosan = ReleasePosteriorFamily.low_rank_family(
        mechanism_id="chitosan_ritger_peppas",
        decoder_handle="ChitosanRitgerPeppas.simulate_release",
        z_center=[0.52, 0.68],
        z_basis=[
            [0.15, -0.02],
            [0.03, 0.09],
        ],
        z_scale=[0.20, 0.07],
        assay_context={
            "time_unit": "hours",
            "locked_time_grid_points": len(chitosan_lock["t_grid_hours"]),
            "release_object": "cumulative_fraction",
        },
        observation_context={
            "observation_mode": "locked prospective sparse release",
            "forecast_targets": ["curve", "coverage", "timescale stress test"],
        },
        metadata={
            "mechanism_family": "chitosan_hand_specified_family",
            "latent_coordinates": "release exponent / prefactor-like state",
            "example_source": "locked chitosan scaffold schema example",
        },
    )

    return [
        ("plga_biphasic_family.json", plga),
        ("liposome_weibull_family.json", liposome),
        ("chitosan_ritger_peppas_family.json", chitosan),
    ]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    lines = ["=== 84 -- release posterior-family examples ===", ""]
    for filename, family in build_examples():
        path = OUTDIR / filename
        family.to_json(path)
        summary = family.summary_dict()
        summary["filename"] = filename
        rows.append(summary)
        lines.extend(
            [
                f"{filename}:",
                f"  mechanism_id  : {summary['mechanism_id']}",
                f"  decoder_handle: {summary['decoder_handle']}",
                f"  latent_dim    : {summary['latent_dim']}",
                f"  rank          : {summary['rank']}",
                f"  assay_keys    : {', '.join(summary['assay_keys'])}",
                f"  obs_keys      : {', '.join(summary['observation_keys'])}",
                "",
            ]
        )
    pd.DataFrame(rows).to_csv(OUTDIR / "summary.csv", index=False)
    (OUTDIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
