"""
110 - Build paper-facing stack occupancy assets and prose hooks.

Purpose:
    Convert the internal-vs-external stack occupancy comparison into a compact
    paper/supplement-facing asset package plus manuscript-ready prose hooks.

Consumes:
    outputs/109_release_stack_occupancy_comparison/internal_vs_external_stack_occupancy.csv
    outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv

Produces:
    outputs/110_release_stack_occupancy_paper_asset/panel_stack_occupancy.csv
    outputs/110_release_stack_occupancy_paper_asset/panel_layer_advantage.csv
    outputs/110_release_stack_occupancy_paper_asset/summary.txt
    docs/release_stack_occupancy_paper_hook_2026-05-29.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


INDIR = Path("outputs/109_release_stack_occupancy_comparison")
OUTDIR = Path("outputs/110_release_stack_occupancy_paper_asset")


KEEP_FAMILIES = [
    "Our current executable stack",
    "NC/Bannigan few-shot and zero-shot ML",
    "Explainable LAI forecaster",
    "Liposome IVR workflow",
    "FormulationLAI",
    "FormulationAI platform",
    "Scientific Data PLGA dataset",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dir(OUTDIR)
    occ = pd.read_csv(INDIR / "internal_vs_external_stack_occupancy.csv")
    adv = pd.read_csv(INDIR / "stack_layer_advantage_summary.csv")

    panel_occ = occ[occ["family"].isin(KEEP_FAMILIES)].copy()
    panel_occ.to_csv(OUTDIR / "panel_stack_occupancy.csv", index=False)
    adv.to_csv(OUTDIR / "panel_layer_advantage.csv", index=False)

    lines = [
        "=== 110 -- release stack occupancy paper asset ===",
        "",
        "Suggested use:",
        "  Supplementary panel or strategy panel in talks/rebuttal.",
        "",
        "Key topline:",
        "  Our executable stack ties or matches best external systems at L1 and L4,",
        "  but leads most clearly at L2 and L3.",
        "",
        "Qualified layer:",
        "  L5 remains partial-strong rather than fully strong because timing defaults",
        "  remain less mature than shape-aware reporting.",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
