# Drug Release 70+ Workstream Map

Date: `2026-05-30`

Purpose:

```text
Reconstruct the 70+ drug-release workstream from evidence, summarize what
each phase actually did, and provide one canonical navigation layer so the
repo can be used without reverse-engineering dozens of scripts and verdict
notes.
```

This file is the best "start here" entry if someone asks:

- what happened from `70` onward,
- which files are canonical now,
- which outputs actually matter,
- and which scripts are still historical/provisional rather than headline.

## Short answer

The `70+` stream ended up having four real phases:

1. `70-77`: world-model / diagnostics / prereg / gate-building phase
2. `78-89`: shared release benchmark + cross-mechanism object/UQ layer
3. `90-96, 109-116`: paper/positioning/deck/landscape layer
4. `97-119`: route-consistent duration/shape uncertainty push

The current best single summary of the whole line of work is:

- [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)

The current best Chinese strategy package is:

- [drug_release_strategy_deck_zh_2026-05-29.pptx](D:/release-foundation/outputs/113_drug_release_strategy_deck/drug_release_strategy_deck_zh_2026-05-29.pptx)
- [drug_release_executive_brief_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_executive_brief_zh_2026-05-29.md)
- [drug_release_strategy_answer_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_strategy_answer_zh_2026-05-29.md)

The current best technical boundary package is:

- [full_cell_bridge_panel.png](D:/release-foundation/outputs/92_release_strategy_figures/full_cell_bridge/full_cell_bridge_panel.png)
- [coverage_audit.csv](D:/release-foundation/outputs/119_full_cell_bridge_coverage_audit/coverage_audit.csv)
- [full_cell_bridge_panel_verdict_2026-05-29.md](D:/release-foundation/docs/full_cell_bridge_panel_verdict_2026-05-29.md)

## 1. Phase map

### Phase A — `70-77`

This was the transition from "world model / active observer / Nature-route"
thinking into stricter diagnostics and prospective discipline.

Canonical files to remember:

- [71_release_world_model.py](D:/release-foundation/scripts/71_release_world_model.py)
- [72_canonical_benchmark.py](D:/release-foundation/scripts/72_canonical_benchmark.py)
- [72_canonical_benchmark_v2.py](D:/release-foundation/scripts/72_canonical_benchmark_v2.py)
- [73_manuscript_readiness_matrix.py](D:/release-foundation/scripts/73_manuscript_readiness_matrix.py)
- [75_chitosan_prospective_eval.py](D:/release-foundation/scripts/75_chitosan_prospective_eval.py)
- [76_casp_conformal_recalibration.py](D:/release-foundation/scripts/76_casp_conformal_recalibration.py)
- [77_regime_benefit_loocv.py](D:/release-foundation/scripts/77_regime_benefit_loocv.py)

What Phase A really contributed:

- it moved the project from loose world-model ambition toward audited
  forecasting claims
- it locked in the idea that prospective discipline matters
- it created the calibration and regime-analysis assets later reused by the
  unified release stack

What is **not** the right reading now:

- do not treat `70-77` as the current repo headline
- do not treat the world-model files as the current primary claim package

### Phase B — `78-89`

This is where the project stopped being "just a PLGA story" and became a
shared release-intelligence stack.

Canonical files:

- [78_liposome_ivr_intake.py](D:/release-foundation/scripts/78_liposome_ivr_intake.py)
- [79_liposome_prediction_export.py](D:/release-foundation/scripts/79_liposome_prediction_export.py)
- [80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)
- [81_release_timescale_eval.py](D:/release-foundation/scripts/81_release_timescale_eval.py)
- [82_release_capability_snapshot.py](D:/release-foundation/scripts/82_release_capability_snapshot.py)
- [83_plga_prediction_export.py](D:/release-foundation/scripts/83_plga_prediction_export.py)
- [84_release_posterior_family_examples.py](D:/release-foundation/scripts/84_release_posterior_family_examples.py)
- [85_release_posterior_family_export.py](D:/release-foundation/scripts/85_release_posterior_family_export.py)
- [86_release_casp_family_export.py](D:/release-foundation/scripts/86_release_casp_family_export.py)
- [87_release_conformal_family_export.py](D:/release-foundation/scripts/87_release_conformal_family_export.py)
- [88_release_uncertainty_snapshot.py](D:/release-foundation/scripts/88_release_uncertainty_snapshot.py)
- [89_release_uq_timescale_gap_snapshot.py](D:/release-foundation/scripts/89_release_uq_timescale_gap_snapshot.py)
- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)

What Phase B established:

- a shared benchmark contract
- a first non-PLGA bridge through liposome
- a shared posterior-family object
- a calibrated-family panel
- explicit visibility into the curve/UQ/timescale gap

This is the real backbone of the current unified-release story.

### Phase C — `90-96`, `109-116`

This phase translated the technical stack into paper and positioning assets.

Canonical files:

- [90_release_paper_figure_assets.py](D:/release-foundation/scripts/90_release_paper_figure_assets.py)
- [91_plot_release_paper_figure2.py](D:/release-foundation/scripts/91_plot_release_paper_figure2.py)
- [92_plot_release_paper_figure4.py](D:/release-foundation/scripts/92_plot_release_paper_figure4.py)
- [93_plot_release_paper_figure3.py](D:/release-foundation/scripts/93_plot_release_paper_figure3.py)
- [94_plot_release_paper_figure5.py](D:/release-foundation/scripts/94_plot_release_paper_figure5.py)
- [95_plot_release_paper_figure1.py](D:/release-foundation/scripts/95_plot_release_paper_figure1.py)
- [96_plot_release_competitor_landscape.py](D:/release-foundation/scripts/96_plot_release_competitor_landscape.py)
- [109_release_stack_occupancy_comparison.py](D:/release-foundation/scripts/109_release_stack_occupancy_comparison.py)
- [110_release_stack_occupancy_paper_asset.py](D:/release-foundation/scripts/110_release_stack_occupancy_paper_asset.py)
- [111_plot_release_stack_occupancy_panel.py](D:/release-foundation/scripts/111_plot_release_stack_occupancy_panel.py)
- [112_plot_release_external_methods_venues_panel.py](D:/release-foundation/scripts/112_plot_release_external_methods_venues_panel.py)
- [113_build_drug_release_strategy_deck.py](D:/release-foundation/scripts/113_build_drug_release_strategy_deck.py)
- [115_plot_release_unification_boundary_panel.py](D:/release-foundation/scripts/115_plot_release_unification_boundary_panel.py)
- [116_plot_release_competitor_response_panel.py](D:/release-foundation/scripts/116_plot_release_competitor_response_panel.py)

What Phase C established:

- external field map
- competitor / borrow / defend framing
- paper figures 1-5
- stack-occupancy framing
- Chinese strategic deck and talk assets

This is the current "how to explain it" layer.

### Phase D — `97-119`

This is the most important technical extension after the main stack existed.
It asked:

> can the shared calibrated-family object emit route-consistent duration/shape uncertainty, and how far does that survive beyond curve-space?

Canonical files:

- [97_release_stack_readiness_audit.py](D:/release-foundation/scripts/97_release_stack_readiness_audit.py)
- [98_release_family_duration_shape_bridge.py](D:/release-foundation/scripts/98_release_family_duration_shape_bridge.py)
- [99_release_family_duration_shape_cell_bridge.py](D:/release-foundation/scripts/99_release_family_duration_shape_cell_bridge.py)
- [100_selected_cell_bridge_summary.py](D:/release-foundation/scripts/100_selected_cell_bridge_summary.py)
- [101_selected_cell_bridge_summary_v2.py](D:/release-foundation/scripts/101_selected_cell_bridge_summary_v2.py)
- [102_release_timing_failure_audit.py](D:/release-foundation/scripts/102_release_timing_failure_audit.py)
- [104_release_threshold_target_summary.py](D:/release-foundation/scripts/104_release_threshold_target_summary.py)
- [105_release_target_layer_contract.py](D:/release-foundation/scripts/105_release_target_layer_contract.py)
- [106_release_target_registry_export.py](D:/release-foundation/scripts/106_release_target_registry_export.py)
- [107_release_shape_target_contract.py](D:/release-foundation/scripts/107_release_shape_target_contract.py)
- [108_release_stack_manifest.py](D:/release-foundation/scripts/108_release_stack_manifest.py)
- [114_plot_selected_cell_bridge_panel.py](D:/release-foundation/scripts/114_plot_selected_cell_bridge_panel.py)
- [117_full_cell_bridge_summary.py](D:/release-foundation/scripts/117_full_cell_bridge_summary.py)
- [118_plot_full_cell_bridge_panel.py](D:/release-foundation/scripts/118_plot_full_cell_bridge_panel.py)
- [119_full_cell_bridge_coverage_audit.py](D:/release-foundation/scripts/119_full_cell_bridge_coverage_audit.py)
- [release_target_registry.py](D:/release-foundation/release_target_registry.py)

What Phase D established:

- selected-curve bridge exists
- selected-cell bridge exists
- timing failure taxonomy exists
- target-layer contract and registry exist
- stack manifest exists
- current full `9-cell` route-consistent cell-panel exists
- `liposome + internal181` are now full matched-curve in that panel
- only `cross321` remains sampled within-cell

This is the current hardest-evidence layer.

## 2. What is canonical now

If the repo feels too crowded, the shortest canonical read path is:

1. Chinese strategic answer:
   [drug_release_executive_brief_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_executive_brief_zh_2026-05-29.md)
2. Master technical / strategic map:
   [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)
3. Current deck:
   [drug_release_strategy_deck_zh_2026-05-29.pptx](D:/release-foundation/outputs/113_drug_release_strategy_deck/drug_release_strategy_deck_zh_2026-05-29.pptx)
4. Current hardest technical boundary:
   [full_cell_bridge_panel.png](D:/release-foundation/outputs/92_release_strategy_figures/full_cell_bridge/full_cell_bridge_panel.png)
5. Current route-consistent coverage boundary:
   [coverage_audit.csv](D:/release-foundation/outputs/119_full_cell_bridge_coverage_audit/coverage_audit.csv)

## 3. What to treat as support rather than headline

Support-only / process-heavy files include:

- duplicated `v2` / `b` scripts that helped patch or validate earlier results
- earlier selected-cell bridge summaries now superseded by the current
  `full_cell_bridge_panel`
- older threshold-only framing that is now weaker than the current
  route-consistent shape-aware framing

That means:

- use [selected_cell_bridge_panel.png](D:/release-foundation/outputs/92_release_strategy_figures/selected_cell_bridge/selected_cell_bridge_panel.png)
  as support
- prefer [full_cell_bridge_panel.png](D:/release-foundation/outputs/92_release_strategy_figures/full_cell_bridge/full_cell_bridge_panel.png)
  as the current headline bridge panel

## 4. Current strongest conclusions

From the whole `70+` stream, the most robust current conclusions are:

1. The field is still mostly doing direct prediction, sparse-prefix forecasting,
   workflow-assay design, or optimization/platform systems.
2. The right bold framing is:
   `shared release-intelligence stack`
3. The wrong bold framing is:
   `universal solved drug-release model`
4. Our strongest moat is still at:
   `L2 partial-observation benchmark` and `L3 shared posterior-family object`
5. The strongest current unified target layer is shape-aware, not deep-threshold timing-aware.
6. The hardest remaining technical gap is still:
   `benchmark-side route-consistent duration/UQ`, especially on the remaining `cross321` cells.

## 5. Why the repo feels crowded

The repo is crowded because the workstream did three things in sequence:

- built real technical infrastructure
- built a full external positioning layer on top of it
- then pushed the hardest technical boundary again

So the crowding is not random. It is the residue of:

1. technical stack construction
2. claim-boundary auditing
3. paper/deck translation

The safest cleanup move right now is **not** to relocate files.
The safest move is to maintain canonical navigation and avoid breaking links.

That is what this file is for.

## 6. Recommended next cleanup move

If we keep organizing later, the next useful cleanup step is:

- keep all current files in place
- continue treating this map plus the master dossier as the navigation layer
- only archive or relocate older support scripts after the current claim
  boundary is fully stable

Right now, the repo is better served by:

`clear canonical entrypoints`

than by:

`aggressive physical file movement`
