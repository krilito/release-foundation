# Drug Release Figure Execution Queue

Date: `2026-05-29`

Purpose:

```text
Convert the figure map and data manifest into an execution queue:
which figure to build first, why it matters, what dependencies it has, and
what exact check would tell us the panel is ready.
```

This is an execution note, not a plotting script.

## 1. Recommended order

Build order should be:

1. Figure 2
2. Figure 4
3. Figure 3
4. Figure 5
5. Figure 1

This order is deliberate.

- Figure 2 gives the cleanest audited empirical core.
- Figure 4 proves the story is not PLGA-only.
- Figure 3 prevents overclaim.
- Figure 5 upgrades the evidence culture.
- Figure 1 should be drawn last because it depends on the final narrative.

## 2. Figure-by-figure queue

## Figure 2 — PLGA audited forecasting core

Status:

- `completed first rendered draft`
- assets: [outputs/90_release_paper_figure_assets/figure2](D:/release-foundation/outputs/90_release_paper_figure_assets/figure2)
- renders: [outputs/91_release_paper_figures/figure2](D:/release-foundation/outputs/91_release_paper_figures/figure2)

Why first:

- strongest current empirical center
- easiest to defend
- likely the first figure reviewers will inspect for substance

Dependencies:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv](D:/release-foundation/outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv)

Ready check:

- panel A shows early-only versus formulation-only gap clearly
- panel B shows theta-minus-direct gain without ambiguous legends
- panel C shows middle-layer gain without implying "always wins"

Main risk:

- overcomplicating the panel and losing the central point

## Figure 4 — First non-PLGA bridge

Status:

- `completed first rendered draft`
- assets: [outputs/90_release_paper_figure_assets/figure4](D:/release-foundation/outputs/90_release_paper_figure_assets/figure4)
- renders: [outputs/91_release_paper_figures/figure4](D:/release-foundation/outputs/91_release_paper_figures/figure4)

Why second:

- this is what lets the paper escape "just PLGA"

Dependencies:

- [outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv)
- [outputs/80_liposome_group_by_api_direct_et_q/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_direct_et_q/metrics_summary.csv)
- [outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv)
- [outputs/80_liposome_group_by_method_direct_et_q/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_direct_et_q/metrics_summary.csv)
- [outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv)

Ready check:

- split families are visually separated
- the plot makes clear that liposome is a bridge, not complete validation
- t50 caveat remains visible

Main risk:

- accidentally drawing the figure as if it proves universal route superiority

## Figure 3 — Timing and shape reality check

Status:

- `completed first rendered draft`
- assets: [outputs/90_release_paper_figure_assets/figure3](D:/release-foundation/outputs/90_release_paper_figure_assets/figure3)
- renders: [outputs/91_release_paper_figures/figure3](D:/release-foundation/outputs/91_release_paper_figures/figure3)

Why third:

- this is the honesty figure that stabilizes the whole paper

Dependencies:

- [outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv)
- [outputs/82_release_capability_snapshot/plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)
- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)

Ready check:

- one panel shows tally-style route ranking
- one panel shows shape-object split clearly
- one callout example makes the "curve fit is not duration" point obvious

Main risk:

- turning the figure into a confusing wall of metrics

## Figure 5 — Shared uncertainty and prospective discipline

Status:

- `completed first rendered draft`
- assets: [outputs/90_release_paper_figure_assets/figure5](D:/release-foundation/outputs/90_release_paper_figure_assets/figure5)
- renders: [outputs/91_release_paper_figures/figure5](D:/release-foundation/outputs/91_release_paper_figures/figure5)

Why fourth:

- this figure upgrades the paper from a point-prediction benchmark to a
  stronger evidence culture

Dependencies:

- [outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv](D:/release-foundation/outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv)
- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

Ready check:

- panel A shows the nine-cell calibrated-family panel cleanly
- panel B picks only one or two representative conformal shifts
- panel C is clearly labeled as locked prospective discipline, not revealed
  success

Main risk:

- overloading the figure with too many UQ details

## Figure 1 — Problem framing and method object

Status:

- `completed first rendered draft`
- renders: [outputs/91_release_paper_figures/figure1](D:/release-foundation/outputs/91_release_paper_figures/figure1)
- support note: [unified_release_intelligence_contract_2026-05-29.md](D:/release-foundation/docs/unified_release_intelligence_contract_2026-05-29.md)

Why last:

- this figure should be drawn after the empirical story is stable

Dependencies:

- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md)
- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [drug_release_paper_positioning_playbook_2026-05-29.md](D:/release-foundation/docs/drug_release_paper_positioning_playbook_2026-05-29.md)

Ready check:

- field-default schematic and our schematic differ in one intuitive glance
- the "shared object + mechanism-specific decoder" layer is visible

Main risk:

- making the concept figure more grandiose than the empirical paper supports

## 3. What should wait for supplement

Keep out of the main-figure queue unless the paper grows:

1. full raw UQ benchmark tables
2. posterior-family example JSON exports
3. the full UQ-to-timescale gap table
4. the stack-occupancy strategy panel

These belong in:

- [outputs/62_fib_casp_benchmark](D:/release-foundation/outputs/62_fib_casp_benchmark)
- [outputs/66_ensemble_casp](D:/release-foundation/outputs/66_ensemble_casp)
- [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)
- [outputs/89_release_uq_timescale_gap_snapshot](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot)
- [outputs/110_release_stack_occupancy_paper_asset](D:/release-foundation/outputs/110_release_stack_occupancy_paper_asset)

## 4. Minimal execution package if plotting starts now

The smallest strong package is:

1. refine Figure 2 typography only if journal layout demands it
2. refine Figure 4 caveat wording only if bridge framing needs softening
3. refine Figure 3 only if a second representative divergence cell is desired
4. refine Figure 5 only if UQ detail needs compression for target journal

At that point, the paper would already have a complete empirical backbone.

## 5. Optional strategy / supplement panel

Status:

- `ready for plotting`
- assets: [outputs/110_release_stack_occupancy_paper_asset](D:/release-foundation/outputs/110_release_stack_occupancy_paper_asset)

Use case:

- supplementary outside-in support
- slide deck positioning panel
- rebuttal figure if reviewers ask what is genuinely new at the systems level

Main claim:

- the project's strongest moat is L2-L3 stack occupancy rather than a claim
  of universal metric leadership

## Bottom line

The figure program is no longer blocked by "not knowing what to show." It is
now a sequencing problem: build the audited empirical core first, then the
bridge, then the honesty figure, then the evidence-culture upgrade.
