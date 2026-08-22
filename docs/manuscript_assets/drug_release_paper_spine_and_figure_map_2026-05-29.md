# Drug Release Paper Spine and Figure Map

Date: `2026-05-29`

Purpose:

```text
Turn the current audited release evidence into a paper-ready skeleton:
1. section-by-section paper spine,
2. main figures and the exact claim each figure carries,
3. supplementary-only items,
4. which outputs are the authoritative sources for each panel,
5. which claims should never be attached to a given figure.
```

This is the bridge from positioning to manuscript assembly.

## 1. Executive verdict

If the paper were assembled now, the cleanest scientific arc would be:

1. the field mostly does direct release regression
2. sparse early release is the key forecasting signal
3. a mechanism-routed middle object changes the problem in a useful way
4. this is not only a PLGA-local story because the interface survives liposome
5. the framework already has a shared uncertainty object and prospective
   discipline
6. but release-duration and uncertainty are not yet universally solved

That arc is strong, honest, and ambitious enough.

## 2. Recommended paper spine

## 2.1 Introduction

Target job:

- establish the field default
- establish the gap
- define the bolder contribution

Core positioning sentence:

> Current release ML is still dominated by direct supervised prediction from
> descriptors or descriptors plus a few early observations; we instead frame
> drug release as a shared partially observed inference problem with
> mechanism-specific decoders and explicit feasible release-state families.

Must cite:

- `NC / Bannigan`
- release-specific ML review
- computational pharmaceutics umbrella
- dataset / schema maturity

Supporting file:

- [drug_release_paper_positioning_playbook_2026-05-29.md](D:/release-foundation/docs/drug_release_paper_positioning_playbook_2026-05-29.md)

## 2.2 Results 1: Early observations are the dominant forecasting signal

Target job:

- prove that the problem is not mainly solved by pure formulation regression

Primary evidence:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)

Claim carried:

> Sparse early release is the dominant OOD forecasting signal in the audited
> PLGA benchmarks.

## 2.3 Results 2: The mechanism route is not decorative

Target job:

- prove that `mechanism state / theta-family` is scientifically active, not a
  narrative wrapper around direct-Q regression

Primary evidence:

- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv](D:/release-foundation/outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv)

Claim carried:

> A mechanism-routed middle object improves over matched direct curve routes on
> the audited PLGA forecasting core.

## 2.4 Results 3: Better curve fit does not equal solved release duration

Target job:

- keep the paper honest
- show that route superiority depends on the target object

Primary evidence:

- [outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv)
- [outputs/82_release_capability_snapshot/plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)
- [docs/timescale_layer_verdict_2026-05-29.md](D:/release-foundation/docs/timescale_layer_verdict_2026-05-29.md)

Claim carried:

> Curve resemblance, release timing, and release-shape prediction are related
> but non-identical targets.

## 2.5 Results 4: First non-PLGA bridge

Target job:

- prove the interface is no longer PLGA-only

Primary evidence:

- [outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv)
- [outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv)
- [docs/liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)

Claim carried:

> The shared release-intelligence interface survives a first non-PLGA
> retrospective benchmark.

## 2.6 Results 5: Shared uncertainty and prospective discipline

Target job:

- show that the project already exceeds pure point-prediction culture

Primary evidence:

- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

Claim carried:

> The framework already supports calibrated-family artifacts across a real
> PLGA + liposome panel and follows prospective lock discipline ahead of
> chitosan reveal.

## 2.7 Discussion

Target job:

- make the bold unified-release claim
- state the honest limits

Primary evidence:

- [unified_release_capability_snapshot_2026-05-29.md](D:/release-foundation/docs/unified_release_capability_snapshot_2026-05-29.md)
- [outputs/89_release_uq_timescale_gap_snapshot/summary.txt](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/summary.txt)

Core discussion sentence:

> The evidence supports a move toward unified drug-release intelligence, but
> not a claim that release-duration prediction or cross-mechanism uncertainty
> is already solved.

## 3. Main figure map

The paper should probably have `5` main figures.

## Figure 1 — Problem framing and method object

Panels:

1. schematic of current field default:
   `descriptors -> release`
2. schematic of our core object:
   `formulation/material + sparse prefix -> feasible state family -> decoder -> future release`
3. object-level unification layer:
   shared benchmark contract + mechanism-specific decoders + posterior-family

Primary claim:

> The contribution is a different problem formulation, not just another
> regressor.

Primary sources:

- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md)
- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [drug_release_paper_positioning_playbook_2026-05-29.md](D:/release-foundation/docs/drug_release_paper_positioning_playbook_2026-05-29.md)

Do not claim from this figure:

- empirical superiority
- solved cross-mechanism generalization

## Figure 2 — PLGA audited forecasting core

Panels:

1. formulation-only vs early-only vs formulation+early input ablation
2. theta-route vs matched direct-Q comparison
3. middle-layer gain audit summary

Primary claim:

> Early release is the dominant forecasting signal, and the mechanism route
> is empirically active on audited PLGA tasks.

Primary sources:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv](D:/release-foundation/outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv)

Do not claim from this figure:

- cross-mechanism success
- solved release duration

## Figure 3 — Timing and shape reality check

Panels:

1. PLGA canonical OOD timescale tallies
2. PLGA release-shape tallies
3. representative route-ranking divergence example

Primary claim:

> Better curve fit does not automatically imply better release-duration or
> shape prediction.

Primary sources:

- [outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv)
- [outputs/82_release_capability_snapshot/plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)
- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)

Do not claim from this figure:

- our route is the universal best duration predictor

## Figure 4 — First non-PLGA bridge

Panels:

1. liposome `group_by_API` comparison
2. liposome `group_by_release_method` comparison
3. liposome shape descriptors

Primary claim:

> The shared release-intelligence interface is no longer only a PLGA story.

Primary sources:

- [outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv)
- [outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv)
- [outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv)

Do not claim from this figure:

- complete cross-mechanism validation
- universal route superiority

## Figure 5 — Shared uncertainty + prospective discipline

Panels:

1. calibrated-family panel summary across PLGA + liposome
2. representative raw vs global/local conformal shift
3. chitosan lock / preregistration timeline box

Primary claim:

> The framework already supports shared uncertainty artifacts and stronger
> prospective evidence discipline than a purely retrospective benchmark paper.

Primary sources:

- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv](D:/release-foundation/outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

Do not claim from this figure:

- uncertainty-aware duration is solved
- prospective cross-mechanism success is complete

## 4. Supplementary map

These items are useful, but should not carry the headline.

## Supplementary S1

Full benchmark tables and extra split details

Sources:

- [outputs/38d_baselines_groupkfold](D:/release-foundation/outputs/38d_baselines_groupkfold)
- [outputs/62_fib_casp_benchmark](D:/release-foundation/outputs/62_fib_casp_benchmark)
- [outputs/66_ensemble_casp](D:/release-foundation/outputs/66_ensemble_casp)

## Supplementary S2

Posterior-family schema examples and export demos

Sources:

- [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)
- [outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2](D:/release-foundation/outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2)
- [outputs/87_release_conformal_family_export](D:/release-foundation/outputs/87_release_conformal_family_export)

## Supplementary S3

UQ-to-timescale gap disclosure

Sources:

- [outputs/89_release_uq_timescale_gap_snapshot/summary.txt](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/summary.txt)
- [outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv)

This is scientifically important, but better placed as an honest limitation or
systems gap note than as the headline figure.

## 5. Figure-to-claim guardrails

These are the claims that should never be attached to the current paper
package.

1. `universal solved drug-release model`
2. `mechanism route universally dominates all release metrics`
3. `cross-mechanism prospective validation complete`
4. `uncertainty-aware duration solved`
5. `active measurement already wins as the mainline method`

## 6. Minimal paper if written now

If the paper needed to be assembled from current evidence without waiting for
chitosan reveal, the smallest honest version would be:

- Figure 1
- Figure 2
- Figure 3
- Figure 4
- a reduced Figure 5 focused on calibrated-family panel + prospective lock

and the title / abstract should stay at:

> shared partially observed release intelligence

not:

> universal cross-mechanism drug-release foundation model

## 7. Bottom line

The current evidence is already enough for a coherent, ambitious, non-local
release paper.

The main challenge is no longer "what could the story be?"
It is:

> assembling the story so that every figure carries one strong claim and no
> figure is asked to prove more than the evidence supports.
