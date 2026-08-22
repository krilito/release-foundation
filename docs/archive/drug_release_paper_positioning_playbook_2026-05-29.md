# Drug Release Paper Positioning Playbook

Date: `2026-05-29`

Purpose:

```text
Convert the external field map and current internal evidence into a
paper-facing positioning guide:
1. who must appear in the Introduction,
2. who belongs in Results comparisons,
3. who belongs in Discussion / future work,
4. what exact claim layer each comparator licenses,
5. what wording is safe versus overreach.
```

This is a writing and positioning manual, not a benchmark artifact.

## 1. Executive verdict

If we wrote the paper today, the cleanest positioning is:

> Most release ML still treats the task as descriptor regression or
> descriptor-plus-early-point forecasting. Our contribution is to recast
> release as a shared partially observed inference problem with
> mechanism-specific decoders and a feasible release-state family.

That sentence is the center of gravity.

Everything else should support it.

## 2. Who must be in the Introduction

These are not optional. If they are absent, the paper will look poorly
positioned.

## 2.1 NC / Bannigan

Use for:

- the few-shot / zero-shot deployment frame
- the strongest simple public baseline identity
- proof that release forecasting with early sparse data is already a live
  problem

Why Introduction:

- reviewers will know or quickly find it
- it anchors the field's current "practical ML for release" story

How to position:

> Prior work framed long-acting injectable release as a few-shot / zero-shot
> supervised prediction problem, but remained primarily direct-regression
> based.

Evidence anchor:

- [Nature Communications 2023](https://www.nature.com/articles/s41467-022-35343-w)

## 2.2 Release-specific standard ML review

Use for:

- showing what "standard ML for release" currently means
- naming the default model family: ANN, RF, XGBoost, SVM, ensembles

Why Introduction:

- it defines reviewer priors

How to position:

> Recent reviews still describe drug-release prediction largely through
> supervised tabular ML models rather than shared partially observed
> release-intelligence systems.

Evidence anchor:

- [Computer Methods and Programs in Biomedicine 2025 review](https://www.sciencedirect.com/science/article/pii/S0010482525001064)

## 2.3 Computational pharmaceutics / formulation-AI umbrella

Use for:

- justifying a bigger ambition than one benchmark
- linking release prediction to broader hybrid formulation science

Why Introduction:

- this gives permission for boldness without sounding detached from the field

How to position:

> The broader computational-pharmaceutics literature motivates integrating
> mechanism-aware simulation, ML, and formulation design workflows, but has
> not yet organized controlled-release prediction around a shared partially
> observed inference benchmark.

Evidence anchors:

- [Advanced Drug Delivery Reviews 2021](https://www.sciencedirect.com/science/article/pii/S0168365921004363)
- [Advanced Drug Delivery Reviews 2025](https://www.sciencedirect.com/science/article/pii/S0169409X25002467)

## 2.4 Dataset / schema paper

Use for:

- legitimizing the PLGA data foundation
- showing that the field is mature enough for benchmark discipline

How to position:

> The release field is beginning to develop reusable datasets and metadata
> structures, which makes shared benchmark contracts increasingly necessary.

Evidence anchor:

- [Scientific Data 2025 PLGA dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873201/)

## 3. Who should appear in Results comparisons

These are the comparators that matter to the main empirical story.

## 3.1 Direct-Q and descriptor baselines

These must remain the main empirical foil because they answer the same local
question as the project.

Use for:

- proving that the mechanism route is not decorative
- showing that sparse early release matters more than formulation-only OOD

Evidence anchors:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)

Recommended role:

> main internal empirical comparison

## 3.2 Liposome bridge

Use for:

- proving this is no longer only a PLGA story
- showing that the same interface survives a non-PLGA release family

Evidence anchors:

- [docs/liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)
- [outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv)
- [outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv)

Recommended role:

> first cross-mechanism retrospective bridge

## 3.3 Timescale / shape split

Use for:

- preventing the paper from collapsing back to curve fit only
- showing that route superiority depends on the target object

Evidence anchors:

- [outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv)
- [outputs/82_release_capability_snapshot/plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)
- [outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv)
- [docs/timescale_layer_verdict_2026-05-29.md](D:/release-foundation/docs/timescale_layer_verdict_2026-05-29.md)

Recommended role:

> main qualifier that keeps the Results section honest and stronger

## 3.4 Prospective lock status

Use for:

- proving evidence discipline even before reveal

Evidence anchors:

- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

Recommended role:

> prospective-discipline result, not completed validation result

## 4. Who belongs mainly in Discussion

These are important, but they should not carry the main empirical claim.

## 4.1 FormulationLAI and platform-like systems

Use for:

- explaining why a platform / decision-support future matters
- acknowledging that workflow value is a real frontier

Why Discussion:

- they are adjacent strategically, not the cleanest matched benchmark rival

Evidence anchor:

- [FormulationLAI 2026](https://www.sciencedirect.com/science/article/pii/S0168365925010326)

## 4.2 FormulationAI-like systems

Use for:

- showing that the field rewards actionable formulation workflows
- motivating decision-support phrasing

Evidence anchor:

- [FormulationAI 2023](https://pubmed.ncbi.nlm.nih.gov/37991246/)

## 4.3 PINN / future hybrid lines

Use for:

- future method space
- signaling that the field may move toward stronger physics-aware deep models

Evidence anchor:

- [PINN preprint 2026](https://arxiv.org/abs/2602.09963)

## 5. Safe wording versus overreach

## 5.1 Safe wording now

These are supported by current evidence.

### Safe S1

> Sparse early release observations are a dominant OOD forecasting signal.

Supported by:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)

### Safe S2

> A mechanism-routed middle object improves over matched direct curve routes
> on the audited PLGA forecasting core.

Supported by:

- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)

### Safe S3

> The project supports shared release-intelligence framing through a common
> benchmark contract, a posterior-family object, and a first non-PLGA bridge.

Supported by:

- [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)
- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [docs/liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)

### Safe S4

> The project already includes prospective lock discipline ahead of wet-lab
> reveal.

Supported by:

- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)

## 5.2 Wording that is too large now

### Unsafe U1

> We solved drug release prediction across mechanisms.

Why blocked:

- no completed prospective non-PLGA reveal
- no leave-one-mechanism-out success

### Unsafe U2

> Our route is the universal best predictor of release duration.

Why blocked:

- PLGA and liposome both show route-ranking divergence across curve, timing,
  and shape

Evidence:

- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)

### Unsafe U3

> Uncertainty is solved across mechanisms.

Why blocked:

- calibrated family panel exists, but uncertainty-aware duration / shape is
  still missing

Evidence:

- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/89_release_uq_timescale_gap_snapshot/summary.txt](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/summary.txt)

## 6. Recommended paper spine if written now

The cleanest high-level structure would be:

1. `Problem framing`
   release forecasting is still mostly treated as direct regression
2. `Core method`
   formulation/material + sparse prefix -> feasible state family -> decoder
3. `Audited PLGA evidence`
   early signal, direct-vs-mechanism, timing/shape split
4. `First non-PLGA bridge`
   liposome under the same shared interface
5. `Prospective discipline`
   chitosan lock and preregistration
6. `Discussion`
   toward unified drug-release intelligence, but not yet a universal solved
   model

## 7. Bottom line

If the paper were written today, the strongest stance would be:

> We are not introducing another direct release regressor. We are showing that
> controlled release can be organized as a shared partially observed inference
> problem with mechanism-specific decoders, explicit feasible-state objects,
> and cross-mechanism benchmark discipline.

That is strong enough to be ambitious, and still consistent with the current
evidence.
