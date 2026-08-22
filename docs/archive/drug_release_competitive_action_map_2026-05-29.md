# Drug Release Competitive Action Map

Date: `2026-05-29`

Purpose:

```text
Turn the external field map into an execution map:
1. who we should fight head-on,
2. who we should borrow from,
3. who we should defend against strategically,
4. which local evidence already supports each move,
5. what exact next step would strengthen our position.
```

This is the bridge from external landscape to project strategy.

## 1. Executive verdict

The field does not currently have one single dominant "drug-release AI"
competitor. It has three different kinds of pressure:

1. `benchmark pressure`
2. `platform pressure`
3. `infrastructure pressure`

Those pressures should not be answered the same way.

The current project should:

- `fight` direct-release benchmark rivals on partial-observation forecasting
- `borrow` mechanism-summary and schema discipline from workflow / dataset papers
- `defend` against platform-style systems by moving toward decision-support
  language without losing benchmark discipline

## 2. The three pressure types

## 2.1 Benchmark pressure

These are systems that compete on the narrow question:

```text
Can you predict release curves better than the usual ML baseline?
```

Main examples:

- [Bannigan et al., Nature Communications 2023](https://www.nature.com/articles/s41467-022-35343-w)
- [PLGA interpretable two-stage ML, Pharmaceuticals 2026](https://www.mdpi.com/1424-8247/19/5/767)

What they threaten:

- simpler release-prediction story
- easier few-shot / zero-shot communication
- established benchmark identity

What our current local evidence says:

- sparse early release is the strongest OOD signal:
  [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
- a mechanism-routed middle object is not cosmetic:
  [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- the project already has stronger object design than direct-Q baselines:
  [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)

Correct move:

> fight them on `partial-observation release forecasting`, not on vague
> "better ML" language.

## 2.2 Platform pressure

These systems are dangerous because they may win on workflow value even if
their release forecasting core is shallower.

Main examples:

- [FormulationLAI, Journal of Controlled Release 2026](https://www.sciencedirect.com/science/article/pii/S0168365925010326)
- [FormulationAI, Briefings in Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/37991246/)
- active-learning / optimization lines summarized in
  [Advanced Drug Delivery Reviews 2025](https://www.sciencedirect.com/science/article/pii/S0169409X25002467)

What they threaten:

- translational story
- design-loop utility
- user-facing decision-support framing

What our current local evidence says:

- we already have the ingredients for a stronger scientific core:
  - shared benchmark API:
    [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)
  - shared posterior-family object:
    [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
  - release timing and shape reporting:
    [scripts/81_release_timescale_eval.py](D:/release-foundation/scripts/81_release_timescale_eval.py)
- but our decision-support layer is still not the headline product

Correct move:

> do not try to out-platform them right now; instead, reframe our outputs as
> uncertainty-aware release decisions and assay decisions.

## 2.3 Infrastructure pressure

These systems shape what the field treats as reusable or benchmarkable.

Main examples:

- [Scientific Data PLGA dataset 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873201/)
- [Digital Discovery liposome workflow 2025](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00112a)
- [Computational pharmaceutics review 2021](https://www.sciencedirect.com/science/article/pii/S0168365921004363)

What they threaten:

- if they define the field's data conventions first, our algorithm story looks
  narrower than it is

What our current local evidence says:

- we already have the start of an internal benchmark federation:
  - PLGA shared exports:
    [scripts/83_plga_prediction_export.py](D:/release-foundation/scripts/83_plga_prediction_export.py)
  - liposome shared intake:
    [scripts/78_liposome_ivr_intake.py](D:/release-foundation/scripts/78_liposome_ivr_intake.py)
  - shared uncertainty panel:
    [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)

Correct move:

> borrow aggressively. This is not where we need originality theater.

## 3. Action classes: fight, borrow, defend

## 3.1 Fight

These are the exact questions where we should try to beat the field cleanly.

### Fight F1

```text
Given sparse early observations, can the mechanism route beat direct release
regression on harder OOD forecasting tasks?
```

Current support:

- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/82_release_capability_snapshot/plga_theta_gain_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_theta_gain_snapshot.csv)

Main rival:

- NC / Bannigan

### Fight F2

```text
Can the same release-intelligence interface survive at least one non-PLGA
benchmark?
```

Current support:

- [outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv)
- [outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv)
- [docs/liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)

Main rival:

- the general field expectation that release models are material-family-local

### Fight F3

```text
Can release forecasting be represented as a feasible-state family rather than
only a direct point estimate?
```

Current support:

- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv](D:/release-foundation/outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv)

Main rival:

- the field's default direct-regression framing

## 3.2 Borrow

These are areas where the fastest winning move is to copy the good idea and
integrate it into our stronger release-intelligence frame.

### Borrow B1

`few-shot / zero-shot` communication from NC

Why:

- simple
- field-recognizable
- deployment-friendly

### Borrow B2

mechanism-specific assay summary discipline from liposome IVR workflow

Why:

- non-PLGA release data is messy
- mechanism-appropriate summaries make bridges more credible

### Borrow B3

metadata-first intake discipline from Scientific Data / AI-ready data work

Why:

- benchmark federation will otherwise stay ad hoc

### Borrow B4

decision-support language from FormulationAI / optimization platforms

Why:

- this prevents us from looking like "just another benchmark paper"

## 3.3 Defend

These are the ways the field can beat us if we do not move.

### Defend D1

Against `platform capture`

Risk:

- more translational systems may look more valuable than a stronger scientific
  core

Defense:

- phrase outputs as release decisions:
  - future release range
  - release timescale
  - burst / tail behavior
  - assay value of the next observation

### Defend D2

Against `data-layer capture`

Risk:

- if others define reusable release-data standards first, our framework looks
  bespoke

Defense:

- keep shared export contracts central:
  - [scripts/78_liposome_ivr_intake.py](D:/release-foundation/scripts/78_liposome_ivr_intake.py)
  - [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)
  - [scripts/83_plga_prediction_export.py](D:/release-foundation/scripts/83_plga_prediction_export.py)

### Defend D3

Against `metric capture`

Risk:

- the field can reduce everything back to pointwise curve fit

Defense:

- keep timing / shape / UQ layers visible:
  - [scripts/81_release_timescale_eval.py](D:/release-foundation/scripts/81_release_timescale_eval.py)
  - [outputs/89_release_uq_timescale_gap_snapshot/summary.txt](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/summary.txt)

## 4. Where we are already ahead

These are the slots where the project has a real edge now.

1. `shared partially observed release benchmark discipline`
2. `explicit feasible-state / posterior-family object`
3. `mechanism-routed versus direct-route audit culture`
4. `prospective lock discipline`

Evidence:

- [docs/unified_release_capability_snapshot_2026-05-29.md](D:/release-foundation/docs/unified_release_capability_snapshot_2026-05-29.md)
- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)

## 5. Where we are still vulnerable

These are the slots where the field can still out-position us.

1. `simplicity of public benchmark narrative`
2. `decision-support and translational packaging`
3. `benchmark-wide uncertainty-aware duration / shape`
4. `completed prospective cross-mechanism validation`

Evidence of the current biggest systems gap:

- [outputs/89_release_uq_timescale_gap_snapshot/summary.txt](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/summary.txt)

## 6. The next three moves that matter most

These are the most leverage-rich moves given the external field shape.

### Move M1

Close the gap between calibrated family artifacts and duration / shape outputs.

Why:

- that is the cleanest place where the field cannot simply answer with
  "we also have a regressor"

Evidence gap anchor:

- [outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv)

### Move M2

Treat liposome as a first-class non-PLGA bridge in every high-level story.

Why:

- it is the current proof that we are not only a PLGA-local benchmark

Evidence anchor:

- [docs/liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)

### Move M3

Prepare the chitosan reveal as a credibility jump, not just one more eval.

Why:

- it is the hardest external answer to benchmark-only skepticism

Evidence anchor:

- [docs/drug_release_closeout_2026-05-28.md](D:/release-foundation/docs/drug_release_closeout_2026-05-28.md)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

## 7. Bottom line

The external field is crowded, but not in one way.

- benchmark rivals should be fought
- infrastructure papers should be borrowed from
- platform systems should be defended against by moving toward decision support

That means the project's best strategic identity is now:

> not the best single-material regressor,
> but the strongest evidence-backed path toward unified drug-release
> intelligence.
