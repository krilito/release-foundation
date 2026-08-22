# Drug Release Claim Map And Build Queue

Date: `2026-05-28`

Purpose:

```text
Make the unification program operational by defining:
1. which claims are legal now,
2. which claims require new evidence,
3. which benchmarks must be built next,
4. which code / interface pieces are the highest-value queue.
```

This file is the bridge from strategy to execution.

## A. Claim map

The project now has enough evidence to separate claims into four tiers.

## Tier C0 — safe now

These claims are supportable from current audited PLGA and locked chitosan
assets.

### C0.1

> Sparse early release observations are highly informative for forecasting
> later release.

Evidence:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)

### C0.2

> A mechanism-routed middle object outperforms matched direct curve regression
> routes on the current PLGA benchmarks.

Evidence:

- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv](D:/release-foundation/outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv)

### C0.3

> The project is organized around feasible mechanism-state inference rather
> than only direct point prediction.

Evidence:

- [docs/drug_release_closeout_2026-05-28.md](D:/release-foundation/docs/drug_release_closeout_2026-05-28.md)
- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md)

### C0.4

> Prospective evidence discipline is already present through locked
> predictions and pre-registration.

Evidence:

- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [outputs/67_chitosan_prospective/HASHES.txt](D:/release-foundation/outputs/67_chitosan_prospective/HASHES.txt)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

## Tier C1 — bold but only as program / framing

These are valid as forward-looking positioning, not as already-proven results.

### C1.1

> Drug release can be unified at the posterior-object and benchmark-interface
> level, while keeping mechanism-specific decoders.

Status:

- valid as strategy
- not yet proven as finished system

Evidence:

- [docs/drug_release_landscape_and_unification_2026-05-28.md](D:/release-foundation/docs/drug_release_landscape_and_unification_2026-05-28.md)
- [docs/unified_drug_release_program_2026-05-28.md](D:/release-foundation/docs/unified_drug_release_program_2026-05-28.md)

### C1.2

> The project is moving toward a shared release-intelligence framework rather
> than another narrow release regressor.

Status:

- valid as comparative framing
- not yet final proof

## Tier C2 — needs one more benchmark / one more result family

These are the high-value near-term claims.

### C2.1

> The same release-intelligence interface works across at least two distinct
> release mechanisms.

Needed evidence:

1. liposome intake aligned and benchmarked through the same benchmark API
2. PLGA and liposome results written in the same output schema

Current state:

- substantially satisfied for retrospective benchmarking
- PLGA and liposome now both run through the shared benchmark contract for
  point-prediction, threshold timing, and release-shape descriptors
- still incomplete for shared uncertainty reporting

### C2.2

> The method predicts release duration / timescale, not only pointwise curves.

Needed evidence:

1. explicit `t50 / t80 / burst / tail` extraction
2. benchmark tables for timescale metrics
3. at least one practical case showing formulation/material changes map to
   release-duration changes

Current state:

- no longer curve-only
- explicit `t10 / t50 / t80` timing and burst/tail-style descriptors now exist
  on both PLGA and liposome through the same evaluator
- still incomplete as a full claim because uncertainty-aware duration
  reporting and prospective validation are not finished

### C2.3

> The framework provides mechanism-aware uncertainty that remains useful
> across mechanisms.

Needed evidence:

1. same uncertainty object on PLGA + liposome + ideally chitosan reveal
2. honest coverage + width comparison

Current state:

- the current full PLGA + liposome panel now has calibrated family artifacts via
  `76 -> 87`
- still partial because the uncertainty layer is not benchmark-wide,
  timescale-aware, or prospectively validated
- chitosan pending reveal

## Tier C3 — do not claim now

These are currently off limits.

### C3.1

> universal drug-release foundation model

Why blocked:

- no benchmark federation completed,
- no broad mechanism coverage,
- no large-scale data federation,
- no clean leave-one-mechanism-out success.

### C3.2

> active measurement is already the best audited mainline method

Why blocked:

- current adaptive route is not the strongest stable headline asset
- leakage concerns and evaluation cleanliness remain part of the recent audit

### C3.3

> cross-mechanism validation is complete

Why blocked:

- liposome not operationalized end-to-end
- chitosan outcome not yet revealed

## B. Benchmark map

This section says exactly which benchmark jobs matter next.

## B1. Main release benchmark spine

These should be the canonical release benchmarks for the unification route.

| Dataset / mechanism | Role | Current tier | Needed next step |
|---|---|---|---|
| PLGA 321 | main retrospective benchmark | usable now | keep as core |
| Bannigan 181 | second PLGA benchmark / transfer check | usable now | keep as cross-source PLGA support |
| Chitosan batch 1 | prospective anchor | locked, pending reveal | run preregistered reveal |
| Liposome IVR | first non-PLGA retrospective benchmark | intake not complete | align and benchmark |

## B2. Optional release expansions

These are not required for the immediate paper but matter for program growth.

| Candidate | Reason | Priority |
|---|---|---|
| hydrogel release datasets | mechanism contrast and timescale diversity | medium |
| coating / colon-targeted release | cheap-observation and delayed-release behavior | medium |
| Raman / spectroscopy linked release | future observation-modality expansion | lower now, high future value |

## B3. Non-release validation spine

This is for the bigger long-term route, not the immediate release paper.

| Domain | Role | Current status |
|---|---|---|
| battery degradation | first non-release structural validation | planned only |

This remains useful, but it is not the next blocking task for the release
program itself.

## C. Build queue

This is the most important section.

It answers: if we want the project to become unified release intelligence,
what do we build first?

## Queue 1 — release benchmark API

Why first:

- without a shared API, all unification talk stays rhetorical

Status on `2026-05-29`:

- minimal executable skeleton landed in
  [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)
- smoke artifact written to
  [outputs/80_release_partial_observation_benchmark_smoke](D:/release-foundation/outputs/80_release_partial_observation_benchmark_smoke)

Deliverables:

1. common aligned dataset contract for release tasks
2. common benchmark output contract
3. common early/future split contract

Suggested artifact:

```text
scripts/80_release_partial_observation_benchmark.py
```

Expected inputs:

```text
formulations.csv
curves_long.csv
mechanism_metadata.json
optional_theta_targets.*
```

Expected outputs:

```text
metrics_summary.csv
metrics_by_curve.csv
timescale_summary.csv
prediction_trajectories.csv
uq_summary.csv
split_metadata.json
```

## Queue 2 — liposome intake and adapter

Why second:

- liposome is the nearest real cross-mechanism proof

Status on `2026-05-29`:

- intake landed in
  [scripts/78_liposome_ivr_intake.py](D:/release-foundation/scripts/78_liposome_ivr_intake.py)
- benchmark-ready assets written to
  [outputs/78_liposome_ivr_intake](D:/release-foundation/outputs/78_liposome_ivr_intake)
  with `33` retained curves, `8` APIs, and `3` release methods
- shared benchmark API validation without predictions completed for
  [outputs/80_liposome_api_split](D:/release-foundation/outputs/80_liposome_api_split)
  and
  [outputs/80_liposome_method_split](D:/release-foundation/outputs/80_liposome_method_split)
- shared-format liposome predictions landed in
  [scripts/79_liposome_prediction_export.py](D:/release-foundation/scripts/79_liposome_prediction_export.py)
- first end-to-end `group_by_API` comparison through the shared benchmark now exists:
  [outputs/80_liposome_group_by_api_our_et_refined](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined)
  versus
  [outputs/80_liposome_group_by_api_direct_et_q](D:/release-foundation/outputs/80_liposome_group_by_api_direct_et_q)
- second split-family check now exists for `group_by_release_method`:
  [outputs/80_liposome_group_by_method_our_et_refined](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined)
  versus
  [outputs/80_liposome_group_by_method_direct_et_q](D:/release-foundation/outputs/80_liposome_group_by_method_direct_et_q)
- interpretation is recorded in
  [liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)
- full liposome benchmark coverage across all schemes / UQ is still pending

Deliverables:

1. aligned liposome `formulations + curves + kinetic targets`
2. decoder or kinetic-adapter wrapper
3. same benchmark API compatibility

Suggested artifacts:

```text
scripts/78_liposome_ivr_intake.py
weibull_simulator.py or liposome adapter successor
outputs/78_liposome_ivr_intake/
```

Success condition:

PLGA and liposome both run through the same benchmark script.

## Queue 3 — timescale-first evaluation layer

Why third:

- this is closest to the actual product / scientific need

Status on `2026-05-29`:

- dedicated evaluator landed in
  [scripts/81_release_timescale_eval.py](D:/release-foundation/scripts/81_release_timescale_eval.py)
- first explicit liposome timescale verdict written in
  [timescale_layer_verdict_2026-05-29.md](D:/release-foundation/docs/timescale_layer_verdict_2026-05-29.md)
- the current layer now distinguishes all-threshold timing from
  forecast-relevant timing after the early-observation window
- burst and tail descriptors now land in the same evaluator:
  - `burst_fraction_at_early_window`
  - `post_window_release`
  - `residual_tail_at_tmax`
  - `tail_auc_after_early_window`

Deliverables:

1. `t10 / t50 / t80` extraction utilities
2. burst and tail descriptors
3. summary metrics for duration prediction

Suggested artifact:

```text
scripts/81_release_timescale_eval.py
```

This script should consume prediction trajectories and observed trajectories,
not retrain models.

## Queue 4 — common posterior-family object

Why fourth:

- after benchmark federation, define the shared algorithmic object formally

Status on `2026-05-29`:

- minimal executable schema now exists in
  [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- example mechanism instances now exist in
  [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)
- benchmark-side point-estimate family exports now exist for the current
  mechanism routes via
  [85_release_posterior_family_export.py](D:/release-foundation/scripts/85_release_posterior_family_export.py)
- selected low-rank family exports now exist directly from FIB / ensemble UQ
  via
  [86_release_casp_family_export.py](D:/release-foundation/scripts/86_release_casp_family_export.py)
- selected conformal-attached family exports now exist via
  [87_release_conformal_family_export.py](D:/release-foundation/scripts/87_release_conformal_family_export.py)
- schema rationale is recorded in
  [release_posterior_family_schema_2026-05-29.md](D:/release-foundation/docs/release_posterior_family_schema_2026-05-29.md)
- still not wired into the full benchmark / UQ stack as the canonical
  calibrated-family output object benchmark-wide

Deliverables:

1. one output schema for PLGA / liposome / chitosan posterior-family objects
2. downstream simulation and uncertainty code consuming the same object

Suggested schema:

```json
{
  "mechanism_id": "",
  "z_center": [],
  "z_basis": [],
  "z_scale": [],
  "decoder_handle": "",
  "assay_context": {},
  "metadata": {}
}
```

This should live in code, not just docs, once Queue 1-3 are stable.

## Queue 5 — active measurement revival

Why fifth:

- important, but not before release benchmark federation and timescale layer

Deliverables:

1. choose next assay time under equal budget
2. report benefit on future uncertainty or timescale uncertainty

Important:

Do not center the release program on this until the simpler release
intelligence core is fully clean.

## D. What not to build first

These are seductive, but should not be first in queue.

1. `universal neural observer`
   - too much risk, not enough benchmark structure

2. `battery integration before liposome`
   - useful for the broader scientific-dynamics route, but liposome is the
     nearest release-specific cross-mechanism proof

3. `new huge PLGA model zoo`
   - low strategic return relative to benchmark federation

4. `full platform UI`
   - decision-support framing is important, but interface work should follow
     stable benchmark and output objects

## E. Decision gates

These gates tell us what story becomes legal after each milestone.

## Gate G1 — liposome benchmark complete

Needed:

- liposome intake aligned
- same benchmark API as PLGA
- honest benchmark outputs written

Current status on `2026-05-29`:

- substantially satisfied for point-prediction benchmarking
- strengthened by two split families (`group_by_API`, `group_by_release_method`)
- not yet complete for uncertainty-aware coverage

Enables:

- claim that the release-intelligence interface is no longer PLGA-only

## Gate G2 — timescale layer complete

Needed:

- `t50 / t80 / burst / tail` metrics written for at least PLGA + liposome

Current status on `2026-05-29`:

- substantially satisfied for curve/timing/shape reporting
- explicit `t10 / t50 / t80` timing layer exists and is already informative on liposome and four canonical PLGA OOD cells
- burst and tail descriptors now exist in the same shared evaluator
- current PLGA panel already shows route-ranking divergence:
  - mechanism route wins median curve fit more consistently
  - direct route often wins future-only threshold timing
- still missing uncertainty-aware duration summaries and a broader non-PLGA shape/timescale panel beyond liposome

Enables:

- claim that the framework targets release duration, not just curve fit

## Gate G3 — chitosan reveal passes prereg primary endpoint

Needed:

- formal reveal using preregistered evaluator

Enables:

- claim that the framework survived at least one locked prospective
  cross-material / cross-mechanism stress test

## Gate G4 — posterior-family object shared in code

Needed:

- PLGA / liposome / chitosan all emit one common object shape

Current status on `2026-05-29`:

- substantially advanced but not complete
- the common object shape now exists in code and has executable examples for
  PLGA, liposome, and chitosan
- current mechanism-route benchmark exports can now emit point-estimate
  posterior-family objects
- selected CASP UQ cells can now emit low-rank family objects
- selected calibrated family artifacts now exist for:
  - `cross321 / group_by_drug`
  - `liposome / group_by_drug`
- still incomplete because this is still a partial panel rather than the full
  benchmark/UQ stack, and the conformal attachment lives in decoder-output
  space rather than as latent-space recalibration

Enables:

- strongest honest version of "unified release intelligence"

## F. Recommended immediate next three tasks

If we want maximum strategic movement, the next three tasks should be:

1. `extend calibrated-family export beyond selected FIB cells`
2. `extend cross-mechanism uncertainty reporting through the shared benchmark`
3. `prepare the chitosan reveal package so the preregistered readout is one command`

That order matters.

## Bottom line

The project is already beyond "just another release predictor" in scientific
intent, but it is not yet beyond that in executable system form.

The conversion path is now clearer:

1. claim discipline,
2. benchmark federation,
3. liposome bridge,
4. timescale layer,
5. uncertainty and prospective closure,
6. shared posterior-family code object.

The first four are now substantially underway in executable form. The next
identity jump comes from closing uncertainty and prospective evidence without
breaking the current evidence discipline.
