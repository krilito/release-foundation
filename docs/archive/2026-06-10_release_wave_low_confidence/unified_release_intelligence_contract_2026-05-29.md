# Unified Release-Intelligence Contract

Date: `2026-05-29`

Purpose:

```text
Make the "unified drug-release intelligence" claim concrete as a systems
contract:
1. what the external field is currently solving,
2. what layers can actually be unified now,
3. what must remain mechanism-specific,
4. what evidence already supports each layer locally,
5. what is still missing before stronger claims become legal.
```

This is the executable interpretation of the project's boldest honest claim.

## 1. One-sentence contract

> A unified drug-release system should share data intake, partially observed
> benchmark rules, posterior-family objects, and reporting outputs, while
> leaving release-curve decoding mechanism-specific.

That sentence is the contract.

## 2. Why the field is still fragmented

The current external field mostly solves one of six narrower problems:

1. `descriptor-only supervised prediction`
2. `descriptor + early-point few-shot forecasting`
3. `kinetic-fit / mechanism-summary workflows`
4. `physics / simulator + ML hybrid design`
5. `active-learning / optimization systems`
6. `data / platform / schema layers`

Reference:

- [drug_release_algorithm_taxonomy_and_unification_bets_2026-05-29.md](D:/release-foundation/docs/drug_release_algorithm_taxonomy_and_unification_bets_2026-05-29.md)

That means most published systems do **not** yet share the same problem
definition, even when they all claim to be doing "drug release ML."

## 3. The five-layer unification stack

The project should treat unification as a layered stack, not a single model.

### Layer 1 — Shared intake schema

All release mechanisms should be representable as:

- formulation / material context
- assay context
- time grid
- fractional release observations

Current local evidence:

- [78_liposome_ivr_intake.py](D:/release-foundation/scripts/78_liposome_ivr_intake.py)
- [80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)
- [outputs/78_liposome_ivr_intake](D:/release-foundation/outputs/78_liposome_ivr_intake)

### Layer 2 — Shared partially observed benchmark

All mechanisms should be evaluable under the same task families:

- `descriptor-only`
- `early-observation-assisted`
- `OOD split families`
- `prospective lock discipline`

Current local evidence:

- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)

### Layer 3 — Shared posterior-family object

All mechanisms should emit a shared object that records:

- which mechanism family is active
- the feasible release-state center
- any low-rank family basis / scale
- partial-observation context
- calibration context
- decoder handle

Current local evidence:

- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [release_posterior_family_schema_2026-05-29.md](D:/release-foundation/docs/release_posterior_family_schema_2026-05-29.md)
- [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)
- [outputs/87_release_conformal_family_export](D:/release-foundation/outputs/87_release_conformal_family_export)

### Layer 4 — Mechanism-specific decoders

This layer is intentionally **not** unified in parameter semantics.

Examples:

- `PLGA` biphasic ODE
- `liposome` Weibull kinetic adapter
- `chitosan` Ritger-Peppas-style scaffold

Current local evidence:

- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md)
- [outputs/84_release_posterior_family_examples/summary.csv](D:/release-foundation/outputs/84_release_posterior_family_examples/summary.csv)

### Layer 5 — Shared decision-support / reporting layer

All mechanisms should emit the same classes of downstream outputs:

- future release curve
- timing metrics such as `t10 / t50 / t80`
- shape descriptors such as burst / tail
- uncertainty summaries
- eventually next-measurement or assay-support outputs

This layer should now be understood as:

- shared target classes
- shared reporting schema
- mechanism-aware default thresholds
- horizon-aware exclusions for overly deep targets

Current local evidence:

- [81_release_timescale_eval.py](D:/release-foundation/scripts/81_release_timescale_eval.py)
- [88_release_uncertainty_snapshot.py](D:/release-foundation/scripts/88_release_uncertainty_snapshot.py)
- [outputs/89_release_uq_timescale_gap_snapshot/summary.txt](D:/release-foundation/outputs/89_release_uq_timescale_gap_snapshot/summary.txt)
- [threshold_target_redesign_verdict_2026-05-29.md](D:/release-foundation/docs/threshold_target_redesign_verdict_2026-05-29.md)
- [release_target_layer_contract_2026-05-29.md](D:/release-foundation/docs/release_target_layer_contract_2026-05-29.md)
- [release_shape_target_contract_2026-05-29.md](D:/release-foundation/docs/release_shape_target_contract_2026-05-29.md)
- [release_target_registry_verdict_2026-05-29.md](D:/release-foundation/docs/release_target_registry_verdict_2026-05-29.md)
- [release_stack_manifest_verdict_2026-05-29.md](D:/release-foundation/docs/release_stack_manifest_verdict_2026-05-29.md)

## 4. What is shared and what is not

### Shared now

- curve schema
- benchmark task definition
- posterior-family serialization
- curve / timing / shape reporting classes
- coverage and calibration bookkeeping

### Not shared now, and should not be forced to be shared

- raw physical parameter meaning
- decoder equation
- mechanism-specific state dimension
- assay timescale
- one universal route winner across all target objects

That boundary is what keeps the unified claim legal.

## 5. How external families map onto this contract

### Descriptor-only predictors

Usually occupy:

- Layer 1 partially
- Layer 2 weakly
- almost never Layer 3

### Few-shot early-observation forecasters

Usually occupy:

- Layer 1
- Layer 2 strongly
- often still skip explicit Layer 3

### Kinetic-fit / mechanism-summary workflows

Usually occupy:

- Layer 1
- Layer 4 strongly
- only partially Layer 5

### Physics / simulator + ML hybrid systems

Can occupy:

- Layers 1, 4, and a design-oriented version of Layer 5
- but often without a shared partially observed benchmark or shared family
  object

### Platform / schema systems

Usually occupy:

- Layer 1 strongly
- parts of Layer 5 in user-facing form
- but without proving forecasting or uncertainty claims

This mapping is why the project still has open ground: almost nobody owns the
whole stack.

## 6. What is already true locally

The strongest honest local sentence right now is:

> We already have a shared intake and benchmark layer, a first non-PLGA
> bridge, a shared posterior-family object, a nine-cell calibrated-family
> panel across PLGA and liposome, and a locked prospective chitosan package.

Evidence bundle:

- [unified_release_capability_snapshot_2026-05-29.md](D:/release-foundation/docs/unified_release_capability_snapshot_2026-05-29.md)
- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/91_release_paper_figures/figure1](D:/release-foundation/outputs/91_release_paper_figures/figure1)

## 7. What is still not legal to claim

This contract does **not** yet justify saying:

1. `one universal solved release model`
2. `cross-mechanism forecasting is complete`
3. `uncertainty-aware duration is solved`
4. `all mechanisms share one raw physical latent state`
5. `active measurement is already the audited winner`

Those remain beyond current evidence.

## 8. What would upgrade the claim next

Three upgrades matter most:

1. `route-consistent uncertainty-aware duration / shape`
2. `completed chitosan reveal`
3. `a second non-PLGA bridge beyond liposome`

Until then, the right ambition is:

> unified drug-release intelligence stack

not:

> universal solved drug-release foundation model

## Bottom line

The unification target should be:

```text
shared interface
+ shared partially observed benchmark
+ shared posterior-family object
+ mechanism-specific decoders
+ shared reporting and decision-support layer
```

That is already bolder than most of the field, and it is also much more
grounded than pretending a single black-box release model has already solved
everything.
