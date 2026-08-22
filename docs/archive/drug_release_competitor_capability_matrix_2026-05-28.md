# Drug Release Competitor Capability Matrix

Date: `2026-05-28`

Purpose:

```text
Compare representative external works and platforms against our current
release-foundation stack at the capability level rather than by one metric.
```

This file answers:

1. who is a real competitor,
2. which capability layers they occupy,
3. where we are already stronger,
4. where we are still weaker,
5. what we should borrow instead of merely trying to "beat" them.

## Rating key

- `yes` = clearly present and central
- `partial` = present, but limited, indirect, or not central
- `no` = not present or not evidenced
- `pending` = plausible in our program, not yet proven

## Capability dimensions

| Capability | Meaning |
|---|---|
| `multi_mechanism_scope` | More than one release-mechanism family is explicitly supported |
| `early_obs_forecast` | Sparse early observations are used to forecast later release |
| `full_curve_target` | Predicts or evaluates the whole release trajectory, not just one endpoint |
| `mechanism_latent` | Uses an explicit latent mechanism / kinetic state instead of only direct curve regression |
| `calibrated_uq` | Reports or builds uncertainty with calibration discipline |
| `active_measurement` | Recommends what to measure next under limited budget |
| `prospective_lock` | Locked prospective predictions before wet-lab reveal |
| `data_platform_value` | Contributes reusable data schema / dataset / workflow infrastructure |
| `timescale_first` | Treats release duration / timescale as a primary object rather than only pointwise curve fit |
| `decision_support` | Aims at actionable formulation or assay decisions, not only retrospective prediction |

## External competitor matrix

| System / paper | Type | multi_mechanism_scope | early_obs_forecast | full_curve_target | mechanism_latent | calibrated_uq | active_measurement | prospective_lock | data_platform_value | timescale_first | decision_support | Main role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bannigan / NC LAI 2023 | paper + code | no | yes | yes | no | no | no | partial | no | partial | partial | direct release-prediction benchmark |
| Explainable PLGA release ML follow-up | paper | no | yes | yes | no | no | no | no | no | partial | partial | interpretable PLGA forecasting |
| Liposome IVR / accelerated IVR | dataset + workflow | partial | partial | yes | partial | no | no | no | yes | yes | partial | non-PLGA data and kinetic workflow |
| AI-ready release data foundation paper | data / standards paper | partial | no | no | no | no | no | no | yes | no | partial | schema and standardization |
| Small hydrogel / nanofiber / nanoparticle ML papers | paper family | no | partial | yes | partial | no | no | no | no | partial | partial | narrow material-family modeling |
| PINN / physics-aware release papers | paper family | no | partial | yes | partial | partial | no | no | no | partial | partial | early physics-aware deep learning |
| FormulationAI / formulation strategy platforms | platform | no | no | no | no | no | no | no | partial | no | yes | broader formulation decision support |

## Our current project matrix

Evidence base for this row:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv](D:/release-foundation/outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv)
- [outputs/62_fib_casp_benchmark/aggregate_table.csv](D:/release-foundation/outputs/62_fib_casp_benchmark/aggregate_table.csv)
- [outputs/66_ensemble_casp/aggregate_table.csv](D:/release-foundation/outputs/66_ensemble_casp/aggregate_table.csv)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)
- [docs/drug_release_closeout_2026-05-28.md](D:/release-foundation/docs/drug_release_closeout_2026-05-28.md)

| System / paper | Type | multi_mechanism_scope | early_obs_forecast | full_curve_target | mechanism_latent | calibrated_uq | active_measurement | prospective_lock | data_platform_value | timescale_first | decision_support | Main role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| release-foundation current audited core | code + benchmarks | partial | yes | yes | yes | partial | partial | yes | partial | partial | partial | mechanism-routed partial-observation release forecasting |

Interpretation:

- `multi_mechanism_scope = partial`
  because PLGA is real, chitosan is prediction-locked but not yet revealed,
  and liposome is a viable path but not yet benchmark-complete.
- `calibrated_uq = partial`
  because CASP exists and is meaningful, but cross-mechanism and hard OOD
  calibration are not solved.
- `active_measurement = partial`
  because the route exists as a research branch, but is not the strongest
  audited headline asset.
- `timescale_first = partial`
  because the practical goal is timescale prediction, but current benchmark
  outputs are still more curve-centric than timescale-centric.

## Head-to-head conclusions

## 1. Versus Bannigan / NC

Where we are stronger:

- explicit middle mechanism object
- audited direct-Q vs mechanism-route comparison
- stronger partial-observation scientific framing
- better path toward calibrated uncertainty
- stronger prospective-evidence discipline through lock + pre-registration

Where they are stronger:

- simpler and easier-to-explain supervised prediction story
- already-published benchmark identity
- cleaner immediate "few-shot vs zero-shot" communication

What we should borrow:

- clear few-shot / zero-shot language
- deployment-simple framing
- presentation simplicity

What we should not imitate:

- remaining purely direct-regression-centric

## 2. Versus liposome IVR / accelerated IVR

Where we are stronger:

- stronger forecasting object
- stronger mechanistic ambition
- stronger uncertainty / posterior-family direction

Where they are stronger:

- non-PLGA data ecosystem
- release-data organization
- mechanism-appropriate kinetic summary discipline

What we should borrow:

- dataset curation style
- liposome-specific decoder / kinetic adapter thinking
- benchmark-ready data packaging

## 3. Versus AI-ready release data standardization papers

Where we are stronger:

- forecasting and inference ambition
- algorithmic novelty potential

Where they are stronger:

- schema discipline
- interoperability mindset
- argument that the field needs shared data structure before grand models

What we should borrow:

- metadata-first design
- release benchmark federation
- explicit intake criteria for datasets

## 4. Versus FormulationAI / formulation decision platforms

Where we are stronger:

- deeper release forecasting object
- mechanism-aware structure
- direct fit to controlled-release scientific question

Where they are stronger:

- platform mentality
- user-facing decision-support framing
- broader workflow value proposition

What we should borrow:

- move from "paper model" to "decision-support system" thinking
- formulate outputs as actionable decisions, not only predictions

## 5. Versus small narrow material-family ML papers

Where we are stronger:

- broader scientific ambition
- more rigorous benchmarking culture
- potential cross-mechanism extensibility

Where they are stronger:

- often easier narrative
- lower burden of proof
- cleaner alignment between model and one material system

What we should borrow:

- local mechanism humility
- narrower subclaims when evidence is still immature

## Where we can already claim advantage

These are the dimensions where we have a defensible edge now, if phrased
carefully.

1. `mechanism-routed partial-observation forecasting`
2. `explicit feasible-state family rather than only direct regression`
3. `audit discipline on direct-Q vs middle-layer route`
4. `prospective lock behavior`

These are not yet universal-field claims, but they are real.

## Where we cannot yet claim advantage

These are dimensions where the current evidence is still too weak.

1. `clean cross-mechanism success`
2. `universal release model`
3. `solved calibrated UQ across mechanisms`
4. `active measurement as a winning audited mainline method`
5. `release-timescale-first superiority`

If we claim these now, we overreach.

## The most valuable borrowing opportunities

This is the part that matters most for strategy.

### Borrow immediately

1. `Few-shot / zero-shot language` from NC
2. `Dataset packaging and kinetic summaries` from accelerated IVR
3. `Data-schema rigor` from AI-ready release-data work
4. `Decision-support framing` from FormulationAI-style platforms

### Borrow later

1. `Physics-aware deep learning` from PINN-type work
2. `Cheap observation modalities` from Raman / soft-sensor lines

## What the field is missing that we can still occupy

The strongest open space is not:

```text
best single-material predictor
```

It is:

```text
shared partially observed release-intelligence framework
```

Concretely, the open space is the combination of:

- benchmark federation,
- mechanism adapter registry,
- common posterior-family object,
- uncertainty discipline,
- prospective evidence behavior,
- future timescale and assay-decision outputs.

No external competitor in this matrix clearly occupies that combined slot yet.

## Implication for our next claims

The right comparative claim is:

> We are not just another release regressor; the project is moving toward a
> shared release-intelligence layer that most current competitors do not even
> attempt.

The wrong comparative claim is:

> We already beat the whole field on every release task.

## Bottom line

At the capability level, the project is already stronger than direct-release
competitors on `object design`, but still weaker on `cross-mechanism proof`
and `simplicity of demonstrated deployment`.

That means the next winning move is not another tiny PLGA metric chase.
It is to complete:

1. one real non-PLGA benchmark,
2. one timescale-first evaluation layer,
3. one shared benchmark API across release mechanisms.

If those land, we stop looking like "a clever PLGA paper" and start looking
like the first serious organizing system for ML-driven drug-release science.
