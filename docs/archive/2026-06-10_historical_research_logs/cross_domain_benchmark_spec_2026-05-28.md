# Cross-domain benchmark specification

Date: 2026-05-28
Author: Codex
Purpose: make the Nature-level strategy executable by defining the minimum dataset interface, benchmark protocol, and first execution order across drug release and non-release scientific dynamics.

## Why this file exists

The strategy document says the project should become:

> calibrated posterior-family inference and active observation for partially observed scientific dynamical systems.

This file defines what a dataset must look like before it can support that claim.

If a dataset cannot be expressed through this interface, it can still be useful, but it does not belong in the main Nature-level benchmark.

## Canonical task template

Every domain must be converted into this form:

```text
entity_id
static descriptors x
time grid t
trajectory y(t)
early observation mask M_early
future evaluation mask M_future
optional latent or fitted kinetic state theta
optional mechanism / condition group labels
measurement cost c(t, measurement_type)
```

The model sees:

```text
x + y(t) for t in M_early
```

It must output:

```text
predictive distribution p(y_future | x, y_early)
posterior family over latent state theta or theta-like kinetic coordinates
recommended next measurement(s)
```

The score is not only RMSE. It must include:

1. point error,
2. cov90 / cov50,
3. interval width,
4. calibration-vs-sharpness tradeoff,
5. benefit of active measurement under identical information budget,
6. grouped OOD performance when metadata permits.

## Non-negotiable leakage rule

No observation timepoint may also be scored as a future prediction target.

Allowed protocols:

1. **Early-fixed protocol**
   - observations: `t <= T_obs`
   - evaluation: `t > T_obs`

2. **Adaptive-with-exclusion protocol**
   - active policy may select arbitrary `t`
   - selected `t` is removed from the evaluation set for that curve
   - report the remaining evaluation grid per curve

3. **Prediction-after-last-observation protocol**
   - if active policy may select day 21, evaluation must begin after day 21

For current release work, use protocol 1 first. It is easiest to defend.

## Dataset readiness tiers

### Tier 0: reject for this paper

Reject from main benchmark if any are true:

- no trajectory,
- no entity-level grouping,
- no long-horizon target,
- target is a one-off table row rather than a time-evolving state,
- measurement order is not meaningful,
- licensing unclear.

### Tier 1: intake only

Dataset has promise but needs auditing:

- units unclear,
- duplicated IDs,
- missing descriptors,
- sparse or irregular time grids,
- raw curves need digitization,
- fitted parameters exist but raw trajectory alignment is uncertain.

### Tier 2: retrospective benchmark

Dataset can be used for retrospective model comparison:

- entity IDs stable,
- descriptors available,
- trajectory table available,
- early/future split definable,
- at least one grouped split definable or a strong reason why not.

### Tier 3: main paper evidence

Dataset supports a main figure:

- Tier 2 plus,
- leakage audit passed,
- bootstrap CIs available,
- calibration and interval width reported,
- comparison includes point predictor, calibrated posterior-family method, and active/fixed observation budgets.

### Tier 4: prospective evidence

Dataset or experiment supports prospective validation:

- predictions locked before measurement,
- code and inputs hashed,
- primary endpoint pre-registered,
- failure thresholds pre-specified,
- no post-hoc metric switching.

Chitosan batch 1 is currently the only Tier 4 asset.

## Current local asset audit

### Chitosan prospective

Local files:

```text
outputs/67_chitosan_prospective/predictions.csv
outputs/67_chitosan_prospective/lock_metadata.json
outputs/67_chitosan_prospective/theta_targets.npz
outputs/67_chitosan_prospective/HASHES.txt
docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md
```

Evidence:

- Git tag exists: `prospective-chitosan-batch1-2026-05-28`
- Pre-registration exists.
- HASHES.txt exists.

Benchmark tier:

```text
Tier 4 prospective evidence, pending wet-lab outcome.
```

Use:

Main validation figure if wet-lab data arrives and is analyzed exactly according to the registration.

### Liposome accelerated IVR

Local files:

```text
data/external/accelerated_IVR/repo/accelerated_IVR-main/data/clean/ML_7_features_df.csv
data/external/accelerated_IVR/repo/accelerated_IVR-main/data/clean/ML_9_features_df.csv
data/external/accelerated_IVR/repo/accelerated_IVR-main/data/clean/weibull_params.csv
data/external/accelerated_IVR/repo/accelerated_IVR-main/results/fitting/drug_release_exp.csv
data/external/accelerated_IVR/repo/accelerated_IVR-main/data/unprocessed/backend_data.csv
```

Observed table shapes:

```text
ML_7_features_df.csv            77 x 8
ML_9_features_df.csv            78 x 10
weibull_params.csv             169 x 4
drug_release_exp.csv          3332 x 3
backend_data.csv               271 x 15
```

Important columns:

```text
backend_data: formulation_ID, IVR_ID, release_method, media_pH,
              media_temp_oC, drug_loading, structure_type, Z_average_nm,
              PDI, zeta_potential, API_ID, API_name, weighted_Mw, weighted_Tm

drug_release_exp: time (Hrs), release_percent, file_name

weibull_params: ID, alpha, beta, Time_units
```

Benchmark tier:

```text
Tier 1 -> Tier 2 after ID alignment audit.
```

Main blocker:

`drug_release_exp.csv` uses `file_name` like `data/drug_release/3..csv`; this must be mapped cleanly to `IVR_ID` / `ID` before modeling.

Use:

First non-PLGA retrospective release-mechanism benchmark.

### Bannigan / LAI 181

Local files:

```text
data/external/bannigan_lai_181/Dataset_14_feat.tsv
data/external/bannigan_lai_181/Dataset_17_feat.tsv
```

Observed issue:

The `.tsv` files have an Excel/zip file header (`PK`), so they should be read as Excel files or renamed internally for clarity. Do not parse them as UTF-8 TSV.

Benchmark tier:

```text
Tier 2 if already represented by cleaned local data;
Tier 1 if re-ingesting from these external files.
```

Use:

Second PLGA / LAI benchmark; not the main novelty but useful for stability.

### PLGA local core

Local files:

```text
data/formulations.csv
data/curves_long.csv
data/theta_bank.csv
data/Dataset_17_feat_augmented.csv
```

Benchmark tier:

```text
Tier 2 retrospective benchmark, but active/adaptive claims currently require leakage-fixed reruns.
```

Use:

Main development benchmark and ablation surface.

## First non-release target: battery degradation

The battery route should be treated as the first real test of whether the algorithm escaped drug release.

### Candidate sources

- MATR battery cycle-life data: https://data.matr.io/1/
- NASA Prognostics Center of Excellence battery datasets: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- CALCE battery datasets: https://calce.umd.edu/battery-data

### Required adapter

Convert battery data into:

```text
entity_id = cell_id
x = protocol / temperature / charge-discharge descriptors
t = cycle number
y(t) = capacity or normalized capacity
theta = optional fitted degradation coordinates
M_early = cycles <= N_early
M_future = cycles > N_early
group labels = protocol / batch / temperature
measurement cost = additional cycles or expensive diagnostics
```

### Minimum smoke benchmark

Use three early budgets:

```text
N_early in {20, 50, 100}
```

Predict:

```text
capacity trajectory after N_early
end-of-life cycle
knee point if robustly definable
```

Report:

```text
RMSE / MAE for trajectory
EOL absolute error
cov90 and interval width
random split and grouped-by-protocol split if metadata supports it
active next-cycle / next-diagnostic utility smoke test
```

### What would count as success

Not "beating all battery papers."

Success for this project is narrower:

1. same posterior-family object works without domain-specific hacks,
2. uncertainty is calibrated after conformal adjustment,
3. early-budget curves behave sensibly,
4. grouped split does not catastrophically collapse,
5. active measurement utility chooses interpretable information-rich regions.

If those pass, battery can become Figure 4 in the cross-domain paper.

## Shared benchmark outputs

Every domain-specific script should write the same file set:

```text
outputs/<script_name>/
  intake_summary.json
  split_metadata.json
  metrics_summary.csv
  metrics_by_entity.csv
  prediction_trajectories.csv
  calibration_summary.csv
  active_measurement_log.csv
  README.md
```

Required `intake_summary.json` fields:

```json
{
  "dataset": "",
  "domain": "",
  "n_entities": 0,
  "n_measurements": 0,
  "time_unit": "",
  "target_unit": "",
  "descriptor_columns": [],
  "trajectory_columns": [],
  "id_columns": [],
  "group_columns": [],
  "early_windows": [],
  "future_windows": [],
  "known_leakage_risks": [],
  "benchmark_tier": "",
  "recommended_role": ""
}
```

## First implementation order

### Step 1: fix current release evidence

Do this before adding new claims:

1. repair adaptive leakage in `72_canonical_benchmark_v2.py`,
2. repair adaptive leakage in `77b_regime_benefit_loocv_v2.py`,
3. repair dataset-aware regime label mapping in `74b_regime_verification_v2.py`,
4. rerun PLGA benchmark with paired bootstrap CI.

### Step 2: promote liposome to Tier 2

Create a liposome intake script:

```text
scripts/78_liposome_ivr_intake.py
```

It should:

1. map `file_name` to numeric `ID`,
2. join with `weibull_params.csv`,
3. join with `backend_data.csv` by `IVR_ID` / `ID`,
4. report curve length and time range,
5. write `outputs/78_liposome_ivr_intake/intake_summary.json`,
6. export aligned `liposome_curves_long.csv` and `liposome_formulations.csv` into output, not into `data/`.

### Step 3: create battery intake

Create:

```text
scripts/79_battery_intake.py
```

Initially it may support one source only. Prefer MATR if easy, NASA/CALCE if access is simpler.

Do not overbuild. The first goal is to prove the interface can represent a non-release system.

### Step 4: common evaluator

Only after liposome and battery intake pass, create:

```text
scripts/80_partial_observation_benchmark.py
```

It should consume an already-aligned dataset and run the shared benchmark protocol.

## Decision gate

After Steps 1-3, decide:

| Evidence | Paper route |
|---|---|
| PLGA clean + chitosan prospective + liposome clean + battery smoke | Cross-domain Nature Methods / NMI route alive |
| PLGA clean + chitosan prospective + liposome clean, battery weak | Strong drug-release method paper |
| PLGA adaptive claim collapses, chitosan passes | Prospective calibrated transfer paper |
| Chitosan fails calibration | Honest limitation paper plus redesign batch 2 |

## Bottom line

The generalization route is plausible, but only if it is operationalized as a shared benchmark interface.

The next real proof is not another name for the algorithm. It is:

```text
Can liposome and battery data enter the same partial-observation benchmark
without rewriting the scientific claim?
```

If yes, the project has a Nature-level spine.
If no, it remains a high-quality drug-release project.
