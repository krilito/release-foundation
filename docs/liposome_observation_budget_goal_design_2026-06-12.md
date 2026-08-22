# Liposome Observation-Budget Goal Design

Date: 2026-06-12

Purpose: define the next `/goal` so follow-up agents test a second drug-release
system with the same discipline used for the frozen PLGA information-budget
package.

## Goal Objective

Test whether the PLGA information-budget logic transfers to a more complete
non-PLGA release system, using liposome in vitro release as the second
retrospective validation case.

Recommended `/goal` objective:

```text
Build and verify a staged liposome release-forecasting evidence package. First
test whether static formulation and assay descriptors can predict whole release
curves. If static prediction fails under strict splits, diagnose the bottleneck
and then test whether measured early release observations improve future
prediction and uncertainty contraction. Keep the claim limited to
system-specific information-budgeted forecasting; do not claim a foundation
model, learned mechanism, or PLGA-to-liposome mechanistic transfer.
```

## Starting Point

Use the frozen PLGA result as the template, not as a pretrained mechanism.

Primary PLGA reference:

```text
docs/plga_information_budget_freeze_reproduction_2026-06-12.md
```

Primary liposome data pool:

```text
outputs/149_release_caveat_augmented_cumulative_v1/
```

Fallback clean pool:

```text
outputs/148_release_main_cumulative_v1/
```

Current local audit expectation:

```text
Liposome has about 209 curves, every curve has at least 6 timepoints, and key
fields such as API name, structure type, drug molecular weight, pH, temperature,
particle size, PDI, and zeta potential are partially available. Source split is
likely weak because the corpus may be single-source; API and structure splits
are the important strict tests.
```

## Hard Scope

In scope:

- liposome release curves only
- whole-curve prediction from static descriptors
- strict split feasibility by API, structure type, pH, temperature, and source
- bottleneck decomposition after static prediction fails
- early-observation budgets `k = 0, 1, 2, 3, 5`
- empirical uncertainty contraction and conservative stopping only after point
  prediction passes data-integrity checks
- comparison to PLGA at the level of information structure, not mechanism

Out of scope:

- HA/PVP prospective hydrogel claims
- new LNN, KAN, MoE, PINN, symbolic-regression, or neural-ODE development before
  the static and early-observation baselines are exhausted
- cross-system pretraining claims
- claiming liposome validates PLGA physics
- changing dependencies, CUDA setup, or `pyproject.toml`
- using future heldout release values to select a deployable stopping policy

## Research Logic

The sequence is deliberately asymmetric:

```text
1. Try the simple direct thing first.
2. If it works, the system may be statically identifiable.
3. If it fails, diagnose why.
4. Only then test whether early release observations reveal the missing state.
```

This avoids the main failure mode: starting with a complex model and mistaking
architecture behavior for a scientific result.

## Core Hypotheses

H0. Liposome curves may be more statically identifiable than PLGA because
descriptor fields such as API, structure type, particle size, pH, temperature,
and assay condition may carry stronger release-state information.

H1. If static-only prediction fails under strict API or structure splits, the
failure should be decomposable into descriptor incompleteness, split/source
shift, shape-family insufficiency, or model-class limitation.

H2. If measured early release points improve future prediction after static
failure, liposome supports the broader claim that drug-release prediction is an
observation-budget problem, not just a static formulation regression problem.

H3. If early observations do not help, the project should not force the PLGA
story onto liposome. The correct conclusion is that this system has a different
information bottleneck or the current corpus is not suitable.

## Required Experiments

### E0. Candidate Data Audit

Question: is liposome suitable as the second retrospective system?

Required outputs:

| File | Purpose |
|---|---|
| `dataset_completeness.csv` | curve count, timepoint count, time range, valid release fraction summary |
| `descriptor_completeness.csv` | missingness and unique-count audit for static descriptors |
| `split_candidate_summary.csv` | feasibility of API, structure, pH, temperature, and source splits |
| `timepoint_coverage.csv` | available early-point budgets per curve |
| `candidate_decision_table.csv` | proceed / caution / reject decision with reasons |
| `report.md` | concise interpretation and next experiment recommendation |
| `lock_metadata.json` | git hash, CLI args, input paths, seed, checks |

Acceptance gate:

```text
Proceed only if there are enough curves, enough valid timepoints, and at least
one strict split axis with meaningful heldout groups. If this fails, stop and
choose another system instead of forcing liposome.
```

### E1. Static Whole-Curve Prediction

Question: can static formulation and assay descriptors predict liposome release
curves without early observations?

Features to test:

| Feature group | Examples |
|---|---|
| drug/API | `API_name`, `API_ID`, `Drug_Mw` |
| carrier/structure | `structure_type`, lipid/carrier descriptors if available |
| colloid | `Particle_Size`, `PDI`, `zeta_potential` |
| medium/assay | `media_pH`, `media_temp_oC`, release method / assay fields if available |

Splits:

- grouped cross-validation by API
- grouped cross-validation by structure type if group sizes allow
- pH or temperature group split as sensitivity
- source split only if more than one source is present

Baselines:

- global median curve
- API-local median curve where allowed by split
- shape-family prior, such as Weibull / Hill / biexponential, fit only on the
  training fold
- direct static descriptor model

Required outputs:

| File | Purpose |
|---|---|
| `static_summary_by_split.csv` | RMSE, MAE, R2, median per-curve RMSE |
| `static_per_curve_metrics.csv` | curve-level errors and metadata |
| `static_feature_set_ablation.csv` | feature-group contribution |
| `static_baseline_contrasts.csv` | paired gains vs baselines |
| `static_prediction_report.md` | decision: static success, partial success, or failure |

Acceptance gate:

```text
If static-only prediction is strong under strict API/structure splits, do not
pretend early observations are necessary. Interpret liposome as a contrast case
against PLGA.
```

### E2. Bottleneck Decomposition

Question: if static prediction fails, what exactly failed?

Required decomposition:

| Bottleneck | Diagnostic |
|---|---|
| descriptor insufficiency | missingness and feature-set ablation cannot explain curve identity |
| API shift | performance collapses on unseen API groups |
| structure shift | performance collapses on unseen LUV/SUV/MLV-like groups |
| assay/source shift | pH, temperature, method, or source groups dominate residuals |
| shape-family insufficiency | shape priors fail even with train-fold fitting |
| model limitation | simple models lose but stronger static models improve without leakage |

Required outputs:

| File | Purpose |
|---|---|
| `bottleneck_by_group.csv` | residuals grouped by API, structure, pH, temperature |
| `bottleneck_feature_missingness.csv` | relationship between missing descriptors and errors |
| `shape_family_failure_table.csv` | shape-prior errors by split |
| `bottleneck_decision_table.csv` | reasoned next step: early observations, better descriptors, or stop |

Acceptance gate:

```text
Early-observation experiments are justified only if static failure remains after
reasonable baselines and leakage-safe splits.
```

### E3. Early-Observation Budget

Question: do measured early liposome release observations reduce future error?

Budgets:

```text
k = 0, 1, 2, 3, 5
```

Required contrasts:

| Contrast | Interpretation |
|---|---|
| static-only vs static+measured-early | value of observed early Q beyond descriptors |
| early-only vs static-only | how much release-state information is in early Q itself |
| static+predicted-early vs static+measured-early | whether static descriptors can synthesize the same early state |

Required outputs:

| File | Purpose |
|---|---|
| `budget_curve.csv` | metric summary by split, method, and budget |
| `budget_contrasts.csv` | early-observation gains |
| `budget_contrast_bootstrap_ci.csv` | paired bootstrap CIs |
| `per_curve_metrics.csv` | curve-level predictions and errors |
| `early_budget_report.md` | interpretation and PLGA comparison |

Acceptance signal:

```text
Measured early Q should improve strict-split future prediction beyond static
descriptors. If gains appear only under random splits, treat them as leakage or
source-memory evidence, not transfer.
```

### E4. Uncertainty And Stopping

Question: if early observations help, do they reduce empirical uncertainty and
support lower-cost IVR schedules?

Required outputs:

| File | Purpose |
|---|---|
| `uncertainty_by_budget.csv` | cov50, cov90, width50, width90 |
| `uncertainty_contraction.csv` | width ratios vs `k=0` |
| `timepoint_value.csv` | single-point and cumulative early-point values |
| `stopping_tradeoff_table.csv` | saved observations vs future error and coverage |
| `uncertainty_stopping_report.md` | final decision |

Acceptance gate:

```text
Only claim stopping-rule value if a predeclared conservative rule saves
observations while staying within bounded RMSE and coverage tolerance relative
to the `k=5` measured-early baseline.
```

## First Script To Build

Create:

```text
scripts/108_liposome_candidate_audit.py
```

Default output:

```text
outputs/108_liposome_candidate_audit/
```

Required CLI:

```text
--pool outputs/149_release_caveat_augmented_cumulative_v1
--out outputs/108_liposome_candidate_audit
--seed 0
--min-timepoints 6
```

Required checks:

- every analyzed curve has at least `--min-timepoints` valid timepoints
- raw `release_fraction` is preserved; clipping is allowed only for model input
- no nonfinite `time_days` or `release_fraction` enters metrics
- split summaries report train/heldout curve counts for every candidate group
- `lock_metadata.json` records git hash, command args, input files, row counts,
  and all data-integrity warnings

## Verification Discipline

Every experiment must end with a verification block:

```text
Data integrity:
- row counts match expected filtered pool
- no heldout group appears in training fold
- no nonfinite metrics
- each curve has target points strictly after context points

Reproducibility:
- seed recorded
- git hash recorded
- CLI args recorded
- input file hashes or modified times recorded

Interpretation:
- claim is one of: supported, weakened, rejected, or not testable
- next action is explicitly bounded
```

## Anti-Drift Rules

Do not let agents drift into:

- training neural models before E0/E1 is complete
- treating liposome as proof of PLGA mechanism
- using random splits as headline evidence
- comparing methods without equal target windows
- optimizing features on heldout groups
- adding dependencies for convenience
- writing a success narrative before the failure modes are measured

## Decision Tree

```text
E0 fails
  -> stop liposome; choose a more complete system.

E0 passes and E1 static succeeds under strict split
  -> liposome is a contrast case: static descriptors are comparatively
     sufficient. Write contrast against PLGA.

E0 passes and E1 static fails, E2 finds clear descriptor/source bottleneck
  -> report bottleneck; improve descriptors or split definition before models.

E0 passes and E1 static fails, E2 does not explain it, E3 early observations help
  -> extend PLGA information-budget story to liposome.

E3 early observations do not help
  -> negative result; do not force transfer. Liposome corpus may be unsuitable
     or governed by unobserved assay variables.
```

## Expected Final Claim Forms

Allowed strong claim:

```text
Across PLGA and liposome retrospective corpora, release prediction quality is
limited by what is observed: static descriptors can be insufficient, and early
release measurements can reveal curve-specific state needed for future
forecasting.
```

Allowed contrast claim:

```text
PLGA and liposome differ in their information bottlenecks. PLGA requires early
release observations to expose missing kinetic state, while liposome may be more
or less statically identifiable depending on strict API and structure splits.
```

Forbidden claim:

```text
The PLGA model transfers mechanistically to liposome.
```

