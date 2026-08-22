# PLGA Observation-Budget Goal Design

Date: 2026-06-11

Purpose: define the next `/goal` so the project moves from a prediction sprint
to a publishable information-budget / experimental-design result.

## Goal Objective

Implement and verify the PLGA observation-budget experiment package that
quantifies how measured early release observations reduce future prediction
error and uncertainty, and whether those observations support practical
early-stopping rules.

Recommended `/goal` objective:

```text
Build scripts/103_plga_observation_value_uncertainty.py and its report package.
Use the committed 91-102 PLGA evidence chain to quantify observation-budget
value, uncertainty contraction, timepoint value, and stopping-rule feasibility
for PLGA release forecasting. Keep the claim limited to information-budgeted
sparse-observation forecasting; do not introduce new foundation-model or
cross-system-transfer claims.
```

## Starting Point

Use the committed evidence chain on branch
`codex/plga-information-budget-evidence-cleanup`, commit `9d6c93b`.

Primary inputs:

| Input | Role |
|---|---|
| `outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv` | direct static / early / static+early per-curve future errors |
| `outputs/97_plga_direct_early_blackbox_control/summary_by_method.csv` | budget-level direct control summary |
| `outputs/99_plga_static_to_early_proxy_probe/per_curve_metrics.csv` | measured-vs-static-predicted early comparison |
| `outputs/99_plga_static_to_early_proxy_probe/proxy_summary.csv` | early proxy quality |
| `outputs/101_lai_author_style_on_our_splits/summary_by_method.csv` | author-style fixed early baseline on our splits |
| `outputs/102_plga_information_budget_claim_table/unified_model_table.csv` | consolidated model table with `budget_kind` and `observation_protocol` |
| `outputs/102_plga_information_budget_claim_table/claim_evidence_table.csv` | current locked claims and remaining risks |

Do not depend on uncommitted `scripts/80_*.py` through `scripts/90_*.py` for
this goal.

## Hard Scope

This goal is about PLGA, not release-corpus transfer.

In scope:

- strict PLGA splits already used by `91`-`102`:
  - `source-group-kfold`
  - `source-dataset-lodo`
  - random-kfold only as an upper-bound / sanity view
- measured early observations vs static-only and static-predicted early proxy
- uncertainty contraction and calibration
- timepoint value and observation budget curves
- stopping-rule simulation
- direct model as primary deployable predictor
- middle-layer and shape-prior routes only as diagnostics / agreement checks

Out of scope:

- new LNN / CNP / KAN model development
- release-corpus cross-system transfer claims
- changing `pyproject.toml` / CUDA dependencies
- claiming a foundation model, world model, or mechanism discovery
- using future heldout release values to choose a deployable policy

## Core Hypotheses

H1. Measured early observations reduce future error under strict splits beyond
the effect of restricting the target window.

H2. Measured early observations reduce future uncertainty interval width while
maintaining acceptable calibration.

H3. Static descriptors cannot currently synthesize the same early state; a
static-predicted early proxy remains worse than measured early observations.

H4. Some early timepoints provide more value than others, enabling a ranked
measurement schedule.

H5. Some curves can stop early once predicted uncertainty is sufficiently small,
with bounded loss in future prediction accuracy.

## Required Experiments

### E1. Window-Matched Observation-Budget Curve

Question: how much error reduction is caused by measured early observations at
equal future target windows?

Budgets:

```text
k = 0, 1, 2, 3, 5
```

Required contrasts:

| Contrast | Interpretation |
|---|---|
| `static_time(k)` vs `static_early_time(k)` | value of measured early Q beyond static descriptors |
| `static_time(k)` vs `early_time(k)` | value of early Q without static descriptors |
| `static_early_time(k)` vs `pred_static_early_time(k)` | measured early state vs static-synthesized early state |

Metrics:

- per-curve RMSE and MAE
- median and mean RMSE by split/budget/method
- paired bootstrap CI for RMSE gains
- number of heldout curves contributing to each contrast

Minimum output:

- `budget_curve.csv`
- `budget_contrasts.csv`
- `budget_contrast_bootstrap_ci.csv`
- `budget_curve.md`

### E2. Uncertainty Contraction

Question: does early Q reduce uncertainty, not just point error?

Preferred first version:

- use residual bootstrap / split-conformal intervals on per-curve future errors,
  grouped by split, method, and budget.
- if ensemble predictions are unavailable, do not pretend to have Bayesian
  uncertainty. Use empirical predictive intervals and state the limitation.

Metrics:

- cov50
- cov90
- median interval width
- mean interval width
- width ratio: `width(k) / width(0)`
- CRPS approximation only if prediction samples are available

Minimum output:

- `uncertainty_by_budget.csv`
- `uncertainty_contraction.csv`
- `calibration_summary.csv`
- `uncertainty_report.md`

Acceptance signal:

```text
Under source-group and source-dataset splits, measured early observations should
show width contraction with no severe cov90 collapse.
```

### E3. Timepoint Value

Question: which early measurements are worth taking?

Candidate policies:

- earliest `k` observed points
- fixed author-style times: `0.25`, `0.5`, `1.0`
- single-point policies: first only, second only, third only, fifth only
- cumulative policies: first 1, first 2, first 3, first 5

Metrics:

- gain per observation
- gain per elapsed day if time metadata permits
- marginal gain from adding the next point
- rank stability across strict splits

Minimum output:

- `timepoint_value.csv`
- `marginal_value.csv`
- `timepoint_rank_by_split.csv`
- `timepoint_value_report.md`

Important caveat:

If current stored predictions do not contain enough alternative single-point
policies, implement this as a clean extension of script `97`, not as a new
model class.

### E4. Stopping-Rule Simulation

Question: can a curve stop early once future prediction is sufficiently certain?

Candidate stopping rules:

| Rule | Stop condition |
|---|---|
| width threshold | predictive interval width below a fixed threshold |
| marginal gain threshold | last additional point gives less than `epsilon` RMSE or width gain |
| stable prediction | median future prediction changes less than threshold after adding a point |
| conservative never-stop baseline | always collect all `k=5` points |

Metrics:

- average number of early points used
- median stopping time
- future RMSE after stop
- cov90 after stop
- saved-observation fraction relative to k=5

Minimum output:

- `stopping_policy_results.csv`
- `stopping_policy_by_split.csv`
- `stopping_tradeoff_table.csv`
- `stopping_rule_report.md`

Acceptance signal:

```text
At least one conservative stopping rule should save observations while staying
within a predeclared RMSE and coverage tolerance relative to the k=5 measured
early baseline.
```

## Required Script

Create:

```text
scripts/103_plga_observation_value_uncertainty.py
```

Default output:

```text
outputs/103_plga_observation_value_uncertainty/
```

Required files:

| File | Purpose |
|---|---|
| `budget_curve.csv` | split/method/budget metric summary |
| `budget_contrasts.csv` | measured early vs static/proxy contrasts |
| `budget_contrast_bootstrap_ci.csv` | paired bootstrap CIs |
| `uncertainty_by_budget.csv` | empirical interval coverage and width |
| `uncertainty_contraction.csv` | width ratios and contraction metrics |
| `timepoint_value.csv` | value of candidate observation schedules |
| `stopping_policy_results.csv` | per-policy stopping tradeoffs |
| `claim_update_table.csv` | which claims are strengthened, weakened, or unchanged |
| `lock_metadata.json` | git hash, args, input file paths, seed, checks |
| `report.md` | concise interpretation and limitations |

Optional figures:

- `budget_curve.png`
- `uncertainty_contraction.png`
- `stopping_tradeoff.png`

## Implementation Constraints

1. Seed all stochastic resampling with `--seed`; default `0`.
2. Do not modify model training scripts unless the current outputs lack a
   required policy table.
3. Do not add dependencies.
4. Do not change `pyproject.toml`.
5. Do not use heldout future observations to choose deployable stopping rules.
   Future errors may only be used for retrospective evaluation.
6. All headline claims must use strict splits:
   - `source-group-kfold`
   - `source-dataset-lodo`
7. Random-kfold may be reported only as an upper-bound / sanity check.
8. Oracle rows from script `96` and `98` must be labeled diagnostic.
9. Static-predicted early rows must remain separated from measured early rows.
10. Author fixed-time early rows must remain separated from earliest-k rows via
    `budget_kind` / `observation_protocol`.

## Statistical Constraints

Use paired comparisons wherever possible.

Minimum bootstrap design:

```text
unit = unified_curve_id
resamples = 2000 by default, configurable by --n-bootstrap
stratify/report by split_kind
compare methods only on shared curve IDs
```

Report:

- point estimate
- 95% CI
- number of paired curves
- fraction of curves improved

Do not claim significance from unpaired aggregate medians.

## Kill Criteria

Stop and report a negative result if any of these happen:

1. measured early observations do not improve strict-split window-matched
   future RMSE relative to static-only after reproducing the `97` contrast.
2. uncertainty intervals shrink only by losing calibration severely.
3. stopping rules save observations only by causing unacceptable RMSE or cov90
   degradation.
4. static-predicted early proxy matches measured early under strict paired
   comparisons, which would weaken the missing-state claim.

Negative-result framing:

```text
PLGA release under the current descriptor set is information-limited, but the
available early observations do not yet support reliable experimental-design
decisions beyond point prediction.
```

## Success Criteria

The goal is complete only when:

- `scripts/103_plga_observation_value_uncertainty.py --help` succeeds.
- `python -m py_compile scripts/103_plga_observation_value_uncertainty.py`
  succeeds.
- the script runs with default inputs and writes all required output files.
- no unexpected NaN/inf appears in required metrics.
- strict-split paired bootstrap tables are present.
- uncertainty contraction and calibration are both reported.
- stopping-rule results include saved-observation and error/coverage tradeoffs.
- `report.md` states what is supported, what failed, and what remains outside
  scope.
- docs are updated to point from `docs/INDEX.md` to this goal result or its
  replacement.

## Recommended `/goal` Prompt

```text
Implement docs/plga_observation_budget_goal_design_2026-06-11.md. Create and
verify scripts/103_plga_observation_value_uncertainty.py plus the required
outputs under outputs/103_plga_observation_value_uncertainty/. Preserve the
scope constraints: PLGA only, no pyproject changes, no 80-90 transfer-chain
claims, no foundation-model framing. Finish by updating docs/INDEX.md and
recording the result in memory.
```

## Expected Paper Conversion

If successful, the paper can say:

```text
For PLGA release forecasting, static formulation descriptors under-identify the
future curve. A small number of measured early release observations acts as an
experimental state probe: it reduces future error, contracts predictive
uncertainty, and supports conservative stopping rules. The contribution is an
information-budget framework for sparse-observation release forecasting, not a
claim that a new model class universally beats black-box predictors.
```
