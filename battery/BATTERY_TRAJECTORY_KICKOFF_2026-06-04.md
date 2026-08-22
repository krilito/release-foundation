---
date: 2026-06-04
scope: battery_trajectory_extension
status: kickoff
depends_on:
  - battery/README.md
  - battery/data/README.md
  - battery/scripts/79_battery_intake.py
  - docs/cross_domain_benchmark_spec_2026-05-28.md
  - docs/workspace_domain_boundaries_2026-05-28.md
  - docs/kgso_kinetic_grammar_state_observer_plan_2026-06-04.md
---

# Battery Trajectory Kickoff

This is the first concrete move away from a drug-release-only story.

The target is not battery material screening and not battery SOTA. The target is:

```text
low-data scientific trajectory state identification under partial observation
```

Battery cycling is the right second domain because it has the same mathematical object:

```text
static/protocol descriptors x
early trajectory observations y(t <= T_obs)
latent degradation state theta
future trajectory y(t > T_obs)
measurement cost = more cycles / more diagnostic tests
```

## 1. Boundary

Keep these workspaces separate:

| Workspace | Object |
|---|---|
| `D:\release-foundation` root | drug-release trajectory inference |
| `D:\release-foundation\battery` | battery capacity-fade / cycle-life trajectory inference |
| `D:\battery` | battery material screening and professor-facing material candidates |

Do not copy `D:\battery` electrolyte screening tables into this route.

## 2. Why Battery Works

Drug release and battery cycling both ask:

```text
Given a short early trajectory, what future state family remains feasible?
```

Drug release:

```text
t = release time
y(t) = cumulative release
future = later release curve / t50 / t80
```

Battery:

```text
t = cycle number
y(t) = capacity / normalized capacity / resistance
future = capacity fade trajectory / EOL cycle / knee point
```

The shared method claim should be:

```text
sparse observations contract a feasible kinetic/degradation state family and improve calibrated future trajectory prediction
```

not:

```text
we made a battery model
```

## 3. Dataset Priority

Use public trajectory data first. No wet experiment is required for this route.

### Candidate A: MATR / TRI battery cycle-life data

Role:

- preferred first source if structured cycle summaries are easy to access
- likely best fit for early-cycle -> cycle-life style benchmarks

Why:

- modern battery cycling dataset
- public page describes structured aging cycling data and BEEP processed summaries
- closer to the intended early-cycle prediction task than small toy aging sets

Risk:

- access/API/download path must be verified locally
- license is not necessarily unrestricted for all uses; record it before modeling

### Candidate B: CALCE battery data

Role:

- fallback or second source
- useful for temperature/protocol grouped splits

Why:

- public CALCE pages expose many cycle-life/storage/temperature files
- clear protocol axes such as SOC window, C-rate, and temperature

Risk:

- parsing may be heterogeneous
- some files may require custom column mapping before `79_battery_intake.py`

### Candidate C: NASA PCoE battery data

Role:

- small robust smoke source, not necessarily the main battery benchmark

Why:

- official PCoE repository has battery aging and randomized battery usage datasets
- direct S3 links exist for some classic datasets

Risk:

- some NASA datasets currently have unavailable direct downloads
- classic files may be MATLAB structs rather than clean cycle-summary CSV

## 4. Current Local Entry Point

Existing script:

```text
battery/scripts/79_battery_intake.py
```

It consumes a CSV with one row per `(cell, cycle)` and writes:

```text
outputs/79_battery_intake/intake_summary.json
outputs/79_battery_intake/battery_curves_long.csv
outputs/79_battery_intake/battery_cell_metadata.csv
outputs/79_battery_intake/README.md
```

Template output already created:

```text
outputs/79_battery_intake_template/
```

Template columns:

```text
cell_id
cycle_number
capacity
normalized_capacity
resistance
coulombic_efficiency
temperature_C
protocol_id
```

Metadata columns:

```text
cell_id
dataset
chemistry
protocol_id
charge_rate
discharge_rate
temperature_C
nominal_capacity
source
```

## 5. First Benchmark Object

Minimum battery task:

```text
observe cycles <= N_early
predict cycles > N_early
```

Initial early budgets:

```text
N_early in {20, 50, 100}
```

Targets:

1. future normalized-capacity trajectory
2. end-of-life cycle, e.g. first cycle where normalized capacity <= 0.8
3. knee point only if a robust definition is pre-registered

Metrics:

- future-only trajectory RMSE / MAE
- future-only per-cell R2, but do not over-trust it when future variance is tiny
- EOL absolute error
- bad-tail fraction
- cov50 / cov90
- interval width
- grouped split by protocol / batch / temperature when possible

Non-negotiable rule:

```text
No observed early cycle may be scored as a future prediction target.
```

## 6. Baselines To Beat

Start simple.

### Baseline 0: last observed capacity

Predict the last observed normalized capacity for all future cycles.

Purpose:

- catches fake wins where the future horizon is short or flat

### Baseline 1: per-protocol mean trajectory

Use train cells in the same protocol group; align by cycle; predict group mean.

Purpose:

- tests whether metadata grouping already solves most of the task

### Baseline 2: early linear / exponential fade fit

Fit a tiny degradation curve to the prefix only.

Purpose:

- battery version of the killer prefix-only baseline

### Baseline 3: simple tree or ridge descriptor prior

Map protocol descriptors to EOL / degradation parameters without early observations.

Purpose:

- isolates value of early trajectory observations

### Candidate method: Battery-KGSO v0

Use:

```text
descriptor/protocol prior
prefix-only degradation-state update
monotone capacity-fade decoder
future-only scoring
state-contraction diagnostics
```

Do not use neural CNP/CDE until v0 beats the tiny baselines or explains why it cannot.

## 7. Battery Degradation Grammar

The first decoder does not need electrochemistry.

Start with monotone empirical families:

```text
capacity(t) = q_inf + (q0 - q_inf) * exp(-a * t^b)
capacity(t) = q0 - a * t^b
capacity(t) = q0 - slow_linear*t - knee_amp*sigmoid((t-knee)/width)
```

Constraints:

- normalized capacity should usually be non-increasing after smoothing
- EOL threshold must be defined before evaluation
- fitted decoder is a state representation, not truth

Potential state coordinates:

```text
theta = q0, fade_rate, fade_exponent, q_inf, knee_cycle, knee_width, knee_amp
```

## 8. First 7-Day Execution Plan

### Day 1: source selection

Decide first source by this order:

1. MATR if structured cycle summaries are accessible without manual scraping
2. CALCE if direct tabular downloads are simpler
3. NASA classic battery dataset if `.mat` parsing is faster than web cleanup

Deliverable:

```text
battery/data/source_selection_2026-06-04.md
```

### Day 2: intake adapter

Convert the chosen source into the canonical CSV expected by `79_battery_intake.py`.

Deliverables:

```text
outputs/79_battery_intake/intake_summary.json
outputs/79_battery_intake/battery_curves_long.csv
outputs/79_battery_intake/battery_cell_metadata.csv
```

### Day 3: corpus audit

Measure:

- number of cells
- cycles per cell
- capacity range
- EOL availability
- protocol/batch groups
- missingness
- monotonicity/noise

Deliverable:

```text
outputs/79_battery_intake/README.md
```

expanded with a short readiness verdict.

### Day 4: future-only split guard

Write or reuse an evaluator that explicitly records:

```text
early_mask
future_mask
max_observed_cycle
min_scored_cycle
```

Add an assertion:

```text
min_scored_cycle > max_observed_cycle
```

### Day 5: tiny baselines

Run:

- last-observed hold
- protocol mean trajectory
- prefix-only exponential/power fade

No KGSO claim before these exist.

### Day 6: Battery-KGSO v0

Run only if baseline tables exist.

Outputs:

- prediction trajectories
- EOL predictions
- state-contraction diagnostics
- calibration if uncertainty bands are included

### Day 7: decision note

Write:

```text
docs/battery_smoke_verdict_2026-06-xx.md
```

Decision:

- battery route alive
- battery route useful only as supplement
- battery route blocked by data/interface

## 9. Kill Criteria

Stop or demote battery if:

- no clean cell-cycle trajectory table can be built quickly
- early/future split is ambiguous
- all useful signal comes from protocol labels and not partial observations
- prefix-only tiny baselines beat the observer
- uncertainty bands are uncalibrated and cannot be conformalized
- grouped-by-protocol split collapses completely

## 10. What This Gives Us

If this works, the paper is no longer:

```text
drug release predictor
```

It becomes:

```text
partial-observation state inference for scientific degradation/release trajectories
```

That does not need wet lab as the only proof route.

It needs:

- clean future-only protocols
- public cross-domain data
- calibrated uncertainty
- honest failure modes
- active measurement value

That is a better game.
