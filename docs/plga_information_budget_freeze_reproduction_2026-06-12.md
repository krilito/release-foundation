# PLGA Information-Budget Freeze And Reproduction Contract

Date: 2026-06-12

Status: frozen research package for the current PLGA manuscript route.

## One-Sentence Freeze

The current PLGA contribution is an information-budget / experimental-design
result:

```text
Static formulation descriptors under-identify PLGA release. Measured early
release observations reveal missing kinetic state, reduce future prediction
error and empirical uncertainty, identify useful observation budgets, and enable
conservative early stopping.
```

Do not reframe this as:

```text
a foundation model
a true PLGA mechanism discovery
a Bayesian posterior over hidden PLGA states
a mechanism model that beats all black boxes
```

## Frozen Claim Set

Canonical source:

```text
outputs/107_plga_route_agreement_audit/claim_route_audit.csv
```

| Claim ID | Location | Status | Strength | Frozen meaning |
|---|---|---|---|---|
| C1 | Main text | supported | strong | static formulation descriptors under-identify PLGA release |
| C2 | Main text | supported | strong | measured early Q improves future prediction |
| C3 | Main or strong supplement | supported | strong | static descriptors cannot synthesize the same early state |
| C4 | Main text | supported | strong | measured early Q contracts empirical uncertainty |
| C5 | Main figure or supplement | supported | strong | point2 is the first major information jump; point5 is strongest absolute observation |
| C6 | Main or supplement | supported | moderate-to-strong | conservative stopping is feasible in simulation |
| C7 | Discussion / limitations | supported with limits | moderate | direct black-box wins prediction; mechanism/middle-layer story must stay modest |

## Key Numbers To Reuse

### E1. Window-Matched Observation Value

Source:

```text
outputs/103_plga_observation_value_uncertainty/budget_contrasts.csv
```

Strict split k5 static -> measured static+early RMSE gains:

| Split | Static RMSE | Measured static+early RMSE | Gain |
|---|---:|---:|---:|
| source-group-kfold | 0.267 | 0.160 | 0.107 |
| source-dataset-lodo | 0.272 | 0.166 | 0.106 |

Interpretation:

```text
Measured early Q adds information beyond static descriptors under equal future
target windows.
```

### E2. Empirical Uncertainty Contraction

Source:

```text
outputs/104_plga_empirical_uncertainty_contraction/uncertainty_contraction.csv
```

Window-matched k5 width90 contraction:

| Split | Width90 contraction | cov90 after measured early |
|---|---:|---:|
| source-group-kfold | 0.353 | 0.856 |
| source-dataset-lodo | 0.259 | 0.928 |

Interpretation:

```text
Early Q reduces empirical future-error width. This is an empirical
fold-jackknife error envelope, not Bayesian uncertainty.
```

### E3. Timepoint Value

Sources:

```text
outputs/105_plga_timepoint_value/timepoint_rank_by_split.csv
outputs/105_plga_timepoint_value/marginal_value.csv
```

Best single observed point by median RMSE gain:

| Split | Best single point | Gain |
|---|---|---:|
| source-group-kfold | point5 | 0.056 |
| source-dataset-lodo | point5 | 0.087 |

First major marginal jump:

| Split | Added point | Marginal gain |
|---|---:|---:|
| source-group-kfold | 2 | 0.026 |
| source-dataset-lodo | 2 | 0.035 |

Interpretation:

```text
The first observed point is weak. The second point is the first major
information jump. Later points, especially point5, carry larger absolute future
state information.
```

### E4. Conservative Stopping

Source:

```text
outputs/106_plga_stopping_rule_simulation/stopping_tradeoff_table.csv
```

Recommended primary rule:

```text
width90_abs_le_0.70
```

Meaning:

```text
Stop when other-fold calibrated empirical width90 <= 0.70.
```

Result:

| Metric | Value |
|---|---:|
| mean saved observations vs k5 | 3.000 |
| max median RMSE delta vs k5 | 0.014 |
| min cov90 delta vs k5 | -0.022 |
| mean stop k | 2.000 |

Interpretation:

```text
A conservative calibrated rule can save about three early observations while
staying inside the predeclared error and coverage tolerance.
```

## Reproduction Anchors

Current commit chain:

```text
c1c8637 Add PLGA route agreement E5 audit
fe1c5bf Add PLGA stopping-rule E4 simulation
a2111a8 Add PLGA timepoint value E3
e7c51db Add PLGA empirical uncertainty contraction E2
1d30265 Add PLGA observation-budget E1 curve
d9b8be8 Archive transfer probes and define PLGA observation goal
9d6c93b Consolidate PLGA information-budget evidence chain
```

Primary scripts:

| Script | Role |
|---|---|
| `scripts/102_plga_information_budget_claim_table.py` | pre-E1 claim table from 91-101 evidence |
| `scripts/103_plga_observation_value_uncertainty.py` | E1 point-error observation budget |
| `scripts/104_plga_empirical_uncertainty_contraction.py` | E2 empirical uncertainty contraction |
| `scripts/105_plga_timepoint_value.py` | E3 timepoint value and marginal observation value |
| `scripts/106_plga_stopping_rule_simulation.py` | E4 stopping-rule simulation |
| `scripts/107_plga_route_agreement_audit.py` | E5 route agreement and manuscript claim table |

Primary outputs:

| Output directory | Role |
|---|---|
| `outputs/102_plga_information_budget_claim_table/` | baseline claim table |
| `outputs/103_plga_observation_value_uncertainty/` | E1 |
| `outputs/104_plga_empirical_uncertainty_contraction/` | E2 |
| `outputs/105_plga_timepoint_value/` | E3 |
| `outputs/106_plga_stopping_rule_simulation/` | E4 |
| `outputs/107_plga_route_agreement_audit/` | E5 and final claim audit |

## Locked Reproduction Commands

These commands reproduce the frozen E1-E5 package from the already generated
upstream PLGA evidence outputs (`91`, `96`, `97`, `99`, `101`, `102`):

```powershell
python scripts\103_plga_observation_value_uncertainty.py
python scripts\104_plga_empirical_uncertainty_contraction.py
python scripts\105_plga_timepoint_value.py --out outputs\105_plga_timepoint_value --splits source-group-kfold,source-dataset-lodo --max-index 5 --n-estimators 150
python scripts\106_plga_stopping_rule_simulation.py
python scripts\107_plga_route_agreement_audit.py
```

Validation commands:

```powershell
python -m py_compile scripts\103_plga_observation_value_uncertainty.py
python -m py_compile scripts\104_plga_empirical_uncertainty_contraction.py
python -m py_compile scripts\105_plga_timepoint_value.py
python -m py_compile scripts\106_plga_stopping_rule_simulation.py
python -m py_compile scripts\107_plga_route_agreement_audit.py
```

Expected no-error check files:

```text
outputs/103_plga_observation_value_uncertainty/data_checks.csv
outputs/104_plga_empirical_uncertainty_contraction/data_checks.csv
outputs/105_plga_timepoint_value/data_checks.csv
outputs/106_plga_stopping_rule_simulation/data_checks.csv
outputs/107_plga_route_agreement_audit/data_checks.csv
```

Each should report:

```text
unexpected_nonfinite = {}
```

Allowed non-finite values are limited to known R2/context-time fields.

## Upstream Reproduction Boundary

The frozen E1-E5 package assumes the following upstream outputs already exist:

```text
outputs/91_plga_static_feature_ceiling_*
outputs/96_plga_early_middle_layer_selector_probe/
outputs/97_plga_direct_early_blackbox_control/
outputs/99_plga_static_to_early_proxy_probe/
outputs/101_lai_author_style_on_our_splits/
outputs/102_plga_information_budget_claim_table/
```

If those are regenerated, rerun `103` through `107` before changing any
manuscript number.

## What Is Valuable

The valuable research result is not that one model is universally better.

The valuable result is this structure:

```text
descriptor ceiling -> early observation value -> uncertainty contraction ->
timepoint prioritization -> conservative stopping -> route agreement audit
```

This gives a defensible PLGA paper spine:

1. PLGA static descriptors are insufficient under strict source shift.
2. Early measured Q acts as a probe of missing kinetic state.
3. The value is measurable under equal future windows.
4. Static descriptors cannot currently synthesize the same early state.
5. Early observations reduce empirical uncertainty, not just point error.
6. The second observed point is the first major information jump.
7. Conservative calibrated stopping can reduce measurement burden.

## What Must Not Be Claimed

Do not claim:

- foundation model,
- universal drug-release model,
- true PLGA mechanism identified from Q(t),
- Bayesian posterior uncertainty,
- mechanism/middle-layer model beats direct black-box controls,
- prospective wet-lab stopping protocol already validated.

Correct wording:

```text
retrospective PLGA information-budget evidence supporting sparse-observation
forecasting and conservative experimental-design rules
```

## Next Work After Freeze

The next work item is:

```text
PLGA information-budget manuscript outline and figure/table plan
```

Do not restart broad model exploration before that scaffold exists.
