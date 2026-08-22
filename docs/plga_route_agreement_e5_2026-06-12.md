# PLGA E5 Route Agreement Audit

Date: 2026-06-12

## Purpose

This note closes the PLGA observation-budget supplement chain by asking:

```text
Do the direct, middle-layer, proxy, uncertainty, timepoint, and stopping-rule
routes support the same manuscript story?
```

Implemented by:

```text
scripts/107_plga_route_agreement_audit.py
```

Output directory:

```text
outputs/107_plga_route_agreement_audit/
```

## Scope

E5 does not train a new model. It merges the existing PLGA evidence chain:

| Route | Role |
|---|---|
| script 96 | middle-layer early-conditioned routing |
| script 97 | direct static / early / static+early black-box controls |
| script 99 | static-to-early proxy failure |
| script 101 | author-style fixed early baseline on our splits |
| script 102 | original PLGA information-budget claim table |
| script 103 | E1 point-error observation budget |
| script 104 | E2 empirical uncertainty contraction |
| script 105 | E3 timepoint value |
| script 106 | E4 stopping-rule simulation |

## Output Files

| File | Meaning |
|---|---|
| `claim_route_audit.csv` | manuscript claim, status, strength, evidence, risk |
| `route_agreement_matrix.csv` | which route supports which claim |
| `manuscript_claim_table.md` | compact manuscript-facing claim table |
| `remaining_risks.csv` | remaining limitations by claim |
| `data_checks.csv` | finite-value checks |
| `lock_metadata.json` | inputs, git hash, locked framing |
| `report.md` | generated route-audit report |

## Final Verdict

The PLGA story should be closed as:

```text
an information-budget / experimental-design result
```

not as:

```text
a mechanism-learning victory over black boxes
a foundation model
a Bayesian posterior over PLGA hidden states
```

## Manuscript Claim Set

| Claim ID | Status | Strength | Location | Short claim |
|---|---|---|---|---|
| C1 | supported | strong | Main text | static formulation descriptors under-identify PLGA release |
| C2 | supported | strong | Main text | measured early Q improves future prediction |
| C3 | supported | strong | Main or strong supplement | static descriptors cannot synthesize the same early state |
| C4 | supported | strong | Main text | measured early Q contracts empirical uncertainty |
| C5 | supported | strong | Main figure or supplement | point2 is the first major information jump; point5 is strongest absolute observation |
| C6 | supported | moderate-to-strong | Main or supplement | conservative stopping can save observations with bounded loss |
| C7 | supported with limits | moderate | Discussion / limitations | direct models win prediction; mechanism story must be framed modestly |

## Key Numbers

E1:

```text
k5 static -> measured static+early RMSE gain:
source-group-kfold = 0.107
source-dataset-lodo = 0.106
```

E2:

```text
k5 width90 contraction:
source-group-kfold = 0.353
source-dataset-lodo = 0.259
```

E3:

```text
best single observed point:
source-group-kfold = point5, gain 0.056
source-dataset-lodo = point5, gain 0.087

point2 marginal gain:
source-group-kfold = 0.026
source-dataset-lodo = 0.035
```

E4:

```text
recommended rule = width90_abs_le_0.70
mean saved observations = 3.000
max median RMSE delta vs k5 = 0.014
min cov90 delta vs k5 = -0.022
```

Route limitation:

```text
Direct controls beat matched middle-layer controls on strict splits.
Therefore the deployable claim is sparse-observation forecasting / experimental
design, not mechanism routing superiority.
```

## Writing Rule

Use this framing:

```text
Formulation descriptors under-identify PLGA release. Measured early release
observations expose a missing kinetic state, reduce future prediction error and
empirical uncertainty, identify useful measurement budgets, and enable a
conservative early-stopping simulation.
```

Avoid this framing:

```text
We learned the true PLGA mechanism.
We built a foundation model.
Our mechanism model beats all black boxes.
```

## Next Work

The next useful action is no longer another PLGA model experiment. The next
useful action is:

```text
write the PLGA information-budget manuscript outline and figure/table plan
```

Only after that should we decide whether a prediction-level stable-stopping
rule or fixed-calendar-day timing experiment is worth adding.

## Verification

Commands run:

```powershell
python -m py_compile scripts\107_plga_route_agreement_audit.py
python scripts\107_plga_route_agreement_audit.py
```

`data_checks.csv` reports no unexpected non-finite values.
