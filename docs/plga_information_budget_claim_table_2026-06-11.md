# PLGA Information-Budget Claim Table

Date: 2026-06-11

Script:

`scripts/102_plga_information_budget_claim_table.py`

Output:

`outputs/102_plga_information_budget_claim_table/`

## Purpose

This file locks the current route after the exploratory sprint.

The project should no longer be framed as "we built a better release predictor"
or "we built a foundation model." The evidence now supports a narrower and
stronger claim:

```text
For PLGA release prediction, early observations reveal missing release state
that static formulation descriptors do not currently identify.
```

## Main Evidence

The strict-split, window-matched direct comparison is the cleanest evidence:

| Split | Static after k=5 window | Static + measured early k=5 | Gain |
|---|---:|---:|---:|
| source-group-kfold | 0.267 | 0.160 | 0.107 |
| source-dataset-lodo | 0.272 | 0.166 | 0.106 |

This avoids the target-window mismatch of comparing k=0 against k=5.

The author-style reproduction independently supports the same direction:

| Split | Author zero-shot | Author fixed early | Gain |
|---|---:|---:|---:|
| source-group-kfold | 0.205 | 0.144 | 0.061 |
| source-dataset-lodo | 0.218 | 0.147 | 0.071 |

The static-to-early proxy test shows that the early signal cannot currently be
synthesized from available static descriptors:

| Split | Measured early k=5 | Static-predicted early k=5 | Penalty |
|---|---:|---:|---:|
| source-group-kfold | 0.160 | 0.256 | 0.096 |
| source-dataset-lodo | 0.166 | 0.290 | 0.124 |

## Current Claims

1. Static descriptors alone have a strict-split ceiling.
2. Measured early observations reveal missing release state.
3. Static descriptors cannot currently synthesize that early state.
4. The middle-layer route has diagnostic value but is not the strongest
   deployable predictor yet.
5. The paper should be framed as information-budgeted sparse-observation
   forecasting, not as a foundation-model claim.

## What This Means

The next paper-shaped target is not:

```text
new model beats all black boxes
```

The target is:

```text
what is the value of each early observation, and how can we use the fewest
measurements to obtain enough future-release predictability?
```

## Remaining Missing Piece

The strongest remaining gap is uncertainty / stopping-rule analysis.

To become a full experimental-design paper, the next script should quantify:

- uncertainty contraction as early observations are added,
- which early time points provide the largest value,
- whether some curves/systems can stop early,
- and whether direct, shape-prior, and middle-layer models agree on stopping.

Until that is done, the result is a strong information-bottleneck story but not
yet a complete experimental-design method.

