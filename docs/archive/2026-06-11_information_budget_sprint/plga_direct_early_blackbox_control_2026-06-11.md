# PLGA Direct Early-Q Black-Box Control

Date: 2026-06-11

## Question

Does the early-conditioned middle-layer route beat a matched direct black-box
control when both receive the same early observations?

This is the required control after the script 96 result.

## Matched Design

All methods use:

- the same PLGA curves,
- the same split definitions,
- the same early-observation budgets `k = 0, 1, 2, 3, 5`,
- the same future-only scoring rule,
- and no scoring on observed context points.

Direct controls:

- `direct_static_time`: static descriptors plus query time.
- `direct_early_time`: early observations plus query time.
- `direct_static_early_time`: static descriptors, early observations, and query time.

Middle-layer comparator:

- `middle_early_residual_family`: script 96 deployable early-residual family
  selector.
- `middle_future_oracle_family`: diagnostic upper bound, not deployable.

## Script

`scripts/97_plga_direct_early_blackbox_control.py`

Main output:

`outputs/97_plga_direct_early_blackbox_control/report.md`

## Main Result

| split | best middle RMSE | best direct RMSE | winning direct method | middle future-oracle |
|---|---:|---:|---|---:|
| random-kfold | 0.109 | 0.086 | `direct_static_early_time`, k=5 | 0.088 |
| source-group-kfold | 0.174 | 0.152 | `direct_early_time`, k=5 | 0.142 |
| source-dataset-lodo | 0.187 | 0.152 | `direct_early_time`, k=5 | 0.160 |

## Interpretation

The early-conditioned middle-layer route improves over static middle-layer
mapping, but it does not beat the matched direct early-Q black-box control.

This means:

- Early observations are strongly informative.
- The current deployable middle-layer selector is not yet competitive as a
  predictor.
- The middle route remains useful diagnostically because the gap between
  deployable middle and future-oracle middle shows selector/parameterization
  room.
- But optimization should not start from model capacity yet; the first problem
  is whether the middle object can add information beyond a direct early-Q
  regressor.

## Decision

Do not claim middle-layer superiority from script 96 alone.

Before optimizing the middle model, run a more focused failure decomposition:

```text
direct early-Q error
vs
middle deployable error
vs
middle future-oracle error
by curve, source split, and budget
```

Only optimize the middle model if the decomposition shows a real regime where
the middle oracle beats direct and the deployable selector is the main gap.

