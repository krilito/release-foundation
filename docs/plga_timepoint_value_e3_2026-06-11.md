# PLGA E3 Observed Timepoint Value

Date: 2026-06-11

## Purpose

This note records the third PLGA observation-budget supplement:

```text
Which early measured release points are worth taking?
```

Implemented by:

```text
scripts/105_plga_timepoint_value.py
```

Output directory:

```text
outputs/105_plga_timepoint_value/
```

## Scope

This experiment re-runs the script 97 direct black-box control with alternative
observed-context policies. It keeps the same PLGA pool, strict split
assignments, static feature design, future-only scoring, and direct ExtraTrees
model family.

Default strict splits:

```text
source-group-kfold
source-dataset-lodo
```

Main policy families:

| Policy | Meaning |
|---|---|
| `firstK` | use the first K observed release points |
| `pointK` | use only the K-th observed release point |
| `first(K-1)_cutK` | marginal ablation: keep target window after point K but remove point K from context |

The primary contrast is:

```text
direct_static_time vs direct_static_context_time
```

Static baselines are target-window matched. For example, `point5` is compared
against static-only predictions on future points after the fifth observation.

## Headline Result

Static-only vs measured static+context:

| Split | Policy | Baseline RMSE | Measured-context RMSE | RMSE gain |
|---|---|---:|---:|---:|
| source-dataset-lodo | first1 | 0.214 | 0.211 | 0.001 |
| source-dataset-lodo | first2 | 0.231 | 0.183 | 0.037 |
| source-dataset-lodo | first3 | 0.255 | 0.186 | 0.055 |
| source-dataset-lodo | first5 | 0.276 | 0.170 | 0.088 |
| source-dataset-lodo | point1 | 0.214 | 0.211 | 0.001 |
| source-dataset-lodo | point2 | 0.231 | 0.186 | 0.033 |
| source-dataset-lodo | point3 | 0.255 | 0.182 | 0.056 |
| source-dataset-lodo | point5 | 0.276 | 0.171 | 0.087 |
| source-group-kfold | first1 | 0.218 | 0.216 | 0.002 |
| source-group-kfold | first2 | 0.225 | 0.191 | 0.025 |
| source-group-kfold | first3 | 0.239 | 0.187 | 0.041 |
| source-group-kfold | first5 | 0.268 | 0.167 | 0.071 |
| source-group-kfold | point1 | 0.218 | 0.216 | 0.002 |
| source-group-kfold | point2 | 0.225 | 0.191 | 0.029 |
| source-group-kfold | point3 | 0.239 | 0.182 | 0.043 |
| source-group-kfold | point5 | 0.268 | 0.178 | 0.056 |

Interpretation:

```text
The first observed point carries little information.
The second point is the first useful observation.
Later single points, especially point 3 and point 5, carry more absolute future-state information.
The best absolute policy is first5 / point5, but the best gain per elapsed day is around point2.
```

## Ranking

Ranked by median RMSE gain:

| Split | Policy kind | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---|---|---|---|
| source-dataset-lodo | cumulative | first5 | first3 | first2 | first1 |
| source-dataset-lodo | single | point5 | point3 | point2 | point1 |
| source-group-kfold | cumulative | first5 | first3 | first2 | first1 |
| source-group-kfold | single | point5 | point3 | point2 | point1 |

Ranked by gain per elapsed day, point2 is strongest because it is early and
already creates a real information jump.

## Marginal Value

Marginal ablation uses the same target window before and after adding the new
point. Example: `first1_cut2` and `first2` are both scored only after point 2,
so the difference isolates the value of measuring point 2.

| Split | Added point | Median RMSE gain | Improved curves | Window matched |
|---|---:|---:|---:|---|
| source-dataset-lodo | 2 | 0.035 | 0.727 | true |
| source-dataset-lodo | 3 | 0.010 | 0.629 | true |
| source-dataset-lodo | 4 | 0.010 | 0.695 | true |
| source-dataset-lodo | 5 | 0.009 | 0.633 | true |
| source-group-kfold | 2 | 0.026 | 0.675 | true |
| source-group-kfold | 3 | 0.009 | 0.633 | true |
| source-group-kfold | 4 | 0.010 | 0.661 | true |
| source-group-kfold | 5 | 0.013 | 0.629 | true |

The second observation is the largest marginal jump. Points 3 to 5 still help,
but their marginal gains are smaller and similar in magnitude.

## Claim Status

Supported:

- E3 agrees with E1/E2: `k=1` is not enough.
- The first informative measurement is the second observed release point.
- Later observations carry larger absolute state information, especially
  point 3 and point 5.
- Marginal ablation confirms the second point is the largest new information
  jump under identical future windows.

Not claimed:

- Universal calendar-day schedule. These are observed-index policies, not fixed
  day policies.
- Wet-lab stopping rule. E4 must test stopping explicitly.
- Mechanistic hidden-state identity.

## Verification

Commands run:

```powershell
python -m py_compile scripts\105_plga_timepoint_value.py
python scripts\105_plga_timepoint_value.py --out outputs\105_plga_timepoint_value --splits source-group-kfold,source-dataset-lodo --max-index 5 --n-estimators 150
```

`data_checks.csv` reports no unexpected non-finite values, and
`target_times_after_context` reports no future-window violations.
