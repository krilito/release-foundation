# PLGA E1 Observation-Budget Curve

Date: 2026-06-11

Script:

`scripts/103_plga_observation_value_uncertainty.py`

Output:

`outputs/103_plga_observation_value_uncertainty/`

## Purpose

This is the first experiment from
`docs/plga_observation_budget_goal_design_2026-06-11.md`.

It locks the future prediction window and asks:

```text
At the same future target window, how much predictive value is added by
measured early release observations?
```

## Inputs

| Input | Role |
|---|---|
| `outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv` | measured early vs static direct-control errors |
| `outputs/99_plga_static_to_early_proxy_probe/per_curve_metrics.csv` | measured early vs static-predicted early proxy errors |

## Required Outputs

| Output | Status |
|---|---|
| `budget_curve.csv` | written |
| `budget_contrasts.csv` | written |
| `budget_contrast_bootstrap_ci.csv` | written |
| `budget_curve.md` | written |
| `paired_curve_deltas.csv` | written as audit detail |
| `data_checks.csv` | written |
| `lock_metadata.json` | written |

## Headline Result

Measured early observations become strongly valuable once the context contains
at least two early points. A single early point does not reliably improve the
strict-split future prediction.

### Static vs Measured Static+Early

Positive gain means measured early observations improve future RMSE over
static-only at the same future target window.

| Split | k | Static RMSE | Static+Measured Early RMSE | Gain | 95% CI |
|---|---:|---:|---:|---:|---|
| source-group-kfold | 1 | 0.217 | 0.218 | -0.000 | [-0.004, 0.004] |
| source-group-kfold | 2 | 0.228 | 0.191 | 0.037 | [0.025, 0.051] |
| source-group-kfold | 3 | 0.237 | 0.188 | 0.049 | [0.035, 0.066] |
| source-group-kfold | 5 | 0.267 | 0.160 | 0.107 | [0.086, 0.128] |
| source-dataset-lodo | 1 | 0.214 | 0.216 | -0.003 | [-0.007, 0.003] |
| source-dataset-lodo | 2 | 0.232 | 0.184 | 0.049 | [0.036, 0.058] |
| source-dataset-lodo | 3 | 0.254 | 0.188 | 0.067 | [0.051, 0.083] |
| source-dataset-lodo | 5 | 0.272 | 0.166 | 0.106 | [0.085, 0.128] |

### Static-Predicted Early vs Measured Early

Positive gain means measured early observations beat static-predicted early
proxy features.

| Split | k | Static-Predicted Early RMSE | Measured Early RMSE | Gain | 95% CI |
|---|---:|---:|---:|---:|---|
| source-group-kfold | 1 | 0.220 | 0.220 | 0.000 | [-0.000, 0.000] |
| source-group-kfold | 2 | 0.217 | 0.192 | 0.025 | [0.013, 0.036] |
| source-group-kfold | 3 | 0.224 | 0.185 | 0.039 | [0.024, 0.054] |
| source-group-kfold | 5 | 0.256 | 0.165 | 0.091 | [0.075, 0.111] |
| source-dataset-lodo | 1 | 0.216 | 0.216 | 0.000 | [-0.000, 0.000] |
| source-dataset-lodo | 2 | 0.237 | 0.186 | 0.051 | [0.043, 0.064] |
| source-dataset-lodo | 3 | 0.253 | 0.186 | 0.068 | [0.053, 0.082] |
| source-dataset-lodo | 5 | 0.290 | 0.163 | 0.127 | [0.112, 0.140] |

## Interpretation

The result strengthens the information-budget framing:

1. The key gain is not merely from excluding early targets; the static-only
   model is evaluated on the same future window and still loses once k >= 2.
2. k=1 is too weak to identify the missing release state.
3. k=2 and k=3 already show measurable state information.
4. k=5 is the strongest current budget and reproduces the prior claim-table
   gains.
5. Static-predicted early features do not replace measured early observations,
   especially under strict splits and larger budgets.

## What This Does Not Yet Prove

This is still only E1.

It does not yet prove:

- uncertainty contraction,
- optimal early timepoint choice,
- stopping-rule feasibility,
- or a mechanism model advantage over direct early-Q prediction.

Those remain the next experiments in
`docs/plga_observation_budget_goal_design_2026-06-11.md`.

## Verification

Executed:

```powershell
python -m py_compile scripts\103_plga_observation_value_uncertainty.py
python scripts\103_plga_observation_value_uncertainty.py
```

`data_checks.csv` reports no unexpected non-finite values. Allowed missingness
is confined to optional R2/context-time fields.
