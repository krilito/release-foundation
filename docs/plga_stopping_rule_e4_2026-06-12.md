# PLGA E4 Stopping-Rule Simulation

Date: 2026-06-12

## Purpose

This note records the fourth PLGA observation-budget supplement:

```text
Can a PLGA release curve stop early once future prediction is sufficiently certain?
```

Implemented by:

```text
scripts/106_plga_stopping_rule_simulation.py
```

Output directory:

```text
outputs/106_plga_stopping_rule_simulation/
```

## Scope

This experiment uses E3 per-curve outputs and simulates stopping rules on the
strict PLGA splits:

```text
source-group-kfold
source-dataset-lodo
```

Stopping decisions are calibrated from other folds. Heldout future RMSE is used
only for evaluation, not for choosing whether a heldout curve stops.

Reference baseline:

```text
never_stop_k5 = always collect first five observed release points
```

## Acceptance Criteria

A stopping rule is accepted only if, relative to `never_stop_k5`, it satisfies:

| Criterion | Threshold |
|---|---:|
| mean saved observations | >= 1.000 |
| median RMSE delta vs k5 | <= 0.025 |
| cov90 delta vs k5 | >= -0.050 |

These thresholds are intentionally conservative enough to reject rules that
save measurements by letting prediction quality collapse.

## Output Files

| File | Meaning |
|---|---|
| `stopping_policy_results.csv` | per-curve stopped policy, error, coverage, and saved observations |
| `stopping_policy_by_split.csv` | split-level stopping tradeoffs |
| `stopping_tradeoff_table.csv` | rule-level pass/fail table across strict splits |
| `calibration_by_fold.csv` | other-fold width calibration for each stop index |
| `marginal_calibration_by_fold.csv` | other-fold marginal gain calibration |
| `data_checks.csv` | finite-value checks |
| `lock_metadata.json` | CLI, git hash, acceptance thresholds, leakage guard |
| `stopping_rule_report.md` | generated markdown report |

## Accepted Rules

| Rule | Family | Mean saved obs | Max median RMSE delta | Min cov90 delta | Mean stop k |
|---|---|---:|---:|---:|---:|
| `next_marginal_gain_le_0.015` | next marginal gain | 3.319 | 0.023 | -0.022 | 1.681 |
| `width90_abs_le_0.70` | width threshold | 3.000 | 0.014 | -0.022 | 2.000 |
| `next_marginal_gain_le_0.010` | next marginal gain | 2.543 | 0.007 | 0.004 | 2.457 |
| `last_marginal_gain_le_0.015` | last marginal gain | 2.319 | 0.012 | 0.000 | 2.681 |
| `width90_ratio_to_k5_le_1.05` | width ratio | 1.637 | 0.009 | -0.032 | 3.363 |
| `last_marginal_gain_le_0.010` | last marginal gain | 1.589 | 0.010 | 0.008 | 3.411 |

## Recommended Primary Rule

Use this as the main E4 evidence:

```text
width90_abs_le_0.70
```

Reason:

- It is simple and auditable: stop when other-fold calibrated empirical width90
  is <= 0.70.
- It stops at `k = 2` on average.
- It saves 3 observations relative to `k = 5`.
- It passes both strict splits.
- It avoids the over-aggressive behavior of the looser marginal rule that can
  stop too close to `k = 1`.

By split:

| Split | Mean stop k | Saved obs | Stopped RMSE | k5 RMSE | Median RMSE delta | cov90 after stop | cov90 k5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| source-dataset-lodo | 2.000 | 3.000 | 0.182 | 0.170 | 0.000 | 0.910 | 0.932 |
| source-group-kfold | 2.000 | 3.000 | 0.189 | 0.167 | 0.014 | 0.872 | 0.836 |

Interpretation:

```text
A conservative empirical-width rule can stop around the second observed point,
save roughly three early observations, and stay within the predeclared RMSE and
coverage tolerance relative to the k5 reference.
```

## Important Caveats

- This is a simulation over existing PLGA curves, not a prospective wet-lab
  stopping protocol.
- The uncertainty is an empirical future-RMSE envelope, not a Bayesian posterior.
- The stable-prediction rule from the original plan is not implemented because
  E3 does not store future prediction trajectories; implementing it requires a
  prediction-level output table from the direct model.
- Rules are calibrated at split/fold level, not individualized per curve.

## Claim Status

Supported:

- At least one conservative stopping rule saves observations while staying
  within the predeclared error and coverage tolerance.
- E4 converts the E1-E3 observation-budget result into an experimental-design
  claim: not every early point needs to be measured if a conservative calibrated
  stopping rule is used.

Not claimed:

- A final clinical or wet-lab protocol.
- A mechanism posterior over hidden PLGA states.
- Curve-specific stopping based on individualized posterior uncertainty.

## Verification

Commands run:

```powershell
python -m py_compile scripts\106_plga_stopping_rule_simulation.py
python scripts\106_plga_stopping_rule_simulation.py
```

`data_checks.csv` reports no unexpected non-finite values.
