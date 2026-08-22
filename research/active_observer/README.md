# Active Kinetic Observer

Status: 2026-05-27 closeout ledger.

This line studies a sparse-experiment release observer:

```
formulation descriptors
  -> kinetic theta particles
  -> PLGABiphasic ODE rollout
  -> active time-point utility
  -> posterior reweighting after observations
  -> conformal-calibrated future release prediction
```

The motivation is simple: formulation descriptors alone are too weak to
identify the full release trajectory. Early release observations contain direct
information about the latent kinetic regime, so the method treats release
prediction as sequential observation plus physical posterior updating.

## Current Claim

Strong claim:

> A two-point active kinetic observer reduces mechanistic prior error under
> grouped evaluation while preserving calibrated uncertainty, and can approach
> fixed late-time measurements with fewer observations.

Do not claim:

- It universally beats direct point predictors. `directQ` is still the strongest
  point predictor on the random split.
- It universally beats all fixed observation schedules. In current v3 outputs,
  fixed 21d or fixed 14d can be slightly better than active two-point.
- Coverage alone proves usefulness. Interval width and CRPS must be reported.

## Evidence Ledger

### Legacy random split: `outputs_active_observer/metrics_summary.csv`

38 test curves. This is the table that matches the early v2/v3 discussion.

| Strategy | RMSE | Conformal coverage 90 | Width 90 | CRPS |
|---|---:|---:|---:|---:|
| zero_early_prior_only | 0.207 | 0.900 | 0.897 | 0.147 |
| active_one_point_posterior | 0.188 | 0.942 | 0.865 | 0.131 |
| active_two_point_posterior | 0.175 | 0.911 | 0.824 | 0.129 |
| fixed_all_four_point | 0.178 | 0.942 | 0.821 | 0.127 |
| directQ_one_point_fixed_7d | 0.120 | - | - | - |

Interpretation: active two-point beats zero-early prior and narrowly beats fixed
four-point, but does not beat directQ.

### Current v3 random split: `outputs_active_observer_v3/metrics_summary.csv`

38 test curves. This is the current generated v3 output and supersedes stale
single-run numbers in older notes.

| Strategy | RMSE | Conformal coverage 90 | Width 90 | CRPS |
|---|---:|---:|---:|---:|
| directQ_one_point_fixed_7d | 0.120 | - | - | - |
| fixed_21d_conformal | 0.129 | 0.930 | 0.490 | 0.106 |
| active_two_point_conformal | 0.138 | 0.930 | 0.500 | 0.112 |
| fixed_14d_conformal | 0.138 | 0.895 | 0.511 | 0.111 |
| fixed_four_point_conformal | 0.149 | 0.886 | 0.515 | 0.121 |
| zero_early_prior_conformal | 0.164 | 0.877 | 0.548 | - |
| zero_early_prior_only | 0.164 | 0.298 | 0.168 | 0.131 |

Interpretation: conformal calibration fixes under-coverage; active two-point
improves the prior, but fixed 21d is slightly better in point error and CRPS.

### DP group split: `outputs_active_observer_v3_group/metrics_summary.csv`

23 test curves, grouped by `DP_Group`.

| Strategy | RMSE | Conformal coverage 90 | Width 90 | CRPS |
|---|---:|---:|---:|---:|
| fixed_14d_conformal | 0.147 | 0.986 | 0.967 | 0.123 |
| fixed_21d_conformal | 0.151 | 0.986 | 0.955 | 0.125 |
| active_two_point_conformal | 0.155 | 0.986 | 0.920 | 0.134 |
| zero_early_prior_conformal | 0.158 | 1.000 | 0.962 | - |
| directQ_one_point_fixed_7d | 0.182 | - | - | - |

Interpretation: group split flips the directQ comparison, but active still does
not beat the best fixed late-time baseline.

### Polymer-family GroupKFold: `outputs_active_observer/groupkfold_results.csv`

148 evaluated curves across 5 folds.

| Strategy | N | RMSE | Conformal coverage 90 | Width 90 | CRPS |
|---|---:|---:|---:|---:|---:|
| prior_only | 148 | 0.226 | 0.924 | 1.087 | 0.169 |
| active_one_point | 148 | 0.194 | 0.976 | 1.143 | 0.143 |
| active_two_point | 148 | 0.185 | 0.973 | 1.076 | 0.142 |

Family-level active two-point RMSE:

| Family | N | RMSE | Conformal coverage 90 |
|---|---:|---:|---:|
| PLGA | 88 | 0.178 | 0.980 |
| PVL-co-PAVL | 38 | 0.168 | 1.000 |
| PCL | 10 | 0.137 | 1.000 |
| PLA-co-PALA | 4 | 0.479 | 0.700 |
| PLGA-co-PALA | 4 | 0.308 | 0.750 |
| PLGA-co-PAVL | 4 | 0.213 | 1.000 |

Interpretation: this is the strongest evidence for the direction. The observer
improves the mechanistic prior under grouped evaluation. The weak small-family
rows should be shown as limitation, not hidden.

## Known Risks

- Interval width is still too large in GroupKFold. Width around 1.08 is
  conservative for a bounded release fraction, so the next optimization target
  should be CRPS and interval sharpness, not only coverage.
- DirectQ remains a strong point-prediction baseline. Keep it in the paper as a
  deliberately strong non-UQ comparator.
- Current v3 random split and legacy random split are different evidence sets.
  Do not mix their numbers in one table without labels.
- Active utility sometimes collapses to early burst plus late erosion windows.
  This can be a scientific finding, but it needs an ablation against fixed 14d
  and fixed 21d.
- Small polymer families have too few curves for stable family-wise claims.

## File Map

| Path | Role |
|---|---|
| `scripts/68_prepare_data.py` | Builds `formulations.csv`, `curves_long.csv`, and `theta_bank.csv` from fitted curve assets. |
| `scripts/68_active_kinetic_observer.py` | Legacy observer and GroupKFold evaluation path. |
| `scripts/68_groupkfold_eval.py` | Convenience wrapper for grouped evaluation. |
| `scripts/68_plot_active_observer.py` | Visualization for observer outputs. |
| `scripts/69_active_observer_v3.py` | Current conformal plus two-point sequential observer. |
| `research/active_observer/CLAUDE.md` | Constraints for future agents touching this line. |
| `docs/plan_70_active_kinetic_observer_route.md` | Paper-facing route and next experimental gates. |

## Next Gates

1. Add fixed 14d, fixed 21d, fixed four-point, and directQ to the 5-fold
   polymer-family GroupKFold table.
2. Add paired bootstrap or Wilcoxon tests for active two-point versus fixed
   baselines.
3. Optimize/report CRPS and interval width; do not treat coverage as sufficient.
4. Produce four paper figures: posterior shrinkage, selected time distribution,
   representative predicted curves, and family-wise generalization.
5. Keep the paper claim centered on calibrated sequential observation, not on
   beating directQ in point RMSE.
