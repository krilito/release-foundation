# PLGA E2 Empirical Uncertainty Contraction

Date: 2026-06-11

## Purpose

This note records the second PLGA observation-budget supplement:

```text
Does measured early Q reduce empirical future-error uncertainty, not only point error?
```

Implemented by:

```text
scripts/104_plga_empirical_uncertainty_contraction.py
```

Output directory:

```text
outputs/104_plga_empirical_uncertainty_contraction/
```

## Scope

This is not Bayesian posterior uncertainty. The script builds a fold-jackknife
empirical future-RMSE envelope from the already locked per-curve outputs of
scripts 97 and 99.

Width definition:

```text
width90 = 2 * q90(future_rmse from calibration folds)
```

Coverage is checked by whether heldout per-curve future RMSE falls below the
fold-calibrated q90 error envelope.

## Inputs

| Input | Role |
|---|---|
| `outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv` | direct static / early / static+early per-curve future errors |
| `outputs/99_plga_static_to_early_proxy_probe/per_curve_metrics.csv` | static-predicted early proxy vs measured early comparison |

## Output Files

| File | Meaning |
|---|---|
| `uncertainty_by_budget.csv` | empirical width and coverage by split, budget, and method |
| `uncertainty_contraction.csv` | width ratios for static vs measured early contrasts |
| `calibration_summary.csv` | cov50/cov90/cov95 and calibration status |
| `fold_jackknife_envelopes.csv` | per-curve calibrated error envelopes |
| `data_checks.csv` | finite-value checks |
| `lock_metadata.json` | CLI, git hash, and interpretation lock |
| `uncertainty_report.md` | generated markdown report |

## Headline Result

Window-matched static-only vs measured static+early:

| Split | Budget | Width90 ratio | Width90 contraction | cov90 after measured early |
|---|---:|---:|---:|---:|
| source-dataset-lodo | 1 | 1.005 | -0.005 | 0.871 |
| source-dataset-lodo | 2 | 0.810 | 0.190 | 0.896 |
| source-dataset-lodo | 3 | 0.773 | 0.227 | 0.928 |
| source-dataset-lodo | 5 | 0.741 | 0.259 | 0.928 |
| source-group-kfold | 1 | 1.019 | -0.019 | 0.845 |
| source-group-kfold | 2 | 0.844 | 0.156 | 0.865 |
| source-group-kfold | 3 | 0.722 | 0.278 | 0.841 |
| source-group-kfold | 5 | 0.647 | 0.353 | 0.856 |

Interpretation:

```text
k = 1 does not contract uncertainty.
k >= 2 contracts empirical future-error width.
k = 5 gives the strongest contraction.
```

This matches E1's point-error pattern: the first early observation is usually
not enough, while two or more early observations expose a missing kinetic state.

## Proxy Comparison

Static-predicted early proxy also separates from measured early as budget grows:

| Split | Budget | Proxy predicted -> measured width90 ratio | Width90 contraction |
|---|---:|---:|---:|
| source-dataset-lodo | 1 | 1.000 | -0.000 |
| source-dataset-lodo | 2 | 0.740 | 0.260 |
| source-dataset-lodo | 3 | 0.691 | 0.309 |
| source-dataset-lodo | 5 | 0.610 | 0.390 |
| source-group-kfold | 1 | 1.000 | 0.000 |
| source-group-kfold | 2 | 0.807 | 0.193 |
| source-group-kfold | 3 | 0.701 | 0.299 |
| source-group-kfold | 5 | 0.670 | 0.330 |

So the static feature proxy does not recover the same uncertainty reduction as
actually observing early release.

## Claim Status

Supported:

- Measured early Q reduces empirical future-error width under strict splits
  when `k >= 2`.
- The contraction happens without severe cov90 collapse; source-dataset is
  mostly acceptable and source-group is watch-to-acceptable.
- Static-predicted early Q cannot replace measured early Q at higher budgets.

Not claimed:

- Bayesian uncertainty.
- Mechanistic posterior over PLGA hidden states.
- Universal stopping rule.

## Verification

Commands run:

```powershell
python -m py_compile scripts\104_plga_empirical_uncertainty_contraction.py
python scripts\104_plga_empirical_uncertainty_contraction.py
```

`data_checks.csv` reports no unexpected non-finite values.
