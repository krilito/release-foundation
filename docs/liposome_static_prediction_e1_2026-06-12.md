# Liposome Static Whole-Curve Prediction E1

Date: 2026-06-12

Status: completed first static-only prediction gate for the second
drug-release system.

## Purpose

Test the first branch of the liposome route:

```text
Can static formulation and assay descriptors predict the full liposome release
curve under strict non-source splits?
```

No early release observations are used as model inputs.

## Reproduction

Script:

```text
scripts/109_liposome_static_whole_curve_prediction.py
```

Command used for the first E1 run:

```powershell
python scripts\109_liposome_static_whole_curve_prediction.py --pool outputs\149_release_caveat_augmented_cumulative_v1 --out outputs\109_liposome_static_whole_curve_prediction --seed 0 --n-estimators 80 --max-shape-nfev 1000
```

Output directory:

```text
outputs/109_liposome_static_whole_curve_prediction/
```

Generated files:

| File | Role |
|---|---|
| `static_summary_by_split.csv` | split-level metrics for baselines and static models |
| `static_per_curve_metrics.csv` | curve-level RMSE / MAE / R2 |
| `static_feature_set_ablation.csv` | direct static feature-set comparison |
| `static_baseline_contrasts.csv` | paired gains vs global median curve |
| `split_audit.csv` | split feasibility and eligible curve counts |
| `data_checks.csv` | leakage and metric-integrity checks |
| `lock_metadata.json` | git hash, inputs, CLI args, feature sets, split axes |
| `static_prediction_report.md` | concise interpretation |

## Main Result

E1 decision:

```text
static_failure
```

Static features improve over a global median curve, but they do not predict
well enough to call liposome statically identifiable under strict splits.

Primary static model:

```text
method = direct_et_point
feature_set = all_no_source
```

Results:

| Split | Median RMSE | Weighted RMSE | Median MAE | Curves |
|---|---:|---:|---:|---:|
| API group | 0.310 | 0.342 | 0.259 | 181 |
| pH group | 0.274 | 0.333 | 0.225 | 205 |
| temperature group | 0.244 | 0.315 | 0.205 | 193 |
| release-method group | 0.227 | 0.359 | 0.194 | 206 |
| structure-type group | 0.196 | 0.292 | 0.163 | 209 |

Baseline context:

| Split | Global median RMSE | Weibull static-theta RMSE | Primary direct static RMSE |
|---|---:|---:|---:|
| API group | 0.332 | 0.292 | 0.310 |
| pH group | 0.332 | 0.264 | 0.274 |
| temperature group | 0.353 | 0.285 | 0.244 |
| release-method group | 0.402 | 0.302 | 0.227 |
| structure-type group | 0.333 | 0.223 | 0.196 |

Best observed direct feature set varied by split:

| Split | Best method | Feature set | Median RMSE |
|---|---|---|---:|
| API group | direct ET | drug_api | 0.227 |
| pH group | direct ET | structure_medium | 0.258 |
| temperature group | direct ET | structure_medium | 0.208 |
| release-method group | direct ET | all_no_source | 0.227 |
| structure-type group | direct ET | all_no_source | 0.196 |

This split-dependence means E1 should not be summarized as "static features do
nothing." A more precise reading is:

```text
Static descriptors contain useful signal, but the signal is not stable enough
across strict heldout regimes to close the liposome prediction problem.
```

## Verification

Checks from `data_checks.csv`:

| Check | Status | Evidence |
|---|---|---|
| finite point rows | pass | 3332 points, 209 curves |
| no heldout group in training fold | pass | overlap_folds=0 |
| no release observation inputs | pass | no release columns used as features |
| no nonfinite metrics | pass | nonfinite_metric_cells=0 |
| required methods present | pass | global median, API-local median, direct ET, Weibull static theta |

Validation commands:

```powershell
python -m py_compile scripts\109_liposome_static_whole_curve_prediction.py
python scripts\109_liposome_static_whole_curve_prediction.py --pool outputs\149_release_caveat_augmented_cumulative_v1 --out outputs\109_liposome_static_whole_curve_prediction --seed 0 --n-estimators 80 --max-shape-nfev 1000
```

## Interpretation

The static-first gate did its job. It found signal, but not enough.

Therefore the route should move to E2:

```text
decompose the static failure before adding early release observations
```

The immediate E2 questions are:

- Is failure dominated by unseen API groups?
- Is failure dominated by structure type, pH, temperature, or release method?
- Are errors concentrated in curves with missing particle size, PDI, or zeta?
- Does the Weibull shape prior fail on the same curves as direct ET?
- Are the "good" feature sets good on the same curves, or is this a
  split-dependent feature-mixture problem?

Do not proceed to E3 early-observation budgets until E2 identifies whether the
bottleneck is descriptor insufficiency, assay/source shift, shape-family
insufficiency, or model-class limitation.

