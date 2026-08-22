# Liposome Bottleneck Decomposition E2

Date: 2026-06-12

Status: completed bottleneck decomposition after E1 static-only failure.

## Purpose

Decompose why liposome static whole-curve prediction failed before moving to
early observations.

This stage answers:

```text
What failed: API transfer, structure transfer, assay-condition shift,
descriptor missingness, shape-family prior, or static feature recipe stability?
```

It does not train a new predictor and does not use early release observations.

## Reproduction

Script:

```text
scripts/110_liposome_bottleneck_decomposition.py
```

Command:

```powershell
python scripts\110_liposome_bottleneck_decomposition.py --e1 outputs\109_liposome_static_whole_curve_prediction --pool outputs\149_release_caveat_augmented_cumulative_v1 --out outputs\110_liposome_bottleneck_decomposition --seed 0
```

Output directory:

```text
outputs/110_liposome_bottleneck_decomposition/
```

Generated files:

| File | Role |
|---|---|
| `bottleneck_by_group.csv` | group-level residuals by API, structure, pH, temperature, and release method |
| `bottleneck_feature_missingness.csv` | missing-vs-present descriptor error comparisons |
| `shape_family_failure_table.csv` | direct static model vs Weibull static-theta agreement |
| `feature_set_curve_agreement.csv` | which static feature set wins per curve/split |
| `bottleneck_decision_table.csv` | structured E2 decision and next action |
| `data_checks.csv` | E2 integrity checks |
| `lock_metadata.json` | git hash, inputs, args, row counts |
| `bottleneck_report.md` | concise interpretation |

## Main Decision

E2 decision:

```text
early-observation justification = justified
```

Static failure remains after leakage-safe strict splits and reasonable static
baselines. The next stage should be E3 early-observation budget, but only as an
information-bottleneck diagnostic, not as a neural-model sprint.

## Bottleneck Summary

| Bottleneck | Status | Evidence |
|---|---|---|
| API shift | supported | API-group split primary median RMSE = 0.310; best direct static feature set still 0.227 |
| structure shift | partial | structure-type split primary median RMSE = 0.196 |
| assay condition shift | supported | pH = 0.274; temperature = 0.244; release method = 0.227 |
| descriptor missingness | modifier | largest eligible missing-present median RMSE delta = -0.178; missingness changes error structure but is not one-way proof of failure |
| shape-family insufficiency | supported | median both-high-error fraction = 0.444; Weibull does not rescue static failure |
| feature-mixture/model limitation | supported | median primary-is-best fraction = 0.166; median possible feature-choice gain = 0.059 |
| early-observation justification | justified | all primary strict-split static medians remain above partial-success threshold |

## Worst Residual Regimes

The worst residual groups are concentrated around high-temperature and
self-quenching / assay-related regimes:

| Split | Axis | Group | Median RMSE | Excess vs split |
|---|---|---|---:|---:|
| release-method split | temperature | 41.0 | 0.789 | 0.562 |
| release-method split | temperature | 42.0 | 0.736 | 0.509 |
| release-method split | release method | self-quenching | 0.706 | 0.478 |
| release-method split | temperature | 42.4 | 0.694 | 0.467 |
| release-method split | API | carboxyfluorescein | 0.679 | 0.452 |
| temperature split | temperature | 42.0 | 0.648 | 0.404 |
| temperature split | release method | self-quenching | 0.642 | 0.397 |

This points to an assay-condition bottleneck, not just a generic model-capacity
problem.

## Shape Prior Check

Weibull static theta is a useful control but not a solution.

| Split | Direct static RMSE | Weibull static RMSE | Shape better fraction | Both high-error fraction |
|---|---:|---:|---:|---:|
| API group | 0.310 | 0.292 | 0.564 | 0.448 |
| pH group | 0.274 | 0.264 | 0.571 | 0.444 |
| temperature group | 0.244 | 0.285 | 0.368 | 0.456 |
| release-method group | 0.227 | 0.302 | 0.350 | 0.388 |
| structure-type group | 0.196 | 0.223 | 0.330 | 0.325 |

Because direct static and Weibull static often fail together, the bottleneck is
not solved by switching to a simple shape-family prior.

## Feature-Set Instability

The winning static feature set varies by split:

| Split | Most frequent curve-level winner | Fraction |
|---|---|---:|
| API group | drug/API | 0.309 |
| pH group | colloid | 0.341 |
| temperature group | structure/medium | 0.352 |
| release-method group | all descriptors | 0.272 |
| structure-type group | all descriptors | 0.311 |

This supports the user's earlier intuition: different curves appear to depend
on different descriptor combinations. But E2 does not yet justify a MoE model.
The next cleaner test is whether measured early release observations collapse
this curve-specific ambiguity.

## Verification

Checks from `data_checks.csv`:

| Check | Status | Evidence |
|---|---|---|
| E1 methods available | pass | global median, direct ET, Weibull static theta available |
| primary static rows present | pass | primary_rows = 994 |
| required output tables nonempty | pass | all E2 tables generated |
| no nonfinite core metrics | pass | nonfinite_core_cells = 0 |
| decision has next action | pass | decision_rows = 7 |

Validation commands:

```powershell
python -m py_compile scripts\110_liposome_bottleneck_decomposition.py
python scripts\110_liposome_bottleneck_decomposition.py --e1 outputs\109_liposome_static_whole_curve_prediction --pool outputs\149_release_caveat_augmented_cumulative_v1 --out outputs\110_liposome_bottleneck_decomposition --seed 0
```

## Interpretation

Liposome static prediction fails for multiple reasons:

- unseen API transfer remains hard;
- pH, temperature, and release-method shifts carry major residual structure;
- descriptor missingness modifies errors but does not fully explain them;
- the Weibull static prior does not rescue failure;
- no single static feature recipe is stable across curves.

Therefore the next justified experiment is E3:

```text
static descriptors + measured early release observations -> future release
```

The claim should stay narrow:

```text
E3 tests whether early Q reveals missing curve-specific state in liposome, just
as PLGA suggested. It is not a claim of mechanism transfer or foundation-model
learning.
```

