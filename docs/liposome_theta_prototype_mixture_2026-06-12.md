# Liposome Theta/Prototype Mixture Probe

Date: 2026-06-12

This note records the second middle-layer experiment for liposome release,
implemented in:

`../scripts/113_liposome_theta_prototype_mixture.py`

Local outputs:

`../outputs/113_liposome_theta_prototype_mixture/`

## Why This Exists

Script `112_liposome_middle_layer_probe.py` showed that the original
3-class kinetic middle layer has real signal, but is too coarse under strict
group splits. This script tests the next finer version:

```text
static descriptors / static+early Q
    -> weights over train-fold Weibull theta prototypes
    -> mixed Q(t)
```

The middle layer is no longer only `Slow / Medium / Fast`. Each train-fold
curve can act as a local Weibull theta prototype.

## Main Run

```powershell
python scripts/113_liposome_theta_prototype_mixture.py `
  --out outputs/113_liposome_theta_prototype_mixture `
  --seed 0 `
  --early-window-h 6 `
  --n-estimators 300 `
  --prototype-ks 3 5 10 20 `
  --refine-nfev 30
```

The run uses the same 6 h early window as script 112. After requiring valid
static features, Weibull parameters, kinetic class labels, at least two early
points, and at least two future points, the main run has `104` curves.

Splits:

| Split | Role |
|---|---|
| `stratified_5fold` | paper-like/easy diagnostic |
| `group_by_API` | API-transfer stress test |
| `group_by_release_method` | assay-method shift stress test |

## Tested Methods

| Method family | Meaning |
|---|---|
| `global_median_weibull` | train-fold median alpha/beta |
| `early_only_weibull_fit` | fit alpha/beta to early Q, initialized from train-fold median |
| `static_knn_proto_mix_k*` | static features select K train theta prototypes and mix their curves |
| `static_early_knn_proto_mix_k*` | static features plus early Q select K train theta prototypes |
| `static_early_knn_proto_theta_refined_k*` | use prototype mixture theta as initialization, then fit to early Q |
| `static_early_et_theta` | learned ExtraTrees theta mapper from static+early features |

Prototype neighbors are selected only from train folds.

## Main Result

Best median future RMSE, in release-percent units:

| Split | Best method | Median future RMSE pct | Interpretation |
|---|---|---:|---|
| `stratified_5fold` | `static_early_et_theta` | 4.992 | learned theta mapping is strong in the easy split |
| `group_by_API` | `static_early_knn_proto_theta_refined_k5` | 8.522 | refined prototype initialization ties early-only |
| `group_by_release_method` | `static_early_knn_proto_theta_refined_k5` | 8.522 | refined prototype initialization ties early-only |

Best direct prototype-mixture rows:

| Split | Best mixture method | Median future RMSE pct |
|---|---|---:|
| `stratified_5fold` | `static_early_knn_proto_mix_k3` | 6.384 |
| `group_by_API` | `static_early_knn_proto_mix_k3` | 11.411 |
| `group_by_release_method` | `static_early_knn_proto_mix_k5` | 12.744 |

## Decision Deltas

Negative means method A is better than method B.

| Split | Question | Delta future RMSE pct |
|---|---|---:|
| `group_by_API` | static prototype mixture vs global median | -4.489 |
| `group_by_API` | early Q routing vs static routing | -5.932 |
| `group_by_API` | best prototype mixture vs early-only fit | +2.889 |
| `group_by_API` | refined prototype init vs early-only fit | -0.000 |
| `group_by_API` | continuous static+early theta vs best prototype mixture | -1.939 |
| `group_by_release_method` | static prototype mixture vs global median | -7.446 |
| `group_by_release_method` | early Q routing vs static routing | -5.216 |
| `group_by_release_method` | best prototype mixture vs early-only fit | +4.222 |
| `group_by_release_method` | refined prototype init vs early-only fit | -0.000 |
| `group_by_release_method` | continuous static+early theta vs best prototype mixture | -3.048 |
| `stratified_5fold` | static prototype mixture vs global median | -10.806 |
| `stratified_5fold` | early Q routing vs static routing | -4.023 |
| `stratified_5fold` | best prototype mixture vs early-only fit | -2.138 |
| `stratified_5fold` | refined prototype init vs early-only fit | -0.000 |
| `stratified_5fold` | continuous static+early theta vs best prototype mixture | -1.392 |

## Interpretation

The finer prototype middle layer improves over crude global priors and early Q
clearly helps choose better local prototypes. However, under strict API and
release-method splits, direct prototype mixture still does not beat early-only
Weibull fitting.

This narrows the path:

1. The useful middle layer is likely continuous theta / local shape state, not
   the 3-class kinetic label alone.
2. Early observations are doing most of the hard work once at least two early
   points are available.
3. Static descriptors help in easier splits and as routing signals, but are not
   yet robust enough under API/method transfer.

## Decision

Do not escalate to a complex MoE/KAN/PINN model yet. The next justified
experiment is an observation-budget E3 table that compares:

```text
early-only fit
static+early theta model
static+early prototype mixture
static+early refined prototype initialization
```

across `k = 0, 1, 2, 3, 5` measured early observations.

If the prototype or theta middle layer only helps after `k >= 2`, the story is
again information-budgeted state identification rather than static feature
prediction.
