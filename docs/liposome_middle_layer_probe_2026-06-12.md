# Liposome Middle-Layer Probe

Date: 2026-06-12

This note records the first active middle-layer experiment for the liposome
system, implemented in:

`../scripts/112_liposome_middle_layer_probe.py`

Local outputs:

`../outputs/112_liposome_middle_layer_probe/`

## Question

Can an intermediate kinetic representation improve liposome release prediction?

The tested forms are:

```text
static descriptors -> kinetic class -> class Weibull prototype -> Q(t)
static descriptors -> class probabilities -> weighted Weibull prototypes -> Q(t)
static descriptors + early Q -> class probabilities -> weighted prototypes -> Q(t)
static descriptors + early Q -> Weibull alpha/beta -> Q(t)
```

This is not a mechanism claim. The middle layer is supervised by the original
Yanes et al. kinetic-class / Weibull-parameter pipeline.

## Data And Split

Input root:

`../data/external/accelerated_IVR/repo/accelerated_IVR-main/`

Main run:

```powershell
python scripts/112_liposome_middle_layer_probe.py `
  --out outputs/112_liposome_middle_layer_probe `
  --seed 0 `
  --early-window-h 6 `
  --n-estimators 300 `
  --refine-nfev 30
```

The 6 h early window is used as the main setting because a 24 h window leaves
too few curves with at least two future points. After requiring valid static
features, Weibull parameters, kinetic class, at least two early points, and at
least two future points, the main run has `104` curves.

Splits:

| Split | Role |
|---|---|
| `stratified_5fold` | paper-like diagnostic, easiest setting |
| `group_by_API` | API-transfer stress test |
| `group_by_release_method` | assay-method shift stress test |

## Models

Class-prototype middle layer:

```text
classifier predicts kinetic class;
train-fold class median alpha/beta defines a Weibull prototype;
prototype generates Q(t).
```

Soft-prototype middle layer:

```text
classifier predict_proba gives class weights;
prediction is the weighted sum of train-fold Weibull prototype curves.
```

Early-conditioned middle layer:

```text
static descriptors + interpolated early Q points -> class/theta predictor.
```

Continuous-theta control:

```text
ExtraTrees predicts log(alpha), log(beta);
optional early refinement fits alpha/beta to early Q only.
```

All prototypes and theta medians are computed from train folds only.

## Main Result

Best median future RMSE, in release-percent units:

| Split | Best method | Median future RMSE pct | Interpretation |
|---|---|---:|---|
| `stratified_5fold` | `static_early_et_theta` | 4.992 | early Q + theta model is strong in the easiest split |
| `group_by_API` | `static_et_theta_early_refined` | 8.522 | early-refined Weibull theta is best; class layer trails |
| `group_by_release_method` | `early_only_weibull_fit` | 8.522 | early-only fit is already near the ceiling |

Best class-middle-layer rows:

| Split | Best class method | Median future RMSE pct |
|---|---|---:|
| `stratified_5fold` | `static_early_soft_logreg_class_proto` | 7.094 |
| `group_by_API` | `static_early_soft_logreg_class_proto` | 9.935 |
| `group_by_release_method` | `static_early_hard_logreg_class_proto` | 12.844 |

## Decision Deltas

Negative means the first method is better.

| Split | Question | Delta future RMSE pct |
|---|---|---:|
| `group_by_API` | best hard class prototype vs global median | -2.683 |
| `group_by_API` | best soft class vs best hard class | -1.018 |
| `group_by_API` | early Q soft class vs static soft class | -8.197 |
| `group_by_API` | best early class middle layer vs early-only Weibull fit | +1.413 |
| `group_by_release_method` | best hard class prototype vs global median | -8.958 |
| `group_by_release_method` | best soft class vs best hard class | +2.805 |
| `group_by_release_method` | early Q soft class vs static soft class | -5.452 |
| `group_by_release_method` | best early class middle layer vs early-only Weibull fit | +4.322 |
| `stratified_5fold` | best hard class prototype vs global median | -9.200 |
| `stratified_5fold` | best soft class vs best hard class | -0.299 |
| `stratified_5fold` | early Q soft class vs static soft class | -4.619 |
| `stratified_5fold` | best early class middle layer vs early-only Weibull fit | -1.428 |

## Interpretation

The middle layer is useful as a diagnostic, but the current kinetic-class layer
is too coarse to be the final predictor.

What seems supported:

1. Kinetic-class prototypes beat a global Weibull median in all three split
   families, so the author middle layer contains real release-shape signal.
2. Early Q improves the soft class middle layer in all three split families,
   supporting the idea that early observations route curves to better kinetic
   states.
3. Under strict group splits, early-only / continuous-theta Weibull routes
   remain stronger than class prototypes. The class layer is not enough.

What is not supported yet:

1. Do not claim the class middle layer beats early observation fitting under
   strict transfer.
2. Do not claim a physical mechanism was learned.
3. Do not claim the liposome result proves cross-system transfer.

## Next Step

Move from a 3-class kinetic middle layer to a finer theta/shape middle layer:

```text
static descriptors + early Q -> local alpha/beta or prototype-mixture weights
```

The immediate next experiment should test whether mixture over Weibull theta
neighbors, rather than hard KMeans classes, beats early-only fitting under
`group_by_API` and `group_by_release_method`.
