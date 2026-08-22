# Research log - accelerated IVR extension

Date: `2026-05-26`

## Goal

Use the `danielyanes22/accelerated_IVR` liposome IVR dataset to test
whether our release-quality paradigm extends beyond PLGA.

Fair comparison:

```text
Yanes et al.:
features -> Fast/Medium/Slow kinetic class

Curve-level version of their method:
features -> predicted class -> train-fold class prototype curve

Our extension:
features + early release -> continuous Weibull alpha/beta -> full curve
```

## Scripts added

### `52_accelerated_ivr_ingestion_audit.py`

Purpose:

- align experimental release curves,
- backend descriptors,
- Weibull parameters,
- kinetic class labels,
- and early-window availability.

Key output:

```text
outputs/52_accelerated_ivr_ingestion_audit/summary.txt
```

Dataset audit:

```text
n_release_curves             209
n_with_backend               208
n_with_weibull               169
n_with_cluster               169
n_complete_features_7        129
n_complete_features_9        126
n_clean_forecast_any         104
n_clean_forecast_6h          104
n_clean_forecast_12h          93
n_clean_forecast_24h          33
n_clean_forecast_48h          13
n_clean_forecast_72h           8
```

Interpretation:

- The dataset is usable for a liposome curve benchmark.
- The realistic early windows are `6h` and `12h`.
- `24h` is only a small reference subset.
- `48h` and `72h` are too small for serious benchmarking.

### `54_accelerated_ivr_weibull_forecast.py`

Purpose:

Run a curve-R2 comparison under the same folds:

1. Their deployable class-prototype curve:

```text
features -> class classifier -> train-fold class median Weibull curve
```

2. Their oracle class-prototype upper bound:

```text
true class -> train-fold class median Weibull curve
```

3. Direct curve ML:

```text
features + early Q -> Q grid
```

4. Our continuous kinetic-state route:

```text
features + early Q -> Weibull alpha/beta -> curve
```

5. Early-only Weibull fit:

```text
early Q only -> Weibull alpha/beta -> curve
```

CV schemes:

```text
stratified_5fold
group_by_API
group_by_release_method
```

## Main result

The class-prototype curve baseline is weak as a curve predictor. This is
not surprising: it was designed to classify kinetic type, not forecast a
trajectory.

Focused full-curve median R2:

```text
6h early window, n=104

scheme                   best class-prototype   oracle class-prototype   early/Weibull fit   direct Q
stratified_5fold          0.568                  0.668                    0.914               0.897
group_by_API              0.163                  0.627                    0.914               0.805
group_by_release_method  -0.138                  0.560                    0.914               0.729
```

```text
12h early window, n=93

scheme                   best class-prototype   oracle class-prototype   early/Weibull fit   direct Q
stratified_5fold          0.544                  0.670                    0.956               0.940
group_by_API              0.342                  0.735                    0.956               0.870
group_by_release_method   0.017                  0.510                    0.956               0.817
```

```text
24h early window, n=33

scheme                   best class-prototype   oracle class-prototype   early/Weibull fit   direct Q
stratified_5fold          0.465                  0.711                    0.889               0.922
group_by_API              0.543                  0.665                    0.889               0.867
group_by_release_method  -0.201                  0.292                    0.889               0.560
```

## Important caveat

The strongest curve method is not the RF/ET predicted Weibull theta.
After early-only local refinement, RF/ET-initialized theta and a simple
early-only Weibull fit converge to nearly the same result.

That means the honest conclusion is:

```text
For this liposome IVR dataset, sparse early release observations are
already sufficient to calibrate a Weibull kinetic state.
```

This is still valuable. It supports the broader paradigm:

```text
early observation -> kinetic state -> full release curve
```

But it does not support a claim that formulation descriptors alone or
RF/ET theta prediction are the source of the liposome improvement.

## Future-R2 caution

Future-only R2 after the early window is often unstable or extremely
negative because many liposome curves have low variance after they begin
to plateau. Therefore:

- use full-curve R2 as the main trajectory metric,
- report future-only R2 or RMSE as a stress check,
- do not overinterpret large negative future-only R2 values without
  checking the denominator variance.

## Interpretation versus Yanes et al.

Fair statement:

> Yanes et al. show that liposome IVR profiles can be reduced to
> predictable kinetic classes. We extend the same kinetic-middle-layer
> idea to curve-level forecasting by fitting a continuous Weibull state
> from sparse early release observations.

Do not say:

```text
Our R2 beats their balanced accuracy.
```

Better:

```text
Their deployable class labels can be decoded into coarse prototype
curves, but this loses trajectory detail. A continuous Weibull state
calibrated from early release recovers full curves much more accurately.
```

## Current verdict

This is a successful first cross-mechanism diagnostic.

It does not prove a general foundation model yet, but it proves that the
PLGA paper's core abstraction travels:

```text
do not learn raw curves first;
learn/calibrate a kinetic state, then decode the curve.
```

For liposomes, the first kinetic state is Weibull `(alpha, beta)` rather
than the PLGA 9-parameter ODE.

## 55 surrogate-observer audit

Question:

```text
If early release is the strongest observer, can cheaper descriptors or
rapid QC features replace it?
```

Implemented:

```text
scripts/55_accelerated_ivr_surrogate_observer.py
```

Compared observers:

```text
formulation_prior:
  pH, temperature, drug loading, API type, weighted MW, weighted Tm

plus_size:
  formulation_prior + Z-average size

plus_qc:
  plus_size + PDI + zeta potential

plus_method_structure:
  plus_qc + release method + liposome structure type

all_proxy_plus_earlyQ:
  all proxy features + early release

early_release_gold:
  early release only -> Weibull fit
```

Focused full-curve median R2:

```text
6h early window

scheme                   early gold   all proxy + earlyQ   best proxy without earlyQ
stratified_5fold          0.906        0.871                0.557
group_by_API              0.906        0.801                0.190
group_by_release_method   0.906        0.708                0.185
```

```text
12h early window

scheme                   early gold   all proxy + earlyQ   best proxy without earlyQ
stratified_5fold          0.950        0.916                0.727
group_by_API              0.950        0.791                0.248
group_by_release_method   0.950        0.716               -0.094
```

Interpretation:

- Cheap surrogate observers contain some signal under random CV.
- They do **not** replace early release under held-out API or held-out
  release-method shifts.
- Adding proxy features to early Q can still be useful in deployment,
  but early release remains the dominant observer.

The honest claim is:

```text
Early release is the gold observer of kinetic state.
Current formulation/QC proxies are weak priors, not substitutes.
```

Scientific framing:

```text
The next data frontier is not a bigger model. It is finding richer
surrogate observers: spectroscopy, imaging, membrane rigidity,
lamellarity, morphology, encapsulation quality, or other rapid QC
signals that approach the information content of early release.
```
