# Liposome Curve Dictionary Static Probe

Date: 2026-06-12

This note records the first static-feature curve-dictionary experiment,
implemented in:

`../scripts/114_liposome_curve_dictionary_static_probe.py`

Local outputs:

`../outputs/114_liposome_curve_dictionary_static_probe/`

## Question

Can a model learn a reusable "curve language" from observed release curves, then
predict the curve-language coefficients from static features only?

The strict version tested here is:

```text
train curves only -> PCA/NMF curve dictionary
train curves -> dictionary coefficients
static features -> dictionary coefficients
predicted coefficients -> predicted Q(t)
```

For test curves, the model never sees the test curve before prediction. Test
curves are used only for evaluation. Oracle reconstruction rows are included
only to measure whether the dictionary itself can represent held-out curve
shapes.

## Main Run

```powershell
python scripts/114_liposome_curve_dictionary_static_probe.py `
  --out outputs/114_liposome_curve_dictionary_static_probe `
  --seed 0 `
  --grid-size 80 `
  --components 2 3 5 8 `
  --n-estimators 300
```

Data source:

`../data/external/accelerated_IVR/repo/accelerated_IVR-main/`

The run uses `111` curves after requiring the author Weibull/cluster data and
the seven static ML features used in the Yanes et al. classifier.

Splits:

| Split | Role |
|---|---|
| `stratified_5fold` | easiest paper-like diagnostic |
| `group_by_API` | drug/API transfer stress test |
| `group_by_release_method` | assay-method transfer stress test |

## Baselines And Methods

| Method | Meaning |
|---|---|
| `global_mean_curve` | train-fold mean curve on the 0-24 h grid |
| `static_et_direct_grid` | ExtraTrees directly predicts all grid points |
| `static_et_weibull_theta` | ExtraTrees predicts Weibull alpha/beta |
| `oracle_pca/nmf_reconstruct_c*` | test curve encoded by train-fold dictionary; representation ceiling only |
| `static_*_pca/nmf_coeff_c*` | static features predict dictionary coefficients |

## Main Result

The curve dictionary itself is strong:

| Split | Best oracle dictionary | Median RMSE pct | Median R2 |
|---|---|---:|---:|
| `group_by_API` | `oracle_pca_reconstruct_c8` | 1.547 | 0.995 |
| `group_by_release_method` | `oracle_pca_reconstruct_c8` | 1.547 | 0.992 |
| `stratified_5fold` | `oracle_pca_reconstruct_c8` | 1.197 | 0.994 |

This means the release curves can be compressed into a small curve language.
The representation is not the bottleneck.

Static-feature coefficient prediction is weaker but nonzero:

| Split | Best static dictionary predictor | Median RMSE pct | Median R2 |
|---|---|---:|---:|
| `group_by_API` | `static_et_pca_coeff_c3` | 15.892 | 0.216 |
| `group_by_release_method` | `static_et_nmf_coeff_c2` | 13.983 | 0.282 |
| `stratified_5fold` | `static_et_nmf_coeff_c5` | 9.578 | 0.693 |

## Decision Deltas

Negative means method A is better.

| Split | Question | Delta RMSE pct |
|---|---|---:|
| `group_by_API` | oracle dictionary vs global mean | -16.996 |
| `group_by_API` | static dictionary coefficients vs global mean | -2.651 |
| `group_by_API` | dictionary coefficients vs direct grid | -1.203 |
| `group_by_API` | dictionary coefficients vs static Weibull theta | -3.868 |
| `group_by_release_method` | oracle dictionary vs global mean | -19.269 |
| `group_by_release_method` | static dictionary coefficients vs global mean | -6.832 |
| `group_by_release_method` | dictionary coefficients vs direct grid | -0.929 |
| `group_by_release_method` | dictionary coefficients vs static Weibull theta | -1.770 |
| `stratified_5fold` | oracle dictionary vs global mean | -16.393 |
| `stratified_5fold` | static dictionary coefficients vs global mean | -8.012 |
| `stratified_5fold` | dictionary coefficients vs direct grid | -0.560 |
| `stratified_5fold` | dictionary coefficients vs static Weibull theta | +1.563 |

## Interpretation

The user's idea partially works.

What works:

1. A compact curve language exists. PCA/NMF dictionaries can reconstruct
   held-out curves with very low error when the test curve is allowed to be
   encoded.
2. Static features can predict useful dictionary coefficients. This beats the
   global mean and direct grid prediction in all split families.
3. Under strict API and release-method splits, dictionary coefficients also
   beat static Weibull theta, suggesting the curve language can be more flexible
   than a two-parameter shape family.

What does not fully work:

1. The gap between oracle reconstruction and static coefficient prediction is
   large. The curve language is good, but static descriptors do not determine
   the coefficient vector precisely.
2. In the easier stratified split, static Weibull theta is still stronger than
   the best dictionary coefficient predictor.
3. Some NMF fits emitted convergence warnings at 2000 iterations, so the most
   robust reading should lean on PCA and ExtraTrees coefficient trends.

## Decision

This is a promising middle-layer result, not a final predictor.

It supports the idea that:

```text
release curves have a learnable low-dimensional language;
static features can predict part of that language;
missing curve-specific state still limits strict transfer.
```

The next justified experiment is to add observation budgets to the same curve
dictionary:

```text
static features + k early observations -> dictionary coefficients -> Q(t)
```

with `k = 0, 1, 2, 3, 5`, compared against:

```text
global mean
static dictionary coefficients
static Weibull theta
early-only Weibull fit
static+early theta
```

If early observations close the oracle gap, this becomes a strong second-system
version of the PLGA information-budget story.
