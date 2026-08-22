# Plan 52 - accelerated IVR extension

Date: `2026-05-26`

## Why this dataset first

The `danielyanes22/accelerated_IVR` repository is the cleanest first
cross-mechanism expansion target because it already contains:

- liposome IVR formulation / method descriptors,
- digitized experimental release curves,
- fitted Weibull parameters,
- kinetic class labels from PCA + KMeans,
- and baseline ML classifier results.

This lets us do a fair comparison:

```text
Yanes et al.:
features -> kinetic class

Our extension:
features + early release -> kinetic parameters -> full release curve
```

The goal is not to claim their classifier is wrong. The goal is to move
from classifying release type to forecasting the full release trajectory.

## What they do

Their pipeline is:

```text
literature liposome IVR curves
        ->
fit six kinetic models
        ->
keep Weibull fits with acceptable quality
        ->
simulate 0-24h Weibull curves from alpha/beta
        ->
PCA on simulated Weibull curves
        ->
KMeans k=3
        ->
Fast / Medium / Slow kinetic class
        ->
ML classifier from formulation + IVR conditions to class
```

Main clean tables:

```text
data/unprocessed/backend_data.csv       271 x 15
data/clean/weibull_params.csv           169 x 4
data/clean/ML_7_features_df.csv          77 x 8
data/clean/ML_9_features_df.csv          78 x 10
results/fitting/drug_release_exp.csv   3332 x 3
```

Their reported clean ML target is `cluster`, not full release curve.
Their 5-fold stratified CV test balanced accuracy is about:

```text
XGBoost              0.697
KNN                  0.695
RandomForest         0.677
LogisticRegression   0.668
```

## Our hypothesis

Their work already supports the shared premise:

```text
release curves are easier to learn through a kinetic middle layer.
```

Our stronger test is:

```text
early release + descriptors can recover a continuous kinetic state
well enough to forecast the full curve, not merely classify it as
Fast / Medium / Slow.
```

For liposomes, the first kinetic state should be Weibull:

```text
theta_lipo = (alpha, beta)
```

Do not force the PLGA ODE onto liposome IVR. Use their Weibull layer
first because it is already validated inside their pipeline.

## Stage 1 - ingestion audit

New script:

```text
scripts/52_accelerated_ivr_ingestion_audit.py
```

Inputs:

```text
data/external/accelerated_IVR/repo/accelerated_IVR-main/
```

Outputs:

```text
outputs/52_accelerated_ivr_ingestion_audit/
  curve_inventory.csv
  aligned_curve_table.csv
  aligned_weibull_table.csv
  dataset_summary.csv
  summary.txt
```

Checks:

- Parse `file_name` into IVR ID.
- Convert all time to hours using their processed `drug_release_exp.csv`.
- Confirm release percent ranges.
- Count time points per curve.
- Count usable early observations at candidate windows:
  - 6h,
  - 12h,
  - 24h,
  - 48h,
  - 72h.
- Join curves to:
  - `backend_data.csv`,
  - `weibull_params.csv`,
  - `3_PCA_KMC.csv`.
- Mark curves as:
  - `clean_forecast`: enough early + later points + descriptors + Weibull,
  - `class_only`: descriptors + cluster but insufficient full curve,
  - `reject`: ID mismatch, too few points, invalid release range, or missing descriptors.

Success criterion:

```text
At least 60 clean_forecast curves.
```

If fewer than 60, use it only as an external stress test, not as a
standalone benchmark.

## Stage 2 - reproduce their task

New script:

```text
scripts/53_accelerated_ivr_classification_baseline.py
```

Goal:

Reproduce their classifier task using their clean feature table, then
run a leakage-aware variant.

Baselines:

- majority class,
- LogisticRegression,
- KNN,
- RF,
- ExtraTrees,
- XGBoost only if already installed; do not add dependency for this.

CV schemes:

```text
stratified_5fold              # matches their paper
group_by_API                  # stronger, held-out drug
group_by_release_method       # IVR-method OOD
group_by_structure_type       # liposome-structure OOD if feasible
```

Metrics:

- balanced accuracy,
- macro F1,
- MCC.

Why this matters:

If their 0.69 balanced accuracy collapses under group-by-API or
group-by-method, we know their task is partly interpolation. That does
not invalidate the paper; it gives us a stronger OOD benchmark.

## Stage 3 - our full-curve task

New script:

```text
scripts/54_accelerated_ivr_weibull_forecast.py
```

Task:

```text
features + early Q -> Weibull alpha/beta -> full curve
```

Compare against:

1. **Direct-Q ML**
   - features + early Q -> release vector on common time grid.

2. **Weibull theta ML**
   - features + early Q -> alpha/beta -> Weibull curve.

3. **Early-only Weibull refinement**
   - fit alpha/beta using only early observations.

4. **Hybrid**
   - tree ensemble predicts alpha/beta,
   - early observations lightly refine alpha/beta,
   - Weibull decodes full curve.

5. **Class-conditioned hybrid**
   - predict class probability,
   - use class-specific alpha/beta priors or branch models,
   - refine with early observations.

CV schemes:

```text
random_5fold
group_by_API
group_by_release_method
```

Early windows:

```text
6h
12h
24h
48h
72h
```

Primary metric:

```text
median full-curve R^2
```

Secondary metrics:

- RMSE,
- fraction R^2 >= 0,
- tail failure rate,
- early-window cost,
- theta plausibility / boundary hits.

Success criterion:

```text
Hybrid Weibull theta forecast beats direct-Q ML in random CV and at
least one OOD scheme, while using <= 24h or <= 48h early observations.
```

Stronger claim if achieved:

```text
The same kinetic-middle-layer paradigm transfers from PLGA to liposome
IVR without using a PLGA-specific ODE.
```

## Stage 4 - "exceed them" fairly

We should not say:

```text
Our R^2 beats their balanced accuracy.
```

That is not a fair comparison.

Fair ways to exceed:

1. **Same task improvement**
   Improve kinetic class prediction under their stratified CV and under
   harder group CV.

2. **Harder task**
   Show full-curve forecasting, where they only classify kinetic type.

3. **Earlier decision**
   Show that <=24h or <=48h early data is enough to forecast the rest
   of the curve.

4. **Mechanistic continuity**
   Predict continuous Weibull alpha/beta instead of discrete class only.

5. **OOD clarity**
   Report held-out API / held-out IVR method performance, which their
   main pipeline does not emphasize.

## Expected paper framing

If it works:

```text
Yanes et al. demonstrated that liposome IVR profiles can be reduced to
predictable kinetic classes. We extend this idea to observer-corrected
full-trajectory forecasting by predicting a continuous kinetic state
(Weibull alpha/beta) from formulation descriptors and sparse early
release observations.
```

This would make the larger project claim much stronger:

```text
The release-quality paradigm is not PLGA-specific. It can use a
mechanism-appropriate kinetic decoder: PLGA ODE for PLGA, Weibull
surrogate for liposome IVR, and later a mechanism-specific simulator
for each release platform.
```

## Stop conditions

Do not force the dataset if:

- clean forecast curves < 60,
- most curves have too few post-early observations,
- ID alignment is ambiguous,
- early windows are too irregular to compare fairly,
- direct-Q ML dominates theta->Weibull in all schemes.

If any stop condition hits, keep the dataset as a literature-control /
external stress-test, not as the main expansion evidence.

## First execution result

Implemented:

```text
scripts/52_accelerated_ivr_ingestion_audit.py
scripts/54_accelerated_ivr_weibull_forecast.py
scripts/55_accelerated_ivr_surrogate_observer.py
```

Main audit:

```text
clean forecast curves:
  6h   104
  12h   93
  24h   33
```

Main curve-R2 result:

```text
6h full-curve median R2:
  best class-prototype:  0.568 / 0.163 / -0.138
  early Weibull fit:     0.914 / 0.914 /  0.914

12h full-curve median R2:
  best class-prototype:  0.544 / 0.342 /  0.017
  early Weibull fit:     0.956 / 0.956 /  0.956
```

Each row is:

```text
stratified_5fold / group_by_API / group_by_release_method
```

Important correction:

```text
RF/ET-initialized theta + early refinement is not meaningfully better
than early-only Weibull fitting.
```

So the honest liposome finding is not "tree models predict liposome
theta." It is:

```text
sparse early release observations calibrate a continuous Weibull kinetic
state that decodes full liposome IVR curves far better than class
prototype curves.
```

## Surrogate-observer follow-up

Script `55_accelerated_ivr_surrogate_observer.py` tests whether cheaper
proxy features can replace early release.

Result:

```text
6h full-curve median R2:
  early release gold:      0.906 / 0.906 / 0.906
  all proxy + early Q:     0.871 / 0.801 / 0.708
  best proxy, no early Q:  0.557 / 0.190 / 0.185

12h full-curve median R2:
  early release gold:      0.950 / 0.950 / 0.950
  all proxy + early Q:     0.916 / 0.791 / 0.716
  best proxy, no early Q:  0.727 / 0.248 / -0.094
```

Rows are:

```text
stratified_5fold / group_by_API / group_by_release_method
```

Verdict:

```text
Current formulation/QC proxies are weak priors, not true replacements
for early release. They help in interpolation but fail under OOD shifts.
```

This gives the next data-collection target:

```text
find richer rapid-QC observers that can approach early-release
information content.
```

Detailed record:

```text
docs/research_log_accelerated_ivr_2026-05-26.md
```
