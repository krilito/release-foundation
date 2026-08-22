# Liposome Latent Gap Observation-Budget

Date: 2026-06-12

## Question

Does measured early release close the latent-state gap between static descriptors and oracle curve-language coefficients?

This experiment evaluates:

```text
X -> z
X + Y_k -> z
oracle Q(t) -> z
```

where `z` is a train-fold PCA curve dictionary coefficient vector and `Y_k` is the first `k` measured release observations.

## Method

- Data: Yanes et al. accelerated IVR liposome release curves.
- Unit: one curve per sample.
- Grid: 0-24 h with 80 points.
- Primary representation: PCA dictionary with 8 components.
- Sensitivity representation: PCA dictionary with 3 components.
- Splits: stratified 5-fold, group by API, and group by release method.
- Predictors: ExtraTrees as the primary small-data nonlinear model; Ridge as a linear sanity check.
- Uncertainty: bootstrap ExtraTrees ensemble intervals, reported as empirical intervals rather than Bayesian posteriors.

## Primary Result

| Split | Model | k | Median future RMSE | Oracle gap closed pct | width90 ratio |
|---|---|---:|---:|---:|---:|
| group_by_API | `static_plus_obs_et_pca_c8` | 0 | 17.285 | 0.0 | 1.000 |
| group_by_API | `static_plus_obs_et_pca_c8` | 1 | 17.195 | 0.6 | 1.444 |
| group_by_API | `static_plus_obs_et_pca_c8` | 2 | 16.134 | 7.2 | 1.340 |
| group_by_API | `static_plus_obs_et_pca_c8` | 3 | 16.300 | 6.2 | 1.061 |
| group_by_API | `static_plus_obs_et_pca_c8` | 5 | 14.119 | 19.8 | 1.123 |
| group_by_API | `static_plus_obs_ridge_pca_c8` | 0 | 25.751 | 0.0 | NA |
| group_by_API | `static_plus_obs_ridge_pca_c8` | 1 | 26.408 | -2.7 | NA |
| group_by_API | `static_plus_obs_ridge_pca_c8` | 2 | 18.425 | 29.9 | NA |
| group_by_API | `static_plus_obs_ridge_pca_c8` | 3 | 17.160 | 35.1 | NA |
| group_by_API | `static_plus_obs_ridge_pca_c8` | 5 | 11.802 | 57.0 | NA |
| group_by_release_method | `static_plus_obs_et_pca_c8` | 0 | 18.016 | 0.0 | 1.000 |
| group_by_release_method | `static_plus_obs_et_pca_c8` | 1 | 17.646 | 2.2 | 1.696 |
| group_by_release_method | `static_plus_obs_et_pca_c8` | 2 | 17.349 | 4.0 | 1.125 |
| group_by_release_method | `static_plus_obs_et_pca_c8` | 3 | 17.469 | 3.3 | 1.148 |
| group_by_release_method | `static_plus_obs_et_pca_c8` | 5 | 17.598 | 2.5 | 1.002 |
| group_by_release_method | `static_plus_obs_ridge_pca_c8` | 0 | 16.389 | 0.0 | NA |
| group_by_release_method | `static_plus_obs_ridge_pca_c8` | 1 | 17.193 | -5.4 | NA |
| group_by_release_method | `static_plus_obs_ridge_pca_c8` | 2 | 18.068 | -11.2 | NA |
| group_by_release_method | `static_plus_obs_ridge_pca_c8` | 3 | 15.543 | 5.6 | NA |
| group_by_release_method | `static_plus_obs_ridge_pca_c8` | 5 | 12.669 | 24.5 | NA |
| stratified_5fold | `static_plus_obs_et_pca_c8` | 0 | 11.077 | 0.0 | 1.000 |
| stratified_5fold | `static_plus_obs_et_pca_c8` | 1 | 11.677 | -6.0 | 1.250 |
| stratified_5fold | `static_plus_obs_et_pca_c8` | 2 | 7.463 | 35.5 | 0.936 |
| stratified_5fold | `static_plus_obs_et_pca_c8` | 3 | 6.391 | 46.0 | 0.926 |
| stratified_5fold | `static_plus_obs_et_pca_c8` | 5 | 5.898 | 50.7 | 0.816 |
| stratified_5fold | `static_plus_obs_ridge_pca_c8` | 0 | 14.393 | 0.0 | NA |
| stratified_5fold | `static_plus_obs_ridge_pca_c8` | 1 | 15.028 | -4.8 | NA |
| stratified_5fold | `static_plus_obs_ridge_pca_c8` | 2 | 10.027 | 32.4 | NA |
| stratified_5fold | `static_plus_obs_ridge_pca_c8` | 3 | 8.970 | 40.2 | NA |
| stratified_5fold | `static_plus_obs_ridge_pca_c8` | 5 | 9.131 | 38.9 | NA |

## Decision Table

| Split | Question | Value | Passed |
|---|---|---:|---:|
| group_by_API | Does X -> z with et beat global mean at k=0? | 17.285 | True |
| group_by_API | Does X + Y_k -> z improve monotonically or near-monotonically for et? | 1.000 | True |
| group_by_API | How much oracle gap is closed at k=1 for et? | 0.570 | True |
| group_by_API | How much oracle gap is closed at k=2 for et? | 7.183 | True |
| group_by_API | How much oracle gap is closed at k=3 for et? | 6.154 | True |
| group_by_API | How much oracle gap is closed at k=5 for et? | 19.763 | True |
| group_by_API | Does ensemble interval width contract with k for et? | 1.123 | False |
| group_by_API | Does strict group split still show gap closure for et? | 19.763 | True |
| group_by_API | Does X -> z with ridge beat global mean at k=0? | 25.751 | False |
| group_by_API | Does X + Y_k -> z improve monotonically or near-monotonically for ridge? | 0.750 | True |
| group_by_API | How much oracle gap is closed at k=1 for ridge? | -2.704 | False |
| group_by_API | How much oracle gap is closed at k=2 for ridge? | 29.905 | True |
| group_by_API | How much oracle gap is closed at k=3 for ridge? | 35.095 | True |
| group_by_API | How much oracle gap is closed at k=5 for ridge? | 56.963 | True |
| group_by_API | Does strict group split still show gap closure for ridge? | 56.963 | True |
| group_by_release_method | Does X -> z with et beat global mean at k=0? | 18.016 | True |
| group_by_release_method | Does X + Y_k -> z improve monotonically or near-monotonically for et? | 1.000 | True |
| group_by_release_method | How much oracle gap is closed at k=1 for et? | 2.245 | True |
| group_by_release_method | How much oracle gap is closed at k=2 for et? | 3.999 | True |
| group_by_release_method | How much oracle gap is closed at k=3 for et? | 3.265 | True |
| group_by_release_method | How much oracle gap is closed at k=5 for et? | 2.489 | True |
| group_by_release_method | Does ensemble interval width contract with k for et? | 1.002 | False |
| group_by_release_method | Does strict group split still show gap closure for et? | 2.489 | True |
| group_by_release_method | Does X -> z with ridge beat global mean at k=0? | 16.389 | True |
| group_by_release_method | Does X + Y_k -> z improve monotonically or near-monotonically for ridge? | 0.500 | False |
| group_by_release_method | How much oracle gap is closed at k=1 for ridge? | -5.410 | False |
| group_by_release_method | How much oracle gap is closed at k=2 for ridge? | -11.158 | False |
| group_by_release_method | How much oracle gap is closed at k=3 for ridge? | 5.591 | True |
| group_by_release_method | How much oracle gap is closed at k=5 for ridge? | 24.529 | True |
| group_by_release_method | Does strict group split still show gap closure for ridge? | 24.529 | True |
| stratified_5fold | Does X -> z with et beat global mean at k=0? | 11.077 | True |
| stratified_5fold | Does X + Y_k -> z improve monotonically or near-monotonically for et? | 0.750 | True |
| stratified_5fold | How much oracle gap is closed at k=1 for et? | -5.985 | False |
| stratified_5fold | How much oracle gap is closed at k=2 for et? | 35.527 | True |
| stratified_5fold | How much oracle gap is closed at k=3 for et? | 46.004 | True |
| stratified_5fold | How much oracle gap is closed at k=5 for et? | 50.650 | True |
| stratified_5fold | Does ensemble interval width contract with k for et? | 0.816 | True |
| stratified_5fold | Does X -> z with ridge beat global mean at k=0? | 14.393 | True |
| stratified_5fold | Does X + Y_k -> z improve monotonically or near-monotonically for ridge? | 0.750 | True |
| stratified_5fold | How much oracle gap is closed at k=1 for ridge? | -4.760 | False |
| stratified_5fold | How much oracle gap is closed at k=2 for ridge? | 32.372 | True |
| stratified_5fold | How much oracle gap is closed at k=3 for ridge? | 40.162 | True |
| stratified_5fold | How much oracle gap is closed at k=5 for ridge? | 38.858 | True |

## Interpretation Rule

A positive gap-closure curve supports the claim that early observations identify release state that static descriptors miss. A weak or negative strict-split result means the available static descriptors and early points do not transfer reliably across held-out drug or assay groups.

Do not read this as a mechanism-learning or foundation-model result. It is a release-state inference diagnostic.

## Output Anchor

`../outputs/115_liposome_latent_gap_observation_budget/`
