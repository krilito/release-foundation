# Liposome Design Utility Benchmark

Date: 2026-06-12

## Question

Can a static release prior guide candidate selection before any release observation, even when exact curve RMSE is limited?

This answers a different question from curve prediction:

```text
prediction utility != exact curve RMSE
```

## Method

- Candidate predictors: global mean, author-style class prototype, static Weibull theta, static PCA coefficient prior, static direct grid, and bootstrap ensemble uncertainty prior.
- Targets: anti-burst, sustained-mid, high-final, and avoid-failure windows defined by train-fold quantiles.
- Ranking metrics: top-k hit rate, enrichment over random, regret, failure avoidance, AUROC, AUPRC, Brier, and ECE.

## Best Top-20pct Rows

| Split | Target | Method | lambda | Hit rate | Enrichment | Failure avoidance | AUROC |
|---|---|---|---:|---:|---:|---:|---:|
| group_by_API | anti_burst | `static_ensemble_uncertainty_prior` | 0.25 | 0.330 | 0.667 | 0.782 | 0.423 |
| group_by_API | anti_burst | `static_ensemble_uncertainty_prior` | 0.50 | 0.330 | 0.667 | 0.848 | 0.423 |
| group_by_API | anti_burst | `global_mean_curve` | 0.00 | 0.318 | 0.670 | 0.582 | 0.500 |
| group_by_API | avoid_failure | `static_curve_dictionary_coeff` | 0.00 | 0.848 | 1.436 | 0.848 | 0.728 |
| group_by_API | avoid_failure | `static_et_direct_grid` | 0.00 | 0.848 | 1.436 | 0.848 | 0.663 |
| group_by_API | avoid_failure | `author_class_proto` | 0.00 | 0.782 | 1.236 | 0.782 | 0.634 |
| group_by_API | high_final | `static_ensemble_uncertainty_prior` | 0.00 | 0.515 | 1.537 | 0.448 | 0.730 |
| group_by_API | high_final | `static_curve_dictionary_coeff` | 0.00 | 0.337 | 0.775 | 0.515 | 0.674 |
| group_by_API | high_final | `static_ensemble_uncertainty_prior` | 0.25 | 0.337 | 0.775 | 0.582 | 0.730 |
| group_by_API | sustained_mid | `static_curve_dictionary_coeff` | 0.00 | 0.406 | 1.594 | 0.800 | 0.532 |
| group_by_API | sustained_mid | `static_et_direct_grid` | 0.00 | 0.339 | 1.094 | 0.800 | 0.673 |
| group_by_API | sustained_mid | `static_ensemble_uncertainty_prior` | 0.25 | 0.322 | 1.543 | 0.733 | 0.752 |
| group_by_release_method | anti_burst | `static_curve_dictionary_coeff` | 0.00 | 0.717 | 1.761 | 0.950 | 0.571 |
| group_by_release_method | anti_burst | `static_ensemble_uncertainty_prior` | 0.00 | 0.700 | 1.684 | 0.917 | 0.500 |
| group_by_release_method | anti_burst | `static_et_direct_grid` | 0.00 | 0.700 | 1.684 | 0.950 | 0.577 |
| group_by_release_method | avoid_failure | `static_curve_dictionary_coeff` | 0.00 | 0.950 | 1.663 | 0.950 | 0.630 |
| group_by_release_method | avoid_failure | `static_et_direct_grid` | 0.00 | 0.950 | 1.663 | 0.950 | 0.634 |
| group_by_release_method | avoid_failure | `static_weibull_theta` | 0.00 | 0.933 | 1.636 | 0.933 | 0.620 |
| group_by_release_method | high_final | `static_ensemble_uncertainty_prior` | 0.25 | 0.305 | 1.357 | 0.905 | 0.774 |
| group_by_release_method | high_final | `static_ensemble_uncertainty_prior` | 0.50 | 0.305 | 1.357 | 0.905 | 0.774 |
| group_by_release_method | high_final | `static_et_direct_grid` | 0.00 | 0.305 | 1.357 | 0.876 | 0.791 |
| group_by_release_method | sustained_mid | `static_curve_dictionary_coeff` | 0.00 | 0.393 | 1.442 | 0.883 | 0.690 |
| group_by_release_method | sustained_mid | `static_ensemble_uncertainty_prior` | 0.00 | 0.319 | 1.153 | 0.867 | 0.600 |
| group_by_release_method | sustained_mid | `static_ensemble_uncertainty_prior` | 0.50 | 0.319 | 1.153 | 0.850 | 0.600 |
| stratified_5fold | anti_burst | `static_curve_dictionary_coeff` | 0.00 | 0.760 | 2.370 | 1.000 | 0.750 |
| stratified_5fold | anti_burst | `static_ensemble_uncertainty_prior` | 0.00 | 0.760 | 2.370 | 1.000 | 0.769 |
| stratified_5fold | anti_burst | `static_ensemble_uncertainty_prior` | 0.25 | 0.760 | 2.370 | 1.000 | 0.769 |
| stratified_5fold | avoid_failure | `static_curve_dictionary_coeff` | 0.00 | 1.000 | 1.277 | 1.000 | 0.912 |
| stratified_5fold | avoid_failure | `static_ensemble_uncertainty_prior` | 0.25 | 1.000 | 1.277 | 1.000 | 0.876 |
| stratified_5fold | avoid_failure | `static_ensemble_uncertainty_prior` | 0.50 | 1.000 | 1.277 | 1.000 | 0.876 |
| stratified_5fold | high_final | `static_et_direct_grid` | 0.00 | 0.760 | 2.351 | 0.680 | 0.822 |
| stratified_5fold | high_final | `static_curve_dictionary_coeff` | 0.00 | 0.720 | 2.205 | 0.680 | 0.819 |
| stratified_5fold | high_final | `static_ensemble_uncertainty_prior` | 0.00 | 0.720 | 2.210 | 0.640 | 0.828 |
| stratified_5fold | sustained_mid | `static_ensemble_uncertainty_prior` | 0.25 | 0.640 | 1.982 | 0.840 | 0.764 |
| stratified_5fold | sustained_mid | `static_ensemble_uncertainty_prior` | 0.50 | 0.640 | 1.982 | 0.880 | 0.764 |
| stratified_5fold | sustained_mid | `static_et_direct_grid` | 0.00 | 0.600 | 1.724 | 0.760 | 0.735 |

## Decision Table

| Split | Target | Question | Value | Passed |
|---|---|---|---:|---:|
| group_by_API | anti_burst | Does static prior beat random/global selection? | -0.046 | False |
| group_by_API | anti_burst | Does dictionary prior beat direct grid? | 0.000 | True |
| group_by_API | anti_burst | Does uncertainty-aware ranking lower burst/failure risk? | 0.200 | True |
| group_by_API | anti_burst | Does strict group split retain design utility? | 0.549 | False |
| group_by_API | avoid_failure | Does static prior beat random/global selection? | 0.267 | True |
| group_by_API | avoid_failure | Does dictionary prior beat direct grid? | 0.000 | True |
| group_by_API | avoid_failure | Does uncertainty-aware ranking lower burst/failure risk? | 0.000 | True |
| group_by_API | avoid_failure | Does strict group split retain design utility? | 1.436 | True |
| group_by_API | high_final | Does static prior beat random/global selection? | 0.046 | True |
| group_by_API | high_final | Does dictionary prior beat direct grid? | 0.015 | True |
| group_by_API | high_final | Does uncertainty-aware ranking lower burst/failure risk? | 0.133 | True |
| group_by_API | high_final | Does strict group split retain design utility? | 0.775 | False |
| group_by_API | sustained_mid | Does static prior beat random/global selection? | 0.165 | True |
| group_by_API | sustained_mid | Does dictionary prior beat direct grid? | 0.067 | True |
| group_by_API | sustained_mid | Does uncertainty-aware ranking lower burst/failure risk? | 0.000 | True |
| group_by_API | sustained_mid | Does strict group split retain design utility? | 1.594 | True |
| group_by_release_method | anti_burst | Does static prior beat random/global selection? | 0.026 | True |
| group_by_release_method | anti_burst | Does dictionary prior beat direct grid? | 0.017 | True |
| group_by_release_method | anti_burst | Does uncertainty-aware ranking lower burst/failure risk? | 0.000 | True |
| group_by_release_method | anti_burst | Does strict group split retain design utility? | 1.761 | True |
| group_by_release_method | avoid_failure | Does static prior beat random/global selection? | 0.095 | True |
| group_by_release_method | avoid_failure | Does dictionary prior beat direct grid? | 0.000 | True |
| group_by_release_method | avoid_failure | Does uncertainty-aware ranking lower burst/failure risk? | -0.005 | False |
| group_by_release_method | avoid_failure | Does strict group split retain design utility? | 1.663 | True |
| group_by_release_method | high_final | Does static prior beat random/global selection? | 0.198 | True |
| group_by_release_method | high_final | Does dictionary prior beat direct grid? | -0.017 | False |
| group_by_release_method | high_final | Does uncertainty-aware ranking lower burst/failure risk? | 0.062 | True |
| group_by_release_method | high_final | Does strict group split retain design utility? | 1.149 | True |
| group_by_release_method | sustained_mid | Does static prior beat random/global selection? | 0.162 | True |
| group_by_release_method | sustained_mid | Does dictionary prior beat direct grid? | 0.195 | True |
| group_by_release_method | sustained_mid | Does uncertainty-aware ranking lower burst/failure risk? | 0.000 | True |
| group_by_release_method | sustained_mid | Does strict group split retain design utility? | 1.442 | True |
| stratified_5fold | anti_burst | Does static prior beat random/global selection? | 0.280 | True |
| stratified_5fold | anti_burst | Does dictionary prior beat direct grid? | 0.000 | True |
| stratified_5fold | anti_burst | Does uncertainty-aware ranking lower burst/failure risk? | 0.000 | True |
| stratified_5fold | avoid_failure | Does static prior beat random/global selection? | 0.160 | True |
| stratified_5fold | avoid_failure | Does dictionary prior beat direct grid? | 0.000 | True |
| stratified_5fold | avoid_failure | Does uncertainty-aware ranking lower burst/failure risk? | 0.040 | True |
| stratified_5fold | high_final | Does static prior beat random/global selection? | 0.480 | True |
| stratified_5fold | high_final | Does dictionary prior beat direct grid? | -0.040 | False |
| stratified_5fold | high_final | Does uncertainty-aware ranking lower burst/failure risk? | 0.000 | True |
| stratified_5fold | sustained_mid | Does static prior beat random/global selection? | 0.320 | True |
| stratified_5fold | sustained_mid | Does dictionary prior beat direct grid? | -0.080 | False |
| stratified_5fold | sustained_mid | Does uncertainty-aware ranking lower burst/failure risk? | 0.040 | True |

## Interpretation

A positive result means static descriptors may still guide experimental prioritization even when they cannot reconstruct exact curves. A weak strict-split result means static design utility is source- or condition-dependent and should not be treated as robust transfer.

## Output Anchor

`../outputs/116_liposome_design_utility_benchmark/`