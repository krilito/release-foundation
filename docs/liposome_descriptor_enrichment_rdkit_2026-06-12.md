# Liposome Descriptor Enrichment RDKit Audit

Date: 2026-06-12

## Question

Is the static `X -> z` gap caused by weak molecular and interaction descriptors?

## Coverage

- RDKit available: True
- PubChem coverage across API names: 1.000

## Primary Results

| Split | Target | Feature set | z RMSE | decoded RMSE | top20 hit | enrichment |
|---|---|---|---:|---:|---:|---:|
| group_by_API | avoid_failure | `baseline_X` | 56.041 | 17.264 | 0.867 | 1.455 |
| group_by_API | avoid_failure | `baseline_X_plus_interaction_features` | 60.858 | 18.163 | 0.867 | 1.455 |
| group_by_API | avoid_failure | `baseline_X_plus_molecular_descriptors` | 71.285 | 19.097 | 0.667 | 1.050 |
| group_by_API | avoid_failure | `baseline_X_plus_molecular_descriptors_plus_interaction_features` | 72.638 | 19.630 | 0.733 | 1.172 |
| group_by_API | sustained_mid | `baseline_X` | 56.041 | 17.264 | 0.412 | 1.372 |
| group_by_API | sustained_mid | `baseline_X_plus_interaction_features` | 60.858 | 18.163 | 0.273 | 0.594 |
| group_by_API | sustained_mid | `baseline_X_plus_molecular_descriptors` | 71.285 | 19.097 | 0.321 | 1.024 |
| group_by_API | sustained_mid | `baseline_X_plus_molecular_descriptors_plus_interaction_features` | 72.638 | 19.630 | 0.321 | 1.024 |
| group_by_release_method | avoid_failure | `baseline_X_plus_molecular_descriptors` | 46.754 | 13.776 | 0.967 | 1.689 |
| group_by_release_method | avoid_failure | `baseline_X_plus_molecular_descriptors_plus_interaction_features` | 47.496 | 13.886 | 0.967 | 1.689 |
| group_by_release_method | avoid_failure | `baseline_X` | 54.700 | 15.104 | 0.967 | 1.689 |
| group_by_release_method | avoid_failure | `baseline_X_plus_interaction_features` | 55.714 | 14.482 | 0.967 | 1.689 |
| group_by_release_method | sustained_mid | `baseline_X_plus_molecular_descriptors` | 46.754 | 13.776 | 0.233 | 0.856 |
| group_by_release_method | sustained_mid | `baseline_X_plus_molecular_descriptors_plus_interaction_features` | 47.496 | 13.886 | 0.193 | 0.775 |
| group_by_release_method | sustained_mid | `baseline_X` | 54.700 | 15.104 | 0.388 | 1.326 |
| group_by_release_method | sustained_mid | `baseline_X_plus_interaction_features` | 55.714 | 14.482 | 0.421 | 1.535 |
| stratified_5fold | avoid_failure | `baseline_X_plus_molecular_descriptors_plus_interaction_features` | 31.895 | 9.545 | 1.000 | 1.277 |
| stratified_5fold | avoid_failure | `baseline_X` | 33.525 | 9.812 | 1.000 | 1.277 |
| stratified_5fold | avoid_failure | `baseline_X_plus_interaction_features` | 34.162 | 10.196 | 1.000 | 1.277 |
| stratified_5fold | avoid_failure | `baseline_X_plus_molecular_descriptors` | 34.864 | 9.710 | 0.960 | 1.218 |
| stratified_5fold | sustained_mid | `baseline_X_plus_molecular_descriptors_plus_interaction_features` | 31.895 | 9.545 | 0.600 | 1.724 |
| stratified_5fold | sustained_mid | `baseline_X` | 33.525 | 9.812 | 0.600 | 1.724 |
| stratified_5fold | sustained_mid | `baseline_X_plus_interaction_features` | 34.162 | 10.196 | 0.520 | 1.504 |
| stratified_5fold | sustained_mid | `baseline_X_plus_molecular_descriptors` | 34.864 | 9.710 | 0.560 | 1.614 |

## Decision Table

| Split | Target | Question | delta z RMSE | delta design hit | Passed |
|---|---|---|---:|---:|---:|
| group_by_API | avoid_failure | Do molecular descriptors improve X -> z? | 15.243 | -0.200 | False |
| group_by_API | avoid_failure | Do interaction features improve beyond molecular descriptors? | 1.354 | 0.067 | True |
| group_by_API | avoid_failure | Does improvement persist under group_by_API without API identity? | 15.243 | -0.200 | False |
| group_by_API | sustained_mid | Do molecular descriptors improve X -> z? | 15.243 | -0.091 | False |
| group_by_API | sustained_mid | Do interaction features improve beyond molecular descriptors? | 1.354 | 0.000 | False |
| group_by_API | sustained_mid | Does improvement persist under group_by_API without API identity? | 15.243 | -0.091 | False |
| group_by_release_method | avoid_failure | Do molecular descriptors improve X -> z? | -7.946 | 0.000 | True |
| group_by_release_method | avoid_failure | Do interaction features improve beyond molecular descriptors? | 0.742 | 0.000 | False |
| group_by_release_method | sustained_mid | Do molecular descriptors improve X -> z? | -7.946 | -0.155 | True |
| group_by_release_method | sustained_mid | Do interaction features improve beyond molecular descriptors? | 0.742 | -0.040 | False |
| stratified_5fold | avoid_failure | Do molecular descriptors improve X -> z? | 1.339 | -0.040 | False |
| stratified_5fold | avoid_failure | Do interaction features improve beyond molecular descriptors? | -2.969 | 0.040 | True |
| stratified_5fold | sustained_mid | Do molecular descriptors improve X -> z? | 1.339 | -0.040 | False |
| stratified_5fold | sustained_mid | Do interaction features improve beyond molecular descriptors? | -2.969 | 0.040 | True |

## Interpretation Rule

If molecular descriptors improve strict group-by-API rows, the static prior was information-limited by weak molecular representation. If they do not, the missing state is more likely process, morphology, microstructure, batch, or hidden assay information.

Raw API identity is excluded from the main group-by-API comparison. Identity rows are only upper-bound sensitivity checks.

## Output Anchor

`../outputs/117_liposome_descriptor_enrichment_rdkit/`