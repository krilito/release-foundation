# PLGA Missing-State Residual Map

Date: 2026-06-13

## Question

What release-state residual remains when static PLGA descriptors attempt to predict curve-derived release states?

This is not a candidate-measurement ranking experiment. It freezes the residual map required before RSF/MSVS.

## Protocol

- Sample unit is one formulation curve.
- Real ragged curves are interpolated to one shared 0-90 day grid.
- State spaces: PCA-8, PCA-3, and Weibull log-theta.
- Static priors: global mean, ExtraTrees, Ridge sanity check.
- Splits: random curve folds, group-by-DOI, and group-by-formulation-method.
- DOI is excluded from static features and used only as a source-structure check.

## Best Rows

| split                       | state_space   | prior_model   |   z_rmse |   decoded_curve_rmse |   high_residual_rate |
|:----------------------------|:--------------|:--------------|---------:|---------------------:|---------------------:|
| group_by_DOI                | pca3          | global_mean   |   0.9602 |               0.1842 |               0.2188 |
| group_by_DOI                | pca8          | global_mean   |   0.6042 |               0.1911 |               0.2188 |
| group_by_DOI                | weibull       | global_mean   |   1.324  |               0.2531 |               0.2188 |
| group_by_Formulation_Method | pca3          | global_mean   |   0.9598 |               0.1849 |               0.1875 |
| group_by_Formulation_Method | pca8          | global_mean   |   0.6071 |               0.1931 |               0.1797 |
| group_by_Formulation_Method | weibull       | global_mean   |   1.3855 |               0.2563 |               0.1328 |
| random_5fold                | pca3          | extra_trees   |   0.7915 |               0.1512 |               0.6406 |
| random_5fold                | pca8          | extra_trees   |   0.4975 |               0.1567 |               0.6172 |
| random_5fold                | weibull       | extra_trees   |   1.106  |               0.1906 |               0.4688 |

## Representation Agreement

| split                       | prior_model   |   mean_spearman |   mean_jaccard |   n_pairs |
|:----------------------------|:--------------|----------------:|---------------:|----------:|
| group_by_DOI                | extra_trees   |          0.6017 |         0.8479 |        15 |
| group_by_DOI                | global_mean   |          0.6383 |         0.5353 |        15 |
| group_by_DOI                | ridge         |          0.5626 |         0.9347 |        15 |
| group_by_Formulation_Method | extra_trees   |          0.4848 |         0.7556 |         9 |
| group_by_Formulation_Method | global_mean   |          0.4505 |         0.5764 |         9 |
| group_by_Formulation_Method | ridge         |          0.6213 |         0.9019 |         9 |
| random_5fold                | extra_trees   |          0.8089 |         0.7535 |        15 |
| random_5fold                | global_mean   |          0.6112 |         0.4549 |        15 |
| random_5fold                | ridge         |          0.8361 |         0.7931 |        15 |

## Source Structure

| split                       | state_space   | prior_model   | source_column      |   eta_squared |   shuffle_null95 |
|:----------------------------|:--------------|:--------------|:-------------------|--------------:|-----------------:|
| group_by_DOI                | pca3          | extra_trees   | DOI                |        0.348  |           0.3056 |
| group_by_DOI                | pca3          | ridge         | Formulation Method |        0.0862 |           0.0473 |
| group_by_DOI                | pca8          | extra_trees   | DOI                |        0.332  |           0.3179 |
| group_by_DOI                | pca8          | ridge         | DOI                |        0.2799 |           0.2774 |
| group_by_DOI                | pca8          | ridge         | Formulation Method |        0.0812 |           0.0491 |
| group_by_DOI                | weibull       | extra_trees   | DOI                |        0.4019 |           0.2791 |
| group_by_DOI                | weibull       | extra_trees   | Drug               |        0.3866 |           0.2508 |
| group_by_DOI                | weibull       | global_mean   | DOI                |        0.3968 |           0.2701 |
| group_by_DOI                | weibull       | global_mean   | Formulation Method |        0.0965 |           0.0418 |
| group_by_DOI                | weibull       | global_mean   | Drug               |        0.3781 |           0.2418 |
| group_by_DOI                | weibull       | ridge         | DOI                |        0.3522 |           0.284  |
| group_by_DOI                | weibull       | ridge         | Formulation Method |        0.0657 |           0.0494 |
| group_by_DOI                | weibull       | ridge         | Drug               |        0.3372 |           0.2624 |
| group_by_Formulation_Method | pca3          | ridge         | DOI                |        0.8362 |           0.3233 |
| group_by_Formulation_Method | pca3          | ridge         | Drug               |        0.8453 |           0.2874 |
| group_by_Formulation_Method | pca8          | ridge         | DOI                |        0.8384 |           0.3174 |
| group_by_Formulation_Method | pca8          | ridge         | Drug               |        0.8472 |           0.2859 |
| group_by_Formulation_Method | weibull       | extra_trees   | DOI                |        0.3863 |           0.2563 |
| group_by_Formulation_Method | weibull       | extra_trees   | Formulation Method |        0.0502 |           0.0419 |
| group_by_Formulation_Method | weibull       | extra_trees   | Drug               |        0.303  |           0.2411 |
| group_by_Formulation_Method | weibull       | global_mean   | DOI                |        0.4271 |           0.3027 |
| group_by_Formulation_Method | weibull       | global_mean   | Formulation Method |        0.0929 |           0.0469 |
| group_by_Formulation_Method | weibull       | global_mean   | Drug               |        0.4001 |           0.2649 |
| group_by_Formulation_Method | weibull       | ridge         | DOI                |        0.9118 |           0.325  |
| group_by_Formulation_Method | weibull       | ridge         | Drug               |        0.9138 |           0.2817 |
| random_5fold                | pca3          | ridge         | DOI                |        0.3331 |           0.3052 |
| random_5fold                | pca3          | ridge         | Drug               |        0.3042 |           0.2738 |
| random_5fold                | pca8          | ridge         | DOI                |        0.336  |           0.2854 |
| random_5fold                | pca8          | ridge         | Drug               |        0.3102 |           0.2753 |
| random_5fold                | weibull       | extra_trees   | DOI                |        0.3378 |           0.2967 |
| random_5fold                | weibull       | extra_trees   | Formulation Method |        0.0773 |           0.0386 |
| random_5fold                | weibull       | extra_trees   | Drug               |        0.3099 |           0.2733 |
| random_5fold                | weibull       | global_mean   | DOI                |        0.3925 |           0.269  |
| random_5fold                | weibull       | global_mean   | Formulation Method |        0.0963 |           0.0446 |
| random_5fold                | weibull       | global_mean   | Drug               |        0.3673 |           0.2599 |
| random_5fold                | weibull       | ridge         | DOI                |        0.3242 |           0.304  |
| random_5fold                | weibull       | ridge         | Formulation Method |        0.0643 |           0.0517 |
| random_5fold                | weibull       | ridge         | Drug               |        0.287  |           0.2818 |

## Decision Table

| decision                                 | status   | evidence                                                                                |
|:-----------------------------------------|:---------|:----------------------------------------------------------------------------------------|
| state_representation_outputs_exist       | pass     | summary rows=27; pca baseline rows=6                                                    |
| static_X_beats_global_state_prior        | pass     | best z_rmse gain vs global_mean=0.1945                                                  |
| strict_group_split_static_signal_exists  | warn     | best strict z_rmse gain vs global_mean=-0.0144                                          |
| residuals_stable_across_representations  | pass     | mean Spearman=0.668; mean high-residual Jaccard=0.831                                   |
| source_or_method_structure_present       | warn     | source/method residual associations exceeding shuffle null=38; DOI hits=16              |
| no_MSVS_or_candidate_measurement_scoring | pass     | 120 outputs residual maps and source structure only; measurement scoring starts in 121. |
| data_checks_pass                         | pass     | failed checks=[]                                                                        |

## Output Anchor

`../outputs/120_plga_missing_state_residual_map/`
