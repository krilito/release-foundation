# Liposome Author Baseline Audit

Date: 2026-06-12

This note records the public baseline from Yanes et al., "A machine learning
workflow to accelerate the design of in vitro release tests from liposomes"
([paper DOI](https://doi.org/10.1039/D5DD00112A);
[GitHub](https://github.com/danielyanes22/accelerated_IVR)).

## Bottom Line

The original authors do have open code, and we reproduced their classifier
evaluation locally from:

`data/external/accelerated_IVR/repo/accelerated_IVR-main`

Command:

```powershell
python -m experiments.testing_scores
```

The strongest reproduced local classifier was:

| Model | Test balanced accuracy | Test F1 | Test MCC |
|---|---:|---:|---:|
| XGBClassifier | 0.697 | 0.702 | 0.547 |
| KNeighborsClassifier | 0.695 | 0.713 | 0.564 |
| RandomForestClassifier | 0.677 | 0.676 | 0.506 |

This is a mandatory author baseline, but it is not the same endpoint as our
liposome route. Their public model predicts a discrete kinetic class from
static features. Our current question is continuous curve prediction under
strict transfer splits and early-observation budgets.

## What Their Model Predicts

Their pipeline converts release curves into kinetic classes and then trains
classifiers to predict those classes.

In operational terms:

1. compile liposome IVR profiles from literature,
2. fit kinetic models to release curves,
3. keep Weibull parameters for acceptable fits,
4. cluster Weibull-derived release behavior with PCA/KMeans,
5. attach the cluster label back to static formulation/IVR descriptors,
6. train classifiers to predict the cluster label.

The actual reproduced script `experiments/testing_scores.py` reads:

`data/clean/ML_7_features_df.csv`

The local file has `77` rows and these columns:

```text
media_pH, media_temp_oC, drug_loading, Z_average_nm,
API_type, weighted_Mw, weighted_Tm, cluster
```

So the supervised task is:

```text
7 static numeric descriptors -> kinetic class
```

It is not:

```text
metadata + early Q(t) -> future continuous Q(t)
```

## Their Training Logic

The reproduced author script does the following:

| Stage | Actual code behavior | Strict note |
|---|---|---|
| Target | `cluster` label from the authors' kinetic-class pipeline | Useful kinetic summary, not measured mechanism |
| Features | all columns except `cluster` from `ML_7_features_df.csv` | Static-only, no early release observations |
| Scaling | `StandardScaler().fit_transform(X)` before CV | Minor preprocessing leakage under strict ML practice |
| Label encoding | `LabelEncoder().fit_transform(y)` | Fine for class labels |
| Split | `StratifiedKFold(n_splits=5, shuffle=True, random_state=15)` | Easier than group-by-API or group-by-method transfer |
| Models | DT, SVC, GaussianNB, KNN, LogisticRegression, RF, XGB | Standard classifiers |
| Metrics | balanced accuracy, micro F1, MCC | Classification metrics, not curve RMSE |
| Outputs | classifier pickles and train/test CV CSVs | External artifacts remain untracked |

One small repository inconsistency: `experiments/__init__.py` says
`testing_scores.py` uses `ML_9_features_df.csv`, but the actual executable
script reads `ML_7_features_df.csv`. We treat executed code as authoritative.

## Why This Matters For Us

If our liposome result only matches this author baseline after converting it
to curve space, then our liposome work is probably just reproducing their
static kinetic-class story.

If our early-observation E3 improves over the converted author baseline under
the same group splits, then the claim becomes stronger:

```text
static liposome formulation/IVR descriptors give a coarse kinetic class,
but measured early release exposes additional curve-specific state.
```

This mirrors the frozen PLGA story without pretending the mechanisms are the
same.

## Required Fair Conversion

The next baseline should convert their classifier into a curve-level predictor:

```text
static features -> predicted kinetic class
predicted class -> train-fold median Weibull alpha/beta
alpha/beta -> predicted Q(t)
```

It must be scored with the same target curves and split families as our
liposome E1/E3 experiments.

Required variants:

| Baseline | Split | Role |
|---|---|---|
| `author_stratified_class_proto` | stratified 5-fold | paper-like diagnostic |
| `author_group_by_API_class_proto` | hold out API groups | strict API-transfer baseline |
| `author_group_by_release_method_class_proto` | hold out method groups | strict assay-shift baseline |
| `author_oracle_class_proto` | true test class label | class-prototype upper bound, not deployable |

## Audit Outputs

Generated locally under:

`outputs/111_liposome_author_baseline_audit/`

Key files:

| File | Role |
|---|---|
| `author_repo_audit.csv` | repo URL, paper URL, license/dependency metadata |
| `author_dataset_summary.csv` | local data shapes and cluster counts |
| `author_classifier_cv_summary.csv` | reproduced train/test classifier metrics |
| `author_classifier_fold_results.csv` | per-fold reproduced classifier results |
| `author_training_logic.csv` | row-wise explanation of their prediction/training logic |
| `author_curve_baseline_plan.csv` | contract for the next curve-space author baseline |
| `data_checks.csv` | pass/fail integrity checks |
| `lock_metadata.json` | script, seed, git hash, paths |
| `report.md` | generated human-readable audit report |

All checks passed in `data_checks.csv`.

## Decision

Before liposome E3 is interpreted, implement the curve-space author baseline.
The author reproduction alone is not enough, because it scores class labels
instead of future release curves.
