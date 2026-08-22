# Original LAI Author Model Audit

Date: 2026-06-11

External repository:

`D:\BaiduNetdisk\long-acting-injectables-main`

Audit script:

`scripts/100_lai_author_model_audit.py`

Fair reproduction on our splits:

`scripts/101_lai_author_style_on_our_splits.py`

Output:

`outputs/100_lai_author_model_audit/`

## What The Original Model Is

The original project for "Machine Learning Models to Accelerate the Design of
Polymeric Long-Acting Injectables" is a tabular point-regression benchmark.
It does not model a release curve as a continuous function.

It trains models of the form:

```text
features at a query time -> release fraction at that time
```

There are two released regimes:

- zero-shot: static descriptors + `Time`
- few-shot: static descriptors + `Time` + fixed early releases
  `T=0.25`, `T=0.5`, `T=1.0`

Both datasets contain 3783 point rows, 181 experimental curves, and 34
drug-polymer groups.

## Original Evaluation Protocol

The released `NESTED_CV_.py` uses:

- outer split: `GroupShuffleSplit(test_size=0.2)` by `DP_Group`
- inner split: `GroupKFold(n_splits=10)` by `DP_Group`
- tuning metric: negative MAE
- headline stored metric: point-level MAE

This is better than random point splitting, but it is not the same as our
source-dataset or source-group transfer splits.

## Released Nested-CV Results

Best author models from released CV pickles:

| Regime | Best model | Median test MAE | Median point RMSE | Median curve RMSE |
|---|---|---:|---:|---:|
| few-shot | LGBM | 0.112 | 0.158 | 0.134 |
| few-shot | NGB | 0.113 | 0.155 | 0.140 |
| few-shot | RF | 0.116 | 0.159 | 0.135 |
| zero-shot | LGBM | 0.158 | 0.210 | 0.209 |
| zero-shot | NGB | 0.163 | 0.221 | 0.205 |
| zero-shot | RF | 0.170 | 0.222 | 0.201 |

The key comparison is not the absolute number, because author MAE is pointwise
and our current reports emphasize future-only per-curve RMSE. The key result
is the gap:

```text
zero-shot LGBM median MAE 0.158
few-shot LGBM median MAE  0.112
gain from fixed early points ~0.046 MAE
```

## Relation To Our Current Route

This independently supports our current thesis:

Early release observations are not a small detail. They are a major information
source.

Our script 97 found the same pattern under our stricter future-only scoring:

- source-group k=5 static direct RMSE: 0.267
- source-group k=5 static+early direct RMSE: 0.160
- source-dataset k=5 static direct RMSE: 0.272
- source-dataset k=5 static+early direct RMSE: 0.166

Our script 99 then tested whether static descriptors could synthesize those
early observations:

- source-group k=5 predicted-early RMSE: 0.256
- source-dataset k=5 predicted-early RMSE: 0.290

So the original author model and our probes agree on the important direction:

```text
static descriptors alone are weak;
measured early release points reveal missing state.
```

## Practical Caveats

The released trained RF pickle is not portable under the current sklearn
version, because the tree node dtype changed. The nested-CV result pickles load
cleanly as pandas DataFrames and are sufficient for auditing reported behavior.

The author prediction notebooks also fit `StandardScaler` on the original full
dataset before scaling prediction rows. For a fair benchmark on our splits, we
should retrain author-style models inside each fold rather than use the saved
full-data trained pickle.

## Decision

Use the original model as a baseline family, not as a direct score to beat.

We reproduced the author-style feature sets on our exact split assignments in
`scripts/101_lai_author_style_on_our_splits.py`.

The reproduction uses released LGBM-style hyperparameters, retrains inside each
of our folds, and scores only points later than 1.0 day so the fixed early
features cannot trivially predict themselves.

| Split | Author zero-shot RMSE | Author fixed-early RMSE | Gain |
|---|---:|---:|---:|
| random-kfold | 0.109 | 0.112 | -0.003 |
| source-group-kfold | 0.205 | 0.144 | 0.061 |
| source-dataset-lodo | 0.218 | 0.147 | 0.071 |

This is strong independent support for the project direction: fixed early
observations are especially valuable under source shift.

## Next Baseline Step

If we continue this line, compare three families on the same folds and the same
future-only target set:

1. zero-shot author-style: static descriptors + query time.
2. few-shot author-style: static descriptors + query time + fixed/interpolated
   early releases at 0.25, 0.5, 1.0 days.
3. our dynamic-budget direct early-Q controls from script 97.
4. middle-layer selector from script 96.
