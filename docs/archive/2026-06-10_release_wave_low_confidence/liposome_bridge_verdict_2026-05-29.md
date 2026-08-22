# Liposome Bridge Verdict

Date: `2026-05-29`

Purpose:

```text
Record the first non-PLGA bridge result after the shared release benchmark API
and liposome intake landed.
```

This is not a polished paper note. It is the evidence ledger for whether the
project is actually moving from PLGA-only release prediction to a broader
release-intelligence framework.

## 1. What was built

### Shared benchmark base

- [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)

### Liposome intake

- [scripts/78_liposome_ivr_intake.py](D:/release-foundation/scripts/78_liposome_ivr_intake.py)
- [outputs/78_liposome_ivr_intake](D:/release-foundation/outputs/78_liposome_ivr_intake)

Key retained dataset facts from
[summary.txt](D:/release-foundation/outputs/78_liposome_ivr_intake/summary.txt):

- `33` retained curves
- `293` observed points
- `8` APIs
- `3` release methods
- cluster mix: `Fast 4`, `Medium 19`, `Slow 10`

### Shared-format liposome predictions

- [scripts/79_liposome_prediction_export.py](D:/release-foundation/scripts/79_liposome_prediction_export.py)

Methods exported under `group_by_API`:

- [our_ET_weibull_theta_early_refined](D:/release-foundation/outputs/79_liposome_prediction_export/group_by_API/our_ET_weibull_theta_early_refined/summary.txt)
- [direct_ET_Q_grid](D:/release-foundation/outputs/79_liposome_prediction_export/group_by_API/direct_ET_Q_grid/summary.txt)

### Shared benchmark evaluation outputs

- [outputs/80_liposome_group_by_api_our_et_refined](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined)
- [outputs/80_liposome_group_by_api_direct_et_q](D:/release-foundation/outputs/80_liposome_group_by_api_direct_et_q)
- [outputs/80_liposome_group_by_method_our_et_refined](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined)
- [outputs/80_liposome_group_by_method_direct_et_q](D:/release-foundation/outputs/80_liposome_group_by_method_direct_et_q)

## 2. Main numerical result

Under the shared benchmark API, using `group_by_API` held-out groups:

### Mechanism route

From
[metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv):

- pooled `R² = 0.9031`
- pooled `RMSE = 0.0849`
- pooled `MAE = 0.0483`
- median curve `R² = 0.8885`
- fraction of curves with `R² >= 0` = `0.9091`

From
[timescale_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/timescale_summary.csv):

- `t10` MAE `0.55 h`
- `t50` MAE `5.45 h`
- `t80` MAE `18.70 h`

### Direct curve route

From
[metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_direct_et_q/metrics_summary.csv):

- pooled `R² = 0.8640`
- pooled `RMSE = 0.1005`
- pooled `MAE = 0.0720`
- median curve `R² = 0.8671`
- fraction of curves with `R² >= 0` = `0.9394`

From
[timescale_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_api_direct_et_q/timescale_summary.csv):

- `t10` MAE `1.05 h`
- `t50` MAE `2.22 h`
- `t80` MAE `34.34 h`

## 3. What this means

### 3.1 Good news

The project now has a **real non-PLGA mechanism path** running through the
shared release benchmark language:

- aligned intake
- shared split metadata
- shared prediction schema
- shared pooled/curve metrics
- shared timescale metrics

That is a real step toward unified release intelligence. It is no longer just
a PLGA architecture story.

### 3.2 More important news

The liposome bridge is **not** a clean replay of the PLGA story.

On liposome `group_by_API`, the mechanism-routed Weibull path is stronger on:

- pooled fit
- pooled error
- median curve `R²`

But it is **not automatically stronger** on all timescale metrics:

- direct-Q is better on `t50`
- mechanism route is better on `t10`
- `t80` is unstable for both, with limited valid curves

So the honest conclusion is:

> The mechanism-routed path survives the first non-PLGA benchmark and remains
> competitive to strong on full-curve prediction, but its advantage does not
> transfer uniformly to release-timescale estimation.

## 3.2 Split robustness check

The `group_by_release_method` split adds a useful second view.

From
[metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv):

- pooled `R² = 0.9031`
- pooled `RMSE = 0.0849`
- pooled `MAE = 0.0483`
- median curve `R² = 0.8885`

From
[metrics_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_direct_et_q/metrics_summary.csv):

- pooled `R² = 0.4570`
- pooled `RMSE = 0.2009`
- pooled `MAE = 0.1563`
- median curve `R² = 0.5844`

This matters because it shows:

- the mechanism-routed route is at least stable across both liposome split families tested so far
- the direct-Q route is more fragile to the split definition
- the first non-PLGA bridge does **not** support the claim that direct-Q is the generally safer cross-mechanism default

### 3.3 Timescale nuance remains

Even with that robustness advantage, timescale is still mixed.

From
[timescale_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_our_et_refined/timescale_summary.csv):

- `t10` MAE `0.55 h`
- `t50` MAE `5.45 h`
- `t80` MAE `18.70 h`

From
[timescale_summary.csv](D:/release-foundation/outputs/80_liposome_group_by_method_direct_et_q/timescale_summary.csv):

- `t10` MAE `2.85 h`
- `t50` MAE `2.55 h`
- `t80` valid curves only `2`

So the best current summary is:

> mechanism routing looks more stable for full-curve prediction across splits,
> while release-timescale estimation still needs its own dedicated design and
> should not be assumed solved as a side effect of better curve fit.

### 3.4 Why the negative future-R² in the export summaries does not overturn this

The export summaries in
[fold_curve_metadata.csv](D:/release-foundation/outputs/79_liposome_prediction_export/group_by_API/our_ET_weibull_theta_early_refined/fold_curve_metadata.csv)
and
[fold_curve_metadata.csv](D:/release-foundation/outputs/79_liposome_prediction_export/group_by_API/direct_ET_Q_grid/fold_curve_metadata.csv)
show very negative median `future_R²`.

That metric is still useful, but it is fragile here because:

- many liposome curves have very few future points after the `24 h` early window
- some held-out curves have low future variance, making `R²` unstable
- the shared benchmark pooled/curve metrics use the full observed trajectory

So for the bridge verdict, the shared benchmark tables are the stronger
evidence.

## 4. Claim impact

### Now stronger than before

We can now say:

> The project has a shared release benchmark interface that already runs on
> PLGA/chitosan assets and a non-PLGA liposome mechanism family.

### Still not legal

We still cannot say:

- `cross-mechanism validation complete`
- `mechanism route dominates direct-Q across mechanisms`
- `timescale prediction solved across mechanisms`
- `universal drug-release model`

## 5. What to do next

The next high-value steps are now clearer:

1. run at least one more liposome scheme through the shared benchmark
   preferably `group_by_release_method`
2. add a compact comparison table combining PLGA and liposome
3. add uncertainty for liposome if we want a real cross-mechanism UQ claim
4. continue the chitosan prospective path as the locked stress test

## Bottom line

The liposome bridge worked.

But it worked in the way that matters scientifically:

- it made the project more general,
- it made the claim set more honest,
- and it showed that cross-mechanism success will not be a simple replay of
  PLGA.
