# Timescale Layer Verdict

Date: `2026-05-29`

Purpose:

```text
Record what changed once release timescale was evaluated as a first-class
forecast object rather than inferred indirectly from curve-fit summaries.
```

Key artifacts:

- [scripts/81_release_timescale_eval.py](D:/release-foundation/scripts/81_release_timescale_eval.py)
- [outputs/81_liposome_group_by_api_our_et_refined](D:/release-foundation/outputs/81_liposome_group_by_api_our_et_refined)
- [outputs/81_liposome_group_by_api_direct_et_q](D:/release-foundation/outputs/81_liposome_group_by_api_direct_et_q)
- [outputs/81_cross321_group_by_drug_theta_rf_ztheta_leaf2](D:/release-foundation/outputs/81_cross321_group_by_drug_theta_rf_ztheta_leaf2)
- [outputs/81_cross321_group_by_drug_direct_et_q](D:/release-foundation/outputs/81_cross321_group_by_drug_direct_et_q)

## 1. What this layer adds

The previous timescale reporting in the shared benchmark was useful but still
mixed together:

1. thresholds reached before the early-observation window
2. thresholds that actually need forecasting after the early window

That matters because a method can look good on `t10` or `t50` simply because
those thresholds often happen in the already-observed region.

Script `81` now separates:

- `all thresholds`
- `forecast-relevant only`, defined as `t_obs > early window`

## 2. Liposome result under `group_by_API`

### Mechanism route

From [summary.txt](D:/release-foundation/outputs/81_liposome_group_by_api_our_et_refined/summary.txt):

- all-threshold `t10` MAE `0.553 h`
- all-threshold `t50` MAE `5.452 h`
- all-threshold `t80` MAE `18.700 h`

Forecast-relevant only:

- `t10`: `n = 0`
- `t50`: `n = 2`, MAE `50.280 h`
- `t80`: `n = 10`, MAE `23.911 h`

### Direct curve route

From [summary.txt](D:/release-foundation/outputs/81_liposome_group_by_api_direct_et_q/summary.txt):

- all-threshold `t10` MAE `1.046 h`
- all-threshold `t50` MAE `2.220 h`
- all-threshold `t80` MAE `34.335 h`

Forecast-relevant only:

- `t10`: `n = 0`
- `t50`: `n = 0`
- `t80`: `n = 4`, MAE `42.026 h`

## 3. What this means

This changes the interpretation in an important way.

### 3.1 The old `t50` comparison was partly misleading

Before this layer, the direct route looked better on liposome `t50` because
the all-threshold summary said:

- mechanism `5.45 h`
- direct `2.22 h`

But once we restrict to thresholds that actually happen after the early
window, that comparison almost disappears:

- mechanism has only `2` forecast-relevant `t50` curves
- direct has `0`

So the earlier `t50` advantage for direct-Q should **not** be interpreted as
"direct-Q is better at forecasting release duration". It mostly reflects
thresholds reached in or before the observed prefix.

### 3.2 The real future-forecast object is much harder

For liposome, the first threshold that clearly remains a forecasting problem
after the `24 h` window is usually `t80`, not `t10`.

On that object:

- mechanism route future-only `t80` MAE is `23.911 h`
- direct-Q future-only `t80` MAE is `42.026 h`

That is much more aligned with the broader curve-fit story: the mechanism
route remains the stronger future-forecast route.

### 3.3 Release duration is not solved yet

Even the better mechanism route still has large future-only timing error:

- future-only `t50` MAE `50.280 h` on a tiny valid subset
- future-only `t80` MAE `23.911 h`

So this layer strengthens the project scientifically, but it also forces a
more honest claim boundary:

> The framework now explicitly targets release timescale, but robust
> cross-mechanism timescale prediction is still an open problem, not a solved
> result.

## 4. Strategic implication

This is exactly why `timescale-first` needed to become its own queue item.

Without this layer, the project would be tempted to overread curve-fit gains
as practical control-release-duration gains. With this layer, we can now say:

1. the project has started to evaluate release duration explicitly
2. the evaluation is more honest than a raw curve-only story
3. future threshold timing still needs dedicated algorithmic work

## 5. Expanded PLGA panel says the ranking split is real, not anecdotal

To avoid overfitting the interpretation to liposome, the same shared pipeline
was used on four canonical PLGA OOD cells:

- [cross321 / group_by_drug](D:/release-foundation/outputs/83_plga_prediction_export/cross321/group_by_drug/formulation_plus_early/theta_RF_ztheta_leaf2/summary.txt)
- [cross321 / group_by_polymer](D:/release-foundation/outputs/83_plga_prediction_export/cross321/group_by_polymer/formulation_plus_early/theta_early_select/summary.txt)
- [internal181 / group_by_drug](D:/release-foundation/outputs/83_plga_prediction_export/internal181/group_by_drug/formulation_plus_early/theta_early_select/summary.txt)
- [internal181 / group_by_polymer](D:/release-foundation/outputs/83_plga_prediction_export/internal181/group_by_polymer/formulation_plus_early/theta_early_select/summary.txt)

Across those four cells:

- mechanism route wins median curve `R²` in `4/4`
- mechanism route wins pooled `R²` in `1/4`
- mechanism route wins future-only `t10` MAE in `2/4`
- mechanism route wins future-only `t50` MAE in `1/4`
- mechanism route wins future-only `t80` MAE in `0/4`

The detailed table is now centralized in
[plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv).

That means the cross-mechanism truth is even more important than the liposome
warning alone:

> route rankings are not stable across evaluation objects.

A method can win:

- median curve fit

but lose:

- pooled fit
- future threshold timing

or vice versa.

This is not just a liposome quirk and not just one PLGA exception. It now
appears across multiple canonical PLGA OOD cells as well.

## Bottom line

The timescale layer was worth adding.

It did two things at once:

- made the project more aligned with the real scientific/product question
- prevented an overclaim that any single route had already "won" release
  timescale prediction in general

That is good evidence discipline, not a negative result.

## 6. Shape descriptors now sharpen the same point

After the dedicated threshold-timing layer was working, the same script was
extended to emit:

- `burst_fraction_at_early_window`
- `post_window_release`
- `residual_tail_at_tmax`
- `tail_auc_after_early_window`

These are not all equally "predictive" objects:

- `burst_fraction_at_early_window` is partly descriptive because the early
  window is often observed
- the other three are much closer to the real sustained-release question

The aggregated panel in
[plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)
and
[liposome_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv)
shows the same deeper lesson:

- inside PLGA, mechanism routing is usually better on `post-window release`
  and `residual tail`, but not consistently on `tail AUC`
- inside liposome, even those shape wins depend on the split family

So the broader verdict is now stronger:

> There is no single current route that can honestly be called the universal
> winner on curve fit, threshold timing, and release-shape descriptors at the
> same time.
