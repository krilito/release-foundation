# Selected Family Duration/Shape Bridge Verdict

Date: `2026-05-29`

Purpose:

```text
Record the first executable bridge from a calibrated ReleasePosteriorFamily
artifact to duration/shape uncertainty on real curves, while keeping clear
that this is still a selected-curve prototype rather than a benchmark-wide
resolved layer.
```

Primary outputs:

- [cross321_2 bridge](D:/release-foundation/outputs/98_release_family_duration_shape_bridge/cross321_2_bridge)
- [liposome_050 bridge](D:/release-foundation/outputs/98_release_family_duration_shape_bridge/liposome_050_bridge)

Code:

- [98_release_family_duration_shape_bridge.py](D:/release-foundation/scripts/98_release_family_duration_shape_bridge.py)

## 1. What is now true

This is the first local proof that the current shared family object can be
decoded into:

- threshold timing intervals
- burst / post-window / residual-tail / tail-AUC intervals

without falling back to a separate direct-Q or point-only route.

That matters because the previous audited gap was:

> route-consistent uncertainty-aware duration / shape = 0

at the benchmark-cell level.

Reference:

- [route_consistent_uq_duration_shape_gap_2026-05-29.md](D:/release-foundation/docs/route_consistent_uq_duration_shape_gap_2026-05-29.md)

## 2. What was demonstrated

### PLGA example: `cross321_2`

Family source:

- [cross321_2.json](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug/cross321_2.json)

Observed trajectory source:

- [prediction_trajectories.csv](D:/release-foundation/outputs/80_cross321_group_by_drug_theta_rf_ztheta_leaf2/prediction_trajectories.csv)

Bridge outputs:

- [threshold_interval_table.csv](D:/release-foundation/outputs/98_release_family_duration_shape_bridge/cross321_2_bridge/threshold_interval_table.csv)
- [descriptor_interval_table.csv](D:/release-foundation/outputs/98_release_family_duration_shape_bridge/cross321_2_bridge/descriptor_interval_table.csv)

Key observation:

- `t10` is covered under raw/global/local
- `t50` is missed by raw but covered after global/local inflation
- all four descriptor objects are intervalized and covered in this selected
  curve

### Liposome example: `liposome_050`

Family source:

- [liposome_050.json](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_drug/liposome_050.json)

Observed trajectory source:

- [prediction_trajectories.csv](D:/release-foundation/outputs/80_liposome_group_by_api_our_et_refined/prediction_trajectories.csv)

Bridge outputs:

- [threshold_interval_table.csv](D:/release-foundation/outputs/98_release_family_duration_shape_bridge/liposome_050_bridge/threshold_interval_table.csv)
- [descriptor_interval_table.csv](D:/release-foundation/outputs/98_release_family_duration_shape_bridge/liposome_050_bridge/descriptor_interval_table.csv)

Key observation:

- `t10` is intervalized and covered
- descriptor intervals are emitted for all four shape objects
- `t50 / t80` can remain unclosed because the lower family band does not cross
  the threshold in the selected case

That is still useful, because it shows the bridge is real rather than purely
symbolic.

## 3. What this does NOT prove

This prototype does **not** yet justify saying:

1. benchmark-wide route-consistent duration UQ exists
2. calibrated duration/shape reporting is solved
3. the current bridge is publication-ready as a headline result

It only proves:

> the shared family object is now capable of emitting duration/shape
> uncertainty on selected real curves through the actual mechanism decoder.

## 4. Strategic meaning

This is exactly the kind of progress that narrows the biggest remaining moat
gap without pretending it is already closed.

Before this bridge, the status was:

- curve panels: yes
- timing/shape panels: yes
- calibrated-family panels: yes
- route-consistent duration UQ: no

After this bridge, the honest status is:

- benchmark-wide audit: still `no`
- selected executable bridge: now `yes`

That is a real step forward.

## 5. Best next move

The next expansion should not be "more prose about the gap."

It should be:

> generalize the selected-curve bridge into a cell-level bridge for one
> representative PLGA split and one representative liposome split.

That would be the first serious attempt to convert this prototype into a
benchmark-layer result.

## Bottom line

The project now has its first executable route-consistent duration/shape
bridge on real calibrated-family artifacts.

That does not finish the gap, but it moves the project from:

> we know what is missing

to:

> we have started to cross it with real decoder-based outputs.
