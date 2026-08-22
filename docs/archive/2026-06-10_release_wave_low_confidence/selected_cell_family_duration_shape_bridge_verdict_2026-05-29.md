# Selected Cell Family Duration/Shape Bridge Verdict

Date: `2026-05-29`

Purpose:

```text
Record the first benchmark-adjacent extension of the family-to-duration/shape
bridge from single curves to representative cells.
```

Primary outputs:

- [cross321_group_by_drug_cell_bridge](D:/release-foundation/outputs/99_release_family_duration_shape_cell_bridge/cross321_group_by_drug_cell_bridge)
- [liposome_group_by_drug_cell_bridge](D:/release-foundation/outputs/99_release_family_duration_shape_cell_bridge/liposome_group_by_drug_cell_bridge)

Code:

- [99_release_family_duration_shape_cell_bridge.py](D:/release-foundation/scripts/99_release_family_duration_shape_cell_bridge.py)

## 1. What is now true

We are no longer limited to a selected-curve claim.

The shared calibrated-family object can now be evaluated at a chosen-cell
level for:

- threshold timing interval coverage
- burst / post-window / residual-tail / tail-AUC interval coverage

This is still not benchmark-wide, but it is the first benchmark-adjacent
evidence tier for the route-consistent duration/shape bridge.

## 2. PLGA chosen cell: `cross321 / group_by_drug`

Outputs:

- [threshold_summary.csv](D:/release-foundation/outputs/99_release_family_duration_shape_cell_bridge/cross321_group_by_drug_cell_bridge/threshold_summary.csv)
- [descriptor_summary.csv](D:/release-foundation/outputs/99_release_family_duration_shape_cell_bridge/cross321_group_by_drug_cell_bridge/descriptor_summary.csv)

Observed pattern:

- `t10` coverage is already high:
  - raw: `0.947`
  - global: `0.947`
  - local: `0.789`
- `t50` is middling:
  - raw: `0.684`
  - global/local: `0.632`
- `t80` is poor:
  - raw: `0.263`
  - global/local: `0.000`

Shape descriptors are much more encouraging:

- burst: `1.0` coverage across raw/global/local
- post-window: `1.0` coverage across raw/global/local
- residual-tail: `0.684` raw, `1.0` global/local
- tail-AUC: `0.737` raw, `0.895` global/local

Interpretation:

> the current bridge looks substantially better for shape-aware uncertainty
> than for high-threshold timing uncertainty.

## 3. Liposome chosen cell: `group_by_drug`

Outputs:

- [threshold_summary.csv](D:/release-foundation/outputs/99_release_family_duration_shape_cell_bridge/liposome_group_by_drug_cell_bridge/threshold_summary.csv)
- [descriptor_summary.csv](D:/release-foundation/outputs/99_release_family_duration_shape_cell_bridge/liposome_group_by_drug_cell_bridge/descriptor_summary.csv)

Observed pattern:

- `t10` coverage: `0.833` across raw/global/local
- `t50` coverage: `0.5` across raw/global/local
- `t80` coverage: `0.167` across raw/global/local

Shape descriptors are stronger:

- all four descriptors reach `1.0` coverage under all three interval modes

Interpretation:

> the liposome chosen-cell bridge again suggests that shape uncertainty is a
> more mature object than deep-threshold timing uncertainty.

## 4. What this means for the gap

This result upgrades the previous state:

- selected-curve bridge: `yes`
- selected-cell bridge: `yes`
- benchmark-wide route-consistent duration UQ: still `no`

So the project has moved one evidence tier upward, but not yet to a fully
audited benchmark layer.

## 5. Why this is still useful

This chosen-cell result tells us two things that matter immediately:

1. the shared family object is not merely capable of emitting duration/shape
   uncertainty in isolated examples
2. the hardest unsolved target is still high-threshold timing, especially
   `t80`, not the entire shape layer

That sharpens the next technical target substantially.

## 6. Best next move

The next sensible move is not "declare the gap solved."

It is:

> add one more representative PLGA cell and one more representative liposome
> cell, then decide whether a reduced benchmark-layer audit is justified.

## Bottom line

The route-consistent duration/shape bridge has now advanced from:

- selected curve

to:

- selected cell

This is real progress toward unified release decision-support, but it still
falls short of a benchmark-wide solved layer.
