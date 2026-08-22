# Expanded Selected Cell Bridge Pattern Verdict

Date: `2026-05-29`

Purpose:

```text
State whether the selected-cell family-duration/shape bridge pattern
still holds after adding the two canonical internal181 PLGA OOD cells.
```

Primary outputs:

- [threshold_cross_cell.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/threshold_cross_cell.csv)
- [descriptor_cross_cell.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/descriptor_cross_cell.csv)
- [pattern_snapshot.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/pattern_snapshot.csv)

Code:

- [101_selected_cell_bridge_summary_v2.py](D:/release-foundation/scripts/101_selected_cell_bridge_summary_v2.py)

## 1. What changed

The chosen-cell bridge no longer stops at:

- `C321-D`
- `C321-P`
- `Lipo-D`
- `Lipo-P`

It now also includes:

- `I181-D`
- `I181-P`

So the pattern is now being judged over a `6-cell` selected panel spanning:

- two PLGA datasets
- two PLGA OOD partition families
- one liposome bridge with two split families

## 2. The repeated pattern still holds

The main repeated finding survives expansion:

> family-based uncertainty is substantially more mature for shape
> descriptors than for deep-threshold timing targets.

Cross-cell means from [pattern_snapshot.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/pattern_snapshot.csv):

### Threshold timing

- global `t10`: mean coverage `0.583`
- global `t50`: mean coverage `0.272`
- global `t80`: mean coverage `0.056`

- local `t10`: mean coverage `0.515`
- local `t50`: mean coverage `0.272`
- local `t80`: mean coverage `0.056`

- raw `t10`: mean coverage `0.458`
- raw `t50`: mean coverage `0.281`
- raw `t80`: mean coverage `0.099`

### Shape descriptors

- global burst: mean coverage `0.864`
- global post-window: mean coverage `0.885`
- global residual-tail: mean coverage `0.986`
- global tail-AUC: mean coverage `0.895`

- local burst: mean coverage `0.744`
- local post-window: mean coverage `0.801`
- local residual-tail: mean coverage `0.993`
- local tail-AUC: mean coverage `0.958`

- raw burst: mean coverage `0.521`
- raw post-window: mean coverage `0.643`
- raw residual-tail: mean coverage `0.721`
- raw tail-AUC: mean coverage `0.689`

The absolute level changes after adding harder PLGA cells, but the ordering does not:

`shape >> deep-threshold timing`

## 3. What the new cells add

The new internal181 cells make two things clearer.

First, `I181-D` is harsh on timing:

- raw `t10`: `0.083`
- raw `t50`: `0.000`
- raw `t80`: `0.000`
- global `t10`: `0.333`

But even there, shape descriptors are materially more stable:

- global residual-tail: `0.958`
- global tail-AUC: `0.833`
- local residual-tail: `1.000`
- local tail-AUC: `1.000`

Second, `I181-P` keeps the same pattern:

- global/local `t10`: `0.500`
- all `t50/t80` variants: `0.000`
- descriptor coverage still stays much higher, especially for:
  - global residual-tail: `0.958`
  - global tail-AUC: `0.958`
  - local residual-tail: `0.958`
  - local tail-AUC: `0.958`

That means the earlier four-cell signal was not just a liposome-vs-PLGA accident.

## 4. What this now justifies

We can now say, with stronger evidence than before:

1. the selected-cell bridge pattern survives beyond one PLGA dataset
2. the current route-consistent family bridge is already informative for
   shape-aware uncertainty
3. the hardest unresolved bottleneck remains deep-threshold timing,
   especially `t80`-style targets

## 5. What this still does not justify

This still does **not** justify saying:

1. benchmark-wide route-consistent duration UQ is established
2. timing uncertainty is broadly solved across PLGA OOD families
3. liposome and PLGA are now equally mature in the bridge layer

## Bottom line

The selected-cell bridge pattern survives expansion to a `6-cell` panel:

> shape-aware uncertainty continues to look substantially closer than
> deep-threshold timing uncertainty, and the new internal181 cells make that
> gap even harder to dismiss.
