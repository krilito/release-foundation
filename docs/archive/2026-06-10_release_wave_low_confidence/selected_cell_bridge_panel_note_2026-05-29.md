# Selected-Cell Bridge Panel Note

Date: `2026-05-29`

Purpose:

```text
Attach one direct visual support figure to the selected-cell family bridge
story so the main pattern is no longer only stated in prose:
shape-aware uncertainty is materially more mature than deep-threshold
timing uncertainty.
```

Primary figure:

- [selected_cell_bridge_panel.png](D:/release-foundation/outputs/92_release_strategy_figures/selected_cell_bridge/selected_cell_bridge_panel.png)

Supporting tables:

- [pattern_snapshot.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/pattern_snapshot.csv)
- [threshold_cross_cell.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/threshold_cross_cell.csv)
- [descriptor_cross_cell.csv](D:/release-foundation/outputs/101_selected_cell_bridge_summary_v2/descriptor_cross_cell.csv)

## What the panel makes visible

### Panel A

The cross-cell means show the main ordering directly:

- threshold timing:
  - global `t10 / t50 / t80 = 0.583 / 0.272 / 0.056`
- shape descriptors:
  - global `burst / post-window / residual-tail / tail-AUC = 0.864 / 0.885 / 0.986 / 0.895`

So the current bridge is already informative for:

- `burst`
- `post-window`
- `residual-tail`
- `tail-AUC`

but is still weak for:

- `t50`
- especially `t80`

### Panel B

The per-cell heatmap shows the same pattern is not coming from one lucky cell.

- `C321-D` is the strongest timing case
- `C321-P`, `I181-D`, and `I181-P` all collapse much earlier on threshold timing
- `Lipo-D` and `Lipo-P` still keep strong shape coverage even when timing is only moderate

That is exactly why the current honest statement is:

`shared shape-aware target layer is already much closer than universal deep-threshold timing layer.`

## What this panel supports

This figure supports saying:

1. selected-cell route-consistent bridge evidence now has a direct visual summary
2. shape-aware uncertainty is substantially more mature than deep-threshold timing
3. universal `t80` remains an illegal abstraction under the current evidence

## What it still does not support

It still does **not** support saying:

1. benchmark-wide route-consistent duration UQ is solved
2. timing targets are uniformly mature across mechanisms
3. shape-aware target layer is already complete rather than just ahead

## Related entry points

- [expanded_selected_cell_bridge_pattern_verdict_2026-05-29.md](D:/release-foundation/docs/expanded_selected_cell_bridge_pattern_verdict_2026-05-29.md)
- [threshold_target_redesign_verdict_2026-05-29.md](D:/release-foundation/docs/threshold_target_redesign_verdict_2026-05-29.md)
- [release_shape_target_contract_2026-05-29.md](D:/release-foundation/docs/release_shape_target_contract_2026-05-29.md)
