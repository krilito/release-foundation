# Full Cell Bridge Panel Verdict

Date: `2026-05-29`

Primary figure:

- [full_cell_bridge_panel.png](D:/release-foundation/outputs/92_release_strategy_figures/full_cell_bridge/full_cell_bridge_panel.png)

Supporting tables:

- [pattern_snapshot.csv](D:/release-foundation/outputs/117_full_cell_bridge_summary/pattern_snapshot.csv)
- [threshold_cross_cell.csv](D:/release-foundation/outputs/117_full_cell_bridge_summary/threshold_cross_cell.csv)
- [descriptor_cross_cell.csv](D:/release-foundation/outputs/117_full_cell_bridge_summary/descriptor_cross_cell.csv)

## Why this figure matters

The previous selected-cell bridge established the basic pattern on a
representative `6-cell` panel.

This figure pushes the same route-consistent bridge to the full current
`9-cell` calibrated-family cell panel by adding:

- `C321-R`
- `I181-R`
- `Lipo-R`

That is not benchmark-wide completion, but it is materially closer to a
panel-level route-consistent audit than the earlier selected-only view.

The current coverage boundary is mixed rather than uniform:

- the three `liposome` cells plus all three `internal181` cells now have full matched-curve coverage
- the remaining three `cross321` cells still use within-cell curve subsampling

## What the full 9-cell panel now shows

### 1. Shape-aware uncertainty still stays ahead

The main ordering does not invert under broader coverage:

- shape descriptors remain much more stable than deep-threshold timing
- the strongest shared target-layer story is still shape-oriented

### 2. The broader panel becomes more honest, not more convenient

The random-fold additions make the story more realistic:

- `C321-R` is a relatively strong timing cell
- `Lipo-R` keeps substantial timing coverage alive, including `t80`
- `I181-R` remains an extremely harsh timing-failure case

So the result is not "everything is now good." The result is:

`route-consistent cell-level uncertainty now spans the whole current 9-cell panel, and the heterogeneity becomes clearer rather than disappearing.`

### 3. This narrows the gap, but does not close the benchmark-side one

This figure supports saying:

- route-consistent duration/shape uncertainty is no longer just a two-cell or
  six-cell anecdote
- it now spans the full current calibrated-family cell panel

It still does **not** support saying:

- benchmark-wide duration UQ is solved
- universal timing targets are now mature
- the hard deep-threshold timing problem has gone away

## Strategic reading

The unification boundary moves one step:

- before: selected-cell bridge only
- now: full current 9-cell cell-bridge panel

But the main caution remains unchanged:

`shape-aware unification is the stronger current path; deep-threshold timing remains the harder unresolved layer.`
