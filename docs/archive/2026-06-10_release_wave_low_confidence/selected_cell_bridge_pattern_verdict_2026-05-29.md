# Selected Cell Bridge Pattern Verdict

Date: `2026-05-29`

Purpose:

```text
State the first repeated pattern emerging from the selected-cell
family-duration/shape bridge layer.
```

Primary outputs:

- [threshold_cross_cell.csv](D:/release-foundation/outputs/100_selected_cell_bridge_summary/threshold_cross_cell.csv)
- [descriptor_cross_cell.csv](D:/release-foundation/outputs/100_selected_cell_bridge_summary/descriptor_cross_cell.csv)
- [pattern_snapshot.csv](D:/release-foundation/outputs/100_selected_cell_bridge_summary/pattern_snapshot.csv)

Code:

- [100_selected_cell_bridge_summary.py](D:/release-foundation/scripts/100_selected_cell_bridge_summary.py)

## 1. What is now repeatable

The selected-cell bridge is no longer just a single PLGA cell and a single
liposome cell.

It now spans four representative cells:

- `C321-D`
- `C321-P`
- `Lipo-D`
- `Lipo-P`

That is enough to ask whether a stable pattern is beginning to repeat.

## 2. The repeated pattern

The clearest repeated pattern is:

> family-based uncertainty is substantially more mature for shape descriptors
> than for deep-threshold timing targets.

Cross-cell means from [pattern_snapshot.csv](D:/release-foundation/outputs/100_selected_cell_bridge_summary/pattern_snapshot.csv):

### Threshold timing

- global `t10`: mean coverage `0.667`
- global `t50`: mean coverage `0.408`
- global `t80`: mean coverage `0.083`

- local `t10`: mean coverage `0.627`
- local `t50`: mean coverage `0.408`
- local `t80`: mean coverage `0.083`

- raw `t10`: mean coverage `0.667`
- raw `t50`: mean coverage `0.421`
- raw `t80`: mean coverage `0.149`

### Shape descriptors

- global burst: mean coverage `0.921`
- global post-window: mean coverage `0.921`
- global residual-tail: mean coverage `1.000`
- global tail-AUC: mean coverage `0.895`

- local burst: mean coverage `0.855`
- local post-window: mean coverage `0.868`
- local residual-tail: mean coverage `1.000`
- local tail-AUC: mean coverage `0.947`

That gap is large enough that it should now be treated as a real structural
pattern, not just noise.

## 3. Split sensitivity

The bridge layer also appears more split-sensitive for PLGA than for liposome.

Most obvious example:

- `C321-D` has usable `t10 / t50` coverage
- `C321-P` collapses badly on timing and even degrades some raw shape coverage
- `Lipo-D` and `Lipo-P` are nearly identical in the current chosen-cell bridge

Interpretation:

> the hardest local route-consistent UQ problem may not be "all mechanisms"
> in the abstract, but certain harder OOD partition families inside PLGA.

## 4. What this now justifies

We can now say, with evidence stronger than a one-off example:

1. the calibrated family object can emit duration/shape uncertainty on real
   curves
2. the bridge survives beyond a single cherry-picked curve
3. shape-aware uncertainty looks much closer than high-threshold timing
   uncertainty

## 5. What this still does not justify

This still does **not** justify saying:

1. route-consistent duration UQ is solved
2. benchmark-wide duration/shape uncertainty is established
3. the bridge is stable across all OOD split families

## 6. Best next move

The next move should focus on the hardest repeated failure:

> improve or better characterize `t80`-style deep-threshold timing
> uncertainty before trying to headline full release decision-support.

## Bottom line

The selected-cell bridge now shows a repeatable signal:

> shape-aware uncertainty is already emerging;
> deep-threshold timing uncertainty is still the bottleneck.

That is one of the most useful strategic findings produced so far, because it
tells us exactly where the unified-release stack is closest to becoming
real decision-support, and where it is still not ready.
