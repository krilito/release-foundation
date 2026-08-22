# Release Shape-Target Contract

Date: `2026-05-29`

Purpose:

```text
Turn the selected-cell bridge shape evidence into a concrete mechanism-aware
contract for shared shape targets.
```

Primary outputs:

- [mechanism_shape_aggregate.csv](D:/release-foundation/outputs/107_release_shape_target_contract/mechanism_shape_aggregate.csv)
- [mechanism_shape_scorecard.csv](D:/release-foundation/outputs/107_release_shape_target_contract/mechanism_shape_scorecard.csv)
- [summary.txt](D:/release-foundation/outputs/107_release_shape_target_contract/summary.txt)

Code:

- [107_release_shape_target_contract.py](D:/release-foundation/scripts/107_release_shape_target_contract.py)

## 1. One-sentence contract

> Shape targets are currently the most mature shared decision-support objects
> in the unified release-intelligence stack.

## 2. Mechanism-level verdict

### PLGA

PLGA shape targets are clearly stronger than PLGA timing targets.

From [mechanism_shape_aggregate.csv](D:/release-foundation/outputs/107_release_shape_target_contract/mechanism_shape_aggregate.csv):

- `burst`: mean `0.796`, min `0.684` -> `secondary_only`
- `post_window`: mean `0.827`, min `0.684` -> `candidate_default`
- `residual_tail`: mean `0.979`, min `0.958` -> `candidate_default`
- `tail_auc`: mean `0.843`, min `0.684` -> `candidate_default`

So the strongest PLGA shape default is:

- `residual_tail`

with `post_window` and `tail_auc` also behaving like plausible defaults.

### Liposome

All four audited liposome shape targets behave like strong pilot defaults in
the current selected-cell bridge panel:

- `burst`: `1.000`
- `post_window`: `1.000`
- `residual_tail`: `1.000`
- `tail_auc`: `1.000`

This does not prove liposome shape UQ is solved benchmark-wide.
But it does mean the shape layer is much more mature than the liposome timing
layer.

### Chitosan

No shape target is verified yet.
Current status remains:

> unverified prospective placeholder

## 3. Why this matters

This sharpens the unification story again.

If someone asks, "what is already genuinely shared across mechanisms?",
the strongest answer is no longer timing.
It is:

- curve-space objects
- shape-target reporting classes
- mechanism-aware target registry semantics

## 4. Bottom line

The unified stack is now strongest at:

> shared shape targets plus mechanism-aware timing anchors,
> not one universal deep-threshold timing target.
