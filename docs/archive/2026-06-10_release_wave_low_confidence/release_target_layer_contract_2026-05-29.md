# Release Target-Layer Contract

Date: `2026-05-29`

Purpose:

```text
Turn the threshold redesign audit into a concrete mechanism-aware contract
for which timing targets can honestly serve as shared defaults.
```

Primary outputs:

- [mechanism_target_aggregate.csv](D:/release-foundation/outputs/105_release_target_layer_contract/mechanism_target_aggregate.csv)
- [mechanism_target_scorecard.csv](D:/release-foundation/outputs/105_release_target_layer_contract/mechanism_target_scorecard.csv)
- [summary.txt](D:/release-foundation/outputs/105_release_target_layer_contract/summary.txt)

Code:

- [105_release_target_layer_contract.py](D:/release-foundation/scripts/105_release_target_layer_contract.py)

## 1. One-sentence contract

> A unified release-intelligence stack may share timing target classes, but it
> should not force one deep-threshold default across all mechanisms.

## 2. Mechanism-level verdict

### PLGA

Current evidence does **not** support any fully stable universal timing
default across the audited PLGA OOD families.

The least-bad anchor is:

- `t50`

But even that should be described as:

> the least-bad shared PLGA timing anchor, not a solved or universally stable
> PLGA timing target.

From [mechanism_target_aggregate.csv](D:/release-foundation/outputs/105_release_target_layer_contract/mechanism_target_aggregate.csv):

- `t50`: truth `1.000`, closed `0.381`, covered `0.158` -> `secondary_only`
- `t60`: truth `0.979`, closed `0.218`, covered `0.105` -> `secondary_only`
- `t70`: truth `0.979`, closed `0.129`, covered `0.039` -> `not_default`
- `t80`: truth `0.979`, closed `0.013`, covered `0.000` -> `not_default`

`t60` and `t70` are exploratory only.
`t80` should not be the default shared PLGA target.

### Liposome

Under the current assay horizon and selected-cell bridge panel, the most
honest *pilot anchor* is:

- `t50`

But the current evidence still does **not** make it a strong shared default.

From [mechanism_target_aggregate.csv](D:/release-foundation/outputs/105_release_target_layer_contract/mechanism_target_aggregate.csv):

- `t50`: truth `0.667`, closed `0.500`, covered `0.500` -> `secondary_only`
- `t60`: truth `0.500`, closed `0.167`, covered `0.167` -> `secondary_only`
- `t70`: truth `0.500`, closed `0.167`, covered `0.167` -> `secondary_only`
- `t80`: truth `0.500`, closed `0.167`, covered `0.167` -> `secondary_only`

So the liposome message is:

> `t50` is the best current liposome-local timing anchor under the present
> assay horizon, but none of the audited liposome thresholds yet behaves like
> a robust cross-mechanism shared default.

### Chitosan

No timing target is verified yet.

Current prospective evidence is still only:

- preregistered curve-space interval coverage
- no revealed timing-target audit yet

So the honest status is:

> unverified prospective placeholder

## 3. Why this is still a unification result

This is not a retreat from unification.
It is a sharper definition of what unification should mean.

The unified layer should share:

- target classes
- reporting interfaces
- benchmark rules
- decision-support semantics

But it should allow:

- mechanism-aware default targets
- horizon-aware exclusions for overly deep thresholds
- provisional mechanism-local anchors when no threshold is yet strong enough
  to serve as a genuine shared default

That is still much bolder than the current field, which usually does not
define any shared target layer at all.

## 4. Bottom line

The current best contract is:

> shared timing target classes, mechanism-aware defaults or pilot anchors,
> and no forced universal `t80`.
