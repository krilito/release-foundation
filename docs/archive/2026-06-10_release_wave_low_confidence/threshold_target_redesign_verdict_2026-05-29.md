# Threshold Target Redesign Verdict

Date: `2026-05-29`

Purpose:

```text
Judge whether t50/t60/t70/t80 behave like equally honest shared timing
targets across the selected PLGA and liposome bridge panel.
```

Primary outputs:

- [cell_threshold_metrics.csv](D:/release-foundation/outputs/104_release_threshold_target_summary/cell_threshold_metrics.csv)
- [pattern_by_threshold.csv](D:/release-foundation/outputs/104_release_threshold_target_summary/pattern_by_threshold.csv)
- [threshold_target_scorecard.csv](D:/release-foundation/outputs/104_release_threshold_target_summary/threshold_target_scorecard.csv)
- [summary.txt](D:/release-foundation/outputs/104_release_threshold_target_summary/summary.txt)

Code:

- [104_release_threshold_target_summary.py](D:/release-foundation/scripts/104_release_threshold_target_summary.py)

## 1. Question

After the timing failure taxonomy, the next question is not "can we make
`t80` look better?" but:

> which deep-threshold timing targets are actually honest shared decision
> targets across PLGA and liposome?

## 2. Verdict

The selected-cell target audit indicates a clear ordering:

`t50` > `t60` > `t70` >> `t80`

for a shared cross-mechanism timing layer.

From [threshold_target_scorecard.csv](D:/release-foundation/outputs/104_release_threshold_target_summary/threshold_target_scorecard.csv),
using the global-method cross-cell means:

- `t50`: truth `0.889`, closed `0.421`, covered `0.272`
- `t60`: truth `0.819`, closed `0.201`, covered `0.126`
- `t70`: truth `0.819`, closed `0.141`, covered `0.082`
- `t80`: truth `0.819`, closed `0.064`, covered `0.056`

The point is not that `t50` is universally solved. It is not.
The point is that as the threshold deepens, three things decay together:

1. the fraction of curves whose truth reaches the threshold
2. the fraction of family intervals that actually close
3. the resulting coverage

That makes `t80` a poor default shared timing target for a general
release-intelligence layer.

More precisely, the biggest collapse from `t50 -> t80` is not truth
reachability. It is interval closure:

- truth reachability only drops from `0.889` to `0.819`
- interval closure drops from `0.421` to `0.064`

So the practical bottleneck is:

> family-based release bands can often still support the existence of the
> target in the truth, but cannot close a useful interval at deep thresholds.

## 3. Why this matters

This changes the right "bold but legal" design target.

Instead of promising:

> one generic deep-threshold duration target across all mechanisms

the more honest unified target layer is:

> a shared timing family built around horizon-aware and mechanism-aware
> threshold choices, where `t50` or `t60` may be safer default targets than
> `t80`.

That is a stronger and more useful conclusion than simply saying "t80 is
hard," because it points to a redesign principle:

> choose shared timing targets partly by interval-closure behavior, not only
> by biological interpretability.

## 4. Bottom line

The unified release stack can still aim big, but it should aim big at the
right layer:

> shared timing targets are plausible;
> `t80` is not yet an honest universal default among them.
