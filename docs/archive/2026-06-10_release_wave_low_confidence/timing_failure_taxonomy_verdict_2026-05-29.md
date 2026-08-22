# Timing Failure Taxonomy Verdict

Date: `2026-05-29`

Purpose:

```text
Explain why route-consistent timing uncertainty is still weak in the
selected-cell bridge layer by classifying the actual failure modes.
```

Primary outputs:

- [classified_rows.csv](D:/release-foundation/outputs/102_release_timing_failure_audit/classified_rows.csv)
- [failure_mode_by_threshold_method.csv](D:/release-foundation/outputs/102_release_timing_failure_audit/failure_mode_by_threshold_method.csv)
- [failure_mode_by_cell.csv](D:/release-foundation/outputs/102_release_timing_failure_audit/failure_mode_by_cell.csv)
- [summary.txt](D:/release-foundation/outputs/102_release_timing_failure_audit/summary.txt)

Code:

- [102_release_timing_failure_audit.py](D:/release-foundation/scripts/102_release_timing_failure_audit.py)

## 1. Question

The selected-cell bridge now shows a repeated pattern:

> shape-aware uncertainty looks much stronger than deep-threshold timing
> uncertainty.

The important follow-up question is whether timing fails because:

1. calibration is slightly off on otherwise closed intervals, or
2. the family-based release bands often do not even reach the deep threshold
   in a way that yields a closed interval.

## 2. Verdict

The failure taxonomy indicates that `t80` weakness is primarily a
`reachability / open-interval` problem, not a small closed-interval
miscalibration problem.

From [failure_mode_by_threshold_method.csv](D:/release-foundation/outputs/102_release_timing_failure_audit/failure_mode_by_threshold_method.csv):

- `t80 raw`
  - `truth_reached_but_interval_open_upper = 0.755`
  - `truth_not_reached_but_interval_open_upper = 0.082`
  - `covered = 0.071`
- `t80 global`
  - `truth_reached_but_interval_open_upper = 0.888`
  - `truth_not_reached_but_interval_open_upper = 0.082`
  - `covered = 0.020`
- `t80 local`
  - `truth_reached_but_interval_open_upper = 0.898`
  - `truth_not_reached_but_interval_open_upper = 0.082`
  - `covered = 0.020`

So the dominant failure mode is not "a closed interval that slightly misses."
It is "the upper side stays open because the lower release band never reaches
the threshold."

In other words:

> the hard part is not merely widening an already valid timing interval.
> the hard part is that the lower family release band often never reaches
> the deep threshold at all, or the truth itself does not reach the
> threshold inside the observed assay horizon.

## 3. Why this matters for unification

This sharpens the program-level interpretation.

If deep-threshold timing mostly failed because of closed-interval misses,
the next move would mainly be better conformal scaling.

But if deep-threshold timing mostly fails because intervals remain open or
the truth does not reach the threshold, then the real bottleneck sits deeper:

- decoder trajectory reachability
- assay horizon limits
- threshold choice relative to mechanism-specific release completion

That means the unified release stack should not promise:

> one generic timing-UQ layer automatically solves all threshold targets

Instead, the more honest next step is:

> mechanism-aware or horizon-aware timing targets, especially for deep
> thresholds like `t80`.

There is also a secondary distinction between datasets, visible in
[failure_mode_by_cell.csv](D:/release-foundation/outputs/102_release_timing_failure_audit/failure_mode_by_cell.csv):

- In `C321-D`, `C321-P`, `I181-D`, and `I181-P`, `t80` is dominated by
  `truth_reached_but_interval_open_upper`
- In `Lipo-D` and `Lipo-P`, `t80` includes a larger
  `truth_not_reached_but_interval_open_upper` component, meaning the real
  curve often does not reach `80%` release within the assay horizon

That means the same headline failure has at least two sub-mechanisms:

1. the truth reaches the threshold, but the lower family band does not
2. the truth itself often fails to reach the threshold within the assay
   horizon

Those are not the same scientific problem, and they should not be treated as
one generic calibration issue.

## 4. Bottom line

The timing bottleneck is now more precisely diagnosed:

> `t80` is weak mainly because the route-consistent family bands often do not
> yield a closed deep-threshold interval, not because they are narrowly
> miscentered around the right answer.

For `t50`, the picture is mixed:

- raw `t50` is mostly `closed_interval_too_early = 0.704`
- global/local `t50` are mostly `truth_reached_but_interval_open_upper`
  at `0.592 / 0.633`

So even before `t80`, the route-consistent timing problem is already partly a
reachability issue, and conformal widening alone does not fix it cleanly.
