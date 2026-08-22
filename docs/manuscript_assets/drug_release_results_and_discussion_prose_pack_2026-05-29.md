# Drug Release Results and Discussion Prose Pack

Date: `2026-05-29`

Purpose:

```text
Provide first-pass manuscript prose for the main Results and Discussion
sections using only currently audited evidence.
```

This is a writing starter, not a polished manuscript.

## Results 1

### Sparse early release dominates out-of-distribution forecasting signal

```text
Across the audited PLGA benchmarks, sparse early release observations carried
the dominant out-of-distribution forecasting signal. In the cross321
benchmark, the best formulation-only median R2 was 0.672 under group-by-drug
and 0.666 under group-by-polymer, whereas early-only routes reached 0.941 and
0.949, respectively. A similar pattern held in the independent internal181
benchmark, where formulation-only median R2 values were 0.661
(group-by-drug) and 0.525 (group-by-polymer), compared with 0.932 and 0.907
for early-only forecasting. These results indicate that pure formulation
regression remains substantially weaker under harder out-of-distribution split
families, and that sparse early release observations provide the main signal
that makes long-horizon forecasting viable.
```

Evidence:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)

## Results 2

### A mechanism-routed middle object improves over matched direct curve routes

```text
We next asked whether the mechanism-routed middle object was scientifically
active or merely narrative. Across the audited PLGA task family, the
best-performing theta route exceeded the matched best-performing direct-Q
route in all 18 audited cells, with theta-minus-direct median R2 gains ranging
from +0.002 to +0.121. In the hardest formulation-plus-early internal181
group-by-polymer cell, for example, the best direct route reached a median R2
of 0.801, whereas the best mechanism route reached 0.922, for a gain of
+0.121. Even under formulation-only settings, where all routes weakened, the
mechanism path remained modestly but consistently better than the matched
direct route. These results argue that the middle object is not merely a
storytelling device, but a useful representation for the audited PLGA
forecasting problem.
```

Evidence:

- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv](D:/release-foundation/outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv)

## Results 3

### Better curve fit does not automatically imply better duration or shape prediction

```text
Although the mechanism route was the more consistent winner on median
curve-level resemblance, this did not imply universal superiority on release
timing or shape. Across four canonical PLGA out-of-distribution
formulation-plus-early cells, the mechanism route won median curve R2 in 4/4
cells but pooled R2 in only 1/4 cells. Future-only threshold timing showed an
even sharper split: the mechanism route won t10 mean absolute error in 2/4
cells, t50 in 1/4 cells, and t80 in 0/4 cells. Shape descriptors were also
mixed. The mechanism route won burst error in 3/4 cells and post-window and
residual-tail error in 4/4 cells, but won tail-area-under-the-curve error in
only 1/4 cells. Together, these results show that curve resemblance, release
timing, and tail-shape behavior are related but non-identical objects, and
that release-duration claims must be audited explicitly rather than inherited
from improved curve-fit metrics.
```

Evidence:

- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)
- [outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv)
- [outputs/82_release_capability_snapshot/plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)

## Results 4

### A shared release-intelligence interface survives a first non-PLGA retrospective bridge

```text
To test whether the release-intelligence interface was still viable beyond
PLGA, we carried it into a first non-PLGA retrospective bridge on liposome
release. Under group-by-API splits, the mechanism route achieved a pooled R2
of 0.903 versus 0.864 for the matched direct route, with t10 and t80 errors
of 0.55 h and 18.70 h compared with 1.05 h and 34.34 h for direct-Q. Under
the more difficult group-by-release-method split, the gap widened further in
curve-space, with pooled R2 values of 0.903 for the mechanism route and 0.457
for direct-Q. However, the same liposome bridge also reinforced the main
timing/shape lesson: route superiority remained target-dependent, with direct-Q
still better on t50 in both split families and on all four shape descriptors
under group-by-API. The main implication is therefore not that one universal
route wins every metric, but that the same release-intelligence interface is
already viable in at least one non-PLGA release family.
```

Evidence:

- [outputs/82_release_capability_snapshot/summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)
- [docs/liposome_bridge_verdict_2026-05-29.md](D:/release-foundation/docs/liposome_bridge_verdict_2026-05-29.md)

## Results 5

### Shared uncertainty artifacts and prospective discipline extend the framework beyond point prediction

```text
The project also already exceeds a pure point-prediction benchmark culture in
two ways. First, a calibrated-family panel now exists across nine PLGA and
liposome benchmark cells. Across this panel, representative split-conformal
global coverage values included 0.902 for cross321/group-by-drug, 0.919 for
cross321/group-by-polymer, 0.946 for internal181/group-by-polymer, and 0.923
for liposome/random_5fold. These results show that calibrated-family artifacts
are now a real output layer rather than only raw uncertainty tables. Second,
the project already follows prospective lock discipline ahead of wet-lab
reveal. The preregistered chitosan prospective batch is locked for 12 curves,
6 formulations, and 2 drugs, with the primary endpoint defined in advance as
aggregate cov90 >= 0.83. Together, these two elements show that the framework
has already moved beyond retrospective point forecasting, even though
prospective cross-mechanism success has not yet been completed.
```

Evidence:

- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
- [outputs/67_chitosan_prospective/lock_metadata.json](D:/release-foundation/outputs/67_chitosan_prospective/lock_metadata.json)
- [docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md](D:/release-foundation/docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md)

## Discussion

### Discussion opener

```text
Taken together, these results support a stronger view of drug release than a
pure regression benchmark framing allows. Current external work is still
dominated by descriptor-based supervised prediction, descriptor-plus-prefix
forecasting, and mechanism-local surrogate workflows. In contrast, the present
results support organizing release as a shared partially observed inference
problem in which sparse early observations constrain feasible release-state
families that are decoded by mechanism-specific models. This is the level at
which a unified drug-release intelligence program is already defensible.
```

### Discussion limits paragraph

```text
At the same time, the evidence places clear limits on what can be claimed.
First, route superiority is target-dependent: better curve resemblance does not
automatically imply better release-duration or tail-shape prediction. Second,
although the framework now supports a real calibrated-family panel across PLGA
and liposome, uncertainty-aware duration and shape reporting is still not
route-consistent. Third, the chitosan prospective batch is locked but not yet
revealed, so prospective cross-mechanism validation remains incomplete. The
appropriate conclusion is therefore not that drug release has been universally
solved, but that the field can now be organized around a shared
release-intelligence interface with clear next steps toward stronger
cross-mechanism and prospective evidence.
```

### Discussion future-direction paragraph

```text
The most important next step is not another point-prediction benchmark table,
but closure of the remaining systems gap between calibrated-family artifacts
and duration/shape outputs. Once uncertainty-aware release timing and tail
behavior are emitted through the same shared object, the framework will be in a
much stronger position to compete not only with direct benchmark rivals, but
also with platform-style formulation systems that derive their strength from
workflow value. In that sense, the main strategic target is not merely a
better PLGA regressor, but the first serious unified drug-release
intelligence stack.
```

### Discussion outside-in paragraph

```text
This interpretation is also supported by an outside-in comparison against the
current external field on the same five-layer surface. The main distinction of
the present system is not simply that it competes as another release
predictor, but that it occupies more of the release-intelligence stack at
once. The strongest structural edge appears at the shared partially observed
benchmark layer and the shared posterior-family object layer, where leading
benchmark and platform families remain partial or absent. By contrast, the
reporting layer remains more qualified: although shared curve, uncertainty,
and shape-target artifacts now exist, deep-threshold timing defaults remain
less mature than the benchmark/object layers. The appropriate positioning
claim is therefore one of stack occupancy rather than universal metric
leadership.
```

Evidence:

- [outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv)
- [release_stack_occupancy_paper_hook_2026-05-29.md](D:/release-foundation/docs/release_stack_occupancy_paper_hook_2026-05-29.md)

## Bottom line

These paragraphs are intentionally conservative in claim wording and aggressive
in scientific framing. They are designed so the project can sound bigger
without asking the current evidence to prove more than it actually proves.
