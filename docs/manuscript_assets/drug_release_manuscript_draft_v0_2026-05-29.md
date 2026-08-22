# Drug Release Manuscript Draft V0

Date: `2026-05-29`

Purpose:

```text
Provide one continuous manuscript-style draft that merges the current
positioning, abstract, Introduction, Results, and Discussion prose into a
single paper-facing document.
```

Status:

```text
This is a conservative draft built only from currently audited evidence.
It is meant to accelerate real manuscript writing, not to finalize wording.
```

## Title

`Shared partially observed drug-release intelligence with mechanism-specific decoders`

## One-sentence claim

> We do not present another direct release regressor; we show that controlled
> release can be organized as a shared partially observed inference problem
> with explicit feasible-state objects, mechanism-specific decoders, and
> mechanism-aware reporting contracts.

## Abstract

```text
Machine-learning studies of drug release still largely frame the problem as
direct supervised prediction from formulation descriptors, or from
descriptors augmented with a few early release measurements. We instead
recast release forecasting as a shared partially observed inference problem in
which formulation/material context and sparse early observations constrain a
feasible release-state family that is decoded by mechanism-specific release
models. On audited PLGA benchmarks, sparse early release observations were
the dominant out-of-distribution forecasting signal, and mechanism-routed
middle objects outperformed matched direct curve-regression routes on the core
forecasting task. At the same time, explicit timing and shape analyses showed
that better curve fit does not automatically imply better release-duration or
tail-shape prediction, motivating evaluation beyond pointwise trajectory error
alone. We then carried the same release-intelligence interface into a first
non-PLGA retrospective bridge on liposome release, where the shared benchmark
contract, shared output schema, and mechanism-specific decoder remained
viable. The framework also already supports calibrated-family uncertainty
artifacts across a real PLGA plus liposome panel, a mechanism-aware reporting
target layer, and prospective lock discipline ahead of a preregistered
chitosan reveal. Together, these results support a move toward unified
drug-release intelligence while also clarifying the current limits of that
claim, especially for deep-threshold timing defaults and prospective
cross-mechanism validation.
```

## Introduction

```text
Machine-learning studies of drug release have grown rapidly, but the dominant
problem formulation remains relatively narrow. Most published systems still
treat release prediction as supervised regression from formulation
descriptors, or from descriptors augmented with a few early observations,
typically using tabular model families such as random forests, gradient
boosting, support vector machines, or multilayer perceptrons. This framing
has already proved useful for practical few-shot forecasting problems,
including long-acting injectable release, but it still leaves the field
largely organized around direct curve prediction rather than around a shared
inference object.

At the same time, adjacent parts of the field are evolving in different
directions. Mechanism-aware release workflows and kinetic-summary pipelines
have become increasingly important for non-PLGA systems such as liposomes,
while broader computational-pharmaceutics and formulation-AI programs are
pushing toward hybrid simulation/ML design loops, platformization, and
decision support. In parallel, curated release datasets and AI-ready metadata
standards are beginning to make reusable benchmark contracts realistic. Taken
together, these developments suggest that the key open problem is no longer
simply whether a slightly better release regressor can be trained, but whether
drug release can be organized as a shared partially observed inference problem
that remains compatible with mechanism-specific decoding and cross-mechanism
benchmark discipline.

That question is difficult because release forecasting is intrinsically
underdetermined at realistic data scales. Formulation descriptors alone often
generalize weakly under harder out-of-distribution split families, while
sparse early release observations can be highly informative but do not
uniquely determine a mechanistic explanation. Direct curve regression absorbs
this ambiguity into the target, but does not make the latent uncertainty
explicit. For a field that increasingly cares about timing, duration, tail
behavior, assay planning, and translational design decisions, a purely direct
prediction object is therefore too limited.

Here we recast controlled-release forecasting as a shared partially observed
inference problem. Instead of mapping descriptors directly to future release,
we use formulation/material context together with sparse early release
observations to constrain a feasible release-state family that is decoded by
mechanism-specific release models. The framework already supports a shared
partially observed benchmark contract, a shared posterior-family object, a
mechanism-aware target registry, calibrated-family artifacts, and a first
non-PLGA bridge on liposome release. What is being unified is not one literal
raw mechanistic parameterization across all release families, but the
benchmark, object, and reporting interfaces through which release forecasting
can become comparable across mechanisms.

The current evidence supports this framing at a meaningful but still bounded
level. On audited PLGA benchmarks, sparse early release observations dominate
out-of-distribution forecasting signal, and mechanism-routed middle objects
improve over matched direct curve-regression routes on the core forecasting
task. A first non-PLGA bridge on liposome release shows that the same release
interface can survive a second mechanism family, while calibrated-family
artifacts and preregistered prospective lock discipline extend the framework
beyond pure point prediction. At the same time, route superiority is already
known to depend on whether the target is curve resemblance, release timing,
or release shape, and prospective cross-mechanism validation remains
incomplete. The goal of this work is therefore not to claim a universal
solved drug-release model, but to show that a unified drug-release
intelligence layer is now technically and scientifically plausible.
```

## Results

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

## Discussion

```text
Taken together, these results support a stronger view of drug release than a
pure regression benchmark framing allows. Current external work is still
dominated by descriptor-based supervised prediction, descriptor-plus-prefix
forecasting, and mechanism-local surrogate workflows. In contrast, the present
results support organizing release as a shared partially observed inference
problem in which sparse early observations constrain feasible release-state
families that are decoded by mechanism-specific models. Viewed against the
current external field on the same five-layer surface, the main novelty of the
present system is not universal metric leadership but stack occupancy. The
strongest structural advantage appears at the shared partially observed
benchmark layer and the shared posterior-family object layer, where leading
benchmark and platform families remain partial or absent.

At the same time, the evidence places clear limits on the strongest version of
the unification claim. First, route superiority remains target-dependent:
better curve resemblance does not automatically imply better release duration
or tail-shape prediction. Second, the current reporting layer is strongest
for shape-aware targets and more qualified for deep-threshold timing, with
t80 in particular behaving poorly as a universal default target. Third, the
chitosan prospective package is locked but not yet revealed, so prospective
cross-mechanism validation remains incomplete. The appropriate conclusion is
therefore not that drug release has been universally solved, but that the
field can now be organized around a shared release-intelligence stack with
mechanism-aware reporting defaults and clear next steps toward stronger
cross-mechanism and prospective evidence.

The most important next step is not another point-prediction benchmark table,
but closure of the remaining systems gap between calibrated-family artifacts
and duration/shape outputs. Once uncertainty-aware release timing and tail
behavior are emitted through the same shared object and target layer, the
framework will be in a much stronger position to compete not only with direct
benchmark rivals, but also with platform-style formulation systems that draw
their strength from workflow value. In that sense, the main strategic target
is not merely a better PLGA regressor, but the first serious unified
drug-release intelligence stack.
```

## Evidence anchors

- [drug_release_opening_sections_draft_2026-05-29.md](D:/release-foundation/docs/drug_release_opening_sections_draft_2026-05-29.md)
- [drug_release_results_and_discussion_prose_pack_2026-05-29.md](D:/release-foundation/docs/drug_release_results_and_discussion_prose_pack_2026-05-29.md)
- [drug_release_manuscript_positioning_addendum_2026-05-29.md](D:/release-foundation/docs/drug_release_manuscript_positioning_addendum_2026-05-29.md)
- [drug_release_positioning_draft_2026-05-29.md](D:/release-foundation/docs/drug_release_positioning_draft_2026-05-29.md)
- [release_stack_occupancy_paper_hook_2026-05-29.md](D:/release-foundation/docs/release_stack_occupancy_paper_hook_2026-05-29.md)

## Bottom line

If you want one file that is closest to a full manuscript narrative without
jumping between multiple prose packs, use this one.
