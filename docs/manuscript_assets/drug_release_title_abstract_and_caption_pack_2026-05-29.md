# Drug Release Title, Abstract, and Caption Pack

Date: `2026-05-29`

Purpose:

```text
Provide a paper-ready writing starter pack from current evidence:
1. candidate titles,
2. one-sentence paper claims,
3. abstract draft,
4. Results subsection headers,
5. first-pass main-figure captions.
```

This is a draft-writing scaffold, not a final manuscript.

## 1. Recommended title band

The title should stay ambitious but below the "universal solved model" line.

## 1.1 Strongest current title

`Shared partially observed drug-release intelligence with mechanism-specific decoders`

Why this is safest:

- bold enough to signal more than PLGA regression
- consistent with current benchmark and object-level evidence
- does not overclaim solved cross-mechanism generalization

## 1.2 Slightly more method-forward title

`Mechanism-routed partial-observation forecasting for drug release`

Why use it:

- cleaner if the paper stays more benchmark/method oriented
- easier for a release/prediction audience

## 1.3 Slightly more framework-forward title

`Toward unified drug-release intelligence from sparse early observations`

Why use it:

- stronger program signal
- better if liposome + prospective discipline become central in the narrative

Why it is riskier:

- easier to read as broader than current evidence if the abstract is not
  disciplined

## 2. One-sentence claim options

## 2.1 Safest one-sentence claim

> We show that controlled-release forecasting can be organized as a shared
> partially observed inference problem in which sparse early release
> observations constrain mechanism-specific feasible-state families rather than
> only direct curve regressors.

## 2.2 More paper-style one-sentence claim

> Across audited PLGA benchmarks and a first liposome bridge, sparse early
> release dominates out-of-distribution forecasting and supports a
> mechanism-routed release-intelligence interface with shared uncertainty
> artifacts.

## 3. Abstract draft

```text
Machine-learning studies of drug release still largely frame the task as
direct supervised prediction from formulation descriptors, or from descriptors
augmented with a few early release measurements. We instead formulate release
forecasting as a shared partially observed inference problem in which
formulation and sparse early observations constrain a feasible mechanism-state
family that is decoded by mechanism-specific release models. On audited PLGA
benchmarks, sparse early release observations were the dominant
out-of-distribution forecasting signal, and mechanism-routed middle objects
outperformed matched direct curve-regression routes on the core forecasting
task. At the same time, explicit timing and shape analyses showed that better
curve fit does not automatically imply better release-duration or tail-shape
prediction, motivating evaluation beyond pointwise trajectory error alone. We
then carried the same release-intelligence interface into a first non-PLGA
retrospective bridge on liposome release, where the shared benchmark contract,
shared output schema, and mechanism-specific decoder remained viable. The
framework also already supports calibrated-family uncertainty artifacts across
a real PLGA plus liposome panel and follows prospective lock discipline ahead
of a preregistered chitosan reveal. Together, these results support a move
toward unified drug-release intelligence while also delimiting what remains
unsolved, particularly uncertainty-aware duration and cross-mechanism
prospective validation.
```

## 4. Results subsection headers

These are the cleanest current section heads if the paper were assembled now.

1. `Sparse early release dominates out-of-distribution forecasting signal`
2. `A mechanism-routed middle object improves over matched direct curve routes`
3. `Release timing and shape are not equivalent to curve-fit accuracy`
4. `A shared release-intelligence interface survives a first non-PLGA bridge`
5. `Shared calibrated-family artifacts and prospective lock discipline extend beyond point prediction`

## 5. Figure caption starters

## Figure 1

`Figure 1 | Drug release as a shared partially observed inference problem.`

Suggested caption:

```text
Current release ML is commonly posed as direct prediction from formulation
descriptors, optionally augmented with sparse early release observations. We
instead frame release forecasting as inference over a feasible mechanism-state
family conditioned on formulation/material context and sparse early prefix
data, followed by mechanism-specific decoding into future release trajectories,
timing summaries, and uncertainty objects.
```

## Figure 2

`Figure 2 | Sparse early release is the dominant forecasting signal and activates a mechanism-routed middle layer.`

Suggested caption:

```text
Across audited PLGA benchmarks, pure formulation-only prediction degraded
substantially under harder out-of-distribution split families, whereas sparse
early release observations carried the strongest forecasting signal. Matched
route comparisons further showed that a mechanism-routed middle object
outperformed direct curve-regression routes on the core audited PLGA task,
indicating that the middle layer is scientifically active rather than purely
narrative.
```

## Figure 3

`Figure 3 | Better curve fit does not automatically imply better duration or shape prediction.`

Suggested caption:

```text
Canonical PLGA timing and shape panels revealed that route ranking diverges
across curve resemblance, threshold timing, and tail-shape descriptors. These
results show that release-duration and release-shape prediction should be
audited explicitly rather than treated as automatic by-products of improved
curve fit.
```

## Figure 4

`Figure 4 | A first non-PLGA bridge supports a shared release-intelligence interface.`

Suggested caption:

```text
Using the same benchmark contract and shared output interface, liposome
release served as a first non-PLGA retrospective bridge. The mechanism-routed
interface remained viable across two split families, demonstrating that the
project is no longer only a PLGA-local release predictor, while also showing
that route superiority remains target-dependent across timing and shape
objects.
```

## Figure 5

`Figure 5 | Shared uncertainty artifacts and prospective discipline extend the framework beyond point prediction.`

Suggested caption:

```text
The current framework already emits calibrated-family artifacts across a real
PLGA plus liposome panel and follows preregistered lock discipline for a
pending chitosan reveal. These results place the project beyond pure
point-prediction benchmarking while also making clear that uncertainty-aware
duration prediction and completed cross-mechanism prospective validation
remain open.
```

## 6. Guardrail phrases for the abstract and title

These are phrases to prefer:

- `shared partially observed release intelligence`
- `mechanism-specific decoders`
- `feasible release-state family`
- `first non-PLGA bridge`
- `prospective lock discipline`

These are phrases to avoid:

- `universal drug-release foundation model`
- `solves release prediction across mechanisms`
- `universal best predictor of release duration`
- `cross-mechanism validation complete`

## 7. Bottom line

If writing started today, the strongest package would be:

- title in the `shared partially observed release intelligence` band
- abstract centered on `problem reformulation + audited PLGA + liposome bridge + UQ/prospective discipline`
- captions that let each figure prove one claim only

That is the highest-ambition writing stance still consistent with the current
evidence.
