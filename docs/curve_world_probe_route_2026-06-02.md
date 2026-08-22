## Curve-World Probe Route

Date: `2026-06-02`

Purpose:

- freeze the old "observer / world-model" rhetoric
- test whether a new curve-pretrained route deserves to exist
- avoid building a large latent-dynamics stack before the data and shape
  assumptions are audited

## Working Claim

The new route is only worth pursuing if all three are true:

1. the real release-curve corpus is large and diverse enough to support a
   shared trajectory grammar
2. simple low-dimensional shape families do **not** already explain nearly all
   curve variation
3. some cross-system shape structure survives domain shift and is not merely
   a single-dataset artifact

If any of these fail, the route should collapse back to a smaller object:

- parameter-family regression
- shape-family regression
- or a domain-local release predictor with no "world model" claim

## Probe Order

### Probe 1 — Corpus Audit

Question:

```text
How much real trajectory supervision do we actually have?
```

Outputs:

- number of curves
- number of formulation groups / systems
- points per curve distribution
- time-span distribution
- release-range distribution
- monotonicity / saturation sanity checks

Passing intuition:

- more than a single narrow family of curves
- enough diversity that "shared grammar" is not obviously vacuous

### Probe 2 — Shape-Family Audit

Question:

```text
Do simple parametric families already explain the curves?
```

Candidate families:

- Weibull
- double exponential / biphasic saturating family
- Korsmeyer-Peppas style family
- simple piecewise logistic / saturating family

Decision gate:

- if a tiny family explains ~95%+ of the structured variation, the new route
  should shrink to shape-parameter prediction rather than a learned world model

### Probe 3 — Cross-System Shape Transfer

Question:

```text
Is there any shared shape grammar across systems, or only within-system fit?
```

Minimum check:

- learn a shared low-dimensional basis / family on one subset
- test reconstruction / transfer on held-out system families

Decision gate:

- if low-dimensional shared shape transfer already works, the "grammar" may be
  simpler than a neural latent-dynamics model
- if it fails but structured residuals remain, a larger learned dynamics route
  becomes more plausible

## Branch Policy

This branch is for probe-first exploration only.

Do not:

- add a large latent world model yet
- add phase-interpretation claims
- add symbolic training constraints to the main loop yet

Do:

- measure data reality first
- measure shape-family sufficiency second
- only then decide whether a new structured trajectory model is justified

## Immediate Artifacts

1. `scripts/120_curve_corpus_audit.py`
2. `outputs/120_curve_corpus_audit/`
3. follow-up scripts for Probe 2 and Probe 3 only if Probe 1 does not already
   show the route is starved for real trajectory supervision
