# PLGA Early Middle-Layer Selector Probe

Date: 2026-06-11

## Question

Can early release observations improve the middle-layer route beyond static
descriptor-only mapping?

This probe tests:

```text
static descriptors + first k observed release points
    -> shape-family theta mapper
    -> early-residual family selector
    -> future release prediction
```

The score is future-only. For budget `k`, observed context points are excluded
from evaluation; only points strictly after the kth observation are scored.

## Script

`scripts/96_plga_early_middle_layer_selector_probe.py`

Main output:

`outputs/96_plga_early_middle_layer_selector_probe/report.md`

## Main Result

| split | static k=0 RMSE | best early-residual RMSE | budget | future-oracle RMSE |
|---|---:|---:|---:|---:|
| random-kfold | 0.130 | 0.109 | 5 | 0.088 |
| source-group-kfold | 0.216 | 0.174 | 5 | 0.142 |
| source-dataset-lodo | 0.220 | 0.187 | 5 | 0.160 |

Interpretation:

- Early observations improve the middle-layer route under all three split
  regimes.
- The improvement survives strict source-group and source-dataset splits.
- The deployable early-residual selector still leaves a visible gap to the
  diagnostic future-oracle selector.

## What This Means

This supports the old diagnosis:

```text
early Q is not merely another input feature;
early Q constrains the latent release-state / theta layer.
```

The current result is stronger than the static feature-combo gate result:

- static combo oracle is strong,
- static gate cannot reliably choose the right combo,
- early-conditioned middle-layer mapping recovers useful signal.

## Guardrails

This is not yet a final claim against black-box models.

Required next comparison:

```text
same split
same future-only target
same early budgets
early-middle-layer route
    vs
direct early-Q black-box route
```

Until that matched benchmark is run, the valid claim is:

> Early-conditioned middle-layer mapping improves over static middle-layer
> mapping and supports the release-regime observability diagnosis.

