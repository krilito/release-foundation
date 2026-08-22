# PLGA Middle vs Direct Complementarity and MoE Screen

Date: 2026-06-11

## Question

Are the direct early-Q black-box route and the middle-layer route good on the
same curves, or do they fail on different curves?

If they are complementary, a mixture-of-experts gate may be useful. If their
errors are mostly aligned, MoE is likely just added complexity.

## Script

`scripts/98_plga_middle_direct_complementarity_moe.py`

Main output:

`outputs/98_plga_middle_direct_complementarity_moe/report.md`

## Core Read

| split | best pair-oracle gain | middle win rate on direct-bad q75 | best deployable gate gain | decision |
|---|---:|---:|---:|---|
| random-kfold | 0.010 | 37.1% | +0.000 | no MoE claim |
| source-group-kfold | 0.027 | 62.9% | +0.001 | diagnostic complementarity only |
| source-dataset-lodo | 0.022 | 40.3% | -0.000 | no MoE claim |

Pair-oracle means choosing the better of direct and middle after seeing the
heldout future error. It is diagnostic, not deployable.

## Are They Good On The Same Curves?

Mostly yes, with one important exception.

- Random split: middle/direct RMSE Spearman correlation is high
  (`~0.67-0.72` across budgets), so their easy/hard curves largely overlap.
- Source-dataset split: correlation is also high at the best budget
  (`~0.70` at `k=5`), and the deployable gate does not recover useful gain.
- Source-group split: complementarity is real around `k=2-5`; at `k=5`,
  middle wins on `62.9%` of the direct-worst quartile curves.

So the useful signal is not global. It is concentrated in source-group style
OOD cases where direct early-Q extrapolation struggles.

## MoE Result

The first deployable MoE gate does not yet justify a claim.

Best full-fold deployable gate:

- source-group, `k=5`, ExtraTrees gate:
  - MoE RMSE `0.169`
  - fixed direct RMSE `0.170`
  - middle RMSE `0.174`
  - oracle pair RMSE `0.127`

This says:

```text
There is real oracle complementarity,
but the current gate barely identifies it.
```

## Decision

Do not build a larger MoE yet.

Next useful step is not bigger experts; it is selector diagnostics:

```text
Which direct-bad curves does middle rescue?
What static or early-observation features distinguish them?
Can a gate identify only that subset without hurting the rest?
```

Only after that should model optimization start.

