# PLGA Early-Observation Route Sync

Date: 2026-06-11

## One-Sentence State

Early observations are useful, but the current middle-layer route has not yet
beaten matched direct early-Q black-box controls; the next bottleneck is the
selector, not expert capacity.

## Artifacts

Scripts:

- `scripts/96_plga_early_middle_layer_selector_probe.py`
- `scripts/97_plga_direct_early_blackbox_control.py`
- `scripts/98_plga_middle_direct_complementarity_moe.py`

Reports:

- `outputs/96_plga_early_middle_layer_selector_probe/report.md`
- `outputs/97_plga_direct_early_blackbox_control/report.md`
- `outputs/98_plga_middle_direct_complementarity_moe/report.md`

Project notes:

- `plga_early_middle_layer_selector_probe_2026-06-11.md`
- `plga_direct_early_blackbox_control_2026-06-11.md`
- `plga_middle_direct_complementarity_moe_2026-06-11.md`

## Result 1: Early Points Improve The Middle Layer

Script 96 tested:

```text
static descriptors + first k observed release points
    -> shape-family theta mapper
    -> early-residual family selector
    -> future release prediction
```

Future-only RMSE improved over the static middle-layer route:

| split | static k=0 | best early-middle | budget | future-oracle |
|---|---:|---:|---:|---:|
| random-kfold | 0.130 | 0.109 | 5 | 0.088 |
| source-group-kfold | 0.216 | 0.174 | 5 | 0.142 |
| source-dataset-lodo | 0.220 | 0.187 | 5 | 0.160 |

Interpretation:

```text
early Q is useful as a latent release-state / theta constraint.
```

## Result 2: Direct Early-Q Black-Box Still Wins

Script 97 added matched direct controls on the same splits, budgets, and
future-only targets.

| split | best middle | best direct | direct method |
|---|---:|---:|---|
| random-kfold | 0.109 | 0.086 | `direct_static_early_time`, k=5 |
| source-group-kfold | 0.174 | 0.152 | `direct_early_time`, k=5 |
| source-dataset-lodo | 0.187 | 0.152 | `direct_early_time`, k=5 |

Interpretation:

```text
Do not claim middle-layer predictive superiority yet.
Do not optimize model capacity before explaining this gap.
```

## Result 3: MoE Has Diagnostic But Not Deployable Signal

Script 98 checked whether direct and middle are good on the same curves.

| split | pair-oracle gain | middle wins on direct-bad q75 | best deployable gate gain |
|---|---:|---:|---:|
| random-kfold | 0.010 | 37.1% | +0.000 |
| source-group-kfold | 0.027 | 62.9% | +0.001 |
| source-dataset-lodo | 0.022 | 40.3% | -0.000 |

Interpretation:

- Random and source-dataset splits mostly share easy/hard curves between direct
  and middle.
- Source-group splits contain real complementarity: middle rescues many
  direct-bad curves.
- The current deployable gate barely captures that complementarity.

## Current Decision

Do not build a larger MoE yet.

Do not optimize LNN/KAN/deeper selector yet.

The next useful experiment is a focused selector diagnostic:

```text
1. Identify curves where direct early-Q is bad and middle is good.
2. Compare them against curves where both are good or both are bad.
3. Audit static descriptors, early slopes, source groups, duration, release span,
   and shape-family assignments.
4. Train a narrow gate only for the source-group regime if separability exists.
```

## Claim Boundary

Allowed claim:

> Early observations expose release-state information; middle-layer prediction
> improves over static middle-layer mapping, and source-group OOD contains
> diagnostic complementarity between direct and middle routes.

Not allowed yet:

> The middle-layer route beats black-box early-observation prediction.

Not allowed yet:

> MoE solves the route-selection problem.
