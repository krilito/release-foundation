# PLGA Static-to-Early Proxy Probe

Date: 2026-06-11

## Question

Can static formulation descriptors infer the early release signal well enough
to replace measured early observation points?

This is the required precondition before spending effort on symbolic
regression or PINN distillation of a formula such as:

```text
static descriptors -> early Q/slope proxy -> future release
```

## Experiment

Script: `scripts/99_plga_static_to_early_proxy_probe.py`

Output: `outputs/99_plga_static_to_early_proxy_probe/`

The probe trains an ExtraTrees proxy from static descriptors to early release
features. Early sampling times are treated as known; only early Q values and
early slopes are predicted. A matched direct future model is then evaluated
under three conditions:

- `static_time`: static descriptors plus query time only.
- `true_static_early_time`: static descriptors plus measured early Q/slope.
- `pred_static_early_time`: the same true-early-trained model, but test early
  Q/slope features are replaced by static-predicted proxy features.

Splits reuse the existing PLGA controls:

- random k-fold
- source-group k-fold
- source-dataset LODO

Budgets: `1,2,3,5`.

## Main Results

The strict-split answer is mostly negative.

| Split | Budget | Static RMSE | True Early RMSE | Predicted Early RMSE | Recovery |
|---|---:|---:|---:|---:|---:|
| random-kfold | 5 | 0.107 | 0.088 | 0.130 | -1.15 |
| source-group-kfold | 5 | 0.267 | 0.165 | 0.256 | 0.11 |
| source-dataset-lodo | 5 | 0.272 | 0.163 | 0.290 | -0.16 |

Proxy quality tells the same story. Median early-Q proxy R2 is positive under
random split but negative under source-group and source-dataset splits:

- random k=5: median Q proxy R2 = 0.492
- source-group k=5: median Q proxy R2 = -0.229
- source-dataset k=5: median Q proxy R2 = -1.147

## Interpretation

Static descriptors can partially memorize early-release behavior inside random
splits, but they do not robustly reconstruct early observations under stricter
source shifts. Measured early Q/slope remains a real information source rather
than something current static descriptors can reliably synthesize.

This weakens the immediate case for symbolic regression or PINN as a
replacement for early observation. A symbolic formula distilled from the
current static descriptors would probably learn source-specific correlations,
not recover hidden microstructure or process state.

## Decision

Do not move symbolic regression or PINN into the main training loop.

Use them only as offline distillation after a static-to-early proxy has proven
positive under source-group or source-dataset splits. For now, the better path
is:

1. Treat measured early observations as the key latent-state reveal.
2. Use source-shift splits as the claim boundary.
3. If we want to remove early measurements, expand static descriptors with
   process and microstructure variables first.
4. Re-run this proxy probe after descriptor expansion.

