# Sprint 1 Status Note

Date checked: 2026-05-22

## Working branch

Current branch:

```text
phase2-sprint1-burst
```

This branch contains the ADR-026 continuous-burst ODE experiment.

## Goal

Sprint 1 asks whether replacing the old instant burst jump with a continuous
burst term can remove the structural Q0 residual while preserving SBI
calibration and deployment performance.

Old 8-parameter layout:

```text
[log_kw, log_kh, log_alpha, log_kd, log_ke, m_crit, q_burst, Q_max]
```

Current 9-parameter layout:

```text
[log_kw, log_kh, log_alpha, log_kd, log_ke, m_crit, q_burst, log_tau_burst, Q_max]
```

Model-form change:

```text
old: Q(0) = q_burst
new: Q(0) = 0
     dQ/dt += (q_burst / tau_burst) * exp(-t / tau_burst) * (Q_max - Q)
```

## Environment status

CUDA torch was restored after the venv had been overwritten with CPU torch.

Verified environment:

```text
torch 2.11.0+cu128
torch.cuda.is_available() == True
CUDA runtime 12.8
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

Do not use `uv run python ...` for training, SBC, or deployment evaluation in
this branch. Use:

```powershell
.\.venv\Scripts\python.exe <script>
```

## Code cleanup completed

- `README.md` now avoids `uv run` in the CUDA-sensitive quickstart.
- `scripts/02_oracle_sweep.py`, `scripts/03_identifiability_replay.py`,
  `scripts/04_train_curve_posterior.py`, and
  `scripts/05_sbc_curve_posterior.py` now show venv-python commands.
- `scripts/08_pipeline_sanity.py` was updated from the old 8-parameter
  q_burst-as-Q0 assumption to the current 9-parameter ADR-026 model.
- `posterior.py` now explicitly moves trained/loaded SBI estimators to their
  target device and moves `sample()` / `log_prob()` inputs to the estimator's
  actual device. This fixed the CUDA/CPU mismatch in `scripts/10_eval_deployment.py`.
- `tests/test_posterior.py` now includes CUDA load/sample/log_prob regression
  coverage when CUDA is available.
- `scripts/10_eval_deployment.py --cross-doi` now honors `--device` instead of
  hard-coding CPU for the loaded `r_psi`.

## Verification

Completed checks:

```text
pytest: 40 passed
ruff:  posterior.py, tests/test_posterior.py, scripts/10_eval_deployment.py pass
```

Pipeline sanity:

```text
output: outputs/sprint1_08_pipeline_sanity
q_burst KS p = 0.0733 PASS
Q_max   KS p = 0.2611 PASS
verdict: PIPELINE HEALTHY
```

## New trusted Sprint 1 artifacts

All trusted Sprint 1 outputs are under `outputs/sprint1_*`.

Stage 1 curve posterior:

```text
output: outputs/sprint1_04_curve_posterior/posterior.pt
device: cuda
n_simulations: 50000
elapsed: 824.5 s
final_train_loss: -3.1147
final_val_loss:   -2.9471
```

Multi-seed SBC:

```text
output: outputs/sprint1_05_sbc_multiseed/summary.txt
n_sbc: 300
n_posterior_samples: 1000
seeds: 10
primary criterion: c2st_ranks max <= 0.60
overall verdict: PASS
```

Per-parameter c2st max:

```text
log_kw          0.5583 PASS
log_kh          0.5600 PASS
log_alpha       0.5417 PASS
log_kd          0.5667 PASS
log_ke          0.5667 PASS
m_crit          0.5900 PASS
q_burst         0.5467 PASS
log_tau_burst   0.5767 PASS
Q_max           0.5817 PASS
```

Stage 2 descriptor posterior:

```text
output: outputs/sprint1_09_descriptor_posterior/posterior.pt
curves kept: 133 high-quality of 181
n_pairs: 8512
device: cuda
teacher sampling: 36.2 s
training: 140.7 s
final_train_loss: -2.8026
final_val_loss:   -2.8521
```

Internal leave-one-drug-out deployment:

```text
output: outputs/sprint1_10_eval_deployment/summary.txt
curves kept: 133
folds: 21 drugs
wallclock: 44.6 min
median R^2: +0.6382
mean R^2:   +0.4042
median MAE: 0.1488
median Q0 residual: +0.0000
verdict: PASS vs internal 0.50 median R^2 gate
```

GroupKFold-by-curve deployment:

```text
output: outputs/sprint1_10_eval_deployment/by_curve/summary.txt
curves kept: 133
folds: 5
wallclock: 9.3 min
median R^2: +0.9413
mean R^2:   +0.8095
median MAE: 0.0622
median Q0 residual: +0.0000
verdict: no regression signal
```

Cross-DOI 321 deployment:

```text
output: outputs/sprint1_10_eval_deployment/cross_doi_summary.txt
curves kept: 259 high-quality of 321
wallclock: 146.5 s
median R^2: +0.3502
mean R^2:   -0.0033
median MAE: 0.2067
median Q0 residual: +0.0000
n curves R^2 > 0:    166 / 259
n curves R^2 > 0.30: 134 / 259
verdict: PASS vs median R^2 > 0.30, but mean R^2 is slightly negative
```

## Current verdict

Sprint 1 has passed the main technical acceptance checks:

- CUDA environment is repaired.
- The 9-parameter continuous-burst simulator is test-covered.
- The SBI pipeline trains and runs on CUDA.
- Multi-seed SBC passes by the project primary c2st criterion.
- Q0 residual median is now zero in internal and cross-DOI evaluations.
- Internal LODO median R² remains above the 0.50 target.
- Cross-DOI median R² remains above the 0.30 target.

The important unresolved caveat is cross-DOI robustness: median R² passes, but
mean R² is slightly negative. That means a subset of 321 curves is still failing
badly enough to drag the average below zero. Do not overclaim "external
generalization solved"; the correct claim is narrower:

```text
ADR-026 removes the Q0 model-form artifact and preserves median deployment
performance, but cross-DOI tail failures still need error analysis.
```

## Next scientific step

Do not change model architecture yet. First inspect the 321 failure tail:

```powershell
.\.venv\Scripts\python.exe scripts\<new_or_existing_error_analysis>.py `
  --metrics outputs\sprint1_10_eval_deployment\cross_doi_per_curve_metrics.csv
```

## 2026-05-23 follow-up status

The failure-tail work has now moved past pure diagnosis. The project should
not restart from the original `median passes / mean slightly negative`
interpretation.

### Strongest current Stage-2 SBI candidate

The strongest descriptor-to-parameter route is now:

```text
missingness-aware + weighted distillation
short name: weighted_mask1
```

Trusted result:

```text
output: outputs/21_eval_plga_fast2x_mask1/cross_doi_summary.txt
median R²: +0.5404
mean R²:   +0.1160
median MAE: 0.1874
```

Interpretation:

```text
This is the best current SBI deployment candidate on 321.
Do not keep treating the original sprint1 descriptor posterior as the
only live Stage-2 route.
```

### World-model exploration status

Three partial-observation routes were explored.

1. `scripts/23_prefix_world_model_benchmark.py`

   Result to keep:

   ```text
   outputs/23_prefix_world_model_benchmark_smoke32_relaxed/summary.txt
   ```

   Main finding:

   ```text
   the current full-curve q_phi is not prefix-aware;
   coercing it into prefix updates causes OOD behavior and unstable
   rejection sampling.
   ```

2. `scripts/24_prefix_curve_ssl_baseline.py`

   Result to keep:

   ```text
   outputs/24_prefix_curve_ssl_baseline_full259/summary.txt
   ```

   Headline:

   ```text
   1d prefix median R² = +0.081
   3d prefix median R² = +0.349
   7d prefix median R² = +0.045
   ```

   Meaning:

   ```text
   curve-only partial-observation forecasting carries real transfer signal;
   3d is the current sweet spot.
   ```

3. `scripts/25_prefix_theta_regressor.py`

   Result to keep:

   ```text
   outputs/25_prefix_theta_regressor_full259/summary.txt
   ```

   Headline:

   ```text
   1d prefix median R² = -1.632
   3d prefix median R² = +0.273
   7d prefix median R² = -0.586
   ```

   Meaning:

   ```text
   prefix-aware latent inference is viable,
   but the current 7-parameter theta bottleneck is weaker than the
   curve-only SSL baseline.
   ```

### Current recommended next step

The project should now pivot from fixed-prefix baselines to a real
partial-observation world model:

```text
masked-time / irregular-observation model
181 full curves -> random masking -> full-curve reconstruction
321 sparse observations -> predict the hidden suffix
```

This is the right successor to scripts 23–25. It is more faithful to the
"observe -> summarize -> simulate -> compare" idea than the existing
full-curve q_phi.

### First full script26 attempt

Implemented:

```text
scripts/26_iterative_world_model.py
```

This version used:

```text
- Transformer observation encoder
- teacher theta from full-curve q_phi on 181
- bounded theta branch
- learned residual branch
- one residual-guided refinement step
```

Result kept at:

```text
outputs/26_iterative_world_model_v2/summary.txt
```

Headline:

```text
1d stage0 median R² = -1.631
3d stage0 median R² = -0.059
7d stage0 median R² = -0.247

1d stage1 median R² = -2.293
3d stage1 median R² = -0.287
7d stage1 median R² = -1.053
```

Interpretation:

```text
The first true iterative world-model attempt did not beat the simpler
curve-only SSL baseline from script24.
The refinement step over-corrected on external 321.
Current evidence favors learned curve state over explicit theta+residual
as the next world-model direction.
```

Questions to answer before any new modeling:

- Are the worst 321 curves concentrated in a few release-shape regimes?
- Are failures linked to descriptors imputed from 181 means?
- Are failures mostly extrapolation in polymer MW, DLC, LA/GA, or time support?
- Would stratified reporting make the median PASS scientifically honest, or is
  the mean R² failure exposing a real deployment boundary?

## Artifacts not to use as evidence

These older artifacts came from CPU torch or interrupted runs and should be
treated only as failure-history records:

```text
outputs/04_curve_posterior/posterior.pt
outputs/04_curve_posterior/training_log.txt
outputs/sprint1_run_20260522_104249.log
```
