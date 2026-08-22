# optimize/ — Algorithm Optimization Module

## Results Summary

### 1. Zero-shot (descriptor → curve, cross321)

| Method | median R² |
|--------|-----------|
| Current NPE (weighted_mask1) | 0.540 |
| **MLP θ regression + normalization** | **0.629** |
| ExtraTrees direct Q | 0.585 |

Key fix: normalize θ targets before training.

### 2. Early validation (Q at 1,3,5,7 days → full curve, cross321)

| Method | future R² | frac ≥ 0 |
|--------|-----------|----------|
| ET early Q only | 0.43 | 0.58 |
| ET formulation + early Q | 0.53 | 0.62 |
| **Oracle θ → simulate (upper bound)** | **0.97** | 0.76 |

Oracle θ achieves 0.97 — the simulator's extrapolation is near-perfect.
The bottleneck is the early-Q-to-theta mapping.

### 3. What doesn't work

- End-to-end curve loss through simulator (ODE gradient explosion)
- Interaction features at n=259 (overfitting)
- NPE rejection sampling on small datasets

## Files

| File | Purpose |
|------|---------|
| `zeroshot_v2.py` | Zero-shot: normalized θ regression |
| `early_validation.py` | Early Q → full curve extrapolation |
| `temporal_embed.py` | Causal conv embedding |
| `calibration.py` | Temperature scaling + CRPS |
| `interaction_encoder.py` | Pairwise features |
| `fib_fast.py` | Sparse-grid FIB |
| `distill_fib.py` | FIB distillation |

## Run

```powershell
cd D:\release-foundation

# Zero-shot
.\.venv\Scripts\python.exe optimize\zeroshot_v2.py --device cpu --epochs 200

# Early validation
.\.venv\Scripts\python.exe optimize\early_validation.py --device cpu --epochs 200
```
