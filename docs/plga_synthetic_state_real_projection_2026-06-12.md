# PLGA Synthetic-State Real Projection Bridge

Date: 2026-06-12

Script: `scripts/119_plga_synthetic_state_real_projection.py`

Output anchor: `outputs/119_plga_synthetic_state_real_projection/`

## Purpose

This experiment bridges the synthetic hidden-state audit to real 321 PLGA
release curves.

The question is:

```text
Can a synthetic release world with known hidden states define H_like
release-state coordinates, and do real PLGA curves projected into those
coordinates align with table-level formulation/process proxies?
```

The answer is partial:

```text
yes, real curves show simulation-aligned proxy structure;
but the 118 synthetic world does not cover most real 321 curve shapes.
```

Therefore 119 is a cautious bridge diagnostic, not a validated real-PLGA
hidden-state model.

## Inputs And Outputs

Real data:

```text
D:/chemical-world-model-v0/datset/321PLGA/
A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles/
mp_dataset_initial.xlsx
```

Primary real subset:

```text
321 real formulation curves loaded
128 curves projectable with max time >= 30 days
```

Main outputs:

| Output | Role |
|---|---|
| `synthetic_inversion_metrics.csv` | verifies synthetic `Q(t) -> H_like` inversion |
| `real_projected_states.csv` | projected real 321 curves in H_like coordinates |
| `proxy_validation_summary.csv` | H_like vs table proxies with shuffle null |
| `nearest_synthetic_neighbors.csv` | case-level synthetic analogs |
| `projection_coverage_summary.csv` | real-to-synthetic curve support and OOD fraction |
| `curve_embedding_baseline_summary.csv` | H_like vs PCA curve embedding vs Hill/Weibull theta |
| `decision_table.csv` | claim-strength gate |

## Synthetic Inversion

The synthetic inversion sanity check uses the 118 synthetic world and trains:

```text
full_Q -> H_like
early_Q_k3 -> H_like
early_Q_k5 -> H_like
x_only -> H_like
```

All-coordinate held-out performance:

| Mode | R2 | RMSE |
|---|---:|---:|
| full_Q | 0.322 | 0.191 |
| early_Q_k3 | 0.215 | 0.206 |
| early_Q_k5 | 0.289 | 0.196 |
| x_only | 0.277 | 0.198 |

This is enough for a diagnostic projection, but not strong enough to claim
accurate recovery of individual physical hidden variables.

## Proxy Validation

The real projected H_like coordinates show table-proxy structure:

```text
proxy associations exceeding shuffle null: 103
best numeric association strength: 0.575
DOI / formulation-method group hits: 25
```

Examples of associations exceeding shuffle null include:

- DOI group structure across several H_like coordinates.
- Drug Loading Capacity aligned with multiple H_like coordinates.
- Initial Drug-to-Polymer Ratio aligned with multiple H_like coordinates.
- Polymer Mw aligned with some H_like coordinates.

This supports the claim that real PLGA curves contain projection-accessible
release-state structure. It does not prove the coordinates are literal
porosity, tortuosity, residual solvent, or autocatalysis.

## Coverage Gate

The coverage gate is the most important limitation:

| Metric | Value |
|---|---:|
| n_real_curves | 321 |
| n_projectable_full | 128 |
| n_projectable_k3 | 128 |
| n_projectable_k5 | 128 |
| median nearest synthetic curve distance | 0.569 |
| p90 nearest synthetic curve distance | 1.188 |
| synthetic self-neighbor p90 | 0.227 |
| fraction within synthetic p90 | 0.086 |
| fraction flagged OOD | 0.914 |

This means most real 321 curves sit outside the curve-shape support of the
current 118 synthetic world.

The safe interpretation is:

```text
119 finds real proxy structure after projection,
but 118 is not yet a validated simulator covering the 321 corpus.
```

## Embedding Baseline

The embedding baseline asks whether H_like is merely another curve embedding.

Compared embeddings:

```text
H_like
PCA_curve_embedding
Hill_Weibull_theta
```

Key pattern:

- H_like is competitive for Drug Loading Capacity, Initial Drug-to-Polymer
  Ratio, Polymer Mw, Formulation Method, and DOI.
- PCA curve embedding beats H_like for some proxies, especially DOI and some
  table descriptors.
- Hill/Weibull theta also captures DOI and some formulation structure.

This means H_like is not uniquely dominant. It is a useful simulation-aligned
coordinate system, but generic curve-language embeddings also capture part of
the same real-data structure.

That is not a failure. It supports the larger thesis:

```text
the observable release curve carries state information that static descriptors
do not fully encode.
```

## Decision Table

| Decision | Status | Evidence |
|---|---|---|
| synthetic Q to H inversion reliable | pass | full_Q all-coordinate R2 = 0.322 |
| real H_like aligns with table proxies | pass | 103 associations exceed shuffle null |
| associations exceed shuffle null | pass | n_hits = 103 |
| curve-projected H_like differs from X-only H_like | pass | RMSE = 0.248 |
| method or DOI group structure present | pass | 25 DOI/method hits |
| projection coverage supported | warn | fraction flagged OOD = 0.914 |
| safe to use as bridge evidence | warn | use only as cautious bridge diagnostic |

## Manuscript Use

Allowed claim:

```text
Projection of real 321 PLGA curves into simulation-informed H_like coordinates
reveals associations with table-level formulation/process proxies, indicating
that real release curves carry state-like information beyond reported static
descriptors.
```

Required caveat:

```text
Because 91.4% of projectable real curves are outside the current synthetic
curve-support threshold, this result is hypothesis-generating and cannot be
used as validation of the 118 synthetic simulator.
```

Forbidden claim:

```text
We inferred real porosity, tortuosity, residual solvent, or autocatalysis from
the literature curves.
```

The right role for 119 is a bridge figure or supplementary diagnostic between:

```text
118 synthetic hidden-state audit
and
real PLGA information-budget evidence.
```
