# Synthetic Hidden-State Identifiability Audit

Date: 2026-06-12

Script: `scripts/118_synthetic_hidden_state_identifiability_audit.py`

Output anchor: `outputs/118_synthetic_hidden_state_identifiability_audit/`

## Purpose

This is a diagnostic simulation, not a new prediction route.

The question is:

```text
If drug-release curves are driven by hidden microstructure/process variables
that are absent from ordinary static formulation tables, can we reproduce the
same bottleneck pattern seen in PLGA and liposome?
```

The synthetic world makes the true hidden release state observable to us:

- reported static descriptors: drug/polymer/formulation/medium numeric fields
- hidden state: porosity, surface drug fraction, size dispersion, residual
  solvent, MW dispersion, tortuosity, water uptake, autocatalysis
- release curve: monotone cumulative `Q(t)` generated from both reported and
  hidden variables

The model does not get hidden variables unless the row is explicitly marked as
an upper bound.

## Experimental Design

Each sample is one release curve. There is no timepoint-level split.

Primary split:

```text
source_group_5fold
```

This holds out synthetic sources to mimic paper/lab/source shift.

Prediction target:

```text
future Q(t)
```

Budgets:

```text
k = 0, 1, 2, 3, 5
```

The early observation schedule is:

```text
0.25, 1, 3, 7, 14 days
```

Compared inputs:

| Input | Meaning |
|---|---|
| `core_X` | minimal drug/polymer/formulation descriptors |
| `reported_X` | core descriptors plus medium/context fields |
| `reported_X_plus_proxy_H` | reported X plus noisy hidden-state proxies |
| `true_hidden_upper` | reported X plus the true hidden state; upper bound only |
| `reported_X_plus_early_Q` | reported X plus measured early release points |
| `reported_X_plus_Hhat_from_early_Q` | reported X plus hidden state estimated from early release points |

Primary model:

```text
ExtraTreesRegressor
```

Ridge is also run as a sanity check.

## Primary Source-Group Result

For `source_group_5fold` with ExtraTrees:

| k | reported_X RMSE | true_hidden_upper RMSE | early_Q RMSE | Hhat_from_early_Q RMSE |
|---:|---:|---:|---:|---:|
| 0 | 0.0748 | 0.0470 | 0.0748 | 0.0753 |
| 1 | 0.0749 | 0.0472 | 0.0741 | 0.0747 |
| 2 | 0.0835 | 0.0517 | 0.0739 | 0.0752 |
| 3 | 0.0890 | 0.0549 | 0.0568 | 0.0618 |
| 5 | 0.0723 | 0.0473 | 0.0295 | 0.0377 |

Interpretation:

- Reported static X leaves a large information gap.
- True hidden variables close that gap, so the synthetic world really is
  state-limited rather than model-limited.
- Early Q closes the gap sharply by k=3 and exceeds the hidden-variable upper
  row by k=5 because it observes the realized trajectory, not only the latent
  drivers. This does not mean early Q is more physical than true H; it means
  early Q is an empirical release-state measurement containing X, H, source
  effect, noise realization, model residuals, and equivalent curve state.
- Inferring hidden state from early Q is plausible but lossy: it trails direct
  early-Q prediction, which is expected.

## Decision Table

| Decision | Status | Evidence |
|---|---|---|
| reported X has an information gap | pass | k0 reported_X RMSE 0.0748 vs true_hidden_upper RMSE 0.0470 |
| noisy hidden proxies improve static X | pass | k0 proxy_H RMSE 0.0620 vs reported_X RMSE 0.0748 |
| early Q closes hidden gap by k=3 | pass | direct early-Q gap closure 94.3% |
| early Q can exceed hidden upper row | pass | k5 direct early-Q gap closure 171.2% |
| early Q to hidden state is plausible but lossy | pass | Hhat gap closure k3 79.8%, k5 138.5% |
| hidden recovery improves with observation budget | pass | mean hidden R2 k0 0.250 to k5 0.365 |

## Hidden-Strength Sensitivity Defense

The defense layer reruns the synthetic world with the same random seed and
same source/X structure while changing only the strength of hidden-state
control over curve formation.

| hidden_strength | reported_X k0 RMSE | true_H k0 RMSE | noisy_proxy_H k0 RMSE | early_Q k3 RMSE | early_Q k5 RMSE | reported_X - true_H gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.0460 | 0.0391 | 0.0461 | 0.0442 | 0.0268 | 0.0068 |
| 1.0 | 0.0745 | 0.0528 | 0.0674 | 0.0608 | 0.0324 | 0.0218 |
| 1.8 | 0.0961 | 0.0583 | 0.0749 | 0.0666 | 0.0356 | 0.0377 |

This is the key reviewer-defense result:

```text
as hidden state becomes more dominant,
reported_X-only becomes less sufficient,
true/proxy state measurements become more valuable,
and early release observations remain strong release-state measurements.
```

Percent gap closure can exceed 100% when the denominator is small or when
early Q directly observes the realized trajectory. Therefore the safer
manuscript emphasis is the absolute RMSE trend and the monotonic growth of the
reported_X-to-true_H gap, not the idea that early Q is "more physical" than H.

## What This Supports

This supports the project's current framing:

```text
Drug release prediction is release-state inference under limited descriptors
and costly observations.
```

It gives a controlled explanation for the real-data pattern:

1. Static descriptors can fail even when the curve is fully mechanistic in the
   synthetic ground truth.
2. Adding the right hidden microstructure/process variables would help.
3. If those variables are unmeasured, early release points can act as empirical
   release-state observations.
4. A model that maps early Q into a latent state is principled, but it must be
   judged as state inference, not as generic curve regression.

## What This Does Not Prove

This does not prove that real PLGA release is driven by exactly these hidden
variables.

It does not validate porosity, residual solvent, tortuosity, or autocatalysis
as fitted physical states in the real corpus.

It does not justify returning to generic neural world-model training.

It says only:

```text
The information-budget explanation is internally coherent: hidden release
state can produce static-feature failure, and early observations can recover
an equivalent predictive release state enough to forecast future release.
```

## Consequence For The Paper

This result should be used as a conceptual control figure or supplementary
diagnostic:

```text
In a synthetic world where the ground-truth hidden release state is known,
ordinary static descriptors under-identify future release; measured early
release observations close the gap because they provide empirical state
measurements.
```

It should not become the headline contribution. The headline remains the
real-data information-budget evidence across PLGA and liposome.
