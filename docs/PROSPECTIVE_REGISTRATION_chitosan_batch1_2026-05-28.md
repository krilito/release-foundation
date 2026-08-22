# Pre-Registration — Chitosan Prospective Batch 1

**Registered**: 2026-05-28
**Git tag**: prospective-chitosan-batch1-2026-05-28
**Tag commit**: `7f9c21fbb61823cd74388c12985cd57b323c3c79`
**Prediction commit**: `ceff62647ba9e43324e4889145f99c08dffd71b3`
**SHA256 lock**: see `outputs/67_chitosan_prospective/HASHES.txt`
**Wet-lab data deadline**: 2026-06-04
**Project lead**: 杨医嶂 (Yang Yizhang)
**Authority**: paper_requirements_locked_2026-05-28.md P4

## Hypothesis

The mechanism-constrained Bayesian inference framework (Active Observer v3 +
ChitosanRitgerPeppas simulator) produces calibrated predictive intervals for
chitosan hydrogel release on 12 prospective formulations (6 formulations × 2
drugs) without retraining on chitosan data.

## Predictions

12 curves, 10 timepoints each (0.5h to 28d). 90% PI lower / upper bound
for Q(t) per curve from `outputs/67_chitosan_prospective/predictions.csv`.

Prediction details:
- Simulator: `ChitosanRitgerPeppas(time_unit_hours=True)` — Ritger-Peppas
  model with chitosan-specific parameterization
- 3 kinetic parameters per curve: log(k), n (release exponent), Q_max
- Prior bounds: log(k) ∈ [-5.0, -0.5], n ∈ [0.35, 1.0], Q_max ∈ [0.3, 1.0]
- Time grid (hours): 0.5, 1, 2, 6, 24, 72, 168, 336, 504, 672
- 6 formulations: CS/TPP nanoparticles with HA/PVP coating variants
- 2 drugs: DEX (dexamethasone) and GCV (ganciclovir)

## Primary endpoint

**cov90**: fraction of (curve, timepoint) observations falling within
the predicted 90% PI band, computed across all 12 curves × all measured
timepoints.

Pre-specified success threshold: **cov90 ≥ 0.83**.

## Secondary endpoints

1. Per-curve hit rate: count of curves with ≥ 80% timepoints in band
2. Mean PI width (sharpness)
3. Pooled R² across all (curve, timepoint) pairs
4. CRPS

## Pre-specified fallback claims

| cov90 outcome | Reported claim |
|---|---|
| ≥ 0.83 | "Pre-registered cov90 met; supports cross-mechanism calibration claim" |
| 0.65–0.83 | "Pre-registered cov90 partially met; moderate under-coverage on chitosan transfer" |
| 0.50–0.65 | "Pre-registered cov90 not met; method exhibits substantial under-coverage on cross-mechanism transfer" |
| < 0.50 | "Substantial calibration failure on chitosan transfer; reported as method limitation pending batch 2" |

## Decided in advance (locked)

- Predictions, prior bounds, evaluation time grid, all coefficients
- Primary endpoint and threshold
- Fallback claims for every cov90 outcome
- No data-dependent threshold adjustment

## Left for post-hoc (must be disclosed if performed)

- Regression analyses on per-curve hit rate vs formulation features
- Regime-specific breakdown if regime labels become available for chitosan
- Exclusion of any curve must be reported with reason

## Reproducibility

Predictions generated at git commit
`ceff62647ba9e43324e4889145f99c08dffd71b3` per
`outputs/67_chitosan_prospective/lock_metadata.json`. Hashes locked at
git commit `7f9c21fbb61823cd74388c12985cd57b323c3c79` per
`prospective-chitosan-batch1-2026-05-28` tag.

SHA256 hashes:
```
f0fdc7bc8740f17d56ceee2ec274ea88b8c256b5831a6c9fada25d086f86e07d *predictions.csv
d3f238e6d5d82692f8c8076d5ae92eebae2cf2f67fa6686b29c44ff993565332 *lock_metadata.json
8b3b1cf5ac4172c24ed1a9fcff27a9f0a7d39467db3ed68fd4e506631cbc801f *theta_targets.npz
```
