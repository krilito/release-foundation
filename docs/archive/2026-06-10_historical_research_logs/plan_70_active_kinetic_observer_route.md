# Route 70: Active Kinetic Observer

Date: 2026-05-27.

## One-line Direction

When formulation descriptors are under-informative, use a small number of early
release measurements to actively identify latent release kinetics, update a
mechanistic particle posterior, and predict the remaining curve with calibrated
uncertainty.

This is the route that turns the project from "descriptor-only release
prediction" into "sparse experimental observation of release dynamics."

## Paper Positioning

Working title:

> Uncertainty-calibrated active kinetic observation for sparse drug-release
> curve prediction

Best claim:

> Two active release measurements can substantially improve a mechanistic prior
> under grouped evaluation while retaining calibrated uncertainty.

Claim boundary:

- Do not sell this as the global point-prediction champion.
- Sell it as a decision-oriented observer with uncertainty, posterior updating,
  and experiment-selection logic.
- Treat directQ, XGB, RF, and LightGBM as strong descriptor/early-point
  predictors, not as the same class of method.

## Method Stack

| Layer | Current implementation | Paper role |
|---|---|---|
| Mechanism | `PLGABiphasic.simulate_numpy()` | Physics-constrained rollout |
| Prior | ExtraTrees tree particles plus KNN similar-formulation particles | Descriptor-conditioned kinetic prior |
| Observation | One-point and two-point release measurements | Sparse experimental evidence |
| Update | Particle likelihood reweighting | Mechanistic posterior update |
| Selection | Variance-reduction utility with time-cost penalty | Active experimental design |
| Calibration | Split/local conformal intervals | Coverage guarantee for sparse data |
| Baselines | zero-early, fixed time, fixed four-point, directQ | Reviewer defense |

## Current Evidence

| Evaluation | Best support for route | Main caveat |
|---|---|---|
| Legacy random split, 38 curves | active two-point `0.175` beats zero prior `0.207` and fixed four-point `0.178` | directQ remains `0.120` |
| Current v3 random split, 38 curves | active two-point `0.138` improves zero prior `0.164` with coverage `0.930` | fixed 21d is better at `0.129` |
| DP group split, 23 curves | active two-point `0.155` beats directQ `0.182` | fixed 14d/21d remain slightly better |
| Polymer-family GroupKFold, 148 curves | active two-point improves prior `0.226 -> 0.185`, coverage `0.973` | intervals are wide, width90 `1.076` |

The strongest table for the route is the 148-curve polymer-family GroupKFold,
because it tests grouped generalization rather than another random split.

## Route Gates

| Gate | Requirement | Current status |
|---|---|---|
| G1: prior update helps | posterior RMSE below zero-early prior | Passed in all current ledgers |
| G2: uncertainty calibrated | conformal coverage90 at least 0.85 | Passed, but sometimes conservative |
| G3: active competitive with fixed | active two-point close to best fixed schedule | Partially passed |
| G4: strong baseline defense | directQ, fixed14, fixed21, fixed four included | Incomplete for GroupKFold |
| G5: paper-grade statistics | paired CI or significance test | Not done |
| G6: sharp intervals | width/CRPS acceptable, not only coverage | Not done |

## Next Execution Plan

1. Close the GroupKFold baseline gap.
   Add directQ, fixed 14d, fixed 21d, and fixed four-point to the same
   polymer-family 5-fold table as prior/active.

2. Add paired statistics.
   For each curve, compare active two-point against prior-only, fixed 14d,
   fixed 21d, fixed four-point, and directQ where available. Report paired
   bootstrap confidence intervals for RMSE difference and CRPS difference.

3. Make uncertainty sharper.
   Keep coverage90 above 0.85, but optimize and report width90, width80, and
   CRPS. A wide conformal interval is honest but not yet useful.

4. Turn active time selection into a scientific result.
   Test whether selected times consistently map to burst, diffusion, and
   erosion windows. If the policy collapses to 0.5d plus 21d, frame it as
   discovered information windows and benchmark it against fixed windows.

5. Connect to the larger world-model story.
   Use the observer as the experimental-decision module. RF/XGB/LGB/directQ
   answer "what curve is likely"; Active Observer answers "what should I measure
   next to know the curve."

## Figures Needed

| Figure | Message |
|---|---|
| Posterior shrinkage | Early observations reduce plausible kinetic trajectories |
| Active time distribution | The method discovers information-rich release windows |
| Representative curves | Conformal intervals are calibrated and visually honest |
| Family-wise GroupKFold | Generalization works for major families and fails transparently for tiny families |
| Baseline frontier | directQ wins point RMSE in random split, observer wins calibrated decision utility |

## Recommended Manuscript Spine

1. Descriptor-only prediction is under-identified for PLGA release.
2. A mechanistic particle prior represents plausible latent kinetics.
3. Sparse observations update the kinetic posterior without retraining.
4. Active utility chooses informative time points under experimental cost.
5. Conformal calibration makes the uncertainty reportable.
6. Grouped evaluation shows posterior improvement and reveals extrapolation
   limits.

This route is worth keeping as a core paper module. It should not be diluted
into another model-zoo comparison.
