# 41 - Mechanistic Feasibility Criterion

Date: `2026-05-25`

## Why this exists

The current strongest predictor is not a deep end-to-end model. It is:

```text
formulation descriptors + Q(1,3,5,7)
        -> RF
        -> theta_9
        -> PLGABiphasic simulator
        -> full release curve
```

Script `38d` showed this decodes curves very well:

```text
RF->theta median curve R^2:
  random            0.957
  held-out drug     0.910
  held-out polymer  0.928
```

Script `39` then showed that per-parameter theta recovery is weak:

```text
median theta R^2 across params:
  random            +0.216
  held-out drug     +0.036
  held-out polymer  -0.023
```

This is not a contradiction. It is the main point:

```text
theta_pred != theta_oracle
but
ODE(theta_pred) ~= observed release curve
```

Therefore theta R^2 is the wrong criterion for the next method. It
measures whether we copied one arbitrary oracle least-squares solution,
not whether we found a physically credible mechanism explanation.

## Working principle

The project should stop treating theta as a unique label.

Better framing:

```text
A release curve admits many mathematical theta fits.
The useful task is to find the subset that is:
  - decodable to the observed curve,
  - physically valid,
  - compatible with formulation descriptors,
  - stable under noise / resampling,
  - cheap to infer.
```

Call this the **mechanistic feasibility criterion**.

## Candidate score

For a candidate parameter vector `theta` and formulation/early observation
pair `(x, y_early)`, define a score like:

```text
score(theta | x, y_early)
  = curve_decoding_score(theta)
  - physics_penalty(theta)
  - descriptor_mismatch(theta, x)
  - instability_penalty(theta)
  - compute_cost(theta)
```

Where:

- `curve_decoding_score`: full-curve or held-out-late R^2 / RMSE after
  ODE simulation.
- `physics_penalty`: violates monotonicity, Q bounds, burst timing, or
  prior support.
- `descriptor_mismatch`: theta lies in a region rarely compatible with
  similar formulations.
- `instability_penalty`: theta changes wildly under bootstrap, added
  observation noise, or nearby early time subsets.
- `compute_cost`: number of free parameters, restarts, solver calls, or
  time needed to recover the explanation.

The target is not:

```text
minimize ||theta_pred - theta_oracle||
```

The target is:

```text
maximize feasible decodability per unit cost
```

## Proposed experiment

Generate a large candidate bank, then learn the fastest reliable shortcut.

For each curve:

1. Sample or optimize many candidate theta vectors.
2. Simulate each candidate through the ODE.
3. Score each candidate by the feasibility criterion above.
4. Keep the top candidate family, not only one best point.
5. Train a fast model to map `(x, y_early)` to either:
   - the best theta candidate,
   - a small candidate set,
   - or a low-dimensional active subset plus fitted values.

This directly matches the user's intuition:

```text
fit many possible explanations,
let the model discover which parameter combinations are stable,
fast,
low-cost,
and still decode correctly.
```

## What to compare

Use 38d's current winner as the baseline:

```text
B2 RF->theta:
  random            0.957
  held-out drug     0.910
  held-out polymer  0.928
```

New methods should be compared on:

1. Curve R^2 under the same three CV schemes.
2. Tail robustness: fraction with R^2 >= 0 and R^2 >= 0.9.
3. Stability: variation of selected theta under bootstrap/noise.
4. Cost: number of free parameters and ODE evaluations.
5. Interpretability: active set / regime / descriptor consistency.

## First implementation sketch

Candidate generators:

- RF point prediction from `38d`.
- RF ensemble tree-level theta samples.
- CFE multistart fits on `t <= 7d`.
- Active-set fits from `29` style greedy subsets.
- Regime-conditioned fits from `36` style modal subsets.
- Random prior samples around descriptor-neighborhood theta banks.

Candidate filters:

- Must stay within prior bounds.
- Must produce monotone Q in `[0, Q_max]`.
- Must fit early observations.
- Must not require all 9 free parameters unless the lower-cost
  alternatives fail.

The first script should probably be:

```text
scripts/40_theta_candidate_feasibility.py
```

Keep it diagnostic first. Do not introduce a new model until the score
actually separates good and bad theta candidates.

## First execution result

Implemented as:

```text
scripts/40_theta_candidate_feasibility.py
```

Full run on cross321 (`n=259`, 5-fold random / group-by-drug /
group-by-polymer):

```text
Method            random   by_drug  by_polymer
B1 CFE            0.8605   0.8605   0.8605
B2 RF->theta      0.9512   0.9124   0.9140
B3 RF->Q          0.8978   0.8599   0.8644
40 feasibility    0.9350   0.9248   0.9251
E oracle          0.9976   0.9976   0.9976
```

Cost:

```text
40 feasibility median active params = 3
40 feasibility median nfev          = 8
```

Interpretation:

- The first feasibility selector is not a universal R^2 upgrade over
  RF->theta.
- It is a useful mechanism-compression result: OOD median R^2 is slightly
  above RF->theta while the chosen explanation usually uses only 2-4
  active parameters.
- The remaining problem is tail robustness. The next version should
  either improve the score/candidate family or learn a fast selector
  from the candidate bank.

## Paper-level framing

This becomes the method story if it works:

> We do not seek a unique ODE parameter vector. Instead, we infer a
> descriptor-conditioned feasible family of mechanistic explanations and
> choose the most stable low-cost member that decodes the observed release
> trajectory.

Short version:

```text
Prediction is X -> feasible theta family -> ODE decoding,
not X -> unique theta.
```

## Input-source guardrail from 45

The `45` input-source ablation changes how the feasibility story should
be worded.

Best median R^2:

```text
cross321
  random:      formulation_only 0.909 | early_only 0.959 | formulation_plus_early 0.963
  by_drug:     formulation_only 0.672 | early_only 0.941 | formulation_plus_early 0.922
  by_polymer:  formulation_only 0.666 | early_only 0.949 | formulation_plus_early 0.944

internal181
  random:      formulation_only 0.932 | early_only 0.946 | formulation_plus_early 0.971
  by_drug:     formulation_only 0.661 | early_only 0.932 | formulation_plus_early 0.938
  by_polymer:  formulation_only 0.525 | early_only 0.907 | formulation_plus_early 0.922
```

Implication:

- The high-performing theta bridge is primarily early-release informed,
  not formulation-only.
- Formulation descriptors are still useful as a conditioning/prior signal,
  especially on internal181, but they are not sufficient for OOD by
  themselves.
- A future feasibility criterion should treat early observations as the
  kinetic evidence and formulation descriptors as the feasible-domain prior.

## Direct-curve control from 46

Script `46_direct_curve_rf_input_ablation.py` tests the most important
control: remove theta entirely and train pure tabular ML to predict the
release curve grid directly.

Best direct-curve median R^2 vs best theta-route median R^2:

```text
cross321
  random:      direct 0.837/0.907/0.925 vs theta 0.909/0.959/0.963
  by_drug:     direct 0.619/0.887/0.879 vs theta 0.672/0.941/0.922
  by_polymer:  direct 0.664/0.885/0.893 vs theta 0.666/0.949/0.944

internal181
  random:      direct 0.866/0.870/0.920 vs theta 0.932/0.946/0.971
  by_drug:     direct 0.620/0.852/0.886 vs theta 0.661/0.932/0.938
  by_polymer:  direct 0.467/0.834/0.801 vs theta 0.525/0.907/0.922
```

Each row lists `formulation_only / early_only / formulation_plus_early`.

Implication:

- Theta is not an arbitrary middle layer. It is a useful ODE-decodable
  bottleneck that beats direct curve regression from the same inputs.
- This gives the paper a stronger method claim than "RF is strong":
  early observations + tree ensemble + theta bottleneck + ODE decoding
  outperforms direct RF/ET curve prediction.

## Regime-gated theta CV guardrail from 50

Script `50_regime_gated_theta_cv.py` tested the next obvious extension
after the wet-curve result in `49`: train a fast/slow gate from train-
fold t50, fit branch-specific theta mappers, and compare them to global
RF/ET theta models under random / held-out-drug / held-out-polymer CV.

Result:

```text
No early refinement:
  regime gating alone usually does not beat global RF/ET theta.

With 5 early-only least-squares evaluations:
  cross321 by_drug:     global zavg 0.927 -> regime-hard-refined 0.943
  cross321 by_polymer:  global zavg 0.935 -> regime-hard-refined 0.946
  internal181 by_polymer: global zavg 0.918 -> regime-hard-refined 0.929
  internal181 by_drug:  global-refined wins over regime-refined
```

Implication:

- The general upgrade is **early-Q local theta refinement**, not the
  regime gate by itself.
- Regime-aware branches are useful as an OOD/wet stress-case option,
  especially held-out polymer, but should not be sold as a universal
  predictor.
- This supports the mechanistic-feasibility direction: predict a theta
  family first, then use available early observations as a cheap physical
  selection/refinement signal.
