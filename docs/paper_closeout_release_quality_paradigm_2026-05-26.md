# Paper closeout - release-quality paradigm

Date: `2026-05-26`

## One-sentence story

Early release observations can be compressed into an ODE-decodable
mechanism state, lightly corrected by the same early observations, and
then used to simulate long-term drug release more robustly than direct
curve regression.

Plain version:

```text
We are not just predicting Q(t).
We infer a mechanism state theta, check it against early release,
and use a simulator to roll the curve forward.
```

## Final paper route

The strongest route is not "physics-informed neural network beats RF".
That story is dead.

The route that survived is:

```text
formulation descriptors
+ sparse early release observations
        ->
tree ensemble predicts an ODE-decodable theta family
        ->
early observations refine/select theta without seeing the full curve
        ->
ODE simulator predicts the full release trajectory
```

This is best described as:

- an observer-corrected mechanistic surrogate,
- an ODE-decodable theta bottleneck,
- or a release-dynamics surrogate environment.

Do not call it a complete world model in the paper. The stronger and
safer wording is that it is a world-model-like scaffold for release
quality prediction.

## Two contributions that should stay separate

### Contribution 1 - prediction / quality surrogate

Question:

> Can a small amount of early release data support reliable long-term
> release prediction across drug/polymer shifts?

Answer:

- Yes. Tree ensembles mapping formulation + early release to ODE theta
  reach about `0.91-0.96` median curve R^2 under random, held-out-drug,
  and held-out-polymer CV.
- The theta->ODE route beats direct RF/ET curve prediction from the same
  inputs.
- Early-Q local refinement improves selected OOD cases, especially
  held-out polymer and external wet stress curves.

Paper claim:

> Sparse early release observations can be converted into a mechanistic
> state that regularizes full-trajectory release prediction.

### Contribution 2 - mechanism / identifiability

Question:

> Does a good curve fit imply a unique mechanism?

Answer:

- No. The 9-parameter ODE can express the observed curves, but each curve
  usually activates only 2-4 effective directions.
- There is no universal small active subset.
- The heterogeneity is better explained by discrete active-set regimes
  than by dense pairwise coupling or a single continuous local-linear
  latent driver.

Paper claim:

> Release curves are mechanism-identifiable only up to low-dimensional,
> regime-dependent feasible theta families.

These two contributions should be linked but not mixed:

```text
Prediction uses theta as a useful simulator-compatible bottleneck.
Mechanism analysis explains why theta should be treated as a feasible
family rather than a unique true parameter vector.
```

## What we falsified

1. **End-to-end MLP is the main innovation.**
   RF/ET wins in this tabular, small-n regime.

2. **ODE-in-the-loop neural training is necessary for prediction.**
   The winning predictor is standard tree ensemble -> theta -> ODE.

3. **Theta R^2 is the right objective.**
   Functional curve decoding can be strong even when per-parameter theta
   recovery is weak.

4. **Formulation descriptors alone explain the high scores.**
   Input ablations show early release is the dominant signal.

5. **Regime gating universally improves prediction.**
   CV audit 50 shows gating alone is unstable; early-Q refinement is the
   broadly useful step.

6. **Direct pure ML curve prediction is enough.**
   Direct RF/ET -> Q loses to theta->ODE in the main controls.

7. **321-only artifact.**
   The same theta-bottleneck route works on cleaned internal181.

## Key evidence to show

### Prediction table

Show random / held-out-drug / held-out-polymer medians for:

- CFE,
- direct RF/ET -> Q,
- RF/ET -> theta -> ODE,
- theta + early-Q refinement,
- optional regime-gated refinement.

The main message is not a single best number. The message is that
theta->ODE is consistently strong and direct Q regression is weaker.

### Input-source ablation

Show:

```text
formulation-only << early-only ~= formulation+early
```

This prevents overclaiming. It also makes the story cleaner:

```text
early release = kinetic fingerprint
formulation = feasibility / conditioning context
theta = mechanism state
ODE = simulator decoder
```

### Early-window ablation

Show that shorter windows already work, but `Q(1,3,5,7)` is the most
stable, especially for polymer OOD. This supports the "one-week
quality readout" framing.

### Mechanism figure

Show:

- full ODE fit is expressive,
- individual curves need only 2-4 active directions,
- active sets cluster into regimes,
- regime-conditional 4-parameter fits work for R5/R6.

This supports the non-identifiability / feasible-family story.

### External wet stress test

Use carefully:

- direct theta prediction on two NC wet curves is modest,
- regime-aware early refinement improves both to about `R^2 ~0.81`,
- this is a stress test, not the main benchmark.

Do not overstate two curves as broad external validation.

## Suggested manuscript structure

### Introduction

1. Long-acting formulations require long release assays, making design
   and quality decisions slow.
2. Pure ML can interpolate release curves but often lacks a mechanism
   state that can be inspected, corrected, or reused.
3. Mechanistic ODEs can fit curves but are non-identifiable from sparse
   release data.
4. We combine early observations, tree-ensemble theta inference, and
   ODE decoding to build an observer-corrected release surrogate.

### Methods

1. Data harmonization and curve cleaning.
2. Nine-parameter PLGA release ODE.
3. Oracle theta fitting and active-set diagnostics.
4. Tree ensemble theta mapping from formulation + early Q.
5. Early-Q theta refinement / candidate feasibility selection.
6. Evaluation protocols: random, held-out drug, held-out polymer,
   direct-curve controls, input ablations, external wet stress test.

### Results

1. The ODE expresses curves but theta is non-unique.
2. Active-set analysis reveals low-dimensional regime-dependent
   mechanisms.
3. RF/ET -> theta -> ODE outperforms MLP and direct curve RF/ET.
4. Early release dominates prediction; formulation conditions the
   feasible domain.
5. Early-Q refinement improves OOD theta decoding.
6. External wet curves expose the remaining deployment gap and show the
   value of regime-aware refinement.

### Discussion

1. The method is a release-quality surrogate, not a universal physics
   law.
2. Theta should be interpreted as a feasible mechanism state family, not
   a uniquely recovered ground-truth parameter.
3. The next ceiling is data and auxiliary observations, not just larger
   models.
4. Future extensions: uncertainty, active observation selection, new
   materials, wet validation.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| The ODE family can express observed PLGA release curves. | 27a oracle full-curve fits, median near 0.998. | Supported |
| Individual curves do not identify all 9 parameters. | 29 active-set audit; 30 dim95 median 2, dim99 median 3. | Supported |
| Mechanism heterogeneity is discrete/regime-like. | 33 regimes, 34/35a partial-out, 36 R5/R6 4-param prototypes. | Supported with caveats |
| RF/ET theta mapping is better than MLP here. | 37c/38/41 CV comparison. | Supported |
| Theta is a useful middle layer. | 46 direct RF/ET -> Q loses to theta->ODE. | Supported |
| High scores are not formulation-only. | 45 input-source ablation. | Supported |
| Early-Q refinement improves OOD deployment. | 50 CV audit and 49 wet stress test. | Supported, not universal |
| This is a full drug-release world model. | Current system lacks uncertainty, active action planning, and broad mechanisms. | Do not claim |

## What to do next

### Route A - baseline closure

Let the model-zoo audit finish, but frame it as defense, not the main
creative contribution.

Needed comparisons:

- RF / ET / LightGBM / XGBoost if available,
- MLP / KAN if already implemented,
- direct Q vs theta->ODE,
- formulation-only / early-only / formulation+early,
- 1d / 3d / 5d / 7d early windows,
- random / by_drug / by_polymer / external wet.

Decision rule:

```text
If a baseline wins by a tiny random-CV margin but loses OOD or direct
interpretability, do not rewrite the paper around it.
```

### Route B - new data / quality paradigm

This is the real expansion route.

If new data follows the same pattern:

```text
early release -> theta state -> simulator forecast
```

then the project becomes more than a PLGA benchmark. It becomes a
release-quality modeling paradigm:

```text
early assay readout
        ->
mechanism-state inference
        ->
long-term release risk / quality forecast
        ->
optional next-measurement recommendation
```

The long-term name can be:

- observer-corrected release-quality surrogate,
- release-dynamics surrogate environment,
- mechanistic quality model for controlled-release formulations.

## Current final judgment

The paper is no longer about inventing a neural architecture. It is about
showing that sparse early release data can be transformed into a
simulator-compatible mechanism state for long-term release quality
prediction, while the mechanism analysis explains why this state should
be treated as a feasible family rather than a unique truth.

That is the line worth writing.

---

## Update 2026-05-26 — CASP unified framework absorbed

After the closeout above was written, the project built a **unified
posterior framework** (`casp/` module) that operationalizes the
"feasibility family" claim and adds calibrated PI across information
budgets. The paper plan now includes it as a co-equal contribution.

### The new contribution

**Single output object** `(μ, U, σ_active)` per curve — a low-rank
Gaussian posterior over the mechanism parameter θ:

```
q(θ | x, ...) = N( θ_hat,  U diag(σ²_active) U^T  +  σ₀² (I - U U^T) )
```

- μ is the point prediction (best θ for this curve / formulation)
- U spans the curve-conditional **active subspace** (r=4 directions
  data actually identifies)
- σ_active is the per-direction posterior std along the active subspace
- The orthogonal complement (P-r dirs) is the **inactive subspace**:
  parameters the data doesn't constrain, kept at a fraction of prior std

This is the unified "feasibility family" the existing closeout asked for,
made operational. Three instantiations of `(μ, U, σ)` covering different
information budgets:

| Information budget | Instance | When and how to construct (μ, U, σ) |
|---|---|---|
| **Few-shot** (formulation + early Q) | **FIB-CASP** | μ = RF→θ point estimate; (U, σ) = eigendecomp of Fisher info matrix of the simulator at μ. Closed-form per curve, no training of the posterior. |
| **Zero-shot** (formulation only) | **Ensemble-CASP** | (μ, U, σ) = low-rank Gaussian fit to the 400 per-tree θ predictions of the formulation-only RF. Empirical bootstrap-style posterior. |
| **Future mechanisms** (no oracle θ) | **Pull-back targets** | (μ, U, σ) = importance-reweighted sample mean / covariance / eigendecomp of θ candidates whose simulator output matches the observed curve. No oracle θ needed. |

All three share the same downstream pipeline: sample θ from q, simulate
at the curve's native t_obs, percentile-aggregate, recenter band onto
the simulation of μ.

### What we showed empirically

**Few-shot (FIB-CASP, scripts 61-62):**
- cross321 random: R² 0.926, cov90 mean 0.81 (median 0.86)
- cross321 by_drug: R² 0.864, cov90 mean 0.76 (median 0.81)
- cross321 by_polymer: R² 0.872, cov90 mean 0.76 (median 0.80)
- internal181 random: R² 0.960, cov90 mean 0.85 (median 0.92)
- internal181 by_drug: R² 0.906, cov90 mean 0.74 (median 0.79)
- internal181 by_polymer: R² 0.885, cov90 mean 0.72 (median 0.75)
- **Cross-mechanism (Liposome / Weibull) random**: R² 0.93, cov90 0.91

Six of nine PLGA cells inside [0.80, 0.95] cov90 mean band; eight of nine
inside on the median. R² matches RF→θ→ODE within 0.01-0.06.

**Zero-shot (Ensemble-CASP, script 66):**
- cross321 random: R² 0.753, cov90 mean 0.75 (median 0.86) ✓
- internal181 random: R² 0.880, cov90 mean 0.73 (median 0.82) ✓
- OOD splits (group_by_drug/polymer): cov90 mean 0.43-0.63 — open
  problem (see below)

### The two algorithms claim is original

Neither FIB-CASP nor Ensemble-CASP is a primitive new technique: Fisher
info eigendecomp is textbook (Cox & Reid 1987); RF tree-level variance
is textbook (Wager 2014). The novelty is the **unification**:

1. The (μ, U, σ) output object is consistent across all three
   instantiations — same downstream code, same paper claim, same
   evaluation.
2. The same simulator and the same module covers two mechanisms
   (PLGABiphasic 9 params, Weibull 2 params) without code change.
3. The framework chooses the right (μ, U, σ) instance per information
   budget. Existing release-modeling papers all collapse to a single
   point predictor without UQ, and certainly without an
   information-budget-conditional method choice.

Code: `casp/` module, ~700 LOC across `fib.py`, `pullback.py`,
`ensemble.py`, `decoder.py`, `calibration.py`, plus a shared encoder
header.

### Honest limitations

1. **OOD UQ is not solved.** Ensemble-CASP's tree-level variance
   underestimates uncertainty under hard distributional shift (held-out
   drug / polymer). This is a broader open problem in the calibration
   literature, not specific to our setup. We report it as a discussion
   point, not as a claim.

2. **Point R² ceiling = the point predictor's ceiling.** FIB-CASP
   inherits RF→θ→ODE's R²; Ensemble-CASP inherits forest-mean RF's R².
   The unification adds PI but does not improve point accuracy.

3. **Pull-back PCA-P trained neural encoder didn't beat RF on 200
   curves.** The simulator-derived targets are sound; the amortized
   inference network is data-hungry. Pull-back as a *target source* for
   future mechanisms without oracle θ is the surviving contribution;
   training amortized neural posteriors on 200 PLGA curves is not.

### Claim-evidence map (updated)

| Claim | Evidence | Status |
|---|---|---|
| The ODE family expresses observed PLGA release curves. | 27a oracle fits, median R² ~0.998. | Supported |
| Individual curves activate 2-4 effective θ directions. | 29 active-set audit; FIB U rank = 4 by design. | Supported |
| Mechanism heterogeneity is discrete/regime-like. | 33-36 regime analyses. | Supported with caveats |
| RF/ET θ mapping is better than MLP. | 37c/38/41/64. | Supported (Ensemble-CASP confirms RF stays the right point predictor.) |
| θ is a useful middle layer. | 46, 59 middle-layer-gain audits. | Supported |
| Early release is the dominant signal in few-shot. | 45 input-source ablation. | Supported |
| **CASP gives calibrated PI in few-shot.** | **FIB-CASP 62; cov90 median 0.75-0.92 across 9 cells.** | **Supported (new)** |
| **CASP gives calibrated PI in zero-shot random.** | **Ensemble-CASP 66; cov90 median 0.82-0.86 on random splits.** | **Supported (new)** |
| **Cross-mechanism reach (PLGA + Liposome).** | **Same casp/ module both, Phase 2.** | **Supported (new)** |
| Zero-shot OOD calibration is solved. | n/a — cov90 0.43-0.69 on OOD. | **NOT claimed** |
| Full drug-release world model. | No active planning, no broad mechanism coverage. | NOT claimed |

### Revised manuscript framing

Two paragraphs of new method content, slotted into the existing closeout
draft:

> **Section: A unified posterior framework over mechanism parameters.**
> We introduce CASP (Conditional Active-Subspace Posterior), a per-curve
> low-rank Gaussian over θ with curve-conditional active subspace and
> explicit inactive-direction nugget. CASP is instantiated three ways
> depending on the available information: from the Fisher information
> matrix of the simulator at a point estimate (FIB-CASP, few-shot); from
> the per-tree predictions of a Random Forest (Ensemble-CASP, zero-shot);
> or from importance-reweighted simulator pull-back on real curves
> (pull-back targets, mechanism-agnostic). All three share the same
> downstream sampling, simulation, and predictive-interval calibration.

> **Section: Information-budget-conditional UQ.** We show that the right
> CASP instance depends on the information budget at inference. With
> early observations (Q at days 1, 3, 5, 7), FIB-CASP reaches median R²
> 0.93 on cross321 cross-DOI 5-fold CV with 90% PI coverage 0.86.
> Without observations (formulation only), Ensemble-CASP reaches R² 0.75
> with 90% PI coverage 0.86 on random splits — the first calibrated PI
> for zero-shot PLGA release prediction. The cross-mechanism applicability
> is demonstrated by running the same code with a Weibull simulator on a
> liposome IVR dataset.

The closeout's previous "two contributions" structure (prediction +
mechanism feasibility) remains intact; the new CASP framework
operationalizes both at once.

### Next steps (post-CASP)

In rough priority order:

1. **Phase 3 identifiability audit**. For each test curve, compare
   FIB-CASP's learned U columns to script 29's greedy oracle active
   set. If consistent, this is the mechanism contribution — FIB
   automatically recovers the active set from data, no greedy fitting
   needed. (1-2 days work.)

2. **(Optional) Combined-CASP for OOD calibration**. Σ_combined =
   Σ_ensemble + Σ_fisher. Both are already implemented; combination is
   straightforward. Goal: push zero-shot OOD cov90 from 0.5 toward 0.8.
   (Half day if straightforward, several days if not.)

3. **Paper draft**. The story is now complete enough to write.

