# Information-Budget Reframing Memo - 2026-06-10

Purpose:
- preserve the 2026-06-10 reframing after adversarial Claude review;
- stop measuring the project only by "RMSE beats black boxes";
- define the next evidence route around information bottlenecks,
  observation budgets, uncertainty contraction, and fair black-box controls.

## 0. Bottom Line

The project should not be framed as:

```text
our mechanistic model universally beats direct black-box models on RMSE
```

That target is both too weak scientifically and too hard statistically after
early release points are observed. The stronger conclusion is:

```text
PLGA formulation descriptors alone do not identify the release curve.
Sparse early release observations expose the missing coarse-grained kinetic
state.
Conditioned on those early observations, point prediction becomes close to an
information-saturated smooth extrapolation problem.
```

The next project target is therefore:

```text
quantify how much information each early wet-lab observation adds,
how fast future uncertainty contracts,
when experiments can stop,
and whether mechanism-state posteriors beat shape-constrained black boxes on
calibrated uncertainty and extrapolation.
```

## 1. The Core Scientific Reframing

The user summarized the empirical situation:

```text
Without early PLGA release observations, complete prediction is basically not
possible.
With early observations, our route and black-box routes are close.
Generic world models need large data and collapse here.
```

These are not failures. They are information-structure results.

### 1.1 Formulation-only prediction is under-identified

The formulation table misses variables that strongly control PLGA release:

- microstructure and porosity;
- molecular-weight distribution;
- autocatalytic degradation state;
- phase separation and morphology;
- drug-polymer interactions;
- batch and processing effects.

So a pure `formulation -> curve` map is under-determined. If formulation-only
prediction fails, that is not automatically a model failure; it may be the
correct identifiability conclusion.

### 1.2 Early release is a coarse-grained latent-state observation

Early Q(t) is not just another feature. It indirectly observes the hidden
kinetic state that the formulation descriptors did not encode.

This explains why early-release features dominate the audits:

- `outputs/45_input_source_ablation/input_source_summary.csv`
- `outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv`
- `outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv`

### 1.3 RMSE saturates after early Q is known

Once early release points are available, the remaining task is often a
low-dimensional, monotone, saturating extrapolation problem. Correctly tuned
black boxes, functional baselines, and mechanism decoders can therefore land
close in point RMSE.

This means a small RMSE gap after early-Q conditioning is not the right main
scientific object.

## 2. What Survives From The Current Project

Current closeout object:

```text
formulation + sparse early release
    -> feasible mechanism state / theta family
    -> simulator decoder
    -> full-trajectory prediction
    (+ optional posterior / UQ layer)
```

This should now be interpreted as an information-bottleneck route, not as a
guaranteed point-RMSE champion.

Surviving facts:

- early release is the dominant signal;
- formulation-only prediction is limited by missing state variables;
- `theta` / mechanism-state families are useful if they explain uncertainty
  contraction and experiment design;
- CASP / posterior-family work remains useful only when judged with sharpness,
  CRPS / interval score, and direct black-box conformal baselines;
- chitosan is a prospective new-mechanism demonstration, not proof that a
  trained PLGA observer transfers unchanged.

## 3. What Has Been Killed Or Demoted

### Active Observer as headline point predictor

Demoted.

The corrected 4-way benchmark:

```text
DirectQ-fixed       RMSE 0.173, 4 fixed observations
DirectQ-adaptive    RMSE 0.167, 2 adaptive observations
Active-fixed        RMSE 0.200, 4 fixed observations
Active-adaptive     RMSE 0.169, 2 adaptive observations
```

Adaptive selection survives. Particle-filter inference does not currently beat
matched Direct-Q as a point predictor.

### RSSM / generic world model

Killed as headline.

Generic learned latent dynamics need more data than this project has. The RSSM
collapse is a useful negative result, not a method victory.

### Regime explains benefit

Killed as main claim.

Regime structure may remain supplementary, but it does not explain Active
Observer benefit under the corrected tests.

### Foundation-model / complete cross-mechanism language

Not allowed yet.

The project does not have enough jointly trained multi-mechanism data to use
that language.

## 4. The Main Statistical Weakness To Fix

The existing "best-route" evidence is exploratory.

`scripts/59_middle_layer_gain_audit.py` uses `_best_by_cell()`:

```python
idx = df.groupby(group_cols, dropna=False)["median"].idxmax()
```

So the current `best_route` mean gain (`0.058`), median gain (`0.054`), and
fraction positive (`1.0`) should be treated as an oracle / hindsight selection
mouth. They are useful for discovery, but not paper-proof evidence.

Paper-facing evidence needs:

- one frozen theta / decoder route chosen before test evaluation;
- the same early timepoints for every method;
- the same future grid;
- the same split files;
- the same HPO budget class;
- paired bootstrap CIs;
- multiple-comparison correction when comparing many cells.

## 5. New Winning Definition

Do not collapse the project into one score. Use five battlefields.

### A. Point prediction fairness lock

Question:

```text
Under identical split, observation budget, future grid, and HPO budget, does
the frozen mechanism-state route beat Direct-Q / RF->Q / fPCA-Ridge?
```

Honest possible outcome:

```text
after early-Q conditioning, methods tie within uncertainty
```

That is not a failure if the information-budget result is strong.

### B. Observation-budget / information-gain curve

Question:

```text
How much future uncertainty is removed by each additional early wet-lab
timepoint?
```

Required budgets:

- formulation only;
- 1 early point;
- 2 early points;
- 4 early points;
- dense early window if available.

Primary figure:

```text
observation budget -> future uncertainty contraction
```

Metrics:

- future RMSE;
- cov90 / cov50;
- interval width;
- CRPS or interval score;
- posterior / theta-family uncertainty;
- information gain per additional timepoint.

### C. Stopping rule / experiment design

Question:

```text
When can a PLGA release experiment stop because the future trajectory is
already identified enough?
```

This is where a mechanism-state posterior can be more useful than a plain
point predictor. A black box may also do adaptive selection, but a mechanism
posterior can tie the decision to latent-state uncertainty.

### D. Mechanism vs shape-prior ablation

Question:

```text
Is the gain caused by physical mechanism state, or only by a monotone,
saturating shape prior?
```

Required controls:

- Direct-Q with monotone / saturation constraints;
- fPCA / spline / isotonic / Weibull / bi-exponential shape baselines;
- theta-route under the same observation and HPO budget;
- paired CIs for the remaining delta.

If shape-prior baselines erase the theta-route gain, the project should be
reframed as shape-constrained sparse-curve forecasting, not mechanism-state
inference.

### E. Prospective / new-mechanism demonstration

Question:

```text
Can the workflow survive outside retrospective PLGA tables?
```

Current status:

- chitosan predictions are locked but not revealed;
- primary endpoint is aggregate cov90 >= 0.83;
- liposome is small-N support only;
- no "cross-mechanism complete" claim.

## 6. Next Sprint

### Step 1 - Leave-one-system-out x early-observation budget scan

This is the least speculative next experiment.

For each held-out drug / polymer / system group, scan:

```text
formulation-only -> 1pt -> 2pt -> 4pt -> dense early window
```

Report:

- future RMSE;
- interval width;
- cov90 / cov50;
- CRPS or interval score;
- posterior entropy / theta-family uncertainty;
- information gain per extra timepoint.

### Step 2 - Fair black-box leaderboard

Build one locked report joining:

- Direct-Q fixed/adaptive;
- RF->theta / theta->decoder;
- fPCA-Ridge;
- GlobalMean;
- shape-prior baselines;
- FIB-CASP point prediction and intervals;
- optional Claude / frontier LLM stress test.

All rows must share:

- split file;
- early timepoint set;
- future evaluation grid;
- bootstrap protocol;
- HPO budget class;
- lock metadata.

### Step 3 - Leakage / causality harness

Fail closed if:

- methods use different early timepoint sets;
- normalization uses final plateau or full-curve information;
- scalers / fPCA / preprocessors are not train-fold-only;
- adaptive selectors see future observations;
- HPO budgets are unequal without being labeled.

### Step 4 - Shape-prior ablation

Run this before writing a mechanism-heavy title.

If the shape-prior control wins or ties, change the story. Do not protect a
mechanism claim that the data do not support.

### Step 5 - Use Claude as adversary, not as authority

Claude / frontier models are useful for:

- finding comparator unfairness;
- attacking split and observation-budget logic;
- proposing leakage tests;
- acting as a frozen in-context black-box stress test if exact model route and
  fallback behavior are logged.

They are not scientific substitutes for paired CIs, calibration metrics, or
prospective wet-lab validation.

## 7. Reporting Spine

### One sentence

PLGA sparse-release prediction is information-limited: formulation descriptors
alone do not identify release trajectories, early release acts as a
coarse-grained kinetic-state observation, and the next contribution is to
quantify how observation budget contracts future uncertainty.

### Three minutes

1. We tested whether sparse early release can identify enough kinetic state to
   forecast future release.
2. Formulation-only prediction is under-identified because key PLGA
   microstructure variables are missing.
3. Early Q(t) is the dominant signal because it observes the missing
   coarse-grained kinetic state.
4. After early Q is given, pure point-RMSE comparisons saturate; Direct-Q,
   fPCA, and mechanism decoders can be close.
5. We killed overclaimed routes: Active Observer as point-prediction winner,
   RSSM / world model as headline, and regime-benefit explanation.
6. The remaining scientific object is information budget: how many
   measurements are needed, when to stop, and whether mechanism-state
   uncertainty beats shape-constrained black boxes.

## 8. Practical Rule For Future Agents

Do not optimize the project narrative around "RMSE beats black boxes" unless a
new locked leaderboard proves it under equal budgets.

Default to this question instead:

```text
What information does each new observation add, and how does it contract the
future release state family?
```

