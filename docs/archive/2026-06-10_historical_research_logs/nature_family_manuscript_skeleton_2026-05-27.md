# Manuscript skeleton, 2026-05-27

Working target: Nature Methods / Nature Communications presubmission.

This is a claim-locked working draft. It should not be treated as final prose.
Every result sentence below is tied to a gate output or figure source. Do not
upgrade any claim without adding new evidence.

## Title candidates

Preferred:

> Sparse release observations reveal mechanistic states for uncertainty-aware
> drug-release forecasting

Alternative:

> Mechanism-constrained observers infer drug-release dynamics from sparse
> early measurements

Avoid in title and abstract:

- world model
- foundation model
- universal drug-release AI
- active learning as a point-prediction champion

## One-sentence claim

Sparse early release observations constrain drug-release trajectories to
low-dimensional, regime-dependent feasible kinetic states; decoding these
states through a mechanism simulator yields statistically positive gains over
direct curve regression and provides useful, though not uniformly nominal,
predictive uncertainty.

## Abstract draft

Long-acting drug formulations are commonly optimized through dense release
experiments, yet formulation descriptors alone often provide too little
information to predict full release trajectories under distribution shift. We
developed a mechanism-constrained release dynamics observer that converts sparse
early release measurements into a posterior over kinetic states and decodes the
resulting state family through release simulators. Across PLGA release datasets,
the inferred kinetic middle layer improved median curve prediction over direct
Q(t) regression in all RF/ET formulation-plus-early cells and in five of six
LightGBM cells, with the cross321 group-by-polymer LightGBM split remaining
inconclusive. Simple two-point classical extrapolators remained far below the
sparse-forecasting performance threshold. Active-set and regime-conditioned
fitting analyses showed that dominant release regimes are well represented by
four-parameter feasible kinetic states (R5 median R2 = 0.9904; R6 median R2 =
0.9899), supporting a low-dimensional mechanism-state interpretation rather
than a unique high-dimensional parameter estimate. Raw FIB-CASP predictive
intervals under-covered in several cells, but proper split-conformal
recalibration restored near-nominal random-split cov90 for cross321,
internal181, and liposome (0.910, 0.905, and 0.912 after local recalibration)
with explicit interval-width costs. PLGA OOD coverage was substantially
repaired after recalibration, whereas liposome OOD remained weak in both point
accuracy and coverage. Active observation maintained nominal interval coverage
and was non-inferior to fixed conformal schedules, but non-inferiority to
directQ fixed-time baselines was not demonstrated at the present sample size.
These results support sparse-observation, mechanism-constrained release
forecasting as a practical route toward uncertainty-aware experimental
decision-making; prospective chitosan validation remains the decisive wet-lab
gate for a stronger biological claim.

## Results architecture

### Result 1: Sparse early observations are informative, but classical sparse extrapolation is insufficient

Claim:

Sparse early release points carry the dominant predictive signal, but simple
curve-only extrapolators do not solve full-trajectory forecasting from one or
two early measurements.

Evidence:

- `scripts/71_nature_gate_diagnostics.py`
- `outputs/71_nature_gate_diagnostics/g2_classical_sparse_summary.csv`
- Fig. 3c from `scripts/74_make_green_claim_figures.py`

Draft result text:

We first asked whether the sparse-observation problem could be solved by simple
curve extrapolation alone. Classical two-point baselines, including linear,
Higuchi, Korsmeyer-Peppas, Weibull, persistence, and Gaussian-process RBF
extrapolators, were fit only to early observations and evaluated on future
points. The best two-point future-only median R2 was 0.512, below the
predefined 0.85 red line for a practically sufficient sparse forecaster. This
negative control indicates that sparse early measurements are not, by
themselves, trivially convertible into accurate future trajectories without a
stronger kinetic-state representation.

Boundary:

This is not a claim that no Bayesian time-series method could improve the
baseline. It is a claim that common low-data curve extrapolators do not explain
the main results.

### Result 2: Release curves collapse onto low-dimensional, regime-dependent feasible kinetic states

Claim:

PLGA release trajectories identify low-dimensional feasible families of kinetic
parameters, not unique nine-dimensional mechanisms.

Evidence:

- `scripts/33_active_set_regimes.py`
- `scripts/36_r6_regime_prototype.py`
- `outputs/36_r5_regime_prototype/summary.txt`
- `outputs/36_r6_regime_prototype/summary.txt`
- Fig. 2 from `scripts/74_make_green_claim_figures.py`

Draft result text:

Active-set analysis of fitted PLGA curves revealed recurring parameter
activation patterns rather than a diffuse use of all nine kinetic coordinates.
The dominant R5 and R6 regimes were then tested by a regime-conditioned fitting
experiment in which only four modal parameters were free and the remaining
parameters were fixed to a leave-one-out regime median. R5 was captured by a
four-parameter state with median R2 = 0.9904, close to the nine-parameter oracle
(0.9966) and the per-curve greedy four-parameter upper bound (0.9946). R6
showed similar near-oracle behavior (regime-conditioned median R2 = 0.9899 vs.
oracle 0.9980), although it remained a partial pass relative to the per-curve
greedy four-parameter fit. These results support a soft regime interpretation:
release curves constrain a low-dimensional feasible kinetic state within a
regime, rather than revealing a single fully identifiable parameter vector.
The R5 tail also showed larger within-regime heterogeneity than the median
suggests, so the regime-conditioned state should be interpreted as a dominant
feasible representation rather than a uniformly sufficient parameterization for
every curve.

Boundary:

Do not write this as a hard taxonomy of immutable release classes. R6 is a
partial pass, and within-regime heterogeneity remains visible in the tails.

### Result 3: The kinetic middle layer improves prediction over direct Q(t) regression

Claim:

The theta/ODE middle layer contributes predictive structure beyond simply
training a regressor from descriptors and early Q values directly to Q(t).

Evidence:

- `scripts/71_nature_gate_diagnostics.py`
- `outputs/71_nature_gate_diagnostics/g1_middle_layer_bootstrap.csv`
- `outputs/71_nature_gate_diagnostics/g1_lgbm_bootstrap.csv`
- Fig. 3a-b from `scripts/74_make_green_claim_figures.py`

Draft result text:

We next tested whether the mechanism layer was merely a decorative
reparameterization. For matched dataset, split, and input-mode cells, we
compared direct Q(t) regression against a two-step kinetic route in which the
model first predicted kinetic parameters and then decoded them through the PLGA
simulator. In formulation-plus-early cells, RF/ET best-route comparisons showed
positive paired-bootstrap middle-layer gains in all six matched cells. The same
test with LightGBM passed in five of six cells, with the cross321
group-by-polymer split remaining inconclusive. Thus, the kinetic middle layer
does not merely rename the target; it contributes a reproducible, modest
predictive gain under matched evaluation. Its value is therefore not only the
size of the point-metric gain, but the creation of a shared, simulator-decodable
state space that supports the regime, uncertainty, active-observation, and
prospective-validation analyses below.

Boundary:

The gain is statistically positive but not enormous. The manuscript should
emphasize structural robustness and interpretability, not a large point-R2
breakthrough.

### Result 4: Conformal recalibration repairs raw FIB-CASP under-coverage with explicit width cost

Claim:

Raw FIB-CASP intervals are informative but too narrow; proper split-conformal
recalibration restores near-nominal pointwise random-split coverage and
substantially repairs PLGA OOD pointwise coverage, while liposome OOD remains
unsolved.

Evidence:

- `scripts/72_casp_active_gate_diagnostics.py`
- `outputs/72_casp_active_gate_diagnostics/casp_bootstrap_gate.csv`
- `outputs/62_fib_casp_benchmark/aggregate_table.csv`
- `scripts/76_casp_conformal_recalibration.py`
- `outputs/76_casp_conformal_recalibration/summary.csv`

Draft result text:

To quantify uncertainty around the mechanism-state prediction, we evaluated
FIB-CASP predictive intervals across PLGA and liposome cells. The raw
random-split intervals were useful but under-covered in PLGA (cov90 = 0.805 for
cross321 and 0.854 for internal181; liposome = 0.913). We therefore applied a
proper split-conformal recalibration: within each outer split, the training fold
was split again into model-fit and calibration subsets, and the learned interval
scale was applied only to the untouched outer test fold. Local conformal
recalibration restored near-nominal pointwise random-split coverage for
cross321, internal181, and liposome (cov90 = 0.910, 0.905, and 0.912) at median
interval-width inflation factors of 1.89, 1.33, and 1.47. PLGA OOD pointwise
coverage was also substantially repaired after recalibration (local cov90 =
0.882-0.909 across drug/polymer group splits), although at larger width cost. In
contrast, liposome OOD remained weak (local cov90 = 0.795; median R2 = 0.540),
defining a clear boundary for cross-mechanism generalization.

Boundary:

Do not write that raw FIB-CASP is a calibrated posterior. The defensible claim
is that split-conformal recalibration repairs empirical coverage with measurable
width cost, and that cross-mechanism OOD calibration remains unsolved.

### Result 5: Active observation is an experimental-decision module, not a point-prediction champion

Claim:

The active observer preserves nominal coverage and is non-inferior to fixed
conformal schedules, but it should not be presented as the best point predictor.

Evidence:

- `scripts/72_casp_active_gate_diagnostics.py`
- `outputs/72_casp_active_gate_diagnostics/active_pairwise_gate.csv`
- `outputs_active_observer_v3/metrics_summary.csv`

Draft result text:

Finally, we evaluated whether sequential observation can support experimental
decision-making. At n=38 test curves, the active two-point observer maintained
above-nominal cov90 coverage and was non-inferior to fixed conformal schedules
under a 10% relative RMSE margin. It was also clearly superior to the zero-early
prior. However, active-vs-directQ non-inferiority was not demonstrated at the
current sample size, and directQ fixed-time baselines remained strong point
predictors. We therefore position the active observer as an uncertainty-aware
observation scheduling module rather than a point-prediction champion.

Boundary:

Do not use "active learning" as the main novelty unless a future experiment
shows robust superiority over strong fixed schedules and directQ baselines.

### Result 6: Decoder extensibility and prospective validation

Claim:

The framework is extensible to other release decoders, but broad
cross-mechanism generalization and prospective biological validation remain
separate gates.

Evidence:

- Liposome: `outputs/72_casp_active_gate_diagnostics/casp_bootstrap_gate.csv`
- Chitosan locked predictions: `outputs/67_chitosan_prospective/`
- Chitosan reveal evaluator: `scripts/75_chitosan_prospective_eval.py`

Draft result text:

As a first non-PLGA decoder test, the same uncertainty workflow was applied to a
liposome IVR dataset using a Weibull simulator. Random-split liposome results
were strong (median R2 = 0.932; raw cov90 = 0.913; conformal local cov90 =
0.912), but OOD liposome splits collapsed to median R2 = 0.540 and local cov90
= 0.795. The dataset is also small and the two-parameter Weibull decoder is much
simpler than the PLGA ODE. We therefore treat liposome as evidence for decoder
extensibility in random splits and as a negative boundary for broad
cross-mechanism OOD claims, not as foundation-model behavior. In parallel,
chitosan predictions have been locked before wet-curve comparison, and a reveal
script predefines the curve-level and coverage gates. Those experiments will
determine whether the present computational story can be upgraded to a
prospective wet-lab validation claim.

Boundary:

No chitosan result should be written until observed release curves have been
compared through `scripts/75_chitosan_prospective_eval.py`.

## Figure legends draft

### Figure 2. Release curves identify regime-dependent low-dimensional kinetic states

**a,** Occupancy of active-set regimes across internal181 and cross321 PLGA
curves. **b,** Frequency with which each kinetic parameter appears in the
k=4 active set within each regime. **c,** Curve R2 distributions for R5 and R6
under nine-parameter oracle fits, per-curve greedy four-parameter fits, and
regime-conditioned four-parameter fits. The dotted line marks R2 = 0.95.
**d,** Per-curve loss in R2 for regime-conditioned four-parameter fits relative
to the nine-parameter oracle. Horizontal bars mark medians; vertical bars mark
interquartile ranges.

Main message:

Dominant regimes retain near-oracle fit quality with four active kinetic
coordinates, supporting a low-dimensional feasible-state interpretation.

### Figure 3. The kinetic middle layer improves prediction and classical sparse extrapolation is insufficient

**a,** Paired-bootstrap median R2 gain for the RF/ET kinetic middle-layer route
over direct Q(t) regression in formulation-plus-early cells. **b,** Same
analysis using LightGBM. **c,** Best classical two-point curve-only
extrapolator in each dataset/window, evaluated on future observations. Error
bars show bootstrap 95% confidence intervals for the median; the dashed line
marks the predefined 0.85 future-R2 threshold.

Main message:

The mechanism layer adds modest but reproducible predictive structure, while
simple two-point extrapolators do not explain the sparse-forecasting result.
Panels a-b and panel c answer distinct control questions: a-b test whether the
mechanism layer adds value over direct Q(t) regression, whereas c tests whether
curve-only sparse extrapolation is already sufficient.

## Methods skeleton

### Data sources

Describe cross321, internal181, liposome IVR, and chitosan prospective
experiment separately. Report curve counts and split schemes from the gate
outputs. Explicitly state which datasets are real release curves and which
predictions are locked prospective outputs.

### Mechanism simulators

Describe PLGABiphasic as a mechanism-constrained release simulator with
hydration, hydrolysis/autocatalysis, diffusion, erosion, burst, and capacity
terms. Describe Weibull and chitosan Ritger-Peppas decoders as mechanism-specific
decoders, not as evidence for a universal simulator.

### Middle-layer gain

Define MLG:

> MLG = median R2(theta -> simulator -> Q) - median R2(direct Q regression)

State that comparisons are matched by dataset, input mode, and split scheme and
that paired bootstrap resampling is performed over curves.

### Regime-conditioned fitting

Define active sets, regime discovery, and the R5/R6 regime-conditioned
four-parameter prototype. Include leave-one-out median fixing of non-modal
parameters to avoid leakage.

### FIB-CASP uncertainty

Describe FIB posterior construction, predictive sampling, interval generation,
raw coverage metrics, and split-conformal recalibration. Explicitly note that
cov90 is empirical pointwise coverage per observed time point, not a
simultaneous whole-curve band; raw FIB intervals under-covered in several PLGA
cells, and recalibration restores pointwise coverage by widening intervals.

### Active observer

Describe particle generation, utility-based observation selection, posterior
reweighting, conformal interval calibration, and paired comparisons against
fixed schedules and directQ baselines.

### Chitosan reveal

Describe locked prediction provenance and the reveal rule. State the integrity
warning: pass thresholds were documented in the 2026-05-27 readiness plan but
were not present in the original lock metadata unless future lock files are
updated.

## Discussion skeleton

1. Sparse release forecasting is not merely a data-regression problem; the
   useful object is a feasible kinetic-state family.
2. Non-identifiability is not only a limitation. It can be reframed as
   structured uncertainty over physically decodable states, although the
   practical value of this framing depends on calibration quality that is
   currently incomplete under OOD splits.
3. The current framework is best viewed as a mechanism-constrained observer,
   not a world model in the learned-dynamics/planning sense.
4. OOD calibration remains the central limitation: PLGA OOD can be repaired
   with wider conformal intervals, but liposome OOD remains weak.
5. Chitosan prospective validation is the decisive next experimental gate.

## Claim language guardrails

Allowed:

- mechanism-constrained observer
- sparse-observation release forecasting
- feasible kinetic state
- uncertainty-aware forecast
- decoder extensibility
- prospective locked prediction

Avoid:

- foundation model
- world model in title/abstract
- universal release predictor
- solved OOD calibration
- active learning as strongest point predictor
- fully identifiable kinetic parameters

## Immediate next writing tasks

1. Convert Fig. 2 and Fig. 3 into polished figure-caption pairs.
2. Draft Introduction using the gap: sparse formulation descriptors cannot
   identify release dynamics without early curve observations.
3. Draft Methods subsections for middle-layer gain and regime-conditioned fits.
4. Wait for chitosan observed curves before writing any prospective result.
