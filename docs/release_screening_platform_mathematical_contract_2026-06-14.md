# Release Screening Platform Mathematical Contract

Date: 2026-06-14

Status: mathematical contract for process-route facts, audit-only process
metadata, optional state-proxy measurements, observation-budget updates, and
future-window evaluation in the observation-budgeted release screening
platform.

## Purpose

This document extends `release_state_inference_formalism_2026-06-13.md`.
It does not replace the existing formalism:

```text
S_c = (Z_c, e_c, g_c, d_c)
```

Instead, it defines the screening-specific objects that must remain separated:

```text
X_i^base      = base descriptors available without extra characterization
R_i^model     = factual process-route information allowed in the main model
R_i^audit     = expert-authored or derived audit metadata, not a default model input
H_i^available = optional state proxies that could be measured
H_i^obs       = optional state proxies actually measured under the current budget
O_i^(k)       = early release observations available under observation budget k
z_i^(0)       = release-state estimate from base descriptors and route facts
z_i^(H)       = release-state estimate after optional state-proxy measurements
z_i^(k)       = release-state estimate after optional state proxies and early Q
U             = screening utility
```

The purpose is to prevent the platform route from hiding unresolved choices
inside implementation:

```text
which process information is factual input versus authored audit judgment;
which variables are available at design time versus only after extra measurement;
which timepoints are observed context versus truly held-out future targets.
```

## Symbol And Boundary Table

| Object | Example contents | Available stage | Measurement cost | Allowed in main model | Allowed use |
|---|---|---|---|---|---|
| `X_i^base` | drug, material, planned formulation, medium descriptors | design stage | low / already known | yes | prior |
| `R_i^model` | factual unit operations, order, reported conditions, explicit unknown tokens | design or process record stage | low | yes | route prior |
| `R_i^audit` | mechanism hypotheses, curve-symptom hypotheses, confound-risk, support counts | audit stage | authored / derived | no | audit only |
| `H_i^available` | optional state-proxy menu such as particle size, actual loading, porosity | post-process measurement menu | optional / costly | no by itself | measurement design |
| `H_i^obs` | measured optional state proxies actually obtained | post-process, current budget | optional / costly | only when explicitly declared | state update |
| `O_i^(k)` | early release points within budget `k` | release experiment | costly / time-dependent | only under declared budget `k` | early update |
| full future `Q_i(t)` | held-out future curve values | unavailable at inference | oracle | no | target only |

## Core Objects

For curve `i`, define the full observed release curve:

```text
Y_i = {(t_ij, Q_i(t_ij))}_{j=1}^{m_i}
```

where `Y_i` is one release curve, not independent timepoint rows.

### Base descriptors

Base descriptors are:

```text
X_i^base = (D_i, M_i, F_i^plan, E_i)
```

where:

| Symbol | Meaning |
|---|---|
| `D_i` | drug or payload descriptors |
| `M_i` | material or carrier descriptors |
| `F_i^plan` | planned formulation descriptors available without extra post-process characterization |
| `E_i` | medium and environment descriptors |

`X_i^base` must exclude optional state proxies that require extra measurement.
For example:

```text
planned drug-to-polymer ratio may belong to F_i^plan;
actual drug loading, encapsulation efficiency, measured particle size, zeta potential,
porosity, swelling, morphology, crystallinity, residual solvent, or lamellarity do not
belong to X_i^base unless the contract explicitly declares them as already measured
and pays their measurement cost through H_i^obs.
```

### Optional state proxies

The optional state-proxy menu is:

```text
H_i^available = {all optional state proxies that could in principle be measured}
```

Examples include:

```text
particle size
particle-size distribution
actual drug loading
encapsulation efficiency
porosity
swelling
mesh-size proxy
crystallinity
surface-associated drug
residual solvent
morphology
lamellarity
zeta potential
```

The measured subset under the current budget is:

```text
H_i^obs subseteq H_i^available
```

and may be empty:

```text
H_i^obs = empty set
```

### Process information

Raw process information is:

```text
P_i = raw process description or record
```

The model-facing route facts are:

```text
R_i^model = tau_model(P_i)
```

`R_i^model` may contain only factual information obtainable from the process
record or standard normalization of that record, for example:

```text
raw or normalized process text
route family
unit-operation sequence
operation order
reported process conditions
reported additions or removals
post-processing operations
explicit unknown or not-reported tokens
```

The audit-only route channel is:

```text
R_i^audit = tau_audit(P_i, external audit notes)
```

`R_i^audit` includes derived or authored metadata such as:

```text
expected state-effect hypotheses
expected curve-symptom hypotheses
mechanistic interpretation
evidence strength
mapping confidence used for audit
confound-risk annotations
source-leakage risk
route-support counts
alternative explanations
expert-authored mechanism labels
```

`R_i^audit` is not a default input to the main model.

### Curve-derived state

Inside one declared state space `S_c`, the curve-derived state is:

```text
z_Q,i = e_c(Y_i)
```

This remains a curve-derived operational state, not automatically a unique
physical hidden variable.

## Information States

The platform must declare which information state is active.

### Information state 0: design / base prior

```text
I_i^(0) = {X_i^base, R_i^model}
pi_i^(0)(z) = p(z_i | X_i^base, R_i^model)
z_i^(0) = E[z_i | X_i^base, R_i^model]
```

`I_i^(0)` is the design-stage or base prior state. It is allowed to use base
descriptors and factual route information only.

It must not use:

```text
H_i^obs
O_i^(k)
R_i^audit
full future Q_i(t)
```

### Information state H: optional state-proxy update

```text
I_i^(H) = I_i^(0) union H_i^obs
z_i^(H) = u_H(z_i^(0), H_i^obs)
```

This state is valid only when the measurement budget explicitly includes the
observed optional proxies `H_i^obs`.

### Information state k: early-release update

Early release observations under budget `k` are:

```text
O_i^(k) = {(t_ij, Q_i(t_ij))}_{j=1}^k
```

The early-updated information state is:

```text
I_i^(k) = I_i^(H) union O_i^(k)
z_i^(k) = u_Q(z_i^(H), O_i^(k))
```

If no optional state proxy is measured, then:

```text
H_i^obs = empty set
I_i^(H) = I_i^(0)
```

### Pre-observation language boundary

The phrase "pre-observation" is ambiguous and must not be used without a budget
declaration. Use one of the following instead:

```text
pre-additional-measurement: before optional state-proxy measurements H_i^obs
pre-release-observation: before early release observations O_i^(k)
```

Likewise, "pre-experimental ranking" is valid only after the contract declares
whether ranking uses:

```text
I_i^(0) only
or
I_i^(H) with a stated optional-measurement budget
```

## Release-State Interpretation Boundary

`z` can be interpreted at three levels. Claims must not jump levels.

| Level | Object | Allowed claim | Required support |
|---|---|---|---|
| 1 | curve-shape state | captures burst, slope, plateau, tail, timescale | decoder and symptom audit |
| 2 | predictive release-route state | reflects route-conditioned release behavior in an operational state space | route-fact association plus strict split |
| 3 | physical state hypothesis | relates to porosity, mesh size, residual solvent, surface drug | independent state-proxy measurement |

Without Level 3 support, the model may recommend measuring porosity-like or
swelling-like proxies, but it may not claim to infer real porosity or
swelling.

## Prior, Update, And Residual Objects

The decoded prior curve from the base information state is:

```text
Q_i^(0)(t) = g_c(z_i^(0), t)
```

The decoded curve after optional state-proxy measurements is:

```text
Q_i^(H)(t) = g_c(z_i^(H), t)
```

The decoded curve after early-release update is:

```text
Q_i^(k)(t) = g_c(z_i^(k), t)
```

Residuals are defined inside one declared state space only:

```text
r_i^(0) = d_c(z_Q,i, z_i^(0))
r_i^(H) = d_c(z_Q,i, z_i^(H))
r_i^(k) = d_c(z_Q,i, z_i^(k))
```

These residuals must not be subtracted across PCA, Weibull, Hill, ODE, or
other state spaces without an explicit shared metric or decoded curve-domain
comparison.

## Observation Windows And Future-Only Evaluation

For sample `i` under observation budget `k`, define:

```text
T_i,k^obs    = observed early timepoints used inside O_i^(k)
T_i,k^future = held-out future timepoints not observed under budget k
tau_i,k      = max T_i,k^obs, when O_i^(k) is non-empty
```

These sets must satisfy:

```text
T_i,k^obs intersection T_i,k^future = empty set
```

If evaluation focuses on a target window `T_target`, define:

```text
T_i,k^eval = T_target intersection T_i,k^future
```

Observed early points may be used for conditioning, but they must not
contribute to the primary future-prediction score.

### State-space loss

For any information state `I`, define the state loss:

```text
L_z(I) = RMSE_i(d_c(z_Q,i, E[z_i | I]))
```

For example:

```text
L_z(I^(0))
L_z(I^(H))
L_z(I^(k))
```

### Primary curve-domain loss

The primary future-curve loss after budget `k` is:

```text
L_Q^future(I^(k)) =
RMSE over i and t in T_i,k^future
of g_c(z_i^(k), t) versus Q_i(t)
```

If a target window is used, the primary target-future loss is:

```text
L_Q^target-future(I^(k)) =
RMSE over i and t in T_i,k^eval
of g_c(z_i^(k), t) versus Q_i(t)
```

### Observed-window reconstruction metric

Observed-window reconstruction may be reported only as:

```text
L_Q^obs(I^(k)) =
RMSE over i and t in T_i,k^obs
of g_c(z_i^(k), t) versus Q_i(t)
```

This metric is debug-only. It may serve as:

```text
debug metric
assimilation diagnostic
context-reconstruction diagnostic
```

It is not a primary observation-value metric.

## Observation Value Layers

Observation value must be reported at more than one layer.

### Future-curve value

Future-curve value answers whether early observations improve truly unknown
future prediction:

```text
V_Q^future(k) =
L_Q^future(reference information state) - L_Q^future(I^(k))
```

or, for target windows:

```text
V_Q^target-future(k) =
L_Q^target-future(reference information state) - L_Q^target-future(I^(k))
```

### Decision value

Decision value answers whether early observations improve screening quality:

```text
target-window success classification
stop or go decision accuracy
false-stop rate
false-go rate
decision calibration
expected decision loss
```

This value is not interchangeable with full-curve RMSE.

### Uncertainty value

Uncertainty value answers whether early observations contract uncertainty:

```text
posterior uncertainty reduction
prediction interval width reduction
entropy reduction
```

### Observation efficiency

Observation efficiency answers whether the gain is worth the cost:

```text
gain per observed point
gain per unit measurement cost
samples required to reach a fixed decision confidence
```

Even if future RMSE improves only modestly, early observation may still have
decision value or uncertainty value. Those claims must be reported separately.

## Matched-Window Discipline

Any comparison between models or information budgets must preserve:

```text
same held-out samples
same evaluation timepoints
same target window
same censoring rule
same normalization rule
same information-budget declaration
```

The following comparison is prohibited:

```text
static-only model evaluated on unknown full or future points
versus
early-conditioned model evaluated partly on points already given as input
```

## Process Route Boundary

The route tokenizer is split into:

```text
tau_model: P_i -> R_i^model
tau_audit: P_i -> R_i^audit
```

`R_i^model` is the only default process input in:

```text
z_i^(0) = f_theta(X_i^base, R_i^model)
```

An additive diagnostic decomposition may be written as:

```text
z_i^(0) = f_X(X_i^base) + f_R(R_i^model) + f_XR(X_i^base, R_i^model)
```

This decomposition is not assumed causal. It is an audit device for testing
whether route facts add information beyond base descriptors and source labels.

`R_i^audit` may be used for:

```text
stratified evaluation
error analysis
conditional permutation
source-confounding analysis
mechanistic consistency audit
reporting
eligibility gate
claim downgrading
```

It must not be used for:

```text
main-model training
main-model inference
default route-token baseline construction
```

The following outputs are audit artifacts, not default model features:

```text
process_to_state_hypothesis_matrix.csv
process_to_curve_symptom_matrix.csv
```

If future work studies expert-authored priors, it must be reported as a
separate ablation:

```text
route facts only
vs
route facts + expert prior
```

Only the second condition may support a claim about expert-prior value. It must
not be rewritten as proof that route facts alone are sufficient.

## Confound-Risk Channel

The following variables belong to the audit channel by default:

```text
confound_risk
source_leakage_risk
route_source_specificity
single-source route flag
mapping ambiguity risk
support-count warnings
DOI or source concentration
```

These variables may be used only for:

```text
reporting
eligibility gate
subgroup audit
sensitivity analysis
conditional permutation
claim downgrading
```

They are not default predictive features.

If a route appears in only one DOI or source, it may remain as a source-bound
hypothesis, but it cannot support a cross-source route-language claim.

## Shape-Prior Baseline Taxonomy

The contract distinguishes three different shape-family objects.

### 1. Legal pre-observation shape prior

A legal pre-observation shape prior uses training data only, for example:

```text
global Weibull parameter prior
system-conditioned training-only Hill prior
training-only mixture of monotone shape families
```

It must not use any release point from the held-out test sample unless the
baseline is explicitly declared as early-conditioned.

### 2. Legal early-fit shape baseline

A legal early-fit shape baseline uses only the currently observed early points:

```text
phi_hat_i^(k) =
argmin over phi
sum over j <= k
loss(g(t_ij; phi), Q_ij)
```

This baseline must be scored only on:

```text
T_i,k^future
or
T_i,k^eval
```

It must not receive any future point from the same sample.

### 3. Full-curve oracle fit

Fitting Weibull, Hill, or another shape family to the full true test curve is:

```text
oracle ceiling
post-hoc descriptive fit
shape-family adequacy audit
```

It is audit-only and must not be used as:

```text
pre-observation baseline
deployable comparator
primary Phase 3 model
```

## Identifiability Boundary

The mapping from physical state to curve is many-to-one:

```text
H_i^phys -> z_Q,i -> Q_i(t)
```

Different physical states may produce similar release curves. Therefore:

```text
z_Q is identifiable from Q(t) only up to the chosen ReleaseStateSpace.
z_i^(0), z_i^(H), and z_i^(k) are identifiable only as predictive coordinates.
physical hidden variables require independent proxies.
```

Sufficient conditions for useful operational state estimates include:

```text
1. e_c and g_c reconstruct Q(t) with acceptable error.
2. z_i^(0) improves over a global or legal shape prior under strict splits.
3. route-fact contribution survives source, DOI, drug, and process audits.
4. optional proxies or early O_i^(k) reduce residual, uncertainty, or decision risk.
5. any physical interpretation is supported by proxy measurements.
```

Failure modes include:

```text
shape-prior equivalence:
  route-fact model does not beat legal training-only shape priors.

source memorization:
  route-fact model helps random split but fails leave-source-out.

non-identifiable state:
  multiple route states decode to indistinguishable curves and no proxy anchors exist.

descriptor starvation:
  common base descriptors are too sparse to support z_i^(0).
```

## Screening Utility

A screening task must define target and failure events.

Let:

```text
A_i = desired target event
B_i = failure event
C(I) = measurement cost under information state I
```

Examples:

```text
A_i: Q_i(24h) in [0.40, 0.70]
B_i: Q_i(1h) > 0.20
B_i: Q_i(24h) < 0.50
```

A simple expected utility under information state `I_i^(k)` is:

```text
U_i(I_i^(k)) =
  w_success * P(A_i | I_i^(k))
- w_failure * P(B_i | I_i^(k))
- C(I_i^(k))
```

with cost decomposed as needed, for example:

```text
C(I_i^(k)) =
cost_of_optional_state_proxies(H_i^obs)
+ timepoint_cost(O_i^(k))
```

The decision rule is:

```text
go_i(I) if P(A_i | I) >= alpha_go
         and P(B_i | I) <= beta_fail

stop_i(I) if P(B_i | I) >= alpha_stop

measure_more_i(I) if E[U_i(I_next) - U_i(I)] > incremental_cost(I_next)
```

Default thresholds for preregistration are:

```text
alpha_go = 0.80
beta_fail = 0.20
alpha_stop = 0.80
```

These are defaults, not universal biological truths. A wet-lab campaign must
declare its target window, risk weights, and information budget before outcome
evaluation.

## Information Value Of Measurements

For a candidate additional measurement `m`, define expected information value:

```text
EIV(m | I) = E[U_i(I with m observed)] - U_i(I) - cost(m)
```

Measurement `m` may be:

```text
an additional release timepoint
an optional state proxy such as swelling, SEM porosity, DSC Tg, GPC, residual solvent
a process-adjacent measurement such as size distribution or rheology
```

Measurement recommendations are valid only as recommendations:

```text
high EIV != proven causal variable
high EIV != discovered physical hidden state
```

## Required Benchmarks

The platform must report at least four objective families:

| Objective | Metric family | Interpretation |
|---|---|---|
| State inference | `L_z`, residual norm, gap closure | whether `z_i^(0)`, `z_i^(H)`, or `z_i^(k)` approaches `z_Q,i` |
| Future-curve prediction | `L_Q^future`, `L_Q^target-future`, MAE, qmax or timing errors | whether decoded future curves improve |
| Decision utility | hit rate, enrichment, regret, expected utility, false-stop and false-go | whether screening decisions improve |
| Uncertainty or efficiency | interval width, entropy, gain per point, gain per cost | whether observations are worth the budget |

Minimal comparisons should be expressible through declared information states,
including:

```text
global prior
legal pre-observation shape prior
static only: I_i^(0) without route facts if desired
static + route: I_i^(0)
static + optional state proxy: I_i^(H) without early Q
early only: O_i^(k) with no base descriptor advantage beyond the declared baseline
static + early
static + route + early
static + route + optional state proxy
static + route + optional state proxy + early
legal early-fit shape baseline
```

Strict split comparisons include:

```text
leave-source-out
leave-drug-out
leave-process-out
leave-system-out when possible
```

## Neural Entry Condition

A neural model is justified only if the task has been narrowed to:

```text
learn f_theta(I_i^(k)) -> z_i^(k)
```

and the following are already true:

```text
descriptor ontology has usable coverage;
route facts R_i^model are separated from audit metadata R_i^audit;
base descriptors X_i^base are separated from optional proxies H_i^obs;
future-only evaluation is declared;
legal shape priors are separated from oracle fits;
classical route-fact baselines show a gap worth closing.
```

The neural model must be evaluated against:

```text
legal shape priors
ExtraTrees or ridge on declared information states
legal early-fit shape baseline
route-fact classical baseline
```

If neural performance improves only random splits, or only after audit metadata
enters the input, it is not cross-system learning. It is a negative control for
source memorization or leakage.

## Mathematical Gate For 123

Experiment 123 should not train a neural model. It should not train any new
main-model route. It should produce the formal inputs that make later modeling
legitimate:

```text
canonical descriptor schema
coverage tensor: dataset x descriptor_layer x field
route token taxonomy
state-effect hypothesis matrix
curve-symptom matrix
utility specification table
data reality checkpoint
```

The following gates must be true before Experiment 123 starts:

```text
1. R_i^model and R_i^audit are separated.
2. X_i^base and optional state proxies are separated.
3. Every descriptor has an availability stage.
4. Early-observation value uses future-only or target-future-only scoring.
5. Legal shape priors, legal early-fit baselines, and full-curve oracle fits are separated.
6. Confound-risk variables belong to the audit channel only.
7. This contract and the design contract use consistent terminology.
8. Experiment 123 itself does not train any new model.
```

Decision rows should include:

```text
common_descriptor_coverage_sufficient?
process_route_tokenization_feasible?
route_facts_not_source_only?
base_vs_optional_measurement_boundary_declared?
future_only_evaluation_declared?
shape_baseline_taxonomy_declared?
utility_function_declared?
neural_entry_allowed?
```

The default answer to `neural_entry_allowed?` should be `false` until the
groundwork tables pass.

## Prohibited Interpretations

Do not write:

```text
the model discovered a named physical microstructure
the model inferred an unmeasured physical state
process route tokens prove mechanism
audit annotations are route facts
screening utility is equivalent to full-curve RMSE
oracle shape fits are deployable pre-observation baselines
neural transfer proves large-scale universal learning
```

Allowed wording:

```text
the model learns an operational or predictive release-state coordinate
route facts provide a process-conditioned prior
optional state proxies update the release-state estimate when measured
early observations update the release-state estimate
state-proxy recommendations are testable measurement hypotheses
decision utility is evaluated under declared target windows, costs, and information budgets
```
