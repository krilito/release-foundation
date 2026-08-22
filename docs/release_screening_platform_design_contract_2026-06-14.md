# Release Screening Platform Design Contract

Date: 2026-06-14

Status: design contract for the observation-budgeted release screening platform
route, with explicit boundaries between model-facing route facts, audit-only
process metadata, optional state-proxy measurements, and future-only
evaluation.

## Purpose

This document defines the design route for a drug-release screening platform.
It replaces the loose question:

```text
Can a model predict drug release across systems?
```

with the stricter platform question:

```text
Can base descriptors, factual process-route tokens, optional state proxies,
and early release observations support experiment-aware release screening?
```

The platform is not a new black-box predictor. It is a staged inference and
decision system:

```text
base descriptor language + route facts
-> operational release-state prior
-> optional state-proxy update
-> early-observation update
-> risk / utility decision
```

## Scope

The platform is allowed to support:

- design-stage candidate ranking under a declared information budget
- future-release forecasting after early observations
- stop / go / measure-more decisions
- recommendation of missing material, process, or state-proxy measurements
- wet-lab validation of decision logic

The platform is not allowed to claim:

- foundation-model behavior
- discovered physical hidden variables
- causal process effects without experimental validation
- cross-system zero-shot prediction from weak static descriptors
- neural architecture novelty before phase gates pass
- route-token value when the signal actually comes from audit metadata

## Shared Contract Vocabulary

This design contract uses the same symbols as the mathematical contract:

```text
X_i^base      = base descriptors available without extra characterization
R_i^model     = factual process-route information allowed in the main model
R_i^audit     = audit-only process metadata, hypotheses, and confound labels
H_i^available = optional state proxies that could be measured
H_i^obs       = optional state proxies actually measured
O_i^(k)       = early release observations under budget k
z_i^(0)       = release-state estimate from X_i^base and R_i^model
z_i^(H)       = release-state estimate after H_i^obs
z_i^(k)       = release-state estimate after H_i^obs and O_i^(k)
T_i,k^obs     = observed early timepoints
T_i,k^future  = held-out future timepoints
T_i,k^eval    = held-out future timepoints inside a declared target window
```

These symbols must keep the same meaning across both contracts.

## Platform Flow

```mermaid
flowchart TB
    accTitle: Release Screening Platform
    accDescr: The platform separates base descriptors, route facts, audit-only metadata, optional state-proxy measurements, early release observations, and future-window screening decisions.

    candidates([Candidate formulations])
    base_descriptors[X_i^base]
    route_facts[R_i^model]
    audit_channel[R_i^audit audit only]
    prior_state[z_i^(0)]
    optional_proxy{H_i^obs measured?}
    proxy_update[z_i^(H)]
    early_obs{O_i^(k) available?}
    release_update[z_i^(k)]
    utility[Risk and utility score]
    decision{Screening decision}
    collect_more[Measure more]
    go_forward[Continue candidate]
    stop_candidate[Stop candidate]
    wet_lab[Wet-lab validation]

    candidates --> base_descriptors
    candidates --> route_facts
    candidates --> audit_channel
    base_descriptors --> prior_state
    route_facts --> prior_state
    prior_state --> optional_proxy
    optional_proxy -->|No| early_obs
    optional_proxy -->|Yes| proxy_update
    proxy_update --> early_obs
    early_obs -->|No| utility
    early_obs -->|Yes| release_update
    release_update --> utility
    utility --> decision
    decision -->|Uncertain| collect_more
    decision -->|Promising| go_forward
    decision -->|High risk| stop_candidate
    go_forward --> wet_lab
    stop_candidate --> wet_lab
```

`R_i^audit` is displayed only to show that the audit channel exists. It is not
a default input to `z_i^(0)`, `z_i^(H)`, or `z_i^(k)`.

## Information-Budget Boundary

The platform must separate three information states:

```text
I_i^(0) = {X_i^base, R_i^model}
I_i^(H) = I_i^(0) union H_i^obs
I_i^(k) = I_i^(H) union O_i^(k)
```

This means:

```text
design-stage prior != post-process optional measurement state != early-release update state
```

The contract must never use one vague phrase such as "pre-observation prior" to
cover all three settings.

## Phase Gates

| Phase | Question | Required output | Pass condition | Fail action |
|---|---|---|---|---|
| 1 | Is there enough shared base descriptor language? | descriptor ontology, availability-stage table, coverage audit | `X_i^base` is declared and common fields cover enough curve-level samples for a stated task | downgrade to data-standardization / information-budget result |
| 2 | Can process be tokenized into route facts without audit leakage? | `R_i^model` taxonomy, `R_i^audit` taxonomy, separation rule | route facts exist beyond DOI or source identity, and audit metadata is excluded from the default model | keep process as source-confound audit only |
| 2.5A | What are the legal information states and release-state objects? | `I_i^(0)`, `I_i^(H)`, `I_i^(k)` plus `z_i^(0)`, `z_i^(H)`, `z_i^(k)` definition | base prior, optional measurement update, and early-release update are separated | do not train route model |
| 2.5B | What decision is the platform optimizing? | utility, target-window, and matched-window evaluation specification | target windows, risks, costs, thresholds, and future-only evaluation rules are explicit | do not claim screening platform |
| 3 | Do route facts improve a minimal model beyond legal shape priors? | route-fact benchmark with legal baselines only | `R_i^model` adds strict-split value beyond legal training-only shape priors | do not start neural model |
| 4 | Do optional proxies or early observations improve future prediction or decisions? | observation-budget benchmark | `H_i^obs` or `O_i^(k)` improves future-only, target-future, decision, uncertainty, or efficiency metrics under matched costs | restrict claim to the value layer that actually improves |
| 5 | Is neural learning justified after boundary cleanup? | neural gate benchmark | neural model beats classical and legal shape baselines under strict transfer without audit leakage | neural route remains negative control |
| 6 | Does wet lab validate decisions? | prospective wet-lab decision test | model-selected decisions beat random or global heuristics under the declared information budget | platform claim remains exploratory |

## Phase 1: Descriptor Ontology And Data Reality Checkpoint

Phase 1 is a research audit, not a naming exercise. It must report which
columns exist often enough to support cross-system learning and at what
information stage they become available.

Canonical descriptor layers:

| Layer | Examples | Role | Default channel |
|---|---|---|---|
| Drug | molecular weight, LogP, TPSA, pKa, Tm, charge, solubility | payload transport and partition prior | `X_i^base` |
| Material / carrier | PLGA Mw, LA/GA, lipid Tm, polymer concentration, HA/PVP/chitosan composition | carrier transport and matrix prior | `X_i^base` |
| Planned formulation | target drug-to-polymer ratio, target loading recipe, planned composition | intended formulation state | `X_i^base` |
| Medium | pH, temperature, buffer, ionic strength, enzyme, sink condition | release environment | `X_i^base` |
| Process record | route family, unit-operation order, reported conditions, additions, removals, drying, curing | route prior | `R_i^model` |
| Optional state proxy | measured particle size, actual loading, encapsulation efficiency, zeta, porosity, swelling, residual solvent, morphology, lamellarity, crystallinity | optional post-process update | `H_i^available` or `H_i^obs` |
| Curve | burst, early slope, t50, plateau, tail, AUC, qmax | observed release behavior | target only |

The checkpoint must answer:

```text
which fields belong to X_i^base versus H_i^available?
which fields are planned descriptors versus post-process measurements?
which descriptors cover >50% of eligible curves?
which route facts cover >30% of eligible curves?
which optional state proxies are missing too often for empirical use?
which datasets have curves but weak base descriptor language?
which datasets are worth collecting because they fill ontology gaps?
```

If only a small common descriptor set survives, the correct conclusion is that
drug-release ML is limited by reporting standards and hidden-state absence.
That is a valid result.

## Phase 2: Process Route Tokenization

Process must not be treated as a plain categorical shortcut, and it must not
smuggle expert-authored audit judgment into the default model.

The intended representation is:

```text
process text or record
-> factual route normalization
-> R_i^model
```

Audit objects are a separate route:

```text
process text or record
-> audit interpretation and confound analysis
-> R_i^audit
```

### `R_i^model`: allowed in the main model

`R_i^model` may include only information available from raw process records or
their factual normalization:

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

### `R_i^audit`: audit-only

`R_i^audit` includes but is not limited to:

```text
expected state-effect hypotheses
expected curve-symptom hypotheses
mechanistic interpretation
evidence strength
mapping confidence used for audit
confound-risk annotations
source-specificity risk
route-support counts
alternative explanations
expert-authored mechanism labels
```

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

The following outputs are audit products, not default model features:

```text
process_to_state_hypothesis_matrix.csv
process_to_curve_symptom_matrix.csv
```

If future work studies expert priors, it must be an explicit ablation:

```text
route facts only
vs
route facts + expert prior
```

Only the second condition may support a claim about expert-prior value.

## Phase 2.5A: Release-State Prior Definition

Before any route model is trained, the legal state objects must be:

```text
z_i^(0) = E[z_i | X_i^base, R_i^model]
z_i^(H) = u_H(z_i^(0), H_i^obs)
z_i^(k) = u_Q(z_i^(H), O_i^(k))
```

with:

```text
I_i^(0) = {X_i^base, R_i^model}
I_i^(H) = I_i^(0) union H_i^obs
I_i^(k) = I_i^(H) union O_i^(k)
```

`z_i^(0)` is not a measured physical state. It is an operational or predictive
release-state coordinate in a declared `ReleaseStateSpace`.

Physical interpretation requires independent state-proxy support. Without that,
the strongest allowed wording is:

```text
predictive release-state coordinate
operational release-state representation
process-conditioned prior over release behavior
```

## Phase 2.5B: Screening Utility Definition

The platform must define the decision before optimizing it.

Minimal decision objects:

| Object | Example |
|---|---|
| Target window | `Q_24h` in 40-70% |
| Failure event | `Q_1h` burst above 20% |
| Risk threshold | `P(failure) >= 0.8` |
| Go threshold | `P(target success) >= 0.8` and `P(failure) <= 0.2` |
| Optional measurement cost | each state-proxy characterization has declared cost |
| Observation cost | each release timepoint has declared cost |
| Measure-more rule | expected utility gain exceeds incremental measurement cost |

Without this specification, the platform cannot claim screening value even if
curve RMSE improves.

## Phase 3: Minimal Route Model

The first model is deliberately small. It tests whether factual route tokens
carry information beyond ordinary base descriptors and legal shape priors.

Required baseline taxonomy:

```text
global prior
legal pre-observation shape prior
static only
static + route facts
static + optional state proxy
static + route facts + optional state proxy
legal early-fit shape baseline, when early Q is allowed
full-curve oracle fit marked audit-only
```

The full-curve oracle fit may be used only as:

```text
oracle ceiling
post-hoc descriptive fit
shape-family adequacy audit
```

It must not be used as:

```text
pre-observation baseline
deployable comparator
primary Phase 3 model
```

Primary splits:

```text
leave-source-out
leave-drug-out
leave-process-out
leave-system-out, if enough systems exist
```

If route facts only help random splits, they are source proxies, not platform
mechanism.

If route signal appears only after `R_i^audit` enters the input, that is audit
leakage, not route-language success.

## Phase 4: Observation-Budget Update

Phase 4 tests whether optional state proxies or early release observations
improve the platform decision, not merely reconstruction of already observed
points.

Required comparisons should cover the declared information states:

```text
static only
static + route
static + optional state proxy
early only
static + early
static + route + early
static + route + optional state proxy
static + route + optional state proxy + early
legal early-fit shape baseline
```

Primary outputs must distinguish value layers:

```text
future-window RMSE or MAE
target-future RMSE or MAE
target-window success probability
failure probability
expected utility
uncertainty contraction
gain per point
gain per cost
stop / go / measure-more accuracy
false-stop rate
false-go rate
```

Observed early points may be used for conditioning, but they must not
contribute to the primary future-prediction score.

Observed-window reconstruction error may be reported only as:

```text
debug metric
assimilation diagnostic
context-reconstruction diagnostic
```

### Matched-window discipline

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

## Phase 5: Neural Entry Gate

Neural models enter only after Phases 1-4 show a real gap that the neural model
is designed to close, and only after the information boundaries are already
clean.

Permitted neural role:

```text
learn a transferable operational release-state language and observation update
```

Prohibited neural role:

```text
serve as a larger black-box regressor for weak descriptors
or
recover signal from audit-only metadata leakage
```

Minimum benchmark:

```text
neural encoder on declared information states only
vs legal pre-observation shape prior
vs legal early-fit shape baseline
vs ExtraTrees or ridge on the same information state
vs route-fact classical model
```

The neural route survives only if it improves strict transfer or decision
utility under the declared information budget.

## Phase 6: Wet-Lab Decision Validation

The planned HA/PVP/chitosan wet experiment should validate decision logic, not
merely curve prediction.

Prospective validation design:

```text
model selects high-risk candidates under a declared information state
model selects promising candidates under a declared information state
model selects uncertain candidates needing O_i^(k)
model recommends the highest-value optional state proxy from H_i^available
wet lab evaluates stop / go / measure-more decisions
```

This makes the wet experiment a platform validation rather than a small curve
prediction demonstration.

## 123 Boundary Gate

Before Experiment 123 starts, the following must be true:

```text
1. R_i^model and R_i^audit are separated.
2. X_i^base and optional state proxies are separated.
3. Every descriptor has an availability stage.
4. Early-observation value uses future-only or target-future-only scoring.
5. Legal shape priors, legal early-fit baselines, and full-curve oracle fits are separated.
6. Confound-risk belongs to the audit channel only.
7. This contract and the mathematical contract use the same terminology.
8. Experiment 123 does not train any new model.
```

Failing any item means 123 remains a documentation and audit prerequisite, not
a modeling experiment.

## Required Next Artifact

The next numbered artifact should be:

```text
123_release_screening_platform_formal_groundwork
```

It should be documentation and audit first, not model training. It should
produce:

```text
canonical_descriptor_schema.csv
dataset_descriptor_coverage.csv
process_route_taxonomy.csv
process_to_state_hypothesis_matrix.csv
process_to_curve_symptom_matrix.csv
utility_specification.csv
data_reality_checkpoint.csv
decision_table.csv
report.md
```

Those outputs document the contract and the audit trail. They do not authorize
model training by themselves.

## Kill Criteria

Stop before neural modeling if:

```text
common base descriptor coverage is too weak;
route facts collapse to DOI or source labels;
R_i^audit cannot be kept out of the default model;
X_i^base cannot be separated from optional state proxies;
future-only evaluation cannot be enforced;
legal shape priors and oracle fits are still mixed;
utility function is not specified;
route facts do not beat legal shape priors under strict splits;
early observations improve observed-point reconstruction but not future or decision value.
```

These failures are not wasted work. They define the reporting and
data-standard barrier that prevents drug-release ML from becoming a reliable
screening platform.
