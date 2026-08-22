# Unified Drug Release Program

Date: `2026-05-28`

Purpose:

```text
Turn the drug-release landscape audit into a concrete program:
- what is the right unifying object,
- what current competitors already occupy,
- what our current assets do and do not prove,
- what build sequence would make a real unified release system plausible.
```

This is not a paper draft. It is a strategy document.

## One-sentence position

We should try to unify **drug-release intelligence**, not pretend we already
have one universal drug-release equation or one universal neural network.

## The central distinction

There are three very different ambitions people keep conflating.

### 1. Better PLGA prediction

This is:

```text
PLGA descriptors + maybe early release -> future PLGA release
```

This is publishable, but narrow.

### 2. Unified drug-release intelligence

This is:

```text
formulation / material / assay context
+ sparse early observations
        ->
feasible latent release-state family
        ->
mechanism-conditioned decoder
        ->
future release + timescale + uncertainty + next assay value
```

This is much bigger and is the right bold target.

### 3. Universal release foundation model

This would mean:

```text
one model already handles all release mechanisms, assays, and materials
with little or no mechanism-specific structure
```

This is not defensible yet.

The right move is to aim for **2**, while refusing to claim **3** until the
evidence becomes real.

## What the field actually does now

From the landscape memo
([drug_release_landscape_and_unification_2026-05-28.md](D:/release-foundation/docs/drug_release_landscape_and_unification_2026-05-28.md)),
the current field clusters into five operating modes:

1. `task-specific tabular ML`
2. `few-shot release continuation from early points`
3. `mechanism or kinetic fitting within one material family`
4. `data / platform / standardization` work
5. `early-stage physics-aware deep learning`

Nobody visible has yet assembled the following full stack:

```text
shared release data interface
+ shared partial-observation benchmark
+ shared latent posterior object
+ mechanism-specific decoder registry
+ calibrated uncertainty
+ active measurement policy
+ cross-mechanism prospective validation
```

That is the empty space.

## The correct unifying object

The thing to unify is not raw `theta`.
The thing to unify is not one shared ODE either.
The thing to unify is the **release-state inference problem**.

## Proposed object

```text
Input:
  x_formulation
  x_material
  x_assay
  y_early(t)

Inference object:
  q(z_release | x_formulation, x_material, x_assay, y_early)

Decoder:
  p(y_future | z_release, decoder_mechanism, x_assay)

Decision layer:
  U(next_measurement | q, budget, assay_cost)
```

Where:

- `z_release` is a feasible latent release state,
- `decoder_mechanism` may be PLGA, chitosan, liposome, hydrogel, coating, etc.,
- the benchmark outputs are not just curve RMSE but also timescale and
  uncertainty.

## Why this is the right abstraction

Because it separates what is likely common from what is likely different.

Likely common across release systems:

- sparse early evidence,
- partial observability,
- multiple latent explanations,
- long-horizon forecast need,
- assay-cost constraints,
- need for uncertainty,
- practical interest in release timescale.

Likely mechanism-specific:

- exact decoder equations,
- latent-coordinate semantics,
- relevant state variables,
- dominant release regimes,
- meaningful assay interventions.

So the shared object should sit **above** the decoder and **below** raw data.

## Current project status against that object

## What we already have

### A. Shared partial-observation logic

Current release work already operationalizes:

```text
early observations -> future curve forecast
```

Evidence:

- [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
- [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)
- [outputs/38d_baselines_groupkfold/summary.txt](D:/release-foundation/outputs/38d_baselines_groupkfold/summary.txt)

### B. A real middle object

The current PLGA line already has a nontrivial middle object:

```text
formulation + early Q -> theta / feasible theta family -> decoder trajectory
```

That is the strongest thing we have that most competitors do not.

### C. ReleaseSimulator family architecture

The repo architecture is already pointed in the right direction:

- `ReleaseSimulator` subclasses
- `FormulationEncoder` abstraction
- cross-mechanism extension planned in architecture

Evidence:

- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md)

### D. Early benchmark interface thinking

The repo already has the beginning of a shared benchmark abstraction:

- [cross_domain_benchmark_spec_2026-05-28.md](D:/release-foundation/docs/cross_domain_benchmark_spec_2026-05-28.md)

This is not yet a release-only benchmark federation, but it is the right
design instinct.

## What we do not yet have

### 1. A shared release latent coordinate

Right now `theta` is still mechanism-local.

PLGA `theta` is not directly the same object as:

- chitosan swelling/diffusion parameters,
- liposome Weibull parameters,
- hydrogel release-mechanism parameters.

So if we want real unification, we need one of two things:

1. a shared higher-level release coordinate, or
2. a mechanism-tagged latent family with a common posterior interface.

Option 2 is easier and more honest now.

### 2. A decoder registry across mechanisms

Right now the architecture imagines this, but the project does not yet have a
real audited release-decoder library containing:

- `PLGABiphasic`
- `ChitosanRitgerPeppas` or successor
- `LiposomeWeibull` or IVR-specific decoder
- future hydrogel / coating decoders

### 3. A common release benchmark contract

We do not yet have one canonical release benchmark script that can consume
multiple release mechanisms through the same benchmark API.

That is a critical missing piece if we want to unify the field.

### 4. Release-timescale-first evaluation

The user's stated real-world goal is not "fit every point beautifully".
It is:

```text
predict how long controlled release lasts under material / ratio changes
```

That means the future unified program needs explicit release-timescale metrics:

- `t10`
- `t50`
- `t80`
- time to cross application-specific threshold
- burst magnitude
- tail persistence

Current release evidence is stronger on full-curve forecasting than on
timescale-as-primary-endpoint.

### 5. Cross-mechanism success with clean evidence

This is the biggest scientific gap.

Current state:

- PLGA: strong retrospective evidence
- Chitosan: prospective lock exists, reveal pending
- Liposome: data path exists, benchmark not yet fully operationalized

Until at least one non-PLGA mechanism is cleanly benchmarked, "unified drug
release" remains a program, not a proven result.

## What current competitors already occupy

This section matters because "bold" only works if we know which ground is
already taken.

## Competitor layer 1: direct release prediction

Occupied by:

- Bannigan / NC line
- explainable PLGA release ML papers
- small system-specific hydrogel / nanofiber / nanoparticle ML papers

Their value proposition:

```text
simple inputs -> useful release prediction
```

If we compete here only on a small metric gain, we lose strategically.

## Competitor layer 2: release-data standardization

Occupied by:

- AI-ready release data standardization work
- liposome IVR database builders

Their value proposition:

```text
clean data schema + reusable data assets
```

We should not ignore this. If we want to unify release, we need to absorb it.

## Competitor layer 3: platformized formulation AI

Occupied by:

- FormulationAI
- formulation strategy design platforms
- broader pharma-AI design tools

Their value proposition:

```text
decision support, not just one prediction model
```

This is a warning: if we only think in terms of "paper model", we may miss
the real strategic level.

## Competitor layer 4: future physics-aware deep learning

Occupied weakly, but starting to appear:

- PINN or hybrid physics-ML release modeling

This is not yet the center of the field, but it is an obvious future
convergence direction.

## The right bolder move

The right bolder move is not:

> "We have a better PLGA model"

and not:

> "We already unified all drug release"

It is:

> "We are building the first unified partially observed release-intelligence
> framework with a shared posterior object and mechanism-specific decoders."

That claim is:

- stronger than a benchmark paper,
- more concrete than "world model",
- closer to what the field is structurally missing,
- still compatible with our current evidence if we phrase it carefully.

## Program design

This is the part that turns ambition into a build sequence.

## Phase U1 — release benchmark federation

Goal:

```text
One release dataset interface and one benchmark contract across mechanisms.
```

Artifacts:

- release dataset schema
- benchmark split schema
- early/future mask schema
- timescale metric schema
- uncertainty metric schema

Minimum mechanism set:

- PLGA 321
- Bannigan 181
- liposome IVR
- chitosan prospective / retrospective as available

Success condition:

One benchmark runner can score all four through the same top-level API.

## Phase U2 — mechanism adapter registry

Goal:

```text
Every release family plugs in as a sibling decoder.
```

Artifacts:

- `PLGABiphasic`
- `ChitosanRitgerPeppas` or revised chitosan decoder
- `LiposomeWeibull` / IVR decoder
- placeholder hydrogel decoder interface

Success condition:

All benchmarked mechanisms obey one decoder contract:

```text
simulate(params, t_grid, assay_context) -> release trajectory
fit_or_pullback(curve) -> feasible latent targets
```

## Phase U3 — common posterior-family object

Goal:

```text
Mechanism-specific latent targets, common posterior interface.
```

Important design choice:

Do **not** force one universal physical `theta` now.

Instead define:

```text
PosteriorFamily = {
  mechanism_id,
  z_center,
  z_basis,
  z_scale,
  decoder_handle,
  assay_context,
}
```

This keeps the output object shared even when semantics differ.

Success condition:

PLGA, liposome, and chitosan can all emit the same downstream object shape.

## Phase U4 — release-timescale intelligence

Goal:

```text
Move evaluation and decision support from pointwise curve fit to practical
release-duration prediction.
```

New outputs:

- predictive distribution over `t50`
- predictive distribution over `t80`
- predicted burst fraction
- probability of being "fast", "intermediate", or "sustained"
- assay recommendation to reduce uncertainty in timescale

Success condition:

The system can answer:

> under this material and ratio, how long will release last and how uncertain
> are we?

That is much closer to the actual user need.

## Phase U5 — active measurement and cheap observation

Goal:

```text
Choose what to measure next, not only what to predict next.
```

Candidate modalities:

- next timepoint in IVR
- cheap early assay panel
- spectroscopy / Raman / imaging proxy if available

Success condition:

The system can justify one additional measurement under equal budget.

## What paper stories are possible

## Story A — safe release paper

```text
PLGA mechanism-state forecasting + chitosan prospective stress test
```

Pros:

- easiest to close

Cons:

- leaves the bigger unification opportunity mostly implicit

## Story B — bold but honest release-intelligence paper

```text
Unified partially observed release intelligence:
shared posterior-family object, mechanism-specific decoders,
PLGA retrospective core, chitosan prospective anchor, liposome retrospective extension
```

Pros:

- stronger scientific identity
- more field-defining

Cons:

- requires liposome and chitosan evidence to be clean enough

## Story C — premature foundation-model story

Not recommended now.

Why:

- the data federation is not yet big enough,
- the decoder set is incomplete,
- the cross-mechanism success is not yet proven,
- our learned observer route is not yet strong enough.

## The strongest near-term position

If we want to be bold without lying, the strongest near-term position is:

> **release intelligence framework**
>
> sparse early observations are converted into a calibrated feasible
> release-state family, decoded by mechanism-specific simulators or kinetic
> adapters, and used to forecast future release duration and uncertainty
> across multiple controlled-release systems

That is the correct big claim to build toward.

## Borrow list: what to steal from the field

Not code theft. Idea theft.

### From Bannigan / NC

- clean few-shot vs zero-shot separation
- simple deployment framing
- clear baseline culture

### From explainable PLGA ML papers

- reviewer-friendly interpretability
- explicit early-release importance story

### From liposome IVR ecosystem

- data curation discipline
- kinetic-family summaries
- non-PLGA benchmark bridge

### From AI-ready release data papers

- metadata-first thinking
- schema discipline
- benchmark federation mindset

### From FormulationAI / formulation platforms

- decision-support framing
- platform perspective beyond one model

### From PINN / physics-aware ML

- deeper future integration of mechanistic structure with learned inference
- but only after data and benchmarks are good enough

## Hard claim boundaries

What we can say now as a program:

- unified release intelligence is a plausible and strategically strong target
- the right common object is posterior-family inference over release state
- mechanism-specific decoders are compatible with that goal

What we cannot say now:

- one universal release model is already solved
- current evidence proves cross-mechanism mastery
- learned neural observer has already beaten simpler mechanism-routed methods

## Bottom line

Yes, we should try to be bolder.

But the boldness should be:

```text
unify the release inference problem
```

not:

```text
pretend all release mechanisms already collapse into one finished model
```

The program to build is:

1. benchmark federation,
2. mechanism adapter registry,
3. common posterior-family object,
4. release-timescale intelligence,
5. active measurement.

If those five pieces come together, then the project genuinely becomes more
than a PLGA paper. It becomes a candidate organizing framework for the drug
release ML field.
