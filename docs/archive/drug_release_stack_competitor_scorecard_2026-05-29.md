# Drug Release Stack Competitor Scorecard

Date: `2026-05-29`

Purpose:

```text
Put the external field on the same stack we are using internally:
1. who is doing direct release prediction,
2. who is building workflow/platform value,
3. who is shaping data infrastructure,
4. who actually threatens a "unified release-intelligence" position,
5. where we should fight, borrow, or defend.
```

Machine-readable companion:

- [drug_release_stack_competitor_scorecard_2026-05-29.csv](D:/release-foundation/docs/drug_release_stack_competitor_scorecard_2026-05-29.csv)

## 1. Executive verdict

The field now has three different pressure types, and they should not be
treated as one contest.

### Pressure A — direct benchmark pressure

These systems ask:

> given descriptors and maybe a few early points, can we predict release?

Main representatives:

- `NC / Bannigan`
- 2026 explainable LAI forecasting paper
- 2026 larger-scale PLGA supervised predictor

This is where we must `fight`.

### Pressure B — workflow / platform pressure

These systems ask:

> can we accelerate formulation design, assay planning, or optimization even
> if the underlying release model is not the deepest one?

Main representatives:

- `FormulationLAI`
- `FormulationAI`
- hydrogel active learning
- broader AI/ML drug-delivery optimization systems

This is where we must `defend` our narrative and borrow what is useful.

### Pressure C — infrastructure pressure

These systems ask:

> can we package release data well enough that the field can actually become
> benchmarkable and interoperable?

Main representatives:

- Scientific Data PLGA dataset
- AI-ready IVR / formulation data work
- liposome IVR workflow packaging

This is where we should mostly `borrow`.

## 2. Stack-level reading

The current project is trying to occupy a stack position that almost nobody
else owns end-to-end:

```text
shared schema
+ shared partially observed benchmark
+ shared posterior-family object
+ mechanism-specific decoders
+ shared timing / shape / uncertainty reporting
```

Most external works occupy only part of that stack.

### Direct supervised predictors

Examples:

- [Predictive Modelling of Drug Release from PLGA Microparticles Using Artificial Intelligence and Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803328/)
- [Predicting early and complete drug release from long-acting injectables using explainable machine learning](https://www.nature.com/articles/s41598-026-08092-5)
- [Bannigan et al., Nature Communications 2023](https://www.nature.com/articles/s41467-022-35343-w)

Pattern:

- they mostly live in `descriptor -> release` or `descriptor + early points -> release`
- they exert real benchmark pressure
- they usually do **not** introduce a shared feasible-state object

Strategic implication:

> We should beat or at least clearly differentiate from these on the
> forecasting core, because this is where reviewers will compare us first.

### Mechanism-aware workflow papers

Example:

- [A machine learning workflow to accelerate the design of in vitro release tests from liposomes](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00112a)

Pattern:

- strong on mechanism-local assay discipline
- useful as the first non-PLGA bridge
- not yet a shared partially observed forecasting benchmark

Strategic implication:

> This is more of a bridge benchmark and borrowing target than a direct rival.

### Platform and optimization systems

Examples:

- [FormulationLAI](https://www.sciencedirect.com/science/article/pii/S0168365925010326)
- [FormulationAI](https://pubmed.ncbi.nlm.nih.gov/37991246/)
- [Automated active learning to optimize hydrogel drug release profiles](https://pubmed.ncbi.nlm.nih.gov/40325116/)
- [AI/ML guided optimization in drug delivery review](https://www.sciencedirect.com/science/article/pii/S0169409X25002467)

Pattern:

- often stronger on translational workflow value than on clean forecasting
  architecture
- can outflank a benchmark paper by looking more practically useful

Strategic implication:

> If we only present ourselves as a benchmark model, these systems can steal
> the "why should anyone care?" slot.

That is why the shift toward `release intelligence` and eventually
`decision-support` matters.

### Data and schema infrastructure

Examples:

- [Scientific Data PLGA dataset 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873201/)
- [Making in vitro release and formulation data AI-ready: A foundation for streamlined nanomedicine development](https://www.sciencedirect.com/science/article/pii/S0010482525001064)

Pattern:

- they do not beat forecasting models directly
- but they shape what becomes tractable, reproducible, and benchmarkable

Strategic implication:

> We should not compete with these as if they were forecasting rivals; we
> should absorb their discipline into our intake and benchmark layers.

## 3. What this means for the bold claim

The scorecard makes the boldness boundary clearer.

### What is defensible now

`unified drug-release intelligence`

Because we already have evidence for:

- shared intake
- shared partially observed benchmark language
- shared posterior-family object
- a first non-PLGA bridge
- calibrated-family artifacts across a `PLGA + liposome` panel

Supporting local evidence:

- [unified_release_intelligence_contract_2026-05-29.md](D:/release-foundation/docs/unified_release_intelligence_contract_2026-05-29.md)
- [unified_release_capability_snapshot_2026-05-29.md](D:/release-foundation/docs/unified_release_capability_snapshot_2026-05-29.md)
- [outputs/91_release_paper_figures/figure1](D:/release-foundation/outputs/91_release_paper_figures/figure1)

### What is not defensible now

`universal solved release model`

The scorecard shows why:

- direct supervised rivals still define the benchmark expectation
- workflow platforms still own much of the translational narrative
- infrastructure papers still own much of the metadata discipline
- we still lack completed cross-mechanism prospective evidence

## 4. Fight / borrow / defend map

### Fight

- NC / Bannigan
- explainable LAI few-shot forecasting
- new larger-scale PLGA supervised predictors

Reason:

- these are the closest benchmark rivals on the exact scientific task

### Borrow

- liposome IVR workflow
- Scientific Data PLGA curation
- AI-ready IVR/formulation data standards
- computational pharmaceutics / hybrid modeling language

Reason:

- they improve our stack without forcing us into their narrower problem
  definitions

### Defend

- FormulationLAI
- FormulationAI
- hydrogel active-learning systems
- broader AI/ML optimization narratives

Reason:

- these can make a pure forecasting paper look strategically smaller unless
  we show the path toward decision-support

## 5. The most important open ground

No one clearly owns this combined slot:

```text
shared partially observed release benchmark
+ explicit feasible-state family
+ mechanism-specific decoders
+ unified timing / shape / uncertainty reporting
+ prospective lock discipline
```

That is the space we should continue trying to occupy.

## Bottom line

The scorecard says the field is not converging on one dominant release
intelligence stack yet.

That is exactly why being bolder still makes sense.

But the scorecard also says the boldness has to be precise:

> fight direct predictors on forecasting,
> borrow infrastructure and mechanism discipline,
> defend against platform systems by moving toward release intelligence and
> eventual decision-support.
