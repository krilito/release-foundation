# Drug Release Landscape And Unification Memo

Date: `2026-05-28`

Purpose:

```text
Answer three practical questions for the release-foundation project:
1. What are people actually doing now in ML + drug release?
2. Can the field be unified into one bigger algorithmic object?
3. Who are the real competitors, and what can we borrow from them?
```

This note is intentionally blunt. It is not a hype memo.

## Executive answer

### What the field is doing now

Current ML + drug release work is mostly one of the following:

1. `tabular descriptors -> release output`
2. `tabular descriptors + a few early release points -> later release output`
3. `time-curve fitting surrogates` over one specific material family
4. `formulation design / property prediction` adjacent to release, but not full release forecasting
5. `review / platform / data-standardization` work that prepares the ground for broader AI

The field is **not** currently organized around a single unified mechanism-aware release model.

### Can drug release be unified?

Yes, but only at the **interface and posterior-object level**, not as one monolithic decoder right now.

The bold but still defensible version is:

```text
partial release observations + formulation / assay context
        ->
feasible latent release-state family
        ->
mechanism-conditioned decoder
        ->
future release trajectory / timescale / uncertainty
        ->
next-measurement utility
```

The wrong version is:

```text
one universal neural network already solves PLGA, liposome, chitosan,
hydrogels, coatings, and every assay without mechanism adapters
```

The current literature does not support that stronger claim.

### Where we can be bolder

The project can credibly aim at:

- a **unified release intelligence interface**
- a **shared posterior-family object**
- a **mechanism-specific decoder library**
- a **common partial-observation benchmark**
- a **common uncertainty and timescale evaluation layer**

That is much bigger than "one more PLGA predictor", but still more honest than "drug-release foundation model" today.

## What the literature is actually doing

## 1. General trend: computational pharmaceutics, not unified release intelligence

The clearest broad review is the computational pharmaceutics review in
`Advanced Drug Delivery Reviews`, which frames the area as AI/ML helping
pharmaceutical formulation, process, and performance prediction rather than
as a unified release-state inference problem.

Source:

- `Computational pharmaceutics: computational methods and modeling in the development of pharmaceutical formulations`
  - https://www.sciencedirect.com/science/article/pii/S0168365921004363

Takeaway:

- The field already accepts ML in formulation science.
- But the dominant framing is still `task-by-task prediction`, not one common
  dynamical object spanning mechanisms.

## 2. Release-specific review: ANN and ensemble methods dominate

A recent review focused directly on polymeric drug-delivery release prediction
summarizes the dominant algorithm families as:

- artificial neural networks,
- random forests / ensemble methods,
- boosting-style tabular learners,
- feature-engineered supervised prediction.

Source:

- `Utilizing machine learning for predicting drug release from polymeric drug delivery systems`
  - https://www.sciencedirect.com/science/article/pii/S0010482525001064

Practical takeaway:

- Most groups are still solving release as a `supervised regression` problem.
- The field is not yet crowded with strong mechanism-state or active-observer
  methods.
- That creates room for us, but only if the evidence is clean.

## 3. Data-standardization is emerging as a real bottleneck

One of the most important signals is not a model paper but a data paper:

- `Making in vitro release and formulation data AI-ready: A foundation for machine learning development in drug delivery`
  - PDF: https://research.chalmers.se/publication/548309/file/548309_Fulltext.pdf

This is important because it says out loud what we keep running into:

- release data are fragmented,
- assay conditions vary,
- time grids vary,
- descriptors are incomplete,
- most datasets are too small and too heterogeneous for naive "foundation model"
  claims.

Practical takeaway:

- The road to unification starts with `data interface + benchmark interface`,
  not with a giant model first.

## Competitor map

The point here is not whether every paper is directly stronger than us.
The point is to identify what role each paper plays in the landscape.

| Competitor / inspiration | What they do | Why it matters | Main limitation | Strategic lesson |
|---|---|---|---|---|
| Bannigan et al., Nature Communications 2023 | ML models for polymeric long-acting injectables; few-shot and zero-shot tabular prediction | This is still the most visible direct ML-for-release benchmark line | No mechanism-state bottleneck; zero-shot external wet prediction is weak; not cross-mechanism | Direct competitor for PLGA release prediction, but not for mechanism-aware release intelligence |
| Bannigan follow-up explainable ML on PLGA MPs | Predicts early and complete release from literature-mined PLGA dataset with explainable tabular ML | Strong evidence that descriptor + early-release ML is publishable and useful | Still mostly system-specific supervised learning | The field likes interpretable tabular models; we need to beat them on object, not just one metric |
| Yanes et al. liposome IVR / accelerated IVR | Standardizes liposome release data, fits Weibull kinetics, clusters curves, benchmarks ML classifiers | This is one of the cleanest non-PLGA release data ecosystems | Not a unified cross-mechanism latent-state system | Excellent first non-PLGA benchmark and strong adapter target |
| PLGA nanoparticle integrated ML + in vitro studies | Small experimental ML tied to one release system | Shows assay-coupled ML is publishable in formulation journals | Narrow scope, small data, usually not reusable | Good reminder that many papers win with much smaller claims than ours |
| Hybrid ML for temperature-responsive hydrogels | Predicts profiles, kinetics, and mechanism labels in one hydrogel family | Closer to "mechanism-aware" than most papers | Still family-specific, not a shared release interface | We should watch mechanism-labeled hydrogel work as a structural cousin |
| Acetalated dextran nanofiber ML release modeling | Supervised prediction of release from a single scaffold family | Shows even small systems publish with disciplined ML | Material-specific; not a transferable release object | Another proof that most of the field is local, not universal |
| Raman-spectrum -> release soft sensor papers | Cheap observation modality predicts release or release quality | Directly relevant to future active observation / cheap sensing | Often modality-specific and small | This is one of the best future expansion routes beyond simple Q(t) sampling |
| Physics-informed neural network release papers | Apply PINN-style constraints to release PDE/ODE structure | Signals that physics-aware deep learning is entering the space | Still early, sparse, and not the dominant paradigm yet | Useful inspiration, but not yet the standard we need to beat |
| FormulationAI / FormulationDT / AI formulation platforms | Broader AI for formulation design, optimization, and recommendation | Competitors at the platform level, not just per-paper level | Usually broader than release; often not release-curve-grounded | We should think platform and workflow, not only manuscript |

## Evidence for specific competitor roles

### A. Bannigan / NC line

Primary source:

- `Machine Learning Models to Accelerate the Design of Polymeric Long-Acting Injectables`
  - https://www.nature.com/articles/s41467-022-35343-w
- Code:
  - https://github.com/aspuru-guzik-group/long-acting-injectables

What it is:

- Few-shot and zero-shot ML over LAI release data.
- Traditional tabular model zoo with `RF / LGBM / XGB / SVR / PLS / NGB`.
- Public code makes clear that the work is fundamentally `descriptors (+ maybe early release) -> release`.

What it is not:

- not a mechanism-state model,
- not an active-observation model,
- not a cross-mechanism release intelligence system.

### B. Explainable ML over larger PLGA release datasets

Relevant signal:

- `Predicting early and complete drug release from long-acting injectables by using explainable machine learning`
  - indexed at PubMed / Medline search results
  - https://pubmed.ncbi.nlm.nih.gov/?term=Predicting+early+and+complete+drug+release+from+long-acting+injectables+by+using+explainable+machine+learning

Why it matters:

- It shows the community values `early-release + full-release` prediction.
- It sits very close to our current scientific problem.

### C. Liposome IVR / accelerated IVR

Sources:

- `A machine learning workflow to accelerate the design of in vitro release tests from liposomes`
  - https://pubmed.ncbi.nlm.nih.gov/?term=A+machine+learning+workflow+to+accelerate+the+design+of+in+vitro+release+tests+from+liposomes
- GitHub:
  - https://github.com/danielyanes22/nanomed_IVR_data
  - https://github.com/danielyanes22/accelerated_IVR

Why it matters:

- This is one of the best concrete cross-mechanism release competitors.
- It also shows a useful pattern: `clean dataset + kinetic summary + ML workflow`
  can become a credible publishable asset even without a grand model.

### D. AI-ready release data standardization

Source:

- `Making in vitro release and formulation data AI-ready: A foundation for machine learning development in drug delivery`
  - https://research.chalmers.se/publication/548309/file/548309_Fulltext.pdf

Why it matters:

- This is one of the strongest external arguments that a unification project
  should start with data schema, metadata, and benchmark interface.

### E. Broader formulation-AI platforms

Sources:

- `FormulationAI: A machine learning-powered web-based platform for integrated preformulation analysis and pharmaceutical formulation design`
  - https://pubmed.ncbi.nlm.nih.gov/?term=FormulationAI%3A+A+machine+learning-powered+web-based+platform+for+integrated+preformulation+analysis+and+pharmaceutical+formulation+design
- `AI-directed formulation strategy design initiates rational drug development`
  - https://www.nature.com/articles/s41467-024-53515-8

Why they matter:

- They are not direct release-curve competitors.
- But they are strong examples of how the field is moving from
  `single-model paper` toward `decision platform`.

## What is actually unifiable in drug release

This is the key question.

## Level 1: unify the benchmark and data interface

This is feasible now.

Shared objects:

- formulation / material descriptors,
- assay context,
- sparse early observations,
- future release target,
- curve-level and time-scale-level metrics,
- uncertainty metrics,
- observation budget.

This should become a common dataset schema and benchmark protocol.

## Level 2: unify the posterior object

This is also feasible now.

Shared object:

```text
q(z | x_formulation, assay, y_early)
```

where `z` is not "the one true universal parameter set", but a
mechanism-conditioned feasible latent release state.

That object can be shared even when the decoder differs by mechanism.

## Level 3: unify the decoder family behind an adapter layer

This is feasible, but only as a family of siblings:

- `PLGA` decoder
- `liposome / Weibull-like or IVR-specific` decoder
- `chitosan swelling / diffusion` decoder
- `hydrogel` decoder
- `coating / colon-targeted coating` decoder

This is the `ReleaseSimulator` family idea, not a single universal equation.

## Level 4: unify active observation

Potentially feasible.

Shared question:

```text
Given current uncertainty over future release, what is the next assay time
or observation modality with the highest expected information value?
```

This can be common across mechanisms if the utility is defined over the
posterior-family object rather than over one mechanism's raw parameters.

## Level 5: one universal model with no mechanism adapters

Not credible right now.

Why not:

- datasets are too fragmented,
- assay metadata are too inconsistent,
- mechanisms are too heterogeneous,
- current literature is not anywhere near this standard,
- our own evidence is not there yet.

So the honest unification claim is:

> **unified release intelligence interface and posterior object, with
> mechanism-specific decoders**

not:

> **already-universal drug-release foundation model**

## Where our current project is stronger than the field

If we stay disciplined, the project already has several things most papers do
not have:

1. `partial-observation forecasting`
   - Most papers are still one-shot supervised prediction systems.

2. `mechanism-state bottleneck`
   - Most papers do not force predictions through a decoder-compatible latent
     object.

3. `audited direct-Q vs mechanism-route comparison`
   - Most papers do not do this level of ablation.

4. `prospective lock mindset`
   - Even if chitosan reveal fails, pre-registration and hash-locking are
     stronger evidence behavior than the norm in this literature.

5. `posterior-family / calibrated UQ ambition`
   - The literature is still dominated by point predictors.

## Where we are still weaker than the field

We should also be honest about what we still do not have.

1. `clean cross-mechanism success`
   - Liposome is still early.
   - Chitosan reveal is still pending.

2. `simple deployable success story`
   - Competitors like NC are easier to explain: "descriptors + early points ->
     release."
   - Our story is deeper but easier to overcomplicate.

3. `fully convincing learned observer`
   - Our tree/theta/ODE route is stronger than our learned world-model route.

4. `large standardized benchmark federation`
   - The field is fragmented; we have not yet become the benchmark hub that
     others must use.

## Strategic position: how bold can we be?

## Too timid

Too timid would be:

> another PLGA release predictor with slightly better RMSE

That leaves too much value on the table.

## Too aggressive

Too aggressive would be:

> universal release foundation model across all biomaterials and mechanisms

Current evidence does not support this.

## The right boldness

The right boldness is:

> a unified partially observed drug-release intelligence framework
> that converts sparse early measurements into a calibrated feasible
> release-state family, then decodes future release through
> mechanism-specific simulators or kinetic adapters

This is:

- bigger than a benchmark paper,
- more defensible than a foundation-model claim,
- structurally extensible to multiple mechanisms,
- close to what the external literature is missing.

## Journal landscape

These are the most relevant non-pure-OA or not-mostly-OA journals for this
project's current release direction.

| Journal | Why it fits | IF / OA signal | Role |
|---|---|---|---|
| Journal of Controlled Release | Best topical fit for release forecasting and mechanism-aware drug delivery | IF `11.5`; Gold OA `22.04%` | Best release-first target |
| International Journal of Pharmaceutics | Strong for formulation + release + methods | IF `5.2`; Gold OA `20.47%` | Practical high-fit target |
| Molecular Pharmaceutics | Good if mechanism-state and formulation-science balance are strong | IF `4.5`; Gold OA `12.79%` | Good methods / formulation bridge |
| Drug Delivery and Translational Research | Good if prospective or translational angle becomes real | IF `5.5`; Gold OA `27.38%` | Good if chitosan reveal works |
| European Journal of Pharmaceutics and Biopharmaceutics | Good pharmaceutical-science home with release relevance | IF `4.3`; Gold OA `28.61%` | Solid fallback / realistic fit |
| Journal of Drug Delivery Science and Technology | Strongly on-topic, slightly lower prestige | IF `4.9`; Gold OA `7.21%` | Conservative release-specific option |

Sources:

- Journal of Controlled Release
  - https://www.ablesci.com/journal/detail?id=5JW48D
- International Journal of Pharmaceutics
  - https://www.ablesci.com/journal/detail?id=rwqKB5
- Molecular Pharmaceutics
  - https://www.ablesci.com/journal/detail?id=rRQQwD
- Drug Delivery and Translational Research
  - https://www.ablesci.com/journal/detail?id=5x8Ng5
- European Journal of Pharmaceutics and Biopharmaceutics
  - https://www.ablesci.com/journal/detail?id=DGL2vr
- Journal of Drug Delivery Science and Technology
  - https://www.ablesci.com/journal/detail?id=DZ69XD

Not preferred for the current objective:

- `Advanced Drug Delivery Reviews`
  - too review-oriented for this paper
- `Pharmaceutics`
  - highly OA-heavy, less aligned if we explicitly prefer non-pure-OA
- `Computers in Biology and Medicine`
  - usable as an algorithmic outlet, but not the natural home for a release
    paper

## Recommended next moves

1. Keep the current release paper framed around the audited PLGA core and the
   chitosan prospective reveal.

2. Do **not** market the current asset as a finished universal release model.

3. Start building the unification program around three artifacts:
   - a common release dataset interface,
   - a common partial-observation benchmark,
   - a mechanism-adapter simulator library.

4. Use `liposome IVR` as the first real non-PLGA mechanism benchmark.

5. If chitosan reveal succeeds, write the paper as:

```text
early measurement -> feasible release-state family -> future release forecast
```

not as:

```text
one neural network solves all release systems
```

## Bottom line

The field is still mostly doing `small-system supervised ML`.

That means:

- there is real room for a stronger unifying object,
- but the winning move is not a premature foundation-model slogan,
- the winning move is a **shared release-intelligence interface with
  mechanism-specific decoders and calibrated partial-observation inference**.

That is the bold version that still matches the evidence.
