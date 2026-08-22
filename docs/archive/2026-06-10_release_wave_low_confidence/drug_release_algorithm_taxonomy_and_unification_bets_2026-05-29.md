# Drug Release Algorithm Taxonomy and Unification Bets

Date: `2026-05-29`

Purpose:

```text
Answer the next-level strategic question:
not just "who published what",
but what algorithm families the external field is actually using,
what decision problems those families solve,
which ones are true competitors,
which ones are dangerous adjacent systems,
and where a unified drug-release program can still occupy open ground.
```

This file is a field-playbook, not a benchmark result.

## 1. Executive verdict

The external field is not one thing. It is currently split into six
algorithmic families:

1. `descriptor-only supervised predictors`
2. `descriptor + early-observation few-shot forecasters`
3. `kinetic-fit / mechanism-summary workflows`
4. `physics / simulator + ML hybrid design loops`
5. `active-learning / optimization systems`
6. `data / platform / schema layers`

Almost nobody currently unifies all of these into one release-intelligence
stack.

That is the real opening.

The right bold claim is:

> unify drug release at the interface, task, posterior-family, and
> reporting layers while keeping mechanism-specific decoders.

The wrong bold claim is:

> one universal black-box model has already solved all release systems.

## 2. What the field is actually doing now

## 2.1 Family A: descriptor-only supervised predictors

This is still the default external baseline family.

Typical input:

- formulation descriptors
- drug descriptors
- polymer / excipient / process variables
- sometimes time as an explicit input

Typical output:

- pointwise release percentage
- a full release profile reconstructed from predicted points
- occasionally a few summary kinetic endpoints

Typical algorithms:

- `RF`
- `XGBoost`
- `LightGBM`
- `SVR`
- `ANN / MLP`
- elastic-net / linear baselines

Representative works:

- [Bannigan et al., Nature Communications 2023](https://www.nature.com/articles/s41467-022-35343-w)
- [Polymeric drug-release ML review, Computer Methods and Programs in Biomedicine 2025](https://www.sciencedirect.com/science/article/pii/S0010482525001064)
- [Interpretable Two-Stage ML for PLGA Microspheres, Pharmaceuticals 2026](https://www.mdpi.com/1424-8247/19/5/767)

Why this family wins today:

- it matches small tabular datasets
- it is easy to explain
- it is easy to publish and deploy

What it does **not** naturally solve:

- one-to-many mechanism ambiguity
- prospective uncertainty discipline
- a shared cross-mechanism latent object
- active assay design

Short verdict:

> this family owns the current "standard ML for release" identity, but not the
> deeper release-intelligence slot.

## 2.2 Family B: descriptor + early-observation few-shot forecasters

This is the closest direct neighbor to the current project.

Typical input:

- descriptors of the candidate formulation
- one or a few early release observations from the target curve

Typical output:

- future release curve
- curve-level accuracy under few-shot deployment

Representative works:

- [Bannigan et al., Nature Communications 2023](https://www.nature.com/articles/s41467-022-35343-w)
- our current audited PLGA core:
  - [outputs/45_input_source_ablation/input_source_summary.csv](D:/release-foundation/outputs/45_input_source_ablation/input_source_summary.csv)
  - [outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv](D:/release-foundation/outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv)

Why this family matters:

- it maps directly onto real deployment
- it acknowledges that pure formulation-only prediction is weak
- it can be much stronger than descriptor-only OOD extrapolation

What is still missing externally:

- explicit feasible mechanism-state family
- shared uncertainty object
- shared cross-mechanism benchmark contract

Short verdict:

> this is where the current scientific action is for realistic release
> forecasting, but most external systems still treat it as direct regression.

## 2.3 Family C: kinetic-fit / mechanism-summary workflows

This family does not always forecast future release from sparse prefix data.
Instead, it fits or summarizes release behavior using mechanism-appropriate
kinetic models and uses those summaries downstream.

Typical input:

- dense or moderately dense release curves
- assay protocol metadata
- sometimes formulation descriptors

Typical output:

- fitted kinetic parameters
- curve cluster labels such as fast / medium / slow
- optimized release-testing conditions

Representative works:

- [A machine learning workflow to accelerate the design of in vitro release tests from liposomes, Digital Discovery 2025](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00112a)
- [Computational pharmaceutics review, Advanced Drug Delivery Reviews 2021](https://www.sciencedirect.com/science/article/pii/S0168365921004363)

Why this family matters:

- it respects mechanism-specific assay structure
- it often yields better scientific summaries than raw pointwise regression
- it creates cleaner non-PLGA benchmark assets

What it usually does **not** solve:

- sparse-prefix future forecasting
- unified cross-mechanism inference objects
- calibrated deployment uncertainty

Short verdict:

> this family is less of a direct forecasting competitor and more of a serious
> source of mechanism discipline that the field underuses.

## 2.4 Family D: physics / simulator + ML hybrid design loops

This family is broader than release prediction. It tries to combine
simulators, mechanistic priors, or physiology with ML-guided formulation
search.

Typical input:

- formulation descriptors
- mechanistic simulation outputs
- PBPK / PD or related domain simulators
- assay or in vivo targets

Typical output:

- formulation proposals
- optimized composition ranges
- downstream performance estimates

Representative works:

- [FormulationLAI, Journal of Controlled Release 2026](https://www.sciencedirect.com/science/article/pii/S0168365925010326)
- [Computational pharmaceutics review, Advanced Drug Delivery Reviews 2021](https://www.sciencedirect.com/science/article/pii/S0168365921004363)

Why this family matters:

- it is where platform-level ambition starts
- it can absorb simulation, wet-lab constraints, and optimization in one loop

What it usually does **not** yet prove:

- that a shared partially observed release-forecast task is solved
- that one release-state object works cleanly across mechanisms

Short verdict:

> this is not the cleanest direct benchmark rival today, but it is the most
> dangerous adjacent direction if it becomes release-forecast-native.

## 2.5 Family E: active-learning / optimization systems

This family is usually not trying to model the full release trajectory as the
primary object. It tries to choose experiments, propose formulations, or drive
closed-loop optimization.

Typical input:

- a candidate design space
- sparse assay feedback
- objective functions like target release, stability, or efficacy

Typical output:

- which experiment to run next
- which formulation region to search next
- which candidate is expected to dominate

Representative evidence:

- [Machine learning guided optimization in drug delivery, Advanced Drug Delivery Reviews 2025](https://www.sciencedirect.com/science/article/pii/S0169409X25002467)
- [Active learning for hydrogel formulation discovery, ACS Applied Materials & Interfaces 2025](https://pubmed.ncbi.nlm.nih.gov/40325116/)

Why this family matters:

- this is the route from prediction to real lab acceleration
- it competes on workflow value, not only retrospective metrics

What it usually does **not** yet provide:

- a rigorous shared release benchmark
- a release-state posterior-family artifact
- a cross-mechanism audited forecasting stack

Short verdict:

> if we stay only at retrospective forecasting, this family can outflank us on
> practical value even without a deeper release model.

## 2.6 Family F: data / platform / schema layers

This family is often underestimated because it does not always look like
algorithm innovation.

Typical output:

- AI-ready datasets
- metadata standards
- web platforms
- interoperable intake and benchmark conventions

Representative works:

- [Dataset and formulation parameters of drug-loaded PLGA microparticles, Scientific Data 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873201/)
- [FormulationAI, Briefings in Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/37991246/)

Why this family matters:

- it shapes what becomes benchmarkable
- it lowers the barrier to adoption
- it can win mindshare before the best algorithm wins science

Short verdict:

> this family does not usually beat a forecasting model head-to-head, but it
> can become the operating system of the field.

## 3. What is the actual mainstream by problem type

The field can be summarized more cleanly by *decision problem* than by model
name.

| Problem type | Current mainstream answer | Typical journals |
|---|---|---|
| `Only descriptors available` | tree ensembles or ANN on tabular formulation data | Nature Communications, CMPB, Pharmaceuticals |
| `A few early release points available` | few-shot supervised curve forecasting | Nature Communications, formulation-focused method papers |
| `Need mechanistic assay summaries` | kinetic fitting, clustering, release-regime grouping | Digital Discovery, release workflow papers |
| `Need formulation search / optimization` | active learning, Bayesian or surrogate-guided design loops | ADDR, optimization/platform papers |
| `Need adoption / reuse / interoperability` | datasets, schema papers, platforms | Scientific Data, Briefings in Bioinformatics |

That means the field is not yet asking one unified question. It is asking at
least five partially overlapping ones.

## 4. So can we try to unify the whole drug-release space?

`Yes, but only if we unify the right things.`

The most important mistake would be trying to unify *raw mechanism parameters*
too early.

The correct unification ladder is:

### Level 0: no unification

Each material family has its own dataset, model, metrics, and story.

This is where most of the field still lives.

### Level 1: shared curve schema

All mechanisms can at least be represented as:

- assay context
- formulation context
- time grid
- fractional release curve

External field strength here:

- data papers and curated datasets

### Level 2: shared partially observed benchmark

All mechanisms can be evaluated under:

- descriptor-only
- early-observation-assisted
- OOD split families
- prospective lock discipline

This is the first real release-intelligence layer.

### Level 3: shared posterior-family object

All mechanisms emit:

- a feasible release-state family
- uncertainty metadata
- decoder handle
- observation context

Current local evidence that this level is already real:

- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)
- [outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2](D:/release-foundation/outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2)
- [outputs/85_liposome_group_by_api_our_et_refined](D:/release-foundation/outputs/85_liposome_group_by_api_our_et_refined)
- [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)

### Level 4: shared decision-support layer

All mechanisms can drive:

- forecasted future release
- duration / tail / burst summaries
- uncertainty-aware decisions
- assay selection or formulation search

This level is not solved locally yet, but it is the correct place to be bold.

Short verdict:

> yes, we can try to unify the whole drug-release space, but the unification
> target should be the release-intelligence stack, not a single raw model.

## 5. Who is the real competition now

The real competition splits into three groups.

## 5.1 Direct benchmark competitors

These compete on the exact question:

```text
given a formulation and maybe a few early points,
can you predict release better than the standard baseline?
```

Main names:

- [Bannigan et al., Nature Communications 2023](https://www.nature.com/articles/s41467-022-35343-w)
- [PLGA interpretable two-stage ML, Pharmaceuticals 2026](https://www.mdpi.com/1424-8247/19/5/767)

These are the most important head-to-head rivals.

## 5.2 Dangerous adjacent systems

These are not one-to-one benchmarking rivals today, but they can become more
strategically dangerous if they absorb release forecasting.

Main names:

- [FormulationLAI, Journal of Controlled Release 2026](https://www.sciencedirect.com/science/article/pii/S0168365925010326)
- [FormulationAI, Briefings in Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/37991246/)
- active-learning / optimization lines summarized in
  [ADDR 2025](https://www.sciencedirect.com/science/article/pii/S0169409X25002467)

Why they are dangerous:

- they compete on workflow value
- they can look more translational than a benchmark paper
- they can become the user-facing operating layer of the field

## 5.3 Borrow-first infrastructure neighbors

These are better treated as assets to borrow from than enemies to beat.

Main names:

- [Scientific Data PLGA dataset 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873201/)
- [Digital Discovery liposome workflow 2025](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00112a)
- [Computational pharmaceutics review 2021](https://www.sciencedirect.com/science/article/pii/S0168365921004363)

What to borrow:

- data packaging
- assay metadata discipline
- mechanism-appropriate kinetic summaries
- hybrid workflow language

## 6. What we should borrow, and what we should refuse

### Borrow now

1. `few-shot / zero-shot deployment language` from NC
2. `non-PLGA intake and kinetic-summary discipline` from liposome IVR work
3. `metadata-first benchmark federation` from dataset / schema papers
4. `decision-support framing` from FormulationAI and adjacent platforms
5. `closed-loop optimization ambition` from active-learning systems

### Refuse now

1. pretending direct regression alone is enough
2. pretending one black-box model should erase mechanism differences
3. claiming solved cross-mechanism generalization before reveal-quality evidence
4. replacing benchmark discipline with vague platform hype

## 7. Where the open ground still is

No external line clearly owns this combined slot:

```text
shared partially observed release benchmark
+ mechanism-specific decoders
+ shared posterior-family object
+ curve / timescale / uncertainty reporting
+ prospective lock behavior
+ future decision-support layer
```

That is exactly the slot we should try to occupy.

This is why the right ambition is not:

```text
"the best PLGA regressor"
```

It is:

```text
"the first serious unified drug-release intelligence stack"
```

## 8. Bottom line

The external field is currently strong on:

- supervised release regression
- a few-shot forecasting baselines
- workflow-specific kinetic summaries
- formulation platforms
- dataset and schema infrastructure

The external field is currently weak on:

- shared partially observed release benchmarks
- explicit feasible release-state families
- cross-mechanism object design
- unified curve + timescale + uncertainty reporting
- prospective discipline tied to the same shared object

So yes, we should be bolder.

But the right boldness is:

> unify drug release as a release-intelligence problem

not:

> claim that one universal model has already solved all release mechanisms.
