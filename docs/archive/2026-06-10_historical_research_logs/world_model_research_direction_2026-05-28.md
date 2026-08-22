# World-Model Research Direction - 2026-05-28

This document records the direction I actually want, instead of the direction
 that is easiest to inherit from the current drug-release literature.

It has two jobs:

1. define the research identity clearly,
2. extract algorithm papers that can grow from the current codebase without
   pretending that release prediction itself is the final goal.

## Part I - Research Direction Statement

## One-line identity

I want to study how a model can build a reality-constrained internal model of a
 partially observed world.

Not just how to predict the next observation.
Not just how to fit a curve.
Not just how to look useful on an application benchmark.

The real question is:

> How should a model represent what is true, what is ambiguous, and what must
> be measured next in a dynamical system that is only partially observed?

## Why this direction matters

Current AI systems, especially language-first systems, often fail in a very
 specific way:

- they can sound coherent without being state-consistent,
- they can interpolate observations without learning the governing structure,
- they can output a single confident answer when the world only supports a
  feasible family of answers,
- and they often do not know the difference between uncertainty, ambiguity, and
  ignorance.

I do not want to work only on "better prediction."
I want to work on reality-constrained inference.

## What "truth" means here

Truth is not only "the prediction matches the observation."

There are at least four levels:

1. observation truth
   The predicted observation matches what is seen.

2. state truth
   The internal state representation aligns with the external latent state.

3. mechanistic truth
   The model captures the structure that actually governs evolution, not just a
   shortcut correlation.

4. epistemic truth
   The model is honest about what is identifiable, what is only weakly
   constrained, and what remains ambiguous.

A system can succeed at level 1 while failing at 2, 3, or 4.
That failure mode is one of the main things I want to study.

## Working research object

The working object is not "drug release."

The working object is:

> partially observed mechanistic dynamical systems with ambiguity,
> non-identifiability, uncertainty, and limited experimental budget.

Drug release is one micro-world.
Chitosan, PLGA, liposomes, hydrogels, and related systems are candidate
 scientific testbeds.

These domains matter because they provide:

- real partial observation,
- real distribution shift,
- real mechanism ambiguity,
- real intervention choices,
- and at least some external truth constraints.

They are useful because they are small, structured worlds.
They are not the final intellectual destination.

## What kind of papers I want

I want algorithm papers centered on:

- posterior family inference under partial observation,
- active identification of hidden mechanistic state,
- uncertainty and ambiguity under OOD shift,
- low-rank feasible-state representations,
- and decision-aware measurement selection.

I do not want to build my identity around:

- release-curve incrementalism,
- benchmark chasing inside a narrow application silo,
- or stitched pipelines dressed up as new algorithms.

## Operational principle

For any future project, I should ask:

1. what is the central inference object?
2. what truth level is being tested?
3. what ambiguity is real and should not be collapsed away?
4. what new observation would change the belief state?
5. would this still be an interesting algorithmic problem if the application
   domain changed?

If the answer to 5 is no, the project is too trapped inside the domain.

## Part II - What Can Be Derived From The Current Project

The current codebase is not wasted.
It already contains several assets that can be reframed as algorithmic rather
 than purely application-specific:

- mechanism simulators,
- partial-observation forecasting,
- posterior inference over latent kinetics,
- active versus inactive directions,
- OOD and calibration diagnostics,
- active observation selection,
- and cross-decoder extensibility.

The key move is to extract the algorithmic object from the release framing.

## Paper A - Feasible Posterior Families Under Partial Observation

## Working title

Feasible posterior family inference for partially observed mechanistic
 dynamical systems

## Core object

Infer a low-rank posterior family over latent mechanistic states, rather than a
 single point estimate, from sparse partial observations.

## Why this is real algorithmically

This is not a generic stack of modules if the paper is centered on one object:

> a low-rank feasible posterior family that separates identifiable active
> directions from weakly constrained inactive directions.

That object can be formalized, constructed, updated, and evaluated.

## What from the current project supports it

- FIB-CASP
- active-set and regime analyses
- feasible-family language already present in manuscript materials
- calibration and OOD audits
- multiple decoders already present

## What the contribution would be

1. formalize why point estimates are wrong objects under partial observation,
2. define active versus inactive latent directions,
3. construct a low-rank feasible posterior family,
4. evaluate not only prediction but uncertainty truthfulness under OOD.

## Why it is not just a release paper

Drug release becomes one example of a broader problem:

- sparse observation,
- latent mechanistic ambiguity,
- physically decodable state family,
- and uncertainty-sensitive forecasting.

## Minimum experiments needed

- point estimate vs feasible family
- full-rank vs low-rank family
- no simulator constraint vs simulator-constrained family
- calibration under random and grouped OOD splits
- one non-PLGA decoder to show method portability

## Main risk

If the paper is written as "RF + simulator + Fisher + conformal," it collapses
 back into pipeline territory.

The method must be organized around the posterior-family object, not around the
 implementation stack.

## Paper B - Active State Identification by Sparse Measurement

## Working title

Active state identification in partially observed mechanistic systems

## Core object

Select the next observation to maximally reduce ambiguity over latent
 mechanistic state under measurement cost.

## Why this is real algorithmically

The paper is not "active release prediction."
It is about belief-state reduction:

- prior over latent states,
- observation-conditioned posterior update,
- utility for measurement choice,
- and decision under experimental budget.

The central object is:

> a utility-guided posterior update policy over hidden dynamical state.

## What from the current project supports it

- Route 70 active observer logic
- posterior shrinkage idea
- sparse observation schedules
- conformal uncertainty for decision support
- grouped evaluation already partially in place

## What the contribution would be

1. define sparse observation as a state-identification problem,
2. compare active measurements to fixed schedules and zero-observation priors,
3. evaluate reduction in ambiguity, not just point-RMSE,
4. connect measurement choice to interpretable temporal information windows.

## Why it matters for the larger world-model agenda

This paper studies a core world-model capability:

> how a model should decide what to observe next when it knows that its current
> state belief is incomplete.

That is much closer to scientific intelligence than passive prediction.

## Minimum experiments needed

- prior-only vs fixed-time vs active selection
- posterior spread reduction
- calibration after update
- ablation on utility definition
- behavior under grouped OOD evaluation

## Main risk

If framed as a point-prediction contest, the idea becomes smaller than it is.
The paper must foreground belief refinement and measurement utility.

## Paper C - Truth Benchmarks For Scientific World Models

## Working title

Truth-aware evaluation for world models in scientific dynamical systems

## Core object

A benchmark and evaluation taxonomy that separates observation truth, state
 truth, mechanistic truth, and epistemic truth.

## Why this is real algorithmically

Most existing world-model benchmarks overemphasize visual realism, short-horizon
 rollout quality, or task success.

This paper would ask a harder question:

> when does a model look right while representing the world wrongly?

That is an evaluation contribution, not merely a dataset note.

## What from the current project supports it

- simulator-based latent-state worlds
- known non-identifiability structure
- partial observations
- feasible families rather than unique latent targets
- prediction, calibration, and active-observation modules

## What the contribution would be

1. define the four truth levels formally,
2. build quantitative metrics for each where possible,
3. show concrete failure modes where observation fit is good but latent or
   epistemic truth is bad,
4. position scientific micro-worlds as a complementary benchmark family for
   world-model research.

## Why it matters

If the long-term goal is truth-aware world models, then better evaluation is
 not optional.
This paper would help define what "world understanding" should mean beyond
 surface prediction.

## Minimum experiments needed

- models with similar observational accuracy but different latent truthfulness
- controlled ambiguity cases
- intervention or counterfactual checks where possible
- uncertainty honesty metrics
- at least one active-observation example showing truth can improve with
  better evidence

## Main risk

This paper needs very clean definitions.
If the taxonomy is vague, it becomes philosophy instead of algorithmic
 evaluation.

## Part III - Priority Order

If I want the cleanest path from the current codebase to an algorithmic
 identity, the order should be:

1. Paper A: Feasible Posterior Families
2. Paper B: Active State Identification
3. Paper C: Truth Benchmark for Scientific World Models

Why:

- Paper A gives the cleanest core inference object.
- Paper B extends that object into decision-making.
- Paper C expands the agenda into a broader world-model program.

## Part IV - What To Stop Doing

If the goal is this research direction, then I should stop spending serious
 energy on:

- tiny point-metric improvements on release benchmarks,
- model-zoo inflation without a central method object,
- domain-specific framing that shrinks the generality of the question,
- and claims of algorithmic novelty when the work is still only a pipeline.

## Final reminder

The current project is not useless.

Its value is that it already contains the skeleton of a better research
 direction:

- latent state,
- partial observation,
- non-identifiability,
- uncertainty,
- active measurement,
- and simulator-constrained belief update.

The task now is not to abandon everything.
The task is to rename the object correctly, formalize it, and stop letting the
 application domain define the ceiling.
