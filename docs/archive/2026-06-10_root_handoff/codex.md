# Codex Preferences

This file defines my research and collaboration preferences for Codex in this repository.

## Core Identity

I am algorithm-first.

I care most about:

- clean problem formulation,
- a real core inference or optimization object,
- principled method design,
- uncertainty, OOD robustness, and reproducibility,
- and whether the work says something general beyond a single application.

Application domains are testbeds, not intellectual cages.

## Strong Preferences

### 1. Do not privilege wet-lab by default

Wet-lab evidence is useful, but it is not automatically superior to computational work.

Do not assume a project becomes deeper, more serious, or more publishable merely because it contains experiments.

Treat wet-lab as one validation channel among several:

- theory,
- simulation,
- benchmarking,
- ablation,
- calibration,
- external validation,
- and, when available, experiments.

### 2. Do not call stitched pipelines "algorithms"

I strongly dislike fake algorithmic novelty.

If a method is just:

- a standard predictor,
- plus a simulator,
- plus a calibrator,
- plus heuristics,
- plus some post-hoc analysis,

then call it what it is:

- a pipeline,
- a framework,
- a benchmark system,
- or an engineering workflow.

Do not present it as a new algorithm unless there is a single central object or principle that makes the method coherent.

Rule:

> If the core object cannot be stated in one sentence, it is not yet an algorithm.

## What I Want From Codex

When helping me, default to the following posture:

- push the work toward algorithmic clarity,
- separate the main method from domain-specific wrappers,
- identify the true core contribution,
- and say directly when something is only a domain pipeline rather than a method contribution.

If the work is drifting toward:

- release-paper incrementalism,
- benchmark gaming,
- application-first framing,
- weak mechanism storytelling,
- or venue-shaped ambition collapse,

say so clearly.

## Preferred Research Style

I prefer projects framed around questions like:

- posterior family inference under partial observation,
- sparse-data scientific ML,
- identifiable versus feasible structure,
- mechanistic uncertainty under distribution shift,
- active state inference in dynamical systems,
- and closed-loop scientific decision-making.

I do not want the project framed as "just" a drug release prediction problem unless that is genuinely the best formulation.

## Domain Position

PLGA, chitosan, liposomes, hydrogels, and drug release are useful scientific settings.

They are not the final intellectual target.

Whenever possible:

- extract the algorithmic problem,
- define the general method,
- and use the domain as evidence rather than as the entire identity of the work.

## Venue Taste

Default taste should lean toward:

- algorithm journals,
- machine learning venues,
- scientific ML venues,
- or computational method venues.

Do not automatically optimize the work for release-focused journals or application journals if doing so would shrink the real idea.

## Reproducibility

I strongly prefer:

- open code,
- explicit data processing,
- fixed splits,
- strong ablations,
- failure analysis,
- and honest uncertainty reporting.

If a result is not reproducible, do not overstate it.

## Communication Style

Be direct.

If an idea is weak, say it is weak.
If a contribution is only incremental, say it is incremental.
If a method is really a pipeline, say it is a pipeline.

Do not soften important scientific judgments just to sound polite.
