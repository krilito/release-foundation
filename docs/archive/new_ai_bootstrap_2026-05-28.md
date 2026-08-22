# New AI Bootstrap - 2026-05-28

Use this document to start a new AI session from zero without losing project
 context and without polluting the current working tree.

## Project location

Primary repository:

`D:\release-foundation`

## Non-negotiable workspace rule

Do not work directly on the current dirty working tree.

Before doing anything substantial, create either:

1. a new git branch plus a separate worktree, or
2. a clean copied working folder if worktree is unavailable.

Preferred option:

```powershell
cd D:\release-foundation
git worktree add ..\release-foundation-worldmodel -b codex/worldmodel-bootstrap HEAD
cd ..\release-foundation-worldmodel
```

Minimum acceptable option:

```powershell
cd D:\release-foundation
git switch -c codex/worldmodel-bootstrap
```

Reason:

- the current repo already has local modifications,
- the active thread is still exploratory,
- and new work should not silently contaminate the existing branch or files.

## Files to read first

Read these in order:

1. `D:\release-foundation\CLAUDE.md`
2. `D:\release-foundation\AGENTS.md`
3. `D:\release-foundation\codex.md`
4. `D:\release-foundation\docs\research_nonnegotiables_2026-05-28.md`
5. `D:\release-foundation\docs\world_model_research_direction_2026-05-28.md`
6. `D:\release-foundation\docs\plan_70_active_kinetic_observer_route.md`

Optional but useful context:

- `D:\release-foundation\docs\nature_family_manuscript_skeleton_2026-05-27.md`
- `D:\release-foundation\docs\paper_closeout_release_quality_paradigm_2026-05-26.md`

## Current strategic direction

The project should no longer be treated as "just a drug release prediction
 project."

The intended direction is:

> algorithm-first research on truth-aware world models for partially observed
> mechanistic dynamical systems

Domain examples such as PLGA, chitosan, liposomes, and hydrogels are testbeds.
They are not the final intellectual target.

## What the user actually wants

The user is algorithm-first and strongly dislikes two things:

1. the idea that wet-lab automatically makes work superior,
2. ordinary model stitching being mislabeled as a novel algorithm.

The new AI should therefore push toward:

- a real core inference object,
- partial-observation latent-state inference,
- identifiable versus feasible structure,
- uncertainty and OOD robustness,
- active measurement / belief update,
- and world-model framing grounded in reality rather than application rhetoric.

Do not drift back into release-paper incrementalism.

## The most promising derived directions from the current project

These are the three best routes already identified:

1. `Feasible Posterior Families Under Partial Observation`
2. `Active State Identification by Sparse Measurement`
3. `Truth-Aware Evaluation for Scientific World Models`

The immediate best route is:

`Feasible Posterior Families Under Partial Observation`

because it has the cleanest algorithmic core object and uses the most existing
 project assets.

## Existing assets in the repository

The current repo already contains useful ingredients:

- mechanism simulators,
- partial-observation forecasting logic,
- latent-state / kinetic posterior work,
- active and inactive direction analyses,
- calibration and OOD diagnostics,
- active observation ideas,
- and multiple decoder families.

These are not worthless.
The main task is to extract the algorithmic object and stop letting the release
 application define the ceiling.

## What not to do

Do not spend time on:

- tiny point-metric improvements on release benchmarks,
- expanding a model zoo without a core method object,
- writing the work as a JCR-style release paper,
- or presenting a workflow stack as if it were a new algorithm.

## Immediate mission for the new AI

Start by turning the current direction into a clean algorithm project proposal.

Specifically:

1. inspect the current code and documents,
2. identify the single central inference object for the best route,
3. write a crisp one-page proposal for
   `Feasible Posterior Families Under Partial Observation`,
4. define:
   - problem statement,
   - core object,
   - proposed method,
   - evaluation protocol,
   - why it is not merely a stitched pipeline,
5. only then propose code changes or new experiments.

## Suggested first concrete deliverable

Create a new markdown note in the clean branch/worktree with a title like:

`docs\paper_a_feasible_posterior_family_outline.md`

This note should contain:

- one-sentence problem definition,
- one-sentence method definition,
- contribution bullets,
- experiment table,
- and a section called `Why this is an algorithmic paper rather than a release paper`.

## Suggested startup prompt for the new AI

You are starting fresh in `D:\release-foundation`.

Before doing any work:

- create a new branch and preferably a new git worktree so you do not pollute
  the current dirty working tree,
- then read `CLAUDE.md`, `AGENTS.md`, `codex.md`,
  `docs\research_nonnegotiables_2026-05-28.md`, and
  `docs\world_model_research_direction_2026-05-28.md`.

Your job is not to continue writing a release-paper story.
Your job is to extract the cleanest algorithmic project that can grow from the
 current repository.

Prioritize the route:

`Feasible Posterior Families Under Partial Observation`

Treat PLGA / chitosan / liposome / hydrogel as scientific testbeds, not as the
 identity of the method.

Do not call a stitched pipeline a new algorithm unless you can identify one
 central inference object that governs the method.

Your first deliverable is a one-page paper outline defining:

- the formal problem,
- the core inference object,
- the method contribution,
- the evaluation plan,
- and the argument for why the work is algorithmic rather than merely applied.
