---
date: 2026-06-03
scope: algorithm_to_repo_mapping
depends_on:
  - docs/algorithm_shortlist_for_plga_release_2026-06-03.md
---

# Algorithm to repo mapping for our PLGA release setting

This note answers a narrower question than the shortlist:

```text
If we actually pick one of those algorithm families, where does it plug into
this repo, and which existing scripts should be treated as the landing zone?
```

## Current code lanes that matter

## Lane A — direct benchmark / canonical split

Primary entry:

- [scripts/72_canonical_benchmark_v2.py](D:/release-foundation/scripts/72_canonical_benchmark_v2.py:1)

What it is:

- fixed canonical split
- direct predictor vs active posterior benchmark
- paper-grade shared benchmark table

Meaning:

- if we want a new **direct-Q baseline** that reviewers can compare against
  ExtraTrees / DirectQ on the same mouth, this is the cleanest insertion point

## Lane B — few-shot / sparse-prefix forecasting

Primary entries:

- [scripts/23_prefix_world_model_benchmark.py](D:/release-foundation/scripts/23_prefix_world_model_benchmark.py:1)
- [scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py:1)

What they are:

- `23`: the concrete PLGA prefix benchmark on external 321
- `80`: the mechanism-agnostic benchmark API for partial-observation release
  forecasting

Meaning:

- if we want an algorithm for `few early points -> future curve`, this is the
  correct lane

## Lane C — world-model / richer sequence route

Primary entries:

- [scripts/26_iterative_world_model.py](D:/release-foundation/scripts/26_iterative_world_model.py:1)
- [docs/refactor_27_npe.md](D:/release-foundation/docs/refactor_27_npe.md:1)

What they are:

- `26`: the heavy learned latent/prefix route that already struggled
- `27` refactor note: a cleaner amortized-SBI-on-prefix alternative

Meaning:

- if we add sequence models, they should probably *replace or simplify* this
  lane, not stack on top of it

## Lane D — functional baseline lane

Primary entry:

- [scripts/38b_canonical_functional_baselines.py](D:/release-foundation/scripts/38b_canonical_functional_baselines.py:1)

What it is:

- current "functional baseline" slot on the canonical split
- presently pragmatic fPCA-Ridge and GlobalMean, not a serious sparse FDA
  mixed-effects baseline

Meaning:

- if we want to show we respected sparse-functional-statistics literature, this
  is where that work belongs

## Lane E — mechanism-plus-residual route

Primary entry:

- [docs/plan_27d_residual_neural_ode.md](D:/release-foundation/docs/plan_27d_residual_neural_ode.md:1)

What it is:

- an already-articulated plan for a low-capacity residual on top of the PLGA
  ODE, explicitly rejecting free latent ODE replacement

Meaning:

- if we pick UDE / physics-residual ML, do not invent a new branch story; the
  repo already has the right framing and kill criteria

## Best-fit mapping by algorithm family

## 1. `Deep Lattice` / `UMNN`

### Best repo landing zone

- primary: `scripts/72_canonical_benchmark_v2.py`
- secondary: `scripts/80_release_partial_observation_benchmark.py`

### Why this is the right landing zone

This family is a **direct-Q constrained baseline**. It belongs in the shared
benchmark lane, not the world-model lane.

### Minimal implementation idea

Add a new method equivalent to:

```text
MonotonicDirectQ-fixed
inputs  = [x, early observations if allowed, t_query]
output  = cumulative release Q(t_query)
constraint = monotone in t_query
```

### What not to do

- do not bury it inside `26_iterative_world_model.py`
- do not present it as a mechanism-aware model
- do not let it depend on post-hoc isotonic repair if the whole point is to
  test native monotonicity

### Effort / risk

- effort: low to medium
- scientific upside: high as a clean benchmark
- code risk: low

### My recommendation

If we run exactly one new baseline in the next short cycle, this should be it.

## 2. `GP-ConvCNP` / `Sequential Neural Process`

### Best repo landing zone

- primary: `scripts/23_prefix_world_model_benchmark.py`
- benchmark wrapper: `scripts/80_release_partial_observation_benchmark.py`

### Why this is the right landing zone

This family is for **partial observation / sparse prefix continuation**. It is
not a replacement for the descriptor-only canonical benchmark.

### Minimal implementation idea

Replace the current heuristic update object with:

```text
context   = {(t_i, Q_i)} early observed points + formulation features
target    = future Q(t) on query times
head      = predictive mean + uncertainty
```

The benchmark should still be reported on the same 1d / 3d / 7d suffix task.

### What not to do

- do not call it a "world model"
- do not force it through theta unless the experiment specifically asks
  whether prefix information should update a mechanistic posterior
- do not expect this family to be the clean answer to descriptor-only
  zero-shot cross-DOI

### Effort / risk

- effort: medium
- scientific upside: high if the paper leans few-shot
- code risk: medium

### My recommendation

This is the best candidate if the paper narrative becomes
`sparse-prefix OOD forecasting`, not `descriptor-only release regression`.

## 3. `Neural CDE`

### Best repo landing zone

- primary: `scripts/26_iterative_world_model.py`
- cleaner alternative: the `27` prefix-posterior lane

### Why this is the right landing zone

Neural CDE is mostly an **encoder choice for irregular observations**. It is a
better substitute for GRU/Transformer prefix encoders than a standalone
benchmark family.

### Minimal implementation idea

Swap the observed-token encoder with a continuous-time encoder:

```text
observed prefix -> Neural CDE encoder -> latent state
latent state -> posterior over theta or direct future-curve decoder
```

### What not to do

- do not first add it to the direct-Q lane
- do not use it as an excuse to resurrect a huge latent-dynamics stack before
  proving the benchmark value

### Effort / risk

- effort: medium to high
- scientific upside: medium
- code risk: medium to high

### My recommendation

Only do this after the team decides that the problem is truly the prefix
encoder, not the target object or the evaluation framing.

## 4. Functional mixed-effects / sparse FDA baseline

### Best repo landing zone

- primary: `scripts/38b_canonical_functional_baselines.py`

### Why this is the right landing zone

The repo already has a slot for "functional baselines", but the current B6 is
still a pragmatic PCA-regression baseline, not a serious grouped sparse FDA
competitor.

### Minimal implementation idea

Add one honest grouped functional baseline:

```text
curve(t) = population mean(t)
         + fixed effects of formulation descriptors
         + random study / DOI effect(t) or grouped curve effect
         + residual
```

This can be a reference implementation in R or Python if needed; it does not
need to be glamorous.

### What not to do

- do not oversell it as the future of the repo
- do not expect it to be the strongest predictive method
- do not mix its role with the mechanistic route

### Effort / risk

- effort: medium
- scientific upside: medium as a defense baseline
- code risk: low to medium

### My recommendation

If reviewers are likely to ask "did you compare to sparse functional
statistics?", this is the right defensive addition.

## 5. `UDE` / physics-residual neural ODE

### Best repo landing zone

- primary: `docs/plan_27d_residual_neural_ode.md`
- implementation target: new simulator subclass + new `27d` scripts

### Why this is the right landing zone

The repo already has the correct conceptual structure:
keep the PLGA ODE, add a small residual, and reject free latent replacement.

### Minimal implementation idea

Follow the existing `27d` plan:

- keep PLGA parametric ODE dominant
- add low-capacity residual on `dQ/dt`
- infer residual coefficients jointly with theta
- evaluate first on the hard tail / boundary-pinned subset

### What not to do

- do not restart from a free neural residual
- do not bypass the existing monotonicity and `Q_max` safety logic
- do not treat this as the next benchmark tweak; it is a method program

### Effort / risk

- effort: high
- scientific upside: high
- code risk: high

### My recommendation

This is the strongest serious research move, but not the cheapest next
experiment.

## If we had to choose by horizon

## Next 1-2 weeks

Do:

- add a **monotonic direct-Q baseline** in the `72` benchmark lane

Maybe:

- add a more serious functional baseline in `38b`

Avoid:

- jumping straight to UDE
- reviving the heavy world-model lane first

## Next 2-6 weeks

Do:

- test one **CNP-family sparse-prefix forecaster** on the `23/80` lane

Maybe:

- test a cleaner prefix-posterior route from the `27` refactor line

Avoid:

- mixing direct-Q and few-shot prefix claims in the same first experiment

## Bigger method cycle

Do:

- pursue `27d` physics-residual UDE only if the simpler constrained baselines
  still leave a real unexplained cross-DOI tail gap

## My concrete call

If I were choosing the order for this repo today:

1. `Deep Lattice` baseline on `72`
2. `GP-ConvCNP` few-shot experiment on `23/80`
3. grouped sparse-FDA baseline on `38b`
4. `27d` residual-UDE only after the above

That order respects the current evidence:

- tree baselines are still real competitors
- the deployable mode is increasingly few-shot / sparse-prefix
- world-model rhetoric is already considered too heavy in several local notes
- the mechanistic residual route is promising, but expensive enough that it
  should be justified by a remaining benchmark gap rather than curiosity alone

## Browser note

I attempted to use the in-app Browser (`@chrome` path) for this search via the
Browser plugin workflow, but the browser runtime failed twice during setup with
an immediate Node/browser session exit on this machine. The algorithm choices
above therefore rely on verified web search results and local repo evidence,
not on a live in-app browser session.
