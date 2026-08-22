---
date: 2026-06-03
scope: implementation_options
depends_on:
  - docs/algorithm_shortlist_for_plga_release_2026-06-03.md
  - docs/algorithm_repo_mapping_for_plga_release_2026-06-03.md
---

# Current implementation options for the shortlisted algorithms

This note answers a practical follow-up:

```text
Which of the shortlisted algorithm families have usable current
implementations, and which are "paper only" for our purposes?
```

## Current repo stack

Current dependency fact from [pyproject.toml](D:/release-foundation/pyproject.toml:1):

- PyTorch-based core stack
- `sbi` already installed as the posterior/inference backbone
- `torchdiffeq` already installed
- no TensorFlow dependency in the current project file

This matters because:

- `UMNN` and `torchcde` are stack-aligned
- `TensorFlow Lattice` is scientifically attractive but ecosystem-misaligned

## 1. `Deep Lattice` / monotonic lattice models

## Current implementation status

This family is the most implementation-ready of the shortlist.

Primary sources checked:

- TensorFlow Lattice overview:
  <https://www.tensorflow.org/lattice>
- TensorFlow Lattice overview guide:
  <https://www.tensorflow.org/lattice/overview>
- TensorFlow Lattice GitHub:
  <https://github.com/tensorflow/lattice>

What exists now:

- maintained TensorFlow package
- official tutorials and API docs
- native monotonicity constraints
- calibrated piecewise-linear input layers
- lattice layers and deep lattice stacks

## Fit to our repo

### Good news

- this is the easiest serious route to a **native monotonic direct-Q**
  baseline
- it matches the direct benchmark lane well
- it avoids post-hoc isotonic cleanup being the main shape-enforcement method

### Bad news

- it is TensorFlow-first, while this repo is strongly PyTorch-centered
- so the first version will either:
  - live as a separate benchmark script, or
  - require reimplementing the core idea in PyTorch

## Practical verdict

```text
Best "ready now" constrained baseline, but not in our native framework.
```

## 2. `UMNN`

## Current implementation status

Primary sources checked:

- official paper:
  <https://arxiv.org/abs/1908.05164>
- official implementation:
  <https://github.com/AWehenkel/UMNN>
- PyPI package:
  <https://pypi.org/project/UMNN/1.65/>

What exists now:

- official author repo
- published Python package
- PyTorch implementation

## Fit to our repo

### Good news

- PyTorch-native path is better aligned with this repo than TensorFlow Lattice
- conceptually cleaner than hand-written isotonic projection
- a good candidate if we want monotonicity without switching ecosystems

### Bad news

- the official code is older and more research-code-like than TensorFlow
  Lattice
- the repo is built around monotonic transforms and flows, not a clean
  drop-in release-regression recipe

## Practical verdict

```text
Best PyTorch monotonic option, but higher adaptation cost than TensorFlow
Lattice.
```

If the question is "which monotonic approach matches our current stack better",
UMNN wins.

If the question is "which one looks easiest to prototype correctly", TensorFlow
Lattice wins.

## 3. `CNP` / `Neural Process` family

## Current implementation status

Primary sources checked:

- DeepMind neural-processes repo:
  <https://github.com/google-deepmind/neural-processes>
- Neural Process Family repo:
  <https://github.com/YannDubs/Neural-Process-Family>
- CNP paper:
  <https://proceedings.mlr.press/v80/garnelo18a.html>

What exists now:

- official DeepMind notebook implementations for CNP / NP / ANP
- a broader PyTorch/JAX-oriented Neural Process Family ecosystem

## Fit to our repo

### Good news

- this family is exactly matched to few-shot function prediction
- there is existing code to start from instead of inventing the model family
- uncertainty is built into the object, which matters for sparse-prefix use

### Bad news

- most official code is notebook/demo style
- direct support for our exact object
  (`formulation features + irregular prefix -> cumulative future curve`)
  still needs task-specific adaptation
- monotonic cumulative output is not free

## Practical verdict

```text
Algorithmically correct for sparse-prefix forecasting, but not "plug and
play".
```

This is a good second-wave experiment, not the cheapest immediate baseline.

## 4. `GP-ConvCNP`

## Current implementation status

Primary sources checked:

- paper:
  <https://proceedings.mlr.press/v161/petersen21a.html>
- arXiv:
  <https://arxiv.org/abs/2106.04967>

What exists now:

- strong primary paper
- searchable ecosystem overlap with Neural Process Family codebases

What is missing from the current check:

- I did not verify a single clearly maintained official standalone GP-ConvCNP
  repo with the same confidence as TensorFlow Lattice or torchcde

## Fit to our repo

### Good news

- among NP-family methods, this is especially attractive for time-series
  generalization
- it is closer to our sparse-prefix release task than plain kNN or generic RNN

### Bad news

- implementation certainty is lower than for TensorFlow Lattice / UMNN /
  torchcde
- likely requires more adaptation and more reading before use

## Practical verdict

```text
Scientifically very relevant, implementation path less turnkey.
```

## 5. `Neural CDE`

## Current implementation status

Primary sources checked:

- paper:
  <https://papers.nips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html>
- official library:
  <https://github.com/patrick-kidger/torchcde>

What exists now:

- official PyTorch library
- explicit irregular-time use case
- ecosystem compatibility with neural differential equation tooling

## Fit to our repo

### Good news

- this is the cleanest current implementation path for irregular-time prefix
  encoding
- fully aligned with PyTorch
- better engineering maturity than a one-off research code dump

### Bad news

- it solves the encoder problem, not the whole benchmark problem
- if we use it too early, we risk upgrading the wrong layer of the system

## Practical verdict

```text
Most implementation-ready option for irregular prefix encoding.
```

If we decide the next experiment is specifically "replace GRU/Transformer-style
prefix encoding with a continuous-time encoder", this is the best current tool.

## 6. `UDE` / residual neural ODE

## Current implementation status

Primary sources checked:

- UDE paper:
  <https://arxiv.org/abs/2001.04385>
- SciML UDE docs:
  <https://sciml.readthedocs.io/en/latest/UDE.html>
- DiffEqFlux docs:
  <https://docs.sciml.ai/DiffEqFlux/>

What exists now:

- mature Julia/SciML ecosystem
- official docs and examples
- strong differentiable ODE tooling

## Fit to our repo

### Good news

- conceptually the strongest implementation ecosystem for UDEs is real and
  mature
- the repo already has a good local plan for a low-capacity residual route

### Bad news

- the strongest official ecosystem is Julia-first, not PyTorch-first
- moving to SciML would be a bigger workflow choice, not a small benchmark add
- a PyTorch-only residual-ODE route is possible, but it is more custom

## Practical verdict

```text
Real ecosystem exists, but it is not the easiest drop-in for this repo.
```

That makes UDE a serious method program, not the next quick experiment.

## 7. Monotonic GP

## Current implementation status

Primary sources checked:

- classic paper:
  <https://proceedings.mlr.press/v9/riihimaki10a.html>

What exists now:

- strong theory anchor
- newer monotonic-GP research lines exist

What is missing from the current check:

- I did not verify one obvious, current, official, maintained implementation
  path that looks simpler than the alternatives above for our exact use case

## Fit to our repo

### Good news

- statistically elegant
- attractive for uncertainty and monotonicity together

### Bad news

- weaker immediate implementation path than TFL / UMNN / torchcde
- not the cheapest way to get a competitive first result

## Practical verdict

```text
Interesting research option, weak immediate engineering option.
```

## Immediate takeaway

If we classify the current options by "can we realistically prototype this
soon in this repo?", the order is:

1. `Neural CDE` for irregular prefix encoding
2. `UMNN` for PyTorch-native monotonic direct-Q
3. `TensorFlow Lattice` for easiest monotonic benchmark in a separate lane
4. `CNP` / `GP-ConvCNP` for sparse-prefix forecasting, but with more task
   adaptation
5. `UDE` for a larger method cycle
6. monotonic GP as a lower-priority research curiosity

If we classify by "best algorithmic fit to the scientific problem", the order
is different:

1. monotonic direct-Q (`Deep Lattice` / `UMNN`) for the direct benchmark lane
2. `GP-ConvCNP` / CNP-family for sparse-prefix forecasting
3. `Neural CDE` as an encoder upgrade once the task framing is fixed
4. `UDE` for the serious mechanism-plus-ML route

## My concrete recommendation

If the next step must be **runnable soon in our current stack**:

- try `UMNN` before TensorFlow Lattice
- try `torchcde` if we decide the next experiment is a prefix encoder study

If the next step must be **the cleanest scientific benchmark move**:

- add one monotonic direct-Q benchmark first, even if it means temporarily
  stepping outside the main PyTorch stack

## Browser note

I again used primary web sources for this check instead of a live in-app
Browser session because the Browser runtime on this machine previously failed
during setup. The sources above were chosen to maximize official docs / official
repos rather than secondary commentary.
