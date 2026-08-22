---
from_browser_search: true
date: 2026-06-03
scope: plga_release_prediction
---

# Algorithm shortlist for our PLGA release setting

## Why this note exists

The question is not "what is a strong generic curve model?".
The question is "what algorithm family is actually matched to the current
failure mode of this repo?"

Current repo evidence says the real constraints are:

- sparse trajectories and sparse formulations
- cumulative release `Q(t)` must be monotone and bounded
- leakage-sensitive grouped evaluation across DOI / source
- the key unresolved weakness is **cross-DOI tail robustness**, not median
  in-domain fit
- direct tabular trees are still the best median cross-DOI baseline, while
  SBI earns its keep mainly on the bad tail

Relevant local anchors:

- [ARCHITECTURE.md](D:/release-foundation/ARCHITECTURE.md:8)
- [SPRINT1_STATUS_NOTE.md](D:/release-foundation/SPRINT1_STATUS_NOTE.md:203)
- [DECISIONS.md](D:/release-foundation/DECISIONS.md:1624)
- [DECISIONS.md](D:/release-foundation/DECISIONS.md:1627)
- [DECISIONS.md](D:/release-foundation/DECISIONS.md:1928)

## External sources checked

- Bannigan et al., *Machine learning models to accelerate the design of
  polymeric long-acting injectables*, Nature Communications 2023.
  Public benchmark identity for LAI release prediction; tree models are a
  serious baseline, not a straw man.
  Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9832011/>

- Kapoor and Narayanan, *Leakage and the reproducibility crisis in
  machine-learning-based science*, Patterns 2023.
  Use this framing for any "non-independent train/test split" argument.
  Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/>

- Wehenkel and Louppe, *Unconstrained Monotonic Neural Networks*, NeurIPS
  2019.
  Source: <https://papers.nips.cc/paper_files/paper/2019/hash/2a084e55c87b1ebcdaad1f62fdbbac8e-Abstract.html>

- You et al., *Deep Lattice Networks and Partial Monotonic Functions*,
  NeurIPS 2017.
  Source: <https://papers.nips.cc/paper/6891-deep-lattice-networks-and-partial-monotonic-functions>

- Riihimaki and Vehtari, *Gaussian processes with monotonicity information*,
  AISTATS 2010.
  Source: <https://proceedings.mlr.press/v9/riihimaki10a.html>

- Garnelo et al., *Conditional Neural Processes*, ICML 2018.
  Source: <https://proceedings.mlr.press/v80/garnelo18a.html>

- Petersen et al., *GP-ConvCNP: Better generalization for conditional
  convolutional Neural Processes on time series data*, AISTATS 2021.
  Source: <https://proceedings.mlr.press/v161/petersen21a.html>

- Kidger et al., *Neural Controlled Differential Equations for Irregular Time
  Series*, NeurIPS 2020.
  Source: <https://papers.nips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html>

- Rackauckas et al., *Universal Differential Equations for Scientific Machine
  Learning*, 2020.
  Source: <https://arxiv.org/abs/2001.04385>

- Greven et al., *Functional Linear Mixed Models for Irregularly or Sparsely
  Sampled Data*, 2015.
  Source: <https://arxiv.org/abs/1508.01686>

## Ranked shortlist

## 1. Monotonic direct-curve models: `Deep Lattice` or `UMNN`

### Best fit when

- we want a direct `descriptor + time -> Q(t)` model
- we care about monotonicity and boundedness without post-hoc projection
- we want a reviewer-friendly replacement for hand-made isotonic cleanup

### Why it matches us

This is the cleanest answer to the current "cumulative release must never go
down" requirement. It solves a real structural issue in direct-Q models
without pretending to solve every generalization problem at once.

### What it will and will not fix

- likely fixes curve legality and reduces ugly post-processing
- gives a fairer direct-Q benchmark than unconstrained MLP/RNN variants
- does **not** automatically solve DOI shift or the cross-DOI bad tail

### Recommendation

This is the best **near-line experiment** if we want something stronger than
`isotonic/PAVA` but much cheaper than a full architecture rewrite.

## 2. `GP-ConvCNP` / `Sequential Neural Process`

### Best fit when

- the main deployment object is "a few early release points -> the rest of
  the curve"
- we want a principled few-shot curve continuation model
- we want uncertainty by design rather than as a bolt-on

### Why it matches us

Your repo already leans toward few-shot / prefix-aware release forecasting in
multiple places. CNP-style models are designed for exactly that regime:
observe a small context set, predict a full target function.

### What it will and will not fix

- much better matched than kNN for sparse-context curve continuation
- handles irregular observation sets naturally
- still needs an explicit monotonicity story if the output is cumulative
- less aligned with the pure descriptor-only cross-DOI story

### Recommendation

This is the best **next-generation model family** if the project chooses to
center the paper on few-shot release continuation rather than descriptor-only
zero-shot prediction.

## 3. `Neural CDE`

### Best fit when

- the hard part is irregular sampling times, varying prefix length, and masked
  observations
- we want a stronger encoder for early observed release history

### Why it matches us

Neural CDEs are purpose-built for irregular time series. That is more on-point
than trying to force a standard GRU/LSTM to act continuous-time.

### What it will and will not fix

- better time encoding than naive sequence models for irregular release data
- useful as a prefix/history encoder inside a larger model
- does not by itself encode monotonic cumulative output
- heavier than the monotonic direct-Q option

### Recommendation

Use this if the immediate next target is **better prefix encoding**, not as the
first thing to try for a descriptor-only benchmark.

## 4. Study-aware functional mixed models / sparse FDA

### Best fit when

- we want a strong, honest statistical baseline under sparse curves
- we want to model source / DOI grouping explicitly
- we care more about trustworthy grouped generalization than neural novelty

### Why it matches us

This family is unusually honest for our data geometry:
small `n`, sparse curves, grouped source effects, and no fantasy of giant-data
deep learning.

### What it will and will not fix

- gives a good anti-hype baseline for grouped sparse curves
- may absorb some DOI-specific variation better than a naive pooled model
- probably not the top raw-performance winner
- not the clearest route to a "foundation model" story

### Recommendation

This is the best **reviewer-defense baseline** to add if we want to show we
did not ignore the sparse-functional-statistics literature.

## 5. `UDE` / residual neural ODE

### Best fit when

- we want to keep the PLGA simulator but let ML learn the missing part
- we are ready for a real mechanism-plus-ML upgrade

### Why it matches us

This is the most principled way to upgrade MEP-like ideas without collapsing
back into a pure black box. It respects the repo's architecture direction much
better than another Weibull bridge.

### What it will and will not fix

- strongest long-term alignment with the repo north star
- can target hard tails as residual misspecification rather than replacing the
  whole simulator
- much more expensive scientifically and implementation-wise
- too heavy for the very next benchmarking step

### Recommendation

This is a **Phase 2 method move**, not the first response to the current
cross-DOI benchmark gap.

## What not to spend much more time on

- `kNN` curve retrieval as a main answer to sparse-context generalization
- new `Weibull` variants presented as if they were a new model family
- `isotonic/PAVA` presented as a method contribution rather than a cheap shape
  repair baseline
- `fPCA` as the main route for cross-DOI deployment; local repo evidence
  already says it degrades badly on cross-DOI

## Concrete decision rule

If the project asks:

- "What is the cheapest serious upgrade to direct-Q?"
  Pick `Deep Lattice` or `UMNN`.

- "What is the right model class for early-point -> future-curve forecasting?"
  Pick `GP-ConvCNP` first, `Neural CDE` second.

- "What is the most principled mechanism+ML upgrade?"
  Pick `UDE`.

- "What statistical baseline proves we respected sparse grouped curve theory?"
  Add a functional mixed-effects baseline.

## My actual recommendation for this repo

The highest-value order is:

1. Add **one monotonic direct-Q baseline**:
   `Deep Lattice` preferred for simplicity and reviewer readability.
2. If the team wants a genuinely new model line, test **one CNP-family
   few-shot curve model** on prefix forecasting.
3. Keep `UDE` as the serious mechanistic upgrade, not as a knee-jerk response
   to the current benchmark gap.

That ordering matches the current repo evidence:

- direct trees remain the real practical benchmark
- the unresolved scientific problem is cross-DOI tail robustness
- the current project should not jump to the heaviest possible model before
  installing the right constrained baselines
