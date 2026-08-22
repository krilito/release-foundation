# 29/30 — Minimal Active Set + Local Effective Dimension

Author: Codex
Date: 2026-05-24
Status: Implementation plan, approved to execute

---

## 0. Why this exists

We now have two facts at the same time:

1. The current 9-parameter PLGA ODE can fit most full curves very well.
2. The inferred parameter vectors are not obviously unique.

Those facts are not contradictory. They point to a more precise question:

> Does one curve really need all 9 parameters, or does it only activate a
> smaller number of effective mechanism directions?

This plan turns that question into two diagnostics.

---

## 1. Core hypothesis

For many PLGA curves, especially sparse or shape-simple curves, the observed
release profile may only identify a **small active subset** of the 9
parameters, or equivalently a **low local effective dimension** around the
best-fit solution.

If true, then:

- per-curve point estimates of all 9 parameters are over-specific,
- many "different" theta vectors are really moving along sloppy directions,
- a feasibility filter should eventually act on **which directions are active**
  and **which families are admissible**, not only on raw parameter values.

---

## 2. Experiment A — Minimal Active Set Audit

### Question

How many parameters must deviate from a dataset-level reference mechanism to
recover a given curve?

### Important design choice

We must **not** fix inactive parameters to that curve's own 9-D full-fit
solution. If we did, every subset would trivially recover the full fit.

Instead:

- define a dataset-level reference parameter vector `theta_ref`
  (median oracle theta over the dataset),
- for each curve, allow only `k` active parameters to move,
- hold all other parameters fixed at `theta_ref`,
- ask how much fit quality can be recovered with `k = 1, 2, 3, 4`.

### Optimization strategy

Exhaustive search over all subsets is too expensive. We therefore use a
**greedy forward selection** diagnostic:

1. start from `theta_ref`,
2. at step `k`, try adding each remaining parameter,
3. optimize only the active subset,
4. keep the parameter that gives the best fit improvement,
5. repeat until `k_max = 4`.

This is not the mathematically exact best subset. It is a practical lower-cost
diagnostic of whether a curve behaves like a low-active-DOF system.

### Output

Per curve:

- full 9-D oracle fit quality,
- reference-only fit quality (`k=0`),
- best greedy subsets at `k=1..4`,
- smallest `k` that gets within `ΔR² <= 0.02` of the full fit.

Dataset summary:

- fraction of curves explained by `<=1`, `<=2`, `<=3`, `<=4` active params,
- which parameters are most often selected first / second / third,
- comparison between internal 181 and cross-DOI 321.

### Interpretation

- if many curves reach near-full-fit by `k=3` or `k=4`, the system is
  effectively low-dimensional per curve,
- if most curves still need more than 4 active parameters, the full 9-D family
  is not just decorative.

---

## 3. Experiment B — Local Effective Dimension Audit

### Question

Near the full-fit solution, how many local parameter directions actually change
the curve in a meaningful way?

### Method

At the full-fit theta for each curve:

1. compute the Jacobian of `Q(t_obs)` with respect to theta,
2. scale theta directions by prior range so units are comparable,
3. run SVD on the scaled Jacobian,
4. inspect singular value decay.

### Metrics

Per curve:

- `dim95`: smallest number of singular directions explaining 95% of local
  sensitivity energy,
- `dim99`: same for 99%,
- count of singular values above 5% / 1% of the top singular value,
- condition number summary.

Dataset summary:

- median / p25 / p75 of `dim95`,
- distribution of local effective dimension across internal 181 vs 321.

### Interpretation

- if `dim95` is often around `3-4`, then the curve locally only "sees" a small
  number of mechanism directions even though the global model has 9 params,
- if `dim95` stays near `7-9`, then the identifiability problem is not mainly
  "too many irrelevant parameters".

---

## 4. Why these two together

Experiment A asks a **global practical question**:

> How many parameters do I need to move away from a canonical reference to
> explain this curve?

Experiment B asks a **local geometric question**:

> Around the best fit, how many directions actually matter?

If both say "about 3-4", that is strong evidence that per-curve effective
dimension is lower than 9.

If A says "3-4" but B says "7-9", then a small active subset can recover the
curve globally, but the local basin is still high-dimensional.

If A says "7-9" but B says "3-4", then local sensitivity is low-dimensional
but the chosen global reference `theta_ref` is too crude.

---

## 5. Scope

Both experiments should cover:

- internal 181 PLGA benchmark,
- matched high-quality 321 cross-DOI PLGA set.

This matters because internal and external curves may have different active
mechanism patterns.

---

## 6. Outputs

### Script 29

`scripts/29_minimal_active_set_audit.py`

Produces:

- `outputs/29_minimal_active_set_audit/full_fit_bank.csv`
- `outputs/29_minimal_active_set_audit/per_curve_active_set.csv`
- `outputs/29_minimal_active_set_audit/dataset_summary.csv`
- `outputs/29_minimal_active_set_audit/summary.txt`

### Script 30

`scripts/30_local_effective_dimension_audit.py`

Produces:

- `outputs/30_local_effective_dimension_audit/per_curve_dimension.csv`
- `outputs/30_local_effective_dimension_audit/dataset_summary.csv`
- `outputs/30_local_effective_dimension_audit/summary.txt`

---

## 7. Decision use

These are diagnostics, not end-user models.

If they show strong low-dimensional structure, the next step is not "predict
all 9 theta better". The next step is to redesign the feasibility filter
around:

- active mechanism directions,
- low-dimensional admissible families,
- simpler explanations preferred by Occam's razor.
