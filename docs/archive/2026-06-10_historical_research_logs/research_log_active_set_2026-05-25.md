## Research Log - Active-set / effective-dimension branch

Date: `2026-05-25`

### Goal of this branch

Clarify whether the current PLGA 9-parameter mechanism family is:

- globally necessary,
- but locally over-complete for individual curves.

The deeper question is whether each curve really identifies all 9 parameters, or only a smaller active subset / lower-dimensional slice.

### Work completed

#### 1. `27a` review

Reviewed `scripts/27a_ode_expressivity_audit.py` and checked current outputs.

Main judgment:

- The full-curve fit numbers are numerically credible.
- The present verdict text is too strong.

Verified:

- `per_curve.csv` has `259` rows and `259` unique `fid`
- sampled `RMSE` / `R^2` values match recomputation from raw observations
- current tail-flag merge is clean

Concern:

- `except Exception: continue` silently swallows optimizer failures
- summary language over-claims that posterior architecture is now the bottleneck

#### 2. `28` descriptor-feasible-prior probe

Script:

- `scripts/28_descriptor_feasible_prior.py`

Purpose:

- test whether descriptor neighborhoods narrow the feasible theta region

Outcome:

- weak-to-moderate narrowing signal
- not strong enough to justify immediate training-time hardwiring

#### 3. `29` minimal active-set audit

Script:

- `scripts/29_minimal_active_set_audit.py`

Key implementation note:

- inactive parameters are fixed to a dataset-level `theta_ref`
- this avoids the trivial mistake of fixing inactive parameters to that curve's own full optimum

Infrastructure added:

- internal full-fit checkpointing
- active-set progress checkpointing
- resume support for long runs

Strong run:

- `outputs/29_minimal_active_set_audit_r5/summary.txt`

Headline result:

- many curves recover near-full-fit quality with only `2-4` active parameters

#### 4. `30` local effective-dimension audit

Script:

- `scripts/30_local_effective_dimension_audit.py`

Purpose:

- compute local Jacobian/SVD dimension around full-fit thetas

Strong run:

- `outputs/30_local_effective_dimension_audit_r5/summary.txt`

Headline result:

- `dim95 median = 2`
- `dim99 median = 3`

Interpretation:

- each curve likely lives on a low-dimensional local mechanism manifold

#### 5. `31` active-set co-occurrence summary

Script:

- `scripts/31_active_set_cooccurrence.py`

Outputs:

- `outputs/31_active_set_cooccurrence/active_set_cooccurrence_k4.png`
- `outputs/31_active_set_cooccurrence/summary.txt`

Headline result:

- there is no single universal 4-parameter subset
- but there is a clear shared backbone

### What seems settled

1. The current 9-parameter ODE class is not grossly incapable.
2. A single curve usually does not require all 9 effective degrees of freedom.
3. Different curves do not all use the same small subset.
4. The right next abstraction is probably not "all 9 parameters are independent".

### What is still unresolved

1. Are the observed active subsets best explained by:
   - parameter coupling,
   - latent mechanism drivers,
   - or local model misspecification in hard tails?

2. Are some parameters:
   - backbone/global,
   - and others conditional/rarely active?

3. Should the next modeling move happen at the level of:
   - structured prior,
   - latent-driver reparameterization,
   - or extra dynamical state variables?

### Current working interpretation

Best current summary:

- the global family needs multiple mechanism knobs,
- but each individual curve only sees a small, curve-dependent subset of them,
- with a recurring shared backbone around transport, burst, and capacity behavior.

That means the next research move should likely emphasize:

- parameter interaction,
- coupling,
- and structured low-dimensional mechanism drivers,

rather than jumping directly to a universal plausibility score or thermodynamics-style criterion.

### Recommended next step

Short version:

1. Study parameter interaction before defining the final scoring rule.
2. Treat `27a` as expressivity evidence, not as a final architectural verdict.
3. Push toward `few latent drivers -> 9 mechanism parameters` as the next conceptual prototype.
