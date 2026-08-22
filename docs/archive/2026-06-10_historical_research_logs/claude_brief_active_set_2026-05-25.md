## PLGA parameter-activity brief for Claude

### What has been verified

1. `27a` shows that the current 9-parameter `PLGABiphasic` family has strong **full-curve expressivity** on matched external `321`.
   - Output: `outputs/27a_ode_expressivity_audit/summary.txt`
   - `overall median R^2 = 0.9976`
   - `frac well-fit (R^2 > 0.9) = 0.950`
   - `fast_short median R^2 = 0.9814`, but `fast_short frac well-fit = 0.788`

2. `27a` numeric results were independently checked and appear correct as a full-curve fit audit.
   - `per_curve.csv` values for `RMSE` and `R^2` match recomputation from raw `(t_obs, q_obs)` and `simulate_numpy`.
   - Current run has `259` rows and `259` unique `fid`.
   - Tail-label merge is clean in the current run.

3. `27a` should **not** be over-interpreted.
   - The current verdict text says `GO 27 vanilla NPE` and implies the bottleneck is posterior architecture.
   - That is stronger than the evidence.
   - Boundary-hit load is high:
     - median `boundary_hits = 3`
     - `57/259` curves have `>=5` boundary hits
     - the hardest tail curves are often pinned on many bounds simultaneously
   - Better reading:
     - `27a` rules out **gross ODE incapacity**
     - `27a` does **not** rule out local misspecification in hard tails

4. `29` shows that many curves do **not** need all 9 parameters to recover near-full-fit quality.
   - Output: `outputs/29_minimal_active_set_audit_r5/summary.txt`
   - On `internal181`:
     - `k=2` close-to-full for `67.5%`
     - `k=3` close-to-full for `82.5%`
     - `k=4` close-to-full for `90.0%`
   - On `cross321`:
     - `k=2` close-to-full for `61.0%`
     - `k=3` close-to-full for `83.9%`
     - `k=4` close-to-full for `96.3%`

5. `30` shows local effective dimension is much smaller than 9.
   - Output: `outputs/30_local_effective_dimension_audit_r5/summary.txt`
   - `dim95 median = 2` on both `internal181` and `cross321`
   - `dim99 median = 3` on both datasets
   - This supports the interpretation that each curve lives on a low-dimensional local slice of the global 9-parameter family

6. `31` shows that different curves do **not** use exactly the same active subset.
   - Output: `outputs/31_active_set_cooccurrence/summary.txt`
   - The most common `k=4` active set is `Q_max,log_kw,log_kh,log_kd`, but it is rare:
     - `internal181`: `11/160 = 6.9%`
     - `cross321`: `10/218 = 4.6%`
   - So there is no single universal 4-parameter subset.
   - However, there is a shared backbone:
     - `cross321` inclusion rates at `k=4`:
       - `log_kd = 0.794`
       - `q_burst = 0.761`
       - `log_kw = 0.679`
       - `Q_max = 0.555`
       - `log_kh = 0.500`
     - Rarely active:
       - `log_alpha = 0.037`
       - `m_crit = 0.124`

### Current interpretation

The best current reading is:

- The 9-parameter ODE family is useful globally.
- A single curve usually activates only `2-4` effective directions.
- Different curves appear to occupy different low-dimensional slices of the same global mechanism family.
- There is a shared backbone, but not a single fixed active subset for every curve.

This suggests the main problem is no longer:

- `Can the ODE draw the curve?`

but rather:

- `Which subset of mechanism directions is active for this curve?`
- `Which parameters are shared backbone parameters versus conditional/specialized parameters?`
- `Should the parameterization be reorganized into a few latent drivers rather than 9 independent knobs?`

### Important caveats

1. `29` is a greedy active-set diagnostic, not a proof of globally optimal sparse subsets.
   - It proves curve-specific low-dimensional structure exists.
   - It does not prove a unique active subset for each curve.

2. `27a` is a full-curve oracle fit, not a deployment condition.
   - High full-curve `R^2` does not imply identifiable or robust inference from descriptors or prefixes.

3. Boundary-hit patterns in `27a` matter.
   - They may indicate prior-edge dependence, local misspecification, or unresolved coupling between parameters.

### Questions for Claude

1. Given `27a + 29 + 30 + 31`, is the next principled step:
   - a parameter-interaction / latent-driver model,
   - or a scoring-rule / Occam-style mechanism filter?

2. Does it make more sense to model:
   - `few latent mechanism drivers -> 9 theta parameters`,
   rather than treating all 9 parameters as independent?

3. Should some parameters be treated as:
   - backbone parameters,
   - and others as conditionally active or optional?

4. How should we interpret `27a`'s combination of:
   - excellent full-curve fit,
   - but many boundary hits and weaker hard-tail support?

5. If we introduce parameter interactions, should that happen first as:
   - a structured prior / coupling layer,
   - or as new dynamical state variables?

### Files to inspect

- `scripts/27a_ode_expressivity_audit.py`
- `scripts/29_minimal_active_set_audit.py`
- `scripts/30_local_effective_dimension_audit.py`
- `scripts/31_active_set_cooccurrence.py`
- `outputs/27a_ode_expressivity_audit/summary.txt`
- `outputs/29_minimal_active_set_audit_r5/summary.txt`
- `outputs/30_local_effective_dimension_audit_r5/summary.txt`
- `outputs/31_active_set_cooccurrence/summary.txt`
- `outputs/31_active_set_cooccurrence/active_set_cooccurrence_k4.png`

### My current stance

I would not jump to a thermodynamics-style global scoring principle yet.

The evidence currently points first to **parameter interaction structure**:

- a shared core mechanism backbone,
- curve-specific active slices,
- likely non-independent parameter roles,
- and a need to reorganize the 9-parameter family into a smaller number of structured drivers.

So my current recommendation is:

1. Study parameter interaction first.
2. Only after that, define a mechanism plausibility criterion.
3. Keep `27a` as evidence for expressivity, but stop using it as proof that posterior architecture is the only remaining bottleneck.
