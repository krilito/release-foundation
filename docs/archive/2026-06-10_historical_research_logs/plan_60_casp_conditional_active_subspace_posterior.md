# Plan 60 — CASP: Conditional Active-Subspace Posterior

Date: `2026-05-26`. Branch: `phase2-sprint1-burst`.

## Status

User explicitly requested an **original algorithm**, not another stitched
RF/MLP recipe. The model zoo (plan_51) and curve-observer hybrids (56-58)
are all sklearn-primitive concatenations and are intentionally stopped.

CASP is the next-tier proposal.

## Scope reframe (2026-05-26, after design start)

User stated: the actual target is **the entire drug release domain**, not
just PLGA. This forces three architectural commitments before any code is
written:

1. **`casp/` is simulator-agnostic.** It imports `ReleaseSimulator` (the
   ABC from `simulator.py`), never a concrete subclass. PLGABiphasic is
   passed in at runtime. Switching to Weibull / Liposome / future LNP
   simulator must be a single line change.
2. **Phase 2 includes a non-PLGA mechanism (Weibull liposome IVR).** This
   is the load-bearing demonstration that the method is general, not a
   PLGA artifact. Codex's accelerated_IVR pipeline (scripts 52-55) already
   has cleaned Weibull oracle parameters and 93-104 curves — adequate for
   a method demonstration.
3. **Warm-start is optional.** PLGA has oracle theta bank (script 29);
   Liposome has Weibull fits (script 54). Future mechanisms may have
   neither. CASP must run without warm-start, just slower / less stable.

This raises ambition (universal release-mechanism posterior) and tightens
the architecture (no PLGA-specific shortcut leaks). It does not change the
core math or the v0 development order — PLGA first, Liposome second.

## Problem statement

The project has been stuck on two facts that current methods cannot reconcile:

1. Tree ensembles (`38d`) reach `R^2 = 0.957` on cross321 curves, but per-
   parameter `theta` recovery is weak (`+0.216` to `-0.023`, script `39`).
2. The 9-parameter ODE is **identifiable per-curve only in 2-4 directions**
   (script `29`: `dim95` median = 2, `dim99` median = 3). The remaining 5-7
   directions are sloppy — different theta values give the same curve.

Every existing predictor outputs a single theta point. That throws away
the identifiability structure. RF makes confident point predictions even
on sloppy directions where there is no signal. The paper closeout names
this explicitly:

> Current system lacks uncertainty, active action planning, and broad
> mechanisms.

## Method — CASP

For each curve indexed by formulation descriptors `x` and early observations
`y_early = Q(1, 3, 5, 7)`, learn a **conditional low-rank Gaussian posterior**:

```text
q(theta | x, y_early) = N(theta | mu(x, y_early), U(x, y_early) S U(x, y_early)^T + sigma_0^2 I)
```

where

- `mu(x, y_early) in R^9` — center of the feasibility family,
- `U(x, y_early) in R^{9 x r}` — orthonormal columns spanning the
  curve-conditional active subspace (the directions that actually move
  the predicted curve),
- `S = diag(sigma_active^2) in R^{r x r}` — variance on active directions,
- `sigma_0^2 I` — small nugget on sloppy directions, fixed and equal to
  the local prior variance projected onto the null space of `U`.

`r` is the active dimensionality. v0 fixes `r = 4` based on `29`'s
`dim95 = 2 / dim99 = 3` finding (one slot of headroom). v1 learns `r`
per-curve via gumbel-softmax over `r in {1, 2, 3, 4, 5, 6}`.

### Forward

```text
sample      epsilon ~ N(0, I_r)
theta       = mu + U @ diag(sigma_active) @ epsilon       # active draw
            + sigma_0 * v_inactive                        # inactive nugget
Q_hat(t)    = PLGABiphasic.simulate(theta, t)             # torchdiffeq, differentiable
```

### Training objective

```text
L(x, y_early, y_full) =
    E_{theta ~ q} [ ||y_full - Q_hat(theta, t_full)||^2 / sigma_obs^2 ]
  + beta * KL[ q(theta | x, y_early) || prior_PLGA(theta) ]
  + gamma * ||U(x, y_early) - U_oracle_29(curve)||_F^2     # warm-start, decayed
  + lambda * orthogonality_loss(U)
```

The KL term is critical and is the **first reason this is not a stitched
algorithm**: in the inactive subspace, the posterior is forced toward the
prior, so the model learns to acknowledge non-identifiability rather than
hallucinate certainty there. In the active subspace, the data term
dominates and the posterior tightens.

The warm-start term uses the oracle theta and active set already computed
in `outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv`. It is decayed
to zero over the first 100 epochs so the final model is not anchored to
oracle.

### Encoder

```text
inputs  : x (10/13-d), y_early (4-d)
trunk   : MLP([x, y_early]) -> h in R^64
heads   :
    mu_raw  in R^9     -> sigmoid -> mu in prior_box
    U_raw   in R^{9*r} -> reshape -> QR decomp -> U with orthonormal cols
    log_sigma in R^r   -> exp -> sigma_active
```

The QR step is what makes `U` interpretable as an orthonormal basis. It is
differentiable in PyTorch (`torch.linalg.qr`).

## Why CASP is original, not stitched

Three claims, each verifiable:

1. **The output object is new.** No prior PLGA-release predictor outputs
   `(mu, U, sigma)` with curve-conditional active subspace. RF/ET output
   theta points. SBI posteriors output full-rank densities. CASP outputs a
   curve-specific *rank-r* density — neither of these.

2. **The objective is new.** The ELBO with KL on the inactive subspace is
   not present in any existing script. The closest is `27_partial_curve_npe`
   (full posterior, no rank constraint, no active-subspace head). The
   inactive-direction prior pull is the specific mechanism that enforces
   identifiability awareness.

3. **Existing methods are special cases of CASP.**
   - RF -> theta (B2): set `U = 0`, `sigma_active = 0`. Then CASP reduces
     to a deterministic theta point regression.
   - SBI posterior (script 09/27): set `r = 9` (full rank, no inactive
     subspace). Then CASP reduces to full-rank amortized posterior
     inference without the active-subspace bias.
   - Active-set audit (script 29): the per-curve `r` and `U` that CASP
     learns should match the oracle active set on a per-curve basis. This
     gives the second contribution (mechanism feasibility) for free.

This is the structural property of an original method: it subsumes the
prior art as edge cases.

## Experimental plan

### Phase 1 — v0 minimal viable CASP (target: 5 days)

Single script `scripts/60_casp_v0.py`. Scope:

- cross321 only, `r = 4` fixed, formulation_plus_early input
- random_5fold CV
- decoder = full PLGABiphasic via torchdiffeq
- warm-start from full_fit_bank.csv

Success criterion:
- median `R^2 >= 0.92` (within `~0.03` of RF best 0.957)
- **90% predicted-interval coverage in `[0.85, 0.95]`** on held-out fold
- training stable (no NaN, no obvious mode collapse on `U`)

If these don't hold, debug or fall back to surrogate decoder (small MLP
trained to approximate PLGABiphasic, faster autograd, lower fidelity).

### Phase 2 — full benchmark grid + cross-mechanism (target: 5-7 days)

Script `scripts/61_casp_benchmark.py`. Scope:

- **Mechanism A — PLGABiphasic (9 params)**:
  - both PLGA datasets (cross321 + cleaned internal181)
  - all 3 CV schemes (random, by_drug, by_polymer)
  - both input modes (early_only, formulation_plus_early)
  - compare against codex's existing best per cell (38d / 42 / 50)
- **Mechanism B — Weibull (2 params) on accelerated_IVR liposome**:
  - random_5fold + group_by_API + group_by_release_method (matching script 54)
  - early window = 12h (codex's strongest, full R² 0.956)
  - compare against script 54's `early-only Weibull fit` baseline (R² 0.956)
  - **claim**: CASP matches Weibull fit on R² but produces calibrated PI
    that the direct fit cannot

The cross-mechanism evidence is what makes this a method paper and not
"another PLGA benchmark"; without it the originality claim weakens.

Outputs:
- `outputs/61_casp_benchmark/scheme_method_summary.csv` — direct comparison
  with 38d / 42 / 50 winners
- `outputs/61_casp_benchmark/calibration_curves.png` — reliability plot
  per cell, the new metric RF cannot produce
- `outputs/61_casp_benchmark/active_dim_distribution.png` — learned vs
  oracle active dim distribution

### Phase 3 — paper claims (target: 3-5 days)

Two evaluations specific to CASP's claimed novelty:

1. **Identifiability recovery audit**:
   for each test curve, compare CASP's learned `U(x, y_early)` with the
   oracle active set from script 29. Report fraction of curves where
   CASP's top-r directions are within the oracle's top-(r+1) directions.

2. **OOD calibration audit**:
   on the NC wet curves (script 48), report whether CASP's PI covers the
   observed curve. RF/ET produce no PI. This is a new evidence type.

### Phase 4 — v1 learned rank (optional, 3-5 days)

If v0 results justify the investment, add gumbel-softmax over `r`. Test:
does per-curve learned `r` correlate with oracle dim95? Does it improve
catastrophic failure rate further?

## Honest research risks and fallbacks

Listed in order of likelihood:

### Risk R1 — Differentiable ODE training is unstable

Estimate: 30% probability.

The torchdiffeq adjoint method on 9-param stiff ODEs has known issues with
dopri5 step rejection during training. PLGABiphasic uses `rtol=1e-5, atol=1e-6`
which is fine for inference but may produce noisy gradients.

Fallback: train a small MLP surrogate `Q_hat = f(theta, t)` on simulator
outputs first (fast, accurate, smooth gradients), then use the surrogate
inside CASP. Lose some fidelity, gain stability. About 1 extra day to
implement.

### Risk R2 — Mode collapse on U

Estimate: 25%.

If the warm-start is too aggressive, U just becomes oracle U and never
adapts to new curves. If too weak, the model converges to a trivial U
(e.g., aligned with descriptors, not with curve-relevant directions).

Mitigation: log U eigenvalue spectrum every 10 epochs, anneal warm-start
weight, ensure orthogonality penalty is active.

### Risk R3 — CASP loses to RF on point R^2

Estimate: 35%.

Possible. The benefit of CASP is on calibration and identifiability, not
necessarily on R^2. If CASP gives `R^2 = 0.93` vs RF's `0.957`, the paper
must still sell the calibration + interpretability story. This is honest
but the user should know.

Fallback narrative: CASP is the **principled** baseline; tree ensembles
are useful for raw point R^2 but cannot quantify identifiability or
uncertainty. Both belong in the paper.

### Risk R4 — Per-curve active dim r is unstable across seeds

Estimate: 20% (only relevant in v1).

If gumbel-softmax over `r` gives different active-dim distributions on
seed=0 vs seed=1, the mechanism claim weakens.

Mitigation: report distribution over 5 seeds; if unstable, fall back to
fixed `r = 4` and remove the learned-r claim from the paper.

## Decision rules and stop conditions

After Phase 1:
- if v0 R^2 < 0.85 even with surrogate decoder, abort CASP, return to
  tree-ensemble + uncertainty wrappers (e.g., quantile RF or conformal
  prediction). The paper falls back to "principled UQ on top of RF".
- if v0 R^2 in [0.85, 0.92] and calibration is good, accept narrower
  paper framing: "CASP matches RF point accuracy while providing
  calibrated uncertainty".
- if v0 R^2 >= 0.92 AND calibration is good, proceed to Phase 2 with full
  ambition.

After Phase 3:
- if CASP wins on calibration but loses on R^2, paper is dual-method.
- if CASP wins on both, RF becomes a baseline.

## File layout

```text
docs/plan_60_casp_conditional_active_subspace_posterior.md    (this file)
scripts/60_casp_v0.py                                          (Phase 1, PLGA only)
scripts/61_casp_benchmark.py                                   (Phase 2, PLGA + Liposome)
scripts/62_casp_identifiability_audit.py                       (Phase 3)
casp/                                                          (new module, simulator-agnostic)
    __init__.py
    encoder.py            # mu/U/sigma heads; reads n_params from simulator
    decoder.py            # ReleaseSimulator wrapper, optional surrogate per-mechanism
    elbo.py               # objective with KL on inactive directions; simulator-agnostic
    train.py              # training loop, warm-start scheduler (warm-start = optional)
    calibration.py        # PI coverage / reliability diagrams
outputs/60_casp_v0/...
outputs/61_casp_benchmark/{plga,liposome}/...
outputs/62_casp_identifiability/...
```

`casp/` is a new module (not just a script) because:
1. v0 -> v1 -> v2 will share the encoder/decoder/elbo.
2. **PLGA -> Liposome -> future mechanisms** all reuse the same encoder
   and ELBO; only the simulator changes. Module structure makes this a
   one-line swap. Single-file scripts (the codex pattern) would force
   copy-paste across mechanisms — the exact "stitching" the user rejected.

This is also why "stitched algorithms" stopped working: the project is
past the single-file complexity ceiling, AND past the single-mechanism
scope.

## Anti-self-deception checklist

- [ ] R^2 measured on the EXACT same fold indices as 38d / 42, so the
      comparison is paired.
- [ ] PI coverage reported on each fold separately, not just overall
      (overall can hide per-fold miscalibration).
- [ ] Active-subspace recovery measured against script 29 oracle, not
      against CASP's own consistency.
- [ ] Multiple seeds (>= 3) on every reported number.
- [ ] Per-curve U logged for >= 10 representative curves and inspected
      manually for sensibility before claiming identifiability recovery.
- [ ] Compare against RF + bootstrap (cheap UQ baseline) before claiming
      CASP wins on calibration. RF + bootstrap is the honest UQ comparator.
