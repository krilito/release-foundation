# 27d — Physics-residual Neural ODE for amortized SBI

Author: Claude (Opus 4.7), session 6
Date: 2026-05-24
Status: Research plan, M1 ready to implement

---

## Goal

Extend `PLGABiphasic` ODE with a small learned residual `f_ψ(state, t)` added
to `dQ/dt`. Jointly amortize-infer `(θ, ψ)` via sbi.NPE. Address the failure
mode 27a / 27b / 27c isolated: the 9-D parametric ODE expresses ~95% of 321
in full-curve fit (27a) but fails on the fast/short boundary-pinning subset
in partial-observation inference (27c).

The residual is NOT a free MLP. It is a **low-dimensional basis expansion
on a fixed time grid** with strong L2 shrinkage toward zero. Identifiability
risk (NN steals θ's job) is mitigated structurally, not just by tuning.

This is the project's first attempt at "physics + learned residual" in an
amortized Bayesian setting. The framing that survives reviewer scrutiny:

> Amortized SBI for joint posterior over parametric kinetic parameters and
> low-dimensional model-error correction, enabling cross-domain release
> prediction with calibrated uncertainty over both known physics and
> missing dynamics.

---

## Why this, not the alternatives

| Option | What | Why not (primary) |
|---|---|---|
| Free MLP residual on dQ/dt | high capacity NN | identifiability collapse (script 26's mistake) |
| **Low-dim basis residual** | 5–8 Fourier modes | **first-choice; bounded capacity, smooth, interpretable** |
| Multiplicative modulator | scalar × parametric | can't add NEW dynamics (triphasic lag etc.) |
| State-dep residual on (dh, dm, dQ) | three channels | over-parameterized; save for v2 if v1 partial |
| Mechanism mixture | 2–3 parametric variants | safer but lower scientific upside; FALLBACK B |
| NeuralCDE / Latent ODE | replace physics | throws away 95% of what works |

Low-dim Fourier basis = the minimum architectural change that COULD capture
the missing modes (triphasic lag, secondary burst, morphology transition)
WITHOUT enabling NN-takes-over failure.

---

## Architecture

### Residual definition

```
N_modes = 5  (default; configurable 3–10)
residual(t, c) = sum_{k=1..N_modes} c_k * phi_k(t)
where phi_k(t) = sin(k * pi * t / T_max)   # zero at t=0 (preserves Q(0)=0)
```

`c = (c_1, ..., c_{N_modes})` are per-curve coefficients. They become part
of the SBI-inferred parameter vector.

### Modified ODE

```
dh/dt = kw * (1 - h)                                    # unchanged
dm/dt = -kh * h * m * (1 + alpha * (1 - m))             # unchanged
dQ/dt = kd * h * (Q_max - Q)
      + ke * sigmoid((m_crit - m)/eps_m) * (Q_max - Q)
      + (q_burst/tau_burst) * exp(-t/tau_burst) * (Q_max - Q)
      + residual(t, c) * (Q_max - Q)                    # NEW
```

The residual is gated by `(Q_max - Q)` so it cannot push Q past Q_max. It
can only redistribute release timing within the saturating envelope.

### Theta layout (9 → 9 + N_modes)

```
theta = [
    log_kw, log_kh, log_alpha, log_kd, log_ke,    # base 5 rates
    m_crit, q_burst, log_tau_burst, Q_max,        # base 4 other
    c_1, ..., c_{N_modes},                        # NEW residual coefs
]
```

### Prior on residual coefficients

```
c_k ~ Normal(0, sigma_c)
sigma_c = 0.05   # shrinkage scale; default = "residual rate is small"
```

This is a soft prior that drags coefficients toward zero. The data has to
push them away from zero to overcome the prior.

NOTE: sbi.NPE prior must be a single Distribution. Compose as:

```
prior_base = sim.prior()              # Independent Uniform over 9 params
prior_resid = Normal(zeros(N_modes), sigma_c * ones(N_modes))
prior_joint = block_concat(prior_base, prior_resid)
# implementation: use sbi.utils.MultipleIndependent
```

---

## Training pipeline

Same as 27c, with three changes:

1. `theta_dim = 9 + N_modes`. Sample from joint prior. Simulate via
   augmented ODE.
2. Density estimator unchanged (MAF; sbi handles higher theta_dim).
3. Add a residual-magnitude penalty as a diagnostic, not loss:
   `mean(c**2)` per batch, log it.

NPE training stays identical. The novelty is in the simulator, not the
inference net.

---

## Milestones, kill criteria, deliverables

Each milestone is a self-contained experiment with a binary pass/fail.

### M1 -- forward simulator + sanity (1 week)

**Build**:
- `simulator.py` gets a new `PLGABiphasicResidual` subclass (or constructor
  flag `n_residual_modes`). Honors AGENTS.md #1 (mechanism subclass).
- `scripts/27d_residual_sim_sanity.py`:
  - simulate 1000 curves with c=0 → must match `PLGABiphasic` bit-exactly
  - simulate 1000 curves with random c~Normal(0, 0.05) → curves still in
    [0, Q_max], still monotone non-decreasing, still Q(0)=0
  - plot 20 random curves with vs without residual

**Pass**:
- bit-exact match when c=0 (within 1e-6 RMSE)
- physical invariants hold for random c
- residual magnitude in plotted curves is visible but not dominant

**Kill**: if c≠0 violates monotonicity / Q_max bound → residual gating is
wrong, fix gating or switch to multiplicative modulator architecture.

**Output**: `outputs/27d_sim_sanity/{sanity.csv, curves.png}`

---

### M2 -- joint SBI training + SBC (1-2 weeks)

**Build**:
- `scripts/27d_residual_npe.py` -- copy of `27c_sbi_npe_partial.py` with:
  - simulator = `PLGABiphasicResidual(n_modes=5)`
  - prior = joint via `MultipleIndependent`
  - theta_dim = 14
  - same MAF, same FCEmbedding, same training
- Run SBC at fixed prefix 1/3/7 d on synthetic held-out

**Pass**:
- G1 (SBC KS p > 0.05 all 14 dims, all 3 prefixes) — strict
- residual coef posteriors have mean ~0 on held-out synthetic where true
  c was sampled from prior (sanity)

**Kill (HARD)**:
- if after 4 separate hyperparameter attempts (n_modes ∈ {3,5,8}, sigma_c
  ∈ {0.02, 0.05, 0.10}, num_transforms ∈ {5,8}) SBC still fails ALL
  configurations → augmented theta + bounded prior is fundamentally
  hard for MAF on this simulator. Switch to **fallback B (mechanism
  mixture)**. Document in DECISIONS.md as ADR-028 (NeuralODE residual SBI
  attempted, failed, mixture chosen).

**Output**: `outputs/27d_residual_npe/{posterior.pt, sbc/, training_log.txt}`

---

### M3 -- 321 evaluation + per-subgroup gates (1 week)

**Build**:
- evaluation harness reuses 27c's 321 loader + subgroup classification
- new diagnostics:
  - per-curve `||c_hat||_2` (median across posterior samples)
  - correlation: does `||c_hat||` correlate with v1's per-curve residual
    in 27a per_curve.csv? (high correlation = residual is doing what we
    want; low = residual is doing something orthogonal/random)
  - subgroup median R^2 at 3d, with comparison to 27c

**Pass tiers (in increasing order of success)**:
- **G3a**: 3d `fast_short` median R^2 from -1.60 (27c) to >= -0.50 (huge
  improvement, residual captures the missing mode partially)
- **G3b**: 3d overall median R^2 >= 0.349 (matches baseline 24)
- **G3c**: 3d `fast_short` median R^2 > 0
- **G3d**: 3d overall median R^2 > 0.45 (clearly beats baseline 24)

**Kill**:
- if G3a not met → residual is not learning the missing physics. Either
  (a) basis is wrong (try B-spline or piecewise), or (b) the missing
  mode isn't smooth (try Option 1 free MLP), or (c) fall back to B.
  Limit: 2 more attempts before fallback.

**Output**: `outputs/27d_residual_npe/{summary.txt, per_curve_metrics.csv,
subgroup_summary.csv, residual_magnitude_analysis.png}`

---

### M4 -- refinement + paper-prep (2-3 weeks)

Conditional on M3 ≥ G3b (matches baseline 24):

- Ablation: 0 modes vs 3 vs 5 vs 8 vs free-MLP residual
- Posterior visualization: posterior over c projected to 2D for 5 example
  curves (well-fit + boundary-pinned)
- Cross-DOI calibration: same diagnostics on a held-out subset of 321
  not used in any training/eval
- Interpretation: what physical phenomenon does each Fourier mode
  correspond to in the worst-fit cases? Manual inspection of 20 cases
- If chitosan data arrives: extend `PLGABiphasicResidual` pattern to
  `ChitosanSwellingResidual`, demonstrate cross-mechanism residual
  framework

---

## Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| R1: NN steals theta's job | medium | high | low-dim basis + shrinkage prior + 0-init pass |
| R2: SBC fails on 14-D | high | high | M2 kill criterion → fallback B |
| R3: residual fits noise not signal | medium | medium | M3 correlation check vs 27a residuals |
| R4: training time blows up | low | low | theta_dim only +5; expect 1.5x of 27c |
| R5: chitosan data delayed | medium | low (for A) | A doesn't depend on it; scope adjusts |

---

## What this is NOT

To prevent later scope creep:

- NOT a NeuralODE replacement of the parametric ODE. The parametric
  ODE stays as the dominant term.
- NOT a free MLP residual. The basis is fixed; SBI infers coefficients only.
- NOT a multi-task / cross-mechanism model in M1-M3. Chitosan extension
  is M4+, conditional on PLGA results.
- NOT an inverse-design framework. Capability extension is a separate
  paper-prep activity, also conditional.

---

## What I need from the user before M1

1. Confirm `n_modes=5` and `sigma_c=0.05` defaults are acceptable, or push
   back with reasoning.
2. Confirm M2 kill criterion (4 hyperparameter attempts) is acceptable
   gate to fallback B.
3. Confirm `PLGABiphasicResidual` as a subclass of `PLGABiphasic` (not a
   separate mechanism) is consistent with AGENTS.md #1 — I think yes,
   because the residual is parameterized within the same theta space.

---

## Code skeleton (M1 deliverable, ready to flesh out)

```python
# simulator.py (add to end)

class PLGABiphasicResidual(PLGABiphasic):
    """PLGABiphasic with low-dim Fourier residual on dQ/dt.

    Joint theta = (base_9_params, residual_coefs[N_modes]).
    residual(t) = sum_k c_k * sin(k * pi * t / T_max)
    gated by (Q_max - Q) so it preserves saturation invariant.
    """

    def __init__(self, n_modes: int = 5, sigma_c: float = 0.05,
                 t_max: float = 90.0, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.sigma_c = sigma_c
        self.t_max = t_max
        self.n_params = 9 + n_modes
        self.param_names = (
            super().param_names + [f"c_{k+1}" for k in range(n_modes)]
        )

    def prior(self) -> Distribution:
        from sbi.utils import MultipleIndependent
        base_prior = super().prior()    # Independent Uniform, 9D
        resid_prior = Independent(
            Normal(torch.zeros(self.n_modes),
                   self.sigma_c * torch.ones(self.n_modes)),
            1,
        )
        # MultipleIndependent stacks per-event-dim
        return MultipleIndependent([base_prior, resid_prior])

    def _residual(self, t: Tensor, c: Tensor) -> Tensor:
        # t: scalar (or (T,)); c: (B, N_modes); returns (B,) or (B, T)
        k = torch.arange(1, self.n_modes + 1, device=c.device, dtype=c.dtype)
        # phi: (B, N_modes)
        phi = torch.sin(k * torch.pi * t / self.t_max)
        if phi.ndim == 1:  # scalar t
            return (c * phi).sum(dim=-1)
        else:
            return (c.unsqueeze(-2) * phi.unsqueeze(0)).sum(dim=-1)

    def _vector_field(self, t, state, theta):
        base_theta = theta[..., :9]
        c = theta[..., 9:]
        # call parent for parametric part
        dstate_base = super()._vector_field(t, state, base_theta)
        # add residual to dQ component only
        Q = state[..., 2]
        Q_max = base_theta[..., 8]
        free = (Q_max - Q).clamp(min=0.0)
        resid = self._residual(t, c) * free
        # additive on dQ channel
        dstate = dstate_base.clone()
        dstate[..., 2] = dstate_base[..., 2] + resid
        return dstate

    # simulate() and simulate_numpy() inherit unchanged
```

M1 script implements this + sanity checks. M2 takes 27c verbatim, swaps
`PLGABiphasic()` for `PLGABiphasicResidual()`, runs.

---

## Honest assessment

This is research, not engineering. Failure rate at M2 is genuinely 30-50%
in my estimate — bounded prior + augmented theta + MAF has known
calibration issues at higher dim. If M2 fails 4 attempts, fallback B is
not a defeat; it's a clean experimental result ("we attempted joint
NeuralODE-SBI; found it uncalibrated; mixture decomposition worked
instead"). Either outcome is publishable methodology.

Total realistic timeline: **5-7 weeks from M1 start to M3 verdict**, then
2-3 weeks M4 if M3 passes. Total **2-2.5 months** for a paper-ready result.
