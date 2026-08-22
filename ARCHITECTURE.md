# ARCHITECTURE

The technical vision. Phase plan, module responsibilities, what this is
and what this is not.

## Problem statement

Drug release prediction from formulation descriptors is, mathematically, an
amortized inverse problem:

    x (descriptors)  →  θ (kinetic parameters)  →  Q(t) (release trajectory)

The middle arrow is conditional on a kinetic model. Three things make this
hard at our data scale:

1. **Sparse trajectories.** Median ~20 time points per curve in our PLGA data.
2. **Sparse formulations.** 181 curves in the internal benchmark; ~10⁴
   curves across all of public PLGA literature.
3. **Non-identifiability.** Different θ values produce nearly identical
   Q(t), so the descriptor-to-parameter map is one-to-many in unknown
   directions.

Cascade approaches (fit θ per curve → train a regressor x → θ → predict
Q(t|θ)) bake the non-identifiability into the supervised target and
generalize poorly across formulations. This codebase replaces the cascade
with amortized SBI: a single neural posterior `q_φ(θ | x, Q)` trained
jointly against a mechanism-constrained simulator.

## North star

A foundation model for cross-mechanism drug release dynamics, where one
posterior network handles PLGA, LNP, liposomes, microspheres, and hydrogels,
conditioned on a mechanism-class token plus a unified formulation
tokenization.

## Phase plan

### Phase 1 — PLGA-only amortized SBI (0–6 months)

- Mechanism-constrained ODE simulator (`PLGABiphasic`, 7 parameters)
- MLP descriptor encoder
- Neural Posterior Estimator via `sbi` (NPE-C)
- Grouped-holdout evaluation against the legacy SRDS-MEP baseline
- Simulation-based calibration (SBC) for identifiability diagnostics

**Deliverable:** method paper. **Compute:** single 4060 sufficient.

### Phase 2 — Multi-mechanism extension (6–14 months)

- Add LNP, liposome, microsphere simulators as `ReleaseSimulator` subclasses
- Replace MLP encoder with a transformer over formulation tokens
  (drug, polymer, processing, route)
- Literature-mined dataset (target 2k–5k curves across mechanisms)
- Unified amortized posterior with mechanism-class conditioning

**Deliverable:** systems paper. **Compute:** A100 × 1, rented.

### Phase 3 — Foundation model (14–24 months)

- 10M–100M parameter backbone
- Counterfactual head: predict ΔQ under intervention on x
- Mechanism-disentangled latent space
- Zero-shot generalization to held-out mechanism classes (validation)

**Deliverable:** flagship paper. **Compute:** A100 × 4 or H100 × 1.

## Module responsibilities

| Module          | Phase 1 contents                                 | Phase 2+ extension                                          |
|-----------------|--------------------------------------------------|-------------------------------------------------------------|
| `simulator.py`  | `ReleaseSimulator` ABC + `PLGABiphasic`          | Add LNP, liposome, microsphere subclasses                   |
| `encoder.py`    | `FormulationEncoder` ABC + MLP encoder           | Transformer encoder over (drug, polymer, processing) tokens |
| `posterior.py`  | `sbi.NPE_C` wrapper                              | Larger conditional flow; mechanism-class conditioning       |
| `data.py`       | PLGA 181 loader + synthetic sampler              | Literature-mined dataset; multi-mechanism unified schema    |
| `diagnostics.py`| SBC, posterior coverage, identifiability         | Posterior calibration across mechanism classes              |
| `train.py`      | NPE training loop, single mechanism              | Multi-mechanism joint training with class-balanced batching |
| `eval.py`       | Grouped-holdout R² vs legacy MEP                 | Leave-one-mechanism-out; zero-shot evaluation               |

## The PLGA biphasic ODE (Phase 1)

State variables, all dimensionless:

    h(t):  hydration fraction          ∈ [0, 1]
    m(t):  polymer MW ratio M(t)/M(0)  ∈ [0, 1]
    Q(t):  cumulative drug released    ∈ [0, 1]

ODE system (rates in 1/day):

    dh/dt = kw · (1 − h)
    dm/dt = −kh · h · m · (1 + α · (1 − m))             # autocatalytic
    dQ/dt = kd · h · (1 − Q)                            # hydration-gated diffusion
          + ke · σ((m_crit − m) / ε_m) · (1 − Q)        # erosion above threshold

Initial conditions: `h(0)=0`, `m(0)=1`, `Q(0)=q_burst`.

Seven free parameters: `log_kw, log_kh, log_α, log_kd, log_ke, m_crit, q_burst`.

The choice of three state variables (vs. Q alone) and the autocatalytic
term are non-cosmetic; see ADR-002 in DECISIONS.md.

## What this codebase is NOT

- Not a symbolic regression toolkit. PySR-style search is retired.
- Not a generic curve-fitting pipeline; mechanism-faithful ODEs are required.
- Not a from-scratch SBI implementation. We depend on `sbi` for proven NPE.
- Not (yet) a deployable formulation-design tool. That comes after Phase 3.
