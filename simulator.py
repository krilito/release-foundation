"""
Drug-release mechanism simulators.

This module owns the *forward model* used by amortized SBI. Each release
mechanism class is a subclass of `ReleaseSimulator`. Phase 1 ships
`PLGABiphasic`; later phases add LNP, liposome, microsphere, hydrogel.

Why ABCs here. The simulator is one of two designated extension points
(the other is `FormulationEncoder` in encoder.py). Keeping all mechanisms
behind a common interface lets training loops, posterior networks, and
evaluation scripts remain mechanism-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform
from torchdiffeq import odeint


class ReleaseSimulator(ABC):
    """A drug-release mechanism as a differentiable ODE forward model.

    Subclasses define:
      * kinetic parameters θ of dimension `n_params`,
      * a prior over θ,
      * a forward map (θ, t) → Q(t) ∈ [0, 1].

    The forward map MUST be differentiable w.r.t. θ. We rely on torchdiffeq
    for adaptive ODE integration with autograd support.
    """

    n_params: int
    param_names: list[str]

    @abstractmethod
    def prior(self) -> Distribution:
        """Prior over θ ∈ R^{n_params}. Used by amortized SBI training."""
        ...

    @abstractmethod
    def simulate(
        self,
        theta: Tensor,                # (B, n_params)
        t: Tensor,                    # (T,) in days, must be sorted ascending
        x: Tensor | None = None,      # (B, D_x) optional descriptor conditioning
    ) -> Tensor:                      # (B, T) cumulative release in [0, 1]
        ...

    def sample_prior(self, n: int) -> Tensor:
        return self.prior().sample((n,))


class PLGABiphasic(ReleaseSimulator):
    """PLGA biphasic release: hydration → autocatalytic hydrolysis → erosion.

    State variables (dimensionless):
        h(t):  hydration fraction          ∈ [0, 1]
        m(t):  polymer MW ratio M(t)/M(0)  ∈ [0, 1]
        Q(t):  cumulative drug released    ∈ [0, 1]

    ODE (rates in 1/day):
        dh/dt = kw · (1 − h)
        dm/dt = −kh · h · m · (1 + α · (1 − m))
        dQ/dt = kd · h · (Q_max − Q)
              + ke · σ((m_crit − m) / ε_m) · (Q_max − Q)
              + (q_burst / τ_burst) · exp(−t / τ_burst) · (Q_max − Q)

    Initial conditions:
        h(0) = 0,   m(0) = 1,   Q(0) = 0

    Why three state variables, not just Q. Erosion onset depends on the
    *current* polymer MW, not the initial polymer MW. The legacy SRDS-MEP
    kernel used `tau_eff = (MW_init / 10000) * tau` to fake this; that
    conflates descriptor with latent state and cannot express autocatalysis.
    See ADR-002 in DECISIONS.md.

    Why the (1 + α(1 − m)) autocatalysis term. PLGA hydrolysis generates
    carboxylic-acid end groups that lower local pH and accelerate further
    hydrolysis (Göpferich 1996; Siepmann & Siepmann 2008). Without this
    term we cannot reproduce the characteristic slow → fast transition.

    Why Q_max instead of saturating at 1. The Day 1.5 residual diagnostic
    on ID 142 (n=64, the most densely sampled curve) showed a persistent
    late-time negative-residual band: the ODE drove Q → 1 while the real
    curve plateaued below 1. PLGA microparticles routinely fail to release
    100 % due to residual adsorption, inaccessible pore volume, and
    polymer-encapsulated drug. See ADR-007 in DECISIONS.md.

    Why a continuous burst term (Phase-2 Sprint-1, ADR-026). The legacy
    Q(0)=q_burst delta introduced a +0.06–0.11 systematic Q0 residual on
    both internal LODO and cross-DOI evaluations (~50–90 % of total MAE).
    Real release curves are 0 at t=0 by definition. The continuous burst
    term integrates to Q_max·(1−exp(−q_burst)) ≈ q_burst·Q_max for small
    q_burst, with timescale τ_burst, generalizing the old delta as τ→0.

    Theta layout (9 params, log-scale on rates and burst timescale):
        [log_kw, log_kh, log_alpha, log_kd, log_ke,
         m_crit, q_burst, log_tau_burst, Q_max]
    """

    n_params = 9
    param_names = [
        "log_kw", "log_kh", "log_alpha",
        "log_kd", "log_ke",
        "m_crit", "q_burst", "log_tau_burst", "Q_max",
    ]

    # Sharpness of the erosion-onset gate. Fixed (not learned).
    # Learning it tends to collapse to either hard threshold (zero-gradient
    # regions) or no gate at all (identifiability collapse).
    EPS_M = 0.05

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-6,
        method: str = "dopri5",
    ) -> None:
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def prior(self) -> Distribution:
        """Uniform prior; log-scale on rate parameters.

        Bounds cover PLGA microsphere and implant release windows ranging
        from ~1 day to ~6 months. Rate parameters span orders of magnitude,
        hence log-uniform.
        """
        lows = torch.tensor([
            -3.0,    # log_kw         -> kw ≈ 0.05 /day  (slow hydration)
            -4.0,    # log_kh         -> kh ≈ 0.018 /day (slow hydrolysis)
            -2.0,    # log_alpha      -> α  ≈ 0.13       (weak autocatalysis)
            -5.0,    # log_kd         -> kd ≈ 0.0067 /day
            -3.0,    # log_ke         -> ke ≈ 0.05 /day
             0.05,   # m_crit
             0.0,    # q_burst        (no burst)
            -3.0,    # log_tau_burst  -> τ_burst ≈ 0.05 d ≈ 1.2 h (ADR-026)
             0.50,   # Q_max          (50 % asymptotic release)
        ])
        # Upper bounds tightened in ADR-015/016 after round-4/5 SBC traced
        # calibration failures to immediate-saturation samples from the
        # fast-rate prior corner (ADR-012). The caps sit at the upper edge
        # of published PLGA ranges -- fast but not unphysical.
        #
        # log_tau_burst ∈ [-3, 0] → τ_burst ∈ [0.05, 1.0] day (1.2 h - 24 h).
        # Lower bound: τ < 0.05 d makes the burst effectively a step function
        # (resolvable by 64-point t_grid). Upper bound: τ > 1 d overlaps
        # with kd-driven diffusion timescale, risking identifiability
        # collapse with log_kd (ADR-026 identifiability concern #1).
        highs = torch.tensor([
             1.0,    # log_kw         -> kw ≈ 2.7 /day   (ADR-015: was 2.0)
             -1.0,   # log_kh         -> kh ≈ 0.37 /day  (ADR-015: was 0.0)
             1.0,    # log_alpha      -> α  ≈ 2.7        (ADR-015: was 2.0)
             0.0,    # log_kd         -> kd ≈ 1.0 /day   (ADR-016: was 1.0)
             2.0,    # log_ke         -> ke ≈ 7.4 /day
             0.50,   # m_crit
             0.30,   # q_burst        (up to 30 % burst; bounded below Q_max_lo)
             0.0,    # log_tau_burst  -> τ_burst ≤ 1.0 day (ADR-026)
             1.00,   # Q_max          (complete release)
        ])
        return Independent(Uniform(lows, highs), 1)

    def _vector_field(self, t: Tensor, state: Tensor, theta: Tensor) -> Tensor:
        # state: (B, 3) = (h, m, Q);  theta: (B, 9)
        h, m, Q = state[..., 0], state[..., 1], state[..., 2]
        (log_kw, log_kh, log_alpha, log_kd, log_ke,
         m_crit, q_burst, log_tau_burst, Q_max) = theta.unbind(-1)

        kw = log_kw.exp()
        kh = log_kh.exp()
        alpha = log_alpha.exp()
        kd = log_kd.exp()
        ke = log_ke.exp()
        tau_burst = log_tau_burst.exp()

        dh = kw * (1.0 - h)
        dm = -kh * h * m * (1.0 + alpha * (1.0 - m))
        erosion_gate = torch.sigmoid((m_crit - m) / self.EPS_M)
        # Q saturates at Q_max < 1 to model incomplete release; the clamp
        # keeps the rate non-negative if a numerical wiggle pushes Q above
        # Q_max momentarily.
        free = (Q_max - Q).clamp(min=0.0)
        # Continuous burst term (ADR-026). The (Q_max - Q) coupling means
        # ∫burst_rate dt → Q_max·(1 - exp(-q_burst)) ≈ q_burst·Q_max for
        # small q_burst, with effective release window ~τ_burst.
        burst_rate = (q_burst / tau_burst) * torch.exp(-t / tau_burst) * free
        dQ = kd * h * free + ke * erosion_gate * free + burst_rate

        return torch.stack([dh, dm, dQ], dim=-1)

    def simulate(
        self,
        theta: Tensor,
        t: Tensor,
        x: Tensor | None = None,
    ) -> Tensor:
        if theta.ndim != 2 or theta.shape[1] != self.n_params:
            raise ValueError(
                f"theta must have shape (B, {self.n_params}), got {tuple(theta.shape)}"
            )
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D, got shape {tuple(t.shape)}")

        B = theta.shape[0]
        device = theta.device
        Q_max = theta[..., 8]   # index 8 in 9-param layout; see param_names
        # Q(0) = 0 (ADR-026): replaces the legacy Q(0) = q_burst delta.
        # Burst release is now driven continuously by the burst term in
        # _vector_field, not by the initial condition.
        state0 = torch.stack(
            [
                torch.zeros(B, device=device),
                torch.ones(B, device=device),
                torch.zeros(B, device=device),
            ],
            dim=-1,
        )

        def ode_fn(tt: Tensor, ss: Tensor) -> Tensor:
            return self._vector_field(tt, ss, theta)

        traj = odeint(ode_fn, state0, t, rtol=self.rtol, atol=self.atol, method=self.method)
        # traj: (T, B, 3)
        Q = traj[..., 2].transpose(0, 1)  # (B, T)
        # Enforce the two physical invariants that the continuous ODE
        # guarantees but adaptive float32 integration can violate by
        # ~O(1e-4):
        #   1. Q ≥ 0 (clamp below)
        #   2. Q monotone non-decreasing (cummax)
        #   3. Q ≤ Q_max per curve (torch.minimum with Q_max)
        # cummax is differentiable: the gradient routes through the
        # argmax index at each timestep, so SBI / NPE backprop still
        # works.
        Q = torch.cummax(Q.clamp(min=0.0), dim=1).values
        return torch.minimum(Q, Q_max.unsqueeze(-1))

    # ------------------------------------------------------------------
    # SciPy backend for offline diagnostics (non-differentiable, faster).
    # MUST stay numerically equivalent to .simulate(). This invariant is
    # enforced by test_torch_and_numpy_backends_agree in tests/.
    # ------------------------------------------------------------------
    def simulate_numpy(self, theta: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
        """SciPy DOP853 forward pass for offline NLS-style diagnostics.

        Equivalent in math to `.simulate()` but ~10x faster on CPU because
        it skips torch autograd plumbing. Use in oracle-sweep,
        identifiability-replay, and any other diagnostic script that does
        not need gradients.

        Args:
            theta: shape (n_params,) for a single parameter vector.
            t_obs: shape (T,) sorted ascending in days.

        Returns:
            Q: shape (T,) cumulative release in [0, Q_max].
        """
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"theta must have shape ({self.n_params},), got {theta.shape}"
            )

        (log_kw, log_kh, log_alpha, log_kd, log_ke,
         m_crit, q_burst, log_tau_burst, Q_max) = theta
        kw, kh, alpha, kd, ke = np.exp([log_kw, log_kh, log_alpha, log_kd, log_ke])
        tau_burst = float(np.exp(log_tau_burst))
        eps_m = self.EPS_M

        def vector_field(tt: float, state: np.ndarray) -> list[float]:
            h, m, Q = state
            dh = kw * (1.0 - h)
            dm = -kh * h * m * (1.0 + alpha * (1.0 - m))
            gate_arg = np.clip((m_crit - m) / eps_m, -60.0, 60.0)
            erosion_gate = 1.0 / (1.0 + np.exp(-gate_arg))
            free = max(Q_max - Q, 0.0)
            burst_rate = (q_burst / tau_burst) * np.exp(-tt / tau_burst) * free
            dQ = kd * h * free + ke * erosion_gate * free + burst_rate
            return [dh, dm, dQ]

        sol = solve_ivp(
            vector_field,
            (0.0, float(t_obs[-1])),
            [0.0, 1.0, 0.0],   # Q(0) = 0 per ADR-026
            t_eval=t_obs,
            method="DOP853",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success or sol.y.shape[1] != len(t_obs):
            return np.full_like(t_obs, 1e6, dtype=float)
        return np.minimum(np.clip(sol.y[2], 0.0, None), Q_max)


class PLGABiphasicResidual(PLGABiphasic):
    """PLGABiphasic with a low-dimensional Fourier residual on dQ/dt.

    Why this class exists (research plan 27d, milestone M1):
        27a found the 9-D PLGA ODE expresses ~95% of 321 cross-DOI curves
        in full-curve fit, but 27c showed amortized SBI on partial
        observations collapses on the boundary-pinned (fast / short)
        subset. The interpretation: the parametric form is locally
        misspecified for that subset, and amortized inference makes the
        mismatch worse (more confident wrong as training converges).

        This class adds a learned residual term to dQ/dt, parameterized
        as a low-dimensional Fourier sine basis with zero boundary at
        t=0 (preserves Q(0)=0). Coefficients are per-curve and become
        part of theta for joint SBI inference. Strong shrinkage prior
        keeps the residual small when the parametric form already fits.

    Modified ODE:
        dQ/dt = parametric(theta_base, state)
              + (sum_{k=1..n_modes} c_k * sin(k*pi*t / t_max)) * (Q_max - Q)

        The (Q_max - Q) gate is essential: it preserves the saturation
        invariant. The residual can redistribute release within the
        envelope but cannot push Q past Q_max or below 0 (combined with
        the cummax invariant inherited from .simulate()).

    Joint theta layout (n_params = 9 + n_modes):
        [<inherited 9 base params>, c_1, c_2, ..., c_{n_modes}]

    Why Fourier sine basis specifically:
        - sin(k * pi * t / t_max) = 0 at t=0 for all k -> Q(0)=0 preserved
        - smooth -> matches "missing physics is gradual" assumption
        - bounded amplitude (|sin| <= 1) -> c_k IS the residual scale
        - orthogonal -> coefficients are independent under L2 prior

    Why a low-dim BASIS (not a free MLP):
        Free NN residuals collapse identifiability (the NN steals theta's
        job; this was script 26's failure mode). Capping at n_modes = 5
        means the residual can express at most 5 smooth temporal patterns;
        anything theta could already explain is much cheaper for the
        optimizer to put on theta. See ADR-027 (TODO if M3 passes) for
        the empirical validation.
    """

    EPS_M = PLGABiphasic.EPS_M

    def __init__(
        self,
        n_modes: int = 5,
        sigma_c: float = 0.05,
        t_max: float = 90.0,
        rtol: float = 1e-5,
        atol: float = 1e-6,
        method: str = "dopri5",
    ) -> None:
        super().__init__(rtol=rtol, atol=atol, method=method)
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")
        if sigma_c <= 0:
            raise ValueError(f"sigma_c must be > 0, got {sigma_c}")
        if t_max <= 0:
            raise ValueError(f"t_max must be > 0, got {t_max}")
        self.n_modes = n_modes
        self.sigma_c = sigma_c
        self.t_max = t_max
        # Shadow class-level attrs as instance attrs (n_params, param_names).
        self.n_params = 9 + n_modes
        self.param_names = list(PLGABiphasic.param_names) + [
            f"c_{k+1}" for k in range(n_modes)
        ]

    # ------------------------------------------------------------------
    # Prior: joint over (base 9, residual coefs n_modes)
    # ------------------------------------------------------------------
    def prior(self) -> Distribution:
        """Joint prior. Base 9 stays Uniform per parent; coefs Normal(0, sigma_c).

        Returned as `sbi.utils.MultipleIndependent` so sbi.NPE accepts it as
        a single prior object. Sample shape: (..., 9 + n_modes).
        """
        # Local import: simulator.py otherwise needs no sbi dependency.
        from sbi.utils import MultipleIndependent

        base_prior = super().prior()
        resid_prior = Independent(
            torch.distributions.Normal(
                torch.zeros(self.n_modes),
                self.sigma_c * torch.ones(self.n_modes),
            ),
            1,
        )
        return MultipleIndependent([base_prior, resid_prior])

    # ------------------------------------------------------------------
    # Torch vector field: adds residual to dQ, inherits dh / dm
    # ------------------------------------------------------------------
    def _vector_field(self, t: Tensor, state: Tensor, theta: Tensor) -> Tensor:
        base_theta = theta[..., :9]
        c = theta[..., 9:]                                  # (B, n_modes)
        dstate_base = super()._vector_field(t, state, base_theta)  # (B, 3)

        # Fourier basis at scalar t (torchdiffeq passes t as 0-D tensor).
        k = torch.arange(1, self.n_modes + 1, device=c.device, dtype=c.dtype)
        phi = torch.sin(k * torch.pi * t / self.t_max)       # (n_modes,)
        residual = (c * phi.unsqueeze(0)).sum(dim=-1)        # (B,)

        # Gate by free = (Q_max - Q) >= 0 to preserve saturation invariant.
        Q = state[..., 2]
        Q_max = base_theta[..., 8]
        free = (Q_max - Q).clamp(min=0.0)

        dh = dstate_base[..., 0]
        dm = dstate_base[..., 1]
        dQ = dstate_base[..., 2] + residual * free
        return torch.stack([dh, dm, dQ], dim=-1)

    # ------------------------------------------------------------------
    # SciPy backend: inlines base computation + residual (no super() in
    # closures). Stays numerically equivalent to .simulate() by
    # construction; invariant enforced by 27d sanity script.
    # ------------------------------------------------------------------
    def simulate_numpy(self, theta: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"theta must have shape ({self.n_params},), got {theta.shape}"
            )
        base_theta = theta[:9]
        c = np.asarray(theta[9:], dtype=float)
        (log_kw, log_kh, log_alpha, log_kd, log_ke,
         m_crit, q_burst, log_tau_burst, Q_max) = base_theta
        kw, kh, alpha, kd, ke = np.exp([log_kw, log_kh, log_alpha, log_kd, log_ke])
        tau_burst = float(np.exp(log_tau_burst))
        eps_m = self.EPS_M
        k_arr = np.arange(1, self.n_modes + 1, dtype=float)

        def vector_field(tt: float, state: np.ndarray) -> list[float]:
            h, m, Q = state
            dh = kw * (1.0 - h)
            dm = -kh * h * m * (1.0 + alpha * (1.0 - m))
            gate_arg = np.clip((m_crit - m) / eps_m, -60.0, 60.0)
            erosion_gate = 1.0 / (1.0 + np.exp(-gate_arg))
            free = max(Q_max - Q, 0.0)
            burst_rate = (q_burst / tau_burst) * np.exp(-tt / tau_burst) * free
            residual = float((c * np.sin(k_arr * np.pi * tt / self.t_max)).sum())
            dQ = (kd * h * free
                  + ke * erosion_gate * free
                  + burst_rate
                  + residual * free)
            return [dh, dm, dQ]

        sol = solve_ivp(
            vector_field,
            (0.0, float(t_obs[-1])),
            [0.0, 1.0, 0.0],
            t_eval=t_obs,
            method="DOP853",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success or sol.y.shape[1] != len(t_obs):
            return np.full_like(t_obs, 1e6, dtype=float)
        return np.minimum(np.clip(sol.y[2], 0.0, None), Q_max)
