"""Ritger-Peppas release simulator — for chitosan hydrogels and matrix systems.

The Ritger-Peppas model (Korsmeyer-Peppas, 1983) is the standard
empirical kinetic form for polymeric matrix release:

    Q(t) = Q_max * min(k * t^n, 1)

where:
    n in [0.43, 0.85]  for cylindrical or spherical Fickian diffusion
                       to anomalous transport regimes
    n -> 1             pure case-II (erosion-dominated, zero-order release)
    k                  rate constant (1/time^n)
    Q_max in (0, 1]    asymptotic release fraction

Three free parameters in log/linear space:
    theta = [log_k, n, Q_max]

This is the third concrete ReleaseSimulator subclass, after PLGABiphasic
(9 params, stiff ODE) and WeibullSimulator (2 params, closed-form). All
three use the same casp/fib/ensemble/pullback machinery without code
change.

Prior bounds calibrated to chitosan-DEX / chitosan-GCV / chitosan-HA-PVP
hydrogel literature (typical Q reaches 60-95% within 7-28 days at 37C
PBS; n usually 0.45-0.7 for chitosan matrices; k spans ~2 orders of
magnitude across formulations).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform

from simulator import ReleaseSimulator


class ChitosanRitgerPeppas(ReleaseSimulator):
    """Ritger-Peppas Q(t) = Q_max * min(k * t^n, 1).

    Defaults the time unit to HOURS (matching chitosan release literature),
    but the simulator is unit-agnostic — interpretation is the caller's
    responsibility. Just be consistent: t_obs in hours implies k in 1/h^n.

    Theta layout (3 params):
        [log_k, n, Q_max]

    Prior bounds (calibrated to chitosan-DEX/GCV/hydrogel literature,
    assuming t in HOURS):
        log_k  in [-5.0, -0.5]    -> k in [0.0067, 0.61] / h^n
        n      in [0.35, 1.00]    -> Fickian (~0.43) ... case-II (~1.0)
        Q_max  in [0.30, 1.00]    -> 30%-100% asymptotic release

    Coverage check at t=24h (typical reporting point):
        Slow corner (log_k=-5, n=0.4): k*24^n ~= 0.025  -> Q24h ~= 2.5% of Q_max
        Fast corner (log_k=-0.5, n=1.0): k*24^n ~= 14.5 -> saturated by 2h
        These cover all reasonable chitosan formulations on the 1h-28d window.

    If t is in DAYS instead of hours, log_k prior shifts by n * log(24).
    For Fickian (n=0.5) that's ~+1.78 — i.e., log_k in [-3.2, 1.3] for days.
    Pass `time_unit_hours=False` to use day-based prior automatically.
    """

    n_params = 3
    param_names = ["log_k", "n", "Q_max"]

    def __init__(self, time_unit_hours: bool = True) -> None:
        self.time_unit_hours = time_unit_hours

    def prior(self) -> Distribution:
        if self.time_unit_hours:
            lows = torch.tensor([-5.0, 0.35, 0.30], dtype=torch.float32)
            highs = torch.tensor([-0.5, 1.00, 1.00], dtype=torch.float32)
        else:
            # Shift log_k by ~+1.78 for n=0.5 (= n * log(24)); use a wider
            # band to cover the n-dependent shift exactly.
            lows = torch.tensor([-3.5, 0.35, 0.30], dtype=torch.float32)
            highs = torch.tensor([1.5, 1.00, 1.00], dtype=torch.float32)
        return Independent(Uniform(lows, highs), 1)

    def simulate(self, theta: Tensor, t: Tensor, x: Tensor | None = None) -> Tensor:
        """theta: (B, 3). t: (T,) in time units consistent with k. Returns (B, T)."""
        if theta.ndim != 2 or theta.shape[1] != 3:
            raise ValueError(f"theta must have shape (B, 3), got {tuple(theta.shape)}")
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D, got {tuple(t.shape)}")
        log_k = theta[:, 0:1]                                  # (B, 1)
        n = theta[:, 1:2]                                      # (B, 1)
        Q_max = theta[:, 2:3]                                  # (B, 1)
        k = log_k.exp()                                        # (B, 1)
        # Avoid 0^n = NaN by clamping t (also matches SimulatorDecoder's
        # t=0 prepending convention).
        t_safe = t.unsqueeze(0).clamp_min(1e-8)                # (1, T)
        raw = k * t_safe.pow(n)                                # (B, T)
        # Q_max * min(raw, 1) — smooth max via clamp; gradient is the
        # subgradient at the kink.
        Q = Q_max * raw.clamp(max=1.0)
        # Set Q(t=0) to 0 explicitly.
        zero_mask = (t == 0.0).unsqueeze(0)
        Q = torch.where(zero_mask.expand_as(Q), torch.zeros_like(Q), Q)
        return Q.clamp(0.0, 1.0)

    def simulate_numpy(self, theta: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
        if theta.shape != (3,):
            raise ValueError(f"theta must have shape (3,), got {theta.shape}")
        log_k, n, Q_max = float(theta[0]), float(theta[1]), float(theta[2])
        k = float(np.exp(log_k))
        t = np.asarray(t_obs, dtype=float)
        t_safe = np.maximum(t, 1e-8)
        raw = k * np.power(t_safe, n)
        Q = Q_max * np.clip(raw, None, 1.0)
        Q = np.where(t == 0.0, 0.0, Q)
        return np.clip(Q, 0.0, 1.0)
