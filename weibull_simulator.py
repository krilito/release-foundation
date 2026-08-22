"""Weibull release simulator — for Liposome IVR and other mechanisms
where the kinetic state is well captured by Weibull (alpha, beta).

Implements the same ReleaseSimulator interface as PLGABiphasic so
casp/fib.py treats it identically (simulator-agnostic FIB construction).

The Weibull form is closed-form:

    Q(t) = 1 - exp(-t^beta / alpha),  t >= 0

Parameters in log-space for positivity:
    theta = [log_alpha, log_beta]

The prior is uniform on log-space, with bounds chosen to cover the
Yanes et al. accelerated_IVR fitted Weibull parameter distribution.

This module is the second concrete subclass of `ReleaseSimulator`
required for the cross-mechanism CASP/FIB demonstration. Both
PLGABiphasic (9 params) and WeibullSimulator (2 params) use the same
`fib_posterior(...)` machinery without modification.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor
from torch.distributions import Distribution, Independent, Uniform

from simulator import ReleaseSimulator


class WeibullSimulator(ReleaseSimulator):
    """Weibull dissolution-style release.

    Q(t) = 1 - exp(-t^beta / alpha)

    Coverage of the Yanes accelerated_IVR fitted parameter range
    (n=169 fitted Weibull curves):
        alpha ~ [0.5, 500]    -> log_alpha ~ [-0.7, 6.2]
        beta  ~ [0.2, 3.0]    -> log_beta  ~ [-1.6, 1.1]

    We pad both bounds slightly to admit edge cases.
    """

    n_params = 2
    param_names = ["log_alpha", "log_beta"]

    def prior(self) -> Distribution:
        lows = torch.tensor([-2.0, -2.0], dtype=torch.float32)
        highs = torch.tensor([8.0, 2.0], dtype=torch.float32)
        return Independent(Uniform(lows, highs), 1)

    def simulate(self, theta: Tensor, t: Tensor, x: Tensor | None = None) -> Tensor:
        """theta: (B, 2). t: (T,). Returns (B, T)."""
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError(f"theta must have shape (B, 2), got {tuple(theta.shape)}")
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D, got {tuple(t.shape)}")
        log_alpha = theta[:, 0:1]
        log_beta = theta[:, 1:2]
        alpha = log_alpha.exp().clamp_min(1e-8)             # (B, 1)
        beta = log_beta.exp().clamp_min(1e-8)               # (B, 1)
        # t may include 0 (matches SimulatorDecoder convention); avoid 0^beta
        # which is well-defined as 0 for beta > 0 but produces NaN gradients.
        t_safe = t.unsqueeze(0).clamp_min(1e-8)             # (1, T)
        # Q = 1 - exp(-t^beta / alpha); use stable form via expm1.
        x_inner = (t_safe.pow(beta)) / alpha                # (B, T)
        Q = -torch.expm1(-x_inner)                          # = 1 - exp(-x_inner)
        # Where t == 0, set Q to 0 explicitly.
        zero_mask = (t == 0.0).unsqueeze(0)
        Q = torch.where(zero_mask.expand_as(Q), torch.zeros_like(Q), Q)
        return Q.clamp(0.0, 1.0)

    def simulate_numpy(self, theta: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
        if theta.shape != (2,):
            raise ValueError(f"theta must have shape (2,), got {theta.shape}")
        alpha = max(float(np.exp(theta[0])), 1e-8)
        beta = max(float(np.exp(theta[1])), 1e-8)
        t = np.asarray(t_obs, dtype=float)
        t_safe = np.maximum(t, 1e-8)
        Q = 1.0 - np.exp(-(t_safe ** beta) / alpha)
        Q = np.where(t == 0.0, 0.0, Q)
        return np.clip(Q, 0.0, 1.0)
