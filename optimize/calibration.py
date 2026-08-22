"""Posterior temperature scaling for SBI calibration.

Problem: NPE posteriors can be overconfident (too narrow) or
underconfident (too wide). SBC's c2st_ranks detects rank bias but
not variance miscalibration. Coverage plots (90% PI should cover
90% of true curves) are the gold standard.

Solution: learn a scalar temperature tau that rescales the posterior
width around the point estimate. tau > 1 widens, tau < 1 narrows.

    theta_calibrated = mu + (theta_raw - mu) * tau

Calibrated by minimizing CRPS on a held-out calibration set.
CRPS is preferred over NLL because it is proper: the optimal
temperature minimizes the expected CRPS when the posterior is
well-specified.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class TemperatureScaler(nn.Module):
    """Learnable scalar temperature for posterior calibration."""

    def __init__(self, init_tau: float = 1.0) -> None:
        super().__init__()
        # log_tau so tau is always positive via exp
        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def forward(self, theta: Tensor, mu: Tensor) -> Tensor:
        """Rescale theta around mu by temperature.

        Args:
            theta: (S, P) posterior samples.
            mu: (P,) point estimate.
        Returns:
            theta_cal: (S, P) calibrated samples.
        """
        tau = self.tau
        return mu + (theta - mu) * tau


def crps_empirical(theta_samples: Tensor, theta_true: Tensor) -> Tensor:
    """Empirical CRPS between posterior samples and a true value.

    CRPS = E|theta - theta_true| - 0.5 * E|theta - theta'|
    where theta, theta' are iid from the posterior.

    Lower is better. Proper scoring rule: optimal when the
    predictive distribution matches the true distribution.

    Args:
        theta_samples: (S, P) posterior samples.
        theta_true: (P,) true parameter.
    Returns:
        crps: scalar.
    """
    S = theta_samples.shape[0]
    # Term 1: mean absolute error
    abs_err = (theta_samples - theta_true).abs().mean()
    # Term 2: pairwise mean absolute difference
    # For efficiency, subsample if S is large
    if S > 200:
        idx = torch.randperm(S)[:200]
        samples_sub = theta_samples[idx]
    else:
        samples_sub = theta_samples
    S_sub = samples_sub.shape[0]
    # pairwise diff: (S, S, P) -> mean |diff|
    diff = samples_sub.unsqueeze(0) - samples_sub.unsqueeze(1)  # (S, S, P)
    pairwise_abs = diff.abs().mean()
    return abs_err - 0.5 * pairwise_abs


def calibrate_temperature(
    theta_samples: Tensor,     # (N, S, P) posterior samples per curve
    theta_true: Tensor,        # (N, P) ground truth (e.g., oracle fit)
    n_steps: int = 500,
    lr: float = 0.05,
    init_tau: float = 1.0,
) -> float:
    """Learn optimal temperature by minimizing mean CRPS.

    Returns the calibrated tau value.
    """
    scaler = TemperatureScaler(init_tau)
    optimizer = torch.optim.Adam(scaler.parameters(), lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        total_crps = torch.tensor(0.0)
        mu = theta_true  # use oracle as point estimate for calibration
        for i in range(theta_samples.shape[0]):
            cal = scaler(theta_samples[i], mu[i])
            total_crps = total_crps + crps_empirical(cal, theta_true[i])
        loss = total_crps / theta_samples.shape[0]
        loss.backward()
        optimizer.step()

    return float(scaler.tau.detach())


def coverage_at_level(
    theta_samples: Tensor,  # (S, P)
    theta_true: Tensor,     # (P,)
    level: float = 0.9,
) -> bool:
    """Check if theta_true falls within the level% PI of theta_samples."""
    lo = torch.quantile(theta_samples, (1 - level) / 2, dim=0)
    hi = torch.quantile(theta_samples, 1 - (1 - level) / 2, dim=0)
    return bool(((theta_true >= lo) & (theta_true <= hi)).all())


def effective_sample_size(theta_samples: Tensor) -> float:
    """Estimate ESS via the pairwise-distance heuristic.

    ESS = (sum ||theta_i - theta_mean||)^2 / sum ||theta_i - theta_mean||^2
    This is the Kish ESS applied to the L2-norm of centered samples.
    """
    mu = theta_samples.mean(dim=0, keepdim=True)
    dists = (theta_samples - mu).norm(dim=1)  # (S,)
    return float(dists.sum() ** 2 / (dists ** 2).sum())
