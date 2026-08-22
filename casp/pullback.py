"""Simulator pull-back: per-curve target Gaussians for PCA-P.

For each real observed curve c = (t_obs, q_obs), construct a target
low-rank Gaussian over theta:

    1. Sample N candidate theta_n from the prior
    2. Simulate Q_n = simulator(theta_n, c.t_obs)
    3. Compute likelihood weights w_n ~ exp(- ||Q_n - q_obs||^2 / 2 sigma^2)
    4. Top-k selection (SIR step) keeps the curve-compatible candidates
    5. Refine: optional local jitter + reweight around top-k center
    6. Empirical low-rank Gaussian fit: weighted mean, covariance, eigh
    7. Return (mu_target, U_target, sigma_active_target)

The output object has the same (mu, U, sigma) shape as CASPEncoder's
forward, so a downstream amortized regressor can train against it
directly with a KL loss.

This module is simulator-agnostic — takes a ReleaseSimulator instance.
The same pull-back works for PLGABiphasic, WeibullSimulator, future
mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from simulator import ReleaseSimulator


@dataclass
class PullbackTarget:
    mu: np.ndarray             # (P,) — weighted mean theta
    U: np.ndarray              # (P, r) — orthonormal top-r covariance eigvecs
    sigma_active: np.ndarray   # (r,) — sqrt eigvals on top-r dirs
    log_evidence: float        # log p(curve) ~ logsumexp(log_w) - log(N)
    ess: float                 # effective sample size
    n_kept: int                # top-k after SIR
    method: str                # "sir" or "sir+local"


def _simulate_batch(
    simulator: ReleaseSimulator,
    theta: np.ndarray,            # (N, P)
    t_obs: np.ndarray,            # (T,)
    batch_size: int = 128,
) -> np.ndarray:                  # (N, T)
    """Run simulator on N theta candidates at curve's native t_obs.

    Always prepends t=0 for numerical correctness (the SimulatorDecoder
    convention) and drops the first output column.
    """
    if t_obs[0] != 0.0:
        t_full = np.concatenate([[0.0], t_obs])
        slice_start = 1
    else:
        t_full = t_obs.copy()
        slice_start = 0
    t_full_t = torch.tensor(t_full, dtype=torch.float32)

    N = theta.shape[0]
    out = np.zeros((N, len(t_obs)), dtype=np.float32)
    for s in range(0, N, batch_size):
        chunk = torch.tensor(theta[s:s + batch_size], dtype=torch.float32)
        with torch.no_grad():
            Q = simulator.simulate(chunk, t_full_t).cpu().numpy()
        out[s:s + batch_size] = Q[:, slice_start:]
    return out


def _weighted_lowrank_gaussian(
    theta: np.ndarray,            # (M, P) - kept candidates
    w: np.ndarray,                # (M,)  - normalized weights
    rank: int,
    prior_low: np.ndarray | None = None,
    prior_high: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, U, sigma_active) from weighted samples.

    mu in prior-units, U orthonormal columns in prior-units,
    sigma_active = sqrt of top-r eigenvalues of the weighted covariance.
    """
    mu = (theta * w[:, None]).sum(axis=0)                    # (P,)
    delta = theta - mu[None, :]                              # (M, P)
    Sigma = (delta * w[:, None]).T @ delta                   # (P, P)
    Sigma = 0.5 * (Sigma + Sigma.T)
    # Eigendecomp; clamp tiny / negative eigvals for numerical stability.
    eigvals, eigvecs = np.linalg.eigh(Sigma)                 # ascending
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]      # descending
    eigvals = np.clip(eigvals, 1e-10, None)
    r = min(rank, eigvecs.shape[1])
    U = eigvecs[:, :r].astype(np.float32)
    sigma_active = np.sqrt(eigvals[:r]).astype(np.float32)
    # If prior bounds available, cap sigma at prior std along that direction.
    if prior_low is not None and prior_high is not None:
        prior_std = (prior_high - prior_low) / np.sqrt(12.0)
        prior_std_along_U = np.sqrt(((U ** 2) * (prior_std ** 2)[:, None]).sum(axis=0))
        sigma_active = np.minimum(sigma_active, prior_std_along_U.astype(np.float32))
    return mu.astype(np.float32), U, sigma_active


def pullback_target(
    simulator: ReleaseSimulator,
    t_obs: np.ndarray,
    q_obs: np.ndarray,
    n_prior_samples: int = 2000,
    sigma_likelihood: float = 0.05,
    top_k: int = 100,
    rank: int = 4,
    seed: int = 0,
    use_local_refine: bool = True,
    local_jitter: float = 0.10,        # fraction of prior std
    n_local: int = 1000,
) -> PullbackTarget:
    """Construct one target Gaussian for a single curve.

    Pipeline:
      1. Sample n_prior_samples theta ~ uniform prior.
      2. Compute log-likelihoods on (t_obs, q_obs).
      3. Keep top_k.
      4. If use_local_refine: re-sample n_local around top_k mean with
         jitter, reweight, merge with original top_k by weight.
      5. Fit low-rank Gaussian to weighted survivors.

    The two-stage (prior + local) sampler matters because the 9-D PLGA
    prior is spread over orders-of-magnitude rate constants; uniform
    sampling gives very low ESS for tightly-constrained curves. Local
    refinement around early survivors is a cheap importance-tempering
    trick that boosts effective ESS by 10-100x.
    """
    rng = np.random.default_rng(seed)
    P = simulator.n_params

    prior = simulator.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    prior_std = (highs - lows) / np.sqrt(12.0)

    # Stage 1: prior samples
    theta_prior = rng.uniform(lows, highs, size=(n_prior_samples, P)).astype(np.float32)
    Q_prior = _simulate_batch(simulator, theta_prior, t_obs)
    log_w_prior = -((Q_prior - q_obs[None, :]) ** 2).sum(axis=1) / (2 * sigma_likelihood ** 2)

    # Top-k from prior pool
    idx_top = np.argsort(log_w_prior)[-top_k:]
    theta_top = theta_prior[idx_top]
    log_w_top = log_w_prior[idx_top]

    method = "sir"
    if use_local_refine:
        # Stage 2: local jitter around top-k mean (proposal = Gaussian in
        # prior-scale units). Normalize log-weights by stable max-subtract,
        # use top-k as the center distribution.
        w_top = np.exp(log_w_top - log_w_top.max())
        w_top = w_top / w_top.sum()
        center = (theta_top * w_top[:, None]).sum(axis=0)
        cov_scale = (local_jitter * prior_std) ** 2
        theta_local = rng.multivariate_normal(
            center, np.diag(cov_scale), size=n_local
        ).astype(np.float32)
        # Clamp to prior box
        theta_local = np.clip(theta_local, lows + 1e-4, highs - 1e-4)
        Q_local = _simulate_batch(simulator, theta_local, t_obs)
        log_w_local = -((Q_local - q_obs[None, :]) ** 2).sum(axis=1) / (2 * sigma_likelihood ** 2)
        # Merge: combine prior-top-k with local samples by weight
        theta_combined = np.concatenate([theta_top, theta_local], axis=0)
        log_w_combined = np.concatenate([log_w_top, log_w_local])
        # SIR on the merged pool
        idx_merge = np.argsort(log_w_combined)[-top_k:]
        theta_top = theta_combined[idx_merge]
        log_w_top = log_w_combined[idx_merge]
        method = "sir+local"

    # Normalize weights
    log_w_norm = log_w_top - log_w_top.max()
    w = np.exp(log_w_norm)
    w_norm = w / w.sum()
    ess = float(1.0 / (w_norm ** 2).sum())

    # Log evidence (up to a constant)
    log_evidence = float(np.log(np.mean(np.exp(log_w_top - log_w_top.max()))) + log_w_top.max())

    # Fit low-rank Gaussian
    mu, U, sigma_active = _weighted_lowrank_gaussian(
        theta_top.astype(np.float64), w_norm, rank=rank,
        prior_low=lows, prior_high=highs,
    )

    return PullbackTarget(
        mu=mu, U=U, sigma_active=sigma_active,
        log_evidence=log_evidence, ess=ess,
        n_kept=int(top_k), method=method,
    )
