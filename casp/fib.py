"""FIB — Fisher-Information Bottleneck posterior factorization.

Per-curve construction of (mu, U, sigma_active, sigma_inactive) from:
  - a point estimate theta_hat (e.g. from RF -> theta in 38d),
  - the differentiable simulator (any ReleaseSimulator subclass),
  - prior precision regularizer alpha (small ridge against singular F),
  - target rank r (active subspace dimension).

The Fisher information matrix F = J^T J / sigma_obs^2 at theta_hat is the
canonical local identifiability object (Cox & Reid 1987). Its top-r
eigenvectors define the active subspace; the reciprocal-sqrt eigenvalues
give Laplace-approximation standard deviations on those directions.

This module is the *deterministic* alternative to plan_60's variational
encoder. No training; closed-form per-curve construction. Works on any
dataset size that supports a reasonable point estimator.
"""
from __future__ import annotations

import torch
from torch import Tensor

from simulator import ReleaseSimulator


def simulator_jacobian(
    simulator: ReleaseSimulator,
    theta: Tensor,                # (P,)
    t: Tensor,                    # (T,)
) -> Tensor:
    """d Q_t / d theta_p at theta. Shape (T, P).

    Uses torch.func.jacrev for batched-by-output. Falls back to manual
    autograd if jacrev not available.
    """
    theta = theta.detach().clone().requires_grad_(True)

    def fwd(th: Tensor) -> Tensor:
        return simulator.simulate(th.unsqueeze(0), t).squeeze(0)        # (T,)

    try:
        J = torch.func.jacrev(fwd)(theta)                               # (T, P)
    except Exception:
        # Manual fallback
        Q = fwd(theta)
        T_ = Q.shape[0]
        cols = []
        for ti in range(T_):
            grad = torch.autograd.grad(Q[ti], theta, retain_graph=(ti < T_ - 1))[0]
            cols.append(grad)
        J = torch.stack(cols, dim=0)
    return J.detach()


def fib_posterior(
    simulator: ReleaseSimulator,
    theta_hat: Tensor,            # (P,)
    t: Tensor,                    # (T,)
    sigma_obs: float = 0.05,
    rank: int = 4,
    alpha: float = 1e-2,
    sigma_0_frac: float = 0.05,
    prior_low: Tensor | None = None,
    prior_high: Tensor | None = None,
) -> dict[str, Tensor]:
    """Construct (mu, U, sigma_active, U_inactive, sigma_inactive_scalar).

    Returns a dict matching the CASP encoder output convention, so the
    rest of the pipeline (sampling, decoding, PI evaluation) is shared.

    Steps:
      1. J = d Q/d theta at theta_hat            shape (T, P)
      2. F = J^T J / sigma_obs^2                 shape (P, P)
      3. F_reg = F + alpha * Sigma_prior^{-1}    shape (P, P)
      4. eigh(F_reg) -> (lambda_i, V_i)          sorted ascending; flip to descending
      5. top-r: U = V[:, :r], sigma = 1/sqrt(lambda[:r])
      6. inactive: U_inactive = V[:, r:]; sigma_0 = sigma_0_frac * prior_std
    """
    P = simulator.n_params
    J = simulator_jacobian(simulator, theta_hat, t)                     # (T, P)
    F = (J.T @ J) / (sigma_obs ** 2)                                    # (P, P)

    # Prior precision: assume diagonal with std = prior_width / sqrt(12)
    # (Uniform prior std). If priors not provided, use identity * alpha.
    if prior_low is not None and prior_high is not None:
        prior_std = (prior_high - prior_low) / (12.0 ** 0.5)
        prior_prec_diag = 1.0 / (prior_std ** 2 + 1e-12)
        F_reg = F + alpha * torch.diag(prior_prec_diag)
    else:
        F_reg = F + alpha * torch.eye(P, device=F.device, dtype=F.dtype)
        prior_std = torch.ones(P, device=F.device, dtype=F.dtype)

    # Eigendecomp; symmetric so use eigh.
    F_reg_sym = 0.5 * (F_reg + F_reg.T)
    eigvals, eigvecs = torch.linalg.eigh(F_reg_sym)                     # ascending
    # Flip to descending (high info -> stiff -> active).
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # Numerical safety: clamp eigvals to be positive (the ridge ensures
    # they should be, but rounding noise can push tiny ones to ~0).
    eigvals = eigvals.clamp(min=1e-8)

    r = min(rank, P)
    U_active = eigvecs[:, :r]                                           # (P, r)
    sigma_active = 1.0 / torch.sqrt(eigvals[:r])                        # (r,)

    U_inactive = eigvecs[:, r:]                                         # (P, P-r)
    sigma_inactive_scalar = sigma_0_frac * prior_std.mean()             # scalar

    # Cap sigma_active at the prior std along that direction. Without this
    # cap, a very-poorly-identified direction inside the top-r (low lambda)
    # could give sigma >> prior std, which is non-physical.
    prior_std_active = torch.sqrt((U_active.pow(2) * prior_std.pow(2).unsqueeze(-1)).sum(dim=0))
    sigma_active = torch.minimum(sigma_active, prior_std_active)

    return {
        "mu": theta_hat.detach(),
        "U": U_active,
        "U_inactive": U_inactive,
        "sigma": sigma_active,
        "sigma_inactive_scalar": sigma_inactive_scalar,
        "eigvals_full": eigvals,
        "eigvecs_full": eigvecs,
    }


def sample_theta_fib(
    q: dict[str, Tensor], n_samples: int,
    prior_low: Tensor | None = None,
    prior_high: Tensor | None = None,
    soft_box_beta: float = 20.0,
    soft_box_margin: float = 0.05,
) -> Tensor:
    """Sample theta ~ q. Returns (S, P) for single curve, or (S, B, P) for batch.

    Soft-clamps into prior box to keep simulator stable (same soft_box as encoder.py).
    """
    mu = q["mu"]
    U = q["U"]
    U_inactive = q["U_inactive"]
    sigma_active = q["sigma"]
    sigma_inactive = q["sigma_inactive_scalar"]
    P = mu.shape[-1]
    r = sigma_active.shape[-1]
    if mu.dim() == 1:
        eps_a = torch.randn(n_samples, r, device=mu.device, dtype=mu.dtype)
        eps_i = torch.randn(n_samples, P - r, device=mu.device, dtype=mu.dtype)
        delta_a = (U @ (eps_a * sigma_active).T).T                       # (S, P)
        delta_i = (U_inactive @ (eps_i * sigma_inactive).T).T            # (S, P)
        theta = mu.unsqueeze(0) + delta_a + delta_i
    else:
        # batched
        B = mu.shape[0]
        eps_a = torch.randn(n_samples, B, r, device=mu.device, dtype=mu.dtype)
        eps_i = torch.randn(n_samples, B, P - r, device=mu.device, dtype=mu.dtype)
        delta_a = torch.einsum("bpr,sbr->sbp", U, eps_a * sigma_active.unsqueeze(0))
        delta_i = torch.einsum("bpi,sbi->sbp", U_inactive, eps_i * sigma_inactive)
        theta = mu.unsqueeze(0) + delta_a + delta_i

    if prior_low is not None and prior_high is not None:
        width = (prior_high - prior_low)
        margin = soft_box_margin * width
        inner_low = prior_low + margin
        inner_high = prior_high - margin
        sp_lo = torch.nn.functional.softplus(soft_box_beta * (theta - inner_low))
        sp_hi = torch.nn.functional.softplus(soft_box_beta * (theta - inner_high))
        theta = inner_low + (sp_lo - sp_hi) / soft_box_beta
    return theta
