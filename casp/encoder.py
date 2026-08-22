"""CASPEncoder — produces (mu, U, sigma_active) per curve.

Inputs: features x (formulation + early Q concatenated).
Outputs: a low-rank Gaussian factorization of q(theta | x).

  mu in [prior_low, prior_high]^n_params   — squashed center
  U  in R^{n_params x r} orthonormal       — active subspace columns (QR'd)
  sigma_active in R^r positive             — std-dev along active dirs

The QR step makes U interpretable as an orthonormal basis spanning the
curve-conditional active subspace. It is fully differentiable in PyTorch.

Design choice: r is fixed at construction time. v1 (later) will learn r
per-curve with gumbel-softmax over {1..r_max}. For v0, fixed r is enough
to validate the architecture.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CASPEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_params: int,
        prior_low: Tensor,
        prior_high: Tensor,
        rank: int = 4,
        hidden: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
        sigma_min: float = 1e-3,
        sigma_max: float = 1.0,
        sigma_inactive_frac: float = 0.1,
    ) -> None:
        super().__init__()
        assert prior_low.shape == (n_params,)
        assert prior_high.shape == (n_params,)
        assert 1 <= rank <= n_params, f"rank {rank} must be in [1, n_params={n_params}]"

        self.n_params = n_params
        self.rank = rank
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Inactive-direction nugget: fixed scalar, expressed as a fraction
        # of per-parameter prior std. Without this, the model can place U
        # along sloppy (insensitive) directions of theta-space — sigma can
        # then grow without hurting recon, and U never finds the stiff
        # directions that actually matter. The nugget penalizes sloppy-U
        # because Q variance from inactive dirs becomes uncontrollable.
        prior_std_each = (prior_high - prior_low) / (12.0 ** 0.5)        # (P,)
        sigma_inactive = sigma_inactive_frac * prior_std_each            # (P,)
        self.register_buffer("sigma_inactive", sigma_inactive)

        layers: list[nn.Module] = []
        d_in = n_features
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        self.trunk = nn.Sequential(*layers)

        self.head_mu = nn.Linear(hidden, n_params)
        self.head_U = nn.Linear(hidden, n_params * rank)
        self.head_logsigma = nn.Linear(hidden, rank)

        # Prior box for sigmoid squash on mu.
        self.register_buffer("prior_low", prior_low.clone())
        self.register_buffer("prior_high", prior_high.clone())

        # Small init on U head so the initial subspace is near-identity-ish
        # rather than wildly arbitrary.
        nn.init.normal_(self.head_U.weight, std=0.02)
        nn.init.zeros_(self.head_U.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return dict with keys: mu (B, P), U (B, P, r), sigma (B, r).

        U columns are orthonormal: U^T U = I_r per batch element.
        """
        B = x.shape[0]
        P = self.n_params
        r = self.rank

        h = self.trunk(x)

        # mu: sigmoid-squashed into prior box (with small epsilon margin
        # so we don't sit on the boundary).
        eps_box = 1e-3
        s = torch.sigmoid(self.head_mu(h))
        s = eps_box + (1.0 - 2.0 * eps_box) * s
        mu = self.prior_low + (self.prior_high - self.prior_low) * s        # (B, P)

        # U_full: complete (B, P, P) orthonormal basis. First r columns are
        # the active subspace U_active; remaining (P - r) columns span the
        # inactive (sloppy) subspace. Built by padding the raw (P, r) head
        # with random Gaussians and doing complete QR; the random padding
        # is differentiable-friendly because gradients only flow through
        # the first r columns of the input to QR.
        prior_width = (self.prior_high - self.prior_low).clamp_min(1e-6)    # (P,)
        U_raw = self.head_U(h).reshape(B, P, r)
        U_raw = U_raw * prior_width.view(1, P, 1)
        # Pad with non-learned random columns to get a full (P, P) matrix.
        # The QR's "complete" mode then gives an orthonormal basis where
        # the first r columns span the same subspace as U_raw and the
        # remaining (P-r) columns are the inactive complement.
        pad = torch.randn(B, P, P - r, device=h.device, dtype=h.dtype)
        U_raw_full = torch.cat([U_raw, pad], dim=-1)                        # (B, P, P)
        Q, R = torch.linalg.qr(U_raw_full, mode="reduced")                  # Q: (B, P, P)
        sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        U_full = Q * sign                                                   # (B, P, P)
        U = U_full[..., :r]                                                 # active (B, P, r)
        U_inactive = U_full[..., r:]                                        # inactive (B, P, P-r)

        # sigma_active: softplus + bound. Bounded to keep KL well-conditioned
        # and prevent posterior collapse to delta or explosion.
        raw = self.head_logsigma(h)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)  # (B, r)

        return {"mu": mu, "U": U, "U_inactive": U_inactive, "sigma": sigma}

    def sample_theta(
        self, q: dict[str, Tensor], n_samples: int, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw theta from q. Returns shape (n_samples, B, P).

        theta = soft_box(mu + U @ diag(sigma) @ eps),   eps ~ N(0, I_r).

        soft_box is a softplus-based smooth clamp that is near-identity
        well inside the prior box and asymptotes at the boundary. This
        keeps gradients alive everywhere (unlike torch.clamp) and ensures
        the simulator never sees out-of-box theta that would trigger
        ODE solver stiffness (e.g. log_tau_burst < -3 makes the burst
        term singular at t=0).

        The soft_box biases the q distribution slightly: samples in the
        boundary tails are pulled inward. This is acceptable because the
        boundary represents physically unreasonable parameter values
        anyway. The KL term still operates on the unsquashed Gaussian
        in latent eps space, so the variational lower bound is well-defined.

        Inactive (sloppy) directions: not added here. They are handled by
        the ELBO's KL term, not by sampling.
        """
        mu = q["mu"]                                    # (B, P)
        U = q["U"]                                      # (B, P, r)
        U_inactive = q["U_inactive"]                    # (B, P, P-r)
        sigma = q["sigma"]                              # (B, r)
        B, P = mu.shape
        r = sigma.shape[-1]
        # Active: learned sigma along U columns.
        eps_a = torch.randn(n_samples, B, r, device=mu.device, generator=generator)
        delta_a = torch.einsum("bpr,sbr->sbp", U, eps_a * sigma.unsqueeze(0))
        # Inactive: fixed-scalar nugget along the orthogonal complement.
        # The nugget is in raw-theta units and propagates through any stiff
        # direction the model failed to identify; this is what forces U to
        # capture the actually sensitive directions of Q wrt theta.
        eps_i = torch.randn(n_samples, B, P - r, device=mu.device, generator=generator)
        # sigma_inactive is per-original-parameter; project into the
        # inactive basis: scalar = sqrt( mean_p (sigma_inactive_p^2 * U_inactive_p_i^2) ).
        # Simpler equivalent: keep a single scalar = mean(sigma_inactive),
        # treating the inactive subspace as homogeneous. Sufficient for v0.
        scalar_inactive = self.sigma_inactive.mean()
        delta_i = torch.einsum("bpi,sbi->sbp", U_inactive, eps_i * scalar_inactive)
        theta_raw = mu.unsqueeze(0) + delta_a + delta_i
        return _soft_box(theta_raw, self.prior_low, self.prior_high, beta=20.0)


def _soft_box(
    x: Tensor, low: Tensor, high: Tensor, beta: float = 20.0, margin_frac: float = 0.05
) -> Tensor:
    """Smooth soft clamp into [low + margin, high - margin].

    Identity-like inside the inner box, saturates at the inner boundary.
    The margin matters for stiffness-sensitive parameters (e.g. log_tau_burst
    at its lower bound makes the burst term singular at t=0); a 5% margin
    keeps the simulator stable while losing little expressive range.
    """
    width = (high - low)
    margin = margin_frac * width
    inner_low = low + margin
    inner_high = high - margin
    sp_lo = torch.nn.functional.softplus(beta * (x - inner_low))
    sp_hi = torch.nn.functional.softplus(beta * (x - inner_high))
    return inner_low + (sp_lo - sp_hi) / beta
