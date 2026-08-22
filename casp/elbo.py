"""CASPLoss — ELBO with KL on inactive directions.

L(x, y_full) =
    recon_w  * E_{theta ~ q}[ ||y_full - decode(theta, t)||^2 / sigma_obs^2 ]
  + kl_w     * KL[ q_active || N(0, sigma_prior_active^2) ]    # active dirs
  + nug_w    * KL[ N(0, sigma_0^2 I) || N(0, sigma_prior_inactive^2 I) ]
  + orth_w   * ||U^T U - I_r||_F^2
  + warm_w   * warm_start_penalty(mu, U; oracle)

The recon term is the data fit. The KL on active dirs prevents posterior
collapse (sigma_active -> 0). The inactive KL is *fixed* (no learnable
sigma there), so it shows up as a constant offset — kept for completeness
of the ELBO form but doesn't affect gradients (omitted from the actual
loss to keep the loss scale interpretable).

The orthogonality term is a safety net: torch.linalg.qr should already
return orthonormal U, but small numerical drift over many forwards can
add up. Cheap insurance, kw=1e-3.

Warm-start: optional. When `oracle_mu` and `oracle_active_mask` are
provided, the loss adds a penalty pulling mu toward oracle and U columns
toward the oracle active-direction basis. Decayed externally by the
training loop.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _kl_diag_gauss(mu_q: Tensor, sigma_q: Tensor, sigma_p: Tensor) -> Tensor:
    """KL( N(mu_q, sigma_q^2) || N(0, sigma_p^2) ) elementwise, per-batch.

    Shapes: mu_q (B, K), sigma_q (B, K), sigma_p (K,) or scalar.
    Returns (B,) tensor of KL summed over K.
    """
    if sigma_p.dim() == 0:
        sigma_p = sigma_p.expand_as(sigma_q[0])
    var_q = sigma_q.pow(2)
    var_p = sigma_p.pow(2).unsqueeze(0)
    kl = 0.5 * (
        (mu_q.pow(2) + var_q) / var_p
        - 1.0
        + var_p.log()
        - var_q.log()
    )
    return kl.sum(dim=-1)


class CASPLoss(nn.Module):
    def __init__(
        self,
        prior_low: Tensor,
        prior_high: Tensor,
        sigma_obs: float = 0.05,
        recon_w: float = 1.0,
        kl_w: float = 1.0,
        orth_w: float = 1e-3,
        warm_w_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.recon_w = recon_w
        self.kl_w = kl_w
        self.orth_w = orth_w
        self.warm_w_init = warm_w_init
        self.sigma_obs = sigma_obs

        # Prior std-dev per parameter dim, taken as (prior_width / sqrt(12))
        # — std of a uniform distribution. Used in the KL term.
        prior_width = (prior_high - prior_low).clamp_min(1e-6)
        prior_std = prior_width / (12.0 ** 0.5)            # (P,)
        self.register_buffer("prior_std", prior_std)
        self.register_buffer("prior_low", prior_low.clone())
        self.register_buffer("prior_high", prior_high.clone())

    def forward(
        self,
        q: dict[str, Tensor],
        Q_pred: Tensor,                          # (S, B, T) — simulated curves from theta samples
        y_obs: Tensor,                           # (B, T)    — observed curves on same t grid
        obs_mask: Tensor | None = None,          # (B, T) bool, optional. True = include in recon.
        warm_w: float | None = None,             # if None, use self.warm_w_init
        oracle_mu: Tensor | None = None,         # (B, P), optional warm-start target for mu
    ) -> dict[str, Tensor]:
        """Return a dict with keys: total, recon, kl_active, orth, warm.
        Each entry is a scalar except .total. .total is the optimization target.
        """
        S, B, T = Q_pred.shape
        device = Q_pred.device

        # --- reconstruction (MC over S samples) ---
        y_exp = y_obs.unsqueeze(0).expand(S, B, T)
        sq = (Q_pred - y_exp).pow(2)             # (S, B, T)
        if obs_mask is not None:
            mask = obs_mask.unsqueeze(0).expand(S, B, T).float()
            sq = sq * mask
            denom = mask.sum(dim=-1).clamp_min(1.0)   # (S, B)
            per_curve = sq.sum(dim=-1) / denom        # (S, B)
        else:
            per_curve = sq.mean(dim=-1)               # (S, B)
        recon = (per_curve / (self.sigma_obs ** 2)).mean()

        # --- KL on active directions ---
        # q on active dirs has mean 0 (the displacement, not mu itself)
        # in the active basis; this KL only governs sigma_active not
        # collapsing. We use prior_std projected onto U directions as the
        # "active prior std" — a per-curve quantity.
        mu = q["mu"]                              # (B, P)
        U = q["U"]                                # (B, P, r)
        sigma = q["sigma"]                        # (B, r)
        # prior std along active dirs: ||diag(prior_std) U||_col
        prior_std_active = torch.sqrt(
            (U.pow(2) * self.prior_std.view(1, -1, 1).pow(2)).sum(dim=1)
        )                                         # (B, r)
        mu_q_active = torch.zeros_like(sigma)     # displacement = 0 by construction
        kl_active = _kl_diag_gauss(mu_q_active, sigma, prior_std_active.mean(dim=0))
        kl_active = kl_active.mean()

        # --- orthogonality safety net ---
        UtU = torch.einsum("bpr,bps->brs", U, U)  # (B, r, r)
        eye = torch.eye(U.shape[-1], device=device).unsqueeze(0)
        orth = (UtU - eye).pow(2).sum(dim=(-1, -2)).mean()

        # --- warm-start (optional) ---
        if oracle_mu is None:
            warm = torch.zeros((), device=device)
        else:
            # MSE in prior-normalized coordinates so all dims contribute
            # comparably regardless of prior width.
            width = (self.prior_high - self.prior_low).clamp_min(1e-6).unsqueeze(0)
            mu_norm = (mu - self.prior_low) / width
            oracle_norm = (oracle_mu - self.prior_low) / width
            warm = (mu_norm - oracle_norm).pow(2).mean()
        w_warm = self.warm_w_init if warm_w is None else warm_w

        total = (
            self.recon_w * recon
            + self.kl_w * kl_active
            + self.orth_w * orth
            + w_warm * warm
        )
        return {
            "total": total,
            "recon": recon.detach(),
            "kl_active": kl_active.detach(),
            "orth": orth.detach(),
            "warm": warm.detach(),
        }
