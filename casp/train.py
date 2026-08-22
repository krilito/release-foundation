"""CASP training loop.

Reusable across PLGA / Liposome / future mechanisms. The loop is
simulator-agnostic — it only sees encoder, decoder, loss, and a tensor
dataset of (X, y_full, oracle_mu_optional).

Three design choices made here, with their rationale:

  1. No hard clamp on sampled theta. Earlier prototypes clamped theta to
     prior box before decode, which kills gradients at the boundary. The
     simulator is numerically stable for theta excursions (exp / sigmoid /
     clamp inside the vector field), and the KL term + sigmoid-squashed
     mu give the encoder enough pressure to stay sensible. A small soft
     out-of-box penalty is added as belt-and-suspenders.

  2. Warm-start weight decays linearly to 0 over `warm_decay_epochs`. The
     warm-start exists to anchor mu near oracle at the start, not to
     replace learning. Decay so the final model is data-driven, not
     oracle-driven.

  3. sigma_obs (observation noise) is learnable, scalar, log-parameterized.
     Fixing it at 0.05 made recon dominate by ~25x in the v0 forward.
     Letting the model learn it lets the recon/KL balance find itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    n_samples: int = 4                # MC draws per forward
    warm_decay_epochs: int = 50       # warm_w -> 0 by this epoch
    oob_w: float = 1e-2               # out-of-box soft penalty weight
    learn_sigma_obs: bool = True
    log_sigma_obs_init: float = -3.0  # sigma_obs = exp(-3) ~= 0.05
    log_every: int = 10
    seed: int = 0


class _LearnableSigmaObs(nn.Module):
    def __init__(self, log_init: float) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(float(log_init)))

    def sigma(self) -> Tensor:
        # Clamp to avoid log(0) instabilities; range covers 0.01..1.0.
        return torch.clamp(self.log_sigma, min=-4.6, max=0.0).exp()


def _oob_penalty(theta: Tensor, prior_low: Tensor, prior_high: Tensor) -> Tensor:
    """Soft penalty for theta outside prior box. Smooth, differentiable."""
    # theta: (S, B, P) or (B, P).  prior_*: (P,)
    over = (theta - prior_high).clamp(min=0.0)
    under = (prior_low - theta).clamp(min=0.0)
    return (over.pow(2) + under.pow(2)).sum(dim=-1).mean()


def train_casp(
    encoder: nn.Module,
    decoder: nn.Module,
    loss_fn: nn.Module,
    X_train: Tensor,                # (N, F)
    y_train: Tensor,                # (N, T)
    t_grid: Tensor,                 # (T,)
    oracle_mu_train: Tensor | None, # (N, P) or None
    prior_low: Tensor,              # (P,)
    prior_high: Tensor,             # (P,)
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
    log_fn: callable = print,
) -> dict[str, list[float]]:
    """Train CASP on a single fold. Returns history dict of per-epoch losses.

    Modifies encoder, decoder, loss_fn in place.
    """
    torch.manual_seed(cfg.seed)
    encoder.to(device); decoder.to(device); loss_fn.to(device)
    X_train = X_train.to(device); y_train = y_train.to(device)
    t_grid = t_grid.to(device)
    prior_low = prior_low.to(device); prior_high = prior_high.to(device)
    if oracle_mu_train is not None:
        oracle_mu_train = oracle_mu_train.to(device)

    sigma_obs_module = _LearnableSigmaObs(cfg.log_sigma_obs_init).to(device)
    if not cfg.learn_sigma_obs:
        sigma_obs_module.log_sigma.requires_grad_(False)

    params = list(encoder.parameters()) + list(sigma_obs_module.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    N = X_train.shape[0]
    history: dict[str, list[float]] = {
        "total": [], "recon": [], "kl_active": [], "warm": [], "oob": [], "sigma_obs": []
    }

    for epoch in range(cfg.epochs):
        # warm-start weight linear decay to 0
        if cfg.warm_decay_epochs > 0:
            warm_w = loss_fn.warm_w_init * max(0.0, 1.0 - epoch / cfg.warm_decay_epochs)
        else:
            warm_w = loss_fn.warm_w_init

        encoder.train()
        perm = torch.randperm(N, device=device)
        ep_logs: dict[str, list[float]] = {k: [] for k in history}

        for s in range(0, N, cfg.batch_size):
            sel = perm[s:s + cfg.batch_size]
            x_b = X_train[sel]
            y_b = y_train[sel]
            oracle_b = oracle_mu_train[sel] if oracle_mu_train is not None else None

            q = encoder(x_b)
            theta = encoder.sample_theta(q, n_samples=cfg.n_samples)        # (S, B, P)
            Q_hat = decoder(theta, t_grid)                                  # (S, B, T)

            # Update loss_fn's sigma_obs from learnable scalar each step.
            loss_fn.sigma_obs = float(sigma_obs_module.sigma().detach())    # for logging
            sigma_obs = sigma_obs_module.sigma()

            # Inline recon with learnable sigma_obs (loss_fn.forward uses a
            # static value; bypass for sigma_obs gradient flow).
            S_, B_, T_ = Q_hat.shape
            y_exp = y_b.unsqueeze(0).expand(S_, B_, T_)
            recon = ((Q_hat - y_exp).pow(2) / sigma_obs.pow(2)).mean()
            # KL on active dirs — reuse loss_fn machinery
            out = loss_fn(q, Q_hat, y_b, oracle_mu=oracle_b, warm_w=warm_w)
            kl_active = out["kl_active"]
            orth = out["orth"]
            warm = out["warm"]

            # Out-of-box penalty
            oob = _oob_penalty(theta, prior_low, prior_high)

            total = (
                loss_fn.recon_w * recon
                + loss_fn.kl_w * kl_active
                + loss_fn.orth_w * orth
                + warm_w * warm
                + cfg.oob_w * oob
            )
            # Add a tiny log-sigma_obs prior to keep sigma_obs from drifting
            # too high (which would let recon get arbitrarily cheap).
            total = total + 1e-3 * sigma_obs_module.log_sigma.pow(2)

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            ep_logs["total"].append(float(total.detach()))
            ep_logs["recon"].append(float(recon.detach()))
            ep_logs["kl_active"].append(float(kl_active.detach()))
            ep_logs["warm"].append(float(warm.detach()))
            ep_logs["oob"].append(float(oob.detach()))
            ep_logs["sigma_obs"].append(float(sigma_obs.detach()))

        for k in history:
            history[k].append(sum(ep_logs[k]) / max(len(ep_logs[k]), 1))

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            log_fn(
                f"[ep {epoch+1:3d}] total={history['total'][-1]:+.3f}  "
                f"recon={history['recon'][-1]:+.3f}  "
                f"kl={history['kl_active'][-1]:+.3f}  "
                f"warm={history['warm'][-1]:+.3f} (w={warm_w:.2f})  "
                f"oob={history['oob'][-1]:+.3f}  "
                f"sigma_obs={history['sigma_obs'][-1]:.3f}"
            )

    return history
