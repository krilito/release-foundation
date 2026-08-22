"""Tests for simulator.py.

These are *physics sanity checks*, not unit tests in the traditional sense.
A green test suite here means the ODE produces monotone, bounded,
biphasic-shaped release curves — the minimum bar before we point an
amortized SBI network at it.
"""
from __future__ import annotations

import torch

from simulator import PLGABiphasic


def test_prior_has_right_dimensionality() -> None:
    sim = PLGABiphasic()
    theta = sim.sample_prior(16)
    assert theta.shape == (16, sim.n_params)
    assert len(sim.param_names) == sim.n_params


def test_curves_are_in_unit_interval() -> None:
    sim = PLGABiphasic()
    theta = sim.sample_prior(32)
    t = torch.linspace(0.0, 60.0, 50)
    Q = sim.simulate(theta, t)
    assert Q.shape == (32, 50)
    assert (Q >= 0.0).all()
    assert (Q <= 1.0).all()


def test_curves_are_monotone_nondecreasing() -> None:
    """Q(t) must never decrease. If it does, the (1 − Q) saturation factor
    in the ODE has been broken or there is a numerical instability."""
    sim = PLGABiphasic()
    theta = sim.sample_prior(64)
    t = torch.linspace(0.0, 90.0, 100)
    Q = sim.simulate(theta, t)
    diffs = Q[:, 1:] - Q[:, :-1]
    # Allow a tiny numerical tolerance for adaptive ODE solver wiggle.
    assert (diffs >= -1e-4).all(), f"min diff: {diffs.min().item()}"


def test_initial_value_is_zero() -> None:
    """ADR-026: Q(0) = 0 by construction; burst is now in the rate equation."""
    sim = PLGABiphasic()
    theta = sim.sample_prior(8)
    t = torch.tensor([0.0, 1.0])
    Q = sim.simulate(theta, t)
    assert torch.allclose(Q[:, 0], torch.zeros(8), atol=1e-5)


def test_zero_release_rates_with_burst_saturate_at_burst_asymptote() -> None:
    """If kd and ke are both ~0, Q(t) reaches Q_max·(1−exp(−q_burst)).

    ADR-026: with (Q_max - Q) coupling on the burst term, the asymptote
    after the burst window completes (t >> τ_burst) is
        Q(∞) = Q_max · (1 − exp(−q_burst))
    which differs from q_burst·Q_max by ≤14% in the prior range
    q_burst ≤ 0.30.
    """
    sim = PLGABiphasic()
    theta = sim.sample_prior(4).clone()
    theta[:, 3] = -20.0   # log_kd  ~ 0
    theta[:, 4] = -20.0   # log_ke  ~ 0
    # Run well past τ_burst (≤ 1 day) so burst is fully released.
    t = torch.linspace(0.0, 30.0, 50)
    Q = sim.simulate(theta, t)
    q_burst = theta[:, 6]
    Q_max = theta[:, 8]
    expected = Q_max * (1.0 - torch.exp(-q_burst))
    assert torch.allclose(Q[:, -1], expected, atol=2e-3), (
        f"Burst-only asymptote mismatch. Got Q[-1]={Q[:, -1].tolist()} vs "
        f"expected Q_max·(1−exp(−q_burst))={expected.tolist()}"
    )


def test_burst_term_recovers_old_behavior_at_small_tau() -> None:
    """ADR-026: as τ_burst → 0, the burst term concentrates near t=0.

    With log_tau_burst = -3 (τ ≈ 0.05 d ≈ 1.2 h), evaluation at t=0.5 d
    should already see Q ≈ Q_max·(1−exp(−q_burst)) (burst fully released).
    """
    sim = PLGABiphasic()
    theta = sim.sample_prior(4).clone()
    theta[:, 3] = -20.0   # log_kd  ~ 0
    theta[:, 4] = -20.0   # log_ke  ~ 0
    theta[:, 7] = -3.0    # log_tau_burst = -3 -> τ ≈ 0.05 d
    t = torch.tensor([0.0, 0.5])
    Q = sim.simulate(theta, t)
    q_burst = theta[:, 6]
    Q_max = theta[:, 8]
    expected = Q_max * (1.0 - torch.exp(-q_burst))
    # Allow 5% relative tolerance: at t=0.5d (= 10τ), exp(-t/τ) ≈ 4.5e-5,
    # so >99.99 % of burst has integrated.
    assert torch.allclose(Q[:, 1], expected, atol=2e-3), (
        f"At small τ_burst, Q(0.5d) should ≈ Q_max·(1−exp(−q_burst)). "
        f"Got {Q[:, 1].tolist()} vs {expected.tolist()}"
    )


def test_q_max_caps_late_time_release() -> None:
    """Saturation point is Q_max, not 1. This is the ADR-007 ODE feature."""
    sim = PLGABiphasic()
    theta = sim.sample_prior(4).clone()
    theta[:, 0] = 2.0     # log_kw  large -> fast hydration
    theta[:, 1] = 0.0     # log_kh  -> fast hydrolysis so erosion gate opens
    theta[:, 3] = 1.0     # log_kd
    theta[:, 4] = 2.0     # log_ke  large -> reach saturation within t_max
    theta[:, 6] = 0.0     # q_burst = 0
    theta[:, 7] = -1.0    # log_tau_burst (irrelevant when q_burst = 0)
    theta[:, 8] = 0.6     # Q_max  = 0.6
    t = torch.linspace(0.0, 200.0, 50)
    Q = sim.simulate(theta, t)
    assert (Q[:, -1] <= 0.6 + 1e-3).all(), f"Q exceeded Q_max=0.6: {Q[:, -1].tolist()}"
    assert (Q[:, -1] >= 0.55).all(), f"Q failed to approach Q_max: {Q[:, -1].tolist()}"


def test_torch_and_numpy_backends_agree() -> None:
    """The SciPy diagnostic backend must produce the same Q(t) as torchdiffeq.

    If this test ever fails, the two backends have drifted. Fix whichever
    backend was edited; never silently update the test bound.
    """
    import numpy as np

    sim = PLGABiphasic()
    theta = sim.sample_prior(4)
    t = torch.linspace(0.0, 60.0, 30)

    Q_torch = sim.simulate(theta, t).detach().numpy()
    Q_numpy = np.stack(
        [sim.simulate_numpy(theta[i].numpy(), t.numpy()) for i in range(4)]
    )

    # Adaptive ODE solvers with different stepping policies will not match
    # bit-for-bit. 1e-3 is a tight bound that catches genuine drift.
    assert np.allclose(Q_torch, Q_numpy, atol=1e-3), (
        f"Backend drift detected. Max abs diff: {np.abs(Q_torch - Q_numpy).max():.4e}"
    )


def test_simulate_is_differentiable() -> None:
    """Amortized SBI requires gradients through the simulator (or at least
    posterior gradients via reparameterization). Verify autograd flows."""
    sim = PLGABiphasic()
    theta = sim.sample_prior(2).requires_grad_(True)
    t = torch.linspace(0.0, 10.0, 20)
    Q = sim.simulate(theta, t)
    loss = Q.mean()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
