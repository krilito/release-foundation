"""Smoke tests for CurvePosterior and DescriptorPosterior.

These do not validate calibration (that's the job of scripts/05_sbc_*).
They confirm that the training pipeline runs end-to-end on a tiny budget
and that the trained object exposes the expected API.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from encoder import FormulationFeaturizer, MLPFormulationEncoder
from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid
from simulator import PLGABiphasic


def test_synthetic_pair_generation_has_right_shape() -> None:
    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 60.0, 32)
    cp = CurvePosterior(simulator=sim, t_grid=t_grid)
    theta, Q = cp.generate_synthetic_pairs(n=16, batch_size=8)
    assert theta.shape == (16, sim.n_params)
    assert Q.shape == (16, 32)
    assert (Q >= 0).all() and (Q <= 1.0).all()


def test_obs_noise_is_applied_by_default() -> None:
    """Synthetic pairs must carry observation noise (ADR-014).

    Compare noisy vs noiseless generation: difference per-element should
    be on the order of `noise_sigma`, not exactly zero.
    """
    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 60.0, 32)
    cp = CurvePosterior(simulator=sim, t_grid=t_grid, noise_sigma=0.03)

    torch.manual_seed(0)
    _, Q_noisy = cp.generate_synthetic_pairs(n=8, seed=42, add_noise=True)
    torch.manual_seed(0)
    _, Q_clean = cp.generate_synthetic_pairs(n=8, seed=42, add_noise=False)

    diff = (Q_noisy - Q_clean).abs()
    assert diff.max() > 0.0, "noise was not applied despite noise_sigma > 0"
    # Most elements should be within ~3σ of clean; some are clipped.
    assert diff.max() < 5 * cp.noise_sigma, (
        f"noise magnitude unrealistic: max diff {diff.max().item()} "
        f"vs sigma {cp.noise_sigma}"
    )


def test_noise_sigma_zero_is_identity() -> None:
    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 60.0, 16)
    cp = CurvePosterior(simulator=sim, t_grid=t_grid, noise_sigma=0.0)
    theta = sim.sample_prior(4)
    with torch.no_grad():
        Q = sim.simulate(theta, t_grid)
    Q_noised = cp.add_obs_noise(Q)
    assert torch.equal(Q, Q_noised)


def test_train_and_sample_smoke() -> None:
    """End-to-end with a tiny budget: train for a few epochs, then sample.

    A small flow on 256 synthetic samples will not be calibrated. The point
    here is that the pipeline runs and produces output of the right shape.
    """
    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 60.0, 16)
    cp = CurvePosterior(
        simulator=sim,
        t_grid=t_grid,
        flow="maf",
        hidden_features=16,
        num_transforms=2,
    )
    cp.train(
        n_simulations=256,
        training_batch_size=64,
        max_num_epochs=3,
        seed=0,
        stop_after_epochs=10,
        verbose=False,
    )
    assert cp.is_trained
    # Sample for a fake observation drawn from the simulator.
    theta_true = sim.sample_prior(1)
    Q_obs = sim.simulate(theta_true, t_grid)[0]
    samples = cp.sample(Q_obs, n_samples=50)
    assert samples.shape == (50, sim.n_params)


def test_save_and_load_round_trip(tmp_path) -> None:
    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 60.0, 16)
    cp = CurvePosterior(
        simulator=sim,
        t_grid=t_grid,
        flow="maf",
        hidden_features=16,
        num_transforms=2,
    )
    cp.train(
        n_simulations=256,
        training_batch_size=64,
        max_num_epochs=3,
        seed=0,
        stop_after_epochs=10,
        verbose=False,
    )

    ckpt_path = tmp_path / "cp.pt"
    cp.save(ckpt_path)

    cp2 = CurvePosterior.load(ckpt_path, simulator=sim)
    assert cp2.is_trained
    Q_obs = sim.simulate(sim.sample_prior(1), t_grid)[0]
    samples = cp2.sample(Q_obs, n_samples=20)
    assert samples.shape == (20, sim.n_params)

    if torch.cuda.is_available():
        cp_cuda = CurvePosterior.load(ckpt_path, simulator=sim, device="cuda")
        samples_cuda = cp_cuda.sample(Q_obs.cpu(), n_samples=5)
        assert samples_cuda.shape == (5, sim.n_params)
        assert samples_cuda.is_cuda


# ----------------------------------------------------------------------
# interpolate_to_grid
# ----------------------------------------------------------------------
def test_interpolate_holds_last_value_past_observed_tail() -> None:
    # Observed: t = 0..30 with Q linear 0 to 0.9. Grid: 0..90 64 pts.
    t_obs = torch.linspace(0.0, 30.0, 10)
    Q_obs = torch.linspace(0.0, 0.9, 10)
    t_grid = torch.linspace(0.0, 90.0, 64)
    Q_grid, quality, meta = interpolate_to_grid(t_obs, Q_obs, t_grid)
    assert Q_grid.shape == (64,)
    # Past the observed support, must hold at last value 0.9.
    past_mask = t_grid > 30.0
    assert torch.allclose(Q_grid[past_mask], torch.full((int(past_mask.sum()),), 0.9), atol=1e-5)
    # Within [0, 30], linear interpolation: at t=15, Q ~ 0.45.
    idx_15 = int(torch.argmin((t_grid - 15.0).abs()))
    assert abs(Q_grid[idx_15].item() - 0.45) < 0.05
    # tmax=30, lastQ=0.9 -> high (lastQ >= 0.8)
    assert quality == "high"
    assert meta["tmax"] == 30.0


def test_interpolate_drops_points_past_t_max_days() -> None:
    # Observed extends to 180 days; only 0..90 must be used.
    t_obs = torch.tensor([0.0, 30.0, 90.0, 120.0, 180.0])
    Q_obs = torch.tensor([0.0, 0.5, 0.85, 0.92, 0.99])
    t_grid = torch.linspace(0.0, 90.0, 64)
    Q_grid, quality, meta = interpolate_to_grid(t_obs, Q_obs, t_grid, t_max_days=90.0)
    assert meta["n_obs_kept"] == 3  # 0, 30, 90 only
    assert abs(Q_grid[-1].item() - 0.85) < 1e-5
    # tmax=90 >= 60 -> high
    assert quality == "high"


def test_interpolate_low_quality_flag() -> None:
    # tmax=20 < 60 and lastQ=0.4 < 0.7 -> low
    t_obs = torch.linspace(0.0, 20.0, 5)
    Q_obs = torch.linspace(0.0, 0.4, 5)
    t_grid = torch.linspace(0.0, 90.0, 32)
    _, quality, meta = interpolate_to_grid(t_obs, Q_obs, t_grid)
    assert quality == "low"
    assert meta["tmax"] == 20.0


def test_interpolate_medium_quality_flag() -> None:
    # tmax=40 < 60, lastQ=0.75 in [0.7, 0.8) -> medium
    t_obs = torch.tensor([0.0, 10.0, 25.0, 40.0])
    Q_obs = torch.tensor([0.0, 0.3, 0.6, 0.75])
    t_grid = torch.linspace(0.0, 90.0, 32)
    _, quality, _ = interpolate_to_grid(t_obs, Q_obs, t_grid)
    assert quality == "medium"


def test_interpolate_rejects_too_few_points() -> None:
    t_obs = torch.tensor([0.0])
    Q_obs = torch.tensor([0.0])
    t_grid = torch.linspace(0.0, 90.0, 16)
    with pytest.raises(ValueError):
        interpolate_to_grid(t_obs, Q_obs, t_grid)


# ----------------------------------------------------------------------
# DescriptorPosterior — end-to-end smoke
# ----------------------------------------------------------------------
@pytest.fixture
def small_descriptor_df() -> pd.DataFrame:
    rows = []
    polymers = ["DEX-PLGA", "PTX-PVL-co-PAVL", "5-FU-PLGA",
                "TAA-PLA-co-PALA", "CBD-PCL"]
    for i, dp in enumerate(polymers):
        rows.append({
            "DP_Group": dp,
            "LA/GA": float(i % 3) * 0.5,
            "Polymer_MW": 20000.0 + 5000.0 * i,
            "CL Ratio": 0.05 * (i % 2),
            "Drug_Tm": 200.0 + 10.0 * i,
            "Drug_Pka": 8.0 + 0.5 * i,
            "Initial D/M ratio": 0.1 + 0.05 * i,
            "DLC": 0.1 + 0.02 * i,
            "SA-V": 400.0 + 10.0 * i,
            "SE": float(i % 2),
            "Drug_Mw": 200.0 + 50.0 * i,
            "Drug_TPSA": 60.0 + 10.0 * i,
            "Drug_NHA": float(4 + i),
            "Drug_LogP": -1.0 + 0.5 * i,
        })
    return pd.DataFrame(rows)


def test_descriptor_posterior_train_and_sample_smoke(
    small_descriptor_df: pd.DataFrame, tmp_path
) -> None:
    """End-to-end with a tiny budget: train teacher q_φ, distill r_ψ,
    sample, save, load. Does not validate calibration."""
    sim = PLGABiphasic()
    t_grid = torch.linspace(0.0, 60.0, 16)

    # 1. Tiny Stage-1 teacher.
    cp = CurvePosterior(
        simulator=sim, t_grid=t_grid,
        flow="maf", hidden_features=16, num_transforms=2,
    )
    cp.train(
        n_simulations=256, training_batch_size=64, max_num_epochs=3,
        seed=0, stop_after_epochs=10, verbose=False,
    )

    # 2. Featurizer + encoder.
    feat = FormulationFeaturizer.fit(small_descriptor_df)
    enc = MLPFormulationEncoder(input_dim=feat.input_dim, embed_dim=16)

    # 3. Build teacher targets on synthetic Q (no real curves needed here).
    theta_truth = sim.sample_prior(len(small_descriptor_df))
    with torch.no_grad():
        Q_real_proxy = sim.simulate(theta_truth, t_grid)
    Q_real_proxy = cp.add_obs_noise(Q_real_proxy)

    # 4. Train Stage-2.
    rp = DescriptorPosterior(
        simulator=sim, featurizer=feat, encoder=enc,
        flow="maf", hidden_features=16, num_transforms=2,
    )
    targets = rp.build_teacher_targets(cp, Q_real_proxy, K=4, seed=0)
    assert targets.shape == (len(small_descriptor_df), 4, sim.n_params)

    x = feat.transform(small_descriptor_df)
    log = rp.train(
        x=x, theta_targets=targets,
        training_batch_size=32, max_num_epochs=3,
        stop_after_epochs=10, verbose=False,
    )
    assert rp.is_trained
    assert log["n_pairs"] == len(small_descriptor_df) * 4

    # 5. Sample.
    samples = rp.sample(x[0], n_samples=30)
    assert samples.shape == (30, sim.n_params)

    # 6. Save / load round-trip.
    ckpt = tmp_path / "rp.pt"
    rp.save(ckpt)
    rp2 = DescriptorPosterior.load(ckpt, simulator=sim)
    assert rp2.is_trained
    samples2 = rp2.sample(x[0], n_samples=20)
    assert samples2.shape == (20, sim.n_params)

    if torch.cuda.is_available():
        rp_cuda = DescriptorPosterior.load(ckpt, simulator=sim, device="cuda")
        samples_cuda = rp_cuda.sample(x[0].cpu(), n_samples=5)
        assert samples_cuda.shape == (5, sim.n_params)
        assert samples_cuda.is_cuda

        log_prob = rp_cuda.log_prob(samples_cuda[:2].cpu(), x[0].cpu())
        assert log_prob.shape == (2,)
        assert log_prob.is_cuda


def test_descriptor_posterior_validates_input_dim(small_descriptor_df: pd.DataFrame) -> None:
    sim = PLGABiphasic()
    feat = FormulationFeaturizer.fit(small_descriptor_df)
    # encoder built with wrong input_dim -> constructor must reject
    bad_enc = MLPFormulationEncoder(input_dim=feat.input_dim + 1, embed_dim=16)
    with pytest.raises(ValueError):
        DescriptorPosterior(simulator=sim, featurizer=feat, encoder=bad_enc)
