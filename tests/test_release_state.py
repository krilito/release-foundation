from __future__ import annotations

import numpy as np

from release_state import (
    ObservationBudgetAnalyzer,
    PCACurveStateSpace,
    TreePrior,
    WeibullStateSpace,
)


def _shared_grid() -> np.ndarray:
    return np.linspace(0.0, 24.0, 48)


def _low_rank_monotone_curves(n_curves: int = 10) -> np.ndarray:
    t = _shared_grid()
    basis = np.vstack(
        [
            1.0 - np.exp(-t / 2.5),
            1.0 - np.exp(-np.square(t / 8.0)),
            t / t.max(),
        ]
    )
    weights = np.linspace(0.2, 1.1, n_curves * 3).reshape(n_curves, 3)
    weights = weights / weights.sum(axis=1, keepdims=True)
    curves = 0.9 * (weights @ basis)
    return np.maximum.accumulate(curves, axis=1)


def test_pca_state_space_round_trip_reconstructs_low_rank_curves() -> None:
    t = _shared_grid()
    curves = _low_rank_monotone_curves()
    state_space = PCACurveStateSpace(n_components=3).fit(curves, t)

    z = state_space.encode(curves, t)
    reconstructed = state_space.decode(z, t)

    assert z.shape == (10, 3)
    assert reconstructed.shape == curves.shape
    assert np.sqrt(np.mean((reconstructed - curves) ** 2)) < 1e-10


def test_weibull_state_space_recovers_known_curve_family() -> None:
    t = _shared_grid()
    state_space = WeibullStateSpace()
    true_states = np.array(
        [
            [np.log(3.0), np.log(1.1), 0.2],
            [np.log(7.0), np.log(1.6), -0.4],
        ]
    )
    curves = state_space.decode(true_states, t)

    estimated_states = state_space.encode(curves, t)
    decoded = state_space.decode(estimated_states, t)

    assert estimated_states.shape == true_states.shape
    assert np.sqrt(np.mean((decoded - curves) ** 2)) < 1e-3


def test_residual_zero_for_identical_states() -> None:
    state_space = WeibullStateSpace()
    states = np.array([[np.log(4.0), np.log(1.2), 0.1]])

    residual = state_space.residual(states, states)

    assert residual.shape == states.shape
    assert np.allclose(residual, 0.0)


def test_pca_residual_symptoms_have_time_domain_shape() -> None:
    t = _shared_grid()
    curves = _low_rank_monotone_curves()
    state_space = PCACurveStateSpace(n_components=3).fit(curves, t)
    z = state_space.encode(curves, t)
    residual = state_space.residual(z[:2], z[2:4])

    symptoms = state_space.residual_symptoms(residual, t)

    assert symptoms.shape == (2, len(t))


def test_tree_prior_fit_predict_smoke() -> None:
    curves = _low_rank_monotone_curves()
    t = _shared_grid()
    z = PCACurveStateSpace(n_components=3).fit(curves, t).encode(curves, t)
    x = np.column_stack([np.linspace(0.0, 1.0, len(curves)), np.ones(len(curves))])

    prior = TreePrior(n_estimators=10, random_state=0).fit(x, z)
    pred = prior.predict(x[:2])

    assert pred.shape == (2, 3)
    assert np.isfinite(pred).all()


def test_gap_closed_returns_negative_when_observations_hurt() -> None:
    analyzer = ObservationBudgetAnalyzer()

    value = analyzer.gap_closed(loss_k=0.25, loss_static=0.20, loss_oracle=0.10)

    assert value < 0.0
