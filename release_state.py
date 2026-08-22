from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor


ArrayLike = np.ndarray


def _as_2d(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {array.shape}")
    return array


def _as_time_grid(times: ArrayLike) -> np.ndarray:
    grid = np.asarray(times, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("time grid is empty")
    return grid


def _monotone_release(values: ArrayLike) -> np.ndarray:
    curves = np.clip(_as_2d(values), 0.0, 1.2)
    return np.maximum.accumulate(curves, axis=1)


def _logit(values: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: ArrayLike) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _weibull_curve(times: np.ndarray, log_tau: float, log_beta: float, qmax_raw: float) -> np.ndarray:
    tau = np.exp(log_tau)
    beta = np.exp(log_beta)
    qmax = 1.2 * _sigmoid(qmax_raw)
    safe_t = np.maximum(times, 0.0)
    return qmax * (1.0 - np.exp(-np.power(safe_t / tau, beta)))


class ReleaseStateSpace(ABC):
    """Coordinate system where curve states, priors, and residuals are defined."""

    name: str
    dim: int

    def fit(self, curves: ArrayLike, times: ArrayLike) -> ReleaseStateSpace:
        return self

    @abstractmethod
    def encode(self, curves: ArrayLike, times: ArrayLike) -> np.ndarray:
        """Map release curves Q(t) into coordinates z."""

    @abstractmethod
    def decode(self, states: ArrayLike, times: ArrayLike) -> np.ndarray:
        """Map coordinates z back into release curves Q_hat(t)."""

    def residual(self, z_curve: ArrayLike, z_prior: ArrayLike) -> np.ndarray:
        """Natural residual in this state space."""
        return _as_2d(z_curve) - _as_2d(z_prior)

    def residual_symptoms(self, residuals: ArrayLike, times: ArrayLike) -> np.ndarray:
        """Decode state residuals into time-domain release symptoms when possible."""
        zeros = np.zeros_like(_as_2d(residuals))
        return self.decode(residuals, times) - self.decode(zeros, times)


class PCACurveStateSpace(ReleaseStateSpace):
    """Train-fold PCA dictionary for cumulative release curves."""

    def __init__(self, n_components: int = 8, random_state: int = 0, name: str | None = None):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.name = name or f"pca{self.n_components}"
        self.dim = self.n_components
        self._pca: PCA | None = None

    def fit(self, curves: ArrayLike, times: ArrayLike) -> PCACurveStateSpace:
        matrix = _monotone_release(curves)
        if matrix.shape[0] < self.n_components:
            raise ValueError("PCA state space needs at least n_components training curves")
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(matrix)
        return self

    def encode(self, curves: ArrayLike, times: ArrayLike) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCACurveStateSpace must be fit on train-fold curves first")
        return self._pca.transform(_monotone_release(curves))

    def decode(self, states: ArrayLike, times: ArrayLike) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCACurveStateSpace must be fit on train-fold curves first")
        return _monotone_release(self._pca.inverse_transform(_as_2d(states)))

    @property
    def components_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCACurveStateSpace must be fit before reading components")
        return self._pca.components_

    def residual_symptoms(self, residuals: ArrayLike, times: ArrayLike) -> np.ndarray:
        """Project state residuals into time-domain curve effects.

        For PCA this is equivalent to ``decode(r) - decode(0)`` because
        inverse PCA is linear.  The direct matrix multiply avoids a
        redundant ``_monotone_release`` clip inside ``decode``.
        """
        if self._pca is None:
            raise RuntimeError("PCACurveStateSpace must be fit on train-fold curves first")
        return _as_2d(residuals) @ self._pca.components_


class WeibullStateSpace(ReleaseStateSpace):
    """Three-parameter Weibull release state in log/logit coordinates."""

    name = "weibull_logtheta"
    dim = 3

    def encode(self, curves: ArrayLike, times: ArrayLike) -> np.ndarray:
        grid = _as_time_grid(times)
        matrix = _monotone_release(curves)
        states = []
        for curve in matrix:
            states.append(self._fit_one(grid, curve))
        return np.asarray(states, dtype=float)

    def decode(self, states: ArrayLike, times: ArrayLike) -> np.ndarray:
        grid = _as_time_grid(times)
        decoded = [_weibull_curve(grid, *state) for state in _as_2d(states)]
        return _monotone_release(np.asarray(decoded, dtype=float))

    def _fit_one(self, times: np.ndarray, curve: np.ndarray) -> np.ndarray:
        finite = np.isfinite(times) & np.isfinite(curve)
        x = times[finite]
        y = np.clip(curve[finite], 0.0, 1.2)
        if x.size < 3:
            return self._fallback_state(x, y)
        p0 = self._fallback_state(x, y)
        bounds = (
            [np.log(1e-3), np.log(0.1), _logit(np.array([0.02 / 1.2]))[0]],
            [np.log(1e4), np.log(8.0), _logit(np.array([0.999]))[0]],
        )
        try:
            params, _ = curve_fit(_weibull_curve, x, y, p0=p0, bounds=bounds, maxfev=8000)
            return np.asarray(params, dtype=float)
        except (RuntimeError, ValueError, FloatingPointError):
            return p0

    def _fallback_state(self, times: np.ndarray, curve: np.ndarray) -> np.ndarray:
        if times.size == 0 or curve.size == 0:
            return np.array([np.log(1.0), np.log(1.0), 0.0], dtype=float)
        qmax = float(np.clip(np.nanmax(curve), 0.02, 1.2))
        target = 0.632 * qmax
        order = np.argsort(times)
        sorted_t = np.maximum(times[order], 1e-3)
        sorted_q = curve[order]
        idx = int(np.nanargmin(np.abs(sorted_q - target)))
        tau = float(np.clip(sorted_t[idx], 1e-3, 1e4))
        qmax_raw = float(_logit(np.array([np.clip(qmax / 1.2, 1e-6, 1.0 - 1e-6)]))[0])
        return np.array([np.log(tau), np.log(1.0), qmax_raw], dtype=float)


class PLGAODEStateSpace(ReleaseStateSpace):
    """Wrapper for PLGA ODE theta encoders; residuals stay in log-theta space."""

    def __init__(
        self,
        dim: int,
        encoder: Callable[[ArrayLike, ArrayLike], ArrayLike] | None = None,
        decoder: Callable[[ArrayLike, ArrayLike], ArrayLike] | None = None,
        name: str = "plga_ode_logtheta",
    ):
        self.name = name
        self.dim = int(dim)
        self._encoder = encoder
        self._decoder = decoder

    def encode(self, curves: ArrayLike, times: ArrayLike) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("PLGAODEStateSpace needs an encoder callable")
        encoded = _as_2d(np.asarray(self._encoder(curves, times), dtype=float))
        if encoded.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} theta columns, got {encoded.shape[1]}")
        return encoded

    def decode(self, states: ArrayLike, times: ArrayLike) -> np.ndarray:
        if self._decoder is None:
            raise RuntimeError("PLGAODEStateSpace needs a decoder callable")
        return _monotone_release(np.asarray(self._decoder(_as_2d(states), times), dtype=float))

    def residual(self, z_curve: ArrayLike, z_prior: ArrayLike) -> np.ndarray:
        return _as_2d(z_curve) - _as_2d(z_prior)


class StaticPrior(ABC):
    """Predict release-state coordinates from pre-experimental descriptors X."""

    @abstractmethod
    def fit(self, descriptors: ArrayLike, states: ArrayLike) -> StaticPrior:
        """Fit X -> z on train-fold curves only."""

    @abstractmethod
    def predict(self, descriptors: ArrayLike) -> np.ndarray:
        """Predict prior state z_X from descriptors X."""


class TreePrior(StaticPrior):
    """Small-data friendly ExtraTrees prior for X -> z."""

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 2,
        random_state: int = 0,
        n_jobs: int = -1,
    ):
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, descriptors: ArrayLike, states: ArrayLike) -> TreePrior:
        self.model.fit(_as_2d(descriptors), _as_2d(states))
        return self

    def predict(self, descriptors: ArrayLike) -> np.ndarray:
        return _as_2d(self.model.predict(_as_2d(descriptors)))


@dataclass(frozen=True)
class ObservationBudgetAnalyzer:
    """Summarize how much loss is reduced as observations are added."""

    k_col: str = "k"
    loss_col: str = "loss"

    def value_curve(self, loss_table: ArrayLike) -> np.ndarray:
        table = np.asarray(loss_table, dtype=float)
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError("loss_table must have columns [k, loss]")
        order = np.argsort(table[:, 0])
        sorted_table = table[order]
        k = sorted_table[:, 0]
        loss = sorted_table[:, 1]
        baseline = loss[0]
        value = baseline - loss
        marginal = np.r_[np.nan, value[1:] - value[:-1]]
        return np.column_stack([k, loss, value, marginal])

    def gap_closed(self, loss_k: float, loss_static: float, loss_oracle: float) -> float:
        """Percentage of the static-to-oracle gap closed by k observations.

        Returns negative values when observations hurt (loss_k > loss_static),
        which is a real diagnostic signal, not a clipping target.
        """
        denom = loss_static - loss_oracle
        if denom <= 0:
            return np.nan
        return 100.0 * (loss_static - loss_k) / denom
