"""
Active Kinetic Observer v3 — Conformal-calibrated, two-point sequential.

Improvements over v2 (script 68):
  1. Split conformal calibration → coverage_90 target ≥ 0.85
  2. Two-point sequential observer → active_2pt ≈ fixed_4pt
  3. More candidate times (0.5–10d) + time-cost utility → breaks monotonicity
  4. Q_max observation cap → tighter posterior from observed Q
  5. Particle rejuvenation via MCMC jitter when ESS drops

Run:
    .\.venv\Scripts\python.exe scripts\69_active_observer_v3.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Standalone PLGA ODE simulator — avoids importing torch (page file issue).
# Mathematically identical to PLGABiphasic.simulate_numpy() in simulator.py.
from scipy.integrate import solve_ivp  # noqa: E402

_PLGA_EPS_M = 0.05
_PLGA_N_PARAMS = 9
_PLGA_PARAM_NAMES = [
    "log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
    "m_crit", "q_burst", "log_tau_burst", "Q_max",
]
_Q_MAX_IDX = 8  # index of Q_max in the 9-param layout


# ============================================================
# 0. Configuration
# ============================================================

@dataclass
class Config:
    formulations_path: str = "data/formulations.csv"
    curves_path: str = "data/curves_long.csv"
    theta_path: str = "data/theta_bank.csv"
    output_dir: str = "outputs_active_observer_v3"

    id_col: str = "curve_id"
    time_col: str = "time"
    release_col: str = "release"

    theta_cols: tuple[str, ...] = (
        "log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
        "m_crit", "q_burst", "log_tau_burst", "Q_max",
    )

    # Step 1 candidates: early observation window
    candidate_times: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0)
    # Step 2 candidates: later window (includes overlap with step 1 at 7d)
    candidate_times_step2: tuple[float, ...] = (7.0, 10.0, 14.0, 21.0)
    future_eval_start: float = 28.0
    rollout_times: tuple[float, ...] = (
        0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
        14.0, 21.0, 28.0, 42.0, 56.0, 84.0,
    )

    # Split: train / calibrate / test
    test_size: float = 0.25
    cal_size: float = 0.15  # fraction of total for conformal calibration
    random_state: int = 42
    group_split_col: Optional[str] = None  # e.g. "DP_Group" for held-out group

    n_tree_estimators: int = 500
    n_knn_particles: int = 50
    n_tree_particles: int = 150
    jitter_scale: float = 0.5

    # Q_max bias correction
    q_max_correction: float = 0.10

    # Observation noise model
    obs_sigma_abs: float = 0.03
    obs_sigma_rel: float = 0.08
    likelihood_beta: float = 2.0
    min_effective_particles: int = 10

    # Active selection
    max_pseudo_obs_for_utility: int = 100
    # Time-cost lambda: penalize later observations. Higher = stronger penalty.
    # With variance-reduction utility (values ~0.05-0.15), lambda=0.005 gives
    # penalty of 0.005*7=0.035 at 7d (moderate) vs 0.005*21=0.105 at 21d (strong).
    time_cost_lambda: float = 0.005

    # Q_max cap: observed Q at any time gives a lower bound for Q_max
    q_max_cap_margin: float = 0.05
    step2_q_max_cap_margin: Optional[float] = None

    # Conformal calibration
    conformal_target_coverage: float = 0.90
    conformal_method: str = "local"  # "global" or "local" (per-time-bin)

    # Particle rejuvenation: when ESS < threshold, add MCMC jitter
    rejuvenation_ess_threshold: float = 20.0
    rejuvenation_jitter_scale: float = 0.05

    train_direct_q_baseline: bool = True
    verbose: bool = True


# ============================================================
# 1. Utilities (shared with v2, kept identical)
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_release(y: pd.Series) -> pd.Series:
    y = y.astype(float)
    if y.max() > 2.0:
        y = y / 100.0
    return y.clip(0.0, 1.2)


def weighted_var(values: np.ndarray, weights: np.ndarray, axis: int = 0) -> np.ndarray:
    w = weights / (weights.sum() + 1e-12)
    mu = np.sum(values * w[:, None], axis=0)
    return np.sum(w[:, None] * (values - mu[None, :]) ** 2, axis=0)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cw = np.cumsum(w)
    cw = cw / (cw[-1] + 1e-12)
    return np.interp(quantiles, cw, v)


def weighted_curve_quantiles(curves: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    return np.asarray([weighted_quantile(curves[:, j], weights, quantiles) for j in range(curves.shape[1])]).T


def effective_sample_size(weights: np.ndarray) -> float:
    w = weights / (weights.sum() + 1e-12)
    return 1.0 / (np.sum(w ** 2) + 1e-12)


def interp_curve(curve_df: pd.DataFrame, times: np.ndarray, time_col: str, release_col: str) -> np.ndarray:
    d = curve_df.sort_values(time_col)
    t, y = d[time_col].to_numpy(dtype=float), d[release_col].to_numpy(dtype=float)
    if len(t) == 1:
        return np.full_like(times, y[0], dtype=float)
    return np.interp(times, t, y, left=y[0], right=y[-1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def ensemble_crps_unweighted(
    samples: np.ndarray, y_true: np.ndarray, max_particles: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> float:
    n = samples.shape[0]
    if n > max_particles:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(n, size=max_particles, replace=False)
        samples = samples[idx]
        n = max_particles
    term1 = np.mean(np.abs(samples - y_true[None, :]), axis=0)
    diffs = np.abs(samples[:, None, :] - samples[None, :, :])
    term2 = 0.5 * np.mean(diffs, axis=(0, 1))
    return float(np.mean(term1 - term2))


def weighted_resample_indices(weights: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    w = weights / (weights.sum() + 1e-12)
    return rng.choice(len(w), size=n_samples, replace=True, p=w)


# ============================================================
# 2. Data loading
# ============================================================

def load_data(cfg: Config):
    formulations = pd.read_csv(cfg.formulations_path)
    curves = pd.read_csv(cfg.curves_path)
    theta = pd.read_csv(cfg.theta_path)
    curves[cfg.release_col] = normalize_release(curves[cfg.release_col])

    common_ids = (
        set(formulations[cfg.id_col])
        & set(curves[cfg.id_col])
        & set(theta[cfg.id_col])
    )
    formulations = formulations[formulations[cfg.id_col].isin(common_ids)].copy()
    curves = curves[curves[cfg.id_col].isin(common_ids)].copy()
    theta = theta[theta[cfg.id_col].isin(common_ids)].copy()

    if cfg.verbose:
        print(f"[data] formulations: {formulations.shape}")
        print(f"[data] curves long: {curves.shape}")
        print(f"[data] theta bank: {theta.shape}")
        print(f"[data] common curve ids: {len(common_ids)}")
    return formulations, curves, theta


def make_train_cal_test_ids(formulations: pd.DataFrame, cfg: Config):
    """Split into train / calibration / test. Supports random or group-based split."""
    ids = formulations[cfg.id_col].to_numpy()

    if cfg.group_split_col and cfg.group_split_col in formulations.columns:
        groups = formulations[cfg.group_split_col].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        # Leave-one-group-out: hold out the largest group
        group_sizes = {g: (groups == g).sum() for g in unique_groups}
        test_group = max(group_sizes, key=group_sizes.get)
        test_mask = groups == test_group
        train_cal_mask = ~test_mask
        test_ids = set(ids[test_mask])
        train_cal_ids_arr = ids[train_cal_mask]
        # Split remaining into train vs cal
        cal_frac = cfg.cal_size / (1.0 - cfg.test_size)
        n_cal = max(1, int(len(train_cal_ids_arr) * cal_frac))
        rng = np.random.default_rng(cfg.random_state)
        rng.shuffle(train_cal_ids_arr)
        cal_ids = set(train_cal_ids_arr[:n_cal])
        train_ids = set(train_cal_ids_arr[n_cal:])
        return set(train_ids), set(cal_ids), set(test_ids), f"group_split:{cfg.group_split_col}({test_group})"

    # Random split
    train_cal_ids, test_ids = train_test_split(
        ids, test_size=cfg.test_size, random_state=cfg.random_state,
    )
    cal_frac = cfg.cal_size / (1.0 - cfg.test_size)
    train_ids, cal_ids = train_test_split(
        train_cal_ids, test_size=cal_frac, random_state=cfg.random_state + 1,
    )
    return set(train_ids), set(cal_ids), set(test_ids), "random_split"


# ============================================================
# 3. PLGA ODE rollout (standalone, no torch)
# ============================================================

# Prior bounds from simulator.py PLGABiphasic.prior()
_LOWS = np.array([-3.0, -4.0, -2.0, -5.0, -3.0, 0.05, 0.0, -3.0, 0.50])
_HIGHS = np.array([1.0, -1.0, 1.0, 0.0, 2.0, 0.50, 0.30, 0.0, 1.00])


def _simulate_one_plga(theta: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    """SciPy DOP853 forward pass for one theta vector.
    Identical to PLGABiphasic.simulate_numpy() in simulator.py."""
    (log_kw, log_kh, log_alpha, log_kd, log_ke,
     m_crit, q_burst, log_tau_burst, Q_max) = theta
    kw, kh, alpha, kd, ke = np.exp([log_kw, log_kh, log_alpha, log_kd, log_ke])
    tau_burst = float(np.exp(log_tau_burst))
    eps_m = _PLGA_EPS_M

    def vector_field(tt, state):
        h, m, Q = state
        dh = kw * (1.0 - h)
        dm = -kh * h * m * (1.0 + alpha * (1.0 - m))
        gate_arg = np.clip((m_crit - m) / eps_m, -60.0, 60.0)
        erosion_gate = 1.0 / (1.0 + np.exp(-gate_arg))
        free = max(Q_max - Q, 0.0)
        burst_rate = (q_burst / tau_burst) * np.exp(-tt / tau_burst) * free
        dQ = kd * h * free + ke * erosion_gate * free + burst_rate
        return [dh, dm, dQ]

    sol = solve_ivp(
        vector_field,
        (0.0, float(t_obs[-1])),
        [0.0, 1.0, 0.0],
        t_eval=t_obs,
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success or sol.y.shape[1] != len(t_obs):
        return np.full_like(t_obs, 1e6, dtype=float)
    return np.minimum(np.clip(sol.y[2], 0.0, None), Q_max)


def simulate_plga_ode(theta_matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
    curves = []
    for theta in theta_matrix:
        q = _simulate_one_plga(theta, times)
        curves.append(q)
    result = np.asarray(curves, dtype=float)
    return np.clip(result, 0.0, 1.1)


# ============================================================
# 4. Formulation-conditioned theta prior (same as v2)
# ============================================================

class FormulationThetaPrior:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.preprocessor = None
        self.et = None
        self.knn = None
        self.train_X_proc = None
        self.train_theta = None
        self.train_ids = None
        self.theta_mean = None
        self.theta_cov = None
        self.feature_cols = None

    def fit(self, formulations_train: pd.DataFrame, theta_train_curve_level: pd.DataFrame):
        cfg = self.cfg
        df = formulations_train.merge(
            theta_train_curve_level[[cfg.id_col, *cfg.theta_cols]], on=cfg.id_col, how="inner",
        )
        self.feature_cols = [c for c in formulations_train.columns if c != cfg.id_col]
        X = df[self.feature_cols].copy()
        y = df[list(cfg.theta_cols)].to_numpy(dtype=float)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
        ], remainder="drop")

        X_proc = self.preprocessor.fit_transform(X)

        self.et = ExtraTreesRegressor(
            n_estimators=cfg.n_tree_estimators, random_state=cfg.random_state,
            min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1,
        )
        self.et.fit(X_proc, y)

        self.knn = NearestNeighbors(
            n_neighbors=min(cfg.n_knn_particles, len(X_proc)), metric="euclidean",
        )
        self.knn.fit(X_proc)

        self.train_X_proc = X_proc
        self.train_theta = y
        self.train_ids = df[cfg.id_col].to_numpy()
        self.theta_mean = y.mean(axis=0)
        self.theta_cov = np.cov(y.T) + np.eye(y.shape[1]) * 1e-6

        tree_pred_train = np.asarray([est.predict(X_proc)[0] for est in self.et.estimators_[:50]])
        tree_mean_train = tree_pred_train.mean(axis=0)
        self._tree_bias = y.mean(axis=0) - tree_mean_train
        self._tree_resid_std = np.std(y - tree_mean_train, axis=0)

        if cfg.verbose:
            print(f"[prior] fitted on {len(df)} curves, numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")
            print(f"[prior] tree bias (Q_max dim): {self._tree_bias[-1]:.3f}")
        return self

    def transform_one(self, formulation_row: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(formulation_row[self.feature_cols].copy())

    def sample_prior_particles(
        self,
        formulation_row: pd.DataFrame,
        n_tree_particles: Optional[int] = None,
        n_knn_particles: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        if rng is None:
            rng = np.random.default_rng(cfg.random_state)
        n_tree = n_tree_particles or cfg.n_tree_particles
        n_knn = n_knn_particles or cfg.n_knn_particles

        X_proc = self.transform_one(formulation_row)

        tree_idx = rng.choice(len(self.et.estimators_), size=min(n_tree, len(self.et.estimators_)), replace=False)
        tree_particles = np.asarray([self.et.estimators_[i].predict(X_proc)[0] for i in tree_idx])

        k = min(n_knn, len(self.train_X_proc))
        _, nn_idx = self.knn.kneighbors(X_proc, n_neighbors=k)
        knn_particles = self.train_theta[nn_idx[0]]

        particles = np.vstack([tree_particles, knn_particles])

        n_tree_actual = tree_particles.shape[0]
        particles[:n_tree_actual] += self._tree_bias

        if cfg.jitter_scale > 0:
            std = np.maximum(self._tree_resid_std, 1e-4)
            jitter = rng.normal(0.0, cfg.jitter_scale * std, size=particles.shape)
            particles = particles + jitter

        if cfg.q_max_correction > 0:
            q_max_idx = _Q_MAX_IDX
            particles[:, q_max_idx] -= cfg.q_max_correction

        lower = np.percentile(self.train_theta, 0.0, axis=0)
        upper = np.percentile(self.train_theta, 100.0, axis=0)
        span = upper - lower
        lower = lower - 0.1 * span
        upper = upper + 0.1 * span
        q_max_idx = _Q_MAX_IDX
        upper[q_max_idx] = min(upper[q_max_idx], 1.0)
        return np.clip(particles, lower, upper)


def collapse_theta_to_curve_level(theta_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return theta_df.groupby(cfg.id_col)[list(cfg.theta_cols)].median().reset_index()


# ============================================================
# 5. Posterior update, particle rejuvenation, Q_max capping
# ============================================================

def compute_likelihood_weights(predicted_q: np.ndarray, observed_q: float, cfg: Config) -> np.ndarray:
    sigma = max(cfg.obs_sigma_abs, cfg.obs_sigma_rel * max(abs(observed_q), 0.05))
    log_w = -0.5 * cfg.likelihood_beta * (predicted_q - observed_q) ** 2 / (sigma ** 2)
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    return w / (w.sum() + 1e-12)


def rejuvenate_particles(
    theta_particles: np.ndarray, weights: np.ndarray,
    cfg: Config, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """When ESS drops too low, resample + add small MCMC jitter to prevent
    particle degeneracy. This is a lightweight rejuvenation step — not a
    full MCMC chain, just enough to restore diversity."""
    ess = effective_sample_size(weights)
    if ess >= cfg.rejuvenation_ess_threshold:
        return theta_particles, weights

    n = len(weights)
    idx = weighted_resample_indices(weights, n, rng)
    new_particles = theta_particles[idx].copy()

    # Jitter scaled by per-dim spread of current particles
    spread = np.std(new_particles, axis=0) + 1e-6
    jitter = rng.normal(0.0, cfg.rejuvenation_jitter_scale * spread, size=new_particles.shape)
    new_particles = new_particles + jitter

    # Clip to prior bounds
    new_particles = np.clip(new_particles, _LOWS, _HIGHS)

    new_weights = np.ones(n) / n
    return new_particles, new_weights


def posterior_update_from_observation(
    rollout_curves: np.ndarray, times: np.ndarray,
    prior_weights: np.ndarray, obs_time: float, obs_q: float, cfg: Config,
    theta_particles: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Update posterior weights given one observation. Also applies Q_max
    capping and particle rejuvenation if theta_particles are provided.

    Returns: (updated_weights, updated_theta_particles_or_None)
    """
    idx = int(np.argmin(np.abs(times - obs_time)))
    like_w = compute_likelihood_weights(rollout_curves[:, idx], obs_q, cfg)
    post_w = prior_weights * like_w
    post_w = post_w / (post_w.sum() + 1e-12)

    # Q_max capping: observed Q at obs_time gives lower bound for Q_max
    updated_theta = None
    if theta_particles is not None:
        updated_theta = theta_particles.copy()
        q_max_idx = _Q_MAX_IDX
        q_max_lower_bound = obs_q + cfg.q_max_cap_margin
        cap_mask = updated_theta[:, q_max_idx] < q_max_lower_bound
        updated_theta[cap_mask, q_max_idx] = q_max_lower_bound

        # Re-rollout curves for capped particles (only those that changed)
        if cap_mask.any():
            capped_curves = simulate_plga_ode(updated_theta[cap_mask], times)
            rollout_curves = rollout_curves.copy()
            rollout_curves[cap_mask] = capped_curves
            # Recompute likelihood with updated curves
            like_w = compute_likelihood_weights(rollout_curves[:, idx], obs_q, cfg)
            post_w = np.ones(len(post_w)) / len(post_w) * like_w  # reset to flat * likelihood
            post_w = post_w / (post_w.sum() + 1e-12)

    if effective_sample_size(post_w) < cfg.min_effective_particles:
        post_w = 0.5 * post_w + 0.5 * (np.ones(len(post_w)) / len(post_w))
        post_w = post_w / (post_w.sum() + 1e-12)

    # Particle rejuvenation
    if theta_particles is not None and rng is not None:
        updated_theta, post_w = rejuvenate_particles(
            updated_theta if updated_theta is not None else theta_particles,
            post_w, cfg, rng,
        )

    return post_w, updated_theta


# ============================================================
# 6. Active utility functions
# ============================================================

def aggregate_future_uncertainty(
    rollout_curves: np.ndarray, weights: np.ndarray, times: np.ndarray, future_eval_start: float,
) -> float:
    mask = times > future_eval_start
    if mask.sum() == 0:
        raise ValueError("No future evaluation times found.")
    return float(np.mean(weighted_var(rollout_curves[:, mask], weights)))


def active_timepoint_utility(
    rollout_curves: np.ndarray, times: np.ndarray, prior_weights: np.ndarray,
    candidate_time: float, cfg: Config, rng: np.random.Generator,
) -> float:
    """Cheap variance-based utility proxy.

    Uses the correlation between Q(t_candidate) and Q(t_future) to estimate
    how much observing at candidate_time would reduce future uncertainty.

    proxy = mean_var_future * (1 - mean_corr^2)

    where mean_corr^2 is the average squared correlation between the candidate
    time point and each future time point. This is the Bayesian linear regression
    variance reduction formula — no pseudo-observations needed.

    Time cost = lambda * candidate_time (penalize later observations).
    """
    future_mask = times > cfg.future_eval_start
    if future_mask.sum() == 0:
        return -1e9

    idx = int(np.argmin(np.abs(times - candidate_time)))
    q_candidate = rollout_curves[:, idx]
    q_future = rollout_curves[:, future_mask]

    # Weighted statistics
    w = prior_weights / (prior_weights.sum() + 1e-12)
    mu_c = np.sum(w * q_candidate)
    var_c = np.sum(w * (q_candidate - mu_c) ** 2) + 1e-10

    # Per-future-time variance and covariance
    mu_f = np.sum(w[:, None] * q_future, axis=0)
    var_f = np.sum(w[:, None] * (q_future - mu_f[None, :]) ** 2, axis=0) + 1e-10
    cov_cf = np.sum(w[:, None] * (q_candidate - mu_c)[:, None] * (q_future - mu_f[None, :]), axis=0)

    # Squared correlation per future time point
    corr2 = cov_cf ** 2 / (var_c * var_f)
    corr2 = np.clip(corr2, 0.0, 1.0)

    # Expected future variance after observing candidate = var_f * (1 - corr^2)
    residual_var = var_f * (1.0 - corr2)
    current_var = np.mean(var_f)
    expected_var = np.mean(residual_var)

    # Relative information gain
    rel_gain = (current_var - expected_var) / (current_var + 1e-8)

    # Time-cost penalty
    time_penalty = cfg.time_cost_lambda * candidate_time
    return rel_gain - time_penalty


def choose_active_timepoint(
    rollout_curves: np.ndarray, times: np.ndarray, prior_weights: np.ndarray,
    candidate_times: list[float], cfg: Config, rng: np.random.Generator,
) -> tuple[float, dict[float, float]]:
    utilities = {
        float(t): active_timepoint_utility(rollout_curves, times, prior_weights, float(t), cfg, rng)
        for t in candidate_times
    }
    return max(utilities, key=utilities.get), utilities


# ============================================================
# 7. Conformal calibration
# ============================================================

class ConformalCalibrator:
    """Split conformal calibration for prediction intervals.

    On the calibration set, compute nonconformity scores:
        s_i = max(q_lo - y_i, y_i - q_hi)  for each time point

    Then at test time, expand intervals by the (1-alpha) quantile of scores
    to achieve marginal coverage ≥ 1-alpha.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.alpha = 1.0 - cfg.conformal_target_coverage
        self.scores: list[np.ndarray] = []  # per-calibration-curve scores
        self.method = cfg.conformal_method

    def add_calibration_point(
        self,
        pred_q05: np.ndarray, pred_q95: np.ndarray,
        y_true: np.ndarray, times: np.ndarray,
    ):
        """Compute nonconformity score for one calibration curve.
        Only scores at future time points (> future_eval_start) are stored."""
        mask = times > self.cfg.future_eval_start
        if mask.sum() == 0:
            return
        y = y_true[mask]
        lo = pred_q05[mask]
        hi = pred_q95[mask]
        # Nonconformity: how much does the true value exceed the interval?
        score = np.maximum(lo - y, y - hi)  # positive when outside
        self.scores.append(score)

    def compute_adjustment(self) -> float:
        """Compute the conformal adjustment delta such that adding delta to
        interval width achieves target coverage."""
        if not self.scores:
            return 0.0
        all_scores = np.concatenate(self.scores)
        n = len(all_scores)
        # Finite-sample corrected quantile
        q_idx = int(np.ceil((1.0 - self.alpha) * (n + 1))) - 1
        q_idx = min(q_idx, n - 1)
        q_idx = max(q_idx, 0)
        delta = float(np.sort(all_scores)[q_idx])
        return max(delta, 0.0)  # never shrink intervals

    def apply_adjustment(
        self, pred_q05: np.ndarray, pred_q95: np.ndarray, delta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand symmetric intervals by delta."""
        center = 0.5 * (pred_q05 + pred_q95)
        half_width = 0.5 * (pred_q95 - pred_q05) + delta
        return center - half_width, center + half_width


class LocalConformalCalibrator(ConformalCalibrator):
    """Per-time-bin conformal calibration for BOTH 90% and 80% intervals.

    Tracks nonconformity scores separately for each confidence level,
    so coverage_90 and coverage_80 are independently calibrated.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Bin edges for time points — aligned with future_eval_start
        self.time_bins = [0, 42, 56, 84]
        # Separate score tracking for 90% and 80% intervals
        self.bin_scores_90: dict[int, list[float]] = {i: [] for i in range(len(self.time_bins) - 1)}
        self.bin_scores_80: dict[int, list[float]] = {i: [] for i in range(len(self.time_bins) - 1)}

    def _find_bin(self, t: float) -> int:
        for b in range(len(self.time_bins) - 1):
            if self.time_bins[b] < t <= self.time_bins[b + 1]:
                return b
        return len(self.time_bins) - 2

    def add_calibration_point(
        self,
        pred_q05: np.ndarray, pred_q95: np.ndarray,
        y_true: np.ndarray, times: np.ndarray,
        pred_q10: Optional[np.ndarray] = None, pred_q90: Optional[np.ndarray] = None,
    ):
        mask = times > self.cfg.future_eval_start
        if mask.sum() == 0:
            return
        y = y_true[mask]
        t_masked = times[mask]

        # 90% interval scores
        lo90, hi90 = pred_q05[mask], pred_q95[mask]
        for j in range(len(t_masked)):
            score = max(lo90[j] - y[j], y[j] - hi90[j])
            self.bin_scores_90[self._find_bin(t_masked[j])].append(score)

        # 80% interval scores
        if pred_q10 is not None and pred_q90 is not None:
            lo80, hi80 = pred_q10[mask], pred_q90[mask]
            for j in range(len(t_masked)):
                score = max(lo80[j] - y[j], y[j] - hi80[j])
                self.bin_scores_80[self._find_bin(t_masked[j])].append(score)

    def _compute_quantile(self, scores: list[float], alpha: float) -> float:
        if not scores:
            return 0.0
        arr = np.array(scores)
        n = len(arr)
        q_idx = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
        q_idx = min(max(q_idx, 0), n - 1)
        return max(float(np.sort(arr)[q_idx]), 0.0)

    def compute_adjustment_per_bin(self) -> tuple[dict[int, float], dict[int, float]]:
        """Returns (deltas_90, deltas_80) for both confidence levels."""
        alpha_90 = 1.0 - 0.90
        alpha_80 = 1.0 - 0.80
        deltas_90 = {i: self._compute_quantile(s, alpha_90) for i, s in self.bin_scores_90.items()}
        deltas_80 = {i: self._compute_quantile(s, alpha_80) for i, s in self.bin_scores_80.items()}
        return deltas_90, deltas_80

    def apply_adjustment(
        self, pred_q05: np.ndarray, pred_q95: np.ndarray, times: np.ndarray,
        deltas_90: dict[int, float],
        pred_q10: Optional[np.ndarray] = None, pred_q90: Optional[np.ndarray] = None,
        deltas_80: Optional[dict[int, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply per-time-bin adjustment to both 90% and 80% intervals.

        Returns: (adj_q05, adj_q95, adj_q10_or_None, adj_q90_or_None)
        """
        new_lo90 = pred_q05.copy()
        new_hi90 = pred_q95.copy()
        for j in range(len(times)):
            delta = deltas_90.get(self._find_bin(times[j]), 0.0)
            center = 0.5 * (new_lo90[j] + new_hi90[j])
            hw = 0.5 * (new_hi90[j] - new_lo90[j]) + delta
            new_lo90[j] = center - hw
            new_hi90[j] = center + hw

        new_lo80, new_hi80 = None, None
        if pred_q10 is not None and pred_q90 is not None and deltas_80 is not None:
            new_lo80 = pred_q10.copy()
            new_hi80 = pred_q90.copy()
            for j in range(len(times)):
                delta = deltas_80.get(self._find_bin(times[j]), 0.0)
                center = 0.5 * (new_lo80[j] + new_hi80[j])
                hw = 0.5 * (new_hi80[j] - new_lo80[j]) + delta
                new_lo80[j] = center - hw
                new_hi80[j] = center + hw

        return new_lo90, new_hi90, new_lo80, new_hi80


# ============================================================
# 8. Prediction summary and metrics
# ============================================================

def posterior_predictive_summary(rollout_curves: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    mean_curve = np.sum(rollout_curves * weights[:, None], axis=0)
    q_arr = weighted_curve_quantiles(rollout_curves, weights, [0.05, 0.10, 0.50, 0.90, 0.95])
    return {"mean": mean_curve, "q05": q_arr[0], "q10": q_arr[1], "q50": q_arr[2], "q90": q_arr[3], "q95": q_arr[4]}


def evaluate_prediction(
    y_true: np.ndarray, pred_summary: dict[str, np.ndarray], times: np.ndarray, cfg: Config,
    posterior_samples: Optional[np.ndarray] = None, posterior_weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    mask = times > cfg.future_eval_start
    y = y_true[mask]
    result = {
        "future_rmse_mean": rmse(y, pred_summary["mean"][mask]),
        "future_rmse_median": rmse(y, pred_summary["q50"][mask]),
        "future_mae_median": float(np.mean(np.abs(y - pred_summary["q50"][mask]))),
        "coverage_90": coverage(y, pred_summary["q05"][mask], pred_summary["q95"][mask]),
        "coverage_80": coverage(y, pred_summary["q10"][mask], pred_summary["q90"][mask]),
        "width_90": interval_width(pred_summary["q05"][mask], pred_summary["q95"][mask]),
        "width_80": interval_width(pred_summary["q10"][mask], pred_summary["q90"][mask]),
    }
    if posterior_samples is not None and posterior_weights is not None:
        if rng is None:
            rng = np.random.default_rng(cfg.random_state)
        idx = weighted_resample_indices(posterior_weights, 200, rng)
        result["crps"] = ensemble_crps_unweighted(posterior_samples[idx][:, mask], y, rng=rng)
    return result


# ============================================================
# 9. Direct-Q baseline (same as v2)
# ============================================================

class DirectQOnePointBaseline:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.feature_cols = None
        self.future_times = None
        self.model = None
        self.preprocessor = None

    def fit(self, formulations_train: pd.DataFrame, curves_train: pd.DataFrame):
        cfg = self.cfg
        times = np.asarray(cfg.rollout_times, dtype=float)
        future_mask = times > cfg.future_eval_start
        self.future_times = times[future_mask]
        self.feature_cols = [c for c in formulations_train.columns if c != cfg.id_col]

        rows, targets = [], []
        curve_groups = {cid: g for cid, g in curves_train.groupby(cfg.id_col)}
        form_by_id = formulations_train.set_index(cfg.id_col)

        for cid, cdf in curve_groups.items():
            if cid not in form_by_id.index:
                continue
            true_full = interp_curve(cdf, times, cfg.time_col, cfg.release_col)
            y_future = true_full[future_mask]
            form_row = form_by_id.loc[cid].to_dict()
            for obs_t in sorted(set(list(cfg.candidate_times) + list(cfg.candidate_times_step2))):
                obs_q = float(interp_curve(cdf, np.array([obs_t]), cfg.time_col, cfg.release_col)[0])
                row = dict(form_row)
                row["obs_time"] = float(obs_t)
                row["obs_q"] = obs_q
                rows.append(row)
                targets.append(y_future)

        X_aug = pd.DataFrame(rows)
        Y = np.asarray(targets)

        numeric_cols = X_aug.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X_aug.columns if c not in numeric_cols]
        self.preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ], remainder="drop")

        X_proc = self.preprocessor.fit_transform(X_aug)
        self.model = ExtraTreesRegressor(
            n_estimators=500, random_state=cfg.random_state,
            min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1,
        )
        self.model.fit(X_proc, Y)
        if cfg.verbose:
            print(f"[direct-Q] trained on {len(X_aug)} augmented rows")
        return self

    def predict_future(self, formulation_row: pd.DataFrame, obs_time: float, obs_q: float) -> np.ndarray:
        row = formulation_row.drop(columns=[self.cfg.id_col]).copy()
        row["obs_time"] = float(obs_time)
        row["obs_q"] = float(obs_q)
        X_proc = self.preprocessor.transform(row)
        return np.clip(self.model.predict(X_proc)[0], 0.0, 1.1)


# ============================================================
# 10. Main evaluation loop — v3
# ============================================================

def evaluate_active_observer_v3(cfg: Config):
    ensure_dir(cfg.output_dir)
    rng = np.random.default_rng(cfg.random_state)

    formulations, curves, theta = load_data(cfg)
    train_ids, cal_ids, test_ids, split_type = make_train_cal_test_ids(formulations, cfg)

    formulations_train = formulations[formulations[cfg.id_col].isin(train_ids)].copy()
    formulations_cal = formulations[formulations[cfg.id_col].isin(cal_ids)].copy()
    formulations_test = formulations[formulations[cfg.id_col].isin(test_ids)].copy()
    curves_train = curves[curves[cfg.id_col].isin(train_ids)].copy()
    curves_cal = curves[curves[cfg.id_col].isin(cal_ids)].copy()
    curves_test = curves[curves[cfg.id_col].isin(test_ids)].copy()
    theta_train = theta[theta[cfg.id_col].isin(train_ids)].copy()

    theta_curve_level = collapse_theta_to_curve_level(theta_train, cfg)

    if cfg.verbose:
        print(f"[split] train={len(train_ids)}, cal={len(cal_ids)}, test={len(test_ids)}")

    # Fit prior on training data
    prior = FormulationThetaPrior(cfg)
    prior.fit(formulations_train, theta_curve_level)

    # Fit Direct-Q baseline on training data
    direct_q = None
    if cfg.train_direct_q_baseline:
        direct_q = DirectQOnePointBaseline(cfg)
        direct_q.fit(formulations_train, curves_train)

    times = np.asarray(cfg.rollout_times, dtype=float)
    candidate_times = list(cfg.candidate_times)

    # ============================================================
    # Phase A: Conformal calibration on calibration set
    # ============================================================
    if cfg.verbose:
        print("\n[conformal] Computing calibration on held-out calibration set...")

    if cfg.conformal_method == "local":
        calibrator = LocalConformalCalibrator(cfg)
    else:
        calibrator = ConformalCalibrator(cfg)

    curve_groups_cal = {cid: g for cid, g in curves_cal.groupby(cfg.id_col)}

    candidate_times_all = sorted(set(list(cfg.candidate_times) + list(cfg.candidate_times_step2)))

    for _, row in formulations_cal.iterrows():
        cid = row[cfg.id_col]
        if cid not in curve_groups_cal:
            continue

        formulation_row = row.to_frame().T
        true_curve = interp_curve(curve_groups_cal[cid], times, cfg.time_col, cfg.release_col)

        theta_particles = prior.sample_prior_particles(formulation_row=formulation_row, rng=rng)
        rollout_curves = simulate_plga_ode(theta_particles, times)
        n_particles = rollout_curves.shape[0]
        prior_weights = np.ones(n_particles) / n_particles

        # Run active selection on calibration curves (same procedure as test time)
        # to ensure calibration score distribution matches test-time distribution.
        active_t, _ = choose_active_timepoint(
            rollout_curves, times, prior_weights, list(cfg.candidate_times), cfg, rng,
        )
        active_q = float(interp_curve(curve_groups_cal[cid], np.array([active_t]), cfg.time_col, cfg.release_col)[0])
        post_w, cal_theta = posterior_update_from_observation(
            rollout_curves, times, prior_weights, active_t, active_q, cfg,
            theta_particles, rng,
        )
        if cal_theta is not None:
            cal_rollout = simulate_plga_ode(cal_theta, times)
        else:
            cal_rollout = rollout_curves
        summary = posterior_predictive_summary(cal_rollout, post_w)
        calibrator.add_calibration_point(
            summary["q05"], summary["q95"], true_curve, times,
            pred_q10=summary["q10"], pred_q90=summary["q90"],
        )

    if isinstance(calibrator, LocalConformalCalibrator):
        cal_deltas_90, cal_deltas_80 = calibrator.compute_adjustment_per_bin()
        if cfg.verbose:
            print(f"[conformal] Per-bin 90% adjustments: {cal_deltas_90}")
            print(f"[conformal] Per-bin 80% adjustments: {cal_deltas_80}")
    else:
        cal_delta = calibrator.compute_adjustment()
        if cfg.verbose:
            print(f"[conformal] Global adjustment: {cal_delta:.4f}")

    def apply_conformal(summary: dict[str, np.ndarray], times_arr: np.ndarray) -> dict[str, np.ndarray]:
        """Apply conformal calibration to a prediction summary dict."""
        if isinstance(calibrator, LocalConformalCalibrator):
            adj_q05, adj_q95, adj_q10, adj_q90 = calibrator.apply_adjustment(
                summary["q05"], summary["q95"], times_arr, cal_deltas_90,
                pred_q10=summary["q10"], pred_q90=summary["q90"], deltas_80=cal_deltas_80,
            )
        else:
            adj_q05, adj_q95 = calibrator.apply_adjustment(summary["q05"], summary["q95"], cal_delta)
            adj_q10, adj_q90 = summary["q10"], summary["q90"]  # no 80% calibration for global
        out = dict(summary)
        out["q05"] = adj_q05
        out["q95"] = adj_q95
        out["q10"] = adj_q10
        out["q90"] = adj_q90
        return out

    # ============================================================
    # Phase B: Evaluate on test set
    # ============================================================
    if cfg.verbose:
        print("\n[test] Evaluating on test set...")

    all_records = []
    utility_records = []
    prediction_records = []
    curve_groups_test = {cid: g for cid, g in curves_test.groupby(cfg.id_col)}

    for _, row in formulations_test.iterrows():
        cid = row[cfg.id_col]
        if cid not in curve_groups_test:
            continue

        formulation_row = row.to_frame().T
        true_curve = interp_curve(curve_groups_test[cid], times, cfg.time_col, cfg.release_col)

        # --- 1. Prior particles from formulation only ---
        theta_particles = prior.sample_prior_particles(formulation_row=formulation_row, rng=rng)
        rollout_curves = simulate_plga_ode(theta_particles, times)
        n_particles = rollout_curves.shape[0]
        prior_weights = np.ones(n_particles) / n_particles

        # --- 2. Zero-early prior prediction (unadjusted) ---
        prior_summary = posterior_predictive_summary(rollout_curves, prior_weights)
        prior_metrics = evaluate_prediction(true_curve, prior_summary, times, cfg, rollout_curves, prior_weights, rng)
        prior_metrics.update({
            "curve_id": cid, "strategy": "zero_early_prior_only",
            "obs_time": np.nan, "obs_q": np.nan, "ess_after": effective_sample_size(prior_weights),
        })
        all_records.append(prior_metrics)

        # --- 3. Zero-early prior + conformal adjustment ---
        prior_cal_summary = apply_conformal(prior_summary, times)
        prior_cal_metrics = evaluate_prediction(true_curve, prior_cal_summary, times, cfg)
        prior_cal_metrics.update({
            "curve_id": cid, "strategy": "zero_early_prior_conformal",
            "obs_time": np.nan, "obs_q": np.nan, "ess_after": effective_sample_size(prior_weights),
        })
        all_records.append(prior_cal_metrics)

        # --- 4. Active ONE-point selection ---
        active_t, utilities = choose_active_timepoint(
            rollout_curves, times, prior_weights, candidate_times, cfg, rng,
        )
        for t, u in utilities.items():
            utility_records.append({"curve_id": cid, "candidate_time": t, "utility": u, "chosen": float(t == active_t), "step": 1})

        active_q = float(interp_curve(curve_groups_test[cid], np.array([active_t]), cfg.time_col, cfg.release_col)[0])
        active_weights, active_theta = posterior_update_from_observation(
            rollout_curves, times, prior_weights, active_t, active_q, cfg,
            theta_particles, rng,
        )
        # Re-rollout with updated theta if rejuvenation happened
        if active_theta is not None:
            rollout_after_1 = simulate_plga_ode(active_theta, times)
        else:
            rollout_after_1 = rollout_curves

        active_summary = posterior_predictive_summary(rollout_after_1, active_weights)
        active_cal_summary = apply_conformal(active_summary, times)

        active_metrics = evaluate_prediction(true_curve, active_cal_summary, times, cfg, rollout_after_1, active_weights, rng)
        active_metrics.update({
            "curve_id": cid, "strategy": "active_one_point_conformal",
            "obs_time": active_t, "obs_q": active_q, "ess_after": effective_sample_size(active_weights),
        })
        all_records.append(active_metrics)

        for j, t in enumerate(times):
            prediction_records.append({
                "curve_id": cid, "strategy": "active_one_point_conformal", "time": t,
                "true_release": true_curve[j], "pred_mean": active_summary["mean"][j],
                "pred_q05": active_cal_summary["q05"][j], "pred_q50": active_summary["q50"][j],
                "pred_q95": active_cal_summary["q95"][j], "obs_time": active_t, "obs_q": active_q,
            })

        # --- 5. Active TWO-point sequential ---
        # Step 1 already done above (active_t, active_q). Now select step 2
        # from the step-2 candidate pool (later time points).
        step2_candidates = list(cfg.candidate_times_step2)
        # Remove any candidate <= active_t (can't observe in the past)
        step2_candidates = [t for t in step2_candidates if t > active_t]
        if step2_candidates:
            active2_t, utilities2 = choose_active_timepoint(
                rollout_after_1, times, active_weights, step2_candidates, cfg, rng,
            )
            for t, u in utilities2.items():
                utility_records.append({"curve_id": cid, "candidate_time": t, "utility": u, "chosen": float(t == active2_t), "step": 2})

            active2_q = float(interp_curve(curve_groups_test[cid], np.array([active2_t]), cfg.time_col, cfg.release_col)[0])
            cfg_step2 = cfg
            if cfg.step2_q_max_cap_margin is not None:
                cfg_step2 = replace(cfg, q_max_cap_margin=cfg.step2_q_max_cap_margin)
            active2_weights, active2_theta = posterior_update_from_observation(
                rollout_after_1, times, active_weights, active2_t, active2_q, cfg_step2,
                active_theta, rng,
            )
            if active2_theta is not None:
                rollout_after_2 = simulate_plga_ode(active2_theta, times)
            else:
                rollout_after_2 = rollout_after_1

            active2_summary = posterior_predictive_summary(rollout_after_2, active2_weights)
            active2_cal_summary = apply_conformal(active2_summary, times)

            active2_metrics = evaluate_prediction(true_curve, active2_cal_summary, times, cfg, rollout_after_2, active2_weights, rng)
            active2_metrics.update({
                "curve_id": cid, "strategy": "active_two_point_conformal",
                "obs_time": active2_t, "obs_q": active2_q,
                "obs_time_1": active_t, "obs_q_1": active_q,
                "ess_after": effective_sample_size(active2_weights),
            })
            all_records.append(active2_metrics)

            for j, t in enumerate(times):
                prediction_records.append({
                    "curve_id": cid, "strategy": "active_two_point_conformal", "time": t,
                    "true_release": true_curve[j], "pred_mean": active2_summary["mean"][j],
                    "pred_q05": active2_cal_summary["q05"][j], "pred_q50": active2_summary["q50"][j],
                    "pred_q95": active2_cal_summary["q95"][j], "obs_time": active2_t, "obs_q": active2_q,
                })

        # --- 6. Fixed one-point baselines (with conformal) ---
        all_candidate_times = sorted(set(list(cfg.candidate_times) + list(cfg.candidate_times_step2)))
        for fixed_t in all_candidate_times:
            fixed_q = float(interp_curve(curve_groups_test[cid], np.array([fixed_t]), cfg.time_col, cfg.release_col)[0])
            fixed_weights, fixed_theta = posterior_update_from_observation(
                rollout_curves, times, prior_weights, fixed_t, fixed_q, cfg,
                theta_particles, rng,
            )
            if fixed_theta is not None:
                fixed_rollout = simulate_plga_ode(fixed_theta, times)
            else:
                fixed_rollout = rollout_curves

            fixed_summary = posterior_predictive_summary(fixed_rollout, fixed_weights)
            fixed_cal_summary = apply_conformal(fixed_summary, times)

            fixed_metrics = evaluate_prediction(true_curve, fixed_cal_summary, times, cfg, fixed_rollout, fixed_weights, rng)
            fixed_metrics.update({
                "curve_id": cid, "strategy": f"fixed_{fixed_t:g}d_conformal",
                "obs_time": fixed_t, "obs_q": fixed_q, "ess_after": effective_sample_size(fixed_weights),
            })
            all_records.append(fixed_metrics)

        # --- 7. Fixed FOUR-point baseline (1d + 3d + 5d + 7d) ---
        four_point_times = [1.0, 3.0, 5.0, 7.0]
        four_weights = prior_weights.copy()
        four_theta = theta_particles.copy()
        four_rollout = rollout_curves.copy()
        for ft in four_point_times:
            fq = float(interp_curve(curve_groups_test[cid], np.array([ft]), cfg.time_col, cfg.release_col)[0])
            four_weights, four_theta = posterior_update_from_observation(
                four_rollout, times, four_weights, ft, fq, cfg,
                four_theta, rng,
            )
            if four_theta is not None:
                four_rollout = simulate_plga_ode(four_theta, times)

        four_summary = posterior_predictive_summary(four_rollout, four_weights)
        four_cal_summary = apply_conformal(four_summary, times)

        four_metrics = evaluate_prediction(true_curve, four_cal_summary, times, cfg, four_rollout, four_weights, rng)
        four_metrics.update({
            "curve_id": cid, "strategy": "fixed_four_point_conformal",
            "obs_time": 7.0, "obs_q": np.nan, "ess_after": effective_sample_size(four_weights),
        })
        all_records.append(four_metrics)

        # --- 8. Direct-Q baselines ---
        if direct_q is not None:
            for label, obs_t in [("active_time", active_t)] + [(f"fixed_{t:g}d", t) for t in candidate_times]:
                obs_q = float(interp_curve(curve_groups_test[cid], np.array([obs_t]), cfg.time_col, cfg.release_col)[0])
                pred_future = direct_q.predict_future(formulation_row, obs_t, obs_q)
                future_mask = times > cfg.future_eval_start
                y_future = true_curve[future_mask]
                all_records.append({
                    "curve_id": cid, "strategy": f"directQ_one_point_{label}",
                    "obs_time": obs_t, "obs_q": obs_q,
                    "future_rmse_mean": rmse(y_future, pred_future),
                    "future_rmse_median": rmse(y_future, pred_future),
                    "future_mae_median": float(np.mean(np.abs(y_future - pred_future))),
                    "coverage_90": np.nan, "coverage_80": np.nan,
                    "width_90": np.nan, "width_80": np.nan, "crps": np.nan, "ess_after": np.nan,
                })

    # ============================================================
    # Save outputs
    # ============================================================
    metrics_df = pd.DataFrame(all_records)
    utility_df = pd.DataFrame(utility_records)
    pred_df = pd.DataFrame(prediction_records)

    metrics_df.to_csv(os.path.join(cfg.output_dir, "metrics_by_curve.csv"), index=False)
    utility_df.to_csv(os.path.join(cfg.output_dir, "active_utilities.csv"), index=False)
    pred_df.to_csv(os.path.join(cfg.output_dir, "prediction_curves_active.csv"), index=False)

    summary = (
        metrics_df.groupby("strategy").agg(
            n=("curve_id", "count"),
            rmse_mean=("future_rmse_mean", "mean"),
            rmse_median=("future_rmse_mean", "median"),
            mae_median=("future_mae_median", "median"),
            coverage90_mean=("coverage_90", "mean"),
            coverage80_mean=("coverage_80", "mean"),
            width90_mean=("width_90", "mean"),
            width80_mean=("width_80", "mean"),
            crps_mean=("crps", "mean"),
            ess_median=("ess_after", "median"),
        ).reset_index().sort_values("rmse_mean")
    )
    summary.to_csv(os.path.join(cfg.output_dir, "metrics_summary.csv"), index=False)

    # Active utility analysis
    utility_summary = (
        utility_df.groupby(["step", "candidate_time"]).agg(
            mean_utility=("utility", "mean"),
            frac_chosen=("chosen", "mean"),
        ).reset_index()
    )
    utility_summary.to_csv(os.path.join(cfg.output_dir, "utility_summary.csv"), index=False)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    print("\n================ Summary ================")
    print(f"Split type: {split_type}")
    print(f"Train={len(train_ids)}, Cal={len(cal_ids)}, Test={len(test_ids)}")
    print(summary.to_string(index=False))
    print("\n--- Active utility by step ---")
    print(utility_summary.to_string(index=False))
    print(f"\nSaved to {cfg.output_dir}/")

    return metrics_df, summary, utility_df, pred_df


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Active Kinetic Observer v3")
    parser.add_argument("--output-dir", default="outputs_active_observer_v3")
    parser.add_argument("--candidate-times", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    parser.add_argument("--candidate-times-step2", nargs="+", type=float, default=[7.0, 10.0, 14.0, 21.0])
    parser.add_argument("--future-eval-start", type=float, default=28.0)
    parser.add_argument("--time-cost-lambda", type=float, default=0.005)
    parser.add_argument("--q-max-correction", type=float, default=0.10)
    parser.add_argument("--step2-q-max-cap-margin", type=float, default=None)
    parser.add_argument("--jitter-scale", type=float, default=0.5)
    parser.add_argument("--likelihood-beta", type=float, default=2.0)
    parser.add_argument("--conformal-method", choices=["global", "local"], default="local")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cal-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--group-split", type=str, default=None,
                        help="Column name for group-based split (e.g. 'DP_Group')")
    parser.add_argument("--no-direct-q", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        output_dir=args.output_dir,
        candidate_times=tuple(args.candidate_times),
        candidate_times_step2=tuple(args.candidate_times_step2),
        future_eval_start=args.future_eval_start,
        time_cost_lambda=args.time_cost_lambda,
        q_max_correction=args.q_max_correction,
        step2_q_max_cap_margin=args.step2_q_max_cap_margin,
        jitter_scale=args.jitter_scale,
        likelihood_beta=args.likelihood_beta,
        conformal_method=args.conformal_method,
        group_split_col=args.group_split,
        test_size=args.test_size,
        cal_size=args.cal_size,
        random_state=args.random_state,
        train_direct_q_baseline=not args.no_direct_q,
        verbose=not args.quiet,
    )
    evaluate_active_observer_v3(cfg)
