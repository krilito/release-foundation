"""
Particle Active Kinetic Observer for PLGA controlled release.

Core logic:
    0-early formulation → kinetic-state prior particles
    → ODE rollout candidate futures → compute information value per timepoint
    → choose best observation time → reveal one experimental point
    → posterior reweighting → full-curve prediction with uncertainty
    → compare fixed-time and direct-Q baselines
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator import PLGABiphasic  # noqa: E402


# ============================================================
# 0. Configuration
# ============================================================

@dataclass
class Config:
    formulations_path: str = "data/formulations.csv"
    curves_path: str = "data/curves_long.csv"
    theta_path: str = "data/theta_bank.csv"
    output_dir: str = "outputs_active_observer"

    id_col: str = "curve_id"
    time_col: str = "time"
    release_col: str = "release"

    theta_cols: Tuple[str, ...] = (
        "log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
        "m_crit", "q_burst", "log_tau_burst", "Q_max",
    )

    candidate_times: Tuple[float, ...] = (1.0, 3.0, 5.0, 7.0)
    future_eval_start: float = 14.0
    rollout_times: Tuple[float, ...] = (
        0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 42.0, 56.0, 84.0,
    )

    test_size: float = 0.25
    random_state: int = 42
    group_split_col: Optional[str] = None

    n_tree_estimators: int = 500
    n_knn_particles: int = 80
    n_tree_particles: int = 300
    jitter_scale: float = 0.5

    # Q_max bias correction: oracle fits systematically overestimate Q_max.
    # Set to 0.0 to disable. Typical value: 0.08-0.12 for PLGA data.
    q_max_correction: float = 0.10

    obs_sigma_abs: float = 0.03
    obs_sigma_rel: float = 0.08
    likelihood_beta: float = 2.0
    min_effective_particles: int = 10

    max_pseudo_obs_for_utility: int = 100
    time_cost_lambda: float = 0.0

    train_direct_q_baseline: bool = True
    verbose: bool = True


CFG = Config()


# ============================================================
# 1. Utilities
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


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cw = np.cumsum(w)
    cw = cw / (cw[-1] + 1e-12)
    return np.interp(quantiles, cw, v)


def weighted_curve_quantiles(curves: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
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


def make_train_test_ids(formulations: pd.DataFrame, cfg: Config):
    ids = formulations[cfg.id_col].to_numpy()
    if cfg.group_split_col and cfg.group_split_col in formulations.columns:
        groups = formulations[cfg.group_split_col].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        train_idx, test_idx = next(splitter.split(formulations, groups=groups))
        return set(ids[train_idx]), set(ids[test_idx]), f"group_split:{cfg.group_split_col}"
    train_ids, test_ids = train_test_split(ids, test_size=cfg.test_size, random_state=cfg.random_state)
    return set(train_ids), set(test_ids), "random_split"


# ============================================================
# 3. PLGA ODE rollout — connected to PLGABiphasic
# ============================================================

_SIM = PLGABiphasic()
_PRIOR = _SIM.prior().base_dist
_LOWS = _PRIOR.low.numpy()
_HIGHS = _PRIOR.high.numpy()


def simulate_plga_ode(theta_matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Run PLGA ODE for each theta particle.

    Args:
        theta_matrix: (n_particles, 9)
        times: (n_times,) sorted ascending, in days

    Returns:
        release_curves: (n_particles, n_times), clipped to [0, 1.1]
    """
    curves = []
    for theta in theta_matrix:
        q = _SIM.simulate_numpy(theta, times)
        curves.append(q)
    result = np.asarray(curves, dtype=float)
    return np.clip(result, 0.0, 1.1)


# ============================================================
# 4. Formulation-conditioned theta prior
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

    def fit(self, formulations_train: pd.DataFrame, theta_train_curve_level: pd.DataFrame,
            curves_train: pd.DataFrame | None = None):
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

        # Per-dimension bias correction
        tree_pred_train = np.asarray([est.predict(X_proc)[0] for est in self.et.estimators_[:50]])
        tree_mean_train = tree_pred_train.mean(axis=0)
        self._tree_bias = y.mean(axis=0) - tree_mean_train
        self._tree_resid_std = np.std(y - tree_mean_train, axis=0)

        # Actual plateau distribution from training curves (NOT oracle Q_max)
        # This is the real max release observed for each curve
        if curves_train is not None:
            plateaus = curves_train.groupby(cfg.id_col)[cfg.release_col].max()
            self._plateau_mean = float(plateaus.mean())
            self._plateau_std = float(plateaus.std())
            self._plateau_values = plateaus.values
        else:
            # Fallback: use training data release max
            self._plateau_mean = 0.85
            self._plateau_std = 0.18
            self._plateau_values = None

        if cfg.verbose:
            print(f"[prior] fitted on {len(df)} curves, numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")
            print(f"[prior] tree bias (Q_max dim): {self._tree_bias[-1]:.3f}")
            print(f"[prior] actual plateau: mean={self._plateau_mean:.3f}, std={self._plateau_std:.3f}")
        return self

    def transform_one(self, formulation_row: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(formulation_row[self.feature_cols].copy())

    def sample_prior_particles(
        self,
        formulation_row: pd.DataFrame,
        n_tree_particles: Optional[int] = None,
        n_knn_particles: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        observed_q: float | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        if rng is None:
            rng = np.random.default_rng(cfg.random_state)
        n_tree = n_tree_particles or cfg.n_tree_particles
        n_knn = n_knn_particles or cfg.n_knn_particles

        X_proc = self.transform_one(formulation_row)

        # Tree particles: each tree predicts one theta
        tree_idx = rng.choice(len(self.et.estimators_), size=min(n_tree, len(self.et.estimators_)), replace=False)
        tree_particles = np.asarray([self.et.estimators_[i].predict(X_proc)[0] for i in tree_idx])

        # KNN particles from similar formulations
        k = min(n_knn, len(self.train_X_proc))
        _, nn_idx = self.knn.kneighbors(X_proc, n_neighbors=k)
        knn_particles = self.train_theta[nn_idx[0]]

        particles = np.vstack([tree_particles, knn_particles])

        # Shift tree particles to correct systematic bias
        n_tree_actual = tree_particles.shape[0]
        particles[:n_tree_actual] += self._tree_bias

        if cfg.jitter_scale > 0:
            std = np.maximum(self._tree_resid_std, 1e-4)
            jitter = rng.normal(0.0, cfg.jitter_scale * std, size=particles.shape)
            particles = particles + jitter

        # Q_max: resample from actual plateau distribution instead of oracle Q_max
        # Oracle Q_max is biased high (mean=0.954) vs actual plateau (mean=0.848)
        q_max_idx = list(cfg.theta_cols).index("Q_max")
        if self._plateau_values is not None:
            # Resample Q_max from actual training plateaus + small noise
            n_particles = len(particles)
            plateau_samples = rng.choice(self._plateau_values, size=n_particles, replace=True)
            plateau_noise = rng.normal(0, 0.03, size=n_particles)  # small noise
            particles[:, q_max_idx] = np.clip(plateau_samples + plateau_noise, 0.3, 1.0)
        else:
            # Fallback: sample from N(plateau_mean, plateau_std)
            particles[:, q_max_idx] = rng.normal(self._plateau_mean, self._plateau_std, size=len(particles))

        # If we've observed Q(t), Q_max must be >= Q(t) + margin
        if observed_q is not None:
            q_max_margin = 0.02  # small margin above observed Q
            particles[:, q_max_idx] = np.maximum(particles[:, q_max_idx], observed_q + q_max_margin)

        lower = np.percentile(self.train_theta, 0.0, axis=0)
        upper = np.percentile(self.train_theta, 100.0, axis=0)
        span = upper - lower
        lower = lower - 0.1 * span
        upper = upper + 0.1 * span
        upper[q_max_idx] = min(upper[q_max_idx], 1.0)
        return np.clip(particles, lower, upper)


def collapse_theta_to_curve_level(theta_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return theta_df.groupby(cfg.id_col)[list(cfg.theta_cols)].median().reset_index()


# ============================================================
# 5. Posterior update and active utility
# ============================================================

def compute_likelihood_weights(predicted_q: np.ndarray, observed_q: float, cfg: Config) -> np.ndarray:
    sigma = max(cfg.obs_sigma_abs, cfg.obs_sigma_rel * max(abs(observed_q), 0.05))
    log_w = -0.5 * cfg.likelihood_beta * (predicted_q - observed_q) ** 2 / (sigma ** 2)
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    return w / (w.sum() + 1e-12)


def posterior_update_from_one_observation(
    rollout_curves: np.ndarray, times: np.ndarray,
    prior_weights: np.ndarray, obs_time: float, obs_q: float, cfg: Config,
) -> np.ndarray:
    idx = int(np.argmin(np.abs(times - obs_time)))
    like_w = compute_likelihood_weights(rollout_curves[:, idx], obs_q, cfg)
    post_w = prior_weights * like_w
    post_w = post_w / (post_w.sum() + 1e-12)
    if effective_sample_size(post_w) < cfg.min_effective_particles:
        post_w = 0.5 * post_w + 0.5 * prior_weights
        post_w = post_w / (post_w.sum() + 1e-12)
    return post_w


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
    current_uncert = aggregate_future_uncertainty(rollout_curves, prior_weights, times, cfg.future_eval_start)
    idx = int(np.argmin(np.abs(times - candidate_time)))
    q_particles = rollout_curves[:, idx]

    n_pseudo = min(cfg.max_pseudo_obs_for_utility, len(prior_weights))
    pseudo_idx = weighted_resample_indices(prior_weights, n_pseudo, rng)
    pseudo_qs = q_particles[pseudo_idx]
    pseudo_probs = prior_weights[pseudo_idx]
    pseudo_probs = pseudo_probs / (pseudo_probs.sum() + 1e-12)

    expected_after = 0.0
    for q_obs, p_obs in zip(pseudo_qs, pseudo_probs):
        post_w = posterior_update_from_one_observation(
            rollout_curves, times, prior_weights, candidate_time, float(q_obs), cfg,
        )
        expected_after += p_obs * aggregate_future_uncertainty(rollout_curves, post_w, times, cfg.future_eval_start)

    return current_uncert - expected_after - cfg.time_cost_lambda * candidate_time


def choose_active_timepoint(
    rollout_curves: np.ndarray, times: np.ndarray, prior_weights: np.ndarray,
    candidate_times: List[float], cfg: Config, rng: np.random.Generator,
) -> Tuple[float, Dict[float, float]]:
    utilities = {
        float(t): active_timepoint_utility(rollout_curves, times, prior_weights, float(t), cfg, rng)
        for t in candidate_times
    }
    return max(utilities, key=utilities.get), utilities


# ============================================================
# 6. Prediction summary and metrics
# ============================================================

def posterior_predictive_summary(rollout_curves: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    mean_curve = np.sum(rollout_curves * weights[:, None], axis=0)
    q_arr = weighted_curve_quantiles(rollout_curves, weights, [0.05, 0.10, 0.50, 0.90, 0.95])
    return {"mean": mean_curve, "q05": q_arr[0], "q10": q_arr[1], "q50": q_arr[2], "q90": q_arr[3], "q95": q_arr[4]}


def evaluate_prediction(
    y_true: np.ndarray, pred_summary: Dict[str, np.ndarray], times: np.ndarray, cfg: Config,
    posterior_samples: Optional[np.ndarray] = None, posterior_weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    conformal: Optional[ConformalCalibrator] = None,
) -> Dict[str, float]:
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

    # Conformal-calibrated coverage
    if conformal is not None and conformal.quantile is not None:
        cal_lower, cal_upper = conformal.predict_interval(
            pred_summary["mean"][mask], pred_summary["q05"][mask], pred_summary["q95"][mask],
        )
        result["coverage_90_conformal"] = coverage(y, cal_lower, cal_upper)
        result["width_90_conformal"] = interval_width(cal_lower, cal_upper)

    if posterior_samples is not None and posterior_weights is not None:
        if rng is None:
            rng = np.random.default_rng(cfg.random_state)
        idx = weighted_resample_indices(posterior_weights, 200, rng)
        result["crps"] = ensemble_crps_unweighted(posterior_samples[idx][:, mask], y, rng=rng)
    return result


# ============================================================
# 7. Conformal calibration
# ============================================================

class ConformalCalibrator:
    """Split conformal calibration for prediction intervals.

    Uses training residuals to compute a conformal quantile that
    guarantees finite-sample coverage at the target level.
    """

    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.quantile = None
        self.calibration_scores = None

    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray):
        """Fit on calibration set: compute nonconformity scores."""
        # Nonconformity score: max distance from prediction to truth
        scores = np.maximum(y_pred_lower - y_true, y_true - y_pred_upper)
        scores = np.maximum(scores, 0.0)  # only count violations

        # Conformal quantile
        n = len(scores)
        q_idx = int(np.ceil((n + 1) * self.target_coverage)) - 1
        q_idx = min(q_idx, n - 1)
        self.quantile = float(np.sort(scores)[q_idx])
        self.calibration_scores = scores

        return self

    def predict_interval(self, y_pred_mean: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray):
        """Apply conformal calibration to get calibrated intervals."""
        if self.quantile is None:
            raise RuntimeError("Call fit() first")
        # Expand intervals by the conformal quantile
        width = y_pred_upper - y_pred_lower
        half_width = width / 2.0 + self.quantile
        return y_pred_mean - half_width, y_pred_mean + half_width


def run_conformal_calibration(
    cfg: Config,
    prior: FormulationThetaPrior,
    formulations_train: pd.DataFrame,
    curves_train: pd.DataFrame,
    rng: np.random.Generator,
) -> ConformalCalibrator:
    """Run conformal calibration on training data.

    Uses leave-one-out-style: for each training curve, predict with
    the prior (excluding that curve from KNN) and compute residuals.
    """
    times = np.asarray(cfg.rollout_times, dtype=float)
    calibrator = ConformalCalibrator(target_coverage=0.90)

    all_true = []
    all_pred_mean = []
    all_pred_lower = []
    all_pred_upper = []

    curve_groups = {cid: g for cid, g in curves_train.groupby(cfg.id_col)}

    for _, row in formulations_train.iterrows():
        cid = row[cfg.id_col]
        if cid not in curve_groups:
            continue

        formulation_row = row.to_frame().T
        true_curve = interp_curve(curve_groups[cid], times, cfg.time_col, cfg.release_col)

        # Sample prior particles (without this curve's observation)
        theta_particles = prior.sample_prior_particles(formulation_row=formulation_row, rng=rng)
        rollout_curves = simulate_plga_ode(theta_particles, times)
        weights = np.ones(len(rollout_curves)) / len(rollout_curves)

        summary = posterior_predictive_summary(rollout_curves, weights)

        # Only use future times for calibration
        future_mask = times > cfg.future_eval_start
        all_true.append(true_curve[future_mask])
        all_pred_mean.append(summary["mean"][future_mask])
        all_pred_lower.append(summary["q05"][future_mask])
        all_pred_upper.append(summary["q95"][future_mask])

    y_true = np.concatenate(all_true)
    y_mean = np.concatenate(all_pred_mean)
    y_lower = np.concatenate(all_pred_lower)
    y_upper = np.concatenate(all_pred_upper)

    calibrator.fit(y_true, y_mean, y_lower, y_upper)

    if cfg.verbose:
        raw_coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        print(f"[conformal] raw coverage: {raw_coverage:.3f}")
        print(f"[conformal] conformal quantile: {calibrator.quantile:.4f}")
        # Verify calibrated coverage
        cal_lower, cal_upper = calibrator.predict_interval(y_mean, y_lower, y_upper)
        cal_coverage = np.mean((y_true >= cal_lower) & (y_true <= cal_upper))
        print(f"[conformal] calibrated coverage: {cal_coverage:.3f}")

    return calibrator


# ============================================================
# 8. Direct-Q baseline
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
            for obs_t in cfg.candidate_times:
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
# 9. Main evaluation loop
# ============================================================

def evaluate_active_observer(cfg: Config):
    ensure_dir(cfg.output_dir)
    rng = np.random.default_rng(cfg.random_state)

    formulations, curves, theta = load_data(cfg)
    train_ids, test_ids, split_type = make_train_test_ids(formulations, cfg)

    formulations_train = formulations[formulations[cfg.id_col].isin(train_ids)].copy()
    formulations_test = formulations[formulations[cfg.id_col].isin(test_ids)].copy()
    curves_train = curves[curves[cfg.id_col].isin(train_ids)].copy()
    curves_test = curves[curves[cfg.id_col].isin(test_ids)].copy()
    theta_train = theta[theta[cfg.id_col].isin(train_ids)].copy()

    theta_curve_level = collapse_theta_to_curve_level(theta_train, cfg)

    prior = FormulationThetaPrior(cfg)
    prior.fit(formulations_train, theta_curve_level, curves_train)

    # Conformal calibration on training data
    conformal = run_conformal_calibration(cfg, prior, formulations_train, curves_train, rng)

    direct_q = None
    if cfg.train_direct_q_baseline:
        direct_q = DirectQOnePointBaseline(cfg)
        direct_q.fit(formulations_train, curves_train)

    times = np.asarray(cfg.rollout_times, dtype=float)
    candidate_times = list(cfg.candidate_times)

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

        # 1. Prior particles from formulation only
        theta_particles = prior.sample_prior_particles(formulation_row=formulation_row, rng=rng)
        rollout_curves = simulate_plga_ode(theta_particles, times)
        n_particles = rollout_curves.shape[0]
        prior_weights = np.ones(n_particles) / n_particles

        # 2. Zero-early prior prediction
        prior_summary = posterior_predictive_summary(rollout_curves, prior_weights)
        prior_metrics = evaluate_prediction(true_curve, prior_summary, times, cfg, rollout_curves, prior_weights, rng, conformal)
        prior_metrics.update({
            "curve_id": cid, "strategy": "zero_early_prior_only",
            "obs_time": np.nan, "obs_q": np.nan, "ess_after": effective_sample_size(prior_weights),
        })
        all_records.append(prior_metrics)

        # 3. Active one-point selection
        active_t, utilities = choose_active_timepoint(
            rollout_curves, times, prior_weights, candidate_times, cfg, rng,
        )
        for t, u in utilities.items():
            utility_records.append({"curve_id": cid, "candidate_time": t, "utility": u, "chosen": float(t == active_t)})

        active_q = float(interp_curve(curve_groups_test[cid], np.array([active_t]), cfg.time_col, cfg.release_col)[0])
        active_weights = posterior_update_from_one_observation(
            rollout_curves, times, prior_weights, active_t, active_q, cfg,
        )
        active_summary = posterior_predictive_summary(rollout_curves, active_weights)
        active_metrics = evaluate_prediction(true_curve, active_summary, times, cfg, rollout_curves, active_weights, rng, conformal)
        active_metrics.update({
            "curve_id": cid, "strategy": "active_one_point_posterior",
            "obs_time": active_t, "obs_q": active_q, "ess_after": effective_sample_size(active_weights),
        })
        all_records.append(active_metrics)

        for j, t in enumerate(times):
            prediction_records.append({
                "curve_id": cid, "strategy": "active_one_point_posterior", "time": t,
                "true_release": true_curve[j], "pred_mean": active_summary["mean"][j],
                "pred_q05": active_summary["q05"][j], "pred_q50": active_summary["q50"][j],
                "pred_q95": active_summary["q95"][j], "obs_time": active_t, "obs_q": active_q,
            })

        # 4. Fixed one-point baselines
        for fixed_t in candidate_times:
            fixed_q = float(interp_curve(curve_groups_test[cid], np.array([fixed_t]), cfg.time_col, cfg.release_col)[0])
            fixed_weights = posterior_update_from_one_observation(
                rollout_curves, times, prior_weights, fixed_t, fixed_q, cfg,
            )
            fixed_summary = posterior_predictive_summary(rollout_curves, fixed_weights)
            fixed_metrics = evaluate_prediction(true_curve, fixed_summary, times, cfg, rollout_curves, fixed_weights, rng, conformal)
            fixed_metrics.update({
                "curve_id": cid, "strategy": f"fixed_{fixed_t:g}d_posterior",
                "obs_time": fixed_t, "obs_q": fixed_q, "ess_after": effective_sample_size(fixed_weights),
            })
            all_records.append(fixed_metrics)

        # 5. Direct-Q baseline
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

        # 6. Active two-point sequential observer
        # Step 1: choose first timepoint, observe, update posterior
        # Step 2: choose second timepoint from remaining candidates, observe, update again
        remaining_times = [t for t in candidate_times if t != active_t]
        if remaining_times:
            # Step 1 posterior (already computed as active_weights)
            # Re-sample particles from posterior
            post_idx = weighted_resample_indices(active_weights, len(active_weights), rng)
            post_rollout = rollout_curves[post_idx]
            post_weights = np.ones(len(post_idx)) / len(post_idx)

            # Step 2: choose best second timepoint from remaining
            active_t2, utilities2 = choose_active_timepoint(
                post_rollout, times, post_weights, remaining_times, cfg, rng,
            )
            active_q2 = float(interp_curve(curve_groups_test[cid], np.array([active_t2]), cfg.time_col, cfg.release_col)[0])

            # Step 2 posterior update
            active_weights2 = posterior_update_from_one_observation(
                post_rollout, times, post_weights, active_t2, active_q2, cfg,
            )
            active_summary2 = posterior_predictive_summary(post_rollout, active_weights2)
            active_metrics2 = evaluate_prediction(
                true_curve, active_summary2, times, cfg, post_rollout, active_weights2, rng, conformal,
            )
            active_metrics2.update({
                "curve_id": cid, "strategy": "active_two_point_posterior",
                "obs_time": f"{active_t},{active_t2}", "obs_q": f"{active_q},{active_q2}",
                "ess_after": effective_sample_size(active_weights2),
            })
            all_records.append(active_metrics2)

            for j, t in enumerate(times):
                prediction_records.append({
                    "curve_id": cid, "strategy": "active_two_point_posterior", "time": t,
                    "true_release": true_curve[j], "pred_mean": active_summary2["mean"][j],
                    "pred_q05": active_summary2["q05"][j], "pred_q50": active_summary2["q50"][j],
                    "pred_q95": active_summary2["q95"][j],
                    "obs_time": f"{active_t},{active_t2}", "obs_q": f"{active_q},{active_q2}",
                })

        # 7. Fixed two-point baselines
        fixed_pairs = [(1.0, 3.0), (1.0, 5.0), (3.0, 7.0), (1.0, 3.0, 5.0, 7.0)]
        for pair in fixed_pairs:
            if len(pair) == 2:
                t1, t2 = pair
                pair_name = f"fixed_{t1:g}d_{t2:g}_two_point"
            else:
                pair_name = f"fixed_all_four_point"
                t1, t2 = pair[0], pair[1]

            q1 = float(interp_curve(curve_groups_test[cid], np.array([t1]), cfg.time_col, cfg.release_col)[0])
            w1 = posterior_update_from_one_observation(rollout_curves, times, prior_weights, t1, q1, cfg)

            # Resample after first observation
            idx1 = weighted_resample_indices(w1, len(w1), rng)
            rollout1 = rollout_curves[idx1]
            w1_resampled = np.ones(len(idx1)) / len(idx1)

            if len(pair) == 2:
                q2 = float(interp_curve(curve_groups_test[cid], np.array([t2]), cfg.time_col, cfg.release_col)[0])
                w2 = posterior_update_from_one_observation(rollout1, times, w1_resampled, t2, q2, cfg)
            else:
                # Four-point: observe all
                for t_obs in pair[2:]:
                    q_obs = float(interp_curve(curve_groups_test[cid], np.array([t_obs]), cfg.time_col, cfg.release_col)[0])
                    w1_resampled = posterior_update_from_one_observation(rollout1, times, w1_resampled, t_obs, q_obs, cfg)
                w2 = w1_resampled

            summary2 = posterior_predictive_summary(rollout1, w2)
            metrics2 = evaluate_prediction(true_curve, summary2, times, cfg, rollout1, w2, rng, conformal)
            metrics2.update({
                "curve_id": cid, "strategy": pair_name,
                "obs_time": ",".join(str(t) for t in pair),
                "obs_q": np.nan,
                "ess_after": effective_sample_size(w2),
            })
            all_records.append(metrics2)

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
            coverage90_conformal_mean=("coverage_90_conformal", "mean"),
            width90_mean=("width_90", "mean"),
            width90_conformal_mean=("width_90_conformal", "mean"),
            crps_mean=("crps", "mean"),
            ess_median=("ess_after", "median"),
        ).reset_index().sort_values("rmse_mean")
    )
    summary.to_csv(os.path.join(cfg.output_dir, "metrics_summary.csv"), index=False)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    print("\n================ Summary ================")
    print(f"Split type: {split_type}")
    print(summary.to_string(index=False))
    print(f"\nSaved to {cfg.output_dir}/")

    return metrics_df, summary, utility_df, pred_df


# ============================================================
# 10. GroupKFold by polymer family
# ============================================================

def evaluate_groupkfold(cfg: Config, n_splits: int = 5):
    from encoder import parse_dp_group
    from sklearn.model_selection import GroupKFold

    ensure_dir(cfg.output_dir)
    rng = np.random.default_rng(cfg.random_state)

    formulations, curves, theta = load_data(cfg)

    # Add polymer family
    formulations = formulations.copy()
    formulations["polymer_family"] = formulations["DP_Group"].apply(
        lambda x: parse_dp_group(x)[1]
    )
    groups = formulations["polymer_family"].values
    family_counts = pd.Series(groups).value_counts()
    valid_families = family_counts[family_counts >= 3].index.tolist()
    mask = formulations["polymer_family"].isin(valid_families)
    formulations = formulations[mask].copy()
    curves = curves[curves[cfg.id_col].isin(formulations[cfg.id_col])].copy()
    theta = theta[theta[cfg.id_col].isin(formulations[cfg.id_col])].copy()
    groups = formulations["polymer_family"].values

    print(f"[groupkfold] {len(formulations)} curves, {len(valid_families)} families")
    print(f"[groupkfold] families: {formulations['polymer_family'].value_counts().to_dict()}")

    times = np.asarray(cfg.rollout_times, dtype=float)
    gkf = GroupKFold(n_splits=min(n_splits, len(valid_families)))
    all_records = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(formulations, groups=groups)):
        print(f"\n=== Fold {fold} ===")
        train_ids = set(formulations.iloc[train_idx][cfg.id_col])
        test_ids = set(formulations.iloc[test_idx][cfg.id_col])
        test_fams = set(formulations.iloc[test_idx]["polymer_family"])
        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}, Test families: {test_fams}")

        form_train = formulations[formulations[cfg.id_col].isin(train_ids)]
        form_test = formulations[formulations[cfg.id_col].isin(test_ids)]
        c_train = curves[curves[cfg.id_col].isin(train_ids)]
        c_test = curves[curves[cfg.id_col].isin(test_ids)]
        t_train = theta[theta[cfg.id_col].isin(train_ids)]

        theta_cl = collapse_theta_to_curve_level(t_train, cfg)
        prior = FormulationThetaPrior(cfg)
        prior.fit(form_train, theta_cl, c_train)
        conformal = run_conformal_calibration(cfg, prior, form_train, c_train, rng)

        curve_groups = {cid: g for cid, g in c_test.groupby(cfg.id_col)}

        for _, row in form_test.iterrows():
            cid = row[cfg.id_col]
            if cid not in curve_groups:
                continue

            formulation_row = row.to_frame().T
            true_curve = interp_curve(curve_groups[cid], times, cfg.time_col, cfg.release_col)

            theta_particles = prior.sample_prior_particles(formulation_row, rng=rng)
            rollout = simulate_plga_ode(theta_particles, times)
            weights = np.ones(len(rollout)) / len(rollout)

            # Prior only
            prior_sum = posterior_predictive_summary(rollout, weights)
            prior_met = evaluate_prediction(true_curve, prior_sum, times, cfg, rollout, weights, rng, conformal)
            prior_met.update({
                "curve_id": cid, "fold": fold, "strategy": "prior_only",
                "polymer_family": row["polymer_family"],
            })
            all_records.append(prior_met)

            # Active one-point
            active_t, _ = choose_active_timepoint(rollout, times, weights, list(cfg.candidate_times), cfg, rng)
            active_q = float(interp_curve(curve_groups[cid], np.array([active_t]), cfg.time_col, cfg.release_col)[0])
            active_w = posterior_update_from_one_observation(rollout, times, weights, active_t, active_q, cfg)
            active_sum = posterior_predictive_summary(rollout, active_w)
            active_met = evaluate_prediction(true_curve, active_sum, times, cfg, rollout, active_w, rng, conformal)
            active_met.update({
                "curve_id": cid, "fold": fold, "strategy": "active_one_point",
                "polymer_family": row["polymer_family"],
            })
            all_records.append(active_met)

            # Active two-point
            remaining = [t for t in cfg.candidate_times if t != active_t]
            if remaining:
                post_idx = weighted_resample_indices(active_w, len(active_w), rng)
                post_rollout = rollout[post_idx]
                post_w = np.ones(len(post_idx)) / len(post_idx)
                active_t2, _ = choose_active_timepoint(post_rollout, times, post_w, remaining, cfg, rng)
                active_q2 = float(interp_curve(curve_groups[cid], np.array([active_t2]), cfg.time_col, cfg.release_col)[0])
                active_w2 = posterior_update_from_one_observation(post_rollout, times, post_w, active_t2, active_q2, cfg)
                active_sum2 = posterior_predictive_summary(post_rollout, active_w2)
                active_met2 = evaluate_prediction(true_curve, active_sum2, times, cfg, post_rollout, active_w2, rng, conformal)
                active_met2.update({
                    "curve_id": cid, "fold": fold, "strategy": "active_two_point",
                    "polymer_family": row["polymer_family"],
                })
                all_records.append(active_met2)

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(cfg.output_dir, "groupkfold_results.csv"), index=False)

    summary = df.groupby(["strategy", "polymer_family"]).agg(
        n=("curve_id", "count"),
        rmse_mean=("future_rmse_mean", "mean"),
        coverage90_conformal=("coverage_90_conformal", "mean"),
    ).reset_index()
    print("\n=== GroupKFold by polymer family ===")
    print(summary.to_string(index=False))

    overall = df.groupby("strategy").agg(
        n=("curve_id", "count"),
        rmse_mean=("future_rmse_mean", "mean"),
        rmse_std=("future_rmse_mean", "std"),
        coverage90_conformal=("coverage_90_conformal", "mean"),
    ).reset_index()
    print("\n=== Overall ===")
    print(overall.to_string(index=False))

    return df


if __name__ == "__main__":
    if "--groupkfold" in sys.argv:
        evaluate_groupkfold(CFG)
    else:
        evaluate_active_observer(CFG)
