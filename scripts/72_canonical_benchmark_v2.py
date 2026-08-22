"""
Canonical benchmark v2 — 4-way factorial design.

Per docs/executor_brief_2026-05-28_v2_bugfix.md Task D.

Fixes from v1 (scripts/72_canonical_benchmark.py):
  - BUG 2 fix: 4-way factorial. Both adaptive variants use the SAME
    selection algorithm (Active Observer particle variance-reduction).
    Only the final predictor differs (ExtraTrees vs particle posterior).
  - StandardScaler fit on TRAIN only, transform TRAIN and TEST separately
    (related leakage fix).
  - Bootstrap CI computed in the same script (paired, at curve level,
    2000 resamples, seed=0).
  - lock_metadata.json output per paper_requirements_locked.md A2.
  - Pooled R^2 per S2 (not per-curve median).
  - All randomness seeded with --seed (default 0).
  - A3 fix: canonical split now comes from `data/canonical_split_v1.csv`
    instead of being silently re-derived in-script.

Methods (all evaluated on the SAME fixed test fold from
`data/canonical_split_v1.csv`, same future timepoints):
  1. DirectQ-fixed:    ExtraTrees on [x, Q(1), Q(3), Q(5), Q(7)]
  2. DirectQ-adaptive: ExtraTrees on [x, Q(t1*), Q(t2*)] per-curve
                       where (t1*, t2*) selected by particle filter
  3. Active-fixed:     particle filter posterior, observations at
                       [1, 3, 5, 7] (4 observations, no adaptive selection)
  4. Active-adaptive:  particle filter posterior, observations at
                       (t1*, t2*) selected by particle filter

Comparison axes:
  - Predictor:   DirectQ (ExtraTrees) vs Active (particle posterior)
  - Observation: 4 fixed early vs 2 adaptively-selected

Pairwise comparisons (6 total) decompose:
  - Selection effect with each predictor
  - Inference effect with each observation strategy

Run:
    .\\.venv\\Scripts\\python.exe scripts\\72_canonical_benchmark_v2.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.integrate import solve_ivp


# ============================================================
# Constants — same as v1 for ODE comparability
# ============================================================
_PLGA_EPS_M = 0.05
_LOWS = np.array([-3.0, -4.0, -2.0, -5.0, -3.0, 0.05, 0.0, -3.0, 0.50])
_HIGHS = np.array([1.0, -1.0, 1.0, 0.0, 2.0, 0.50, 0.30, 0.0, 1.00])
_Q_MAX_IDX = 8

OUTPUT_DIR = Path("outputs/72_canonical_benchmark_v2")
CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")


# ============================================================
# Data — same loader as v1
# ============================================================

def load_canonical_split():
    split_df = pd.read_csv(CANONICAL_SPLIT_PATH)
    required_cols = {"curve_id", "split_name", "fold_index"}
    missing = required_cols.difference(split_df.columns)
    if missing:
        raise ValueError(f"Canonical split file missing columns: {sorted(missing)}")

    split_df = split_df.copy()
    split_df["curve_id"] = split_df["curve_id"].astype(int)
    split_df["split_name"] = split_df["split_name"].astype(str)
    split_df["fold_index"] = split_df["fold_index"].astype(int)

    if split_df["curve_id"].duplicated().any():
        dupes = split_df.loc[split_df["curve_id"].duplicated(), "curve_id"].tolist()
        raise ValueError(f"Duplicate curve_ids in canonical split: {dupes[:10]}")
    if split_df["fold_index"].duplicated().any():
        dupes = split_df.loc[split_df["fold_index"].duplicated(), "fold_index"].tolist()
        raise ValueError(f"Duplicate fold_index values in canonical split: {dupes[:10]}")

    allowed = {"train", "cal", "test"}
    bad = sorted(set(split_df["split_name"]) - allowed)
    if bad:
        raise ValueError(f"Unexpected split_name values in canonical split: {bad}")

    return split_df.sort_values("fold_index").reset_index(drop=True)


def load_data(split_df=None):
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    theta = pd.read_csv("data/theta_bank.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])

    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]) & set(theta["curve_id"]))
    if split_df is not None:
        split_ids = split_df["curve_id"].tolist()
        missing = sorted(set(split_ids) - set(common_ids))
        if missing:
            raise ValueError(
                "Canonical split references curve_ids not present in formulations/curves/theta: "
                f"{missing[:10]}"
            )
        common_ids = split_ids

    formulations = (
        formulations[formulations["curve_id"].isin(common_ids)]
        .set_index("curve_id")
        .loc[common_ids]
        .reset_index()
    )
    theta = (
        theta[theta["curve_id"].isin(common_ids)]
        .set_index("curve_id")
        .loc[common_ids]
        .reset_index()
    )

    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    theta_matrix = theta.set_index("curve_id").loc[common_ids][theta_cols].to_numpy()

    return formulations, curve_matrix, time_grid, theta_matrix, common_ids


def load_regime_labels(common_ids):
    """Return per-curve regime label (NaN if unknown). Aligns to common_ids order."""
    path = Path("outputs/33_active_set_regimes/regime_assignments.csv")
    if not path.exists():
        return np.full(len(common_ids), np.nan)
    df = pd.read_csv(path)
    df_internal = df[df["dataset"] == "internal181"].copy()
    # internal181 fid corresponds to curve_id 1..181 (1-indexed)
    fid_to_regime = dict(zip(df_internal["fid"], df_internal["regime"]))
    regimes = np.array([fid_to_regime.get(int(cid), np.nan) for cid in common_ids])
    return regimes


def prepare_features_split(formulations, train_idx, test_idx, cal_idx=None):
    """Fit preprocessor on TRAIN only. Transform train/test (and cal) separately.

    This fixes the leak in v1 where StandardScaler was fit on all 181 curves
    before splitting.
    """
    feature_cols = [c for c in formulations.columns if c != "curve_id"]
    X_all = formulations[feature_cols].copy()
    numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_all.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         categorical_cols),
    ], remainder="drop")

    # Fit on training rows only
    X_train_raw = X_all.iloc[train_idx]
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw).astype(np.float32)
    X_test = preprocessor.transform(X_all.iloc[test_idx]).astype(np.float32)
    if cal_idx is not None:
        X_cal = preprocessor.transform(X_all.iloc[cal_idx]).astype(np.float32)
        return X_train, X_cal, X_test, feature_cols, preprocessor
    return X_train, X_test, feature_cols, preprocessor


# ============================================================
# PLGA ODE — same as v1
# ============================================================

def _simulate_one(theta, t_obs):
    (log_kw, log_kh, log_alpha, log_kd, log_ke,
     m_crit, q_burst, log_tau_burst, Q_max) = theta
    kw, kh, alpha, kd, ke = np.exp([log_kw, log_kh, log_alpha, log_kd, log_ke])
    tau_burst = float(np.exp(log_tau_burst))

    def vf(tt, state):
        h, m, Q = state
        dh = kw * (1.0 - h)
        dm = -kh * h * m * (1.0 + alpha * (1.0 - m))
        gate_arg = np.clip((m_crit - m) / _PLGA_EPS_M, -60.0, 60.0)
        erosion_gate = 1.0 / (1.0 + np.exp(-gate_arg))
        free = max(Q_max - Q, 0.0)
        burst_rate = (q_burst / tau_burst) * np.exp(-tt / tau_burst) * free
        dQ = kd * h * free + ke * erosion_gate * free + burst_rate
        return [dh, dm, dQ]

    sol = solve_ivp(vf, (0.0, float(t_obs[-1])), [0.0, 1.0, 0.0],
                    t_eval=t_obs, method="DOP853", rtol=1e-6, atol=1e-8)
    if not sol.success or sol.y.shape[1] != len(t_obs):
        return np.full_like(t_obs, 1e6, dtype=float)
    return np.minimum(np.clip(sol.y[2], 0.0, None), Q_max)


def simulate_batch(theta_matrix, times):
    return np.clip(np.asarray([_simulate_one(t, times) for t in theta_matrix]), 0.0, 1.1)


# ============================================================
# Metrics
# ============================================================

def pooled_rmse(y_true, y_pred):
    """RMSE pooled over all (curve, time) pairs."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pooled_r2(y_true, y_pred):
    """R^2 pooled over all (curve, time) pairs (S2 in requirements doc)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_curve_rmse(y_true, y_pred):
    """Per-curve RMSE; used for paired bootstrap at curve level."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def bootstrap_paired_ci(rmse_per_curve_a, rmse_per_curve_b, n_resamples=2000, seed=0):
    """Paired bootstrap CI for (RMSE_a - RMSE_b).

    Resample at the curve level (S1). Returns (delta_mean, ci_low, ci_high).
    Negative delta means method A is better (lower RMSE).
    """
    rng = np.random.default_rng(seed)
    n = len(rmse_per_curve_a)
    assert n == len(rmse_per_curve_b), "Methods must be evaluated on same curves"

    # Pool curves' (curve, time) squared errors and recompute pooled RMSE per resample
    # But we only have per-curve RMSE here, so resample those
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        # Pooled RMSE on resampled curves: combine per-curve MSEs
        mse_a = (rmse_per_curve_a[idx] ** 2).mean()
        mse_b = (rmse_per_curve_b[idx] ** 2).mean()
        deltas.append(np.sqrt(mse_a) - np.sqrt(mse_b))
    deltas = np.array(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ============================================================
# Particle prior — shared across all Active methods and adaptive selection
# ============================================================

def build_particle_prior(x_test, X_train, theta_train, et_model, knn_model, rng,
                          n_tree=150, n_knn=50):
    """Generate prior particles for a single test curve. Returns particles array."""
    n_tree = min(n_tree, len(et_model.estimators_))
    n_knn = min(n_knn, len(X_train))
    tree_idx = rng.choice(len(et_model.estimators_), size=n_tree, replace=False)
    tree_particles = np.asarray([et_model.estimators_[j].predict(x_test)[0] for j in tree_idx])

    _, nn_idx = knn_model.kneighbors(x_test, n_neighbors=n_knn)
    knn_particles = theta_train[nn_idx[0]]

    particles = np.vstack([tree_particles, knn_particles])

    # Jitter (same recipe as v1)
    et_pred = et_model.predict(X_train)
    resid_std = np.std(theta_train - et_pred, axis=0)
    particles += rng.normal(0, 0.5 * np.maximum(resid_std, 1e-4), particles.shape)
    particles[:, _Q_MAX_IDX] -= 0.10  # Q_max correction (v1 line 204)
    particles = np.clip(particles, _LOWS, _HIGHS)
    return particles


def variance_reduction_score(rollout, weights, future_mask, candidate_idx):
    """Score for choosing a candidate observation time (greedy variance reduction).
    Higher = more informative. From v1 active observer lines 217-231.
    """
    w = weights / (weights.sum() + 1e-12)
    q_c = rollout[:, candidate_idx]
    mu_c = np.sum(w * q_c)
    var_c = np.sum(w * (q_c - mu_c) ** 2) + 1e-10
    q_f = rollout[:, future_mask]
    mu_f = np.sum(w[:, None] * q_f, axis=0)
    var_f = np.sum(w[:, None] * (q_f - mu_f[None, :]) ** 2, axis=0) + 1e-10
    cov_cf = np.sum(w[:, None] * (q_c - mu_c)[:, None] * (q_f - mu_f[None, :]), axis=0)
    corr2 = np.clip(cov_cf ** 2 / (var_c * var_f), 0.0, 1.0)
    return float(np.mean(var_f * corr2) / (np.mean(var_f) + 1e-8))


def apply_observation(particles, rollout, weights, obs_q, idx_obs, time_grid):
    """Update particle weights given an observation. Also caps Q_max from below.
    Returns updated (particles, rollout, weights).
    """
    sigma = max(0.03, 0.08 * max(abs(obs_q), 0.05))
    log_w = -0.5 * 2.0 * (rollout[:, idx_obs] - obs_q) ** 2 / sigma ** 2
    log_w -= log_w.max()
    like_w = np.exp(log_w)
    weights = weights * like_w
    weights /= weights.sum() + 1e-12

    q_max_lb = obs_q + 0.05
    cap_mask = particles[:, _Q_MAX_IDX] < q_max_lb
    if cap_mask.any():
        particles = particles.copy()
        particles[cap_mask, _Q_MAX_IDX] = q_max_lb
        rollout = rollout.copy()
        rollout[cap_mask] = simulate_batch(particles[cap_mask], time_grid)
    return particles, rollout, weights


# ============================================================
# Particle filter inference (shared between Active-fixed and Active-adaptive,
# and adaptive selection used by DirectQ-adaptive)
# ============================================================

def particle_inference(x_test, curves_test_i, X_train, theta_train,
                       et_model, knn_model, time_grid, future_mask,
                       observation_times, rng):
    """Run particle filter posterior inference for one test curve at given
    observation times. Returns (selected_times, weights, particles, rollout)
    so that callers can use the posterior predictive directly OR extract the
    times used for downstream feature construction.

    observation_times: list of pre-specified times to observe at, OR
                       None to use adaptive selection (2 steps).
    """
    particles = build_particle_prior(x_test, X_train, theta_train, et_model,
                                       knn_model, rng)
    rollout = simulate_batch(particles, time_grid)
    weights = np.ones(len(particles)) / len(particles)

    if observation_times is not None:
        # FIXED observation schedule
        selected_times = list(observation_times)
        for ct in selected_times:
            idx_obs = int(np.argmin(np.abs(time_grid - ct)))
            obs_q = float(curves_test_i[idx_obs])
            particles, rollout, weights = apply_observation(
                particles, rollout, weights, obs_q, idx_obs, time_grid)
    else:
        # ADAPTIVE: 2-step greedy variance reduction (matches v1 active observer)
        early_candidates = [1.0, 3.0, 5.0, 7.0]
        late_candidates = [10.0, 14.0]
        selected_times = []

        for step_candidates in [early_candidates, late_candidates]:
            best_t, best_gain = step_candidates[0], -1e9
            for ct in step_candidates:
                idx_c = int(np.argmin(np.abs(time_grid - ct)))
                gain = variance_reduction_score(rollout, weights, future_mask, idx_c)
                if gain > best_gain:
                    best_gain, best_t = gain, ct

            selected_times.append(best_t)
            idx_obs = int(np.argmin(np.abs(time_grid - best_t)))
            obs_q = float(curves_test_i[idx_obs])
            particles, rollout, weights = apply_observation(
                particles, rollout, weights, obs_q, idx_obs, time_grid)

    return selected_times, weights, particles, rollout


# ============================================================
# Method 1: DirectQ-fixed
# ============================================================

def run_directq_fixed(X_train, curves_train, X_test, curves_test,
                       time_grid, obs_times, future_mask, seed):
    obs_idx = [int(np.argmin(np.abs(time_grid - t))) for t in obs_times]
    X_aug_train = np.hstack([X_train, curves_train[:, obs_idx]])
    X_aug_test = np.hstack([X_test, curves_test[:, obs_idx]])
    y_train = curves_train[:, future_mask]
    y_test = curves_test[:, future_mask]

    model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                 max_features="sqrt", bootstrap=True,
                                 n_jobs=-1, random_state=seed)
    model.fit(X_aug_train, y_train)
    y_pred = np.clip(model.predict(X_aug_test), 0.0, 1.1)

    obs_times_used = [list(obs_times) for _ in range(len(X_test))]
    return y_pred, y_test, obs_times_used


# ============================================================
# Method 2: DirectQ-adaptive (per-curve ExtraTrees, shared selection)
# ============================================================

def run_directq_adaptive(X_train, curves_train, X_test, curves_test,
                           theta_train, time_grid, future_mask,
                           selected_times_per_curve, seed):
    """For each test curve, train an ExtraTrees with features
    [x, Q(t1), Q(t2)] using the adaptively-selected times for that curve.

    selected_times_per_curve: list of length n_test, each item is [t1, t2]
                              chosen by the particle filter for that curve.
    """
    y_test = curves_test[:, future_mask]
    n_test = len(X_test)
    y_preds = np.zeros_like(y_test)

    for i in range(n_test):
        t_sel = selected_times_per_curve[i]
        idx_sel = [int(np.argmin(np.abs(time_grid - t))) for t in t_sel]
        X_aug_train_i = np.hstack([X_train, curves_train[:, idx_sel]])
        X_aug_test_i = np.hstack([X_test[i:i+1], curves_test[i:i+1, idx_sel]])

        model_i = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                       max_features="sqrt", bootstrap=True,
                                       n_jobs=-1, random_state=seed)
        model_i.fit(X_aug_train_i, curves_train[:, future_mask])
        y_preds[i] = np.clip(model_i.predict(X_aug_test_i)[0], 0.0, 1.1)

    return y_preds, y_test


# ============================================================
# Method 3 & 4: Active-fixed and Active-adaptive
# ============================================================

def run_active(X_train, curves_train, X_test, curves_test, theta_train,
                time_grid, future_mask, observation_times, seed):
    """observation_times: list of fixed times to observe at, OR None for
    2-step adaptive selection.
    """
    et = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                              max_features="sqrt", bootstrap=True,
                              n_jobs=-1, random_state=seed)
    et.fit(X_train, theta_train)

    knn = NearestNeighbors(n_neighbors=min(50, len(X_train)), metric="euclidean")
    knn.fit(X_train)

    rng = np.random.default_rng(seed)
    y_test = curves_test[:, future_mask]
    n_test = len(X_test)
    y_preds = np.zeros_like(y_test)
    obs_times_used = []

    for i in range(n_test):
        x = X_test[i:i+1]
        selected_times, weights, _, rollout = particle_inference(
            x, curves_test[i], X_train, theta_train, et, knn,
            time_grid, future_mask, observation_times, rng)
        w = weights / (weights.sum() + 1e-12)
        y_preds[i] = np.sum(rollout[:, future_mask] * w[:, None], axis=0)
        obs_times_used.append(selected_times)

    return y_preds, y_test, obs_times_used


# ============================================================
# Adaptive selection helper (run ONCE, share across DirectQ-adaptive and Active-adaptive)
# ============================================================

def compute_adaptive_selections(X_train, curves_train, X_test, curves_test,
                                  theta_train, time_grid, future_mask, seed):
    """Run particle-filter selection per test curve. Returns:
       - selected_times_per_curve: list of [t1, t2] per test curve
       - active_adaptive_preds: predictions from particle posterior (Method 4)
       - active_adaptive_obs_times: same as selected_times_per_curve

    The selected_times_per_curve is consumed by DirectQ-adaptive (Method 2)
    so the two methods see the SAME observations per test curve.
    """
    et = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                              max_features="sqrt", bootstrap=True,
                              n_jobs=-1, random_state=seed)
    et.fit(X_train, theta_train)

    knn = NearestNeighbors(n_neighbors=min(50, len(X_train)), metric="euclidean")
    knn.fit(X_train)

    rng = np.random.default_rng(seed)
    y_test = curves_test[:, future_mask]
    n_test = len(X_test)
    y_preds = np.zeros_like(y_test)
    selected_times_per_curve = []

    for i in range(n_test):
        x = X_test[i:i+1]
        selected_times, weights, _, rollout = particle_inference(
            x, curves_test[i], X_train, theta_train, et, knn,
            time_grid, future_mask, None, rng)  # None = adaptive
        w = weights / (weights.sum() + 1e-12)
        y_preds[i] = np.sum(rollout[:, future_mask] * w[:, None], axis=0)
        selected_times_per_curve.append(selected_times)

    return selected_times_per_curve, y_preds, y_test


# ============================================================
# Main
# ============================================================

def get_git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 70)
    print("Canonical Benchmark v2 — 4-way factorial")
    print("=" * 70)
    print(f"  Seed: {seed}")
    print(f"  Bootstrap resamples: {args.n_bootstrap}")
    print()

    # ---- Load data ----
    split_df = load_canonical_split()
    formulations, curves, time_grid, theta_matrix, common_ids = load_data(split_df=split_df)
    regime_labels = load_regime_labels(common_ids)
    n_curves = len(common_ids)
    print(f"  Loaded {n_curves} curves, {len(time_grid)} timepoints")

    # ---- Canonical split (A3 fixed file) ----
    split_names = split_df["split_name"].to_numpy()
    train_idx = np.flatnonzero(split_names == "train")
    cal_idx = np.flatnonzero(split_names == "cal")
    test_idx = np.flatnonzero(split_names == "test")
    print(f"  Train={len(train_idx)}, Cal={len(cal_idx)}, Test={len(test_idx)}")

    # ---- Preprocess (fit on TRAIN only — fixes related leak) ----
    X_train, X_cal, X_test, feature_cols, preprocessor = prepare_features_split(
        formulations, train_idx, test_idx, cal_idx)
    curves_train, curves_test = curves[train_idx], curves[test_idx]
    theta_train, theta_test = theta_matrix[train_idx], theta_matrix[test_idx]
    regimes_test = regime_labels[test_idx]

    # ---- Evaluation window ----
    early_times = [1.0, 3.0, 5.0, 7.0]
    future_start = 14.0
    future_mask = time_grid > future_start
    print(f"  Future eval times: {time_grid[future_mask]}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Run all 4 methods
    # ============================================================
    results = {}  # method -> dict with y_pred, y_test, obs_times, per_curve_rmse

    # --- Method 1: DirectQ-fixed ---
    print("[1/4] DirectQ-fixed (4 obs at [1,3,5,7]) ...")
    t0 = time.time()
    y_pred, y_test, obs_used = run_directq_fixed(
        X_train, curves_train, X_test, curves_test,
        time_grid, early_times, future_mask, seed)
    pc_rmse = per_curve_rmse(y_test, y_pred)
    results["DirectQ-fixed"] = dict(y_pred=y_pred, y_test=y_test,
                                       obs_times=obs_used, per_curve_rmse=pc_rmse,
                                       rmse=pooled_rmse(y_test, y_pred),
                                       r2=pooled_r2(y_test, y_pred))
    print(f"       RMSE={results['DirectQ-fixed']['rmse']:.4f}, R²={results['DirectQ-fixed']['r2']:.3f}  ({time.time()-t0:.1f}s)")

    # --- Method 4 (run first to get selected_times for Method 2): Active-adaptive ---
    print("[2/4] Active-adaptive (particle, 2 adaptive obs) ...")
    t0 = time.time()
    selected_times, y_pred, y_test = compute_adaptive_selections(
        X_train, curves_train, X_test, curves_test, theta_train,
        time_grid, future_mask, seed)
    pc_rmse = per_curve_rmse(y_test, y_pred)
    results["Active-adaptive"] = dict(y_pred=y_pred, y_test=y_test,
                                          obs_times=selected_times, per_curve_rmse=pc_rmse,
                                          rmse=pooled_rmse(y_test, y_pred),
                                          r2=pooled_r2(y_test, y_pred))
    print(f"       RMSE={results['Active-adaptive']['rmse']:.4f}, R²={results['Active-adaptive']['r2']:.3f}  ({time.time()-t0:.1f}s)")

    # --- Method 2: DirectQ-adaptive (uses same selected_times) ---
    print("[3/4] DirectQ-adaptive (per-curve ExtraTrees, shared 2 adaptive obs) ...")
    t0 = time.time()
    y_pred, y_test = run_directq_adaptive(
        X_train, curves_train, X_test, curves_test, theta_train,
        time_grid, future_mask, selected_times, seed)
    pc_rmse = per_curve_rmse(y_test, y_pred)
    results["DirectQ-adaptive"] = dict(y_pred=y_pred, y_test=y_test,
                                           obs_times=selected_times, per_curve_rmse=pc_rmse,
                                           rmse=pooled_rmse(y_test, y_pred),
                                           r2=pooled_r2(y_test, y_pred))
    print(f"       RMSE={results['DirectQ-adaptive']['rmse']:.4f}, R²={results['DirectQ-adaptive']['r2']:.3f}  ({time.time()-t0:.1f}s)")

    # --- Method 3: Active-fixed (particle with 4 forced obs) ---
    print("[4/4] Active-fixed (particle, 4 obs at [1,3,5,7]) ...")
    t0 = time.time()
    y_pred, y_test, obs_used = run_active(
        X_train, curves_train, X_test, curves_test, theta_train,
        time_grid, future_mask, early_times, seed)
    pc_rmse = per_curve_rmse(y_test, y_pred)
    results["Active-fixed"] = dict(y_pred=y_pred, y_test=y_test,
                                       obs_times=obs_used, per_curve_rmse=pc_rmse,
                                       rmse=pooled_rmse(y_test, y_pred),
                                       r2=pooled_r2(y_test, y_pred))
    print(f"       RMSE={results['Active-fixed']['rmse']:.4f}, R²={results['Active-fixed']['r2']:.3f}  ({time.time()-t0:.1f}s)")

    # ============================================================
    # Summary table
    # ============================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    method_order = ["DirectQ-fixed", "DirectQ-adaptive", "Active-fixed", "Active-adaptive"]
    print(f"{'Method':<22} {'RMSE':>8} {'R²':>8} {'n_obs':>7}")
    print("-" * 50)
    n_obs_per_method = {"DirectQ-fixed": 4, "Active-fixed": 4,
                          "DirectQ-adaptive": 2, "Active-adaptive": 2}
    for m in method_order:
        print(f"{m:<22} {results[m]['rmse']:>8.4f} {results[m]['r2']:>8.3f} "
              f"{n_obs_per_method[m]:>7d}")

    # ============================================================
    # Pairwise bootstrap CI
    # ============================================================
    print()
    print("=" * 70)
    print("PAIRWISE COMPARISONS (paired bootstrap, 95% CI; negative Δ = A wins)")
    print("=" * 70)
    pairs = [
        ("DirectQ-fixed", "DirectQ-adaptive", "Selection effect, DirectQ predictor"),
        ("Active-fixed", "Active-adaptive", "Selection effect, Active predictor"),
        ("DirectQ-fixed", "Active-fixed", "Inference effect, fixed obs"),
        ("DirectQ-adaptive", "Active-adaptive", "Inference effect, adaptive obs"),
        ("DirectQ-fixed", "Active-adaptive", "Combined (original v1 +27% claim)"),
        ("DirectQ-adaptive", "Active-fixed", "Disentanglement check"),
    ]
    pairwise_rows = []
    print(f"{'A':<20} {'B':<20} {'ΔRMSE':>8} {'CI low':>8} {'CI high':>8} {'sig':>5}  {'note'}")
    print("-" * 110)
    for a, b, note in pairs:
        d_mean, d_lo, d_hi = bootstrap_paired_ci(
            results[a]["per_curve_rmse"], results[b]["per_curve_rmse"],
            n_resamples=args.n_bootstrap, seed=seed)
        sig = "✓" if (d_lo > 0 or d_hi < 0) else "—"
        rel = (d_mean / results[a]["rmse"]) * 100
        print(f"{a:<20} {b:<20} {d_mean:>+8.4f} {d_lo:>+8.4f} {d_hi:>+8.4f} {sig:>5}  {note}")
        pairwise_rows.append(dict(
            method_a=a, method_b=b,
            delta_rmse=d_mean, ci_low=d_lo, ci_high=d_hi,
            significant_95=bool(d_lo > 0 or d_hi < 0),
            relative_pct=rel,
            note=note))

    # ============================================================
    # Pre-decided decision rule (per executor brief Task D)
    # ============================================================
    print()
    print("=" * 70)
    print("PRE-DECIDED DECISION RULE OUTCOME")
    print("=" * 70)
    aa = results["Active-adaptive"]["rmse"]
    df = results["DirectQ-fixed"]["rmse"]
    delta = (df - aa) / df * 100  # positive = Active-adaptive wins
    # CI on this delta (paired bootstrap of DirectQ-fixed minus Active-adaptive)
    d_mean, d_lo, d_hi = bootstrap_paired_ci(
        results["DirectQ-fixed"]["per_curve_rmse"],
        results["Active-adaptive"]["per_curve_rmse"],
        n_resamples=args.n_bootstrap, seed=seed)
    ci_low_pct = d_lo / df * 100
    ci_high_pct = d_hi / df * 100

    if d_lo > 0 and delta > 5:
        decision = "Active-adaptive wins by >5% (CI > 0): combined active + Bayesian inference effective"
    elif d_lo > 0 and 2 <= delta <= 5:
        decision = "Active-adaptive wins by 2-5% (CI > 0): honest but moderate claim"
    elif d_lo <= 0 <= d_hi:
        decision = "CI crosses 0: Active Observer death possible; reconsider main contribution"
    elif d_hi < 0:
        decision = "DirectQ-adaptive or DirectQ-fixed beats Active-adaptive: Active Observer is genuinely worse, major finding"
    else:
        decision = f"Active-adaptive wins by {delta:.1f}% but unclear category"

    print(f"  Active-adaptive vs DirectQ-fixed:")
    print(f"    ΔRMSE = {d_mean:+.4f}  (CI [{d_lo:+.4f}, {d_hi:+.4f}])")
    print(f"    Relative: {delta:+.1f}%  (CI [{ci_low_pct:+.1f}%, {ci_high_pct:+.1f}%])")
    print(f"    Decision: {decision}")

    # ============================================================
    # Write outputs
    # ============================================================
    # summary.csv
    summary_rows = []
    for m in method_order:
        summary_rows.append(dict(
            method=m,
            n_test=len(test_idx),
            n_obs=n_obs_per_method[m],
            rmse=results[m]["rmse"],
            r2_pooled=results[m]["r2"],
            per_curve_rmse_mean=float(results[m]["per_curve_rmse"].mean()),
            per_curve_rmse_median=float(np.median(results[m]["per_curve_rmse"])),
        ))
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # pairwise_comparisons.csv
    pd.DataFrame(pairwise_rows).to_csv(OUTPUT_DIR / "pairwise_comparisons.csv", index=False)

    # per_curve_results.csv (one row per (curve, method))
    per_curve_rows = []
    for m in method_order:
        for i, ti in enumerate(test_idx):
            obs_t = results[m]["obs_times"][i] if i < len(results[m]["obs_times"]) else []
            per_curve_rows.append(dict(
                curve_id=int(common_ids[ti]),
                regime=float(regimes_test[i]) if not np.isnan(regimes_test[i]) else None,
                method=m,
                rmse=float(results[m]["per_curve_rmse"][i]),
                obs_times_used=",".join(f"{t:g}" for t in obs_t),
            ))
    pd.DataFrame(per_curve_rows).to_csv(OUTPUT_DIR / "per_curve_results.csv", index=False)

    # lock_metadata.json
    lock = dict(
        script="scripts/72_canonical_benchmark_v2.py",
        generated_at_utc=pd.Timestamp.utcnow().isoformat(),
        git_hash=get_git_hash(),
        seed=seed,
        n_bootstrap=args.n_bootstrap,
        n_curves_total=int(n_curves),
        n_train=int(len(train_idx)),
        n_cal=int(len(cal_idx)),
        n_test=int(len(test_idx)),
        split_source=str(CANONICAL_SPLIT_PATH),
        train_idx=train_idx.tolist(),
        cal_idx=cal_idx.tolist(),
        test_idx=test_idx.tolist(),
        early_times_fixed=early_times,
        future_start=future_start,
        eval_times=time_grid[future_mask].tolist(),
        methods=method_order,
        pre_decided_decision=decision,
        delta_rmse_active_vs_directq_fixed=dict(
            delta=d_mean, ci_low=d_lo, ci_high=d_hi,
            relative_pct=delta, relative_ci_pct=[ci_low_pct, ci_high_pct]),
        note="v1 BUG 1 (GRU leak) not in scope here — h-features omitted in v2. "
             "v1 BUG 2 fixed via 4-way factorial. Preprocessor fit on train only.",
    )
    with open(OUTPUT_DIR / "lock_metadata.json", "w") as f:
        json.dump(lock, f, indent=2, default=float)

    print()
    print(f"Outputs in {OUTPUT_DIR}/")
    print(f"  summary.csv, pairwise_comparisons.csv, per_curve_results.csv, lock_metadata.json")


if __name__ == "__main__":
    main()
