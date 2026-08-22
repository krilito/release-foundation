"""Canonical benchmark v2 — BUGFIX version.

Fixes from code review:
  BUG 1: GRU features now trained ONLY on train set (no test leakage)
  BUG 2: Active Observer and Direct-Q use SAME observation inputs
         (both use fixed early times [1,3,5,7d] — fair comparison)
  BUG 3: StandardScaler fit only on train set
  BUG 4: Bootstrap CI included in script

Run:
    .\.venv\Scripts\python.exe scripts\72b_canonical_benchmark_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PLGA_EPS_M = 0.05
_LOWS = np.array([-3.0, -4.0, -2.0, -5.0, -3.0, 0.05, 0.0, -3.0, 0.50])
_HIGHS = np.array([1.0, -1.0, 1.0, 0.0, 2.0, 0.50, 0.30, 0.0, 1.00])
_Q_MAX_IDX = 8


def sim_one(theta, t_obs):
    (lk, lh, la, ld, le, mc, qb, lt, qm) = theta
    kw, kh, al, kd, ke = np.exp([lk, lh, la, ld, le])
    tb = float(np.exp(lt))

    def vf(tt, st):
        h, m, Q = st
        dh = kw * (1 - h)
        dm = -kh * h * m * (1 + al * (1 - m))
        ga = np.clip((mc - m) / _PLGA_EPS_M, -60, 60)
        eg = 1 / (1 + np.exp(-ga))
        fr = max(qm - Q, 0)
        br = (qb / tb) * np.exp(-tt / tb) * fr
        dQ = kd * h * fr + ke * eg * fr + br
        return [dh, dm, dQ]

    sol = solve_ivp(vf, (0, float(t_obs[-1])), [0, 1, 0],
                    t_eval=t_obs, method="DOP853", rtol=1e-6, atol=1e-8)
    if not sol.success or sol.y.shape[1] != len(t_obs):
        return np.full_like(t_obs, 1e6)
    return np.minimum(np.clip(sol.y[2], 0, None), qm)


def sim_batch(thetas, times):
    return np.clip(np.asarray([sim_one(t, times) for t in thetas]), 0, 1.1)


def per_curve_r2(y_true, y_pred):
    r2s = []
    for i in range(len(y_true)):
        ss_r = np.sum((y_true[i] - y_pred[i]) ** 2)
        ss_t = np.sum((y_true[i] - y_true[i].mean()) ** 2) + 1e-10
        r2s.append(1.0 - ss_r / ss_t)
    return np.array(r2s)


def evaluate(y_true, y_pred, label):
    r2s = per_curve_r2(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_pooled = 1 - ss_res / ss_tot
    print(f"  {label:40s} R²_med={np.median(r2s):.3f}  R²_pooled={r2_pooled:.3f}  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}  R²>0={np.mean(r2s>0):.1%}")
    return {"r2_median": np.median(r2s), "r2_pooled": r2_pooled,
            "rmse": rmse, "mae": mae, "frac_positive": np.mean(r2s > 0)}


def main():
    print("Loading data...")
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    theta_df = pd.read_csv("data/theta_bank.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]) & set(theta_df["curve_id"]))
    formulations = formulations[formulations["curve_id"].isin(common_ids)].copy()
    theta_df = theta_df[theta_df["curve_id"].isin(common_ids)].copy()

    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    theta_matrix = theta_df.set_index("curve_id").loc[common_ids][theta_cols].to_numpy()

    # Features — fit preprocessor ONLY on train set (BUG 3 fix)
    feature_cols = [c for c in formulations.columns if c != "curve_id"]
    X_df = formulations[feature_cols].copy()
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    # Canonical split
    idx_all = np.arange(len(common_ids))
    train_cal_idx, test_idx = train_test_split(idx_all, test_size=0.25, random_state=42)
    train_idx, cal_idx = train_test_split(train_cal_idx, test_size=0.2, random_state=42)
    print(f"  Train={len(train_idx)}, Cal={len(cal_idx)}, Test={len(test_idx)}")

    # BUG 3 FIX: fit preprocessor ONLY on train set
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
    ])
    X_train = preprocessor.fit_transform(X_df.iloc[train_idx]).astype(np.float32)
    X_test = preprocessor.transform(X_df.iloc[test_idx]).astype(np.float32)
    X_all = preprocessor.transform(X_df).astype(np.float32)  # for GRU training

    curves_train, curves_test = curve_matrix[train_idx], curve_matrix[test_idx]
    theta_train = theta_matrix[train_idx]

    early_times = [1.0, 3.0, 5.0, 7.0]
    future_start = 14.0
    future_mask = time_grid > future_start
    early_idx = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    y_test_f = curves_test[:, future_mask]

    print(f"\n  Early observations: {early_times}")
    print(f"  Future evaluation: t > {future_start}")
    print(f"  Future time points: {time_grid[future_mask]}")
    print()

    print("=" * 70)
    print("CANONICAL BENCHMARK v2 (BUGFIX)")
    print("=" * 70)

    # ============================================================
    # BUG 2 FIX: Both methods use SAME fixed early observations
    # ============================================================

    # Method 1: Direct-Q with fixed early observations
    X_aug_train = np.hstack([X_train, curves_train[:, early_idx]])
    X_aug_test = np.hstack([X_test, curves_test[:, early_idx]])
    y_train_f = curves_train[:, future_mask]

    et_dq = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                 max_features="sqrt", bootstrap=True,
                                 n_jobs=-1, random_state=42)
    et_dq.fit(X_aug_train, y_train_f)
    y_pred_dq = np.clip(et_dq.predict(X_aug_test), 0.0, 1.1)
    results_dq = evaluate(y_test_f, y_pred_dq, "Direct-Q (ExtraTrees)")

    # Method 2: Active Observer with FIXED early observations (same as Direct-Q)
    # BUG 2 FIX: NOT adaptive selection — fair comparison
    et_prior = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                    max_features="sqrt", bootstrap=True,
                                    n_jobs=-1, random_state=42)
    et_prior.fit(X_train, theta_train)
    knn = NearestNeighbors(n_neighbors=min(50, len(X_train)), metric="euclidean").fit(X_train)
    rng = np.random.default_rng(42)
    resid_std = np.std(theta_train - et_prior.predict(X_train), axis=0)

    y_pred_ao = []
    for i in range(len(X_test)):
        x = X_test[i:i+1]
        tree_idx = rng.choice(len(et_prior.estimators_), size=150, replace=False)
        tp = np.asarray([et_prior.estimators_[j].predict(x)[0] for j in tree_idx])
        _, ni = knn.kneighbors(x, n_neighbors=min(50, len(X_train)))
        kp = theta_train[ni[0]]
        particles = np.vstack([tp, kp])
        particles += rng.normal(0, 0.5 * np.maximum(resid_std, 1e-4), particles.shape)
        particles[:, _Q_MAX_IDX] -= 0.10
        particles = np.clip(particles, _LOWS, _HIGHS)
        rollout = sim_batch(particles, time_grid)
        n_p = len(particles)
        weights = np.ones(n_p) / n_p

        # FIXED observations at [1d, 3d, 5d, 7d] — same as Direct-Q
        for obs_t in early_times:
            obs_idx = int(np.argmin(np.abs(time_grid - obs_t)))
            obs_q = curves_test[i, obs_idx]
            sig = max(0.03, 0.08 * max(abs(obs_q), 0.05))
            lw = -0.5 * 2 * (rollout[:, obs_idx] - obs_q) ** 2 / sig ** 2
            lw -= lw.max()
            weights *= np.exp(lw)
            weights /= weights.sum() + 1e-12
            # Q_max cap
            qlb = obs_q + 0.05
            cm = particles[:, _Q_MAX_IDX] < qlb
            particles[cm, _Q_MAX_IDX] = qlb
            if cm.any():
                rollout[cm] = sim_batch(particles[cm], time_grid)

        w = weights / (weights.sum() + 1e-12)
        y_pred_ao.append(np.sum(rollout[:, future_mask] * w[:, None], axis=0))

    y_pred_ao = np.array(y_pred_ao)
    results_ao = evaluate(y_test_f, y_pred_ao, "Active Observer (fixed 4pt)")

    # Method 3: Direct-Q + GRU h-features (BUG 1 FIX: train GRU on train only)
    print("\n  Training GRU on TRAIN set only (BUG 1 fix)...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_71", "scripts/71_release_world_model.py")
    _71 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_71)

    # BUG 1 FIX: train GRU on train set only
    model_gru = _71.ReleaseWorldModel(n_features=X_train.shape[1], latent_dim=8, hidden_dim=64, deter_dim=64)
    optimizer = torch.optim.Adam(model_gru.parameters(), lr=3e-4, weight_decay=1e-5)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    curves_train_t = torch.tensor(curves_train, dtype=torch.float32)
    times_t = torch.tensor(time_grid, dtype=torch.float32)
    mask_t = torch.ones(len(X_train), len(time_grid), dtype=torch.bool)

    for epoch in range(300):
        model_gru.train()
        perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), 32):
            idx = perm[start:start+32]
            loss = model_gru.training_loss(X_train_t[idx], curves_train_t[idx], times_t, mask_t[idx],
                                            kl_weight=0.01, mono_weight=0.5)
            optimizer.zero_grad()
            loss["total"].backward()
            nn.utils.clip_grad_norm_(model_gru.parameters(), 1.0)
            optimizer.step()

    # Extract h(t=14d) for train and test SEPARATELY
    model_gru.eval()
    early_indices = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    early_times_arr = time_grid[early_indices]

    def extract_h(X_data, curves_data):
        h_features = []
        with torch.no_grad():
            for i in range(len(X_data)):
                x = torch.tensor(X_data[i:i+1], dtype=torch.float32)
                eq = torch.tensor(curves_data[i:i+1, early_indices], dtype=torch.float32)
                h, z = model_gru.rssm.init_state(1, "cpu")
                for j, t_obs in enumerate(early_times_arr):
                    dt_val = t_obs - (early_times_arr[j-1] if j > 0 else 0.0)
                    dt = torch.full((1, 1), dt_val)
                    h = model_gru.rssm.dynamics_gru(torch.cat([z, x, dt], dim=-1), h)
                    q_mu, q_ls = model_gru.rssm.posterior(h, eq[:, j:j+1], dt)
                    z = model_gru.rssm.sample(q_mu, q_ls)
                h_features.append(h.numpy()[0])
        return np.array(h_features)

    h_train = extract_h(X_train, curves_train)
    h_test = extract_h(X_test, curves_test)

    X_aug_train_h = np.hstack([X_train, curves_train[:, early_idx], h_train])
    X_aug_test_h = np.hstack([X_test, curves_test[:, early_idx], h_test])
    et_dq_h = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                   max_features="sqrt", bootstrap=True,
                                   n_jobs=-1, random_state=42)
    et_dq_h.fit(X_aug_train_h, y_train_f)
    y_pred_dq_h = np.clip(et_dq_h.predict(X_aug_test_h), 0.0, 1.1)
    results_dq_h = evaluate(y_test_f, y_pred_dq_h, "Direct-Q + h(t=14d) (train-only GRU)")

    # ============================================================
    # Bootstrap CI
    # ============================================================
    print("\n--- Bootstrap CI (2000 resamples) ---")
    rmse_dq_per = np.sqrt(np.mean((y_test_f - y_pred_dq) ** 2, axis=1))
    rmse_ao_per = np.sqrt(np.mean((y_test_f - y_pred_ao) ** 2, axis=1))
    n_boot = 2000
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(len(rmse_dq_per), len(rmse_dq_per), replace=True)
        deltas.append(np.mean(rmse_dq_per[idx]) - np.mean(rmse_ao_per[idx]))
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    print(f"  Δ RMSE (DQ - AO) = {np.mean(deltas):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Relative: {np.mean(deltas)/np.mean(rmse_dq_per)*100:.1f}% [{ci_lo/np.mean(rmse_dq_per)*100:.1f}%, {ci_hi/np.mean(rmse_dq_per)*100:.1f}%]")
    print(f"  Significant: {'YES' if ci_lo > 0 else 'NO'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (BUGFIX VERSION)")
    print("=" * 70)
    print(f"\n  {'Method':45s} {'RMSE':>8s} {'R²_pooled':>10s}")
    print("  " + "-" * 65)
    for name, r in [("Direct-Q (ExtraTrees)", results_dq),
                     ("Active Observer (fixed 4pt)", results_ao),
                     ("Direct-Q + h(t=14d) (train-only GRU)", results_dq_h)]:
        print(f"  {name:45s} {r['rmse']:8.4f} {r['r2_pooled']:10.3f}")

    print("\n  NOTE: Active Observer uses SAME fixed 4 observations as Direct-Q.")
    print("  This is a FAIR method comparison (BUG 2 fix).")


if __name__ == "__main__":
    main()
