"""73b - Diagnostics on the fixed canonical split file.

This is the safe A3-follow-up companion to `72_canonical_benchmark_v2.py`.
It preserves the spirit of `scripts/73_diagnostics.py` but avoids touching
that user-modified file. The key difference is that the train/cal/test split
comes from `data/canonical_split_v1.csv` instead of being re-derived in code.

Diagnostics:
  1. Bootstrap CI on the RMSE gap between Direct-Q and Active Observer
  2. Pooled R² over all curve-time pairs
  3. Regime PCA / silhouette on GRU h(t=14d)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PLGA_EPS_M = 0.05
_LOWS = np.array([-3.0, -4.0, -2.0, -5.0, -3.0, 0.05, 0.0, -3.0, 0.50])
_HIGHS = np.array([1.0, -1.0, 1.0, 0.0, 2.0, 0.50, 0.30, 0.0, 1.00])
_Q_MAX_IDX = 8
CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")


def sim_one(theta, t_obs):
    (lk, lh, la, ld, le, mc, qb, lt, qm) = theta
    kw, kh, al, kd, ke = np.exp([lk, lh, la, ld, le])
    tb = float(np.exp(lt))

    def vf(tt, st):
        h, m, q = st
        dh = kw * (1 - h)
        dm = -kh * h * m * (1 + al * (1 - m))
        ga = np.clip((mc - m) / _PLGA_EPS_M, -60, 60)
        eg = 1 / (1 + np.exp(-ga))
        fr = max(qm - q, 0)
        br = (qb / tb) * np.exp(-tt / tb) * fr
        dq = kd * h * fr + ke * eg * fr + br
        return [dh, dm, dq]

    sol = solve_ivp(
        vf,
        (0, float(t_obs[-1])),
        [0, 1, 0],
        t_eval=t_obs,
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success or sol.y.shape[1] != len(t_obs):
        return np.full_like(t_obs, 1e6)
    return np.minimum(np.clip(sol.y[2], 0, None), qm)


def sim_batch(thetas, times):
    return np.clip(np.asarray([sim_one(t, times) for t in thetas]), 0, 1.1)


def load_canonical_split() -> pd.DataFrame:
    split_df = pd.read_csv(CANONICAL_SPLIT_PATH)
    required = {"curve_id", "split_name", "fold_index"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Canonical split missing columns: {sorted(missing)}")

    split_df = split_df.copy()
    split_df["curve_id"] = split_df["curve_id"].astype(int)
    split_df["split_name"] = split_df["split_name"].astype(str)
    split_df["fold_index"] = split_df["fold_index"].astype(int)

    if split_df["curve_id"].duplicated().any():
        raise ValueError("Duplicate curve_id values in canonical split")
    if split_df["fold_index"].duplicated().any():
        raise ValueError("Duplicate fold_index values in canonical split")

    allowed = {"train", "cal", "test"}
    bad = sorted(set(split_df["split_name"]) - allowed)
    if bad:
        raise ValueError(f"Unexpected split names in canonical split: {bad}")

    return split_df.sort_values("fold_index").reset_index(drop=True)


def main(seed: int):
    # === Load data ===
    split_df = load_canonical_split()
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    theta_df = pd.read_csv("data/theta_bank.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]) & set(theta_df["curve_id"]))
    split_ids = split_df["curve_id"].tolist()
    missing = sorted(set(split_ids) - set(common_ids))
    if missing:
        raise ValueError(f"Canonical split contains curve_ids missing from tables: {missing[:10]}")

    common_ids = split_ids
    formulations = (
        formulations[formulations["curve_id"].isin(common_ids)]
        .set_index("curve_id")
        .loc[common_ids]
        .reset_index()
    )
    theta_df = (
        theta_df[theta_df["curve_id"].isin(common_ids)]
        .set_index("curve_id")
        .loc[common_ids]
        .reset_index()
    )

    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke", "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    theta_matrix = theta_df.set_index("curve_id").loc[common_ids][theta_cols].to_numpy()

    feature_cols = [c for c in formulations.columns if c != "curve_id"]
    x_df = formulations[feature_cols].copy()
    num_cols = x_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in x_df.columns if c not in num_cols]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])
    x = preprocessor.fit_transform(x_df).astype(np.float32)

    split_names = split_df["split_name"].to_numpy()
    train_idx = np.flatnonzero(split_names == "train")
    cal_idx = np.flatnonzero(split_names == "cal")
    test_idx = np.flatnonzero(split_names == "test")
    print(f"Canonical split source: {CANONICAL_SPLIT_PATH}")
    print(f"Train={len(train_idx)}, Cal={len(cal_idx)}, Test={len(test_idx)}")

    x_train, x_test = x[train_idx], x[test_idx]
    curves_train, curves_test = curve_matrix[train_idx], curve_matrix[test_idx]
    theta_train = theta_matrix[train_idx]

    future_mask = time_grid > 14.0
    early_times = [1.0, 3.0, 5.0, 7.0]
    early_idx = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    y_test_f = curves_test[:, future_mask]

    # === Direct-Q ===
    x_aug_train = np.hstack([x_train, curves_train[:, early_idx]])
    x_aug_test = np.hstack([x_test, curves_test[:, early_idx]])
    y_train_f = curves_train[:, future_mask]
    et_dq = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )
    et_dq.fit(x_aug_train, y_train_f)
    y_pred_dq = np.clip(et_dq.predict(x_aug_test), 0.0, 1.1)
    rmse_dq_per = np.sqrt(np.mean((y_test_f - y_pred_dq) ** 2, axis=1))

    # === Active Observer ===
    et_prior = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )
    et_prior.fit(x_train, theta_train)
    knn = NearestNeighbors(n_neighbors=min(50, len(x_train)), metric="euclidean").fit(x_train)
    rng = np.random.default_rng(seed)
    resid_std = np.std(theta_train - et_prior.predict(x_train), axis=0)

    y_pred_ao = []
    for i in range(len(x_test)):
        x_row = x_test[i:i + 1]
        tree_idx = rng.choice(len(et_prior.estimators_), size=150, replace=False)
        tp = np.asarray([et_prior.estimators_[j].predict(x_row)[0] for j in tree_idx])
        _, ni = knn.kneighbors(x_row, n_neighbors=min(50, len(x_train)))
        kp = theta_train[ni[0]]
        particles = np.vstack([tp, kp])
        particles += rng.normal(0, 0.5 * np.maximum(resid_std, 1e-4), particles.shape)
        particles[:, _Q_MAX_IDX] -= 0.10
        particles = np.clip(particles, _LOWS, _HIGHS)
        rollout = sim_batch(particles, time_grid)
        n = len(particles)
        weights = np.ones(n) / n

        for cands in [early_times, [t for t in [7, 10, 14, 21] if t > early_times[-1]]]:
            if not cands:
                continue
            best_t, best_gain = cands[0], -1e9
            for ct in cands:
                ci = int(np.argmin(np.abs(time_grid - ct)))
                qc = rollout[:, ci]
                w = weights / (weights.sum() + 1e-12)
                mc = np.sum(w * qc)
                vc = np.sum(w * (qc - mc) ** 2) + 1e-10
                qf = rollout[:, future_mask]
                mf = np.sum(w[:, None] * qf, axis=0)
                vf_ = np.sum(w[:, None] * (qf - mf[None, :]) ** 2, axis=0) + 1e-10
                ccf = np.sum(w[:, None] * (qc - mc)[:, None] * (qf - mf[None, :]), axis=0)
                c2 = np.clip(ccf ** 2 / (vc * vf_), 0, 1)
                gain = np.mean(vf_ * c2) / (np.mean(vf_) + 1e-8)
                if gain > best_gain:
                    best_gain, best_t = gain, ct

            obs_q = curves_test[i, int(np.argmin(np.abs(time_grid - best_t)))]
            oi = int(np.argmin(np.abs(time_grid - best_t)))
            sig = max(0.03, 0.08 * max(abs(obs_q), 0.05))
            lw = -0.5 * 2 * (rollout[:, oi] - obs_q) ** 2 / sig ** 2
            lw -= lw.max()
            weights *= np.exp(lw)
            weights /= weights.sum() + 1e-12
            qlb = obs_q + 0.05
            cm = particles[:, _Q_MAX_IDX] < qlb
            particles[cm, _Q_MAX_IDX] = qlb
            if cm.any():
                rollout[cm] = sim_batch(particles[cm], time_grid)

        w = weights / (weights.sum() + 1e-12)
        y_pred_ao.append(np.sum(rollout[:, future_mask] * w[:, None], axis=0))

    y_pred_ao = np.array(y_pred_ao)
    rmse_ao_per = np.sqrt(np.mean((y_test_f - y_pred_ao) ** 2, axis=1))

    # === (i) Bootstrap CI ===
    print("=== (i) Bootstrap CI on RMSE gap ===")
    n_boot = 5000
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(len(rmse_dq_per), len(rmse_dq_per), replace=True)
        deltas.append(np.mean(rmse_dq_per[idx]) - np.mean(rmse_ao_per[idx]))
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    mean_delta = np.mean(deltas)
    print(f"  Delta RMSE (Direct-Q - Active) = {mean_delta:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Significant: {'YES' if ci_lo > 0 else 'NO'}")
    print(
        "  Relative improvement: "
        f"{mean_delta / np.mean(rmse_dq_per) * 100:.1f}% "
        f"[{ci_lo / np.mean(rmse_dq_per) * 100:.1f}%, {ci_hi / np.mean(rmse_dq_per) * 100:.1f}%]"
    )
    print()

    # === (ii) Pooled R2 ===
    print("=== (ii) Pooled R2 (all curve-time pairs) ===")
    ss_res_dq = np.sum((y_test_f - y_pred_dq) ** 2)
    ss_res_ao = np.sum((y_test_f - y_pred_ao) ** 2)
    ss_tot = np.sum((y_test_f - y_test_f.mean()) ** 2)
    r2_dq = 1 - ss_res_dq / ss_tot
    r2_ao = 1 - ss_res_ao / ss_tot
    print(f"  Direct-Q pooled R2:        {r2_dq:.4f}")
    print(f"  Active Observer pooled R2: {r2_ao:.4f}")
    print()

    # === (iii) Regime PCA ===
    print("=== (iii) Regime PCA on GRU h(t=14d) ===")
    regime_df = pd.read_csv("outputs/33_active_set_regimes/regime_assignments.csv")
    regime_map = dict(zip(regime_df["fid"], regime_df["regime"]))
    regime_labels = np.array([regime_map.get(cid, -1) for cid in common_ids])
    valid = regime_labels > 0
    print(f"  Curves with regime labels: {valid.sum()} / {len(common_ids)}")

    spec = importlib.util.spec_from_file_location("_71", "scripts/71_release_world_model.py")
    _71 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_71)

    model_gru = _71.ReleaseWorldModel(n_features=x.shape[1], latent_dim=8, hidden_dim=64, deter_dim=64)
    optimizer = torch.optim.Adam(model_gru.parameters(), lr=3e-4, weight_decay=1e-5)
    x_t = torch.tensor(x, dtype=torch.float32)
    curves_t = torch.tensor(curve_matrix, dtype=torch.float32)
    times_t = torch.tensor(time_grid, dtype=torch.float32)
    mask_t = torch.ones(len(x), len(time_grid), dtype=torch.bool)

    for _epoch in range(300):
        model_gru.train()
        perm = torch.randperm(len(x))
        for start in range(0, len(x), 32):
            idx = perm[start:start + 32]
            loss = model_gru.training_loss(x_t[idx], curves_t[idx], times_t, mask_t[idx], kl_weight=0.01, mono_weight=0.5)
            optimizer.zero_grad()
            loss["total"].backward()
            nn.utils.clip_grad_norm_(model_gru.parameters(), 1.0)
            optimizer.step()

    h_14d = []
    early_indices = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    early_times_arr = time_grid[early_indices]

    model_gru.eval()
    with torch.no_grad():
        for i in range(len(x)):
            x_row = torch.tensor(x[i:i + 1], dtype=torch.float32)
            eq = torch.tensor(curve_matrix[i:i + 1, early_indices], dtype=torch.float32)
            h, z = model_gru.rssm.init_state(1, "cpu")
            for j, t_obs in enumerate(early_times_arr):
                dt_val = t_obs - (early_times_arr[j - 1] if j > 0 else 0.0)
                dt = torch.full((1, 1), dt_val)
                h = model_gru.rssm.dynamics_gru(torch.cat([z, x_row, dt], dim=-1), h)
                q_mu, q_ls = model_gru.rssm.posterior(h, eq[:, j:j + 1], dt)
                z = model_gru.rssm.sample(q_mu, q_ls)
            h_14d.append(h.numpy()[0])

    h_14d = np.array(h_14d)
    h_valid = h_14d[valid]
    x_valid = x[valid]
    reg_valid = regime_labels[valid]

    def perm_test(features, labels, n=1000):
        real = silhouette_score(features, labels)
        null = [silhouette_score(features, rng.permutation(labels)) for _ in range(n)]
        p = np.mean(np.array(null) >= real)
        return real, p

    real_h, p_h = perm_test(h_valid, reg_valid)
    real_x, p_x = perm_test(x_valid, reg_valid)

    print(f"  GRU h(t=14d) silhouette: {real_h:.3f} (p={p_h:.4f})")
    print(f"  x alone silhouette:      {real_x:.3f} (p={p_x:.4f})")
    print(f"  Delta silhouette (h-x):  {real_h - real_x:.3f}")
    print()

    pca = PCA(n_components=2)
    pcs_h = pca.fit_transform(h_valid)
    print("  Per-regime PCA centers (GRU h):")
    for r in sorted(np.unique(reg_valid)):
        m = reg_valid == r
        print(f"    Regime {r} (n={m.sum()}): PC1={pcs_h[m, 0].mean():.3f}, PC2={pcs_h[m, 1].mean():.3f}")

    print()
    if real_h > real_x + 0.10 and p_h < 0.01:
        print("  VERDICT: GRU automatically rediscovers regime structure (STRONG)")
    elif real_h > real_x + 0.03:
        print("  VERDICT: Weak regime signal in GRU h (SUPPLEMENTARY)")
    elif abs(real_h - real_x) <= 0.03:
        print("  VERDICT: Regime primarily encoded in x, GRU adds little (NO CLAIM)")
    else:
        print("  VERDICT: GRU blurs regime boundaries (REGIME IS FORMULATION-LEVEL)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed; split comes from data/canonical_split_v1.csv",
    )
    main(parser.parse_args().seed)
