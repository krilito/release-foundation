"""Verification: regime = identifiability cluster + regime drives active benefit."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import kruskal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PLGA_EPS_M = 0.05
_LOWS = np.array([-3.0,-4.0,-2.0,-5.0,-3.0,0.05,0.0,-3.0,0.50])
_HIGHS = np.array([1.0,-1.0,1.0,0.0,2.0,0.50,0.30,0.0,1.00])
_Q_MAX_IDX = 8
THETA_COLS = ["log_kw","log_kh","log_alpha","log_kd","log_ke","m_crit","q_burst","log_tau_burst","Q_max"]


def sim_one(theta, t_obs):
    (lk,lh,la,ld,le,mc,qb,lt,qm) = theta
    kw,kh,al,kd,ke = np.exp([lk,lh,la,ld,le])
    tb = float(np.exp(lt))
    def vf(tt, st):
        h,m,Q = st
        dh=kw*(1-h); dm=-kh*h*m*(1+al*(1-m))
        ga=np.clip((mc-m)/_PLGA_EPS_M,-60,60); eg=1/(1+np.exp(-ga))
        fr=max(qm-Q,0); br=(qb/tb)*np.exp(-tt/tb)*fr
        dQ=kd*h*fr+ke*eg*fr+br; return [dh,dm,dQ]
    sol = solve_ivp(vf,(0,float(t_obs[-1])),[0,1,0],t_eval=t_obs,method="DOP853",rtol=1e-6,atol=1e-8)
    if not sol.success or sol.y.shape[1]!=len(t_obs): return np.full_like(t_obs,1e6)
    return np.minimum(np.clip(sol.y[2],0,None),qm)


def sim_batch(thetas, times):
    return np.clip(np.asarray([sim_one(t,times) for t in thetas]),0,1.1)


def perm_pvalue(features, labels, n=2000, metric="euclidean", rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    real = silhouette_score(features, labels, metric=metric)
    null = [silhouette_score(features, rng.permutation(labels), metric=metric) for _ in range(n)]
    return real, np.mean(np.array(null) >= real)


def main():
    # Load data
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    theta_df = pd.read_csv("data/theta_bank.csv")
    regime_df = pd.read_csv("outputs/33_active_set_regimes/regime_assignments.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0: curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    time_grid = np.array([0.25,0.5,1.0,2.0,3.0,5.0,7.0,10.0,14.0,21.0,28.0,42.0,56.0,84.0])
    regime_map = dict(zip(regime_df["fid"], regime_df["regime"]))
    active_set_map = dict(zip(regime_df["fid"], regime_df["active_set_k4"]))
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]) & set(theta_df["curve_id"]) & set(regime_map.keys()))

    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"]==cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i,:] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    theta_matrix = theta_df.set_index("curve_id").loc[common_ids][THETA_COLS].to_numpy()
    regime_labels = np.array([regime_map[cid] for cid in common_ids])

    # Features
    feature_cols = [c for c in formulations.columns if c != "curve_id"]
    X_df = formulations[formulations["curve_id"].isin(common_ids)][feature_cols].copy()
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])
    X = preprocessor.fit_transform(X_df).astype(np.float32)

    # ============================================================
    # Verification 1: Active-set silhouette
    # ============================================================
    print("=" * 60)
    print("VERIFICATION 1: Active-set silhouette")
    print("=" * 60)

    # Convert active_set_k4 to binary vectors
    active_sets = np.zeros((len(common_ids), len(THETA_COLS)))
    for i, cid in enumerate(common_ids):
        as_str = active_set_map[cid]
        if pd.isna(as_str):
            continue
        params = [p.strip() for p in str(as_str).split(",")]
        for p in params:
            if p in THETA_COLS:
                active_sets[i, THETA_COLS.index(p)] = 1.0

    print(f"  Active set shape: {active_sets.shape}")
    print(f"  Mean active params per curve: {active_sets.sum(axis=1).mean():.1f}")
    print()

    # Silhouette on active-set binary vectors (Hamming distance)
    rng_perm = np.random.default_rng(42)
    sil_active, p_active = perm_pvalue(active_sets, regime_labels, metric="hamming", rng=rng_perm)
    print(f"  Active-set silhouette (Hamming): {sil_active:.3f} (p={p_active:.4f})")

    # Also try Jaccard
    from sklearn.metrics import pairwise_distances
    D_jaccard = pairwise_distances(active_sets, metric="jaccard")
    sil_jaccard = silhouette_score(D_jaccard, regime_labels, metric="precomputed")
    print(f"  Active-set silhouette (Jaccard): {sil_jaccard:.3f}")
    print()

    # Compare all spaces
    print("  Comparison across all spaces:")
    theta_std = StandardScaler().fit_transform(theta_matrix)
    for name, feats, metric in [
        ("Formulation x", X, "euclidean"),
        ("Theta (oracle)", theta_std, "euclidean"),
        ("Active set (binary)", active_sets, "hamming"),
    ]:
        sil, p = perm_pvalue(feats, regime_labels, metric=metric, rng=rng_perm)
        print(f"    {name:25s}: sil={sil:+.3f}, p={p:.4f}")

    print()

    # ============================================================
    # Verification 2: Regime ↔ Active Observer benefit
    # ============================================================
    print("=" * 60)
    print("VERIFICATION 2: Regime drives Active Observer benefit")
    print("=" * 60)

    # Canonical split
    idx_all = np.arange(len(common_ids))
    train_cal_idx, test_idx = train_test_split(idx_all, test_size=0.25, random_state=42)
    train_idx, cal_idx = train_test_split(train_cal_idx, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    curves_train, curves_test = curve_matrix[train_idx], curve_matrix[test_idx]
    theta_train = theta_matrix[train_idx]
    regime_test = regime_labels[test_idx]

    future_mask = time_grid > 14.0
    early_times = [1.0, 3.0, 5.0, 7.0]
    early_idx = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    y_test_f = curves_test[:, future_mask]

    # Direct-Q baseline
    X_aug_train = np.hstack([X_train, curves_train[:, early_idx]])
    X_aug_test = np.hstack([X_test, curves_test[:, early_idx]])
    y_train_f = curves_train[:, future_mask]
    et_dq = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42)
    et_dq.fit(X_aug_train, y_train_f)
    y_pred_dq = np.clip(et_dq.predict(X_aug_test), 0.0, 1.1)
    rmse_dq = np.sqrt(np.mean((y_test_f - y_pred_dq)**2, axis=1))

    # Active Observer
    et_prior = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42)
    et_prior.fit(X_train, theta_train)
    knn = NearestNeighbors(n_neighbors=min(50, len(X_train)), metric="euclidean").fit(X_train)
    rng = np.random.default_rng(42)
    resid_std = np.std(theta_train - et_prior.predict(X_train), axis=0)

    rmse_ao = []
    for i in range(len(X_test)):
        x = X_test[i:i+1]
        tree_idx = rng.choice(len(et_prior.estimators_), size=150, replace=False)
        tp = np.asarray([et_prior.estimators_[j].predict(x)[0] for j in tree_idx])
        _, ni = knn.kneighbors(x, n_neighbors=min(50, len(X_train)))
        kp = theta_train[ni[0]]
        particles = np.vstack([tp, kp])
        particles += rng.normal(0, 0.5*np.maximum(resid_std, 1e-4), particles.shape)
        particles[:, _Q_MAX_IDX] -= 0.10
        particles = np.clip(particles, _LOWS, _HIGHS)
        rollout = sim_batch(particles, time_grid)
        n = len(particles); weights = np.ones(n)/n
        for step, cands in enumerate([early_times, [t for t in [7,10,14,21] if t > early_times[-1]]]):
            if not cands: continue
            best_t, best_gain = cands[0], -1e9
            for ct in cands:
                ci = int(np.argmin(np.abs(time_grid-ct)))
                qc = rollout[:,ci]; w = weights/(weights.sum()+1e-12)
                mc = np.sum(w*qc); vc = np.sum(w*(qc-mc)**2)+1e-10
                qf = rollout[:,future_mask]; mf = np.sum(w[:,None]*qf,axis=0)
                vf_ = np.sum(w[:,None]*(qf-mf[None,:])**2,axis=0)+1e-10
                ccf = np.sum(w[:,None]*(qc-mc)[:,None]*(qf-mf[None,:]),axis=0)
                c2 = np.clip(ccf**2/(vc*vf_),0,1)
                gain = np.mean(vf_*c2)/(np.mean(vf_)+1e-8)
                if gain > best_gain: best_gain, best_t = gain, ct
            obs_q = curves_test[i, int(np.argmin(np.abs(time_grid-best_t)))]
            oi = int(np.argmin(np.abs(time_grid-best_t)))
            sig = max(0.03, 0.08*max(abs(obs_q),0.05))
            lw = -0.5*2*(rollout[:,oi]-obs_q)**2/sig**2; lw -= lw.max()
            weights *= np.exp(lw); weights /= weights.sum()+1e-12
            qlb = obs_q+0.05; cm = particles[:,_Q_MAX_IDX]<qlb
            particles[cm,_Q_MAX_IDX]=qlb
            if cm.any(): rollout[cm] = sim_batch(particles[cm], time_grid)
        w = weights/(weights.sum()+1e-12)
        rmse_ao.append(np.sqrt(np.mean((np.sum(rollout[:,future_mask]*w[:,None],axis=0) - y_test_f[i])**2)))
    rmse_ao = np.array(rmse_ao)

    benefit = rmse_dq - rmse_ao  # positive = active is better

    print(f"\n  Per-regime Active Observer benefit (RMSE reduction):")
    regime_benefits = {}
    for r in sorted(np.unique(regime_test)):
        mask = regime_test == r
        if mask.sum() >= 2:
            b = benefit[mask]
            se = b.std() / np.sqrt(mask.sum())
            regime_benefits[r] = b
            print(f"    Regime {r} (N={int(mask.sum()):2d}): benefit = {b.mean():+.4f} +/- {se:.4f}  "
                  f"[{b.mean()-1.96*se:+.4f}, {b.mean()+1.96*se:+.4f}]")

    # Kruskal-Wallis test
    groups = [v for v in regime_benefits.values() if len(v) >= 2]
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        print(f"\n  Kruskal-Wallis: H={H:.3f}, p={p_kw:.4f}")
        if p_kw < 0.01:
            print("  -> STRONG: regime significantly drives Active Observer benefit")
        elif p_kw < 0.05:
            print("  -> MODERATE: regime partially explains Active Observer benefit")
        else:
            print("  -> WEAK: regime does not significantly explain Active Observer benefit")

    # Correlation: active-set dimensionality vs benefit
    active_dim = active_sets[test_idx].sum(axis=1)
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(active_dim, benefit)
    print(f"\n  Spearman(active_dim, benefit): rho={rho:.3f}, p={p_rho:.4f}")
    if p_rho < 0.05:
        print("  -> Active-set dimensionality correlates with Active Observer benefit")


if __name__ == "__main__":
    main()
