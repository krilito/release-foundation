"""Task E: cross321 LOOCV under 4-way factorial design.

Per docs/executor_brief_2026-05-28_v2_bugfix.md Task E.

4 methods per curve:
  1. DirectQ-fixed:    ExtraTrees on [x, Q(1), Q(3), Q(5), Q(7)]
  2. DirectQ-adaptive: ExtraTrees on [x, Q(t1*), Q(t2*)]
  3. Active-fixed:     particle filter, obs at [1, 3, 5, 7]
  4. Active-adaptive:  particle filter, obs at (t1*, t2*)

Both adaptive variants use the SAME selection algorithm.

Run:
    .\.venv\Scripts\python.exe scripts\77b_regime_benefit_loocv_v2.py
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import kruskal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PLGA_EPS_M = 0.05
_LOWS = np.array([-3.0, -4.0, -2.0, -5.0, -3.0, 0.05, 0.0, -3.0, 0.50])
_HIGHS = np.array([1.0, -1.0, 1.0, 0.0, 2.0, 0.50, 0.30, 0.0, 1.00])
_Q_MAX_IDX = 8
THETA_COLS = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
              "m_crit", "q_burst", "log_tau_burst", "Q_max"]
SEED = 0
N_BOOT = 2000
OUT_DIR = Path("outputs/77_regime_benefit_loocv_v2")


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


def load_cross321():
    xlsx_path = (
        r"D:\chemical-world-model-v0\datset\321PLGA"
        r"\A Dataset on Formulation Parameters and Characteristics of "
        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
    )
    _321_RENAME = {
        "Formulation Index": "Experimental_index",
        "Drug MW": "Drug_Mw", "Drug TPSA": "Drug_TPSA",
        "Drug LogP": "Drug_LogP", "Polymer MW": "Polymer_MW",
        "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
        "Drug Loading Capacity": "DLC",
    }
    theta_df = pd.read_csv("outputs/27a_ode_expressivity_audit/per_curve.csv")
    theta_df = theta_df.rename(columns={"fid": "curve_id"})
    regime_df = pd.read_csv("outputs/33_active_set_regimes/regime_assignments.csv")
    # Filter to cross321 only to avoid fid collision with internal181
    regime_df_cross = regime_df[regime_df["dataset"] == "cross321"]
    regime_map = dict(zip(regime_df_cross["fid"], regime_df_cross["regime"]))

    raw = pd.read_excel(xlsx_path).rename(columns=_321_RENAME)
    raw = raw.rename(columns={"Experimental_index": "curve_id", "Time": "time", "Release": "release"})
    for col in ["Particle Size", "Drug Encapsulation Efficiency", "Solubility Enhancer Concentration"]:
        if col in raw.columns:
            raw = raw.drop(columns=[col])
    raw["Polymer_MW"] = pd.to_numeric(raw["Polymer_MW"], errors="coerce") * 1000
    raw["DLC"] = pd.to_numeric(raw["DLC"], errors="coerce") / 100
    raw["release"] = pd.to_numeric(raw["release"], errors="coerce")
    if raw["release"].max() > 2.0:
        raw["release"] = raw["release"] / 100.0
    raw["release"] = raw["release"].clip(0.0, 1.2)

    feature_cols = ["LA/GA", "Polymer_MW", "CL Ratio", "Drug_Tm", "Drug_Pka",
                    "Initial D/M ratio", "DLC", "SA-V", "SE",
                    "Drug_Mw", "Drug_TPSA", "Drug_NHA", "Drug_LogP"]
    for col in feature_cols:
        if col not in raw.columns:
            raw[col] = np.nan

    form = raw.groupby("curve_id").first().reset_index()
    form = form[["curve_id", *feature_cols]].copy()
    common_ids = sorted(set(form["curve_id"]) & set(theta_df["curve_id"]))
    form = form[form["curve_id"].isin(common_ids)].copy()
    theta_df = theta_df[theta_df["curve_id"].isin(common_ids)].copy()

    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = raw[raw["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    theta_matrix = theta_df.set_index("curve_id").loc[common_ids][THETA_COLS].to_numpy()
    regime_labels = np.array([regime_map.get(cid, -1) for cid in common_ids])

    X_df = form[feature_cols].copy()
    for col in feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    all_nan = X_df.columns[X_df.isna().all()].tolist()
    if all_nan:
        X_df = X_df.drop(columns=all_nan)
    X_df = X_df.fillna(X_df.mean()).fillna(0)

    return X_df, curve_matrix, theta_matrix, regime_labels, common_ids, time_grid


def select_adaptive_obs(particles, weights, rollout, time_grid, future_mask, candidate_times, n_obs, cfg_rng):
    """Variance-reduction utility — same algorithm for all adaptive methods."""
    selected = []
    w = weights.copy()
    for _ in range(n_obs):
        remaining = [t for t in candidate_times if t not in selected]
        if not remaining:
            break
        best_t, best_gain = remaining[0], -1e9
        for ct in remaining:
            ci = int(np.argmin(np.abs(time_grid - ct)))
            qc = rollout[:, ci]
            wt = w / (w.sum() + 1e-12)
            mc = np.sum(wt * qc)
            vc = np.sum(wt * (qc - mc) ** 2) + 1e-10
            qf = rollout[:, future_mask]
            mf = np.sum(wt[:, None] * qf, axis=0)
            vf_ = np.sum(wt[:, None] * (qf - mf[None, :]) ** 2, axis=0) + 1e-10
            ccf = np.sum(wt[:, None] * (qc - mc)[:, None] * (qf - mf[None, :]), axis=0)
            c2 = np.clip(ccf ** 2 / (vc * vf_), 0, 1)
            gain = np.mean(vf_ * c2) / (np.mean(vf_) + 1e-8)
            if gain > best_gain:
                best_gain, best_t = gain, ct
        selected.append(best_t)
    return selected


def update_particle_posterior(particles, rollout, weights, obs_times, obs_values, time_grid, cfg_rng):
    """Update particle weights with observations."""
    w = weights.copy()
    for obs_t, obs_q in zip(obs_times, obs_values):
        oi = int(np.argmin(np.abs(time_grid - obs_t)))
        sig = max(0.03, 0.08 * max(abs(obs_q), 0.05))
        lw = -0.5 * 2 * (rollout[:, oi] - obs_q) ** 2 / sig ** 2
        lw -= lw.max()
        w *= np.exp(lw)
        w /= w.sum() + 1e-12
        # Q_max cap
        qlb = obs_q + 0.05
        cm = particles[:, _Q_MAX_IDX] < qlb
        particles[cm, _Q_MAX_IDX] = qlb
        if cm.any():
            rollout[cm] = sim_batch(particles[cm], time_grid)
    return w


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading cross321 data...")
    X_df, curves, theta_matrix, regime_labels, common_ids, time_grid = load_cross321()
    N = len(common_ids)
    print(f"  {N} curves, {len(time_grid)} time points, {X_df.shape[1]} features")

    future_mask = time_grid > 14.0
    fixed_times = [1.0, 3.0, 5.0, 7.0]
    fixed_idx = [int(np.argmin(np.abs(time_grid - t))) for t in fixed_times]
    candidate_times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0]

    # Results storage
    results = {m: np.zeros(N) for m in ["DQ_fixed", "DQ_adaptive", "Active_fixed", "Active_adaptive"]}
    obs_times_record = {m: [] for m in ["DQ_adaptive", "Active_adaptive"]}

    print(f"\nRunning 4-way LOOCV on {N} curves...")
    for i in range(N):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N} done...")

        train_idx = [j for j in range(N) if j != i]
        X_train = X_df.iloc[train_idx].to_numpy().astype(np.float32)
        X_test = X_df.iloc[i:i+1].to_numpy().astype(np.float32)
        curves_train = curves[train_idx]
        curves_test = curves[i:i+1]
        theta_train = theta_matrix[train_idx]
        y_test_f = curves_test[0, future_mask]

        # Fit preprocessor on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Generate prior particles for this test curve
        et_prior = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                        max_features="sqrt", bootstrap=True,
                                        n_jobs=-1, random_state=SEED)
        et_prior.fit(X_train_s, theta_train)
        knn = NearestNeighbors(n_neighbors=min(50, len(X_train_s)), metric="euclidean").fit(X_train_s)
        resid_std = np.std(theta_train - et_prior.predict(X_train_s), axis=0)

        tree_idx = rng.choice(len(et_prior.estimators_), size=150, replace=False)
        tp = np.asarray([et_prior.estimators_[j].predict(X_test_s)[0] for j in tree_idx])
        _, ni = knn.kneighbors(X_test_s, n_neighbors=min(50, len(X_train_s)))
        kp = theta_train[ni[0]]
        particles = np.vstack([tp, kp])
        particles += rng.normal(0, 0.5 * np.maximum(resid_std, 1e-4), particles.shape)
        particles[:, _Q_MAX_IDX] -= 0.10
        particles = np.clip(particles, _LOWS, _HIGHS)
        rollout = sim_batch(particles, time_grid)
        n_p = len(particles)
        init_weights = np.ones(n_p) / n_p

        # --- Method 1: DirectQ-fixed ---
        X_aug_train = np.hstack([X_train_s, curves_train[:, fixed_idx]])
        X_aug_test = np.hstack([X_test_s, curves_test[0:1, fixed_idx]])
        y_train_f = curves_train[:, future_mask]
        et_dq = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                     max_features="sqrt", bootstrap=True,
                                     n_jobs=-1, random_state=SEED)
        et_dq.fit(X_aug_train, y_train_f)
        y_pred = np.clip(et_dq.predict(X_aug_test)[0], 0.0, 1.1)
        results["DQ_fixed"][i] = np.sqrt(np.mean((y_test_f - y_pred) ** 2))

        # --- Adaptive selection (shared for methods 2 and 4) ---
        adaptive_obs = select_adaptive_obs(
            particles.copy(), init_weights.copy(), rollout.copy(),
            time_grid, future_mask, candidate_times, 2, rng
        )

        # --- Method 2: DirectQ-adaptive ---
        obs_values_adaptive = [float(curves_test[0, int(np.argmin(np.abs(time_grid - t)))]) for t in adaptive_obs]
        X_aug_train_adapt = np.hstack([X_train_s, curves_train[:, [int(np.argmin(np.abs(time_grid - t))) for t in adaptive_obs]]])
        X_aug_test_adapt = np.hstack([X_test_s, np.array(obs_values_adaptive).reshape(1, -1)])
        et_dq_adapt = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                            max_features="sqrt", bootstrap=True,
                                            n_jobs=-1, random_state=SEED)
        et_dq_adapt.fit(X_aug_train_adapt, y_train_f)
        y_pred_adapt = np.clip(et_dq_adapt.predict(X_aug_test_adapt)[0], 0.0, 1.1)
        results["DQ_adaptive"][i] = np.sqrt(np.mean((y_test_f - y_pred_adapt) ** 2))
        obs_times_record["DQ_adaptive"].append(adaptive_obs)

        # --- Method 3: Active-fixed ---
        w_fixed = update_particle_posterior(
            particles.copy(), rollout.copy(), init_weights.copy(),
            fixed_times,
            [float(curves_test[0, idx]) for idx in fixed_idx],
            time_grid, rng
        )
        y_pred_af = np.sum(rollout[:, future_mask] * (w_fixed / (w_fixed.sum() + 1e-12))[:, None], axis=0)
        results["Active_fixed"][i] = np.sqrt(np.mean((y_test_f - y_pred_af) ** 2))

        # --- Method 4: Active-adaptive ---
        w_adaptive = update_particle_posterior(
            particles.copy(), rollout.copy(), init_weights.copy(),
            adaptive_obs, obs_values_adaptive, time_grid, rng
        )
        y_pred_aa = np.sum(rollout[:, future_mask] * (w_adaptive / (w_adaptive.sum() + 1e-12))[:, None], axis=0)
        results["Active_adaptive"][i] = np.sqrt(np.mean((y_test_f - y_pred_aa) ** 2))
        obs_times_record["Active_adaptive"].append(adaptive_obs)

    # Save results
    git_hash = os.popen("git rev-parse HEAD").read().strip()

    per_curve_df = pd.DataFrame({
        "curve_id": [common_ids[i] for i in range(N)],
        "regime": regime_labels,
        "DQ_fixed": results["DQ_fixed"],
        "DQ_adaptive": results["DQ_adaptive"],
        "Active_fixed": results["Active_fixed"],
        "Active_adaptive": results["Active_adaptive"],
    })
    per_curve_df.to_csv(OUT_DIR / "per_curve_results.csv", index=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (4-way LOOCV)")
    print(f"{'='*60}")
    for m in ["DQ_fixed", "DQ_adaptive", "Active_fixed", "Active_adaptive"]:
        r = results[m]
        print(f"  {m:20s}: RMSE={r.mean():.4f} (median={np.median(r):.4f})")

    # Pairwise bootstrap
    pairs = [
        ("DQ_fixed", "DQ_adaptive", "Selection effect, DQ"),
        ("Active_fixed", "Active_adaptive", "Selection effect, Active"),
        ("DQ_fixed", "Active_fixed", "Inference effect, fixed"),
        ("DQ_adaptive", "Active_adaptive", "Inference effect, adaptive"),
        ("DQ_fixed", "Active_adaptive", "Combined (original claim)"),
        ("DQ_adaptive", "Active_fixed", "Disentanglement"),
    ]
    pairwise_rows = []
    print(f"\n{'='*60}")
    print("PAIRWISE COMPARISONS")
    print(f"{'='*60}")
    for a, b, note in pairs:
        da = results[a]
        db = results[b]
        deltas = []
        for _ in range(N_BOOT):
            idx = rng.choice(N, N, replace=True)
            deltas.append(da[idx].mean() - db[idx].mean())
        deltas = np.array(deltas)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        sig = "YES" if (ci_lo > 0 or ci_hi < 0) else "NO"
        pairwise_rows.append({"A": a, "B": b, "delta_rmse": float(np.mean(deltas)),
                              "ci_low": float(ci_lo), "ci_high": float(ci_hi),
                              "significant": sig, "note": note})
        print(f"  {a:20s} vs {b:20s}: Δ={np.mean(deltas):+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig}  ({note})")

    pd.DataFrame(pairwise_rows).to_csv(OUT_DIR / "pairwise_comparisons.csv", index=False)

    # Per-regime analysis
    print(f"\n{'='*60}")
    print("PER-REGIME BENEFIT")
    print(f"{'='*60}")
    benefit = results["DQ_fixed"] - results["Active_adaptive"]
    valid_regimes = sorted(r for r in np.unique(regime_labels) if r > 0 and (regime_labels == r).sum() >= 3)
    regime_groups = []
    for r in valid_regimes:
        mask = regime_labels == r
        b = benefit[mask]
        regime_groups.append(b)
        print(f"  Regime {r} (N={mask.sum():3d}): benefit = {b.mean():+.4f}")

    if len(regime_groups) >= 2:
        H, p_kw = kruskal(*regime_groups)
        k = len(regime_groups)
        eta_sq = (H - k + 1) / (N - k) if N > k else 0
        print(f"\n  Kruskal-Wallis: H={H:.3f}, p={p_kw:.4f}, eta²={eta_sq:.4f}")

        if p_kw < 0.05:
            decision = "Regime × benefit significant; regime-aware method selection viable"
        elif p_kw < 0.15:
            decision = "Underpowered; do not expand further"
        else:
            decision = "No significant regime × benefit interaction"
        print(f"  Decision: {decision}")
    else:
        H, p_kw, eta_sq, decision = 0, 1.0, 0, "Insufficient regimes"

    # Save summary
    summary = {
        "n_curves": N,
        "methods": {m: float(results[m].mean()) for m in results},
        "pairwise": pairwise_rows,
        "kruskal_H": float(H),
        "kruskal_p": float(p_kw),
        "eta_squared": float(eta_sq),
        "decision": decision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    lock_meta = {
        "script": "scripts/77b_regime_benefit_loocv_v2.py",
        "seed": SEED,
        "n_boot": N_BOOT,
        "loocv_definition": "leave-one-curve-out",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
    }
    with open(OUT_DIR / "lock_metadata.json", "w") as f:
        json.dump(lock_meta, f, indent=2)

    print(f"\n  All outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
