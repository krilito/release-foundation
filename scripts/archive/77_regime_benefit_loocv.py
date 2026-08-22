"""Task 4: cross321 LOOCV regime benefit analysis.

Re-test E4 (regime × Active Observer benefit) with larger N (~250 curves).
Leave-one-curve-out CV on the cross-DOI 321 dataset.
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
OUT_DIR = Path("outputs/77_regime_benefit_loocv")


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
    """Load cross-DOI 321 data with formulation features, curves, theta, regimes."""
    xlsx_path = (
        r"D:\chemical-world-model-v0\datset\321PLGA"
        r"\A Dataset on Formulation Parameters and Characteristics of "
        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
    )

    # Column rename from 321 xlsx to internal schema (matches script 10)
    _321_RENAME = {
        "Formulation Index": "Experimental_index",
        "Drug MW": "Drug_Mw",
        "Drug TPSA": "Drug_TPSA",
        "Drug LogP": "Drug_LogP",
        "Polymer MW": "Polymer_MW",
        "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
        "Drug Loading Capacity": "DLC",
    }

    # Load oracle theta
    theta_df = pd.read_csv("outputs/27a_ode_expressivity_audit/per_curve.csv")
    theta_df = theta_df.rename(columns={"fid": "curve_id"})

    # Load regime labels
    regime_df = pd.read_csv("outputs/33_active_set_regimes/regime_assignments.csv")
    regime_map = dict(zip(regime_df["fid"], regime_df["regime"]))

    # Load cross-DOI data from xlsx
    raw = pd.read_excel(xlsx_path)
    raw = raw.rename(columns=_321_RENAME)
    raw = raw.rename(columns={"Experimental_index": "curve_id", "Time": "time", "Release": "release"})

    # Drop unused columns
    for col in ["Particle Size", "Drug Encapsulation Efficiency", "Solubility Enhancer Concentration"]:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    # Unit conversions (matches script 10)
    raw["Polymer_MW"] = pd.to_numeric(raw["Polymer_MW"], errors="coerce") * 1000  # kDa -> Da
    raw["DLC"] = pd.to_numeric(raw["DLC"], errors="coerce") / 100  # % -> fraction

    # Normalize release
    raw["release"] = pd.to_numeric(raw["release"], errors="coerce")
    if raw["release"].max() > 2.0:
        raw["release"] = raw["release"] / 100.0
    raw["release"] = raw["release"].clip(0.0, 1.2)

    # Feature columns
    feature_cols = [
        "LA/GA", "Polymer_MW", "CL Ratio", "Drug_Tm", "Drug_Pka",
        "Initial D/M ratio", "DLC", "SA-V", "SE",
        "Drug_Mw", "Drug_TPSA", "Drug_NHA", "Drug_LogP",
    ]

    # Add missing columns with NaN (will be imputed)
    for col in feature_cols:
        if col not in raw.columns:
            raw[col] = np.nan

    # Get unique formulations
    form = raw.groupby("curve_id").first().reset_index()
    form = form[["curve_id", *feature_cols]].copy()

    # Common IDs with theta
    common_ids = sorted(set(form["curve_id"]) & set(theta_df["curve_id"]))
    print(f"  Cross321 curves with theta: {len(common_ids)}")

    # Filter
    form = form[form["curve_id"].isin(common_ids)].copy()
    theta_df = theta_df[theta_df["curve_id"].isin(common_ids)].copy()

    # Build curve matrix
    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = raw[raw["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    # Theta matrix
    theta_matrix = theta_df.set_index("curve_id").loc[common_ids][THETA_COLS].to_numpy()

    # Regime labels
    regime_labels = np.array([regime_map.get(cid, -1) for cid in common_ids])

    # Features — impute missing with column mean, drop all-NaN columns
    X_df = form[feature_cols].copy()
    for col in feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    # Drop columns that are entirely NaN
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    if all_nan_cols:
        print(f"  Dropping all-NaN columns: {all_nan_cols}")
        X_df = X_df.drop(columns=all_nan_cols)
    X_df = X_df.fillna(X_df.mean())
    # Final safety: fill any remaining NaN with 0
    X_df = X_df.fillna(0)

    return X_df, curve_matrix, theta_matrix, regime_labels, common_ids, time_grid, list(X_df.columns)


def run_loocv():
    """Run leave-one-curve-out CV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading cross321 data...")
    X_df, curves, theta_matrix, regime_labels, common_ids, time_grid, feature_cols = load_cross321()
    N = len(common_ids)
    print(f"  {N} curves, {len(time_grid)} time points, {len(feature_cols)} features")
    print(f"  Regime distribution: {dict(zip(*np.unique(regime_labels[regime_labels > 0], return_counts=True)))}")

    future_mask = time_grid > 14.0
    early_times = [1.0, 3.0, 5.0, 7.0]
    early_idx = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]

    # Preprocessing
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
    ])
    X = preprocessor.fit_transform(X_df).astype(np.float32)

    benefits = []
    regimes_out = []
    rmse_dq_all = []
    rmse_ao_all = []

    print(f"\nRunning LOOCV on {N} curves...")
    for i in range(N):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N} done...")

        # Leave one out
        train_idx = [j for j in range(N) if j != i]
        test_idx = [i]

        X_train, X_test = X[train_idx], X[test_idx]
        curves_train, curves_test = curves[train_idx], curves[test_idx]
        theta_train = theta_matrix[train_idx]
        y_test_f = curves_test[0, future_mask]

        # Direct-Q
        X_aug_train = np.hstack([X_train, curves_train[:, early_idx]])
        X_aug_test = np.hstack([X_test, curves_test[:, early_idx]])
        y_train_f = curves_train[:, future_mask]
        et_dq = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                     max_features="sqrt", bootstrap=True,
                                     n_jobs=-1, random_state=SEED)
        et_dq.fit(X_aug_train, y_train_f)
        y_pred_dq = np.clip(et_dq.predict(X_aug_test)[0], 0.0, 1.1)
        rmse_dq = np.sqrt(np.mean((y_test_f - y_pred_dq) ** 2))

        # Active Observer
        et_prior = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                        max_features="sqrt", bootstrap=True,
                                        n_jobs=-1, random_state=SEED)
        et_prior.fit(X_train, theta_train)
        knn = NearestNeighbors(n_neighbors=min(50, len(X_train)), metric="euclidean").fit(X_train)
        resid_std = np.std(theta_train - et_prior.predict(X_train), axis=0)

        x = X_test[0:1]
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

        for step, cands in enumerate([early_times, [t for t in [7, 10, 14, 21] if t > early_times[-1]]]):
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

            obs_q = curves_test[0, int(np.argmin(np.abs(time_grid - best_t)))]
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
        y_pred_ao = np.sum(rollout[:, future_mask] * w[:, None], axis=0)[0]
        rmse_ao = np.sqrt(np.mean((y_test_f - y_pred_ao) ** 2))

        benefits.append(rmse_dq - rmse_ao)
        regimes_out.append(regime_labels[i])
        rmse_dq_all.append(rmse_dq)
        rmse_ao_all.append(rmse_ao)

    benefits = np.array(benefits)
    regimes_out = np.array(regimes_out)
    rmse_dq_all = np.array(rmse_dq_all)
    rmse_ao_all = np.array(rmse_ao_all)

    # Analysis
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    valid_regimes = sorted(r for r in np.unique(regimes_out) if r > 0 and (regimes_out == r).sum() >= 3)
    print(f"\n  Regimes with N >= 3: {valid_regimes}")

    per_regime = []
    print(f"\n  Per-regime benefit:")
    for r in valid_regimes:
        mask = regimes_out == r
        b = benefits[mask]
        se = b.std() / np.sqrt(mask.sum())
        # Bootstrap CI
        boot_means = []
        for _ in range(N_BOOT):
            idx = rng.choice(len(b), len(b), replace=True)
            boot_means.append(b[idx].mean())
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        per_regime.append({
            "regime": int(r), "n": int(mask.sum()),
            "mean_benefit": float(b.mean()),
            "ci_low": float(ci_lo), "ci_high": float(ci_hi),
        })
        print(f"    Regime {r} (N={mask.sum():3d}): benefit = {b.mean():+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Kruskal-Wallis
    groups = [benefits[regimes_out == r] for r in valid_regimes]
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        k = len(groups)
        eta_sq = (H - k + 1) / (N - k) if N > k else 0
    else:
        H, p_kw, eta_sq = 0, 1.0, 0

    print(f"\n  Kruskal-Wallis: H={H:.3f}, p={p_kw:.4f}, eta²={eta_sq:.4f}")

    # Decision rule
    if p_kw < 0.05:
        decision = "ADR-027 reversal triggered: regime re-elevated to main paper as supporting claim"
    elif p_kw < 0.15:
        decision = "Still underpowered; do not expand further (would suggest p-hacking); regime stays supplementary"
    else:
        decision = "E4 hypothesis rejected; supplementary note: no significant regime × benefit interaction detected"
    print(f"\n  Decision: {decision}")

    # Save outputs
    git_hash = os.popen("git rev-parse HEAD").read().strip()
    summary = {
        "kruskal_H": float(H),
        "kruskal_p": float(p_kw),
        "eta_squared": float(eta_sq),
        "decision_rule_outcome": decision,
        "n_curves": N,
        "n_regimes_tested": len(valid_regimes),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    per_regime_df = pd.DataFrame(per_regime)
    per_regime_df.to_csv(OUT_DIR / "per_regime_table.csv", index=False)

    lock_meta = {
        "script": "scripts/77_regime_benefit_loocv.py",
        "seed": SEED,
        "n_boot": N_BOOT,
        "loocv_definition": "leave-one-curve-out",
        "rmse_timegrid": "t > 14d (same as 72_canonical_benchmark.py)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
    }
    with open(OUT_DIR / "lock_metadata.json", "w") as f:
        json.dump(lock_meta, f, indent=2)

    # Box plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        data_to_plot = [benefits[regimes_out == r] for r in valid_regimes]
        bp = ax.boxplot(data_to_plot, labels=[f"R{r}\n(N={(regimes_out==r).sum()})" for r in valid_regimes])
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Benefit (RMSE_DirectQ - RMSE_Active)")
        ax.set_xlabel("Regime")
        ax.set_title(f"Regime × Active Observer Benefit (LOOCV, N={N})\nKruskal-Wallis p={p_kw:.4f}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "benefit_by_regime.png", dpi=150)
        plt.close(fig)
        print(f"\n  Saved box plot to {OUT_DIR / 'benefit_by_regime.png'}")
    except ImportError:
        print("\n  matplotlib not available, skipping box plot")

    print(f"\n  All outputs saved to {OUT_DIR}/")
    return summary


if __name__ == "__main__":
    run_loocv()
