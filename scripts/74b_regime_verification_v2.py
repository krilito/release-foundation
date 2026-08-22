"""Task C: Regime verification v2 — theta-space as independent test.

Drops the tautological active-set silhouette (regimes were clustered ON
active-set Jaccard distance; measuring silhouette in that space is circular).

Tests whether regimes are also separable in independent feature spaces:
  - Theta-space (oracle kinetic parameters)
  - x-space (formulation features)
  - Curve-shape space (handcrafted features)

Framing: A NEGATIVE silhouette supports the interpretation that regime
structure is orthogonal to parameter values. A POSITIVE silhouette would
mean regimes are also clusterable by parameter values.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("outputs/74_regime_verification_v2")
N_PERM = 5000
SEED = 0


def perm_pvalue(features, labels, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    real = silhouette_score(features, labels)
    null = [silhouette_score(features, rng.permutation(labels)) for _ in range(n)]
    p = np.mean(np.array(null) >= real)
    return real, p


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    theta_df = pd.read_csv("data/theta_bank.csv")
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    regime_df = pd.read_csv("outputs/33_active_set_regimes/regime_assignments.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    # Filter to internal181 only to avoid fid collision with cross321
    regime_df_int = regime_df[regime_df["dataset"] == "internal181"]
    regime_map = dict(zip(regime_df_int["fid"], regime_df_int["regime"]))
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"])
                        & set(theta_df["curve_id"]) & set(regime_map.keys()))

    regime_labels = np.array([regime_map[cid] for cid in common_ids])
    valid = regime_labels > 0
    regime_valid = regime_labels[valid]

    # Theta space
    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    theta_matrix = theta_df.set_index("curve_id").loc[common_ids][theta_cols].to_numpy()
    theta_std = StandardScaler().fit_transform(theta_matrix)[valid]

    # x-space (formulation features)
    feature_cols = [c for c in formulations.columns if c != "curve_id"]
    X_df = formulations.set_index("curve_id").loc[common_ids][feature_cols].copy()
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]
    X_num = X_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    x_std = StandardScaler().fit_transform(X_num)[valid]

    # Curve-shape space
    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    def extract_curve_features(Q):
        plateau = max(Q[-1], 0.01)
        t50 = np.interp(0.5, Q, time_grid) if plateau > 0.5 else np.nan
        auc = np.trapezoid(Q, time_grid) / (time_grid[-1] - time_grid[0])
        max_rate = np.max(np.gradient(Q, time_grid))
        t7_idx = int(np.argmin(np.abs(time_grid - 7.0)))
        early_late = Q[t7_idx] / max(plateau, 1e-3)
        t1_idx = int(np.argmin(np.abs(time_grid - 1.0)))
        burst = Q[t1_idx]
        return [plateau, t50, auc, max_rate, early_late, burst]

    curve_feats = np.array([extract_curve_features(curve_matrix[i]) for i in range(len(common_ids))])
    for j in range(curve_feats.shape[1]):
        mask = np.isnan(curve_feats[:, j])
        if mask.any():
            curve_feats[mask, j] = np.nanmean(curve_feats[:, j])
    curve_std = StandardScaler().fit_transform(curve_feats)[valid]

    # Compute silhouettes
    print("=" * 60)
    print("REGIME VERIFICATION v2 (independent feature spaces)")
    print("=" * 60)
    print(f"  Curves with regime labels: {valid.sum()}")
    print(f"  N_PERM = {N_PERM}")
    print()

    results = {}
    for name, feats in [("Theta (oracle params)", theta_std),
                         ("Formulation x", x_std),
                         ("Curve-shape", curve_std)]:
        sil, p = perm_pvalue(feats, regime_valid)
        results[name] = {"silhouette": sil, "p_value": p}
        print(f"  {name:25s}: sil={sil:+.3f} (p={p:.4f})")

    # PCA visualization data
    pca_theta = PCA(n_components=2).fit_transform(theta_std)
    pca_x = PCA(n_components=2).fit_transform(x_std)

    # Decision rule
    sil_theta = results["Theta (oracle params)"]["silhouette"]
    if sil_theta > 0.2:
        decision = "Regime IS theta-cluster; reframe entire regime claim"
    elif sil_theta >= -0.10:
        decision = "Theta orthogonal to regime; ADR-027 supplementary framing valid"
    else:
        decision = "Regime explicitly anti-aligned with theta; novel finding worth highlighting"

    print(f"\n  Decision: {decision}")

    # Save
    git_hash = os.popen("git rev-parse HEAD").read().strip()
    summary = {
        "sil_theta": float(results["Theta (oracle params)"]["silhouette"]),
        "p_theta": float(results["Theta (oracle params)"]["p_value"]),
        "sil_x": float(results["Formulation x"]["silhouette"]),
        "p_x": float(results["Formulation x"]["p_value"]),
        "sil_curve": float(results["Curve-shape"]["silhouette"]),
        "p_curve": float(results["Curve-shape"]["p_value"]),
        "decision": decision,
        "n_curves": int(valid.sum()),
        "n_perm": N_PERM,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lock_meta = {
        "script": "scripts/74b_regime_verification_v2.py",
        "note": "Active-set silhouette dropped as tautological (regimes clustered on active-set Jaccard)",
        "n_perm": N_PERM,
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
    }
    with open(OUT_DIR / "lock_metadata.json", "w") as f:
        json.dump(lock_meta, f, indent=2)

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
