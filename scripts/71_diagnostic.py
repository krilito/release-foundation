"""Diagnostics for RSSM World Model — 3 checks before deciding to continue."""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
_spec = importlib.util.spec_from_file_location("_71", Path(__file__).resolve().parent / "71_release_world_model.py")
_71 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_71)

def main():
    formulations, curves, time_grid = _71.load_all_curves()
    X, feature_cols = _71.prepare_features(formulations)
    idx_train, idx_test = train_test_split(range(len(X)), test_size=0.2, random_state=42)
    X_train, X_test = X[idx_train], X[idx_test]
    curves_train, curves_test = curves[idx_train], curves[idx_test]

    model = _71.ReleaseWorldModel(n_features=X.shape[1], latent_dim=8, hidden_dim=64, deter_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    curves_t = torch.tensor(curves_train, dtype=torch.float32)
    times_t = torch.tensor(time_grid, dtype=torch.float32)
    observe_mask = torch.ones(len(X_train), len(time_grid), dtype=torch.bool)

    # === DIAGNOSTIC 1: KL Health ===
    print("=== DIAGNOSTIC 1: KL Health Check ===")
    for epoch in range(300):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_kl = 0; epoch_recon = 0; n_b = 0
        for start in range(0, len(X_train), 32):
            idx = perm[start:start+32]
            loss_dict = model.training_loss(X_t[idx], curves_t[idx], times_t, observe_mask[idx], kl_weight=0.01, mono_weight=0.5)
            optimizer.zero_grad(); loss_dict["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            epoch_kl += loss_dict["kl"].item(); epoch_recon += loss_dict["recon"].item(); n_b += 1
        if (epoch+1) % 100 == 0:
            kl_per_dim = epoch_kl/n_b/8
            print(f"  Epoch {epoch+1}: recon={epoch_recon/n_b:.4f}, KL={epoch_kl/n_b:.4f}, KL/dim={kl_per_dim:.4f}")
    final_kl_dim = kl_per_dim  # save for summary

    # === DIAGNOSTIC 1b: Posterior vs Prior variance ===
    print("\n=== DIAGNOSTIC 1b: Posterior vs Prior ===")
    model.eval()
    with torch.no_grad():
        prior_stats, post_stats, _ = model._process_sequence(
            torch.tensor(X_test[:10], dtype=torch.float32),
            torch.tensor(curves_test[:10], dtype=torch.float32),
            times_t, observe_mask[:10])
        print(f"  Prior mu var: {prior_stats['mu'].var(0).mean():.4f}")
        print(f"  Post mu var:  {post_stats['mu'].var(0).mean():.4f}")
        print(f"  Prior sigma:  {prior_stats['log_sigma'].exp().mean():.4f}")
        print(f"  Post sigma:   {post_stats['log_sigma'].exp().mean():.4f}")

    # === DIAGNOSTIC 2: Posterior Coverage ===
    print("\n=== DIAGNOSTIC 2: Posterior Coverage ===")
    future_mask = time_grid > 14.0
    early_indices = [2, 4, 5, 6]
    early_times_arr = time_grid[early_indices]
    cov90_list = []; cov50_list = []
    std_list = []
    for i in range(len(X_test)):
        x = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        eq = torch.tensor(curves_test[i:i+1, early_indices], dtype=torch.float32)
        q_mean, q_std = model.predict(x, eq, early_times_arr, time_grid, n_samples=50)
        yt = curves_test[i, future_mask]; yp = q_mean[future_mask]; ys = q_std[future_mask]
        cov90_list.append(np.mean((yt >= yp-1.645*ys) & (yt <= yp+1.645*ys)))
        cov50_list.append(np.mean((yt >= yp-0.674*ys) & (yt <= yp+0.674*ys)))
        std_list.append(np.mean(ys))
    cov90_mean = float(np.mean(cov90_list))
    pred_std_mean = float(np.mean(std_list))
    print(f"  Coverage 90%: mean={cov90_mean:.3f}, median={np.median(cov90_list):.3f}")
    print(f"  Coverage 50%: mean={np.mean(cov50_list):.3f}, median={np.median(cov50_list):.3f}")
    print(f"  Pred std: mean={pred_std_mean:.4f}, min={np.min(std_list):.4f}, max={np.max(std_list):.4f}")

    # Write summary JSON
    summary = {
        "cov90": round(cov90_mean, 3),
        "kl_per_dim": round(float(final_kl_dim), 3),
        "pred_std": round(pred_std_mean, 3),
    }
    out_dir = Path("outputs/71_diagnostic_bugfix")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {out_dir / 'summary.json'}")

    # === DIAGNOSTIC 3: Latent Space Audit ===
    print("\n=== DIAGNOSTIC 3: Latent Space Audit ===")
    model.eval()
    all_z = []
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.tensor(X_test[i:i+1], dtype=torch.float32)
            eq = torch.tensor(curves_test[i:i+1, early_indices], dtype=torch.float32)
            h, z = model.rssm.init_state(1, "cpu")
            for j, t_obs in enumerate(early_times_arr):
                dt_val = t_obs - (early_times_arr[j-1] if j > 0 else 0.0)
                dt = torch.full((1, 1), dt_val)
                h = model.rssm.dynamics_gru(torch.cat([z, x, dt], dim=-1), h)
                q_mu, q_ls = model.rssm.posterior(h, eq[:, j:j+1], dt)
                z = model.rssm.sample(q_mu, q_ls)
            all_z.append(z.numpy()[0])
    all_z = np.array(all_z)
    pca = PCA(n_components=4)
    z_pca = pca.fit_transform(all_z)
    print(f"  PCA explained var: {pca.explained_variance_ratio_}")
    print(f"  z std per dim: {all_z.std(axis=0)}")

    # Check polymer separation
    formulations_test = formulations.iloc[idx_test]
    def get_polymer(dp):
        parts = str(dp).split("-")
        for k in range(len(parts)):
            c = "-".join(parts[k:])
            if any(c.startswith(kw) for kw in ["PLGA","PCL","PLA","PEA"]) or "PVL" in c: return c
        return str(dp)
    polymers = np.array([get_polymer(g) for g in formulations_test["DP_Group"].values])
    for p in np.unique(polymers):
        m = polymers == p
        if m.sum() > 2:
            zm = z_pca[m].mean(0)
            print(f"    {p} (n={m.sum()}): PCA1={zm[0]:.3f}, PCA2={zm[1]:.3f}")

if __name__ == "__main__":
    main()
