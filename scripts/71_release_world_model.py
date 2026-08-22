"""
Drug Release World Model — RSSM-based latent dynamics.

Architecture follows Hafner et al. (PlaNet, ICML 2019 / Dreamer, ICLR 2020):
  - Deterministic state: h_t = GRU(h_{t-1}, z_{t-1}, x)
  - Stochastic state: z_t ~ N(mu, sigma) (prior or posterior)
  - Decoder: Q(t) = dec(z_t, h_t)
  - Training: reconstruction + KL regularization

Simplified for drug release (no actions, no rewards, fixed formulation):
  - x = formulation features (constant over time)
  - o_t = Q(t) observed release at time t
  - Goal: from early Q observations, predict full curve

Run:
    .\.venv\Scripts\python.exe scripts\71_release_world_model.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================
# Data (reuse from 70)
# ============================================================

def load_all_curves() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]))
    formulations = formulations[formulations["curve_id"].isin(common_ids)].copy()

    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    return formulations, curve_matrix, time_grid


def prepare_features(formulations: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_cols = [c for c in formulations.columns if c != "curve_id"]
    X = formulations[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
    ], remainder="drop")
    X_proc = preprocessor.fit_transform(X)
    return X_proc.astype(np.float32), feature_cols


# ============================================================
# RSSM: Recurrent State-Space Model
# ============================================================

class RSSM(nn.Module):
    """Recurrent State-Space Model for drug release.

    Core idea from PlaNet/Dreamer:
      - Deterministic state h_t captures temporal history
      - Stochastic state z_t captures uncertainty
      - Prior p(z_t | h_t) predicts next state without observation
      - Posterior q(z_t | o_t, h_t) refines with actual observation

    For drug release:
      - No actions (formulation is fixed)
      - Observations are release values Q(t)
      - Time is continuous (passed as dt to dynamics)
    """

    def __init__(self, n_features: int, latent_dim: int = 8,
                 hidden_dim: int = 64, deter_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim

        # --- Prior: p(z_t | h_t) ---
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu + log_sigma
        )

        # --- Posterior: q(z_t | o_t, h_t) ---
        self.obs_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # (Q_obs, dt)
            nn.ReLU(),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu + log_sigma
        )

        # --- Deterministic dynamics: h_t = GRU(h_{t-1}, [z_{t-1}, x, dt]) ---
        self.dynamics_gru = nn.GRUCell(
            input_size=latent_dim + n_features + 1,  # z + x + dt
            hidden_size=deter_dim,
        )

        # --- Decoder: Q(t) = dec(z_t, h_t) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Q ∈ [0, 1]
        )

    def prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(z_t | h_t) — predict next state without observation."""
        out = self.prior_net(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-5.0, 2.0)
        return mu, log_sigma

    def posterior(self, h: torch.Tensor, obs: torch.Tensor,
                  dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z_t | o_t, h_t) — refine with actual observation."""
        obs_emb = self.obs_encoder(torch.cat([obs, dt], dim=-1))
        out = self.posterior_net(torch.cat([h, obs_emb], dim=-1))
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-5.0, 2.0)
        return mu, log_sigma

    def sample(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        sigma = log_sigma.exp()
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def init_state(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial deterministic state h₀ and stochastic state z₀."""
        h0 = torch.zeros(batch_size, self.deter_dim, device=device)
        z0 = torch.zeros(batch_size, self.latent_dim, device=device)
        return h0, z0


# ============================================================
# Release World Model
# ============================================================

class ReleaseWorldModel(nn.Module):
    """Drug release world model with RSSM dynamics.

    Training: teacher-forced on full curves
    Inference: observe early points → posterior → forward simulate
    """

    def __init__(self, n_features: int, latent_dim: int = 8,
                 hidden_dim: int = 64, deter_dim: int = 64):
        super().__init__()
        self.rssm = RSSM(n_features, latent_dim, hidden_dim, deter_dim)
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim

    def _process_sequence(
        self,
        x: torch.Tensor,          # (B, D) formulation features
        q_obs: torch.Tensor,       # (B, T) observed Q values
        times: torch.Tensor,       # (T,) time points
        observe_mask: torch.Tensor, # (B, T) bool, which times are observed
    ) -> tuple[dict, dict]:
        """Run RSSM over a sequence, collecting prior/posterior states.

        Returns:
            prior_stats: {mu, log_sigma} at each time step
            states: {h, z, q_pred} at each time step
        """
        B = x.shape[0]
        T = len(times)
        device = x.device

        h, z = self.rssm.init_state(B, device)
        prior_mus, prior_log_sigmas = [], []
        post_mus, post_log_sigmas = [], []
        q_preds = []

        for t in range(T):
            # Compute dt
            dt_val = times[t] - (times[t-1] if t > 0 else 0.0)
            dt = torch.full((B, 1), dt_val, device=device)

            # Deterministic dynamics
            gru_input = torch.cat([z, x, dt], dim=-1)
            h = self.rssm.dynamics_gru(gru_input, h)

            # Prior
            p_mu, p_log_sigma = self.rssm.prior(h)
            prior_mus.append(p_mu)
            prior_log_sigmas.append(p_log_sigma)

            # Posterior (if observed)
            obs = q_obs[:, t:t+1]  # (B, 1)
            q_mu, q_log_sigma = self.rssm.posterior(h, obs, dt)
            post_mus.append(q_mu)
            post_log_sigmas.append(q_log_sigma)

            # Use posterior during training (teacher forcing)
            z = self.rssm.sample(q_mu, q_log_sigma)

            # Decode
            q_pred = self.rssm.decoder(torch.cat([z, h], dim=-1))
            q_preds.append(q_pred)

        prior_stats = {
            "mu": torch.stack(prior_mus, dim=1),        # (B, T, latent_dim)
            "log_sigma": torch.stack(prior_log_sigmas, dim=1),
        }
        post_stats = {
            "mu": torch.stack(post_mus, dim=1),
            "log_sigma": torch.stack(post_log_sigmas, dim=1),
        }
        states = {
            "q_pred": torch.cat(q_preds, dim=1),  # (B, T)
        }
        return prior_stats, post_stats, states

    def training_loss(
        self,
        x: torch.Tensor,
        q_true: torch.Tensor,
        times: torch.Tensor,
        observe_mask: torch.Tensor,
        kl_weight: float = 0.1,
        mono_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss: reconstruction + KL + monotonicity.

        Args:
            x: (B, D) formulation features
            q_true: (B, T) true release curves
            times: (T,) time points
            observe_mask: (B, T) which times are observed
            kl_weight: weight for KL divergence
            mono_weight: weight for monotonicity penalty
        """
        prior_stats, post_stats, states = self._process_sequence(
            x, q_true, times, observe_mask,
        )

        # Reconstruction loss (only at observed times)
        q_pred = states["q_pred"]
        recon_loss = ((q_pred - q_true) ** 2 * observe_mask.float()).sum() / observe_mask.float().sum().clamp(min=1)

        # KL divergence: KL[posterior || prior]
        kl = self._kl_divergence(
            post_stats["mu"], post_stats["log_sigma"],
            prior_stats["mu"], prior_stats["log_sigma"],
        )
        kl_loss = kl.mean()

        # Monotonicity penalty: Q(t+1) >= Q(t)
        q_diff = q_pred[:, 1:] - q_pred[:, :-1]
        mono_penalty = F.relu(-q_diff).mean()

        total = recon_loss + kl_weight * kl_loss + mono_weight * mono_penalty
        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl_loss,
            "mono": mono_penalty,
        }

    def _kl_divergence(self, q_mu, q_log_sigma, p_mu, p_log_sigma):
        """KL[N(q_mu, q_sigma) || N(p_mu, p_sigma)].

        q_log_sigma = ln(sigma_q), p_log_sigma = ln(sigma_p)
        KL = ln(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2)/(2*sigma_p^2) - 1/2
        """
        q_var = q_log_sigma.exp().pow(2)  # sigma_q^2
        p_var = p_log_sigma.exp().pow(2)  # sigma_p^2
        kl = (p_log_sigma - q_log_sigma
              + (q_var + (q_mu - p_mu).pow(2)) / (2 * p_var)
              - 0.5)
        return kl.sum(dim=-1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        early_q: torch.Tensor,
        early_times: torch.Tensor,
        all_times: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict full curve from early observations.

        Steps:
          1. Run posterior on early observations → infer z_T_early
          2. From z_T_early, run prior dynamics forward → predict future
          3. Return mean and std of predictions

        Args:
            x: (1, D) formulation features
            early_q: (1, n_obs) early release values
            early_times: (n_obs,) observation times
            all_times: (T,) all time points to predict

        Returns:
            q_mean: (T,) mean prediction
            q_std: (T,) std over samples
        """
        device = x.device
        B = 1
        T = len(all_times)

        # Phase 1: Run posterior on early observations
        h, z = self.rssm.init_state(B, device)
        for i, t_obs in enumerate(early_times):
            dt_val = t_obs - (early_times[i-1] if i > 0 else 0.0)
            dt = torch.full((B, 1), dt_val, device=device)
            gru_input = torch.cat([z, x, dt], dim=-1)
            h = self.rssm.dynamics_gru(gru_input, h)
            obs = early_q[:, i:i+1]
            q_mu, q_log_sigma = self.rssm.posterior(h, obs, dt)
            z = self.rssm.sample(q_mu, q_log_sigma)

        # Phase 2: Forward simulate from last early time
        last_early_t = early_times[-1]
        future_mask = all_times > last_early_t + 0.01  # avoid float comparison
        future_times = all_times[future_mask]
        future_indices = np.where(future_mask)[0]

        q_samples = []
        for _ in range(n_samples):
            h_sample = h.clone()
            # Start from posterior sample at last observation time
            z_sample = self.rssm.sample(q_mu, q_log_sigma)
            q_trajectory = []
            prev_t = last_early_t

            for j, t_future in enumerate(future_times):
                dt_val = t_future - prev_t
                dt = torch.full((B, 1), dt_val, device=device)
                gru_input = torch.cat([z_sample, x, dt], dim=-1)
                h_sample = self.rssm.dynamics_gru(gru_input, h_sample)

                # Decode current state BEFORE overwriting z with prior
                q_pred = self.rssm.decoder(torch.cat([z_sample, h_sample], dim=-1))
                q_trajectory.append(q_pred.item())

                # Now step z forward using prior dynamics for next iteration
                p_mu, p_log_sigma = self.rssm.prior(h_sample)
                z_sample = self.rssm.sample(p_mu, p_log_sigma)
                prev_t = t_future

            q_samples.append(q_trajectory)

        q_samples = np.array(q_samples)  # (n_samples, n_future)
        q_mean = q_samples.mean(axis=0)
        q_std = q_samples.std(axis=0)

        # Combine: early times = observed, future times = predicted
        full_mean = np.zeros(T)
        full_std = np.zeros(T)
        for i, t in enumerate(all_times):
            early_match = np.where(np.abs(early_times - t) < 0.01)[0]
            if len(early_match) > 0:
                full_mean[i] = early_q[0, early_match[0]].item()
                full_std[i] = 0.0
            else:
                fi = np.where(np.abs(future_times - t) < 0.01)[0]
                if len(fi) > 0:
                    full_mean[i] = q_mean[fi[0]]
                    full_std[i] = q_std[fi[0]]
                else:
                    # Interpolate or use last known
                    full_mean[i] = q_mean[-1] if len(q_mean) > 0 else 0.5
                    full_std[i] = q_std[-1] if len(q_std) > 0 else 0.1

        return full_mean, full_std


# ============================================================
# Training
# ============================================================

def train_world_model(
    model: ReleaseWorldModel,
    X_train: np.ndarray,
    curves_train: np.ndarray,
    time_grid: np.ndarray,
    n_epochs: int = 500,
    lr: float = 3e-4,
    batch_size: int = 32,
    kl_weight: float = 0.1,
    mono_weight: float = 1.0,
    device: str = "cpu",
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    curves_t = torch.tensor(curves_train, dtype=torch.float32, device=device)
    times_t = torch.tensor(time_grid, dtype=torch.float32, device=device)
    observe_mask = torch.ones(len(X_train), len(time_grid), dtype=torch.bool, device=device)

    n = len(X_train)
    losses = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = X_t[idx]
            q_batch = curves_t[idx]
            mask_batch = observe_mask[idx]

            loss_dict = model.training_loss(
                x_batch, q_batch, times_t, mask_batch,
                kl_weight=kl_weight, mono_weight=mono_weight,
            )
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}")

    return losses


# ============================================================
# Evaluation
# ============================================================

def evaluate_world_model(
    model: ReleaseWorldModel,
    X_test: np.ndarray,
    curves_test: np.ndarray,
    time_grid: np.ndarray,
    early_times: list[float],
    future_start: float = 14.0,
    n_samples: int = 50,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    future_mask = time_grid > future_start
    early_indices = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    early_times_arr = time_grid[early_indices]

    all_preds_mean = []
    all_preds_std = []

    for i in range(len(X_test)):
        x = torch.tensor(X_test[i:i+1], dtype=torch.float32, device=device)
        early_q = torch.tensor(curves_test[i:i+1, early_indices], dtype=torch.float32, device=device)
        q_mean, q_std = model.predict(x, early_q, early_times_arr, time_grid, n_samples=n_samples)
        all_preds_mean.append(q_mean)
        all_preds_std.append(q_std)

    all_preds_mean = np.array(all_preds_mean)
    all_preds_std = np.array(all_preds_std)

    y_true = curves_test[:, future_mask]
    y_pred = all_preds_mean[:, future_mask]

    per_curve_r2 = []
    for i in range(len(y_true)):
        ss_r = np.sum((y_true[i] - y_pred[i]) ** 2)
        ss_t = np.sum((y_true[i] - y_true[i].mean()) ** 2) + 1e-10
        per_curve_r2.append(1.0 - ss_r / ss_t)

    return {
        "r2_median": float(np.median(per_curve_r2)),
        "r2_mean": float(np.mean(per_curve_r2)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "frac_positive": float(np.mean(np.array(per_curve_r2) > 0)),
    }


def evaluate_directq(
    X_train, curves_train, X_test, curves_test,
    time_grid, early_times, future_start=14.0,
) -> dict[str, float]:
    future_mask = time_grid > future_start
    early_indices = [int(np.argmin(np.abs(time_grid - t))) for t in early_times]
    early_train = curves_train[:, early_indices]
    early_test = curves_test[:, early_indices]
    X_aug_train = np.hstack([X_train, early_train])
    X_aug_test = np.hstack([X_test, early_test])
    y_train = curves_train[:, future_mask]
    y_test = curves_test[:, future_mask]

    model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                 max_features="sqrt", bootstrap=True,
                                 n_jobs=-1, random_state=42)
    model.fit(X_aug_train, y_train)
    y_pred = np.clip(model.predict(X_aug_test), 0.0, 1.1)

    per_curve_r2 = []
    for i in range(len(y_test)):
        ss_r = np.sum((y_test[i] - y_pred[i]) ** 2)
        ss_t = np.sum((y_test[i] - y_test[i].mean()) ** 2) + 1e-10
        per_curve_r2.append(1.0 - ss_r / ss_t)

    return {
        "r2_median": float(np.median(per_curve_r2)),
        "r2_mean": float(np.mean(per_curve_r2)),
        "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
        "frac_positive": float(np.mean(np.array(per_curve_r2) > 0)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Release World Model (RSSM)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--deter-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--mono-weight", type=float, default=0.5)
    parser.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    parser.add_argument("--future-start", type=float, default=14.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    print("Loading data...")
    formulations, curves, time_grid = load_all_curves()
    X, feature_cols = prepare_features(formulations)
    print(f"  {len(X)} curves, {len(time_grid)} time points, {X.shape[1]} features")
    print(f"  Early: {args.early_times}, Future: t > {args.future_start}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    wm_results = []
    dq_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{args.n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        curves_train, curves_test = curves[train_idx], curves[test_idx]

        # Train world model
        print("  Training world model (RSSM)...")
        model = ReleaseWorldModel(
            n_features=X.shape[1],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            deter_dim=args.deter_dim,
        )
        train_world_model(
            model, X_train, curves_train, time_grid,
            n_epochs=args.epochs, lr=args.lr,
            kl_weight=args.kl_weight, mono_weight=args.mono_weight,
            device=args.device,
        )
        wm_metrics = evaluate_world_model(
            model, X_test, curves_test, time_grid,
            args.early_times, args.future_start,
            n_samples=args.n_samples, device=args.device,
        )
        wm_results.append(wm_metrics)
        print(f"  World Model: R²={wm_metrics['r2_median']:.3f}, RMSE={wm_metrics['rmse']:.4f}")

        # Direct-Q baseline
        print("  Training Direct-Q baseline...")
        dq_metrics = evaluate_directq(
            X_train, curves_train, X_test, curves_test,
            time_grid, args.early_times, args.future_start,
        )
        dq_results.append(dq_metrics)
        print(f"  Direct-Q:    R²={dq_metrics['r2_median']:.3f}, RMSE={dq_metrics['rmse']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, results in [("World Model (RSSM)", wm_results), ("Direct-Q (ExtraTrees)", dq_results)]:
        r2_med = np.median([r["r2_median"] for r in results])
        r2_mean = np.mean([r["r2_mean"] for r in results])
        rmse_mean = np.mean([r["rmse"] for r in results])
        frac_pos = np.mean([r["frac_positive"] for r in results])
        print(f"\n  {name}:")
        print(f"    R² median:    {r2_med:.3f}")
        print(f"    R² mean:      {r2_mean:.3f}")
        print(f"    RMSE:         {rmse_mean:.4f}")
        print(f"    R² > 0 frac:  {frac_pos:.1%}")


if __name__ == "__main__":
    main()
