"""
Latent Dynamics Model for Drug Release — v1 minimal validation.

Core idea: learn release dynamics directly from data, without fitting
kinetic parameters (theta). The model learns a latent state z(t) that
evolves over time, conditioned on formulation features.

Architecture:
    Encoder:  (formulation x, early Q) → z₀ ∈ R^k
    Dynamics: z_{t+1} = GRU(z_t, x)
    Decoder:  z_t → Q(t)

Comparison: Direct-Q baseline (ExtraTrees, same data split).

Run:
    .\.venv\Scripts\python.exe scripts\70_latent_dynamics_v1.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================
# Data loading
# ============================================================

def load_all_curves() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load formulations + curves, return aligned data.

    Returns:
        formulations: (n, features) with curve_id
        curves_matrix: (n, T) release values on common time grid
        time_grid: (T,) time points
    """
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")

    # Normalize release
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    # Common time grid: use the rollout_times from the active observer
    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                          14.0, 21.0, 28.0, 42.0, 56.0, 84.0])

    # Interpolate each curve onto the common grid
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]))
    formulations = formulations[formulations["curve_id"].isin(common_ids)].copy()

    curve_matrix = np.zeros((len(common_ids), len(time_grid)))
    for i, cid in enumerate(common_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        if len(t) == 1:
            curve_matrix[i, :] = y[0]
        else:
            curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])

    return formulations, curve_matrix, time_grid


def prepare_features(formulations: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract and scale formulation features."""
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
# Latent Dynamics Model
# ============================================================

class LatentDynamicsModel(nn.Module):
    """Learn release dynamics in latent space.

    Given early Q observations + formulation features, infer initial
    latent state z₀, then forward-simulate using learned dynamics.
    """

    def __init__(self, n_features: int, latent_dim: int = 4,
                 hidden_dim: int = 64, n_gru_layers: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (x, early_Q) → z₀
        # early_Q is variable-length, so we use a small net per observation
        self.obs_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_features + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Dynamics: GRU conditioned on formulation
        self.dynamics = nn.GRU(
            input_size=n_features + 1,  # formulation + time_delta
            hidden_size=latent_dim,
            num_layers=n_gru_layers,
            batch_first=True,
        )

        # Decoder: z_t → Q(t)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Q ∈ [0, 1]
        )

    def encode(self, x: torch.Tensor, early_q: torch.Tensor,
               early_times: torch.Tensor) -> torch.Tensor:
        """Infer initial latent state from formulation + early observations.

        Args:
            x: (B, D) formulation features
            early_q: (B, n_obs) observed Q values
            early_times: (n_obs,) observation times

        Returns:
            z₀: (B, latent_dim)
        """
        B, n_obs = early_q.shape
        # Encode each observation
        obs_emb = self.obs_encoder(early_q.unsqueeze(-1))  # (B, n_obs, 16)
        # Pool observations (mean)
        obs_pooled = obs_emb.mean(dim=1)  # (B, 16)
        # Concatenate with formulation features
        enc_input = torch.cat([x, obs_pooled], dim=-1)
        z0 = self.encoder(enc_input)
        return z0

    def forward_simulate(self, z0: torch.Tensor, x: torch.Tensor,
                         times: torch.Tensor) -> torch.Tensor:
        """Forward-simulate latent state and decode to Q.

        Args:
            z0: (B, latent_dim) initial state
            x: (B, D) formulation features
            times: (T,) time points to simulate

        Returns:
            q_pred: (B, T) predicted release
        """
        B = z0.shape[0]
        T = len(times)

        # Build GRU input: for each time step, input = [x, dt]
        # dt is time since previous step
        dt = torch.zeros(T, device=times.device)
        dt[0] = times[0]
        dt[1:] = times[1:] - times[:-1]

        # Expand x and dt for batch
        x_expanded = x.unsqueeze(1).expand(B, T, -1)  # (B, T, D)
        dt_expanded = dt.unsqueeze(0).unsqueeze(-1).expand(B, T, 1)  # (B, T, 1)
        gru_input = torch.cat([x_expanded, dt_expanded], dim=-1)  # (B, T, D+1)

        # GRU forward
        h0 = z0.unsqueeze(0)  # (1, B, latent_dim) for single layer
        gru_out, _ = self.dynamics(gru_input, h0)  # (B, T, latent_dim)

        # Decode each time step
        q_pred = self.decoder(gru_out).squeeze(-1)  # (B, T)
        return q_pred

    def forward(self, x: torch.Tensor, early_q: torch.Tensor,
                early_times: torch.Tensor, all_times: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode → simulate → decode."""
        z0 = self.encode(x, early_q, early_times)
        q_pred = self.forward_simulate(z0, x, all_times)
        return q_pred


# ============================================================
# Direct-Q baseline (ExtraTrees)
# ============================================================

class DirectQBaseline:
    """ExtraTrees: (formulation + early Q) → future Q."""

    def __init__(self):
        self.model = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            bootstrap=True, n_jobs=-1, random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X_test), 0.0, 1.1)


# ============================================================
# Training and evaluation
# ============================================================

def train_latent_model(
    model: LatentDynamicsModel,
    X_train: np.ndarray,
    curves_train: np.ndarray,
    time_grid: np.ndarray,
    early_indices: list[int],
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> list[float]:
    """Train latent dynamics model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    curves_t = torch.tensor(curves_train, dtype=torch.float32, device=device)
    times_t = torch.tensor(time_grid, dtype=torch.float32, device=device)
    early_times_t = times_t[early_indices]
    early_indices_t = torch.tensor(early_indices, dtype=torch.long, device=device)

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
            curves_batch = curves_t[idx]
            early_q = curves_batch[:, early_indices_t]

            q_pred = model(x_batch, early_q, early_times_t, times_t)
            loss = nn.functional.mse_loss(q_pred, curves_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}")

    return losses


def evaluate_latent_model(
    model: LatentDynamicsModel,
    X_test: np.ndarray,
    curves_test: np.ndarray,
    time_grid: np.ndarray,
    early_indices: list[int],
    future_start: float = 14.0,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate latent dynamics model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        curves_t = torch.tensor(curves_test, dtype=torch.float32, device=device)
        times_t = torch.tensor(time_grid, dtype=torch.float32, device=device)
        early_times_t = times_t[early_indices]
        early_indices_t = torch.tensor(early_indices, dtype=torch.long, device=device)

        early_q = curves_t[:, early_indices_t]
        q_pred = model(X_t, early_q, early_times_t, times_t).cpu().numpy()

    future_mask = time_grid > future_start
    y_true = curves_test[:, future_mask]
    y_pred = q_pred[:, future_mask]

    # Per-curve R² (the only meaningful R² for release curves)
    per_curve_r2 = []
    for i in range(len(y_true)):
        ss_r = np.sum((y_true[i] - y_pred[i]) ** 2)
        ss_t = np.sum((y_true[i] - y_true[i].mean()) ** 2) + 1e-10
        per_curve_r2.append(1.0 - ss_r / ss_t)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        "r2_median": float(np.median(per_curve_r2)),
        "r2_mean": float(np.mean(per_curve_r2)),
        "rmse": float(rmse),
        "mae": float(mae),
        "frac_positive": float(np.mean(np.array(per_curve_r2) > 0)),
    }


def evaluate_directq(
    X_train: np.ndarray, curves_train: np.ndarray,
    X_test: np.ndarray, curves_test: np.ndarray,
    time_grid: np.ndarray, early_indices: list[int],
    future_start: float = 14.0,
) -> dict[str, float]:
    """Evaluate Direct-Q baseline."""
    future_mask = time_grid > future_start
    early_times = time_grid[early_indices]

    # Build augmented features: formulation + early Q values
    early_train = curves_train[:, early_indices]
    early_test = curves_test[:, early_indices]

    X_aug_train = np.hstack([X_train, early_train])
    X_aug_test = np.hstack([X_test, early_test])

    y_train = curves_train[:, future_mask]
    y_test = curves_test[:, future_mask]

    baseline = DirectQBaseline()
    baseline.fit(X_aug_train, y_train)
    y_pred = baseline.predict(X_aug_test)

    per_curve_r2 = []
    for i in range(len(y_test)):
        ss_r = np.sum((y_test[i] - y_pred[i]) ** 2)
        ss_t = np.sum((y_test[i] - y_test[i].mean()) ** 2) + 1e-10
        per_curve_r2.append(1.0 - ss_r / ss_t)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    return {
        "r2_median": float(np.median(per_curve_r2)),
        "r2_mean": float(np.mean(per_curve_r2)),
        "rmse": float(rmse),
        "mae": float(mae),
        "frac_positive": float(np.mean(np.array(per_curve_r2) > 0)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Latent Dynamics v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-times", nargs="+", type=float,
                        default=[1.0, 3.0, 5.0, 7.0])
    parser.add_argument("--future-start", type=float, default=14.0)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    print("Loading data...")
    formulations, curves, time_grid = load_all_curves()
    X, feature_cols = prepare_features(formulations)
    n_curves, T = curves.shape
    print(f"  {n_curves} curves, {T} time points, {X.shape[1]} features")

    # Map early times to indices
    early_indices = []
    for t in args.early_times:
        idx = int(np.argmin(np.abs(time_grid - t)))
        if time_grid[idx] not in early_indices:
            early_indices.append(idx)
    early_indices = sorted(set(int(np.argmin(np.abs(time_grid - t))) for t in args.early_times))
    print(f"  Early observation indices: {early_indices} → times: {time_grid[early_indices]}")

    # Cross-validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    latent_results = []
    directq_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{args.n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        curves_train, curves_test = curves[train_idx], curves[test_idx]

        # Train latent dynamics model
        print("  Training latent dynamics model...")
        model = LatentDynamicsModel(
            n_features=X.shape[1],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )
        train_latent_model(
            model, X_train, curves_train, time_grid, early_indices,
            n_epochs=args.epochs, lr=args.lr, device=args.device,
        )
        latent_metrics = evaluate_latent_model(
            model, X_test, curves_test, time_grid, early_indices,
            future_start=args.future_start, device=args.device,
        )
        latent_results.append(latent_metrics)
        print(f"  Latent: R²={latent_metrics['r2_median']:.3f}, RMSE={latent_metrics['rmse']:.4f}")

        # Train Direct-Q baseline
        print("  Training Direct-Q baseline...")
        dq_metrics = evaluate_directq(
            X_train, curves_train, X_test, curves_test,
            time_grid, early_indices, future_start=args.future_start,
        )
        directq_results.append(dq_metrics)
        print(f"  DirectQ: R²={dq_metrics['r2_median']:.3f}, RMSE={dq_metrics['rmse']:.4f}")

    # Diagnostic: what does the model actually predict?
    print("\n--- Diagnostic (last fold) ---")
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        times_t = torch.tensor(time_grid, dtype=torch.float32)
        early_q = torch.tensor(curves_test[:, early_indices], dtype=torch.float32)
        q_pred_last = model(X_t, early_q, times_t[early_indices], times_t).numpy()
    future_mask = time_grid > args.future_start
    y_true_f = curves_test[:, future_mask]
    y_pred_f = q_pred_last[:, future_mask]
    print(f"  True future range: [{y_true_f.min():.3f}, {y_true_f.max():.3f}], mean={y_true_f.mean():.3f}")
    print(f"  Pred future range: [{y_pred_f.min():.3f}, {y_pred_f.max():.3f}], mean={y_pred_f.mean():.3f}")
    pred_stds = y_pred_f.std(axis=1)
    print(f"  Per-curve pred std: mean={pred_stds.mean():.4f}, min={pred_stds.min():.4f}, max={pred_stds.max():.4f}")
    for i in range(min(3, len(y_true_f))):
        rmse_i = np.sqrt(np.mean((y_true_f[i] - y_pred_f[i])**2))
        print(f"  Curve {i}: RMSE={rmse_i:.4f}")
        print(f"    True: {' '.join(f'{v:.3f}' for v in y_true_f[i])}")
        print(f"    Pred: {' '.join(f'{v:.3f}' for v in y_pred_f[i])}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Data: {n_curves} curves, {args.n_folds}-fold CV")
    print(f"  Early observations: {args.early_times}")
    print(f"  Future evaluation: t > {args.future_start}")
    print()

    for name, results in [("Latent Dynamics", latent_results), ("Direct-Q (ExtraTrees)", directq_results)]:
        r2_med = np.median([r["r2_median"] for r in results])
        r2_mean = np.mean([r["r2_mean"] for r in results])
        rmse_mean = np.mean([r["rmse"] for r in results])
        mae_mean = np.mean([r["mae"] for r in results])
        frac_pos = np.mean([r["frac_positive"] for r in results])
        print(f"  {name}:")
        print(f"    R² median:     {r2_med:.3f}")
        print(f"    R² mean:       {r2_mean:.3f}")
        print(f"    RMSE:          {rmse_mean:.4f}")
        print(f"    MAE:           {mae_mean:.4f}")
        print(f"    R² > 0 frac:   {frac_pos:.1%}")
        print()


if __name__ == "__main__":
    main()
