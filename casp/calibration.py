"""Calibration / PI-coverage helpers for CASP.

The new evaluation primitive this paper introduces is per-curve uncertainty:
for each test curve, sample S draws from q(theta | x, y_early), simulate
each, and compute predictive intervals on Q(t). Coverage of the observed
curve is the new metric RF / ET cannot produce.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


@torch.no_grad()
def predict_with_uncertainty(
    encoder: nn.Module,
    decoder: nn.Module,
    X: Tensor,                   # (N, F)
    t_grid: Tensor,              # (T,)
    n_samples: int = 64,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Return Q-sample tensor (N, T, n_samples) plus aggregate statistics.

    Use n_samples large enough for stable quantile estimation (>= 50).
    """
    encoder.eval(); decoder.eval()
    encoder.to(device); decoder.to(device)
    X = X.to(device); t_grid = t_grid.to(device)

    q = encoder(X)
    theta = encoder.sample_theta(q, n_samples=n_samples)              # (S, N, P)
    Q = decoder(theta, t_grid)                                        # (S, N, T)
    Q = Q.permute(1, 2, 0).cpu().numpy()                              # (N, T, S)

    mean = Q.mean(axis=-1)
    median = np.median(Q, axis=-1)
    lo90 = np.percentile(Q, 5.0, axis=-1)
    hi90 = np.percentile(Q, 95.0, axis=-1)
    lo50 = np.percentile(Q, 25.0, axis=-1)
    hi50 = np.percentile(Q, 75.0, axis=-1)

    return {
        "Q_samples": Q,           # (N, T, S)
        "mean": mean,             # (N, T)
        "median": median,
        "lo90": lo90,
        "hi90": hi90,
        "lo50": lo50,
        "hi50": hi50,
    }


def per_curve_r2(y_obs: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-curve R^2 on the rows of y_obs vs y_pred.  shape (N, T) -> (N,)."""
    ss_res = ((y_obs - y_pred) ** 2).sum(axis=-1)
    y_mean = y_obs.mean(axis=-1, keepdims=True)
    ss_tot = ((y_obs - y_mean) ** 2).sum(axis=-1)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return r2


def coverage(
    y_obs: np.ndarray,            # (N, T)
    lo: np.ndarray,               # (N, T)
    hi: np.ndarray,               # (N, T)
) -> np.ndarray:
    """Per-curve fraction of timepoints inside [lo, hi]. shape (N,)."""
    inside = (y_obs >= lo) & (y_obs <= hi)
    return inside.mean(axis=-1)


def summarize_calibration(
    y_obs: np.ndarray,
    pred: dict[str, np.ndarray],
) -> dict[str, float]:
    """Single-scalar summary used in fold reports."""
    r2_mean = per_curve_r2(y_obs, pred["mean"])
    r2_median = per_curve_r2(y_obs, pred["median"])
    cov90 = coverage(y_obs, pred["lo90"], pred["hi90"])
    cov50 = coverage(y_obs, pred["lo50"], pred["hi50"])
    pi_width_90 = (pred["hi90"] - pred["lo90"]).mean(axis=-1)

    return {
        "r2_mean_median": float(np.nanmedian(r2_mean)),
        "r2_mean_mean": float(np.nanmean(r2_mean)),
        "r2_median_median": float(np.nanmedian(r2_median)),
        "frac_R2_gte_0": float((r2_mean >= 0).mean()),
        "frac_R2_gte_0p5": float((r2_mean >= 0.5).mean()),
        "frac_R2_gte_0p9": float((r2_mean >= 0.9).mean()),
        "cov90_mean": float(cov90.mean()),
        "cov90_median": float(np.median(cov90)),
        "cov50_mean": float(cov50.mean()),
        "pi_width_90_median": float(np.median(pi_width_90)),
    }
