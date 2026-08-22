"""Ensemble-CASP — per-curve posterior from a tree ensemble.

For each test curve x:
  1. RF predicts theta per individual tree in the forest (n_estimators
     predictions per curve, NOT just the averaged forest prediction)
  2. Fit a low-rank Gaussian to those n_estimators theta samples
  3. Output (mu, U, sigma) — same shape as FIB / pull-back targets
  4. Sample, simulate, get PI band

The (mu, U, sigma) here encodes formulation-conditional uncertainty
in theta: where trees agree, sigma is small; where trees disagree,
sigma is large. This is exactly the right UQ for zero-shot prediction,
because:

  - Fisher info gives identifiability of theta GIVEN a fixed curve point;
    it cannot tell you "the formulation didn't constrain theta enough."
  - Tree-ensemble variance gives uncertainty over theta GIVEN the
    formulation, which IS the zero-shot uncertainty source.

Combined with the simulator push-forward, this yields per-curve PI
that widens whenever the formulation poorly identifies theta — exactly
what a zero-shot predictor needs to honestly report.

This is the second concrete instantiation of the unified (mu, U, sigma)
output object, alongside FIB and pull-back. All three share the same
downstream sampling + simulator + recenter pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class EnsembleTarget:
    mu: np.ndarray             # (P,) — forest mean
    U: np.ndarray              # (P, r) — orthonormal top-r covariance eigvecs
    sigma_active: np.ndarray   # (r,) — sqrt eigvals on top-r dirs
    n_trees_used: int          # =n_estimators
    spectrum: np.ndarray       # full eigenvalues (P,), descending


def per_tree_predictions(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """Return shape (n_test, n_trees, n_outputs)."""
    preds = np.stack([est.predict(X) for est in rf.estimators_], axis=1)
    return preds


def ensemble_posterior(
    theta_per_tree: np.ndarray,    # (T_trees, P)
    rank: int,
    prior_low: np.ndarray | None = None,
    prior_high: np.ndarray | None = None,
    nugget: float = 1e-6,
) -> EnsembleTarget:
    """Fit a low-rank Gaussian to a set of tree-level theta predictions.

    Standard sample mean / covariance / eigendecomp — this is exactly
    a Laplace approximation to the empirical bootstrap posterior, just
    with bagged trees instead of explicit bootstrap.
    """
    n_trees, P = theta_per_tree.shape
    mu = theta_per_tree.mean(axis=0)                               # (P,)
    delta = theta_per_tree - mu                                    # (T, P)
    Sigma = (delta.T @ delta) / max(n_trees - 1, 1)
    Sigma = 0.5 * (Sigma + Sigma.T) + nugget * np.eye(P)
    eigvals, eigvecs = np.linalg.eigh(Sigma)                       # ascending
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]            # descending
    eigvals = np.clip(eigvals, 1e-10, None)

    r = min(rank, P)
    U = eigvecs[:, :r].astype(np.float32)
    sigma_active = np.sqrt(eigvals[:r]).astype(np.float32)

    # Optional: cap sigma at prior std along each direction (avoids
    # uninformative directions giving non-physical posterior spread).
    if prior_low is not None and prior_high is not None:
        prior_std = (prior_high - prior_low) / np.sqrt(12.0)
        prior_std_along_U = np.sqrt(((U ** 2) * (prior_std ** 2)[:, None]).sum(axis=0))
        sigma_active = np.minimum(sigma_active, prior_std_along_U.astype(np.float32))

    return EnsembleTarget(
        mu=mu.astype(np.float32), U=U, sigma_active=sigma_active,
        n_trees_used=n_trees, spectrum=eigvals.astype(np.float32),
    )
