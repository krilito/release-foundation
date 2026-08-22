"""CASP — Conditional Active-Subspace Posterior.

A simulator-agnostic variational posterior over mechanism parameters
that outputs a low-rank Gaussian per curve. The active subspace is
curve-conditional; inactive directions are pulled toward the prior so
the model does not hallucinate certainty on sloppy directions.

Three claims structure this module (see docs/plan_60_...):
  1. Output object is new:    (mu, U, sigma_active) per curve, not a
     theta point.
  2. Training objective is new: simulator-decoded ELBO with KL on the
     inactive subspace.
  3. Existing methods are special cases:
       RF -> theta   = CASP with r=0 (point estimate)
       SBI posterior = CASP with r=n_params (full rank, no inactive KL)

Simulator-agnostic from day one: the module never imports a concrete
ReleaseSimulator subclass. Pass `simulator: ReleaseSimulator` at
construction time. PLGABiphasic / Weibull / future mechanisms all
plug in without code change.
"""

from casp.encoder import CASPEncoder
from casp.decoder import SimulatorDecoder
from casp.elbo import CASPLoss
from casp.fib import fib_posterior, sample_theta_fib, simulator_jacobian
from casp.pullback import pullback_target, PullbackTarget
from casp.ensemble import ensemble_posterior, per_tree_predictions, EnsembleTarget

__all__ = [
    "CASPEncoder", "SimulatorDecoder", "CASPLoss",
    "fib_posterior", "sample_theta_fib", "simulator_jacobian",
    "pullback_target", "PullbackTarget",
    "ensemble_posterior", "per_tree_predictions", "EnsembleTarget",
]
