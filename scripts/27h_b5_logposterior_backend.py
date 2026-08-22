"""27h - Sampler-agnostic log-posterior backend for the B5 HMC benchmark.

Purpose:
    B5 still lacks an HMC/NUTS dependency path, but the benchmark should not
    wait on sampler choice to define the posterior. This script provides the
    fixed-four canonical-mouth log posterior for the locked 27f subset package.

Posterior definition:
    - prior: `PLGABiphasic.prior()` uniform box
    - observations: fixed four early releases at t = [1, 3, 5, 7] days
    - likelihood: independent Gaussian with the same heteroskedastic sigma
      rule used by Active Observer:
        sigma(t, q) = max(0.03, 0.08 * max(|q|, 0.05))

Outputs:
    outputs/27h_b5_logposterior_backend/
        curve_summary.csv
        lock_metadata.json
        map_smoke_summary.json      # if --map-smoke is used

Notes:
    - This is not an HMC implementation.
    - It is the backend that a future HMC / NUTS runner should call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator import PLGABiphasic


DEFAULT_SUBSET_DIR = Path("outputs/27f_hmc_subset_export_test")
DEFAULT_OUT = Path("outputs/27h_b5_logposterior_backend")
OBS_TIMES = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)


def get_git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=D:/release-foundation", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_subset(subset_dir: Path) -> pd.DataFrame:
    mouth = pd.read_csv(subset_dir / "fixed_four_mouth.csv")
    manifest = pd.read_csv(subset_dir / "subset_manifest.csv")
    df = manifest.merge(mouth, on="curve_id", how="inner")
    required = {
        "curve_id",
        "subset_role",
        "split_name",
        "obs_release_t1",
        "obs_release_t3",
        "obs_release_t5",
        "obs_release_t7",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Subset package missing columns: {sorted(missing)}")
    return df.sort_values("curve_id").reset_index(drop=True)


def sigma_rule(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.maximum(0.03, 0.08 * np.maximum(np.abs(q), 0.05))


def sigma_rule_torch(q: torch.Tensor) -> torch.Tensor:
    q = q.to(dtype=torch.float32)
    return torch.maximum(
        torch.full_like(q, 0.03),
        0.08 * torch.maximum(q.abs(), torch.full_like(q, 0.05)),
    )


def make_logposterior(sim: PLGABiphasic, obs_t: np.ndarray, obs_q: np.ndarray):
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(float)
    highs = prior.high.numpy().astype(float)
    sigmas = sigma_rule(obs_q)
    log_norm = -0.5 * np.log(2.0 * np.pi * sigmas**2)

    def log_prior(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        inside = np.all(theta >= lows) and np.all(theta <= highs)
        if not inside:
            return float("-inf")
        volume = np.prod(highs - lows)
        return float(-np.log(volume))

    def log_likelihood(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if not (np.all(theta >= lows) and np.all(theta <= highs)):
            return float("-inf")
        pred = sim.simulate_numpy(theta, obs_t)
        resid = (obs_q - pred) / sigmas
        return float(np.sum(log_norm - 0.5 * resid**2))

    def log_posterior(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return lp
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return float("-inf")
        return lp + ll

    def residuals(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        pred = sim.simulate_numpy(theta, obs_t)
        return (obs_q - pred) / sigmas

    return lows, highs, sigmas, log_prior, log_likelihood, log_posterior, residuals


def box_transform_from_unconstrained(z: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(z)
    return lows + (highs - lows) * sig


def make_torch_potential(sim: PLGABiphasic, obs_t: np.ndarray, obs_q: np.ndarray):
    prior = sim.prior().base_dist
    lows = prior.low.to(dtype=torch.float32)
    highs = prior.high.to(dtype=torch.float32)
    span = highs - lows
    obs_t_t = torch.tensor(obs_t, dtype=torch.float32)
    obs_q_t = torch.tensor(obs_q, dtype=torch.float32)
    sigma_t = sigma_rule_torch(obs_q_t)
    log_norm = -0.5 * torch.log(2.0 * torch.pi * sigma_t**2)
    log_prior_const = -torch.sum(torch.log(span))

    def potential_fn(z_unconstrained: torch.Tensor) -> torch.Tensor:
        theta = box_transform_from_unconstrained(z_unconstrained, lows, highs)
        pred = sim.simulate(theta.unsqueeze(0), obs_t_t)[0]
        resid = (obs_q_t - pred) / sigma_t
        log_like = torch.sum(log_norm - 0.5 * resid**2)
        # theta = low + span * sigmoid(z); include full box Jacobian explicitly
        log_jac = torch.sum(torch.log(span) + torch.nn.functional.logsigmoid(z_unconstrained) + torch.nn.functional.logsigmoid(-z_unconstrained))
        log_post = log_prior_const + log_like + log_jac
        return -log_post

    return {
        "lows": lows,
        "highs": highs,
        "obs_t": obs_t_t,
        "obs_q": obs_q_t,
        "sigma": sigma_t,
        "potential_fn": potential_fn,
    }


def fit_map_smoke(
    sim: PLGABiphasic,
    obs_t: np.ndarray,
    obs_q: np.ndarray,
    n_restarts: int,
    seed: int,
) -> dict[str, object]:
    lows, highs, sigmas, log_prior, log_likelihood, log_posterior, residuals = make_logposterior(sim, obs_t, obs_q)
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lows + highs)
    best_theta: np.ndarray | None = None
    best_cost = np.inf

    for k in range(n_restarts):
        x0 = mid if k == 0 else rng.uniform(lows, highs)
        try:
            res = least_squares(
                residuals,
                x0=x0,
                bounds=(lows, highs),
                method="trf",
                max_nfev=400,
            )
        except Exception:
            continue
        cost = float(np.sum(res.fun**2))
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best_theta = res.x.copy()

    if best_theta is None:
        raise RuntimeError("MAP smoke fit failed for all restarts")

    pred = sim.simulate_numpy(best_theta, obs_t)
    return {
        "theta_map": best_theta.tolist(),
        "pred_obs": pred.tolist(),
        "obs_q": obs_q.tolist(),
        "sigma": sigmas.tolist(),
        "log_prior": float(log_prior(best_theta)),
        "log_likelihood": float(log_likelihood(best_theta)),
        "log_posterior": float(log_posterior(best_theta)),
        "weighted_residual_norm_sq": float(best_cost),
    }


def torch_smoke(
    sim: PLGABiphasic,
    obs_t: np.ndarray,
    obs_q: np.ndarray,
    theta_reference: np.ndarray | None = None,
) -> dict[str, object]:
    backend = make_torch_potential(sim, obs_t, obs_q)
    lows = backend["lows"]
    highs = backend["highs"]
    span = highs - lows
    if theta_reference is None:
        theta_ref = 0.5 * (lows + highs)
    else:
        theta_ref = torch.tensor(theta_reference, dtype=torch.float32)
    frac = ((theta_ref - lows) / span).clamp(1e-5, 1.0 - 1e-5)
    z0 = torch.log(frac) - torch.log1p(-frac)
    z0 = z0.clone().detach().requires_grad_(True)
    potential = backend["potential_fn"](z0)
    potential.backward()
    grad = z0.grad.detach().cpu().numpy()
    theta_eval = box_transform_from_unconstrained(z0.detach(), lows, highs).detach().cpu().numpy()
    return {
        "theta_eval": theta_eval.tolist(),
        "obs_q": backend["obs_q"].detach().cpu().numpy().tolist(),
        "sigma": backend["sigma"].detach().cpu().numpy().tolist(),
        "potential": float(potential.detach().cpu().item()),
        "grad_norm": float(np.linalg.norm(grad)),
        "grad": grad.tolist(),
        "all_grad_finite": bool(np.isfinite(grad).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--map-smoke", action="store_true")
    parser.add_argument("--torch-smoke", action="store_true")
    parser.add_argument("--curve-id", type=int, default=None)
    parser.add_argument("--map-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sim = PLGABiphasic()
    df = load_subset(args.subset_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    curve_rows = []
    for _, row in df.iterrows():
        obs_q = np.array(
            [
                row["obs_release_t1"],
                row["obs_release_t3"],
                row["obs_release_t5"],
                row["obs_release_t7"],
            ],
            dtype=float,
        )
        _, _, sigmas, _, _, _, _ = make_logposterior(sim, OBS_TIMES, obs_q)
        curve_rows.append(
            {
                "curve_id": int(row["curve_id"]),
                "subset_role": row["subset_role"],
                "split_name": row["split_name"],
                "obs_release_t1": float(obs_q[0]),
                "obs_release_t3": float(obs_q[1]),
                "obs_release_t5": float(obs_q[2]),
                "obs_release_t7": float(obs_q[3]),
                "sigma_t1": float(sigmas[0]),
                "sigma_t3": float(sigmas[1]),
                "sigma_t5": float(sigmas[2]),
                "sigma_t7": float(sigmas[3]),
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(args.out / "curve_summary.csv", index=False)

    metadata = {
        "script": "scripts/27h_b5_logposterior_backend.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "git_hash": get_git_hash(),
        "subset_dir": str(args.subset_dir),
        "n_curves": int(len(curve_df)),
        "obs_times_days": OBS_TIMES.tolist(),
        "likelihood_sigma_rule": "sigma=max(0.03, 0.08*max(|q|,0.05))",
        "prior_source": "PLGABiphasic.prior() uniform box",
        "notes": [
            "This is a sampler-agnostic backend for B5, not an HMC implementation.",
            "Future HMC/NUTS runners should consume the 27f subset package and this same log posterior.",
        ],
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.map_smoke:
        if args.curve_id is None:
            curve_id = int(curve_df.iloc[0]["curve_id"])
        else:
            curve_id = int(args.curve_id)
        row = curve_df.loc[curve_df["curve_id"] == curve_id]
        if row.empty:
            raise ValueError(f"curve_id {curve_id} not present in subset package")
        row = row.iloc[0]
        obs_q = np.array(
            [
                row["obs_release_t1"],
                row["obs_release_t3"],
                row["obs_release_t5"],
                row["obs_release_t7"],
            ],
            dtype=float,
        )
        summary = fit_map_smoke(
            sim=sim,
            obs_t=OBS_TIMES,
            obs_q=obs_q,
            n_restarts=args.map_restarts,
            seed=args.seed,
        )
        summary["curve_id"] = curve_id
        (args.out / "map_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[27h] map smoke wrote {args.out / 'map_smoke_summary.json'}")

    if args.torch_smoke:
        if args.curve_id is None:
            curve_id = int(curve_df.iloc[0]["curve_id"])
        else:
            curve_id = int(args.curve_id)
        row = curve_df.loc[curve_df["curve_id"] == curve_id]
        if row.empty:
            raise ValueError(f"curve_id {curve_id} not present in subset package")
        row = row.iloc[0]
        obs_q = np.array(
            [
                row["obs_release_t1"],
                row["obs_release_t3"],
                row["obs_release_t5"],
                row["obs_release_t7"],
            ],
            dtype=float,
        )
        theta_ref_df = None
        theta_path = args.subset_dir / "theta_reference.csv"
        if theta_path.exists():
            theta_ref_df = pd.read_csv(theta_path)
            theta_ref_df = theta_ref_df.loc[theta_ref_df["curve_id"] == curve_id]
        theta_ref = None
        if theta_ref_df is not None and not theta_ref_df.empty:
            theta_cols = sim.param_names
            theta_ref = theta_ref_df.iloc[0][theta_cols].to_numpy(dtype=float)
        summary = torch_smoke(
            sim=sim,
            obs_t=OBS_TIMES,
            obs_q=obs_q,
            theta_reference=theta_ref,
        )
        summary["curve_id"] = curve_id
        (args.out / "torch_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[27h] torch smoke wrote {args.out / 'torch_smoke_summary.json'}")

    print(f"[27h] wrote {args.out / 'curve_summary.csv'}")


if __name__ == "__main__":
    main()
