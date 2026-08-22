"""27m - Multi-chain diagnostics for the dependency-free B5 HMC fallback.

Purpose:
    Posterior-overlap evidence is only as convincing as chain quality. This
    script runs multiple independent `27j` chains for selected locked curves
    and reports split-R-hat diagnostics for the unconstrained HMC fallback.

Outputs:
    outputs/27m_b5_chain_diagnostics/
        posterior_draws_curve_<id>_chain_<k>.csv
        per_chain_diagnostics.csv
        per_curve_param_diagnostics.csv
        per_curve_summary.csv
        overall_summary.json

Notes:
    - This script is diagnostic, not a benchmark headline producer.
    - Split-R-hat is the first quality gate; ESS can be added later if needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


SCRIPT_27J = Path("scripts/27j_b5_hmc_torch_runner.py")
DEFAULT_OUT = Path("outputs/27m_b5_chain_diagnostics")
DEFAULT_CURVE_IDS = [11, 20]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute split-R-hat per parameter.

    chains: shape (n_chains, n_draws, n_params)
    """
    if chains.ndim != 3:
        raise ValueError(f"Expected chains shape (C, N, D), got {chains.shape}")
    c, n, d = chains.shape
    if c < 2 or n < 4:
        return np.full(d, np.nan)

    n_even = n - (n % 2)
    chains = chains[:, :n_even, :]
    half = n_even // 2
    split = np.concatenate([chains[:, :half, :], chains[:, half:, :]], axis=0)
    m = split.shape[0]
    n = split.shape[1]

    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    W = chain_vars.mean(axis=0)
    B = n * chain_means.var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    rhat = np.sqrt(np.maximum(var_hat / np.maximum(W, 1e-12), 0.0))
    return rhat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-ids", nargs="*", type=int, default=DEFAULT_CURVE_IDS)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--leapfrog-steps", type=int, default=6)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--target-accept", type=float, default=0.65)
    parser.add_argument("--adapt-rate", type=float, default=0.02)
    parser.add_argument("--use-reasonable-step-init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mod27j = load_module(SCRIPT_27J, "script27j_diag")
    backend = mod27j.load_backend_module()
    subset = backend.load_subset(mod27j.DEFAULT_SUBSET_DIR)

    param_rows: list[dict[str, object]] = []
    chain_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for curve_id in args.curve_ids:
        row = subset.loc[subset["curve_id"] == int(curve_id)]
        if row.empty:
            raise ValueError(f"curve_id {curve_id} not found in locked claim subset")
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

        sim = backend.PLGABiphasic()
        torch_backend = backend.make_torch_potential(sim, backend.OBS_TIMES, obs_q)
        lows = torch_backend["lows"]
        highs = torch_backend["highs"]
        potential_fn = torch_backend["potential_fn"]

        theta_path = mod27j.DEFAULT_SUBSET_DIR / "theta_reference.csv"
        theta_ref = None
        if theta_path.exists():
            theta_ref_df = pd.read_csv(theta_path)
            theta_ref_df = theta_ref_df.loc[theta_ref_df["curve_id"] == int(curve_id)]
            if not theta_ref_df.empty:
                theta_ref = theta_ref_df.iloc[0][sim.param_names].to_numpy(dtype=float)
        if theta_ref is None:
            theta_ref_t = 0.5 * (lows + highs)
        else:
            theta_ref_t = torch.tensor(theta_ref, dtype=torch.float32)
        frac = ((theta_ref_t - lows) / (highs - lows)).clamp(1e-5, 1.0 - 1e-5)
        q_init = torch.log(frac) - torch.log1p(-frac)

        chain_draws = []
        chain_accepts = []
        for chain_id in range(args.num_chains):
            draws_z, diagnostics = mod27j.hmc_sample(
                potential_fn=potential_fn,
                q_init=q_init,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                leapfrog_steps=args.leapfrog_steps,
                step_size=args.step_size,
                adapt_step_size=True,
                use_reasonable_step_init=args.use_reasonable_step_init,
                target_accept=args.target_accept,
                adapt_rate=args.adapt_rate,
                seed=args.seed + int(curve_id) * 100 + chain_id,
            )
            draws_z_t = torch.tensor(draws_z, dtype=torch.float32)
            theta = backend.box_transform_from_unconstrained(draws_z_t, lows, highs).detach().cpu().numpy()
            chain_draws.append(theta)
            chain_accepts.append(float(diagnostics["accept_rate_post"]))
            theta_df = pd.DataFrame(theta, columns=sim.param_names)
            theta_df.insert(0, "chain_id", chain_id)
            theta_df.to_csv(args.out / f"posterior_draws_curve_{curve_id}_chain_{chain_id}.csv", index=False)
            chain_rows.append(
                {
                    "curve_id": int(curve_id),
                    "DP_Group": row["DP_Group"],
                    "chain_id": int(chain_id),
                    "initial_step_size": float(diagnostics["initial_step_size"]),
                    "adapted_initial_step_size": float(diagnostics["adapted_initial_step_size"]),
                    "final_step_size": float(diagnostics["final_step_size"]),
                    "accept_rate_total": float(diagnostics["accept_rate_total"]),
                    "accept_rate_warmup": float(diagnostics["accept_rate_warmup"]),
                    "accept_rate_post": float(diagnostics["accept_rate_post"]),
                    "mean_delta_H_post": float(diagnostics["mean_delta_H_post"]),
                    "median_delta_H_post": float(diagnostics["median_delta_H_post"]),
                }
            )

        chains = np.asarray(chain_draws)  # (C, N, D)
        rhat = split_rhat(chains)
        mean_accept = float(np.mean(chain_accepts))
        max_rhat = float(np.nanmax(rhat))
        for name, val in zip(sim.param_names, rhat):
            param_rows.append(
                {
                    "curve_id": int(curve_id),
                    "DP_Group": row["DP_Group"],
                    "param": name,
                    "split_rhat": float(val),
                }
            )
        curve_rows.append(
            {
                "curve_id": int(curve_id),
                "DP_Group": row["DP_Group"],
                "num_chains": int(args.num_chains),
                "draws_per_chain": int(args.num_samples),
                "mean_accept_rate_post": mean_accept,
                "max_split_rhat": max_rhat,
                "median_split_rhat": float(np.nanmedian(rhat)),
            }
        )

    chain_df = pd.DataFrame(chain_rows).sort_values(["curve_id", "chain_id"])
    param_df = pd.DataFrame(param_rows)
    curve_df = pd.DataFrame(curve_rows).sort_values("curve_id")
    chain_df.to_csv(args.out / "per_chain_diagnostics.csv", index=False)
    param_df.to_csv(args.out / "per_curve_param_diagnostics.csv", index=False)
    curve_df.to_csv(args.out / "per_curve_summary.csv", index=False)

    overall = {
        "curve_ids": [int(x) for x in args.curve_ids],
        "num_chains": int(args.num_chains),
        "draws_per_chain": int(args.num_samples),
        "mean_of_max_split_rhat": float(curve_df["max_split_rhat"].mean()),
        "worst_split_rhat": float(param_df["split_rhat"].max()),
    }
    (args.out / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print(curve_df.to_string(index=False))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
