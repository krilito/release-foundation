"""27i - Pyro HMC/NUTS runner for the B5 canonical subset benchmark.

Purpose:
    This is the first concrete B5 sampler runner scaffold. It depends on
    `pyro-ppl`, but the script is written so that the repo can carry the
    runner before the dependency is installed.

Backend:
    - subset / mouth / log-posterior definition come from
      `27h_b5_logposterior_backend.py`
    - the sampler operates on an unconstrained 9D latent `z`, mapped to
      bounded PLGA theta by a sigmoid box transform

Outputs (once pyro is installed and the script is run):
    outputs/27i_b5_hmc_pyro_runner/
        posterior_draws_curve_<id>.csv
        summary_curve_<id>.json

Notes:
    - This script is intentionally lazy-imported so it remains syntax-valid
      before `pyro-ppl` is installed.
    - It is not yet counted as B5 completion; it is the runner side of the
      now-frozen 27h backend.
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


SCRIPT_27H = Path("scripts/27h_b5_logposterior_backend.py")
DEFAULT_SUBSET_DIR = Path("outputs/27f_hmc_subset_export_test")
DEFAULT_OUT = Path("outputs/27i_b5_hmc_pyro_runner")


def load_backend_module():
    spec = importlib.util.spec_from_file_location("b5_backend_27h", SCRIPT_27H)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_27H}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_pyro():
    try:
        import pyro  # noqa: F401
        from pyro.infer.mcmc import MCMC, HMC, NUTS
        return MCMC, HMC, NUTS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyro-ppl is not installed. "
            "Minimal proposed dependency route for B5 is `uv add pyro-ppl` "
            "after DECISIONS.md receives a matching ADR entry."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--curve-id", type=int, required=True)
    parser.add_argument("--kernel", choices=("nuts", "hmc"), default="nuts")
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backend = load_backend_module()
    subset = backend.load_subset(args.subset_dir)
    row = subset.loc[subset["curve_id"] == int(args.curve_id)]
    if row.empty:
        raise ValueError(f"curve_id {args.curve_id} not found in subset package {args.subset_dir}")
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

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[27i] dry run")
        print(f"[27i] curve_id        : {args.curve_id}")
        print(f"[27i] kernel          : {args.kernel}")
        print(f"[27i] num_samples     : {args.num_samples}")
        print(f"[27i] warmup_steps    : {args.warmup_steps}")
        print(f"[27i] num_chains      : {args.num_chains}")
        print(f"[27i] step_size       : {args.step_size}")
        print(f"[27i] theta_dim       : {len(lows)}")
        return

    MCMC, HMC, NUTS = import_pyro()
    torch.manual_seed(args.seed)

    if args.kernel == "nuts":
        kernel = NUTS(potential_fn=potential_fn, adapt_step_size=True, adapt_mass_matrix=True)
    else:
        kernel = HMC(potential_fn=potential_fn, step_size=args.step_size, num_steps=10)

    mcmc = MCMC(
        kernel,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        num_chains=args.num_chains,
        disable_progbar=False,
    )
    initial_params = {"": torch.zeros_like(lows)}
    mcmc.run(init_params=initial_params)
    samples = mcmc.get_samples()

    # Pyro returns unconstrained z samples; map them back into theta space.
    z = samples[""]
    theta = backend.box_transform_from_unconstrained(z, lows, highs).detach().cpu().numpy()
    theta_df = pd.DataFrame(theta, columns=sim.param_names)
    theta_df.to_csv(args.out / f"posterior_draws_curve_{args.curve_id}.csv", index=False)

    summary = {
        "curve_id": int(args.curve_id),
        "kernel": args.kernel,
        "num_samples": int(args.num_samples),
        "warmup_steps": int(args.warmup_steps),
        "num_chains": int(args.num_chains),
        "posterior_mean": theta_df.mean().to_dict(),
        "posterior_std": theta_df.std(ddof=1).to_dict(),
    }
    (args.out / f"summary_curve_{args.curve_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[27i] wrote posterior draws for curve {args.curve_id} to {args.out}")


if __name__ == "__main__":
    main()
