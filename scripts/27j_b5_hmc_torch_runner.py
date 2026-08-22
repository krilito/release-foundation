"""27j - Dependency-free torch HMC runner for the B5 canonical subset.

Purpose:
    Provide a first real B5 posterior runner without adding any new sampler
    dependency. This is the fallback route explicitly allowed by the roadmap
    ("numpyro/pyro or hand-written NUTS"). Here we start with hand-written HMC.

Backend:
    - subset / mouth / differentiable potential come from
      `27h_b5_logposterior_backend.py`
    - the chain runs in the unconstrained 9D latent `z`
    - `z -> theta` uses the same sigmoid box transform as 27h / 27i

Outputs:
    outputs/27j_b5_hmc_torch_runner/
        posterior_draws_curve_<id>.csv
        summary_curve_<id>.json

Notes:
    - This is plain HMC, not NUTS.
    - The goal is to establish a real posterior-sampling path for B5 without
      dependency churn. If this works, it becomes the first evidence-bearing
      fallback while the Pyro path remains available.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


SCRIPT_27H = Path("scripts/27h_b5_logposterior_backend.py")
DEFAULT_SUBSET_DIR = Path("outputs/27f_hmc_subset_export_test")
DEFAULT_OUT = Path("outputs/27j_b5_hmc_torch_runner")


def load_backend_module():
    spec = importlib.util.spec_from_file_location("b5_backend_27h", SCRIPT_27H)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_27H}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def value_and_grad(potential_fn, z: torch.Tensor) -> tuple[float, torch.Tensor]:
    z_var = z.clone().detach().requires_grad_(True)
    value = potential_fn(z_var)
    value.backward()
    grad = z_var.grad.detach()
    return float(value.detach().cpu().item()), grad


def leapfrog(potential_fn, q: torch.Tensor, p: torch.Tensor, step_size: float, num_steps: int) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    q_new = q.clone()
    p_new = p.clone()

    U, grad = value_and_grad(potential_fn, q_new)
    if not torch.isfinite(grad).all() or not math.isfinite(U):
        return q, -p, float("inf"), False

    p_new = p_new - 0.5 * step_size * grad
    for step in range(num_steps):
        q_new = q_new + step_size * p_new
        U, grad = value_and_grad(potential_fn, q_new)
        if not torch.isfinite(grad).all() or not math.isfinite(U):
            return q, -p, float("inf"), False
        if step != num_steps - 1:
            p_new = p_new - step_size * grad
    p_new = p_new - 0.5 * step_size * grad
    return q_new, -p_new, U, True


def proposal_log_accept(
    potential_fn,
    q: torch.Tensor,
    step_size: float,
    num_steps: int,
    p0: torch.Tensor,
    current_U: float | None = None,
) -> tuple[float, torch.Tensor, float, bool]:
    if current_U is None:
        current_U, _ = value_and_grad(potential_fn, q)
    current_K = 0.5 * float(torch.sum(p0 * p0).cpu().item())
    q_prop, p_prop, prop_U, ok = leapfrog(potential_fn, q, p0, step_size, num_steps)
    if not ok or not math.isfinite(prop_U):
        return float("-inf"), q, float("inf"), False
    prop_K = 0.5 * float(torch.sum(p_prop * p_prop).cpu().item())
    delta_H = (prop_U + prop_K) - (current_U + current_K)
    return min(0.0, -delta_H), q_prop, prop_U, True


def find_reasonable_step_size(
    potential_fn,
    q_init: torch.Tensor,
    step_size_guess: float,
    seed: int,
    max_trials: int = 20,
) -> float:
    """Heuristic epsilon finder from the NUTS warmup recipe."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    step = float(np.clip(step_size_guess, 1e-4, 0.5))
    current_U, _ = value_and_grad(potential_fn, q_init)
    if not math.isfinite(current_U):
        raise RuntimeError("Initial potential is not finite")

    p0 = torch.randn(q_init.shape, dtype=q_init.dtype, device=q_init.device, generator=rng)
    log_alpha, _, _, ok = proposal_log_accept(
        potential_fn=potential_fn,
        q=q_init,
        step_size=step,
        num_steps=1,
        p0=p0,
        current_U=current_U,
    )
    trials = 0
    while (not ok or not math.isfinite(log_alpha)) and trials < max_trials:
        step *= 0.5
        p0 = torch.randn(q_init.shape, dtype=q_init.dtype, device=q_init.device, generator=rng)
        log_alpha, _, _, ok = proposal_log_accept(
            potential_fn=potential_fn,
            q=q_init,
            step_size=step,
            num_steps=1,
            p0=p0,
            current_U=current_U,
        )
        trials += 1
    if not ok or not math.isfinite(log_alpha):
        return float(np.clip(step, 1e-4, 0.25))

    direction = 1.0 if log_alpha > math.log(0.5) else -1.0
    trials = 0
    while trials < max_trials:
        proposed = step * (2.0 if direction > 0 else 0.5)
        if proposed < 1e-4 or proposed > 0.5:
            break
        p0 = torch.randn(q_init.shape, dtype=q_init.dtype, device=q_init.device, generator=rng)
        new_log_alpha, _, _, ok = proposal_log_accept(
            potential_fn=potential_fn,
            q=q_init,
            step_size=proposed,
            num_steps=1,
            p0=p0,
            current_U=current_U,
        )
        if not ok or not math.isfinite(new_log_alpha):
            if direction > 0:
                break
            step = proposed
            trials += 1
            continue
        if (direction > 0 and new_log_alpha < math.log(0.5)) or (
            direction < 0 and new_log_alpha > math.log(0.5)
        ):
            break
        step = proposed
        trials += 1
    return float(np.clip(step, 1e-4, 0.25))


def hmc_sample(
    potential_fn,
    q_init: torch.Tensor,
    num_samples: int,
    warmup_steps: int,
    leapfrog_steps: int,
    step_size: float,
    adapt_step_size: bool,
    use_reasonable_step_init: bool,
    target_accept: float,
    adapt_rate: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = q_init.clone().detach()
    current_U, _ = value_and_grad(potential_fn, q)
    if not math.isfinite(current_U):
        raise RuntimeError("Initial potential is not finite")

    total_iters = warmup_steps + num_samples
    draws: list[np.ndarray] = []
    accepts = 0
    accepts_warmup = 0
    energy_errors: list[float] = []
    accepted_post = 0
    initial_step_size = float(step_size)
    step_size_curr = float(np.clip(step_size, 1e-4, 0.25))
    if use_reasonable_step_init and adapt_step_size and warmup_steps > 0:
        step_size_curr = find_reasonable_step_size(
            potential_fn=potential_fn,
            q_init=q,
            step_size_guess=step_size_curr,
            seed=seed + 17,
        )
    adapted_initial_step_size = float(step_size_curr)

    for it in range(total_iters):
        p0 = torch.randn_like(q)
        log_alpha, q_prop, prop_U, ok = proposal_log_accept(
            potential_fn=potential_fn,
            q=q,
            step_size=step_size_curr,
            num_steps=leapfrog_steps,
            p0=p0,
            current_U=current_U,
        )
        accepted = False
        delta_H = float("inf")
        alpha = 0.0
        if ok and math.isfinite(prop_U) and math.isfinite(log_alpha):
            delta_H = -log_alpha if log_alpha < 0.0 else 0.0
            alpha = min(1.0, math.exp(log_alpha))
            if math.log(rng.uniform()) < log_alpha:
                q = q_prop
                current_U = prop_U
                accepted = True

        if it < warmup_steps:
            if accepted:
                accepts_warmup += 1
            if adapt_step_size:
                step_size_curr *= math.exp(adapt_rate * ((1.0 if accepted else 0.0) - target_accept))
                step_size_curr = float(np.clip(step_size_curr, 1e-4, 0.25))
        else:
            if accepted:
                accepted_post += 1
            draws.append(q.detach().cpu().numpy().copy())
            if math.isfinite(delta_H):
                energy_errors.append(delta_H)

        if accepted:
            accepts += 1

    draws_arr = np.asarray(draws)
    diagnostics = {
        "accept_rate_total": accepts / total_iters,
        "accept_rate_warmup": accepts_warmup / max(warmup_steps, 1),
        "accept_rate_post": accepted_post / max(num_samples, 1),
        "initial_step_size": initial_step_size,
        "adapted_initial_step_size": adapted_initial_step_size,
        "final_step_size": step_size_curr,
        "mean_delta_H_post": float(np.mean(energy_errors)) if energy_errors else float("nan"),
        "median_delta_H_post": float(np.median(energy_errors)) if energy_errors else float("nan"),
    }
    return draws_arr, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--curve-id", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--leapfrog-steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept", type=float, default=0.65)
    parser.add_argument("--adapt-rate", type=float, default=0.02)
    parser.add_argument("--no-adapt-step-size", action="store_true")
    parser.add_argument("--use-reasonable-step-init", action="store_true")
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

    theta_path = args.subset_dir / "theta_reference.csv"
    theta_ref = None
    if theta_path.exists():
        theta_ref_df = pd.read_csv(theta_path)
        theta_ref_df = theta_ref_df.loc[theta_ref_df["curve_id"] == int(args.curve_id)]
        if not theta_ref_df.empty:
            theta_ref = theta_ref_df.iloc[0][sim.param_names].to_numpy(dtype=float)
    if theta_ref is None:
        theta_ref_t = 0.5 * (lows + highs)
    else:
        theta_ref_t = torch.tensor(theta_ref, dtype=torch.float32)
    frac = ((theta_ref_t - lows) / (highs - lows)).clamp(1e-5, 1.0 - 1e-5)
    q_init = torch.log(frac) - torch.log1p(-frac)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[27j] dry run")
        print(f"[27j] curve_id          : {args.curve_id}")
        print(f"[27j] num_samples       : {args.num_samples}")
        print(f"[27j] warmup_steps      : {args.warmup_steps}")
        print(f"[27j] leapfrog_steps    : {args.leapfrog_steps}")
        print(f"[27j] step_size         : {args.step_size}")
        print(f"[27j] theta_dim         : {len(lows)}")
        return

    t0 = time.time()
    draws_z, diagnostics = hmc_sample(
        potential_fn=potential_fn,
        q_init=q_init,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        leapfrog_steps=args.leapfrog_steps,
        step_size=args.step_size,
        adapt_step_size=not args.no_adapt_step_size,
        use_reasonable_step_init=args.use_reasonable_step_init,
        target_accept=args.target_accept,
        adapt_rate=args.adapt_rate,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    draws_z_t = torch.tensor(draws_z, dtype=torch.float32)
    theta = backend.box_transform_from_unconstrained(draws_z_t, lows, highs).detach().cpu().numpy()
    theta_df = pd.DataFrame(theta, columns=sim.param_names)
    theta_df.to_csv(args.out / f"posterior_draws_curve_{args.curve_id}.csv", index=False)

    summary = {
        "curve_id": int(args.curve_id),
        "num_samples": int(args.num_samples),
        "warmup_steps": int(args.warmup_steps),
        "leapfrog_steps": int(args.leapfrog_steps),
        "initial_step_size": float(args.step_size),
        "elapsed_sec": float(elapsed),
        **diagnostics,
        "posterior_mean": theta_df.mean().to_dict(),
        "posterior_std": theta_df.std(ddof=1).to_dict(),
    }
    (args.out / f"summary_curve_{args.curve_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[27j] wrote posterior draws for curve {args.curve_id} to {args.out}")


if __name__ == "__main__":
    main()
