"""27l - Small-panel B5 audit with dependency-free HMC vs amortized 27c.

Purpose:
    Turn the current B5 work from isolated curve smokes into a deterministic
    small-panel audit. This script:
      1. runs the dependency-free HMC fallback (`27j`) on a small locked panel
      2. compares those draws against the amortized 27c posterior (`27k` logic)
      3. exports one panel-level summary

Default panel:
    Four claim-test curves with distinct chemistries / groups:
      - 11   DEX-PLGA
      - 17   QRC-PCL
      - 20   PTX-PVL-co-PAVL
      - 34   TAA-PLGA

Outputs:
    outputs/27l_b5_small_panel_audit/
        posterior_draws_curve_<id>.csv
        summary_curve_<id>.json
        per_curve_param_overlap.csv
        per_curve_summary.csv
        overall_summary.json
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
SCRIPT_27K = Path("scripts/27k_b5_hmc_vs_amortized_compare.py")
DEFAULT_OUT = Path("outputs/27l_b5_small_panel_audit")
DEFAULT_CURVE_IDS = [11, 17, 20, 34]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-ids", nargs="*", type=int, default=DEFAULT_CURVE_IDS)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--leapfrog-steps", type=int, default=6)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--target-accept", type=float, default=0.65)
    parser.add_argument("--adapt-rate", type=float, default=0.02)
    parser.add_argument("--n-amortized-samples", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mod27j = load_module(SCRIPT_27J, "script27j_panel")
    mod27k = load_module(SCRIPT_27K, "script27k_panel")
    backend = mod27j.load_backend_module()
    subset = backend.load_subset(mod27j.DEFAULT_SUBSET_DIR)

    # Run HMC fallback on the panel.
    panel_summaries = []
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

        draws_z, diagnostics = mod27j.hmc_sample(
            potential_fn=potential_fn,
            q_init=q_init,
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            leapfrog_steps=args.leapfrog_steps,
            step_size=args.step_size,
            adapt_step_size=True,
            target_accept=args.target_accept,
            adapt_rate=args.adapt_rate,
            seed=args.seed + int(curve_id),
        )
        draws_z_t = torch.tensor(draws_z, dtype=torch.float32)
        theta = backend.box_transform_from_unconstrained(draws_z_t, lows, highs).detach().cpu().numpy()
        theta_df = pd.DataFrame(theta, columns=sim.param_names)
        theta_df.to_csv(args.out / f"posterior_draws_curve_{curve_id}.csv", index=False)

        summary = {
            "curve_id": int(curve_id),
            "DP_Group": row["DP_Group"],
            "num_samples": int(args.num_samples),
            "warmup_steps": int(args.warmup_steps),
            "leapfrog_steps": int(args.leapfrog_steps),
            "initial_step_size": float(args.step_size),
            **diagnostics,
            "posterior_mean": theta_df.mean().to_dict(),
            "posterior_std": theta_df.std(ddof=1).to_dict(),
        }
        (args.out / f"summary_curve_{curve_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        panel_summaries.append(
            {
                "curve_id": int(curve_id),
                "DP_Group": row["DP_Group"],
                "accept_rate_post": float(diagnostics["accept_rate_post"]),
                "final_step_size": float(diagnostics["final_step_size"]),
            }
        )

    # Compare panel HMC draws against amortized 27c posterior.
    sys.argv = [
        "27k_b5_hmc_vs_amortized_compare.py",
        "--device",
        args.device,
        "--n-amortized-samples",
        str(args.n_amortized_samples),
        "--out",
        str(args.out),
        "--hmc-dirs",
        str(args.out),
    ]
    mod27k.main()

    # Enrich overall summary with panel metadata.
    overall_path = args.out / "overall_summary.json"
    overall = json.loads(overall_path.read_text(encoding="utf-8"))
    overall["panel_curve_ids"] = [int(x) for x in args.curve_ids]
    overall["panel_acceptance"] = panel_summaries
    overall["panel_groups"] = {int(x["curve_id"]): x["DP_Group"] for x in panel_summaries}
    overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
