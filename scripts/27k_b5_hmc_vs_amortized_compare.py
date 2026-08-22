"""27k - Compare B5 HMC fallback draws against amortized 27c posterior.

Purpose:
    The first B5 acceptance target is not just "did HMC run", but whether the
    ground-truth posterior and the amortized posterior agree on the same curve
    and the same mouth. This script compares saved HMC draws from `27j` (or
    any compatible draw CSV) against the 27c amortized posterior on the locked
    internal canonical 7d mouth.

Outputs:
    outputs/27k_b5_hmc_vs_amortized_compare/
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


SCRIPT_27E = Path("scripts/27e_internal_canonical_npe_fixed_eval.py")
DEFAULT_POSTERIOR_PT = Path("outputs/27c_sbi_npe_partial/posterior.pt")
DEFAULT_OUT = Path("outputs/27k_b5_hmc_vs_amortized_compare")
DEFAULT_HMC_DIRS = [
    Path("outputs/27j_b5_hmc_torch_runner_smoke"),
    Path("outputs/27j_b5_hmc_torch_runner_smoke_curve20"),
]


def load_27e_module():
    spec = importlib.util.spec_from_file_location("script27e_compare", SCRIPT_27E)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_27E}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def interval_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q05": np.quantile(samples, 0.05, axis=0),
        "q50": np.quantile(samples, 0.50, axis=0),
        "q95": np.quantile(samples, 0.95, axis=0),
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros(samples.shape[1]),
    }


def overlap_fraction(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    union = max(hi_a, hi_b) - min(lo_a, lo_b)
    if union <= 1e-12:
        return 1.0
    return inter / union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-pt", type=Path, default=DEFAULT_POSTERIOR_PT)
    parser.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-amortized-samples", type=int, default=400)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hmc-dirs", nargs="*", type=Path, default=DEFAULT_HMC_DIRS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mod27e = load_27e_module()
    sim, posterior_obj, cond_grid = mod27e.build_partial_npe_posterior(
        args.posterior_pt,
        args.config,
        args.device,
        "27c",
        residual_n_modes=5,
        residual_sigma_c=0.015,
    )
    split_df = mod27e.load_canonical_split()
    _, _, _, curves_df = mod27e.load_internal_curve_matrix(split_df)

    hmc_files: list[Path] = []
    for hdir in args.hmc_dirs:
        if hdir.exists():
            hmc_files.extend(sorted(hdir.glob("posterior_draws_curve_*.csv")))
    if not hmc_files:
        raise FileNotFoundError("No HMC draw CSVs found in the provided hmc dirs")

    param_names = list(sim.param_names)
    overlap_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for draw_path in hmc_files:
        curve_id = int(draw_path.stem.split("_")[-1])
        hmc_df = pd.read_csv(draw_path)
        hmc_samples = hmc_df[param_names].to_numpy(dtype=float)

        x_obs = mod27e.build_obs_vector(curves_df, curve_id, cond_grid).to(args.device)
        amortized_samples = posterior_obj.sample(
            (args.n_amortized_samples,),
            x=x_obs,
            show_progress_bars=False,
        ).cpu().numpy()

        hmc_stats = interval_stats(hmc_samples)
        amp_stats = interval_stats(amortized_samples)

        overlap_vals = []
        mean_deltas = []
        median_deltas = []
        for j, name in enumerate(param_names):
            ov = overlap_fraction(
                float(hmc_stats["q05"][j]),
                float(hmc_stats["q95"][j]),
                float(amp_stats["q05"][j]),
                float(amp_stats["q95"][j]),
            )
            overlap_vals.append(ov)
            mean_delta = float(hmc_stats["mean"][j] - amp_stats["mean"][j])
            median_delta = float(hmc_stats["q50"][j] - amp_stats["q50"][j])
            mean_deltas.append(abs(mean_delta))
            median_deltas.append(abs(median_delta))
            overlap_rows.append(
                {
                    "curve_id": curve_id,
                    "param": name,
                    "hmc_q05": float(hmc_stats["q05"][j]),
                    "hmc_q50": float(hmc_stats["q50"][j]),
                    "hmc_q95": float(hmc_stats["q95"][j]),
                    "amortized_q05": float(amp_stats["q05"][j]),
                    "amortized_q50": float(amp_stats["q50"][j]),
                    "amortized_q95": float(amp_stats["q95"][j]),
                    "interval_overlap_fraction": float(ov),
                    "mean_delta": mean_delta,
                    "median_delta": median_delta,
                }
            )

        curve_rows.append(
            {
                "curve_id": curve_id,
                "n_hmc_draws": int(len(hmc_samples)),
                "n_amortized_draws": int(len(amortized_samples)),
                "mean_interval_overlap_fraction": float(np.mean(overlap_vals)),
                "min_interval_overlap_fraction": float(np.min(overlap_vals)),
                "max_abs_mean_delta": float(np.max(mean_deltas)),
                "max_abs_median_delta": float(np.max(median_deltas)),
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)
    curve_df = pd.DataFrame(curve_rows).sort_values("curve_id")
    overlap_df.to_csv(args.out / "per_curve_param_overlap.csv", index=False)
    curve_df.to_csv(args.out / "per_curve_summary.csv", index=False)

    overall = {
        "posterior_pt": str(args.posterior_pt),
        "n_curves_compared": int(len(curve_df)),
        "curve_ids": curve_df["curve_id"].tolist(),
        "mean_of_mean_interval_overlap_fraction": float(curve_df["mean_interval_overlap_fraction"].mean()),
        "min_curve_mean_interval_overlap_fraction": float(curve_df["mean_interval_overlap_fraction"].min()),
        "worst_param_overlap_fraction": float(overlap_df["interval_overlap_fraction"].min()),
    }
    (args.out / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print(curve_df.to_string(index=False))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
