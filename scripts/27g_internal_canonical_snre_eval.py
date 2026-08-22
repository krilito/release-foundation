"""27g - Internal canonical same-mouth SNRE comparator for B4.

Purpose:
    B4 requires an SBI gold-standard comparator on the same canonical mouth as
    `72_canonical_benchmark_v2.py`. `27e` already gives same-mouth NPE reads
    for `27c/27d`; this script adds the missing SNRE route.

Training mouth:
    - synthetic-only theta ~ prior
    - conditioning input = concat(Q_masked, mask) on the 64-point PLGA grid
    - continuous prefix sampling over [0.5, 14.0] days, matching 27c

Evaluation mouth:
    - fixed canonical split from `data/canonical_split_v1.csv`
    - conditioning input = the actual internal curve truncated at 7d
    - target = future trajectory for times > 14d on the 14-point internal grid
    - point prediction = simulator rollout from posterior-median theta

Outputs:
    outputs/27g_internal_canonical_snre_eval/summary.csv
    outputs/27g_internal_canonical_snre_eval/per_curve_metrics.csv
    outputs/27g_internal_canonical_snre_eval/pairwise_vs_72.csv
    outputs/27g_internal_canonical_snre_eval/summary.txt
    outputs/27g_internal_canonical_snre_eval/lock_metadata.json

Notes:
    - This is the first honest same-mouth SNRE comparator in the repo.
    - For tractability, the first run can use `--quick`; that should be
      interpreted as a directional readout, not a final submission number.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sbi.inference import SNRE
from sbi.neural_nets import classifier_nn
from sbi.neural_nets.embedding_nets import FCEmbedding


SCRIPT_27C = Path("scripts/27c_sbi_npe_partial.py")
SCRIPT_27E = Path("scripts/27e_internal_canonical_npe_fixed_eval.py")

DEFAULT_OUT = Path("outputs/27g_internal_canonical_snre_eval")
CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")
DEFAULT_BENCH72 = Path("outputs/72_canonical_benchmark_v2")
FUTURE_START = 14.0
PREFIX_DAYS = 7.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_curve_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def bootstrap_paired_ci(rmse_a: np.ndarray, rmse_b: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(rmse_a)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        mse_a = (rmse_a[idx] ** 2).mean()
        mse_b = (rmse_b[idx] ** 2).mean()
        deltas.append(np.sqrt(mse_a) - np.sqrt(mse_b))
    deltas = np.asarray(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sim", type=int, default=None)
    parser.add_argument("--n-posterior-samples", type=int, default=64)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--prefix-low", type=float, default=0.5)
    parser.add_argument("--prefix-high", type=float, default=14.0)
    parser.add_argument("--sample-with", choices=("mcmc", "rejection", "vi", "importance"), default="importance")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_sim = 4000
        args.n_posterior_samples = 48

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mod27c = load_module(SCRIPT_27C, "script27c")
    mod27e = load_module(SCRIPT_27E, "script27e")

    overall_t0 = time.time()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n_sim = args.n_sim or cfg["synthetic"]["n_pairs"]
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    noise_sigma = cfg["posterior"].get("noise_sigma", 0.03)

    sim = mod27c.PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    prior_base = sim.prior().base_dist
    prior = torch.distributions.Independent(
        torch.distributions.Uniform(
            prior_base.low.to(args.device),
            prior_base.high.to(args.device),
        ),
        1,
    )

    print(f"[27g] device                : {args.device}")
    print(f"[27g] n_simulations         : {n_sim}")
    print(f"[27g] n_posterior_samples   : {args.n_posterior_samples}")
    print(f"[27g] sample_with           : {args.sample_with}")
    print(f"[27g] training prefix range : [{args.prefix_low}, {args.prefix_high}]")

    embedding_x = FCEmbedding(
        input_dim=2 * t_grid.numel(),
        output_dim=cfg["posterior"].get("embedding_output_dim", 16),
        num_layers=cfg["posterior"].get("embedding_layers", 3),
        num_hiddens=cfg["posterior"].get("embedding_hidden", 64),
    )
    classifier_builder = classifier_nn(
        model="resnet",
        hidden_features=cfg["posterior"]["hidden_features"],
        embedding_net_x=embedding_x,
    )
    inference = SNRE(prior=prior, classifier=classifier_builder, device=args.device, show_progress_bars=True)

    print(f"[27g] generating {n_sim} synthetic pairs...")
    t_gen = time.time()
    theta_train, x_train = mod27c._generate_synthetic_pairs(
        sim=sim,
        t_grid=t_grid,
        n=n_sim,
        noise_sigma=noise_sigma,
        prefix_low=args.prefix_low,
        prefix_high=args.prefix_high,
        seed=args.seed,
        device=args.device,
    )
    print(f"[27g] synthetic generation done in {time.time()-t_gen:.1f}s")

    inference.append_simulations(theta_train.to(args.device), x_train.to(args.device))
    print("[27g] training SNRE...")
    t_train = time.time()
    if args.quick:
        max_epochs = 20
        stop_after = 8
    else:
        max_epochs = cfg["train"]["max_epochs"]
        stop_after = cfg["train"]["early_stopping_patience"]
    ratio_estimator = inference.train(
        training_batch_size=cfg["train"]["batch_size"],
        max_num_epochs=max_epochs,
        validation_fraction=0.1,
        stop_after_epochs=stop_after,
        show_train_summary=True,
    )
    train_elapsed = time.time() - t_train
    print(f"[27g] training done in {train_elapsed/60:.1f} min")

    posterior = inference.build_posterior(ratio_estimator, sample_with=args.sample_with)

    split_df = mod27e.load_canonical_split()
    curve_matrix, time_grid_internal, split_ids, curves_long = mod27e.load_internal_curve_matrix(split_df)
    future_mask = time_grid_internal > FUTURE_START
    test_mask = split_df["split_name"].to_numpy() == "test"
    test_ids = [int(cid) for cid in split_df.loc[test_mask, "curve_id"].tolist()]
    curve_id_to_index = {int(cid): i for i, cid in enumerate(split_ids)}

    mask = mod27c._make_prefix_mask(PREFIX_DAYS, t_grid).to(args.device)
    y_true = np.zeros((len(test_ids), int(future_mask.sum())), dtype=float)
    y_pred = np.zeros_like(y_true)

    print(f"[27g] evaluating {len(test_ids)} canonical test curves at prefix={PREFIX_DAYS}d...")
    for i, cid in enumerate(test_ids):
        idx = curve_id_to_index[cid]
        cdf = curves_long[curves_long["curve_id"] == cid].sort_values("time")
        t_obs = cdf["time"].to_numpy(dtype=float)
        q_obs = cdf["release"].to_numpy(dtype=float)
        q_interp64 = np.interp(
            t_grid.cpu().numpy(),
            t_obs,
            q_obs,
            left=q_obs[0],
            right=q_obs[-1],
        )
        q_full = torch.tensor(q_interp64, dtype=torch.float32)
        q_prefix = q_full * mask.cpu()
        x_obs = torch.cat([q_prefix, mask.cpu()], dim=-1).to(args.device)

        theta_samples = posterior.sample(
            (args.n_posterior_samples,),
            x=x_obs,
            show_progress_bars=False,
        ).detach().cpu()
        theta_median = theta_samples.median(dim=0).values.numpy()

        curve_pred = sim.simulate_numpy(theta_median, time_grid_internal)
        y_true[i] = curve_matrix[idx][future_mask]
        y_pred[i] = curve_pred[future_mask]

    rmse_pc = per_curve_rmse(y_true, y_pred)
    summary_df = pd.DataFrame(
        [
            {
                "method": "SNRE-fixed7d",
                "n_test": int(len(test_ids)),
                "prefix_days": PREFIX_DAYS,
                "n_posterior_samples": int(args.n_posterior_samples),
                "rmse": pooled_rmse(y_true, y_pred),
                "r2_pooled": pooled_r2(y_true, y_pred),
                "per_curve_rmse_mean": float(rmse_pc.mean()),
                "per_curve_rmse_median": float(np.median(rmse_pc)),
            }
        ]
    )
    summary_df.to_csv(args.out / "summary.csv", index=False)

    per_curve_df = pd.DataFrame(
        {
            "curve_id": test_ids,
            "method": "SNRE-fixed7d",
            "rmse": rmse_pc,
            "obs_times_used": ["1,3,5,7"] * len(test_ids),
        }
    )
    per_curve_df.to_csv(args.out / "per_curve_metrics.csv", index=False)

    bench72_path = DEFAULT_BENCH72 / "per_curve_results.csv"
    if not bench72_path.exists():
        raise FileNotFoundError(f"Expected {bench72_path} for pairwise comparison")
    bench72_df = pd.read_csv(bench72_path)
    pairwise_rows: list[dict[str, object]] = []
    for method, g in bench72_df.groupby("method"):
        g = g.assign(curve_id=g["curve_id"].astype(int)).set_index("curve_id").loc[test_ids]
        ref_rmse = g["rmse"].to_numpy(dtype=float)
        delta_mean, ci_low, ci_high = bootstrap_paired_ci(rmse_pc, ref_rmse, args.n_bootstrap, args.seed)
        pairwise_rows.append(
            {
                "baseline_method": "SNRE-fixed7d",
                "reference_method": method,
                "delta_rmse_baseline_minus_reference": float(delta_mean),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "baseline_better": bool(ci_high < 0.0),
                "reference_better": bool(ci_low > 0.0),
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(args.out / "pairwise_vs_72.csv", index=False)

    metadata = {
        "script": "scripts/27g_internal_canonical_snre_eval.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "git_hash": get_git_hash(),
        "seed": int(args.seed),
        "device": args.device,
        "n_sim": int(n_sim),
        "n_posterior_samples": int(args.n_posterior_samples),
        "sample_with": args.sample_with,
        "prefix_days_eval": float(PREFIX_DAYS),
        "canonical_split_path": str(CANONICAL_SPLIT_PATH),
        "future_eval_times": time_grid_internal[future_mask].tolist(),
        "train_wallclock_min": float(train_elapsed / 60.0),
        "total_wallclock_min": float((time.time() - overall_t0) / 60.0),
        "notes": [
            "Training mouth matches 27c: synthetic-only, continuous prefix sampling, concat(Q_masked, mask).",
            "Evaluation mouth matches 27e/72: internal canonical split, fixed 7d prefix, future times >14d.",
            "Quick mode is a directional same-mouth comparator, not a final submission number.",
        ],
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "=== 27g -- internal canonical SNRE comparator ===",
        "",
        f"device                : {args.device}",
        f"n_simulations         : {n_sim}",
        f"n_posterior_samples   : {args.n_posterior_samples}",
        f"sample_with           : {args.sample_with}",
        f"prefix_days           : {PREFIX_DAYS}",
        f"train wallclock       : {train_elapsed/60:.1f} min",
        f"total wallclock       : {(time.time()-overall_t0)/60:.1f} min",
        "",
        "Summary:",
        summary_df.to_string(index=False),
        "",
        "Pairwise vs 72:",
        pairwise_df.to_string(index=False),
    ]
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
