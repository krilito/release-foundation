"""69b - Empirical-particle SBC for Active Observer v3 on the canonical split.

Purpose:
    S6 requires Simulation-Based Calibration (SBC) on the actual Active
    Observer posterior, not only on the older curve-posterior scripts. The
    Active Observer prior is not an analytic distribution; it is an empirical
    formulation-conditioned particle prior built from ExtraTrees + KNN.

    This script therefore runs an honest approximation to SBC for the claimed
    Active Observer posterior:

      1. Fit the formulation-conditioned particle prior on the canonical
         TRAIN split only.
      2. Draw held-out canonical TEST formulation rows x.
      3. Build the empirical prior particle cloud p_hat(theta | x).
      4. Sample theta* uniformly from that particle cloud.
      5. Simulate noisy observations from theta* under the same observation
         model used by Active Observer.
      6. Run the actual Active Observer posterior update mechanism
         (active-two-point by default, or fixed-four-point).
      7. Check rank uniformity of theta* against posterior samples.

Important limitations:
    - This is not analytic-prior SBC in the strict Talts et al. sense because
      the claimed prior is itself empirical and formulation-conditioned.
    - Conformal adjustment is intentionally excluded here. Conformal calibrates
      predictive intervals, not the latent theta posterior.
    - Interpret the result as calibration of the actual particle posterior
      mechanism that the paper would claim, on held-out canonical formulations.

Outputs:
    outputs/69b_active_observer_v3_sbc/rank_histograms.png
    outputs/69b_active_observer_v3_sbc/rank_ecdf.png
    outputs/69b_active_observer_v3_sbc/sbc_summary.csv
    outputs/69b_active_observer_v3_sbc/per_replicate.csv
    outputs/69b_active_observer_v3_sbc/summary.txt
    outputs/69b_active_observer_v3_sbc/lock_metadata.json

Run:
    python scripts/69b_active_observer_v3_sbc.py --device-note cpu
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")
ACTIVE_OBSERVER_SCRIPT = Path("scripts/69_active_observer_v3.py")
DEFAULT_OUT = Path("outputs/69b_active_observer_v3_sbc")
FIXED_FOUR_TIMES = (1.0, 3.0, 5.0, 7.0)


def load_active_observer_module():
    spec = importlib.util.spec_from_file_location("active_observer_v3", ACTIVE_OBSERVER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {ACTIVE_OBSERVER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_canonical_split(path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(path)
    required = {"curve_id", "split_name", "fold_index"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Canonical split missing columns: {sorted(missing)}")
    split_df = split_df.copy()
    split_df["curve_id"] = split_df["curve_id"].astype(int)
    split_df["split_name"] = split_df["split_name"].astype(str)
    split_df["fold_index"] = split_df["fold_index"].astype(int)
    return split_df.sort_values("fold_index").reset_index(drop=True)


def sample_noisy_observation(true_q: float, cfg, rng: np.random.Generator) -> float:
    sigma = max(cfg.obs_sigma_abs, cfg.obs_sigma_rel * max(abs(true_q), 0.05))
    return float(np.clip(true_q + rng.normal(0.0, sigma), 0.0, 1.1))


def draw_posterior_samples(theta_particles: np.ndarray, weights: np.ndarray, n_samples: int, rng: np.random.Generator, ao) -> np.ndarray:
    idx = ao.weighted_resample_indices(weights, n_samples, rng)
    return theta_particles[idx]


def randomized_rank(true_value: float, posterior_samples: np.ndarray, rng: np.random.Generator) -> int:
    less = int(np.sum(posterior_samples < true_value))
    equal = int(np.sum(np.isclose(posterior_samples, true_value, atol=1e-12, rtol=0.0)))
    if equal <= 0:
        return less
    return less + int(rng.integers(0, equal + 1))


def run_strategy_active_two_point(theta_particles: np.ndarray, times: np.ndarray, true_curve: np.ndarray, cfg, rng: np.random.Generator, ao):
    rollout_curves = ao.simulate_plga_ode(theta_particles, times)
    weights = np.ones(len(theta_particles)) / len(theta_particles)

    active_t, _ = ao.choose_active_timepoint(
        rollout_curves, times, weights, list(cfg.candidate_times), cfg, rng,
    )
    obs_idx_1 = int(np.argmin(np.abs(times - active_t)))
    active_q = sample_noisy_observation(float(true_curve[obs_idx_1]), cfg, rng)
    weights_1, theta_1 = ao.posterior_update_from_observation(
        rollout_curves, times, weights, active_t, active_q, cfg, theta_particles, rng,
    )
    rollout_1 = ao.simulate_plga_ode(theta_1, times)

    step2_candidates = [t for t in cfg.candidate_times_step2 if t > active_t]
    if not step2_candidates:
        return theta_1, weights_1, {
            "obs_time_1": float(active_t),
            "obs_q_1": float(active_q),
            "obs_time_2": np.nan,
            "obs_q_2": np.nan,
            "ess_after": float(ao.effective_sample_size(weights_1)),
        }

    active2_t, _ = ao.choose_active_timepoint(
        rollout_1, times, weights_1, step2_candidates, cfg, rng,
    )
    obs_idx_2 = int(np.argmin(np.abs(times - active2_t)))
    active2_q = sample_noisy_observation(float(true_curve[obs_idx_2]), cfg, rng)
    cfg_step2 = copy.deepcopy(cfg)
    if hasattr(cfg, "_step2_q_max_cap_margin_override"):
        cfg_step2.q_max_cap_margin = float(cfg._step2_q_max_cap_margin_override)
    weights_2, theta_2 = ao.posterior_update_from_observation(
        rollout_1, times, weights_1, active2_t, active2_q, cfg_step2, theta_1, rng,
    )
    return theta_2, weights_2, {
        "obs_time_1": float(active_t),
        "obs_q_1": float(active_q),
        "obs_time_2": float(active2_t),
        "obs_q_2": float(active2_q),
        "ess_after": float(ao.effective_sample_size(weights_2)),
    }


def run_strategy_active_one_point(theta_particles: np.ndarray, times: np.ndarray, true_curve: np.ndarray, cfg, rng: np.random.Generator, ao):
    rollout_curves = ao.simulate_plga_ode(theta_particles, times)
    weights = np.ones(len(theta_particles)) / len(theta_particles)
    active_t, _ = ao.choose_active_timepoint(
        rollout_curves, times, weights, list(cfg.candidate_times), cfg, rng,
    )
    obs_idx = int(np.argmin(np.abs(times - active_t)))
    active_q = sample_noisy_observation(float(true_curve[obs_idx]), cfg, rng)
    weights_1, theta_1 = ao.posterior_update_from_observation(
        rollout_curves, times, weights, active_t, active_q, cfg, theta_particles, rng,
    )
    return theta_1, weights_1, {
        "obs_time_1": float(active_t),
        "obs_q_1": float(active_q),
        "obs_time_2": np.nan,
        "obs_q_2": np.nan,
        "ess_after": float(ao.effective_sample_size(weights_1)),
    }


def run_strategy_prior_only(theta_particles: np.ndarray, cfg, ao):
    weights = np.ones(len(theta_particles)) / len(theta_particles)
    return theta_particles, weights, {
        "obs_time_1": np.nan,
        "obs_q_1": np.nan,
        "obs_time_2": np.nan,
        "obs_q_2": np.nan,
        "ess_after": float(ao.effective_sample_size(weights)),
    }


def run_strategy_fixed_four(theta_particles: np.ndarray, times: np.ndarray, true_curve: np.ndarray, cfg, rng: np.random.Generator, ao):
    theta_curr = theta_particles.copy()
    rollout_curr = ao.simulate_plga_ode(theta_curr, times)
    weights_curr = np.ones(len(theta_curr)) / len(theta_curr)
    obs_values = []

    for obs_t in FIXED_FOUR_TIMES:
        obs_idx = int(np.argmin(np.abs(times - obs_t)))
        obs_q = sample_noisy_observation(float(true_curve[obs_idx]), cfg, rng)
        obs_values.append((float(obs_t), float(obs_q)))
        weights_curr, theta_curr = ao.posterior_update_from_observation(
            rollout_curr, times, weights_curr, obs_t, obs_q, cfg, theta_curr, rng,
        )
        rollout_curr = ao.simulate_plga_ode(theta_curr, times)

    return theta_curr, weights_curr, {
        "obs_time_1": obs_values[0][0],
        "obs_q_1": obs_values[0][1],
        "obs_time_2": obs_values[-1][0],
        "obs_q_2": obs_values[-1][1],
        "ess_after": float(ao.effective_sample_size(weights_curr)),
    }


def plot_rank_histograms(ranks: np.ndarray, param_names: list[str], n_posterior_samples: int, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
    axes = axes.ravel()
    expected = ranks.shape[0] / 20.0
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(ranks[:, i], bins=20, range=(0, n_posterior_samples), edgecolor="black", alpha=0.85)
        ax.axhline(expected, color="red", linestyle="--", linewidth=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("rank")
        if i % 3 == 0:
            ax.set_ylabel("count")
    for i in range(len(param_names), len(axes)):
        axes[i].axis("off")
    fig.suptitle("69b Active Observer posterior SBC rank histograms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_ecdf(ranks: np.ndarray, param_names: list[str], n_posterior_samples: int, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    uniform_x = np.linspace(0.0, 1.0, 200)
    for i, name in enumerate(param_names):
        ax = axes[i]
        u = (ranks[:, i] + 1.0) / (n_posterior_samples + 1.0)
        u_sorted = np.sort(u)
        y = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
        ax.plot(u_sorted, y, label="empirical")
        ax.plot(uniform_x, uniform_x, linestyle="--", color="red", label="uniform")
        ax.set_title(name, fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("ECDF")
        if i >= 6:
            ax.set_xlabel("normalized rank")
    for i in range(len(param_names), len(axes)):
        axes[i].axis("off")
    fig.suptitle("69b Active Observer posterior SBC ECDF")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canonical-split", type=Path, default=CANONICAL_SPLIT_PATH)
    parser.add_argument("--n-sbc", type=int, default=200)
    parser.add_argument("--n-posterior-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=("prior_only", "active_one_point", "active_two_point", "fixed_four_point"), default="active_two_point")
    parser.add_argument("--n-tree-particles", type=int, default=None)
    parser.add_argument("--n-knn-particles", type=int, default=None)
    parser.add_argument("--likelihood-beta", type=float, default=None)
    parser.add_argument("--q-max-correction", type=float, default=None)
    parser.add_argument("--q-max-cap-margin", type=float, default=None)
    parser.add_argument("--step2-q-max-cap-margin", type=float, default=None)
    parser.add_argument("--rejuvenation-ess-threshold", type=float, default=None)
    parser.add_argument("--device-note", type=str, default="cpu", help="Metadata only; script is numpy/scipy based.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    ao = load_active_observer_module()
    cfg = ao.Config(verbose=False)
    if args.n_tree_particles is not None:
        cfg.n_tree_particles = int(args.n_tree_particles)
    if args.n_knn_particles is not None:
        cfg.n_knn_particles = int(args.n_knn_particles)
    if args.likelihood_beta is not None:
        cfg.likelihood_beta = float(args.likelihood_beta)
    if args.q_max_correction is not None:
        cfg.q_max_correction = float(args.q_max_correction)
    if args.q_max_cap_margin is not None:
        cfg.q_max_cap_margin = float(args.q_max_cap_margin)
    if args.step2_q_max_cap_margin is not None:
        cfg._step2_q_max_cap_margin_override = float(args.step2_q_max_cap_margin)
    if args.rejuvenation_ess_threshold is not None:
        cfg.rejuvenation_ess_threshold = float(args.rejuvenation_ess_threshold)
    times = np.asarray(cfg.rollout_times, dtype=float)

    split_df = load_canonical_split(args.canonical_split)
    train_ids = set(split_df.loc[split_df["split_name"] == "train", "curve_id"].astype(int))
    test_ids = split_df.loc[split_df["split_name"] == "test", "curve_id"].astype(int).tolist()

    formulations, curves, theta = ao.load_data(cfg)
    theta_curve_level = ao.collapse_theta_to_curve_level(theta, cfg)
    formulations_train = formulations[formulations[cfg.id_col].isin(train_ids)].copy()
    theta_train = theta_curve_level[theta_curve_level[cfg.id_col].isin(train_ids)].copy()
    formulations_test = (
        formulations[formulations[cfg.id_col].isin(test_ids)]
        .set_index(cfg.id_col)
        .loc[test_ids]
        .reset_index()
    )

    prior = ao.FormulationThetaPrior(cfg).fit(formulations_train, theta_train)

    param_names = list(cfg.theta_cols)
    ranks = np.zeros((args.n_sbc, len(param_names)), dtype=int)
    replicate_rows = []

    for rep in range(args.n_sbc):
        formulation_row = formulations_test.iloc[[int(rng.integers(0, len(formulations_test)))]]  # keep DataFrame shape
        curve_id = int(formulation_row.iloc[0][cfg.id_col])
        theta_particles = prior.sample_prior_particles(formulation_row=formulation_row, rng=rng)
        true_idx = int(rng.integers(0, len(theta_particles)))
        true_theta = theta_particles[true_idx].copy()
        true_curve = ao.simulate_plga_ode(true_theta[None, :], times)[0]

        if args.strategy == "active_two_point":
            posterior_particles, posterior_weights, meta = run_strategy_active_two_point(
                theta_particles, times, true_curve, cfg, rng, ao,
            )
        elif args.strategy == "prior_only":
            posterior_particles, posterior_weights, meta = run_strategy_prior_only(
                theta_particles, cfg, ao,
            )
        elif args.strategy == "active_one_point":
            posterior_particles, posterior_weights, meta = run_strategy_active_one_point(
                theta_particles, times, true_curve, cfg, rng, ao,
            )
        else:
            posterior_particles, posterior_weights, meta = run_strategy_fixed_four(
                theta_particles, times, true_curve, cfg, rng, ao,
            )

        posterior_samples = draw_posterior_samples(
            posterior_particles, posterior_weights, args.n_posterior_samples, rng, ao,
        )

        for dim, name in enumerate(param_names):
            ranks[rep, dim] = randomized_rank(float(true_theta[dim]), posterior_samples[:, dim], rng)

        qmax_dim = param_names.index("Q_max")
        qmax_rank = int(ranks[rep, qmax_dim])
        qmax_norm_rank = float((qmax_rank + 1.0) / (args.n_posterior_samples + 1.0))
        replicate_rows.append({
            "replicate": rep,
            "curve_id": curve_id,
            "strategy": args.strategy,
            "obs_time_1": meta["obs_time_1"],
            "obs_q_1": meta["obs_q_1"],
            "obs_time_2": meta["obs_time_2"],
            "obs_q_2": meta["obs_q_2"],
            "ess_after": meta["ess_after"],
            "true_Q_max": float(true_theta[qmax_dim]),
            "posterior_Q_max_mean": float(posterior_samples[:, qmax_dim].mean()),
            "posterior_Q_max_median": float(np.median(posterior_samples[:, qmax_dim])),
            "qmax_rank": qmax_rank,
            "qmax_norm_rank": qmax_norm_rank,
        })

    replicate_df = pd.DataFrame(replicate_rows)
    replicate_df.to_csv(args.out / "per_replicate.csv", index=False)

    summary_rows = []
    for dim, name in enumerate(param_names):
        u = (ranks[:, dim] + 1.0) / (args.n_posterior_samples + 1.0)
        ks = kstest(u, "uniform")
        summary_rows.append({
            "param": name,
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "mean_normalized_rank": float(u.mean()),
            "std_normalized_rank": float(u.std(ddof=0)),
            "pass_p_gt_0p05": bool(ks.pvalue > 0.05),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out / "sbc_summary.csv", index=False)

    plot_rank_histograms(ranks, param_names, args.n_posterior_samples, args.out / "rank_histograms.png")
    plot_rank_ecdf(ranks, param_names, args.n_posterior_samples, args.out / "rank_ecdf.png")

    n_pass = int(summary_df["pass_p_gt_0p05"].sum())
    summary_lines = [
        "=== 69b Active Observer posterior SBC ===",
        f"strategy             : {args.strategy}",
        f"canonical_split      : {args.canonical_split}",
        f"device_note          : {args.device_note}",
        f"n_sbc                : {args.n_sbc}",
        f"n_posterior_samples  : {args.n_posterior_samples}",
        f"seed                 : {args.seed}",
        f"n_tree_particles     : {cfg.n_tree_particles}",
        f"n_knn_particles      : {cfg.n_knn_particles}",
        f"likelihood_beta      : {cfg.likelihood_beta}",
        f"q_max_correction     : {cfg.q_max_correction}",
        f"q_max_cap_margin     : {cfg.q_max_cap_margin}",
        f"step2_qmax_cap_margin: {getattr(cfg, '_step2_q_max_cap_margin_override', cfg.q_max_cap_margin)}",
        f"rejuvenation_ess_thr : {cfg.rejuvenation_ess_threshold}",
        f"train_curves         : {len(train_ids)}",
        f"test_formulations    : {len(test_ids)}",
        f"passed_dims_p_gt_0p05: {n_pass} / {len(param_names)}",
        "",
        "Per-parameter KS summary:",
        summary_df.to_string(index=False),
    ]
    (args.out / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    metadata = {
        "script": "scripts/69b_active_observer_v3_sbc.py",
        "strategy": args.strategy,
        "canonical_split": str(args.canonical_split),
        "n_sbc": args.n_sbc,
        "n_posterior_samples": args.n_posterior_samples,
        "seed": args.seed,
        "n_tree_particles": cfg.n_tree_particles,
        "n_knn_particles": cfg.n_knn_particles,
        "likelihood_beta": cfg.likelihood_beta,
        "q_max_correction": cfg.q_max_correction,
        "q_max_cap_margin": cfg.q_max_cap_margin,
        "step2_q_max_cap_margin": getattr(cfg, "_step2_q_max_cap_margin_override", cfg.q_max_cap_margin),
        "rejuvenation_ess_threshold": cfg.rejuvenation_ess_threshold,
        "train_curves": len(train_ids),
        "test_formulations": len(test_ids),
        "device_note": args.device_note,
        "method_note": "empirical-particle SBC on Active Observer posterior; conformal excluded",
    }
    with open(args.out / "lock_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[69b] wrote {args.out / 'sbc_summary.csv'}")
    print(f"[69b] wrote {args.out / 'rank_histograms.png'}")
    print(f"[69b] wrote {args.out / 'rank_ecdf.png'}")


if __name__ == "__main__":
    main()
