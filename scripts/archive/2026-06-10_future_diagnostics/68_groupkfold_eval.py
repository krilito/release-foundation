"""
GroupKFold evaluation of active observer by polymer family.

Validates that the model generalizes across different polymer types
and there's no information leakage from same-polymer curves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder import parse_dp_group
import importlib.util

# Import from 68_active_kinetic_observer.py (can't use normal import due to numeric prefix)
_spec = importlib.util.spec_from_file_location(
    "active_observer", str(Path(__file__).parent / "68_active_kinetic_observer.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Config = _mod.Config
FormulationThetaPrior = _mod.FormulationThetaPrior
load_data = _mod.load_data
collapse_theta_to_curve_level = _mod.collapse_theta_to_curve_level
simulate_plga_ode = _mod.simulate_plga_ode
interp_curve = _mod.interp_curve
posterior_predictive_summary = _mod.posterior_predictive_summary
evaluate_prediction = _mod.evaluate_prediction
posterior_update_from_one_observation = _mod.posterior_update_from_one_observation
choose_active_timepoint = _mod.choose_active_timepoint
weighted_resample_indices = _mod.weighted_resample_indices
effective_sample_size = _mod.effective_sample_size
run_conformal_calibration = _mod.run_conformal_calibration
ensure_dir = _mod.ensure_dir


def run_groupkfold(cfg: Config, n_splits: int = 5):
    ensure_dir(cfg.output_dir)
    rng = np.random.default_rng(cfg.random_state)

    formulations, curves, theta = load_data(cfg)

    # Add polymer family
    formulations['polymer_family'] = formulations['DP_Group'].apply(
        lambda x: parse_dp_group(x)[1]
    )
    groups = formulations['polymer_family'].values
    unique_groups = np.unique(groups)
    print(f"[groupkfold] polymer families: {unique_groups}")
    print(f"[groupkfold] distribution:\n{pd.Series(groups).value_counts()}")

    # Only use families with enough data
    family_counts = pd.Series(groups).value_counts()
    valid_families = family_counts[family_counts >= 3].index.tolist()
    mask = formulations['polymer_family'].isin(valid_families)
    formulations = formulations[mask].copy()
    curves = curves[curves[cfg.id_col].isin(formulations[cfg.id_col])].copy()
    theta = theta[theta[cfg.id_col].isin(formulations[cfg.id_col])].copy()
    groups = formulations['polymer_family'].values

    print(f"\n[groupkfold] using {len(formulations)} curves from {len(valid_families)} families")

    times = np.asarray(cfg.rollout_times, dtype=float)
    gkf = GroupKFold(n_splits=min(n_splits, len(valid_families)))

    all_records = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(formulations, groups=groups)):
        print(f"\n=== Fold {fold} ===")
        train_ids = set(formulations.iloc[train_idx][cfg.id_col])
        test_ids = set(formulations.iloc[test_idx][cfg.id_col])
        test_families = set(formulations.iloc[test_idx]['polymer_family'])
        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}, Test families: {test_families}")

        form_train = formulations[formulations[cfg.id_col].isin(train_ids)]
        form_test = formulations[formulations[cfg.id_col].isin(test_ids)]
        curves_train = curves[curves[cfg.id_col].isin(train_ids)]
        curves_test = curves[curves[cfg.id_col].isin(test_ids)]
        theta_train = theta[theta[cfg.id_col].isin(train_ids)]

        theta_cl = collapse_theta_to_curve_level(theta_train, cfg)
        prior = FormulationThetaPrior(cfg)
        prior.fit(form_train, theta_cl, curves_train)

        # Conformal calibration
        conformal = run_conformal_calibration(cfg, prior, form_train, curves_train, rng)

        curve_groups = {cid: g for cid, g in curves_test.groupby(cfg.id_col)}

        for _, row in form_test.iterrows():
            cid = row[cfg.id_col]
            if cid not in curve_groups:
                continue

            formulation_row = row.to_frame().T
            true_curve = interp_curve(curve_groups[cid], times, cfg.time_col, cfg.release_col)

            # Prior
            theta_particles = prior.sample_prior_particles(formulation_row, rng=rng)
            rollout = simulate_plga_ode(theta_particles, times)
            weights = np.ones(len(rollout)) / len(rollout)

            # Active one-point
            active_t, _ = choose_active_timepoint(rollout, times, weights, list(cfg.candidate_times), cfg, rng)
            active_q = float(interp_curve(curve_groups[cid], np.array([active_t]), cfg.time_col, cfg.release_col)[0])
            active_w = posterior_update_from_one_observation(rollout, times, weights, active_t, active_q, cfg)
            active_sum = posterior_predictive_summary(rollout, active_w)
            active_met = evaluate_prediction(true_curve, active_sum, times, cfg, rollout, active_w, rng, conformal)
            active_met.update({
                "curve_id": cid, "fold": fold, "strategy": "active_one_point",
                "polymer_family": row['polymer_family'],
                "obs_time": active_t, "ess_after": effective_sample_size(active_w),
            })
            all_records.append(active_met)

            # Active two-point
            remaining = [t for t in cfg.candidate_times if t != active_t]
            if remaining:
                post_idx = weighted_resample_indices(active_w, len(active_w), rng)
                post_rollout = rollout[post_idx]
                post_w = np.ones(len(post_idx)) / len(post_idx)
                active_t2, _ = choose_active_timepoint(post_rollout, times, post_w, remaining, cfg, rng)
                active_q2 = float(interp_curve(curve_groups[cid], np.array([active_t2]), cfg.time_col, cfg.release_col)[0])
                active_w2 = posterior_update_from_one_observation(post_rollout, times, post_w, active_t2, active_q2, cfg)
                active_sum2 = posterior_predictive_summary(post_rollout, active_w2)
                active_met2 = evaluate_prediction(true_curve, active_sum2, times, cfg, post_rollout, active_w2, rng, conformal)
                active_met2.update({
                    "curve_id": cid, "fold": fold, "strategy": "active_two_point",
                    "polymer_family": row['polymer_family'],
                    "obs_time": f"{active_t},{active_t2}",
                    "ess_after": effective_sample_size(active_w2),
                })
                all_records.append(active_met2)

            # Prior only
            prior_sum = posterior_predictive_summary(rollout, weights)
            prior_met = evaluate_prediction(true_curve, prior_sum, times, cfg, rollout, weights, rng, conformal)
            prior_met.update({
                "curve_id": cid, "fold": fold, "strategy": "prior_only",
                "polymer_family": row['polymer_family'],
                "obs_time": np.nan, "ess_after": effective_sample_size(weights),
            })
            all_records.append(prior_met)

    # Save
    df = pd.DataFrame(all_records)
    df.to_csv(f"{cfg.output_dir}/groupkfold_results.csv", index=False)

    summary = df.groupby(["strategy", "polymer_family"]).agg(
        n=("curve_id", "count"),
        rmse_mean=("future_rmse_mean", "mean"),
        coverage90_conformal=("coverage_90_conformal", "mean"),
    ).reset_index()
    print("\n=== GroupKFold Summary ===")
    print(summary.to_string(index=False))

    overall = df.groupby("strategy").agg(
        n=("curve_id", "count"),
        rmse_mean=("future_rmse_mean", "mean"),
        rmse_std=("future_rmse_mean", "std"),
        coverage90_conformal=("coverage_90_conformal", "mean"),
    ).reset_index()
    print("\n=== Overall ===")
    print(overall.to_string(index=False))

    return df


if __name__ == "__main__":
    run_groupkfold(Config())
