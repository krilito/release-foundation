"""69e - Canonical AO candidate rerun with additional 50% interval reporting.

Purpose:
    The current AO v3 candidate artifact reports 90% and 80% interval metrics,
    which is enough for a partial S4 closeout but still leaves the strict
    `cov50` requirement open. This wrapper reruns the same paper-facing AO
    candidate while preserving the original script untouched, and adds:

      - q25 / q75 posterior summary bands
      - per-curve coverage_50 / width_50
      - aggregate summary file with 90 / 80 / 50 interval metrics

Important scope:
    - 90% / 80% intervals remain the same conformal-calibrated bands as 69
    - the 50% band is the raw weighted central posterior interval (q25-q75);
      it is reported honestly as an additional sharpness / concentration view,
      not as a separately conformal-calibrated band

Outputs:
    outputs_active_observer_v3_beta1_step2nocap_qcorr015_cov50/
        metrics_by_curve.csv            # now includes coverage_50 / width_50
        metrics_summary.csv             # original 69 summary
        metrics_summary_cov50.csv       # new summary with 50% metrics
        summary_cov50.txt
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_69 = Path("scripts/69_active_observer_v3.py")
DEFAULT_OUTPUT_DIR = "outputs_active_observer_v3_beta1_step2nocap_qcorr015_cov50"


def _load_module():
    spec = importlib.util.spec_from_file_location("ao69_cov50_mod", SCRIPT_69)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_69}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_patches(mod) -> None:
    original_weighted_curve_quantiles = mod.weighted_curve_quantiles
    original_rmse = mod.rmse
    original_coverage = mod.coverage
    original_interval_width = mod.interval_width
    original_weighted_resample_indices = mod.weighted_resample_indices
    original_crps = mod.ensemble_crps_unweighted

    def posterior_predictive_summary_cov50(rollout_curves: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
        mean_curve = np.sum(rollout_curves * weights[:, None], axis=0)
        q_arr = original_weighted_curve_quantiles(
            rollout_curves,
            weights,
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        )
        return {
            "mean": mean_curve,
            "q05": q_arr[0],
            "q10": q_arr[1],
            "q25": q_arr[2],
            "q50": q_arr[3],
            "q75": q_arr[4],
            "q90": q_arr[5],
            "q95": q_arr[6],
        }

    def evaluate_prediction_cov50(
        y_true: np.ndarray,
        pred_summary: dict[str, np.ndarray],
        times: np.ndarray,
        cfg,
        posterior_samples: np.ndarray | None = None,
        posterior_weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        mask = times > cfg.future_eval_start
        y = y_true[mask]
        result = {
            "future_rmse_mean": original_rmse(y, pred_summary["mean"][mask]),
            "future_rmse_median": original_rmse(y, pred_summary["q50"][mask]),
            "future_mae_median": float(np.mean(np.abs(y - pred_summary["q50"][mask]))),
            "coverage_90": original_coverage(y, pred_summary["q05"][mask], pred_summary["q95"][mask]),
            "coverage_80": original_coverage(y, pred_summary["q10"][mask], pred_summary["q90"][mask]),
            "coverage_50": original_coverage(y, pred_summary["q25"][mask], pred_summary["q75"][mask]),
            "width_90": original_interval_width(pred_summary["q05"][mask], pred_summary["q95"][mask]),
            "width_80": original_interval_width(pred_summary["q10"][mask], pred_summary["q90"][mask]),
            "width_50": original_interval_width(pred_summary["q25"][mask], pred_summary["q75"][mask]),
        }
        if posterior_samples is not None and posterior_weights is not None:
            if rng is None:
                rng = np.random.default_rng(cfg.random_state)
            idx = original_weighted_resample_indices(posterior_weights, 200, rng)
            result["crps"] = original_crps(posterior_samples[idx][:, mask], y, rng=rng)
        return result

    mod.posterior_predictive_summary = posterior_predictive_summary_cov50
    mod.evaluate_prediction = evaluate_prediction_cov50


def _write_cov50_summary(metrics_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = (
        metrics_df.groupby("strategy")
        .agg(
            n=("curve_id", "count"),
            rmse_mean=("future_rmse_mean", "mean"),
            rmse_median=("future_rmse_mean", "median"),
            mae_median=("future_mae_median", "median"),
            coverage90_mean=("coverage_90", "mean"),
            coverage80_mean=("coverage_80", "mean"),
            coverage50_mean=("coverage_50", "mean"),
            coverage50_median=("coverage_50", "median"),
            width90_mean=("width_90", "mean"),
            width80_mean=("width_80", "mean"),
            width50_mean=("width_50", "mean"),
            width50_median=("width_50", "median"),
            crps_mean=("crps", "mean"),
            ess_median=("ess_after", "median"),
        )
        .reset_index()
        .sort_values("rmse_mean")
    )
    summary.to_csv(out_dir / "metrics_summary_cov50.csv", index=False)
    lines = ["=== 69e -- AO candidate summary with 50% interval reporting ===", ""]
    lines.append("Important note: 50% interval is raw weighted central posterior q25-q75, not separately conformal-calibrated.")
    lines.append("")
    for _, r in summary.iterrows():
        lines.extend(
            [
                f"{r['strategy']}:",
                f"  n              = {int(r['n'])}",
                f"  rmse_mean      = {r['rmse_mean']:.4f}",
                f"  cov90 / cov80  = {r['coverage90_mean']:.3f} / {r['coverage80_mean']:.3f}",
                f"  cov50 mean/med = {r['coverage50_mean']:.3f} / {r['coverage50_median']:.3f}",
                f"  width90 / 80   = {r['width90_mean']:.3f} / {r['width80_mean']:.3f}",
                f"  width50 mean/med = {r['width50_mean']:.3f} / {r['width50_median']:.3f}",
            ]
        )
        if np.isfinite(r["crps_mean"]):
            lines.append(f"  crps_mean      = {r['crps_mean']:.4f}")
        lines.append("")
    (out_dir / "summary_cov50.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mod = _load_module()
    _install_patches(mod)
    cfg = mod.Config(
        output_dir=args.output_dir,
        likelihood_beta=1.0,
        step2_q_max_cap_margin=0.0,
        q_max_correction=0.15,
        conformal_method="local",
        verbose=not args.dry_run,
    )
    if args.dry_run:
        print("[69e] would run AO candidate with cov50 reporting")
        print(cfg)
        return

    metrics_df, _, utility_df, pred_df = mod.evaluate_active_observer_v3(cfg)
    out_dir = Path(cfg.output_dir)
    _write_cov50_summary(metrics_df, out_dir)
    print((out_dir / "summary_cov50.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
