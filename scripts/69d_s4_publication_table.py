"""69d - Publication-facing S4 coverage/sharpness table on the canonical split.

Purpose:
    Convert the current Active Observer paper-candidate output into a single
    submission-facing table that reports coverage + sharpness on the canonical
    test fold, and anchors those numbers against a naive empirical train-band
    baseline as required by S4.

Current scope:
    - uses an already-run canonical AO candidate directory
    - aggregates per-curve coverage / width into mean + median summaries
    - adds a naive "always use training-band quantiles" comparator on the same
      future evaluation mouth used by script 69 (`times > future_eval_start`)
    - if the AO artifact includes `coverage_50` / `width_50`, the same table
      will extend the naive comparison to the 50% central band as well

Outputs:
    outputs/69d_s4_publication_table/
        summary.csv
        summary.txt
        naive_train_band_per_curve.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")
FORMULATIONS_PATH = Path("data/formulations.csv")
CURVES_PATH = Path("data/curves_long.csv")
DEFAULT_ACTIVE_DIR = Path("outputs_active_observer_v3_beta1_step2nocap_qcorr015")
DEFAULT_OUT = Path("outputs/69d_s4_publication_table")


def _load_curve_matrix(split_ids: list[int], rollout_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = pd.read_csv(CURVES_PATH)
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    matrix = np.zeros((len(split_ids), len(rollout_times)), dtype=float)
    for i, cid in enumerate(split_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        if cdf.empty:
            raise ValueError(f"curve_id {cid} missing from curves_long.csv")
        t = cdf["time"].to_numpy(dtype=float)
        y = cdf["release"].to_numpy(dtype=float)
        matrix[i, :] = np.interp(rollout_times, t, y, left=y[0], right=y[-1])
    return matrix, curves


def _coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def _interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def _load_active_tables(active_dir: Path) -> tuple[pd.DataFrame, dict]:
    metrics_by_curve = pd.read_csv(active_dir / "metrics_by_curve.csv")
    config = json.loads((active_dir / "config.json").read_text(encoding="utf-8"))
    return metrics_by_curve, config


def _aggregate_existing(metrics_by_curve: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    sub = metrics_by_curve.loc[metrics_by_curve["strategy"].isin(strategies)].copy()
    agg_spec: dict[str, tuple[str, str]] = {
        "n_curves": ("curve_id", "nunique"),
        "coverage90_mean": ("coverage_90", "mean"),
        "coverage90_median": ("coverage_90", "median"),
        "coverage80_mean": ("coverage_80", "mean"),
        "coverage80_median": ("coverage_80", "median"),
        "width90_mean": ("width_90", "mean"),
        "width90_median": ("width_90", "median"),
        "width80_mean": ("width_80", "mean"),
        "width80_median": ("width_80", "median"),
        "crps_mean": ("crps", "mean"),
        "rmse_mean": ("future_rmse_mean", "mean"),
    }
    if {"coverage_50", "width_50"}.issubset(sub.columns):
        agg_spec.update(
            {
                "coverage50_mean": ("coverage_50", "mean"),
                "coverage50_median": ("coverage_50", "median"),
                "width50_mean": ("width_50", "mean"),
                "width50_median": ("width_50", "median"),
            }
        )
    grouped = sub.groupby("strategy", sort=False).agg(**agg_spec).reset_index()
    grouped["source"] = "existing_ao_output"
    return grouped


def _build_naive_train_band(
    rollout_times: np.ndarray,
    future_eval_start: float,
    train_ids: list[int],
    test_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_matrix, _ = _load_curve_matrix(train_ids, rollout_times)
    test_matrix, _ = _load_curve_matrix(test_ids, rollout_times)
    future_mask = rollout_times > future_eval_start

    q05 = np.quantile(train_matrix, 0.05, axis=0)
    q10 = np.quantile(train_matrix, 0.10, axis=0)
    q25 = np.quantile(train_matrix, 0.25, axis=0)
    q75 = np.quantile(train_matrix, 0.75, axis=0)
    q90 = np.quantile(train_matrix, 0.90, axis=0)
    q95 = np.quantile(train_matrix, 0.95, axis=0)
    q50 = np.quantile(train_matrix, 0.50, axis=0)

    rows: list[dict[str, float | int | str]] = []
    for i, cid in enumerate(test_ids):
        y = test_matrix[i, future_mask]
        lo90 = q05[future_mask]
        hi90 = q95[future_mask]
        lo80 = q10[future_mask]
        hi80 = q90[future_mask]
        lo50 = q25[future_mask]
        hi50 = q75[future_mask]
        rows.append(
            {
                "curve_id": int(cid),
                "strategy": "naive_train_band",
                "coverage_90": _coverage(y, lo90, hi90),
                "coverage_80": _coverage(y, lo80, hi80),
                "coverage_50": _coverage(y, lo50, hi50),
                "width_90": _interval_width(lo90, hi90),
                "width_80": _interval_width(lo80, hi80),
                "width_50": _interval_width(lo50, hi50),
                "future_rmse_mean": float(np.sqrt(np.mean((y - q50[future_mask]) ** 2))),
                "crps": np.nan,
            }
        )

    per_curve = pd.DataFrame(rows)
    agg = {
        "strategy": "naive_train_band",
        "n_curves": int(per_curve["curve_id"].nunique()),
        "coverage90_mean": float(per_curve["coverage_90"].mean()),
        "coverage90_median": float(per_curve["coverage_90"].median()),
        "coverage80_mean": float(per_curve["coverage_80"].mean()),
        "coverage80_median": float(per_curve["coverage_80"].median()),
        "coverage50_mean": float(per_curve["coverage_50"].mean()),
        "coverage50_median": float(per_curve["coverage_50"].median()),
        "width90_mean": float(per_curve["width_90"].mean()),
        "width90_median": float(per_curve["width_90"].median()),
        "width80_mean": float(per_curve["width_80"].mean()),
        "width80_median": float(per_curve["width_80"].median()),
        "width50_mean": float(per_curve["width_50"].mean()),
        "width50_median": float(per_curve["width_50"].median()),
        "crps_mean": float("nan"),
        "rmse_mean": float(per_curve["future_rmse_mean"].mean()),
        "source": "empirical_train_band",
    }
    return per_curve, agg


def _write_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.copy()
    naive = summary.loc[summary["strategy"] == "naive_train_band"]
    if naive.empty:
        raise ValueError("naive_train_band row missing from summary")
    naive_width90 = float(naive.iloc[0]["width90_mean"])
    naive_cov90 = float(naive.iloc[0]["coverage90_mean"])
    summary["width90_vs_naive"] = summary["width90_mean"] / max(naive_width90, 1e-12)
    summary["delta_cov90_vs_naive"] = summary["coverage90_mean"] - naive_cov90
    has_50 = {"coverage50_mean", "width50_mean"}.issubset(summary.columns)
    if has_50:
        naive_width50 = float(naive.iloc[0]["width50_mean"])
        naive_cov50 = float(naive.iloc[0]["coverage50_mean"])
        summary["width50_vs_naive"] = summary["width50_mean"] / max(naive_width50, 1e-12)
        summary["delta_cov50_vs_naive"] = summary["coverage50_mean"] - naive_cov50
    summary.to_csv(out_dir / "summary.csv", index=False)

    lines = [
        "=== 69d -- S4 publication-facing coverage/sharpness table ===",
        "",
        "Canonical mouth:",
        "  - test fold = data/canonical_split_v1.csv (38 curves)",
        "  - future evaluation = rollout times > future_eval_start (current AO candidate: > 28d)",
        "  - naive baseline = empirical training-band quantiles on the same mouth",
        "",
    ]
    if has_50:
        lines.extend(
            [
                "Important limitation:",
                "  - 50% interval here is the raw weighted central posterior q25-q75 from the AO rerun",
                "  - it is useful for S4 reporting, but unlike the 90/80 bands it is not separately conformal-calibrated",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Important limitation:",
                "  - current AO artifact exposes 90% and 80% intervals, not 50% intervals",
                "  - this table therefore closes naive-band + sharpness for 90/80, but does NOT yet prove the S4 cov50 requirement",
                "",
            ]
        )
    for _, r in summary.iterrows():
        lines.extend(
            [
                f"{r['strategy']}:",
                f"  n_curves         = {int(r['n_curves'])}",
                f"  cov90 mean/med   = {r['coverage90_mean']:.3f} / {r['coverage90_median']:.3f}",
                f"  cov80 mean/med   = {r['coverage80_mean']:.3f} / {r['coverage80_median']:.3f}",
                f"  width90 mean/med = {r['width90_mean']:.3f} / {r['width90_median']:.3f}",
                f"  width80 mean/med = {r['width80_mean']:.3f} / {r['width80_median']:.3f}",
                f"  width90 vs naive = {r['width90_vs_naive']:.3f}",
                f"  delta cov90      = {r['delta_cov90_vs_naive']:+.3f}",
            ]
        )
        if has_50:
            lines.extend(
                [
                    f"  cov50 mean/med   = {r['coverage50_mean']:.3f} / {r['coverage50_median']:.3f}",
                    f"  width50 mean/med = {r['width50_mean']:.3f} / {r['width50_median']:.3f}",
                    f"  width50 vs naive = {r['width50_vs_naive']:.3f}",
                    f"  delta cov50      = {r['delta_cov50_vs_naive']:+.3f}",
                ]
            )
        if np.isfinite(r["crps_mean"]):
            lines.append(f"  crps mean        = {r['crps_mean']:.4f}")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-dir", type=Path, default=DEFAULT_ACTIVE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_by_curve, config = _load_active_tables(args.active_dir)
    split_df = pd.read_csv(CANONICAL_SPLIT_PATH)
    train_ids = split_df.loc[split_df["split_name"] == "train", "curve_id"].astype(int).tolist()
    test_ids = split_df.loc[split_df["split_name"] == "test", "curve_id"].astype(int).tolist()
    rollout_times = np.asarray(config["rollout_times"], dtype=float)
    future_eval_start = float(config["future_eval_start"])

    strategies = [
        "active_two_point_conformal",
        "active_one_point_conformal",
        "fixed_four_point_conformal",
        "zero_early_prior_conformal",
        "zero_early_prior_only",
    ]
    existing = _aggregate_existing(metrics_by_curve, strategies)
    naive_per_curve, naive_row = _build_naive_train_band(
        rollout_times=rollout_times,
        future_eval_start=future_eval_start,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    naive_per_curve.to_csv(args.out / "naive_train_band_per_curve.csv", index=False)

    summary = pd.concat([existing, pd.DataFrame([naive_row])], ignore_index=True)
    order = {name: i for i, name in enumerate(strategies + ["naive_train_band"])}
    summary["sort_key"] = summary["strategy"].map(order).fillna(999)
    summary = summary.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    _write_summary(summary, args.out)
    print((args.out / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
