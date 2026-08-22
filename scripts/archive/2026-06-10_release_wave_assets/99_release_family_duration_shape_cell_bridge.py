"""
99 - Cell-level family duration/shape bridge on calibrated-family artifacts.

Purpose:
    Extend the selected-curve bridge into a representative-cell summary.
    For each calibrated family in a chosen cell, decode sampled trajectories
    through the real simulator, derive threshold and shape intervals, and
    summarize coverage / width across the cell.

Important scope:
    This is a benchmark-adjacent prototype on chosen cells, not yet a full
    benchmark-wide replacement for script 97.

Consumes:
    - outputs/87_release_conformal_family_export/<dataset>/<scheme>/*.json
    - outputs/80_.../prediction_trajectories.csv

Produces:
    outputs/99_release_family_duration_shape_cell_bridge/<label>/
        threshold_curve_table.csv
        threshold_summary.csv
        descriptor_curve_table.csv
        descriptor_summary.csv
        summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_posterior_family import ReleasePosteriorFamily
from simulator import PLGABiphasic
from weibull_simulator import WeibullSimulator


EARLY_WINDOW_CANDIDATES = (
    "early_window_h",
    "early_window_h_meta",
    "early_window_days",
    "early_window_days_meta",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family-dir", type=Path, required=True)
    ap.add_argument("--trajectory-csv", type=Path, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-grid", type=int, default=192)
    ap.add_argument("--thresholds", type=str, default="0.1,0.5,0.8")
    ap.add_argument("--max-curves", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outroot", type=Path, default=Path("outputs/99_release_family_duration_shape_cell_bridge"))
    return ap.parse_args()


def _pick_early_window_col(df: pd.DataFrame) -> str | None:
    for col in EARLY_WINDOW_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _time_to_threshold(t: np.ndarray, q: np.ndarray, target: float) -> float:
    mask = np.isfinite(t) & np.isfinite(q)
    if mask.sum() < 2:
        return float("nan")
    t_use = t[mask]
    q_use = np.maximum.accumulate(np.clip(q[mask], 0.0, 1.2))
    if q_use.max() < target:
        return float("nan")
    hit = int(np.argmax(q_use >= target))
    if hit == 0:
        return float(t_use[0])
    q0, q1 = q_use[hit - 1], q_use[hit]
    t0, t1 = t_use[hit - 1], t_use[hit]
    if q1 <= q0:
        return float(t1)
    frac = float((target - q0) / (q1 - q0))
    return float(t0 + frac * (t1 - t0))


def _interp_q_at(t: np.ndarray, q: np.ndarray, t_query: float) -> float:
    mask = np.isfinite(t) & np.isfinite(q)
    if mask.sum() < 2 or not np.isfinite(t_query):
        return float("nan")
    t_use = t[mask]
    q_use = np.maximum.accumulate(np.clip(q[mask], 0.0, 1.2))
    order = np.argsort(t_use)
    t_use = t_use[order]
    q_use = q_use[order]
    if t_query <= t_use[0]:
        return float(q_use[0])
    if t_query >= t_use[-1]:
        return float(q_use[-1])
    return float(np.interp(t_query, t_use, q_use))


def _tail_auc_after(t: np.ndarray, q: np.ndarray, t_start: float) -> float:
    mask = np.isfinite(t) & np.isfinite(q)
    if mask.sum() < 2 or not np.isfinite(t_start):
        return float("nan")
    t_use = t[mask]
    q_use = np.maximum.accumulate(np.clip(q[mask], 0.0, 1.2))
    order = np.argsort(t_use)
    t_use = t_use[order]
    q_use = q_use[order]
    t0 = max(float(t_start), float(t_use[0]))
    t1 = float(t_use[-1])
    if t1 <= t0:
        return 0.0
    tail_t = np.concatenate([[t0], t_use[(t_use > t0) & (t_use < t1)], [t1]])
    tail_q = np.array([_interp_q_at(t_use, q_use, x) for x in tail_t], dtype=float)
    unreleased = np.clip(1.0 - tail_q, 0.0, 1.2)
    return float(np.trapz(unreleased, tail_t))


def _make_simulator(handle: str):
    if handle == "PLGABiphasic.simulate_numpy":
        return PLGABiphasic()
    if handle == "WeibullSimulator.simulate_numpy":
        return WeibullSimulator()
    raise ValueError(f"Unsupported decoder handle: {handle}")


def _sample_theta(family: ReleasePosteriorFamily, n: int, rng: np.random.Generator) -> np.ndarray:
    center = np.asarray(family.z_center, dtype=float)
    if family.rank == 0:
        z = np.repeat(center[None, :], n, axis=0)
    else:
        basis = np.asarray(family.z_basis, dtype=float)
        scale = np.asarray(family.z_scale, dtype=float)
        coeff = rng.normal(size=(n, family.rank))
        z = center[None, :] + (coeff * scale[None, :]) @ basis
    sim = _make_simulator(family.decoder_handle)
    prior = sim.prior().base_dist
    low = prior.low.detach().cpu().numpy()
    high = prior.high.detach().cpu().numpy()
    eps = 1e-4 * (high - low)
    return np.clip(z, low + eps, high - eps)


def _local_scale_curve(t: np.ndarray, family: ReleasePosteriorFamily) -> np.ndarray:
    edges = np.asarray(family.calibration_context["local_scale_bin_edges_t_norm"], dtype=float)
    values = np.asarray(family.calibration_context["local_scale_values"], dtype=float)
    t_max = float(family.assay_context["t_max"])
    t_norm = np.clip(t / t_max, 0.0, 1.0)
    scale = np.empty_like(t_norm)
    for i in range(len(values)):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (t_norm >= lo) & (t_norm < hi)
        scale[mask] = values[i]
    scale[t_norm >= edges[-2]] = values[-1]
    return scale


def _bands_from_samples(q_samples: np.ndarray, family: ReleasePosteriorFamily, t_grid: np.ndarray) -> dict[str, np.ndarray]:
    q_med = np.nanmedian(q_samples, axis=0)
    q_lo_raw = np.nanquantile(q_samples, 0.05, axis=0)
    q_hi_raw = np.nanquantile(q_samples, 0.95, axis=0)
    global_scale = float(family.calibration_context["global_scale"])
    scale_local = _local_scale_curve(t_grid, family)

    bands = {
        "median": np.maximum.accumulate(np.clip(q_med, 0.0, 1.0)),
        "raw_lo": np.maximum.accumulate(np.clip(q_lo_raw, 0.0, 1.0)),
        "raw_hi": np.maximum.accumulate(np.clip(q_hi_raw, 0.0, 1.0)),
    }
    bands["global_lo"] = np.maximum.accumulate(np.clip(bands["median"] - global_scale * (bands["median"] - bands["raw_lo"]), 0.0, 1.0))
    bands["global_hi"] = np.maximum.accumulate(np.clip(bands["median"] + global_scale * (bands["raw_hi"] - bands["median"]), 0.0, 1.0))
    bands["local_lo"] = np.maximum.accumulate(np.clip(bands["median"] - scale_local * (bands["median"] - bands["raw_lo"]), 0.0, 1.0))
    bands["local_hi"] = np.maximum.accumulate(np.clip(bands["median"] + scale_local * (bands["raw_hi"] - bands["median"]), 0.0, 1.0))
    return bands


def _descriptor_true(t_obs: np.ndarray, q_obs: np.ndarray, early_window: float) -> dict[str, float]:
    q_early = _interp_q_at(t_obs, q_obs, early_window)
    q_final = float(np.maximum.accumulate(np.clip(q_obs, 0.0, 1.2))[-1])
    return {
        "burst": q_early,
        "post_window": q_final - q_early,
        "residual_tail": 1.0 - q_final,
        "tail_auc": _tail_auc_after(t_obs, q_obs, early_window),
    }


def _descriptor_interval(t_grid: np.ndarray, bands: dict[str, np.ndarray], early_window: float, method: str) -> dict[str, tuple[float, float]]:
    lo_key = f"{method}_lo"
    hi_key = f"{method}_hi"
    q_lo = bands[lo_key]
    q_hi = bands[hi_key]
    q_early_lo = _interp_q_at(t_grid, q_lo, early_window)
    q_early_hi = _interp_q_at(t_grid, q_hi, early_window)
    q_final_lo = float(q_lo[-1])
    q_final_hi = float(q_hi[-1])
    return {
        "burst": (q_early_lo, q_early_hi),
        "post_window": (q_final_lo - q_early_hi, q_final_hi - q_early_lo),
        "residual_tail": (1.0 - q_final_hi, 1.0 - q_final_lo),
        "tail_auc": (_tail_auc_after(t_grid, q_hi, early_window), _tail_auc_after(t_grid, q_lo, early_window)),
    }


def _safe_mean(x: pd.Series) -> float:
    arr = x.to_numpy(dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _select_matched_families(
    family_dir: Path,
    trajectory_curve_ids: set[str],
    max_curves: int,
    rng: np.random.Generator,
) -> tuple[list[ReleasePosteriorFamily], int]:
    matched: list[ReleasePosteriorFamily] = []
    for path in sorted(family_dir.glob("*.json")):
        family = ReleasePosteriorFamily.from_json(path)
        curve_id = str(family.metadata["curve_id"])
        if curve_id in trajectory_curve_ids:
            matched.append(family)
    matched_total = len(matched)
    if max_curves > 0 and matched_total > max_curves:
        idx = np.sort(rng.choice(matched_total, size=max_curves, replace=False))
        matched = [matched[i] for i in idx]
    return matched, matched_total


def main() -> None:
    args = parse_args()
    outdir = args.outroot / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    traj = pd.read_csv(args.trajectory_csv)
    rng = np.random.default_rng(args.seed)
    traj_curve_ids = set(traj["curve_id"].astype(str).unique().tolist())
    families, matched_total = _select_matched_families(
        args.family_dir,
        traj_curve_ids,
        args.max_curves,
        rng,
    )

    threshold_rows: list[dict[str, object]] = []
    descriptor_rows: list[dict[str, object]] = []

    for family in families:
        curve_id = str(family.metadata["curve_id"])
        sub = traj[traj["curve_id"] == curve_id].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("t")
        t_obs = sub["t"].to_numpy(dtype=float)
        q_obs = sub["Q_obs"].to_numpy(dtype=float)
        early_col = _pick_early_window_col(sub)
        early_window = float(sub[early_col].dropna().iloc[0]) if early_col and sub[early_col].notna().any() else np.nan
        t_max = float(family.assay_context["t_max"])
        t_grid = np.unique(np.concatenate([np.linspace(0.0, t_max, args.n_grid), t_obs]))
        theta_samples = _sample_theta(family, args.n_samples, rng)
        sim = _make_simulator(family.decoder_handle)
        q_samples = np.vstack([sim.simulate_numpy(theta, t_grid) for theta in theta_samples])
        bands = _bands_from_samples(q_samples, family, t_grid)

        for thr in thresholds:
            t_true = _time_to_threshold(t_obs, q_obs, thr)
            for method in ("raw", "global", "local"):
                t_lo = _time_to_threshold(t_grid, bands[f"{method}_hi"], thr)
                t_hi = _time_to_threshold(t_grid, bands[f"{method}_lo"], thr)
                threshold_rows.append(
                    {
                        "curve_id": curve_id,
                        "method": method,
                        "threshold": thr,
                        "t_true": t_true,
                        "t_interval_low": t_lo,
                        "t_interval_high": t_hi,
                        "covered": bool(np.isfinite(t_true) and np.isfinite(t_lo) and np.isfinite(t_hi) and (t_lo <= t_true <= t_hi)),
                        "interval_width": t_hi - t_lo if np.isfinite(t_lo) and np.isfinite(t_hi) else np.nan,
                    }
                )

        desc_true = _descriptor_true(t_obs, q_obs, early_window)
        for method in ("raw", "global", "local"):
            desc_int = _descriptor_interval(t_grid, bands, early_window, method)
            for name, true_val in desc_true.items():
                lo, hi = desc_int[name]
                descriptor_rows.append(
                    {
                        "curve_id": curve_id,
                        "method": method,
                        "descriptor": name,
                        "true_value": true_val,
                        "interval_low": lo,
                        "interval_high": hi,
                        "covered": bool(np.isfinite(true_val) and np.isfinite(lo) and np.isfinite(hi) and (lo <= true_val <= hi)),
                        "interval_width": hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                    }
                )

    threshold_df = pd.DataFrame(threshold_rows)
    descriptor_df = pd.DataFrame(descriptor_rows)
    threshold_df.to_csv(outdir / "threshold_curve_table.csv", index=False)
    descriptor_df.to_csv(outdir / "descriptor_curve_table.csv", index=False)

    threshold_summary = (
        threshold_df.groupby(["method", "threshold"], as_index=False)
        .agg(
            n_curves=("curve_id", "nunique"),
            coverage=("covered", _safe_mean),
            median_width=("interval_width", _safe_mean),
        )
    )
    descriptor_summary = (
        descriptor_df.groupby(["method", "descriptor"], as_index=False)
        .agg(
            n_curves=("curve_id", "nunique"),
            coverage=("covered", _safe_mean),
            median_width=("interval_width", _safe_mean),
        )
    )
    threshold_summary.to_csv(outdir / "threshold_summary.csv", index=False)
    descriptor_summary.to_csv(outdir / "descriptor_summary.csv", index=False)

    lines = [
        "=== 99 -- family duration/shape cell bridge ===",
        "",
        f"label: {args.label}",
        f"family_dir: {args.family_dir.as_posix()}",
        f"trajectory_csv: {args.trajectory_csv.as_posix()}",
        f"matched_curve_count: {matched_total}",
        f"selected_curve_count: {len(families)}",
        f"n_curves_used: {threshold_df['curve_id'].nunique() if not threshold_df.empty else 0}",
        f"n_samples_per_curve: {args.n_samples}",
        "",
        "Interpretation",
        "  1. This is a chosen-cell bridge from calibrated family artifacts to cell-level duration/shape coverage summaries.",
        "  2. It remains benchmark-adjacent rather than benchmark-wide, but it is the first attempt to summarize family-based duration/shape intervals across multiple real curves.",
        "  3. The critical question is no longer whether the object can emit one selected interval; it is whether cell-level coverage is coherent enough to justify widening the audit.",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
