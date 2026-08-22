"""
98 - Prototype family-to-duration/shape bridge on selected curves.

Purpose:
    Take a calibrated ReleasePosteriorFamily artifact, decode a sampled family
    through the real mechanism-specific simulator, and produce the first
    route-consistent duration/shape uncertainty summaries for selected curves.

Important scope:
    This is a selected-curve prototype bridge, not yet a benchmark-wide audit.
    Its job is to prove that the current object contract can begin emitting
    timing/shape uncertainty without falling back to a separate point route.

Consumes:
    - outputs/87_release_conformal_family_export/.../*.json
    - outputs/80_.../prediction_trajectories.csv

Produces:
    outputs/98_release_family_duration_shape_bridge/<label>/
        band_trajectories.csv
        threshold_interval_table.csv
        descriptor_interval_table.csv
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
    ap.add_argument("--family-json", type=Path, required=True)
    ap.add_argument("--trajectory-csv", type=Path, required=True)
    ap.add_argument("--curve-id", type=str, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--n-grid", type=int, default=256)
    ap.add_argument("--thresholds", type=str, default="0.1,0.5,0.8")
    ap.add_argument("--outroot", type=Path, default=Path("outputs/98_release_family_duration_shape_bridge"))
    ap.add_argument("--seed", type=int, default=0)
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
    q0 = _interp_q_at(t_use, q_use, t0)
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
        basis = np.asarray(family.z_basis, dtype=float)  # (rank, dim)
        scale = np.asarray(family.z_scale, dtype=float)  # (rank,)
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


def _build_curve_bands(q_samples: np.ndarray, family: ReleasePosteriorFamily) -> dict[str, np.ndarray]:
    q_med = np.nanmedian(q_samples, axis=0)
    q_lo_raw = np.nanquantile(q_samples, 0.05, axis=0)
    q_hi_raw = np.nanquantile(q_samples, 0.95, axis=0)

    global_scale = float(family.calibration_context["global_scale"])
    lo_global = np.clip(q_med - global_scale * (q_med - q_lo_raw), 0.0, 1.0)
    hi_global = np.clip(q_med + global_scale * (q_hi_raw - q_med), 0.0, 1.0)

    return {
        "median": np.maximum.accumulate(np.clip(q_med, 0.0, 1.0)),
        "raw_lo": np.maximum.accumulate(np.clip(q_lo_raw, 0.0, 1.0)),
        "raw_hi": np.maximum.accumulate(np.clip(q_hi_raw, 0.0, 1.0)),
        "global_lo": np.maximum.accumulate(lo_global),
        "global_hi": np.maximum.accumulate(hi_global),
    }


def _add_local_bands(bands: dict[str, np.ndarray], t_grid: np.ndarray, family: ReleasePosteriorFamily) -> dict[str, np.ndarray]:
    q_med = bands["median"]
    q_lo_raw = bands["raw_lo"]
    q_hi_raw = bands["raw_hi"]
    scale = _local_scale_curve(t_grid, family)
    lo_local = np.clip(q_med - scale * (q_med - q_lo_raw), 0.0, 1.0)
    hi_local = np.clip(q_med + scale * (q_hi_raw - q_med), 0.0, 1.0)
    bands["local_lo"] = np.maximum.accumulate(lo_local)
    bands["local_hi"] = np.maximum.accumulate(hi_local)
    return bands


def _descriptor_obs(t_obs: np.ndarray, q_obs: np.ndarray, early_window: float) -> dict[str, float]:
    q_early = _interp_q_at(t_obs, q_obs, early_window)
    q_final = float(np.maximum.accumulate(np.clip(q_obs, 0.0, 1.2))[-1])
    return {
        "burst": q_early,
        "post_window": q_final - q_early,
        "residual_tail": 1.0 - q_final,
        "tail_auc": _tail_auc_after(t_obs, q_obs, early_window),
    }


def _descriptor_from_bands(t_grid: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, early_window: float) -> dict[str, tuple[float, float]]:
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


def main() -> None:
    args = parse_args()
    outdir = args.outroot / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    family = ReleasePosteriorFamily.from_json(args.family_json)
    traj = pd.read_csv(args.trajectory_csv)
    sub = traj[traj["curve_id"] == args.curve_id].copy()
    if sub.empty:
        raise ValueError(f"curve_id {args.curve_id} not found in {args.trajectory_csv}")
    sub = sub.sort_values("t")

    t_obs = sub["t"].to_numpy(dtype=float)
    q_obs = sub["Q_obs"].to_numpy(dtype=float)
    early_col = _pick_early_window_col(sub)
    early_window = float(sub[early_col].dropna().iloc[0]) if early_col is not None and sub[early_col].notna().any() else np.nan
    t_max = float(family.assay_context["t_max"])
    t_grid = np.unique(np.concatenate([np.linspace(0.0, t_max, args.n_grid), t_obs]))

    rng = np.random.default_rng(args.seed)
    theta_samples = _sample_theta(family, args.n_samples, rng)
    sim = _make_simulator(family.decoder_handle)
    q_samples = np.vstack([sim.simulate_numpy(theta, t_grid) for theta in theta_samples])

    bands = _build_curve_bands(q_samples, family)
    bands = _add_local_bands(bands, t_grid, family)

    band_df = pd.DataFrame(
        {
            "t": t_grid,
            "q_median": bands["median"],
            "q_raw_lo": bands["raw_lo"],
            "q_raw_hi": bands["raw_hi"],
            "q_global_lo": bands["global_lo"],
            "q_global_hi": bands["global_hi"],
            "q_local_lo": bands["local_lo"],
            "q_local_hi": bands["local_hi"],
        }
    )
    band_df.to_csv(outdir / "band_trajectories.csv", index=False)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    threshold_rows: list[dict[str, object]] = []
    t_obs_map = {thr: _time_to_threshold(t_obs, q_obs, thr) for thr in thresholds}

    for thr in thresholds:
        t_true = t_obs_map[thr]
        for method, lo_key, hi_key in [
            ("raw", "raw_lo", "raw_hi"),
            ("global", "global_lo", "global_hi"),
            ("local", "local_lo", "local_hi"),
        ]:
            t_early = _time_to_threshold(t_grid, bands[hi_key], thr)
            t_late = _time_to_threshold(t_grid, bands[lo_key], thr)
            covered = bool(np.isfinite(t_true) and np.isfinite(t_early) and np.isfinite(t_late) and (t_early <= t_true <= t_late))
            threshold_rows.append(
                {
                    "curve_id": args.curve_id,
                    "method": method,
                    "threshold": thr,
                    "t_true": t_true,
                    "t_interval_low": t_early,
                    "t_interval_high": t_late,
                    "covered": covered,
                    "interval_width": t_late - t_early if np.isfinite(t_early) and np.isfinite(t_late) else np.nan,
                }
            )
    pd.DataFrame(threshold_rows).to_csv(outdir / "threshold_interval_table.csv", index=False)

    obs_desc = _descriptor_obs(t_obs, q_obs, early_window)
    descriptor_rows: list[dict[str, object]] = []
    for method, lo_key, hi_key in [
        ("raw", "raw_lo", "raw_hi"),
        ("global", "global_lo", "global_hi"),
        ("local", "local_lo", "local_hi"),
    ]:
        desc_int = _descriptor_from_bands(t_grid, bands[lo_key], bands[hi_key], early_window)
        for name, true_val in obs_desc.items():
            lo, hi = desc_int[name]
            descriptor_rows.append(
                {
                    "curve_id": args.curve_id,
                    "method": method,
                    "descriptor": name,
                    "true_value": true_val,
                    "interval_low": lo,
                    "interval_high": hi,
                    "covered": bool(np.isfinite(true_val) and np.isfinite(lo) and np.isfinite(hi) and (lo <= true_val <= hi)),
                    "interval_width": hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                }
            )
    pd.DataFrame(descriptor_rows).to_csv(outdir / "descriptor_interval_table.csv", index=False)

    lines = [
        "=== 98 -- family duration/shape bridge ===",
        "",
        f"curve_id: {args.curve_id}",
        f"mechanism_id: {family.mechanism_id}",
        f"decoder_handle: {family.decoder_handle}",
        f"n_samples: {args.n_samples}",
        f"t_max: {t_max}",
        f"early_window: {early_window}",
        "",
        "Interpretation",
        "  1. This is a selected-curve bridge from calibrated family artifact to duration/shape intervals.",
        "  2. Intervals are produced by decoding sampled families through the real simulator and inflating release-space bands with the attached conformal scales.",
        "  3. This does not yet prove benchmark-wide route-consistent duration UQ, but it shows the shared object can begin to emit it.",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
