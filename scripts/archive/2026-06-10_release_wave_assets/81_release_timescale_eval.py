"""
81 - Dedicated release-timescale evaluation from shared trajectory tables.

Purpose:
    Evaluate release-duration / threshold timing as a first-class object.
    Unlike the lightweight timescale summary inside script 80, this script
    explicitly distinguishes:

    1. global threshold timing error
    2. forecast-relevant timing error, where the threshold is reached after
       the early-observation window

Consumes:
    prediction_trajectories.csv from the shared release benchmark layer.

Produces:
    timescale_curve_table.csv
    timescale_summary_all.csv
    timescale_summary_future_only.csv
    descriptor_curve_table.csv
    descriptor_summary.csv
    summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EARLY_WINDOW_CANDIDATES = (
    "early_window_h",
    "early_window_h_meta",
    "early_window_days",
    "early_window_days_meta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--thresholds", type=str, default="0.1,0.5,0.8")
    parser.add_argument(
        "--time-unit",
        choices=("auto", "hours", "days", "as_is"),
        default="auto",
    )
    return parser.parse_args()


def _resolve_time_unit(df: pd.DataFrame, requested: str) -> tuple[str, float]:
    if requested == "hours":
        return "hours", 1.0
    if requested == "days":
        return "days", 1.0
    if requested == "as_is":
        return "as_is", 1.0

    for col in ("time_unit_standard", "time_unit", "time_units"):
        if col in df.columns:
            value = str(df[col].dropna().iloc[0]).strip().lower()
            if value in ("hours", "days", "as_is"):
                return value, 1.0
    return "as_is", 1.0


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
    if q_use[hit] < target:
        return float("nan")
    if hit == 0:
        return float(t_use[0])
    q0, q1 = q_use[hit - 1], q_use[hit]
    t0, t1 = t_use[hit - 1], t_use[hit]
    if q1 <= q0:
        return float(t1)
    frac = float((target - q0) / (q1 - q0))
    return float(t0 + frac * (t1 - t0))


def _summary_from_subset(sub: pd.DataFrame) -> dict[str, float]:
    err = sub["signed_error"].to_numpy(dtype=float)
    abs_err = sub["abs_error"].to_numpy(dtype=float)
    mask = np.isfinite(err)
    if not mask.any():
        return {
            "n_valid": 0,
            "mae_t": np.nan,
            "rmse_t": np.nan,
            "median_abs_error_t": np.nan,
            "bias_t": np.nan,
        }
    return {
        "n_valid": int(mask.sum()),
        "mae_t": float(np.nanmean(abs_err)),
        "rmse_t": float(np.sqrt(np.nanmean(err[mask] ** 2))),
        "median_abs_error_t": float(np.nanmedian(abs_err)),
        "bias_t": float(np.nanmean(err)),
    }


def _numeric_error_summary(err: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(err)
    if not mask.any():
        return {
            "n_valid": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs_error": np.nan,
            "bias": np.nan,
        }
    use = err[mask]
    abs_use = np.abs(use)
    return {
        "n_valid": int(mask.sum()),
        "mae": float(np.nanmean(abs_use)),
        "rmse": float(np.sqrt(np.nanmean(use ** 2))),
        "median_abs_error": float(np.nanmedian(abs_use)),
        "bias": float(np.nanmean(use)),
    }


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


def build_curve_table(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    early_col = _pick_early_window_col(df)
    rows: list[dict[str, object]] = []

    for curve_id, sub in df.groupby("curve_id", sort=True):
        sub = sub.sort_values("t")
        t = sub["t"].to_numpy(dtype=float)
        q_obs = sub["Q_obs"].to_numpy(dtype=float)
        q_pred = sub["Q_pred"].to_numpy(dtype=float)
        early_window = float(sub[early_col].dropna().iloc[0]) if early_col is not None and sub[early_col].notna().any() else np.nan

        for target in thresholds:
            t_obs = _time_to_threshold(t, q_obs, target)
            t_pred = _time_to_threshold(t, q_pred, target)
            forecast_relevant = bool(np.isfinite(early_window) and np.isfinite(t_obs) and t_obs > early_window)
            rows.append(
                {
                    "curve_id": curve_id,
                    "threshold": target,
                    "time_unit": str(sub["time_unit_standard"].dropna().iloc[0]) if "time_unit_standard" in sub.columns and sub["time_unit_standard"].notna().any() else "",
                    "early_window": early_window,
                    "t_obs": t_obs,
                    "t_pred": t_pred,
                    "signed_error": t_pred - t_obs if np.isfinite(t_obs) and np.isfinite(t_pred) else np.nan,
                    "abs_error": abs(t_pred - t_obs) if np.isfinite(t_obs) and np.isfinite(t_pred) else np.nan,
                    "forecast_relevant": forecast_relevant,
                }
            )
    return pd.DataFrame(rows)


def build_descriptor_table(df: pd.DataFrame) -> pd.DataFrame:
    early_col = _pick_early_window_col(df)
    rows: list[dict[str, object]] = []

    for curve_id, sub in df.groupby("curve_id", sort=True):
        sub = sub.sort_values("t")
        t = sub["t"].to_numpy(dtype=float)
        q_obs = sub["Q_obs"].to_numpy(dtype=float)
        q_pred = sub["Q_pred"].to_numpy(dtype=float)
        early_window = float(sub[early_col].dropna().iloc[0]) if early_col is not None and sub[early_col].notna().any() else np.nan
        q_early_obs = _interp_q_at(t, q_obs, early_window)
        q_early_pred = _interp_q_at(t, q_pred, early_window)
        q_final_obs = float(np.maximum.accumulate(np.clip(q_obs, 0.0, 1.2))[-1]) if len(q_obs) else np.nan
        q_final_pred = float(np.maximum.accumulate(np.clip(q_pred, 0.0, 1.2))[-1]) if len(q_pred) else np.nan
        post_release_obs = q_final_obs - q_early_obs if np.isfinite(q_final_obs) and np.isfinite(q_early_obs) else np.nan
        post_release_pred = q_final_pred - q_early_pred if np.isfinite(q_final_pred) and np.isfinite(q_early_pred) else np.nan
        residual_tail_obs = 1.0 - q_final_obs if np.isfinite(q_final_obs) else np.nan
        residual_tail_pred = 1.0 - q_final_pred if np.isfinite(q_final_pred) else np.nan
        tail_auc_obs = _tail_auc_after(t, q_obs, early_window)
        tail_auc_pred = _tail_auc_after(t, q_pred, early_window)
        rows.append(
            {
                "curve_id": curve_id,
                "time_unit": str(sub["time_unit_standard"].dropna().iloc[0]) if "time_unit_standard" in sub.columns and sub["time_unit_standard"].notna().any() else "",
                "early_window": early_window,
                "t_max": float(np.nanmax(t)) if len(t) else np.nan,
                "burst_fraction_obs": q_early_obs,
                "burst_fraction_pred": q_early_pred,
                "burst_fraction_error": q_early_pred - q_early_obs if np.isfinite(q_early_obs) and np.isfinite(q_early_pred) else np.nan,
                "post_window_release_obs": post_release_obs,
                "post_window_release_pred": post_release_pred,
                "post_window_release_error": post_release_pred - post_release_obs if np.isfinite(post_release_obs) and np.isfinite(post_release_pred) else np.nan,
                "residual_tail_obs": residual_tail_obs,
                "residual_tail_pred": residual_tail_pred,
                "residual_tail_error": residual_tail_pred - residual_tail_obs if np.isfinite(residual_tail_obs) and np.isfinite(residual_tail_pred) else np.nan,
                "tail_auc_obs": tail_auc_obs,
                "tail_auc_pred": tail_auc_pred,
                "tail_auc_error": tail_auc_pred - tail_auc_obs if np.isfinite(tail_auc_obs) and np.isfinite(tail_auc_pred) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_descriptors(descriptor_table: pd.DataFrame) -> pd.DataFrame:
    spec = {
        "burst_fraction_error": "burst_fraction_at_early_window",
        "post_window_release_error": "post_window_release",
        "residual_tail_error": "residual_tail_at_tmax",
        "tail_auc_error": "tail_auc_after_early_window",
    }
    rows: list[dict[str, object]] = []
    for col, label in spec.items():
        summary = _numeric_error_summary(descriptor_table[col].to_numpy(dtype=float))
        summary["descriptor"] = label
        rows.append(summary)
    cols = ["descriptor", "n_valid", "mae", "rmse", "median_abs_error", "bias"]
    return pd.DataFrame(rows)[cols]


def summarize(curve_table: pd.DataFrame, future_only: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold, sub in curve_table.groupby("threshold", sort=True):
        use = sub[sub["forecast_relevant"]] if future_only else sub
        summary = _summary_from_subset(use)
        summary["threshold"] = threshold
        rows.append(summary)
    cols = ["threshold", "n_valid", "mae_t", "rmse_t", "median_abs_error_t", "bias_t"]
    return pd.DataFrame(rows)[cols]


def write_summary_text(
    outdir: Path,
    curve_table: pd.DataFrame,
    all_summary: pd.DataFrame,
    future_summary: pd.DataFrame,
    descriptor_summary: pd.DataFrame,
) -> None:
    lines = [
        "=== 81 -- release timescale evaluation ===",
        "",
        f"curves           : {curve_table['curve_id'].nunique()}",
        f"threshold rows   : {len(curve_table)}",
        "",
        "--- all thresholds ---",
    ]
    for row in all_summary.itertuples(index=False):
        lines.append(
            f"  thr={row.threshold:.2f} n={row.n_valid:<3d} "
            f"mae={row.mae_t:.3f} rmse={row.rmse_t:.3f} median_abs={row.median_abs_error_t:.3f} bias={row.bias_t:.3f}"
        )
    lines.extend(["", "--- forecast-relevant only (t_obs > early window) ---"])
    for row in future_summary.itertuples(index=False):
        lines.append(
            f"  thr={row.threshold:.2f} n={row.n_valid:<3d} "
            f"mae={row.mae_t:.3f} rmse={row.rmse_t:.3f} median_abs={row.median_abs_error_t:.3f} bias={row.bias_t:.3f}"
        )
    lines.extend(["", "--- release-shape descriptors ---"])
    for row in descriptor_summary.itertuples(index=False):
        lines.append(
            f"  {row.descriptor}: n={row.n_valid:<3d} "
            f"mae={row.mae:.3f} rmse={row.rmse:.3f} median_abs={row.median_abs_error:.3f} bias={row.bias:.3f}"
        )
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    df = pd.read_csv(args.trajectory_csv)
    if "Q_pred" not in df.columns or "Q_obs" not in df.columns or "t" not in df.columns:
        raise KeyError("trajectory csv must contain columns t, Q_obs, Q_pred")

    time_unit, _ = _resolve_time_unit(df, args.time_unit)
    if "time_unit_standard" not in df.columns:
        df["time_unit_standard"] = time_unit

    curve_table = build_curve_table(df=df, thresholds=thresholds)
    all_summary = summarize(curve_table=curve_table, future_only=False)
    future_summary = summarize(curve_table=curve_table, future_only=True)
    descriptor_table = build_descriptor_table(df=df)
    descriptor_summary = summarize_descriptors(descriptor_table=descriptor_table)

    curve_table.to_csv(args.outdir / "timescale_curve_table.csv", index=False)
    all_summary.to_csv(args.outdir / "timescale_summary_all.csv", index=False)
    future_summary.to_csv(args.outdir / "timescale_summary_future_only.csv", index=False)
    descriptor_table.to_csv(args.outdir / "descriptor_curve_table.csv", index=False)
    descriptor_summary.to_csv(args.outdir / "descriptor_summary.csv", index=False)
    write_summary_text(
        outdir=args.outdir,
        curve_table=curve_table,
        all_summary=all_summary,
        future_summary=future_summary,
        descriptor_summary=descriptor_summary,
    )


if __name__ == "__main__":
    main()
