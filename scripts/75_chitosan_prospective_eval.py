"""
75 - Evaluate locked chitosan prospective predictions against wet curves.

This script is the preregistered reveal step for script 67. It compares
the locked predictions in outputs/67_chitosan_prospective/predictions.csv
against observed release curves after the wet experiment is complete.

Primary endpoint, per
docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md:
    - aggregate cov90 across all (curve, timepoint) observations
    - preregistered success threshold: cov90 >= 0.83

Secondary / descriptive outputs:
    - per-curve cov90 hit rate
    - fraction of curves with >= 80% points inside the 90% PI
    - pooled R^2 across all (curve, timepoint) pairs
    - mean 90% / 50% PI width
    - empirical CRPS estimated from the locked theta family in
      outputs/67_chitosan_prospective/theta_targets.npz

Inputs:
    outputs/67_chitosan_prospective/predictions.csv
    outputs/67_chitosan_prospective/theta_targets.npz
    observed CSV with columns:
        formulation, drug, t_hours, Q_observed
    Optional accepted aliases:
        time_h / time_hours / t_days
        release / Q / q_observed / observed

Outputs:
    outputs/75_chitosan_prospective_eval/per_point.csv
    outputs/75_chitosan_prospective_eval/per_curve_metrics.csv
    outputs/75_chitosan_prospective_eval/gate_summary.csv
    outputs/75_chitosan_prospective_eval/observed_template.csv (with --write-template)
    outputs/75_chitosan_prospective_eval/summary.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chitosan_simulator import ChitosanRitgerPeppas


REQUIRED_KEYS = ("formulation", "drug")
DEFAULT_TIME_GRID_HOURS = np.array([0.5, 1, 2, 6, 24, 72, 168, 336, 504, 672], dtype=float)
DEFAULT_PREREG_DOC = Path("docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md")
DEFAULT_CURVE_HIT_THRESHOLD = 0.80
DEFAULT_CRPS_SAMPLES = 256
DEFAULT_SEED = 0


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    yh = y_pred[mask]
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, q_floor: float) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= q_floor)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])))


def _normalize_release(y: pd.Series) -> pd.Series:
    out = y.astype(float)
    if out.max() > 2.0:
        out = out / 100.0
    return out.clip(0.0, 1.2)


def _empirical_crps(samples: np.ndarray, y_true: float) -> float:
    if samples.size == 0 or not np.isfinite(y_true):
        return float("nan")
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return float("nan")
    term_1 = np.mean(np.abs(s - y_true))
    term_2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
    return float(term_1 - term_2)


def _load_lock_metadata(predictions: Path) -> dict[str, object]:
    lock_path = predictions.parent / "lock_metadata.json"
    if not lock_path.exists():
        return {}
    return json.loads(lock_path.read_text(encoding="utf-8"))


def _load_preregistered_min_cov90(doc_path: Path) -> tuple[float, str]:
    if not doc_path.exists():
        return 0.83, f"fallback default (missing {doc_path})"
    text = doc_path.read_text(encoding="utf-8")
    match = re.search(
        r"Pre-specified success threshold:\s*\*\*cov90\s*[≥>=]\s*([0-9.]+)\*\*",
        text,
    )
    if match:
        return float(match.group(1)), str(doc_path)
    return 0.83, f"fallback default (unparsed {doc_path})"


def _standardize_observed(raw: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower().strip(): c for c in raw.columns}

    def pick(*names: str) -> str:
        for name in names:
            if name.lower() in col_map:
                return col_map[name.lower()]
        raise KeyError(f"expected one of columns {names}; found {list(raw.columns)}")

    formulation_col = pick("formulation", "formulation_id", "formula", "sample")
    drug_col = pick("drug", "drug_name", "api")
    if "t_hours" in col_map:
        t_hours = raw[col_map["t_hours"]].astype(float)
    elif "time_h" in col_map:
        t_hours = raw[col_map["time_h"]].astype(float)
    elif "time_hours" in col_map:
        t_hours = raw[col_map["time_hours"]].astype(float)
    elif "t_days" in col_map:
        t_hours = raw[col_map["t_days"]].astype(float) * 24.0
    elif "time_days" in col_map:
        t_hours = raw[col_map["time_days"]].astype(float) * 24.0
    else:
        raise KeyError("observed CSV needs t_hours/time_h/time_hours or t_days/time_days")

    q_col = pick("Q_observed", "q_observed", "release", "Q", "observed")
    out = pd.DataFrame(
        {
            "formulation": raw[formulation_col].astype(str).str.strip(),
            "drug": raw[drug_col].astype(str).str.strip(),
            "t_hours": t_hours,
            "Q_observed": _normalize_release(raw[q_col]),
        }
    )
    out = (
        out.dropna(subset=["formulation", "drug", "t_hours", "Q_observed"])
        .groupby(["formulation", "drug", "t_hours"], as_index=False)
        .agg({"Q_observed": "mean"})
        .sort_values(["formulation", "drug", "t_hours"])
        .reset_index(drop=True)
    )
    return out


def write_template(pred: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for formulation, drug in pred[list(REQUIRED_KEYS)].drop_duplicates().itertuples(index=False, name=None):
        for t in DEFAULT_TIME_GRID_HOURS:
            rows.append(
                {
                    "formulation": formulation,
                    "drug": drug,
                    "t_hours": float(t),
                    "Q_observed": "",
                }
            )
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def interpolate_locked_predictions(pred: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pred_keys = set(pred[list(REQUIRED_KEYS)].drop_duplicates().itertuples(index=False, name=None))
    obs_keys = set(obs[list(REQUIRED_KEYS)].drop_duplicates().itertuples(index=False, name=None))
    missing = sorted(obs_keys - pred_keys)
    if missing:
        raise KeyError(f"observed curves missing from locked predictions: {missing}")

    pred_cols = ["Q_predicted_point", "Q_lo90", "Q_hi90", "Q_lo50", "Q_hi50"]
    for key, obs_sub in obs.groupby(list(REQUIRED_KEYS), sort=True):
        formulation, drug = key
        pred_sub = pred[(pred["formulation"] == formulation) & (pred["drug"] == drug)].sort_values("t_hours")
        t_pred = pred_sub["t_hours"].to_numpy(dtype=float)
        out = obs_sub.copy()
        for col in pred_cols:
            out[col] = np.interp(
                obs_sub["t_hours"].to_numpy(dtype=float),
                t_pred,
                pred_sub[col].to_numpy(dtype=float),
                left=float(pred_sub[col].iloc[0]),
                right=float(pred_sub[col].iloc[-1]),
            )
        rows.extend(out.to_dict("records"))
    merged = pd.DataFrame(rows)
    merged["inside90"] = (
        (merged["Q_observed"] >= merged["Q_lo90"]) & (merged["Q_observed"] <= merged["Q_hi90"])
    )
    merged["inside50"] = (
        (merged["Q_observed"] >= merged["Q_lo50"]) & (merged["Q_observed"] <= merged["Q_hi50"])
    )
    merged["pi_width_90"] = merged["Q_hi90"] - merged["Q_lo90"]
    merged["pi_width_50"] = merged["Q_hi50"] - merged["Q_lo50"]
    merged["abs_error"] = (merged["Q_observed"] - merged["Q_predicted_point"]).abs()
    return merged.sort_values(["formulation", "drug", "t_hours"]).reset_index(drop=True)


def _simulate_shifted_samples(
    mu: np.ndarray,
    U: np.ndarray,
    sigma: np.ndarray,
    t_hours: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    sim: ChitosanRitgerPeppas,
) -> np.ndarray:
    eps = rng.standard_normal((n_samples, U.shape[1]))
    delta = (U @ (eps * sigma).T).T
    theta_samples = np.clip(mu[None, :] + delta, prior_low, prior_high)

    t_eval = np.asarray(t_hours, dtype=np.float32)
    t_full = torch.tensor(np.concatenate([[0.0], t_eval]), dtype=torch.float32)
    theta_t = torch.tensor(theta_samples, dtype=torch.float32)
    with torch.no_grad():
        q_full = sim.simulate(theta_t, t_full).cpu().numpy()
    q_samples = q_full[:, 1:]

    q_point = sim.simulate_numpy(mu, np.concatenate([[0.0], t_eval]))[1:]
    recenter = q_point - q_samples.mean(axis=0)
    return np.clip(q_samples + recenter[None, :], 0.0, 1.2)


def attach_crps(
    per_point: pd.DataFrame,
    theta_targets_path: Path,
    predictions_path: Path,
    n_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    out = per_point.copy()
    out["crps"] = np.nan

    if not theta_targets_path.exists():
        return out, float("nan")

    bundle = np.load(theta_targets_path, allow_pickle=True)
    lock = _load_lock_metadata(predictions_path)
    prior_low = np.asarray(lock.get("prior_low", [-5.0, 0.35, 0.30]), dtype=np.float32)
    prior_high = np.asarray(lock.get("prior_high", [-0.5, 1.0, 1.0]), dtype=np.float32)
    labels = [str(x) for x in bundle["formulations"]]
    label_to_idx = {label: i for i, label in enumerate(labels)}

    sim = ChitosanRitgerPeppas(time_unit_hours=True)
    rng = np.random.default_rng(seed)

    for key, sub in out.groupby(list(REQUIRED_KEYS), sort=True):
        formulation, drug = key
        label = f"{formulation}_{drug}"
        idx = label_to_idx.get(label)
        if idx is None:
            continue

        mu = np.asarray(bundle["mu"][idx], dtype=np.float32)
        U = np.asarray(bundle["U"][idx], dtype=np.float32)
        sigma = np.asarray(bundle["sigma"][idx], dtype=np.float32)
        t_hours = sub["t_hours"].to_numpy(dtype=float)
        shifted_samples = _simulate_shifted_samples(
            mu=mu,
            U=U,
            sigma=sigma,
            t_hours=t_hours,
            prior_low=prior_low,
            prior_high=prior_high,
            n_samples=n_samples,
            rng=rng,
            sim=sim,
        )
        crps_vals = [
            _empirical_crps(shifted_samples[:, j], float(y))
            for j, y in enumerate(sub["Q_observed"].to_numpy(dtype=float))
        ]
        out.loc[sub.index, "crps"] = crps_vals

    return out, float(out["crps"].mean()) if out["crps"].notna().any() else float("nan")


def per_curve_metrics(
    per_point: pd.DataFrame,
    mape_q_floor: float,
    curve_hit_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, sub in per_point.groupby(list(REQUIRED_KEYS), sort=True):
        formulation, drug = key
        y = sub["Q_observed"].to_numpy(dtype=float)
        yh = sub["Q_predicted_point"].to_numpy(dtype=float)
        cov90 = _coverage(y, sub["Q_lo90"].to_numpy(dtype=float), sub["Q_hi90"].to_numpy(dtype=float))
        rows.append(
            {
                "formulation": formulation,
                "drug": drug,
                "n_obs": int(len(sub)),
                "t_min_hours": float(sub["t_hours"].min()),
                "t_max_hours": float(sub["t_hours"].max()),
                "r2": _r2(y, yh),
                "rmse": _rmse(y, yh),
                "mae": _mae(y, yh),
                "mape_q_ge_floor": _mape(y, yh, mape_q_floor),
                "cov90": cov90,
                "cov50": _coverage(y, sub["Q_lo50"].to_numpy(dtype=float), sub["Q_hi50"].to_numpy(dtype=float)),
                "mean_width_90": float(sub["pi_width_90"].mean()),
                "mean_width_50": float(sub["pi_width_50"].mean()),
                "crps_mean": float(sub["crps"].mean()) if sub["crps"].notna().any() else float("nan"),
                "curve_cov90_ge_threshold": bool(np.isfinite(cov90) and cov90 >= curve_hit_threshold),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(
    per_point: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    min_cov90: float,
    curve_hit_threshold: float,
) -> pd.DataFrame:
    y = per_point["Q_observed"].to_numpy(dtype=float)
    yh = per_point["Q_predicted_point"].to_numpy(dtype=float)
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    pooled_r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    endpoint = per_point.sort_values("t_hours").groupby(["formulation", "drug"], as_index=False).tail(1)
    n_curve_hits = int(curve_metrics["curve_cov90_ge_threshold"].sum())
    n_curves = int(len(curve_metrics))
    agg_cov90 = float(per_point["inside90"].mean())
    agg_cov50 = float(per_point["inside50"].mean())
    overall_pass = bool(agg_cov90 >= min_cov90)

    return pd.DataFrame(
        [
            {
                "n_curves": n_curves,
                "aggregate_cov90": agg_cov90,
                "min_cov90": min_cov90,
                "primary_endpoint_pass": overall_pass,
                "aggregate_cov50": agg_cov50,
                "endpoint_cov90": float(endpoint["inside90"].mean()),
                "curve_hit_threshold": curve_hit_threshold,
                "n_curve_cov90_ge_threshold": n_curve_hits,
                "curve_hit_rate_ge_threshold": float(n_curve_hits / n_curves) if n_curves else float("nan"),
                "mean_pi_width_90": float(per_point["pi_width_90"].mean()),
                "mean_pi_width_50": float(per_point["pi_width_50"].mean()),
                "pooled_r2": pooled_r2,
                "crps_mean": float(per_point["crps"].mean()) if per_point["crps"].notna().any() else float("nan"),
            }
        ]
    )


def write_summary(
    out: Path,
    summary_df: pd.DataFrame,
    curves: pd.DataFrame,
    args: argparse.Namespace,
    prereg_source: str,
) -> None:
    summary = summary_df.iloc[0]
    lock = _load_lock_metadata(args.predictions)
    has_lock_rule = bool(lock.get("preregistered_pass_rule"))

    lines = [
        "=== 75 -- chitosan prospective validation reveal ===",
        "",
        "--- locked prediction provenance ---",
        f"  locked commit : {lock.get('git_hash', 'unknown')}",
        f"  locked at UTC : {lock.get('generated_at_utc', 'unknown')}",
        f"  lock script   : {lock.get('script', 'unknown')}",
        "",
        "--- preregistration integrity ---",
        f"  primary endpoint source doc : {prereg_source}",
        f"  pass rule duplicated in lock_metadata.json : {'YES' if has_lock_rule else 'NO'}",
        f"  theta family for CRPS        : {args.theta_targets}",
        "",
        "--- preregistered primary endpoint ---",
        f"  aggregate cov90 >= {args.min_cov90:.2f}",
        "",
        "--- primary result ---",
        f"  curves evaluated : {int(summary['n_curves'])}",
        f"  aggregate cov90  : {summary['aggregate_cov90']:.3f} "
        f"({'PASS' if summary['primary_endpoint_pass'] else 'FAIL'})",
        f"  aggregate cov50  : {summary['aggregate_cov50']:.3f}",
        f"  endpoint cov90   : {summary['endpoint_cov90']:.3f}",
        "",
        "--- secondary / descriptive endpoints ---",
        f"  curve hit rate (cov90 >= {summary['curve_hit_threshold']:.2f}) : "
        f"{int(summary['n_curve_cov90_ge_threshold'])}/{int(summary['n_curves'])} "
        f"({summary['curve_hit_rate_ge_threshold']:.3f})",
        f"  pooled R2                                         : {summary['pooled_r2']:+.3f}",
        f"  mean PI width 90                                  : {summary['mean_pi_width_90']:.3f}",
        f"  mean PI width 50                                  : {summary['mean_pi_width_50']:.3f}",
        f"  CRPS mean                                         : {summary['crps_mean']:.4f}",
        "",
        "--- per-curve metrics ---",
    ]

    for row in curves.sort_values(["drug", "formulation"]).itertuples(index=False):
        lines.append(
            f"  {row.formulation}/{row.drug:<3} "
            f"R2={row.r2:+.3f} RMSE={row.rmse:.3f} "
            f"MAPE={row.mape_q_ge_floor:.3f} cov90={row.cov90:.3f} "
            f"curve_hit={'YES' if row.curve_cov90_ge_threshold else 'NO'}"
        )

    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=Path("outputs/67_chitosan_prospective/predictions.csv"))
    ap.add_argument("--theta-targets", type=Path, default=Path("outputs/67_chitosan_prospective/theta_targets.npz"))
    ap.add_argument("--preregistration-doc", type=Path, default=DEFAULT_PREREG_DOC)
    ap.add_argument("--observed-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/75_chitosan_prospective_eval"))
    ap.add_argument("--write-template", action="store_true")
    ap.add_argument("--mape-q-floor", type=float, default=0.05)
    ap.add_argument("--curve-hit-threshold", type=float, default=DEFAULT_CURVE_HIT_THRESHOLD)
    ap.add_argument("--min-cov90", type=float, default=None)
    ap.add_argument("--crps-samples", type=int, default=DEFAULT_CRPS_SAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    prereg_min_cov90, prereg_source = _load_preregistered_min_cov90(args.preregistration_doc)
    if args.min_cov90 is None:
        args.min_cov90 = prereg_min_cov90
    elif args.min_cov90 != prereg_min_cov90:
        print("[WARN] Non-default --min-cov90 provided; primary gate is no longer the preregistered default.")
        print(f"       Requested {args.min_cov90:.3f}; preregistered value is {prereg_min_cov90:.3f} from {prereg_source}")

    args.out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)

    if args.write_template:
        template = args.out / "observed_template.csv"
        write_template(pred, template)
        print(f"Wrote observed-data template: {template}")
        if args.observed_csv is None:
            return

    if args.observed_csv is None:
        raise SystemExit("Provide --observed-csv or use --write-template.")

    raw_obs = pd.read_csv(args.observed_csv)
    obs = _standardize_observed(raw_obs)
    per_point = interpolate_locked_predictions(pred, obs)
    per_point, _ = attach_crps(
        per_point=per_point,
        theta_targets_path=args.theta_targets,
        predictions_path=args.predictions,
        n_samples=args.crps_samples,
        seed=args.seed,
    )
    curves = per_curve_metrics(
        per_point=per_point,
        mape_q_floor=args.mape_q_floor,
        curve_hit_threshold=args.curve_hit_threshold,
    )
    summary_df = aggregate_metrics(
        per_point=per_point,
        curve_metrics=curves,
        min_cov90=args.min_cov90,
        curve_hit_threshold=args.curve_hit_threshold,
    )

    per_point.to_csv(args.out / "per_point.csv", index=False)
    curves.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary_df.to_csv(args.out / "gate_summary.csv", index=False)
    write_summary(args.out, summary_df, curves, args, prereg_source)
    print((args.out / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
