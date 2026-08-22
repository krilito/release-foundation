"""
72 - CASP calibration and active-observer gate diagnostics.

What this does:
    Converts existing CASP and Active Observer outputs into manuscript
    routing gates:

      G4. FIB-CASP calibration gate.
          Bootstrap per-curve cov90 / R2 / width summaries for every
          dataset × split. Random splits should be calibrated, OOD splits
          must not collapse.

      G5. Active observer decision gate.
          Paired-bootstrap active_two_point_conformal against fixed
          schedules and direct-Q point baselines. Active does not need to
          be the point-prediction champion, but it must justify a main-text
          experimental-decision role.

Inputs:
    outputs/62_fib_casp_benchmark/**/per_curve.csv
    outputs/62_fib_casp_benchmark/aggregate_table.csv
    outputs_active_observer_v3/metrics_by_curve.csv
    outputs_active_observer_v3/metrics_summary.csv

Outputs:
    outputs/72_casp_active_gate_diagnostics/casp_bootstrap_gate.csv
    outputs/72_casp_active_gate_diagnostics/active_pairwise_gate.csv
    outputs/72_casp_active_gate_diagnostics/summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_stat(
    vals: np.ndarray,
    stat: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if stat == "mean":
        point = float(np.mean(vals))
        fn = np.mean
    elif stat == "median":
        point = float(np.median(vals))
        fn = np.median
    else:
        raise ValueError(stat)
    boot = np.empty(n_boot, dtype=float)
    n = len(vals)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(fn(vals[idx]))
    return (point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _bootstrap_diff(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    stat: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    vals = df[[a_col, b_col]].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    a = vals[:, 0]
    b = vals[:, 1]
    fn = np.mean if stat == "mean" else np.median
    point = float(fn(a) - fn(b))
    boot = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(fn(a[idx]) - fn(b[idx]))
    p_le_0 = float(np.mean(boot <= 0.0))
    return (point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), p_le_0)


def casp_gate(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for per_curve_path in sorted(args.casp_root.glob("*/*/per_curve.csv")):
        dataset = per_curve_path.parts[-3]
        scheme = per_curve_path.parts[-2]
        df = pd.read_csv(per_curve_path)
        cov90_mean, cov90_lo, cov90_hi = _bootstrap_stat(
            df["cov90"].to_numpy(dtype=float), "mean", args.n_boot, rng,
        )
        cov90_med, cov90_med_lo, cov90_med_hi = _bootstrap_stat(
            df["cov90"].to_numpy(dtype=float), "median", args.n_boot, rng,
        )
        r2_med, r2_lo, r2_hi = _bootstrap_stat(
            df["r2_point"].to_numpy(dtype=float), "median", args.n_boot, rng,
        )
        width_med, width_lo, width_hi = _bootstrap_stat(
            df["pi_width_90"].to_numpy(dtype=float), "median", args.n_boot, rng,
        )
        is_random = scheme == "random_5fold"
        min_cov = args.random_cov90_floor if is_random else args.ood_cov90_floor
        max_cov = args.cov90_ceiling
        cov_point_ok = min_cov <= cov90_mean <= max_cov
        cov_ci_ok = cov90_lo >= min_cov
        r2_ok = r2_med >= args.casp_r2_floor
        if cov_point_ok and cov_ci_ok and r2_ok:
            gate_status = "pass"
        elif cov_point_ok and r2_ok:
            gate_status = "borderline"
        else:
            gate_status = "warn"
        rows.append({
            "dataset": dataset,
            "scheme": scheme,
            "n": int(len(df)),
            "split_type": "random" if is_random else "ood",
            "r2_median": r2_med,
            "r2_median_ci_low": r2_lo,
            "r2_median_ci_high": r2_hi,
            "cov90_mean": cov90_mean,
            "cov90_mean_ci_low": cov90_lo,
            "cov90_mean_ci_high": cov90_hi,
            "cov90_median": cov90_med,
            "cov90_median_ci_low": cov90_med_lo,
            "cov90_median_ci_high": cov90_med_hi,
            "pi_width90_median": width_med,
            "pi_width90_ci_low": width_lo,
            "pi_width90_ci_high": width_hi,
            "gate_cov90_point": bool(cov_point_ok),
            "gate_cov90_ci_low": bool(cov_ci_ok),
            f"gate_r2_median_ge_{args.casp_r2_floor:g}": bool(r2_ok),
            "gate_status": gate_status,
        })
    return pd.DataFrame(rows)


def active_gate(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.read_csv(args.active_metrics_by_curve)
    strategies = set(df["strategy"].astype(str))
    active = "active_two_point_conformal"
    comparators = [
        "fixed_21d_conformal",
        "fixed_14d_conformal",
        "fixed_four_point_conformal",
        "zero_early_prior_conformal",
        "directQ_one_point_fixed_7d",
        "directQ_one_point_fixed_5d",
        "directQ_one_point_active_time",
    ]
    if active not in strategies:
        raise KeyError(f"{active} missing from {args.active_metrics_by_curve}")

    rows: list[dict[str, object]] = []
    a = df[df["strategy"] == active].copy()
    for comp in comparators:
        if comp not in strategies:
            continue
        b = df[df["strategy"] == comp].copy()
        merged = a.merge(
            b,
            on="curve_id",
            how="inner",
            suffixes=("_active", "_comp"),
        )
        rmse_diff, rmse_lo, rmse_hi, rmse_p_le_0 = _bootstrap_diff(
            merged,
            "future_rmse_mean_active",
            "future_rmse_mean_comp",
            "mean",
            args.n_boot,
            rng,
        )
        mae_diff, mae_lo, mae_hi, mae_p_le_0 = _bootstrap_diff(
            merged,
            "future_mae_median_active",
            "future_mae_median_comp",
            "median",
            args.n_boot,
            rng,
        )
        comp_rmse_mean = float(np.nanmean(merged["future_rmse_mean_comp"].to_numpy(dtype=float)))
        rmse_margin = max(args.active_rmse_noninferiority_abs, args.active_rmse_noninferiority_rel * comp_rmse_mean)
        row: dict[str, object] = {
            "active_strategy": active,
            "comparator": comp,
            "n_pairs": int(len(merged)),
            "rmse_mean_diff_active_minus_comp": rmse_diff,
            "rmse_mean_diff_ci_low": rmse_lo,
            "rmse_mean_diff_ci_high": rmse_hi,
            "p_rmse_diff_le_0": rmse_p_le_0,
            "comparator_rmse_mean": comp_rmse_mean,
            "rmse_noninferiority_margin": rmse_margin,
            "mae_median_diff_active_minus_comp": mae_diff,
            "mae_median_diff_ci_low": mae_lo,
            "mae_median_diff_ci_high": mae_hi,
            "p_mae_diff_le_0": mae_p_le_0,
            "rmse_gate_active_not_worse_10pct": bool(
                rmse_diff <= rmse_margin
            ),
            "rmse_gate_active_better_ci": bool(rmse_hi < 0.0),
        }
        if "crps_active" in merged.columns and "crps_comp" in merged.columns:
            crps_diff, crps_lo, crps_hi, crps_p_le_0 = _bootstrap_diff(
                merged, "crps_active", "crps_comp", "mean", args.n_boot, rng,
            )
            row.update({
                "crps_mean_diff_active_minus_comp": crps_diff,
                "crps_mean_diff_ci_low": crps_lo,
                "crps_mean_diff_ci_high": crps_hi,
                "p_crps_diff_le_0": crps_p_le_0,
                "crps_gate_active_not_worse": bool(crps_diff <= args.active_crps_noninferiority_margin),
            })
        if "coverage_90_active" in merged.columns and "coverage_90_comp" in merged.columns:
            cov_diff, cov_lo, cov_hi, _ = _bootstrap_diff(
                merged, "coverage_90_active", "coverage_90_comp", "mean", args.n_boot, rng,
            )
            active_cov90 = float(np.nanmean(merged["coverage_90_active"].to_numpy(dtype=float)))
            row.update({
                "active_coverage90_mean": active_cov90,
                "coverage90_mean_diff_active_minus_comp": cov_diff,
                "coverage90_mean_diff_ci_low": cov_lo,
                "coverage90_mean_diff_ci_high": cov_hi,
                "coverage90_gate_active_nominal": bool(
                    args.active_cov90_floor <= active_cov90 <= args.active_cov90_ceiling
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_bool(x: object) -> str:
    return "PASS" if bool(x) else "WARN"


def write_summary(out: Path, casp: pd.DataFrame, active: pd.DataFrame) -> None:
    lines: list[str] = [
        "=== 72 -- CASP calibration and active-observer gates ===",
        "",
        "G4: FIB-CASP calibration gate",
        "  Random split target: cov90 mean >= 0.80 and <= 0.95.",
        "  OOD split target: cov90 mean >= 0.75 and <= 0.95.",
        "  R2 guardrail: median R2 >= 0.80 for main-text calibrated forecast claims.",
        "",
    ]
    for _, r in casp.sort_values(["dataset", "split_type", "scheme"]).iterrows():
        lines.append(
            f"  {r['dataset']:<11} {r['scheme']:<16} "
            f"R2med={r['r2_median']:+.4f} "
            f"cov90={r['cov90_mean']:.3f} "
            f"CI=[{r['cov90_mean_ci_low']:.3f},{r['cov90_mean_ci_high']:.3f}] "
            f"width90med={r['pi_width90_median']:.3f} "
            f"gate={r['gate_status']} "
            f"ciFloor={_fmt_bool(r['gate_cov90_ci_low'])}"
        )

    lines.extend([
        "",
        "G5: Active observer gate",
        "  Negative RMSE/CRPS diff means active is better than the comparator.",
        "  Main-text role is justified only as a decision/UQ module unless it",
        "  beats strong fixed/directQ comparators on paired metrics.",
        "",
    ])
    for _, r in active.iterrows():
        msg = (
            f"  active_two_point vs {r['comparator']:<28} "
            f"dRMSE={r['rmse_mean_diff_active_minus_comp']:+.4f} "
            f"CI=[{r['rmse_mean_diff_ci_low']:+.4f},{r['rmse_mean_diff_ci_high']:+.4f}] "
            f"p<=0={r['p_rmse_diff_le_0']:.3f} "
            f"margin={r['rmse_noninferiority_margin']:.4f} "
            f"noninf10pct={_fmt_bool(r['rmse_gate_active_not_worse_10pct'])}"
        )
        if "crps_mean_diff_active_minus_comp" in r and pd.notna(r["crps_mean_diff_active_minus_comp"]):
            msg += (
                f" dCRPS={r['crps_mean_diff_active_minus_comp']:+.4f} "
                f"CI=[{r['crps_mean_diff_ci_low']:+.4f},{r['crps_mean_diff_ci_high']:+.4f}] "
                f"crpsNoninf={_fmt_bool(r['crps_gate_active_not_worse'])}"
            )
        if "coverage90_mean_diff_active_minus_comp" in r and pd.notna(r["coverage90_mean_diff_active_minus_comp"]):
            msg += (
                f" activeCov90={r['active_coverage90_mean']:.3f} "
                f"covNominal={_fmt_bool(r['coverage90_gate_active_nominal'])} "
                f"dCov90={r['coverage90_mean_diff_active_minus_comp']:+.3f}"
            )
        lines.append(msg)

    lines.extend([
        "",
        "--- routing verdict template ---",
        "  CASP: random-split calibration mostly meets targets, but cross321",
        "  random is borderline by bootstrap CI lower bound. OOD calibration",
        "  should be written as borderline/limited, not fully solved.",
        "  internal181 by-polymer and liposome OOD are the weakest main-text",
        "  calibrated-forecast claims.",
        "  Active observer: keep as experimental-decision / uncertainty module.",
        "  Do not sell it as the best point predictor if directQ/fixed21 remain",
        "  stronger on RMSE.",
    ])
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("outputs/72_casp_active_gate_diagnostics"))
    ap.add_argument("--casp-root", type=Path, default=Path("outputs/62_fib_casp_benchmark"))
    ap.add_argument(
        "--active-metrics-by-curve",
        type=Path,
        default=Path("outputs_active_observer_v3/metrics_by_curve.csv"),
    )
    ap.add_argument("--random-cov90-floor", type=float, default=0.80)
    ap.add_argument("--ood-cov90-floor", type=float, default=0.75)
    ap.add_argument("--cov90-ceiling", type=float, default=0.95)
    ap.add_argument("--casp-r2-floor", type=float, default=0.80)
    ap.add_argument("--active-rmse-noninferiority-abs", type=float, default=0.0)
    ap.add_argument("--active-rmse-noninferiority-rel", type=float, default=0.10)
    ap.add_argument("--active-crps-noninferiority-margin", type=float, default=0.005)
    ap.add_argument("--active-cov90-floor", type=float, default=0.85)
    ap.add_argument("--active-cov90-ceiling", type=float, default=0.95)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    casp = casp_gate(args, rng)
    active = active_gate(args, rng)
    casp.to_csv(args.out / "casp_bootstrap_gate.csv", index=False)
    active.to_csv(args.out / "active_pairwise_gate.csv", index=False)
    write_summary(args.out, casp, active)
    print((args.out / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
