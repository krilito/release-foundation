"""
71 - Nature-family gate diagnostics.

What this does:
    Consolidates the first hard go/no-go checks for a Nature-family
    manuscript route:

      G1. Is the kinetic middle layer real under paired bootstrap?
          Recompute theta->ODE minus direct-Q gains from per-curve outputs,
          not just summary tables.

      G2. Do classical sparse-curve models already solve the 1/2-point
          forecasting task?
          Fit simple Higuchi, Korsmeyer-Peppas, Weibull, and linear sparse
          baselines using only early interpolated observations, then evaluate
          on the full/future observed curve.

      G3. Do R5/R6 regime-conditional prototypes clear the existing gate?
          Parse script 36 summaries into a compact manuscript-routing table.

Inputs:
    Existing outputs from scripts 36, 45, 46, 59, and 62.
    Raw PLGA curve sources used by scripts 41/42 for the classical baselines.

Outputs:
    outputs/71_nature_gate_diagnostics/g1_middle_layer_bootstrap.csv
    outputs/71_nature_gate_diagnostics/g1_lgbm_bootstrap.csv
    outputs/71_nature_gate_diagnostics/g2_classical_sparse_per_curve.csv
    outputs/71_nature_gate_diagnostics/g2_classical_sparse_summary.csv
    outputs/71_nature_gate_diagnostics/g3_regime_gate_summary.csv
    outputs/71_nature_gate_diagnostics/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _interp_at(t_obs: np.ndarray, q_obs: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.interp(times, t_obs, q_obs, left=q_obs[0], right=q_obs[-1])


def _bootstrap_gain(
    df: pd.DataFrame,
    direct_col: str,
    theta_col: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    vals = df[[direct_col, theta_col]].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return {
            "n_curves": 0,
            "gain": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_gain_le_0": float("nan"),
        }
    direct = vals[:, 0]
    theta = vals[:, 1]
    gain = float(np.median(theta) - np.median(direct))
    boot = np.empty(n_boot, dtype=float)
    n = len(vals)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.median(theta[idx]) - np.median(direct[idx]))
    return {
        "n_curves": int(n),
        "gain": gain,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_gain_le_0": float(np.mean(boot <= 0.0)),
    }


def _theta_per_curve_path(root: Path, dataset: str, input_mode: str) -> Path:
    return root / f"{dataset}_{input_mode}" / "per_curve.csv"


def g1_rf_et_bootstrap(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    direct = pd.read_csv(args.direct_per_curve)
    best = pd.read_csv(args.best_route_summary)
    rows: list[dict[str, object]] = []

    for _, cell in best.iterrows():
        dataset = str(cell["dataset"])
        input_mode = str(cell["input_mode"])
        scheme = str(cell["scheme"])
        direct_method = str(cell["direct_best_method"])
        theta_method = str(cell["theta_best_method"])
        direct_col = f"r2_{direct_method}"
        theta_col = f"r2_{theta_method}"
        theta_path = _theta_per_curve_path(args.theta_root, dataset, input_mode)
        theta = pd.read_csv(theta_path)
        sub_direct = direct[
            (direct["dataset"] == dataset)
            & (direct["input_mode"] == input_mode)
            & (direct["scheme"] == scheme)
        ].copy()
        sub_theta = theta[theta["scheme"] == scheme].copy()
        missing = [c for c in [direct_col] if c not in sub_direct.columns]
        missing += [c for c in [theta_col] if c not in sub_theta.columns]
        if missing:
            raise KeyError(f"missing columns for {dataset}/{input_mode}/{scheme}: {missing}")
        merged = sub_direct[["scheme", "fold", "fid", direct_col]].merge(
            sub_theta[["scheme", "fold", "fid", theta_col]],
            on=["scheme", "fold", "fid"],
            how="inner",
        )
        stats = _bootstrap_gain(merged, direct_col, theta_col, args.n_boot, rng)
        rows.append({
            "family": "RF_ET_best_route",
            "dataset": dataset,
            "input_mode": input_mode,
            "scheme": scheme,
            "direct_method": direct_method,
            "theta_method": theta_method,
            **stats,
            "gate_ci_low_gt_0": bool(stats["ci_low"] > 0.0),
        })
    return pd.DataFrame(rows)


def g1_lgbm_bootstrap(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    per = pd.read_csv(args.lgbm_per_curve)
    cells = pd.read_csv(args.lgbm_gain_summary)
    rows: list[dict[str, object]] = []
    for _, cell in cells.iterrows():
        dataset = str(cell["dataset"])
        input_mode = str(cell["input_mode"])
        scheme = str(cell["scheme"])
        theta_method = str(cell["theta_best_method"])
        direct_col = "r2_LGBM_direct_Q"
        theta_col = f"r2_{theta_method}"
        sub = per[
            (per["dataset"] == dataset)
            & (per["input_mode"] == input_mode)
            & (per["scheme"] == scheme)
        ].copy()
        missing = [c for c in [direct_col, theta_col] if c not in sub.columns]
        if missing:
            raise KeyError(f"missing LGBM columns for {dataset}/{input_mode}/{scheme}: {missing}")
        stats = _bootstrap_gain(sub, direct_col, theta_col, args.n_boot, rng)
        rows.append({
            "family": "LGBM",
            "dataset": dataset,
            "input_mode": input_mode,
            "scheme": scheme,
            "direct_method": "LGBM_direct_Q",
            "theta_method": theta_method,
            **stats,
            "gate_ci_low_gt_0": bool(stats["ci_low"] > 0.0),
        })
    return pd.DataFrame(rows)


def _predict_constant(t: np.ndarray, q_early: np.ndarray) -> np.ndarray:
    return np.full_like(t, float(q_early[-1]), dtype=float)


def _poly_fit_predict(t_fit: np.ndarray, q_fit: np.ndarray, t_eval: np.ndarray, transform: str) -> np.ndarray:
    if transform == "time":
        x_fit = t_fit
        x_eval = t_eval
    elif transform == "sqrt_time":
        x_fit = np.sqrt(np.maximum(t_fit, 1e-9))
        x_eval = np.sqrt(np.maximum(t_eval, 1e-9))
    else:
        raise ValueError(transform)

    if len(t_fit) < 2:
        slope = q_fit[0] / max(float(x_fit[0]), 1e-9)
        pred = slope * x_eval
    else:
        coef = np.polyfit(x_fit, q_fit, deg=1)
        pred = np.polyval(coef, x_eval)
    return np.clip(pred, 0.0, 1.0)


def _kp_predict(t_fit: np.ndarray, q_fit: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    mask = (t_fit > 0) & (q_fit > 1e-6) & (q_fit < 0.999)
    if mask.sum() >= 2:
        x = np.log(t_fit[mask])
        y = np.log(q_fit[mask])
        n, log_k = np.polyfit(x, y, deg=1)
        n = float(np.clip(n, 0.05, 1.5))
        k = float(np.clip(math.exp(log_k), 1e-6, 2.0))
    else:
        return np.full_like(t_eval, np.nan, dtype=float)
    return np.clip(k * np.maximum(t_eval, 0.0) ** n, 0.0, 1.0)


def _weibull_qmax1_predict(t_fit: np.ndarray, q_fit: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    t_fit = np.maximum(t_fit.astype(float), 1e-6)
    q_fit = np.clip(q_fit.astype(float), 1e-6, 0.999)
    if len(t_fit) == 1:
        beta = 1.0
        tau = float(t_fit[0] / max(-math.log(1.0 - q_fit[0]), 1e-6))
    else:
        def residual(z: np.ndarray) -> np.ndarray:
            log_tau, log_beta = z
            tau = math.exp(float(log_tau))
            beta = math.exp(float(log_beta))
            pred = 1.0 - np.exp(-((t_fit / tau) ** beta))
            return pred - q_fit

        z0 = np.array([math.log(max(float(np.median(t_fit)), 1e-3)), math.log(1.0)])
        res = least_squares(
            residual,
            z0,
            bounds=([math.log(1e-3), math.log(0.1)], [math.log(1000.0), math.log(5.0)]),
            max_nfev=200,
        )
        tau = math.exp(float(res.x[0]))
        beta = math.exp(float(res.x[1]))
    return np.clip(1.0 - np.exp(-((np.maximum(t_eval, 0.0) / tau) ** beta)), 0.0, 1.0)


def _gp_predict(t_fit: np.ndarray, q_fit: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=7.0, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        optimizer=None,
        random_state=0,
    )
    gp.fit(t_fit.reshape(-1, 1), q_fit)
    pred = gp.predict(t_eval.reshape(-1, 1))
    return np.clip(pred, 0.0, 1.0)


def _classical_predictions(t_fit: np.ndarray, q_fit: np.ndarray, t_eval: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "persistence": _predict_constant(t_eval, q_fit),
        "linear_time": _poly_fit_predict(t_fit, q_fit, t_eval, "time"),
        "higuchi": _poly_fit_predict(t_fit, q_fit, t_eval, "sqrt_time"),
        "korsmeyer_peppas": _kp_predict(t_fit, q_fit, t_eval),
        "weibull_qmax1": _weibull_qmax1_predict(t_fit, q_fit, t_eval),
        "gp_rbf": _gp_predict(t_fit, q_fit, t_eval),
    }


def _load_curve_records_for_g2(args: argparse.Namespace) -> list[dict[str, object]]:
    mod41 = _load_script(Path("scripts/41_theta_target_scaling_audit.py"), "script41_for_71")
    mod42 = _load_script(Path("scripts/42_internal181_tree_theta_audit.py"), "script42_for_71")

    common = SimpleNamespace(
        cross_doi_data=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        full_fit_bank=args.full_fit_bank,
        regime_csv=args.regime_csv,
        t_grid_max_days=args.t_grid_max_days,
        early_times=[1.0, 3.0, 5.0, 7.0],
    )
    _, cross_map, *_ = mod41._load_dataset(common)

    desc, internal_map, _ = mod42._load_clean_internal(
        data_csv=args.internal_data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )
    keep_internal = set(desc[mod42.INTERNAL_FID_COL].astype(int).tolist())

    out: list[dict[str, object]] = []
    for fid, curve in cross_map.items():
        out.append({
            "dataset": "cross321",
            "fid": int(fid),
            "t_obs": curve.t_obs.astype(float),
            "q_obs": curve.q_obs.astype(float),
        })
    for fid, curve in internal_map.items():
        if int(fid) not in keep_internal:
            continue
        out.append({
            "dataset": "internal181",
            "fid": int(fid),
            "t_obs": curve.t_obs.astype(float),
            "q_obs": curve.q_obs.astype(float),
        })
    return out


def _bootstrap_median_ci(vals: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    boot = np.empty(n_boot, dtype=float)
    n = len(vals)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.median(vals[idx]))
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def g2_classical_sparse(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = {
        "one_point_1d": np.array([1.0], dtype=float),
        "two_point_1_3d": np.array([1.0, 3.0], dtype=float),
        "two_point_1_7d": np.array([1.0, 7.0], dtype=float),
        "four_point_1_3_5_7d": np.array([1.0, 3.0, 5.0, 7.0], dtype=float),
    }
    rows: list[dict[str, object]] = []
    for rec in _load_curve_records_for_g2(args):
        t_obs = rec["t_obs"]
        q_obs = rec["q_obs"]
        for window_name, early_times in windows.items():
            q_early = _interp_at(t_obs, q_obs, early_times)
            cutoff = float(np.max(early_times))
            future_mask = t_obs > cutoff
            if future_mask.sum() < 2:
                continue
            preds = _classical_predictions(early_times, q_early, t_obs)
            for method, pred in preds.items():
                rows.append({
                    "dataset": rec["dataset"],
                    "fid": rec["fid"],
                    "window": window_name,
                    "method": method,
                    "n_fit_points": int(len(early_times)),
                    "full_r2": _r2(q_obs, pred),
                    "future_r2": _r2(q_obs[future_mask], pred[future_mask]),
                    "future_rmse": _rmse(q_obs[future_mask], pred[future_mask]),
                    "q_at_cutoff": float(q_early[-1]),
                    "t_max": float(np.max(t_obs)),
                })
    per_curve = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for keys, sub in per_curve.groupby(["dataset", "window", "method"], sort=True):
        dataset, window, method = keys
        vals = sub["future_r2"].dropna().to_numpy(dtype=float)
        full_vals = sub["full_r2"].dropna().to_numpy(dtype=float)
        rmse = sub["future_rmse"].dropna().to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_median_ci(vals, args.n_boot, rng)
        if len(vals) == 0:
            median = mean = p25 = p10 = frac = float("nan")
        else:
            median = float(np.median(vals))
            mean = float(np.mean(vals))
            p25 = float(np.percentile(vals, 25))
            p10 = float(np.percentile(vals, 10))
            frac = float(np.mean(vals >= 0.85))
        summary_rows.append({
            "dataset": dataset,
            "window": window,
            "method": method,
            "n": int(len(vals)),
            "future_r2_median": median,
            "future_r2_ci_low": ci_low,
            "future_r2_ci_high": ci_high,
            "future_r2_mean": mean,
            "future_r2_p25": p25,
            "future_r2_p10": p10,
            "future_frac_above_0.85": frac,
            "full_r2_median": float(np.median(full_vals)) if len(full_vals) else float("nan"),
            "future_rmse_median": float(np.median(rmse)) if len(rmse) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    return per_curve, summary


def _parse_regime_summary(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    first = re.search(r"=== 36 -- R(\d+) regime-conditional prototype ===", text)
    regime = int(first.group(1)) if first else -1

    def _median(label: str) -> float:
        pat = rf"{re.escape(label)}\s+([-+]?\d+\.\d+)"
        match = re.search(pat, text)
        return float(match.group(1)) if match else float("nan")

    return {
        "regime": f"R{regime}",
        "oracle9_median": _median("(a) oracle 9"),
        "greedy4_median": _median("(b) greedy 4 (per-curve)"),
        "regime_cond4_median": _median("(c) R6-cond 4"),
        "zero_shot_median": _median("(d) R6 zero-shot"),
        "pass_count": int(text.count("PASS")),
        "fail_count": int(text.count("FAIL")),
        "raw_verdict": "partial" if "PARTIAL PASS" in text else (
            "pass" if "SINGLE-REGIME PROTOTYPE WORKS" in text else "fail"
        ),
    }


def g3_regime_gate(args: argparse.Namespace) -> pd.DataFrame:
    rows = [_parse_regime_summary(path) for path in args.regime_summaries]
    df = pd.DataFrame(rows)
    df["near_oracle_delta"] = df["regime_cond4_median"] - df["oracle9_median"]
    df["vs_greedy_delta"] = df["regime_cond4_median"] - df["greedy4_median"]
    df["gate_median_ge_0.95"] = df["regime_cond4_median"] >= 0.95
    df["gate_near_oracle_0.01"] = df["near_oracle_delta"].abs() <= 0.01
    return df


def _best_classical(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.groupby(["dataset", "window"])["future_r2_median"].idxmax()
    return summary.loc[idx].sort_values(["dataset", "window"]).reset_index(drop=True)


def write_summary(
    out: Path,
    g1: pd.DataFrame,
    g1_lgbm: pd.DataFrame,
    g2_summary: pd.DataFrame,
    g3: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "=== 71 -- Nature-family gate diagnostics ===",
        "",
        "G1: middle-layer gain bootstrap",
        "  Pass rule for manuscript headline: formulation_plus_early cells should",
        "  have bootstrap 95% CI lower bound > 0, especially OOD splits.",
        "",
        "--- RF/ET best-route formulation_plus_early ---",
    ]
    g1_focus = g1[g1["input_mode"] == "formulation_plus_early"].copy()
    for _, r in g1_focus.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {r['dataset']:<11} {r['scheme']:<16} "
            f"gain={r['gain']:+.4f} CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
            f"p<=0={r['p_gain_le_0']:.3f} pass={bool(r['gate_ci_low_gt_0'])}"
        )
    lines.append("")
    lines.append("--- LGBM formulation_plus_early ---")
    lgbm_focus = g1_lgbm[g1_lgbm["input_mode"] == "formulation_plus_early"].copy()
    for _, r in lgbm_focus.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {r['dataset']:<11} {r['scheme']:<16} "
            f"gain={r['gain']:+.4f} CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
            f"p<=0={r['p_gain_le_0']:.3f} pass={bool(r['gate_ci_low_gt_0'])}"
        )

    best_classic = _best_classical(g2_summary)
    lines.extend([
        "",
        "G2: classical sparse-curve baseline",
        "  Red flag rule: if a simple classical model reaches future-R2 median",
        "  >= 0.85 with only 2 early points, the paper cannot claim that sparse",
        "  forecasting itself is hard without qualification.",
        "",
        "--- best classical model by dataset/window (future-only R2) ---",
    ])
    for _, r in best_classic.iterrows():
        flag = "RED_FLAG" if r["future_r2_median"] >= 0.85 and "two_point" in str(r["window"]) else ""
        lines.append(
            f"  {r['dataset']:<11} {r['window']:<22} {r['method']:<18} "
            f"median={r['future_r2_median']:+.4f} "
            f"CI=[{r['future_r2_ci_low']:+.4f},{r['future_r2_ci_high']:+.4f}] "
            f"p25={r['future_r2_p25']:+.4f} "
            f"frac>=.85={r['future_frac_above_0.85']:.2f} {flag}"
        )

    lines.extend([
        "",
        "G3: R5/R6 regime-conditional prototypes",
        "  Pass rule: regime-cond4 median >= 0.95 and within 0.01 of oracle9.",
        "",
    ])
    for _, r in g3.sort_values("regime").iterrows():
        lines.append(
            f"  {r['regime']:<3} cond4={r['regime_cond4_median']:.4f} "
            f"oracle={r['oracle9_median']:.4f} delta={r['near_oracle_delta']:+.4f} "
            f"vs_greedy={r['vs_greedy_delta']:+.4f} "
            f"median>=.95={bool(r['gate_median_ge_0.95'])} "
            f"near_oracle={bool(r['gate_near_oracle_0.01'])} "
            f"verdict={r['raw_verdict']}"
        )

    lines.extend([
        "",
        "--- routing verdict template ---",
        "  If G1 and G3 pass but G2 shows classical 2-point models are strong,",
        "  frame the paper around calibrated mechanism-state/UQ and decision",
        "  support, not raw sparse forecasting accuracy.",
        "  If G1 fails, the middle-layer claim must be demoted.",
        "  If G3 fails, regime discreteness becomes a diagnostic observation,",
        "  not the main scientific discovery.",
    ])
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("outputs/71_nature_gate_diagnostics"))

    ap.add_argument("--theta-root", type=Path, default=Path("outputs/45_input_source_ablation"))
    ap.add_argument("--direct-per-curve", type=Path, default=Path("outputs/46_direct_curve_rf_input_ablation/per_curve.csv"))
    ap.add_argument("--best-route-summary", type=Path, default=Path("outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv"))
    ap.add_argument("--lgbm-per-curve", type=Path, default=Path("outputs/62_lgbm_middle_layer_gain/per_curve.csv"))
    ap.add_argument("--lgbm-gain-summary", type=Path, default=Path("outputs/62_lgbm_middle_layer_gain/middle_layer_gain.csv"))

    ap.add_argument("--cross-doi-data", type=Path, default=Path(
        r"D:\chemical-world-model-v0\datset\321PLGA"
        r"\A Dataset on Formulation Parameters and Characteristics of "
        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
    ))
    ap.add_argument("--matched-fids-csv", type=Path, default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"))
    ap.add_argument("--internal-data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument("--full-fit-bank", type=Path, default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"))
    ap.add_argument("--regime-csv", type=Path, default=Path("outputs/33_active_set_regimes/regime_assignments.csv"))
    ap.add_argument("--fit-filter-r2", type=float, default=0.9)
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument(
        "--regime-summaries",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/36_r5_regime_prototype/summary.txt"),
            Path("outputs/36_r6_regime_prototype/summary.txt"),
        ],
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    g1 = g1_rf_et_bootstrap(args, rng)
    g1_lgbm = g1_lgbm_bootstrap(args, rng)
    g2_per_curve, g2_summary = g2_classical_sparse(args, rng)
    g3 = g3_regime_gate(args)

    g1.to_csv(args.out / "g1_middle_layer_bootstrap.csv", index=False)
    g1_lgbm.to_csv(args.out / "g1_lgbm_bootstrap.csv", index=False)
    g2_per_curve.to_csv(args.out / "g2_classical_sparse_per_curve.csv", index=False)
    g2_summary.to_csv(args.out / "g2_classical_sparse_summary.csv", index=False)
    g3.to_csv(args.out / "g3_regime_gate_summary.csv", index=False)
    write_summary(args.out, g1, g1_lgbm, g2_summary, g3)
    print((args.out / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
