"""
118 - Synthetic hidden-state identifiability audit.

Purpose:
    Build a controlled synthetic PLGA-like release world where hidden
    microstructure/process variables are known. Test whether common static
    descriptors fail for the same reason as real release corpora: they do not
    contain the release state, while early Q(t) observations partially recover it.

Consumes:
    None. This is a diagnostic simulation.

Produces:
    outputs/118_synthetic_hidden_state_identifiability_audit/
      per_curve_metrics.csv
      summary_by_input.csv
      hidden_recovery.csv
      gap_closure.csv
      hidden_strength_sensitivity.csv (when --run-hidden-strength-sensitivity is used)
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This is not a deployable predictor and not a new model family. It is a
    release-state inference sanity check: if hidden state drives release, then
    X-only prediction should be bounded, true hidden variables should close the
    gap, and early observations should act as noisy state measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_OUT = Path("outputs/118_synthetic_hidden_state_identifiability_audit")

CORE_FEATURES = [
    "drug_mw",
    "drug_logp",
    "drug_tpsa",
    "plga_mw_kda",
    "la_ga_ratio",
    "drug_loading_pct",
    "particle_size_um",
]

CONTEXT_FEATURES = [
    "medium_pH",
    "temperature_c",
    "agitation_rpm",
    "solvent_strength",
    "surfactant_pct",
]

HIDDEN_FEATURES = [
    "porosity",
    "surface_drug_fraction",
    "particle_size_cv",
    "residual_solvent",
    "mw_dispersion",
    "tortuosity",
    "water_uptake_rate",
    "autocatalysis_strength",
]

PROXY_FEATURES = [f"proxy_{name}" for name in HIDDEN_FEATURES]
MODEL_NAMES = ("extra_trees", "ridge")
EARLY_OBS_DAYS = np.asarray([0.25, 1.0, 3.0, 7.0, 14.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic hidden-state release audit.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-curves", type=int, default=900)
    parser.add_argument("--n-sources", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--max-time-days", type=float, default=90.0)
    parser.add_argument("--noise-sigma", type=float, default=0.025)
    parser.add_argument("--hidden-strength", type=float, default=1.0)
    parser.add_argument("--source-shift", type=float, default=0.55)
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument(
        "--run-hidden-strength-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a lightweight low/medium/high hidden-state sensitivity defense table.",
    )
    parser.add_argument("--sensitivity-hidden-strengths", type=float, nargs="+", default=[0.4, 1.0, 1.8])
    parser.add_argument("--sensitivity-curves", type=int, default=360)
    parser.add_argument("--sensitivity-estimators", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_curves = min(args.n_curves, 240)
        args.n_sources = min(args.n_sources, 8)
        args.n_estimators = min(args.n_estimators, 60)
    return args


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def monotone_clip(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.clip(y, 0.0, 1.0))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if y.size < 2 or float(np.var(y)) <= 1e-12:
        return float("nan")
    return float(r2_score(y, pred))


def make_model(model_name: str, seed: int, n_estimators: int) -> Any:
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(f"unknown model {model_name}")


def generate_synthetic_world(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n = args.n_curves
    source_id = rng.integers(0, args.n_sources, size=n)
    source_effects = rng.normal(0.0, args.source_shift, size=(args.n_sources, len(HIDDEN_FEATURES)))

    drug_mw = rng.lognormal(np.log(420.0), 0.32, size=n)
    drug_logp = rng.normal(2.2, 1.0, size=n)
    drug_tpsa = rng.normal(75.0, 28.0, size=n).clip(10.0, 180.0)
    plga_mw_kda = rng.lognormal(np.log(45.0), 0.45, size=n)
    la_ga_ratio = rng.beta(4.0, 3.0, size=n)
    drug_loading_pct = rng.uniform(1.0, 25.0, size=n)
    particle_size_um = rng.lognormal(np.log(32.0), 0.7, size=n)

    medium_pH = rng.normal(7.25, 0.35, size=n).clip(5.5, 8.2)
    temperature_c = rng.choice([25.0, 37.0, 45.0], size=n, p=[0.18, 0.68, 0.14])
    agitation_rpm = rng.normal(95.0, 35.0, size=n).clip(0.0, 220.0)
    solvent_strength = rng.beta(2.0, 4.0, size=n)
    surfactant_pct = rng.lognormal(np.log(0.35), 0.7, size=n).clip(0.0, 3.0)

    z_loading = (drug_loading_pct - drug_loading_pct.mean()) / drug_loading_pct.std()
    z_mw = (np.log(plga_mw_kda) - np.log(plga_mw_kda).mean()) / np.log(plga_mw_kda).std()
    z_size = (np.log(particle_size_um) - np.log(particle_size_um).mean()) / np.log(particle_size_um).std()
    z_logp = (drug_logp - drug_logp.mean()) / drug_logp.std()
    z_tpsa = (drug_tpsa - drug_tpsa.mean()) / drug_tpsa.std()
    src = source_effects[source_id]
    eps = rng.normal(0.0, 0.85, size=(n, len(HIDDEN_FEATURES)))
    hs = args.hidden_strength

    hidden = np.column_stack(
        [
            sigmoid(hs * (0.65 * z_loading - 0.35 * z_mw + 0.35 * solvent_strength + src[:, 0] + eps[:, 0])),
            sigmoid(hs * (0.85 * z_loading + 0.45 * z_logp + 0.55 * solvent_strength + src[:, 1] + eps[:, 1])),
            sigmoid(hs * (0.75 * z_size + 0.35 * surfactant_pct + src[:, 2] + eps[:, 2])),
            sigmoid(hs * (0.7 * solvent_strength - 0.45 * surfactant_pct + src[:, 3] + eps[:, 3])),
            sigmoid(hs * (0.55 * z_mw + 0.4 * la_ga_ratio + src[:, 4] + eps[:, 4])),
            sigmoid(hs * (0.7 * z_mw + 0.45 * z_size - 0.55 * surfactant_pct + src[:, 5] + eps[:, 5])),
            sigmoid(hs * (-0.45 * z_logp + 0.45 * z_tpsa + 0.4 * (medium_pH - 7.2) + src[:, 6] + eps[:, 6])),
            sigmoid(hs * (0.5 * la_ga_ratio + 0.55 * (medium_pH - 7.2) + 0.35 * src[:, 7] + eps[:, 7])),
        ]
    )

    h = {name: hidden[:, i] for i, name in enumerate(HIDDEN_FEATURES)}
    times = np.geomspace(0.25, args.max_time_days, args.grid_size)
    t = times[None, :]

    burst = 0.03 + 0.36 * sigmoid(
        2.1 * h["surface_drug_fraction"]
        + 1.25 * h["porosity"]
        + 0.85 * h["residual_solvent"]
        - 0.9 * h["tortuosity"]
        + 0.35 * z_loading
    )
    qmax = 0.62 + 0.36 * sigmoid(
        1.4 * h["porosity"]
        - 0.75 * z_logp
        + 0.65 * surfactant_pct
        - 0.55 * h["residual_solvent"]
        + 0.25 * (temperature_c - 37.0) / 8.0
    )
    log_k = (
        -3.35
        + 1.45 * h["porosity"]
        - 1.25 * h["tortuosity"]
        - 0.15 * z_size
        + 0.75 * h["water_uptake_rate"]
        + 0.32 * (temperature_c - 37.0) / 8.0
        + 0.42 * (medium_pH - 7.2)
        + 0.75 * h["autocatalysis_strength"]
    )
    k_eff = np.exp(log_k)
    shape_n = 0.48 + 1.65 * sigmoid(
        0.95 * h["autocatalysis_strength"] + 0.7 * h["mw_dispersion"] - 0.6 * h["tortuosity"]
    )
    tau_burst = 0.35 + 2.75 * (1.0 - h["surface_drug_fraction"]) + 0.6 * h["particle_size_cv"]

    curves = burst[:, None] * (1.0 - np.exp(-t / tau_burst[:, None]))
    curves += (qmax - burst)[:, None] * (1.0 - np.exp(-np.power(k_eff[:, None] * t, shape_n[:, None])))
    curves += rng.normal(0.0, args.noise_sigma, size=curves.shape)
    curves = np.vstack([monotone_clip(row) for row in curves])

    df = pd.DataFrame(
        {
            "curve_id": [f"synthetic_{i:05d}" for i in range(n)],
            "source_id": source_id,
            "drug_mw": drug_mw,
            "drug_logp": drug_logp,
            "drug_tpsa": drug_tpsa,
            "plga_mw_kda": plga_mw_kda,
            "la_ga_ratio": la_ga_ratio,
            "drug_loading_pct": drug_loading_pct,
            "particle_size_um": particle_size_um,
            "medium_pH": medium_pH,
            "temperature_c": temperature_c,
            "agitation_rpm": agitation_rpm,
            "solvent_strength": solvent_strength,
            "surfactant_pct": surfactant_pct,
        }
    )
    for i, name in enumerate(HIDDEN_FEATURES):
        df[name] = hidden[:, i]
        df[f"proxy_{name}"] = np.clip(hidden[:, i] + rng.normal(0.0, 0.18, size=n), 0.0, 1.0)

    return df, times, curves


def context_indices(times: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0:
        return np.asarray([], dtype=int)
    wanted = EARLY_OBS_DAYS[:budget]
    indices = [int(np.argmin(np.abs(times - day))) for day in wanted]
    return np.asarray(sorted(set(indices)), dtype=int)


def early_feature_matrix(curves: np.ndarray, times: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0:
        return np.empty((len(curves), 0))
    parts = []
    for idx in context_indices(times, budget):
        parts.append(np.full(len(curves), np.log1p(times[idx])))
        parts.append(curves[:, idx])
    return np.column_stack(parts)


def future_mask(times: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0:
        return np.ones(len(times), dtype=bool)
    indices = context_indices(times, budget)
    return times > times[indices[-1]]


def make_splits(df: pd.DataFrame, seed: int) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    random_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(random_cv.split(df), start=1):
        splits.append(("random_5fold", fold, train_idx, test_idx))

    n_groups = int(df["source_id"].nunique())
    group_cv = GroupKFold(n_splits=min(5, n_groups))
    for fold, (train_idx, test_idx) in enumerate(group_cv.split(df, groups=df["source_id"]), start=1):
        splits.append(("source_group_5fold", fold, train_idx, test_idx))
    return splits


def feature_block(df: pd.DataFrame, kind: str, budget: int, curves: np.ndarray, times: np.ndarray) -> np.ndarray:
    if kind == "core_X":
        cols = CORE_FEATURES
    elif kind == "reported_X":
        cols = CORE_FEATURES + CONTEXT_FEATURES
    elif kind == "reported_X_plus_proxy_H":
        cols = CORE_FEATURES + CONTEXT_FEATURES + PROXY_FEATURES
    elif kind == "true_hidden_upper":
        cols = CORE_FEATURES + CONTEXT_FEATURES + HIDDEN_FEATURES
    elif kind == "reported_X_plus_early_Q":
        base = df[CORE_FEATURES + CONTEXT_FEATURES].to_numpy(dtype=float)
        return np.column_stack([base, early_feature_matrix(curves, times, budget)])
    else:
        raise ValueError(f"unknown feature kind {kind}")
    return df[cols].to_numpy(dtype=float)


def evaluate_curve_predictors(
    df: pd.DataFrame,
    times: np.ndarray,
    curves: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    splits = make_splits(df, args.seed)
    feature_kinds = [
        "core_X",
        "reported_X",
        "reported_X_plus_proxy_H",
        "true_hidden_upper",
        "reported_X_plus_early_Q",
        "reported_X_plus_Hhat_from_early_Q",
    ]

    for split_kind, fold, train_idx, test_idx in splits:
        train_sources = set(df.iloc[train_idx]["source_id"])
        test_sources = set(df.iloc[test_idx]["source_id"])
        if split_kind.startswith("source") and train_sources.intersection(test_sources):
            raise RuntimeError("source-group split leaked sources")

        for budget in args.budgets:
            fmask = future_mask(times, budget)
            h_train = df.iloc[train_idx][HIDDEN_FEATURES].to_numpy(dtype=float)
            h_test = df.iloc[test_idx][HIDDEN_FEATURES].to_numpy(dtype=float)
            h_inputs_train = feature_block(df.iloc[train_idx], "reported_X_plus_early_Q", budget, curves[train_idx], times)
            h_inputs_test = feature_block(df.iloc[test_idx], "reported_X_plus_early_Q", budget, curves[test_idx], times)

            for model_name in MODEL_NAMES:
                h_model = make_model(model_name, args.seed + 1000 + fold + budget, args.n_estimators)
                h_model.fit(h_inputs_train, h_train)
                h_pred_train = np.clip(h_model.predict(h_inputs_train), 0.0, 1.0)
                h_pred_test = np.clip(h_model.predict(h_inputs_test), 0.0, 1.0)
                hidden_rows.append(
                    {
                        "split_kind": split_kind,
                        "fold": fold,
                        "budget": budget,
                        "model": model_name,
                        "input_kind": "reported_X_plus_early_Q_to_H",
                        "hidden_mean_r2": safe_r2(h_test.ravel(), h_pred_test.ravel()),
                        "hidden_rmse": rmse(h_test, h_pred_test),
                    }
                )
                for j, hidden_name in enumerate(HIDDEN_FEATURES):
                    hidden_rows.append(
                        {
                            "split_kind": split_kind,
                            "fold": fold,
                            "budget": budget,
                            "model": model_name,
                            "input_kind": f"recover_{hidden_name}",
                            "hidden_mean_r2": safe_r2(h_test[:, j], h_pred_test[:, j]),
                            "hidden_rmse": rmse(h_test[:, j], h_pred_test[:, j]),
                        }
                    )

                for feature_kind in feature_kinds:
                    if feature_kind == "reported_X_plus_Hhat_from_early_Q":
                        x_train = np.column_stack(
                            [df.iloc[train_idx][CORE_FEATURES + CONTEXT_FEATURES].to_numpy(dtype=float), h_pred_train]
                        )
                        x_test = np.column_stack(
                            [df.iloc[test_idx][CORE_FEATURES + CONTEXT_FEATURES].to_numpy(dtype=float), h_pred_test]
                        )
                    else:
                        x_train = feature_block(df.iloc[train_idx], feature_kind, budget, curves[train_idx], times)
                        x_test = feature_block(df.iloc[test_idx], feature_kind, budget, curves[test_idx], times)

                    model = make_model(model_name, args.seed + 10 * fold + budget, args.n_estimators)
                    model.fit(x_train, curves[train_idx])
                    pred = np.vstack([monotone_clip(row) for row in model.predict(x_test)])

                    for row_pos, curve_idx in enumerate(test_idx):
                        y_true = curves[curve_idx, fmask]
                        y_pred = pred[row_pos, fmask]
                        per_rows.append(
                            {
                                "curve_id": df.iloc[curve_idx]["curve_id"],
                                "source_id": int(df.iloc[curve_idx]["source_id"]),
                                "split_kind": split_kind,
                                "fold": fold,
                                "budget": budget,
                                "model": model_name,
                                "input_kind": feature_kind,
                                "future_start_day": float(times[fmask][0]),
                                "future_rmse": rmse(y_true, y_pred),
                                "future_mae": mae(y_true, y_pred),
                                "future_r2": safe_r2(y_true, y_pred),
                                "is_true_hidden_upper_bound": feature_kind == "true_hidden_upper",
                            }
                        )
    return pd.DataFrame(per_rows), pd.DataFrame(hidden_rows)


def summarize(per_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        per_curve.groupby(["split_kind", "budget", "model", "input_kind"], as_index=False)
        .agg(
            n_curves=("curve_id", "nunique"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_mae=("future_mae", "mean"),
            median_future_r2=("future_r2", "median"),
        )
        .sort_values(["split_kind", "model", "budget", "mean_future_rmse"])
    )

    gap_rows: list[dict[str, Any]] = []
    for (split_kind, budget, model), block in summary.groupby(["split_kind", "budget", "model"]):
        lookup = dict(zip(block["input_kind"], block["mean_future_rmse"]))
        base = lookup.get("reported_X")
        upper = lookup.get("true_hidden_upper")
        denom = None if base is None or upper is None else base - upper
        for input_kind, rmse_value in lookup.items():
            if denom is None or denom <= 1e-12:
                gap = float("nan")
                guard = "reported_X_not_worse_than_true_hidden_guardrail"
            else:
                gap = 100.0 * (base - rmse_value) / denom
                guard = "ok"
            gap_rows.append(
                {
                    "split_kind": split_kind,
                    "budget": budget,
                    "model": model,
                    "input_kind": input_kind,
                    "reported_X_rmse": base,
                    "true_hidden_rmse": upper,
                    "method_rmse": rmse_value,
                    "hidden_gap_closed_pct": gap,
                    "guardrail": guard,
                }
            )
    return summary, pd.DataFrame(gap_rows)


def make_decision_table(summary: pd.DataFrame, gap: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    primary = summary[(summary["split_kind"] == "source_group_5fold") & (summary["model"] == "extra_trees")]
    gap_primary = gap[(gap["split_kind"] == "source_group_5fold") & (gap["model"] == "extra_trees")]

    def mean_rmse(input_kind: str, budget: int) -> float:
        row = primary[(primary["input_kind"] == input_kind) & (primary["budget"] == budget)]
        if row.empty:
            return float("nan")
        return float(row["mean_future_rmse"].iloc[0])

    def mean_gap(input_kind: str, budget: int) -> float:
        row = gap_primary[(gap_primary["input_kind"] == input_kind) & (gap_primary["budget"] == budget)]
        if row.empty:
            return float("nan")
        return float(row["hidden_gap_closed_pct"].iloc[0])

    hidden_primary = hidden[
        (hidden["split_kind"] == "source_group_5fold")
        & (hidden["model"] == "extra_trees")
        & (hidden["input_kind"] == "reported_X_plus_early_Q_to_H")
    ]
    hidden_by_budget = hidden_primary.groupby("budget")["hidden_mean_r2"].mean()

    reported0 = mean_rmse("reported_X", 0)
    true0 = mean_rmse("true_hidden_upper", 0)
    proxy0 = mean_rmse("reported_X_plus_proxy_H", 0)
    gap3 = mean_gap("reported_X_plus_early_Q", 3)
    gap5 = mean_gap("reported_X_plus_early_Q", 5)
    hhat3 = mean_gap("reported_X_plus_Hhat_from_early_Q", 3)
    hhat5 = mean_gap("reported_X_plus_Hhat_from_early_Q", 5)
    h0 = float(hidden_by_budget.get(0, np.nan))
    h5 = float(hidden_by_budget.get(5, np.nan))

    rows = [
        {
            "decision": "reported_X_has_information_gap",
            "status": "pass" if reported0 - true0 > 0.015 else "warn",
            "evidence": f"k0 reported_X RMSE={reported0:.4f}; true_hidden_upper RMSE={true0:.4f}",
        },
        {
            "decision": "noisy_hidden_proxies_improve_static_X",
            "status": "pass" if proxy0 < reported0 else "warn",
            "evidence": f"k0 proxy_H RMSE={proxy0:.4f}; reported_X RMSE={reported0:.4f}",
        },
        {
            "decision": "early_Q_closes_hidden_gap_by_k3",
            "status": "pass" if gap3 >= 50.0 else "warn",
            "evidence": f"k3 direct early-Q gap closure={gap3:.1f}%",
        },
        {
            "decision": "early_Q_can_exceed_hidden_upper_for_future_prediction",
            "status": "pass" if gap5 > 100.0 else "warn",
            "evidence": f"k5 direct early-Q gap closure={gap5:.1f}%",
        },
        {
            "decision": "early_Q_to_hidden_state_is_plausible_but_lossy",
            "status": "pass" if hhat3 >= 50.0 and hhat5 >= 100.0 else "warn",
            "evidence": f"Hhat gap closure k3={hhat3:.1f}%, k5={hhat5:.1f}%",
        },
        {
            "decision": "hidden_recovery_improves_with_observation_budget",
            "status": "pass" if h5 > h0 else "warn",
            "evidence": f"mean hidden R2 k0={h0:.3f}; k5={h5:.3f}",
        },
    ]
    return pd.DataFrame(rows)


def run_hidden_strength_sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_seed = args.seed + 20_000
    for hidden_strength in args.sensitivity_hidden_strengths:
        sens_args = argparse.Namespace(**vars(args))
        sens_args.seed = base_seed
        sens_args.n_curves = args.sensitivity_curves
        sens_args.n_estimators = args.sensitivity_estimators
        sens_args.hidden_strength = hidden_strength
        sens_args.budgets = [0, 3, 5]

        df, times, curves = generate_synthetic_world(sens_args)
        per_curve, hidden = evaluate_curve_predictors(df, times, curves, sens_args)
        summary, gap = summarize(per_curve)

        primary = summary[(summary["split_kind"] == "source_group_5fold") & (summary["model"] == "extra_trees")]
        gap_primary = gap[(gap["split_kind"] == "source_group_5fold") & (gap["model"] == "extra_trees")]

        def rmse_value(input_kind: str, budget: int) -> float:
            block = primary[(primary["input_kind"] == input_kind) & (primary["budget"] == budget)]
            if block.empty:
                return float("nan")
            return float(block["mean_future_rmse"].iloc[0])

        def gap_value(input_kind: str, budget: int) -> float:
            block = gap_primary[(gap_primary["input_kind"] == input_kind) & (gap_primary["budget"] == budget)]
            if block.empty:
                return float("nan")
            return float(block["hidden_gap_closed_pct"].iloc[0])

        hidden_primary = hidden[
            (hidden["split_kind"] == "source_group_5fold")
            & (hidden["model"] == "extra_trees")
            & (hidden["input_kind"] == "reported_X_plus_early_Q_to_H")
        ]
        hidden_by_budget = hidden_primary.groupby("budget")["hidden_mean_r2"].mean()

        rows.append(
            {
                "hidden_strength": hidden_strength,
                "n_curves": sens_args.n_curves,
                "n_estimators": sens_args.n_estimators,
                "reported_X_k0_rmse": rmse_value("reported_X", 0),
                "true_hidden_k0_rmse": rmse_value("true_hidden_upper", 0),
                "noisy_proxy_H_k0_rmse": rmse_value("reported_X_plus_proxy_H", 0),
                "early_Q_k3_rmse": rmse_value("reported_X_plus_early_Q", 3),
                "early_Q_k5_rmse": rmse_value("reported_X_plus_early_Q", 5),
                "hhat_from_early_Q_k3_rmse": rmse_value("reported_X_plus_Hhat_from_early_Q", 3),
                "hhat_from_early_Q_k5_rmse": rmse_value("reported_X_plus_Hhat_from_early_Q", 5),
                "reported_X_to_true_H_gap_k0": rmse_value("reported_X", 0) - rmse_value("true_hidden_upper", 0),
                "early_Q_k3_gap_closed_pct": gap_value("reported_X_plus_early_Q", 3),
                "early_Q_k5_gap_closed_pct": gap_value("reported_X_plus_early_Q", 5),
                "hhat_k3_gap_closed_pct": gap_value("reported_X_plus_Hhat_from_early_Q", 3),
                "hhat_k5_gap_closed_pct": gap_value("reported_X_plus_Hhat_from_early_Q", 5),
                "hidden_recovery_r2_k0": float(hidden_by_budget.get(0, np.nan)),
                "hidden_recovery_r2_k5": float(hidden_by_budget.get(5, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def write_checks(
    df: pd.DataFrame,
    times: np.ndarray,
    curves: np.ndarray,
    per_curve: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    splits = make_splits(df, args.seed)
    group_ok = True
    for split_kind, _, train_idx, test_idx in splits:
        if split_kind.startswith("source"):
            group_ok &= not bool(
                set(df.iloc[train_idx]["source_id"]).intersection(set(df.iloc[test_idx]["source_id"]))
            )
    checks = [
        {
            "check": "sample_unit_is_curve",
            "status": "pass",
            "detail": f"{len(df)} synthetic curves; no timepoint rows used for splitting",
        },
        {
            "check": "curves_monotone_bounded",
            "status": "pass" if bool((np.diff(curves, axis=1) >= -1e-12).all() and curves.min() >= 0 and curves.max() <= 1) else "fail",
            "detail": "model target curves are cumulative fractions in [0, 1]",
        },
        {
            "check": "early_times_before_future",
            "status": "pass" if all(future_mask(times, k).any() for k in args.budgets) else "fail",
            "detail": f"budgets={args.budgets}; early schedule={EARLY_OBS_DAYS.tolist()} days",
        },
        {
            "check": "source_group_no_overlap",
            "status": "pass" if group_ok else "fail",
            "detail": "GroupKFold uses source_id as held-out group",
        },
        {
            "check": "hidden_not_in_reported_X",
            "status": "pass",
            "detail": "reported_X excludes true hidden variables; true_hidden_upper is explicitly marked",
        },
        {
            "check": "no_nan_primary_metrics",
            "status": "pass" if not per_curve["future_rmse"].isna().any() else "fail",
            "detail": f"rows={len(per_curve)}",
        },
    ]
    return pd.DataFrame(checks)


def write_report(
    out: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    gap: pd.DataFrame,
    hidden: pd.DataFrame,
    checks: pd.DataFrame,
    decisions: pd.DataFrame,
    sensitivity: pd.DataFrame | None = None,
) -> None:
    primary = summary[(summary["split_kind"] == "source_group_5fold") & (summary["model"] == "extra_trees")]
    pivot = (
        primary.pivot_table(index=["budget"], columns="input_kind", values="mean_future_rmse", aggfunc="mean")
        .reset_index()
        .round(4)
    )
    gap_primary = gap[
        (gap["split_kind"] == "source_group_5fold")
        & (gap["model"] == "extra_trees")
        & (gap["input_kind"].isin(["reported_X_plus_early_Q", "reported_X_plus_Hhat_from_early_Q", "true_hidden_upper"]))
    ].copy()
    gap_primary = gap_primary.sort_values(["budget", "input_kind"])[
        ["budget", "input_kind", "method_rmse", "hidden_gap_closed_pct", "guardrail"]
    ].round(4)

    hidden_primary = hidden[
        (hidden["split_kind"] == "source_group_5fold")
        & (hidden["model"] == "extra_trees")
        & (hidden["input_kind"] == "reported_X_plus_early_Q_to_H")
    ][["budget", "hidden_mean_r2", "hidden_rmse"]].groupby("budget", as_index=False).mean().round(4)

    lines = [
        "# Synthetic Hidden-State Identifiability Audit",
        "",
        "This diagnostic creates a synthetic PLGA-like world where the release-driving hidden state is known.",
        "It asks whether the project's real bottleneck pattern can arise from missing microstructure/process variables rather than weak model choice.",
        "",
        "## Primary Source-Group Result",
        "",
        pivot.to_markdown(index=False),
        "",
        "## Hidden Gap Closure",
        "",
        gap_primary.to_markdown(index=False),
        "",
        "## Hidden-State Recovery From Early Q",
        "",
        hidden_primary.to_markdown(index=False),
        "",
        "## Data Checks",
        "",
        checks.to_markdown(index=False),
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
    ]
    if sensitivity is not None and not sensitivity.empty:
        display_cols = [
            "hidden_strength",
            "reported_X_k0_rmse",
            "true_hidden_k0_rmse",
            "noisy_proxy_H_k0_rmse",
            "early_Q_k3_rmse",
            "early_Q_k5_rmse",
            "reported_X_to_true_H_gap_k0",
            "early_Q_k3_gap_closed_pct",
        ]
        lines.extend(
            [
                "## Hidden-Strength Sensitivity",
                "",
                sensitivity[display_cols].round(4).to_markdown(index=False),
                "",
                "This table is a defense layer: as hidden state becomes more important, reported static descriptors become less sufficient and early observations remain valuable.",
                "Percent gap closure can exceed 100% because early Q observes the realized release trajectory, not because it is more physical than the named hidden variables.",
                "",
            ]
        )
    lines.extend(
        [
        "## Interpretation Rule",
        "",
        "- If `true_hidden_upper` is much better than `reported_X`, the synthetic world contains a real information gap.",
        "- If `reported_X_plus_early_Q` closes that gap, early observations are acting as release-state measurements.",
        "- If early Q exceeds the true-hidden row, interpret it as an empirical release-state measurement that contains X, H, source effect, noise realization, and model residuals.",
        "- If `reported_X_plus_Hhat_from_early_Q` approaches direct early-Q prediction, then inferring hidden state from early curves is plausible in principle.",
        "- This does not validate any specific real hidden variable. It tells us what kind of missing information could explain the PLGA/liposome failures.",
        "",
        f"Run settings: n_curves={args.n_curves}, n_sources={args.n_sources}, grid_size={args.grid_size}, seed={args.seed}.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    df, times, curves = generate_synthetic_world(args)
    per_curve, hidden = evaluate_curve_predictors(df, times, curves, args)
    summary, gap = summarize(per_curve)
    decisions = make_decision_table(summary, gap, hidden)
    checks = write_checks(df, times, curves, per_curve, args)
    sensitivity = run_hidden_strength_sensitivity(args) if args.run_hidden_strength_sensitivity else None

    df.to_csv(args.out / "synthetic_metadata.csv", index=False)
    pd.DataFrame({"time_days": times}).to_csv(args.out / "time_grid.csv", index=False)
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_input.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    hidden.to_csv(args.out / "hidden_recovery.csv", index=False)
    gap.to_csv(args.out / "gap_closure.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    if sensitivity is not None:
        sensitivity.to_csv(args.out / "hidden_strength_sensitivity.csv", index=False)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": vars(args) | {"out": str(args.out)},
        "core_features": CORE_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "hidden_features": HIDDEN_FEATURES,
        "proxy_features": PROXY_FEATURES,
        "early_observation_days": EARLY_OBS_DAYS.tolist(),
        "model_names": MODEL_NAMES,
        "interpretation": "diagnostic simulation only; not a deployable predictor",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out, args, summary, gap, hidden, checks, decisions, sensitivity)
    print(f"Wrote synthetic hidden-state audit to {args.out}")


if __name__ == "__main__":
    main()
