"""
40 - Mechanistic feasibility selector for the RF -> theta middle layer.

What this does:
    Script 38d showed that the strongest predictor is RF -> theta -> ODE.
    Script 39 showed that this does not mean RF recovers a unique oracle
    theta. This diagnostic treats theta as a feasible family instead:

        formulation + early Q -> candidate theta bank
        early-only score       -> selected low-cost feasible theta
        full curve             -> evaluation only

Candidate sources:
    1. RF point prediction.
    2. RF tree-level predictions.
    3. Descriptor-neighborhood oracle theta from train folds.
    4. Local active-subset refinements using only early Q.

Anti-leakage rule:
    Candidate generation and selection use only train-fold data plus the
    test curve's first-week observations. Full release curves are used only
    after selection, for evaluation.

Outputs:
    outputs/40_theta_candidate_feasibility/per_candidate.csv
    outputs/40_theta_candidate_feasibility/per_curve_selected.csv
    outputs/40_theta_candidate_feasibility/scheme_summary.csv
    outputs/40_theta_candidate_feasibility/score_component_sensitivity.csv
    outputs/40_theta_candidate_feasibility/summary.txt
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402


DATASET_COL = "dataset"
FID_COL = "fid"

FORMULATION_COL_MAP = {
    "Drug MW": "drug_mw",
    "Drug TPSA": "drug_tpsa",
    "Drug LogP": "drug_logp",
    "Polymer MW": "polymer_mw",
    "LA/GA": "laga",
    "Initial Drug-to-Polymer Ratio": "drug_polymer_ratio",
    "Particle Size": "particle_size",
    "Drug Loading Capacity": "drug_loading",
    "Drug Encapsulation Efficiency": "drug_ee",
    "Solubility Enhancer Concentration": "solubility_enhancer",
}
DRUG_KEY_COLS = ["drug_mw", "drug_tpsa", "drug_logp"]
POLYMER_KEY_COLS = ["polymer_mw", "laga"]

DEFAULT_WEIGHTS = {
    "active_param_count": 0.02,
    "nfev": 0.002,
    "descriptor_z": 0.05,
    "stability_rmse": 0.10,
    "boundary_hits": 0.01,
}
SENSITIVITY_PROFILES = {
    "default": DEFAULT_WEIGHTS,
    "fit_first": {
        "active_param_count": 0.005,
        "nfev": 0.0005,
        "descriptor_z": 0.02,
        "stability_rmse": 0.05,
        "boundary_hits": 0.005,
    },
    "cost_first": {
        "active_param_count": 0.05,
        "nfev": 0.005,
        "descriptor_z": 0.05,
        "stability_rmse": 0.10,
        "boundary_hits": 0.01,
    },
    "stability_first": {
        "active_param_count": 0.02,
        "nfev": 0.002,
        "descriptor_z": 0.10,
        "stability_rmse": 0.25,
        "boundary_hits": 0.01,
    },
}


@dataclass
class CurveRecord:
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray


@dataclass
class ThetaCandidate:
    theta: np.ndarray
    generator: str
    active_idx: tuple[int, ...]
    fixed_theta: np.ndarray
    nfev: int
    source_rank: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_external_records(
    xlsx_path: Path, matched_fids_csv: Path, t_grid_max_days: float
) -> list[CurveRecord]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns={"Formulation Index": "fid"})
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    df = df[df["fid"].astype(int).isin(matched_fids)].copy()
    out: list[CurveRecord] = []
    for fid in sorted(df["fid"].unique().tolist()):
        g = df[df["fid"] == fid].sort_values("Time")
        g = g.groupby("Time", as_index=False).agg({"Release": "mean"})
        t_obs = g["Time"].to_numpy(dtype=float)
        q_obs = np.clip(g["Release"].to_numpy(dtype=float), 0.0, 1.0)
        mask = t_obs <= t_grid_max_days
        t_obs = t_obs[mask]
        q_obs = q_obs[mask]
        if len(t_obs) < 3:
            continue
        out.append(CurveRecord(fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    return out


def _interp_at(t_obs: np.ndarray, q_obs: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.interp(times, t_obs, q_obs, left=q_obs[0], right=q_obs[-1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(float), lows + eps, highs - eps)


def _boundary_hits(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> int:
    margin = 0.01 * (highs - lows)
    return int(np.sum((theta <= lows + margin) | (theta >= highs - margin)))


def _active_set_name(active_idx: tuple[int, ...], param_names: list[str]) -> str:
    if len(active_idx) == len(param_names):
        return "all9"
    if not active_idx:
        return "none"
    return ",".join(param_names[i] for i in active_idx)


def _descriptor_z(theta: np.ndarray, neighbor_theta: np.ndarray) -> float:
    med = np.median(neighbor_theta, axis=0)
    q25 = np.percentile(neighbor_theta, 25, axis=0)
    q75 = np.percentile(neighbor_theta, 75, axis=0)
    robust_sd = (q75 - q25) / 1.349
    fallback_sd = neighbor_theta.std(axis=0)
    scale = np.where(robust_sd > 1e-3, robust_sd, fallback_sd)
    scale = np.where(scale > 1e-3, scale, 1.0)
    return float(np.mean(np.abs((theta - med) / scale)))


def _fit_active_subset_early(
    sim: PLGABiphasic,
    t_early: np.ndarray,
    q_early: np.ndarray,
    theta_fixed: np.ndarray,
    theta_start: np.ndarray,
    active_idx: tuple[int, ...],
    lows: np.ndarray,
    highs: np.ndarray,
    max_nfev: int,
) -> tuple[np.ndarray, int]:
    if not active_idx:
        return theta_fixed.copy(), 0

    idx = np.array(active_idx, dtype=int)
    z0 = np.clip(theta_start[idx], lows[idx], highs[idx])

    def residual(z: np.ndarray) -> np.ndarray:
        theta = theta_fixed.copy()
        theta[idx] = z
        return sim.simulate_numpy(theta, t_early) - q_early

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = least_squares(
            residual,
            x0=z0,
            bounds=(lows[idx], highs[idx]),
            method="trf",
            max_nfev=max_nfev,
        )
    theta = theta_fixed.copy()
    theta[idx] = result.x
    return theta, int(result.nfev)


def _cfe_fit(
    sim: PLGABiphasic,
    t_early: np.ndarray,
    q_early: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    n_restarts: int,
    seed: int,
) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lows + highs)
    best_theta: np.ndarray | None = None
    best_ss = np.inf
    for k in range(n_restarts):
        x0 = mid if k == 0 else rng.uniform(lows, highs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = least_squares(
                    lambda th: sim.simulate_numpy(th, t_early) - q_early,
                    x0=x0,
                    bounds=(lows, highs),
                    method="trf",
                    max_nfev=300,
                )
        except Exception:
            continue
        pred = sim.simulate_numpy(result.x, t_early)
        ss = float(np.sum((pred - q_early) ** 2))
        if np.isfinite(ss) and ss < best_ss:
            best_ss = ss
            best_theta = result.x.copy()
    return best_theta


def _load_dataset(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    dict[int, CurveRecord],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    feature_cols = list(FORMULATION_COL_MAP.values())

    df_meta = pd.read_excel(args.cross_doi_data, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    df_bank = pd.read_csv(args.full_fit_bank)
    df_bank_cross = df_bank[df_bank[DATASET_COL] == "cross321"].copy()
    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )

    curves = _load_external_records(
        xlsx_path=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid_max_days=args.t_grid_max_days,
    )
    curve_map = {c.fid: c for c in curves}

    early_times = np.array(args.early_times, dtype=float)
    late_times = np.array(args.late_grid, dtype=float)
    early_q_list: list[np.ndarray] = []
    late_q_list: list[np.ndarray] = []
    keep_rows: list[int] = []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        early_q_list.append(_interp_at(c.t_obs, c.q_obs, early_times))
        late_q_list.append(_interp_at(c.t_obs, c.q_obs, late_times))
        keep_rows.append(i)

    df = df.loc[keep_rows].reset_index(drop=True)
    early_q = np.stack(early_q_list, axis=0).astype(np.float32)
    late_q = np.stack(late_q_list, axis=0).astype(np.float32)
    x_form = df[feature_cols].to_numpy(dtype=np.float32)
    x_full = np.concatenate([x_form, early_q], axis=1)
    theta_oracle = df[param_names].to_numpy(dtype=np.float32)
    drug_keys = df[DRUG_KEY_COLS].apply(tuple, axis=1).to_numpy()
    polymer_keys = df[POLYMER_KEY_COLS].apply(tuple, axis=1).to_numpy()
    return (
        df,
        curve_map,
        x_full,
        early_q,
        late_q,
        theta_oracle,
        feature_cols,
        drug_keys,
        polymer_keys,
    )


def _tree_candidates(
    rf: RandomForestRegressor,
    x_row: np.ndarray,
    n_tree_candidates: int,
    seed: int,
) -> list[np.ndarray]:
    estimators = list(rf.estimators_)
    if not estimators or n_tree_candidates <= 0:
        return []
    rng = np.random.default_rng(seed)
    take = min(n_tree_candidates, len(estimators))
    indices = rng.choice(len(estimators), size=take, replace=False)
    return [estimators[i].predict(x_row[None, :])[0].astype(float) for i in indices]


def _ranked_active_idx(
    theta_start: np.ndarray,
    theta_ref: np.ndarray,
    theta_scale: np.ndarray,
    k: int,
) -> tuple[int, ...]:
    z = np.abs((theta_start - theta_ref) / theta_scale)
    return tuple(np.argsort(z)[-k:][::-1].tolist())


def _make_candidates(
    sim: PLGABiphasic,
    x_row: np.ndarray,
    t_early: np.ndarray,
    q_early: np.ndarray,
    rf_theta: RandomForestRegressor,
    train_theta: np.ndarray,
    neighbor_ids: np.ndarray,
    theta_ref: np.ndarray,
    theta_scale: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> list[ThetaCandidate]:
    n_tree = min(args.n_tree_candidates, max(0, args.n_candidates // 4))
    n_neighbor = min(args.n_neighbor_candidates, max(1, args.n_candidates // 4))
    n_refine_bases = min(args.n_refine_bases, max(1, args.n_candidates // 32))

    rf_point = _clip_theta(rf_theta.predict(x_row[None, :])[0], lows, highs)
    candidates: list[ThetaCandidate] = [
        ThetaCandidate(
            theta=rf_point,
            generator="rf_point",
            active_idx=tuple(range(len(rf_point))),
            fixed_theta=rf_point.copy(),
            nfev=0,
            source_rank=0,
        )
    ]

    for rank, theta in enumerate(_tree_candidates(rf_theta, x_row, n_tree, seed), start=1):
        theta = _clip_theta(theta, lows, highs)
        candidates.append(
            ThetaCandidate(
                theta=theta,
                generator="rf_tree",
                active_idx=tuple(range(len(theta))),
                fixed_theta=theta.copy(),
                nfev=0,
                source_rank=rank,
            )
        )

    neighbor_take = min(n_neighbor, len(neighbor_ids))
    for rank, train_idx in enumerate(neighbor_ids[:neighbor_take], start=1):
        theta = _clip_theta(train_theta[int(train_idx)], lows, highs)
        candidates.append(
            ThetaCandidate(
                theta=theta,
                generator="descriptor_neighbor",
                active_idx=tuple(range(len(theta))),
                fixed_theta=theta.copy(),
                nfev=0,
                source_rank=rank,
            )
        )

    refine_starts = [rf_point]
    for train_idx in neighbor_ids[: max(0, n_refine_bases - 1)]:
        refine_starts.append(_clip_theta(train_theta[int(train_idx)], lows, highs))

    for source_rank, theta_start in enumerate(refine_starts[:n_refine_bases]):
        for k in range(1, args.k_max + 1):
            active_idx = _ranked_active_idx(theta_start, theta_ref, theta_scale, k)
            try:
                theta_fit, nfev = _fit_active_subset_early(
                    sim=sim,
                    t_early=t_early,
                    q_early=q_early,
                    theta_fixed=theta_ref,
                    theta_start=theta_start,
                    active_idx=active_idx,
                    lows=lows,
                    highs=highs,
                    max_nfev=args.refine_max_nfev,
                )
            except Exception:
                continue
            candidates.append(
                ThetaCandidate(
                    theta=_clip_theta(theta_fit, lows, highs),
                    generator=f"local_refine_k{k}",
                    active_idx=active_idx,
                    fixed_theta=theta_ref.copy(),
                    nfev=nfev,
                    source_rank=source_rank,
                )
            )
    return candidates


def _candidate_early_rows(
    sim: PLGABiphasic,
    candidates: list[ThetaCandidate],
    t_early: np.ndarray,
    q_early: np.ndarray,
    neighbor_theta: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    param_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cand_idx, cand in enumerate(candidates):
        pred_early = sim.simulate_numpy(cand.theta, t_early)
        active_count = len(cand.active_idx)
        row: dict[str, object] = {
            "candidate_idx": cand_idx,
            "generator": cand.generator,
            "source_rank": cand.source_rank,
            "active_set": _active_set_name(cand.active_idx, param_names),
            "active_param_count": active_count,
            "nfev": int(cand.nfev),
            "early_rmse": _rmse(q_early, pred_early),
            "descriptor_z": _descriptor_z(cand.theta, neighbor_theta),
            "boundary_hits": _boundary_hits(cand.theta, lows, highs),
            "stability_rmse": np.nan,
        }
        for i, name in enumerate(param_names):
            row[name] = float(cand.theta[i])
        rows.append(row)
    return rows


def _score_row(row: pd.Series, weights: dict[str, float]) -> float:
    return float(
        row["early_rmse"]
        + weights["active_param_count"] * row["active_param_count"]
        + weights["nfev"] * row["nfev"]
        + weights["descriptor_z"] * row["descriptor_z"]
        + weights["stability_rmse"] * row["stability_rmse"]
        + weights["boundary_hits"] * row["boundary_hits"]
    )


def _stability_rmse(
    sim: PLGABiphasic,
    cand: ThetaCandidate,
    t_early: np.ndarray,
    q_early: np.ndarray,
    t_eval: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> float:
    if args.n_noise_repeats <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    if not args.stability_refit:
        pred_early = sim.simulate_numpy(cand.theta, t_early)
        base_rmse = _rmse(q_early, pred_early)
        deltas: list[float] = []
        for _ in range(args.n_noise_repeats):
            noisy_q = np.clip(
                q_early + rng.normal(0.0, args.noise_sigma, size=q_early.shape),
                0.0,
                1.0,
            )
            deltas.append(abs(_rmse(noisy_q, pred_early) - base_rmse))
        return float(np.median(deltas)) if deltas else 0.0

    base_pred = sim.simulate_numpy(cand.theta, t_eval)
    rmses: list[float] = []
    for _ in range(args.n_noise_repeats):
        noisy_q = np.clip(q_early + rng.normal(0.0, args.noise_sigma, size=q_early.shape), 0.0, 1.0)
        try:
            theta_noisy, _ = _fit_active_subset_early(
                sim=sim,
                t_early=t_early,
                q_early=noisy_q,
                theta_fixed=cand.fixed_theta,
                theta_start=cand.theta,
                active_idx=cand.active_idx,
                lows=lows,
                highs=highs,
                max_nfev=args.stability_max_nfev,
            )
        except Exception:
            return 1.0
        noisy_pred = sim.simulate_numpy(theta_noisy, t_eval)
        rmses.append(_rmse(base_pred, noisy_pred))
    return float(np.median(rmses)) if rmses else 0.0


def _add_selection_scores(
    df: pd.DataFrame,
    weights: dict[str, float],
    eligible_delta: float,
) -> pd.DataFrame:
    out = df.copy()
    min_early = float(out["early_rmse"].min())
    out["eligible"] = out["early_rmse"] <= (min_early + eligible_delta)
    out["selection_score"] = out.apply(_score_row, axis=1, weights=weights)
    return out


def _select_candidate(df: pd.DataFrame) -> pd.Series:
    eligible = df[df["eligible"]].copy()
    if eligible.empty:
        eligible = df.copy()
    eligible = eligible.sort_values(
        ["selection_score", "early_rmse", "active_param_count", "nfev"],
        ascending=[True, True, True, True],
    )
    return eligible.iloc[0]


def _rf_q_curve(
    c: CurveRecord,
    early_times: np.ndarray,
    early_q: np.ndarray,
    late_times: np.ndarray,
    late_pred: np.ndarray,
) -> np.ndarray:
    t_combined = np.concatenate([early_times, late_times])
    q_combined = np.concatenate([early_q, late_pred])
    order = np.argsort(t_combined)
    q = np.interp(
        c.t_obs,
        t_combined[order],
        q_combined[order],
        left=q_combined[order[0]],
        right=q_combined[order[-1]],
    )
    return np.clip(q, 0.0, 1.0)


def _scheme_summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "B1 CFE": "r2_B1_CFE",
        "B2 RF->theta": "r2_B2_RF_theta",
        "B3 RF->Q": "r2_B3_RF_Q",
        "40 feasibility": "r2_40_feasibility",
        "E oracle": "r2_E_oracle",
    }
    rows: list[dict[str, object]] = []
    for scheme, sub in per_curve.groupby("scheme", sort=False):
        for method, col in methods.items():
            vals = sub[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            row = {
                "scheme": scheme,
                "method": method,
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p10": float(np.percentile(vals, 10)),
                "frac_above_0.9": float(np.mean(vals >= 0.9)),
                "frac_above_0.5": float(np.mean(vals >= 0.5)),
                "frac_above_0": float(np.mean(vals >= 0.0)),
            }
            if method == "40 feasibility":
                row["median_active_param_count"] = float(sub["selected_active_param_count"].median())
                row["median_nfev"] = float(sub["selected_nfev"].median())
            rows.append(row)
    return pd.DataFrame(rows)


def _sensitivity_summary(per_candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["scheme", "fold", "fid"]
    for profile, weights in SENSITIVITY_PROFILES.items():
        scored = per_candidate.copy()
        scored["profile_score"] = scored.apply(_score_row, axis=1, weights=weights)
        for scheme, scheme_df in scored.groupby("scheme", sort=False):
            selected_rows = []
            for _, sub in scheme_df.groupby(group_cols, sort=False):
                eligible = sub[sub["eligible"]].copy()
                if eligible.empty:
                    eligible = sub.copy()
                picked = eligible.sort_values(
                    ["profile_score", "early_rmse", "active_param_count", "nfev"],
                    ascending=[True, True, True, True],
                ).iloc[0]
                selected_rows.append(picked)
            sel = pd.DataFrame(selected_rows)
            vals = sel["full_r2_eval"].dropna().to_numpy(dtype=float)
            rows.append({
                "profile": profile,
                "scheme": scheme,
                "n": int(len(vals)),
                "median_r2": float(np.median(vals)),
                "mean_r2": float(np.mean(vals)),
                "p10_r2": float(np.percentile(vals, 10)),
                "frac_above_0": float(np.mean(vals >= 0.0)),
                "median_active_param_count": float(sel["active_param_count"].median()),
                "median_nfev": float(sel["nfev"].median()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument("--late-grid", nargs="+", type=float, default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0])
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=250)
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--n-tree-candidates", type=int, default=8)
    ap.add_argument("--n-neighbor-candidates", type=int, default=8)
    ap.add_argument("--n-refine-bases", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--refine-max-nfev", type=int, default=50)
    ap.add_argument("--stability-top-k", type=int, default=5)
    ap.add_argument("--stability-max-nfev", type=int, default=40)
    ap.add_argument("--n-noise-repeats", type=int, default=2)
    ap.add_argument("--noise-sigma", type=float, default=0.01)
    ap.add_argument(
        "--stability-refit",
        action="store_true",
        help="Refit top candidates under noisy early Q. Slower; default uses a cheap no-refit noise proxy.",
    )
    ap.add_argument("--eligible-delta", type=float, default=0.02)
    ap.add_argument("--cfe-restarts", type=int, default=3)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/40_theta_candidate_feasibility"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    param_names = list(sim.param_names)

    (
        df,
        curve_map,
        x_full,
        early_q,
        late_q,
        theta_oracle,
        _feature_cols,
        drug_keys,
        polymer_keys,
    ) = _load_dataset(args)

    if args.max_curves is not None:
        max_n = min(int(args.max_curves), len(df))
        df = df.iloc[:max_n].reset_index(drop=True)
        x_full = x_full[:max_n]
        early_q = early_q[:max_n]
        late_q = late_q[:max_n]
        theta_oracle = theta_oracle[:max_n]
        drug_keys = drug_keys[:max_n]
        polymer_keys = polymer_keys[:max_n]

    n = len(df)
    fids = df[FID_COL].to_numpy(dtype=int)
    early_times = np.array(args.early_times, dtype=float)
    late_times = np.array(args.late_grid, dtype=float)
    print(f"[40] n_curves={n}, n_candidates_budget={args.n_candidates}", flush=True)

    schemes: list[tuple[str, object | None]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_keys),
        ("group_by_polymer", polymer_keys),
    ]

    per_candidate_rows: list[dict[str, object]] = []
    per_curve_rows: list[dict[str, object]] = []
    cfe_cache: dict[int, np.ndarray | None] = {}

    for scheme_name, groups in schemes:
        print(f"\n[40] ====== scheme: {scheme_name} ======", flush=True)
        if scheme_name == "random_5fold":
            n_splits = min(args.n_folds, n)
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(n)))
        else:
            unique_groups = pd.Series(groups).nunique()
            n_splits = min(args.n_folds, int(unique_groups))
            if n_splits < 2:
                print(f"[40] skipping {scheme_name}: only {unique_groups} unique groups", flush=True)
                continue
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.arange(n), groups=groups))

        for fold_idx, (tr, te) in enumerate(splits):
            print(
                f"[40] {scheme_name} fold {fold_idx + 1}/{len(splits)} "
                f"train={len(tr)} test={len(te)}",
                flush=True,
            )
            if groups is not None:
                train_groups = set(np.asarray(groups, dtype=object)[tr].tolist())
                test_groups = set(np.asarray(groups, dtype=object)[te].tolist())
                overlap = train_groups.intersection(test_groups)
                if overlap:
                    raise RuntimeError(f"group leakage in {scheme_name} fold {fold_idx}: {len(overlap)} shared groups")

            rf_theta = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=None,
                random_state=args.seed + fold_idx,
                n_jobs=-1,
            )
            rf_theta.fit(x_full[tr], theta_oracle[tr])
            theta_rf_te = rf_theta.predict(x_full[te])

            rf_q = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=None,
                random_state=args.seed + 100 + fold_idx,
                n_jobs=-1,
            )
            rf_q.fit(x_full[tr], late_q[tr])
            q_rf_te = rf_q.predict(x_full[te])

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_full[tr])
            x_test_scaled = scaler.transform(x_full[te])
            n_neighbors = min(max(args.n_neighbor_candidates, 10), len(tr))
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(x_train_scaled)
            neighbor_pos = nn.kneighbors(x_test_scaled, return_distance=False)

            theta_ref = np.median(theta_oracle[tr], axis=0).astype(float)
            theta_scale = theta_oracle[tr].std(axis=0).astype(float)
            theta_scale = np.where(theta_scale > 1e-3, theta_scale, 1.0)

            for local_i, j in enumerate(te):
                fid = int(fids[j])
                curve = curve_map[fid]
                q_early = early_q[j].astype(float)
                neighbor_train_indices = tr[neighbor_pos[local_i]]
                neighbor_theta = theta_oracle[neighbor_train_indices].astype(float)

                candidates = _make_candidates(
                    sim=sim,
                    x_row=x_full[j].astype(float),
                    t_early=early_times,
                    q_early=q_early,
                    rf_theta=rf_theta,
                    train_theta=theta_oracle,
                    neighbor_ids=neighbor_train_indices,
                    theta_ref=theta_ref,
                    theta_scale=theta_scale,
                    lows=lows,
                    highs=highs,
                    args=args,
                    seed=args.seed + 10_000 * fold_idx + fid,
                )
                cand_rows = _candidate_early_rows(
                    sim=sim,
                    candidates=candidates,
                    t_early=early_times,
                    q_early=q_early,
                    neighbor_theta=neighbor_theta,
                    lows=lows,
                    highs=highs,
                    param_names=param_names,
                )
                cand_df = pd.DataFrame(cand_rows)
                cand_df = _add_selection_scores(cand_df, DEFAULT_WEIGHTS, args.eligible_delta)

                eligible = cand_df[cand_df["eligible"]].copy()
                if eligible.empty:
                    eligible = cand_df.copy()
                top_for_stability = eligible.sort_values(
                    ["selection_score", "early_rmse"],
                    ascending=[True, True],
                ).head(args.stability_top_k)

                stability_values: dict[int, float] = {}
                for cand_idx in top_for_stability["candidate_idx"].astype(int).tolist():
                    stability_values[cand_idx] = _stability_rmse(
                        sim=sim,
                        cand=candidates[cand_idx],
                        t_early=early_times,
                        q_early=q_early,
                        t_eval=curve.t_obs,
                        lows=lows,
                        highs=highs,
                        args=args,
                        seed=args.seed + 20_000 * fold_idx + 101 * fid + cand_idx,
                    )
                fallback_stability = (
                    max(stability_values.values()) + 0.02
                    if stability_values else 0.0
                )
                cand_df["stability_rmse"] = cand_df["candidate_idx"].map(stability_values).fillna(fallback_stability)
                cand_df = _add_selection_scores(cand_df, DEFAULT_WEIGHTS, args.eligible_delta)
                selected = _select_candidate(cand_df)
                selected_idx = int(selected["candidate_idx"])

                full_r2_values: dict[int, float] = {}
                for cand_idx, cand in enumerate(candidates):
                    pred = sim.simulate_numpy(cand.theta, curve.t_obs)
                    full_r2_values[cand_idx] = _r2(curve.q_obs, pred)
                cand_df["full_r2_eval"] = cand_df["candidate_idx"].map(full_r2_values)
                cand_df["selected"] = cand_df["candidate_idx"] == selected_idx
                cand_df.insert(0, "fid", fid)
                cand_df.insert(0, "fold", fold_idx)
                cand_df.insert(0, "scheme", scheme_name)
                per_candidate_rows.extend(cand_df.to_dict(orient="records"))

                theta_rf = _clip_theta(theta_rf_te[local_i], lows, highs)
                r2_rf_theta = _r2(curve.q_obs, sim.simulate_numpy(theta_rf, curve.t_obs))
                q_rf_q = _rf_q_curve(curve, early_times, q_early, late_times, q_rf_te[local_i])
                r2_rf_q = _r2(curve.q_obs, q_rf_q)

                if fid not in cfe_cache:
                    cfe_cache[fid] = _cfe_fit(
                        sim=sim,
                        t_early=early_times,
                        q_early=q_early,
                        lows=lows,
                        highs=highs,
                        n_restarts=args.cfe_restarts,
                        seed=args.seed + fid,
                    )
                cfe_theta = cfe_cache[fid]
                r2_cfe = (
                    _r2(curve.q_obs, sim.simulate_numpy(cfe_theta, curve.t_obs))
                    if cfe_theta is not None else float("nan")
                )
                theta_oracle_j = theta_oracle[j].astype(float)
                r2_oracle = _r2(curve.q_obs, sim.simulate_numpy(theta_oracle_j, curve.t_obs))
                selected_theta = candidates[selected_idx].theta
                r2_selected = full_r2_values[selected_idx]

                curve_row: dict[str, object] = {
                    "scheme": scheme_name,
                    "fold": fold_idx,
                    "fid": fid,
                    "r2_B1_CFE": r2_cfe,
                    "r2_B2_RF_theta": r2_rf_theta,
                    "r2_B3_RF_Q": r2_rf_q,
                    "r2_40_feasibility": r2_selected,
                    "r2_E_oracle": r2_oracle,
                    "selected_candidate_idx": selected_idx,
                    "selected_generator": str(selected["generator"]),
                    "selected_active_set": str(selected["active_set"]),
                    "selected_active_param_count": int(selected["active_param_count"]),
                    "selected_nfev": int(selected["nfev"]),
                    "selected_early_rmse": float(selected["early_rmse"]),
                    "selected_descriptor_z": float(selected["descriptor_z"]),
                    "selected_stability_rmse": float(selected["stability_rmse"]),
                    "selected_boundary_hits": int(selected["boundary_hits"]),
                    "selected_score": float(selected["selection_score"]),
                    "rf_theta_r2_delta": float(r2_selected - r2_rf_theta),
                }
                for p_idx, name in enumerate(param_names):
                    curve_row[f"selected_{name}"] = float(selected_theta[p_idx])
                per_curve_rows.append(curve_row)

    per_candidate = pd.DataFrame(per_candidate_rows)
    per_curve = pd.DataFrame(per_curve_rows)
    scheme_summary = _scheme_summary(per_curve)
    sensitivity = _sensitivity_summary(per_candidate)

    per_candidate.to_csv(args.out / "per_candidate.csv", index=False)
    per_curve.to_csv(args.out / "per_curve_selected.csv", index=False)
    scheme_summary.to_csv(args.out / "scheme_summary.csv", index=False)
    sensitivity.to_csv(args.out / "score_component_sensitivity.csv", index=False)

    lines = [
        "=== 40 -- mechanistic feasibility selector ===",
        "",
        f"n_curves              : {n}",
        f"n_folds               : {args.n_folds}",
        f"n_estimators          : {args.n_estimators}",
        f"n_candidates_budget   : {args.n_candidates}",
        f"eligible_delta        : {args.eligible_delta:.3f}",
        f"n_noise_repeats       : {args.n_noise_repeats}",
        "",
        "--- median R^2 by scheme x method ---",
    ]
    for scheme in ["random_5fold", "group_by_drug", "group_by_polymer"]:
        sub = scheme_summary[scheme_summary["scheme"] == scheme]
        if sub.empty:
            continue
        lines.append(f"  {scheme}:")
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['method']:<16} median={row['median']:+.4f} "
                f"frac_R2>=0={row['frac_above_0']:.3f}"
            )
    lines.append("")
    lines.append("--- feasibility selector cost ---")
    for scheme, sub in per_curve.groupby("scheme", sort=False):
        lines.append(
            f"  {scheme:<16} median_active={sub['selected_active_param_count'].median():.1f} "
            f"median_nfev={sub['selected_nfev'].median():.1f} "
            f"median_delta_vs_RFtheta={sub['rf_theta_r2_delta'].median():+.4f}"
        )
    lines.append("")
    lines.append("--- interpretation guardrail ---")
    lines.append("  Selection used train-fold data plus early Q only. Full curves were")
    lines.append("  evaluated after selection, so the full-curve R^2 columns are diagnostics,")
    lines.append("  not selection targets.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
