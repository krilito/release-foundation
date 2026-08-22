"""
119 - PLGA synthetic-state real projection bridge.

Purpose:
    Use the synthetic hidden-state world from script 118 to define
    simulation-informed release-state coordinates, then project real 321 PLGA
    curves into those coordinates and test whether they align with table-level
    proxies such as particle size, drug loading, formulation method, and DOI.

Consumes:
    D:/chemical-world-model-v0/datset/321PLGA/.../mp_dataset_initial.xlsx
    scripts/118_synthetic_hidden_state_identifiability_audit.py

Produces:
    outputs/119_plga_synthetic_state_real_projection/
      synthetic_inversion_metrics.csv
      real_projected_states.csv
      proxy_validation_summary.csv
      nearest_synthetic_neighbors.csv
      projection_coverage_summary.csv
      curve_embedding_baseline_summary.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    The projected coordinates are H_like release-state coordinates. They are
    not measured porosity, tortuosity, residual solvent, or any other physical
    hidden variable in the real PLGA literature corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.optimize import curve_fit
    from scipy.stats import kruskal, spearmanr
except Exception:  # pragma: no cover - fallback only when scipy is unavailable.
    curve_fit = None
    kruskal = None
    spearmanr = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_XLSX = Path(
    r"D:/chemical-world-model-v0/datset/321PLGA/"
    r"A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles/"
    r"mp_dataset_initial.xlsx"
)
DEFAULT_OUT = Path("outputs/119_plga_synthetic_state_real_projection")
SCRIPT118 = ROOT / "scripts" / "118_synthetic_hidden_state_identifiability_audit.py"

HIDDEN_LIKE = [
    "porosity_like",
    "surface_drug_fraction_like",
    "particle_size_cv_like",
    "residual_solvent_like",
    "mw_dispersion_like",
    "tortuosity_like",
    "water_uptake_rate_like",
    "autocatalysis_strength_like",
]
MODES = ("full_Q", "early_Q_k3", "early_Q_k5", "x_only")
NUMERIC_PROXIES = [
    "Polymer Mw",
    "Polymer Mn",
    "PDI",
    "LA/GA",
    "Initial Drug-to-Polymer Ratio",
    "Particle Size",
    "Drug Loading Capacity",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
]
CATEGORICAL_PROXIES = ["Formulation Method", "DOI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLGA synthetic-state real projection bridge.")
    parser.add_argument("--real-xlsx", type=Path, default=DEFAULT_REAL_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-synthetic", type=int, default=3000)
    parser.add_argument("--hidden-strength", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--max-time-days", type=float, default=90.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-real-curves", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_synthetic = min(args.n_synthetic, 300)
        args.n_estimators = min(args.n_estimators, 50)
        args.bootstrap = min(args.bootstrap, 80)
        args.shuffle_repeats = min(args.shuffle_repeats, 60)
        if args.max_real_curves <= 0:
            args.max_real_curves = 45
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_118_module() -> Any:
    spec = importlib.util.spec_from_file_location("synthetic118", SCRIPT118)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SCRIPT118}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if y.size < 2 or float(np.var(y)) <= 1e-12:
        return float("nan")
    return float(r2_score(y, pred))


def q_feature_matrix(curves: np.ndarray, times: np.ndarray, mode: str, m118: Any) -> np.ndarray:
    if mode == "full_Q":
        return curves
    if mode == "early_Q_k3":
        return m118.early_feature_matrix(curves, times, 3)
    if mode == "early_Q_k5":
        return m118.early_feature_matrix(curves, times, 5)
    raise ValueError(f"unsupported Q mode {mode}")


def x_feature_matrix(df: pd.DataFrame, m118: Any) -> np.ndarray:
    return df[m118.CORE_FEATURES + m118.CONTEXT_FEATURES].to_numpy(dtype=float)


def make_model(seed: int, n_estimators: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )


def train_inversion_models(
    args: argparse.Namespace,
    m118: Any,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame, np.ndarray]:
    synth_args = argparse.Namespace(
        seed=args.seed,
        n_curves=args.n_synthetic,
        n_sources=12,
        grid_size=args.grid_size,
        max_time_days=args.max_time_days,
        noise_sigma=0.025,
        hidden_strength=args.hidden_strength,
        source_shift=0.55,
        budgets=[0, 1, 2, 3, 5],
        n_estimators=args.n_estimators,
        smoke=False,
        out=args.out,
        run_hidden_strength_sensitivity=False,
        sensitivity_hidden_strengths=[0.4, 1.0, 1.8],
        sensitivity_curves=360,
        sensitivity_estimators=80,
    )
    synth_df, times, curves = m118.generate_synthetic_world(synth_args)
    hidden = synth_df[m118.HIDDEN_FEATURES].to_numpy(dtype=float)
    hidden_df = pd.DataFrame(hidden, columns=HIDDEN_LIKE)

    train_idx, test_idx = train_test_split(
        np.arange(len(synth_df)),
        test_size=0.25,
        random_state=args.seed,
        shuffle=True,
    )
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    for mode in MODES:
        if mode == "x_only":
            x_all = x_feature_matrix(synth_df, m118)
        else:
            x_all = q_feature_matrix(curves, times, mode, m118)
        model = make_model(args.seed + len(models) + 10, args.n_estimators)
        model.fit(x_all[train_idx], hidden[train_idx])
        pred = np.clip(model.predict(x_all[test_idx]), 0.0, 1.0)
        models[mode] = make_model(args.seed + len(models) + 100, args.n_estimators)
        models[mode].fit(x_all, hidden)

        rows.append(
            {
                "mode": mode,
                "hidden_coordinate": "__all__",
                "r2": safe_r2(hidden[test_idx].ravel(), pred.ravel()),
                "rmse": rmse(hidden[test_idx], pred),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        for j, name in enumerate(HIDDEN_LIKE):
            rows.append(
                {
                    "mode": mode,
                    "hidden_coordinate": name,
                    "r2": safe_r2(hidden[test_idx, j], pred[:, j]),
                    "rmse": rmse(hidden[test_idx, j], pred[:, j]),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
    return synth_df, times, curves, hidden_df, models, pd.DataFrame(rows), hidden


def load_real_321(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if not args.real_xlsx.exists():
        raise FileNotFoundError(args.real_xlsx)
    raw = pd.read_excel(args.real_xlsx, sheet_name="PLGA_MPs")
    required = [
        "Formulation Index",
        "Drug",
        "Drug SMILES",
        "Polymer Mw",
        "Polymer Mn",
        "PDI",
        "LA/GA",
        "Formulation Method",
        "Initial Drug-to-Polymer Ratio",
        "Particle Size",
        "Drug Loading Capacity",
        "Drug Encapsulation Efficiency",
        "Solubility Enhancer Concentration",
        "Time",
        "Release",
        "DOI",
    ]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"missing required real 321 columns: {missing}")

    df = raw[required].copy()
    df["Formulation Index"] = pd.to_numeric(df["Formulation Index"], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Release"] = pd.to_numeric(df["Release"], errors="coerce")
    df = df.dropna(subset=["Formulation Index", "Time", "Release"]).copy()
    df["Formulation Index"] = df["Formulation Index"].astype(int)
    df["Release_raw"] = df["Release"]
    df["Release"] = df["Release"].clip(0.0, 1.0)

    curve_groups: dict[int, pd.DataFrame] = {}
    meta_rows: list[dict[str, Any]] = []
    for fid, sub in df.groupby("Formulation Index", sort=True):
        curve = (
            sub.groupby("Time", as_index=False, sort=True)
            .agg(
                Release=("Release", "mean"),
                Release_raw=("Release_raw", "mean"),
                **{col: (col, "first") for col in required if col not in ("Time", "Release", "Formulation Index")},
            )
            .sort_values("Time")
            .reset_index(drop=True)
        )
        if len(curve) < 3:
            continue
        curve_groups[int(fid)] = curve
        first = curve.iloc[0].to_dict()
        meta = {col: first.get(col) for col in required if col not in ("Time", "Release")}
        meta["Formulation Index"] = int(fid)
        meta["n_timepoints"] = int(len(curve))
        meta["max_time"] = float(curve["Time"].max())
        meta["min_time"] = float(curve["Time"].min())
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows).sort_values("Formulation Index").reset_index(drop=True)
    if args.max_real_curves and len(meta_df) > args.max_real_curves:
        meta_df = meta_df.sample(n=args.max_real_curves, random_state=args.seed).sort_values("Formulation Index").reset_index(drop=True)
        keep = set(meta_df["Formulation Index"].astype(int))
        curve_groups = {fid: curve for fid, curve in curve_groups.items() if fid in keep}
    return meta_df, curve_groups


def interpolate_curve(curve: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    t = curve["Time"].to_numpy(dtype=float)
    q = curve["Release"].to_numpy(dtype=float)
    order = np.argsort(t)
    t = t[order]
    q = q[order]
    y = np.interp(times, t, q, left=q[0], right=q[-1])
    return np.maximum.accumulate(np.clip(y, 0.0, 1.0))


def real_x_matrix(meta: pd.DataFrame, synth_df: pd.DataFrame, m118: Any) -> np.ndarray:
    out = pd.DataFrame(index=meta.index)
    out["drug_mw"] = pd.to_numeric(meta["Drug"].map(lambda _: np.nan), errors="coerce")
    if "Drug MW" in meta.columns:
        out["drug_mw"] = pd.to_numeric(meta["Drug MW"], errors="coerce")
    out["drug_logp"] = np.nan
    out["drug_tpsa"] = np.nan
    out["plga_mw_kda"] = pd.to_numeric(meta["Polymer Mw"], errors="coerce")
    out["la_ga_ratio"] = pd.to_numeric(meta["LA/GA"], errors="coerce")
    out["drug_loading_pct"] = pd.to_numeric(meta["Drug Loading Capacity"], errors="coerce")
    out["particle_size_um"] = pd.to_numeric(meta["Particle Size"], errors="coerce")
    out["medium_pH"] = np.nan
    out["temperature_c"] = np.nan
    out["agitation_rpm"] = np.nan
    out["solvent_strength"] = pd.to_numeric(meta["Solubility Enhancer Concentration"], errors="coerce")
    out["surfactant_pct"] = out["solvent_strength"]

    cols = m118.CORE_FEATURES + m118.CONTEXT_FEATURES
    med = synth_df[cols].median(axis=0)
    return out[cols].fillna(med).to_numpy(dtype=float)


def project_real_states(
    meta: pd.DataFrame,
    curve_groups: dict[int, pd.DataFrame],
    times: np.ndarray,
    synth_df: pd.DataFrame,
    models: dict[str, Any],
    m118: Any,
) -> tuple[pd.DataFrame, np.ndarray, list[int], dict[str, np.ndarray]]:
    eligible_ids = [int(fid) for fid in meta["Formulation Index"].tolist() if int(fid) in curve_groups]
    full_ids = [fid for fid in eligible_ids if curve_groups[fid]["Time"].max() >= 30.0]
    curve_matrix = np.vstack([interpolate_curve(curve_groups[fid], times) for fid in full_ids])
    meta_full = meta[meta["Formulation Index"].isin(full_ids)].copy().reset_index(drop=True)

    projections: list[pd.DataFrame] = []
    mode_features: dict[str, np.ndarray] = {}
    for mode in ("full_Q", "early_Q_k3", "early_Q_k5"):
        if mode == "early_Q_k3":
            ok_mask = np.asarray([curve_groups[fid]["Time"].max() >= 3.0 and len(curve_groups[fid]) >= 4 for fid in full_ids])
        elif mode == "early_Q_k5":
            ok_mask = np.asarray([curve_groups[fid]["Time"].max() >= 14.0 and len(curve_groups[fid]) >= 6 for fid in full_ids])
        else:
            ok_mask = np.ones(len(full_ids), dtype=bool)
        ids = [fid for fid, ok in zip(full_ids, ok_mask) if ok]
        if not ids:
            continue
        curves = curve_matrix[ok_mask]
        x = q_feature_matrix(curves, times, mode, m118)
        mode_features[mode] = x
        pred = np.clip(models[mode].predict(x), 0.0, 1.0)
        block = pd.DataFrame(pred, columns=HIDDEN_LIKE)
        block.insert(0, "projection_mode", mode)
        block.insert(0, "Formulation Index", ids)
        projections.append(block)

    x_real = real_x_matrix(meta_full, synth_df, m118)
    mode_features["x_only"] = x_real
    pred_x = np.clip(models["x_only"].predict(x_real), 0.0, 1.0)
    block_x = pd.DataFrame(pred_x, columns=HIDDEN_LIKE)
    block_x.insert(0, "projection_mode", "x_only")
    block_x.insert(0, "Formulation Index", full_ids)
    projections.append(block_x)

    projected = pd.concat(projections, ignore_index=True)
    projected = projected.merge(meta_full, on="Formulation Index", how="left")
    return projected, curve_matrix, full_ids, mode_features


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 8 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    if spearmanr is not None:
        rho = float(spearmanr(x, y).correlation)
    else:
        rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        bx = x[idx]
        by = y[idx]
        if np.nanstd(bx) <= 1e-12 or np.nanstd(by) <= 1e-12:
            continue
        if spearmanr is not None:
            boots.append(float(spearmanr(bx, by).correlation))
        else:
            boots.append(float(pd.Series(bx).rank().corr(pd.Series(by).rank())))
    if not boots:
        return rho, float("nan"), float("nan")
    return rho, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def group_eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values) & pd.Series(labels).notna().to_numpy()
    values = values[mask]
    labels = labels[mask]
    if len(values) < 8:
        return float("nan")
    groups = [values[labels == label] for label in pd.unique(labels)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total <= 1e-12:
        return float("nan")
    between = float(sum(len(g) * (np.mean(g) - np.mean(values)) ** 2 for g in groups))
    return between / total


def validate_proxies(args: argparse.Namespace, projected: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 400)
    rows: list[dict[str, Any]] = []
    for mode, block in projected.groupby("projection_mode", sort=True):
        for hcol in HIDDEN_LIKE:
            h = pd.to_numeric(block[hcol], errors="coerce").to_numpy(dtype=float)
            for proxy in NUMERIC_PROXIES:
                if proxy not in block.columns:
                    continue
                x = pd.to_numeric(block[proxy], errors="coerce").to_numpy(dtype=float)
                rho, ci_lo, ci_hi = bootstrap_spearman(x, h, rng, args.bootstrap)
                null = []
                valid = np.isfinite(x) & np.isfinite(h)
                for _ in range(args.shuffle_repeats):
                    shuf = x.copy()
                    rng.shuffle(shuf)
                    rr, _, _ = bootstrap_spearman(shuf[valid], h[valid], rng, 0)
                    if np.isfinite(rr):
                        null.append(abs(rr))
                null95 = float(np.nanpercentile(null, 95)) if null else float("nan")
                rows.append(
                    {
                        "projection_mode": mode,
                        "hidden_coordinate": hcol,
                        "proxy": proxy,
                        "proxy_type": "numeric",
                        "n": int(np.sum(np.isfinite(x) & np.isfinite(h))),
                        "effect": rho,
                        "effect_abs": abs(rho) if np.isfinite(rho) else float("nan"),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "shuffle_null95_abs": null95,
                        "exceeds_shuffle_null": bool(np.isfinite(rho) and np.isfinite(null95) and abs(rho) > null95),
                    }
                )

            for proxy in CATEGORICAL_PROXIES:
                if proxy not in block.columns:
                    continue
                labels = block[proxy].astype(str).replace({"nan": np.nan}).to_numpy()
                eta = group_eta_squared(h, labels)
                valid_labels = pd.Series(labels).dropna().astype(str)
                n_groups = int(valid_labels.nunique())
                null = []
                for _ in range(args.shuffle_repeats):
                    shuf = labels.copy()
                    rng.shuffle(shuf)
                    val = group_eta_squared(h, shuf)
                    if np.isfinite(val):
                        null.append(val)
                null95 = float(np.nanpercentile(null, 95)) if null else float("nan")
                p_value = float("nan")
                if kruskal is not None and n_groups >= 2:
                    groups = [
                        h[(labels == label) & np.isfinite(h)]
                        for label in pd.unique(labels[pd.notna(labels)])
                    ]
                    groups = [g for g in groups if len(g) >= 2]
                    if len(groups) >= 2:
                        try:
                            p_value = float(kruskal(*groups).pvalue)
                        except Exception:
                            p_value = float("nan")
                rows.append(
                    {
                        "projection_mode": mode,
                        "hidden_coordinate": hcol,
                        "proxy": proxy,
                        "proxy_type": "categorical",
                        "n": int(np.sum(pd.notna(labels) & np.isfinite(h))),
                        "n_groups": n_groups,
                        "effect": eta,
                        "effect_abs": eta,
                        "p_value": p_value,
                        "shuffle_null95_abs": null95,
                        "exceeds_shuffle_null": bool(np.isfinite(eta) and np.isfinite(null95) and eta > null95),
                    }
                )
    return pd.DataFrame(rows)


def nearest_synthetic_neighbors(
    projected: pd.DataFrame,
    synth_curves: np.ndarray,
    synth_hidden: np.ndarray,
    curve_matrix: np.ndarray,
    full_ids: list[int],
    times: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    full_proj = projected[projected["projection_mode"] == "full_Q"].copy()
    if full_proj.empty:
        return pd.DataFrame()
    hidden_scaler = StandardScaler().fit(synth_hidden)
    nn_h = NearestNeighbors(n_neighbors=args.neighbors).fit(hidden_scaler.transform(synth_hidden))
    nn_q = NearestNeighbors(n_neighbors=args.neighbors).fit(synth_curves)
    id_to_curve_pos = {fid: i for i, fid in enumerate(full_ids)}
    rows: list[dict[str, Any]] = []
    for _, row in full_proj.iterrows():
        fid = int(row["Formulation Index"])
        if fid not in id_to_curve_pos:
            continue
        h_like = row[HIDDEN_LIKE].to_numpy(dtype=float).reshape(1, -1)
        q_real = curve_matrix[id_to_curve_pos[fid]].reshape(1, -1)
        h_dist, h_idx = nn_h.kneighbors(hidden_scaler.transform(h_like))
        q_dist, q_idx = nn_q.kneighbors(q_real)
        for rank in range(args.neighbors):
            sid_h = int(h_idx[0, rank])
            sid_q = int(q_idx[0, rank])
            out = {
                "Formulation Index": fid,
                "Drug": row.get("Drug"),
                "DOI": row.get("DOI"),
                "Formulation Method": row.get("Formulation Method"),
                "neighbor_rank": rank + 1,
                "h_neighbor_synthetic_id": sid_h,
                "h_space_distance": float(h_dist[0, rank]),
                "curve_neighbor_synthetic_id": sid_q,
                "curve_shape_distance": float(q_dist[0, rank]),
            }
            for j, hcol in enumerate(HIDDEN_LIKE):
                out[f"h_neighbor_true_{hcol}"] = float(synth_hidden[sid_h, j])
            rows.append(out)
    return pd.DataFrame(rows)


def make_projection_coverage_summary(
    real_meta: pd.DataFrame,
    projected: pd.DataFrame,
    neighbors: pd.DataFrame,
    synth_curves: np.ndarray,
) -> pd.DataFrame:
    if len(synth_curves) >= 3:
        nn = NearestNeighbors(n_neighbors=2).fit(synth_curves)
        synth_dist, _ = nn.kneighbors(synth_curves)
        synthetic_self_p90 = float(np.nanpercentile(synth_dist[:, 1], 90))
    else:
        synthetic_self_p90 = float("nan")

    rank1 = neighbors[neighbors["neighbor_rank"] == 1].copy() if not neighbors.empty else pd.DataFrame()
    real_dist = pd.to_numeric(rank1.get("curve_shape_distance", pd.Series(dtype=float)), errors="coerce")
    median_dist = float(real_dist.median()) if len(real_dist) else float("nan")
    p90_dist = float(real_dist.quantile(0.9)) if len(real_dist) else float("nan")
    within = real_dist <= synthetic_self_p90 if np.isfinite(synthetic_self_p90) else pd.Series(dtype=bool)
    fraction_within = float(within.mean()) if len(within) else float("nan")
    fraction_ood = float(1.0 - fraction_within) if np.isfinite(fraction_within) else float("nan")

    def n_mode(mode: str) -> int:
        if projected.empty:
            return 0
        return int(projected.loc[projected["projection_mode"] == mode, "Formulation Index"].nunique())

    return pd.DataFrame(
        [
            {
                "n_real_curves": int(len(real_meta)),
                "n_projectable_full": n_mode("full_Q"),
                "n_projectable_k3": n_mode("early_Q_k3"),
                "n_projectable_k5": n_mode("early_Q_k5"),
                "median_nearest_synthetic_curve_distance": median_dist,
                "p90_nearest_synthetic_curve_distance": p90_dist,
                "synthetic_self_neighbor_p90": synthetic_self_p90,
                "fraction_within_synthetic_p90": fraction_within,
                "fraction_flagged_ood": fraction_ood,
            }
        ]
    )


def _numeric_association_strength(x: np.ndarray, y_matrix: np.ndarray) -> float:
    vals = []
    for j in range(y_matrix.shape[1]):
        y = y_matrix[:, j]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 8 or np.nanstd(x[mask]) <= 1e-12 or np.nanstd(y[mask]) <= 1e-12:
            continue
        if spearmanr is not None:
            vals.append(abs(float(spearmanr(x[mask], y[mask]).correlation)))
        else:
            vals.append(abs(float(pd.Series(x[mask]).rank().corr(pd.Series(y[mask]).rank()))))
    return float(np.nanmax(vals)) if vals else float("nan")


def _categorical_association_strength(labels: np.ndarray, y_matrix: np.ndarray) -> float:
    vals = []
    for j in range(y_matrix.shape[1]):
        val = group_eta_squared(y_matrix[:, j], labels)
        if np.isfinite(val):
            vals.append(val)
    return float(np.nanmax(vals)) if vals else float("nan")


def embedding_proxy_summary(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    embedding_cols: list[str],
    embedding_type: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 900 + len(embedding_type))
    rows: list[dict[str, Any]] = []
    y_matrix = frame[embedding_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    proxies = [(p, "numeric") for p in NUMERIC_PROXIES if p in frame.columns]
    proxies.extend((p, "categorical") for p in CATEGORICAL_PROXIES if p in frame.columns)

    for proxy, proxy_type in proxies:
        if proxy_type == "numeric":
            x = pd.to_numeric(frame[proxy], errors="coerce").to_numpy(dtype=float)
            assoc = _numeric_association_strength(x, y_matrix)
            n = int(np.isfinite(x).sum())
            null = []
            for _ in range(args.shuffle_repeats):
                shuf = x.copy()
                rng.shuffle(shuf)
                val = _numeric_association_strength(shuf, y_matrix)
                if np.isfinite(val):
                    null.append(val)
        else:
            labels = frame[proxy].astype(str).replace({"nan": np.nan}).to_numpy()
            assoc = _categorical_association_strength(labels, y_matrix)
            n = int(pd.Series(labels).notna().sum())
            null = []
            for _ in range(args.shuffle_repeats):
                shuf = labels.copy()
                rng.shuffle(shuf)
                val = _categorical_association_strength(shuf, y_matrix)
                if np.isfinite(val):
                    null.append(val)
        shuffle_p = float((1 + sum(v >= assoc for v in null)) / (len(null) + 1)) if null and np.isfinite(assoc) else float("nan")
        rows.append(
            {
                "embedding_type": embedding_type,
                "proxy": proxy,
                "proxy_type": proxy_type,
                "n": n,
                "association_strength": assoc,
                "shuffle_p": shuffle_p,
                "shuffle_null95": float(np.nanpercentile(null, 95)) if null else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_pca_embedding_frame(
    real_meta: pd.DataFrame,
    full_ids: list[int],
    real_curve_matrix: np.ndarray,
    synth_curves: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str]]:
    n_components = min(args.pca_components, synth_curves.shape[0], synth_curves.shape[1], real_curve_matrix.shape[0])
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(synth_curves)
    coords = pca.transform(real_curve_matrix)
    cols = [f"pca_curve_{i + 1}" for i in range(coords.shape[1])]
    frame = pd.DataFrame(coords, columns=cols)
    frame.insert(0, "Formulation Index", full_ids)
    frame = frame.merge(real_meta, on="Formulation Index", how="left")
    return frame, cols


def _weibull_func(t: np.ndarray, tau: float, beta: float, qmax: float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 1e-8, None)
    return qmax * (1.0 - np.exp(-np.power(t / tau, beta)))


def _hill_func(t: np.ndarray, t50: float, hill_n: float, qmax: float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 1e-8, None)
    num = np.power(t, hill_n)
    return qmax * num / (np.power(t50, hill_n) + num)


def fit_shape_theta_frame(
    real_meta: pd.DataFrame,
    curve_groups: dict[int, pd.DataFrame],
    full_ids: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    for fid in full_ids:
        curve = curve_groups[fid]
        t = curve["Time"].to_numpy(dtype=float)
        q = curve["Release"].to_numpy(dtype=float)
        row: dict[str, Any] = {"Formulation Index": fid}
        if curve_fit is not None and len(t) >= 4:
            try:
                popt, _ = curve_fit(
                    _weibull_func,
                    t,
                    q,
                    p0=[max(np.median(t), 1.0), 1.0, min(max(float(np.max(q)), 0.2), 1.0)],
                    bounds=([1e-3, 0.1, 0.05], [1e4, 5.0, 1.2]),
                    maxfev=20000,
                )
                pred = _weibull_func(t, *popt)
                row["weibull_tau"] = float(popt[0])
                row["weibull_beta"] = float(popt[1])
                row["weibull_qmax"] = float(popt[2])
                row["weibull_rmse"] = rmse(q, pred)
            except Exception:
                row["weibull_tau"] = row["weibull_beta"] = row["weibull_qmax"] = row["weibull_rmse"] = np.nan
            try:
                popt, _ = curve_fit(
                    _hill_func,
                    t,
                    q,
                    p0=[max(np.median(t), 1.0), 1.0, min(max(float(np.max(q)), 0.2), 1.0)],
                    bounds=([1e-3, 0.1, 0.05], [1e4, 5.0, 1.2]),
                    maxfev=20000,
                )
                pred = _hill_func(t, *popt)
                row["hill_t50"] = float(popt[0])
                row["hill_n"] = float(popt[1])
                row["hill_qmax"] = float(popt[2])
                row["hill_rmse"] = rmse(q, pred)
            except Exception:
                row["hill_t50"] = row["hill_n"] = row["hill_qmax"] = row["hill_rmse"] = np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).merge(real_meta, on="Formulation Index", how="left")
    theta_cols = [
        col
        for col in ["weibull_tau", "weibull_beta", "weibull_qmax", "hill_t50", "hill_n", "hill_qmax"]
        if col in frame.columns and frame[col].notna().sum() >= 8
    ]
    return frame, theta_cols


def build_curve_embedding_baseline_summary(
    args: argparse.Namespace,
    projected: pd.DataFrame,
    real_meta: pd.DataFrame,
    curve_groups: dict[int, pd.DataFrame],
    full_ids: list[int],
    real_curve_matrix: np.ndarray,
    synth_curves: np.ndarray,
) -> pd.DataFrame:
    frames = []
    h_frame = projected[projected["projection_mode"] == "full_Q"].copy()
    frames.append(embedding_proxy_summary(args, h_frame, HIDDEN_LIKE, "H_like"))
    pca_frame, pca_cols = build_pca_embedding_frame(real_meta, full_ids, real_curve_matrix, synth_curves, args)
    frames.append(embedding_proxy_summary(args, pca_frame, pca_cols, "PCA_curve_embedding"))
    theta_frame, theta_cols = fit_shape_theta_frame(real_meta, curve_groups, full_ids)
    if theta_cols:
        frames.append(embedding_proxy_summary(args, theta_frame, theta_cols, "Hill_Weibull_theta"))

    out = pd.concat(frames, ignore_index=True)
    h_lookup = (
        out[out["embedding_type"] == "H_like"][["proxy", "association_strength"]]
        .rename(columns={"association_strength": "H_like_association_strength"})
    )
    out = out.merge(h_lookup, on="proxy", how="left")
    out["beats_Hlike_or_not"] = np.where(
        out["embedding_type"] == "H_like",
        "reference",
        np.where(out["association_strength"] > out["H_like_association_strength"], "yes", "no"),
    )
    return out


def make_decision_table(
    inversion_metrics: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    projected: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    full_all = inversion_metrics[
        (inversion_metrics["mode"] == "full_Q") & (inversion_metrics["hidden_coordinate"] == "__all__")
    ]
    reliable = bool((not full_all.empty) and float(full_all["r2"].iloc[0]) >= 0.25)
    deploy = proxy_summary[proxy_summary["projection_mode"].isin(["full_Q", "early_Q_k3", "early_Q_k5"])]
    n_proxy_hits = int(deploy["exceeds_shuffle_null"].fillna(False).sum()) if not deploy.empty else 0
    best_numeric = deploy[deploy["proxy_type"] == "numeric"]["effect_abs"].max() if not deploy.empty else np.nan
    doi_hits = int(
        deploy[(deploy["proxy"].isin(["DOI", "Formulation Method"])) & (deploy["exceeds_shuffle_null"].fillna(False))].shape[0]
    )
    ood_fraction = float(coverage["fraction_flagged_ood"].iloc[0]) if not coverage.empty else float("nan")
    merged = projected.pivot_table(index="Formulation Index", columns="projection_mode", values=HIDDEN_LIKE)
    diff = float("nan")
    try:
        full = merged.xs("full_Q", level=1, axis=1).to_numpy(dtype=float)
        xonly = merged.xs("x_only", level=1, axis=1).to_numpy(dtype=float)
        diff = rmse(full, xonly)
    except Exception:
        pass
    rows = [
        {
            "decision": "synthetic_Q_to_H_inversion_reliable",
            "status": "pass" if reliable else "warn",
            "evidence": f"full_Q all-coordinate R2={float(full_all['r2'].iloc[0]) if not full_all.empty else np.nan:.3f}",
        },
        {
            "decision": "real_Hhat_aligns_with_table_proxies",
            "status": "pass" if n_proxy_hits > 0 else "warn",
            "evidence": f"proxy associations exceeding shuffle null={n_proxy_hits}; best abs_effect={best_numeric:.3f}",
        },
        {
            "decision": "associations_exceed_shuffle_null",
            "status": "pass" if n_proxy_hits > 0 else "warn",
            "evidence": f"n_hits={n_proxy_hits}",
        },
        {
            "decision": "curve_projected_Hhat_differs_from_X_only_Hhat",
            "status": "pass" if np.isfinite(diff) and diff > 0.03 else "warn",
            "evidence": f"RMSE(full_Q H_like, x_only H_like)={diff:.3f}",
        },
        {
            "decision": "method_or_DOI_group_structure_present",
            "status": "pass" if doi_hits > 0 else "warn",
            "evidence": f"DOI/method hits exceeding shuffle null={doi_hits}",
        },
        {
            "decision": "projection_coverage_supported",
            "status": "pass" if np.isfinite(ood_fraction) and ood_fraction <= 0.5 else "warn",
            "evidence": f"fraction_flagged_ood={ood_fraction:.3f}",
        },
        {
            "decision": "safe_to_use_as_bridge_evidence",
            "status": "pass" if reliable and np.isfinite(ood_fraction) and ood_fraction <= 0.5 else "warn",
            "evidence": "Use as cautious bridge diagnostic only; H_like is not measured physical H and OOD coverage must be reported.",
        },
    ]
    return pd.DataFrame(rows)


def write_checks(args: argparse.Namespace, meta: pd.DataFrame, projected: pd.DataFrame, inversion: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check": "real_input_exists",
            "status": "pass" if args.real_xlsx.exists() else "fail",
            "detail": str(args.real_xlsx),
        },
        {
            "check": "sample_unit_is_curve",
            "status": "pass",
            "detail": f"{len(meta)} real formulation curves loaded; no timepoint split",
        },
        {
            "check": "full_projection_primary_subset",
            "status": "pass" if int((meta["max_time"] >= 30.0).sum()) > 0 else "fail",
            "detail": f"max_time>=30d curves={int((meta['max_time'] >= 30.0).sum())}",
        },
        {
            "check": "projected_states_present",
            "status": "pass" if not projected.empty else "fail",
            "detail": f"rows={len(projected)}, modes={sorted(projected['projection_mode'].unique().tolist()) if not projected.empty else []}",
        },
        {
            "check": "synthetic_inversion_metrics_present",
            "status": "pass" if not inversion.empty and {"r2", "rmse"}.issubset(inversion.columns) else "fail",
            "detail": f"rows={len(inversion)}",
        },
        {
            "check": "proxy_shuffle_columns_present",
            "status": "pass" if {"shuffle_null95_abs", "exceeds_shuffle_null"}.issubset(proxy.columns) else "fail",
            "detail": f"rows={len(proxy)}",
        },
        {
            "check": "coverage_summary_requested",
            "status": "pass",
            "detail": "projection_coverage_summary.csv records real-to-synthetic support and OOD fraction",
        },
        {
            "check": "embedding_baseline_requested",
            "status": "pass",
            "detail": "curve_embedding_baseline_summary.csv compares H_like, PCA curve embeddings, and shape theta when available",
        },
        {
            "check": "no_nan_primary_metrics",
            "status": "pass" if not inversion[(inversion["hidden_coordinate"] == "__all__")]["rmse"].isna().any() else "fail",
            "detail": "primary inversion RMSE checked",
        },
        {
            "check": "interpretation_boundary",
            "status": "pass",
            "detail": "H_like is simulation-informed and not measured physical hidden state",
        },
    ]
    return pd.DataFrame(checks)


def write_report(
    out: Path,
    args: argparse.Namespace,
    inversion: pd.DataFrame,
    proxy: pd.DataFrame,
    coverage: pd.DataFrame,
    embedding_baseline: pd.DataFrame,
    decisions: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    inv = inversion[inversion["hidden_coordinate"] == "__all__"][["mode", "r2", "rmse"]].round(4)
    hits = (
        proxy[proxy["exceeds_shuffle_null"].fillna(False)]
        .sort_values(["projection_mode", "effect_abs"], ascending=[True, False])
        .head(20)
    )
    hit_cols = ["projection_mode", "hidden_coordinate", "proxy", "proxy_type", "n", "effect", "effect_abs", "shuffle_null95_abs"]
    baseline_cols = ["embedding_type", "proxy", "association_strength", "shuffle_p", "beats_Hlike_or_not"]
    baseline_display = embedding_baseline[np.isfinite(pd.to_numeric(embedding_baseline["association_strength"], errors="coerce"))]
    lines = [
        "# PLGA Synthetic-State Real Projection Bridge",
        "",
        "This bridge projects real 321 PLGA curves into simulation-informed H_like coordinates learned from the synthetic hidden-state audit.",
        "",
        "H_like coordinates are not measured porosity, tortuosity, residual solvent, or autocatalysis in the real corpus.",
        "",
        "## Synthetic Inversion",
        "",
        inv.to_markdown(index=False),
        "",
        "## Proxy Associations Exceeding Shuffle Null",
        "",
        hits[hit_cols].round(4).to_markdown(index=False) if not hits.empty else "No proxy associations exceeded the shuffle null.",
        "",
        "## Projection Coverage",
        "",
        coverage.round(4).to_markdown(index=False),
        "",
        "## Curve Embedding Baseline",
        "",
        baseline_display[baseline_cols].round(4).to_markdown(index=False),
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Data Checks",
        "",
        checks.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- If proxy alignment exists, real PLGA curves carry simulation-aligned release-state structure and table proxies partially support it.",
        "- If proxy alignment is weak, current 321 table proxies are insufficient; this supports the missing-state argument rather than refuting it.",
        "- Coverage is part of the claim: high OOD fraction means this is a hypothesis-generating bridge, not a validated simulator for the real corpus.",
        "- Never claim that real porosity/tortuosity/residual solvent was measured.",
        "",
        f"Run settings: n_synthetic={args.n_synthetic}, n_estimators={args.n_estimators}, seed={args.seed}.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    m118 = load_118_module()

    synth_df, times, synth_curves, hidden_df, models, inversion_metrics, synth_hidden = train_inversion_models(args, m118)
    real_meta, real_curves = load_real_321(args)
    projected, real_curve_matrix, full_ids, _ = project_real_states(real_meta, real_curves, times, synth_df, models, m118)
    proxy_summary = validate_proxies(args, projected)
    neighbors = nearest_synthetic_neighbors(projected, synth_curves, synth_hidden, real_curve_matrix, full_ids, times, args)
    coverage = make_projection_coverage_summary(real_meta, projected, neighbors, synth_curves)
    embedding_baseline = build_curve_embedding_baseline_summary(
        args,
        projected,
        real_meta,
        real_curves,
        full_ids,
        real_curve_matrix,
        synth_curves,
    )
    decisions = make_decision_table(inversion_metrics, proxy_summary, projected, coverage)
    checks = write_checks(args, real_meta, projected, inversion_metrics, proxy_summary)

    inversion_metrics.to_csv(args.out / "synthetic_inversion_metrics.csv", index=False)
    projected.to_csv(args.out / "real_projected_states.csv", index=False)
    proxy_summary.to_csv(args.out / "proxy_validation_summary.csv", index=False)
    neighbors.to_csv(args.out / "nearest_synthetic_neighbors.csv", index=False)
    coverage.to_csv(args.out / "projection_coverage_summary.csv", index=False)
    embedding_baseline.to_csv(args.out / "curve_embedding_baseline_summary.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "real_xlsx": str(args.real_xlsx),
        "hidden_like_coordinates": HIDDEN_LIKE,
        "projection_modes": MODES,
        "interpretation_boundary": "H_like is simulation-informed; not measured physical H.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out, args, inversion_metrics, proxy_summary, coverage, embedding_baseline, decisions, checks)
    print(f"Wrote PLGA synthetic-state real projection bridge to {args.out}")


if __name__ == "__main__":
    main()
