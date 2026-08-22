"""
121 - PLGA candidate measurement value scoring.

Purpose:
    Rank candidate measurements for explaining Experiment 120 PLGA
    missing-state residuals, while keeping empirical evidence and future
    hypotheses separate.

Consumes:
    outputs/120_plga_missing_state_residual_map/
    D:/chemical-world-model-v0/datset/321PLGA/.../mp_dataset_initial.xlsx

Produces:
    outputs/121_plga_candidate_measurement_value_scoring/
      candidate_measurement_library.csv
      candidate_signature_matrix.csv
      proxy_validation_summary.csv
      proxy_prediction_gain_summary.csv
      source_confound_audit.csv
      candidate_variable_priority.csv
      null_candidate_score_distribution.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    121 ranks candidate measurements. It does not discover physical hidden
    variables and does not output formulation-level remeasurement plans.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_120 = Path("outputs/120_plga_missing_state_residual_map")
DEFAULT_OUT = Path("outputs/121_plga_candidate_measurement_value_scoring")
DEFAULT_DOC = Path("docs/plga_candidate_measurement_value_scoring_2026-06-13.md")
SCRIPT_120_PATH = ROOT / "scripts" / "120_plga_missing_state_residual_map.py"

PRIMARY_SPLITS = ("group_by_DOI", "group_by_Formulation_Method")
PRIMARY_STATE_SPACES = ("pca8", "pca3", "weibull")
PRIMARY_PRIOR = "extra_trees"
SENSITIVITY_SPLIT = "random_5fold"
SENSITIVITY_PRIOR = "ridge"
RESIDUAL_TARGETS = [
    "residual_norm",
    "burst_residual",
    "early_slope_residual",
    "middle_diffusion_residual",
    "late_tail_residual",
    "qmax_residual",
]
SYMPTOM_TARGETS = [
    "burst_residual",
    "early_slope_residual",
    "middle_diffusion_residual",
    "late_tail_residual",
    "qmax_residual",
]
SOURCE_COLUMNS = ["DOI", "Formulation Method", "Drug"]
COST_PENALTIES = {"low": 0.15, "medium": 0.50, "high": 1.00}


def parse_args() -> argparse.Namespace:
    module_120 = load_script_120()
    parser = argparse.ArgumentParser(description="PLGA candidate measurement value scoring.")
    parser.add_argument("--input-120", type=Path, default=DEFAULT_INPUT_120)
    parser.add_argument("--real-xlsx", type=Path, default=module_120.DEFAULT_REAL_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--shuffle-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--n-null-candidates", type=int, default=100)
    parser.add_argument("--max-curves", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_estimators = min(args.n_estimators, 50)
        args.shuffle_repeats = min(args.shuffle_repeats, 50)
        args.bootstrap_repeats = min(args.bootstrap_repeats, 50)
        args.n_null_candidates = min(args.n_null_candidates, 20)
        if args.max_curves <= 0:
            args.max_curves = 45
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


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {"exists": True, "path": str(path), "size": stat.st_size, "mtime": stat.st_mtime}


def load_script_120():
    spec = importlib.util.spec_from_file_location("script120", SCRIPT_120_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT_120_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["script120"] = module
    spec.loader.exec_module(module)
    return module


def args_for_120(args: argparse.Namespace, lock: dict[str, Any], max_curves: int = 0) -> SimpleNamespace:
    lock_args = lock.get("args", {})
    return SimpleNamespace(
        real_xlsx=args.real_xlsx,
        out=args.input_120,
        doc=Path(lock_args.get("doc", "docs/plga_missing_state_residual_map_2026-06-13.md")),
        seed=int(lock_args.get("seed", args.seed)),
        grid_max_days=float(lock_args.get("grid_max_days", 90.0)),
        grid_size=int(lock_args.get("grid_size", 80)),
        min_timepoints=int(lock_args.get("min_timepoints", 3)),
        min_max_time=float(lock_args.get("min_max_time", 30.0)),
        n_folds=int(lock_args.get("n_folds", 5)),
        n_estimators=int(lock_args.get("n_estimators", args.n_estimators)),
        max_curves=max_curves,
        shuffle_repeats=int(lock_args.get("shuffle_repeats", 200)),
        smoke=False,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y - pred))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3 or float(np.nanvar(y[mask])) <= 1e-12:
        return float("nan")
    return float(r2_score(y[mask], pred[mask]))


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(value, 0.0, 1.0))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 6 or float(np.nanstd(xx[mask])) <= 1e-12 or float(np.nanstd(yy[mask])) <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    res = spearmanr(xx[mask], yy[mask])
    corr = float(res.correlation)
    pval = float(res.pvalue)
    return abs(corr), corr, pval


def group_eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    groups = pd.Series(labels).astype(str).replace({"nan": np.nan})
    mask = np.isfinite(y) & groups.notna().to_numpy()
    y = y[mask]
    g = groups.to_numpy()[mask]
    if len(y) < 8:
        return float("nan")
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1e-12:
        return float("nan")
    between = 0.0
    valid = 0
    for label in pd.unique(g):
        vals = y[g == label]
        if len(vals) < 2:
            continue
        valid += 1
        between += len(vals) * float((np.mean(vals) - np.mean(y)) ** 2)
    if valid < 2:
        return float("nan")
    return float(between / total)


def bh_fdr(pvalues: list[float]) -> list[float]:
    arr = np.asarray([1.0 if not np.isfinite(p) else p for p in pvalues], dtype=float)
    n = len(arr)
    if n == 0:
        return []
    order = np.argsort(arr)
    q = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = n - rank + 1
        running = min(running, arr[idx] * n / true_rank)
        q[idx] = running
    return [float(np.clip(v, 0.0, 1.0)) for v in q]


def load_120_tables(input_dir: Path) -> dict[str, Any]:
    required = [
        "missing_state_residuals.csv",
        "curve_state_table.csv",
        "static_state_prediction.csv",
        "lock_metadata.json",
        "data_checks.csv",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing 120 outputs: {missing}")
    return {
        "residuals": pd.read_csv(input_dir / "missing_state_residuals.csv"),
        "curve_state": pd.read_csv(input_dir / "curve_state_table.csv"),
        "static_prediction": pd.read_csv(input_dir / "static_state_prediction.csv"),
        "checks_120": pd.read_csv(input_dir / "data_checks.csv"),
        "lock": json.loads((input_dir / "lock_metadata.json").read_text(encoding="utf-8")),
    }


def make_candidate_library() -> pd.DataFrame:
    rows = [
        {
            "candidate_variable": "drug_loading_capacity",
            "candidate_family": "drug_distribution",
            "measurement_method": "assay_loading_or_HPLC",
            "evidence_tier": "direct_proxy",
            "proxy_columns": "Drug Loading Capacity",
            "cost_level": "low",
            "destructiveness": "low",
            "expected_information_gain_prior": 0.55,
            "safe_interpretation": "available loading proxy may explain missing-state residuals",
        },
        {
            "candidate_variable": "encapsulation_efficiency",
            "candidate_family": "drug_distribution",
            "measurement_method": "encapsulation_efficiency_assay",
            "evidence_tier": "direct_proxy",
            "proxy_columns": "Drug Encapsulation Efficiency",
            "cost_level": "low",
            "destructiveness": "low",
            "expected_information_gain_prior": 0.50,
            "safe_interpretation": "available encapsulation proxy may explain missing-state residuals",
        },
        {
            "candidate_variable": "initial_drug_polymer_ratio",
            "candidate_family": "formulation_ratio",
            "measurement_method": "reported_formulation_ratio",
            "evidence_tier": "direct_proxy",
            "proxy_columns": "Initial Drug-to-Polymer Ratio",
            "cost_level": "low",
            "destructiveness": "none",
            "expected_information_gain_prior": 0.45,
            "safe_interpretation": "reported ratio is an available formulation proxy",
        },
        {
            "candidate_variable": "PLGA_Mw_Mn_PDI_GPC",
            "candidate_family": "polymer_distribution",
            "measurement_method": "GPC_or_reported_polymer_spec",
            "evidence_tier": "direct_proxy",
            "proxy_columns": "Polymer Mw|Polymer Mn|PDI",
            "cost_level": "medium",
            "destructiveness": "medium",
            "expected_information_gain_prior": 0.65,
            "safe_interpretation": "available molecular-weight proxies may explain residuals",
        },
        {
            "candidate_variable": "LA_GA_ratio",
            "candidate_family": "polymer_composition",
            "measurement_method": "NMR_or_reported_LA_GA",
            "evidence_tier": "direct_proxy",
            "proxy_columns": "LA/GA",
            "cost_level": "medium",
            "destructiveness": "medium",
            "expected_information_gain_prior": 0.60,
            "safe_interpretation": "available LA/GA proxy may explain residuals",
        },
        {
            "candidate_variable": "particle_size_distribution",
            "candidate_family": "particle_morphology",
            "measurement_method": "DLS_laser_diffraction_or_SEM_distribution",
            "evidence_tier": "partial_proxy",
            "proxy_columns": "Particle Size",
            "cost_level": "low",
            "destructiveness": "low",
            "expected_information_gain_prior": 0.70,
            "safe_interpretation": "current table has particle-size mean, not full distribution",
        },
        {
            "candidate_variable": "surface_drug_fraction",
            "candidate_family": "drug_distribution",
            "measurement_method": "surface_wash_or_XPS",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "medium",
            "destructiveness": "medium",
            "expected_information_gain_prior": 0.75,
            "safe_interpretation": "hypothesis only; requires direct experimental measurement",
        },
        {
            "candidate_variable": "SEM_porosity_score",
            "candidate_family": "microstructure",
            "measurement_method": "SEM_image_porosity_score",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "medium",
            "destructiveness": "medium",
            "expected_information_gain_prior": 0.80,
            "safe_interpretation": "hypothesis only; SEM score would test morphology contribution",
        },
        {
            "candidate_variable": "swelling_ratio",
            "candidate_family": "hydration",
            "measurement_method": "mass_uptake_or_swelling_assay",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "low",
            "destructiveness": "medium",
            "expected_information_gain_prior": 0.70,
            "safe_interpretation": "hypothesis only; swelling proxy is not in current table",
        },
        {
            "candidate_variable": "DSC_Tg",
            "candidate_family": "thermal_state",
            "measurement_method": "DSC_glass_transition",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "medium",
            "destructiveness": "medium",
            "expected_information_gain_prior": 0.55,
            "safe_interpretation": "hypothesis only; thermal state requires direct measurement",
        },
        {
            "candidate_variable": "residual_solvent_GC",
            "candidate_family": "process_residue",
            "measurement_method": "GC_residual_solvent",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "high",
            "destructiveness": "high",
            "expected_information_gain_prior": 0.75,
            "safe_interpretation": "hypothesis only; no current proxy supports empirical evidence",
        },
        {
            "candidate_variable": "degradation_mass_loss_or_GPC_after_release",
            "candidate_family": "degradation_state",
            "measurement_method": "mass_loss_or_post_release_GPC",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "high",
            "destructiveness": "high",
            "expected_information_gain_prior": 0.70,
            "safe_interpretation": "hypothesis only; post-release degradation needs validation",
        },
        {
            "candidate_variable": "zeta_or_surface_charge",
            "candidate_family": "surface_chemistry",
            "measurement_method": "zeta_potential",
            "evidence_tier": "no_proxy",
            "proxy_columns": "",
            "cost_level": "low",
            "destructiveness": "low",
            "expected_information_gain_prior": 0.45,
            "safe_interpretation": "hypothesis only; surface charge is not in current table",
        },
        {
            "candidate_variable": "negative_control_random_measurement",
            "candidate_family": "negative_control",
            "measurement_method": "permuted_existing_proxy",
            "evidence_tier": "negative_control",
            "proxy_columns": "Drug Loading Capacity",
            "cost_level": "low",
            "destructiveness": "none",
            "expected_information_gain_prior": 0.00,
            "safe_interpretation": "fixed negative control for readability; null ensemble is authoritative",
        },
    ]
    df = pd.DataFrame(rows)
    df["proxy_available"] = df["evidence_tier"].isin(["direct_proxy", "partial_proxy", "negative_control"])
    fidelity = {"direct_proxy": 1.0, "partial_proxy": 0.5, "no_proxy": 0.0, "negative_control": 1.0}
    df["proxy_fidelity"] = df["evidence_tier"].map(fidelity).astype(float)
    df["measurement_cost_raw"] = df["cost_level"].map(COST_PENALTIES)
    df["measurement_cost_penalty"] = df["measurement_cost_raw"]
    df["evidence_status"] = np.where(df["evidence_tier"].eq("no_proxy"), "hypothesis_only", "proxy_available")
    df.loc[df["evidence_tier"].eq("negative_control"), "evidence_status"] = "negative_control"
    return df


def make_signature_matrix(library: pd.DataFrame) -> pd.DataFrame:
    signatures = {
        "drug_loading_capacity": [0.8, 0.4, 0.1, 0.0, 0.6],
        "encapsulation_efficiency": [-0.6, -0.2, 0.0, 0.0, -0.2],
        "initial_drug_polymer_ratio": [0.7, 0.4, 0.1, 0.0, 0.5],
        "PLGA_Mw_Mn_PDI_GPC": [-0.3, -0.5, -0.5, 0.5, -0.2],
        "LA_GA_ratio": [-0.2, -0.4, -0.4, 0.5, -0.2],
        "particle_size_distribution": [-0.6, -0.6, -0.2, 0.3, -0.1],
        "surface_drug_fraction": [0.9, 0.5, 0.1, 0.0, 0.2],
        "SEM_porosity_score": [0.4, 0.5, 0.9, 0.3, 0.2],
        "swelling_ratio": [0.2, 0.5, 0.9, 0.5, 0.2],
        "DSC_Tg": [-0.2, -0.5, -0.5, -0.2, -0.3],
        "residual_solvent_GC": [0.8, 0.8, 0.3, -0.2, 0.0],
        "degradation_mass_loss_or_GPC_after_release": [0.0, 0.1, 0.4, 0.9, 0.4],
        "zeta_or_surface_charge": [0.3, 0.2, 0.0, 0.0, 0.1],
        "negative_control_random_measurement": [0.1, -0.2, 0.3, -0.1, 0.2],
    }
    rows = []
    for _, row in library.iterrows():
        values = signatures[row["candidate_variable"]]
        rows.append(
            {
                "candidate_variable": row["candidate_variable"],
                "signature_basis": "curve_residual_symptom_delta_Q",
                "burst_residual_signature": values[0],
                "early_slope_residual_signature": values[1],
                "middle_diffusion_residual_signature": values[2],
                "late_tail_residual_signature": values[3],
                "qmax_residual_signature": values[4],
            }
        )
    return pd.DataFrame(rows)


def parse_proxy_columns(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def rebuild_120_data(args: argparse.Namespace, lock: dict[str, Any], max_curves: int = 0):
    module_120 = load_script_120()
    a120 = args_for_120(args, lock, max_curves=max_curves)
    meta, curve_groups = module_120.load_real_321(a120)
    grid = np.linspace(0.0, a120.grid_max_days, a120.grid_size)
    y_grid = module_120.build_curve_matrix(meta, curve_groups, grid)
    splits = module_120.make_splits(meta, a120)
    return module_120, a120, meta, curve_groups, grid, y_grid, splits


def verify_120_oof(
    args: argparse.Namespace,
    tables: dict[str, Any],
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    lock = tables["lock"]
    residuals = tables["residuals"]
    _, _, meta, _, _, _, splits = rebuild_120_data(args, lock, max_curves=0)
    fold_rows = []
    for split, fold, train_idx, test_idx in splits:
        train_ids = set(meta.iloc[train_idx]["Formulation Index"].astype(int))
        test_ids = set(meta.iloc[test_idx]["Formulation Index"].astype(int))
        row: dict[str, Any] = {
            "split": split,
            "fold": int(fold),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_ids": train_ids,
            "test_ids": test_ids,
        }
        if split == "group_by_DOI":
            train_groups = set(meta.iloc[train_idx]["DOI"].astype(str))
            test_groups = set(meta.iloc[test_idx]["DOI"].astype(str))
            row["group_column"] = "DOI"
            row["n_group_overlap"] = len(train_groups & test_groups)
        elif split == "group_by_Formulation_Method":
            train_groups = set(meta.iloc[train_idx]["Formulation Method"].astype(str))
            test_groups = set(meta.iloc[test_idx]["Formulation Method"].astype(str))
            row["group_column"] = "Formulation Method"
            row["n_group_overlap"] = len(train_groups & test_groups)
        else:
            row["group_column"] = ""
            row["n_group_overlap"] = 0
        fold_rows.append(row)
    fold_index = {(r["split"], int(r["fold"])): r for r in fold_rows}

    primary = residuals[
        residuals["split"].isin(PRIMARY_SPLITS)
        & residuals["state_space"].isin(PRIMARY_STATE_SPACES)
        & residuals["prior_model"].eq(PRIMARY_PRIOR)
        & (~residuals["is_oracle"].astype(bool))
    ].copy()

    detail_rows = []
    ok = True
    for (split, state_space, prior_model), block in primary.groupby(["split", "state_space", "prior_model"]):
        counts = block["Formulation Index"].astype(int).value_counts()
        duplicated = counts[counts != 1]
        if not duplicated.empty:
            ok = False
        detail_rows.append(
            {
                "check_scope": f"{split}|{state_space}|{prior_model}",
                "n_rows": len(block),
                "n_unique_curves": int(block["Formulation Index"].nunique()),
                "each_curve_once": bool(duplicated.empty),
            }
        )

    for _, row in primary.iterrows():
        key = (row["split"], int(row["fold"]))
        fold = fold_index.get(key)
        fid = int(row["Formulation Index"])
        in_test = bool(fold and fid in fold["test_ids"])
        in_train = bool(fold and fid in fold["train_ids"])
        group_ok = bool(fold and int(fold["n_group_overlap"]) == 0)
        row_ok = in_test and not in_train and group_ok
        if not row_ok:
            ok = False
        detail_rows.append(
            {
                "check_scope": f"{row['split']}|fold{int(row['fold'])}|curve{fid}",
                "n_rows": 1,
                "n_unique_curves": 1,
                "each_curve_once": True,
                "curve_in_fold_test": in_test,
                "curve_in_fold_train": in_train,
                "group_overlap_zero": group_ok,
                "row_oof_ok": row_ok,
            }
        )
    fold_table = pd.DataFrame(
        [
            {
                "split": r["split"],
                "fold": r["fold"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "group_column": r["group_column"],
                "n_group_overlap": r["n_group_overlap"],
            }
            for r in fold_rows
        ]
    )
    return ok and not primary.empty, pd.DataFrame(detail_rows), fold_table


def prepare_meta_for_scoring(
    args: argparse.Namespace,
    lock: dict[str, Any],
    residuals: pd.DataFrame,
) -> tuple[Any, SimpleNamespace, pd.DataFrame, np.ndarray, np.ndarray, list[tuple[str, int, np.ndarray, np.ndarray]]]:
    max_curves = 0
    module_120, a120, meta, _, grid, y_grid, splits = rebuild_120_data(args, lock, max_curves=max_curves)
    if args.max_curves and len(meta) > args.max_curves:
        primary_ids = (
            residuals[
                residuals["split"].isin(PRIMARY_SPLITS)
                & residuals["prior_model"].eq(PRIMARY_PRIOR)
                & residuals["state_space"].eq("pca8")
            ]["Formulation Index"]
            .astype(int)
            .drop_duplicates()
            .sort_values()
        )
        keep_ids = set(primary_ids.sample(n=min(args.max_curves, len(primary_ids)), random_state=args.seed).tolist())
        mask = meta["Formulation Index"].astype(int).isin(keep_ids).to_numpy()
        meta = meta.loc[mask].reset_index(drop=True)
        y_grid = y_grid[mask]
        a120 = args_for_120(args, lock, max_curves=0)
        splits = module_120.make_splits(meta, a120)
    return module_120, a120, meta, grid, y_grid, splits


def numeric_proxy(meta: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(meta[column], errors="coerce") if column in meta.columns else pd.Series(np.nan, index=meta.index)


def build_feature_matrix(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    blocks_train: list[np.ndarray] = []
    blocks_test: list[np.ndarray] = []
    for col in numeric_cols:
        train = pd.to_numeric(train_meta[col], errors="coerce") if col in train_meta.columns else pd.Series(np.nan, index=train_meta.index)
        test = pd.to_numeric(test_meta[col], errors="coerce") if col in test_meta.columns else pd.Series(np.nan, index=test_meta.index)
        median = 0.0 if train.dropna().empty else float(train.dropna().median())
        blocks_train.append(train.fillna(median).to_numpy(dtype=float)[:, None])
        blocks_test.append(test.fillna(median).to_numpy(dtype=float)[:, None])
    for col in categorical_cols:
        train = train_meta[col].astype(str).fillna("missing") if col in train_meta.columns else pd.Series("missing", index=train_meta.index)
        test = test_meta[col].astype(str).fillna("missing") if col in test_meta.columns else pd.Series("missing", index=test_meta.index)
        cats = sorted(train.unique().tolist())
        cat_to_idx = {cat: i for i, cat in enumerate(cats)}
        tr = np.zeros((len(train), len(cats) + 1), dtype=float)
        te = np.zeros((len(test), len(cats) + 1), dtype=float)
        for i, value in enumerate(train):
            tr[i, cat_to_idx.get(value, len(cats))] = 1.0
        for i, value in enumerate(test):
            te[i, cat_to_idx.get(value, len(cats))] = 1.0
        blocks_train.append(tr)
        blocks_test.append(te)
    if not blocks_train:
        return np.zeros((len(train_meta), 0)), np.zeros((len(test_meta), 0))
    return np.hstack(blocks_train), np.hstack(blocks_test)


def candidate_proxy_frame(
    meta: pd.DataFrame,
    candidate: pd.Series,
    rng: np.random.Generator,
    null_values: np.ndarray | None = None,
) -> pd.DataFrame:
    cols = parse_proxy_columns(candidate["proxy_columns"])
    if candidate["candidate_variable"].startswith("null_candidate_") and null_values is not None:
        return pd.DataFrame({"null_proxy": null_values}, index=meta.index)
    if candidate["candidate_variable"] == "negative_control_random_measurement":
        base = numeric_proxy(meta, "Drug Loading Capacity").to_numpy(dtype=float)
        shuffled = base.copy()
        mask = np.isfinite(shuffled)
        shuffled[mask] = rng.permutation(shuffled[mask])
        return pd.DataFrame({"negative_control_proxy": shuffled}, index=meta.index)
    data = {}
    for col in cols:
        data[col] = numeric_proxy(meta, col).to_numpy(dtype=float)
    return pd.DataFrame(data, index=meta.index)


def normalize_support(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return clip01(float(np.nanmean(arr)))


def max_candidate_stat(
    residuals: pd.DataFrame,
    meta: pd.DataFrame,
    proxy_df: pd.DataFrame,
    candidate_variable: str,
    rng: np.random.Generator | None = None,
    shuffle: bool = False,
) -> dict[str, Any]:
    meta_by_id = meta.set_index("Formulation Index")
    proxy_by_id = proxy_df.copy()
    proxy_by_id.index = meta["Formulation Index"].astype(int).to_numpy()
    if shuffle:
        for col in proxy_by_id.columns:
            values = proxy_by_id[col].to_numpy(dtype=float).copy()
            finite = np.isfinite(values)
            if rng is not None and finite.any():
                values[finite] = rng.permutation(values[finite])
            proxy_by_id[col] = values

    best: dict[str, Any] = {
        "candidate_variable": candidate_variable,
        "observed_max_stat": float("nan"),
        "raw_p": float("nan"),
        "best_proxy_column": "",
        "best_residual_target": "",
        "best_split": "",
        "best_state_space": "",
        "best_effect_sign": float("nan"),
    }
    max_stat = -np.inf
    for split in PRIMARY_SPLITS:
        for state_space in PRIMARY_STATE_SPACES:
            block = residuals[
                residuals["split"].eq(split)
                & residuals["state_space"].eq(state_space)
                & residuals["prior_model"].eq(PRIMARY_PRIOR)
                & (~residuals["is_oracle"].astype(bool))
            ].copy()
            if block.empty:
                continue
            ids = block["Formulation Index"].astype(int).to_numpy()
            for proxy_col in proxy_by_id.columns:
                if proxy_col not in proxy_by_id.columns:
                    continue
                x = proxy_by_id.reindex(ids)[proxy_col].to_numpy(dtype=float)
                for target in RESIDUAL_TARGETS:
                    stat, signed, pval = safe_spearman(x, block[target].to_numpy(dtype=float))
                    if np.isfinite(stat) and stat > max_stat:
                        max_stat = stat
                        best.update(
                            {
                                "observed_max_stat": stat,
                                "raw_p": pval,
                                "best_proxy_column": proxy_col,
                                "best_residual_target": target,
                                "best_split": split,
                                "best_state_space": state_space,
                                "best_effect_sign": np.sign(signed),
                            }
                        )
    return best


def frame_support_rows(
    residuals: pd.DataFrame,
    meta: pd.DataFrame,
    proxy_df: pd.DataFrame,
    candidate_variable: str,
    rng: np.random.Generator,
    shuffle_repeats: int,
) -> list[dict[str, Any]]:
    rows = []
    proxy_by_id = proxy_df.copy()
    proxy_by_id.index = meta["Formulation Index"].astype(int).to_numpy()
    for split in PRIMARY_SPLITS:
        for state_space in PRIMARY_STATE_SPACES:
            block = residuals[
                residuals["split"].eq(split)
                & residuals["state_space"].eq(state_space)
                & residuals["prior_model"].eq(PRIMARY_PRIOR)
                & (~residuals["is_oracle"].astype(bool))
            ].copy()
            if block.empty:
                continue
            ids = block["Formulation Index"].astype(int).to_numpy()
            best_stat = float("nan")
            best_sign = float("nan")
            for proxy_col in proxy_by_id.columns:
                x = proxy_by_id.reindex(ids)[proxy_col].to_numpy(dtype=float)
                for target in RESIDUAL_TARGETS:
                    stat, signed, _ = safe_spearman(x, block[target].to_numpy(dtype=float))
                    if np.isfinite(stat) and (not np.isfinite(best_stat) or stat > best_stat):
                        best_stat = stat
                        best_sign = float(np.sign(signed))
            null = []
            for _ in range(shuffle_repeats):
                max_stat = -np.inf
                for proxy_col in proxy_by_id.columns:
                    x = proxy_by_id.reindex(ids)[proxy_col].to_numpy(dtype=float)
                    finite = np.isfinite(x)
                    xs = x.copy()
                    if finite.any():
                        xs[finite] = rng.permutation(xs[finite])
                    for target in RESIDUAL_TARGETS:
                        stat, _, _ = safe_spearman(xs, block[target].to_numpy(dtype=float))
                        if np.isfinite(stat):
                            max_stat = max(max_stat, stat)
                if np.isfinite(max_stat):
                    null.append(max_stat)
            null95 = float(np.nanpercentile(null, 95)) if null else float("nan")
            rows.append(
                {
                    "candidate_variable": candidate_variable,
                    "split": split,
                    "state_space": state_space,
                    "frame_stat": best_stat,
                    "frame_effect_sign": best_sign,
                    "frame_maxT_95pct": null95,
                    "frame_above_null": bool(np.isfinite(best_stat) and np.isfinite(null95) and best_stat > null95),
                }
            )
    return rows


def proxy_validation(
    residuals: pd.DataFrame,
    meta: pd.DataFrame,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    null_values_by_candidate: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    frame_rows = []
    null_values_by_candidate = null_values_by_candidate or {}
    for _, candidate in candidates.iterrows():
        if candidate["evidence_tier"] == "no_proxy":
            continue
        proxy_df = candidate_proxy_frame(
            meta,
            candidate,
            rng,
            null_values=null_values_by_candidate.get(candidate["candidate_variable"]),
        )
        if proxy_df.empty:
            continue
        best = max_candidate_stat(residuals, meta, proxy_df, candidate["candidate_variable"])
        null = []
        for _ in range(args.shuffle_repeats):
            shuffled = max_candidate_stat(
                residuals,
                meta,
                proxy_df,
                candidate["candidate_variable"],
                rng=rng,
                shuffle=True,
            )
            if np.isfinite(shuffled["observed_max_stat"]):
                null.append(float(shuffled["observed_max_stat"]))
        null95 = float(np.nanpercentile(null, 95)) if null else float("nan")
        obs = best["observed_max_stat"]
        maxp = float((1 + np.sum(np.asarray(null) >= obs)) / (len(null) + 1)) if np.isfinite(obs) and null else float("nan")
        rows.append(
            {
                **best,
                "evidence_tier": candidate["evidence_tier"],
                "n_proxy_columns": int(proxy_df.shape[1]),
                "n_curves_with_any_proxy": int(proxy_df.notna().any(axis=1).sum()),
                "maxT_empirical_p": maxp,
                "maxT_95pct": null95,
                "beats_maxT_95pct": bool(np.isfinite(obs) and np.isfinite(null95) and obs > null95),
                "observed_proxy_support": float(obs) if np.isfinite(obs) and np.isfinite(null95) and obs > null95 else 0.0,
            }
        )
        frame_rows.extend(frame_support_rows(residuals, meta, proxy_df, candidate["candidate_variable"], rng, args.shuffle_repeats))
    out = pd.DataFrame(rows)
    if not out.empty:
        real_mask = out["evidence_tier"].isin(["direct_proxy", "partial_proxy", "negative_control"])
        out["BH_FDR_q"] = np.nan
        out.loc[real_mask, "BH_FDR_q"] = bh_fdr(out.loc[real_mask, "maxT_empirical_p"].tolist())
    return out, pd.DataFrame(frame_rows)


def fit_state_targets(
    module_120: Any,
    a120: SimpleNamespace,
    grid: np.ndarray,
    y_grid: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    state_space: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    space = module_120.make_state_space(state_space, seed)
    space.fit(y_grid[train_idx], grid)
    return space.encode(y_grid[train_idx], grid), space.encode(y_grid[test_idx], grid)


def candidate_numeric_sets(module_120: Any, proxy_cols: list[str]) -> tuple[list[str], list[str]]:
    base = list(module_120.NUMERIC_FEATURES)
    without = [col for col in base if col not in proxy_cols]
    with_candidate = list(dict.fromkeys(without + proxy_cols))
    return without, with_candidate


def train_predict_et(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def prediction_gain_summary(
    module_120: Any,
    meta: pd.DataFrame,
    y_grid: np.ndarray,
    grid: np.ndarray,
    splits: list[tuple[str, int, np.ndarray, np.ndarray]],
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    null_values_by_candidate: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    null_values_by_candidate = null_values_by_candidate or {}
    all_rows = []
    split_names = set(PRIMARY_SPLITS) | {SENSITIVITY_SPLIT}
    for split, fold, train_idx, test_idx in splits:
        if split not in split_names:
            continue
        train_meta = meta.iloc[train_idx].reset_index(drop=True)
        test_meta = meta.iloc[test_idx].reset_index(drop=True)
        for state_space in PRIMARY_STATE_SPACES:
            z_train, z_test = fit_state_targets(
                module_120,
                SimpleNamespace(),
                grid,
                y_grid,
                train_idx,
                test_idx,
                state_space,
                args.seed + fold,
            )
            for _, candidate in candidates.iterrows():
                if candidate["evidence_tier"] == "no_proxy":
                    continue
                proxy_cols = parse_proxy_columns(candidate["proxy_columns"])
                working_train = train_meta.copy()
                working_test = test_meta.copy()
                if candidate["candidate_variable"].startswith("null_candidate_"):
                    vals = null_values_by_candidate.get(candidate["candidate_variable"])
                    if vals is None:
                        continue
                    full = pd.Series(vals, index=meta.index)
                    working_train["null_proxy"] = full.iloc[train_idx].to_numpy(dtype=float)
                    working_test["null_proxy"] = full.iloc[test_idx].to_numpy(dtype=float)
                    proxy_cols = ["null_proxy"]
                elif candidate["candidate_variable"] == "negative_control_random_measurement":
                    base = numeric_proxy(meta, "Drug Loading Capacity").to_numpy(dtype=float)
                    finite = np.isfinite(base)
                    shuffled = base.copy()
                    if finite.any():
                        shuffled[finite] = rng.permutation(shuffled[finite])
                    full = pd.Series(shuffled, index=meta.index)
                    working_train["negative_control_proxy"] = full.iloc[train_idx].to_numpy(dtype=float)
                    working_test["negative_control_proxy"] = full.iloc[test_idx].to_numpy(dtype=float)
                    proxy_cols = ["negative_control_proxy"]

                without_cols, with_cols = candidate_numeric_sets(module_120, proxy_cols)
                xtr_base, xte_base = build_feature_matrix(
                    working_train,
                    working_test,
                    without_cols,
                    list(module_120.CATEGORICAL_FEATURES),
                )
                xtr_cand, xte_cand = build_feature_matrix(
                    working_train,
                    working_test,
                    with_cols,
                    list(module_120.CATEGORICAL_FEATURES),
                )
                pred_base = train_predict_et(xtr_base, z_train, xte_base, args.seed + 1210 + fold, args.n_estimators)
                pred_cand = train_predict_et(xtr_cand, z_train, xte_cand, args.seed + 2210 + fold, args.n_estimators)
                rb = rmse(z_test, pred_base)
                rc = rmse(z_test, pred_cand)

                xtr_min, xte_min = build_feature_matrix(
                    working_train,
                    working_test,
                    [],
                    list(module_120.CATEGORICAL_FEATURES),
                )
                xtr_min_c, xte_min_c = build_feature_matrix(
                    working_train,
                    working_test,
                    proxy_cols,
                    list(module_120.CATEGORICAL_FEATURES),
                )
                pred_min = train_predict_et(xtr_min, z_train, xte_min, args.seed + 3210 + fold, args.n_estimators)
                pred_min_c = train_predict_et(xtr_min_c, z_train, xte_min_c, args.seed + 4210 + fold, args.n_estimators)
                rmin = rmse(z_test, pred_min)
                rmin_c = rmse(z_test, pred_min_c)
                all_rows.append(
                    {
                        "candidate_variable": candidate["candidate_variable"],
                        "split": split,
                        "fold": int(fold),
                        "state_space": state_space,
                        "model": "extra_trees",
                        "rmse_baseline": rb,
                        "rmse_candidate": rc,
                        "strict_prediction_gain": rb - rc,
                        "strict_prediction_gain_fraction": (rb - rc) / rb if np.isfinite(rb) and abs(rb) > 1e-12 else float("nan"),
                        "minimal_baseline_gain_sensitivity": rmin - rmin_c,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "zQ_targets_rehydrated_from_120_protocol": True,
                    }
                )
    return pd.DataFrame(all_rows)


def nonredundancy_summary(
    module_120: Any,
    meta: pd.DataFrame,
    splits: list[tuple[str, int, np.ndarray, np.ndarray]],
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    null_values_by_candidate: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows = []
    null_values_by_candidate = null_values_by_candidate or {}
    for _, candidate in candidates.iterrows():
        if candidate["evidence_tier"] == "no_proxy":
            continue
        proxy_cols = parse_proxy_columns(candidate["proxy_columns"])
        working = meta.copy()
        if candidate["candidate_variable"].startswith("null_candidate_"):
            vals = null_values_by_candidate.get(candidate["candidate_variable"])
            if vals is None:
                continue
            working["null_proxy"] = vals
            proxy_cols = ["null_proxy"]
        elif candidate["candidate_variable"] == "negative_control_random_measurement":
            base = numeric_proxy(meta, "Drug Loading Capacity").to_numpy(dtype=float)
            finite = np.isfinite(base)
            shuffled = base.copy()
            if finite.any():
                shuffled[finite] = rng.permutation(shuffled[finite])
            working["negative_control_proxy"] = shuffled
            proxy_cols = ["negative_control_proxy"]

        r2_values = []
        for proxy_col in proxy_cols:
            y_oof = np.full(len(working), np.nan, dtype=float)
            for split, fold, train_idx, test_idx in splits:
                if split not in PRIMARY_SPLITS:
                    continue
                train_meta = working.iloc[train_idx].reset_index(drop=True)
                test_meta = working.iloc[test_idx].reset_index(drop=True)
                y_train = pd.to_numeric(train_meta[proxy_col], errors="coerce").to_numpy(dtype=float)
                train_valid = np.isfinite(y_train)
                if int(train_valid.sum()) < 6:
                    continue
                without_cols, _ = candidate_numeric_sets(module_120, [proxy_col])
                xtr, xte = build_feature_matrix(
                    train_meta.iloc[train_valid],
                    test_meta,
                    without_cols,
                    list(module_120.CATEGORICAL_FEATURES),
                )
                model = ExtraTreesRegressor(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=2,
                    random_state=args.seed + 5010 + fold,
                    n_jobs=-1,
                )
                model.fit(xtr, y_train[train_valid])
                y_oof[test_idx] = model.predict(xte)
            y_true = pd.to_numeric(working[proxy_col], errors="coerce").to_numpy(dtype=float)
            r2_values.append(safe_r2(y_true, y_oof))
        mean_r2 = float(np.nanmean(r2_values)) if r2_values else float("nan")
        rows.append(
            {
                "candidate_variable": candidate["candidate_variable"],
                "mean_cv_r2_existing_X_to_proxy": mean_r2,
                "nonredundancy": clip01(1.0 - max(0.0, mean_r2)) if np.isfinite(mean_r2) else float("nan"),
                "proxy_columns": "|".join(proxy_cols),
            }
        )
    return pd.DataFrame(rows)


def source_confound_audit(
    meta: pd.DataFrame,
    candidates: pd.DataFrame,
    gain: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    null_values_by_candidate: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows = []
    null_values_by_candidate = null_values_by_candidate or {}
    for _, candidate in candidates.iterrows():
        if candidate["evidence_tier"] == "no_proxy":
            rows.append(
                {
                    "candidate_variable": candidate["candidate_variable"],
                    "proxy_column": "",
                    "association_with_DOI": np.nan,
                    "association_with_Formulation_Method": np.nan,
                    "association_with_Drug": np.nan,
                    "random_split_gain": np.nan,
                    "group_by_DOI_gain": np.nan,
                    "group_by_Method_gain": np.nan,
                    "gain_collapse_random_to_DOI": np.nan,
                    "gain_collapse_random_to_Method": np.nan,
                    "source_confound_penalty": 0.0,
                }
            )
            continue
        proxy_df = candidate_proxy_frame(
            meta,
            candidate,
            rng,
            null_values=null_values_by_candidate.get(candidate["candidate_variable"]),
        )
        if proxy_df.empty:
            continue
        for proxy_col in proxy_df.columns:
            vals = proxy_df[proxy_col].to_numpy(dtype=float)
            assoc_doi = group_eta_squared(vals, meta["DOI"].to_numpy())
            assoc_method = group_eta_squared(vals, meta["Formulation Method"].to_numpy())
            assoc_drug = group_eta_squared(vals, meta["Drug"].to_numpy())
            cand_gain = gain[gain["candidate_variable"].eq(candidate["candidate_variable"])]
            random_gain = float(cand_gain[cand_gain["split"].eq(SENSITIVITY_SPLIT)]["strict_prediction_gain"].mean())
            doi_gain = float(cand_gain[cand_gain["split"].eq("group_by_DOI")]["strict_prediction_gain"].mean())
            method_gain = float(cand_gain[cand_gain["split"].eq("group_by_Formulation_Method")]["strict_prediction_gain"].mean())
            eps = 1e-12
            collapse_doi = max(0.0, random_gain - doi_gain) / max(abs(random_gain), eps) if np.isfinite(random_gain) and np.isfinite(doi_gain) else 0.0
            collapse_method = max(0.0, random_gain - method_gain) / max(abs(random_gain), eps) if np.isfinite(random_gain) and np.isfinite(method_gain) else 0.0
            assoc_values = [v for v in [assoc_doi, assoc_method, assoc_drug] if np.isfinite(v)]
            source_assoc = max(assoc_values) if assoc_values else 0.0
            if not np.isfinite(source_assoc):
                source_assoc = 0.0
            penalty = clip01(0.40 * source_assoc + 0.30 * collapse_doi + 0.30 * collapse_method)
            rows.append(
                {
                    "candidate_variable": candidate["candidate_variable"],
                    "proxy_column": proxy_col,
                    "association_with_DOI": assoc_doi,
                    "association_with_Formulation_Method": assoc_method,
                    "association_with_Drug": assoc_drug,
                    "random_split_gain": random_gain,
                    "group_by_DOI_gain": doi_gain,
                    "group_by_Method_gain": method_gain,
                    "gain_collapse_random_to_DOI": collapse_doi,
                    "gain_collapse_random_to_Method": collapse_method,
                    "source_confound_penalty": penalty,
                }
            )
    return pd.DataFrame(rows)


def residual_signature_matches(
    residuals: pd.DataFrame,
    signatures: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    sig_cols = [f"{name}_signature" for name in SYMPTOM_TARGETS]
    for _, sig in signatures.iterrows():
        vector = sig[sig_cols].to_numpy(dtype=float)
        signed_scores = []
        abs_scores = []
        for split in PRIMARY_SPLITS:
            for state_space in PRIMARY_STATE_SPACES:
                block = residuals[
                    residuals["split"].eq(split)
                    & residuals["state_space"].eq(state_space)
                    & residuals["prior_model"].eq(PRIMARY_PRIOR)
                    & (~residuals["is_oracle"].astype(bool))
                ]
                if block.empty:
                    continue
                weights = block["residual_norm"].to_numpy(dtype=float)
                weights = np.where(np.isfinite(weights), weights, 0.0)
                if float(weights.sum()) <= 1e-12:
                    weights = np.ones(len(block), dtype=float)
                profile = np.average(block[SYMPTOM_TARGETS].to_numpy(dtype=float), axis=0, weights=weights)
                signed_scores.append(cosine_01(vector, profile))
                abs_scores.append(cosine_01(np.abs(vector), np.abs(profile)))
        rows.append(
            {
                "candidate_variable": sig["candidate_variable"],
                "residual_signature_match_signed": float(np.nanmedian(signed_scores)) if signed_scores else float("nan"),
                "residual_signature_match_absolute": float(np.nanmedian(abs_scores)) if abs_scores else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def cosine_01(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if np.linalg.norm(aa) <= 1e-12 or np.linalg.norm(bb) <= 1e-12:
        return 0.5
    cos = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))
    return float((cos + 1.0) / 2.0)


def cross_source_stability(frame_support: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame_support.empty:
        return pd.DataFrame(columns=["candidate_variable", "cross_source_stability", "worst_case_support", "cross_source_relevance"])
    for cand, sub in frame_support.groupby("candidate_variable"):
        supports = pd.to_numeric(sub["frame_stat"], errors="coerce").to_numpy(dtype=float)
        signs = pd.to_numeric(sub["frame_effect_sign"], errors="coerce").to_numpy(dtype=float)
        above = sub["frame_above_null"].astype(bool).to_numpy()
        med = float(np.nanmedian(np.clip(supports, 0.0, 1.0))) if np.isfinite(supports).any() else 0.0
        worst = float(np.nanmin(np.clip(supports, 0.0, 1.0))) if np.isfinite(supports).any() else 0.0
        finite_signs = signs[np.isfinite(signs) & (signs != 0)]
        if finite_signs.size:
            sign_consistency = float(max(np.mean(finite_signs > 0), np.mean(finite_signs < 0)))
        else:
            sign_consistency = 0.0
        frac = float(np.mean(above)) if above.size else 0.0
        stability = clip01(0.50 * med + 0.30 * sign_consistency + 0.20 * frac)
        rows.append(
            {
                "candidate_variable": cand,
                "cross_source_stability": stability,
                "worst_case_support": worst,
                "cross_source_relevance": stability,
            }
        )
    return pd.DataFrame(rows)


def make_null_candidates(
    library: pd.DataFrame,
    meta: pd.DataFrame,
    n_null: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    proxy_templates = []
    for _, row in library[library["evidence_tier"].isin(["direct_proxy", "partial_proxy"])].iterrows():
        for col in parse_proxy_columns(row["proxy_columns"]):
            vals = numeric_proxy(meta, col).to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 8:
                proxy_templates.append((row, col, vals))
    null_rows = []
    values: dict[str, np.ndarray] = {}
    for i in range(n_null):
        template, col, vals = proxy_templates[i % len(proxy_templates)]
        shuffled = vals.copy()
        finite = np.isfinite(shuffled)
        if finite.any():
            shuffled[finite] = rng.permutation(shuffled[finite])
        name = f"null_candidate_{i + 1:03d}"
        values[name] = shuffled
        null_rows.append(
            {
                "candidate_variable": name,
                "candidate_family": "null_candidate",
                "measurement_method": f"permuted_template:{col}",
                "evidence_tier": "negative_control",
                "proxy_available": True,
                "proxy_fidelity": 1.0,
                "proxy_columns": "null_proxy",
                "cost_level": template["cost_level"],
                "measurement_cost_raw": template["measurement_cost_raw"],
                "measurement_cost_penalty": template["measurement_cost_penalty"],
                "destructiveness": template["destructiveness"],
                "expected_information_gain_prior": 0.0,
                "evidence_status": "null_candidate",
                "safe_interpretation": "permuted null candidate for score calibration",
            }
        )
    return pd.DataFrame(null_rows), values


def assemble_priority(
    library: pd.DataFrame,
    signatures: pd.DataFrame,
    proxy_validation_summary: pd.DataFrame,
    gain: pd.DataFrame,
    nonred: pd.DataFrame,
    confound: pd.DataFrame,
    stability: pd.DataFrame,
    signature_match: pd.DataFrame,
    null_score_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = library.merge(signatures[["candidate_variable"]], on="candidate_variable", how="left")
    df = df.merge(signature_match, on="candidate_variable", how="left")
    val_cols = [
        "candidate_variable",
        "observed_proxy_support",
        "maxT_empirical_p",
        "BH_FDR_q",
        "beats_maxT_95pct",
        "observed_max_stat",
    ]
    df = df.merge(proxy_validation_summary[val_cols], on="candidate_variable", how="left")

    primary_gain = gain[gain["split"].isin(PRIMARY_SPLITS)] if not gain.empty else pd.DataFrame()
    if not primary_gain.empty:
        gain_summary = (
            primary_gain.groupby("candidate_variable", as_index=False)
            .agg(
                strict_prediction_gain=("strict_prediction_gain", "mean"),
                strict_prediction_gain_fraction=("strict_prediction_gain_fraction", "mean"),
                strict_positive_state_spaces=("state_space", lambda s: int(primary_gain.loc[s.index].groupby("state_space")["strict_prediction_gain"].mean().gt(0).sum())),
            )
        )
    else:
        gain_summary = pd.DataFrame(columns=["candidate_variable", "strict_prediction_gain", "strict_prediction_gain_fraction", "strict_positive_state_spaces"])
    random_gain = (
        gain[gain["split"].eq(SENSITIVITY_SPLIT)]
        .groupby("candidate_variable", as_index=False)
        .agg(random_split_prediction_gain=("strict_prediction_gain", "mean"))
        if not gain.empty
        else pd.DataFrame(columns=["candidate_variable", "random_split_prediction_gain"])
    )
    min_gain = (
        gain.groupby("candidate_variable", as_index=False).agg(minimal_baseline_gain_sensitivity=("minimal_baseline_gain_sensitivity", "mean"))
        if not gain.empty
        else pd.DataFrame(columns=["candidate_variable", "minimal_baseline_gain_sensitivity"])
    )
    df = df.merge(gain_summary, on="candidate_variable", how="left")
    df = df.merge(random_gain, on="candidate_variable", how="left")
    df = df.merge(min_gain, on="candidate_variable", how="left")
    df = df.merge(nonred[["candidate_variable", "nonredundancy"]], on="candidate_variable", how="left")
    conf = confound.groupby("candidate_variable", as_index=False).agg(
        source_confound_penalty=("source_confound_penalty", "mean"),
        gain_collapse_under_strict_split=("gain_collapse_random_to_DOI", "mean"),
    )
    df = df.merge(conf, on="candidate_variable", how="left")
    df = df.merge(stability, on="candidate_variable", how="left")

    for col in ["source_confound_penalty", "gain_collapse_under_strict_split"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    for col in ["observed_proxy_support", "strict_prediction_gain", "strict_prediction_gain_fraction", "random_split_prediction_gain"]:
        if col not in df.columns:
            df[col] = np.nan

    df["strict_prediction_gain_score"] = df["strict_prediction_gain_fraction"].clip(lower=0.0, upper=1.0)
    evidence_mask = df["evidence_tier"].isin(["direct_proxy", "partial_proxy", "negative_control"])
    df["EvidenceScore_raw"] = np.nan
    df.loc[evidence_mask, "EvidenceScore_raw"] = (
        0.30 * df.loc[evidence_mask, "observed_proxy_support"].fillna(0.0)
        + 0.30 * df.loc[evidence_mask, "strict_prediction_gain_score"].fillna(0.0)
        + 0.20 * df.loc[evidence_mask, "cross_source_stability"].fillna(0.0)
        + 0.20 * df.loc[evidence_mask, "nonredundancy"].fillna(0.0)
        - 0.30 * df.loc[evidence_mask, "source_confound_penalty"].fillna(0.0)
    )
    df.loc[df["evidence_tier"].eq("no_proxy"), ["observed_proxy_support", "strict_prediction_gain", "strict_prediction_gain_fraction"]] = np.nan
    df.loc[df["evidence_tier"].eq("no_proxy"), "EvidenceScore_raw"] = np.nan

    df["evidence_component_if_available"] = np.where(
        df["evidence_tier"].isin(["direct_proxy", "partial_proxy", "negative_control"]),
        df["EvidenceScore_raw"].clip(lower=0.0, upper=1.0).fillna(0.0),
        0.0,
    )
    df["cross_source_relevance"] = df["cross_source_relevance"].fillna(df["residual_signature_match_signed"]).fillna(0.0)
    df["PriorityScore_raw"] = (
        0.30 * df["evidence_component_if_available"]
        + 0.25 * df["residual_signature_match_signed"].fillna(0.0)
        + 0.20 * df["expected_information_gain_prior"].fillna(0.0)
        + 0.15 * df["cross_source_relevance"].fillna(0.0)
        - 0.10 * df["measurement_cost_penalty"].fillna(0.0)
    )
    df["EvidenceScore_display_0_1"] = df["EvidenceScore_raw"].clip(lower=0.0, upper=1.0)
    df["PriorityScore_display_0_1"] = df["PriorityScore_raw"].clip(lower=0.0, upper=1.0)

    if null_score_df is not None and not null_score_df.empty:
        null_e = null_score_df["EvidenceScore_raw"].dropna().to_numpy(dtype=float)
        null_p = null_score_df["PriorityScore_raw"].dropna().to_numpy(dtype=float)
        df["EvidenceScore_null_percentile"] = df["EvidenceScore_raw"].apply(lambda x: percentile_against_null(x, null_e))
        df["PriorityScore_null_percentile"] = df["PriorityScore_raw"].apply(lambda x: percentile_against_null(x, null_p))
        df["EvidenceScore_empirical_p"] = df["EvidenceScore_raw"].apply(lambda x: empirical_p_against_null(x, null_e))
        df["PriorityScore_empirical_p"] = df["PriorityScore_raw"].apply(lambda x: empirical_p_against_null(x, null_p))
        e95 = float(np.nanpercentile(null_e, 95)) if null_e.size else float("nan")
        p95 = float(np.nanpercentile(null_p, 95)) if null_p.size else float("nan")
        df["beats_95pct_null"] = (
            (df["EvidenceScore_raw"].fillna(-np.inf) > e95)
            | (df["PriorityScore_raw"].fillna(-np.inf) > p95)
        )
    else:
        df["EvidenceScore_null_percentile"] = np.nan
        df["PriorityScore_null_percentile"] = np.nan
        df["EvidenceScore_empirical_p"] = np.nan
        df["PriorityScore_empirical_p"] = np.nan
        df["beats_95pct_null"] = False

    real_evidence = df["evidence_tier"].isin(["direct_proxy", "partial_proxy"])
    hypo = df["evidence_tier"].eq("no_proxy")
    df["rank_within_evidence_backed"] = np.nan
    df.loc[real_evidence, "rank_within_evidence_backed"] = (
        df.loc[real_evidence, "EvidenceScore_raw"].rank(ascending=False, method="min")
    )
    df["rank_within_hypothesis_only"] = np.nan
    df.loc[hypo, "rank_within_hypothesis_only"] = (
        df.loc[hypo, "PriorityScore_raw"].rank(ascending=False, method="min")
    )
    df["n_curves_with_proxy"] = df["proxy_columns"].apply(lambda _: np.nan)
    return df


def percentile_against_null(value: float, null: np.ndarray) -> float:
    if not np.isfinite(value) or null.size == 0:
        return float("nan")
    return float(100.0 * np.mean(null <= value))


def empirical_p_against_null(value: float, null: np.ndarray) -> float:
    if not np.isfinite(value) or null.size == 0:
        return float("nan")
    return float((1 + np.sum(null >= value)) / (len(null) + 1))


def fill_proxy_counts(priority: pd.DataFrame, meta: pd.DataFrame, null_values: dict[str, np.ndarray]) -> pd.DataFrame:
    out = priority.copy()
    counts = []
    for _, row in out.iterrows():
        if row["evidence_tier"] == "no_proxy":
            counts.append(0)
            continue
        if row["candidate_variable"].startswith("null_candidate_"):
            vals = null_values.get(row["candidate_variable"], np.array([]))
            counts.append(int(np.isfinite(vals).sum()))
            continue
        proxy_cols = parse_proxy_columns(row["proxy_columns"])
        if row["candidate_variable"] == "negative_control_random_measurement":
            proxy_cols = ["Drug Loading Capacity"]
        if not proxy_cols:
            counts.append(0)
        else:
            counts.append(int(pd.concat([numeric_proxy(meta, col) for col in proxy_cols], axis=1).notna().any(axis=1).sum()))
    out["n_curves_with_proxy"] = counts
    return out


def make_decision_table(
    priority: pd.DataFrame,
    proxy_validation_summary: pd.DataFrame,
    gain: pd.DataFrame,
    data_checks: pd.DataFrame,
    source_confound: pd.DataFrame,
    null_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    checks_pass = data_checks.set_index("check")["status"].to_dict()
    primary_oof = checks_pass.get("all_primary_residuals_are_out_of_fold") == "pass"
    real_evidence = priority[priority["evidence_tier"].isin(["direct_proxy", "partial_proxy"])]
    hypo = priority[priority["evidence_tier"].eq("no_proxy")]
    maxT_hits = proxy_validation_summary[
        proxy_validation_summary["evidence_tier"].isin(["direct_proxy", "partial_proxy"])
        & proxy_validation_summary["beats_maxT_95pct"].fillna(False)
    ]
    rows.append(
        {
            "decision": "primary_residuals_are_strict_OOF?",
            "status": "pass" if primary_oof else "fail",
            "evidence": "120 strict group-fold predictions verified against reconstructed folds",
        }
    )
    rows.append(
        {
            "decision": "existing_table_proxies_explain_residual_beyond_maxT_null?",
            "status": "pass" if not maxT_hits.empty else "warn",
            "evidence": f"evidence-backed candidates beating maxT 95pct={len(maxT_hits)}",
        }
    )
    exceeds_null = real_evidence[real_evidence["EvidenceScore_null_percentile"].fillna(0) >= 95]
    rows.append(
        {
            "decision": "any_evidence_backed_candidate_exceeds_95pct_null?",
            "status": "pass" if not exceeds_null.empty else "warn",
            "evidence": f"n={len(exceeds_null)}",
        }
    )
    primary_gain = gain[gain["split"].isin(PRIMARY_SPLITS)] if not gain.empty else pd.DataFrame()
    positive_gain = primary_gain["strict_prediction_gain"].gt(0).sum() if not primary_gain.empty else 0
    rows.append(
        {
            "decision": "prediction_gain_exists_beyond_full_reported_X_baseline?",
            "status": "pass" if positive_gain > 0 else "warn",
            "evidence": f"positive strict fold/state gain rows={int(positive_gain)}",
        }
    )
    state_positive = (
        primary_gain.groupby("state_space")["strict_prediction_gain"].mean().gt(0).sum()
        if not primary_gain.empty
        else 0
    )
    rows.append(
        {
            "decision": "strict_prediction_gain_exists_in_at_least_two_state_spaces?",
            "status": "pass" if int(state_positive) >= 2 else "warn",
            "evidence": f"state spaces with positive mean strict gain={int(state_positive)}",
        }
    )
    nonred_exists = bool(real_evidence["nonredundancy"].fillna(0).gt(0.25).any())
    rows.append(
        {
            "decision": "nonredundant_proxy_value_exists?",
            "status": "pass" if nonred_exists else "warn",
            "evidence": f"max nonredundancy={real_evidence['nonredundancy'].max():.3f}" if not real_evidence.empty else "no evidence-backed candidates",
        }
    )
    top = real_evidence.sort_values("EvidenceScore_raw", ascending=False).head(3)
    survive = bool((top["source_confound_penalty"].fillna(1.0) < 0.5).any()) if not top.empty else False
    rows.append(
        {
            "decision": "top_candidates_survive_DOI_and_method_audit?",
            "status": "pass" if survive else "warn",
            "evidence": f"top3 mean source penalty={top['source_confound_penalty'].mean():.3f}" if not top.empty else "no top evidence-backed candidates",
        }
    )
    real_best = float(real_evidence["EvidenceScore_raw"].max()) if not real_evidence.empty else float("nan")
    null_best = float(null_scores["EvidenceScore_raw"].max()) if not null_scores.empty else float("nan")
    rows.append(
        {
            "decision": "null_candidates_rank_low?",
            "status": "pass" if np.isfinite(real_best) and np.isfinite(null_best) and real_best > null_best else "warn",
            "evidence": f"best real EvidenceScore={real_best:.3f}; best null={null_best:.3f}",
        }
    )
    separated = hypo["rank_within_hypothesis_only"].notna().all() and real_evidence["rank_within_evidence_backed"].notna().all()
    rows.append(
        {
            "decision": "hypothesis_only_candidates_are_separately_ranked?",
            "status": "pass" if separated else "fail",
            "evidence": "no_proxy candidates are ranked only within hypothesis-only priority",
        }
    )

    gain_not_limited_random = True
    if not primary_gain.empty:
        random_mean = float(gain[gain["split"].eq(SENSITIVITY_SPLIT)]["strict_prediction_gain"].mean())
        strict_mean = float(primary_gain["strict_prediction_gain"].mean())
        gain_not_limited_random = bool(np.isfinite(strict_mean) and strict_mean > 0 and strict_mean >= 0.25 * max(random_mean, 0.0))
    gate_true = (
        primary_oof
        and not exceeds_null.empty
        and int(state_positive) >= 2
        and gain_not_limited_random
        and survive
        and np.isfinite(real_best)
        and np.isfinite(null_best)
        and real_best > null_best
    )
    if gate_true:
        safe = "true"
        status = "pass"
    elif primary_oof and (not hypo.empty or real_evidence.empty or exceeds_null.empty):
        safe = "exploratory_only"
        status = "warn"
    else:
        safe = "false"
        status = "fail"
    rows.append(
        {
            "decision": "safe_to_enter_122_active_planner?",
            "status": status,
            "evidence": safe,
        }
    )
    return pd.DataFrame(rows)


def make_data_checks(
    args: argparse.Namespace,
    tables: dict[str, Any],
    oof_ok: bool,
    priority: pd.DataFrame,
    proxy_validation_summary: pd.DataFrame,
    gain: pd.DataFrame,
    library: pd.DataFrame,
) -> pd.DataFrame:
    required_files = [
        "missing_state_residuals.csv",
        "curve_state_table.csv",
        "static_state_prediction.csv",
        "lock_metadata.json",
        "data_checks.csv",
    ]
    residuals = tables["residuals"]
    checks = [
        ("input_120_outputs_exist", all((args.input_120 / name).exists() for name in required_files), str(args.input_120)),
        ("sample_unit_is_curve", "Formulation Index" in residuals.columns, "121 groups by curve/formulation index"),
        ("uses_120_residuals_not_new_state_definition", True, "primary residuals are read from 120 missing_state_residuals.csv"),
        ("all_primary_residuals_are_out_of_fold", oof_ok, "strict group-fold membership reconstructed from 120 lock metadata"),
        ("residual_fold_matches_120_lock_metadata", oof_ok, "fold ids and test curves match regenerated 120 folds"),
        ("no_curve_used_to_predict_its_own_state", oof_ok, "each strict residual row is in fold test set and not train set"),
        ("no_timepoint_split", True, "all splits are curve-level folds inherited from 120 protocol"),
        (
            "candidate_library_has_cost_signature_and_evidence_tier",
            {"evidence_tier", "measurement_cost_penalty"}.issubset(library.columns),
            f"library rows={len(library)}",
        ),
        (
            "proxy_validation_has_maxT_shuffle_null",
            {"maxT_empirical_p", "maxT_95pct", "beats_maxT_95pct"}.issubset(proxy_validation_summary.columns),
            f"shuffle_repeats={args.shuffle_repeats}",
        ),
        (
            "prediction_gain_uses_curve_level_cv",
            "n_test" in gain.columns and gain["n_test"].min() >= 1 if not gain.empty else False,
            "gain rows are fold-level curve predictions",
        ),
        (
            "prediction_gain_uses_full_reported_X_without_candidate_baseline",
            "rmse_baseline" in gain.columns,
            "baseline excludes only candidate proxy columns from full reported X",
        ),
        (
            "strict_group_splits_reported",
            set(PRIMARY_SPLITS).issubset(set(gain["split"].unique())) if not gain.empty else False,
            ",".join(sorted(gain["split"].unique())) if not gain.empty else "no gain rows",
        ),
        (
            "multiple_null_candidates_present",
            args.n_null_candidates >= 2,
            f"n_null_candidates={args.n_null_candidates}",
        ),
        (
            "evidence_and_hypothesis_rankings_separated",
            priority.loc[priority["evidence_tier"].eq("no_proxy"), "rank_within_evidence_backed"].isna().all()
            and priority.loc[priority["evidence_tier"].isin(["direct_proxy", "partial_proxy"]), "rank_within_hypothesis_only"].isna().all(),
            "rank_within_evidence_backed and rank_within_hypothesis_only are separate",
        ),
        (
            "negative_control_present",
            "negative_control_random_measurement" in set(priority["candidate_variable"]),
            "fixed negative control plus null ensemble",
        ),
        (
            "no_formulation_remeasurement_priority_output",
            not (args.out / "formulation_remeasurement_priority.csv").exists(),
            "121 ranks variables, not formulations",
        ),
        (
            "no_nan_in_required_evidence_components",
            required_evidence_components_finite(priority),
            "NaN EvidenceScore is allowed only for no_proxy candidates",
        ),
        (
            "hypothesis_only_candidates_have_no_fake_proxy_evidence",
            hypothesis_only_guardrail(priority),
            "no_proxy candidates have NaN empirical evidence columns",
        ),
        (
            "interpretation_boundaries_present",
            True,
            "report and docs state candidate measurements only, no physical hidden-variable discovery",
        ),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "pass" if passed else "fail", "detail": detail} for name, passed, detail in checks]
    )


def required_evidence_components_finite(priority: pd.DataFrame) -> bool:
    mask = priority["evidence_tier"].isin(["direct_proxy", "partial_proxy"])
    cols = [
        "observed_proxy_support",
        "strict_prediction_gain",
        "strict_prediction_gain_fraction",
        "nonredundancy",
        "source_confound_penalty",
        "EvidenceScore_raw",
    ]
    return bool(np.isfinite(priority.loc[mask, cols].to_numpy(dtype=float)).all())


def hypothesis_only_guardrail(priority: pd.DataFrame) -> bool:
    hypo = priority[priority["evidence_tier"].eq("no_proxy")]
    if hypo.empty:
        return False
    cols = ["observed_proxy_support", "strict_prediction_gain", "strict_prediction_gain_fraction", "EvidenceScore_raw"]
    return bool(hypo[cols].isna().all().all() and hypo["evidence_status"].eq("hypothesis_only").all())


def write_report(
    out: Path,
    priority: pd.DataFrame,
    decisions: pd.DataFrame,
    data_checks: pd.DataFrame,
    null_scores: pd.DataFrame,
) -> None:
    evidence = priority[priority["evidence_tier"].isin(["direct_proxy", "partial_proxy"])].sort_values("EvidenceScore_raw", ascending=False)
    hypo = priority[priority["evidence_tier"].eq("no_proxy")].sort_values("PriorityScore_raw", ascending=False)
    lines = [
        "# PLGA Candidate Measurement Value Scoring",
        "",
        "121 ranks candidate measurements. 121 does not discover physical hidden variables.",
        "",
        "EvidenceScore represents support from currently available proxies. PriorityScore represents future measurement priority.",
        "A high hypothesis-only PriorityScore is not empirical evidence; it is a testable experimental hypothesis.",
        "DOI/method structure may represent source, protocol, assay, batch, or process confounding.",
        "No candidate is causal without direct experimental validation.",
        "",
        "## Top Evidence-Backed Candidates",
        "",
        evidence[
            [
                "candidate_variable",
                "evidence_tier",
                "EvidenceScore_raw",
                "observed_proxy_support",
                "strict_prediction_gain",
                "source_confound_penalty",
                "EvidenceScore_null_percentile",
            ]
        ]
        .head(8)
        .round(4)
        .to_markdown(index=False),
        "",
        "## Top Hypothesis-Only Candidates",
        "",
        hypo[
            [
                "candidate_variable",
                "PriorityScore_raw",
                "residual_signature_match_signed",
                "expected_information_gain_prior",
                "measurement_cost_penalty",
                "safe_interpretation",
            ]
        ]
        .head(8)
        .round(4)
        .to_markdown(index=False),
        "",
        "## Null Candidate Score Range",
        "",
        null_scores[["EvidenceScore_raw", "PriorityScore_raw"]].describe().round(4).to_markdown()
        if not null_scores.empty
        else "No null scores generated.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Data Checks",
        "",
        data_checks.to_markdown(index=False),
        "",
        "## Interpretation Boundary",
        "",
        "- EvidenceScore answers which currently available proxies have empirical support for explaining missing-state residuals.",
        "- PriorityScore answers which unmeasured variables are worth testing as future experimental hypotheses.",
        "- No formulation-level remeasurement plan is generated here.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(doc: Path) -> None:
    lines = [
        "# PLGA Candidate Measurement Value Scoring",
        "",
        "Date: 2026-06-13",
        "",
        "## Purpose",
        "",
        "Experiment 121 consumes the strict out-of-fold residual map from Experiment 120 and ranks candidate measurement variables.",
        "",
        "It does not redefine the residual, train a new release predictor as the headline object, or output a formulation-level remeasurement plan.",
        "",
        "## Core Distinction",
        "",
        "- EvidenceScore: support from currently available table proxies.",
        "- PriorityScore: future measurement priority.",
        "",
        "No-proxy candidates can rank only as hypotheses. They receive no observed proxy support, no proxy prediction gain, and no EvidenceScore.",
        "",
        "## Guardrails",
        "",
        "- Primary residuals must be strict out-of-fold rows from Experiment 120.",
        "- Proxy validation uses a max-statistic permutation null across proxy columns, residual targets, state spaces, and strict group splits.",
        "- Prediction gain is conditional on all available reported X except the candidate proxy.",
        "- Nonredundancy is computed as how poorly existing X predicts the candidate proxy, not as a duplicate of prediction gain.",
        "- DOI is only a source/confound audit label and is never a candidate measurement.",
        "- Multiple null candidates calibrate score competitiveness.",
        "",
        "## Interpretation Boundary",
        "",
        "121 ranks candidate measurements. 121 does not discover physical hidden variables.",
        "",
        "A high hypothesis-only PriorityScore is not empirical evidence. It is a testable experimental hypothesis.",
        "",
        "DOI/method structure may represent source, protocol, assay, batch, or process confounding.",
        "",
        "No candidate may be described as causal without direct experimental validation.",
        "",
        "## Output Anchor",
        "",
        "`../outputs/121_plga_candidate_measurement_value_scoring/`",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lock(args: argparse.Namespace, out: Path, tables: dict[str, Any], priority: pd.DataFrame) -> None:
    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "input_120": file_meta(args.input_120),
        "real_xlsx": file_meta(args.real_xlsx),
        "source_120_git_hash": tables["lock"].get("git_hash", "unknown"),
        "primary_residual_definition": "r_i = z_Q_i - z_X_i_OOF from 120 missing_state_residuals.csv",
        "primary_frames": [
            f"{split} x {state_space} x {PRIMARY_PRIOR}"
            for split in PRIMARY_SPLITS
            for state_space in PRIMARY_STATE_SPACES
        ],
        "n_real_candidates": int((~priority["candidate_variable"].str.startswith("null_candidate_")).sum()),
        "n_null_candidates": int(priority["candidate_variable"].str.startswith("null_candidate_").sum()),
        "interpretation_boundary": "candidate measurement scoring only; no physical hidden-variable discovery",
        "generated_files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed + 121)
    args.out.mkdir(parents=True, exist_ok=True)

    tables = load_120_tables(args.input_120)
    oof_ok, oof_detail, fold_table = verify_120_oof(args, tables)
    module_120, a120, meta, grid, y_grid, splits = prepare_meta_for_scoring(args, tables["lock"], tables["residuals"])

    residuals = tables["residuals"].copy()
    if args.max_curves:
        keep = set(meta["Formulation Index"].astype(int))
        residuals = residuals[residuals["Formulation Index"].astype(int).isin(keep)].reset_index(drop=True)

    library = make_candidate_library()
    signatures = make_signature_matrix(library)
    null_library, null_values = make_null_candidates(library, meta, args.n_null_candidates, rng)
    scoring_library = pd.concat([library, null_library], ignore_index=True)

    null_signatures = []
    for _, row in null_library.iterrows():
        vec = rng.normal(0, 0.3, size=len(SYMPTOM_TARGETS))
        null_signatures.append(
            {
                "candidate_variable": row["candidate_variable"],
                "signature_basis": "curve_residual_symptom_delta_Q",
                "burst_residual_signature": vec[0],
                "early_slope_residual_signature": vec[1],
                "middle_diffusion_residual_signature": vec[2],
                "late_tail_residual_signature": vec[3],
                "qmax_residual_signature": vec[4],
            }
        )
    scoring_signatures = pd.concat([signatures, pd.DataFrame(null_signatures)], ignore_index=True)

    proxy_summary, frame_support = proxy_validation(residuals, meta, scoring_library, args, rng, null_values)
    gain = prediction_gain_summary(module_120, meta, y_grid, grid, splits, scoring_library, args, rng, null_values)
    nonred = nonredundancy_summary(module_120, meta, splits, scoring_library, args, rng, null_values)
    confound = source_confound_audit(meta, scoring_library, gain, args, rng, null_values)
    stability = cross_source_stability(frame_support)
    signature_match = residual_signature_matches(residuals, scoring_signatures)

    null_priority_initial = assemble_priority(
        null_library,
        scoring_signatures,
        proxy_summary,
        gain,
        nonred,
        confound,
        stability,
        signature_match,
        null_score_df=None,
    )
    null_scores = null_priority_initial.copy()
    priority_all = assemble_priority(
        library,
        signatures,
        proxy_summary,
        gain,
        nonred,
        confound,
        stability,
        signature_match,
        null_score_df=null_scores,
    )
    priority_all = fill_proxy_counts(priority_all, meta, null_values)
    null_scores = fill_proxy_counts(null_scores, meta, null_values)

    data_checks = make_data_checks(args, tables, oof_ok, priority_all, proxy_summary, gain, library)
    decisions = make_decision_table(priority_all, proxy_summary, gain, data_checks, confound, null_scores)

    library.to_csv(args.out / "candidate_measurement_library.csv", index=False)
    signatures.to_csv(args.out / "candidate_signature_matrix.csv", index=False)
    proxy_summary[~proxy_summary["candidate_variable"].str.startswith("null_candidate_", na=False)].to_csv(
        args.out / "proxy_validation_summary.csv", index=False
    )
    gain[~gain["candidate_variable"].str.startswith("null_candidate_", na=False)].to_csv(
        args.out / "proxy_prediction_gain_summary.csv", index=False
    )
    confound[~confound["candidate_variable"].str.startswith("null_candidate_", na=False)].to_csv(
        args.out / "source_confound_audit.csv", index=False
    )
    final_cols = [
        "candidate_variable",
        "candidate_family",
        "measurement_method",
        "evidence_tier",
        "proxy_available",
        "proxy_fidelity",
        "proxy_columns",
        "n_curves_with_proxy",
        "residual_signature_match_signed",
        "residual_signature_match_absolute",
        "observed_proxy_support",
        "strict_prediction_gain",
        "strict_prediction_gain_fraction",
        "minimal_baseline_gain_sensitivity",
        "random_split_prediction_gain",
        "gain_collapse_under_strict_split",
        "cross_source_stability",
        "worst_case_support",
        "nonredundancy",
        "source_confound_penalty",
        "measurement_cost_penalty",
        "EvidenceScore_raw",
        "PriorityScore_raw",
        "EvidenceScore_null_percentile",
        "PriorityScore_null_percentile",
        "EvidenceScore_empirical_p",
        "PriorityScore_empirical_p",
        "maxT_empirical_p",
        "BH_FDR_q",
        "rank_within_evidence_backed",
        "rank_within_hypothesis_only",
        "evidence_status",
        "safe_interpretation",
        "EvidenceScore_display_0_1",
        "PriorityScore_display_0_1",
        "beats_95pct_null",
    ]
    priority_all[final_cols].to_csv(args.out / "candidate_variable_priority.csv", index=False)
    null_scores.to_csv(args.out / "null_candidate_score_distribution.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    data_checks.to_csv(args.out / "data_checks.csv", index=False)
    oof_detail.to_csv(args.out / "oof_verification_detail.csv", index=False)
    fold_table.to_csv(args.out / "reconstructed_120_folds.csv", index=False)
    write_report(args.out, priority_all, decisions, data_checks, null_scores)
    write_doc(args.doc)
    write_lock(args, args.out, tables, pd.concat([priority_all, null_scores], ignore_index=True))

    failed = data_checks[data_checks["status"].eq("fail")]
    if not failed.empty:
        raise RuntimeError(f"121 failed data checks: {failed['check'].tolist()}")
    print(f"Wrote PLGA candidate measurement value scoring to {args.out}")


if __name__ == "__main__":
    main()
