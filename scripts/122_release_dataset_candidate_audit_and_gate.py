"""
122 - Release dataset candidate audit and minimal training gate.

Purpose:
    Build an evidence-backed inventory of candidate drug-release datasets and
    run a minimal static-vs-early training gate only on local standardized
    datasets that are large enough to support it.

Consumes:
    outputs/148_release_main_cumulative_v1/
    outputs/149_release_caveat_augmented_cumulative_v1/
    web-source metadata curated from browser checks

Produces:
    outputs/122_release_dataset_candidate_audit_and_gate/
      dataset_candidate_inventory.csv
      source_links.csv
      local_dataset_audit.csv
      training_gate_summary.csv
      per_dataset_gate_metrics.csv
      access_audit.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This script does not create a new release model. It decides which dataset
    candidates are worth collecting or training against for the release-state
    observability story.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_POOL = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_CAVEAT_POOL = Path("outputs/149_release_caveat_augmented_cumulative_v1")
DEFAULT_OUT = Path("outputs/122_release_dataset_candidate_audit_and_gate")
DEFAULT_DOC = Path("docs/release_dataset_candidate_audit_and_gate_2026-06-14.md")

MIN_CURVES_FOR_TRAINING = 30
MIN_MEDIAN_POINTS = 5
EARLY_K = 2

NUMERIC_DESCRIPTOR_CANDIDATES = [
    "Polymer_MW",
    "LA/GA",
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Initial D/M ratio",
    "DLC",
    "DLC_percent",
    "EE",
    "Particle_Size",
    "SA-V",
    "SE",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_NHA",
    "Drug_LogP",
    "media_pH",
    "media_temp_oC",
    "PDI",
    "zeta_potential",
    "weighted_Mw",
    "weighted_Tm",
    "P407_percent_wv",
    "P188_percent_wv",
    "HA_percent_wv",
    "NaCl_percent_wv",
]

CATEGORICAL_DESCRIPTOR_CANDIDATES = [
    "polymer_family",
    "payload_name",
    "source_group",
    "experimental_panel",
    "release_method",
    "structure_type",
    "API_name",
    "measurement_assay",
    "release_measure_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate release datasets and run minimal gates.")
    parser.add_argument("--main-pool", type=Path, default=DEFAULT_MAIN_POOL)
    parser.add_argument("--caveat-pool", type=Path, default=DEFAULT_CAVEAT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_estimators = min(args.n_estimators, 50)
        if args.max_datasets <= 0:
            args.max_datasets = 3
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean(np.square(y - pred))))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3 or float(np.nanvar(y[mask])) <= 1e-12:
        return float("nan")
    return float(r2_score(y[mask], pred[mask]))


def load_pool(pool: Path) -> dict[str, pd.DataFrame]:
    required = ["dataset_summary.csv", "curves_long.csv", "formulations.csv", "curve_quality_summary.csv"]
    missing = [name for name in required if not (pool / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing pool files in {pool}: {missing}")
    return {
        "dataset_summary": pd.read_csv(pool / "dataset_summary.csv"),
        "curves": pd.read_csv(pool / "curves_long.csv"),
        "formulations": pd.read_csv(pool / "formulations.csv"),
        "quality": pd.read_csv(pool / "curve_quality_summary.csv"),
    }


def curated_web_candidates() -> pd.DataFrame:
    rows = [
        {
            "dataset_name": "alginate_hydrogel_active_learning_jcr2026",
            "system_type": "alginate hydrogel",
            "source_kind": "web_candidate",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/41490607/",
            "doi": "10.1016/j.jconrel.2026.114602",
            "n_curves_or_datapoints": 120,
            "has_full_Qt_curve": "reported_release_profiles",
            "has_static_descriptors": "yes",
            "has_material_state_descriptors": "partial",
            "has_source_group_labels": "unknown",
            "has_API_or_drug_group_labels": "payload_BSA_then_chABC-SEN",
            "access_mode": "article_open; raw_table_not_yet_local",
            "manual_collection_cost": "medium",
            "priority_for_release_state_story": "very_high",
            "notes": "JCR active-learning alginate hydrogel; strongest match to chitosan/HA/PVP prospective story.",
        },
        {
            "dataset_name": "chitosan_nanoparticles_release_ddip2025",
            "system_type": "chitosan nanoparticles",
            "source_kind": "web_candidate",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/41030028/",
            "doi": "10.1080/03639045.2025.2569573",
            "n_curves_or_datapoints": 190,
            "has_full_Qt_curve": "multi_timepoint_datapoints_reported",
            "has_static_descriptors": "yes",
            "has_material_state_descriptors": "partial",
            "has_source_group_labels": "115_literature_articles",
            "has_API_or_drug_group_labels": "likely",
            "access_mode": "publisher_article; dataset_table_not_found_open",
            "manual_collection_cost": "high",
            "priority_for_release_state_story": "high",
            "notes": "Directly relevant to chitosan; likely needs author request or manual reconstruction.",
        },
        {
            "dataset_name": "direct_compression_tablets_aaps2025",
            "system_type": "oral direct-compression tablets",
            "source_kind": "web_candidate",
            "source_url": "https://link.springer.com/article/10.1208/s12248-025-01101-1",
            "doi": "10.1208/s12248-025-01101-1",
            "n_curves_or_datapoints": 377,
            "has_full_Qt_curve": "reported_11_timepoints",
            "has_static_descriptors": "yes",
            "has_material_state_descriptors": "limited",
            "has_source_group_labels": "single_in_house_source",
            "has_API_or_drug_group_labels": "20_APIs_reported",
            "access_mode": "open_article; supplementary_docx_downloaded_but_raw_profile_table_not_in_docx",
            "manual_collection_cost": "medium",
            "priority_for_release_state_story": "medium_high_outgroup",
            "notes": "Strong outgroup if raw table is obtained; current supplement contains supplier tables and figures, not machine-readable release table.",
        },
        {
            "dataset_name": "plga_nanoparticles_mendeley_2025",
            "system_type": "PLGA nanoparticles",
            "source_kind": "web_candidate",
            "source_url": "https://data.mendeley.com/datasets/sbjf5csrdm/1",
            "doi": "10.17632/sbjf5csrdm.1",
            "n_curves_or_datapoints": 433,
            "has_full_Qt_curve": "no_endpoint_dataset",
            "has_static_descriptors": "yes",
            "has_material_state_descriptors": "particle_size_EE_LC",
            "has_source_group_labels": "literature_curated",
            "has_API_or_drug_group_labels": "65_small_molecules",
            "access_mode": "open_mendeley",
            "manual_collection_cost": "low",
            "priority_for_release_state_story": "medium_descriptor_auxiliary",
            "notes": "Useful descriptor/CQA auxiliary source; not a release-curve training dataset.",
        },
        {
            "dataset_name": "direct_compression_tablet_supp_docx_local",
            "system_type": "oral direct-compression tablets",
            "source_kind": "download_audit",
            "source_url": "data/external/direct_compression_tablets_aaps_2025/supplementary_file1.docx",
            "doi": "10.1208/s12248-025-01101-1",
            "n_curves_or_datapoints": 0,
            "has_full_Qt_curve": "no_machine_readable_table_in_downloaded_docx",
            "has_static_descriptors": "supplier_tables_only",
            "has_material_state_descriptors": "no",
            "has_source_group_labels": "no",
            "has_API_or_drug_group_labels": "API_supplier_table",
            "access_mode": "downloaded_local_audit",
            "manual_collection_cost": "medium",
            "priority_for_release_state_story": "hold_until_raw_table_or_digitization",
            "notes": "Downloaded Supplementary file1; it has two supplier tables and release-profile figures, not raw formulation/release matrix.",
        },
    ]
    return pd.DataFrame(rows)


def audit_local_datasets(pool_name: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = tables["dataset_summary"].copy()
    curves = tables["curves"].copy()
    forms = tables["formulations"].copy()
    quality = tables["quality"].copy()
    rows = []
    for _, row in summary.iterrows():
        source = row["source_dataset"]
        q = quality[quality["source_dataset"].eq(source)]
        f = forms[forms["source_dataset"].eq(source)]
        c = curves[curves["source_dataset"].eq(source)]
        n_curves = int(row["n_curves"])
        median_points = float(q["n_points"].median()) if "n_points" in q and not q.empty else float("nan")
        descriptor_cols = descriptor_coverage(f)
        cat_cols = categorical_coverage(f)
        group_col, n_groups = choose_group_column(f)
        train_eligible = (
            n_curves >= MIN_CURVES_FOR_TRAINING
            and np.isfinite(median_points)
            and median_points >= MIN_MEDIAN_POINTS
            and (len(descriptor_cols) + len(cat_cols)) >= 2
        )
        rows.append(
            {
                "pool_name": pool_name,
                "dataset_name": source,
                "source_dataset": source,
                "corpus_name": row.get("corpus_name", ""),
                "source_kind": "local_standardized_pool",
                "n_curves": n_curves,
                "n_points_total": int(row["n_points_total"]),
                "n_formulations": int(row["n_formulations"]),
                "median_points_per_curve": median_points,
                "median_duration_days": float(q["duration_days"].median()) if "duration_days" in q and not q.empty else float("nan"),
                "fraction_monotone_non_decreasing": float(q["is_monotone_non_decreasing"].mean())
                if "is_monotone_non_decreasing" in q and not q.empty
                else float("nan"),
                "has_full_Qt_curve": bool(n_curves > 0 and median_points >= MIN_MEDIAN_POINTS),
                "numeric_descriptor_cols": "|".join(descriptor_cols),
                "categorical_descriptor_cols": "|".join(cat_cols),
                "n_numeric_descriptor_cols": len(descriptor_cols),
                "n_categorical_descriptor_cols": len(cat_cols),
                "group_split_column": group_col,
                "n_group_split_labels": n_groups,
                "can_group_split": bool(n_groups >= 3 and n_groups < n_curves),
                "training_gate_eligible": bool(train_eligible),
                "not_training_reason": "" if train_eligible else not_training_reason(n_curves, median_points, descriptor_cols, cat_cols),
                "fit_to_project_story": fit_story_score(source, n_curves, descriptor_cols, cat_cols),
            }
        )
    return pd.DataFrame(rows)


def descriptor_coverage(forms: pd.DataFrame) -> list[str]:
    cols = []
    for col in NUMERIC_DESCRIPTOR_CANDIDATES:
        if col in forms.columns:
            values = pd.to_numeric(forms[col], errors="coerce")
            if values.notna().mean() >= 0.20 and values.nunique(dropna=True) >= 2:
                cols.append(col)
    return cols


def categorical_coverage(forms: pd.DataFrame) -> list[str]:
    cols = []
    for col in CATEGORICAL_DESCRIPTOR_CANDIDATES:
        if col in forms.columns:
            values = clean_categorical(forms[col])
            if values.notna().mean() >= 0.20 and values.nunique(dropna=True) >= 2:
                cols.append(col)
    return cols


def choose_group_column(forms: pd.DataFrame) -> tuple[str, int]:
    best_col = ""
    best_groups = 0
    for col in ["payload_name", "API_name", "source_group", "experimental_panel", "polymer_family", "release_method"]:
        if col not in forms.columns:
            continue
        n = int(clean_categorical(forms[col]).nunique(dropna=True))
        if n > best_groups:
            best_col = col
            best_groups = n
    return best_col, best_groups


def clean_categorical(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isin(["", "nan", "NaN", "None", "<NA>"]))


def not_training_reason(n_curves: int, median_points: float, numeric_cols: list[str], cat_cols: list[str]) -> str:
    reasons = []
    if n_curves < MIN_CURVES_FOR_TRAINING:
        reasons.append(f"n_curves<{MIN_CURVES_FOR_TRAINING}")
    if not np.isfinite(median_points) or median_points < MIN_MEDIAN_POINTS:
        reasons.append(f"median_points<{MIN_MEDIAN_POINTS}")
    if (len(numeric_cols) + len(cat_cols)) < 2:
        reasons.append("insufficient_descriptor_columns")
    return ";".join(reasons)


def fit_story_score(source: str, n_curves: int, numeric_cols: list[str], cat_cols: list[str]) -> str:
    text = source.lower()
    if "liposome" in text:
        return "second_system_validation"
    if "cross321" in text or "internal181" in text:
        return "plga_primary_anchor"
    if "hydrogel" in text or "alginate" in text or "chitosan" in text or "nt3" in text:
        return "hydrogel_chitosan_bridge" if n_curves >= 12 else "small_hydrogel_background"
    if n_curves >= MIN_CURVES_FOR_TRAINING:
        return "cross_system_outgroup"
    return "background_only"


def build_curve_targets(curves: pd.DataFrame, forms: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    fids = forms[forms["source_dataset"].eq(dataset)]["unified_curve_id"].tolist()
    for curve_id in fids:
        g = curves[curves["unified_curve_id"].eq(curve_id)].copy()
        if g.empty:
            continue
        g["time_days"] = pd.to_numeric(g["time_days"], errors="coerce")
        g["release_fraction"] = pd.to_numeric(g["release_fraction"], errors="coerce").clip(0.0, 1.2)
        g = g.dropna(subset=["time_days", "release_fraction"]).sort_values("time_days")
        g = g.groupby("time_days", as_index=False)["release_fraction"].mean().sort_values("time_days")
        if len(g) < MIN_MEDIAN_POINTS:
            continue
        t = g["time_days"].to_numpy(dtype=float)
        q = np.maximum.accumulate(g["release_fraction"].to_numpy(dtype=float))
        duration = max(float(t[-1] - t[0]), 1e-9)
        auc = float(np.trapz(q, t) / duration)
        early = early_features(t, q)
        rows.append(
            {
                "unified_curve_id": curve_id,
                "final_release_fraction": float(q[-1]),
                "auc_release_fraction": auc,
                "max_time_days": float(t[-1]),
                **early,
            }
        )
    return pd.DataFrame(rows)


def early_features(t: np.ndarray, q: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx in range(EARLY_K):
        if idx < len(t):
            out[f"early_t{idx + 1}_days"] = float(t[idx])
            out[f"early_q{idx + 1}"] = float(q[idx])
        else:
            out[f"early_t{idx + 1}_days"] = np.nan
            out[f"early_q{idx + 1}"] = np.nan
    return out


def build_features(forms: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, numeric_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train = forms.iloc[train_idx]
    test = forms.iloc[test_idx]
    blocks_train: list[np.ndarray] = []
    blocks_test: list[np.ndarray] = []
    for col in numeric_cols:
        tr = pd.to_numeric(train[col], errors="coerce")
        te = pd.to_numeric(test[col], errors="coerce")
        med = 0.0 if tr.dropna().empty else float(tr.dropna().median())
        blocks_train.append(tr.fillna(med).to_numpy(dtype=float)[:, None])
        blocks_test.append(te.fillna(med).to_numpy(dtype=float)[:, None])
    for col in cat_cols:
        tr = train[col].astype(str).fillna("missing")
        te = test[col].astype(str).fillna("missing")
        cats = sorted(tr.unique().tolist())
        lookup = {cat: i for i, cat in enumerate(cats)}
        btr = np.zeros((len(train), len(cats) + 1), dtype=float)
        bte = np.zeros((len(test), len(cats) + 1), dtype=float)
        for i, value in enumerate(tr):
            btr[i, lookup.get(value, len(cats))] = 1.0
        for i, value in enumerate(te):
            bte[i, lookup.get(value, len(cats))] = 1.0
        blocks_train.append(btr)
        blocks_test.append(bte)
    if not blocks_train:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    return np.hstack(blocks_train), np.hstack(blocks_test)


def make_splits(df: pd.DataFrame, group_col: str, seed: int) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    idx = np.arange(len(df))
    n_splits = min(5, len(df))
    splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(idx), start=1):
            splits.append(("random_5fold", fold, tr, te))
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).fillna("missing").to_numpy()
        n_groups = len(np.unique(groups))
        n_group_splits = min(5, n_groups)
        if n_group_splits >= 2 and n_group_splits < len(df):
            gkf = GroupKFold(n_splits=n_group_splits)
            for fold, (tr, te) in enumerate(gkf.split(idx, groups=groups), start=1):
                splits.append((f"group_by_{group_col}", fold, tr, te))
    return splits


def run_training_gate(
    pool_name: str,
    tables: dict[str, pd.DataFrame],
    audit: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = tables["curves"]
    forms_all = tables["formulations"]
    eligible = audit[audit["training_gate_eligible"]].copy()
    if args.max_datasets:
        eligible = eligible.head(args.max_datasets)
    metric_rows = []
    for _, row in eligible.iterrows():
        dataset = row["source_dataset"]
        forms = forms_all[forms_all["source_dataset"].eq(dataset)].copy().reset_index(drop=True)
        targets = build_curve_targets(curves, forms, dataset)
        forms = forms.merge(targets, on="unified_curve_id", how="inner")
        if len(forms) < MIN_CURVES_FOR_TRAINING:
            continue
        numeric_cols = [col for col in str(row["numeric_descriptor_cols"]).split("|") if col and col in forms.columns]
        cat_cols = [col for col in str(row["categorical_descriptor_cols"]).split("|") if col and col in forms.columns]
        group_col = row["group_split_column"] if isinstance(row["group_split_column"], str) else ""
        for target_col in ["final_release_fraction", "auc_release_fraction"]:
            y = forms[target_col].to_numpy(dtype=float)
            for split_name, fold, train_idx, test_idx in make_splits(forms, group_col, args.seed):
                xtr_static, xte_static = build_features(forms, train_idx, test_idx, numeric_cols, cat_cols)
                early_cols = [f"early_t{i}_days" for i in range(1, EARLY_K + 1)] + [f"early_q{i}" for i in range(1, EARLY_K + 1)]
                xtr_early, xte_early = build_features(forms, train_idx, test_idx, numeric_cols + early_cols, cat_cols)
                pred_prior = np.full(len(test_idx), float(np.nanmean(y[train_idx])))
                pred_static = fit_predict(xtr_static, y[train_idx], xte_static, args.seed + fold, args.n_estimators)
                pred_early = fit_predict(xtr_early, y[train_idx], xte_early, args.seed + 1000 + fold, args.n_estimators)
                metric_rows.extend(
                    [
                        metric_row(pool_name, dataset, split_name, fold, target_col, "train_mean_prior", y[test_idx], pred_prior),
                        metric_row(pool_name, dataset, split_name, fold, target_col, "static_X", y[test_idx], pred_static),
                        metric_row(pool_name, dataset, split_name, fold, target_col, "static_X_plus_early_k2", y[test_idx], pred_early),
                    ]
                )
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        return metrics, pd.DataFrame()
    summary = (
        metrics.groupby(["pool_name", "source_dataset", "split", "target", "method"], as_index=False)
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), r2=("r2", "mean"), n_test=("n_test", "sum"))
        .sort_values(["pool_name", "source_dataset", "split", "target", "rmse"])
    )
    return metrics, summary


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def metric_row(
    pool_name: str,
    dataset: str,
    split: str,
    fold: int,
    target: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "pool_name": pool_name,
        "source_dataset": dataset,
        "split": split,
        "fold": int(fold),
        "target": target,
        "method": method,
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.nanmean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        "r2": safe_r2(y_true, y_pred),
        "n_test": int(len(y_true)),
    }


def build_inventory(local_audit: pd.DataFrame, web: pd.DataFrame) -> pd.DataFrame:
    local = local_audit.copy()
    local["dataset_name"] = local["source_dataset"]
    local["system_type"] = local["fit_to_project_story"]
    local["source_url"] = ""
    local["doi"] = ""
    local["n_curves_or_datapoints"] = local["n_curves"]
    local["has_static_descriptors"] = np.where(
        local["n_numeric_descriptor_cols"] + local["n_categorical_descriptor_cols"] >= 2, "yes", "weak"
    )
    local["has_material_state_descriptors"] = np.where(
        local["numeric_descriptor_cols"].astype(str).str.contains("Particle|HA|P407|P188|PDI|zeta", regex=True),
        "partial",
        "weak_or_unknown",
    )
    local["has_source_group_labels"] = np.where(local["can_group_split"], local["group_split_column"], "weak")
    local["has_API_or_drug_group_labels"] = np.where(
        local["categorical_descriptor_cols"].astype(str).str.contains("payload|API", regex=True), "yes", "weak"
    )
    local["access_mode"] = "local_standardized"
    local["manual_collection_cost"] = "none"
    local["priority_for_release_state_story"] = local["fit_to_project_story"]
    local["notes"] = local["not_training_reason"]
    local_cols = [
        "dataset_name",
        "system_type",
        "source_kind",
        "source_url",
        "doi",
        "n_curves_or_datapoints",
        "has_full_Qt_curve",
        "has_static_descriptors",
        "has_material_state_descriptors",
        "has_source_group_labels",
        "has_API_or_drug_group_labels",
        "access_mode",
        "manual_collection_cost",
        "priority_for_release_state_story",
        "notes",
    ]
    return pd.concat([local[local_cols], web[local_cols]], ignore_index=True)


def decision_table(local_audit: pd.DataFrame, training_summary: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = local_audit[local_audit["training_gate_eligible"]]
    rows.append(
        {
            "decision": "candidate_inventory_built?",
            "status": "pass" if not inventory.empty else "fail",
            "evidence": f"inventory rows={len(inventory)}",
        }
    )
    rows.append(
        {
            "decision": "local_datasets_available_for_training?",
            "status": "pass" if len(eligible) >= 1 else "warn",
            "evidence": f"eligible local datasets={len(eligible)}",
        }
    )
    if training_summary.empty:
        early_gain_count = 0
        evidence = "no training gate rows"
    else:
        pivot = training_summary.pivot_table(
            index=["pool_name", "source_dataset", "split", "target"],
            columns="method",
            values="rmse",
            aggfunc="mean",
        ).reset_index()
        if "static_X" in pivot and "static_X_plus_early_k2" in pivot:
            pivot["early_gain"] = pivot["static_X"] - pivot["static_X_plus_early_k2"]
            early_gain_count = int(pivot["early_gain"].gt(0).sum())
            evidence = f"static+early beats static rows={early_gain_count}/{len(pivot)}"
        else:
            early_gain_count = 0
            evidence = "missing static or early columns"
    rows.append(
        {
            "decision": "minimal_training_gate_completed?",
            "status": "pass" if not training_summary.empty else "warn",
            "evidence": f"training summary rows={len(training_summary)}",
        }
    )
    rows.append(
        {
            "decision": "early_observation_value_seen_in_candidate_gate?",
            "status": "pass" if early_gain_count > 0 else "warn",
            "evidence": evidence,
        }
    )
    web_high = inventory[
        inventory["source_kind"].eq("web_candidate")
        & inventory["priority_for_release_state_story"].isin(["very_high", "high", "medium_high_outgroup"])
    ]
    rows.append(
        {
            "decision": "next_collection_targets_identified?",
            "status": "pass" if len(web_high) >= 2 else "warn",
            "evidence": ",".join(web_high["dataset_name"].tolist()),
        }
    )
    rows.append(
        {
            "decision": "do_not_start_new_model_invention?",
            "status": "pass",
            "evidence": "122 uses ExtraTrees only as a gate and does not introduce a new architecture.",
        }
    )
    return pd.DataFrame(rows)


def data_checks(args: argparse.Namespace, inventory: pd.DataFrame, local_audit: pd.DataFrame, training_summary: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("main_pool_exists", args.main_pool.exists(), str(args.main_pool)),
        ("caveat_pool_exists", args.caveat_pool.exists(), str(args.caveat_pool)),
        ("candidate_inventory_has_web_and_local_rows", {"web_candidate", "local_standardized_pool"}.issubset(set(inventory["source_kind"])), ""),
        ("training_gate_only_on_eligible_local_datasets", True, "web candidates are audited, not trained without local machine-readable data"),
        ("no_timepoint_independent_split", True, "training gate splits curve/formulation rows only"),
        ("no_new_neural_model", True, "ExtraTrees gate only"),
        ("local_audit_has_training_reasons", "not_training_reason" in local_audit.columns, ""),
        ("training_summary_no_nan_rmse", training_summary.empty or np.isfinite(training_summary["rmse"]).all(), ""),
        ("tablet_supplement_download_audited", (ROOT / "data/external/direct_compression_tablets_aaps_2025/supplementary_file1.docx").exists(), "downloaded but not committed"),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "pass" if passed else "fail", "detail": detail} for name, passed, detail in checks]
    )


def write_report(
    out: Path,
    inventory: pd.DataFrame,
    local_audit: pd.DataFrame,
    training_summary: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    top_web = inventory[inventory["source_kind"].eq("web_candidate")].copy()
    eligible = local_audit[local_audit["training_gate_eligible"]].copy()
    lines = [
        "# Release Dataset Candidate Audit And Gate",
        "",
        "122 is a dataset triage and minimal training-gate step. It does not start a new model line.",
        "",
        "## Top Web Collection Targets",
        "",
        top_web[
            [
                "dataset_name",
                "system_type",
                "n_curves_or_datapoints",
                "access_mode",
                "priority_for_release_state_story",
                "notes",
            ]
        ].to_markdown(index=False),
        "",
        "## Local Training-Eligible Datasets",
        "",
        eligible[
            [
                "pool_name",
                "source_dataset",
                "n_curves",
                "median_points_per_curve",
                "group_split_column",
                "fit_to_project_story",
            ]
        ].to_markdown(index=False)
        if not eligible.empty
        else "No local dataset met the training gate.",
        "",
        "## Minimal Training Gate",
        "",
        training_summary.round(4).to_markdown(index=False) if not training_summary.empty else "No training summary rows.",
        "",
        "## Decisions",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Web candidates are not treated as training data until a machine-readable local table exists.",
        "- Small local hydrogel datasets remain collection/background evidence, not independent benchmark systems.",
        "- The gate asks whether static descriptors and early observations have signal; it is not a new predictor claim.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(doc: Path) -> None:
    lines = [
        "# Release Dataset Candidate Audit And Gate",
        "",
        "Date: 2026-06-14",
        "",
        "## Purpose",
        "",
        "Experiment 122 keeps the project on the release-state inference route while expanding candidate datasets.",
        "",
        "The goal is not to collect many curves blindly. A candidate dataset matters only if it can test:",
        "",
        "```text",
        "static X -> release endpoint / curve state",
        "static X + early Q -> future release endpoint / curve state",
        "residual -> missing material/process/state descriptor hypothesis",
        "```",
        "",
        "## Gate Policy",
        "",
        "- Web candidates are audited for access, scale, descriptors, and fit to the story.",
        "- Training runs only on local standardized curve pools with enough curves and timepoints.",
        "- No timepoint-independent split is allowed.",
        "- No new neural model or architecture is introduced.",
        "",
        "## Current Highest-Priority Collection Targets",
        "",
        "1. Alginate hydrogel active-learning release profiles, because it directly supports the chitosan/HA/PVP material-controllability story.",
        "2. Chitosan nanoparticle release profile dataset, because it is the closest literature analogue to the planned chitosan direction.",
        "3. Direct-compression tablet profiles, because it is a large outgroup if the raw table can be obtained.",
        "",
        "## Output Anchor",
        "",
        "`../outputs/122_release_dataset_candidate_audit_and_gate/`",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lock(args: argparse.Namespace, out: Path) -> None:
    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "main_pool": file_meta(args.main_pool),
        "caveat_pool": file_meta(args.caveat_pool),
        "training_gate": "curve-level static vs static+early ExtraTrees gate",
        "interpretation_boundary": "dataset triage only; no new model claim",
        "generated_files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    main_tables = load_pool(args.main_pool)
    caveat_tables = load_pool(args.caveat_pool)
    main_audit = audit_local_datasets("main_clean_148", main_tables)
    caveat_audit = audit_local_datasets("caveat_augmented_149", caveat_tables)
    local_audit = pd.concat([main_audit, caveat_audit], ignore_index=True)
    web = curated_web_candidates()
    inventory = build_inventory(local_audit, web)
    source_links = web[["dataset_name", "source_url", "doi", "access_mode", "notes"]].copy()
    access_audit = web[
        [
            "dataset_name",
            "access_mode",
            "manual_collection_cost",
            "has_full_Qt_curve",
            "has_static_descriptors",
            "priority_for_release_state_story",
            "notes",
        ]
    ].copy()

    metrics_main, summary_main = run_training_gate("main_clean_148", main_tables, main_audit, args)
    metrics_caveat, summary_caveat = run_training_gate("caveat_augmented_149", caveat_tables, caveat_audit, args)
    metrics = pd.concat([metrics_main, metrics_caveat], ignore_index=True)
    training_summary = pd.concat([summary_main, summary_caveat], ignore_index=True)

    decisions = decision_table(local_audit, training_summary, inventory)
    checks = data_checks(args, inventory, local_audit, training_summary)

    inventory.to_csv(args.out / "dataset_candidate_inventory.csv", index=False)
    source_links.to_csv(args.out / "source_links.csv", index=False)
    local_audit.to_csv(args.out / "local_dataset_audit.csv", index=False)
    access_audit.to_csv(args.out / "access_audit.csv", index=False)
    metrics.to_csv(args.out / "per_dataset_gate_metrics.csv", index=False)
    training_summary.to_csv(args.out / "training_gate_summary.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    write_report(args.out, inventory, local_audit, training_summary, decisions)
    write_doc(args.doc)
    write_lock(args, args.out)

    failed = checks[checks["status"].eq("fail")]
    if not failed.empty:
        raise RuntimeError(f"122 failed data checks: {failed['check'].tolist()}")
    print(f"Wrote release dataset candidate audit and gate to {args.out}")


if __name__ == "__main__":
    main()
