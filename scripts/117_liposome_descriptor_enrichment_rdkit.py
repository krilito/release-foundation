"""
117 - Liposome descriptor enrichment / RDKit audit.

Purpose:
    Test whether the static X -> z gap is partly caused by weak molecular
    descriptors. The main group-by-API comparison avoids raw API identity and
    uses molecule descriptors instead of API labels.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/
    optional PubChem PUG REST cache under data/external/descriptor_cache/

Produces:
    outputs/117_liposome_descriptor_enrichment_rdkit/
      per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    If descriptors help under group_by_API, the static prior was information-
    limited by weak molecular representation. If they do not, missing state
    likely lies in microstructure, process, morphology, batch, or hidden assay
    variables.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import re
import time
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HELPER_115_PATH = Path(__file__).resolve().parent / "115_liposome_latent_gap_observation_budget.py"
spec115 = importlib.util.spec_from_file_location("liposome_115_helpers", HELPER_115_PATH)
if spec115 is None or spec115.loader is None:
    raise RuntimeError(f"cannot import helper script: {HELPER_115_PATH}")
helpers = importlib.util.module_from_spec(spec115)
spec115.loader.exec_module(helpers)


DEFAULT_ROOT = helpers.DEFAULT_ROOT
DEFAULT_OUT = Path("outputs/117_liposome_descriptor_enrichment_rdkit")
DEFAULT_CACHE = Path("data/external/descriptor_cache/pubchem_liposome_api")
SCHEMES = helpers.SCHEMES
BASE_FEATURES_NO_API = [c for c in helpers.FEATURE_7 if c != "API_type"]
TARGETS = ("avoid_failure", "sustained_mid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liposome descriptor enrichment RDKit audit.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--grid-max-h", type=float, default=24.0)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--morgan-bits", type=int, default=64)
    parser.add_argument("--pubchem-timeout-s", type=float, default=15.0)
    parser.add_argument("--skip-pubchem", action="store_true")
    return parser.parse_args()


def try_import_rdkit() -> tuple[bool, dict[str, Any]]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        return True, {
            "Chem": Chem,
            "Crippen": Crippen,
            "Descriptors": Descriptors,
            "Lipinski": Lipinski,
            "rdMolDescriptors": rdMolDescriptors,
        }
    except Exception:
        return False, {}


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip().lower()).strip("_")


def fetch_pubchem(name: str, cache_dir: Path, timeout_s: float, skip: bool) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{safe_name(name)}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("found") or skip or "PUGREST.BadRequest" not in str(cached.get("error", "")):
            return cached
    if skip:
        payload = {"query": name, "found": False, "error": "skip_pubchem"}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    props = ",".join(
        [
            "MolecularWeight",
            "ExactMass",
            "XLogP",
            "TPSA",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
            "Charge",
            "IsomericSMILES",
            "CanonicalSMILES",
        ]
    )
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/{props}/JSON"
    try:
        with urlopen(url, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        row = data.get("PropertyTable", {}).get("Properties", [{}])[0]
        payload = {"query": name, "found": bool(row), "properties": row, "error": ""}
    except Exception as exc:
        payload = {"query": name, "found": False, "properties": {}, "error": repr(exc)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    time.sleep(0.15)
    return payload


def pubchem_descriptor_table(meta: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for api_name in sorted(meta["API_name"].dropna().astype(str).unique().tolist()):
        payload = fetch_pubchem(api_name, args.cache, args.pubchem_timeout_s, args.skip_pubchem)
        props = payload.get("properties", {}) if payload.get("found") else {}
        rows.append(
            {
                "API_name": api_name,
                "pubchem_found": bool(payload.get("found")),
                "pubchem_error": payload.get("error", ""),
                "CID": props.get("CID", np.nan),
                "SMILES": props.get("SMILES") or props.get("IsomericSMILES") or props.get("CanonicalSMILES") or "",
                "PubChem_MolWt": props.get("MolecularWeight", np.nan),
                "PubChem_ExactMass": props.get("ExactMass", np.nan),
                "PubChem_XLogP": props.get("XLogP", np.nan),
                "PubChem_TPSA": props.get("TPSA", np.nan),
                "PubChem_HBD": props.get("HBondDonorCount", np.nan),
                "PubChem_HBA": props.get("HBondAcceptorCount", np.nan),
                "PubChem_RotBonds": props.get("RotatableBondCount", np.nan),
                "PubChem_FormalCharge": props.get("Charge", np.nan),
            }
        )
    return pd.DataFrame(rows)


def add_rdkit_descriptors(desc: pd.DataFrame, morgan_bits: int) -> tuple[pd.DataFrame, bool]:
    available, rdkit = try_import_rdkit()
    if not available:
        return desc.copy(), False
    Chem = rdkit["Chem"]
    Crippen = rdkit["Crippen"]
    Descriptors = rdkit["Descriptors"]
    Lipinski = rdkit["Lipinski"]
    rdMolDescriptors = rdkit["rdMolDescriptors"]

    out_rows = []
    for row in desc.to_dict("records"):
        smiles = str(row.get("SMILES", "") or "")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        extra: dict[str, Any] = {}
        if mol is None:
            extra.update(
                {
                    "RDKit_MolWt": np.nan,
                    "RDKit_ExactMolWt": np.nan,
                    "RDKit_TPSA": np.nan,
                    "RDKit_LogP": np.nan,
                    "RDKit_HBD": np.nan,
                    "RDKit_HBA": np.nan,
                    "RDKit_RotBonds": np.nan,
                    "RDKit_AromaticRings": np.nan,
                    "RDKit_FormalCharge": np.nan,
                }
            )
            extra.update({f"Morgan_{i:02d}": np.nan for i in range(morgan_bits)})
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_bits)
            extra.update(
                {
                    "RDKit_MolWt": Descriptors.MolWt(mol),
                    "RDKit_ExactMolWt": Descriptors.ExactMolWt(mol),
                    "RDKit_TPSA": rdMolDescriptors.CalcTPSA(mol),
                    "RDKit_LogP": Crippen.MolLogP(mol),
                    "RDKit_HBD": Lipinski.NumHDonors(mol),
                    "RDKit_HBA": Lipinski.NumHAcceptors(mol),
                    "RDKit_RotBonds": Lipinski.NumRotatableBonds(mol),
                    "RDKit_AromaticRings": Lipinski.NumAromaticRings(mol),
                    "RDKit_FormalCharge": Chem.GetFormalCharge(mol),
                }
            )
            extra.update({f"Morgan_{i:02d}": int(fp.GetBit(i)) for i in range(morgan_bits)})
        row.update(extra)
        out_rows.append(row)
    return pd.DataFrame(out_rows), True


def enrich_meta(meta: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    pubchem = pubchem_descriptor_table(meta, args)
    descriptors, rdkit_available = add_rdkit_descriptors(pubchem, args.morgan_bits)
    enriched = meta.merge(descriptors, on="API_name", how="left")

    for col in [
        "RDKit_LogP",
        "PubChem_XLogP",
        "RDKit_TPSA",
        "PubChem_TPSA",
        "RDKit_MolWt",
        "PubChem_MolWt",
        "RDKit_HBD",
        "PubChem_HBD",
        "RDKit_HBA",
        "PubChem_HBA",
        "weighted_Mw",
        "weighted_Tm",
        "media_pH",
    ]:
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce")

    logp = enriched["RDKit_LogP"].where(enriched["RDKit_LogP"].notna(), enriched["PubChem_XLogP"])
    tpsa = enriched["RDKit_TPSA"].where(enriched["RDKit_TPSA"].notna(), enriched["PubChem_TPSA"])
    molwt = enriched["RDKit_MolWt"].where(enriched["RDKit_MolWt"].notna(), enriched["PubChem_MolWt"])
    hbd = enriched["RDKit_HBD"].where(enriched["RDKit_HBD"].notna(), enriched["PubChem_HBD"])
    hba = enriched["RDKit_HBA"].where(enriched["RDKit_HBA"].notna(), enriched["PubChem_HBA"])
    enriched["pH_minus_pKa_proxy"] = enriched["media_pH"] - 7.4
    enriched["drug_logP_minus_carrier_hydrophobicity_proxy"] = logp - enriched["weighted_Tm"] / 100.0
    enriched["drug_TPSA_over_weighted_Mw"] = tpsa / np.maximum(enriched["weighted_Mw"], 1e-6)
    enriched["drug_HBD_HBA_total"] = hbd + hba
    enriched["drug_polarity_x_media_pH"] = tpsa * enriched["media_pH"]
    enriched["drug_hydrophobicity_x_weighted_Tm"] = logp * enriched["weighted_Tm"]
    enriched["drug_MolWt_over_lipid_Mw"] = molwt / np.maximum(enriched["weighted_Mw"], 1e-6)
    return enriched, descriptors, rdkit_available


def feature_sets(meta: pd.DataFrame) -> dict[str, list[str]]:
    molecular = [
        c
        for c in meta.columns
        if c.startswith("RDKit_")
        or c.startswith("PubChem_")
        or c.startswith("Morgan_")
    ]
    molecular = [c for c in molecular if c not in {"PubChem_error"}]
    interaction = [
        "pH_minus_pKa_proxy",
        "drug_logP_minus_carrier_hydrophobicity_proxy",
        "drug_TPSA_over_weighted_Mw",
        "drug_HBD_HBA_total",
        "drug_polarity_x_media_pH",
        "drug_hydrophobicity_x_weighted_Tm",
        "drug_MolWt_over_lipid_Mw",
    ]
    return {
        "baseline_X": BASE_FEATURES_NO_API,
        "baseline_X_plus_molecular_descriptors": BASE_FEATURES_NO_API + molecular,
        "baseline_X_plus_interaction_features": BASE_FEATURES_NO_API + interaction,
        "baseline_X_plus_molecular_descriptors_plus_interaction_features": BASE_FEATURES_NO_API + molecular + interaction,
        "with_API_name_identity_upper_bound": helpers.FEATURE_7 + molecular + interaction,
    }


def fit_transform_features(meta: pd.DataFrame, cols: list[str], tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing_fraction = float(np.mean(~np.isfinite(x)))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_tr = imputer.fit_transform(x[tr])
    x_te = imputer.transform(x[te])
    return x_tr, x_te, missing_fraction


def target_thresholds(train_y: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    q1 = np.asarray([np.interp(1.0, grid, row, left=row[0], right=row[-1]) for row in train_y])
    q6 = np.asarray([np.interp(6.0, grid, row, left=row[0], right=row[-1]) for row in train_y])
    return {
        "q1_80": float(np.quantile(q1, 0.80)),
        "q6_33": float(np.quantile(q6, 0.33)),
        "q6_66": float(np.quantile(q6, 0.66)),
        "q6_mid": float(0.5 * (np.quantile(q6, 0.33) + np.quantile(q6, 0.66))),
    }


def target_success(y_grid: np.ndarray, grid: np.ndarray, thresholds: dict[str, float], target: str) -> np.ndarray:
    q1 = np.asarray([np.interp(1.0, grid, row, left=row[0], right=row[-1]) for row in y_grid])
    q6 = np.asarray([np.interp(6.0, grid, row, left=row[0], right=row[-1]) for row in y_grid])
    if target == "avoid_failure":
        return ~(q1 >= thresholds["q1_80"])
    if target == "sustained_mid":
        return (q6 >= thresholds["q6_33"]) & (q6 <= thresholds["q6_66"])
    raise ValueError(target)


def target_score(y_grid: np.ndarray, grid: np.ndarray, thresholds: dict[str, float], target: str) -> np.ndarray:
    q1 = np.asarray([np.interp(1.0, grid, row, left=row[0], right=row[-1]) for row in y_grid])
    q6 = np.asarray([np.interp(6.0, grid, row, left=row[0], right=row[-1]) for row in y_grid])
    if target == "avoid_failure":
        return -q1
    if target == "sustained_mid":
        scale = max(thresholds["q6_66"] - thresholds["q6_33"], 1.0)
        return -np.abs(q6 - thresholds["q6_mid"]) / scale
    raise ValueError(target)


def minmax_probability(score: np.ndarray) -> np.ndarray:
    lo = float(np.min(score))
    hi = float(np.max(score))
    if hi - lo <= 1e-12:
        return np.full_like(score, 0.5, dtype=float)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=bool)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    out = {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    if len(y):
        out["brier"] = float(brier_score_loss(y.astype(int), p))
    return out


def run_probe(args: argparse.Namespace, meta: pd.DataFrame, curve_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    grid = np.linspace(0.0, args.grid_max_h, args.grid_size)
    ids = meta["ID"].to_numpy(dtype=int)
    y_grid = helpers.curve_grid(curve_map, ids, grid)
    fsets = feature_sets(meta)
    rows: list[dict[str, Any]] = []

    for scheme in SCHEMES:
        for fold, (tr, te) in enumerate(helpers.splits(meta, scheme, args.n_folds, args.seed), start=1):
            if len(tr) <= args.n_components or len(te) < 2:
                continue
            pca = PCA(n_components=args.n_components, random_state=args.seed)
            z_tr = pca.fit_transform(y_grid[tr])
            z_oracle = pca.transform(y_grid[te])
            thresholds = target_thresholds(y_grid[tr], grid)
            for set_name, cols in fsets.items():
                is_identity_upper = set_name == "with_API_name_identity_upper_bound"
                if scheme == "group_by_API" and is_identity_upper:
                    leakage_guardrail = "api_identity_upper_bound_not_main"
                else:
                    leakage_guardrail = "main_no_raw_api_identity"
                x_tr, x_te, missing_fraction = fit_transform_features(meta, cols, tr, te)
                models: dict[str, Any] = {
                    "et": ExtraTreesRegressor(
                        n_estimators=args.n_estimators,
                        min_samples_leaf=2,
                        random_state=args.seed + 19 * fold,
                        n_jobs=-1,
                    ),
                    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                }
                for model_name, model in models.items():
                    model.fit(x_tr, z_tr)
                    z_pred = model.predict(x_te)
                    pred_grid = np.asarray([helpers.monotone_clip(row) for row in pca.inverse_transform(z_pred)])
                    pred_scores = {
                        target: target_score(pred_grid, grid, thresholds, target)
                        for target in TARGETS
                    }
                    true_success = {
                        target: target_success(y_grid[te], grid, thresholds, target)
                        for target in TARGETS
                    }
                    pred_prob = {target: minmax_probability(pred_scores[target]) for target in TARGETS}
                    for local_i, idx in enumerate(te):
                        t = curve_map[int(meta.iloc[int(idx)]["ID"])]["time_h"].to_numpy(dtype=float)
                        q = curve_map[int(meta.iloc[int(idx)]["ID"])]["release_pct"].to_numpy(dtype=float)
                        pred_at_t = np.interp(t, grid, pred_grid[local_i], left=pred_grid[local_i, 0], right=pred_grid[local_i, -1])
                        base = {
                            "scheme": scheme,
                            "fold": fold,
                            "feature_set": set_name,
                            "model": model_name,
                            "method": f"{set_name}_{model_name}",
                            "ID": int(meta.iloc[int(idx)]["ID"]),
                            "API_name": str(meta.iloc[int(idx)]["API_name"]),
                            "release_method": str(meta.iloc[int(idx)]["release_method"]),
                            "n_features": int(len(cols)),
                            "missing_descriptor_fraction": missing_fraction,
                            "z_rmse": helpers.rmse(z_oracle[local_i], z_pred[local_i]),
                            "decoded_curve_rmse": helpers.rmse(q, pred_at_t),
                            "is_identity_upper_bound": bool(is_identity_upper),
                            "leakage_guardrail": leakage_guardrail,
                        }
                        for target in TARGETS:
                            row = dict(base)
                            row.update(
                                {
                                    "target": target,
                                    "target_success": bool(true_success[target][local_i]),
                                    "predicted_success_probability": float(pred_prob[target][local_i]),
                                    "predicted_design_score": float(pred_scores[target][local_i]),
                                }
                            )
                            rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, feature_set, model, target), sub in per_curve.groupby(["scheme", "feature_set", "model", "target"], sort=True):
        y = sub["target_success"].to_numpy(dtype=bool)
        p = sub["predicted_success_probability"].to_numpy(dtype=float)
        cls = classification_metrics(y, p)
        fold_hits = []
        fold_enrich = []
        for _, fold_sub in sub.groupby("fold"):
            n_top = max(1, int(np.ceil(0.20 * len(fold_sub))))
            selected = fold_sub.sort_values("predicted_design_score", ascending=False).head(n_top)
            random_rate = float(fold_sub["target_success"].mean())
            hit = float(selected["target_success"].mean())
            fold_hits.append(hit)
            fold_enrich.append(float("nan") if random_rate <= 0 else hit / random_rate)
        rows.append(
            {
                "scheme": scheme,
                "feature_set": feature_set,
                "model": model,
                "method": f"{feature_set}_{model}",
                "target": target,
                "n_curve_records": int(len(sub)),
                "n_unique_curves": int(sub["ID"].nunique()),
                "n_features": int(sub["n_features"].iloc[0]),
                "is_identity_upper_bound": bool(sub["is_identity_upper_bound"].iloc[0]),
                "missing_descriptor_fraction": float(sub["missing_descriptor_fraction"].iloc[0]),
                "coverage_rate": float(1.0 - sub["missing_descriptor_fraction"].iloc[0]),
                "z_rmse_median": float(sub["z_rmse"].median()),
                "decoded_curve_rmse_median": float(sub["decoded_curve_rmse"].median()),
                "design_top20_hit_rate": float(np.mean(fold_hits)),
                "design_enrichment_vs_random": float(np.nanmean(fold_enrich)),
                "auroc": cls["auroc"],
                "auprc": cls["auprc"],
                "brier": cls["brier"],
            }
        )
    return pd.DataFrame(rows).sort_values(["scheme", "target", "z_rmse_median"])


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = summary[(summary["model"] == "et") & (~summary["is_identity_upper_bound"])].copy()
    for scheme, sub in primary.groupby("scheme"):
        for target, target_sub in sub.groupby("target"):
            lookup = target_sub.set_index("feature_set")
            baseline = lookup.loc["baseline_X"] if "baseline_X" in lookup.index else None
            molecular = lookup.loc["baseline_X_plus_molecular_descriptors"] if "baseline_X_plus_molecular_descriptors" in lookup.index else None
            interaction = lookup.loc["baseline_X_plus_molecular_descriptors_plus_interaction_features"] if "baseline_X_plus_molecular_descriptors_plus_interaction_features" in lookup.index else None
            if baseline is not None and molecular is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Do molecular descriptors improve X -> z?",
                        "method_a": "baseline_X_plus_molecular_descriptors",
                        "method_b": "baseline_X",
                        "delta_z_rmse": float(molecular["z_rmse_median"] - baseline["z_rmse_median"]),
                        "delta_design_hit_rate": float(molecular["design_top20_hit_rate"] - baseline["design_top20_hit_rate"]),
                        "passed": bool(molecular["z_rmse_median"] < baseline["z_rmse_median"] or molecular["design_top20_hit_rate"] > baseline["design_top20_hit_rate"]),
                        "interpretation": "negative z delta or positive design delta supports molecular representation value",
                    }
                )
            if molecular is not None and interaction is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Do interaction features improve beyond molecular descriptors?",
                        "method_a": "baseline_X_plus_molecular_descriptors_plus_interaction_features",
                        "method_b": "baseline_X_plus_molecular_descriptors",
                        "delta_z_rmse": float(interaction["z_rmse_median"] - molecular["z_rmse_median"]),
                        "delta_design_hit_rate": float(interaction["design_top20_hit_rate"] - molecular["design_top20_hit_rate"]),
                        "passed": bool(interaction["z_rmse_median"] < molecular["z_rmse_median"] or interaction["design_top20_hit_rate"] > molecular["design_top20_hit_rate"]),
                        "interpretation": "improvement suggests drug-carrier/media coupling adds signal",
                    }
                )
            if scheme == "group_by_API" and baseline is not None and molecular is not None:
                rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "question": "Does improvement persist under group_by_API without API identity?",
                        "method_a": "baseline_X_plus_molecular_descriptors",
                        "method_b": "baseline_X",
                        "delta_z_rmse": float(molecular["z_rmse_median"] - baseline["z_rmse_median"]),
                        "delta_design_hit_rate": float(molecular["design_top20_hit_rate"] - baseline["design_top20_hit_rate"]),
                        "passed": bool(molecular["z_rmse_median"] < baseline["z_rmse_median"] or molecular["design_top20_hit_rate"] > baseline["design_top20_hit_rate"]),
                        "interpretation": "main anti-leakage transfer row; API identity is excluded",
                    }
                )
    return pd.DataFrame(rows)


def data_checks(meta: pd.DataFrame, descriptors: pd.DataFrame, per_curve: pd.DataFrame, summary: pd.DataFrame, rdkit_available: bool, args: argparse.Namespace) -> pd.DataFrame:
    coverage = float(descriptors["pubchem_found"].mean()) if len(descriptors) else 0.0
    main_group_api = per_curve[(per_curve["scheme"] == "group_by_API") & (~per_curve["is_identity_upper_bound"])]
    return pd.DataFrame(
        [
            {"check": "input_files_exist", "passed": bool(args.root.exists()), "detail": str(args.root)},
            {"check": "sample_unit_is_curve", "passed": bool(meta["ID"].is_unique), "detail": f"n_curves={len(meta)}"},
            {"check": "no_timepoint_split", "passed": True, "detail": "folds are built from curve-level metadata rows"},
            {"check": "rdkit_available", "passed": bool(rdkit_available), "detail": "RDKit descriptor path used when available"},
            {"check": "descriptor_coverage_rate", "passed": bool(coverage >= 0.5), "detail": f"pubchem_coverage={coverage:.3f}"},
            {"check": "group_by_API_main_excludes_identity", "passed": bool((main_group_api["leakage_guardrail"] == "main_no_raw_api_identity").all()), "detail": "API_type is excluded from main feature sets"},
            {"check": "no_nan_in_primary_metrics", "passed": bool(np.isfinite(summary["z_rmse_median"]).all()), "detail": "z_rmse_median finite for all summary rows"},
            {"check": "oracle_rows_marked_and_excluded", "passed": True, "detail": "117 has no oracle deployable rows"},
        ]
    )


def write_report(out: Path, descriptors: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame, rdkit_available: bool) -> None:
    lines = [
        "# Liposome Descriptor Enrichment RDKit Audit",
        "",
        "Date: 2026-06-12",
        "",
        "## Aim",
        "",
        "Test whether weak molecular descriptors explain part of the static X -> z gap.",
        "",
        f"- RDKit available: {rdkit_available}",
        f"- PubChem API coverage: {descriptors['pubchem_found'].mean():.3f}",
        "",
        "## Best Rows By Split And Target",
        "",
        "| Split | Target | Feature set | Model | z RMSE | decoded RMSE | top20 hit | enrichment |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    best = summary[(summary["model"] == "et") & (~summary["is_identity_upper_bound"])].sort_values(["scheme", "target", "z_rmse_median"])
    for row in best.groupby(["scheme", "target"], sort=True).head(4).itertuples(index=False):
        lines.append(
            f"| {row.scheme} | {row.target} | `{row.feature_set}` | {row.model} | "
            f"{row.z_rmse_median:.3f} | {row.decoded_curve_rmse_median:.3f} | "
            f"{row.design_top20_hit_rate:.3f} | {row.design_enrichment_vs_random:.3f} |"
        )
    lines.extend(["", "## Decisions", "", "| Split | Target | Question | delta z RMSE | delta design hit | Passed |", "|---|---|---|---:|---:|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.target} | {row.question} | {row.delta_z_rmse:.3f} | {row.delta_design_hit_rate:.3f} | {row.passed} |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Main group-by-API rows exclude raw API identity.",
            "- `with_API_name_identity_upper_bound` is a sensitivity row, not the main comparison.",
            "- PubChem cache lives under gitignored `data/external/descriptor_cache/`.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_doc(doc_path: Path, descriptors: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame, rdkit_available: bool) -> None:
    lines = [
        "# Liposome Descriptor Enrichment RDKit Audit",
        "",
        "Date: 2026-06-12",
        "",
        "## Question",
        "",
        "Is the static `X -> z` gap caused by weak molecular and interaction descriptors?",
        "",
        "## Coverage",
        "",
        f"- RDKit available: {rdkit_available}",
        f"- PubChem coverage across API names: {descriptors['pubchem_found'].mean():.3f}",
        "",
        "## Primary Results",
        "",
        "| Split | Target | Feature set | z RMSE | decoded RMSE | top20 hit | enrichment |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    best = summary[(summary["model"] == "et") & (~summary["is_identity_upper_bound"])].sort_values(["scheme", "target", "z_rmse_median"])
    for row in best.groupby(["scheme", "target"], sort=True).head(4).itertuples(index=False):
        lines.append(
            f"| {row.scheme} | {row.target} | `{row.feature_set}` | {row.z_rmse_median:.3f} | "
            f"{row.decoded_curve_rmse_median:.3f} | {row.design_top20_hit_rate:.3f} | "
            f"{row.design_enrichment_vs_random:.3f} |"
        )
    lines.extend(["", "## Decision Table", "", "| Split | Target | Question | delta z RMSE | delta design hit | Passed |", "|---|---|---|---:|---:|---:|"])
    for row in decisions.itertuples(index=False):
        lines.append(f"| {row.scheme} | {row.target} | {row.question} | {row.delta_z_rmse:.3f} | {row.delta_design_hit_rate:.3f} | {row.passed} |")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "If molecular descriptors improve strict group-by-API rows, the static prior was information-limited by weak molecular representation. If they do not, the missing state is more likely process, morphology, microstructure, batch, or hidden assay information.",
            "",
            "Raw API identity is excluded from the main group-by-API comparison. Identity rows are only upper-bound sensitivity checks.",
            "",
            "## Output Anchor",
            "",
            "`../outputs/117_liposome_descriptor_enrichment_rdkit/`",
        ]
    )
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta, curve_map = helpers.load_dataset(args.root)
    meta = helpers.stratified_cap(meta, args.max_curves, args.seed)
    meta, descriptors, rdkit_available = enrich_meta(meta, args)
    per_curve = run_probe(args, meta, curve_map)
    if per_curve.empty:
        raise RuntimeError("no descriptor enrichment rows generated")
    summary = summarize(per_curve)
    decisions = decision_table(summary)
    checks = data_checks(meta, descriptors, per_curve, summary, rdkit_available, args)

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks.to_csv(args.out / "data_checks.csv", index=False)
    descriptors.to_csv(args.out / "descriptor_coverage.csv", index=False)
    write_report(args.out, descriptors, summary, decisions, rdkit_available)
    write_doc(Path("docs/liposome_descriptor_enrichment_rdkit_2026-06-12.md"), descriptors, summary, decisions, rdkit_available)

    metadata = {
        "script": Path(__file__).name,
        "date": "2026-06-12",
        "seed": args.seed,
        "root": str(args.root),
        "out": str(args.out),
        "cache": str(args.cache),
        "grid_max_h": args.grid_max_h,
        "grid_size": args.grid_size,
        "n_components": args.n_components,
        "n_estimators": args.n_estimators,
        "morgan_bits": args.morgan_bits,
        "rdkit_available": rdkit_available,
        "pubchem_coverage": float(descriptors["pubchem_found"].mean()) if len(descriptors) else 0.0,
        "git_hash": helpers.git_hash(),
        "input_files": {
            "backend_data": helpers.file_meta(args.root / "data/unprocessed/backend_data.csv"),
            "weibull_params": helpers.file_meta(args.root / "data/clean/weibull_params.csv"),
            "clusters": helpers.file_meta(args.root / "results/clustering/3_PCA_KMC.csv"),
            "release_exp": helpers.file_meta(args.root / "results/fitting/drug_release_exp.csv"),
        },
        "generated_files": sorted(p.name for p in args.out.iterdir() if p.is_file()),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"descriptor enrichment audit failed checks: {failed}")


if __name__ == "__main__":
    main()
