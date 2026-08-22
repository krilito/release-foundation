"""93 - PLGA conditional feature relevance protocol.

Purpose:
    Define the leakage-safe algorithm for testing whether each PLGA curve can
    be predicted from a curve-specific static feature combination.

This script is intentionally dry-run first. It writes the planned feature
combos, leak guards, and analysis protocol. It does not train models unless a
future implementation adds an explicit execution mode.

Core distinction:
    - test_oracle_combo: after-the-fact diagnostic, not deployable evidence.
    - gated_combo: selected before seeing heldout Q(t), deployable evidence.

Produces:
    outputs/93_plga_conditional_feature_relevance_protocol/
      feature_combo_plan.csv
      leak_guard.json
      protocol.md
      lock_metadata.json
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_OUT = Path("outputs/93_plga_conditional_feature_relevance_protocol")

BASE_GROUPS: dict[str, dict[str, list[str]]] = {
    "polymer_core": {
        "numeric": ["Polymer_MW", "LA/GA", "CL Ratio"],
        "categorical": ["polymer_family"],
    },
    "formulation_core": {
        "numeric": ["Initial D/M ratio", "DLC", "DLC_percent", "EE", "Particle_Size", "SA-V", "SE"],
        "categorical": [],
    },
    "drug_physchem": {
        "numeric": ["Drug_Mw", "Drug_TPSA", "Drug_NHA", "Drug_LogP", "Drug_Pka", "Drug_Tm"],
        "categorical": ["payload_name"],
    },
    "environment_protocol": {
        "numeric": ["media_pH", "media_temp_oC"],
        "categorical": ["release_medium_condition", "release_method", "measurement_assay", "light_condition"],
    },
    "source_diagnostic_only": {
        "numeric": [],
        "categorical": ["source_dataset", "source_group"],
    },
}

TARGET_DERIVED_COLUMNS = {
    "alpha",
    "beta",
    "best_model_mae",
    "best_model_mae_value",
    "best_model_aic",
    "best_model_aic_value",
    "best_model_f2",
    "best_model_f2_value",
    "timepoint_count_sufficient",
    "shape_resembles_profile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write PLGA conditional feature-relevance protocol.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--include-source-diagnostic", action="store_true")
    return parser.parse_args()


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def args_to_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def read_formulations(pool_dir: Path) -> pd.DataFrame:
    path = pool_dir / "formulations.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    forms = pd.read_csv(path)
    return forms[forms["source_dataset"].astype(str).isin({"internal181", "cross321"})].copy()


def build_feature_combos(forms: pd.DataFrame, max_combo_size: int, include_source_diagnostic: bool) -> pd.DataFrame:
    group_names = [name for name in BASE_GROUPS if include_source_diagnostic or name != "source_diagnostic_only"]
    rows: list[dict[str, Any]] = []
    combo_id = 0
    for size in range(1, max_combo_size + 1):
        for names in combinations(group_names, size):
            numeric: list[str] = []
            categorical: list[str] = []
            for name in names:
                numeric.extend(BASE_GROUPS[name]["numeric"])
                categorical.extend(BASE_GROUPS[name]["categorical"])
            numeric = sorted(set(numeric))
            categorical = sorted(set(categorical))
            target_overlap = sorted((set(numeric) | set(categorical)) & TARGET_DERIVED_COLUMNS)
            available_cols = [col for col in numeric + categorical if col in forms.columns]
            nonmissing = {
                col: float(forms[col].notna().mean()) if col in forms.columns else 0.0
                for col in numeric + categorical
            }
            rows.append(
                {
                    "combo_id": f"combo_{combo_id:03d}",
                    "combo_name": "+".join(names),
                    "groups": ",".join(names),
                    "numeric_cols": ",".join(numeric),
                    "categorical_cols": ",".join(categorical),
                    "n_numeric": len(numeric),
                    "n_categorical": len(categorical),
                    "n_available_cols": len(available_cols),
                    "min_nonmissing_fraction": min(nonmissing.values()) if nonmissing else 1.0,
                    "contains_source_diagnostic": any(name == "source_diagnostic_only" for name in names),
                    "target_derived_overlap": ",".join(target_overlap),
                    "leak_guard_pass": len(target_overlap) == 0,
                }
            )
            combo_id += 1
    return pd.DataFrame(rows)


def leak_guard(combos: pd.DataFrame) -> dict[str, Any]:
    failed = combos[~combos["leak_guard_pass"].astype(bool)]
    return {
        "prediction_inputs_allowed": [
            "static formulation/material descriptors available before release measurement",
            "categorical formulation/source fields only when explicitly marked diagnostic",
        ],
        "prediction_inputs_forbidden": [
            "early Q(t)",
            "heldout curve fitted theta",
            "heldout curve shape-family fit",
            "any column derived from the full release curve",
            "feature/combo selection based on heldout Q(t)",
        ],
        "test_oracle_combo_status": "diagnostic_only_not_deployable",
        "gated_combo_status": "valid_only_if_gate_uses_static_inputs_and_train-fold labels",
        "failed_combo_count": int(len(failed)),
        "failed_combos": failed[["combo_id", "combo_name", "target_derived_overlap"]].to_dict(orient="records"),
    }


def write_protocol(out: Path, combos: pd.DataFrame, guard: dict[str, Any]) -> None:
    deployable = combos[~combos["contains_source_diagnostic"].astype(bool)].copy()
    diagnostic = combos[combos["contains_source_diagnostic"].astype(bool)].copy()
    lines = [
        "# PLGA Conditional Feature-Relevance Protocol",
        "",
        "Goal: test whether individual PLGA curves can be predicted by different static feature combinations without leaking heldout release information.",
        "",
        "## Two Answers, Not One",
        "",
        "1. `test_oracle_combo`: after evaluating all feature combos on a heldout curve, report the best combo. This answers whether the curve is representable by some static subset. It is diagnostic only.",
        "2. `gated_combo`: train a gate on train-fold out-of-fold best-combo labels, then choose a combo for a heldout curve using only static descriptors. This is the valid predictive claim.",
        "",
        "## Algorithm",
        "",
        "For each outer split:",
        "",
        "1. Fit expert models for each predefined feature combo, shape family, and regressor using only training curves.",
        "2. Inside the training fold, create out-of-fold predictions for every expert.",
        "3. Assign each training curve an OOF `best_combo_label` and `predictability_score = best_oof_rmse / intercept_oof_rmse`.",
        "4. Train a gate from static descriptors to `best_combo_label` or to expert weights.",
        "5. On heldout curves, evaluate three rows:",
        "   - `global_train_best`: combo chosen by train-fold aggregate OOF score.",
        "   - `gated_combo`: combo chosen by the static gate before seeing heldout Q(t).",
        "   - `test_oracle_combo`: best combo after seeing heldout errors, diagnostic only.",
        "6. Report per-curve gap: `gated_rmse - oracle_rmse`. Small gap means the curve-specific static subset is learnable before measurement.",
        "",
        "## Decision Rules",
        "",
        "- If oracle is strong but gate is weak: curve-specific feature relevance exists, but cannot yet be inferred from current descriptors.",
        "- If gate approaches oracle: conditional feature relevance is a real predictive mechanism.",
        "- If both oracle and gate are weak: static descriptors are insufficient for those curves.",
        "- If source-diagnostic combos dominate: current success may be source/protocol memory, not physical transfer.",
        "",
        "## Leak Guard",
        "",
        "```json",
        json.dumps(guard, indent=2),
        "```",
        "",
        "## Deployable Feature Combos",
        "",
        deployable.to_markdown(index=False),
        "",
        "## Diagnostic-Only Feature Combos",
        "",
        diagnostic.to_markdown(index=False) if len(diagnostic) else "No source-diagnostic combos included.",
        "",
    ]
    (out / "protocol.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    forms = read_formulations(args.pool_dir)
    combos = build_feature_combos(forms, args.max_combo_size, args.include_source_diagnostic)
    guard = leak_guard(combos)
    if guard["failed_combo_count"]:
        raise RuntimeError(f"Leak guard failed: {guard['failed_combos']}")
    combos.to_csv(args.out / "feature_combo_plan.csv", index=False)
    (args.out / "leak_guard.json").write_text(json.dumps(guard, indent=2), encoding="utf-8")
    write_protocol(args.out, combos, guard)
    metadata = {
        "script": "scripts/93_plga_conditional_feature_relevance_protocol.py",
        "git_hash": git_hash(),
        "args": args_to_metadata(args),
        "pool_dir": str(args.pool_dir),
        "out": str(args.out),
        "max_combo_size": int(args.max_combo_size),
        "include_source_diagnostic": bool(args.include_source_diagnostic),
        "n_plga_curves": int(forms["unified_curve_id"].nunique()),
        "n_feature_combos": int(len(combos)),
        "note": "Protocol only. Formal experiment intentionally not run.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[93] wrote protocol only:", args.out)
    print(combos[["combo_id", "combo_name", "n_available_cols", "min_nonmissing_fraction", "contains_source_diagnostic"]].to_string(index=False))


if __name__ == "__main__":
    main()
