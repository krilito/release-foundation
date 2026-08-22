"""85 - Shape-prior leakage audit for release transfer baselines.

Purpose:
    Verify that shape-prior baselines in the release transfer probe derive
    their prior parameters only from each split's training curve IDs, even when
    the fallback fit bank contains all curves.

Consumes:
    outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored/
    outputs/158_freeze_theta_targets/all_family_fits.csv

Produces:
    outputs/85_release_shape_prior_leakage_audit/
      shape_source_by_split_family.csv
      baseline_family_availability.csv
      leakage_rows.csv
      summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN = Path("outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored")
DEFAULT_THETA_DIR = Path("outputs/158_freeze_theta_targets")
DEFAULT_OUT = Path("outputs/85_release_shape_prior_leakage_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit shape-prior leakage in release transfer baselines.")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--theta-dir", type=Path, default=DEFAULT_THETA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def shape_family_from_baseline(name: str) -> str | None:
    if name.startswith("ShapePriorIncrementAnchored:"):
        return name.split(":", 1)[1]
    if name.startswith("ShapePrior:"):
        return name.split(":", 1)[1]
    return None


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(values.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def build_source_table(manifest: pd.DataFrame, fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_ids = fits[["unified_curve_id", "family"]].dropna().copy()
    fit_ids["unified_curve_id"] = fit_ids["unified_curve_id"].astype(str)
    fit_ids["family"] = fit_ids["family"].astype(str)
    all_families = sorted(fit_ids["family"].unique())

    source_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    split_cols = ["split_kind", "heldout_system"]
    for (split_kind, heldout_system), split_df in manifest.groupby(split_cols, dropna=False):
        train = split_df[split_df["role"] == "train"].copy()
        test = split_df[split_df["role"] == "test"].copy()
        train_ids = set(train["unified_curve_id"].astype(str))
        test_ids = set(test["unified_curve_id"].astype(str))
        overlap = train_ids & test_ids
        if overlap:
            for curve_id in sorted(overlap):
                leakage_rows.append(
                    {
                        "leak_type": "manifest_train_test_overlap",
                        "split_kind": split_kind,
                        "heldout_system": heldout_system,
                        "family": "",
                        "unified_curve_id": curve_id,
                    }
                )
        if str(split_kind) == "loso-system":
            wrong_system = train[train["system_id"].astype(str) == str(heldout_system)]
            for curve_id in wrong_system["unified_curve_id"].astype(str).unique():
                leakage_rows.append(
                    {
                        "leak_type": "loso_train_contains_heldout_system",
                        "split_kind": split_kind,
                        "heldout_system": heldout_system,
                        "family": "",
                        "unified_curve_id": curve_id,
                    }
                )

        for family in all_families:
            fam = fit_ids[fit_ids["family"] == family]
            source_ids = set(fam["unified_curve_id"]) & train_ids
            leaked_ids = source_ids & test_ids
            heldout_fit_ids = set(fam["unified_curve_id"]) & test_ids
            source_rows.append(
                {
                    "split_kind": split_kind,
                    "heldout_system": heldout_system,
                    "family": family,
                    "n_train_curves": int(len(train_ids)),
                    "n_test_curves": int(len(test_ids)),
                    "n_fit_bank_family_rows": int(len(fam)),
                    "n_shape_source_curves": int(len(source_ids)),
                    "n_heldout_fit_bank_curves": int(len(heldout_fit_ids)),
                    "n_leaked_source_curves": int(len(leaked_ids)),
                    "has_source": bool(len(source_ids) > 0),
                }
            )
            for curve_id in sorted(leaked_ids):
                leakage_rows.append(
                    {
                        "leak_type": "shape_source_contains_test_curve",
                        "split_kind": split_kind,
                        "heldout_system": heldout_system,
                        "family": family,
                        "unified_curve_id": curve_id,
                    }
                )
    return pd.DataFrame(source_rows), pd.DataFrame(leakage_rows)


def build_availability_table(baselines: pd.DataFrame, source_table: pd.DataFrame) -> pd.DataFrame:
    shape = baselines.copy()
    shape["family"] = shape["baseline"].astype(str).map(shape_family_from_baseline)
    shape = shape[shape["family"].notna()].copy()
    if shape.empty:
        return pd.DataFrame()
    availability = (
        shape.groupby(["split_kind", "heldout_system", "budget_kind", "budget_label", "budget_value", "baseline", "family"], dropna=False)
        .agg(n_baseline_curves=("unified_curve_id", "nunique"))
        .reset_index()
    )
    source_cols = [
        "split_kind",
        "heldout_system",
        "family",
        "n_shape_source_curves",
        "n_heldout_fit_bank_curves",
        "n_leaked_source_curves",
        "has_source",
    ]
    availability = availability.merge(source_table[source_cols], on=["split_kind", "heldout_system", "family"], how="left")
    availability["baseline_without_source"] = availability["n_baseline_curves"].gt(0) & ~availability["has_source"].fillna(False)
    availability["baseline_with_leaked_source"] = availability["n_baseline_curves"].gt(0) & availability["n_leaked_source_curves"].fillna(0).gt(0)
    return availability.sort_values(["split_kind", "heldout_system", "budget_kind", "budget_value", "baseline"])


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def write_summary(
    out: Path,
    run: Path,
    theta_dir: Path,
    source_table: pd.DataFrame,
    availability: pd.DataFrame,
    leakage: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    source_summary = (
        source_table.groupby(["split_kind", "family"], dropna=False)
        .agg(
            n_splits=("heldout_system", "nunique"),
            min_shape_source_curves=("n_shape_source_curves", "min"),
            median_shape_source_curves=("n_shape_source_curves", "median"),
            max_leaked_source_curves=("n_leaked_source_curves", "max"),
        )
        .reset_index()
    )
    bad_availability = availability[
        availability["baseline_without_source"].fillna(False) | availability["baseline_with_leaked_source"].fillna(False)
    ].copy()
    lines = [
        "# Shape-Prior Leakage Audit",
        "",
        f"- Source run: `{run}`",
        f"- Fit bank: `{theta_dir / 'all_family_fits.csv'}`",
        f"- Leakage rows: `{len(leakage)}`",
        f"- Baseline availability problems: `{len(bad_availability)}`",
        "",
        "## Source Counts By Split Kind And Family",
        "",
        markdown_table(source_summary, max_rows=20),
        "",
        "## Leakage Rows",
        "",
        markdown_table(leakage, max_rows=20) if len(leakage) else "No leakage rows detected.",
        "",
        "## Availability Problems",
        "",
        markdown_table(bad_availability, max_rows=20) if len(bad_availability) else "No shape baselines were emitted without train-only source curves.",
        "",
        "## Integrity Checks",
        "",
        markdown_table(pd.DataFrame(checks), max_rows=20),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.run / "split_manifest.csv")
    baselines = read_csv(args.run / "baseline_per_curve_metrics.csv")
    fits = read_csv(args.theta_dir / "all_family_fits.csv")

    source_table, leakage = build_source_table(manifest, fits)
    availability = build_availability_table(baselines, source_table)
    if leakage.empty:
        leakage = pd.DataFrame(columns=["leak_type", "split_kind", "heldout_system", "family", "unified_curve_id"])

    checks = [
        finite_numeric_check("shape_source_by_split_family", source_table),
        finite_numeric_check("baseline_family_availability", availability),
        finite_numeric_check("leakage_rows", leakage),
    ]
    bad_numeric = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad_numeric:
        raise RuntimeError(f"NaN/inf detected: {bad_numeric}")

    bad_availability = availability[
        availability["baseline_without_source"].fillna(False) | availability["baseline_with_leaked_source"].fillna(False)
    ]
    if len(leakage) or len(bad_availability):
        raise RuntimeError(f"Shape-prior leakage audit failed: leaks={len(leakage)} availability={len(bad_availability)}")

    source_table.to_csv(args.out / "shape_source_by_split_family.csv", index=False)
    availability.to_csv(args.out / "baseline_family_availability.csv", index=False)
    leakage.to_csv(args.out / "leakage_rows.csv", index=False)
    metadata = {
        "script": "scripts/85_release_shape_prior_leakage_audit.py",
        "run": str(args.run),
        "theta_dir": str(args.theta_dir),
        "out": str(args.out),
        "leakage_rows": int(len(leakage)),
        "availability_problems": int(len(bad_availability)),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out, args.run, args.theta_dir, source_table, availability, leakage, checks)
    print(f"[85] wrote {args.out}")
    print(source_table.groupby(["split_kind", "family"])["n_shape_source_curves"].agg(["min", "median", "max"]).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
