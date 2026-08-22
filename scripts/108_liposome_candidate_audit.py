"""108 - Liposome candidate audit for second-system release forecasting.

Purpose:
    Start the second drug-release system without model drift. This script
    audits whether the liposome IVR corpus is suitable for the staged
    static-first -> bottleneck -> early-observation route defined in
    docs/liposome_observation_budget_goal_design_2026-06-12.md.

Consumes:
    outputs/149_release_caveat_augmented_cumulative_v1/curves_long.csv
    outputs/149_release_caveat_augmented_cumulative_v1/formulations.csv

Produces:
    outputs/108_liposome_candidate_audit/
      dataset_completeness.csv
      descriptor_completeness.csv
      split_candidate_summary.csv
      timepoint_coverage.csv
      candidate_decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This is an E0 gate, not a prediction experiment. It decides whether
    liposome is suitable for the next static whole-curve prediction probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_POOL = Path("outputs/149_release_caveat_augmented_cumulative_v1")
DEFAULT_OUT = Path("outputs/108_liposome_candidate_audit")

DESCRIPTOR_COLUMNS = [
    "source_dataset",
    "source_curve_id",
    "API_name",
    "API_ID",
    "payload_name",
    "structure_type",
    "Drug_Mw",
    "Particle_Size",
    "PDI",
    "zeta_potential",
    "media_pH",
    "media_temp_oC",
    "release_method",
    "measurement_assay",
    "release_medium_condition",
    "curve_level",
    "normalization_basis",
]

SPLIT_AXES = [
    "API_name",
    "API_ID",
    "structure_type",
    "media_pH",
    "media_temp_oC",
    "release_method",
    "source_dataset",
]

EARLY_BUDGETS = [0, 1, 2, 3, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit liposome corpus suitability.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-timepoints", type=int, default=6)
    parser.add_argument("--min-curves", type=int, default=100)
    parser.add_argument("--min-heldout-curves", type=int, default=5)
    parser.add_argument("--min-train-curves", type=int, default=30)
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


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_inputs(pool: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves_path = pool / "curves_long.csv"
    formulations_path = pool / "formulations.csv"
    if not curves_path.exists():
        raise FileNotFoundError(curves_path)
    if not formulations_path.exists():
        raise FileNotFoundError(formulations_path)
    curves = pd.read_csv(curves_path)
    formulations = pd.read_csv(formulations_path)
    required_curves = {"unified_curve_id", "time_days", "release_fraction", "source_dataset"}
    required_formulations = {"unified_curve_id", "polymer_family", "source_dataset"}
    missing_curves = sorted(required_curves - set(curves.columns))
    missing_formulations = sorted(required_formulations - set(formulations.columns))
    if missing_curves:
        raise ValueError(f"curves_long.csv missing required columns: {missing_curves}")
    if missing_formulations:
        raise ValueError(f"formulations.csv missing required columns: {missing_formulations}")
    return curves, formulations


def as_nonempty_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isna() | (out == "") | (out.str.lower() == "nan"))


def value_count_preview(series: pd.Series, n: int = 8) -> str:
    clean = as_nonempty_string(series)
    counts = clean.value_counts(dropna=True).head(n)
    if counts.empty:
        return ""
    return "; ".join(f"{idx}:{int(value)}" for idx, value in counts.items())


def select_liposome(formulations: pd.DataFrame) -> pd.DataFrame:
    family = as_nonempty_string(formulations["polymer_family"])
    mask = family.str.contains("liposome", case=False, na=False)
    liposome = formulations.loc[mask].copy()
    if liposome.empty:
        raise ValueError("No liposome rows found in formulations.csv")
    return liposome


def clean_liposome_curves(curves: pd.DataFrame, liposome_formulations: pd.DataFrame) -> pd.DataFrame:
    ids = set(liposome_formulations["unified_curve_id"].astype(str))
    liposome_curves = curves[curves["unified_curve_id"].astype(str).isin(ids)].copy()
    liposome_curves["time_days_num"] = pd.to_numeric(liposome_curves["time_days"], errors="coerce")
    liposome_curves["release_fraction_num"] = pd.to_numeric(
        liposome_curves["release_fraction"], errors="coerce"
    )
    liposome_curves["valid_metric_row"] = (
        np.isfinite(liposome_curves["time_days_num"])
        & np.isfinite(liposome_curves["release_fraction_num"])
    )
    return liposome_curves


def build_timepoint_coverage(
    liposome_curves: pd.DataFrame,
    liposome_formulations: pd.DataFrame,
    min_timepoints: int,
) -> pd.DataFrame:
    valid = liposome_curves[liposome_curves["valid_metric_row"]].copy()
    grouped = valid.groupby("unified_curve_id", dropna=False)
    coverage = grouped.agg(
        n_rows=("record_id", "size") if "record_id" in valid.columns else ("time_days", "size"),
        n_valid_timepoints=("time_days_num", "nunique"),
        min_time_days=("time_days_num", "min"),
        max_time_days=("time_days_num", "max"),
        min_release_fraction_raw=("release_fraction", "min"),
        max_release_fraction_raw=("release_fraction", "max"),
        min_release_fraction_num=("release_fraction_num", "min"),
        max_release_fraction_num=("release_fraction_num", "max"),
        n_negative_release=("release_fraction_num", lambda s: int((s < 0).sum())),
        n_release_gt1=("release_fraction_num", lambda s: int((s > 1).sum())),
        n_release_gt12=("release_fraction_num", lambda s: int((s > 1.2).sum())),
    ).reset_index()
    coverage["passes_min_timepoints"] = coverage["n_valid_timepoints"] >= min_timepoints
    for budget in EARLY_BUDGETS:
        coverage[f"has_budget_k{budget}"] = coverage["n_valid_timepoints"] > budget
    keep_cols = [
        "unified_curve_id",
        "source_dataset",
        "API_name",
        "API_ID",
        "structure_type",
        "media_pH",
        "media_temp_oC",
        "release_method",
    ]
    available = [col for col in keep_cols if col in liposome_formulations.columns]
    return coverage.merge(liposome_formulations[available], on="unified_curve_id", how="left")


def dataset_completeness(
    formulations: pd.DataFrame,
    liposome_formulations: pd.DataFrame,
    liposome_curves: pd.DataFrame,
    coverage: pd.DataFrame,
    min_timepoints: int,
) -> pd.DataFrame:
    valid_rows = int(liposome_curves["valid_metric_row"].sum())
    total_rows = int(len(liposome_curves))
    release = liposome_curves.loc[liposome_curves["valid_metric_row"], "release_fraction_num"]
    time = liposome_curves.loc[liposome_curves["valid_metric_row"], "time_days_num"]
    rows = [
        ("total_formulation_rows", len(formulations), "all systems in formulations.csv"),
        ("liposome_formulation_rows", len(liposome_formulations), "liposome rows before timepoint gate"),
        ("liposome_curve_ids_in_formulations", liposome_formulations["unified_curve_id"].nunique(), ""),
        ("liposome_curve_ids_in_curves_long", liposome_curves["unified_curve_id"].nunique(), ""),
        ("liposome_curve_rows", total_rows, "raw curve rows for selected IDs"),
        ("valid_metric_rows", valid_rows, "finite time_days and release_fraction rows"),
        (
            "valid_metric_row_fraction",
            valid_rows / total_rows if total_rows else np.nan,
            "fraction used by audit metrics",
        ),
        ("curves_passing_min_timepoints", int(coverage["passes_min_timepoints"].sum()), ""),
        (
            "fraction_curves_passing_min_timepoints",
            float(coverage["passes_min_timepoints"].mean()) if len(coverage) else np.nan,
            f"min_timepoints={min_timepoints}",
        ),
        ("median_valid_timepoints", float(coverage["n_valid_timepoints"].median()), ""),
        ("min_valid_timepoints", int(coverage["n_valid_timepoints"].min()), ""),
        ("max_valid_timepoints", int(coverage["n_valid_timepoints"].max()), ""),
        ("median_max_time_days", float(coverage["max_time_days"].median()), ""),
        ("min_max_time_days", float(coverage["max_time_days"].min()), ""),
        ("max_max_time_days", float(coverage["max_time_days"].max()), ""),
        ("min_release_fraction", float(release.min()), "raw value, not clipped"),
        ("max_release_fraction", float(release.max()), "raw value, not clipped"),
        ("rows_release_lt0", int((release < 0).sum()), ""),
        ("rows_release_gt1", int((release > 1).sum()), ""),
        ("rows_release_gt12", int((release > 1.2).sum()), ""),
        ("min_time_days", float(time.min()), ""),
        ("max_time_days", float(time.max()), ""),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "notes"])


def descriptor_completeness(liposome_formulations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(liposome_formulations)
    for col in DESCRIPTOR_COLUMNS:
        if col not in liposome_formulations.columns:
            rows.append(
                {
                    "column": col,
                    "exists": False,
                    "nonmissing_count": 0,
                    "nonmissing_fraction": 0.0,
                    "unique_nonmissing": 0,
                    "top_values": "",
                    "numeric_min": np.nan,
                    "numeric_median": np.nan,
                    "numeric_max": np.nan,
                }
            )
            continue
        raw = liposome_formulations[col]
        clean = as_nonempty_string(raw)
        numeric = pd.to_numeric(raw, errors="coerce")
        nonmissing = clean.notna()
        rows.append(
            {
                "column": col,
                "exists": True,
                "nonmissing_count": int(nonmissing.sum()),
                "nonmissing_fraction": float(nonmissing.mean()) if n else np.nan,
                "unique_nonmissing": int(clean.dropna().nunique()),
                "top_values": value_count_preview(raw),
                "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "numeric_median": float(numeric.median()) if numeric.notna().any() else np.nan,
                "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def split_candidate_summary(
    liposome_formulations: pd.DataFrame,
    min_heldout_curves: int,
    min_train_curves: int,
) -> pd.DataFrame:
    rows = []
    n_total = liposome_formulations["unified_curve_id"].nunique()
    for axis in SPLIT_AXES:
        if axis not in liposome_formulations.columns:
            rows.append(
                {
                    "split_axis": axis,
                    "group_value": "__COLUMN_MISSING__",
                    "heldout_curves": 0,
                    "train_curves": n_total,
                    "n_groups": 0,
                    "n_feasible_groups_on_axis": 0,
                    "axis_feasible": False,
                    "group_feasible": False,
                    "notes": "column missing",
                }
            )
            continue
        values = as_nonempty_string(liposome_formulations[axis]).fillna("__MISSING__")
        group_counts = values.value_counts(dropna=False).sort_values(ascending=False)
        feasible_count = int(
            ((group_counts >= min_heldout_curves) & ((n_total - group_counts) >= min_train_curves)).sum()
        )
        axis_feasible = feasible_count >= 2 or (axis == "source_dataset" and feasible_count >= 1)
        for group_value, heldout in group_counts.items():
            train = int(n_total - int(heldout))
            group_feasible = int(heldout) >= min_heldout_curves and train >= min_train_curves
            rows.append(
                {
                    "split_axis": axis,
                    "group_value": str(group_value),
                    "heldout_curves": int(heldout),
                    "train_curves": train,
                    "n_groups": int(len(group_counts)),
                    "n_feasible_groups_on_axis": feasible_count,
                    "axis_feasible": bool(axis_feasible),
                    "group_feasible": bool(group_feasible),
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def data_checks(
    liposome_formulations: pd.DataFrame,
    liposome_curves: pd.DataFrame,
    coverage: pd.DataFrame,
    split_summary: pd.DataFrame,
    min_timepoints: int,
) -> pd.DataFrame:
    n_formula_ids = liposome_formulations["unified_curve_id"].nunique()
    n_curve_ids = liposome_curves["unified_curve_id"].nunique()
    checks = [
        {
            "check": "liposome_ids_have_curve_rows",
            "status": "pass" if n_formula_ids == n_curve_ids else "fail",
            "value": f"formulations={n_formula_ids}; curves_long={n_curve_ids}",
            "unexpected_nonfinite": "{}",
        },
        {
            "check": "finite_time_and_release_rows",
            "status": "pass" if liposome_curves["valid_metric_row"].all() else "warn",
            "value": f"{int(liposome_curves['valid_metric_row'].sum())}/{len(liposome_curves)}",
            "unexpected_nonfinite": json.dumps(
                {
                    "bad_rows": int((~liposome_curves["valid_metric_row"]).sum()),
                }
            ),
        },
        {
            "check": "all_analyzed_curves_pass_min_timepoints",
            "status": "pass" if coverage["passes_min_timepoints"].all() else "fail",
            "value": f"min_timepoints={min_timepoints}; pass={int(coverage['passes_min_timepoints'].sum())}/{len(coverage)}",
            "unexpected_nonfinite": "{}",
        },
        {
            "check": "raw_release_fraction_preserved",
            "status": "pass",
            "value": "script audits raw release_fraction and does not clip model inputs",
            "unexpected_nonfinite": "{}",
        },
        {
            "check": "at_least_one_strict_split_axis_feasible",
            "status": "pass" if split_summary["axis_feasible"].any() else "fail",
            "value": ",".join(sorted(split_summary.loc[split_summary["axis_feasible"], "split_axis"].unique())),
            "unexpected_nonfinite": "{}",
        },
    ]
    return pd.DataFrame(checks)


def candidate_decision_table(
    dataset: pd.DataFrame,
    descriptors: pd.DataFrame,
    split_summary: pd.DataFrame,
    min_curves: int,
) -> pd.DataFrame:
    metrics = dict(zip(dataset["metric"], dataset["value"], strict=False))
    descriptor_by_col = descriptors.set_index("column") if not descriptors.empty else pd.DataFrame()

    def frac(col: str) -> float:
        if descriptor_by_col.empty or col not in descriptor_by_col.index:
            return 0.0
        return float(descriptor_by_col.loc[col, "nonmissing_fraction"])

    strict_axes = sorted(
        axis
        for axis in split_summary.loc[split_summary["axis_feasible"], "split_axis"].unique()
        if axis != "source_dataset"
    )
    source_groups = split_summary[split_summary["split_axis"] == "source_dataset"]["n_groups"].max()
    source_groups = int(source_groups) if pd.notna(source_groups) else 0
    rows = [
        {
            "criterion": "enough_liposome_curves",
            "status": "pass" if float(metrics["liposome_curve_ids_in_curves_long"]) >= min_curves else "fail",
            "evidence": f"n_curves={int(float(metrics['liposome_curve_ids_in_curves_long']))}; threshold={min_curves}",
            "next_action": "proceed" if float(metrics["liposome_curve_ids_in_curves_long"]) >= min_curves else "choose another system",
        },
        {
            "criterion": "timepoint_budget_available",
            "status": "pass" if float(metrics["fraction_curves_passing_min_timepoints"]) >= 0.9 else "fail",
            "evidence": f"fraction_pass={float(metrics['fraction_curves_passing_min_timepoints']):.3f}",
            "next_action": "use k=0/1/2/3/5 budgets",
        },
        {
            "criterion": "strict_split_available",
            "status": "pass" if strict_axes else "fail",
            "evidence": f"feasible_axes={';'.join(strict_axes) if strict_axes else 'none'}",
            "next_action": "headline splits must use feasible non-source axes",
        },
        {
            "criterion": "api_descriptor_available",
            "status": "pass" if max(frac("API_name"), frac("API_ID")) >= 0.9 else "warn",
            "evidence": f"API_name={frac('API_name'):.3f}; API_ID={frac('API_ID'):.3f}",
            "next_action": "API-grouped split is suitable if group sizes pass",
        },
        {
            "criterion": "structure_descriptor_available",
            "status": "pass" if frac("structure_type") >= 0.7 else "warn",
            "evidence": f"structure_type={frac('structure_type'):.3f}",
            "next_action": "structure split can be a secondary strict split",
        },
        {
            "criterion": "medium_descriptors_partial",
            "status": "pass" if min(frac("media_pH"), frac("media_temp_oC")) >= 0.7 else "warn",
            "evidence": f"media_pH={frac('media_pH'):.3f}; media_temp_oC={frac('media_temp_oC'):.3f}",
            "next_action": "use pH/temp as covariates and sensitivity split if feasible",
        },
        {
            "criterion": "source_split_available",
            "status": "pass" if source_groups > 1 else "warn",
            "evidence": f"source_dataset_groups={source_groups}",
            "next_action": "do not use source split as headline if only one source exists",
        },
    ]
    blocking = any(row["status"] == "fail" for row in rows[:3])
    rows.append(
        {
            "criterion": "overall_e0_decision",
            "status": "proceed" if not blocking else "reject",
            "evidence": "core gates passed" if not blocking else "one or more core gates failed",
            "next_action": "build E1 static whole-curve prediction" if not blocking else "select a different second system",
        }
    )
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    dataset: pd.DataFrame,
    descriptors: pd.DataFrame,
    split_summary: pd.DataFrame,
    decision: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    metrics = dict(zip(dataset["metric"], dataset["value"], strict=False))
    descriptor_by_col = descriptors.set_index("column")

    def nonmissing(col: str) -> str:
        if col not in descriptor_by_col.index:
            return "missing"
        row = descriptor_by_col.loc[col]
        return f"{int(row['nonmissing_count'])}/{int(float(metrics['liposome_formulation_rows']))} ({float(row['nonmissing_fraction']):.1%})"

    feasible_axes = sorted(split_summary.loc[split_summary["axis_feasible"], "split_axis"].unique())
    strict_axes = [axis for axis in feasible_axes if axis != "source_dataset"]
    overall = decision[decision["criterion"] == "overall_e0_decision"].iloc[0]
    lines = [
        "# Liposome Candidate Audit E0",
        "",
        "Date: 2026-06-12",
        "",
        "## Decision",
        "",
        f"Overall E0 decision: **{overall['status']}**.",
        "",
        str(overall["next_action"]),
        "",
        "## Corpus",
        "",
        f"- Liposome curves in formulations: `{int(float(metrics['liposome_formulation_rows']))}`.",
        f"- Liposome curve IDs with curve rows: `{int(float(metrics['liposome_curve_ids_in_curves_long']))}`.",
        f"- Valid curve rows: `{int(float(metrics['valid_metric_rows']))}` / `{int(float(metrics['liposome_curve_rows']))}`.",
        f"- Curves passing minimum timepoints: `{int(float(metrics['curves_passing_min_timepoints']))}`.",
        f"- Median valid timepoints per curve: `{float(metrics['median_valid_timepoints']):.1f}`.",
        f"- Median max time: `{float(metrics['median_max_time_days']):.3f}` days.",
        f"- Raw release fraction range: `{float(metrics['min_release_fraction']):.3f}` to `{float(metrics['max_release_fraction']):.3f}`.",
        "",
        "## Descriptor Completeness",
        "",
        f"- API name: {nonmissing('API_name')}.",
        f"- API ID: {nonmissing('API_ID')}.",
        f"- Structure type: {nonmissing('structure_type')}.",
        f"- Drug molecular weight: {nonmissing('Drug_Mw')}.",
        f"- Particle size: {nonmissing('Particle_Size')}.",
        f"- PDI: {nonmissing('PDI')}.",
        f"- Zeta potential: {nonmissing('zeta_potential')}.",
        f"- pH: {nonmissing('media_pH')}.",
        f"- Temperature: {nonmissing('media_temp_oC')}.",
        f"- Release method: {nonmissing('release_method')}.",
        "",
        "## Split Feasibility",
        "",
        f"Feasible split axes: `{', '.join(feasible_axes) if feasible_axes else 'none'}`.",
        "",
        "Headline evidence should use non-source strict axes: "
        f"`{', '.join(strict_axes) if strict_axes else 'none'}`.",
        "",
        "Do not use random splits or a single-source split as the headline result.",
        "",
        "## Verification",
        "",
    ]
    for _, row in checks.iterrows():
        lines.append(f"- `{row['check']}`: {row['status']} ({row['value']})")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Build `scripts/109_liposome_static_whole_curve_prediction.py` only if the E0",
            "decision is `proceed`. E1 must test static descriptors before any early",
            "observation or neural-model work.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.default_rng(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    curves, formulations = read_inputs(args.pool)
    liposome_formulations = select_liposome(formulations)
    liposome_curves = clean_liposome_curves(curves, liposome_formulations)
    coverage = build_timepoint_coverage(liposome_curves, liposome_formulations, args.min_timepoints)
    dataset = dataset_completeness(
        formulations,
        liposome_formulations,
        liposome_curves,
        coverage,
        args.min_timepoints,
    )
    descriptors = descriptor_completeness(liposome_formulations)
    split_summary = split_candidate_summary(
        liposome_formulations,
        args.min_heldout_curves,
        args.min_train_curves,
    )
    checks = data_checks(
        liposome_formulations,
        liposome_curves,
        coverage,
        split_summary,
        args.min_timepoints,
    )
    decision = candidate_decision_table(dataset, descriptors, split_summary, args.min_curves)

    dataset.to_csv(out / "dataset_completeness.csv", index=False)
    descriptors.to_csv(out / "descriptor_completeness.csv", index=False)
    split_summary.to_csv(out / "split_candidate_summary.csv", index=False)
    coverage.to_csv(out / "timepoint_coverage.csv", index=False)
    decision.to_csv(out / "candidate_decision_table.csv", index=False)
    checks.to_csv(out / "data_checks.csv", index=False)
    write_report(out, dataset, descriptors, split_summary, decision, checks)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {
            "pool": str(args.pool),
            "out": str(args.out),
            "seed": args.seed,
            "min_timepoints": args.min_timepoints,
            "min_curves": args.min_curves,
            "min_heldout_curves": args.min_heldout_curves,
            "min_train_curves": args.min_train_curves,
        },
        "inputs": {
            "curves_long": file_meta(args.pool / "curves_long.csv"),
            "formulations": file_meta(args.pool / "formulations.csv"),
        },
        "row_counts": {
            "curves_long_rows": int(len(curves)),
            "formulations_rows": int(len(formulations)),
            "liposome_curve_rows": int(len(liposome_curves)),
            "liposome_formulation_rows": int(len(liposome_formulations)),
            "liposome_curve_ids": int(liposome_formulations["unified_curve_id"].nunique()),
        },
        "data_check_status": checks[["check", "status", "value"]].to_dict(orient="records"),
        "overall_decision": decision[decision["criterion"] == "overall_e0_decision"].iloc[0].to_dict(),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
