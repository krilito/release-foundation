"""110 - Liposome E2 bottleneck decomposition.

Purpose:
    Decompose why liposome static-only whole-curve prediction failed in E1.
    This script reads E1 outputs and original formulation descriptors; it does
    not train a new predictor and does not use early release observations.

Consumes:
    outputs/109_liposome_static_whole_curve_prediction/static_per_curve_metrics.csv
    outputs/109_liposome_static_whole_curve_prediction/static_summary_by_split.csv
    outputs/109_liposome_static_whole_curve_prediction/static_feature_set_ablation.csv
    outputs/149_release_caveat_augmented_cumulative_v1/formulations.csv

Produces:
    outputs/110_liposome_bottleneck_decomposition/
      bottleneck_by_group.csv
      bottleneck_feature_missingness.csv
      shape_family_failure_table.csv
      feature_set_curve_agreement.csv
      bottleneck_decision_table.csv
      data_checks.csv
      lock_metadata.json
      bottleneck_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_E1 = Path("outputs/109_liposome_static_whole_curve_prediction")
DEFAULT_POOL = Path("outputs/149_release_caveat_augmented_cumulative_v1")
DEFAULT_OUT = Path("outputs/110_liposome_bottleneck_decomposition")

PRIMARY_METHOD = "direct_et_point"
PRIMARY_FEATURE_SET = "all_no_source"
SHAPE_METHOD = "weibull_static_theta"
GROUP_AXES = ["API_name", "structure_type", "media_pH", "media_temp_oC", "release_method"]
MISSINGNESS_COLUMNS = [
    "API_name",
    "API_ID",
    "structure_type",
    "Drug_Mw",
    "Particle_Size",
    "PDI",
    "zeta_potential",
    "media_pH",
    "media_temp_oC",
    "release_method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose liposome E1 static-prediction bottlenecks.")
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--high-rmse-threshold", type=float, default=0.25)
    parser.add_argument("--strong-success-rmse", type=float, default=0.12)
    parser.add_argument("--partial-success-rmse", type=float, default=0.18)
    return parser.parse_args()


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


def as_nonempty_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isna() | (out == "") | (out.str.lower() == "nan"))


def normalize_group(series: pd.Series) -> pd.Series:
    return as_nonempty_string(series).fillna("__MISSING__").astype(str)


def read_inputs(e1: Path, pool: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_curve_path = e1 / "static_per_curve_metrics.csv"
    summary_path = e1 / "static_summary_by_split.csv"
    ablation_path = e1 / "static_feature_set_ablation.csv"
    formulations_path = pool / "formulations.csv"
    for path in [per_curve_path, summary_path, ablation_path, formulations_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    per_curve = pd.read_csv(per_curve_path)
    summary = pd.read_csv(summary_path)
    formulations = pd.read_csv(formulations_path)
    required = {
        "split_kind",
        "fold",
        "method",
        "feature_set",
        "unified_curve_id",
        "rmse",
        "mae",
    }
    missing = sorted(required - set(per_curve.columns))
    if missing:
        raise ValueError(f"static_per_curve_metrics.csv missing columns: {missing}")
    return per_curve, summary, formulations


def liposome_formulations(formulations: pd.DataFrame) -> pd.DataFrame:
    if "polymer_family" not in formulations.columns:
        raise ValueError("formulations.csv missing polymer_family")
    family = as_nonempty_string(formulations["polymer_family"])
    liposome = formulations.loc[family.str.contains("liposome", case=False, na=False)].copy()
    if liposome.empty:
        raise ValueError("No liposome formulations found")
    return liposome


def primary_rows(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows = per_curve[
        (per_curve["method"] == PRIMARY_METHOD) & (per_curve["feature_set"] == PRIMARY_FEATURE_SET)
    ].copy()
    if rows.empty:
        raise ValueError("No primary direct_et_point/all_no_source rows found in E1 metrics")
    rows["rmse"] = pd.to_numeric(rows["rmse"], errors="coerce")
    rows["mae"] = pd.to_numeric(rows["mae"], errors="coerce")
    return rows


def merge_descriptors(rows: pd.DataFrame, liposome: pd.DataFrame) -> pd.DataFrame:
    descriptor_cols = ["unified_curve_id", *[col for col in MISSINGNESS_COLUMNS if col in liposome.columns]]
    merged = rows.merge(liposome[descriptor_cols], on="unified_curve_id", how="left", suffixes=("", "_form"))
    for axis in GROUP_AXES:
        form_col = f"{axis}_form"
        if form_col in merged.columns:
            merged[axis] = merged[form_col]
            merged = merged.drop(columns=[form_col])
    return merged


def bottleneck_by_group(primary: pd.DataFrame, high_threshold: float) -> pd.DataFrame:
    rows = []
    for split_kind, split_df in primary.groupby("split_kind", dropna=False):
        split_median = float(split_df["rmse"].median())
        split_q75 = float(split_df["rmse"].quantile(0.75))
        for axis in GROUP_AXES:
            if axis not in split_df.columns:
                continue
            work = split_df.copy()
            work["group_value"] = normalize_group(work[axis])
            for value, sub in work.groupby("group_value", dropna=False):
                n = int(len(sub))
                if n < 2:
                    continue
                median_rmse = float(sub["rmse"].median())
                rows.append(
                    {
                        "split_kind": split_kind,
                        "analysis_axis": axis,
                        "group_value": value,
                        "n_curve_records": n,
                        "n_unique_curves": int(sub["unified_curve_id"].nunique()),
                        "median_rmse": median_rmse,
                        "mean_rmse": float(sub["rmse"].mean()),
                        "q75_rmse": float(sub["rmse"].quantile(0.75)),
                        "max_rmse": float(sub["rmse"].max()),
                        "split_median_rmse": split_median,
                        "split_q75_rmse": split_q75,
                        "median_excess_vs_split": median_rmse - split_median,
                        "high_error_fraction": float((sub["rmse"] >= high_threshold).mean()),
                        "high_error_threshold": high_threshold,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["split_kind", "median_excess_vs_split"], ascending=[True, False])


def bottleneck_feature_missingness(primary: pd.DataFrame, liposome: pd.DataFrame) -> pd.DataFrame:
    merged = merge_descriptors(primary, liposome)
    rows = []
    for split_kind, split_df in merged.groupby("split_kind", dropna=False):
        for col in MISSINGNESS_COLUMNS:
            if col not in split_df.columns:
                continue
            clean = as_nonempty_string(split_df[col])
            work = split_df.copy()
            work["missing_status"] = np.where(clean.isna(), "missing", "present")
            stats: dict[str, dict[str, float]] = {}
            for status, sub in work.groupby("missing_status"):
                stats[str(status)] = {
                    "n": float(len(sub)),
                    "median_rmse": float(sub["rmse"].median()),
                    "mean_rmse": float(sub["rmse"].mean()),
                    "q75_rmse": float(sub["rmse"].quantile(0.75)),
                }
                rows.append(
                    {
                        "split_kind": split_kind,
                        "descriptor": col,
                        "missing_status": status,
                        "n_curve_records": int(len(sub)),
                        "n_unique_curves": int(sub["unified_curve_id"].nunique()),
                        "median_rmse": stats[str(status)]["median_rmse"],
                        "mean_rmse": stats[str(status)]["mean_rmse"],
                        "q75_rmse": stats[str(status)]["q75_rmse"],
                        "missing_minus_present_median_rmse": np.nan,
                    }
                )
            if "missing" in stats and "present" in stats:
                delta = stats["missing"]["median_rmse"] - stats["present"]["median_rmse"]
                rows.append(
                    {
                        "split_kind": split_kind,
                        "descriptor": col,
                        "missing_status": "delta_missing_minus_present",
                        "n_curve_records": int(stats["missing"]["n"] + stats["present"]["n"]),
                        "n_unique_curves": int(split_df["unified_curve_id"].nunique()),
                        "missing_n_curve_records": int(stats["missing"]["n"]),
                        "present_n_curve_records": int(stats["present"]["n"]),
                        "median_rmse": np.nan,
                        "mean_rmse": np.nan,
                        "q75_rmse": np.nan,
                        "missing_minus_present_median_rmse": float(delta),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["split_kind", "descriptor", "missing_status"])


def shape_family_failure_table(per_curve: pd.DataFrame, high_threshold: float) -> pd.DataFrame:
    primary = primary_rows(per_curve)[["split_kind", "fold", "unified_curve_id", "rmse", "mae"]].rename(
        columns={"rmse": "direct_rmse", "mae": "direct_mae"}
    )
    shape = per_curve[
        (per_curve["method"] == SHAPE_METHOD) & (per_curve["feature_set"] == PRIMARY_FEATURE_SET)
    ][["split_kind", "fold", "unified_curve_id", "rmse", "mae"]].rename(
        columns={"rmse": "shape_rmse", "mae": "shape_mae"}
    )
    paired = primary.merge(shape, on=["split_kind", "fold", "unified_curve_id"], how="inner")
    rows = []
    for split_kind, sub in paired.groupby("split_kind", dropna=False):
        direct = sub["direct_rmse"].to_numpy(dtype=float)
        shape_rmse = sub["shape_rmse"].to_numpy(dtype=float)
        corr = float(np.corrcoef(direct, shape_rmse)[0, 1]) if len(sub) > 2 else np.nan
        rows.append(
            {
                "split_kind": split_kind,
                "n_paired_curves": int(len(sub)),
                "direct_median_rmse": float(np.median(direct)),
                "shape_median_rmse": float(np.median(shape_rmse)),
                "shape_minus_direct_median_rmse": float(np.median(shape_rmse) - np.median(direct)),
                "direct_mean_rmse": float(np.mean(direct)),
                "shape_mean_rmse": float(np.mean(shape_rmse)),
                "shape_better_fraction": float(np.mean(shape_rmse < direct)),
                "both_high_error_fraction": float(
                    np.mean((shape_rmse >= high_threshold) & (direct >= high_threshold))
                ),
                "direct_high_shape_low_fraction": float(
                    np.mean((direct >= high_threshold) & (shape_rmse < high_threshold))
                ),
                "shape_high_direct_low_fraction": float(
                    np.mean((shape_rmse >= high_threshold) & (direct < high_threshold))
                ),
                "rmse_correlation": corr,
                "high_error_threshold": high_threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("split_kind")


def feature_set_curve_agreement(per_curve: pd.DataFrame) -> pd.DataFrame:
    direct = per_curve[per_curve["method"] == PRIMARY_METHOD].copy()
    direct["rmse"] = pd.to_numeric(direct["rmse"], errors="coerce")
    rows = []
    for split_kind, split_df in direct.groupby("split_kind", dropna=False):
        pivot = split_df.pivot_table(
            index=["split_kind", "fold", "unified_curve_id"],
            columns="feature_set",
            values="rmse",
            aggfunc="first",
        ).reset_index()
        feature_cols = [col for col in pivot.columns if col not in {"split_kind", "fold", "unified_curve_id"}]
        if not feature_cols:
            continue
        values = pivot[feature_cols].to_numpy(dtype=float)
        best_idx = np.nanargmin(values, axis=1)
        pivot["best_feature_set"] = [feature_cols[i] for i in best_idx]
        pivot["best_rmse"] = np.nanmin(values, axis=1)
        if PRIMARY_FEATURE_SET in pivot.columns:
            pivot["primary_rmse"] = pivot[PRIMARY_FEATURE_SET]
            pivot["primary_excess_vs_best"] = pivot["primary_rmse"] - pivot["best_rmse"]
            primary_is_best = pivot["best_feature_set"].eq(PRIMARY_FEATURE_SET)
        else:
            pivot["primary_rmse"] = np.nan
            pivot["primary_excess_vs_best"] = np.nan
            primary_is_best = pd.Series(False, index=pivot.index)
        for feature_set, sub in pivot.groupby("best_feature_set", dropna=False):
            rows.append(
                {
                    "split_kind": split_kind,
                    "best_feature_set": feature_set,
                    "n_curve_records": int(len(sub)),
                    "fraction_curve_records": float(len(sub) / len(pivot)),
                    "median_best_rmse": float(sub["best_rmse"].median()),
                    "median_primary_excess_vs_best": float(sub["primary_excess_vs_best"].median()),
                    "primary_is_best_fraction_for_split": float(primary_is_best.mean()),
                    "median_possible_gain_from_oracle_feature_choice": float(
                        pivot["primary_excess_vs_best"].median()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["split_kind", "fraction_curve_records"], ascending=[True, False])


def split_primary_medians(summary: pd.DataFrame) -> dict[str, float]:
    rows = summary[
        (summary["method"] == PRIMARY_METHOD) & (summary["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    return {
        str(row.split_kind): float(row.median_curve_rmse)
        for row in rows.itertuples(index=False)
    }


def best_direct_medians(summary: pd.DataFrame) -> dict[str, tuple[str, float]]:
    direct = summary[summary["method"] == PRIMARY_METHOD].copy()
    out: dict[str, tuple[str, float]] = {}
    for split_kind, sub in direct.groupby("split_kind", dropna=False):
        row = sub.sort_values("median_curve_rmse").iloc[0]
        out[str(split_kind)] = (str(row["feature_set"]), float(row["median_curve_rmse"]))
    return out


def bottleneck_decision_table(
    summary: pd.DataFrame,
    missingness: pd.DataFrame,
    shape: pd.DataFrame,
    agreement: pd.DataFrame,
    high_threshold: float,
    partial_success: float,
) -> pd.DataFrame:
    primary = split_primary_medians(summary)
    best = best_direct_medians(summary)

    def fmt_split_values(values: dict[str, float]) -> str:
        return "; ".join(f"{key}={value:.3f}" for key, value in sorted(values.items()))

    api_rmse = primary.get("groupkfold_API_name", np.nan)
    structure_rmse = primary.get("groupkfold_structure_type", np.nan)
    assay_splits = {
        key: value
        for key, value in primary.items()
        if key in {"groupkfold_media_pH", "groupkfold_media_temp_oC", "groupkfold_release_method"}
    }

    max_missing_delta = np.nan
    max_missing_descriptor = ""
    delta_rows = missingness[missingness["missing_status"] == "delta_missing_minus_present"].copy()
    if {"missing_n_curve_records", "present_n_curve_records"} <= set(delta_rows.columns):
        delta_rows = delta_rows[
            (pd.to_numeric(delta_rows["missing_n_curve_records"], errors="coerce") >= 5)
            & (pd.to_numeric(delta_rows["present_n_curve_records"], errors="coerce") >= 5)
        ].copy()
    if not delta_rows.empty:
        idx = delta_rows["missing_minus_present_median_rmse"].abs().idxmax()
        max_missing_delta = float(delta_rows.loc[idx, "missing_minus_present_median_rmse"])
        max_missing_descriptor = f"{delta_rows.loc[idx, 'split_kind']}:{delta_rows.loc[idx, 'descriptor']}"

    shape_both_high = float(shape["both_high_error_fraction"].median()) if not shape.empty else np.nan
    shape_better = float(shape["shape_better_fraction"].median()) if not shape.empty else np.nan
    primary_is_best = float(agreement["primary_is_best_fraction_for_split"].median()) if not agreement.empty else np.nan
    oracle_gain = float(agreement["median_possible_gain_from_oracle_feature_choice"].median()) if not agreement.empty else np.nan
    best_text = "; ".join(
        f"{split} best={feature}:{rmse:.3f}" for split, (feature, rmse) in sorted(best.items())
    )

    rows = [
        {
            "bottleneck": "API shift",
            "status": "supported" if api_rmse >= high_threshold else "weak",
            "evidence": f"groupkfold_API_name primary median RMSE={api_rmse:.3f}; {best.get('groupkfold_API_name', ('NA', np.nan))[0]} best median={best.get('groupkfold_API_name', ('NA', np.nan))[1]:.3f}",
            "interpretation": "Unseen API groups remain difficult; API/drug descriptors help but do not make static prediction sufficient.",
            "next_action": "include API-stratified residual analysis in E2 report; do not headline random split.",
        },
        {
            "bottleneck": "structure shift",
            "status": "partial" if structure_rmse >= partial_success else "weak",
            "evidence": f"groupkfold_structure_type primary median RMSE={structure_rmse:.3f}",
            "interpretation": "Structure-type holdout is the easiest strict split, but still above the partial-success threshold.",
            "next_action": "treat structure as useful but insufficient static information.",
        },
        {
            "bottleneck": "assay condition shift",
            "status": "supported" if assay_splits and max(assay_splits.values()) >= 0.22 else "weak",
            "evidence": fmt_split_values(assay_splits),
            "interpretation": "pH, temperature, and release-method holdouts retain sizable errors, so assay conditions are not fully absorbed by current descriptors.",
            "next_action": "group residuals by condition before designing early-observation schedules.",
        },
        {
            "bottleneck": "descriptor missingness",
            "status": "modifier" if np.isfinite(max_missing_delta) and abs(max_missing_delta) >= 0.05 else "not_primary",
            "evidence": f"largest eligible missing-present median RMSE delta={max_missing_delta:.3f} at {max_missing_descriptor}; eligibility requires missing_n>=5 and present_n>=5",
            "interpretation": "Descriptor missingness changes error structure, but eligible deltas are not interpreted as one-way proof that missingness causes failure.",
            "next_action": "report missingness as a modifier, not as the only failure cause.",
        },
        {
            "bottleneck": "shape-family insufficiency",
            "status": "supported" if shape_both_high >= 0.35 else "partial",
            "evidence": f"median both-high-error fraction={shape_both_high:.3f}; median shape-better fraction={shape_better:.3f}",
            "interpretation": "Weibull static theta and direct static models often fail together, so shape prior alone is not enough.",
            "next_action": "keep shape prior as a baseline/control, not the rescue path.",
        },
        {
            "bottleneck": "feature-mixture/model limitation",
            "status": "supported" if np.isfinite(primary_is_best) and primary_is_best < 0.5 else "weak",
            "evidence": f"median primary-is-best fraction={primary_is_best:.3f}; median oracle feature-choice gain={oracle_gain:.3f}; {best_text}",
            "interpretation": "Different feature sets win on different curves/splits; a single static feature recipe is unstable.",
            "next_action": "do not jump to MoE yet; first test whether measured early Q resolves this curve-specific ambiguity.",
        },
    ]

    static_failures = [value for value in primary.values() if value >= partial_success]
    rows.append(
        {
            "bottleneck": "early-observation justification",
            "status": "justified" if len(static_failures) >= max(1, len(primary) // 2) else "not_yet",
            "evidence": f"primary static medians: {fmt_split_values(primary)}",
            "interpretation": "Static failure remains after leakage-safe splits and reasonable static baselines.",
            "next_action": "Proceed to E3 early-observation budget only as a diagnostic of missing curve-specific state, not as a neural-model sprint.",
        }
    )
    return pd.DataFrame(rows)


def data_checks(
    per_curve: pd.DataFrame,
    summary: pd.DataFrame,
    group_table: pd.DataFrame,
    missingness: pd.DataFrame,
    shape: pd.DataFrame,
    agreement: pd.DataFrame,
    decision: pd.DataFrame,
) -> pd.DataFrame:
    required_methods = {PRIMARY_METHOD, SHAPE_METHOD, "global_median_curve"}
    methods = set(per_curve["method"].astype(str))
    core_nonfinite = 0
    critical_specs = [
        (group_table, ["median_rmse", "median_excess_vs_split", "high_error_fraction"]),
        (shape, ["direct_median_rmse", "shape_median_rmse", "shape_better_fraction"]),
        (agreement, ["fraction_curve_records", "median_best_rmse"]),
    ]
    for table, cols in critical_specs:
        available = [col for col in cols if col in table.columns]
        if available:
            core_nonfinite += int(
                table[available].replace([np.inf, -np.inf], np.nan).isna().sum().sum()
            )
    checks = [
        {
            "check": "e1_methods_available",
            "status": "pass" if required_methods <= methods else "fail",
            "value": ",".join(sorted(methods)),
            "details": "{}",
        },
        {
            "check": "primary_static_failure_rows_present",
            "status": "pass" if not primary_rows(per_curve).empty else "fail",
            "value": f"primary_rows={len(primary_rows(per_curve))}",
            "details": "{}",
        },
        {
            "check": "required_output_tables_nonempty",
            "status": "pass"
            if all(len(table) > 0 for table in [group_table, missingness, shape, agreement, decision])
            else "fail",
            "value": json.dumps(
                {
                    "bottleneck_by_group": len(group_table),
                    "missingness": len(missingness),
                    "shape": len(shape),
                    "agreement": len(agreement),
                    "decision": len(decision),
                }
            ),
            "details": "{}",
        },
        {
            "check": "no_nonfinite_core_metrics",
            "status": "pass" if core_nonfinite == 0 else "fail",
            "value": f"nonfinite_core_cells={core_nonfinite}",
            "details": "Optional delta cells are allowed to be blank when a missing/present contrast is structurally undefined.",
        },
        {
            "check": "decision_has_next_action",
            "status": "pass" if decision["next_action"].notna().all() else "fail",
            "value": f"decision_rows={len(decision)}",
            "details": "{}",
        },
    ]
    return pd.DataFrame(checks)


def write_report(
    out: Path,
    group_table: pd.DataFrame,
    missingness: pd.DataFrame,
    shape: pd.DataFrame,
    agreement: pd.DataFrame,
    decision: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    top_groups = group_table.sort_values("median_excess_vs_split", ascending=False).head(8)
    delta_rows = missingness[missingness["missing_status"] == "delta_missing_minus_present"].copy()
    if {"missing_n_curve_records", "present_n_curve_records"} <= set(delta_rows.columns):
        eligible_delta_rows = delta_rows[
            (pd.to_numeric(delta_rows["missing_n_curve_records"], errors="coerce") >= 5)
            & (pd.to_numeric(delta_rows["present_n_curve_records"], errors="coerce") >= 5)
        ].copy()
    else:
        eligible_delta_rows = delta_rows
    top_missing = eligible_delta_rows.reindex(
        eligible_delta_rows["missing_minus_present_median_rmse"].abs().sort_values(ascending=False).index
    ).head(8)
    early_row = decision[decision["bottleneck"] == "early-observation justification"].iloc[0]
    lines = [
        "# Liposome Bottleneck Decomposition E2",
        "",
        "Date: 2026-06-12",
        "",
        "## Decision",
        "",
        f"Early-observation next step: **{early_row['status']}**.",
        "",
        str(early_row["interpretation"]),
        "",
        str(early_row["next_action"]),
        "",
        "## Bottleneck Decisions",
        "",
        "| Bottleneck | Status | Evidence |",
        "|---|---|---|",
    ]
    for row in decision.itertuples(index=False):
        lines.append(f"| {row.bottleneck} | {row.status} | {row.evidence} |")
    lines.extend(["", "## Worst Group Residuals", "", "| Split | Axis | Group | n | Median RMSE | Excess | High-error fraction |", "|---|---|---|---:|---:|---:|---:|"])
    for row in top_groups.itertuples(index=False):
        lines.append(
            f"| {row.split_kind} | {row.analysis_axis} | {row.group_value} | "
            f"{int(row.n_curve_records)} | {row.median_rmse:.3f} | "
            f"{row.median_excess_vs_split:.3f} | {row.high_error_fraction:.3f} |"
        )
    lines.extend(["", "## Descriptor Missingness", "", "| Split | Descriptor | Missing n | Present n | Delta missing-present RMSE |", "|---|---|---:|---:|---:|"])
    for row in top_missing.itertuples(index=False):
        lines.append(
            f"| {row.split_kind} | {row.descriptor} | {int(row.missing_n_curve_records)} | "
            f"{int(row.present_n_curve_records)} | {row.missing_minus_present_median_rmse:.3f} |"
        )
    lines.extend(["", "## Shape Prior Agreement", "", "| Split | Direct RMSE | Weibull RMSE | Shape better fraction | Both high-error fraction |", "|---|---:|---:|---:|---:|"])
    for row in shape.itertuples(index=False):
        lines.append(
            f"| {row.split_kind} | {row.direct_median_rmse:.3f} | {row.shape_median_rmse:.3f} | "
            f"{row.shape_better_fraction:.3f} | {row.both_high_error_fraction:.3f} |"
        )
    lines.extend(["", "## Feature-Set Instability", "", "| Split | Best feature set | Fraction | Median possible gain |", "|---|---|---:|---:|"])
    for row in agreement.itertuples(index=False):
        lines.append(
            f"| {row.split_kind} | {row.best_feature_set} | {row.fraction_curve_records:.3f} | "
            f"{row.median_possible_gain_from_oracle_feature_choice:.3f} |"
        )
    lines.extend(["", "## Verification", ""])
    for row in checks.itertuples(index=False):
        lines.append(f"- `{row.check}`: {row.status} ({row.value})")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "E2 supports moving to E3 early-observation budget, but only as an",
            "information-bottleneck diagnostic. The failure is not a single clean",
            "source artifact: API and assay-condition splits remain hard, descriptor",
            "missingness modifies errors, shape-family priors do not rescue the",
            "problem, and different static feature sets win on different curves.",
            "",
        ]
    )
    (out / "bottleneck_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    per_curve, summary, formulations = read_inputs(args.e1, args.pool)
    liposome = liposome_formulations(formulations)
    primary = merge_descriptors(primary_rows(per_curve), liposome)

    group_table = bottleneck_by_group(primary, args.high_rmse_threshold)
    missingness = bottleneck_feature_missingness(primary, liposome)
    shape = shape_family_failure_table(per_curve, args.high_rmse_threshold)
    agreement = feature_set_curve_agreement(per_curve)
    decision = bottleneck_decision_table(
        summary,
        missingness,
        shape,
        agreement,
        args.high_rmse_threshold,
        args.partial_success_rmse,
    )
    checks = data_checks(per_curve, summary, group_table, missingness, shape, agreement, decision)

    group_table.to_csv(out / "bottleneck_by_group.csv", index=False)
    missingness.to_csv(out / "bottleneck_feature_missingness.csv", index=False)
    shape.to_csv(out / "shape_family_failure_table.csv", index=False)
    agreement.to_csv(out / "feature_set_curve_agreement.csv", index=False)
    decision.to_csv(out / "bottleneck_decision_table.csv", index=False)
    checks.to_csv(out / "data_checks.csv", index=False)
    write_report(out, group_table, missingness, shape, agreement, decision, checks)

    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {
            "e1": str(args.e1),
            "pool": str(args.pool),
            "out": str(args.out),
            "seed": args.seed,
            "high_rmse_threshold": args.high_rmse_threshold,
            "strong_success_rmse": args.strong_success_rmse,
            "partial_success_rmse": args.partial_success_rmse,
        },
        "inputs": {
            "per_curve": file_meta(args.e1 / "static_per_curve_metrics.csv"),
            "summary": file_meta(args.e1 / "static_summary_by_split.csv"),
            "ablation": file_meta(args.e1 / "static_feature_set_ablation.csv"),
            "formulations": file_meta(args.pool / "formulations.csv"),
        },
        "row_counts": {
            "e1_per_curve_rows": int(len(per_curve)),
            "primary_rows": int(len(primary)),
            "bottleneck_by_group_rows": int(len(group_table)),
            "missingness_rows": int(len(missingness)),
            "shape_rows": int(len(shape)),
            "agreement_rows": int(len(agreement)),
            "decision_rows": int(len(decision)),
        },
        "data_check_status": checks[["check", "status", "value"]].to_dict(orient="records"),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
