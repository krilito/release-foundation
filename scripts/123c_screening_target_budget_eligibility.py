"""
123C - Screening target and observation-budget eligibility audit helpers.

This module defines target legality, per-curve target computability, budgeted
future-window eligibility, and source/schedule confounding audits without
training any predictive model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_SCHEMA_COLUMNS = [
    "target_id",
    "target_name",
    "target_family",
    "target_type",
    "definition",
    "formula_or_rule",
    "required_curve_region",
    "required_min_time",
    "required_max_time",
    "minimum_timepoints",
    "thresholds",
    "threshold_source",
    "system_scope",
    "application_scope",
    "uses_full_curve",
    "derived_from_future",
    "allowed_as_model_input",
    "allowed_as_training_target",
    "allowed_for_pre-observation_decision",
    "cross_system_comparable",
    "normalization_rule",
    "censoring_rule",
    "missing_rule",
    "ambiguity_note",
    "status",
]

TARGET_ELIGIBILITY_COLUMNS = [
    "formulation_id",
    "curve_id",
    "source_dataset",
    "source_id",
    "doi",
    "system",
    "target_id",
    "target_value",
    "target_computable",
    "eligibility_status",
    "required_window_observed",
    "minimum_points_met",
    "threshold_crossed",
    "left_censored",
    "right_censored",
    "curve_complete_enough",
    "unit_resolved",
    "normalization_valid",
    "duplicate_curve_flag",
    "source_conflict_flag",
    "ineligibility_reason",
]

BUDGET_ELIGIBILITY_COLUMNS = [
    "formulation_id",
    "curve_id",
    "source_dataset",
    "source_id",
    "system",
    "budget_id",
    "budget_type",
    "k_points",
    "observation_cutoff_time",
    "n_observed_points",
    "n_future_points",
    "observed_time_min",
    "observed_time_max",
    "future_time_min",
    "future_time_max",
    "target_id",
    "target_still_unobserved",
    "future_window_available",
    "target_window_available",
    "minimum_future_points_met",
    "eligible_for_future_evaluation",
    "eligible_for_decision_evaluation",
    "ineligibility_reason",
]


@dataclass
class ScreeningEligibilityResult:
    screening_target_schema: pd.DataFrame
    formulation_target_eligibility: pd.DataFrame
    observation_budget_eligibility: pd.DataFrame
    future_window_coverage: pd.DataFrame
    target_cross_system_comparability: pd.DataFrame
    target_source_confounding_audit: pd.DataFrame
    sampling_schedule_source_audit: pd.DataFrame
    screening_task_feasibility_matrix: pd.DataFrame
    gate_summary: dict[str, Any]
    stats: dict[str, Any]
    budget_schema: pd.DataFrame


def _field_present(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text not in {"", "nan", "NaN", "None", "<NA>"}


def _canonical_source_id(row: pd.Series) -> str:
    for column in ["source_curve_id", "source_group", "formulation_ID", "IVR_ID", "unified_curve_id"]:
        if column in row and _field_present(row[column]):
            return str(row[column]).strip()
    return str(row.get("unified_curve_id", "")).strip()


def _canonical_formulation_id(row: pd.Series) -> str:
    if "formulation_ID" in row and _field_present(row["formulation_ID"]):
        return f"{row['source_dataset']}::formulation_id::{str(row['formulation_ID']).strip()}"
    if str(row.get("source_has_explicit_group", "")).lower() == "true" and _field_present(row.get("source_group", "")):
        return f"{row['source_dataset']}::group::{str(row['source_group']).strip()}"
    return str(row["unified_curve_id"])


def _system_group(polymer_family: str) -> str:
    text = str(polymer_family).lower()
    if "plga" in text:
        return "PLGA"
    if "liposome" in text:
        return "liposome"
    if "hydrogel" in text:
        return "hydrogel"
    if "degrapol" in text:
        return "DegraPol"
    if "film" in text:
        return "film"
    if "microbead" in text:
        return "microbead"
    if "microsphere" in text:
        return "microsphere"
    if "nanoparticle" in text:
        return "nanoparticle"
    if "halloysite" in text:
        return "halloysite"
    if "laponite" in text:
        return "laponite"
    return str(polymer_family).strip() or "system_not_reported"


def _target_schema() -> pd.DataFrame:
    rows = [
        {
            "target_id": "q_day1",
            "target_name": "release fraction at 1 day",
            "target_family": "curve_phenotype",
            "target_type": "continuous",
            "definition": "Cumulative release fraction at 1 day from the observed curve.",
            "formula_or_rule": "linear interpolation on release_fraction versus time_days at t=1 day",
            "required_curve_region": "around_1_day",
            "required_min_time": 1.0,
            "required_max_time": "",
            "minimum_timepoints": 2,
            "thresholds": "",
            "threshold_source": "",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": False,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "insufficient_followup_if_final_time_lt_1d",
            "missing_rule": "not_computable_if_time_bracketing_absent",
            "ambiguity_note": "Absolute 1-day timing is not a universal biological stage across systems.",
            "status": "system_specific",
        },
        {
            "target_id": "t10_days",
            "target_name": "time to 10 percent release",
            "target_family": "curve_phenotype",
            "target_type": "time_to_threshold",
            "definition": "First time cumulative release reaches 10 percent.",
            "formula_or_rule": "first threshold crossing of the monotone cumulative envelope at 0.10",
            "required_curve_region": "threshold_crossing",
            "required_min_time": 0.0,
            "required_max_time": "",
            "minimum_timepoints": 2,
            "thresholds": "0.10",
            "threshold_source": "phenotype_threshold_contract",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "left_censored_if_first_point_ge_threshold; right_censored_if_final_envelope_lt_threshold",
            "missing_rule": "invalid_if_less_than_two_points",
            "ambiguity_note": "Absolute threshold times are heavily system- and schedule-dependent.",
            "status": "conditionally_eligible",
        },
        {
            "target_id": "t25_days",
            "target_name": "time to 25 percent release",
            "target_family": "curve_phenotype",
            "target_type": "time_to_threshold",
            "definition": "First time cumulative release reaches 25 percent.",
            "formula_or_rule": "first threshold crossing of the monotone cumulative envelope at 0.25",
            "required_curve_region": "threshold_crossing",
            "required_min_time": 0.0,
            "required_max_time": "",
            "minimum_timepoints": 2,
            "thresholds": "0.25",
            "threshold_source": "phenotype_threshold_contract",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "left_censored_if_first_point_ge_threshold; right_censored_if_final_envelope_lt_threshold",
            "missing_rule": "invalid_if_less_than_two_points",
            "ambiguity_note": "Threshold time depends on assay horizon and cannot be assumed cross-system equivalent.",
            "status": "conditionally_eligible",
        },
        {
            "target_id": "t50_days",
            "target_name": "time to 50 percent release",
            "target_family": "curve_phenotype",
            "target_type": "time_to_threshold",
            "definition": "First time cumulative release reaches 50 percent.",
            "formula_or_rule": "first threshold crossing of the monotone cumulative envelope at 0.50",
            "required_curve_region": "threshold_crossing",
            "required_min_time": 0.0,
            "required_max_time": "",
            "minimum_timepoints": 2,
            "thresholds": "0.50",
            "threshold_source": "phenotype_threshold_contract",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "left_censored_if_first_point_ge_threshold; right_censored_if_final_envelope_lt_threshold",
            "missing_rule": "invalid_if_less_than_two_points",
            "ambiguity_note": "Useful within system, but absolute 50 percent timing is not a universal stage across mechanisms.",
            "status": "conditionally_eligible",
        },
        {
            "target_id": "final_release_fraction",
            "target_name": "final observed release fraction",
            "target_family": "curve_phenotype",
            "target_type": "continuous",
            "definition": "Final reported cumulative release fraction at the last observed timepoint.",
            "formula_or_rule": "last release_fraction on the observed curve",
            "required_curve_region": "curve_end",
            "required_min_time": 0.0,
            "required_max_time": "",
            "minimum_timepoints": 1,
            "thresholds": "",
            "threshold_source": "",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "right_censored_if_followup_horizon_is_too_short_for_application_claims",
            "missing_rule": "invalid_if_no_release_fraction",
            "ambiguity_note": "Final observed release mixes mechanism with study duration.",
            "status": "system_specific",
        },
        {
            "target_id": "incomplete_release_final_lt80",
            "target_name": "final release below 80 percent",
            "target_family": "curve_phenotype",
            "target_type": "binary",
            "definition": "Whether the final observed cumulative release remains below 80 percent.",
            "formula_or_rule": "1 if final_release_fraction < 0.80 else 0",
            "required_curve_region": "curve_end",
            "required_min_time": 0.0,
            "required_max_time": "",
            "minimum_timepoints": 1,
            "thresholds": "0.80",
            "threshold_source": "phenotype_threshold_contract",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "application interpretation right-censored when followup is short",
            "missing_rule": "invalid_if_no_release_fraction",
            "ambiguity_note": "Final incompleteness can reflect short studies rather than true asymptote.",
            "status": "system_specific",
        },
        {
            "target_id": "normalized_t50_fraction_of_horizon",
            "target_name": "time to 50 percent as fraction of observed horizon",
            "target_family": "system_normalized_phenotype",
            "target_type": "continuous",
            "definition": "Time to 50 percent release divided by the final observed curve horizon.",
            "formula_or_rule": "t50_days / final_time_days",
            "required_curve_region": "threshold_crossing_and_curve_end",
            "required_min_time": 0.0,
            "required_max_time": "",
            "minimum_timepoints": 2,
            "thresholds": "0.50",
            "threshold_source": "phenotype_threshold_contract",
            "system_scope": "all",
            "application_scope": "none",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": True,
            "allowed_for_pre-observation_decision": True,
            "cross_system_comparable": True,
            "normalization_rule": "divide by final observed horizon; audit-only for cross-system claims unless leakage-safe evaluation is feasible",
            "censoring_rule": "inherits t50 censoring",
            "missing_rule": "invalid_if_final_time_nonpositive_or_t50_missing",
            "ambiguity_note": "Normalization uses full-curve horizon and therefore cannot be a pre-observation input.",
            "status": "conditionally_eligible",
        },
        {
            "target_id": "application_target_not_available",
            "target_name": "global application utility target",
            "target_family": "application_conditioned_utility",
            "target_type": "binary",
            "definition": "Placeholder for a real application-conditioned screening target.",
            "formula_or_rule": "not_defined",
            "required_curve_region": "application_specific",
            "required_min_time": "",
            "required_max_time": "",
            "minimum_timepoints": "",
            "thresholds": "",
            "threshold_source": "not_available_locally",
            "system_scope": "unknown",
            "application_scope": "global_application_not_defined",
            "uses_full_curve": True,
            "derived_from_future": True,
            "allowed_as_model_input": False,
            "allowed_as_training_target": False,
            "allowed_for_pre-observation_decision": False,
            "cross_system_comparable": False,
            "normalization_rule": "none",
            "censoring_rule": "not_applicable",
            "missing_rule": "application_target_not_available",
            "ambiguity_note": "No repository-wide application utility threshold is currently justified.",
            "status": "not_supported",
        },
    ]
    return pd.DataFrame(rows, columns=TARGET_SCHEMA_COLUMNS)


def _budget_schema() -> pd.DataFrame:
    rows = [
        {"budget_id": "k0_points", "budget_type": "point_count", "k_points": 0, "cutoff_days": np.nan, "normalized_cutoff": np.nan},
        {"budget_id": "k1_points", "budget_type": "point_count", "k_points": 1, "cutoff_days": np.nan, "normalized_cutoff": np.nan},
        {"budget_id": "k2_points", "budget_type": "point_count", "k_points": 2, "cutoff_days": np.nan, "normalized_cutoff": np.nan},
        {"budget_id": "k3_points", "budget_type": "point_count", "k_points": 3, "cutoff_days": np.nan, "normalized_cutoff": np.nan},
        {"budget_id": "k5_points", "budget_type": "point_count", "k_points": 5, "cutoff_days": np.nan, "normalized_cutoff": np.nan},
        {"budget_id": "abs_6h", "budget_type": "absolute_time", "k_points": np.nan, "cutoff_days": 0.25, "normalized_cutoff": np.nan},
        {"budget_id": "abs_24h", "budget_type": "absolute_time", "k_points": np.nan, "cutoff_days": 1.0, "normalized_cutoff": np.nan},
        {"budget_id": "abs_3d", "budget_type": "absolute_time", "k_points": np.nan, "cutoff_days": 3.0, "normalized_cutoff": np.nan},
        {"budget_id": "abs_7d", "budget_type": "absolute_time", "k_points": np.nan, "cutoff_days": 7.0, "normalized_cutoff": np.nan},
        {"budget_id": "norm_10pct_horizon", "budget_type": "normalized_time", "k_points": np.nan, "cutoff_days": np.nan, "normalized_cutoff": 0.10},
        {"budget_id": "norm_20pct_horizon", "budget_type": "normalized_time", "k_points": np.nan, "cutoff_days": np.nan, "normalized_cutoff": 0.20},
    ]
    return pd.DataFrame(rows)


def _prepare_forms(forms: pd.DataFrame) -> pd.DataFrame:
    prepared = forms.copy()
    prepared["canonical_formulation_id"] = prepared.apply(_canonical_formulation_id, axis=1)
    prepared["curve_id"] = prepared["unified_curve_id"].astype(str)
    prepared["source_id"] = prepared.apply(_canonical_source_id, axis=1)
    prepared["system"] = prepared["polymer_family"].astype(str).map(_system_group)
    prepared["doi"] = ""
    counts = prepared.groupby("canonical_formulation_id")["curve_id"].transform("nunique")
    prepared["duplicate_curve_flag"] = counts.gt(1)
    source_conflict = prepared.groupby("canonical_formulation_id")["source_dataset"].transform("nunique")
    prepared["source_conflict_flag"] = source_conflict.gt(1)
    return prepared


def _schedule_signature(times: np.ndarray) -> str:
    if len(times) == 0:
        return "empty"
    rounded = [f"{float(value):.6g}" for value in np.round(times, 6)]
    return f"n={len(times)}|" + ",".join(rounded[:12]) + ("|..." if len(rounded) > 12 else "")


def _interp_at_time(times: np.ndarray, values: np.ndarray, target_time: float) -> tuple[float | None, str]:
    if len(times) < 2:
        return None, "insufficient_points"
    if target_time < float(times.min()) or target_time > float(times.max()):
        return None, "insufficient_followup"
    exact = np.where(np.isclose(times, target_time))[0]
    if len(exact):
        return float(values[int(exact[0])]), "observed"
    idx = int(np.searchsorted(times, target_time, side="right"))
    if idx <= 0 or idx >= len(times):
        return None, "insufficient_followup"
    t0, t1 = float(times[idx - 1]), float(times[idx])
    v0, v1 = float(values[idx - 1]), float(values[idx])
    if abs(t1 - t0) <= 1e-12:
        return float(v1), "duplicate_time_interpolation"
    frac = (target_time - t0) / (t1 - t0)
    return float(v0 + frac * (v1 - v0)), "interpolated"


def _threshold_time(times: np.ndarray, envelope: np.ndarray, threshold: float) -> dict[str, Any]:
    if len(times) < 2:
        return {
            "value": None,
            "threshold_crossed": False,
            "left_censored": False,
            "right_censored": False,
            "status": "invalid_curve",
        }
    if float(envelope[0]) >= threshold:
        return {
            "value": float(times[0]),
            "threshold_crossed": True,
            "left_censored": True,
            "right_censored": False,
            "status": "left_censored",
        }
    if float(envelope[-1]) < threshold:
        return {
            "value": None,
            "threshold_crossed": False,
            "left_censored": False,
            "right_censored": True,
            "status": "not_reached",
        }
    idx = int(np.argmax(envelope >= threshold))
    t0, t1 = float(times[idx - 1]), float(times[idx])
    q0, q1 = float(envelope[idx - 1]), float(envelope[idx])
    if q1 <= q0 + 1e-12:
        crossing_time = t1
    else:
        crossing_time = t0 + (threshold - q0) * (t1 - t0) / (q1 - q0)
    return {
        "value": float(crossing_time),
        "threshold_crossed": True,
        "left_censored": False,
        "right_censored": False,
        "status": "observed",
    }


def _curve_tables(forms: pd.DataFrame, curves: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    curve_points: dict[str, pd.DataFrame] = {}
    meta_rows: list[dict[str, Any]] = []
    curve_lookup = forms.set_index("curve_id")
    for curve_id, df in curves.groupby("unified_curve_id", sort=True):
        form_row = curve_lookup.loc[str(curve_id)]
        work = df.copy()
        work["time_days"] = pd.to_numeric(work["time_days"], errors="coerce")
        work["release_fraction"] = pd.to_numeric(work["release_fraction"], errors="coerce")
        work = work.dropna(subset=["time_days", "release_fraction"]).sort_values("time_days")
        duplicate_timepoints_flag = bool(work["time_days"].duplicated().any())
        grouped = work.groupby("time_days", as_index=False).agg(
            release_fraction=("release_fraction", "mean"),
            record_count=("record_id", "count"),
            release_unit=("release_unit", lambda s: "|".join(sorted({str(v) for v in s.dropna().astype(str)}))),
            time_unit=("time_unit", lambda s: "|".join(sorted({str(v) for v in s.dropna().astype(str)}))),
        )
        grouped["release_envelope"] = np.maximum.accumulate(grouped["release_fraction"].to_numpy(dtype=float))
        grouped["time_days"] = grouped["time_days"].astype(float)
        grouped["curve_id"] = str(curve_id)
        curve_points[str(curve_id)] = grouped
        raw_release = grouped["release_fraction"].to_numpy(dtype=float)
        raw_diff = np.diff(raw_release) if len(raw_release) >= 2 else np.array([])
        meta_rows.append(
            {
                "curve_id": str(curve_id),
                "formulation_id": form_row["canonical_formulation_id"],
                "source_dataset": form_row["source_dataset"],
                "source_id": form_row["source_id"],
                "doi": form_row["doi"],
                "system": form_row["system"],
                "payload_name": form_row.get("payload_name", ""),
                "polymer_family": form_row.get("polymer_family", ""),
                "duplicate_curve_flag": bool(form_row["duplicate_curve_flag"]),
                "source_conflict_flag": bool(form_row["source_conflict_flag"]),
                "curve_complete_enough": bool(len(grouped) >= 2),
                "unit_resolved": bool(grouped["release_fraction"].notna().all()),
                "normalization_valid": bool(_field_present(form_row.get("normalization_basis", ""))),
                "duplicate_timepoints_flag": duplicate_timepoints_flag,
                "n_points": int(len(grouped)),
                "n_raw_points": int(len(work)),
                "time_min": float(grouped["time_days"].min()) if not grouped.empty else np.nan,
                "time_max": float(grouped["time_days"].max()) if not grouped.empty else np.nan,
                "final_release_fraction": float(grouped["release_fraction"].iloc[-1]) if not grouped.empty else np.nan,
                "max_release_fraction": float(grouped["release_envelope"].iloc[-1]) if not grouped.empty else np.nan,
                "release_unit_signature": "|".join(sorted({str(v) for v in grouped["release_unit"].astype(str)})),
                "time_unit_signature": "|".join(sorted({str(v) for v in grouped["time_unit"].astype(str)})),
                "nonmonotone_step_count": int((raw_diff < -1e-9).sum()),
                "schedule_signature": _schedule_signature(grouped["time_days"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(meta_rows), curve_points


def _curve_target_rows(
    schema: pd.DataFrame,
    curve_meta: pd.DataFrame,
    curve_points: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, meta in curve_meta.iterrows():
        curve_id = str(meta["curve_id"])
        points = curve_points[curve_id]
        times = points["time_days"].to_numpy(dtype=float)
        release = points["release_fraction"].to_numpy(dtype=float)
        envelope = points["release_envelope"].to_numpy(dtype=float)
        final_time = float(times[-1]) if len(times) else np.nan
        final_release = float(release[-1]) if len(release) else np.nan
        for _, target in schema.iterrows():
            row = {
                "formulation_id": meta["formulation_id"],
                "curve_id": curve_id,
                "source_dataset": meta["source_dataset"],
                "source_id": meta["source_id"],
                "doi": meta["doi"],
                "system": meta["system"],
                "target_id": target["target_id"],
                "target_value": "",
                "target_computable": False,
                "eligibility_status": "not_supported",
                "required_window_observed": False,
                "minimum_points_met": bool(len(times) >= int(target["minimum_timepoints"]) if str(target["minimum_timepoints"]).strip() else True),
                "threshold_crossed": False,
                "left_censored": False,
                "right_censored": False,
                "curve_complete_enough": bool(meta["curve_complete_enough"]),
                "unit_resolved": bool(meta["unit_resolved"]),
                "normalization_valid": bool(meta["normalization_valid"]),
                "duplicate_curve_flag": bool(meta["duplicate_curve_flag"]),
                "source_conflict_flag": bool(meta["source_conflict_flag"]),
                "ineligibility_reason": "",
            }
            reasons: list[str] = []
            if target["target_id"] == "application_target_not_available":
                reasons.append("application_target_not_available")
            elif target["target_id"] == "q_day1":
                value, status = _interp_at_time(times, envelope, 1.0)
                row["required_window_observed"] = bool(final_time >= 1.0)
                if value is not None:
                    row["target_value"] = float(value)
                    row["target_computable"] = True
                    row["eligibility_status"] = "system_specific"
                else:
                    reasons.append(status)
            elif target["target_id"] in {"t10_days", "t25_days", "t50_days"}:
                threshold = {"t10_days": 0.10, "t25_days": 0.25, "t50_days": 0.50}[target["target_id"]]
                crossing = _threshold_time(times, envelope, threshold)
                row["required_window_observed"] = True
                row["threshold_crossed"] = bool(crossing["threshold_crossed"])
                row["left_censored"] = bool(crossing["left_censored"])
                row["right_censored"] = bool(crossing["right_censored"])
                if crossing["value"] is not None:
                    row["target_value"] = float(crossing["value"])
                elif crossing["status"] == "not_reached":
                    row["target_value"] = "not_reached"
                if crossing["status"] in {"observed", "left_censored", "not_reached"}:
                    row["target_computable"] = True
                    row["eligibility_status"] = "within_system_only"
                else:
                    reasons.append(crossing["status"])
            elif target["target_id"] == "final_release_fraction":
                row["required_window_observed"] = bool(len(release) >= 1)
                if len(release):
                    row["target_value"] = final_release
                    row["target_computable"] = True
                    row["eligibility_status"] = "system_specific"
                else:
                    reasons.append("invalid_curve")
            elif target["target_id"] == "incomplete_release_final_lt80":
                row["required_window_observed"] = bool(len(release) >= 1)
                if len(release):
                    row["target_value"] = int(final_release < 0.80)
                    row["threshold_crossed"] = bool(final_release >= 0.80)
                    row["right_censored"] = bool(final_release < 0.80)
                    row["target_computable"] = True
                    row["eligibility_status"] = "system_specific"
                else:
                    reasons.append("invalid_curve")
            elif target["target_id"] == "normalized_t50_fraction_of_horizon":
                crossing = _threshold_time(times, envelope, 0.50)
                row["required_window_observed"] = True
                row["threshold_crossed"] = bool(crossing["threshold_crossed"])
                row["left_censored"] = bool(crossing["left_censored"])
                row["right_censored"] = bool(crossing["right_censored"])
                if crossing["value"] is not None and np.isfinite(final_time) and final_time > 0:
                    row["target_value"] = float(crossing["value"] / final_time)
                    row["target_computable"] = True
                    row["eligibility_status"] = "conditionally_eligible"
                elif crossing["status"] == "not_reached":
                    row["target_value"] = "not_reached"
                    row["target_computable"] = True
                    row["eligibility_status"] = "conditionally_eligible"
                else:
                    reasons.append(crossing["status"])
            else:
                reasons.append("unknown_target")

            if not row["minimum_points_met"]:
                row["target_computable"] = False
                reasons.append("minimum_timepoints_not_met")
            if not row["unit_resolved"]:
                row["target_computable"] = False
                reasons.append("release_units_unresolved")
            if target["target_id"] == "normalized_t50_fraction_of_horizon" and not np.isfinite(final_time):
                row["target_computable"] = False
                reasons.append("final_time_unresolved")

            if not row["target_computable"] and not reasons:
                reasons.append("not_computable")
            if row["target_id"] == "application_target_not_available":
                row["eligibility_status"] = "not_supported"
            elif row["target_computable"] and row["eligibility_status"] == "within_system_only" and row["system"] in {"PLGA", "liposome"}:
                row["eligibility_status"] = "conditionally_eligible"
            row["ineligibility_reason"] = "|".join(dict.fromkeys(reason for reason in reasons if reason))
            rows.append(row)
    return pd.DataFrame(rows, columns=TARGET_ELIGIBILITY_COLUMNS)


def _observed_future_points(points: pd.DataFrame, budget: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if budget["budget_type"] == "point_count":
        k = int(budget["k_points"])
        observed = points.iloc[:k].copy()
        future = points.iloc[k:].copy()
        cutoff = float(observed["time_days"].max()) if not observed.empty else 0.0
        return observed, future, cutoff
    if budget["budget_type"] == "absolute_time":
        cutoff = float(budget["cutoff_days"])
        observed = points[points["time_days"].le(cutoff)].copy()
        future = points[points["time_days"].gt(cutoff)].copy()
        return observed, future, cutoff
    cutoff = float(points["time_days"].max()) * float(budget["normalized_cutoff"])
    observed = points[points["time_days"].le(cutoff)].copy()
    future = points[points["time_days"].gt(cutoff)].copy()
    return observed, future, cutoff


def _budget_target_rows(
    target_eligibility: pd.DataFrame,
    budget_schema: pd.DataFrame,
    curve_meta: pd.DataFrame,
    curve_points: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    meta_lookup = curve_meta.set_index("curve_id")
    target_lookup = target_eligibility.set_index(["curve_id", "target_id"])
    rows: list[dict[str, Any]] = []
    for curve_id, points in curve_points.items():
        meta = meta_lookup.loc[curve_id]
        final_time = float(meta["time_max"]) if np.isfinite(meta["time_max"]) else np.nan
        final_release = float(meta["final_release_fraction"]) if np.isfinite(meta["final_release_fraction"]) else np.nan
        envelope = points["release_envelope"].to_numpy(dtype=float)
        times = points["time_days"].to_numpy(dtype=float)
        t10 = _threshold_time(times, envelope, 0.10)
        t25 = _threshold_time(times, envelope, 0.25)
        t50 = _threshold_time(times, envelope, 0.50)
        threshold_events = {
            "t10_days": t10,
            "t25_days": t25,
            "t50_days": t50,
            "normalized_t50_fraction_of_horizon": t50,
        }
        for _, budget in budget_schema.iterrows():
            observed, future, cutoff = _observed_future_points(points, budget)
            obs_times = observed["time_days"].to_numpy(dtype=float)
            future_times = future["time_days"].to_numpy(dtype=float)
            obs_envelope = observed["release_envelope"].to_numpy(dtype=float)
            for target_id in target_lookup.loc[curve_id].index.tolist():
                target_row = target_lookup.loc[(curve_id, target_id)]
                target_still_unobserved = False
                target_window_available = False
                reasons: list[str] = []
                if not bool(target_row["target_computable"]):
                    reasons.append(str(target_row["ineligibility_reason"]) or "target_not_computable")
                if target_id == "final_release_fraction" or target_id == "incomplete_release_final_lt80":
                    target_still_unobserved = bool(np.isfinite(final_time) and cutoff < final_time)
                    target_window_available = target_still_unobserved
                elif target_id == "q_day1":
                    target_still_unobserved = bool(np.isfinite(final_time) and cutoff < 1.0 <= final_time)
                    target_window_available = target_still_unobserved
                elif target_id in threshold_events:
                    crossing = threshold_events[target_id]
                    if crossing["status"] == "invalid_curve":
                        target_still_unobserved = False
                        target_window_available = False
                    elif crossing["status"] == "left_censored":
                        target_still_unobserved = False
                        target_window_available = False
                    elif crossing["status"] == "not_reached":
                        target_still_unobserved = bool(np.isfinite(final_time) and cutoff < final_time)
                        target_window_available = target_still_unobserved
                    else:
                        event_time = float(crossing["value"]) if crossing["value"] is not None else np.nan
                        target_still_unobserved = bool(np.isfinite(event_time) and event_time > cutoff)
                        target_window_available = target_still_unobserved
                else:
                    target_still_unobserved = False
                    target_window_available = False

                future_window_available = bool(len(future) > 0)
                min_future_points = 1
                minimum_future_points_met = bool(len(future) >= min_future_points)
                eligible = bool(
                    target_row["target_computable"]
                    and target_still_unobserved
                    and future_window_available
                    and target_window_available
                    and minimum_future_points_met
                )
                if not target_still_unobserved:
                    reasons.append("target_already_observed_or_not_defined")
                if not future_window_available:
                    reasons.append("no_future_window")
                if future_window_available and not target_window_available:
                    reasons.append("target_window_not_in_future")
                if future_window_available and not minimum_future_points_met:
                    reasons.append("insufficient_future_points")
                rows.append(
                    {
                        "formulation_id": meta["formulation_id"],
                        "curve_id": curve_id,
                        "source_dataset": meta["source_dataset"],
                        "source_id": meta["source_id"],
                        "system": meta["system"],
                        "budget_id": budget["budget_id"],
                        "budget_type": budget["budget_type"],
                        "k_points": int(budget["k_points"]) if pd.notna(budget["k_points"]) else np.nan,
                        "observation_cutoff_time": cutoff,
                        "n_observed_points": int(len(observed)),
                        "n_future_points": int(len(future)),
                        "observed_time_min": float(obs_times.min()) if len(obs_times) else np.nan,
                        "observed_time_max": float(obs_times.max()) if len(obs_times) else np.nan,
                        "future_time_min": float(future_times.min()) if len(future_times) else np.nan,
                        "future_time_max": float(future_times.max()) if len(future_times) else np.nan,
                        "target_id": target_id,
                        "target_still_unobserved": target_still_unobserved,
                        "future_window_available": future_window_available,
                        "target_window_available": target_window_available,
                        "minimum_future_points_met": minimum_future_points_met,
                        "eligible_for_future_evaluation": eligible,
                        "eligible_for_decision_evaluation": eligible,
                        "ineligibility_reason": "|".join(dict.fromkeys(reason for reason in reasons if reason)),
                    }
                )
    return pd.DataFrame(rows, columns=BUDGET_ELIGIBILITY_COLUMNS)


def _summary_distribution(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "empty"
    return f"median={clean.median():.4f};p10={clean.quantile(0.1):.4f};p90={clean.quantile(0.9):.4f}"


def _source_entropy(series: pd.Series) -> float:
    values = series.astype(str).value_counts(normalize=True)
    if values.empty:
        return 0.0
    return float(-sum(p * log(p, 2) for p in values if p > 0))


def _future_window_coverage(obs_budget: pd.DataFrame, target_eligibility: pd.DataFrame) -> pd.DataFrame:
    merged = obs_budget.merge(
        target_eligibility[
            ["curve_id", "target_id", "target_computable", "right_censored"]
        ],
        on=["curve_id", "target_id"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    groupings = [
        ("overall", [], "all"),
        ("system", ["system"], None),
        ("source_dataset", ["source_dataset"], None),
        ("source_id", ["source_id"], None),
        ("target_budget", ["target_id", "budget_id"], None),
    ]
    for level, cols, fixed_key in groupings:
        if not cols:
            groups = [(fixed_key, merged)]
        else:
            groups = merged.groupby(cols, dropna=False, sort=True)
        for key, df in groups:
            label = fixed_key if fixed_key is not None else "|".join([str(v) for v in (key if isinstance(key, tuple) else (key,))])
            rows.append(
                {
                    "audit_level": level,
                    "audit_key": label,
                    "n_curves": int(df["curve_id"].nunique()),
                    "n_target_computable": int(df[df["target_computable"]]["curve_id"].nunique()),
                    "n_budget_eligible": int(df[df["eligible_for_future_evaluation"]]["curve_id"].nunique()),
                    "n_with_future_window": int(df[df["future_window_available"]]["curve_id"].nunique()),
                    "n_with_target_future_window": int(df[df["target_window_available"]]["curve_id"].nunique()),
                    "eligibility_fraction": float(df["eligible_for_future_evaluation"].mean()),
                    "median_observed_points": float(df["n_observed_points"].median()),
                    "median_future_points": float(df["n_future_points"].median()),
                    "median_observation_cutoff": float(pd.to_numeric(df["observation_cutoff_time"], errors="coerce").median()),
                    "median_future_horizon": float((pd.to_numeric(df["future_time_max"], errors="coerce") - pd.to_numeric(df["observed_time_max"], errors="coerce")).median()),
                    "fraction_right_censored": float(df["right_censored"].fillna(False).mean()),
                    "fraction_target_already_observed": float((~df["target_still_unobserved"]).mean()),
                    "fraction_insufficient_future": float((~df["future_window_available"] | ~df["minimum_future_points_met"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def _target_cross_system(target_schema: pd.DataFrame, target_eligibility: pd.DataFrame, curve_meta: pd.DataFrame) -> pd.DataFrame:
    merged = target_eligibility.merge(
        curve_meta[["curve_id", "time_max", "n_points", "schedule_signature"]],
        on="curve_id",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for _, target in target_schema.iterrows():
        target_id = target["target_id"]
        target_df = merged[merged["target_id"].eq(target_id)].copy()
        for system, df in target_df.groupby("system", sort=True):
            eligible = df[df["target_computable"]].copy()
            n_eligible = int(len(eligible))
            cross_system = False
            reason = "target_not_supported"
            recommended_scope = "not-supported"
            if target["status"] == "not_supported":
                pass
            elif target_id == "normalized_t50_fraction_of_horizon":
                if n_eligible >= 20 and float(eligible["right_censored"].mean()) <= 0.6:
                    cross_system = False
                    reason = "normalization_uses_full_horizon_and_support_is_still_system-imbalanced"
                    recommended_scope = "system-normalized"
                else:
                    reason = "insufficient_support_or_high_censoring"
                    recommended_scope = "within-system-only"
            else:
                reason = "absolute_time_or_final_horizon_not_cross-system_safe"
                recommended_scope = "within-system-only"
            rows.append(
                {
                    "target_id": target_id,
                    "system": system,
                    "n_eligible": n_eligible,
                    "time_scale": float(pd.to_numeric(eligible["time_max"], errors="coerce").median()) if n_eligible else np.nan,
                    "value_distribution": _summary_distribution(eligible["target_value"]),
                    "censoring_fraction": float(eligible["right_censored"].mean()) if n_eligible else 0.0,
                    "measurement_schedule_pattern": f"median_timepoints={float(pd.to_numeric(eligible['n_points'], errors='coerce').median()) if n_eligible else np.nan}",
                    "cross_system_comparable": cross_system,
                    "reason": reason,
                    "recommended_scope": recommended_scope,
                }
            )
    return pd.DataFrame(rows)


def _target_source_confounding(obs_budget: pd.DataFrame, target_eligibility: pd.DataFrame) -> pd.DataFrame:
    merged = obs_budget.merge(
        target_eligibility[["curve_id", "target_id", "target_value", "target_computable"]],
        on=["curve_id", "target_id"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for (target_id, budget_id), df in merged.groupby(["target_id", "budget_id"], sort=True):
        eligible = df[df["eligible_for_future_evaluation"] & df["target_computable"]].copy()
        if eligible.empty:
            rows.append(
                {
                    "target_id": target_id,
                    "budget_id": budget_id,
                    "n_sources": 0,
                    "n_systems": 0,
                    "n_curves": 0,
                    "max_source_fraction": 1.0,
                    "source_entropy": 0.0,
                    "class_balance": "empty",
                    "source_bound_flag": True,
                    "leave_source_evaluation_feasible": False,
                    "cross_source_claim_allowed": False,
                }
            )
            continue
        source_fraction = eligible["source_dataset"].astype(str).value_counts(normalize=True)
        target_numeric = pd.to_numeric(eligible["target_value"], errors="coerce")
        if target_numeric.notna().any():
            class_balance = f"variance={float(target_numeric.var(ddof=0)):.6f}"
        else:
            positive = eligible["target_value"].astype(str).eq("1").mean()
            class_balance = f"positive_fraction={positive:.4f}"
        max_fraction = float(source_fraction.max()) if not source_fraction.empty else 1.0
        leave_source = bool(
            eligible["source_dataset"].nunique() >= 2
            and eligible.groupby("source_dataset")["curve_id"].nunique().min() >= 10
        )
        rows.append(
            {
                "target_id": target_id,
                "budget_id": budget_id,
                "n_sources": int(eligible["source_dataset"].nunique()),
                "n_systems": int(eligible["system"].nunique()),
                "n_curves": int(eligible["curve_id"].nunique()),
                "max_source_fraction": max_fraction,
                "source_entropy": _source_entropy(eligible["source_dataset"]),
                "class_balance": class_balance,
                "source_bound_flag": bool(max_fraction >= 0.80 or eligible["source_dataset"].nunique() < 2),
                "leave_source_evaluation_feasible": leave_source,
                "cross_source_claim_allowed": bool(max_fraction < 0.80 and leave_source),
            }
        )
    return pd.DataFrame(rows)


def _schedule_audit(curve_meta: pd.DataFrame) -> pd.DataFrame:
    signature_source_counts = curve_meta.groupby("schedule_signature")["source_dataset"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    for (source_dataset, system), df in curve_meta.groupby(["source_dataset", "system"], sort=True):
        signature_counts = df["schedule_signature"].astype(str).value_counts()
        most_common = signature_counts.index[0] if not signature_counts.empty else "empty"
        shared = signature_source_counts.get(most_common, 0)
        source_specificity = float(signature_counts.iloc[0] / len(df)) if len(df) else 0.0
        source_identifiable = bool(source_specificity >= 0.80 and shared <= 1)
        rows.append(
            {
                "source_id": source_dataset,
                "system": system,
                "n_curves": int(len(df)),
                "unique_schedule_count": int(df["schedule_signature"].nunique()),
                "median_timepoints": float(df["n_points"].median()),
                "common_time_grid": most_common,
                "schedule_signature": most_common,
                "schedule_source_specificity": source_specificity,
                "source_identifiable_from_schedule": source_identifiable,
                "recommended_mitigation": "leave-source evaluation|common-grid evaluation" if source_identifiable else "monitor_only",
            }
        )
    return pd.DataFrame(rows)


def _task_feasibility(
    obs_budget: pd.DataFrame,
    target_eligibility: pd.DataFrame,
    confounding: pd.DataFrame,
) -> pd.DataFrame:
    merged = obs_budget.merge(
        target_eligibility[["curve_id", "target_id", "target_value", "target_computable"]],
        on=["curve_id", "target_id"],
        how="left",
    )
    conf_lookup = confounding.set_index(["target_id", "budget_id"])
    rows: list[dict[str, Any]] = []
    scopes = ["cross_system"] + [f"within_system::{system}" for system in sorted(merged["system"].dropna().unique())]
    for target_id in sorted(merged["target_id"].unique()):
        for budget_id in sorted(merged["budget_id"].unique()):
            base = merged[(merged["target_id"].eq(target_id)) & (merged["budget_id"].eq(budget_id))].copy()
            for scope in scopes:
                if scope == "cross_system":
                    df = base.copy()
                else:
                    system = scope.split("::", 1)[1]
                    df = base[base["system"].eq(system)].copy()
                computable = df[df["target_computable"]].copy()
                eligible = df[df["eligible_for_future_evaluation"] & df["target_computable"]].copy()
                if computable.empty and eligible.empty:
                    continue
                if scope == "cross_system" and (target_id, budget_id) in conf_lookup.index:
                    conf_row = conf_lookup.loc[(target_id, budget_id)]
                    source_ok = bool(conf_row["cross_source_claim_allowed"])
                    leave_source = bool(conf_row["leave_source_evaluation_feasible"])
                else:
                    source_counts = eligible.groupby("source_dataset")["curve_id"].nunique() if not eligible.empty else pd.Series(dtype=int)
                    max_fraction = float(eligible["source_dataset"].astype(str).value_counts(normalize=True).max()) if not eligible.empty else 1.0
                    source_ok = bool(eligible["source_dataset"].nunique() >= 2 and max_fraction < 0.85)
                    leave_source = bool(not source_counts.empty and source_counts.min() >= 5 and len(source_counts) >= 2)
                leave_system = bool(scope == "cross_system" and eligible["system"].nunique() >= 2 and eligible.groupby("system")["curve_id"].nunique().min() >= 20) if not eligible.empty else False
                future_valid = bool(not eligible.empty)
                numeric = pd.to_numeric(eligible["target_value"], errors="coerce")
                if numeric.notna().any():
                    class_balance = f"variance={float(numeric.var(ddof=0)):.6f}"
                    variability_ok = bool(float(numeric.var(ddof=0)) > 1e-8)
                else:
                    positive = eligible["target_value"].astype(str).eq("1").mean() if not eligible.empty else np.nan
                    class_balance = f"positive_fraction={positive:.4f}" if not np.isnan(positive) else "empty"
                    variability_ok = bool(not np.isnan(positive) and 0.05 <= positive <= 0.95)
                sample_ok = bool(len(eligible) >= 50) if scope == "cross_system" else bool(len(eligible) >= 30)
                scope_ok = bool(leave_system) if scope == "cross_system" else True
                classical = bool(future_valid and source_ok and leave_source and scope_ok and variability_ok and sample_ok)
                recommended_role = "not_feasible"
                gate_reason = []
                if classical:
                    recommended_role = "primary_screening_target" if scope == "cross_system" else "within-system_target"
                else:
                    if future_valid and variability_ok:
                        recommended_role = "audit_only" if scope == "cross_system" else "secondary_target"
                    if not source_ok:
                        gate_reason.append("source_confounding")
                    if not leave_source:
                        gate_reason.append("leave_source_not_feasible")
                    if scope == "cross_system" and not scope_ok:
                        gate_reason.append("leave_system_not_feasible")
                    if not variability_ok:
                        gate_reason.append("class_balance_or_variance_insufficient")
                    if not sample_ok:
                        gate_reason.append("sample_size_insufficient")
                    if not future_valid:
                        gate_reason.append("no_future_window")
                rows.append(
                    {
                        "target_id": target_id,
                        "budget_id": budget_id,
                        "scope": scope,
                        "n_total": int(len(computable)),
                        "n_eligible": int(len(eligible)),
                        "n_sources": int(eligible["source_dataset"].nunique()) if not eligible.empty else 0,
                        "n_systems": int(eligible["system"].nunique()) if not eligible.empty else 0,
                        "class_balance_or_variance": class_balance,
                        "future_window_valid": future_valid,
                        "source_confounding_acceptable": source_ok,
                        "leave_source_feasible": leave_source,
                        "leave_system_feasible": leave_system,
                        "classical_baseline_feasible": classical,
                        "neural_model_feasible": False,
                        "recommended_role": recommended_role,
                        "gate_reason": "|".join(gate_reason) if gate_reason else "",
                    }
                )
    return pd.DataFrame(rows)


def run_screening_target_budget_eligibility(forms: pd.DataFrame, curves: pd.DataFrame, root: Path) -> ScreeningEligibilityResult:
    del root
    prepared_forms = _prepare_forms(forms)
    curve_meta, curve_points = _curve_tables(prepared_forms, curves)
    target_schema = _target_schema()
    target_eligibility = _curve_target_rows(target_schema, curve_meta, curve_points)
    budget_schema = _budget_schema()
    obs_budget = _budget_target_rows(target_eligibility, budget_schema, curve_meta, curve_points)
    future_coverage = _future_window_coverage(obs_budget, target_eligibility)
    comparability = _target_cross_system(target_schema, target_eligibility, curve_meta)
    confounding = _target_source_confounding(obs_budget, target_eligibility)
    schedule_audit = _schedule_audit(curve_meta)
    feasibility = _task_feasibility(obs_budget, target_eligibility, confounding)

    cross_system_available = bool(
        feasibility["scope"].eq("cross_system").any()
        and feasibility.loc[feasibility["scope"].eq("cross_system"), "classical_baseline_feasible"].any()
    )
    within_system_available = bool(
        feasibility["scope"].astype(str).str.startswith("within_system::").any()
        and feasibility.loc[feasibility["scope"].astype(str).str.startswith("within_system::"), "classical_baseline_feasible"].any()
    )
    leave_source_feasible = bool(feasibility["leave_source_feasible"].any())
    gate_summary = {
        "screening_targets_defined": not target_schema.empty,
        "future_only_windows_defined": True,
        "target_eligibility_available": not target_eligibility.empty,
        "budget_eligibility_available": not obs_budget.empty,
        "cross_system_target_available": cross_system_available,
        "within_system_targets_available": within_system_available,
        "leave_source_evaluation_feasible": leave_source_feasible,
        "classical_screening_entry_allowed": within_system_available or cross_system_available,
        "neural_entry_allowed": False,
    }
    stats = {
        "n_total_formulation_rows": int(len(prepared_forms)),
        "n_total_canonical_formulations": int(prepared_forms["canonical_formulation_id"].nunique()),
        "n_total_curves": int(curve_meta["curve_id"].nunique()),
        "n_valid_timepoints": int(curves["release_fraction"].notna().sum()),
        "n_duplicate_timepoint_curves": int(curve_meta["duplicate_timepoints_flag"].sum()),
        "n_nonmonotone_curves": int(curve_meta["nonmonotone_step_count"].gt(0).sum()),
        "targets_defined": sorted(target_schema["target_id"].astype(str).tolist()),
        "targets_rejected": sorted(target_schema[target_schema["status"].eq("not_supported")]["target_id"].astype(str).tolist()),
        "cross_system_targets_available": int(
            feasibility[
                feasibility["scope"].eq("cross_system") & feasibility["classical_baseline_feasible"]
            ]["target_id"].nunique()
        ),
        "within_system_targets_available": int(
            feasibility[
                feasibility["scope"].astype(str).str.startswith("within_system::")
                & feasibility["classical_baseline_feasible"]
            ]["target_id"].nunique()
        ),
        "n_source_bound_targets": int(confounding["source_bound_flag"].sum()),
        "n_leave_source_targets": int(confounding["leave_source_evaluation_feasible"].sum()),
        "n_primary_target_candidates": int(
            feasibility[
                feasibility["recommended_role"].isin({"primary_screening_target", "within-system_target"})
            ]["target_id"].nunique()
        ),
    }
    return ScreeningEligibilityResult(
        screening_target_schema=target_schema,
        formulation_target_eligibility=target_eligibility,
        observation_budget_eligibility=obs_budget,
        future_window_coverage=future_coverage,
        target_cross_system_comparability=comparability,
        target_source_confounding_audit=confounding,
        sampling_schedule_source_audit=schedule_audit,
        screening_task_feasibility_matrix=feasibility,
        gate_summary=gate_summary,
        stats=stats,
        budget_schema=budget_schema,
    )
