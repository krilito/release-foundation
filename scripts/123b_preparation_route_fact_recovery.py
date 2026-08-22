"""
123B - Preparation-route fact recovery helpers.

This module recovers only provenance-traceable preparation facts from local
source files and standardized formulation pools. It does not infer missing
routes from assay descriptors, dataset identity, or release-curve shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
import re
from typing import Any

import pandas as pd


PROVENANCE_COLUMNS = [
    "formulation_id",
    "source_dataset",
    "source_file",
    "source_sheet_or_table",
    "source_row_or_record",
    "source_id",
    "doi",
    "raw_field_name",
    "raw_process_text",
    "raw_text_span",
    "text_context",
    "route_family",
    "route_subfamily",
    "operation_index",
    "unit_operation",
    "operation_role",
    "reported_condition_name",
    "reported_condition_value",
    "reported_condition_unit",
    "material_or_solvent",
    "post_processing_flag",
    "fact_status",
    "mapping_confidence",
    "extraction_method",
    "review_status",
    "model_channel",
    "audit_note",
]

ROUTE_FACT_COLUMNS = [
    "formulation_id",
    "source_dataset",
    "source_id",
    "doi",
    "system",
    "drug",
    "material",
    "route_family",
    "route_subfamily",
    "unit_operation_sequence",
    "n_reported_operations",
    "n_unknown_operations",
    "reported_condition_count",
    "post_processing_sequence",
    "route_mapping_confidence",
    "route_fact_completeness",
    "multi_source_supported",
    "source_bound_route_flag",
    "eligible_for_R_model",
    "ineligibility_reason",
]

ALLOWED_FACT_STATUS = {
    "explicit_fact",
    "partial_fact",
    "ambiguous",
    "not_reported",
    "conflicting",
    "assay_only",
    "audit_only",
}
ALLOWED_CONFIDENCE = {"exact", "high", "medium", "low", "unresolved"}
ALLOWED_EXTRACTION_METHOD = {
    "structured_field",
    "raw_table_text",
    "manual_rule",
    "source_metadata",
    "document_text",
    "legacy_mapping",
}
ALLOWED_MODEL_CHANNEL = {"R_model", "R_audit", "excluded"}

ASSAY_LIKE_FIELDS = [
    "release_method",
    "phase_sequence",
    "measurement_assay",
    "experimental_panel",
    "source_sheet",
    "worksheet_title",
    "source_label_evidence",
    "origin_project_object",
]

CASEINATE_DATASET = "caseinate_gallic_mendeley_8d3973kgb3"
PLGA_DOX_DATASET = "plga_dox_mendeley_h4tt4433w9"


@dataclass
class RouteRecoveryResult:
    provenance: pd.DataFrame
    taxonomy_v2: pd.DataFrame
    route_facts: pd.DataFrame
    coverage: pd.DataFrame
    confounding: pd.DataFrame
    unresolved_queue: pd.DataFrame
    gate_summary: dict[str, Any]
    stats: dict[str, Any]
    raw_sources_inspected: list[str]


def _blank_row() -> dict[str, Any]:
    return {column: "" for column in PROVENANCE_COLUMNS}


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


def _system_value(row: pd.Series) -> str:
    for column in ["polymer_family", "structure_type", "particle_architecture"]:
        if column in row and _field_present(row[column]):
            return str(row[column]).strip()
    return "system_not_reported"


def _drug_value(row: pd.Series) -> str:
    for column in ["payload_name", "API_name"]:
        if column in row and _field_present(row[column]):
            return str(row[column]).strip()
    return "drug_not_reported"


def _material_value(row: pd.Series) -> str:
    if "polymer_family" in row and _field_present(row["polymer_family"]):
        return str(row["polymer_family"]).strip()
    return "material_not_reported"


def _confidence_rank(value: str) -> int:
    order = {"unresolved": 0, "low": 1, "medium": 2, "high": 3, "exact": 4}
    return order.get(str(value), -1)


def _worst_confidence(values: list[str]) -> str:
    if not values:
        return "unresolved"
    return min(values, key=_confidence_rank)


def _sequence_text(route_rows: pd.DataFrame) -> str:
    if route_rows.empty:
        return "operation_sequence_partial"
    view = (
        route_rows.dropna(subset=["operation_index"])
        .sort_values("operation_index")
        [["operation_index", "unit_operation"]]
        .drop_duplicates()
    )
    ops = [str(value).strip() for value in view["unit_operation"] if _field_present(value)]
    return " -> ".join(ops) if ops else "operation_sequence_partial"


def _post_processing_text(route_rows: pd.DataFrame) -> str:
    flags = route_rows[route_rows["post_processing_flag"].astype(str).eq("true")]
    ops = (
        flags.sort_values("operation_index")["unit_operation"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    return " -> ".join(ops) if ops else "not_reported"


def _make_route_row(
    base_row: pd.Series,
    *,
    source_file: str,
    source_sheet_or_table: str,
    source_row_or_record: str,
    raw_field_name: str,
    raw_process_text: str,
    raw_text_span: str,
    text_context: str,
    route_family: str,
    route_subfamily: str,
    operation_index: int | str,
    unit_operation: str,
    operation_role: str,
    reported_condition_name: str,
    reported_condition_value: str,
    reported_condition_unit: str,
    material_or_solvent: str,
    post_processing_flag: bool,
    fact_status: str,
    mapping_confidence: str,
    extraction_method: str,
    review_status: str,
    model_channel: str,
    audit_note: str,
    doi: str = "",
) -> dict[str, Any]:
    row = _blank_row()
    row.update(
        {
            "formulation_id": str(base_row["unified_curve_id"]),
            "source_dataset": str(base_row["source_dataset"]),
            "source_file": source_file,
            "source_sheet_or_table": source_sheet_or_table,
            "source_row_or_record": source_row_or_record,
            "source_id": _canonical_source_id(base_row),
            "doi": doi,
            "raw_field_name": raw_field_name,
            "raw_process_text": raw_process_text,
            "raw_text_span": raw_text_span,
            "text_context": text_context,
            "route_family": route_family,
            "route_subfamily": route_subfamily,
            "operation_index": operation_index,
            "unit_operation": unit_operation,
            "operation_role": operation_role,
            "reported_condition_name": reported_condition_name,
            "reported_condition_value": reported_condition_value,
            "reported_condition_unit": reported_condition_unit,
            "material_or_solvent": material_or_solvent,
            "post_processing_flag": "true" if post_processing_flag else "false",
            "fact_status": fact_status,
            "mapping_confidence": mapping_confidence,
            "extraction_method": extraction_method,
            "review_status": review_status,
            "model_channel": model_channel,
            "audit_note": audit_note,
        }
    )
    return row


def _caseinate_rows(forms: pd.DataFrame, root: Path) -> tuple[list[dict[str, Any]], str]:
    doc_path = root / "data/external/8d3973kgb3_mendeley/Antioxidant capacity and release kinetics of active film based on sodium caseinate and gallic acid/Draft Rehan et al. repository data.docx"
    if not doc_path.exists():
        return [], str(doc_path)
    import docx

    document = docx.Document(doc_path)
    paragraph = ""
    for para in document.paragraphs:
        text = " ".join(str(para.text).split())
        if "Caseinate films were prepared by solution casting method" in text:
            paragraph = text
            break
    if not paragraph:
        return [], str(doc_path)

    specs = [
        (1, "active_dissolution", "gallic acid dissolved in freshly prepared 50 ml Tris buffer (0.02 M, pH: 8) at room temperature for 45 min under continuous stirring", "gallic acid|Tris buffer"),
        (2, "matrix_dissolution", "10 g of sodium caseinate and glycerol (10% w/w of caseinate) were dissolved in Tris buffer (75 ml) at 68±2°C under stirring for 2 h", "sodium caseinate|glycerol|Tris buffer"),
        (3, "blending", "the film forming solution was allowed to cool at room temperature, followed by the incorporation of gallic acid solution and stirring", "gallic acid solution"),
        (4, "casting", "FFS was casted onto the petri plates (120×120 mm)", "petri plates"),
        (5, "drying", "allowed to dry for 24 h in the climatic chamber at 57±2% relative humidity and 30°C", "climatic chamber"),
    ]
    condition_rows = [
        (1, "tris_buffer_volume", "50", "ml", "Tris buffer"),
        (1, "tris_buffer_molarity", "0.02", "M", "Tris buffer"),
        (1, "tris_buffer_ph", "8", "pH", "Tris buffer"),
        (1, "mix_time", "45", "min", "gallic acid solution"),
        (1, "mix_temperature", "room_temperature", "", "gallic acid solution"),
        (2, "caseinate_mass", "10", "g", "sodium caseinate"),
        (2, "glycerol_loading", "10", "% w/w", "glycerol"),
        (2, "tris_buffer_volume", "75", "ml", "Tris buffer"),
        (2, "mix_temperature", "68±2", "°C", "sodium caseinate solution"),
        (2, "mix_time", "2", "h", "sodium caseinate solution"),
        (4, "casting_surface_size", "120×120", "mm", "petri plates"),
        (5, "dry_time", "24", "h", "climatic chamber"),
        (5, "dry_temperature", "30", "°C", "climatic chamber"),
        (5, "relative_humidity", "57±2", "%", "climatic chamber"),
    ]

    rows: list[dict[str, Any]] = []
    dataset_rows = forms[forms["source_dataset"].eq(CASEINATE_DATASET)]
    for _, base_row in dataset_rows.iterrows():
        for operation_index, operation_role, snippet, material_text in specs:
            rows.append(
                _make_route_row(
                    base_row,
                    source_file=str(doc_path.relative_to(root)),
                    source_sheet_or_table="paragraph:2.1.Film formulation",
                    source_row_or_record="2.1.Film formulation",
                    raw_field_name="document_paragraph",
                    raw_process_text=paragraph,
                    raw_text_span=snippet,
                    text_context="preparation_method",
                    route_family="solvent_casting",
                    route_subfamily="aqueous_solution_casting",
                    operation_index=operation_index,
                    unit_operation={
                        1: "active_dissolution",
                        2: "matrix_dissolution",
                        3: "blending",
                        4: "casting",
                        5: "drying",
                    }[operation_index],
                    operation_role=operation_role,
                    reported_condition_name="",
                    reported_condition_value="",
                    reported_condition_unit="",
                    material_or_solvent=material_text,
                    post_processing_flag=operation_index == 5,
                    fact_status="explicit_fact",
                    mapping_confidence="exact",
                    extraction_method="document_text",
                    review_status="needs_quick_human_spotcheck",
                    model_channel="R_model",
                    audit_note="Explicit formulation-method paragraph provides route sequence and conditions.",
                )
            )
        for operation_index, condition_name, value, unit, material_text in condition_rows:
            rows.append(
                _make_route_row(
                    base_row,
                    source_file=str(doc_path.relative_to(root)),
                    source_sheet_or_table="paragraph:2.1.Film formulation",
                    source_row_or_record="2.1.Film formulation",
                    raw_field_name="document_paragraph",
                    raw_process_text=paragraph,
                    raw_text_span=condition_name,
                    text_context="reported_condition",
                    route_family="solvent_casting",
                    route_subfamily="aqueous_solution_casting",
                    operation_index=operation_index,
                    unit_operation={
                        1: "active_dissolution",
                        2: "matrix_dissolution",
                        3: "blending",
                        4: "casting",
                        5: "drying",
                    }[operation_index],
                    operation_role="reported_condition",
                    reported_condition_name=condition_name,
                    reported_condition_value=value,
                    reported_condition_unit=unit,
                    material_or_solvent=material_text,
                    post_processing_flag=operation_index == 5,
                    fact_status="explicit_fact",
                    mapping_confidence="exact",
                    extraction_method="document_text",
                    review_status="needs_quick_human_spotcheck",
                    model_channel="R_model",
                    audit_note="Reported numeric or categorical preparation condition from the local source document.",
                )
            )
    return rows, str(doc_path)


def _plga_dox_rows(forms: pd.DataFrame, root: Path) -> tuple[list[dict[str, Any]], str]:
    source_root = root / "data/external/h4tt4433w9_mendeley/extracted"
    if not source_root.exists():
        return [], str(source_root)
    title_dirs = [path for path in source_root.iterdir() if path.is_dir()]
    if not title_dirs:
        return [], str(source_root)
    title = title_dirs[0].name
    source_file = str(title_dirs[0].relative_to(root))
    route_text = title
    rows: list[dict[str, Any]] = []
    dataset_rows = forms[
        forms["source_dataset"].eq(PLGA_DOX_DATASET)
        & forms.get("is_free_drug_control", pd.Series(False, index=forms.index)).astype(str).ne("True")
    ]
    for _, base_row in dataset_rows.iterrows():
        rows.append(
            _make_route_row(
                base_row,
                source_file=source_file,
                source_sheet_or_table="directory_title",
                source_row_or_record=title_dirs[0].name,
                raw_field_name="source_directory_name",
                raw_process_text=route_text,
                raw_text_span="prepared by coaxial electrospraying",
                text_context="source_title_metadata",
                route_family="electrospraying",
                route_subfamily="coaxial_electrospraying",
                operation_index=1,
                unit_operation="coaxial_electrospraying",
                operation_role="particle_formation",
                reported_condition_name="",
                reported_condition_value="",
                reported_condition_unit="",
                material_or_solvent="not_reported",
                post_processing_flag=False,
                fact_status="explicit_fact",
                mapping_confidence="high",
                extraction_method="document_text",
                review_status="needs_human_confirmation_if_used_for_public_claim",
                model_channel="R_model",
                audit_note="Source title explicitly states the nanoparticle preparation route, but no local method body was recovered here.",
            )
        )
    return rows, source_file


def _assay_only_rows(forms: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, base_row in forms.iterrows():
        for field in ASSAY_LIKE_FIELDS:
            if field not in base_row or not _field_present(base_row[field]):
                continue
            text = str(base_row[field]).strip()
            rows.append(
                _make_route_row(
                    base_row,
                    source_file="outputs/149_release_caveat_augmented_cumulative_v1/formulations.csv",
                    source_sheet_or_table="formulations",
                    source_row_or_record=str(base_row["unified_curve_id"]),
                    raw_field_name=field,
                    raw_process_text=text,
                    raw_text_span=text,
                    text_context="standardized_pool_field",
                    route_family="route_family_unknown",
                    route_subfamily="route_subfamily_unknown",
                    operation_index="",
                    unit_operation="not_a_preparation_route",
                    operation_role="assay_or_audit_candidate",
                    reported_condition_name="",
                    reported_condition_value="",
                    reported_condition_unit="",
                    material_or_solvent="",
                    post_processing_flag=False,
                    fact_status="assay_only" if field in {"release_method", "phase_sequence", "measurement_assay"} else "audit_only",
                    mapping_confidence="exact",
                    extraction_method="structured_field",
                    review_status="auto_classified",
                    model_channel="excluded",
                    audit_note="Standardized field is release-assay or audit metadata and is therefore banned from R_model.",
                )
            )
    return rows


def _unresolved_rows(forms: pd.DataFrame, resolved_ids: set[str], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_hints = {
        "liposome_ivr209": "Local backend metadata exists, but it contains release-assay and formulation-state fields only; local preparation methods are absent.",
        "cross321": "Standardized PLGA curve corpus has descriptors but no local per-formulation preparation-method text.",
        "internal181": "Standardized PLGA descriptor corpus has group labels but no local per-formulation preparation-method text.",
        CASEINATE_DATASET: "Resolved from local document text.",
        PLGA_DOX_DATASET: "Partially resolved from local source-title text for nanoparticle rows only.",
    }
    for _, base_row in forms.iterrows():
        formulation_id = str(base_row["unified_curve_id"])
        if formulation_id in resolved_ids:
            continue
        source_dataset = str(base_row["source_dataset"])
        note = source_hints.get(source_dataset, "No formulation-level preparation method was recovered from the local standardized pool or inspected local source files.")
        rows.append(
            _make_route_row(
                base_row,
                source_file="outputs/149_release_caveat_augmented_cumulative_v1/formulations.csv",
                source_sheet_or_table="formulations",
                source_row_or_record=formulation_id,
                raw_field_name="preparation_route_missing",
                raw_process_text="route_unresolved",
                raw_text_span="route_unresolved",
                text_context="missing_local_preparation_fact",
                route_family="route_family_unknown",
                route_subfamily="route_subfamily_unknown",
                operation_index="",
                unit_operation="operation_sequence_partial",
                operation_role="missing_route_fact",
                reported_condition_name="source_document_status",
                reported_condition_value="source_document_required",
                reported_condition_unit="",
                material_or_solvent="solvent_not_reported",
                post_processing_flag=False,
                fact_status="not_reported",
                mapping_confidence="unresolved",
                extraction_method="legacy_mapping",
                review_status="needs_source_recovery",
                model_channel="excluded",
                audit_note=note,
            )
        )
    return rows


def _taxonomy_rows() -> pd.DataFrame:
    rows = [
        ("emulsion_solvent_removal", "single_emulsion", "emulsification", "particle_formation", "solvent|surfactant|mixing_rate|phase_ratio|temperature|unknown_condition", "polymeric_particle", "single emulsion|o/w emulsion", "release_dialysis|sample_and_separate|assay_filtration", "Emulsion-mediated particle formation followed by solvent removal."),
        ("emulsion_solvent_removal", "single_emulsion", "solvent_evaporation", "solvent_removal", "time|temperature|pressure|unknown_condition", "polymeric_particle", "evaporation|solvent evaporation", "release_medium_evaporation", "Removal of volatile organic solvent after emulsification."),
        ("nanoprecipitation", "classical_nanoprecipitation", "antisolvent_addition", "particle_formation", "solvent|antisolvent|addition_rate|temperature|unknown_condition", "polymeric_particle", "solvent displacement|nanoprecipitation", "release_medium_exchange", "Particle formation by solvent displacement into an antisolvent."),
        ("spray_drying", "generic_spray_drying", "spray_drying", "particle_formation", "inlet_temperature|feed_rate|atomization|unknown_condition", "powder_particle", "spray drying", "assay_drying_step", "Atomized drying-based particle or powder formation."),
        ("thin_film_hydration", "generic_thin_film_hydration", "lipid_film_formation", "film_formation", "solvent|evaporation|temperature|unknown_condition", "liposome", "thin film hydration|lipid film", "release_film_dissolution", "Lipid film creation before hydration."),
        ("thin_film_hydration", "generic_thin_film_hydration", "hydration", "vesicle_formation", "buffer|temperature|time|unknown_condition", "liposome", "hydration", "release_medium_hydration", "Hydration of a dried film to form vesicles."),
        ("extrusion", "liposome_extrusion", "extrusion", "size_control", "membrane_size|passes|temperature|unknown_condition", "liposome", "extrusion", "release_assay_filter_exchange", "Membrane extrusion used to narrow vesicle size distribution."),
        ("crosslinking", "enzymatic_crosslinking", "precursor_mixing", "precursor_preparation", "polymer|payload|buffer|unknown_condition", "hydrogel", "precursor mixing", "assay_mix_before_readout", "Precursor preparation before gelation."),
        ("crosslinking", "enzymatic_crosslinking", "enzymatic_crosslinking", "network_formation", "enzyme|time|temperature|unknown_condition", "hydrogel", "enzymatically crosslinked|enzymatic crosslinking", "release_crosslink_breakdown_hypothesis", "Network formation driven by an enzymatic crosslinking chemistry."),
        ("casting", "generic_casting", "casting", "shape_setting", "surface|thickness|unknown_condition", "film", "casting", "assay_plate_casting", "Casting of a formulation solution into a shaped film or layer."),
        ("solvent_casting", "aqueous_solution_casting", "active_dissolution", "active_solution_preparation", "solvent|time|temperature|mixing|unknown_condition", "film", "solution casting|solvent casting", "release_simulant_dissolution", "Preparation of the active solution before casting."),
        ("solvent_casting", "aqueous_solution_casting", "matrix_dissolution", "matrix_solution_preparation", "polymer_mass|plasticizer|temperature|time|unknown_condition", "film", "solution casting|solvent casting", "release_simulant_dissolution", "Preparation of the carrier matrix solution before casting."),
        ("solvent_casting", "aqueous_solution_casting", "blending", "blend_finalization", "time|temperature|unknown_condition", "film", "blending", "assay_blending", "Combination of active and matrix solutions before casting."),
        ("solvent_casting", "aqueous_solution_casting", "casting", "shape_setting", "surface|dimensions|unknown_condition", "film", "casting", "assay_plate_casting", "Casting of the final film-forming solution."),
        ("solvent_casting", "aqueous_solution_casting", "drying", "post_processing", "time|temperature|humidity|unknown_condition", "film", "drying", "release_sample_drying_before_weighing", "Controlled drying after film casting."),
        ("freeze_drying", "generic_lyophilization", "freezing", "pre_lyophilization", "temperature|time|unknown_condition", "powder_or_hydrogel", "freezing", "assay_sample_freezing", "Freeze step before lyophilization."),
        ("freeze_drying", "generic_lyophilization", "lyophilization", "post_processing", "pressure|time|temperature|unknown_condition", "powder_or_hydrogel", "freeze drying|lyophilization", "assay_freeze_thaw_cycle", "Water removal by lyophilization."),
        ("compression", "direct_compression", "compression", "tablet_formation", "pressure|dwell_time|unknown_condition", "tablet", "compression|direct compression", "release_test_apparatus_pressure", "Powder compaction into tablets."),
        ("coating", "film_coating", "coating", "surface_processing", "coating_solution|time|unknown_condition", "tablet_or_particle", "coating", "assay_plate_coating", "Post-formation surface coating step."),
        ("self_assembly", "spontaneous_self_assembly", "self_assembly", "particle_formation", "solvent|buffer|temperature|unknown_condition", "nanocarrier", "self assembly", "release_self_quenching_assay", "Carrier formation through spontaneous self-assembly."),
        ("microfluidic_mixing", "focused_flow_mixing", "microfluidic_mixing", "particle_formation", "flow_ratio|total_flow_rate|unknown_condition", "nanocarrier", "microfluidic mixing", "release_microfluidic_assay", "Carrier formation using controlled microfluidic mixing."),
        ("ionic_gelation", "alginate_ionic_gelation", "solution_preparation", "precursor_preparation", "polymer|payload|unknown_condition", "bead_or_hydrogel", "ionic gelation|alginate beads", "release_buffer_phase_switch", "Preparation of the pre-gel solution."),
        ("ionic_gelation", "alginate_ionic_gelation", "ionic_gelation", "network_formation", "crosslinker|time|unknown_condition", "bead_or_hydrogel", "ionic gelation|crosslinking bath", "release_dialysis", "Network formation by ionic crosslinking."),
        ("electrospraying", "coaxial_electrospraying", "coaxial_electrospraying", "particle_formation", "voltage|flow_rate|collector_distance|unknown_condition", "polymeric_particle", "coaxial electrospraying|electrospraying", "assay_electrode_setup", "Particle formation by coaxial electrospraying."),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "route_family",
            "route_subfamily",
            "unit_operation",
            "operation_order_role",
            "condition_schema",
            "system_scope",
            "allowed_synonyms",
            "excluded_assay_contexts",
            "definition",
        ],
    )
    return frame


def _formulation_route_facts(forms: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    r_model = provenance[provenance["model_channel"].eq("R_model")].copy()
    for _, base_row in forms.iterrows():
        formulation_id = str(base_row["unified_curve_id"])
        subset = r_model[r_model["formulation_id"].eq(formulation_id)].copy()
        if subset.empty:
            rows.append(
                {
                    "formulation_id": formulation_id,
                    "source_dataset": str(base_row["source_dataset"]),
                    "source_id": _canonical_source_id(base_row),
                    "doi": "",
                    "system": _system_value(base_row),
                    "drug": _drug_value(base_row),
                    "material": _material_value(base_row),
                    "route_family": "route_family_unknown",
                    "route_subfamily": "route_subfamily_unknown",
                    "unit_operation_sequence": "operation_sequence_partial",
                    "n_reported_operations": 0,
                    "n_unknown_operations": 1,
                    "reported_condition_count": 0,
                    "post_processing_sequence": "not_reported",
                    "route_mapping_confidence": "unresolved",
                    "route_fact_completeness": "source_document_required",
                    "multi_source_supported": False,
                    "source_bound_route_flag": False,
                    "eligible_for_R_model": False,
                    "ineligibility_reason": "route_unresolved_or_not_reported",
                }
            )
            continue

        ops = subset[["operation_index", "unit_operation"]].drop_duplicates()
        confidences = subset["mapping_confidence"].astype(str).tolist()
        condition_count = int(subset["reported_condition_name"].astype(str).str.strip().ne("").sum())
        sequence = _sequence_text(subset)
        completeness = "reported_multi_step_sequence" if int(ops["operation_index"].nunique()) >= 2 else "partial_sequence"
        eligible = bool(
            subset["fact_status"].isin({"explicit_fact", "partial_fact"}).all()
            and subset["mapping_confidence"].isin({"exact", "high", "medium"}).all()
        )
        rows.append(
            {
                "formulation_id": formulation_id,
                "source_dataset": str(base_row["source_dataset"]),
                "source_id": _canonical_source_id(base_row),
                "doi": str(subset["doi"].dropna().astype(str).replace("", pd.NA).dropna().iloc[0]) if subset["doi"].astype(str).str.strip().ne("").any() else "",
                "system": _system_value(base_row),
                "drug": _drug_value(base_row),
                "material": _material_value(base_row),
                "route_family": str(subset["route_family"].iloc[0]),
                "route_subfamily": str(subset["route_subfamily"].iloc[0]),
                "unit_operation_sequence": sequence,
                "n_reported_operations": int(ops["operation_index"].nunique()),
                "n_unknown_operations": 0 if sequence != "operation_sequence_partial" else 1,
                "reported_condition_count": condition_count,
                "post_processing_sequence": _post_processing_text(subset),
                "route_mapping_confidence": _worst_confidence(confidences),
                "route_fact_completeness": completeness,
                "multi_source_supported": False,
                "source_bound_route_flag": False,
                "eligible_for_R_model": eligible,
                "ineligibility_reason": "" if eligible else "mapping_confidence_insufficient",
            }
        )
    return pd.DataFrame(rows, columns=ROUTE_FACT_COLUMNS)


def _source_entropy(series: pd.Series) -> float:
    values = series.astype(str).value_counts(normalize=True)
    if values.empty:
        return 0.0
    return float(-sum(p * log(p, 2) for p in values if p > 0))


def _route_source_confounding(route_facts: pd.DataFrame) -> pd.DataFrame:
    eligible = route_facts[route_facts["eligible_for_R_model"]].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "route_id",
                "n_formulations",
                "n_sources",
                "n_dois",
                "n_drugs",
                "n_materials",
                "n_systems",
                "max_source_fraction",
                "source_entropy",
                "source_bound_route_flag",
                "cross_source_claim_allowed",
            ]
        )

    eligible["route_id"] = eligible["route_family"].astype(str) + "::" + eligible["route_subfamily"].astype(str)
    rows: list[dict[str, Any]] = []
    for route_id, df in eligible.groupby("route_id", sort=True):
        source_fraction = df["source_dataset"].astype(str).value_counts(normalize=True)
        max_fraction = float(source_fraction.max()) if not source_fraction.empty else 1.0
        n_sources = int(df["source_dataset"].nunique())
        n_dois = int(df["doi"].replace("", pd.NA).dropna().nunique())
        source_bound = n_sources <= 1 or max_fraction >= 0.85
        cross_source = bool(n_sources >= 2 and n_dois >= 2 and not source_bound)
        rows.append(
            {
                "route_id": route_id,
                "n_formulations": int(len(df)),
                "n_sources": n_sources,
                "n_dois": n_dois,
                "n_drugs": int(df["drug"].nunique()),
                "n_materials": int(df["material"].nunique()),
                "n_systems": int(df["system"].nunique()),
                "max_source_fraction": max_fraction,
                "source_entropy": _source_entropy(df["source_dataset"]),
                "source_bound_route_flag": source_bound,
                "cross_source_claim_allowed": cross_source,
            }
        )
    return pd.DataFrame(rows)


def _apply_confounding_flags(route_facts: pd.DataFrame, confounding: pd.DataFrame) -> pd.DataFrame:
    updated = route_facts.copy()
    if confounding.empty:
        updated.loc[updated["eligible_for_R_model"], "source_bound_route_flag"] = True
        return updated
    conf_map = confounding.set_index("route_id")
    route_ids = updated["route_family"].astype(str) + "::" + updated["route_subfamily"].astype(str)
    updated["multi_source_supported"] = (
        route_ids.map(conf_map["cross_source_claim_allowed"]).astype("boolean").fillna(False).astype(bool)
    )
    updated["source_bound_route_flag"] = (
        route_ids.map(conf_map["source_bound_route_flag"]).astype("boolean").fillna(False).astype(bool)
    )
    blocked = updated["eligible_for_R_model"] & updated["source_bound_route_flag"]
    updated.loc[blocked, "ineligibility_reason"] = updated.loc[blocked, "ineligibility_reason"].replace("", "single_source_route_support")
    return updated


def _coverage_frame(route_facts: pd.DataFrame) -> pd.DataFrame:
    known = route_facts["eligible_for_R_model"] | route_facts["route_family"].ne("route_family_unknown")
    rows: list[dict[str, Any]] = []

    groupings = [
        ("overall", [], "all"),
        ("source_dataset", ["source_dataset"], None),
        ("system", ["system"], None),
        ("route_family", ["route_family"], None),
        ("source_id_doi", ["source_id", "doi"], None),
        ("drug", ["drug"], None),
        ("material", ["material"], None),
    ]

    for level, group_cols, fixed_label in groupings:
        if not group_cols:
            groups = [(fixed_label, route_facts)]
        else:
            groups = route_facts.groupby(group_cols, dropna=False, sort=True)
        for key, df in groups:
            label = fixed_label if fixed_label is not None else "|".join([str(v) for v in (key if isinstance(key, tuple) else (key,))])
            with_route = df[known.loc[df.index]]
            with_sequence = with_route[with_route["unit_operation_sequence"].ne("operation_sequence_partial")]
            with_conditions = with_route[with_route["reported_condition_count"].gt(0)]
            route_known = with_route[with_route["route_family"].ne("route_family_unknown")]
            if not route_known.empty:
                per_route = route_known.groupby("route_family", sort=True)
                n_sources_per_route = per_route["source_dataset"].nunique().mean()
                n_drugs_per_route = per_route["drug"].nunique().mean()
                n_materials_per_route = per_route["material"].nunique().mean()
                n_systems_per_route = per_route["system"].nunique().mean()
                single_source_fraction = float((per_route["source_dataset"].nunique() <= 1).mean())
                source_determined_fraction = float(route_known["source_bound_route_flag"].mean())
                low_conf_fraction = float(route_known["route_mapping_confidence"].isin({"low", "unresolved"}).mean())
            else:
                n_sources_per_route = 0.0
                n_drugs_per_route = 0.0
                n_materials_per_route = 0.0
                n_systems_per_route = 0.0
                single_source_fraction = 0.0
                source_determined_fraction = 0.0
                low_conf_fraction = 0.0

            rows.append(
                {
                    "audit_level": level,
                    "audit_key": label,
                    "n_formulations": int(len(df)),
                    "n_with_any_route_fact": int(len(with_route)),
                    "route_fact_coverage": float(len(with_route) / max(len(df), 1)),
                    "n_with_sequence": int(len(with_sequence)),
                    "sequence_coverage": float(len(with_sequence) / max(len(df), 1)),
                    "n_with_conditions": int(len(with_conditions)),
                    "condition_coverage": float(len(with_conditions) / max(len(df), 1)),
                    "n_unique_route_families": int(route_known["route_family"].nunique()),
                    "n_sources_per_route": float(n_sources_per_route),
                    "n_drugs_per_route": float(n_drugs_per_route),
                    "n_materials_per_route": float(n_materials_per_route),
                    "n_systems_per_route": float(n_systems_per_route),
                    "fraction_single_source_routes": single_source_fraction,
                    "fraction_source_determined": source_determined_fraction,
                    "fraction_low_confidence": low_conf_fraction,
                    "fraction_assay_only_candidates": float(
                        df["ineligibility_reason"].astype(str).str.contains("assay", regex=False).mean()
                    )
                    if "ineligibility_reason" in df.columns
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _unresolved_queue(forms: pd.DataFrame, route_facts: pd.DataFrame) -> pd.DataFrame:
    merged = forms[["unified_curve_id", "source_dataset", "payload_name", "polymer_family"]].merge(
        route_facts[["formulation_id", "eligible_for_R_model", "route_family", "ineligibility_reason"]],
        left_on="unified_curve_id",
        right_on="formulation_id",
        how="left",
    )
    unresolved = merged[
        merged["eligible_for_R_model"].fillna(False).eq(False)
        | merged["route_family"].fillna("route_family_unknown").eq("route_family_unknown")
    ].copy()
    rows: list[dict[str, Any]] = []
    for dataset, df in unresolved.groupby("source_dataset", sort=True):
        reasons = sorted({str(value) for value in df["ineligibility_reason"].fillna("") if str(value).strip()})
        rows.append(
            {
                "source_dataset": dataset,
                "n_formulations": int(len(df)),
                "payloads": "|".join(sorted({str(v) for v in df["payload_name"].fillna("") if str(v).strip()})) or "not_reported",
                "materials": "|".join(sorted({str(v) for v in df["polymer_family"].fillna("") if str(v).strip()})) or "not_reported",
                "route_status": "route_unresolved",
                "blocking_reason": "|".join(reasons) if reasons else "source_document_required",
                "required_next_action": "recover_local_method_text_or_supplementary_route_fields",
            }
        )
    return pd.DataFrame(rows)


def run_preparation_route_fact_recovery(forms: pd.DataFrame, root: Path) -> RouteRecoveryResult:
    inspected = [
        "outputs/149_release_caveat_augmented_cumulative_v1/formulations.csv",
        "outputs/148_release_main_cumulative_v1/formulations.csv",
        "outputs/123_release_corpus_v1/formulations.csv",
        "data/formulations.csv",
        "data/external/accelerated_IVR/repo/accelerated_IVR-main/data/unprocessed/backend_data.csv",
        "data/external/nanomed_IVR_data/repo/nanomed_IVR_data-main/data/unprocessed/backend_data.csv",
        "data/external/8d3973kgb3_mendeley/Antioxidant capacity and release kinetics of active film based on sodium caseinate and gallic acid/Draft Rehan et al. repository data.docx",
        "data/external/8d3973kgb3_mendeley/Antioxidant capacity and release kinetics of active film based on sodium caseinate and gallic acid/Data repository.xlsx",
        "data/external/h4tt4433w9_mendeley/drug_release_profile.xlsx",
        "data/external/alginate_microbeads_mendeley_hgtphykjnb/Release_Acid and Buffer.xlsx",
        "data/external/kbwcw7w4rn_mendeley/extracted/Research Data/Data2.xlsx",
        "data/external/wtnjj6smjd_mendeley/Roser Posada Data.xlsx",
        "data/external/cs6x4f86f2_mendeley/Data on heparinised poloxamer hydrogel with hyaluronic acid can stabilise and sustain the release of bioactive NT-3/Data for manuscript.xlsx",
        "outputs/127_release_data_asset_registry/asset_registry.csv",
        "outputs/128_external_source_intake_registry/source_intake_registry.csv",
        "outputs/147_layered_release_corpus_registry/layered_asset_registry.csv",
    ]

    caseinate_rows, caseinate_source = _caseinate_rows(forms, root)
    plga_dox_rows, plga_dox_source = _plga_dox_rows(forms, root)
    assay_rows = _assay_only_rows(forms)
    resolved_ids = {row["formulation_id"] for row in caseinate_rows + plga_dox_rows}
    unresolved_rows = _unresolved_rows(forms, resolved_ids, root)

    provenance = pd.DataFrame(caseinate_rows + plga_dox_rows + assay_rows + unresolved_rows, columns=PROVENANCE_COLUMNS)
    taxonomy_v2 = _taxonomy_rows()
    route_facts = _formulation_route_facts(forms, provenance)
    confounding = _route_source_confounding(route_facts)
    route_facts = _apply_confounding_flags(route_facts, confounding)
    coverage = _coverage_frame(route_facts)
    unresolved_queue = _unresolved_queue(forms, route_facts)

    eligible = route_facts[route_facts["eligible_for_R_model"]].copy()
    cross_source_routes = confounding[confounding["cross_source_claim_allowed"]] if not confounding.empty else confounding
    gate_summary = {
        "route_fact_channel_exists": not provenance.empty,
        "formulation_level_route_facts_available": bool(len(eligible) > 0),
        "multi_source_route_support_available": bool(not cross_source_routes.empty),
        "route_source_confounding_acceptable": bool(not confounding.empty and confounding["source_bound_route_flag"].eq(False).all()),
        "leave_source_evaluation_feasible": bool(not cross_source_routes.empty and int(cross_source_routes["route_id"].nunique()) >= 2),
        "route_model_allowed": False,
        "neural_entry_allowed": False,
    }
    stats = {
        "n_formulations_total": int(len(route_facts)),
        "n_formulations_with_any_route_fact": int(route_facts["route_family"].ne("route_family_unknown").sum()),
        "n_formulations_with_sequence": int(route_facts["unit_operation_sequence"].ne("operation_sequence_partial").sum()),
        "n_formulations_with_conditions": int(route_facts["reported_condition_count"].gt(0).sum()),
        "n_route_families": int(route_facts[route_facts["route_family"].ne("route_family_unknown")]["route_family"].nunique()),
        "n_multi_source_routes": int(cross_source_routes["route_id"].nunique()) if not confounding.empty else 0,
        "n_single_source_routes": int(confounding[confounding["source_bound_route_flag"]]["route_id"].nunique()) if not confounding.empty else 0,
        "n_r_model_eligible_formulations": int(len(eligible)),
        "n_unresolved_formulations": int(route_facts["route_family"].eq("route_family_unknown").sum()),
        "n_assay_only_rejected_formulations": int(
            provenance[provenance["fact_status"].eq("assay_only")]["formulation_id"].nunique()
        ),
        "caseinate_source_path": caseinate_source,
        "plga_dox_source_path": plga_dox_source,
    }
    return RouteRecoveryResult(
        provenance=provenance,
        taxonomy_v2=taxonomy_v2,
        route_facts=route_facts,
        coverage=coverage,
        confounding=confounding,
        unresolved_queue=unresolved_queue,
        gate_summary=gate_summary,
        stats=stats,
        raw_sources_inspected=inspected,
    )
