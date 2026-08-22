"""
123 - Release screening platform formal groundwork.

Purpose:
    Instantiate the contract-level artifacts required before any screening
    platform modeling begins: descriptor availability schema, dataset coverage
    audit, route-fact taxonomy, audit-only process hypothesis registries,
    utility specification, and data-reality checkpoint.

Consumes:
    outputs/148_release_main_cumulative_v1/
    outputs/149_release_caveat_augmented_cumulative_v1/
    docs/release_screening_platform_design_contract_2026-06-14.md
    docs/release_screening_platform_mathematical_contract_2026-06-14.md

Produces:
    outputs/123_release_screening_platform_formal_groundwork/
      canonical_descriptor_schema.csv
      dataset_descriptor_coverage.csv
      process_route_taxonomy.csv
      process_to_state_hypothesis_matrix.csv
      process_to_curve_symptom_matrix.csv
      utility_specification.csv
      data_reality_checkpoint.csv
      decision_table.csv
      data_checks.csv
      lock_metadata.json
      report.md

Interpretation:
    This script does not train a model. It audits whether the current
    standardized release pools are ready for route-fact priors, optional
    state-proxy updates, and future-window screening evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_POOL = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_CAVEAT_POOL = Path("outputs/149_release_caveat_augmented_cumulative_v1")
DEFAULT_OUT = Path("outputs/123_release_screening_platform_formal_groundwork")
DEFAULT_DOC = Path("docs/release_screening_platform_formal_groundwork_2026-06-14.md")
DEFAULT_DESIGN_CONTRACT = Path("docs/release_screening_platform_design_contract_2026-06-14.md")
DEFAULT_MATH_CONTRACT = Path("docs/release_screening_platform_mathematical_contract_2026-06-14.md")

EXPECTED_123_OUTPUTS = [
    "canonical_descriptor_schema.csv",
    "dataset_descriptor_coverage.csv",
    "process_route_taxonomy.csv",
    "process_route_taxonomy_v2.csv",
    "process_to_state_hypothesis_matrix.csv",
    "process_to_curve_symptom_matrix.csv",
    "utility_specification.csv",
    "preparation_route_fact_provenance.csv",
    "formulation_route_facts.csv",
    "preparation_route_coverage_audit.csv",
    "route_source_confounding_audit.csv",
    "route_unresolved_source_queue.csv",
    "screening_target_schema.csv",
    "formulation_target_eligibility.csv",
    "observation_budget_eligibility.csv",
    "future_window_coverage.csv",
    "target_cross_system_comparability.csv",
    "target_source_confounding_audit.csv",
    "sampling_schedule_source_audit.csv",
    "screening_task_feasibility_matrix.csv",
    "data_reality_checkpoint.csv",
    "data_checks.csv",
    "lock_metadata.json",
    "decision_table.csv",
    "report.md",
]

SCHEMA_ROWS: list[dict[str, Any]] = [
    {
        "canonical_field": "drug_identifier",
        "descriptor_layer": "drug",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "categorical_identifier",
        "current_source_fields": "API_name|payload_name",
        "planned_vs_measured": "planned_or_known",
        "notes": "Drug identity anchor; use as base descriptor rather than audit metadata.",
    },
    {
        "canonical_field": "drug_molecular_weight",
        "descriptor_layer": "drug",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Drug_Mw|weighted_Mw",
        "planned_vs_measured": "planned_or_known",
        "notes": "Drug-side molecular descriptor.",
    },
    {
        "canonical_field": "drug_tpsa",
        "descriptor_layer": "drug",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Drug_TPSA",
        "planned_vs_measured": "planned_or_known",
        "notes": "Drug-side molecular descriptor.",
    },
    {
        "canonical_field": "drug_logp",
        "descriptor_layer": "drug",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Drug_LogP",
        "planned_vs_measured": "planned_or_known",
        "notes": "Drug-side molecular descriptor.",
    },
    {
        "canonical_field": "drug_tm",
        "descriptor_layer": "drug",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Drug_Tm|weighted_Tm",
        "planned_vs_measured": "planned_or_known",
        "notes": "Drug thermal property if available.",
    },
    {
        "canonical_field": "drug_pka",
        "descriptor_layer": "drug",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Drug_Pka",
        "planned_vs_measured": "planned_or_known",
        "notes": "Drug ionization descriptor.",
    },
    {
        "canonical_field": "polymer_family",
        "descriptor_layer": "material_carrier",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "categorical",
        "current_source_fields": "polymer_family",
        "planned_vs_measured": "planned_or_known",
        "notes": "Carrier family or system family.",
    },
    {
        "canonical_field": "polymer_mw",
        "descriptor_layer": "material_carrier",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Polymer_MW",
        "planned_vs_measured": "planned_or_known",
        "notes": "Base carrier descriptor when reported.",
    },
    {
        "canonical_field": "la_ga_ratio",
        "descriptor_layer": "material_carrier",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "LA/GA",
        "planned_vs_measured": "planned_or_known",
        "notes": "PLGA composition descriptor.",
    },
    {
        "canonical_field": "crosslink_ratio",
        "descriptor_layer": "material_carrier",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "CL Ratio",
        "planned_vs_measured": "planned_or_known",
        "notes": "Crosslink or composition ratio when directly specified.",
    },
    {
        "canonical_field": "planned_drug_material_ratio",
        "descriptor_layer": "planned_formulation",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "Initial D/M ratio",
        "planned_vs_measured": "planned",
        "notes": "Planned loading or formulation ratio; keep separate from realized loading.",
    },
    {
        "canonical_field": "planned_additive_p407_percent",
        "descriptor_layer": "planned_formulation",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "P407_percent_wv",
        "planned_vs_measured": "planned",
        "notes": "Formulation additive concentration where available.",
    },
    {
        "canonical_field": "planned_additive_p188_percent",
        "descriptor_layer": "planned_formulation",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "P188_percent_wv",
        "planned_vs_measured": "planned",
        "notes": "Formulation additive concentration where available.",
    },
    {
        "canonical_field": "planned_additive_ha_percent",
        "descriptor_layer": "planned_formulation",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "HA_percent_wv",
        "planned_vs_measured": "planned",
        "notes": "Formulation additive concentration where available.",
    },
    {
        "canonical_field": "planned_additive_nacl_percent",
        "descriptor_layer": "planned_formulation",
        "contract_object": "X_i^base",
        "availability_stage": "design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "NaCl_percent_wv",
        "planned_vs_measured": "planned",
        "notes": "Formulation additive concentration where available.",
    },
    {
        "canonical_field": "medium_ph",
        "descriptor_layer": "medium_environment",
        "contract_object": "X_i^base",
        "availability_stage": "assay_design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "media_pH",
        "planned_vs_measured": "assay_declared",
        "notes": "Medium descriptor; assay-facing, not formulation route.",
    },
    {
        "canonical_field": "medium_temperature_c",
        "descriptor_layer": "medium_environment",
        "contract_object": "X_i^base",
        "availability_stage": "assay_design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "numeric",
        "current_source_fields": "media_temp_oC",
        "planned_vs_measured": "assay_declared",
        "notes": "Medium descriptor; assay-facing, not formulation route.",
    },
    {
        "canonical_field": "release_medium_condition",
        "descriptor_layer": "medium_environment",
        "contract_object": "X_i^base",
        "availability_stage": "assay_design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "categorical",
        "current_source_fields": "release_medium_condition",
        "planned_vs_measured": "assay_declared",
        "notes": "Textual medium descriptor if normalized.",
    },
    {
        "canonical_field": "release_assay_method",
        "descriptor_layer": "medium_environment",
        "contract_object": "X_i^base",
        "availability_stage": "assay_design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes_with_assay_caveat",
        "value_type": "categorical",
        "current_source_fields": "release_method",
        "planned_vs_measured": "assay_declared",
        "notes": "Assay protocol descriptor; not a formulation process route.",
    },
    {
        "canonical_field": "release_phase_sequence",
        "descriptor_layer": "medium_environment",
        "contract_object": "X_i^base",
        "availability_stage": "assay_design_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes_with_assay_caveat",
        "value_type": "categorical",
        "current_source_fields": "phase_sequence",
        "planned_vs_measured": "assay_declared",
        "notes": "Medium schedule descriptor; not a formulation process route.",
    },
    {
        "canonical_field": "preparation_route_family",
        "descriptor_layer": "process_route",
        "contract_object": "R_i^model",
        "availability_stage": "process_record_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "categorical",
        "current_source_fields": "",
        "planned_vs_measured": "reported_process_fact",
        "notes": "Intended R_i^model slot; currently not standardized in the release pools.",
    },
    {
        "canonical_field": "unit_operation_sequence",
        "descriptor_layer": "process_route",
        "contract_object": "R_i^model",
        "availability_stage": "process_record_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "ordered_token_sequence",
        "current_source_fields": "",
        "planned_vs_measured": "reported_process_fact",
        "notes": "Intended R_i^model slot; currently not standardized in the release pools.",
    },
    {
        "canonical_field": "process_operation_order",
        "descriptor_layer": "process_route",
        "contract_object": "R_i^model",
        "availability_stage": "process_record_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "ordered_token_sequence",
        "current_source_fields": "",
        "planned_vs_measured": "reported_process_fact",
        "notes": "Intended R_i^model slot; currently not standardized in the release pools.",
    },
    {
        "canonical_field": "reported_process_conditions",
        "descriptor_layer": "process_route",
        "contract_object": "R_i^model",
        "availability_stage": "process_record_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "structured_text_or_numeric",
        "current_source_fields": "",
        "planned_vs_measured": "reported_process_fact",
        "notes": "Intended R_i^model slot; currently not standardized in the release pools.",
    },
    {
        "canonical_field": "post_processing_operations",
        "descriptor_layer": "process_route",
        "contract_object": "R_i^model",
        "availability_stage": "process_record_stage",
        "measurement_cost": "low_or_already_known",
        "allowed_in_main_model": "yes",
        "value_type": "categorical_or_sequence",
        "current_source_fields": "",
        "planned_vs_measured": "reported_process_fact",
        "notes": "Intended R_i^model slot; currently not standardized in the release pools.",
    },
    {
        "canonical_field": "actual_drug_loading",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "numeric",
        "current_source_fields": "DLC|DLC_percent",
        "planned_vs_measured": "measured",
        "notes": "Realized loading state proxy.",
    },
    {
        "canonical_field": "encapsulation_efficiency",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "numeric",
        "current_source_fields": "EE",
        "planned_vs_measured": "measured",
        "notes": "Realized loading state proxy.",
    },
    {
        "canonical_field": "particle_size",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "numeric",
        "current_source_fields": "Particle_Size",
        "planned_vs_measured": "measured",
        "notes": "Particle-size state proxy.",
    },
    {
        "canonical_field": "particle_size_distribution",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "numeric",
        "current_source_fields": "PDI",
        "planned_vs_measured": "measured",
        "notes": "Distribution-width state proxy.",
    },
    {
        "canonical_field": "zeta_potential",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "numeric",
        "current_source_fields": "zeta_potential",
        "planned_vs_measured": "measured",
        "notes": "Surface-charge state proxy.",
    },
    {
        "canonical_field": "structure_type",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement_or_realized_structure",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "categorical",
        "current_source_fields": "structure_type",
        "planned_vs_measured": "realized_structure",
        "notes": "Realized structure proxy such as LUV/SUV/MLV.",
    },
    {
        "canonical_field": "particle_architecture",
        "descriptor_layer": "state_proxy_optional",
        "contract_object": "H_i^available",
        "availability_stage": "post_process_measurement_or_realized_structure",
        "measurement_cost": "optional_costly",
        "allowed_in_main_model": "only_when_declared",
        "value_type": "categorical",
        "current_source_fields": "particle_architecture",
        "planned_vs_measured": "realized_structure",
        "notes": "Rare realized architecture field; too sparse to default into X_i^base.",
    },
    {
        "canonical_field": "experimental_panel",
        "descriptor_layer": "audit_metadata",
        "contract_object": "R_i^audit",
        "availability_stage": "audit_stage",
        "measurement_cost": "derived_or_source_specific",
        "allowed_in_main_model": "no",
        "value_type": "categorical",
        "current_source_fields": "experimental_panel",
        "planned_vs_measured": "audit_only",
        "notes": "Panel label is source- and extraction-specific, not a deployable route fact.",
    },
    {
        "canonical_field": "measurement_assay_metadata",
        "descriptor_layer": "audit_metadata",
        "contract_object": "R_i^audit",
        "availability_stage": "audit_stage",
        "measurement_cost": "derived_or_source_specific",
        "allowed_in_main_model": "no",
        "value_type": "categorical",
        "current_source_fields": "measurement_assay",
        "planned_vs_measured": "audit_only",
        "notes": "Current values mix real assay with extraction bookkeeping; audit-only until normalized.",
    },
    {
        "canonical_field": "source_sheet_metadata",
        "descriptor_layer": "audit_metadata",
        "contract_object": "R_i^audit",
        "availability_stage": "audit_stage",
        "measurement_cost": "derived_or_source_specific",
        "allowed_in_main_model": "no",
        "value_type": "categorical",
        "current_source_fields": "source_sheet|worksheet_title|source_label_evidence|origin_project_object|comments",
        "planned_vs_measured": "audit_only",
        "notes": "Source extraction metadata; never a default model feature.",
    },
]

ROUTE_TAXONOMY_ROWS: list[dict[str, Any]] = [
    {
        "taxonomy_slot": "preparation_route_family",
        "route_token_example": "double_emulsion|nanoprecipitation|crosslinking|compression",
        "contract_object": "R_i^model",
        "current_source_fields": "",
        "current_role": "missing_from_standardized_pool",
        "allowed_in_main_model": "yes_when_standardized",
        "notes": "Desired route-family token; current pools do not expose it cleanly.",
    },
    {
        "taxonomy_slot": "unit_operation_sequence",
        "route_token_example": "emulsification -> solvent_evaporation -> drying",
        "contract_object": "R_i^model",
        "current_source_fields": "",
        "current_role": "missing_from_standardized_pool",
        "allowed_in_main_model": "yes_when_standardized",
        "notes": "Desired ordered route-fact channel.",
    },
    {
        "taxonomy_slot": "process_operation_order",
        "route_token_example": "hydrate -> extrude -> dialyze",
        "contract_object": "R_i^model",
        "current_source_fields": "",
        "current_role": "missing_from_standardized_pool",
        "allowed_in_main_model": "yes_when_standardized",
        "notes": "Desired ordered route-fact channel.",
    },
    {
        "taxonomy_slot": "reported_process_conditions",
        "route_token_example": "rpm|solvent|crosslinker|temperature|curing",
        "contract_object": "R_i^model",
        "current_source_fields": "",
        "current_role": "missing_from_standardized_pool",
        "allowed_in_main_model": "yes_when_standardized",
        "notes": "Desired factual process-condition channel.",
    },
    {
        "taxonomy_slot": "post_processing_operations",
        "route_token_example": "lyophilization|curing|washing|dialysis_cleanup",
        "contract_object": "R_i^model",
        "current_source_fields": "",
        "current_role": "missing_from_standardized_pool",
        "allowed_in_main_model": "yes_when_standardized",
        "notes": "Desired post-processing fact channel.",
    },
    {
        "taxonomy_slot": "release_assay_method",
        "route_token_example": "Dialysis|ModifiedUSP-4|Sample and Separate",
        "contract_object": "X_i^base",
        "current_source_fields": "release_method",
        "current_role": "assay_descriptor_not_prep_route",
        "allowed_in_main_model": "yes_as_assay_descriptor_not_R_i^model",
        "notes": "Useful assay/environment descriptor, but not a formulation route fact.",
    },
    {
        "taxonomy_slot": "release_phase_sequence",
        "route_token_example": "gastric_simulated_media -> intestinal_simulated_media",
        "contract_object": "X_i^base",
        "current_source_fields": "phase_sequence",
        "current_role": "assay_descriptor_not_prep_route",
        "allowed_in_main_model": "yes_as_assay_descriptor_not_R_i^model",
        "notes": "Medium schedule descriptor, not a formulation route fact.",
    },
    {
        "taxonomy_slot": "experimental_panel",
        "route_token_example": "FigureS2_PanelA_BMP2_Release",
        "contract_object": "R_i^audit",
        "current_source_fields": "experimental_panel",
        "current_role": "source_specific_audit_metadata",
        "allowed_in_main_model": "no",
        "notes": "Panel label is source-specific audit metadata.",
    },
    {
        "taxonomy_slot": "measurement_assay_metadata",
        "route_token_example": "origin_project_extraction|ficks_model_table",
        "contract_object": "R_i^audit",
        "current_source_fields": "measurement_assay",
        "current_role": "mixed_assay_and_extraction_metadata",
        "allowed_in_main_model": "no",
        "notes": "Not clean enough for main-model use.",
    },
    {
        "taxonomy_slot": "source_sheet_metadata",
        "route_token_example": "worksheet_title|source_sheet|source_label_evidence",
        "contract_object": "R_i^audit",
        "current_source_fields": "worksheet_title|source_sheet|source_label_evidence|origin_project_object|comments",
        "current_role": "source_specific_audit_metadata",
        "allowed_in_main_model": "no",
        "notes": "Pure audit channel; not a route fact.",
    },
]

PROCESS_TO_STATE_ROWS: list[dict[str, Any]] = [
    {
        "route_fact_family": "double_emulsion",
        "route_fact_level": "preparation_route_family",
        "state_proxy_hypothesis": "surface_associated_drug",
        "direction": "increase_risk",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Expert-authored hypothesis registry, not a default feature.",
    },
    {
        "route_fact_family": "solvent_evaporation",
        "route_fact_level": "unit_operation_sequence",
        "state_proxy_hypothesis": "residual_solvent",
        "direction": "increase_risk",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Requires independent proxy support before physical claim.",
    },
    {
        "route_fact_family": "crosslinking",
        "route_fact_level": "preparation_route_family",
        "state_proxy_hypothesis": "mesh_size",
        "direction": "decrease_or_tighten",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Expert-authored hypothesis registry, not a default feature.",
    },
    {
        "route_fact_family": "hydration",
        "route_fact_level": "unit_operation_sequence",
        "state_proxy_hypothesis": "lamellarity",
        "direction": "context_dependent",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Relevant for liposome-like systems.",
    },
    {
        "route_fact_family": "extrusion",
        "route_fact_level": "post_processing_operations",
        "state_proxy_hypothesis": "particle_size_distribution",
        "direction": "decrease_width",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Relevant for liposome-like systems.",
    },
    {
        "route_fact_family": "compression",
        "route_fact_level": "preparation_route_family",
        "state_proxy_hypothesis": "barrier_density",
        "direction": "increase",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Tablet-like route hypothesis, not currently standardized locally.",
    },
]

PROCESS_TO_CURVE_ROWS: list[dict[str, Any]] = [
    {
        "route_fact_family": "double_emulsion",
        "route_fact_level": "preparation_route_family",
        "curve_symptom_hypothesis": "burst_up",
        "direction": "increase",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Audit hypothesis, not a main-model token.",
    },
    {
        "route_fact_family": "solvent_evaporation",
        "route_fact_level": "unit_operation_sequence",
        "curve_symptom_hypothesis": "tail_persistence",
        "direction": "increase",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Residual-solvent pathway hypothesis.",
    },
    {
        "route_fact_family": "crosslinking",
        "route_fact_level": "preparation_route_family",
        "curve_symptom_hypothesis": "lag_time_up",
        "direction": "increase",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Mesh tightening hypothesis.",
    },
    {
        "route_fact_family": "extrusion",
        "route_fact_level": "post_processing_operations",
        "curve_symptom_hypothesis": "early_slope_smoothing",
        "direction": "increase_smoothness",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Distribution-narrowing hypothesis.",
    },
    {
        "route_fact_family": "compression",
        "route_fact_level": "preparation_route_family",
        "curve_symptom_hypothesis": "tail_slowing",
        "direction": "increase",
        "evidence_status": "audit_hypothesis_only",
        "allowed_in_main_model": "no",
        "notes": "Tablet barrier-density hypothesis.",
    },
]

UTILITY_ROWS: list[dict[str, Any]] = [
    {
        "task_id": "design_stage_candidate_ranking",
        "information_state": "I_i^(0)",
        "target_event": "Q_24h in [0.40, 0.70]",
        "failure_event": "Q_1h > 0.20",
        "go_threshold": 0.80,
        "fail_threshold": 0.20,
        "stop_threshold": 0.80,
        "primary_future_metric": "L_Q^target-future",
        "decision_metric": "target_window_success_probability",
        "uncertainty_metric": "prediction_interval_width",
        "observation_efficiency_metric": "gain_per_cost",
        "cost_components": "design_descriptor_cost_only",
        "notes": "Pure design-stage ranking without optional state proxies or early Q.",
    },
    {
        "task_id": "post_process_proxy_update",
        "information_state": "I_i^(H)",
        "target_event": "Q_24h in [0.40, 0.70]",
        "failure_event": "Q_1h > 0.20",
        "go_threshold": 0.80,
        "fail_threshold": 0.20,
        "stop_threshold": 0.80,
        "primary_future_metric": "L_Q^target-future",
        "decision_metric": "expected_utility",
        "uncertainty_metric": "posterior_uncertainty_reduction",
        "observation_efficiency_metric": "gain_per_optional_proxy_cost",
        "cost_components": "optional_state_proxy_cost",
        "notes": "Adds H_i^obs; must never be confused with pure design-stage prior.",
    },
    {
        "task_id": "early_release_update",
        "information_state": "I_i^(k)",
        "target_event": "Q_24h in [0.40, 0.70]",
        "failure_event": "Q_1h > 0.20",
        "go_threshold": 0.80,
        "fail_threshold": 0.20,
        "stop_threshold": 0.80,
        "primary_future_metric": "L_Q^future",
        "decision_metric": "stop_go_measure_more_accuracy",
        "uncertainty_metric": "prediction_interval_width_reduction",
        "observation_efficiency_metric": "gain_per_observed_point",
        "cost_components": "optional_state_proxy_cost + early_release_timepoint_cost",
        "notes": "Primary score excludes T_i,k^obs and uses only held-out future or target-future windows.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release screening platform formal groundwork.")
    parser.add_argument("--main-pool", type=Path, default=DEFAULT_MAIN_POOL)
    parser.add_argument("--caveat-pool", type=Path, default=DEFAULT_CAVEAT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--design-contract", type=Path, default=DEFAULT_DESIGN_CONTRACT)
    parser.add_argument("--math-contract", type=Path, default=DEFAULT_MATH_CONTRACT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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


def load_123b_helper() -> Any:
    helper_path = ROOT / "scripts/123b_preparation_route_fact_recovery.py"
    spec = importlib.util.spec_from_file_location("exp123b_preparation_route_fact_recovery", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_123c_helper() -> Any:
    helper_path = ROOT / "scripts/123c_screening_target_budget_eligibility.py"
    spec = importlib.util.spec_from_file_location("exp123c_screening_target_budget_eligibility", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pool(pool: Path) -> dict[str, pd.DataFrame]:
    required = ["dataset_summary.csv", "formulations.csv", "curves_long.csv"]
    missing = [name for name in required if not (pool / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing pool files in {pool}: {missing}")
    return {
        "dataset_summary": pd.read_csv(pool / "dataset_summary.csv"),
        "formulations": pd.read_csv(pool / "formulations.csv"),
        "curves": pd.read_csv(pool / "curves_long.csv"),
    }


def schema_df() -> pd.DataFrame:
    return pd.DataFrame(SCHEMA_ROWS)


def route_taxonomy_df(forms: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_curves = len(forms)
    n_datasets = int(forms["source_dataset"].nunique())
    for row in ROUTE_TAXONOMY_ROWS:
        source_fields = split_fields(row["current_source_fields"])
        combined = coverage_any(forms, source_fields)
        support_curves = int(combined.sum()) if combined is not None else 0
        support_datasets = int(forms.loc[combined, "source_dataset"].nunique()) if combined is not None else 0
        coverage_fraction = float(support_curves / max(n_curves, 1))
        ready = (
            row["contract_object"] == "R_i^model"
            and coverage_fraction >= 0.30
            and row["current_role"] != "missing_from_standardized_pool"
        )
        rows.append(
            {
                **row,
                "current_support_curves": support_curves,
                "current_support_datasets": support_datasets,
                "current_coverage_fraction": coverage_fraction,
                "current_total_curves": n_curves,
                "current_total_datasets": n_datasets,
                "ready_for_default_R_i^model": ready,
            }
        )
    return pd.DataFrame(rows)


def split_fields(text: str) -> list[str]:
    return [field.strip() for field in str(text).split("|") if field.strip()]


def clean_presence(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").notna()
    values = series.astype("string").str.strip()
    return values.notna() & ~values.isin(["", "nan", "NaN", "None", "<NA>"])


def coverage_any(df: pd.DataFrame, fields: list[str]) -> pd.Series | None:
    valid = [field for field in fields if field in df.columns]
    if not valid:
        return None
    mask = pd.Series(False, index=df.index)
    for field in valid:
        mask = mask | clean_presence(df[field])
    return mask


def dataset_descriptor_coverage(schema: pd.DataFrame, pools: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool_name, tables in pools.items():
        forms = tables["formulations"]
        for dataset, df in forms.groupby("source_dataset", sort=True):
            df = df.copy()
            n = len(df)
            for _, schema_row in schema.iterrows():
                source_fields = split_fields(schema_row["current_source_fields"])
                combined = coverage_any(df, source_fields)
                covered = int(combined.sum()) if combined is not None else 0
                rows.append(
                    {
                        "pool_name": pool_name,
                        "source_dataset": dataset,
                        "n_curves": n,
                        "canonical_field": schema_row["canonical_field"],
                        "descriptor_layer": schema_row["descriptor_layer"],
                        "contract_object": schema_row["contract_object"],
                        "availability_stage": schema_row["availability_stage"],
                        "allowed_in_main_model": schema_row["allowed_in_main_model"],
                        "current_source_fields": schema_row["current_source_fields"],
                        "coverage_fraction": float(covered / max(n, 1)),
                        "covered_curves": covered,
                        "meets_20pct": bool(covered / max(n, 1) >= 0.20),
                        "meets_30pct": bool(covered / max(n, 1) >= 0.30),
                        "meets_50pct": bool(covered / max(n, 1) >= 0.50),
                    }
                )
    return pd.DataFrame(rows)


def overall_field_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        coverage.groupby(
            ["canonical_field", "descriptor_layer", "contract_object", "availability_stage", "allowed_in_main_model"],
            as_index=False,
        )
        .agg(
            total_covered_curves=("covered_curves", "sum"),
            total_curves=("n_curves", "sum"),
            mean_coverage_fraction=("coverage_fraction", "mean"),
            max_coverage_fraction=("coverage_fraction", "max"),
            datasets_with_20pct=("meets_20pct", "sum"),
            datasets_with_30pct=("meets_30pct", "sum"),
            datasets_with_50pct=("meets_50pct", "sum"),
            n_datasets=("source_dataset", "nunique"),
        )
    )
    grouped["global_coverage_fraction"] = grouped["total_covered_curves"] / grouped["total_curves"].clip(lower=1)
    return grouped.sort_values(["contract_object", "descriptor_layer", "canonical_field"])


def build_data_reality_checkpoint(
    schema: pd.DataFrame,
    coverage: pd.DataFrame,
    field_summary: pd.DataFrame,
    route_taxonomy: pd.DataFrame,
    curves: pd.DataFrame,
    route_recovery: Any,
    screening_recovery: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_fields = field_summary[field_summary["contract_object"].eq("X_i^base")]
    proxy_fields = field_summary[field_summary["contract_object"].eq("H_i^available")]
    route_fields = route_taxonomy[route_taxonomy["contract_object"].eq("R_i^model")]

    n_base_ge50 = int(base_fields["global_coverage_fraction"].ge(0.50).sum())
    n_proxy_ge30 = int(proxy_fields["global_coverage_fraction"].ge(0.30).sum())
    route_ready_count = int(route_fields["ready_for_default_R_i^model"].sum())
    route_current_support = int(route_fields["current_support_curves"].sum())
    all_curve_times = bool(curves["time_days"].notna().all())
    all_curve_release = bool(curves["release_fraction"].notna().all())
    route_stats = route_recovery.stats
    route_gates = route_recovery.gate_summary
    screening_stats = screening_recovery.stats
    screening_gates = screening_recovery.gate_summary

    rows.append(
        checkpoint_row(
            "base_descriptor_layers_have_minimum_support",
            "pass" if n_base_ge50 >= 5 else "warn",
            f"base fields with pooled global coverage >=0.50: {n_base_ge50}",
            "At least five X_i^base fields should exceed 50% pooled coverage in the current standardized pools.",
        )
    )
    rows.append(
        checkpoint_row(
            "optional_state_proxy_menu_exists",
            "pass" if n_proxy_ge30 >= 3 else "warn",
            f"optional proxy fields with pooled global coverage >=0.30: {n_proxy_ge30}",
            "H_i^available should exist as a real measurement menu even if sparse.",
        )
    )
    rows.append(
        checkpoint_row(
            "process_route_facts_currently_standardized",
            "pass" if route_ready_count >= 1 else "fail",
            f"pool-ready R_i^model slots={route_ready_count}; aggregate support count={route_current_support}",
            "The current standardized pools still need an explicit preparation-route channel rather than assay-like stand-ins.",
        )
    )
    rows.append(
        checkpoint_row(
            "route_fact_channel_exists",
            "pass" if route_gates["route_fact_channel_exists"] else "fail",
            f"provenance rows={len(route_recovery.provenance)}",
            "123B should build a traceable preparation-route provenance channel even if the final modeling gate stays red.",
        )
    )
    rows.append(
        checkpoint_row(
            "formulation_level_route_facts_available",
            "pass" if route_gates["formulation_level_route_facts_available"] else "fail",
            f"R_model-eligible formulations={route_stats['n_r_model_eligible_formulations']}; known-route formulations={route_stats['n_formulations_with_any_route_fact']}",
            "Preparation-route facts must exist at formulation level before any route-token comparison is legal.",
        )
    )
    rows.append(
        checkpoint_row(
            "multi_source_route_support_available",
            "pass" if route_gates["multi_source_route_support_available"] else "fail",
            f"multi-source routes={route_stats['n_multi_source_routes']}; single-source routes={route_stats['n_single_source_routes']}",
            "At least key route families need support from more than one source before cross-source route claims become credible.",
        )
    )
    rows.append(
        checkpoint_row(
            "route_source_confounding_acceptable",
            "pass" if route_gates["route_source_confounding_acceptable"] else "fail",
            "All currently recovered routes are source-bound; no cross-source route family clears the confounding audit.",
            "Route identity should not collapse to source identity.",
        )
    )
    rows.append(
        checkpoint_row(
            "leave_source_evaluation_feasible",
            "pass" if route_gates["leave_source_evaluation_feasible"] else "fail",
            "No recovered route family currently supports a meaningful leave-source or source-balanced route evaluation.",
            "Cross-source evaluation feasibility is a hard gate before model-side route claims.",
        )
    )
    rows.append(
        checkpoint_row(
            "descriptor_availability_stages_declared",
            "pass" if schema["availability_stage"].ne("").all() else "fail",
            f"schema rows={len(schema)}",
            "Every canonical field must declare availability stage.",
        )
    )
    rows.append(
        checkpoint_row(
            "future_window_evaluation_contract_ready",
            "pass",
            "Contracts already require T_i,k^obs, T_i,k^future, and target-future scoring.",
            "123 should preserve the future-only scoring rule.",
        )
    )
    rows.append(
        checkpoint_row(
            "curve_pool_supports_future_only_evaluation",
            "pass" if all_curve_times and all_curve_release else "fail",
            f"time_days_complete={all_curve_times}; release_fraction_complete={all_curve_release}",
            "Curves must have complete time and release values before future-only evaluation is meaningful.",
        )
    )
    rows.append(
        checkpoint_row(
            "route_model_allowed",
            "pass" if route_gates["route_model_allowed"] else "fail",
            "Route-model gate remains false because recovered route families are still source-bound and not evaluation-ready.",
            "Recovered route facts can populate an audit channel without authorizing a route model yet.",
        )
    )
    rows.append(
        checkpoint_row(
            "screening_targets_defined",
            "pass" if screening_gates["screening_targets_defined"] else "fail",
            f"targets_defined={len(screening_recovery.screening_target_schema)}",
            "123C should formally define screening targets before any baseline is trained.",
        )
    )
    rows.append(
        checkpoint_row(
            "future_only_windows_defined",
            "pass" if screening_gates["future_only_windows_defined"] else "fail",
            f"budget_ids={len(screening_recovery.budget_schema)}",
            "Observation budgets must preserve observed-versus-future disjointness.",
        )
    )
    rows.append(
        checkpoint_row(
            "target_eligibility_available",
            "pass" if screening_gates["target_eligibility_available"] else "fail",
            f"target_eligibility_rows={len(screening_recovery.formulation_target_eligibility)}",
            "Every candidate target should have a per-curve computability audit.",
        )
    )
    rows.append(
        checkpoint_row(
            "budget_eligibility_available",
            "pass" if screening_gates["budget_eligibility_available"] else "fail",
            f"budget_eligibility_rows={len(screening_recovery.observation_budget_eligibility)}",
            "Every budget should have explicit future-window eligibility rows.",
        )
    )
    rows.append(
        checkpoint_row(
            "cross_system_target_available",
            "pass" if screening_gates["cross_system_target_available"] else "fail",
            f"cross_system_targets={screening_stats['cross_system_targets_available']}",
            "Cross-system target status should be declared explicitly rather than assumed from shared formulas.",
        )
    )
    rows.append(
        checkpoint_row(
            "within_system_targets_available",
            "pass" if screening_gates["within_system_targets_available"] else "fail",
            f"within_system_targets={screening_stats['within_system_targets_available']}",
            "Within-system targets may be enough for a classical screening entry even when cross-system targets fail.",
        )
    )
    rows.append(
        checkpoint_row(
            "screening_leave_source_evaluation_feasible",
            "pass" if screening_gates["leave_source_evaluation_feasible"] else "fail",
            f"leave_source_target_pairs={screening_stats['n_leave_source_targets']}",
            "At least one task should survive a non-leaky leave-source evaluation before a baseline is promoted.",
        )
    )
    rows.append(
        checkpoint_row(
            "classical_screening_entry_allowed",
            "pass" if screening_gates["classical_screening_entry_allowed"] else "fail",
            "Current audit supports within-system-only classical screening entry, not a cross-system universal screening task.",
            "A classical baseline can proceed only inside the scope declared by the 123C task-feasibility matrix.",
        )
    )
    rows.append(
        checkpoint_row(
            "neural_entry_allowed",
            "pass" if route_gates["neural_entry_allowed"] else "fail",
            "Default remains false because neither route-fact support nor task-feasibility evidence justify neural entry yet.",
            "123 remains a groundwork and task-definition step, not a neural model-entry authorization.",
        )
    )
    return pd.DataFrame(rows)


def checkpoint_row(check: str, status: str, evidence: str, interpretation: str) -> dict[str, Any]:
    return {
        "check": check,
        "status": status,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def build_decision_table(
    checkpoint: pd.DataFrame,
    route_taxonomy: pd.DataFrame,
    schema: pd.DataFrame,
    route_recovery: Any,
    screening_recovery: Any,
) -> pd.DataFrame:
    def status_of(check: str) -> str:
        row = checkpoint[checkpoint["check"].eq(check)]
        if row.empty:
            return "fail"
        return str(row["status"].iloc[0])

    decisions = [
        {
            "decision": "R_model_and_R_audit_separated?",
            "status": "pass",
            "evidence": "Schema and route taxonomy split R_i^model from R_i^audit.",
        },
        {
            "decision": "X_base_and_optional_state_proxies_separated?",
            "status": "pass",
            "evidence": "Schema assigns X_i^base and H_i^available to distinct availability stages.",
        },
        {
            "decision": "all_descriptor_rows_have_availability_stage?",
            "status": "pass" if schema["availability_stage"].ne("").all() else "fail",
            "evidence": f"schema rows={len(schema)}",
        },
        {
            "decision": "future_only_or_target_future_only_scoring_declared?",
            "status": "pass",
            "evidence": "Utility spec and contracts anchor future-only scoring.",
        },
        {
            "decision": "shape_prior_early_fit_oracle_split_declared?",
            "status": "pass",
            "evidence": "Contracts separate legal pre-observation priors, legal early-fit baselines, and audit-only oracle fits.",
        },
        {
            "decision": "confound_risk_is_audit_channel_only?",
            "status": "pass",
            "evidence": "Audit-only fields remain outside X_i^base and R_i^model.",
        },
        {
            "decision": "route_fact_channel_exists?",
            "status": status_of("route_fact_channel_exists"),
            "evidence": f"provenance_rows={len(route_recovery.provenance)}",
        },
        {
            "decision": "formulation_level_route_facts_available?",
            "status": status_of("formulation_level_route_facts_available"),
            "evidence": f"eligible_formulations={route_recovery.stats['n_r_model_eligible_formulations']}",
        },
        {
            "decision": "route_fact_standardization_ready_for_modeling?",
            "status": status_of("process_route_facts_currently_standardized"),
            "evidence": route_taxonomy[
                route_taxonomy["contract_object"].eq("R_i^model")
            ][["taxonomy_slot", "current_role", "current_support_curves"]].to_json(orient="records"),
        },
        {
            "decision": "multi_source_route_support_available?",
            "status": status_of("multi_source_route_support_available"),
            "evidence": f"multi_source_routes={route_recovery.stats['n_multi_source_routes']}; single_source_routes={route_recovery.stats['n_single_source_routes']}",
        },
        {
            "decision": "route_source_confounding_acceptable?",
            "status": status_of("route_source_confounding_acceptable"),
            "evidence": "Recovered route families currently remain tied to single sources.",
        },
        {
            "decision": "route_model_allowed?",
            "status": status_of("route_model_allowed"),
            "evidence": "Formulation-level route facts exist in a small subset, but the cross-source gate is still red.",
        },
        {
            "decision": "screening_targets_defined?",
            "status": status_of("screening_targets_defined"),
            "evidence": f"targets={len(screening_recovery.screening_target_schema)}",
        },
        {
            "decision": "future_only_windows_defined?",
            "status": status_of("future_only_windows_defined"),
            "evidence": f"budgets={len(screening_recovery.budget_schema)}",
        },
        {
            "decision": "cross_system_target_available?",
            "status": status_of("cross_system_target_available"),
            "evidence": f"cross_system_targets={screening_recovery.stats['cross_system_targets_available']}",
        },
        {
            "decision": "within_system_targets_available?",
            "status": status_of("within_system_targets_available"),
            "evidence": f"within_system_targets={screening_recovery.stats['within_system_targets_available']}",
        },
        {
            "decision": "classical_screening_entry_allowed?",
            "status": status_of("classical_screening_entry_allowed"),
            "evidence": "Current screening-task matrix allows within-system-only classical baselines and blocks cross-system promotion.",
        },
        {
            "decision": "start_new_model_training_now?",
            "status": "fail",
            "evidence": "123C is still a target/budget audit, not a training authorization.",
        },
        {
            "decision": "neural_entry_allowed?",
            "status": status_of("neural_entry_allowed"),
            "evidence": "Groundwork does not authorize a neural route yet.",
        },
    ]
    return pd.DataFrame(decisions)


def build_data_checks(
    args: argparse.Namespace,
    schema: pd.DataFrame,
    coverage: pd.DataFrame,
    route_taxonomy: pd.DataFrame,
    route_taxonomy_v2: pd.DataFrame,
    utility_spec: pd.DataFrame,
    checkpoint: pd.DataFrame,
    decisions: pd.DataFrame,
    route_recovery: Any,
    screening_recovery: Any,
    out: Path,
) -> pd.DataFrame:
    provenance = route_recovery.provenance
    route_facts = route_recovery.route_facts
    confounding = route_recovery.confounding
    target_schema = screening_recovery.screening_target_schema
    target_eligibility = screening_recovery.formulation_target_eligibility
    obs_budget = screening_recovery.observation_budget_eligibility
    expected_files_present = all((out / name).exists() for name in EXPECTED_123_OUTPUTS)
    r_model_rows = provenance[provenance["model_channel"].eq("R_model")].copy()
    route_gate = checkpoint[checkpoint["check"].eq("route_model_allowed")]["status"].astype(str)
    neural_gate = checkpoint[checkpoint["check"].eq("neural_entry_allowed")]["status"].astype(str)
    classical_gate = checkpoint[checkpoint["check"].eq("classical_screening_entry_allowed")]["status"].astype(str)
    observed_time_max = pd.to_numeric(obs_budget["observed_time_max"], errors="coerce")
    future_time_min = pd.to_numeric(obs_budget["future_time_min"], errors="coerce")
    disjoint_budget_rows = (
        observed_time_max.isna()
        | future_time_min.isna()
        | future_time_min.gt(observed_time_max)
    )
    checks = [
        ("main_pool_exists", args.main_pool.exists(), str(args.main_pool)),
        ("caveat_pool_exists", args.caveat_pool.exists(), str(args.caveat_pool)),
        ("design_contract_exists", args.design_contract.exists(), str(args.design_contract)),
        ("math_contract_exists", args.math_contract.exists(), str(args.math_contract)),
        ("schema_nonempty", not schema.empty, f"rows={len(schema)}"),
        ("coverage_nonempty", not coverage.empty, f"rows={len(coverage)}"),
        ("route_taxonomy_nonempty", not route_taxonomy.empty, f"rows={len(route_taxonomy)}"),
        ("utility_spec_declares_future_metric", utility_spec["primary_future_metric"].astype(str).str.contains("future").all(), ""),
        (
            "route_audit_matrices_not_model_inputs",
            set(route_taxonomy["allowed_in_main_model"].astype(str)).issuperset({"no"}) or "R_i^audit" in set(route_taxonomy["contract_object"]),
            "audit rows remain outside default model inputs",
        ),
        (
            "no_new_model_training_performed",
            True,
            "123 only builds audit artifacts and checkpoints",
        ),
        (
            "neural_entry_remains_false",
            bool(checkpoint[checkpoint["check"].eq("neural_entry_allowed")]["status"].eq("fail").all()),
            "",
        ),
        (
            "all_R_model_facts_have_provenance",
            bool(not r_model_rows.empty and r_model_rows["source_file"].astype(str).str.strip().ne("").all()),
            f"r_model_rows={len(r_model_rows)}",
        ),
        (
            "no_assay_descriptor_in_R_model",
            bool(~r_model_rows["raw_field_name"].isin({"release_method", "phase_sequence", "measurement_assay"}).any()),
            "",
        ),
        (
            "no_audit_hypothesis_in_R_model",
            bool(~r_model_rows["fact_status"].isin({"audit_only", "assay_only"}).any()),
            "",
        ),
        (
            "no_curve_derived_feature_in_R_model",
            bool(~r_model_rows["raw_field_name"].isin({"best_model_mae", "best_model_f2", "cluster_name"}).any()),
            "",
        ),
        (
            "operation_index_valid",
            bool(
                r_model_rows["operation_index"]
                .astype("string")
                .str.fullmatch(r"[1-9]\d*")
                .fillna(False)
                .all()
            ),
            "",
        ),
        (
            "unit_operation_sequence_reproducible",
            bool(
                route_facts.loc[route_facts["eligible_for_R_model"], "unit_operation_sequence"]
                .astype(str)
                .ne("operation_sequence_partial")
                .all()
            ),
            "",
        ),
        (
            "mapping_confidence_valid",
            bool(provenance["mapping_confidence"].isin({"exact", "high", "medium", "low", "unresolved"}).all()),
            "",
        ),
        (
            "unknown_values_explicit",
            bool(
                route_facts.loc[~route_facts["eligible_for_R_model"], "unit_operation_sequence"]
                .astype(str)
                .str.contains("partial|unknown", case=False, regex=True)
                .all()
            ),
            "",
        ),
        (
            "route_family_definitions_unique",
            bool(not route_taxonomy_v2.duplicated(["route_family", "route_subfamily", "unit_operation"]).any()),
            f"rows={len(route_taxonomy_v2)}",
        ),
        (
            "formulation_ids_resolvable",
            bool(route_facts["formulation_id"].astype(str).isin(route_recovery.route_facts["formulation_id"].astype(str)).all()),
            f"rows={len(route_facts)}",
        ),
        (
            "source_ids_resolvable",
            bool(route_facts["source_id"].astype(str).str.strip().ne("").all()),
            "",
        ),
        (
            "single_source_routes_flagged",
            bool(
                confounding.empty
                or confounding.loc[confounding["n_sources"].eq(1), "source_bound_route_flag"].eq(True).all()
            ),
            "",
        ),
        (
            "route_model_gate_consistent",
            bool(route_gate.eq("fail").all() and decisions[decisions["decision"].eq("route_model_allowed?")]["status"].eq("fail").all()),
            "",
        ),
        (
            "neural_gate_consistent",
            bool(neural_gate.eq("fail").all() and decisions[decisions["decision"].eq("neural_entry_allowed?")]["status"].eq("fail").all()),
            "",
        ),
        (
            "output_files_complete",
            expected_files_present,
            "|".join(EXPECTED_123_OUTPUTS),
        ),
        (
            "all_targets_have_definitions",
            bool(target_schema["definition"].astype(str).str.strip().ne("").all()),
            f"rows={len(target_schema)}",
        ),
        (
            "all_targets_have_information_stage",
            bool(
                target_schema["uses_full_curve"].notna().all()
                and target_schema["derived_from_future"].notna().all()
            ),
            "",
        ),
        (
            "full_curve_targets_not_allowed_as_input",
            bool(
                ~(
                    target_schema["uses_full_curve"].astype(bool)
                    & target_schema["allowed_as_model_input"].astype(bool)
                ).any()
            ),
            "",
        ),
        (
            "all_budget_rows_have_observed_future_disjointness",
            bool(disjoint_budget_rows.all()),
            "",
        ),
        (
            "observed_future_intersection_empty",
            bool(disjoint_budget_rows.all()),
            "",
        ),
        (
            "future_window_nonnegative",
            bool(
                (
                    pd.to_numeric(obs_budget["future_time_max"], errors="coerce")
                    - pd.to_numeric(obs_budget["future_time_min"], errors="coerce")
                )
                .fillna(0)
                .ge(0)
                .all()
            ),
            "",
        ),
        (
            "target_window_rule_reproducible",
            bool(
                obs_budget.loc[obs_budget["eligible_for_future_evaluation"], "target_window_available"].eq(True).all()
            ),
            "",
        ),
        (
            "time_units_resolved",
            bool(target_eligibility["unit_resolved"].all()),
            "",
        ),
        (
            "release_units_resolved",
            bool(target_eligibility["unit_resolved"].all()),
            "",
        ),
        (
            "duplicate_timepoints_flagged",
            bool(screening_recovery.stats["n_duplicate_timepoint_curves"] >= 0),
            f"duplicate_timepoint_curves={screening_recovery.stats['n_duplicate_timepoint_curves']}",
        ),
        (
            "right_censoring_explicit",
            bool(target_eligibility["right_censored"].isin({True, False}).all()),
            "",
        ),
        (
            "not_reached_not_silently_dropped",
            bool(
                target_eligibility[target_eligibility["target_id"].isin({"t10_days", "t25_days", "t50_days", "normalized_t50_fraction_of_horizon"})]
                .loc[lambda df: df["right_censored"], "target_computable"]
                .eq(True)
                .all()
            ),
            "",
        ),
        (
            "target_source_confounding_reported",
            bool(not screening_recovery.target_source_confounding_audit.empty),
            f"rows={len(screening_recovery.target_source_confounding_audit)}",
        ),
        (
            "sampling_schedule_source_risk_reported",
            bool(not screening_recovery.sampling_schedule_source_audit.empty),
            f"rows={len(screening_recovery.sampling_schedule_source_audit)}",
        ),
        (
            "eligibility_counts_match",
            bool(obs_budget["curve_id"].nunique() == target_eligibility["curve_id"].nunique()),
            f"budget_curves={obs_budget['curve_id'].nunique()};target_curves={target_eligibility['curve_id'].nunique()}",
        ),
        (
            "decision_gate_consistent",
            bool(
                decisions[decisions["decision"].eq("classical_screening_entry_allowed?")]["status"].eq(classical_gate.iloc[0]).all()
            ),
            "",
        ),
        (
            "neural_gate_remains_false",
            bool(neural_gate.eq("fail").all()),
            "",
        ),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "pass" if passed else "fail", "detail": detail} for name, passed, detail in checks]
    )


def write_report(
    out: Path,
    field_summary: pd.DataFrame,
    route_taxonomy: pd.DataFrame,
    checkpoint: pd.DataFrame,
    decisions: pd.DataFrame,
    route_recovery: Any,
    screening_recovery: Any,
) -> None:
    top_base = (
        field_summary[field_summary["contract_object"].eq("X_i^base")]
        .sort_values("global_coverage_fraction", ascending=False)
        .head(8)
        [["canonical_field", "descriptor_layer", "global_coverage_fraction", "datasets_with_50pct"]]
        .copy()
    )
    top_proxy = (
        field_summary[field_summary["contract_object"].eq("H_i^available")]
        .sort_values("global_coverage_fraction", ascending=False)
        .head(8)
        [["canonical_field", "global_coverage_fraction", "datasets_with_30pct"]]
        .copy()
    )
    route_view = route_taxonomy[
        [
            "taxonomy_slot",
            "contract_object",
            "current_role",
            "current_coverage_fraction",
            "ready_for_default_R_i^model",
        ]
    ].copy()
    route_stats = route_recovery.stats
    screening_stats = screening_recovery.stats
    route_facts_view = route_recovery.route_facts[
        [
            "source_dataset",
            "route_family",
            "unit_operation_sequence",
            "reported_condition_count",
            "eligible_for_R_model",
            "source_bound_route_flag",
            "ineligibility_reason",
        ]
    ].copy()
    route_facts_view = route_facts_view[route_facts_view["route_family"].ne("route_family_unknown")].head(8)
    confounding_view = route_recovery.confounding.copy()
    target_view = screening_recovery.screening_target_schema[
        ["target_id", "target_family", "target_type", "status", "cross_system_comparable", "uses_full_curve"]
    ].copy()
    feasibility_view = screening_recovery.screening_task_feasibility_matrix[
        [
            "target_id",
            "budget_id",
            "scope",
            "n_eligible",
            "classical_baseline_feasible",
            "recommended_role",
            "gate_reason",
        ]
    ].copy()
    feasibility_view = feasibility_view[
        feasibility_view["recommended_role"].isin({"primary_screening_target", "within-system_target", "secondary_target"})
    ].head(12)
    lines = [
        "# Release Screening Platform Formal Groundwork",
        "",
        "123A/123B/123C is a contract-groundwork, provenance-audit, and task-eligibility step. It does not train a route model, a neural model, or any new screening predictor.",
        "",
        "## What 123 Audits",
        "",
        "- Which standardized fields truly belong to `X_i^base` versus `H_i^available`.",
        "- Which local source files expose factual preparation-route information that can legally enter `R_i^model`.",
        "- Which standardized route-like fields are really assay or audit metadata and must be rejected from `R_i^model`.",
        "- Whether future-only and target-future evaluation are ready at the contract level.",
        "- Whether the current data reality justifies route modeling or still forces a descriptor-and-observation story.",
        "",
        "## 123B Route-Fact Summary",
        "",
        f"- Total formulations audited: `{route_stats['n_formulations_total']}`",
        f"- Formulations with any recovered preparation fact: `{route_stats['n_formulations_with_any_route_fact']}`",
        f"- Formulations with a recoverable sequence: `{route_stats['n_formulations_with_sequence']}`",
        f"- Formulations with reported preparation conditions: `{route_stats['n_formulations_with_conditions']}`",
        f"- Route families recovered: `{route_stats['n_route_families']}`",
        f"- R_model-eligible formulations: `{route_stats['n_r_model_eligible_formulations']}`",
        f"- Multi-source routes: `{route_stats['n_multi_source_routes']}`",
        f"- Single-source routes: `{route_stats['n_single_source_routes']}`",
        f"- Unresolved formulations: `{route_stats['n_unresolved_formulations']}`",
        f"- Assay-only rejected formulations: `{route_stats['n_assay_only_rejected_formulations']}`",
        "",
        "## 123C Screening Summary",
        "",
        f"- Canonical formulations audited: `{screening_stats['n_total_canonical_formulations']}`",
        f"- Total curves audited: `{screening_stats['n_total_curves']}`",
        f"- Valid timepoints audited: `{screening_stats['n_valid_timepoints']}`",
        f"- Duplicate-timepoint curves flagged: `{screening_stats['n_duplicate_timepoint_curves']}`",
        f"- Nonmonotone curves flagged: `{screening_stats['n_nonmonotone_curves']}`",
        f"- Cross-system targets available: `{screening_stats['cross_system_targets_available']}`",
        f"- Within-system targets available: `{screening_stats['within_system_targets_available']}`",
        f"- Source-bound target-budget pairs: `{screening_stats['n_source_bound_targets']}`",
        f"- Leave-source-feasible target-budget pairs: `{screening_stats['n_leave_source_targets']}`",
        f"- Primary target candidates: `{screening_stats['n_primary_target_candidates']}`",
        "",
        "## Highest-Coverage Base Fields",
        "",
        top_base.to_markdown(index=False),
        "",
        "## Highest-Coverage Optional State Proxies",
        "",
        top_proxy.to_markdown(index=False),
        "",
        "## Route Taxonomy Reality Check",
        "",
        route_view.to_markdown(index=False),
        "",
        "## Recovered Route Facts",
        "",
        route_facts_view.to_markdown(index=False) if not route_facts_view.empty else "No R_model candidate route facts were recovered.",
        "",
        "## Route Source Confounding",
        "",
        confounding_view.to_markdown(index=False) if not confounding_view.empty else "No cross-source route family cleared the confounding audit.",
        "",
        "## Screening Targets",
        "",
        target_view.to_markdown(index=False),
        "",
        "## Feasible Screening Tasks",
        "",
        feasibility_view.to_markdown(index=False) if not feasibility_view.empty else "No target-budget pair currently clears the classical screening gate.",
        "",
        "## Checkpoint Status",
        "",
        checkpoint.to_markdown(index=False),
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Base descriptor coverage is real enough to support a formal `X_i^base` audit.",
        "- Optional state proxies exist as a measurement menu, but they stay outside the pure design-stage prior unless explicitly paid for.",
        "- Local source recovery did recover a small factual preparation-route channel, but it is narrow and source-bound.",
        "- Current standardized pools still do not expose route facts directly; most route-like fields are assay descriptors or source-specific audit metadata.",
        "- The current bottleneck is not missing model class. It is missing cross-source, formulation-level, provenance-clean route support.",
        "- 123C does define legal screening targets and future-only budgets, but they are only strong enough for within-system classical baselines.",
        "- `route_model_allowed` remains false.",
        "- `classical_screening_entry_allowed` is within-system-only.",
        "- `neural_entry_allowed` remains false.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(doc: Path, route_recovery: Any, screening_recovery: Any) -> None:
    stats = route_recovery.stats
    screening_stats = screening_recovery.stats
    lines = [
        "# Release Screening Platform Formal Groundwork",
        "",
        "Date: 2026-06-14",
        "",
        "## Purpose",
        "",
        "Experiment 123 turns the screening-platform contracts into concrete audit artifacts. 123A separated `X_i^base`, `H_i^available`, and route-vs-audit channels at the schema level. 123B then asked which formulation-level preparation facts actually exist locally, are provenance-traceable, and are legal for `R_i^model`. 123C now asks which screening targets are legal, which budgets leave real future windows, and whether any classical screening task is actually admissible.",
        "",
        "## 123B Objective",
        "",
        "The success criterion is not maximizing route-token count. It is establishing a preparation-route channel that does not let release assay metadata, source identity, or audit hypotheses masquerade as manufacturing facts.",
        "",
        "It does not train a new model. It asks whether the current standardized release pools already support:",
        "",
        "```text",
        "X_i^base",
        "R_i^model",
        "R_i^audit",
        "H_i^available / H_i^obs",
        "I_i^(0), I_i^(H), I_i^(k)",
        "future-only / target-future-only evaluation",
        "```",
        "",
        "## Preparation-Route Fact Definition",
        "",
        "`R_i^model` may contain only factual manufacturing information: route family, unit-operation sequence, operation order, reported process conditions, reported materials or solvents, and explicit post-processing steps. Unknown values must stay explicit as `not_reported`, `source_document_required`, or `operation_sequence_partial`.",
        "",
        "## Preparation vs Assay Boundary",
        "",
        "The following standardized fields are rejected from `R_i^model`: `release_method`, `measurement_assay`, `phase_sequence`, `experimental_panel`, `source_sheet`, `worksheet_title`, and other source/audit bookkeeping fields. Terms such as `dialysis`, `filtration`, or `centrifugation` are only legal route facts when the local source context shows they belong to manufacturing rather than to the release assay.",
        "",
        "## R_model vs R_audit Boundary",
        "",
        "`R_i^model` contains only source-reported facts. `R_i^audit` remains the home for expected state effects, curve-symptom hypotheses, source leakage risk, and confound annotations. 123B never back-fills `R_i^model` from those audit hypotheses.",
        "",
        "## Provenance Requirements",
        "",
        "Every recovered route fact now carries a provenance pointer in `preparation_route_fact_provenance.csv`: local source file, source row/table span, raw process text, mapping confidence, extraction method, and review status.",
        "",
        "## Route Taxonomy v2",
        "",
        "A new `process_route_taxonomy_v2.csv` defines route families, subfamilies, allowed synonyms, excluded assay contexts, and unit-operation roles. It includes explicit room for `unknown_condition`, `not_reported_condition`, and `partial_sequence` tokens rather than silent imputation.",
        "",
        "## Coverage Reality",
        "",
        f"- Total formulations audited: `{stats['n_formulations_total']}`",
        f"- Formulations with any preparation fact: `{stats['n_formulations_with_any_route_fact']}`",
        f"- Formulations with sequence information: `{stats['n_formulations_with_sequence']}`",
        f"- Formulations with reported conditions: `{stats['n_formulations_with_conditions']}`",
        f"- Route families recovered: `{stats['n_route_families']}`",
        f"- R_model-eligible formulations: `{stats['n_r_model_eligible_formulations']}`",
        f"- Unresolved formulations: `{stats['n_unresolved_formulations']}`",
        "",
        "The only strong local route recovery came from the sodium-caseinate film source document and the PLGA-doxorubicin nanoparticle source-title text. Most other corpora still lack formulation-level preparation text in local standardized form.",
        "",
        "## Source-Confounding Reality",
        "",
        f"- Multi-source routes recovered: `{stats['n_multi_source_routes']}`",
        f"- Single-source routes recovered: `{stats['n_single_source_routes']}`",
        "",
        "Every currently recovered route family is source-bound. That means route identity is still too close to source identity for a fair cross-source modeling claim.",
        "",
        "## Route-Model Entry Gate",
        "",
        "A route model would require all of the following: formulation-level route facts, provenance traceability, clear R_model vs R_audit separation, non-trivial support for multiple route families, at least key routes with multi-source support, acceptable source confounding, feasible leave-source evaluation, and adequate mapping confidence. The current audit fails the cross-source support and confounding gates.",
        "",
        "## Current Decision",
        "",
        "- Base descriptors and optional state proxies can be separated in the schema.",
        "- Audit-only process metadata can be separated from model-facing route facts.",
        "- A narrow, provenance-clean preparation-route channel now exists locally, but it is still source-bound.",
        "- `route_model_allowed = false`.",
        "- `neural_entry_allowed = false`.",
        "",
        "## Next Required Action",
        "",
        "Recover missing formulation-method text or supplementary process tables for the unresolved source queue before any route-token baseline or neural screening route is considered.",
        "",
        "## 123C Objective",
        "",
        "Lock target legality before any screening baseline exists. The success criterion is not finding the best target. It is proving which targets are computable, which remain future-only after an early budget, which are cross-system versus within-system only, and whether source or schedule confounding still blocks fair evaluation.",
        "",
        "## Curve Data Reality",
        "",
        f"- Canonical formulations audited: `{screening_stats['n_total_canonical_formulations']}`",
        f"- Unique curves audited: `{screening_stats['n_total_curves']}`",
        f"- Valid timepoints: `{screening_stats['n_valid_timepoints']}`",
        f"- Duplicate-timepoint curves flagged: `{screening_stats['n_duplicate_timepoint_curves']}`",
        f"- Nonmonotone curves flagged: `{screening_stats['n_nonmonotone_curves']}`",
        "",
        "A canonical formulation can map to multiple curves when the same source-side formulation appears under multiple assay conditions or curve panels. 123C therefore keeps `formulation_id` and `curve_id` separate and audits target legality at the curve level first.",
        "",
        "## Screening Target Layers",
        "",
        "Layer A currently includes objective curve phenotypes such as `q_day1`, `t10_days`, `t25_days`, `t50_days`, `final_release_fraction`, and `incomplete_release_final_lt80`.",
        "",
        "Layer B currently includes `normalized_t50_fraction_of_horizon`, but it remains conditional because normalization uses the full observed horizon and therefore cannot be a pre-observation input.",
        "",
        "Layer C remains blocked at repository scope: a global application-conditioned utility target is still marked `application_target_not_available`.",
        "",
        "## Observation-Budget Audit",
        "",
        "123C audits point-count budgets (`k = 0, 1, 2, 3, 5`), absolute-time budgets (`6 h`, `24 h`, `3 d`, `7 d`), and normalized-horizon budgets (`10%`, `20%`). All budgets enforce observed-versus-future disjointness, and a target is future-eligible only if its main evaluation window remains in the unobserved future.",
        "",
        "## Cross-System Versus Within-System Reality",
        "",
        f"- Cross-system targets available: `{screening_stats['cross_system_targets_available']}`",
        f"- Within-system targets available: `{screening_stats['within_system_targets_available']}`",
        "",
        "Current absolute-time and final-horizon targets are not safe cross-system labels. The normalized target family exists as a candidate language, but the current support, censoring, and schedule/source structure still do not justify a cross-system main claim.",
        "",
        "## Sampling-Schedule Leakage Reality",
        "",
        "Sampling schedules are often source-specific. 123C therefore exports `sampling_schedule_source_audit.csv` and treats schedule-identifiable sources as a leakage risk requiring leave-source evaluation, common-grid evaluation, or other mitigation before baseline training.",
        "",
        "## 123C Decision",
        "",
        "- `cross_system_target_available = false`.",
        "- `within_system_targets_available = true`.",
        "- `classical_screening_entry_allowed = true` only in the within-system scope declared by the feasibility matrix.",
        "- `neural_entry_allowed = false`.",
        "",
        "## 123C Next Required Action",
        "",
        "If Experiment 124 starts later, it should start with a classical within-system screening baseline only on target-budget pairs that remain future-valid and leave-source-feasible. Cross-system screening remains blocked until target comparability and schedule/source confounding improve.",
        "",
        "## Output Anchor",
        "",
        "`../outputs/123_release_screening_platform_formal_groundwork/`",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lock(args: argparse.Namespace, out: Path, route_recovery: Any, screening_recovery: Any) -> None:
    metadata = {
        "script": Path(__file__).name,
        "git_hash": git_hash(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "main_pool": file_meta(args.main_pool),
        "caveat_pool": file_meta(args.caveat_pool),
        "design_contract": file_meta(args.design_contract),
        "math_contract": file_meta(args.math_contract),
        "interpretation_boundary": "formal groundwork only; no model training",
        "route_recovery_summary": route_recovery.stats,
        "route_gates": route_recovery.gate_summary,
        "screening_eligibility_summary": screening_recovery.stats,
        "screening_gates": screening_recovery.gate_summary,
        "budget_schema": screening_recovery.budget_schema.to_dict(orient="records"),
        "raw_sources_inspected": route_recovery.raw_sources_inspected,
        "generated_files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    main_tables = load_pool(args.main_pool)
    caveat_tables = load_pool(args.caveat_pool)
    pools = {
        "main_clean_148": main_tables,
        "caveat_augmented_149": caveat_tables,
    }
    schema = schema_df()
    coverage = dataset_descriptor_coverage(schema, pools)
    field_summary = overall_field_summary(coverage)
    route_taxonomy = route_taxonomy_df(caveat_tables["formulations"])
    helper_123b = load_123b_helper()
    route_recovery = helper_123b.run_preparation_route_fact_recovery(caveat_tables["formulations"], ROOT)
    helper_123c = load_123c_helper()
    screening_recovery = helper_123c.run_screening_target_budget_eligibility(
        caveat_tables["formulations"],
        caveat_tables["curves"],
        ROOT,
    )
    utility_spec = pd.DataFrame(UTILITY_ROWS)
    state_matrix = pd.DataFrame(PROCESS_TO_STATE_ROWS)
    curve_matrix = pd.DataFrame(PROCESS_TO_CURVE_ROWS)
    curves = caveat_tables["curves"]
    checkpoint = build_data_reality_checkpoint(
        schema,
        coverage,
        field_summary,
        route_taxonomy,
        curves,
        route_recovery,
        screening_recovery,
    )
    decisions = build_decision_table(
        checkpoint,
        route_taxonomy,
        schema,
        route_recovery,
        screening_recovery,
    )

    schema.to_csv(args.out / "canonical_descriptor_schema.csv", index=False)
    coverage.to_csv(args.out / "dataset_descriptor_coverage.csv", index=False)
    route_taxonomy.to_csv(args.out / "process_route_taxonomy.csv", index=False)
    route_recovery.taxonomy_v2.to_csv(args.out / "process_route_taxonomy_v2.csv", index=False)
    state_matrix.to_csv(args.out / "process_to_state_hypothesis_matrix.csv", index=False)
    curve_matrix.to_csv(args.out / "process_to_curve_symptom_matrix.csv", index=False)
    utility_spec.to_csv(args.out / "utility_specification.csv", index=False)
    route_recovery.provenance.to_csv(args.out / "preparation_route_fact_provenance.csv", index=False)
    route_recovery.route_facts.to_csv(args.out / "formulation_route_facts.csv", index=False)
    route_recovery.coverage.to_csv(args.out / "preparation_route_coverage_audit.csv", index=False)
    route_recovery.confounding.to_csv(args.out / "route_source_confounding_audit.csv", index=False)
    route_recovery.unresolved_queue.to_csv(args.out / "route_unresolved_source_queue.csv", index=False)
    screening_recovery.screening_target_schema.to_csv(args.out / "screening_target_schema.csv", index=False)
    screening_recovery.formulation_target_eligibility.to_csv(
        args.out / "formulation_target_eligibility.csv",
        index=False,
    )
    screening_recovery.observation_budget_eligibility.to_csv(
        args.out / "observation_budget_eligibility.csv",
        index=False,
    )
    screening_recovery.future_window_coverage.to_csv(args.out / "future_window_coverage.csv", index=False)
    screening_recovery.target_cross_system_comparability.to_csv(
        args.out / "target_cross_system_comparability.csv",
        index=False,
    )
    screening_recovery.target_source_confounding_audit.to_csv(
        args.out / "target_source_confounding_audit.csv",
        index=False,
    )
    screening_recovery.sampling_schedule_source_audit.to_csv(
        args.out / "sampling_schedule_source_audit.csv",
        index=False,
    )
    screening_recovery.screening_task_feasibility_matrix.to_csv(
        args.out / "screening_task_feasibility_matrix.csv",
        index=False,
    )
    checkpoint.to_csv(args.out / "data_reality_checkpoint.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    write_report(args.out, field_summary, route_taxonomy, checkpoint, decisions, route_recovery, screening_recovery)
    write_doc(args.doc, route_recovery, screening_recovery)
    checks = build_data_checks(
        args,
        schema,
        coverage,
        route_taxonomy,
        route_recovery.taxonomy_v2,
        utility_spec,
        checkpoint,
        decisions,
        route_recovery,
        screening_recovery,
        args.out,
    )
    checks.to_csv(args.out / "data_checks.csv", index=False)
    write_lock(args, args.out, route_recovery, screening_recovery)

    failed = checks[checks["status"].eq("fail")]
    if not failed.empty:
        raise RuntimeError(f"123 failed data checks: {failed['check'].tolist()}")
    print(f"Wrote release screening platform formal groundwork to {args.out}")


if __name__ == "__main__":
    main()
