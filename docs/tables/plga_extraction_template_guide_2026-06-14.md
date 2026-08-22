# PLGA Extraction Template Guide, 2026-06-14

These templates are the operational companion to
`plga_paper_feature_extraction_priority_2026-06-14.md`.

Files:

- `plga_paper_triage_template_2026-06-14.csv`
- `plga_formulation_feature_extraction_template_2026-06-14.csv`
- `plga_paper_triage_example_caseinate_2026-06-14.csv`
- `plga_formulation_feature_extraction_example_caseinate_2026-06-14.csv`

## How To Use

### Step 1: Triage One Paper

Fill one row in `plga_paper_triage_template_2026-06-14.csv` per paper.

Goal:

- decide whether the paper is high-yield for route/process recovery,
- decide whether formulation-to-curve mapping is possible,
- decide whether the paper should move to formulation-level extraction.

Recommended categorical values:

- `curve_mapping_possible`: `yes`, `partial`, `no`, `unknown`
- `has_*` flags: `yes`, `no`, `unknown`
- `priority_tier`: `high_yield`, `medium`, `low`, `defer`
- `triage_status`: `not_started`, `triaged`, `needs_followup`, `completed`
- `triage_confidence`: `low`, `medium`, `high`

### Step 2: Extract Formulation-Level Facts

Fill one row in `plga_formulation_feature_extraction_template_2026-06-14.csv`
per extracted fact, not per formulation.

That means one formulation may produce many rows:

- one row for route family,
- several rows for unit operations,
- several rows for reported process conditions,
- several rows for optional state proxies,
- several rows for base descriptors that were missing from the standardized
  pool.

Use the caseinate example files as the first concrete precedent before opening
new papers.

Recommended categorical values:

- `curve_mapping_status`: `mapped`, `partial`, `unmapped`, `ambiguous`
- `contract_bucket`: `X_i^base`, `R_i^model`, `H_i^available`, `R_i^audit`, `excluded`
- `field_group`: `route_fact`, `process_condition`, `design_descriptor`, `optional_proxy`, `audit_only`, `excluded`
- `is_design_stage_input`: `True`, `False`
- `is_optional_measurement`: `True`, `False`
- `is_curve_derived`: `True`, `False`
- `allowed_in_main_model`: `True`, `False`
- `mapping_confidence`: `exact`, `high`, `medium`, `low`, `unresolved`
- `review_status`: `not_started`, `auto_extracted`, `needs_quick_spotcheck`, `needs_human_confirmation`, `reviewed`

## Field Intent

### Triage template

- `route_family_explicit`: whether the paper explicitly names the preparation
  route
- `shared_route_family_candidate`: whether the route looks likely to recur
  across multiple papers
- `has_formulation_level_process_variation`: whether multiple formulations vary
  process rather than only chemistry
- `has_formula_level_proxy_measurements`: whether size/loading/morphology etc.
  are reported per formulation rather than only in aggregate

### Formulation extraction template

- `normalized_field_name`: standardized field name you want later in the clean
  extraction table
- `contract_bucket`: legal destination of the field in the screening contract
- `field_group`: practical grouping for later filtering
- `exclusion_reason`: required when `contract_bucket=excluded` or
  `allowed_in_main_model=False`

## Minimal Working Rule

When uncertain:

- keep the raw snippet,
- lower `mapping_confidence`,
- set `needs_human_review=True`,
- do not silently promote the fact into `R_i^model`.

If a field smells like assay metadata, schedule identity, or curve-derived
summary, mark it `excluded` or `R_i^audit`, not `R_i^model`.
