# PLGA Paper Collection Lane Contract, 2026-06-14

This document defines a dedicated execution lane for PLGA paper triage,
DOI/PDF matching, formulation-level fact extraction, and curve-mapping
groundwork.

It exists for one reason:

```text
the collection lane must stay separate from model invention,
claim drift, and assay-leakage mistakes
```

## Scope

This lane is allowed to do only the following:

1. build and audit DOI/PDF manifests for the 113-paper PLGA corpus,
2. triage PLGA papers for extraction yield,
3. extract legal `X_i^base`, `R_i^model`, `H_i^available`, and
   `R_i^audit` facts from paper text,
4. improve formulation-to-curve mapping readiness,
5. record unresolved ambiguities explicitly.

This lane is not allowed to:

- train any predictive model,
- start any new experiment number,
- claim route-model readiness,
- infer hidden process facts from release figures alone,
- promote assay metadata into `R_i^model`,
- silently rewrite contract boundaries.

## Primary Inputs

Read these before touching any collection artifact:

1. `README.md`
2. `docs/script_route_decision_map_2026-06-14.md`
3. `docs/release_screening_platform_design_contract_2026-06-14.md`
4. `docs/release_screening_platform_mathematical_contract_2026-06-14.md`
5. `docs/plga_paper_feature_extraction_priority_2026-06-14.md`
6. `docs/tables/plga_extraction_template_guide_2026-06-14.md`

## Canonical Working Files

Use these files as the main collection interface:

- `docs/tables/plga_paper_triage_template_2026-06-14.csv`
- `docs/tables/plga_formulation_feature_extraction_template_2026-06-14.csv`

Use these as precedents:

- `docs/tables/plga_paper_triage_example_caseinate_2026-06-14.csv`
- `docs/tables/plga_formulation_feature_extraction_example_caseinate_2026-06-14.csv`

## Durable Artifact Rule

Do not rely on `data/*.csv` as the only durable output for this lane.

Current repository policy ignores `data/**`, so any crucial corpus manifest
that lives only under `data/` is local-state only, not durable project state.

That means:

- local scratch manifests under `data/` are allowed,
- but any collection result needed by future agents must be mirrored into a
  tracked location such as `docs/` or another committed path,
- or must be summarized in a committed markdown audit note before handoff.

## Output Semantics

### Triage file

`n_formulations_reported` means:

```text
how many formulations the paper itself reports
```

It does not automatically mean:

```text
how many local release curves are already mapped
```

If paper-reported formulations exceed locally digitized curves for the same
DOI, say so explicitly in `curve_mapping_note` or `notes`.

### Formulation extraction file

`source_dataset=cross321` means:

```text
this DOI belongs to the PLGA 321 corpus
```

It does not mean:

```text
the formulation row is already mapped to a specific local curve id
```

Use these combinations honestly:

- `source_dataset=cross321` + `curve_mapping_status=unmapped`
- `source_dataset=cross321` + `local_curve_id=(not mapped yet)`
- `curve_mapping_note=<why the mapping is still unresolved>`

Do not write `"(not in cross321)"` for a DOI already confirmed by the corpus
manifest.

## Allowed Fact Types

### Must prioritize

- route family
- route subfamily
- ordered unit operations
- explicit process settings
- polymer identity and grade
- polymer Mw and LA:GA
- drug identity
- planned composition variables
- post-process morphology / size / EE / loading proxies

### Must demote or exclude

- dialysis setup
- membrane MWCO
- sampling schedule
- HPLC wavelength
- release-medium conditions used only for assay
- curve-derived burst labels
- source bookkeeping labels

If uncertain, keep the raw snippet and lower confidence.

## Worker Procedure

For each paper:

1. confirm the DOI and local PDF,
2. decide whether the paper belongs to the PLGA 113-paper corpus,
3. write or update one triage row,
4. extract formulation-level facts one row at a time,
5. preserve raw provenance,
6. stop if mapping becomes ambiguous and mark it unresolved.

Do not skip directly to bulk extraction if triage is still weak.

## Stop Conditions

The worker should stop and report instead of guessing when:

1. the paper provides release curves but no formulation-level preparation text,
2. multiple paper formulations could map to one local curve with no clean rule,
3. the route is only implied by figure style or author discussion,
4. a field could be either assay metadata or manufacturing metadata and the
   text is not clear,
5. the next step would require model training or new experiment design.

## Collection Success Criteria

This lane is succeeding if it increases one or more of these:

1. number of DOI-confirmed PLGA papers with local PDFs,
2. number of high-yield triaged PLGA papers,
3. number of provenance-clean `R_i^model` facts,
4. number of papers with explicit unresolved mapping notes rather than silent
   ambiguity,
5. number of cross-source repeated route families.

This lane is failing if it mainly increases:

- untracked local scratch files,
- assay metadata rows,
- guessed route labels,
- or pseudo-progress summaries disconnected from committed artifacts.

## Branch Intent

This collection lane should live on a branch dedicated to paper triage,
mapping, and extraction work.

It should not be used as the branch where:

- new model routes are prototyped,
- screening claims are escalated,
- or manuscript headline framing is changed.
