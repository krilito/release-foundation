# GitHub Literature Extraction Skill Scan, 2026-06-14

This note records a quick GitHub scan for reusable skill patterns related to
literature review, paper extraction, and document extraction.

## What Was Found

The scan found useful generic patterns, but no existing skill matched the exact
needs of this repository.

Representative references:

- `Imbad0202/academic-research-skills-codex`
  - Codex-native academic research skill suite
  - useful as an example of packaging and workflow structure
- `claude-office-skills/skills/data-extractor`
  - document extraction skill for turning PDFs and documents into structured
    data
  - useful as an example of extraction-first framing
- `bytedance/deer-flow` `systematic-literature-review`
  - systematic literature review skill with structured paper synthesis
  - useful as an example of metadata schema and multi-paper review flow

## What Was Missing

None of the discovered skills enforce the contract boundary this project needs:

- `X_i^base`
- `R_i^model`
- `H_i^available`
- `R_i^audit`
- `excluded`

They are mostly built for:

- broad literature search
- general review writing
- generic document extraction
- multi-paper synthesis

They do not solve the core local problem:

```text
how to read PLGA formulation papers and extract only legal,
formulation-level, provenance-clean pre-observation fields
without mixing in assay leakage or curve-derived facts
```

## Design Decision

Because of that gap, the project should use a custom local skill rather than
importing a generic GitHub literature-review skill unchanged.

The custom skill should:

1. work directly against the 321-curve / 113-paper PLGA corpus,
2. point to local repo templates and examples,
3. force contract-bucket assignment during extraction,
4. preserve raw provenance for every extracted fact,
5. explicitly reject assay metadata and post hoc curve summaries.

## Current Outcome

That custom skill is now created locally as:

`D:\cache\.codex\skills\plga-paper-extraction`

Its operational repo companions are:

- `docs/plga_paper_feature_extraction_priority_2026-06-14.md`
- `docs/tables/plga_paper_triage_template_2026-06-14.csv`
- `docs/tables/plga_formulation_feature_extraction_template_2026-06-14.csv`
- `docs/tables/plga_paper_triage_example_caseinate_2026-06-14.csv`
- `docs/tables/plga_formulation_feature_extraction_example_caseinate_2026-06-14.csv`
- `docs/tables/plga_extraction_template_guide_2026-06-14.md`
