# PLGA Paper Collection Worker Prompt, 2026-06-14

Use this prompt when handing the PLGA collection lane to another AI that needs
high structure and low freedom.

```text
You are working inside D:\release-foundation on the dedicated PLGA paper
collection lane.

Your task is not modeling.
Your task is not new experiments.
Your task is not paper writing.

Your only job is to improve the PLGA paper-collection state safely.

Before doing anything, read:
1. README.md
2. docs/script_route_decision_map_2026-06-14.md
3. docs/release_screening_platform_design_contract_2026-06-14.md
4. docs/release_screening_platform_mathematical_contract_2026-06-14.md
5. docs/plga_paper_feature_extraction_priority_2026-06-14.md
6. docs/plga_paper_collection_lane_contract_2026-06-14.md
7. docs/tables/plga_extraction_template_guide_2026-06-14.md

Then follow these rules:

- Do not train any model.
- Do not start a new experiment number.
- Do not invent route labels from release figures.
- Do not move assay metadata into R_i^model.
- Do not change CLAUDE.md.
- Do not overwrite user-edited files without checking git status first.
- Do not put the only copy of a crucial manifest under data/, because data/**
  is gitignored.

Main files you are allowed to extend:
- docs/tables/plga_paper_triage_template_2026-06-14.csv
- docs/tables/plga_formulation_feature_extraction_template_2026-06-14.csv

Use these examples first:
- docs/tables/plga_paper_triage_example_caseinate_2026-06-14.csv
- docs/tables/plga_formulation_feature_extraction_example_caseinate_2026-06-14.csv

For each paper:
1. confirm DOI and local PDF status,
2. confirm whether the DOI belongs to the PLGA 113-paper corpus,
3. add or refine one triage row,
4. only then extract formulation-level facts,
5. keep one fact per row,
6. preserve raw snippets and source anchors,
7. if curve mapping is unresolved, say so explicitly.

Use these semantics strictly:
- n_formulations_reported = number reported by the paper
- n_curves_linked_local = number already linked locally
- source_dataset=cross321 can coexist with curve_mapping_status=unmapped
- local_curve_id may stay "(not mapped yet)" if not truly resolved

When uncertain:
- lower confidence
- mark needs_human_review=True
- keep the raw snippet
- stop instead of guessing

At the end of the task, report only:
1. which papers were triaged,
2. how many fact rows were added,
3. which mappings remain unresolved,
4. whether any durable manifest still exists only under data/.
```
