# Release Screening Platform Formal Groundwork

Date: 2026-06-14

## Purpose

Experiment 123 turns the screening-platform contracts into concrete audit artifacts. 123A separated `X_i^base`, `H_i^available`, and route-vs-audit channels at the schema level. 123B then asked which formulation-level preparation facts actually exist locally, are provenance-traceable, and are legal for `R_i^model`. 123C now asks which screening targets are legal, which budgets leave real future windows, and whether any classical screening task is actually admissible.

## 123B Objective

The success criterion is not maximizing route-token count. It is establishing a preparation-route channel that does not let release assay metadata, source identity, or audit hypotheses masquerade as manufacturing facts.

It does not train a new model. It asks whether the current standardized release pools already support:

```text
X_i^base
R_i^model
R_i^audit
H_i^available / H_i^obs
I_i^(0), I_i^(H), I_i^(k)
future-only / target-future-only evaluation
```

## Preparation-Route Fact Definition

`R_i^model` may contain only factual manufacturing information: route family, unit-operation sequence, operation order, reported process conditions, reported materials or solvents, and explicit post-processing steps. Unknown values must stay explicit as `not_reported`, `source_document_required`, or `operation_sequence_partial`.

## Preparation vs Assay Boundary

The following standardized fields are rejected from `R_i^model`: `release_method`, `measurement_assay`, `phase_sequence`, `experimental_panel`, `source_sheet`, `worksheet_title`, and other source/audit bookkeeping fields. Terms such as `dialysis`, `filtration`, or `centrifugation` are only legal route facts when the local source context shows they belong to manufacturing rather than to the release assay.

## R_model vs R_audit Boundary

`R_i^model` contains only source-reported facts. `R_i^audit` remains the home for expected state effects, curve-symptom hypotheses, source leakage risk, and confound annotations. 123B never back-fills `R_i^model` from those audit hypotheses.

## Provenance Requirements

Every recovered route fact now carries a provenance pointer in `preparation_route_fact_provenance.csv`: local source file, source row/table span, raw process text, mapping confidence, extraction method, and review status.

## Route Taxonomy v2

A new `process_route_taxonomy_v2.csv` defines route families, subfamilies, allowed synonyms, excluded assay contexts, and unit-operation roles. It includes explicit room for `unknown_condition`, `not_reported_condition`, and `partial_sequence` tokens rather than silent imputation.

## Coverage Reality

- Total formulations audited: `784`
- Formulations with any preparation fact: `6`
- Formulations with sequence information: `6`
- Formulations with reported conditions: `4`
- Route families recovered: `2`
- R_model-eligible formulations: `6`
- Unresolved formulations: `778`

The only strong local route recovery came from the sodium-caseinate film source document and the PLGA-doxorubicin nanoparticle source-title text. Most other corpora still lack formulation-level preparation text in local standardized form.

## Source-Confounding Reality

- Multi-source routes recovered: `0`
- Single-source routes recovered: `2`

Every currently recovered route family is source-bound. That means route identity is still too close to source identity for a fair cross-source modeling claim.

## Route-Model Entry Gate

A route model would require all of the following: formulation-level route facts, provenance traceability, clear R_model vs R_audit separation, non-trivial support for multiple route families, at least key routes with multi-source support, acceptable source confounding, feasible leave-source evaluation, and adequate mapping confidence. The current audit fails the cross-source support and confounding gates.

## Current Decision

- Base descriptors and optional state proxies can be separated in the schema.
- Audit-only process metadata can be separated from model-facing route facts.
- A narrow, provenance-clean preparation-route channel now exists locally, but it is still source-bound.
- `route_model_allowed = false`.
- `neural_entry_allowed = false`.

## Next Required Action

Recover missing formulation-method text or supplementary process tables for the unresolved source queue before any route-token baseline or neural screening route is considered.

## 123C Objective

Lock target legality before any screening baseline exists. The success criterion is not finding the best target. It is proving which targets are computable, which remain future-only after an early budget, which are cross-system versus within-system only, and whether source or schedule confounding still blocks fair evaluation.

## Curve Data Reality

- Canonical formulations audited: `533`
- Unique curves audited: `784`
- Valid timepoints: `13153`
- Duplicate-timepoint curves flagged: `3`
- Nonmonotone curves flagged: `278`

A canonical formulation can map to multiple curves when the same source-side formulation appears under multiple assay conditions or curve panels. 123C therefore keeps `formulation_id` and `curve_id` separate and audits target legality at the curve level first.

## Screening Target Layers

Layer A currently includes objective curve phenotypes such as `q_day1`, `t10_days`, `t25_days`, `t50_days`, `final_release_fraction`, and `incomplete_release_final_lt80`.

Layer B currently includes `normalized_t50_fraction_of_horizon`, but it remains conditional because normalization uses the full observed horizon and therefore cannot be a pre-observation input.

Layer C remains blocked at repository scope: a global application-conditioned utility target is still marked `application_target_not_available`.

## Observation-Budget Audit

123C audits point-count budgets (`k = 0, 1, 2, 3, 5`), absolute-time budgets (`6 h`, `24 h`, `3 d`, `7 d`), and normalized-horizon budgets (`10%`, `20%`). All budgets enforce observed-versus-future disjointness, and a target is future-eligible only if its main evaluation window remains in the unobserved future.

## Cross-System Versus Within-System Reality

- Cross-system targets available: `0`
- Within-system targets available: `6`

Current absolute-time and final-horizon targets are not safe cross-system labels. The normalized target family exists as a candidate language, but the current support, censoring, and schedule/source structure still do not justify a cross-system main claim.

## Sampling-Schedule Leakage Reality

Sampling schedules are often source-specific. 123C therefore exports `sampling_schedule_source_audit.csv` and treats schedule-identifiable sources as a leakage risk requiring leave-source evaluation, common-grid evaluation, or other mitigation before baseline training.

## 123C Decision

- `cross_system_target_available = false`.
- `within_system_targets_available = true`.
- `classical_screening_entry_allowed = true` only in the within-system scope declared by the feasibility matrix.
- `neural_entry_allowed = false`.

## 123C Next Required Action

If Experiment 124 starts later, it should start with a classical within-system screening baseline only on target-budget pairs that remain future-valid and leave-source-feasible. Cross-system screening remains blocked until target comparability and schedule/source confounding improve.

## Output Anchor

`../outputs/123_release_screening_platform_formal_groundwork/`
