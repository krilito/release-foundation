# PLGA Paper Feature Extraction Priority, 2026-06-14

This note answers one practical question:

```text
If 321 PLGA curves are already cleaned and linked to 113 papers,
what should be collected next from the papers themselves?
```

The answer is not "collect everything." The answer is:

```text
collect only pre-observation descriptors, factual process-route records,
and optional post-process state proxies that can strengthen the release-state
story without introducing assay leakage.
```

## Short Answer

Yes, paper-level extraction is now the right next move.

But the target is not "more columns."

The target is:

1. recover legal `R_i^model` fields from paper methods and supplement tables,
2. strengthen missing `X_i^base` fields that are genuinely available at design
   or process-record stage,
3. collect `H_i^available` proxies as an optional measurement menu,
4. explicitly reject assay metadata and curve-derived features.

Operational templates now live in:

- `docs/tables/plga_paper_triage_template_2026-06-14.csv`
- `docs/tables/plga_formulation_feature_extraction_template_2026-06-14.csv`
- `docs/tables/plga_extraction_template_guide_2026-06-14.md`

## Why This Is The Right Next Step

Current evidence already says:

- static descriptors are not enough under strict PLGA splits,
- measured early observations carry information not synthesized by current
  static `X`,
- route facts are scientifically relevant but locally under-recovered,
- the screening route is blocked mostly by observability and provenance, not by
  lack of model families.

That means the next data effort should improve observability, not just row
count.

## Extraction Unit

The extraction unit is one declared formulation record aligned to one or more
release curves from one paper.

Every extracted field must preserve:

- `paper_id` or DOI,
- formulation identifier as reported in the paper,
- section / table / figure anchor,
- exact raw snippet or normalized paraphrase target,
- unit,
- confidence,
- ambiguity flag when mapping is imperfect.

If the paper does not support formulation-to-curve alignment, the route record
should be marked unresolved rather than guessed.

## Tier 1: Must Extract For `R_i^model`

These are the highest-priority paper fields because they are currently the
largest missing legal input block.

| Group | Examples | Contract bucket |
|---|---|---|
| Route family | single emulsion, double emulsion, nanoprecipitation, electrospraying, solvent casting, spray drying | `R_i^model` |
| Route subfamily | `o/w`, `w/o/w`, coaxial electrospraying, aqueous casting, solvent evaporation subtype | `R_i^model` |
| Unit-operation sequence | dissolve polymer -> dissolve API -> emulsify -> harden -> wash -> dry | `R_i^model` |
| Operation order | which step came first and what was added when | `R_i^model` |
| Reported process conditions | sonication time, homogenization rpm, electrospray voltage, feed rate, temperature, mixing time | `R_i^model` |
| Phase construction | organic phase, aqueous phase, internal water phase, shell/core designation | `R_i^model` |
| Solvents and stabilizers | DCM, acetone, ethyl acetate, PVA, surfactant identity | `R_i^model` |
| Post-processing | solvent evaporation, curing, hardening bath, centrifugation, washing, lyophilization, vacuum drying | `R_i^model` |

These fields should be read from:

- Methods sections
- Supplementary methods
- formulation tables
- figure captions only when they explicitly state preparation facts

These fields should not be inferred from release figures.

## Tier 2: Strengthen `X_i^base`

These are legal static descriptors when they are reported as planned or known
design/process inputs rather than measured after fabrication.

| Group | Examples | Contract bucket |
|---|---|---|
| Drug descriptors | MW, logP, pKa, solubility class, charge state if reported | `X_i^base` |
| Polymer descriptors | PLGA Mw, LA:GA ratio, end-cap chemistry, inherent viscosity, polymer grade | `X_i^base` |
| Planned composition | target drug:polymer ratio, target loading recipe, excipient composition | `X_i^base` |
| Process-input concentrations | polymer concentration, drug concentration, phase volumes, stabilizer concentration | `X_i^base` or `R_i^model` depending on whether it is a design variable or operation condition |
| Medium descriptors | pH, buffer identity, ionic strength, temperature, sink setup if part of experimental environment definition | `X_i^base` |

Rule of thumb:

- if it exists before fabrication as part of the intended recipe, it can live
  in `X_i^base`;
- if it is a recorded manufacturing step or setting, it belongs in
  `R_i^model`.

## Tier 3: Collect As `H_i^available`, Not As Base Inputs

These are worth collecting, but they must stay separate from pure design-stage
inputs.

| Group | Examples | Contract bucket |
|---|---|---|
| Particle geometry | particle size, size distribution, PDI, shell thickness | `H_i^available` |
| Surface and charge | zeta potential, surface chemistry readouts | `H_i^available` |
| Loading outcomes | actual drug loading, encapsulation efficiency | `H_i^available` |
| Morphology | SEM shape class, porosity, roughness, hollow/core-shell confirmation | `H_i^available` |
| Physical state | crystallinity, Tg, residual solvent, degradation onset, swelling | `H_i^available` |

These fields are important because they may explain missing release state.

They are not clean `X_i^base` unless the dataset explicitly declares them as
already measured and available at decision time.

## What Not To Collect Into Model Inputs

These should be recorded only as audit metadata or excluded entirely.

| Bad field type | Examples | Why excluded |
|---|---|---|
| Release-assay metadata | dialysis bag, membrane MWCO, sampling schedule, assay wavelength, HPLC method | assay, not manufacturing |
| Curve-derived labels | burst fraction from fitted curve, t50 estimated from the plotted curve only, latent label computed after seeing full release | future leakage or post hoc target |
| Source bookkeeping | sheet name, file name, paper grouping tags | source proxy, not science |
| Author interpretation labels | "fast release", "sustained", "biphasic" if not source-declared process facts | audit-only, not `R_i^model` |

## Paper Triage Order For The 113 PLGA Papers

Do not deep-read all 113 papers equally at the start.

Prioritize papers with:

1. explicit formulation-method paragraphs,
2. supplementary formulation tables with per-formulation process settings,
3. repeated route families across more than one DOI,
4. multiple formulations that vary one process factor while keeping chemistry
   mostly stable,
5. post-process proxy measurements such as size, loading, or morphology.

Deprioritize papers that provide:

- release curves only,
- assay detail without preparation detail,
- pooled qualitative statements with no formulation-level mapping,
- figures that cannot be aligned to the standardized curve records.

## Recommended First Pass Workflow

### Pass 1: Paper Triage

For each paper, record:

- whether formulation-to-curve alignment is possible,
- whether method text exists,
- whether route family is explicit,
- whether operation settings are tabulated,
- whether optional proxies are reported.

### Pass 2: High-Yield Extraction

Extract Tier 1 `R_i^model` fields first.

Goal:

- maximize cross-source route recovery,
- not maximize total column count.

### Pass 3: Static And Proxy Enrichment

After route extraction stabilizes:

- fill missing `X_i^base` recipe/design fields,
- then collect `H_i^available` proxy measurements.

## Minimum Provenance Fields Per Extracted Fact

Every extracted fact should carry:

- `doi`
- `paper_short_id`
- `source_section`
- `source_table_or_figure`
- `paper_formulation_label`
- `curve_mapping_status`
- `raw_text_snippet`
- `normalized_field_name`
- `normalized_value`
- `normalized_unit`
- `contract_bucket`
- `mapping_confidence`
- `needs_human_review`

## Success Criterion

This extraction pass succeeds if it produces:

1. more multi-source support for shared PLGA route families,
2. more clean separation between `X_i^base`, `R_i^model`, and `H_i^available`,
3. a better observability story for why static-only prediction fails,
4. a legal basis for later within-system screening or route-aware prior work.

It does not need to prove that route tokens alone solve PLGA prediction.

## Current Recommendation

If work starts now, start with paper-level triage and route/process extraction,
not with a new model and not with more curve scraping.

The next useful question is:

```text
Across the 113 PLGA papers, how many formulations expose
cross-source, formulation-level, provenance-clean route facts
and optional state proxies that can legally enter the screening contract?
```
