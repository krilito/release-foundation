# Documentation Index

Last cleanup pass: 2026-06-14.

This file is a reading guide, not a second source of truth. When documents
conflict, use this authority order:

1. current claim table and durable outputs from `outputs/`
2. locked requirements / preregistration docs
3. historical closeout and roadmap notes
4. archived sprint notes

## Quick Paths

If you want the shortest possible re-entry path, use one of these:

### I want the current answer

1. `script_route_decision_map_2026-06-14.md`
2. `plga_information_budget_claim_table_2026-06-11.md`
3. `release_screening_platform_formal_groundwork_2026-06-14.md`

### I want to know why older routes were killed

1. `project_archaeology_inventory_2026-06-12.md`
2. `script_route_decision_map_2026-06-14.md`
3. `README.md`

### I want to continue current work without drifting

1. `README.md`
2. `project_archaeology_inventory_2026-06-12.md`
3. `release_state_inference_formalism_2026-06-13.md`
4. `release_screening_platform_design_contract_2026-06-14.md`
5. `release_screening_platform_mathematical_contract_2026-06-14.md`
6. `release_screening_platform_formal_groundwork_2026-06-14.md`

## Read First

| Path | Purpose |
|---|---|
| `../README.md` | current project framing and reproduction entry points |
| `../AGENTS.md` | project-specific AI/coding rules and current research framing |
| `../DECISIONS.md` | ADR history and project-state ledger |
| `project_archaeology_inventory_2026-06-12.md` | frozen archaeology and method inventory; prevents drift back to retired SRDS, PySR-main, RSSM, LNN/CNP fantasy, or architecture-novelty routes |
| `script_route_decision_map_2026-06-14.md` | decision-oriented script classification: which numbered routes remain main-claim material, which are supplement-only, and which are retired |
| `github_literature_extraction_skill_scan_2026-06-14.md` | quick GitHub scan of literature-extraction skills and why this repo still needs a custom PLGA extraction skill |
| `plga_paper_collection_lane_contract_2026-06-14.md` | dedicated collection-lane contract: what another agent may collect, where durable outputs should live, and what must stay out of scope |
| `plga_paper_collection_worker_prompt_2026-06-14.md` | reusable low-autonomy worker prompt for handing PLGA paper triage / extraction work to another AI |
| `unified_drug_release_ml_problem_solution_2026-06-12.md` | current unified framing: release-state inference, design utility, observation budget, and descriptor enrichment |
| `release_state_inference_formalism_2026-06-13.md` | formal state-space contract for scripts `120+`: defines release state spaces, static priors, observation updates, residuals, and symptoms before RSF/MSVS scoring |
| `plga_information_budget_freeze_reproduction_2026-06-12.md` | frozen PLGA claim/reproduction contract |
| `plga_information_budget_claim_table_2026-06-11.md` | current PLGA route and main evidence table |
| `plga_observation_budget_goal_design_2026-06-11.md` | next `/goal` design: observation value, uncertainty contraction, stopping rules |
| `plga_observation_budget_curve_e1_2026-06-11.md` | first observation-budget result: window-matched early-observation value |
| `plga_empirical_uncertainty_contraction_e2_2026-06-11.md` | second PLGA supplement: empirical uncertainty contraction from early observations |
| `plga_timepoint_value_e3_2026-06-11.md` | third PLGA supplement: value ranking of observed early timepoints |
| `plga_stopping_rule_e4_2026-06-12.md` | fourth PLGA supplement: conservative early-stopping rule simulation |
| `plga_route_agreement_e5_2026-06-12.md` | final PLGA supplement: route agreement audit and manuscript claim set |
| `liposome_observation_budget_goal_design_2026-06-12.md` | next `/goal` design: second-system static-first and early-observation validation |
| `liposome_author_paper_reading_2026-06-12.md` | completed full reading of the Yanes et al. liposome IVR ML paper: purpose, workflow, technical limits, and relevance |
| `liposome_candidate_audit_e0_2026-06-12.md` | completed liposome E0 gate: corpus, descriptors, split feasibility |
| `liposome_static_prediction_e1_2026-06-12.md` | completed liposome E1 gate: static-only prediction failed as sufficient model |
| `liposome_bottleneck_decomposition_e2_2026-06-12.md` | completed liposome E2 gate: bottleneck decomposition justifies early-observation test |
| `liposome_author_baseline_audit_2026-06-12.md` | completed author-baseline audit: public accelerated_IVR code predicts kinetic class, not continuous future curves |
| `liposome_middle_layer_probe_2026-06-12.md` | completed first liposome middle-layer probe: class prototypes have signal but are weaker than early/theta routes under strict splits |
| `liposome_theta_prototype_mixture_2026-06-12.md` | completed finer liposome middle-layer probe: local theta prototypes improve routing but do not beat early-only fitting under strict splits |
| `liposome_curve_dictionary_static_probe_2026-06-12.md` | completed static curve-dictionary probe: curve language is strong, static-to-coefficients works partially under strict splits |
| `liposome_latent_gap_observation_budget_2026-06-12.md` | completed liposome latent-gap probe: early observations close part of the curve-language gap, strongly in stratified split and weakly under release-method transfer |
| `liposome_design_utility_benchmark_2026-06-12.md` | completed liposome design-utility probe: static priors can enrich pre-observation candidate selection even when curve RMSE remains limited |
| `liposome_descriptor_enrichment_rdkit_2026-06-12.md` | completed RDKit/PubChem descriptor enrichment audit: molecular descriptors do not close the strict group-by-API static gap |
| `minimal_drug_release_world_model_synthesis_2026-06-12.md` | current synthesis across PLGA and liposome: partial support for minimal release-state world model framing |
| `synthetic_hidden_state_identifiability_audit_2026-06-12.md` | synthetic hidden-state audit plus sensitivity defense: missing microstructure/process variables can cause static-X failure, and early Q acts as empirical release-state measurement |
| `plga_synthetic_state_real_projection_2026-06-12.md` | real 321 PLGA bridge diagnostic: projects curves into simulation-informed H_like coordinates, finds proxy associations, but flags high OOD coverage |
| `release_state_forensics_algorithm_framework_plan_2026-06-13.md` | proposed original RSF/MSVS framework: residual-to-symptom-to-measurement recommendation route for missing-state discovery |
| `plga_missing_state_residual_map_2026-06-13.md` | completed Experiment 120: PLGA release-state residual map; static priors help in random folds but not strict source/method transfer, with stable residuals and strong source structure |
| `plga_candidate_measurement_value_scoring_2026-06-13.md` | Experiment 121: candidate measurement scoring from strict OOF 120 residuals; separates evidence-backed proxies from hypothesis-only future measurements |
| `release_dataset_candidate_audit_and_gate_2026-06-14.md` | Experiment 122: candidate release-dataset audit plus minimal local training gate; identifies hydrogel/chitosan/tablet collection targets without starting a new model route |
| `plga_paper_feature_extraction_priority_2026-06-14.md` | paper-level PLGA extraction guide: what to collect from the 113 papers as `X_i^base`, `R_i^model`, `H_i^available`, and what to exclude |
| `tables/plga_extraction_template_guide_2026-06-14.md` | companion guide for the PLGA paper-triage and formulation-level extraction CSV templates |
| `tables/plga_paper_triage_template_2026-06-14.csv` | one-row-per-paper triage template for the 113 PLGA papers |
| `tables/plga_formulation_feature_extraction_template_2026-06-14.csv` | one-row-per-fact extraction template for legal `X_i^base`, `R_i^model`, `H_i^available`, and excluded fields |
| `tables/plga_paper_triage_example_caseinate_2026-06-14.csv` | caseinate paper triage example showing one complete high-yield paper row |
| `tables/plga_formulation_feature_extraction_example_caseinate_2026-06-14.csv` | caseinate formulation-level extraction example showing route facts, process conditions, and one excluded assay row |
| `release_screening_platform_design_contract_2026-06-14.md` | design contract for the observation-budgeted release screening platform: descriptor ontology, process routing, utility gates, neural entry gate, and wet-lab validation logic |
| `release_screening_platform_mathematical_contract_2026-06-14.md` | mathematical contract for `z0`, process route tokens, identifiability limits, screening utility, measurement value, and Experiment 123 gates |
| `release_screening_platform_formal_groundwork_2026-06-14.md` | Experiment 123A/123B/123C groundwork result: canonical field schema, route provenance audit, screening target/budget legality, coverage/confounding reality, assay-vs-route boundary, and explicit reason route/neural modeling is still blocked |
| `paper_requirements_locked_2026-05-28.md` | statistical rigor, baselines, reproducibility, kill criteria |
| `research_nonnegotiables_2026-05-28.md` | high-level research constraints |

## Current PLGA Route

The active PLGA paper route is:

```text
static formulation descriptors under-identify release;
measured early observations reveal missing release state;
the research problem is observation-budgeted future-release forecasting.
```

Current main evidence:

- source-group strict split: static-after-k5 `0.267` -> measured-early-k5
  `0.160`, gain `0.107` RMSE.
- source-dataset strict split: static-after-k5 `0.272` -> measured-early-k5
  `0.166`, gain `0.106` RMSE.
- author-style fixed early baseline independently improves strict-split RMSE.
- static-predicted early points fail to replace measured early points.

Primary output anchor:

| Path | Role |
|---|---|
| `../outputs/102_plga_information_budget_claim_table/` | unified claim table |

Supporting output anchors:

| Path | Role |
|---|---|
| `../outputs/97_plga_direct_early_blackbox_control/` | direct static-vs-early control |
| `../outputs/99_plga_static_to_early_proxy_probe/` | static-to-early proxy failure |
| `../outputs/101_lai_author_style_on_our_splits/` | original LAI author-style baseline on our splits |
| `../outputs/96_plga_early_middle_layer_selector_probe/` | middle-layer early-conditioned selector |
| `../outputs/91_plga_static_feature_ceiling_*` | static descriptor ceiling probes |

## Current Next Work

The PLGA E1-E5 supplement chain is complete and frozen. The liposome
second-system chain through descriptor enrichment is also complete. The
synthetic hidden-state audit in `118_synthetic_hidden_state_identifiability_audit.py`
is a completed diagnostic with a hidden-strength sensitivity defense. It
explains why missing microstructure/process state can create the observed
static-feature ceiling, while keeping early Q framed as empirical release-state
measurement rather than physical hidden-variable recovery. The real 321 PLGA
projection bridge in `119_plga_synthetic_state_real_projection.py` finds
simulation-aligned proxy structure but also flags high OOD coverage, so it is
a cautious bridge diagnostic rather than real hidden-variable validation. The
formal contract for scripts `120+` is
`release_state_inference_formalism_2026-06-13.md` plus `../release_state.py`.
The proposed original-method direction is RSF/MSVS in
`release_state_forensics_algorithm_framework_plan_2026-06-13.md`, but RSF is
now explicitly application-layer: script 120 maps same-space release-state
residuals, and MSVS-style candidate measurement scoring begins only after that
residual map exists. Experiment 120 is now complete in
`120_plga_missing_state_residual_map.py` and
`plga_missing_state_residual_map_2026-06-13.md`: static descriptors show
random-fold signal but fail strict DOI/method transfer, residuals are stable
across representations, and DOI/source structure is substantial. Experiment
121 in `121_plga_candidate_measurement_value_scoring.py` and
`plga_candidate_measurement_value_scoring_2026-06-13.md` scores candidate
measurements from those strict OOF residuals. It keeps EvidenceScore
(available proxy support) separate from PriorityScore (future measurement
priority), and no-proxy candidates remain hypothesis-only. Experiment 122 in
`122_release_dataset_candidate_audit_and_gate.py` and
`release_dataset_candidate_audit_and_gate_2026-06-14.md` audits candidate
release datasets and runs a minimal gate only on eligible local standardized
curves. It identifies alginate hydrogel, chitosan nanoparticles, and direct
compression tablets as collection targets, while confirming that the only
current local training-eligible second-system pool remains liposome. The
screening-platform route is now constrained by
`release_screening_platform_design_contract_2026-06-14.md` and
`release_screening_platform_mathematical_contract_2026-06-14.md`: descriptor
coverage, process-route tokenization, `z0` identifiability, and utility
definitions must be settled before neural modeling. Experiment 123 in
`123_release_screening_platform_formal_groundwork.py` and
`release_screening_platform_formal_groundwork_2026-06-14.md` is now complete
through 123C as a formal-groundwork step: it generates the canonical
descriptor schema, dataset coverage audit, route taxonomy, audit-only
hypothesis registries, utility specification, checkpoint tables, a
formulation-level preparation-route provenance audit, and a screening
target/budget feasibility matrix. Its main result is that `X_i^base` and
optional state proxies can be separated cleanly, and a narrow preparation-route
channel can be recovered locally (`6 / 784` formulations; `solvent_casting`
and `electrospraying`). However, every recovered route is still single-source,
so route modeling and any neural screening-platform entry remain blocked by
source confounding rather than model capacity. The current screening green
light is narrower: only within-system PLGA classical screening tasks clear the
123C gate, while cross-system screening remains blocked. The next work should
therefore be unresolved-source method recovery, source-clean process field
extraction, and descriptor/utility/data-reality audit tightening rather than
generic model invention.

Do not start new model-invention branches. In particular, do not reopen SRDS,
PySR-as-main-method, RSSM/world-model training, LNN/CNP transfer fantasy, KAN,
PINN, RL, foundation-model, or generic neural architecture sprints unless a new
preregistered diagnostic first shows which release-state gap is being closed.

Completed liposome route:

1. use `112_liposome_middle_layer_probe.py` as the completed class/theta
   middle-layer starting point,
2. use `113_liposome_theta_prototype_mixture.py` as the completed finer
   prototype-mixture middle-layer test,
3. use `115_liposome_latent_gap_observation_budget.py` as the completed
   static-plus-measured-early PCA coefficient gap-closure test,
4. compare 115 against early-only fit, static+early theta, prototype mixture,
   and refined prototype initialization,
5. use the curve-dictionary route from
   `114_liposome_curve_dictionary_static_probe.py` as the static-only
   dictionary baseline before adding early-observation coefficients,
6. use `116_liposome_design_utility_benchmark.py` as the completed design
   utility test: static priors can enrich some pre-observation target rankings,
7. use `117_liposome_descriptor_enrichment_rdkit.py` as the completed
   descriptor audit: RDKit/PubChem descriptors do not close the strict
   group-by-API static gap,
8. use `minimal_drug_release_world_model_synthesis_2026-06-12.md` as the
   current cross-system synthesis and manuscript-story anchor,
9. use `synthetic_hidden_state_identifiability_audit_2026-06-12.md` only as a
   conceptual control and sensitivity defense for the hidden-state /
   early-observation argument,
10. use `plga_synthetic_state_real_projection_2026-06-12.md` only as a cautious
   real-data bridge: H_like proxy associations are useful, but high OOD coverage
   prevents simulator-validation claims,
11. use `release_state_inference_formalism_2026-06-13.md` and `../release_state.py`
   as the contract for script 120: residuals are valid only inside one
   declared `ReleaseStateSpace`,
12. use `release_state_forensics_algorithm_framework_plan_2026-06-13.md` as the
   critique target for the original algorithm idea: RSF/MSVS should recommend
   missing measurements, not claim discovered physical hidden variables,
13. use `plga_missing_state_residual_map_2026-06-13.md` as the completed 120
   result: it is the residual map prerequisite for 121, not a measurement
   recommendation result,
14. do not reframe this as mechanism transfer, foundation-model learning, or a
   neural architecture sprint.

Liposome E0 anchor:

| Path | Role |
|---|---|
| `../outputs/108_liposome_candidate_audit/` | completed second-system candidate audit; generated locally and ignored by git |
| `../outputs/109_liposome_static_whole_curve_prediction/` | completed E1 static-only prediction gate; generated locally and ignored by git |
| `../outputs/110_liposome_bottleneck_decomposition/` | completed E2 bottleneck decomposition; generated locally and ignored by git |
| `../outputs/111_liposome_author_baseline_audit/` | completed author-baseline reproduction/audit; generated locally and ignored by git |
| `../outputs/112_liposome_middle_layer_probe/` | completed 6 h early-window middle-layer probe; generated locally and ignored by git |
| `../outputs/113_liposome_theta_prototype_mixture/` | completed 6 h finer theta/prototype-mixture probe; generated locally and ignored by git |
| `../outputs/114_liposome_curve_dictionary_static_probe/` | completed static feature-to-curve-dictionary probe; generated locally and ignored by git |
| `../outputs/115_liposome_latent_gap_observation_budget/` | completed early-observation latent gap closure probe; generated locally and ignored by git |
| `../outputs/116_liposome_design_utility_benchmark/` | completed static-prior design utility benchmark; generated locally and ignored by git |
| `../outputs/117_liposome_descriptor_enrichment_rdkit/` | completed descriptor enrichment audit; generated locally and ignored by git |
| `../outputs/118_synthetic_hidden_state_identifiability_audit/` | completed synthetic hidden-state identifiability audit; generated locally and ignored by git |
| `../outputs/119_plga_synthetic_state_real_projection/` | completed real 321 PLGA projection bridge; generated locally and ignored by git |
| `../outputs/120_plga_missing_state_residual_map/` | completed PLGA missing-state residual map; generated locally and ignored by git |
| `../outputs/121_plga_candidate_measurement_value_scoring/` | candidate measurement scoring from 120 strict OOF residuals; generated locally and ignored by git |
| `../outputs/122_release_dataset_candidate_audit_and_gate/` | candidate dataset inventory, access audit, and minimal local training gate; generated locally and ignored by git |

## Supporting And Historical Docs

| Path | Status |
|---|---|
| `drug_release_closeout_2026-05-28.md` | historical mechanism-middle-layer closeout; useful context but no longer the top-level PLGA claim by itself |
| `top_journal_plan_2026-05-30.md` | older roadmap; use only after checking current PLGA route |
| `formal_few_shot_compositional_kinetic_world_model_2026-06-07.md` | world-model manifesto / future-method plan, not current proof |
| `release_data_collection_closeout_2026-06-03.md` | corpus intake status |
| `kgso_kinetic_grammar_state_observer_plan_2026-06-04.md` | concept plan, not current headline evidence |
| `PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md` | chitosan prospective reveal protocol |

## Archived Sprint Notes

The 2026-06-11 exploratory notes were consolidated into the current claim table.
Detailed one-off notes live here:

`archive/2026-06-11_information_budget_sprint/`

Use these only for audit trails or when reconstructing how a result was
derived:

- `cleanup_trace_2026-06-11.md`
- `black_box_baseline_competition_memo_2026-06-10.md`
- `original_lai_author_model_audit_2026-06-11.md`
- `plga_direct_early_blackbox_control_2026-06-11.md`
- `plga_static_to_early_proxy_probe_2026-06-11.md`
- `release_transfer_baseline_comparison_2026-06-11.md`
- and related 2026-06-11 probe notes.

Older historical logs and positioning material remain under `archive/`.

## Git Hygiene Reality

The working tree may contain many untracked numbered scripts and generated
outputs from research probes. Do not stage `outputs/`, `data/`, model pickles,
or cache directories. If committing, stage specific scripts/docs rather than
`git add -A`.

Release-corpus / LNN transfer scripts `80`-`90` are archived under
`../scripts/archive/2026-06-11_release_transfer_lnn_chain/`. They are historical
transfer diagnostics, not part of the active PLGA `91`-`103`
information-budget chain.
