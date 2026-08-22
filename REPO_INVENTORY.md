# Repository Inventory

Audit date: 2026-06-10

Purpose:
- make the top level readable in 5 minutes
- separate current truth from historical notes
- flag git-hygiene issues without executing any git action

## Top Level

### Core code and config

| Path | Purpose |
|---|---|
| `simulator.py` | Single source of truth for the PLGA ODE, with `ReleaseSimulator` and `PLGABiphasic`. |
| `chitosan_simulator.py` | Chitosan-specific simulator used by the prospective bridge workflow. |
| `weibull_simulator.py` | Weibull decoder for the liposome accelerated-IVR route. |
| `encoder.py` | `FormulationEncoder` abstraction plus the current MLP encoder. |
| `posterior.py` | Neural posterior logic and `sbi` wrapper code. |
| `data.py` | Data loading and synthetic sampling helpers. |
| `release_posterior_family.py` | Posterior-family export objects added in the 2026-05-29 release wave. |
| `release_target_registry.py` | Mechanism-aware release target registry added in the 2026-05-29 release wave. |
| `shape_baseline_utils.py` | Shared shape-baseline helper utilities from the release-wave support line. |
| `configs/` | Config source of truth; currently centered on `plga_phase1.yaml`. |
| `pyproject.toml` | Dependency and tool configuration; use `uv`, not `pip`. |
| `uv.lock` | Local lockfile artifact, currently untracked/ignored. |

### Repo guardrails and architecture

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Repo-wide coding doctrine: think before coding, simplicity, surgical diffs, goal-driven verification. |
| `AGENTS.md` | Project-specific hard constraints, coding contract, and research guardrails. |
| `ARCHITECTURE.md` | Long-horizon technical vision and module responsibilities. |
| `DECISIONS.md` | ADR-style decision log and project-state ledger. |
| `README.md` | Main entry point for new readers; currently modified in working tree. |

### Local setup and testing

| Path | Purpose |
|---|---|
| `tests/` | Pytest coverage for core modules. |
| `CUDA_TORCH_RECOVERY.md` | Windows CUDA/PyTorch recovery note for the current local environment. |
| `.python-version` | Python version pin. |
| `.gitignore` | Ignore rules; currently modified in working tree. |
| `.venv/` | Local virtual environment; never commit. |
| `.pytest_cache/` | Pytest cache. |
| `.ruff_cache/` | Ruff cache. |
| `__pycache__/` | Python bytecode cache. |

### Data, outputs, and experiments

| Path | Purpose |
|---|---|
| `data/` | Real datasets plus the fixed canonical split file; real data should remain gitignored per repo rules. |
| `outputs/` | Main experiment output tree; not for commits. |
| `outputs_active_observer/` | Older Active Observer outputs. |
| `outputs_active_observer_v3/` | Current Active Observer v3 outputs, now supplementary/UQ only. |
| `outputs_active_observer_v3_group/` | Group evaluation outputs for the Active Observer branch. |
| `sbi-logs/` | TensorBoard event logs; currently 58 tracked binaries. |
| `scripts/` | Current high-confidence numbered execution surface; historical / low-confidence routes are under `scripts/archive/`. |

### Side workspaces and auxiliary modules

| Path | Purpose |
|---|---|
| `casp/` | Reusable CASP/posterior-family modules. |
| `optimize/` | Optimization/algorithm side-route experiments, not the paper entry point. |
| `battery/` | Battery trajectory extension workspace, separate from the release paper. |
| `drug_release/` | Boundary marker for future drug-release-only migration work. |
| `research/` | Research logs, currently centered on `active_observer`. |
| `vendored/` | Vendored external reference code/material. |
| `_external/` | External reference assets. |

### Handoff and transient notes

| Path | Purpose |
|---|---|
| `docs/archive/2026-06-10_root_handoff/` | Historical root handoffs and local agent notes moved out of the root. |
| `docs/archive/2026-06-10_cleanup_manifest.md` | Ledger for the 2026-06-10 cleanup pass. |
| `.claude/` | Agent protocol and handoff artifacts, not paper-facing documentation. |

## `docs/` Classification

Authority order:
1. durable output files in `outputs/`
2. `docs/drug_release_closeout_2026-05-28.md`
3. active plan / locked requirements docs
4. historical notes and archived material

### 1. Source-of-truth

| Path | Purpose |
|---|---|
| `docs/drug_release_closeout_2026-05-28.md` | Current release-only closeout and allowed/banned claims. |
| `docs/top_journal_plan_2026-05-30.md` | Current roadmap and anti-scope, subordinate to closeout. |
| `docs/paper_requirements_locked_2026-05-28.md` | Locked statistical/reproducibility requirements and kill criteria; path-stable but provisional. |
| `docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md` | Preregistered chitosan reveal protocol and primary endpoint. |
| `docs/research_nonnegotiables_2026-05-28.md` | High-level research guardrails. |

### 2. Active plan / current working support

| Path | Purpose |
|---|---|
| `docs/INDEX.md` | Reader-facing document index aligned to the closeout authority order. |
| `docs/decision_post_bugfix_2026-05-28.md` | Honest post-bugfix result summary. |
| `docs/executor_brief_2026-05-28.md` | Earlier execution brief. |
| `docs/executor_brief_2026-05-28_v2_bugfix.md` | Bugfix execution brief and key audit handoff. |
| `docs/audit_direct_q_discrepancy_2026-05-28.md` | Audit of Direct-Q number discrepancies. |
| `docs/audit_script_33_clustering_2026-05-28.md` | Audit of script 33 clustering inputs. |
| `docs/top_journal_gap_audit_2026-05-30.md` | Current ledger of what is still missing vs already evidenced for top-journal readiness. |
| `docs/b5_dependency_route_2026-05-30.md` | Current dependency-route note for the B5 HMC / NUTS posterior-ground-truth benchmark. |
| `docs/drug_release_claim_map_and_build_queue_2026-05-28.md` | Claim/build queue planning note. |
| `docs/drug_release_70plus_workstream_map_2026-05-30.md` | Workstream map across the large numbered-script surface. |
| `docs/tables/drug_release_70plus_workstream_map_2026-05-30.csv` | Tabular companion to the workstream map. |

### 3. Active but not current paper truth

| Path | Purpose |
|---|---|
| `docs/release_data_collection_closeout_2026-06-03.md` | Release-corpus intake status; relevant for corpus-building, not a paper claim source. |
| `docs/curve_world_probe_route_2026-06-02.md` | Diagnostic route note for curve-world probing. |
| `docs/kgso_kinetic_grammar_state_observer_plan_2026-06-04.md` | KGSO future-method plan; not yet evidenced as the current method. |
| `docs/formal_few_shot_compositional_kinetic_world_model_2026-06-07.md` | Formal world-model research direction; plan/manifesto, not current proof. |
| `docs/task_classification_thread_019e8dec_2026-06-04.md` | Task classification support note. |
| `battery/BATTERY_TRAJECTORY_KICKOFF_2026-06-04.md` | Battery trajectory extension kickoff; separate workspace from the release paper. |
| `battery/data/source_selection_2026-06-04.md` | Battery source selection note; not part of release-paper evidence. |

### 4. Historical handoff / research log / provisional writing assets

These are preserved, but no longer sit at the `docs/` root:

| Path | Purpose |
|---|---|
| `docs/archive/2026-06-10_historical_research_logs/` | Historical briefs, plans, research logs, old direction notes, and workspace-boundary memos. |
| `docs/manuscript_assets/` | Provisional manuscript prose, claim/evidence drafts, paper maps, and figure queue notes. |
| `docs/tables/` | CSV companion tables for workstream maps, paper maps, figure manifests, and claim/evidence drafts. |
| `docs/paper_closeout_release_quality_paradigm_2026-05-26.md` | Older closeout note superseded by the 2026-05-28 release closeout; still modified in the working tree, so left in place. |

### 5. Noise / positioning / stack / verdict wave

These exist under `docs/archive/2026-06-10_release_wave_low_confidence/`, but
should not drive README framing. The folder contains the 2026-05-29
algorithm-taxonomy, verdict, stack, timing, target-contract, panel-note, and
unified-release wave.

### `docs/archive/`

`docs/archive/` contains historical and low-confidence holding folders. The
2026-06-10 cleanup ledger is `docs/archive/2026-06-10_cleanup_manifest.md`.

## `docs/` CSV Files

CSV companions now live in `docs/tables/`:

- `docs/tables/drug_release_70plus_workstream_map_2026-05-30.csv`
- `docs/tables/drug_release_algorithm_taxonomy_2026-05-29.csv`
- `docs/tables/drug_release_claim_evidence_matrix_2026-05-29.csv`
- `docs/tables/drug_release_figure_assembly_checklist_2026-05-29.csv`
- `docs/tables/drug_release_figure_data_manifest_2026-05-29.csv`
- `docs/tables/drug_release_paper_spine_and_figure_map_2026-05-29.csv`

## `scripts/` Summary

### Canonical / paper-relevant entry points

- `38d_baselines_groupkfold.py`
- `46_direct_curve_rf_input_ablation.py`
- `59_middle_layer_gain_audit.py`
- `62_fib_casp_benchmark.py`
- `66_ensemble_casp.py`
- `67_chitosan_prospective_predictions.py`
- `69_active_observer_v3.py`
- `72_canonical_benchmark_v2.py`
- `73_diagnostics.py`
- `74b_regime_verification_v2.py`
- `75_chitosan_prospective_eval.py`
- `76_casp_conformal_recalibration.py`
- `77b_regime_benefit_loocv_v2.py`
- `make_canonical_split.py`

### Current top-level execution surface

The `scripts/` root has been reduced to high-confidence current entry points
and active WIP scripts. Historical/future side routes now live under
`scripts/archive/2026-06-10_*`.

### New support scripts added in the current cleanup / top-journal pass

| Path | Purpose |
|---|---|
| `scripts/27f_export_hmc_subset.py` | Locks the B5 claim-facing HMC subset and mouth. |
| `scripts/27g_internal_canonical_snre_eval.py` | First same-mouth internal SNRE comparator for B4. |
| `scripts/27h_b5_logposterior_backend.py` | Sampler-agnostic B5 log-posterior backend. |
| `scripts/27i_b5_hmc_pyro_runner.py` | Pyro/NUTS runner entrypoint scaffold for B5, pending dependency route. |
| `scripts/27j_b5_hmc_torch_runner.py` | Dependency-free fallback HMC sampler for B5. |
| `scripts/27k_b5_hmc_vs_amortized_compare.py` | Compares B5 fallback-HMC draws to amortized posterior draws. |
| `scripts/27l_b5_small_panel_audit.py` | Small-panel posterior-overlap audit for B5. |
| `scripts/27m_b5_chain_diagnostics.py` | Multi-chain split-R-hat diagnostic harness for B5 fallback HMC. |
| `scripts/69b_active_observer_v3_sbc.py` | Canonical-split SBC harness for the Active Observer posterior. |
| `scripts/69c_active_observer_v3_paper_candidate.py` | Stable rerun wrapper for the current best AO paper candidate. |
| `scripts/69d_s4_publication_table.py` | Canonical publication-facing sharpness + naive-band table for AO UQ claims. |
| `scripts/69e_active_observer_v3_cov50_eval.py` | AO candidate rerun that adds central 50% interval reporting for S4. |

### Archive scripts

| Path | Status |
|---|---|
| `scripts/archive/01_oracle_baseline.py` | Historical baseline evidence. |
| `scripts/archive/72_canonical_benchmark.py` | Superseded by `scripts/72_canonical_benchmark_v2.py`. |
| `scripts/archive/72b_canonical_benchmark_fixed.py` | Historical bugfix checkpoint. |
| `scripts/archive/74_regime_verification.py` | Superseded by `scripts/74b_regime_verification_v2.py`. |
| `scripts/archive/77_regime_benefit_loocv.py` | Superseded by `scripts/77b_regime_benefit_loocv_v2.py`. |
| `scripts/archive/2026-06-10_historical_sbi_active_set/` | 02-36 early SBI, posterior, deployment, active-set, and regime mechanics scripts. |
| `scripts/archive/2026-06-10_historical_prediction_routes/` | 37-58 non-entry prediction, theta, wet-eval, liposome, and curve-hybrid scripts. |
| `scripts/archive/2026-06-10_future_diagnostics/` | falsified/future CASP, PCAP, curve-world, active-observer, and green-claim diagnostics. |

### 80-119 wave

Scripts `80_*` through `119_*` are mostly snapshot/export/figure/stack assets from
the 2026-05-29 release wave. They are archived under
`scripts/archive/2026-06-10_release_wave_assets/` and should not define the repo
entry narrative.

### 120-161 release-corpus / KGSO / world-model wave

Scripts `120_*` through `161_*` are the newer release-corpus, external-source,
layered-pool, hard-gate, theta-freeze, and hierarchical baseline/adaptation
line. They are archived under
`scripts/archive/2026-06-10_release_corpus_world_model/` and should be treated
as a future-method workspace until their outputs are anchored in a closeout or
decision entry.

Current handling rule:

- do not use this wave to rewrite the README or paper claim unless the
  corresponding output manifest is checked first;
- keep it separate from the 2026-05-28 release closeout and the 2026-05-30
  top-journal gap audit;
- if promoted, add a short route document before expanding the entry narrative.

## Git Hygiene Issues

This section is recommendation-only. No git action has been executed.

### 1. Tracked binary logs in `sbi-logs/`

- Current issue: 58 tracked TensorBoard binaries
- Suggested action: `git rm --cached -r sbi-logs/`
- Why: they add churn, are not reviewable, and should remain local artifacts

### 2. Untracked authority documents

- Current issue: closeout, roadmap, preregistration, and several key support docs are still `??`
- Suggested action: add only the authority docs and a minimal support set first
- Why: README and index files should not point at ghost sources

### 3. Modified WIP files in active use

- Current issue: `README.md`, `scripts/README.md`, `AGENTS.md`, `DECISIONS.md`, `.gitignore`, and several key scripts are `M`
- Suggested action: use path-specific staging later; do not use `git add -A`
- Why: avoid sweeping unrelated user work into the same commit

### 4. Root-level transient notes

- Current state: `HANDOFF.md`, `NEXT_CHAT_HANDOFF_ZH.md`, `SPRINT1_*`, and
  `codex.md` have been moved to `docs/archive/2026-06-10_root_handoff/`.
- Why: they are useful as records, but they are not entry-point docs.

### 5. Mixed 2026-06 future-method wave

- Current issue: release-corpus / KGSO / world-model files and battery files
  are mixed into the same dirty worktree as release-paper cleanup.
- Suggested action: keep them on their existing split branches or stage them
  path-specifically by work package.
- Why: otherwise the current paper truth, future-method narrative, and
  cross-domain battery experiment will blur into one unverifiable commit.
