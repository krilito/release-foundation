# release-foundation

Information-budgeted drug-release state inference from sparse descriptors and
costly observations.

## What this is

A research codebase for sparse drug-release forecasting under mechanistic and
curve-language constraints. As of 2026-06-12, the active paper route is not
"we propose a stronger predictor." It is:

```text
drug release prediction is release-state inference under limited
pre-experimental descriptors and costly observations.
```

For PLGA, the locked information-budget line remains in
`docs/plga_information_budget_claim_table_2026-06-11.md`:

```text
static formulation descriptors under-identify PLGA release;
measured early release observations reveal missing release state;
the research question is how much observation budget is needed for reliable
future-release forecasting.
```

The archaeology and method inventory in
`docs/project_archaeology_inventory_2026-06-12.md` freezes retired routes and
prevents drift back to SRDS, PySR-main, RSSM/world-model, LNN/CNP transfer
fantasy, or architecture-novelty sprints. The older mechanism-middle-layer
framing in
`docs/drug_release_closeout_2026-05-28.md` remains useful historical context,
but it is no longer the top-level claim by itself. The current claim is not
"we beat all black boxes"; it is that early observations carry information that
available static descriptors cannot currently synthesize.

As of 2026-06-14, the screening-platform route is also locked behind explicit
contract and groundwork gates:

```text
base descriptors and optional state proxies must stay separated;
factual process-route tokens must stay separated from audit-only mechanism labels;
early-observation value must be scored on held-out future windows only;
route/neural modeling stays blocked until real R_i^model fields are standardized.
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the technical vision and phase plan.
See [CLAUDE.md](CLAUDE.md) + [AGENTS.md](AGENTS.md) before any AI assistant
touches this repo (general LLM-coding guardrails + project-specific hard rules).

## Read This First

If you are reopening this repo after some time away, do not start by scanning
all scripts.

Use this order instead:

1. [docs/script_route_decision_map_2026-06-14.md](docs/script_route_decision_map_2026-06-14.md):
   what is still alive, what is supplement-only, and what is retired
2. [docs/plga_information_budget_claim_table_2026-06-11.md](docs/plga_information_budget_claim_table_2026-06-11.md):
   the strongest current PLGA claim and its core evidence
3. [docs/project_archaeology_inventory_2026-06-12.md](docs/project_archaeology_inventory_2026-06-12.md):
   why older routes were demoted or frozen
4. [docs/release_screening_platform_formal_groundwork_2026-06-14.md](docs/release_screening_platform_formal_groundwork_2026-06-14.md):
   what 123 actually proved and why route/neural entry is still blocked

If you only have 15 minutes, stop after those four.

## Why this problem is hard

Drug release here is an amortized inverse problem with three practical
constraints:

- sparse release curves
- sparse formulation coverage
- non-identifiability: multiple kinetic parameter settings can produce nearly
  the same release trajectory

That is why the repo is organized around a mechanism-constrained middle layer
instead of a direct "fit one theta, then regress theta" cascade.

## Core map

For the current consolidation route, start from:

- `docs/script_route_decision_map_2026-06-14.md`: practical decision map for
  scripts `0-123`; separates main-claim routes, supplement-only routes, and
  retired routes
- `docs/project_archaeology_inventory_2026-06-12.md`: frozen archaeology,
  method inventory, retired routes, and consolidation-only guardrails
- `docs/plga_information_budget_claim_table_2026-06-11.md`: strongest current
  PLGA claim and evidence table
- `docs/minimal_drug_release_world_model_synthesis_2026-06-12.md`: current
  PLGA + liposome synthesis
- `docs/INDEX.md`: current reading order and output anchors

For the screening-platform contract route, start from:

- `docs/release_screening_platform_design_contract_2026-06-14.md`: design-side
  gate for `X_i^base`, `R_i^model`, `H_i^obs`, utility, and wet-lab logic
- `docs/release_screening_platform_mathematical_contract_2026-06-14.md`:
  mathematical boundary for information states, future-only scoring, and
  baseline legality
- `docs/release_screening_platform_formal_groundwork_2026-06-14.md`: current
  123A/123B/123C result; shows which fields are ready, which route facts were
  actually recovered, which screening targets and budgets are legal, which
  candidates were rejected as assay/audit metadata, and why route/neural
  modeling remains blocked
- `scripts/123_release_screening_platform_formal_groundwork.py`: reproducible
  generator for the 123 audit artifacts

For the PLGA information-budget route, the most relevant scripts are:

- `scripts/91_plga_static_feature_ceiling_probe.py`: static descriptor ceiling
- `scripts/96_plga_early_middle_layer_selector_probe.py`: middle-layer
  early-conditioned selector
- `scripts/97_plga_direct_early_blackbox_control.py`: matched direct early-Q
  black-box control
- `scripts/99_plga_static_to_early_proxy_probe.py`: test whether static
  descriptors can synthesize early observations
- `scripts/100_lai_author_model_audit.py`: original LAI author baseline audit
- `scripts/101_lai_author_style_on_our_splits.py`: author-style baseline on
  our strict splits
- `scripts/102_plga_information_budget_claim_table.py`: current claim table

Older PLGA SBI, chitosan prospective, CASP, and Active Observer scripts remain
in the repo as historical or supporting lines; they should not override the
2026-06-11 information-budget framing unless a new decision explicitly changes
the paper route.

## Quickstart

Environment management is via `uv` and `pyproject.toml`, but this Windows
workspace currently depends on a manually repaired CUDA PyTorch wheel. For that
reason, do not use `uv run python ...` for training or SBC jobs; it can
re-resolve torch from PyPI and replace CUDA torch with a CPU wheel.

Create or refresh the environment with `uv`, then call the venv interpreter
directly for actual runs:

```powershell
uv sync
.\.venv\Scripts\python.exe -m pytest
```

Representative local commands:

```powershell
.\.venv\Scripts\python.exe scripts\make_canonical_split.py
.\.venv\Scripts\python.exe scripts\72_canonical_benchmark_v2.py --seed 0
.\.venv\Scripts\python.exe scripts\46_direct_curve_rf_input_ablation.py
.\.venv\Scripts\python.exe scripts\59_middle_layer_gain_audit.py
```

Note: `scripts\make_canonical_split.py` defaults to `--seed 42` on purpose,
because ADR-028 pins the canonical split at that seed.

If dependencies are intentionally resynced, immediately re-check CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Expected healthy state is `torch 2.11.0+cu128`, `True`, CUDA `12.8`. See
[CUDA_TORCH_RECOVERY.md](CUDA_TORCH_RECOVERY.md) if it falls back to CPU.

## Layout

```
simulator.py             ReleaseSimulator ABC + PLGABiphasic concrete ODE
chitosan_simulator.py    Chitosan ReleaseSimulator subclass
weibull_simulator.py     Weibull decoder (liposome accelerated-IVR)
encoder.py               FormulationEncoder ABC + MLP concrete
posterior.py             Neural posterior estimator (sbi wrapper)
data.py                  Formulation loader + synthetic sampler
release_posterior_family.py  Posterior-family export objects
release_target_registry.py   Mechanism-aware release-target registry

scripts/              Numbered one-shot scripts (01_, 02_, ...); see scripts/README.md
configs/              Hydra configs; no magic numbers in code
tests/                pytest
data/                 Real datasets (gitignored)
outputs/              Run outputs (gitignored)
docs/                 closeout, roadmap, preregistration, and audit notes
```

Note: training, SBC/diagnostics, and grouped-holdout evaluation currently live
in numbered `scripts/`, not in standalone `train.py` / `diagnostics.py` /
`eval.py` modules. ARCHITECTURE.md lists those as the intended Phase-1 module
split; they have not been factored out of the scripts yet.

## Reproducing the current information-budget line

These are the commands a new reader should start with:

```powershell
.\.venv\Scripts\python.exe scripts\97_plga_direct_early_blackbox_control.py
.\.venv\Scripts\python.exe scripts\99_plga_static_to_early_proxy_probe.py
.\.venv\Scripts\python.exe scripts\101_lai_author_style_on_our_splits.py
.\.venv\Scripts\python.exe scripts\102_plga_information_budget_claim_table.py
```

Interpretation:

- `97` measures direct static-vs-early observation value.
- `99` tests whether static descriptors can synthesize early state.
- `101` ports the original LAI author zero/few-shot baseline onto our splits.
- `102` writes the current claim/evidence table.

## Status

`DECISIONS.md` is the source of truth for project state; this section is a
pointer, not a second copy.

- The current PLGA paper framing is "early observations reveal missing release
  state and define an information-budgeted forecasting problem" — see
  `docs/plga_information_budget_claim_table_2026-06-11.md`.
- The current cross-system manuscript framing is "minimal release-state
  prototype with partial support," not "new predictor wins" — see
  `docs/project_archaeology_inventory_2026-06-12.md` and
  `docs/minimal_drug_release_world_model_synthesis_2026-06-12.md`.
- Experiment 123 groundwork is now complete through 123C as a
  documentation-and-audit step: it formalizes `X_i^base`, `R_i^model`,
  `R_i^audit`, `H_i^available`, future-only evaluation, and screening
  target/budget legality, then runs a formulation-level preparation-route
  provenance audit plus task-feasibility gate. The current result is narrow
  but actionable: route recovery is real but still blocked for route/neural
  entry (`6 / 784` formulations, `2` route families, all single-source), while
  classical screening entry is only allowed inside a within-system PLGA scope —
  see
  `docs/release_screening_platform_formal_groundwork_2026-06-14.md` and
  `outputs/123_release_screening_platform_formal_groundwork/`.
- Phase 1 (PLGA amortized SBI) reached its success bar — see ADR-021.
- A 2026-05-28 code review found 4 critical bugs (test-set leakage, asymmetric
  Active-vs-DirectQ tasks, tautological regime silhouette, RSSM prior-sampling).
  After fixes, the "Active Observer +27%" headline was retracted; the surviving
  result is adaptive observation selection (4→2 timepoints) plus conformal
  calibration. See ADR-029 and `docs/decision_post_bugfix_2026-05-28.md`.
- The strongest missing piece for the current route is uncertainty contraction
  and stopping-rule analysis.

## Limitations and current scope

- This is not currently a "foundation model" or "world model" paper. Those
  routes are diagnostic/future-method branches unless the data scale and
  transfer evidence change.
- This is not an SRDS, PySR-main, RSSM, LNN/CNP, KAN, PINN, RL, or generic
  neural architecture sprint. Those branches are retired, demoted, or future
  diagnostics unless a locked release-state gap diagnostic reopens them.
- This is not an "Active Observer is the best predictive algorithm" repo. That
  headline was explicitly retracted.
- This is not just a reproduction of the original LAI few-shot LGBM baseline.
  The baseline is used to support the early-observation value claim.

## Document map

Start here:

- [docs/script_route_decision_map_2026-06-14.md](docs/script_route_decision_map_2026-06-14.md)
- [docs/plga_information_budget_claim_table_2026-06-11.md](docs/plga_information_budget_claim_table_2026-06-11.md)
- [docs/INDEX.md](docs/INDEX.md)
- [docs/project_archaeology_inventory_2026-06-12.md](docs/project_archaeology_inventory_2026-06-12.md)
- [docs/release_screening_platform_design_contract_2026-06-14.md](docs/release_screening_platform_design_contract_2026-06-14.md)
- [docs/release_screening_platform_mathematical_contract_2026-06-14.md](docs/release_screening_platform_mathematical_contract_2026-06-14.md)
- [docs/release_screening_platform_formal_groundwork_2026-06-14.md](docs/release_screening_platform_formal_groundwork_2026-06-14.md)
- [docs/minimal_drug_release_world_model_synthesis_2026-06-12.md](docs/minimal_drug_release_world_model_synthesis_2026-06-12.md)
- [docs/drug_release_closeout_2026-05-28.md](docs/drug_release_closeout_2026-05-28.md)
- [docs/paper_requirements_locked_2026-05-28.md](docs/paper_requirements_locked_2026-05-28.md)

Reproducibility rule still in force: CPU-trained outputs are
environment-contaminated and must not be used for scientific conclusions —
retrain on CUDA before SBC and deployment evaluations. See
[CUDA_TORCH_RECOVERY.md](CUDA_TORCH_RECOVERY.md).
