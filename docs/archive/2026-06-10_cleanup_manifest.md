# 2026-06-10 Cleanup Manifest

This manifest records a knowledge-hygiene pass, not a scientific decision.

Rule used:

- keep high-confidence current entry points in the repo or `docs/` root;
- move historical, low-confidence, draft, or side-route material into explicit
  holding folders;
- preserve files instead of deleting when the material may still have audit
  value.

## Kept At The Root

Project-root entry points:

- `README.md`
- `CLAUDE.md`
- `AGENTS.md`
- `ARCHITECTURE.md`
- `DECISIONS.md`
- `CUDA_TORCH_RECOVERY.md`
- `REPO_INVENTORY.md`

Current `docs/` entry points:

- current closeout, roadmap, locked requirements, preregistration, audits, and
  gap ledgers;
- active 2026-06 route notes for release-corpus, KGSO, world-model, and task
  classification, but only as future-method / side-route material.

## Moved

| Destination | Meaning |
|---|---|
| `docs/archive/2026-06-10_root_handoff/` | root-level session handoffs and local agent notes |
| `docs/archive/2026-06-10_historical_research_logs/` | older plans, research logs, historical briefs, and superseded direction notes |
| `docs/archive/2026-06-10_release_wave_low_confidence/` | 2026-05-29 algorithm-taxonomy, verdict, stack, and panel-note wave |
| `docs/manuscript_assets/` | draft manuscript prose, claim/evidence drafts, and figure queue notes |
| `docs/tables/` | CSV companion tables from manuscript and planning documents |
| `scripts/archive/2026-06-10_historical_sbi_active_set/` | tracked `02_*` through `36_*` early SBI, posterior, deployment, active-set, and regime mechanics scripts |
| `scripts/archive/2026-06-10_historical_prediction_routes/` | tracked `37_*` through `58_*` non-entry prediction, theta, wet-eval, liposome, and curve-hybrid scripts |
| `scripts/archive/2026-06-10_future_diagnostics/` | tracked falsified or future CASP, PCAP, curve-world, active-observer, and green-claim diagnostic scripts |
| `scripts/archive/2026-06-10_release_wave_assets/` | untracked `80_*` through `119_*` release-wave export / figure / stack scripts |
| `scripts/archive/2026-06-10_release_corpus_world_model/` | untracked `120_*` through `161_*` release-corpus, hard-gate, theta-freeze, and hierarchical/world-model scripts |

## Not Moved

- Modified code or docs already in active use, including `README.md`,
  `AGENTS.md`, `DECISIONS.md`, `scripts/README.md`, and the modified evaluator
  scripts.
- `data/`, `outputs/`, and `outputs_active_observer*`.
- Current benchmark, posterior/UQ, chitosan, and diagnostic scripts in the
  `27e-27m`, `69b-69e`, and `72-77` ranges.
- Current top-level liposome/export endpoints `78_liposome_ivr_intake.py` and
  `79_liposome_prediction_export.py`.

## Remaining Cleanup Risk

- `sbi-logs/` contains tracked TensorBoard event logs and should be removed
  from git tracking with a path-specific command when ready.
- Several current authority docs are still untracked; stage only the intended
  documentation package, not the whole worktree.
