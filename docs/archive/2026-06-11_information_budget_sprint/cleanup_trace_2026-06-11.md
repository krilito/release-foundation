# Documentation Cleanup Trace

Date: 2026-06-11

Scope: documentation cleanup and handoff tracking for the PLGA
information-budget sprint.

## What Changed

The documentation root was reduced to one current PLGA entry point plus older
historical roadmaps. Detailed 2026-06-11 exploratory notes were moved into this
archive folder so they remain auditable without competing with the current
claim.

Current entry point:

- `../../plga_information_budget_claim_table_2026-06-11.md`

Primary evidence output:

- `../../../outputs/102_plga_information_budget_claim_table/`

## Files Updated

| File | What it now says |
|---|---|
| `../../../README.md` | Frames the project as information-budgeted drug-release forecasting from sparse early observations. Points reproduction to scripts `97`, `99`, `101`, and `102`. States that the current PLGA route is not a foundation-model/world-model claim. |
| `../../../AGENTS.md` | Locks the AI-agent research framing: do not optimize the PLGA story around RMSE supremacy unless a frozen equal-budget leaderboard proves it. Prioritize observation value, uncertainty contraction, stopping rules, and mechanism-vs-shape-prior controls. |
| `../../INDEX.md` | Defines the reading order, current PLGA route, evidence anchors, next work, historical docs, and git hygiene rules. |
| `README.md` | Explains that this archive contains detailed notes from the 2026-06-11 sprint and points back to the current claim table. |
| `plga_early_observation_route_sync_2026-06-11.md` | Updated project-note paths after moving the notes into this archive folder. |

## Files Archived Here

These files are preserved as audit trail / reconstruction material, not as
current top-level claims:

- `black_box_baseline_competition_memo_2026-06-10.md`
- `original_lai_author_model_audit_2026-06-11.md`
- `plga_conditional_feature_relevance_protocol_2026-06-11.md`
- `plga_direct_early_blackbox_control_2026-06-11.md`
- `plga_early_middle_layer_selector_probe_2026-06-11.md`
- `plga_early_observation_route_sync_2026-06-11.md`
- `plga_middle_direct_complementarity_moe_2026-06-11.md`
- `plga_static_feature_ceiling_2026-06-11.md`
- `plga_static_to_early_proxy_probe_2026-06-11.md`
- `release_transfer_baseline_comparison_2026-06-11.md`

## Current Research Record

The cleaned documentation now records this interpretation:

```text
static formulation descriptors under-identify PLGA release;
measured early observations reveal missing release state;
the paper should be framed as observation-budgeted future-release forecasting.
```

The current strongest numerical anchors are:

- source-group strict split: static-after-k5 `0.267` -> measured-early-k5
  `0.160`, gain `0.107` RMSE.
- source-dataset strict split: static-after-k5 `0.272` -> measured-early-k5
  `0.166`, gain `0.106` RMSE.
- author-style fixed early baseline improves strict-split RMSE.
- static-predicted early points do not replace measured early observations.

## Not Cleaned In This Pass

The working tree still contains untracked experiment scripts and a pre-existing
modified `pyproject.toml`. This pass intentionally did not stage, delete, or
rewrite them because the requested scope was documentation cleanup and trace
recording.

If committing this cleanup, stage specific documentation files only. Do not use
`git add -A`.

## Dirty Worktree Audit Added After Cleanup

Follow-up audit scope: `pyproject.toml` plus untracked `scripts/80_*.py` through
`scripts/102_*.py`.

Findings:

- `pyproject.toml` pins `torch==2.7.0+cu128` and adds a PyTorch cu128 uv index.
  This conflicts with the README CUDA health note, which expects
  `torch 2.11.0+cu128`. Treat this as an environment/CUDA trial change, not as
  a clean project dependency decision, unless it is reconciled with
  `uv.lock`, `DECISIONS.md`, and the README CUDA note.
- `uv.lock` already contains `torch==2.7.0+cu128` entries, but `git status`
  does not show it as modified in this pass. The mismatch is therefore between
  current docs and project dependency state, not simply an unstaged lockfile.
- All untracked scripts `80`-`102` passed `python -m py_compile`.
- `scripts/100_lai_author_model_audit.py` has a machine-specific default path:
  `D:/BaiduNetdisk/long-acting-injectables-main`. It is usable as a local audit
  script because it also exposes `--author-root`, but it should not become a
  polished portable baseline without making that path explicit in docs or
  requiring a CLI argument.

Script grouping:

| Range | Role | Cleanup judgment |
|---|---|---|
| `80`-`90` | release-corpus / LNN transfer probes and reports | useful experimental audit chain, but not the current PLGA paper core |
| `91`-`99` | PLGA static-feature ceiling, early-observation, proxy, and MoE probes | current evidence chain supporting the information-budget interpretation |
| `100`-`102` | original-author baseline audit and final claim consolidation | current claim-table support; closest to commit-worthy after review |

Recommended next cleanup:

1. Keep `91`-`102` together if committing the current PLGA evidence line.
2. Move or archive `80`-`90` if the immediate paper route stays PLGA
   information-budget rather than release-corpus transfer.
3. Do not commit the `pyproject.toml` CUDA pin until the expected torch version
   and uv index policy are made consistent with README and project decisions.

Follow-up script-chain audit:

- `plga_91_102_script_chain_audit_2026-06-11.md`

Cleanup after audit:

- `91`-`102` now use portable `git_hash()` helpers.
- `91`-`102` lock metadata now records serialized CLI args.
- `100` now requires `--author-root` or `LAI_AUTHOR_ROOT`.
- `102` now separates numeric budget from observation protocol with
  `budget_kind` and `observation_protocol`.
