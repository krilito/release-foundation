# PLGA 91-102 Script Chain Audit

Date: 2026-06-11

Scope: untracked scripts `scripts/91_*.py` through `scripts/102_*.py`, the
outputs they consume/produce, and whether they are clean enough to support the
current PLGA information-budget claim.

## Audit Verdict

The `91`-`102` chain is not random scratch. It is a coherent evidence chain for
the current claim:

```text
static formulation descriptors under-identify PLGA release;
measured early observations reveal missing release state;
static-predicted early observations do not replace measured early observations.
```

No fatal leakage was found in the main headline contrasts. The chain can support
the current information-budget framing. A follow-up cleanup pass removed the
local author-repo default path, removed hard-coded repository paths in
`git_hash()`, standardized CLI argument capture in lock metadata, and separated
numeric observation budget from observation protocol in script `102`.

## Verification Performed

Commands/checks:

- `python scripts/<91-102>_*.py --help`
- `python -m py_compile scripts/<91-102>_*.py`
- existence check for expected outputs under `outputs/91_*` through
  `outputs/102_plga_information_budget_claim_table/`
- numeric sanity check on key summary CSV files
- reproduction check: reran `scripts/102_plga_information_budget_claim_table.py`
  into `outputs/_audit_102_repro/` and compared the generated
  `budget_value_table.csv` and `claim_evidence_table.csv` against the current
  `outputs/102_plga_information_budget_claim_table/` copies.

Results:

- all `91`-`102` scripts compile.
- all `91`-`102` scripts show CLI help successfully.
- all expected output folders/files exist.
- `102` reproduces the current core claim and budget tables exactly.
- key summary tables have no unexpected non-finite values. Non-finite values in
  `96`/`97` summaries are confined to optional statistics such as R2/MAE fields
  where the script already allows missingness.

## Evidence Chain

| Scripts | Role | Audit status |
|---|---|---|
| `91`-`92` | static descriptor ceiling and consolidation | main static baseline chain; train/test split is respected |
| `93`-`95` | feature-combo protocol, oracle diagnostic, deployable gate screen | diagnostic; useful for understanding static-feature limits, not headline |
| `96` | early-conditioned middle-layer selector | deployable selector plus explicit non-deployable future oracle |
| `97` | matched direct early-Q black-box control | key control; future targets are after context time |
| `98` | middle-vs-direct complementarity / MoE screen | diagnostic; not currently a deployable headline result |
| `99` | static-to-early proxy probe | key negative control: static descriptors fail to synthesize measured early state |
| `100` | original author model audit | useful external baseline context; has local-path portability issue |
| `101` | author-style LGBM on our splits | fairer author-style comparison under our split definitions |
| `102` | final information-budget claim table | reproducible consolidation of current claim |

## Leakage / Split Findings

Main static descriptor chain:

- `91` trains parameter mappers only on train curve shape-fit labels and scores
  heldout raw release observations.
- Numeric imputation/scaling in `91` uses train medians/means/stds.
- Source-group and source-dataset splits are explicit.

Direct early-Q control:

- `97` builds targets only at `time_days > last_context_time`, so context points
  are not evaluated as future targets.
- Train/test curve IDs are derived fold-by-fold from the same split assignments
  used by the static chain.

Author-style comparison:

- `101` fits scalers on train static descriptors only.
- Target points use `time_days > min_target_time` with default `1.0`, so the
  fixed early points are not scored as future targets.

Gate / MoE diagnostics:

- `95` trains the static combo gate on non-current-fold per-curve errors and
  evaluates on the current fold.
- `98` trains the middle-vs-direct gate on non-current-fold delta errors and
  evaluates on the current fold.
- Both are valid diagnostics, but they should not become headline claims without
  clearer preregistration and possibly nested validation.

Oracle rows:

- `96` explicitly labels `future_oracle_family` as non-deployable.
- `102` carries comparability notes and should keep oracle rows out of headline
  deployable rankings.

## Cleanup Applied

These items were identified during audit and addressed before handoff:

1. `scripts/100_lai_author_model_audit.py` now requires `--author-root` or
   `LAI_AUTHOR_ROOT`; it no longer defaults to a machine-specific
   `D:/BaiduNetdisk/...` path.
2. `git_hash()` helpers now run `git rev-parse HEAD` from the resolved
   repository root instead of using `safe.directory=D:/release-foundation`.
3. `91`-`102` lock metadata now records `args` through `args_to_metadata()`.
4. `102` now adds `budget_kind` and `observation_protocol` columns, so
   author fixed-time early observations are not silently conflated with
   earliest-k observations.

Post-cleanup checks:

- all `91`-`102` scripts still compile.
- `100 --help` and `102 --help` succeed.
- rerunning `102` after cleanup leaves `budget_value_table.csv` and
  `claim_evidence_table.csv` unchanged.
- `outputs/102_plga_information_budget_claim_table/` was regenerated after
  cleanup so `unified_model_table.csv` and `lock_metadata.json` reflect the new
  protocol columns and serialized CLI args.

## Non-Blocking Hygiene

- `91` and `97` build one-hot matrices after mapping test-only categories to
  `__UNK__`. This does not use heldout targets and should not affect predictive
  fit, but a stricter implementation would build categorical columns from train
  categories only.
- `95` and `98` are useful but should be described as exploratory diagnostics,
  not as deployable model-selection evidence.
- `100` reads released pickle result frames from the author repository. This is
  acceptable for audit, but not portable unless the external repo/version is
  documented.

## Current Recommendation

Keep `91`-`102` together as the current PLGA evidence chain. Do not mix this
commit with the `pyproject.toml` CUDA pin or the `80`-`90`
release-corpus/LNN transfer chain.
