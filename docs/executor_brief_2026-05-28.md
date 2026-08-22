# Executor Brief — Sprint 0 (2026-05-28)

You are executing concrete tasks for the release-foundation paper project.
Read this file end-to-end before acting. Execute tasks in order. Report
after each task with the format specified at the bottom.

## Context (read once)

Project: mechanism-constrained Bayesian inference for sparse drug release
prediction. Targeting Nature Methods / NMI / Nat Communications.

Authoritative references — read these before touching anything:
- `docs/paper_requirements_locked_2026-05-28.md` — single source of truth
- `AGENTS.md` — hard project constraints
- `CLAUDE.md` — coding discipline (Karpathy 4 principles)
- `DECISIONS.md` — ADR log (next free number: ADR-027)

Just-completed state:
- Active Observer v3 canonical RMSE 0.1446 vs Direct-Q 0.1758 on N=38
  (Δ = 27%, paired bootstrap 95% CI [8%, 47%])
- Verification 1: active-set silhouette = 0.284, p < 10⁻⁴ — below
  pre-set K1 threshold of 0.30 by 0.016
- Verification 2: regime × benefit Kruskal-Wallis p = 0.184 (N=38) —
  above pre-set K1 threshold of 0.05
- K1 (paper_requirements_locked section 8) is triggered. Decision:
  honor K1, demote regime contribution to supplementary, plan LOOCV
  expansion as reversal path.

---

## Task 1 — Hash + tag chitosan batch 1

**Goal**: cryptographic timestamp on 12 chitosan predictions before wet-lab
data arrives. Deadline 2026-05-29 EOD.

**Files to read first**:
- `outputs/67_chitosan_prospective/predictions.csv`
- `outputs/67_chitosan_prospective/lock_metadata.json`
- `outputs/67_chitosan_prospective/theta_targets.npz`

**Execute**:
```bash
cd D:\release-foundation
sha256sum outputs/67_chitosan_prospective/predictions.csv \
          outputs/67_chitosan_prospective/lock_metadata.json \
          outputs/67_chitosan_prospective/theta_targets.npz \
          > outputs/67_chitosan_prospective/HASHES.txt
git add outputs/67_chitosan_prospective/HASHES.txt
git commit -m "Lock chitosan prospective batch 1 (12 curves)"
git tag -a prospective-chitosan-batch1-2026-05-28 \
        -F outputs/67_chitosan_prospective/HASHES.txt
git push && git push --tags
```

**Acceptance**: `git tag -l 'prospective-chitosan-*'` lists the tag;
`git ls-remote --tags origin` confirms remote push.

**Don't**: modify `predictions.csv`, `lock_metadata.json`, or
`theta_targets.npz`. Don't use `git add -A` (uncommitted user work may
exist — see AGENTS.md Hard YES list rule 2).

---

## Task 2 — Write ADR-027

**Goal**: record the K1 decision in the ADR log.

**File**: append to `DECISIONS.md` (do not modify existing content).

**Append exactly this content** (no edits):

```markdown


## ADR-027 — Honor K1 kill criterion: demote regime contribution to supplementary

**Date**: 2026-05-28
**Status**: Adopted (pre-decided per paper_requirements_locked_2026-05-28.md K1)
**Authority**: paper_requirements_locked_2026-05-28.md sections 0, 3, 8

### Decision

Verification 1 (active-set silhouette) returned **0.284 (p < 10⁻⁴)**, below
the pre-set 0.30 threshold by 0.016. Verification 2 (regime × Active
Observer benefit, Kruskal-Wallis) returned **p = 0.184**, above the pre-set
0.05 threshold. Both arms of K1 triggered. We honor K1 without adjustment.

- Identifiability regime contribution (E3) demoted to supplementary.
- Paper main restructured to 3 claims:
  1. Active Observer v3 + canonical benchmark (E1, E2, E9)
  2. Chitosan prospective validation across 3 mechanism families (E7, E8)
  3. RSSM negative result in sparse data regime (E5, E6)

### Reversal trigger

E4 re-evaluation on cross321 LOOCV (N ≈ 250) returning Kruskal-Wallis
p < 0.05. Pre-decided branching:

| LOOCV p | Action |
|---|---|
| p < 0.05 | Reversal: regime re-elevated to main paper as supporting (not headline) claim; write ADR-028 |
| 0.05 ≤ p < 0.15 | Underpowered; do not expand N further (would suggest p-hacking); regime stays supplementary |
| p ≥ 0.15 | E4 hypothesis rejected; supplementary note "no significant regime × benefit interaction detected" |

### Why no K1 adjustment

Post-data threshold adjustment is the canonical HARKing pattern K1 was
designed to prevent. The 0.016 gap to threshold is not large enough to
justify retroactive adjustment without external evidence (e.g., a new
independent measurement) overruling the pre-decision.

### What this does NOT do

This ADR does not delete the regime finding. It sets its location in the
paper (supplementary, not main) until the reversal trigger fires. The
project lead's drafted text describing regime structure remains valid
content; only its section placement changes.
```

**Acceptance**: `tail -60 DECISIONS.md` shows ADR-027 in full; no other
content changed in the file.

---

## Task 3 — Update evidence matrix

**Goal**: reflect E3/E4 measurement status in
`docs/paper_requirements_locked_2026-05-28.md`.

**Modify only these rows** in the evidence-per-claim matrix (section 3):

| # | Old Status text | New Status text |
|---|---|---|
| E3 | `**NOT MEASURED**` | `**MEASURED** (sil=0.284, p<10⁻⁴); demoted to supplementary per ADR-027` |
| E4 | `**NOT MEASURED**` | `**UNDERPOWERED** at N=38 (p=0.184); pending LOOCV expansion per ADR-027` |

**Acceptance**: `git diff docs/paper_requirements_locked_2026-05-28.md`
shows exactly 2 lines changed.

**Don't**: change any other row, add new rows, or modify any other section.

---

## Task 4 — cross321 LOOCV V2 expansion

**Goal**: re-test E4 (regime × Active Observer benefit) with larger N to
potentially trigger ADR-027 reversal.

**Deliverable**: `scripts/77_regime_benefit_loocv.py`

**Inputs**:
- `data/Dataset_17_feat_augmented.csv` or whatever cross321 source the
  canonical benchmark uses — match `scripts/72_canonical_benchmark.py`
- Regime labels from script 33 / 36 outputs in `outputs/33_*` or
  `outputs/36_*` (find the canonical regime label file; document choice
  in a header comment)
- Direct-Q baseline: ExtraTreesRegressor, same hyperparameters as
  canonical benchmark
- Active Observer v3: import from `scripts/69_active_observer_v3.py` or
  equivalent module

**Algorithm**:
1. For each curve in cross321 (target N ≈ 250):
   - Leave one out; train Direct-Q and Active Observer on the rest
   - Predict full release curve for held-out curve
   - Compute RMSE on pre-specified evaluation timepoints (match
     `scripts/72_canonical_benchmark.py` exactly)
   - benefit_i = RMSE_DirectQ_i − RMSE_Active_i
2. Group benefit by regime label (drop regimes with N < 3)
3. Compute:
   - per-regime mean ± bootstrap 95% CI of benefit
   - per-regime N
   - Kruskal-Wallis H statistic + p-value
   - effect size: eta-squared η² = (H − k + 1) / (N − k)
4. Apply pre-decided decision rule (verbatim from ADR-027):
   - p < 0.05: write "ADR-027 reversal triggered" to summary
   - 0.05 ≤ p < 0.15: write "still underpowered, do not expand further"
   - p ≥ 0.15: write "E4 hypothesis rejected"

**Outputs** to `outputs/77_regime_benefit_loocv/`:
- `summary.json`: {kruskal_H, kruskal_p, eta_squared, decision_rule_outcome, n_curves, n_regimes_tested, generated_at_utc, git_hash}
- `per_regime_table.csv`: regime, n, mean_benefit, ci_low, ci_high
- `benefit_by_regime.png`: box plot of benefit per regime
- `lock_metadata.json`: same fields as
  `outputs/67_chitosan_prospective/lock_metadata.json` for reproducibility

**Pre-decisions (LOCK before seeing results)**:
- LOOCV is leave-one-curve-out, not leave-one-timepoint-out
- "RMSE" is computed on the same future-timepoint grid as
  `scripts/72_canonical_benchmark.py`
- Bootstrap CI uses 2000 resamples
- Random seed = 0

**Acceptance**:
- Script runs to completion on full cross321
- All 4 output files present
- `summary.json` contains a decision string from the pre-decided set
- Runtime documented in script header

**Don't**:
- Change LOOCV definition or RMSE timegrid after seeing intermediate results
- Try alternative clustering / regime relabeling
- Re-run with different seed if first result is "borderline"

---

## Task 5 — Audit U1 (Direct-Q RMSE discrepancy)

**Goal**: explain why Direct-Q RMSE appeared as 0.161, 0.120, and 0.176 in
different reports. Lock the canonical number.

**Method**:
1. `grep -rn "Direct" scripts/ outputs/ | grep -i rmse` to find all
   reports
2. For each, identify: which evaluation script, which split, which
   feature set, which seed, which preprocessing
3. Reproduce each number by re-running the source script (or note if
   the script is gone / archived)

**Deliverable**: `docs/audit_direct_q_discrepancy_2026-05-28.md` with:
- Table: number | source script | split | features | seed | reproducible
- Identification of canonical number (currently 0.176 per
  `scripts/72_canonical_benchmark.py`)
- Explanation of each non-canonical number (different split? earlier
  version? bug?)

**Plus**: append ADR-028 to DECISIONS.md locking the canonical Direct-Q
evaluation as `scripts/72_canonical_benchmark.py` (or whichever you
identify).

**Acceptance**: every Direct-Q number ever reported in
chat/doc/output is accounted for in the audit table.

---

## Task 6 — Audit U2 (script 33 clustering input)

**Goal**: confirm the basis for the regime = identifiability cluster claim.

**Method**:
1. Read `scripts/33*.py` (any file matching `scripts/33_*`, including in
   `scripts/archive/`)
2. Identify the feature space passed to the clustering algorithm
3. Trace data flow from raw curves → clustering input
4. Confirm or refute: clustering input is active-set indicator vectors
   (binary, length = n_params)

**Deliverable**: `docs/audit_script_33_clustering_2026-05-28.md` with:
- File/line citations
- Description of the clustering input feature space
- Verdict: does the active-set interpretation hold?
- If not: what's the actual clustering basis? Implications for ADR-027.

**Acceptance**: document answers "what features did script 33 cluster on"
with verifiable citations.

**Escalate**: if the answer is "not active-set", flag immediately — this
invalidates part of the ADR-027 reasoning.

---

## Task 7 — Draft prospective registration document

**Goal**: formal pre-registration for chitosan batch 1 per
`paper_requirements_locked_2026-05-28.md` section 1 P4.

**File**: `docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md`

**Required structure**:

```markdown
# Pre-Registration — Chitosan Prospective Batch 1

**Registered**: 2026-05-28
**Git tag**: prospective-chitosan-batch1-2026-05-28
**SHA256 lock**: see outputs/67_chitosan_prospective/HASHES.txt
**Wet-lab data deadline**: 2026-06-04
**Project lead**: [name]
**Authority**: paper_requirements_locked_2026-05-28.md P4

## Hypothesis

One sentence: the mechanism-constrained Bayesian inference framework
(Active Observer v3 + ChitosanRitgerPeppas simulator) produces calibrated
predictive intervals for chitosan hydrogel release on 12 prospective
formulations (6 formulations × 2 drugs) without retraining on chitosan
data.

## Predictions

12 curves, 10 timepoints each (0.5h to 28d). 90% PI lower / upper bound
for Q(t) per curve from `outputs/67_chitosan_prospective/predictions.csv`.

## Primary endpoint

**cov90**: fraction of (curve, timepoint) observations falling within
the predicted 90% PI band, computed across all 12 curves × all measured
timepoints.

Pre-specified success threshold: **cov90 ≥ 0.83**.

## Secondary endpoints

1. Per-curve hit rate: count of curves with ≥ 80% timepoints in band
2. Mean PI width (sharpness)
3. Pooled R² across all (curve, timepoint) pairs
4. CRPS

## Pre-specified fallback claims

| cov90 outcome | Reported claim |
|---|---|
| ≥ 0.83 | "Pre-registered cov90 met; supports cross-mechanism calibration claim" |
| 0.65–0.83 | "Pre-registered cov90 partially met; moderate under-coverage on chitosan transfer" |
| 0.50–0.65 | "Pre-registered cov90 not met; method exhibits substantial under-coverage on cross-mechanism transfer" |
| < 0.50 | "Substantial calibration failure on chitosan transfer; reported as method limitation pending batch 2" |

## Decided in advance (locked)

- Predictions, prior bounds, evaluation time grid, all coefficients
- Primary endpoint and threshold
- Fallback claims for every cov90 outcome

## Left for post-hoc (must be disclosed if performed)

- Regression analyses on per-curve hit rate vs formulation features
- Regime-specific breakdown if regime labels become available
- Exclusion of any curve must be reported with reason

## Reproducibility

Predictions generated at git commit
`ceff62647ba9e43324e4889145f99c08dffd71b3` per
`outputs/67_chitosan_prospective/lock_metadata.json`. Hashes locked at
git commit [insert from Task 1] per `prospective-chitosan-batch1-2026-05-28`
tag.
```

**Acceptance**: file follows the structure above, references the git
tag from Task 1, fills in actual commit hash.

---

## Reporting format (after each task)

After each task, report exactly this format:

```
TASK [N]: [name]
STATUS: [pass | fail | escalate]
FILES TOUCHED:
  - [path] : [created | modified | unchanged]
ACCEPTANCE: [pass | fail with reason]
DECISION OUTCOME (if applicable): [verbatim from pre-decided set]
NEXT: [task N+1 starting | escalating to user | done]
```

Then stop and start the next task. Do not concatenate task results.

---

## Don't do (hard rules)

1. Don't modify `outputs/67_chitosan_prospective/*` files (predictions
   are locked).
2. Don't change canonical hyperparameters without an ADR.
3. Don't add banned vocabulary (paper_requirements_locked R4):
   "world model", "foundation model", "active sensing",
   "active learning", "by design", "few-shot", "cross-mechanism",
   "emergent", "paradigm" — without precondition.
4. Don't write new claims into the evidence matrix without measurement.
5. Don't run V2 variants until Task 4 completes.
6. Don't use `git add -A`; add specific paths only.
7. Don't skip git hooks (`--no-verify`).
8. Don't read files via `cat` or `Get-Content`; use the Read tool.
9. Don't run `grep` via shell; use the Grep tool.
10. Don't write summaries / explanations into code comments (per CLAUDE.md).

## Escalate to user when

- Acceptance criteria fail and root cause is unclear
- A pre-decided rule needs interpretation (do not interpret silently)
- File modification conflicts with uncommitted user changes (`git status`
  shows `M` on target file)
- A new file outside the scope listed above is required
- Decision-rule outcome would change the paper structure beyond what
  ADR-027 anticipates

---

**End of brief**.
