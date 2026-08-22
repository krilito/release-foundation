# Executor Brief v2 — Bug Fixes + Rerun (2026-05-28 evening)

You are executing bug fixes and re-evaluations. **All decisions in
docs/executor_brief_2026-05-28.md (v1) are now provisional pending the
outcomes here.** Read this file end-to-end before acting.

## Why this brief exists

Code review (transcript in chat) identified 4 CRITICAL bugs in the
scripts that produced this session's headline numbers. LOOCV Task 4
result (Kruskal p=0.031, but Active Observer NEGATIVE benefit in 3/4
regimes) is consistent with the bug review's prediction that the
canonical +27% was a methodological-asymmetry artifact, not a real
improvement. Both signals converge on: **trust no number until bugs
are fixed and code rerun.**

## Bug inventory (with locations)

- **BUG 1**: `scripts/72_canonical_benchmark.py:264-308, 372` — GRU
  feature extractor trained on full data including test set; h-features
  leak test information into Direct-Q + h baseline.
- **BUG 2**: `scripts/72_canonical_benchmark.py:152-166 vs 172-257` —
  Direct-Q uses fixed early observation times [1, 3, 5, 7]; Active
  Observer adaptively chooses 2 best times. Not the same task. User
  decision: Option C (4-way factorial comparison).
- **BUG 3**: `scripts/74_regime_verification.py:109` cross-referenced
  with `scripts/33_active_set_regimes.py:120, 256-260` — regimes were
  clustered ON active-set Jaccard distance; measuring silhouette IN
  active-set space is tautological. User decision: use theta-space as
  independent feature space (with explicit null-result framing).
- **BUG 4**: `scripts/71_release_world_model.py:355-366` — `predict()`
  draws posterior sample at line 355, then OVERWRITES with prior sample
  at lines 365-366. "Posterior collapse" diagnostic measures prior
  variance, not posterior variance.

User-confirmed design decisions (locked):

- **BUG 2 fix**: 4-way factorial — {Direct-Q-fixed, Direct-Q-adaptive,
  Active-fixed, Active-adaptive}. Adaptive variants use the SAME
  observation-time selection algorithm; only the final predictor
  differs. This decomposes "selection ability" from "inference ability".
- **BUG 3 fix**: independent feature space is theta values (oracle
  params). Pre-known sil_theta = -0.057. This is explicitly framed as a
  null result demonstrating that regime is orthogonal to parameter
  values, NOT as a tautology check.

---

## Task A — Write ADR-028 (pause provisional ADR-027)

**Goal**: record that ADR-027 reversal trigger fired BUT execution is
paused pending bug fixes.

**File**: append to `DECISIONS.md`.

**Append exactly**:

```markdown


## ADR-028 — Pause ADR-027 reversal pending bug fixes in source scripts

**Date**: 2026-05-28 (evening)
**Status**: Adopted
**Supersedes**: ADR-027 reversal-trigger execution (NOT ADR-027 itself)

### Context

ADR-027's reversal trigger (LOOCV Kruskal-Wallis p < 0.05) fired with
p = 0.0309. Per ADR-027, this would re-elevate regime contribution to
main paper as supporting claim. However, the LOOCV result also showed
Active Observer NEGATIVE benefit in 3 of 4 regimes (Regime 4: -0.046,
Regime 5: -0.019, Regime 6: -0.055; only Regime 3: +0.027 positive).

Concurrent code review of source scripts (transcript 2026-05-28
afternoon) identified 4 CRITICAL bugs:

- BUG 1: GRU feature extraction trains on test set (leakage)
- BUG 2: Active vs Direct-Q methodologically asymmetric tasks
- BUG 3: Regime silhouette in active-set space is tautological
- BUG 4: RSSM "posterior collapse" diagnostic samples from prior

The negative LOOCV direction is consistent with BUG 2's predicted
artifact: canonical +27% inflated by Active's selection advantage,
which disappears under leave-one-out evaluation with regime-balanced
sample sizes.

### Decision

- ADR-027 reversal execution is PAUSED.
- Regime contribution location (supplementary vs main) is UNDECIDED.
- Active Observer's role (main method vs auxiliary) is UNDECIDED.
- Paper structure decisions are PAUSED.
- All 4 bugs must be fixed per executor_brief_2026-05-28_v2_bugfix.md.
- Canonical benchmark and LOOCV must be rerun under 4-way factorial
  design.
- Only after the corrected reruns may ADR-027 reversal be re-evaluated.

### What this does NOT change

- ADR-027's intent (honor pre-decided K1) stays valid.
- chitosan batch 1 hash and prospective registration (Tasks 1 and 7 of
  v1 brief) remain in force; they are independent of the source-script
  bugs.
- Audits U1 and U2 (Tasks 5 and 6) remain in force.

### Reversal trigger (for this ADR)

If corrected canonical benchmark shows Active Observer +X% with X > 5
AND corrected LOOCV shows non-negative benefit in majority of regimes,
this ADR is rescinded and ADR-027's original reversal proceeds.

If corrected results show Active Observer ≤ 0 benefit on average,
write ADR-029 restructuring paper entirely.
```

**Acceptance**: ADR-028 appended; no other DECISIONS.md content changed.

---

## Task B — Fix BUG 4 (RSSM predict sampling)

**Goal**: correct the sampling path in `predict()` so that posterior
predictive uncertainty is actually measured.

**File**: `scripts/71_release_world_model.py`

**Fix**: In the `predict()` method around lines 355-366, the current
code draws posterior sample and then overwrites with prior sample.
Decide the intended semantics:

- **For inference WITH early observations**: use posterior sample at
  observed timesteps, prior sample at future timesteps.
- **For inference WITHOUT early observations**: prior sample throughout.
- **For posterior predictive uncertainty diagnostic**: posterior sample
  at observed steps, propagated dynamics (NOT prior) at future steps.

The current code conflates these. Read the diagnostic call site
(`scripts/71_diagnostic.py:73`) to determine which mode it expects, and
fix `predict()` so the sample mode matches.

**Acceptance**:
- `git diff scripts/71_release_world_model.py` shows fix in `predict()`
- After fix, re-run `scripts/71_diagnostic.py` and report new numbers
  for KL/dim, cov90, cov50, prediction std

**Pre-decided decision rule** (lock before rerun):

| New cov90 (after fix) | Interpretation |
|---|---|
| 0.30-0.50 | Posterior collapse confirmed; original negative-result claim survives |
| 0.50-0.85 | Partial collapse; nuanced finding, paper writeup needs care |
| > 0.85 | NO posterior collapse; original "RSSM negative result" was a measurement bug, must retract |

---

## Task C — Fix BUG 3 (regime verification with theta-space)

**Goal**: replace tautological active-set silhouette with theta-space
silhouette as the independent test of regime structure.

**File**: rewrite `scripts/74_regime_verification.py` (call it v2 or
overwrite — your choice, but git-commit the rewrite).

**Algorithm**:

1. Load regime labels from `outputs/33_active_set_regimes/regime_assignments.csv`
   (these are clustered ON active-set Jaccard distance; this is now
   acknowledged as a definitional choice, not a discovery).

2. **Drop** the active-set silhouette test entirely (it is tautological).

3. Compute silhouette on theta-space:
   - Load oracle theta per curve from `outputs/27d_residual_npe/` or
     wherever fitted theta lives
   - Standardize theta features
   - Compute silhouette + permutation p-value with n_perm ≥ 5000

4. Compute silhouette on x-space (formulation features):
   - Same standardization as canonical_benchmark
   - n_perm ≥ 5000

5. Compute silhouette on curve-shape features (handcrafted: t50,
   plateau, AUC, max slope, early/late ratio) for completeness.

6. Report all 3 silhouettes side by side. **Frame in script header**:

   > "We test whether the active-set regimes (clustered on
   > identifiability patterns) are also separable in independent
   > feature spaces. The active-set silhouette is omitted as
   > tautological. A POSITIVE silhouette in theta-space or x-space
   > would mean regimes are also clusterable by parameter values or
   > formulations. A NEGATIVE silhouette is the expected null,
   > supporting the interpretation that regime structure is
   > orthogonal to parameter values."

7. Output `outputs/74_regime_verification_v2/{summary.json, lock_metadata.json}`

**Pre-decided decision rule** (lock before rerun):

| Theta-space sil | x-space sil | Interpretation |
|---|---|---|
| > 0.2 | any | Regime IS theta-cluster; reframe entire regime claim |
| -0.10 to 0.20 | any | Theta orthogonal to regime; ADR-027 supplementary framing valid |
| < -0.10 | any | Regime explicitly anti-aligned with theta; novel finding worth highlighting |

**Acceptance**: script runs, outputs all 3 silhouettes with n_perm ≥ 5000,
summary.json contains a decision string from the pre-decided set.

---

## Task D — Fix BUG 1 + BUG 2 in canonical benchmark (4-way factorial)

**Goal**: rewrite canonical benchmark with cross-fitted GRU features
and 4-way factorial comparison of (predictor) × (observation strategy).

**File**: rewrite `scripts/72_canonical_benchmark.py` as
`scripts/72_canonical_benchmark_v2.py` (do NOT delete v1; keep for
audit trail).

**Architecture**:

```python
# 4-way factorial design
methods = {
    "DirectQ-fixed":    direct_q + fixed_observations([1, 3, 5, 7]),
    "DirectQ-adaptive": direct_q + adaptive_observations(n=2),
    "Active-fixed":     active_obs + fixed_observations([1, 3, 5, 7]),
    "Active-adaptive":  active_obs + adaptive_observations(n=2),
}
```

**Crucial constraint for fairness**: the adaptive observation selection
algorithm MUST be the SAME for DirectQ-adaptive and Active-adaptive.
The two methods see the same 2 observation times per curve. Only the
final prediction (ExtraTrees vs particle filter) differs.

**Implementation notes**:

1. **Cross-fitted GRU features** (fix BUG 1):
   - K-fold split (K=5)
   - For each fold k: train GRU on (folds ≠ k), extract h(t=14d) for
     curves in fold k
   - Concatenate across folds → h_all without leakage
   - Then evaluate Direct-Q + h on held-out test split

2. **StandardScaler fit ONLY on training data** (fix related leak):
   - Split first, then fit preprocessor on X_train only, transform
     X_test

3. **Adaptive observation algorithm**:
   - Use the same variance-reduction heuristic from
     `scripts/69_active_observer_v3.py` (or import its function)
   - Apply identically to both DirectQ-adaptive and Active-adaptive
   - Document which timepoints were selected per curve in output

4. **Bootstrap CI in same script** (fix BUG: CI was in another script):
   - Paired bootstrap, 2000 resamples
   - Resample at curve level (not (curve, time) pair level)
   - Pre-decided seed = 0

5. **All 4 methods evaluated on identical**:
   - test curves
   - future timepoint grid
   - RMSE pooled across (curve, time) pairs
   - same observation set (for adaptive variants)

6. **Reproducibility**:
   - `torch.manual_seed(0)`, `np.random.seed(0)`, sklearn random_state=0
   - Write `lock_metadata.json` with git hash, timestamps, args,
     test_idx, train_idx

**Output**: `outputs/72_canonical_benchmark_v2/`
- `summary.csv`: method | n_test | rmse | rmse_ci_low | rmse_ci_high | cov90 | mean_pi_width
- `pairwise_comparisons.csv`: method_a | method_b | Δrmse | CI_low | CI_high | significant
- `per_curve_results.csv`: curve_id | regime | obs_times_used | method | rmse
- `lock_metadata.json`

**Pre-decided decision rule** (lock before rerun):

Compare 4 methods on canonical test split. The 6 pairwise comparisons
identify which axis (predictor / observation strategy) drives any gain:

| Comparison | Tests |
|---|---|
| DirectQ-fixed vs DirectQ-adaptive | Selection ability with simple predictor |
| Active-fixed vs Active-adaptive | Selection ability with particle filter |
| DirectQ-fixed vs Active-fixed | Inference ability with forced early obs |
| DirectQ-adaptive vs Active-adaptive | Inference ability with optimal obs |
| DirectQ-fixed vs Active-adaptive | Original "+27%" claim (combined effect) |
| DirectQ-adaptive vs Active-fixed | Disentanglement check |

Headline:

| Best method | Margin (95% CI) | Paper headline |
|---|---|---|
| Active-adaptive, beats DirectQ-fixed by >5% | CI > 0 | "Combined active + Bayesian inference improves RMSE by X%" |
| Active-adaptive, beats DirectQ-fixed by 2-5% | CI > 0 | Honest but moderate claim |
| Active-adaptive ≤ DirectQ-fixed | CI crosses 0 | Active Observer death; write ADR-029 |
| DirectQ-adaptive ≈ Active-adaptive | both > DirectQ-fixed | "Adaptive observation matters, predictor doesn't" — different paper |
| DirectQ-adaptive > Active-adaptive | — | Active Observer is genuinely worse; major finding |

**Acceptance**: script runs all 4 methods, all 6 pairwise comparisons,
summary.csv populated.

---

## Task E — Rerun LOOCV under 4-way design

**Goal**: replicate Task D's 4-way comparison under LOOCV (N ≈ 250) to
test whether the canonical and LOOCV results agree once bugs are fixed.

**File**: rewrite `scripts/77_regime_benefit_loocv.py` as
`scripts/77_regime_benefit_loocv_v2.py`.

**Algorithm**:
1. For each curve in cross321 (target N ≈ 250):
   - Leave one out
   - Train all 4 methods (cross-fitted GRU for h-features per Task D)
   - Predict held-out curve
   - Record RMSE per method
2. Per-curve benefit decomposition:
   - benefit_selection_dq = RMSE_DirectQ_fixed - RMSE_DirectQ_adaptive
   - benefit_selection_ao = RMSE_Active_fixed  - RMSE_Active_adaptive
   - benefit_inference_fixed = RMSE_DirectQ_fixed - RMSE_Active_fixed
   - benefit_inference_adaptive = RMSE_DirectQ_adaptive - RMSE_Active_adaptive
3. Per-regime Kruskal-Wallis on each benefit metric (4 tests; apply
   Bonferroni: α = 0.05/4 = 0.0125)
4. Output `outputs/77_regime_benefit_loocv_v2/`

**Pre-decided decision rules** (lock before rerun):

| Result | Action |
|---|---|
| Both canonical and LOOCV show Active-adaptive beats DirectQ-fixed by >5% with CI > 0 | Active Observer survives as main method; reverse ADR-027 reversal pause |
| Canonical shows Active wins, LOOCV shows Active loses | DISAGREEMENT — escalate to user, do not write paper |
| Both show Active loses or insignificant | Active Observer dies as main method; restructure paper around regime + chitosan |
| Selection effect strong (any predictor + adaptive wins), inference effect weak | Paper headline becomes "adaptive observation matters, ODE inference does not" — different paper |
| Regime-conditional pattern survives (Active wins in some regimes, loses in others) | Paper restructures around regime-aware method selector |

**Acceptance**: same as Task D but with LOOCV N report.

---

## Task F — Recompute evidence matrix entries

**Goal**: update `docs/paper_requirements_locked_2026-05-28.md` section
3 evidence matrix based on Task B/C/D/E outcomes.

**Rules**:
- ONLY update entries that were affected by bug fixes
- ALL prior numbers must be retracted if the rerun changes them
- Each updated row must cite the rerun output file

**Pre-decided updates**:
- E1 (Active Observer 27% claim): replace with Task D outcome
- E5 (RSSM posterior collapse): replace with Task B outcome
- E6 (RSSM deterministic dynamics): re-verify probe R² is not
  contaminated by Task B's bug fix
- E3 (regime identifiability cluster): explicitly mark as "definitional
  by clustering input; not measured as discovery"
- E4 (regime → benefit Kruskal): replace with Task E outcome
- E10 (regime invisible in x/h): update with Task C theta-space
  result (independent of active-set)

**Acceptance**: matrix reflects new measurements; no row carries a
contaminated number.

---

## Task G — Determine if ADR-027 reversal proceeds

**Goal**: based on Tasks B-F results, decide ADR-027 reversal status.

**Pre-decided rule**:

Read corrected canonical and LOOCV outputs. Apply this truth table:

| Active-adaptive vs DirectQ-fixed (canonical) | Active-adaptive vs DirectQ-fixed (LOOCV) | Action |
|---|---|---|
| Wins ≥ 5%, CI > 0 | Wins ≥ 5%, CI > 0 | Original ADR-027 reversal proceeds: regime to main as supporting claim |
| Wins | Wins but marginal | Reversal proceeds with caveats |
| Wins canonical, loses LOOCV | — | Investigate (escalate) |
| Loses or insignificant in both | — | ADR-029 needed: paper restructures around regime + chitosan; Active Observer demoted to "method we tried" |

**Output**: write `docs/decision_post_bugfix_2026-05-28.md` summarizing
the truth-table outcome and proposing the next ADR (if needed).

**DO NOT WRITE the new ADR yourself**. Surface the proposal to the
project lead via the decision document. The project lead approves
the next ADR.

---

## Reporting format

Same as v1 brief. After each task report:

```
TASK [letter]: [name]
STATUS: pass | fail | escalate
FILES TOUCHED: ...
ACCEPTANCE: pass | fail with reason
DECISION OUTCOME: verbatim from pre-decided rule (if applicable)
NEXT: ...
```

---

## Don't do (additional rules for this brief)

In addition to v1 brief's don't-do list:

1. **Don't modify v1 scripts in place** — create v2 alongside for audit
   trail (`72_canonical_benchmark.py` stays, `72_canonical_benchmark_v2.py`
   is new).
2. **Don't run any rerun until all 4 bugs are fixed** — partial fixes
   produce mixed-contamination data.
3. **Don't merge corrected output paths into v1 output dirs** — keep
   `outputs/72_canonical_benchmark/` and
   `outputs/72_canonical_benchmark_v2/` separate.
4. **Don't write a new ADR proposing paper restructure** — that
   decision is the project lead's (Task G ends in a proposal document,
   not an ADR).
5. **Don't claim a method "wins" if its CI crosses zero**.

## Escalate to user when

In addition to v1 escalation rules:

1. Task D or E shows DISAGREEMENT between canonical and LOOCV (per
   Task E decision rule)
2. Task B reveals cov90 in the "must retract" range (>0.85)
3. Task C reveals theta-space sil > 0.2 (regime IS theta cluster —
   would require entire framing rewrite)
4. Cross-fitted GRU (Task D point 1) crashes or shows numerical
   instability (RSSM trains unstably on per-fold data)

---

**End of brief v2**.
