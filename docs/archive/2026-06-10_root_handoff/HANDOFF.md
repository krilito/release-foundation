# HANDOFF — Sprint 0 closed (post-adversarial-review) → Phase 2 Sprint 1 start

**Date:** 2026-05-22
**Status:** Phase 1 success bar **3/3 PASS** (ADR-021); diagnostic
paper absorbed via vendoring + reimplemented comparators (ADR-023);
Phase 2 success bar locked (ADR-024); **adversarial-review correction
of internal-181 numbers landed (ADR-025) — fPCA / Direct / SBI are
tied within ~0.03 R² on internal matched protocol; SBI uniquely keeps
mean R² > 0 on cross-DOI tail**. Phase 2 Sprint 1 (continuous burst
term, ODE form) is the next concrete work item, gated on Sprint 1
prep (new ADR-026 to author, prior-bounds audit, branch isolation).

This file is a session-snapshot. It will become stale; update or delete it
when Day 4 (Phase 1 close-out or Phase 2 prep) lands. For evergreen project
context read, in order: `CLAUDE.md` → `AGENTS.md` → `ARCHITECTURE.md` →
`DECISIONS.md`.

---

## Where we are

Phase 1 (PLGA-only amortized SBI) is the active phase. MVP-B two-stage
architecture (ADR-009) is **complete**:

- **Stage 1 — `CurvePosterior` (q_φ(θ|Q))** — trained, 8/8 SBC PASS under
  c2st_ranks criterion (ADR-017).
- **Stage 2 — `DescriptorPosterior` (r_ψ(θ|x))** — trained (script 09),
  evaluated leave-one-drug-out (script 10).

Latest validated artifacts:
- `outputs/04_curve_posterior/posterior.pt` — trained Stage 1 model
- `outputs/05_sbc_multiseed/summary.txt` — 8/8 PASS, c2st_ranks ≤ 0.60
  across 10 seeds (ADR-017)
- `outputs/09_descriptor_posterior/posterior.pt` — trained Stage 2 model
  (full 133 high-quality curves, K=64)
- `outputs/09_descriptor_posterior/quality_distribution.csv` — per-curve
  teacher quality flags (high=133, medium=11, low=37)
- `outputs/10_eval_deployment/summary.txt` — leave-one-drug-out R² results
- `outputs/10_eval_deployment/per_curve_metrics.csv` — 133 rows, all the
  metrics needed for slice / dice
- `outputs/10_eval_deployment/predictions/<fid>.npz` — per-curve raw
  trajectories for plotting / audit

Latest commit: `8e2b32b` (script 09). ADR-019 and script 10 not yet committed.

---

## Phase 1 success bar (ADR-006 + ADR-020) — final verdict

| Bar | Status | Evidence |
|---|---|---|
| (1) SBC calibration on 181 | **PASS** | c2st_ranks ≤ 0.60 across all 8 params, 10 seeds (ADR-017) |
| (2') Internal LODO median R² ≥ 0.50 | **PASS** | median R² = +0.657 (ADR-019, ADR-020) |
| (3) Cross-DOI 321 R² > 0.30 | **PASS** | median R² = +0.483 (ADR-021) |

Phase 1 is **3/3 PASS** and complete.

## Sprint 0 (post-Phase-1) — what landed

| Task | Result | Source |
|---|---|---|
| Absorb diagnostic paper | Vendored to `vendored/diagnostic_paper/` | ADR-023 |
| fPCA + Direct LGBM comparators (multi-seed) | `scripts/12_baseline_comparators.py` | ADR-023 / ADR-025 |
| SBI GroupKFold-5 by curve | median R² +0.934 on n=133 high-quality | script 10 `--fold-by curve`, `outputs/10_eval_deployment/by_curve/` |
| **Adversarial-review of ADR-023** | 7 holes found, 3 critical; fold-mismatch in original framing | spawned during Sprint 0 close-out |
| Matched-protocol comparator (SBI's 133 fids + folds) | fPCA +0.940 / SBI +0.934 / Direct +0.921 median R² — all 3 tied within 0.03 | `outputs/12_baseline_comparators/internal-181-matched/` |
| Cross-DOI 321 multi-seed runs | Direct +0.554 / SBI +0.483 / fPCA +0.431 median; SBI unique mean R² > 0 (+0.056) | `outputs/12_baseline_comparators/cross-doi-321/` |
| tmax-stratified + LA/GA + GEF audits | tmax<14d concentrates the negative R² tail; LA/GA extrapolation moot (only 2 curves); GEF was an LODO artifact | `outputs/13_phase1_audits/` |
| Phase 2 success bar locked | tail robustness + cross-DOI vs-Direct gap + UQ + cross-mechanism | ADR-024 |
| ADR-023 numbers corrected | SBI is competitive (not victorious) on internal; cross-DOI tail-robustness is THE differentiator | ADR-025 |

## Phase 2 success bar (ADR-024 + ADR-025 corrections)

| # | Condition | Threshold | Phase 1 baseline | Differentiates SBI? |
|---|---|---|---|---|
| (1) | Calibration | c2st_ranks ≤ 0.60 per mechanism | PASS on PLGA | yes — bypass cannot |
| (2) | Internal median R² | ≥ 0.85 on PLGA 181 GKF-5 | +0.934 | no — fPCA / Direct also clear |
| (3) | Cross-DOI median R² | ≥ Direct LGBM 5-seed median on matched subset | +0.483 vs **+0.554** (gap **0.071** to close) | no — Direct is the bar |
| (4) | **Cross-DOI mean R²** | **> 0** (= the kill feature) | **+0.056** vs Direct −0.030 vs fPCA −0.176 | **YES** — bypass goes negative |
| (5) | 90% credible coverage | ∈ [0.85, 0.95] | not measured (Sprint 1 deliverable) | yes — bypass has no UQ |
| (6) | Cross-mechanism transfer | joint PLGA+LNP SBC PASS | not started (Sprint 5) | yes — bypass needs per-mechanism refit |

---

## Locked-in design (do NOT change without writing a new ADR first)

| Concern | Decision | ADR |
|---|---|---|
| Architecture | Two-stage amortized hierarchical inference (Q→θ→x) | ADR-009 |
| ODE form | 3-state PLGA biphasic with autocatalysis + Q_max + cummax | ADR-002, 007 |
| Q(0)=q_burst delta limitation | Accepted for Phase 1; Phase 2 candidate for continuous burst term | ADR-019 |
| Prior bounds | 8 params; upper bounds per ADR-015 + 016 | ADR-015, 016 |
| t_grid | linspace(0, 90, 64) days | ADR-012 (after ADR-011 reversal) |
| Observation noise | σ_Q = 0.03 Gaussian, clipped to [0,1] | ADR-014 |
| Flow (q_φ) | MAF, hidden=64, transforms=5, FCEmbedding(64→16) | ADR-009, 010 |
| Synthetic count | 50 000 (θ, Q) pairs | ADR-009 |
| 321 dataset | cross-DOI test set only, NOT Phase 1 training | ADR-005 |
| Phase 1 success bar | SBC + leave-one-drug-out R² ≥ 0.50 + cross-DOI R² > 0.30 | ADR-006, ADR-020 |
| SBC verdict criterion | c2st_ranks max ≤ 0.60 across ≥10 seeds | ADR-017 |
| Stage-2 training target | KL distillation `min KL(q_φ||r_ψ)` with K=64 teacher samples | ADR-018 |
| Teacher quality flag | high if tmax≥60 or lastQ≥0.8; low if tmax<90 and lastQ<0.7 | ADR-018 |
| DP_Group encoding | parsed polymer_family one-hot + OTHER; drug_id excluded from features in v0 | ADR-018 |
| Phase 1 R² ceiling | MLP encoder limited; transformer in Phase 2 to close gap | ADR-019 |

---

## Open items (carry forward to Phase 2)

### Cleared by Sprint 0
- ~~Cross-DOI 321 evaluation~~ → ADR-021 PASS at +0.483.
- ~~GEF outlier fold~~ → was a LODO artifact; under per-curve CV GEF
  is median R² +0.513 (ADR-023 finding 3).
- ~~PLGA family hardest~~ → also a LODO artifact; under per-curve CV
  PLGA median R² +0.933, tied with everything else (ADR-023 finding 3).

### Carry to Phase 2 Sprint 1 (continuous burst term)
- **Q(0)=q_burst delta** still produces median Q0 residual = +0.107 on
  cross-DOI. ADR-019 deferred fix; Phase 2 Sprint 1 implements
  `dQ/dt += k_burst·exp(-t/τ_burst)·(Q_max - Q)` with `q_burst → 0`
  initial condition. Must re-do prior bounds audit + SBC after the
  ODE form change. New ADR-025 captures the burst-term decision.

### Carry to Phase 2 Sprint 2 (encoder)
- **Cross-DOI median R² gap to Direct LGBM = 0.065 on 321.** Encoder
  upgrade (SMILES drug embedding + token transformer for polymer +
  process) targets this. Must NOT lose mean R² lead (currently +0.056,
  the only method with positive mean R²).

### Non-blocking (low priority)
- **OTHER polymer bucket dead at training** — ADR-018 technical debt.
- **`best_val_log_prob` still logs as `None`** — ADR-016 technical
  debt.
- **Stage-2 weight_decay not exposed by sbi.NPE.train** — ADR-018.
- **Short-tmax cross-DOI negative R² tail** — 52 / 259 = 20% of
  cross-DOI curves have tmax < 14d and median R² −0.51. Most of the
  apparent cross-DOI "failure" is small-Var(Q_obs) pathology, not
  model failure. Phase 2 evaluation should report R² stratified by
  tmax bin (script 13 pattern).

### Diagnostic-paper preprint (user side)
- 2-hour formatting job to put `D:/诊断论文/` into chemrxiv-ready
  PDF + manifest. Gives ADR-023's framing a DOI anchor. Not blocking.

---

## Phase 2 plan summary

Five sprints, total ~5 months (full-time equivalent). Details in the
Phase 2 plan draft (conversation log), canonicalized into ADRs as
each sprint lands.

**Sprint 1 (PROPOSED — awaits user go-ahead)**: ODE continuous burst term.
**Full design in ADR-026 (Proposed status, not Adopted).** Sprint 1 does
NOT begin until ADR-026 moves to Adopted via explicit user approval.
Risk: ODE form change is ADR-002-locked; identifiability of new
`log_tau_burst` parameter must clear SBC. Estimated wall time
4–7h if SBC passes first try, 12–20h if 1–2 prior-tightening cycles
needed. **Branch isolation (`phase2-sprint1-burst`) recommended** when
Sprint 1 kicks off so master stays at current verified state.

**Sprint 2**: Encoder upgrade (SMILES drug embed + token transformer);
heterogeneous descriptor interface (must accept PLGA + chitosan +
hydrogel descriptor schemas simultaneously).

**Sprint 3 (parallel)**:
- 3a: chitosan nanoparticle simulator + ingest first 3-5 wet-lab curves
      from the user's 大创 project (DCNPs / HDCNPs / PHDCNPs layered NPs)
- 3b: hydrogel simulator (simplest Fickian ODE) + manual curation of
      public hydrogel release dataset (target 50-100 curves; user-side
      data curation work, advisor lab data may be partly usable but
      data cleaning is on us)

**Sprint 4**: Joint posterior + amortized NRE for **mechanism inference**
(PLGA vs chitosan vs hydrogel from observed curve + descriptors).
First NC-tier paper main result.

**Sprint 5 (closed-loop Nature Materials stretch)**: Inverse design head.
Model proposes 5-7 chitosan formulations from feasible parameter space;
user synthesizes; release measured; coverage of model's 90% credible
interval scored. Pass→Nature Materials framing; partial pass→NC; full
fail→JACS fallback.

**Sprint 6**: Phase 2 paper writing (2 months).

Total: 13-15 months, aligned with the user's 大创 timeline ending 2026.6.

## Wake-up briefing (2026-05-22 — user fell asleep mid-Sprint 0 close-out)

While the user was asleep:

1. **Adversarial review**: spawned a general-purpose agent to attack ADR-023's
   "SBI beats both bypass on internal" claim. The agent found 7 holes,
   3 critical. Most importantly: ADR-023 compared SBI (n=133 high-quality
   subset, SBI's own GroupKFold partition) against fPCA / Direct (n=181 full
   set, comparator's own partition). Only 40% of fids landed in the same
   numbered fold. The "matching protocol" claim was apples-to-oranges.

2. **Rigor remediation** (commit `ae2de79`):
   - `scripts/12_baseline_comparators.py` extended with `--restrict-fids-csv`
     (replicate SBI's exact fold partition), multi-seed sweep (5 seeds),
     finite-mask in `_per_group` matching diagnostic paper's `metric_row`,
     cross-DOI nested k re-selection (was hardcoded k=18; re-selects to k=20).
   - Three new outputs: `internal-181-standalone/`, `internal-181-matched/`,
     `cross-doi-321/` (refreshed with matched-set summary).
   - **Headline finding under matched protocol (n=133 SBI fids + SBI folds,
     5-seed median)**: fPCA +0.940 / SBI +0.934 / Direct +0.921 — all tied
     within 0.03 R². **SBI does NOT beat both bypass methods on internal as
     ADR-023 claimed.**
   - Cross-DOI finding still stands: SBI is the ONLY method with mean R²
     > 0 (+0.056), Direct −0.030, fPCA −0.176. **Tail robustness — not
     median R² — is the real Phase-2 differentiator.**
   - **ADR-025** captures the correction (supersedes ADR-023 §2 + §3
     numbers; cross-DOI thesis stands).

3. **Sprint 1 prep** (DECISIONS.md additions, no code changes):
   - **ADR-026 (PROPOSED status)** — detailed Sprint 1 design with
     prior-bound proposal, identifiability concerns, 10-step execution
     sequence, acceptance criteria, alternatives rejected, reversal trigger.
   - **Did NOT modify simulator.py / configs / tests / kick off any
     training runs.** ODE form change is invasive enough to need user
     approval before execution.

**What you need to do when you read this:**

- Skim ADR-025 (the rigor correction) to confirm the "SBI competitive on
  internal, unique on cross-DOI tail" framing is OK.
- Read ADR-026 (the Sprint 1 proposal) carefully. Particular attention to:
  the `q_burst / τ_burst` parameterization choice, the proposed
  `log_tau_burst` prior [-3, 0], and the acceptance criteria S1-a through
  S1-f.
- If both OK → tell me "ADR-026 approved, kick off Sprint 1 on branch"
  and I execute steps 1-10 of the Sprint 1 sequence.
- If something needs to change in ADR-026 → tell me what.
- If you want to defer Sprint 1 and work on Sprint 2 (encoder) or audits
  first → tell me.

## Session 5 close-out (2026-05-22 — Phase 2 framing locked, Sprint 1 awaits next session)

User decisions made during session 5 after the wake-up briefing above:

1. **Phase 2 target tier**: Nature Materials stretch / NC realistic / JACS
   fallback. No graduation deadline → optimize for breakthrough, not speed.
2. **Path B (closed-loop wet+dry validation) accepted in principle.**
   Lock PLGA Sprint 1 first, then extend to multi-mechanism.
3. **Third mechanism = hydrogel.** Reasons:
   - Mechanistically distinct from PLGA biphasic (pure Fickian diffusion
     through swollen network; simplest ODE form among 3 mechanisms).
   - User's advisor (杨超) runs a hydrogel project lineage:
     横向《水凝胶眼贴产品的开发》+ 2024 大创《聚多糖载药可注射水凝胶》.
   - Connects to chitosan story (chitosan can be embedded in hydrogel
     carriers, which is the user's 大创 framing).
   - Public data exists in literature but **manual curation on us**
     (user explicitly noted: 大概率要我们自己清洗数据).
4. **Chitosan wet-lab timeline**: user's 大创 will produce first batch
   "一周左右". Design must use Latin Hypercube sampling over the 6-D
   parameter space (CS_MW, deacetyl_degree, TPP_ratio, HA_coating,
   PVP_coating, drug_class), NOT the sequential DCNPs→HDCNPs→PHDCNPs
   ladder — the latter only covers 3 points in formulation space and
   would starve Sprint 3a simulator training.
5. **Worldmodel-v0 lessons absorbed, code not reused**:
   - F1 (curve_fit oracle inversion → degenerate params) is structurally
     fixed by Phase 1's amortized SBI posterior (no per-curve inversion).
   - F2 (test-curve fine-tuning leakage) → never repeat; Sprint 5
     closed-loop validation must use UQ coverage, not curve fit.
   - SD1-SD5 self-deception patterns → adversarial-review-before-commit
     is now Sprint-exit discipline (ADR-025 process lesson).
   - All MEP2 routing / KAN / curve_fit code from
     `D:/chemical-world-model-v0/` is cleanly cut. Only the conceptual
     experience and FAILURE_LOG.md format transfer to release-foundation.

## Session 6 opening checklist

When the user starts the next conversation:

1. **Confirm hydrogel as 3rd mechanism is still the choice** (low chance
   user changes mind, but always reconfirm).
2. **Ask whether the user has the first batch of chitosan curves yet**
   (they estimated ~1 week). If yes → start parsing the dataset format.
3. **Open ADR-026 (PROPOSED) and walk through it together**. If approved
   on the spot, branch `phase2-sprint1-burst` is the next action; Sprint 1
   step 1 (edit `simulator.py` + adapt `tests/test_simulator.py`) begins.
4. **Hydrogel data curation kickoff**: confirm whether user wants to drive
   data curation themselves (with a junior to help) or delegate the
   literature search to me. Either way, manual cleaning is on the
   user/lab side.
5. **Advisor handshake**: confirm the user has talked to advisor (杨超)
   about (a) using `D:/release-foundation` codebase as the working repo
   for Phase 2; (b) potentially using advisor's hydrogel project data;
   (c) closed-loop wet experiments for Nature Materials stretch.

## Pickup pointers for next session

- Last green commit on master: `384b5cb` (Phase 2 plan + ADR-026 PROPOSED).
- DECISIONS.md ends at ADR-026. ADR-027+ will be Sprint 1 verdict
  (when burst term lands), ADR-028+ Sprint 2 (encoder), etc.
- `outputs/12_baseline_comparators/` has the matched-protocol multi-seed
  numbers. Treat as canonical comparator baselines for any Phase 2
  evaluation.
- `outputs/10_eval_deployment/{summary.txt, by_curve/summary.txt,
  cross_doi_summary.txt}` are the SBI baselines to not regress against.
- World-model-v0 code at `D:/chemical-world-model-v0/` is **not** to be
  reused, only its FAILURE_LOG.md as cautionary reading.
  planned fix for the MLP encoder ceiling identified in ADR-019.
- Out of Phase 1 scope, but doable if the user wants to start parallel.

Recommendation: **A first** (Phase 1 needs closure before Phase 2 prep).

---

## Hard-earned process rules (for the next AI agent in this repo)

1. **Never `uv run`** anything. It will reinstall torch from PyPI (CPU
   wheel) and break CUDA. Always use `.\.venv\Scripts\python.exe`
   directly. The user installed `torch 2.11.0+cu128` manually; it is
   currently working. Leave it alone. (`uv run pytest` is fine; only
   `uv run python script.py` triggers the reinstall path.)

2. **Use PowerShell, not Bash, for invoking the venv python.** Bash on
   Windows mangles the leading `.\` backslash-dot pattern; the call
   becomes `..venvScriptspython.exe` and fails with exit 127. Either
   call `.\.venv\Scripts\python.exe ...` via the PowerShell tool, or
   quote the path. See HANDOFF history of failed background tasks
   before this rule landed.

3. **Check `git status --short` before editing files the user may be
   touching.** AGENTS.md hard rule #2.

4. **Evidence before configuration changes.** AGENTS.md hard rule #1.
   Especially: prior bounds, t_grid, ODE form, simulator. The pattern
   that works: query `outputs/07_synthetic_audit/per_curve_stats.csv`
   for the conditional relationship between the parameter you want to
   change and the failure mode you want to fix. See ADR-016 for the
   canonical example.

5. **The simulator has two backends; the test enforces equivalence.**
   If you edit one of `simulate` or `simulate_numpy`, edit the other
   and keep `test_torch_and_numpy_backends_agree` passing. See ADR-008.

6. **Archive failed experiments before overwriting.** Convention:
   `outputs/archive/<name>_YYYYMMDD/`. Pre-Q_max, 180-day, and 4/8
   noisy SBC runs are kept as ADR evidence; do not delete.

7. **CurvePosterior.load() reseeds torch to 0 internally.** Bug fix
   was ADR-017 for script 05, but the side effect persists: anywhere
   you call `.load()`, do `torch.manual_seed(your_seed)` *after* the
   load, not before. Scripts 05 and 09 both handle this; new
   scripts must too.

---

## If anything is unclear

Read `DECISIONS.md` end to end. Nineteen ADRs document every non-trivial
choice made during Phase 1. The narrative is chronological; you'll see
why we did what we did and what each choice ruled out.

If a future change contradicts an existing ADR: write a new ADR with a
"supersedes ADR-N" header; do not edit the old ADR's status. Append-only.

---

# Session 6 closeout — 2026-05-24

**Strategic frame locked**: user explicitly rejected JACS-floor scope.
Target is NM-tier via "physics + learned residual amortized SBI" research
path. No graduation pressure; 2-3 month research-risk acceptable. Plan
spec: `docs/plan_27d_residual_neural_ode.md`.

## What landed (all uncommitted as of session close)

| Script | Purpose | Status |
|---|---|---|
| `simulator.py` | Added `PLGABiphasicResidual(PLGABiphasic)` subclass with Fourier-sine residual on `dQ/dt`, gated by `(Q_max - Q)` | **M1 PASS** — bit-exact c=0 reduction, 500/500 physical invariants, numpy/torch agree at median 2e-6 |
| `scripts/27a_ode_expressivity_audit.py` | Multi-start NLS full-curve fit on matched 321 (n=259) with subgroup analysis | **DONE** — median R² 0.998, 95% well-fit. **ODE class is NOT the bottleneck on 321.** Refutes prior "extend ODE" hypothesis. |
| `scripts/27b_widened_prior_sbc.py` | SBC under widened prior bounds (`log_kw_hi: 1.0→1.5`, `log_alpha_hi: 1.0→1.5`, `q_burst_hi: 0.30→0.40`) | **DONE** — FAIL: 3/9 KS p<0.05 (log_kw 0.049, log_kd 0.005, q_burst 0.001). Don't widen. `log_kd` failed despite not being widened → identifiability coupling, same family as ADR-015 immediate-saturation issue. |
| `scripts/27_partial_curve_npe.py` (v1) | Hand-rolled Transformer + diagonal Gaussian on partial obs | **FAILED** (kept as ablation reference). 3d overall R²=-0.25, G2 disaster (mean_sd flat 0.73 across 1d/3d/7d). |
| `scripts/27c_sbi_npe_partial.py` | Vanilla sbi.NPE + MAF on partial obs (theta_dim=9) | **DONE** — G2 PASS, G1/G3 FAIL. 3d overall median R² -0.027 (vs baseline 24 +0.349). 3d `neither` subset +0.453 (beats baseline). 3d `fast_short` -1.601. |
| `scripts/27d_residual_sim_sanity.py` | M1 invariant checks for PLGABiphasicResidual | **DONE** — all 3 checks PASS |
| `scripts/27d_residual_amplitude_audit.py` | M1.5 forward-only audit of residual amplitude distribution | **DONE** — accepted residual-scale run is `outputs/27d_residual_amplitude_audit_s0p015/` (`sigma_c=0.015`). The default directory `outputs/27d_residual_amplitude_audit/` is the rejected `sigma_c=0.05` run; `outputs/27d_residual_amplitude_audit_s0p025/` is also rejected. |
| `scripts/27d_residual_npe.py` | M2 joint SBI training over (theta_base, c) | **20k intermediate CPU smoke DONE** — canonical result is `outputs/27d_residual_npe/summary.txt` (CPU, `n_simulations=20000`, `sigma_c=0.015`). Best result so far on the **3d fast_short** subgroup: `+0.119` (vs 27c `-1.601`, +1.72 absolute jump). 3d overall improved to `+0.280` and `neither` to `+0.508`, but **G1 still FAIL, G3b still FAIL, G4 still FAIL**. Treat this as promising subgroup rescue, not a globally solved model. |

## The critical ambiguous finding from 27d 20k intermediate

Source: `outputs/27d_residual_npe/summary.txt`

`||c||_2` median by subgroup at 3d:
- `neither`: 0.0322 (≈ prior std 0.0335)
- `fast_short`: 0.0326 (≈ prior std)

**Residual is essentially unused** — but R² improved dramatically. Two
explanations the data can't yet distinguish:
- (α) Larger network (theta_dim 9→14) provides implicit regularization
- (β) Shrinkage prior is doing the work; residual capacity is decorative

If (β) holds, the paper cannot claim "we learned the missing physics".
Must reframe as "augmented theta + shrinkage achieves SOTA point
predictions on 321" — weaker but real.

## Three queued experiments to disambiguate

ROI-ordered:

1. **n_modes=0 ablation** (CPU ~10 min) — train sbi.NPE+MAF with
   `theta_dim=9` (no residual) but otherwise identical to 27d config
   (continuous prefix [0.5, 14], sigma_c irrelevant). If R² matches 27d's
   +0.28, residual is fully decorative → (β) confirmed. If significantly
   worse, augmented theta is doing real work → (α).
2. **35k intermediate point** (CPU ~20 min) — bracket the training curve.
   If R² still climbing → run full 50k. If plateaued/regressing → 20k is
   sweet spot. Defends against 27c's "more training → more confidently
   wrong" pattern.
3. **G1 fix attempt** (CPU ~30 min) — `embedding_output_dim: 16→32`,
   `num_transforms: 5→8`. If G1 PASS at 7d → paper calibration section
   has a story.

**Recommended next session: run #1 first.** Read
`outputs/27d_residual_npe/summary.txt` and
`outputs/27d_residual_amplitude_audit_s0p015/amplitude_summary.txt`
before launching it. This 10 min CPU investment decides whether paper
framing is "learned residual physics" (true) or "residual is
regularization only" (honest but weaker).

## Other red flag worth investigating

27d 20k shows **7d prefix R² WORSE than 3d** on overall (-0.725 vs
+0.280). Direction-of-information bug — more data should help, not hurt.
Same family as 27c's pattern. May indicate MAF over-specifies posterior
modes when prefix is longer. Not a blocker for next session but should be
diagnosed before paper draft.

## Uncommitted state at session close

```
 M posterior.py                                  (pre-existing, untouched this session)
 M simulator.py                                  (added PLGABiphasicResidual at end of file)
?? docs/                                         (refactor_27_npe.md, plan_27d_residual_neural_ode.md)
?? scripts/27_partial_curve_npe.py               (v1 failed, kept as ablation)
?? scripts/27a_ode_expressivity_audit.py
?? scripts/27b_widened_prior_sbc.py
?? scripts/27c_sbi_npe_partial.py
?? scripts/27d_residual_sim_sanity.py
?? scripts/27d_residual_amplitude_audit.py
?? scripts/27d_residual_npe.py
```

**Nothing committed.** User did not ask. Next session: review and decide
commit strategy (likely group: 27a/b/c as completed diagnostics; 27d set
as research WIP).

## Bugs flagged in passing (not fixed)

1. `scripts/05_sbc_curve_posterior.py:118` uses `2x4=8` subplot for
   `n_params=9` post-ADR-026 — Q_max rank histogram silently dropped from
   plot (KS p still computed). `27b` and `27d` use dynamic ceil-divide
   layout. One-line fix when convenient.
2. `sbi.utils.MultipleIndependent` does not propagate component device.
   Working around with custom `_JointPrior` class in 27d. Saved to memory
   `reference_sbi_device_bug.md`.
3. GPU OOM on `sim.simulate(theta_batch)` when batch=1024 with residual
   ODE on 8GB 4060. Fixed by reducing default batch to 256 + per-batch
   `torch.cuda.empty_cache()`. Note for future: residual ODE has larger
   torchdiffeq state footprint than base.

## CPU vs GPU note

GPU intermittently has 6GB free per nvidia-smi but cannot allocate small
tensors (Windows + multi-process fragmentation). 27d 20k smoke ran on CPU
in 11.7 min total (4.3 min training); acceptable. Future sessions:
default to CPU for 27d until GPU memory state is reliable, or kill all
Python processes and verify before launching.
