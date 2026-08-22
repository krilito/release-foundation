# Paper Requirements — Provisional 2026-05-28

Status note:
- This file is retained at its original path so existing references do not break.
- It is no longer the sole source of truth.
- Use `docs/drug_release_closeout_2026-05-28.md` for the current release-only closeout state.

Working requirements note for what must be done, measured, and avoided to
produce a defensible **Nature Methods / NMI / Nature Communications** tier
submission on mechanism-constrained Bayesian inference for sparse drug
release prediction.

> **Realistic target**: NM / NMI / Nat Comms (acceptance ≤ 15%).
> **Not target**: Nature main / Science main. Reaching that tier requires
> wet-lab inverse design closed-loop + multi-mechanism foundation model,
> which is a 12+ month additional program.

This document is now a provisional working note. If it contradicts current
durable outputs or `docs/drug_release_closeout_2026-05-28.md`, the closeout
document and the durable output files win.

---

## 0. Anti-hype discipline (meta-rules)

These are the rules the assistant and the user must both enforce.

**R1.** Any "if X then claim Y" requires X to be a **measured number**,
not a plausible explanation. Unverified intermediate step → all downstream
claims become "hypothesis to test", not paper-eligible.

**R2.** Framing escalation is not allowed after a single positive
diagnostic. **Two consecutive diagnostics from different angles** required
before upgrading framing.

**R3.** Realistic ceiling stays NM/NMI/Nat Comms through Sprint 4. Only
after all primary endpoints are met may we consider escalating target
journal.

**R4.** Banned vocabulary unless precondition met:

| Banned word | Precondition |
|---|---|
| "World model" | learned latent dynamics **+** action/intervention space **+** planning capability demonstrated |
| "Active sensing" / "Active learning" | expected information gain (EIG) computed, not just variance reduction |
| "Foundation model" | ≥ 3 mechanisms × ≥ 1000 curves jointly trained |
| "By design" | hyperparameter was data-selected with sensitivity sweep |
| "Few-shot" | input does NOT include same-curve observations |
| "Cross-mechanism" | ≥ 2 mechanisms × ≥ 200 curves each, independently held out |
| "Emergent" | unsupervised structure recovered without label leakage, sil > 0.3, p < 0.001 |
| "Paradigm" | reviewers from outside the project agree it is one |

**R5.** When the assistant escalates framing without new measured
evidence, the user invokes **"hype 警报"** and the assistant must revert
and audit.

**R6.** Plausible explanations are stored in `docs/hypotheses.md`, not in
paper outlines. Migrating a hypothesis to paper requires Rule R1.

---

## 1. Pre-registration protocol (multi-batch)

**P1.** Each prospective wet-lab batch must be locked by SHA256 hash +
signed git tag **BEFORE** the corresponding wet-lab measurement begins.

**P2.** **Batch 1 (12 chitosan curves) MUST be hashed and tagged by
2026-05-29 EOD** or the prospective premium is forfeit. Wet-lab data
expected ≤ 2026-06-04.

**P3.** Each subsequent batch follows the same protocol independently.
Multiple independent batches are **stronger** evidence than one large
batch (standard practice in adaptive RCTs).

**P4.** Each batch's pre-registration document must specify, in writing,
before any wet-lab data:
- Hypothesis (one sentence)
- Primary endpoint (one metric, one threshold)
- Secondary endpoints (≤ 3 metrics)
- Pre-specified success threshold
- Honest fallback claim if endpoint not met
- Wet-lab data acquisition deadline
- What is decided in advance vs. what is left for post-hoc

**P5.** Timestamps go to **at least two** of:
- Signed git tag (cryptographic)
- Email to PhD advisor with hashes in body (institutional)
- OSF private registration (third-party)
- Public GitHub commit on a tracked branch

**P6.** Hashes cover: `predictions.csv`, `lock_metadata.json`, and any
auxiliary files (`theta_targets.npz`, etc.). Hash file is itself committed
and tagged.

---

## 2. Statistical rigor (required for every numerical claim)

**S1.** Any RMSE comparison must report **95% CI via paired bootstrap**
(≥ 2000 resamples). CI crossing 0 → comparison is "not significant" and
**cannot be written as a claim**.

**S2.** **R² reported as pooled R²** over (curve, time) pairs, not
per-curve R² median. Per-curve R² on ≤ 10 timepoints is unstable;
"% curves with R² > 0" is not a defensible metric.

**S3.** Any "X% improvement" claim must be evaluated on **≥ 3 splits**:
- canonical random split (locked)
- group_by_drug
- group_by_polymer
- plus held-out chitosan if available

A claim that holds on 1 split but fails on others must be reported with
the split it holds on, explicitly.

**S4.** Coverage claims (cov90, cov50) must report:
- (a) mean
- (b) median
- (c) **interval sharpness** (mean PI width)
- (d) comparison to naive baseline (e.g., always-90%-band of training data)

Wide intervals with high coverage are not a method advantage. Conformal
calibration explicitly trades sharpness for coverage; sharpness must be
reported.

**S5.** Any clustering / latent structure claim requires:
- Silhouette score
- Permutation p-value (≥ 1000 permutations)
- Comparison silhouette in baseline feature space (e.g., `x` alone)
- Explicit feature space used for clustering

**S6.** Any posterior calibration claim requires **Simulation-Based
Calibration (SBC, Talts et al. 2018)**: sample θ ~ prior, simulate y, fit
posterior, check rank uniformity. cov90 alone is not sufficient.

**S7.** Multiple-comparison correction: when reporting K independent
hypothesis tests, apply Bonferroni or Benjamini-Hochberg. Document which.

---

## 3. Evidence-per-claim matrix

For each potential paper claim, this matrix tracks what is measured vs.
what is assumed. **Only "MEASURED" rows may be written as paper claims**.

| # | Potential claim | Required evidence | Status |
|---|---|---|---|
| E1 | Active Observer v3 improves RMSE over Direct-Q by ~27% (CI [8, 47]) | Bootstrap CI on ≥ 3 splits | **RETRACTED**: 4-way factorial shows +12.9% canonical (CI crosses 0) and +0.1% LOOCV (not significant). Inference effect is negative: Active worse than DirectQ. See 72_canonical_benchmark_v2 and 77b_regime_benefit_loocv_v2. |
| E2 | Active Observer v3 achieves cov90 = 0.93 with reasonable sharpness | cov90 + cov50 + mean width + naive baseline comparison | cov90/width reported in `outputs_active_observer_v3/metrics_summary.csv`; naive baseline comparison still missing, so keep supplementary. |
| E3 | Drug release exhibits 6 discrete identifiability regimes | Silhouette in active-set space ≥ 0.3, perm p < 0.001 | **MEASURED** (sil=0.284, p<10⁻⁴); demoted to supplementary per ADR-027 |
| E4 | Identifiability regime structure explains Active Observer benefit | Kruskal-Wallis on per-curve benefit by regime, p < 0.05; N ≥ 5 per regime | **NOT SUPPORTED**: LOOCV N=259 p=0.521. No regime × benefit interaction. See 77b_regime_benefit_loocv_v2. |
| E5 | RSSM stochastic latent suffers posterior collapse in sparse drug release data | KL/dim, posterior μ variance, cov90 | **MEASURED** (BUG 4 fixed): `outputs/71_diagnostic_bugfix/summary.json` currently shows KL/dim=0.028, cov90=0.411. Posterior collapse / under-coverage still supported, but do not cite superseded intermediate numbers. |
| E6 | RSSM deterministic encoding integrates dynamics information beyond static formulation features | Probe R² monotonic with t_query | **MEASURED**: 0.32 → 0.53 → 0.73 |
| E7 | Chitosan prospective: cov90 = X/12 within band | Wet-lab data + pre-registration timestamp | Prediction lock artifacts verified (`outputs/67_chitosan_prospective/HASHES.txt`, git tag `prospective-chitosan-batch1-2026-05-28`); real wet-lab reveal still pending. |
| E8 | Methods generalize across PLGA, liposome, chitosan | Canonical benchmark on all three with same methodology, N ≥ 100 each | **PLGA done, liposome marginal (93), chitosan prediction-only** |
| E9 | Active Observer outperforms learned latent dynamics (RSSM) in sparse regime | Head-to-head on canonical split, same evaluation | **RETRACTED**: original 0.145 vs 0.167 was on buggy split. 4-way factorial shows Active-adaptive RMSE=0.151 (canonical), 0.084 (LOOCV). RSSM comparison needs rerun on corrected split. |
| E10 | Regime structure is invisible in formulation features and learned trajectories | sil_x ≈ sil_h ≈ 0 with negative direction | **MEASURED**: sil_x = -0.18, sil_h = -0.19 |
| E11 | Direct-Q + h(t=14d) features outperform Direct-Q alone | RMSE comparison with bootstrap CI | **RETRACTED**: original 6.7% improvement was BUG 1 (GRU trained on test set). Train-only GRU shows 0.174 → 0.181 (WORSE). See 72b_canonical_benchmark_fixed.py. |
| E12 | Inverse design produces target-matching chitosan formulations validated by wet lab | Hit rate of model-proposed formulations against target profiles | **NOT STARTED** |

---

## 4. Methodological baselines required (before submission)

| ID | Baseline | Why required | Status |
|---|---|---|---|
| B1 | Direct-Q (ExtraTrees) | Point predictor floor | DONE |
| B2 | Active Observer v3 + conformal | Primary method | DONE |
| B3 | RSSM stochastic + deterministic | "We tried world model" defense | DONE |
| B4 | NPE / SNRE from sbi library | SBI gold standard | **REQUIRED, NOT DONE** |
| B5 | HMC on ~50-curve subset | Bayesian posterior ground truth | **REQUIRED, NOT DONE** |
| B6 | fPCA / functional regression | Functional-data baseline | **REQUIRED, status unknown** |
| B7 | Mean baseline (predict global mean) | R² sanity floor | **REQUIRED, status unknown** |
| B8 | Direct-Q + h(t=14d) | Hybrid feature baseline | DONE (no CI yet) |

Any method paper that omits NPE/SNRE comparison in SBI-adjacent work will
be flagged by Cranmer/Brehmer-circle reviewers. B4 is mandatory.

---

## 5. Sensitivity sweeps required

Any hardcoded hyperparameter that influences a headline number must be
swept. Only after sweep can the number be reported as "X under
hyperparameter Y, with monotone behavior across [Y_low, Y_high]".

| ID | Parameter | Sweep range | Why |
|---|---|---|---|
| V1 | Active observer particle count | {100, 200, 400, 800} | claim depends on it |
| V2 | Active observer observation budget | {1, 2, 3, 4} | claim depends on it |
| V3 | CASP rank `r` | {2, 3, 4, 5, 6} | currently hardcoded 4, marketed as "by design" |
| V4 | CASP `σ₀_frac` | {0.01, 0.05, 0.1, 0.2} | inconsistent defaults across modules |
| V5 | Conformal coverage targets | {0.50, 0.80, 0.90, 0.95} | claim is one number, need scan |
| V6 | RSSM KL weight (if revisited) | {0.01, 0.1, 1.0, KL-balanced} | currently posterior-collapsed |

---

## 6. Reproducibility / artifact requirements

**A1.** Every script accepts `--seed` and seeds `torch`, `numpy`, `random`,
and any `sbi` / `scipy` code paths. Default seed: `0`.

**A2.** Every output directory contains `lock_metadata.json` with:
git hash, UTC timestamp, prior bounds, seed, command-line args used,
package versions.

**A3.** Canonical evaluation split is saved as a **fixed file**:
`data/canonical_split_v1.csv` (curve_id, split_name, fold_index). All
methods evaluated on the **same file**.

**A4.** All headline numbers in paper must be traceable to a **single
output file** committed to git. No number lives only in chat logs or
agent outputs.

**A5.** Multi-batch chitosan predictions each in their own dated output
directory: `outputs/67_chitosan_prospective_batchN_YYYY-MM-DD/`.

**A6.** Figure-generating scripts live in `scripts/figures/`. Each figure
script consumes only files in `outputs/` (no recomputation in plotting).

---

## 7. Honest negative result handling

**N1.** RSSM stochastic latent collapse → write as Methods Comparison
("we tried world models, here's why they fail in this regime"), **not
hidden, not headline**.

**N2.** Liposome OOD coverage failure (cov90 mean 0.43-0.69 on group
splits) → discuss explicitly in **Limitations**.

**N3.** Per-curve R² median negative on sparse evaluation → explain in
Methods, use pooled R² as primary metric (S2).

**N4.** If active-set silhouette ≤ 0.3 OR Kruskal p > 0.05 → regime
contribution **demoted to supplementary**, paper restructured to 2
claims (Active Observer + chitosan prospective).

**N5.** If chitosan wet-lab cov90 < 0.5 across all 12 curves → reported
honestly as method limitation. Paper still publishable due to
pre-registration; framing shifts to "honest evaluation of cross-mechanism
transfer limits."

**N6.** "Cherry-picked the split where it worked" is not allowed. If a
method works on 1 of 3 splits, report all 3.

---

## 8. Decision rules / kill criteria

Pre-decided to avoid rationalization after seeing data.

**K1.** If active-set silhouette < 0.3 OR Kruskal p > 0.05 (regime
verifications):
→ Demote regime to supplementary, restructure paper around Active
Observer + chitosan prospective + RSSM negative.

**K2.** If chitosan wet-lab cov90 < 0.5 across batch 1:
→ Reframe paper around PLGA-only with chitosan as "limitation of
cross-mechanism transfer." Batch 2 design must address whatever failed.

**K3.** If bootstrap CI on Active Observer vs Direct-Q crosses 0 on ≥ 2
of 3 splits:
→ "27% improvement" claim killed. Restructure paper around calibrated UQ
and prospective validation only.

**K4.** If NPE/SNRE baseline matches or beats Active Observer:
→ Active Observer is not novel; paper restructured around regime +
prospective contribution + framework convenience.

**K5.** If 2 independent reviewers (e.g., labmate, advisor) cannot follow
the unified story without prompting:
→ Restructure as separate Method paper + separate Discovery paper.

**K6.** If RSSM Dreamer-V3 recipe is attempted and posterior still
collapses:
→ Permanently kill RSSM line. Negative-result paragraph stays.

---

## 9. Sprint schedule (4-6 weeks, target submission 2026-08-19)

### Sprint 0 — this week (2026-05-28 → 2026-06-03)
- [TODAY] Hash batch 1 chitosan predictions + git tag (5 min)
- [TODAY] Run active-set silhouette verification (10 min)
- [TODAY] Run regime × benefit Kruskal verification (1-2 h)
- [Day 2-3] Draft prospective registration document v1
- [Day 4-5] Resolve U1 (Direct-Q 0.161 vs 0.120 discrepancy audit)
- [Day 4-5] Read script 33 to confirm regime clustering input (U2)
- [Day 5-7] Wait for chitosan wet-lab data (expected by 2026-06-04)

### Sprint 1 — week 2-3 (2026-06-04 → 2026-06-17)
- Run canonical benchmark on group_by_drug + group_by_polymer splits
- Run NPE/SNRE baseline (B4)
- Run HMC on 50-curve subset (B5)
- Run sensitivity sweeps V1-V5
- Process chitosan wet-lab data, compute pre-registered cov90 +
  sharpness + per-curve table
- Decide K1, K2, K3 based on above

### Sprint 2 — week 4-5 (2026-06-18 → 2026-07-08)
- Cross-mechanism regime replication (chitosan, liposome)
- SBC on Active Observer posterior (S6)
- Direct-Q + h-features expanded evaluation with CI
- Draft Methods + Results sections
- Plan batch 2 chitosan experiments (intervention-pair design if K2
  passes; safety formulations if K2 fails)

### Sprint 3 — week 6-7 (2026-07-09 → 2026-07-29)
- Lock and hash batch 2 predictions (if applicable)
- Draft Discussion + Introduction
- Internal review by PhD advisor
- Generate all final figures

### Sprint 4 — week 8 (2026-07-30 → 2026-08-19)
- Address advisor feedback
- Polish figures (consistent style, colorblind-safe, vector format)
- Final reproducibility audit (A1-A6)
- Submit to target journal

---

## 10. Existing hard constraints (inherited)

All of the following remain in force and are referenced here for
completeness, not duplicated:

- `AGENTS.md` Hard NO list (no PySR, no KAN, no LGBM mapper, no committing
  outputs/, etc.)
- `AGENTS.md` Hard YES list (evidence before configuration changes, check
  git status before editing user-modified files)
- `AGENTS.md` Active Observer invariant: `simulate_plga_ode` MUST use
  `PLGABiphasic.simulate_numpy()`. No surrogate, no neural ODE, no custom
  solver.
- `CLAUDE.md` Karpathy 4 principles (think before coding, simplicity first,
  surgical changes, goal-driven execution)
- ADR process: any change to ODE form, prior bounds, t_grid requires
  diagnostic measurement before configuration change

---

## 11. Open uncertainties (must resolve before submission)

| ID | Question | Resolution path | Deadline |
|---|---|---|---|
| U1 | Why does Direct-Q RMSE differ between 0.161 and 0.120 reports? | Audit which split / version produced each; lock canonical at 0.176 | Sprint 0 |
| U2 | What features did script 33 cluster on? | Read script 33 source; document feature space | Sprint 0 |
| U3 | Is the "active" claim BOED-defensible or should it be renamed? | Compute EIG once for comparison; decide naming | Sprint 1 |
| U4 | Is CASP included in this paper or held back as separate submission? | Decide based on whether regime + identifiability story coheres with CASP | Sprint 2 |
| U5 | Chitosan batch 2 design: random formulations or intervention-pair design? | Decide after batch 1 wet-lab analysis | Sprint 2 |
| U6 | Should liposome 93 curves be in main paper or supplementary? | Decide based on whether OOD analysis is honest enough for main | Sprint 2 |
| U7 | Does "discrete regime" survive in chitosan and liposome data? | Replicate clustering on each mechanism's data | Sprint 2 |

---

## 12. What this document is NOT

- Not a research log (see `research/active_observer/log_*.md`)
- Not a paper outline (paper outline is pending Sprint 0 verifications)
- Not a hypothesis dump (see `docs/hypotheses.md` if needed)
- Not a list of nice-to-haves (everything here is mandatory or a kill
  criterion)

If a requirement is not in this document, it is not required. If a
requirement appears here, it is mandatory. Changes to this document
require an ADR-style entry in `DECISIONS.md`.

---

**Locked**: 2026-05-28
**Next review**: end of Sprint 1 (2026-06-17)
**Owner**: project lead (you)
**Authority**: this document supersedes prior paper-related framing in
all other docs.
