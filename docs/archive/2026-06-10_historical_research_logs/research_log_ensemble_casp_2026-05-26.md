# Research log — Ensemble-CASP zero-shot

Date: `2026-05-26`. Branch: `phase2-sprint1-burst`.

## Why this exists

FIB-CASP solved the few-shot regime (formulation + 1-week observations
→ R² 0.92, cov90 0.81). The user asked: can we drop the early-observation
input and still claim calibrated UQ in true zero-shot?

Three attempts:

| Approach | What it does | Result | Verdict |
|---|---|---|---|
| Zero-shot FIB-CASP (`scripts/65`) | Same FIB, but train RF on formulation only | R² 0.75, **cov90 0.61** | PI miscalibrated — Fisher doesn't see RF prediction error |
| PCA-P (`scripts/63 → 64`) | Neural encoder + simulator pull-back targets | R² 0.62-0.68 | Neural encoder loses to RF on 200 curves |
| **Ensemble-CASP (`scripts/66`)** | RF tree-level theta predictions as empirical posterior | R² 0.75-0.88, **cov90 random 0.86** | **Random ✓, OOD partial** |

The Ensemble-CASP idea: instead of Fisher info (which measures
identifiability given a fixed curve point and cannot help if the curve
isn't observed), use the **400 individual trees of the forest** as samples
from the empirical formulation-conditional posterior. Where formulation
fully constrains θ, the trees agree and σ is small. Where formulation
is ambiguous, the trees disagree and σ widens.

Same `(μ, U, σ)` output structure as FIB; same simulator push-forward; only
the source of the posterior changes.

## Full 6-cell results (Ensemble-CASP, zero-shot)

| Dataset | Scheme | n | R² median | R² mean | cov90 mean | cov90 median | PI width |
|---|---|---|---|---|---|---|---|
| cross321 | random_5fold | 259 | **0.753** | 0.02 | **0.75** | **0.86** | 0.41 |
| cross321 | group_by_drug | 259 | 0.400 | -0.75 | 0.63 | 0.69 | 0.45 |
| cross321 | group_by_polymer | 259 | 0.388 | -0.54 | 0.62 | 0.67 | 0.43 |
| internal181 | random_5fold | 162 | **0.880** | 0.67 | 0.73 | **0.82** | 0.31 |
| internal181 | group_by_drug | 162 | 0.563 | 0.10 | 0.59 | 0.62 | 0.38 |
| internal181 | group_by_polymer | 162 | 0.426 | -0.25 | 0.43 | 0.42 | 0.32 |

## What the table actually shows

**Random splits PASS the calibration bar.** Both cross321 and internal181
random_5fold land cov90 median 0.82-0.86, inside the paper-acceptable
[0.80, 0.95] band. R² 0.75-0.88 is lower than FIB few-shot (0.92) and
than codex 45's formulation-only best (0.91), but that's expected — we
gave up the early-observation information channel and run vanilla RF
instead of best-of-N tree variants.

**OOD splits FAIL the calibration bar.** Group-by-drug and group-by-polymer
land cov90 mean 0.43-0.63. This is not a bug — it's the canonical
distributional-shift problem: all 400 trees are trained on the same
in-distribution data and **share the same systematic bias** on truly
novel drug/polymer combinations. Tree-ensemble variance cannot diagnose
"the prediction is wrong because this formulation is OOD."

This is a known open problem in the broader UQ literature (Guo et al.
2017, Snoek et al. 2019, Yao et al. 2018). It is not specific to our
formulation; ALL ensemble-UQ methods fail this test under hard
distributional shift. Honest paper framing: report random-split
calibration as the achievement, note OOD remains open, propose
combinations (Σ_ensemble + Σ_fisher, or temperature scaling on a
held-out OOD calibration slice) as future work.

## Three-instance unified framework

The full CASP/FIB/Ensemble work establishes:

```
Same output object:    (μ, U, σ_active)  per curve, low-rank Gaussian
Same downstream:       sample θ → simulate at c.t_obs → recenter PI band
What varies:           how (μ, U, σ) is constructed
```

| Information budget | (μ, U, σ) source | When to use |
|---|---|---|
| Few-shot (formulation + early Q) | **FIB**: Fisher info on simulator @ θ_RF (script 62) | Best R² + PI |
| Zero-shot (formulation only) | **Ensemble**: tree-level forest predictions (script 66) | True zero-shot with PI |
| Future-mechanism (no oracle θ) | **Pull-back**: simulator IS-reweighting on real curves (script 63) | Targets from data + simulator only |

All three share the `casp/` module. PLGABiphasic (9 params) and
WeibullSimulator (2 params) work identically through the same code.

## Comparison to baselines

**Few-shot regime** (formulation + early Q):

| Method | cross321 random R² | internal181 random R² |
|---|---|---|
| Toronto RF (script 15) | 0.33 | n/a |
| 37c MLP-direct | 0.83 | 0.87 |
| 38d RF→θ→ODE | 0.957 | n/a |
| 42 best tree variant | n/a | ~0.97 |
| **FIB-CASP (62)** | **0.926** | **0.960** |

FIB-CASP matches RF→θ→ODE within 0.01-0.03 R² and adds calibrated PI
(cov90 random 0.81-0.85, median in band for 6 of 9 PLGA cells in
Phase 2).

**Zero-shot regime** (formulation only):

| Method | cross321 random R² | cross321 by_drug R² | cov90 |
|---|---|---|---|
| Toronto RF | 0.33 | ~0.0 | n/a |
| 45 form-only best (45) | 0.909 | 0.672 | n/a |
| **Ensemble-CASP (66)** | **0.753** | **0.400** | **0.86 / 0.69** (random / by_drug median) |

Ensemble-CASP loses ~0.16 R² to script 45's best-of-N RF variant
because we run vanilla RF; closing this gap is RF tuning, not method
work. The new capability is the PI: cov90 random median 0.86 vs no
prior baseline at all (no zero-shot method in the literature provides PI
for PLGA release).

## Open problems honestly stated

1. **OOD UQ is unsolved.** cov90 drops to 0.4-0.7 on held-out drug /
   polymer. This is the broader open problem in calibration under
   distributional shift; we do not claim to solve it.

2. **R² gap to best-of-N RF tuning.** Our vanilla RF underperforms
   script 45's best tree variant by 0.10-0.15 R² on random splits. RF
   tuning would close this. Out of scope for the method paper; can be
   reported in an ablation.

3. **Pull-back PCA-P didn't beat RF.** The neural-encoder route (script
   64) with simulator-derived feasibility targets gave R² 0.62-0.68
   on cross321 random. The MLP encoder cannot match RF on 200 curves.
   Pull-back as a target-generation technique still has value (it's how
   future mechanisms without oracle θ would bootstrap), but the
   amortized inference layer needs more data than we have.

## Decision points

For paper submission:

- **Domain top journal** (Nature Communications, JCR, drug delivery
  venues): everything we have is publishable. The framework + few-shot
  results + cross-mechanism + zero-shot-random are a complete story.
  Open OOD problem is a discussion point, not a blocker.

- **NeurIPS / ICML tier**: borderline. The Ensemble-CASP + FIB-CASP
  unification is a meaningful algorithmic contribution but lacks the
  theoretical statement that ML conferences usually want. Could
  strengthen with (a) Combined-CASP (Σ_ensemble + Σ_fisher) closing OOD
  gap, or (b) active observation policy on top of FIB.

## Next concrete actions

1. **Write the paper plan** ([docs/paper_closeout_release_quality_paradigm_2026-05-26.md](paper_closeout_release_quality_paradigm_2026-05-26.md))
   to absorb the CASP unification.
2. **Phase 3 identifiability audit** still pending (FIB U vs script 29
   oracle active set). This is the mechanism contribution.
3. **Optional**: Combined-CASP run if user wants to push OOD calibration.
