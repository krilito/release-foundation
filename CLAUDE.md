# CLAUDE.md

General behavioral guidelines for any AI assistant working in this repo.
Adapted from [Andrej Karpathy's observations on LLM coding pitfalls](https://x.com/karpathy/status/2015883857489522876),
via the [karpathy-guidelines skill](https://github.com/multica-ai/andrej-karpathy-skills) (MIT).

**Read [AGENTS.md](AGENTS.md) after this for project-specific hard constraints.**

These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

In this repo specifically: if the user asks for a model change, **first** identify
whether it's a physics decision (ODE form, prior bounds), an inference decision
(posterior architecture, training objective), or an evaluation decision (splits,
metrics). These have different reversal costs and require different evidence.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify. This codebase grew out of a "patchwork" predecessor; resist the
temptation to re-create that pattern by adding modules before they are needed.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

In this repo specifically: success on numerical / Bayesian work is measured by
**diagnostics**, not by code running without error. SBC calibration plots, posterior
coverage curves, and grouped-holdout R² are the real verifiers — not green
checkmarks from `pytest`.

---

## 5. Project Memory (updated 2026-06-13)

Session context that future assistants must read before touching this repo.

### What happened in the 2026-06-13 formalism + RSF session

1. **Diagnosis:** The project had 15+ branches, 119 numbered scripts, and 50+
   docs, but no unified mathematical object. Every system (PLGA, liposome,
   chitosan) re-invented curve states, priors, residuals, and symptoms inside
   one-off scripts. The root cause was that the Release State Inference
   framework existed only as natural language, never as formal definitions or
   shared code interfaces.

2. **Formalism created:** `docs/release_state_inference_formalism_2026-06-13.md`
   defines seven formal objects:
   - Definition 1: Release State Space `S_c = (Z_c, e_c, g_c, d_c)`
   - Definition 2: Curve-Derived State `z_Q = e_c(Y)`
   - Definition 3: Static Prior `π(z|X)`
   - Definition 4: Observation Update `π(z|X, Y_k)`
   - Definition 5: Information Value `V(k)`
   - Definition 6: Missing-State Residual `r = d_c(z_Q, z_X)`
   - Definition 7: Residual Symptom (Mode A: interpretable params, Mode B:
     decoded time-domain perturbations)

   Hard rule: **do not subtract states from different spaces.** PCA residuals
   and ODE theta residuals live in different Z_c and cannot be compared without
   projecting through a shared metric or decoding to curve space.

3. **Code interface created:** `release_state.py` implements
   `ReleaseStateSpace` (ABC), `PCACurveStateSpace`, `WeibullStateSpace`,
   `PLGAODEStateSpace`, `StaticPrior` (ABC), `TreePrior`, and
   `ObservationBudgetAnalyzer`. Tests in `tests/test_release_state.py`.

4. **RSF is application-layer, not foundation-layer.** The RSF plan doc was
   corrected: RSF (Release-State Forensics) depends on the formalism layer.
   Script 120 = residual map only. MSVS / candidate measurement scoring
   starts in 121. No MSVS allowed inside 120.

5. **120 completed and frozen.** 128 PLGA curves, 3 state spaces (PCA8, PCA3,
   Weibull), 3 priors (global_mean, ExtraTrees, Ridge), 3 split schemes
   (random, group-by-DOI, group-by-Formulation-Method). All 11 data checks
   pass. Key results:
   - `static_X_beats_global_state_prior`: pass (random split only)
   - `strict_group_split_static_signal_exists`: **warn** (best strict gain =
     -0.0144, confirming static X fails under DOI/method transfer)
   - `residuals_stable_across_representations`: pass (mean Spearman = 0.668,
     mean Jaccard = 0.831)
   - `source_or_method_structure_present`: **warn** (38 associations > shuffle
     null, DOI hits = 16)

6. **121 completed and frozen.** Candidate measurement value scoring with 14
   real candidates + 100 null candidates + 1 fixed negative control. All 19
   data checks pass. Strict OOF verification against reconstructed 120 folds.
   Key results:
   - Top evidence-backed: `initial_drug_polymer_ratio` (EvidenceScore 0.352,
     null %ile 99%, source penalty 0.30), `particle_size_distribution`
     (0.346, 99%, 0.21)
   - `drug_loading_capacity` has highest proxy support (0.507) but very high
     source penalty (0.749) — likely source artifact
   - `null_candidates_rank_low`: **warn** (best real 0.352 < best null 0.377)
   - `strict_prediction_gain_in_at_least_two_state_spaces`: **warn** (only 1)
   - `safe_to_enter_122`: **exploratory_only**
   - Evidence/hypothesis rankings are strictly separated; no_proxy candidates
     have NaN EvidenceScore by design

7. **What 121's result means for the paper.** The result is *not* "RSF
   discovered the hidden variables." The result is: existing proxy signal is
   detectable but weak, source confounding is the dominant structure, and
   missing information is likely process/microstructure/assay rather than
   molecular descriptors. This aligns with liposome 117 descriptor enrichment
   negative result.

### Current state for next session

- **120 and 121 are frozen.** Do not re-run or modify unless a new decision in
  DECISIONS.md explicitly reopens them.
- **122 (active remeasurement planner)** is allowed but must be exploratory.
  121 gave `safe_to_enter_122 = exploratory_only`. If 122 is written, it must
  lead with "121 did not pass the following gates" and frame outputs as
  hypotheses, not recommendations.
- **More valuable next step** may be writing 120+121 results directly into the
  manuscript "missing state diagnosis" section rather than proceeding to 122.
- **Long-term code hygiene:** `load_real_321`, `make_splits`,
  `build_curve_matrix`, `make_state_space` should be extracted from
  `scripts/120_*.py` into a shared `plga_data.py` module. 121 currently uses
  `importlib` to import 120 as a module, which works but is fragile.
- **curve_symptoms time windows** (0-7d burst, 7-30d slope, 60+d tail) are
  PLGA-specific. Must be parameterized when extending to liposome/hydrogel.
- **Signature matrix weights** in 121 are expert prior. Sensitivity analysis
  (±50% perturbation) should be prepared for reviewer defense.

### Interpretation boundaries that must not be violated

```
120 maps residuals. It does not identify physical hidden variables.
121 ranks candidate measurements. It does not discover causation.
A high PriorityScore is a testable hypothesis, not empirical evidence.
DOI/method structure may be source, protocol, assay, or batch confounding.
No candidate is causal without direct experimental validation.
```

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, and clarifying questions come **before**
implementation rather than after mistakes.
