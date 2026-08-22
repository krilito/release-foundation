## PLGA regime-conditional architecture brief for Claude

### Where we are

Continuation of the active-set branch (see
[`claude_brief_active_set_2026-05-25.md`](claude_brief_active_set_2026-05-25.md)
and [`research_log_regime_conditional_2026-05-25.md`](research_log_regime_conditional_2026-05-25.md)).

A six-step diagnostic cascade (`32` -> `33` -> `34` -> `35a` -> `35b`
-> `35c`) and a viability prototype (`36`) have established:

1. The 9-parameter ODE family has **real structure** beyond marginals
   and beyond any measured confounder.
2. That structure is **discrete (5 regimes)** and **non-geometric**
   (does not correspond to a local-linear-subspace partition).
3. The closest physical analog is reaction-mechanism class: same
   parameters, qualitatively different dynamical modes.
4. A **regime-conditional architecture is viable** on the two largest
   regimes (R6 n=146 and R5 n=112), with median R^2 within ~0.01 of
   the 9-parameter oracle when using only 4 free modal params + 5
   regime-shared fixed params.
5. The current 27a "verdict" line claiming "posterior architecture is
   the only remaining bottleneck" is too strong. The bottleneck is
   architectural in a more specific way: the posterior should be
   **regime-aware** rather than uniform over the 9-dimensional joint.

### What has been falsified (do not revisit)

- **Pairwise coupling network**: only `log_kw ~ Q_max` is consistent
  across datasets (`32`). 9 params do not form a dense coupling graph.
- **Continuous latent driver `z (k << 9) -> theta_9`**: regimes do not
  separate in v1 direction (`35b`) nor in |v1| magnitude profile (`35c`).
  Even the per-curve overlap between greedy active set and Jacobian
  top-4 is only 30% above chance.
- **Formulation-based regime assignment**: multinomial classifier
  `regime ~ formulation` reaches CV accuracy 0.343 vs top-class baseline
  0.352 (`35a`). Formulation cannot replace regime labels.

### What is justified next

In dependency order:

1. **Extend 36 to R4** (65 curves, third-largest). R2 and R3 are
   smaller and may not have enough sample for a clean prototype.
2. **Replace fixed-median with soft prior** (Gaussian centered on
   regime median, sigma = within-regime std for the 5 non-modal
   params). Re-run on R6 and R5. The ~10% R^2 < 0.9 tail in each
   regime should lift if the soft prior is the right design choice.
3. **Shape-feature regime classifier** for deployment-time regime
   assignment when full-fit theta is unavailable. Input features
   (curve-derived, not formulation-derived): early slope, t50, plateau
   height, burst presence indicator. Output: 5-class soft probabilities.
4. **Mixture-NPE end-to-end**. Choose between two designs:
   - single NPE conditioned on `(curve_features, regime_probs)`
   - true mixture of 5 NPEs, averaged by `regime_probs`
   The former is sample-efficient; the latter is more compositional.
   Decision pending the shape-feature classifier accuracy.

### Open questions for the user (not for Claude to decide alone)

1. **Tail strategy**: for the ~10% per-regime curves that fit poorly
   under regime-conditioning, do we
   (a) accept the architectural simplicity and route those to the
       full-9-param fallback,
   (b) sub-cluster within each regime (likely fragile with current n),
   or (c) commit to the soft-prior design and live with whatever the
   tail looks like after step 2 above?

2. **Regime count**: 33 picked n=6 by excess silhouette but R1 is
   trivial (n=2). Effectively 5 regimes. Is the user OK with this, or
   should we re-examine n=4 / n=5 cuts before building the classifier?

3. **27a verdict text**: should be updated to remove the "GO 27 vanilla
   NPE" / "posterior architecture is the only remaining bottleneck"
   language, which the cascade has falsified. Pending user confirmation
   before editing 27a outputs.

### What Claude should NOT do without explicit user approval

- Start writing the mixture-NPE training code. That depends on shape-
  feature classifier accuracy and tail-strategy decision.
- Re-run 33 with a different regime count. The current n=6 cut is the
  basis for all 34/35a/35b/35c/36 results; changing it invalidates them.
- Modify 27a outputs or verdict text. The verdict is wrong, but
  rewriting prior outputs is a documentation decision the user owns.
- Install `sbi` to enable NPE prototyping. Detected in 35b that the
  environment does not have it; do not modify the environment.

### Files to inspect

- `scripts/32_active_set_substitution.py` ... `scripts/36_r6_regime_prototype.py`
- `outputs/32_active_set_substitution/summary.txt`
- `outputs/33_active_set_regimes/summary.txt`
- `outputs/33_active_set_regimes/regime_active_set_heatmap.png`
- `outputs/34_regime_confound_partial/summary.txt`
- `outputs/35a_regime_formulation_partial/summary.txt`
- `outputs/35b_regime_jacobian_alignment/summary.txt`
- `outputs/35b_regime_jacobian_alignment/regime_v1_heatmap.png`
- `outputs/35c_regime_v1_magnitude/summary.txt`
- `outputs/36_r6_regime_prototype/summary.txt`
- `outputs/36_r5_regime_prototype/summary.txt`

### My recommended priority

Do (1) and (2) above before anything else. They are cheap (each is a
~30-minute compute job) and they decide whether the regime-conditional
plan survives contact with the harder regimes and with soft priors.
Only after (1)+(2) is the shape-feature classifier (step 3) worth
designing, because the classifier's target depends on whether we are
committing to 5 regimes with soft priors or to a more nuanced
sub-clustered scheme.
