## Research Log - Regime-conditional branch

Date: `2026-05-25`

Continues from [`research_log_active_set_2026-05-25.md`](research_log_active_set_2026-05-25.md).

### Goal of this branch

The active-set branch established that:

- 9-parameter ODE family is globally useful (`27a` median R^2 = 0.998)
- Each individual curve usually activates only 2-4 effective directions (`29`, `30`)
- Different curves use different active subsets (`31`)
- There is a "shared backbone" pattern at marginal-inclusion-rate level

This branch asks the next question: **what structure actually drives the
active-set heterogeneity, and what architecture should the next NPE use?**

Three candidate stories were on the table:

1. **Pairwise coupling network** ("free radicals catalyzing each other")
2. **Continuous latent driver** `z (k << 9) -> theta_9`
3. **Discrete regime classes** (different mechanism modes, like reaction
   mechanism classes SN1 / SN2 / E1 / E2)

The branch designed a falsification cascade for each, then a viability
prototype for whichever survived.

### Work completed

#### 32 active-set pairwise substitution test
Script: `scripts/32_active_set_substitution.py`
Output: `outputs/32_active_set_substitution/`

- Null model: row+column-marginal-preserving swap permutation
- Tested k=2, 3, 4
- Result: only `log_kw ~ Q_max` is consistent complement across both
  datasets (z_int=+3.36, z_cross=+3.01 at k=2). No pair crosses the
  Bonferroni line (|z| >= 3.4) on both datasets. Most pairs flip sign
  between datasets.
- **Falsifies story 1**: dense pairwise coupling network does not exist.

#### 33 active-set regime discovery
Script: `scripts/33_active_set_regimes.py`
Output: `outputs/33_active_set_regimes/`

- Jaccard distance + average-linkage hierarchical clustering
- Selection by excess silhouette over marginal-preserving null (not raw
  silhouette, which trivially picks degenerate outlier splits)
- Best n_clusters = 6 (observed silhouette 0.280 vs null p95 0.232, p=0.000)
- R1 (n=2) is an outlier cluster; R2-R6 are physically readable:
  - R2 (28): `log_kh, q_burst, log_ke, Q_max` -- dual-rate diffusion + burst
  - R3 (25): `m_crit, q_burst, log_kw, Q_max` -- erosion-gate dominated
  - R4 (65): `log_ke, log_kd, q_burst, log_kw` -- multi-rate + burst, no Q_max
  - R5 (112): `log_kd, q_burst, log_kw, Q_max` -- classical burst + diffusion + capacity
  - R6 (146): `log_kh, log_kd, Q_max, log_kw` -- smooth diffusion + hydration, no burst

External descriptor reverse-check passed: t_max, fast_regime, short_window
all separate regimes (p < 0.01). But t_max is the strongest separator,
flagging an observation-window confound that 34 had to address.

#### 34 observation-window confound partial-out
Script: `scripts/34_regime_confound_partial.py`
Output: `outputs/34_regime_confound_partial/`

- t_max-stratified chi-square (3 quantile bins) + logistic LR
- Per-stratum: only mid-window stratum significant (t2: chi^2=18, p=0.0026,
  Cramer V=0.50). Short and long windows uninformative -- makes physical
  sense (no dynamics resolved / saturated).
- Stouffer combined p = 0.0019
- Logistic LR (`fast ~ t_max + short + C(regime)` vs without regime):
  chi^2=19.2, df=4, **p=0.0007**
- Verdict: regime predicts `fast_regime` net of observation window.

#### 35a formulation confound partial-out
Script: `scripts/35a_regime_formulation_partial.py`
Output: `outputs/35a_regime_formulation_partial/`

- Nested logistic: M1 (window) -> M2 (window + 10 formulation cols) -> M3 (+ regime)
- M2 -> M3 LR: chi^2=19.3, df=4, **p=0.0007, dAIC=+11.3**
- Reverse direction: multinomial `regime ~ formulation` 5-fold CV
  accuracy = 0.343 vs top-class baseline 0.352 -- **formulation cannot
  predict regime** even at majority-class level.
- Verdict: regime carries information not captured by observation
  window OR formulation descriptors.

#### 35b regime <-> Jacobian principal-direction alignment
Script: `scripts/35b_regime_jacobian_alignment.py`
Output: `outputs/35b_regime_jacobian_alignment/`

- Per-curve local Jacobian (forward-difference, batched -- ~40s for 376 curves)
- v1 sign-aligned right singular vector; abs-cosine distance for silhouette
- Result: silhouette observed -0.106 vs null p95 -0.058, **p=0.48 FAIL**
- Per-regime alignment (mean |v1| vs modal active set, Spearman):
  - R6 = 0.69 (only regime that aligns)
  - R5 = 0.07, R4 = 0.39, R3 = 0.26, R2 = -0.08
- Verdict: regimes do NOT correspond to distinct dominant Jacobian directions.

#### 35c v1 magnitude-profile follow-up
Script: `scripts/35c_regime_v1_magnitude.py`
Output: `outputs/35c_regime_v1_magnitude/`

Wrote after noticing that all regimes had `log_kd`-dominant v1, which
trivially fails direction-based silhouette. Tested an alternative that
should be conceptually closer to what active sets measure:

- Test 1: silhouette on l1-normalized |v1| (magnitude profile, not direction)
  - observed -0.072 vs null mean -0.073 -- essentially **zero signal, p=0.54**
- Test 2: per-curve overlap between greedy 4 and |v1|-top-4
  - mean overlap 2.31 / 4 vs chance baseline 1.78 -- only 30% above chance
- Verdict: **falsifies story 2**. Even the magnitude-profile test is null.
  Continuous latent driver `z -> theta_9` not supported.

#### 36 R6 + R5 regime-conditional fit prototype
Script: `scripts/36_r6_regime_prototype.py` (despite name, generic over regime)
Outputs:
- `outputs/36_r6_regime_prototype/`
- `outputs/36_r6_regime_prototype_v2/` (sanity reproduction)
- `outputs/36_r5_regime_prototype/`

Cheapest possible viability test for story 3 (regime-conditional architecture):
fix the 5 non-modal params at the regime's LOO median, fit only the 4
modal params per curve. Compare to (a) oracle 9, (b) per-curve greedy 4,
(d) zero-shot all-9-fixed.

| Regime | n   | (a) oracle9 median | (b) greedy4 median | **(c) regime-cond4 median** | (d) zero-shot | criteria pass |
|--------|-----|--------------------|--------------------|-----------------------------|---------------|---------------|
| R6     | 146 | 0.9980             | 0.9958             | **0.9899**                  | 0.629         | 2/3 (miss by 0.001) |
| R5     | 112 | 0.9966             | 0.9946             | **0.9904**                  | 0.554         | **3/3**       |

Both regimes: median R^2 close to oracle, 4 free modal params doing real
per-curve work over zero-shot.

Tail (~10% of each regime, R^2 < 0.9): curves whose own best 4-param
greedy set differs from the regime modal -- they look like "regime
boundary" cases that 33's clustering misassigned (e.g. R5 curves whose
greedy set includes `log_alpha` or `log_tau_burst`, neither in R5 modal).

### Integrated interpretation

The PLGA 9-param mechanism family has:

- **discrete** structure (5 well-populated regimes R2-R6)
- **non-geometric** structure (regimes do not correspond to local linear
  parameter subspaces)
- **non-formulation-determined** structure (formulation cannot predict
  regime above majority-class baseline)
- **kinetic-class-like** structure (regimes predict kinetic flags
  beyond all measured confounders)

The closest physical analog is **reaction mechanism class** (SN1 vs SN2
vs E1 vs E2): the same atoms / params can react in qualitatively
different ways, the difference is discrete and nonlinear, and the
mechanism class predicts behavior beyond just "what atoms are present".

The active-set diagnostic (29) and the regime structure (33) are
**algebraic** descriptions of this mechanism heterogeneity. They are
real, but they do not reduce to a local-linear-subspace partition.

### What seems settled

1. The 9-parameter ODE family is globally needed (27a).
2. Per-curve effective degrees of freedom are ~2-4 (29, 30).
3. The reason is **discrete mechanism heterogeneity**, not pairwise
   coupling (32) or continuous latent driver (35b, 35c).
4. There are 5 well-populated mechanism regimes (33), each with a
   physically readable modal active set, separable from observation
   window (34) and from formulation descriptors (35a).
5. Regime-conditional parameterization is **architecturally viable**
   on the two largest regimes (36).

### What is still unresolved

1. **Tail strategy** for the ~10% per-regime fit failures (regime
   boundary curves). Three candidate fixes:
   - soft prior (Gaussian on non-modal, replacing fixed-median),
   - sub-clustering within each regime,
   - fallback to full 9-param for low-confidence curves.

2. **R2, R3, R4 prototype** not yet run. R4 has 65 curves (probably
   workable); R2 (28) and R3 (25) borderline by sample size.

3. **Shape-feature regime classifier** for deployment, since at
   inference time we don't have the full-fit theta to assign regimes.
   35a confirmed formulation descriptors do NOT work as classifier
   input; curve shape features must.

4. **Mixture-NPE architecture** end-to-end design: one regime-aware NPE
   conditioned on (curve, regime_probs), or true mixture-of-NPEs
   averaged by regime probs. Trade-off is sample efficiency vs
   compositionality.

### Recommended next step

In dependency order:

1. Extend 36 to R4 (cheapest remaining regime test).
2. Replace fixed-median with N(median, sigma) soft prior, re-run on R6
   and R5; if tail R^2 lifts above 0.9 for the 10% boundary curves,
   the soft-prior design is justified.
3. Design and prototype the shape-feature regime classifier
   (descriptive features: early slope, t50, plateau, burst presence).
4. If 1-3 all work, write the mixture-NPE plan and start NPE training.

### Caveats

- All five-step partial-out tests (34, 35a) only address measured
  confounders. There could be unmeasured confounders that the regime
  label is tracking instead of mechanism heterogeneity.
- 35b/35c falsified the local-linear-subspace story but did not rule
  out higher-order or nonlinear geometric structure. Regimes might still
  correspond to e.g. a polynomial manifold; this was not tested.
- 36's LOO median is a hard constraint. The true test of the
  regime-conditional architecture is with soft priors (item 2 above).
- The "10% tail" observation depends on the specific 6-regime cut.
  Different regime granularities will redistribute the tail.
