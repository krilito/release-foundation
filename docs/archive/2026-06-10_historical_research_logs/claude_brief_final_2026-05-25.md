## PLGA project closing brief for the next Claude

Date: `2026-05-25`. Supersedes
[`claude_brief_active_set_2026-05-25.md`](claude_brief_active_set_2026-05-25.md)
and
[`claude_brief_regime_conditional_2026-05-25.md`](claude_brief_regime_conditional_2026-05-25.md).

Two research branches have been worked through and have reached
defensible stopping points:

1. **Mechanism branch** (32 -> 36): regime structure of the 9-param
   ODE. Documented in
   [`research_log_regime_conditional_2026-05-25.md`](research_log_regime_conditional_2026-05-25.md).
2. **Prediction branch** (37 -> 38d): formulation -> curve prediction
   under proper CV. Documented in
   [`research_log_prediction_benchmark_2026-05-25.md`](research_log_prediction_benchmark_2026-05-25.md).
3. **Middle-layer audit** (`39`): RF -> theta interpretation and the
   feasibility-criterion pivot. Documented in
   [`plan_41_mechanistic_feasibility_criterion.md`](plan_41_mechanistic_feasibility_criterion.md).
4. **Feasibility selector** (`40`): first executable candidate-theta
   sieve. Documented in
   [`research_log_prediction_benchmark_2026-05-25.md`](research_log_prediction_benchmark_2026-05-25.md).
5. **Tree-ensemble enhancement** (`41`): tests whether the 38d RF->theta
   ceiling can be raised with parameter-aware classical ML.
6. **Internal + mixed-source audits** (`42`, `43`): checks whether the
   tree-ensemble theta route transfers to internal181 and whether pooling
   181/321 helps OOD.
7. **Early-window ablation** (`44`): checks whether `Q(1,3,5,7)` is
   arbitrary or actually best.

This brief tells the next session what is settled, what is open, and
what NOT to redo.

### What is settled (do not redo or re-question without strong reason)

#### Mechanism side
- 9-parameter ODE family (`PLGABiphasic`) has full-curve expressivity
  on cross321 (`27a` median R^2 = 0.998, 95% of curves > 0.9).
- Per-curve effective dimension is 2-4 (`29`, `30`).
- The 9 parameters do not form a dense pairwise coupling network
  (`32`; only `log_kw ~ Q_max` is consistent across datasets).
- 5 well-populated kinetic regimes exist (`33`, n=6 cluster with R1
  trivially small). Modal active sets are physically readable.
- Regimes are not formulation-determined (`35a` CV acc 0.34 vs majority
  baseline 0.35) and not observation-window artifacts (`34` LR p=7e-4).
- Regimes do NOT correspond to linear parameter subspaces (`35b`, `35c`).
  Local Jacobian principal direction is shared (`log_kd` dominant in
  all regimes); regime structure is non-geometric.
- Best mechanistic analogy: discrete reaction-mechanism class
  (SN1 / SN2 / E1 / E2), not "shared backbone + functional groups"
  and not "continuous latent driver".
- Regime-conditional fit prototype (`36`): for both R5 and R6, fixing
  the 5 non-modal parameters at the regime LOO median and fitting only
  the 4 modal parameters per curve reaches median R^2 ~0.99, within
  0.01 of the 9-param oracle. The architecture is viable at the
  fitting-diagnostic level.

#### Prediction side
- Pure formulation -> theta -> curve (37) gives median R^2 = 0.83 with
  18% catastrophic failures. Formulation alone is insufficient.
- Formulation + 4 early observations (37c) gives median R^2 = 0.91
  with 4% catastrophic failures. This is the deployable form.
- Under stricter CV (`37d`, `38d`):
  - random 5-fold        : 0.90-0.96 depending on method
  - GroupKFold by drug   : 0.84-0.91 (best: RF -> theta)
  - GroupKFold by polymer: 0.77-0.93 (best: RF -> theta)
- **Random Forest beats the physics-informed MLP on every CV scheme**.
  The physics-informed end-to-end architectural narrative is falsified
  for this prediction task at n=218.
- RF -> theta is the best single method (median 0.96 random / 0.91
  drug / 0.93 polymer; catastrophic failure rate 2-3%).
- `39` showed RF->theta does not recover oracle theta well parameter by
  parameter (median theta R^2 across params: +0.216 random, +0.036
  drug, -0.023 polymer), even though the decoded curves are strong.
  This means theta R^2 is not the right objective; the right objective
  is feasible decodability.
- `40` implemented the first feasibility selector. Full run (`n=259`)
  gave median R^2 = 0.935 random / 0.925 drug / 0.925 polymer with
  median active params = 3 and median nfev = 8. It is not a universal
  predictor upgrade over RF->theta, but it is a real low-dimensional
  OOD-feasible mechanism explanation.
- `41` showed the 38d RF->theta number is not the ceiling. ExtraTrees
  reaches 0.963 random median, RF target-standardization reaches 0.922
  held-out-drug median, and early-window model selection reaches 0.944
  held-out-polymer median.
- `42` showed the route transfers to cleaned internal181:
  early-select tree ensemble -> theta reaches 0.971 random / 0.938
  held-out drug / 0.922 held-out polymer.
- `43` showed mixed-source training helps OOD with an important leakage
  caveat. Hard leakage was not found, but cross-source polymer descriptor
  overlap exists. Strict aux filtering preserves held-out-drug gains on
  both sources and internal181 held-out-polymer gain; it does not preserve
  the cross321 held-out-polymer gain.
- `44` showed the 1/3/5/7-day window is genuinely best across dataset x
  split combinations. Shorter windows work, especially `Q(1,3,5)`, but
  day 7 adds the most value for polymer OOD.
- Classical fit-then-extrapolate (`38` baseline B1) gives 0.78-0.84.
  The contribution over CFE is the formulation conditioning, not the
  ODE-in-the-loop.

### What is falsified (do not pursue without first re-falsifying)

- Pairwise coupling network ("free-radical catalysis") between the 9
  parameters: `32` ruled this out.
- Continuous latent driver `z (k << 9) -> theta_9`: `35b`, `35c` ruled
  this out at the local-Jacobian level.
- Formulation-predicted regime classification: `35a` showed CV
  accuracy is below the majority-class baseline.
- Physics-informed end-to-end MLP as a SOTA predictor: `38`, `38d`
  showed RF wins on every CV scheme.
- ODE-in-the-loop as a precondition for accurate prediction: `38`
  showed RF directly on Q (no ODE) also beats MLP+ODE.
- Unique-theta supervision as the next objective: `39` showed the RF
  can decode curves well without matching the oracle theta point. The
  next method should optimize a feasibility criterion over candidate
  theta families, not theta R^2.
- Treating the first feasibility selector as a finished predictor:
  `40` is a promising explanation/compression result, but RF->theta is
  still the cleaner prediction baseline in-distribution.
- Treating default RF as final: `41` falsified that. The prediction
  branch should now say "tree ensemble -> theta", with ExtraTrees /
  target-aware RF as the improved baseline family.
- Treating 181 and 321 as isolated stories: `43` mostly falsified that
  for held-out-drug OOD. For held-out-polymer OOD, report the strict
  filtered results, not the optimistic augmented-only result.

### Defensible paper story (two interlocking contributions)

**A. Engineering / prediction contribution**:
First systematic, properly-CV'd benchmark for PLGA release prediction.
The deployable model is formulation + 1 week observations + Random
Forest. Reports R^2 under random, GroupKFold-by-drug, and
GroupKFold-by-polymer schemes. Out-of-distribution numbers (0.91 cross
drug, 0.93 cross polymer) with catastrophic failure rate <3%. Compares
to classical fit-then-extrapolate (the literature workflow) showing
+0.10-0.15 R^2 improvement attributable to formulation conditioning.

**B. Mechanism / science contribution**:
Systematic diagnostic cascade of the 9-parameter ODE family yielding:
5 discrete kinetic regimes with physically readable active sets;
falsification of pairwise coupling, continuous latent driver, and
formulation-predicted regime hypotheses; identification of the best
physical analog as discrete reaction-mechanism class. Reports the
falsification chain (32 -> 33 -> 34 -> 35a -> 35b -> 35c) and the
regime-conditional fit prototype (36) demonstrating mechanism-level
parameter sharing.

The two contributions are independent on purpose:
A uses no physics for prediction; B uses physics to explain mechanism.
Trying to merge them (e.g. regime-conditional NPE) is a potential
future contribution but was not validated in this work.

The newly clarified bridge is not "make theta prediction more exact".
It is "learn a descriptor-conditioned feasible theta family". Candidate
theta vectors should be ranked by curve decoding, physical validity,
descriptor consistency, stability, and compute cost.

The first execution of that bridge (`40`) says: a low-dimensional
selector can preserve OOD median accuracy around 0.925 with only about
3 active parameters. The remaining weakness is tail robustness and the
random-split median gap to RF->theta.

The follow-up prediction audit (`41`) says: the stronger predictive
baseline is not necessarily RF, but the tree-ensemble theta bottleneck.
The simplest upgraded headline is ET->theta = 0.963 random median, with
OOD medians around 0.92-0.94 depending on the selector.

The internal/mixed audits (`42`, `43`) strengthen this: the same route
works on internal181, and mixed-source training improves held-out-drug
OOD on both datasets under strict leakage checks. The most honest story
is now "shared formulation + early release signals map to an ODE-decodable
theta family across data sources", with polymer OOD stated carefully.

The early-window audit (`44`) keeps the "one-week observation" claim
intact. The paper can also mention a shorter-window tradeoff: `Q(1,3,5)`
is already strong, but `Q(7)` materially improves held-out polymer
generalization.

The input-source audit (`45`) is the guardrail against overclaiming.
The high R^2 is not coming from formulation descriptors alone. With
formulation-only inputs, OOD medians drop to ~0.52-0.67; with `Q(1,3,5,7)`
alone, medians are already ~0.91-0.95. On internal181, formulation +
early improves over early-only, but on cross321 OOD early-only is slightly
better in this audit. Paper wording should be: early release supplies
the kinetic fingerprint; formulation descriptors condition/regularize
the theta mapping when the descriptor schema is reliable.

The direct-curve audit (`46`) says the theta layer is still useful.
Pure RF/ET -> Q from the same inputs is consistently below the
theta->ODE route. The largest gap is internal181 held-out polymer with
formulation + early release: direct 0.801 vs theta 0.922. This supports
the claim that theta is an ODE-decodable bottleneck / shape regularizer,
not merely an unnecessary detour.

The NC wet external check (`48`) is sobering. `Prediction.csv` contains
two wet deployment curves, not the full 181 library. Training on cleaned
internal181 and testing on those two curves gives only modest absolute
performance: early-select theta is ~0.47 median R^2 with `T=0.25/T=1.0`
and ~0.45 with `Q(1,3)`. Direct RF/ET->Q is worse, but this is not a
headline benchmark. The quick oracle fit shows `Prediction_1` is ODE-
expressible if the full curve is given, so the gap is external theta
selection/generalization. `Prediction_2` is an extreme fast-burst curve
and is hard even for quick oracle fitting.

The regime-gated theta follow-up (`49`) partially fixes the wet gap.
Inspired by `ME LGBM.py`, it trains a fast/slow gate from train-curve
`t50`, then trains branch-specific theta mappers. With `T=0.25/T=1.0`,
the best fixed method improves to median R^2 ~0.56. With `Q(1,3)` plus
early-only local refinement, the fixed `RF_ET_regime_hard_avg_early_refined`
method reaches ~0.812 on OLA-PLGA and ~0.819 on SA-PLGA. This is not the
same task as NC's one-day prospective setting, but it strongly supports
the next method direction: regime-aware theta selection/refinement.

The CV follow-up (`50`) gives the guardrail. Regime gating by itself is
not a universal improvement over global RF/ET theta models. The broadly
useful operation is light early-only local refinement of theta; regime-
gated refinement helps most in OOD stress settings. With `Q(1,3,5,7)`
and 5 least-squares evaluations, cross321 held-out drug rises from
global RF/ET ztheta median R^2 0.927 to regime-hard-refined 0.943, and
cross321 held-out polymer rises from 0.935 to 0.946. Internal181 held-
out polymer rises from 0.918 to 0.929. Internal181 held-out drug is the
exception: global refinement beats regime refinement. Paper wording
should be: early-Q refinement is the upgrade; regime-aware branching is
useful for selected OOD/wet cases, not a universal law.

The paper closeout document (`paper_closeout_release_quality_paradigm_2026-05-26.md`)
sets the current writing route. The manuscript should separate two
contributions: (1) an observer-corrected theta->ODE release-quality
surrogate for prediction, and (2) active-set/regime analysis showing why
theta should be interpreted as a feasible mechanism family rather than a
unique recovered truth. The future expansion route is new data and
quality-paradigm validation, not more blind model chasing.

The first new-data collection pass is documented in
`data_collection_inventory_2026-05-26.md`. The actionable expansion
path is: keep the manuscript on PLGA 321 + LAI 181 + NC wet curves, then
use the liposome IVR 271 dataset as the first cross-mechanism test. PLGA
NP 433 is useful formulation-property data but not a release-curve
dataset, so do not mix it into Q(t) prediction as if it were equivalent.

The first concrete cross-mechanism plan is
`plan_52_accelerated_ivr_extension.md`. Start with the
`danielyanes22/accelerated_IVR` repo, reproduce their kinetic-class task,
then upgrade to full-curve forecasting via
`features + early Q -> Weibull alpha/beta -> full curve`. A fair "beat
them" claim is not R^2 vs balanced accuracy; it is same-task reproduction
plus a harder task: continuous kinetic-state prediction and full release
trajectory forecasting under random and group-OOD CV.

The first execution of `52/54` is in
`research_log_accelerated_ivr_2026-05-26.md`. It works, with an important
caveat. Their class-prototype curve baseline is much weaker than
continuous Weibull calibration: at 6h, full-curve median R2 is about
0.914 for early-Weibull versus 0.568 / 0.163 / -0.138 for best class-
prototype under stratified / group-by-API / group-by-method. At 12h,
early-Weibull reaches about 0.956. But RF/ET-initialized theta plus
early refinement collapses to the same result as early-only Weibull
fitting, so the honest claim is early observation -> kinetic state ->
curve, not formulation-only ML magic.

The surrogate-observer audit (`55`) answers the next question: can
cheap formulation/QC proxies replace early release? Not with the current
features. At 6h, early-release gold gives full-curve median R2 about
0.906 across the three schemes; the best no-early-Q proxy is only
0.557 / 0.190 / 0.185 under stratified / held-out API / held-out method.
At 12h, early-release gold is about 0.950, while the best no-early-Q
proxy is 0.727 / 0.248 / -0.094. Proxy + early-Q can help some settings,
but the conclusion is clear: current formulation/QC variables are weak
priors, not replacements for early release. The real expansion target is
richer rapid-QC observers such as spectroscopy, imaging, membrane
rigidity, lamellarity, morphology, or encapsulation-quality signals.

The curve-latent observer diagnostic (`56`) tests the user's "let the
model observe curves and learn the curve manifold" idea in the smallest
non-neural form. Train-fold full curves define PCA/KNN curve priors;
test-fold `Q(1,3,5,7)` retrieves/fits a latent curve; test full curves
are evaluation-only. The result is useful but not a replacement for the
theta route. Best full-curve median R2 is about 0.881-0.886 for cross321
OOD and 0.840-0.859 for internal181 OOD, with future-only medians lower
than full-curve medians. The current theta/ODE route remains stronger
(about 0.943-0.946 cross321 OOD and 0.929-0.939 internal181 OOD in
`50`). Paper framing: release curves do have a reusable empirical shape
manifold, but theta->ODE is still the stronger shape regularizer. The
future foundation-model route should be simulator-augmented masked-curve
pretraining and should be judged on held-out future R2, not only full R2.

The small-MLP curve follow-up (`57`) closes the narrow "maybe a better
MLP is enough" question. It trains KNN, ET, MLP(64,64), MLP(128,64), and
an MLP ensemble from `Q(1,3,5,7)` to late-grid Q. MLP helps in selected
OOD settings: cross321 held-out drug/polymer best full-curve medians are
~0.889/~0.893 with MLP ensemble, and internal181 held-out drug reaches
~0.868 with MLP(64,64). But it is not a universal upgrade: random splits
are still best with ET/KNN, internal181 held-out polymer is still best
with ET, and future-only R2 remains far below the theta->ODE route. So
ordinary small MLP is a useful baseline, not the next main method.

The tree + MLP residual hybrid (`58`) tests a more reasonable neural
variant: RF/ET first predicts late Q, then an MLP learns only the
train-fold OOF residual. This avoids in-sample residual leakage. The
result is conditionally useful but not a universal upgrade. Internal181
held-out polymer improves to ~0.852 full-curve median R2 with RF/ET
residual averaging, and random internal181 improves slightly to ~0.878.
But cross321 is mostly best with base trees or ordinary MLP; residual
correction often lowers future-only R2. Verdict: residual hybrid is a
model-zoo diagnostic, not a replacement for theta->ODE.

The PLGA sub-day window audit (`research_log_plga_subday_window_2026-05-26.md`)
answers the 6h/12h/24h question. `Q(0.25,0.5,1.0)` is much stricter than
`Q(1,3,5,7)` and drops performance: cross321 best medians become 0.894
random / 0.759 held-out drug / 0.777 held-out polymer; internal181 best
medians become 0.940 / 0.860 / 0.836. The caveat is crucial: internal181
has true 0.25/0.5/1.0 day observations, while cross321 usually does not
have exact 6h/12h sampling, so those points are partly interpolated.
Paper wording should be "one-week partial-observation forecasting"; do
not call the main PLGA result "very-early prediction". The data horizon
audit still supports that day 7 is not usually the end: combined PLGA
median t_max is ~30 days and only ~12% of curves have Q7 >= 90% of final
release.

### Open questions for the user (decide before writing)

1. **Tail strategy** for the ~10% per-regime fit failures in `36`:
   (a) accept and route to full-9-param fallback,
   (b) sub-cluster within each regime,
   (c) commit to a soft-prior design (Gaussian on non-modal params).
2. **Whether to attempt the regime-conditional RF experiment** (RF
   trained per regime, compared to global RF). This would mechanically
   connect contributions A and B into a single story. ~30 min compute.
   See "Recommended cheap follow-ups" below.
3. **27a verdict text**: the current "posterior architecture is the only
   remaining bottleneck" line in `outputs/27a_ode_expressivity_audit/
   summary.txt` is now falsified by the full cascade. Update?
4. **internal181 with formulation labels**: do we have any source of
   formulation features for internal181? If yes, the cross-dataset
   generalization test becomes possible. If no, paper has to state
   this limitation.
5. **Feasibility criterion v2**: should the next experiment improve
   tail robustness by changing the selector score, candidate family, or
   train a fast selector over the 40 candidate bank? See `plan_41`.
6. **50 final framing**: whether to promote early-only local theta
   refinement as the main deployable method, with regime gating as an
   OOD/wet stress-test option rather than the default predictor.

### Recommended cheap follow-ups (if user wants more before paper)

In priority order, all under 1 hour compute each:

1. **Regime-conditional RF**: train one RF per regime (using `33`
   regime labels), compare to global RF on the same CV schemes. If
   regime-conditional RF beats global RF, the mechanism work directly
   improves the prediction model.
2. **RF uncertainty calibration**: use RF ensemble variance as a
   prediction interval, check calibration against actual errors. Adds
   a "model knows when it doesn't know" claim to the prediction
   contribution.
3. **Active learning angle**: which next time point should a researcher
   measure to most reduce RF predictive uncertainty? Practical for
   deployment, has ML/chemistry crossover.
4. **Better chemistry features**: replace `Drug MW + TPSA + LogP` with
   Mordred / RDKit descriptors. May push RF further but is
   incremental.
5. **Theta candidate feasibility bank**: generate many candidate theta
   explanations per curve, score them by decodability + physical validity
   + descriptor consistency + stability + cost, then learn a fast selector.
   This is the most direct route to the user's "mechanistic law / sieve"
   idea.
6. **RF->theta enhancement audit**: before trying KAN or another neural
   model, test whether classical tree ensembles can raise the current
   0.95-ish RF->theta random median. Compare RF hyperparameters,
   ExtraTrees, and simple RF/ET averaging under the same three CV schemes.
   This was done in `41`; next step is light tuning of the best tree
   ensemble variants.

### What NOT to do without explicit user approval

- Start any work that requires `sbi` (NPE training). Not installed in
  this environment.
- Re-derive the regime count (currently n=6 in `33`, effectively n=5).
  Changing it invalidates all of `34`, `35a`, `35b`, `35c`, `36`.
- Rewrite the prior or simulator. ADRs are protected; refer to
  `DECISIONS.md`.
- Modify `outputs/27a_*` files. The "wrong" verdict line there should
  only be changed with user sign-off (see open question 3).
- Re-run `33` clustering with different metric or linkage; the
  Jaccard/average-linkage choice is what `34`-`36` depend on.

### Files to inspect on first contact

- `docs/research_log_active_set_2026-05-25.md` (oldest, what set up the
  cascade)
- `docs/research_log_regime_conditional_2026-05-25.md` (mechanism branch,
  32-36)
- `docs/research_log_prediction_benchmark_2026-05-25.md` (prediction
  branch, 37-38d)
- `outputs/36_r6_regime_prototype/summary.txt`
- `outputs/36_r5_regime_prototype/summary.txt`
- `outputs/38_prediction_baselines/summary.txt`
- `outputs/38d_baselines_groupkfold/summary.txt`
- `outputs/38d_baselines_groupkfold/degradation_table.png` (single
  picture summarizing the prediction-side story)
- `outputs/39_rf_theta_mapping_audit/summary.txt`
- `outputs/40_theta_candidate_feasibility/summary.txt`
- `outputs/41_theta_target_scaling_audit/summary.txt`
- `outputs/42_internal181_tree_theta_audit/summary.txt`
- `outputs/43_mixed_source_tree_theta_audit/summary.txt`
- `outputs/44_early_window_ablation/early_window_summary.csv`
- `outputs/45_input_source_ablation/input_source_summary.csv`
- `outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv`
- `outputs/48_external_wet_prediction_eval/summary.txt`
- `outputs/48_external_wet_prediction_eval_d1_3/summary.txt`
- `outputs/49_regime_gated_theta_wet_eval/summary.txt`
- `outputs/49_regime_gated_theta_wet_eval_d1_3/summary.txt`
- `outputs/50_regime_gated_theta_cv_refine5/summary.txt`
- `outputs/50_regime_gated_theta_cv_no_refine/summary.txt`
- `docs/plan_41_mechanistic_feasibility_criterion.md`
- `docs/paper_closeout_release_quality_paradigm_2026-05-26.md`
- `docs/data_collection_inventory_2026-05-26.md`
- `docs/plan_52_accelerated_ivr_extension.md`
- `docs/research_log_accelerated_ivr_2026-05-26.md`
- `docs/research_log_curve_latent_observer_2026-05-26.md`
- `docs/research_log_plga_subday_window_2026-05-26.md`

### Honest assessment of the current state

The original architectural narrative ("physics-informed end-to-end NPE
for PLGA release") did not survive contact with proper baselines. What
did survive is two distinct, defensible contributions that should be
written as two threads in the same paper rather than one synthesized
claim. The current numbers are paper-grade if framed honestly. Further
experiments are optional rather than necessary to close the work.
