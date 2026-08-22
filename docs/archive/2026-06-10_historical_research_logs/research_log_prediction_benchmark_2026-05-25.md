## Research Log - Prediction-benchmark branch

Date: `2026-05-25`

Continues from
[`research_log_regime_conditional_2026-05-25.md`](research_log_regime_conditional_2026-05-25.md).

The regime-conditional branch left two unanswered questions:

1. Given the diagnostic insights (5 regimes, non-geometric, kinetic-class-like),
   what can we actually **predict** for a new formulation?
2. Is the regime structure useful at deployment time, or only for analysis?

The prediction-benchmark branch (`37` -> `38d`) answers these and
**overturns the working assumption** that a physics-informed end-to-end
architecture is the right next move.

### Work completed

#### 37 - end-to-end formulation -> theta -> curve (MLP)
- Trained an MLP from 10 cross321 formulation features to theta (with
  sigmoid-squash to prior bounds). Theta-loss training (curve-loss
  end-to-end was infeasibly slow through `torchdiffeq` adaptive ODE on
  CPU). Evaluated by simulating predicted theta and computing curve R^2.
- Single 75/25 stratified split, seed=0.
- Result: median test R^2 = 0.835, but **18% catastrophic failures**
  (R^2 < 0). Beats the train-median baseline by +0.21.
- First real prediction result. Formulation has signal, but not enough.

#### 37c - hybrid (formulation + early observations -> curve)
- Added Q at fixed early times {1, 3, 5, 7} days (interpolated) to the
  MLP input.
- Same hyperparameters; same train/test split.
- Result: median full-curve R^2 = 0.914, p10 R^2 = +0.35,
  **catastrophic failure rate dropped from 18% to 4%**.
- The user's intuition was right: adding ~1 week of observations rescues
  the formulation-only failures. The deployable mode is few-shot, not
  pure formulation->curve.

#### 37d - CV generalization audit
- Re-ran 37c under three CV schemes (random 5-fold, GroupKFold by
  drug, GroupKFold by polymer) to test whether 0.91 was a single-split
  artifact or reflected true generalization.
- cross321 audit: 218 curves from 64 unique drugs and 50 unique
  polymers. Random splits leak: every test curve has training curves
  with the same drug or polymer.
- Result (median R^2 / 5 folds):
  - random_5fold      = 0.904 (37c's 0.914 is robust under reshuffling)
  - group_by_drug     = 0.835 (degraded -0.07)
  - group_by_polymer  = 0.774 (degraded -0.13)
- frac >= 0 stays at 92-96% across schemes (model rarely fails
  catastrophically).
- Verdict at this point: 0.91 is the "interpolation within known
  chemistry" number; the "novel chemistry" number is 0.77-0.84.

#### 38 - prediction baselines
- Compared the 37c MLP against three legitimate baselines on the same
  random 5-fold split:
  - B1: classical fit-then-extrapolate (CFE) -- 9-param ODE bounded
    least-squares on observations at t <= 7d, simulate the rest. The
    standard PLGA-literature workflow with no ML.
  - B2: RandomForest -> theta -> simulate. Same inputs as 37c.
  - B3: RandomForest -> Q directly (no ODE in the loop).
- Pooled median R^2:
  - **B2 RF -> theta : 0.957**  (+0.05 over MLP)
  - **B3 RF -> Q     : 0.921**  (+0.02 over MLP, no ODE used)
  -   37c MLP       : 0.904
  -   B1 CFE         : 0.783
- Catastrophic failure rate:
  - RF -> Q: 0.9%  /  RF -> theta: 2.3%  /  MLP: 4.1%  /  CFE: 8.6%
- Win rate per-curve: RF -> theta beats MLP on 67% of curves; the
  advantage is consistent across all 5 folds, not driven by one lucky
  split.
- **This overturned the working architectural claim**. The
  physics-informed end-to-end MLP is neither the most accurate nor
  the most robust of the tested methods. RF -> theta is consistently
  better; RF directly on Q (skipping the ODE) is also better than MLP.

#### 38d - baselines under group-by-drug / group-by-polymer
- Critical follow-up: does RF's advantage hold out-of-distribution?
- All 4 methods on all 3 CV schemes, same fold structure, same
  hyperparameters.
- Median R^2:
  - RF -> theta: random 0.957 -> drug 0.910 -> polymer 0.928
  - RF -> Q    : random 0.921 -> drug 0.877 -> polymer 0.884
  - 37c MLP   : random 0.904 -> drug 0.835 -> polymer 0.774
  - CFE        : random 0.783 -> drug 0.780 -> polymer 0.839
- RF maintains its lead on every scheme. In fact:
  - **RF -> theta vs MLP gap WIDENS** on polymer (+0.15 vs +0.05)
  - **RF -> theta polymer-drop is only -0.029** vs MLP's -0.130
- CFE is unaffected by chemistry splits because it uses no training
  data; its ~0.78-0.84 is the lower-bound "no ML" reference.

#### 39 - RF -> theta middle-layer mapping audit
- Script: `scripts/39_rf_theta_mapping_audit.py`
- Purpose: pull out the strongest middle layer (`formulation + Q(1,3,5,7)
  -> RF -> theta`) and ask which inputs predict which ODE parameters.
- Outputs: `outputs/39_rf_theta_mapping_audit/`
- Main finding:
  - RF relies mostly on early release shape, not formulation alone.
  - Average normalized feature-group importance:
    - early Q: ~0.72-0.74
    - drug descriptors: ~0.17-0.19
    - polymer descriptors: ~0.03-0.06
    - process/formulation descriptors: ~0.04-0.05
- Important negative finding:
  - Per-parameter theta R^2 is low under OOD splits:
    - random median across params: +0.216
    - group-by-drug median across params: +0.036
    - group-by-polymer median across params: -0.023
  - This is not a failure of curve prediction. The same RF->theta model
    still decodes to high curve R^2 in 38d.
- Interpretation:
  - The RF is not recovering a unique "true theta".
  - It is finding a functionally decodable theta: a parameter vector that
    may differ from the oracle least-squares theta, but still yields the
    correct release curve after ODE simulation.
  - Therefore theta R^2 should be treated as a non-identifiability
    diagnostic, not as the optimization target for the next method.

#### 40 - Mechanistic feasibility selector
- Script: `scripts/40_theta_candidate_feasibility.py`
- Purpose: test the user's "mechanistic law / sieve" idea as a first
  executable criterion. The script generates a family of candidate theta
  explanations, then selects one using only train-fold data plus the test
  curve's early release points.
- Candidate sources:
  - RF point theta.
  - RF tree-level theta samples.
  - Descriptor-neighborhood oracle theta from train folds.
  - Local active-subset refinements using only Q(1,3,5,7).
- Selection score:
  - early RMSE
  - active parameter count
  - ODE evaluation cost
  - descriptor-neighborhood mismatch
  - noise stability
  - prior-boundary hits
- Full-curve release is used only after selection for evaluation.
- Full run (`n=259`, 5-fold, three CV schemes):

```text
Method            random   by_drug  by_polymer
B1 CFE            0.8605   0.8605   0.8605
B2 RF->theta      0.9512   0.9124   0.9140
B3 RF->Q          0.8978   0.8599   0.8644
40 feasibility    0.9350   0.9248   0.9251
E oracle          0.9976   0.9976   0.9976
```

- Cost / interpretability:

```text
40 feasibility median active params:
  random            3
  group_by_drug     3
  group_by_polymer  3

40 feasibility median nfev:
  random            8
  group_by_drug     8
  group_by_polymer  8
```

- Interpretation:
  - 40 does not replace RF->theta as the strongest in-distribution
    predictor: random median is lower (0.9350 vs 0.9512 in this n=259
    run).
  - It slightly improves the OOD medians over RF->theta while compressing
    the explanation to ~3 active parameters.
  - Therefore 40's current contribution is not "better predictor than RF".
    It is "early-only feasible-theta selection can preserve OOD accuracy
    with a low-dimensional mechanism explanation".

#### 41 - Theta target-scaling / tree ensemble enhancement audit
- Script: `scripts/41_theta_target_scaling_audit.py`
- Purpose: answer whether the RF->theta `0.957` random median from 38d
  can be improved without moving to a neural model.
- Key idea:
  - Multi-output tree regressors split on summed target MSE.
  - The 9 ODE parameters have different scales and variances.
  - A parameter-friendly variant standardizes each theta target on the
    train fold before fitting, then inverse-transforms predictions before
    ODE decoding.
- Models tested:
  - `RF_raw`: exact 38d baseline.
  - `RF_ztheta`: RF trained on standardized theta targets.
  - `RF_ztheta_leaf2`: standardized theta + min leaf 2.
  - `ET_raw`: ExtraTrees on raw theta.
  - `ET_ztheta`, `ET_ztheta_leaf2`.
  - `RF_ET_zavg`: average of RF_ztheta and ET_ztheta predictions.
  - `early_select`: choose among RF_ztheta / ET_ztheta / RF_ET_zavg by
    early-window ODE RMSE only.
- Full run (`n=218`, same fold protocol as 38d):

```text
Best median R^2 by scheme:
  random            ET_raw              0.9630  (RF_raw 0.9570)
  group_by_drug     RF_ztheta_leaf2     0.9219  (RF_raw 0.9102)
  group_by_polymer  early_select        0.9436  (RF_raw 0.9281)

Strong single-method candidate:
  ET_raw            0.9630 random / 0.9194 drug / 0.9284 polymer
  RF_ET_zavg        0.9607 random / 0.9163 drug / 0.9409 polymer
```

- Interpretation:
  - Yes, the 0.957 RF->theta result can be improved modestly with pure
    classical ML.
  - ExtraTrees is the cleanest replacement if we want a single simple
    model.
  - Target-standardized theta helps most under drug/polymer OOD, which
    supports the idea that the middle layer should be parameter-aware.
  - The strongest cross-polymer number comes from early-only model
    selection, not from full-curve leakage.

#### 42 - Cleaned internal181 tree-ensemble theta audit
- Script: `scripts/42_internal181_tree_theta_audit.py`
- Purpose: test whether the 321/cross-DOI tree-ensemble theta route
  transfers back to the internal 181 PLGA dataset after 321-style curve
  cleaning.
- Cleaning:
  - group duplicate `(Experimental_index, Time)` rows and average Release
  - clip Release to `[0, 1]`
  - truncate observations to `t <= 90 d`
  - require at least 3 observations
  - align curves with the internal181 full-fit oracle theta bank from 29
  - keep oracle fits with `full_r2 >= 0.95`
- Input:
  - 13 native internal descriptors from `PLGA_CONTINUOUS_COLS`
  - early release points `Q(1,3,5,7)`
- Full run:

```text
n_curves = 169
unique drugs = 22
unique polymers = 8

Best median R^2 by scheme:
  random            early_select   0.9705
  group_by_drug     early_select   0.9380
  group_by_polymer  early_select   0.9224
```

- Single-method anchors:

```text
RF_raw:
  random            0.9628
  group_by_drug     0.9160
  group_by_polymer  0.9078

RF_ET_zavg:
  random            0.9652
  group_by_drug     0.9349
  group_by_polymer  0.9192
```

- Interpretation:
  - The method transfers to internal181 strongly.
  - Early-window model selection is again useful, especially under
    held-out drug/polymer splits.
  - The result supports a unified story: the robust object is not RF
    specifically, but `formulation + early Q -> tree ensemble -> theta
    -> ODE`.

#### 43 - Mixed-source tree-ensemble theta audit
- Script: `scripts/43_mixed_source_tree_theta_audit.py`
- Leakage audit: `outputs/43_mixed_source_tree_theta_audit/leakage_audit.txt`
- Purpose: test whether mixing internal181 and cross321 improves target
  source generalization, or whether source/domain mismatch hurts.
- Protocol:
  - Use only the 7 descriptors shared by both datasets:
    `LA/GA`, `Polymer MW`, initial drug/polymer ratio, DLC, Drug MW,
    Drug TPSA, Drug LogP.
  - Add early release points `Q(1,3,5,7)`.
  - Align units: cross321 polymer MW is converted kDa -> Da and DLC is
    converted % -> fraction.
  - For each target source, compare:
    - `target_only`: train on target-source train fold only.
    - `augmented`: train on target-source train fold plus all curves from
      the other source.
  - Test folds always come from the target source only; this is not pooled
    random CV across both sources.
  - Leakage audit added `augmented_strict`: remove aux-source rows that
    share the held-out drug descriptor or polymer descriptor with the
    target test fold.
- Full run:

```text
internal181 curves : 169
cross321 curves    : 237
features           : 7 shared descriptors + 4 early Q

Best median R^2:

Target cross321:
  random target_only          0.9643
  random augmented            0.9637
  random augmented_strict     0.9637
  drug target_only            0.9360
  drug augmented              0.9466
  drug augmented_strict       0.9466
  polymer target_only         0.9441
  polymer augmented           0.9471
  polymer augmented_strict    0.9426

Target internal181:
  random target_only          0.9728
  random augmented            0.9720
  random augmented_strict     0.9720
  drug target_only            0.9334
  drug augmented              0.9556
  drug augmented_strict       0.9556
  polymer target_only         0.9306
  polymer augmented           0.9548
  polymer augmented_strict    0.9497
```

- Interpretation:
  - Hard leakage check: no target test curve uid appears in training, no
    exact full input vector is shared across sources, and no exact shared
    7-descriptor formulation appears across sources.
  - Cross-source exact drug descriptor overlap is zero.
  - Cross-source exact polymer descriptor overlap exists (11 shared
    polymer descriptor keys), so polymer OOD claims must use
    `augmented_strict`.
  - Mixing does not materially improve random CV.
  - Mixed-source drug OOD gain is robust under strict filtering:
    - cross321 held-out drug: +0.0106
    - internal181 held-out drug: +0.0222
  - Mixed-source polymer OOD is mixed:
    - cross321 held-out polymer: original +0.0030, strict -0.0015
    - internal181 held-out polymer: original +0.0242, strict +0.0191
  - Honest conclusion: source mixing clearly helps held-out drug
    generalization and helps internal181 held-out polymer; cross321
    polymer improvement is not robust under strict aux filtering.

#### 44 - Early observation window ablation
- Outputs: `outputs/44_early_window_ablation/`
- Purpose: test whether `Q(1,3,5,7)` is necessary, or whether fewer
  early observations already give near-final performance.
- Windows tested:
  - `d1`: `Q(1)`
  - `d1_3`: `Q(1,3)`
  - `d1_3_5`: `Q(1,3,5)`
  - `d1_3_5_7`: `Q(1,3,5,7)`
- Datasets:
  - cross321 via script `41`
  - cleaned internal181 via script `42`
- Best median R^2 by window:

```text
cross321
  random:      d1 0.914 | d1_3 0.930 | d1_3_5 0.950 | d1_3_5_7 0.963
  by_drug:     d1 0.765 | d1_3 0.862 | d1_3_5 0.910 | d1_3_5_7 0.922
  by_polymer:  d1 0.792 | d1_3 0.853 | d1_3_5 0.900 | d1_3_5_7 0.944

internal181
  random:      d1 0.952 | d1_3 0.964 | d1_3_5 0.966 | d1_3_5_7 0.971
  by_drug:     d1 0.863 | d1_3 0.905 | d1_3_5 0.928 | d1_3_5_7 0.938
  by_polymer:  d1 0.806 | d1_3 0.870 | d1_3_5 0.915 | d1_3_5_7 0.922
```

- Interpretation:
  - `Q(1)` alone is already informative, especially for random CV, but
    not enough for robust OOD.
  - `Q(1,3,5)` is a strong shorter-window compromise.
  - `Q(7)` still matters, especially for cross321 held-out polymer
    (0.900 -> 0.944).
  - The safest headline remains "one-week observation", but the
    stronger practical message is "performance rises monotonically as
    the early observation window extends from day 1 to day 7".

#### 45 - Input source ablation
- Outputs: `outputs/45_input_source_ablation/`
- Scripts: `41` and `42`, now with `--input-mode`.
- Purpose: test what the high R^2 is actually using:
  formulation descriptors alone, early release alone, or both.
- Modes:
  - `formulation_only`: formulation/descriptors only.
  - `early_only`: `Q(1,3,5,7)` only.
  - `formulation_plus_early`: original input.
- Best median R^2 by input source:

```text
cross321
  random:      formulation_only 0.909 | early_only 0.959 | formulation_plus_early 0.963
  by_drug:     formulation_only 0.672 | early_only 0.941 | formulation_plus_early 0.922
  by_polymer:  formulation_only 0.666 | early_only 0.949 | formulation_plus_early 0.944

internal181
  random:      formulation_only 0.932 | early_only 0.946 | formulation_plus_early 0.971
  by_drug:     formulation_only 0.661 | early_only 0.932 | formulation_plus_early 0.938
  by_polymer:  formulation_only 0.525 | early_only 0.907 | formulation_plus_early 0.922
```

- Interpretation:
  - The high prediction accuracy is mostly carried by early release
    observations, not formulation descriptors alone.
  - Formulation-only is inadequate under OOD: held-out drug/polymer
    medians collapse to ~0.52-0.67.
  - On internal181, formulation + early release improves over early-only
    in all three schemes, so formulation descriptors are useful when
    they are aligned with the target dataset.
  - On cross321 OOD, early-only is slightly better than formulation +
    early in this audit. This means we should not over-claim that
    formulation descriptors always improve OOD.
  - Correct paper framing: early release provides the kinetic
    fingerprint; formulation descriptors condition/regularize the
    theta mapping when the descriptor schema is reliable.

#### 46 - Direct curve RF/ET without theta
- Outputs: `outputs/46_direct_curve_rf_input_ablation/`
- Script: `scripts/46_direct_curve_rf_input_ablation.py`
- Purpose: test whether the theta middle layer is actually useful, or
  whether pure tabular ML can map the same inputs directly to the curve.
- Reconstruction rule:
  - If early Q is an input, direct ML predicts late-grid Q and uses the
    observed early points as anchors.
  - If `formulation_only` is used, direct ML predicts both early and
    late grid Q; no observed early Q is leaked into reconstruction.
- Best direct-curve median R^2 vs best theta-route median R^2:

```text
cross321
  random:
    formulation_only       direct 0.837 | theta 0.909
    early_only             direct 0.907 | theta 0.959
    formulation_plus_early direct 0.925 | theta 0.963
  by_drug:
    formulation_only       direct 0.619 | theta 0.672
    early_only             direct 0.887 | theta 0.941
    formulation_plus_early direct 0.879 | theta 0.922
  by_polymer:
    formulation_only       direct 0.664 | theta 0.666
    early_only             direct 0.885 | theta 0.949
    formulation_plus_early direct 0.893 | theta 0.944

internal181
  random:
    formulation_only       direct 0.866 | theta 0.932
    early_only             direct 0.870 | theta 0.946
    formulation_plus_early direct 0.920 | theta 0.971
  by_drug:
    formulation_only       direct 0.620 | theta 0.661
    early_only             direct 0.852 | theta 0.932
    formulation_plus_early direct 0.886 | theta 0.938
  by_polymer:
    formulation_only       direct 0.467 | theta 0.525
    early_only             direct 0.834 | theta 0.907
    formulation_plus_early direct 0.801 | theta 0.922
```

- Interpretation:
  - Direct RF/ET->Q is strong, but theta->ODE is consistently stronger.
  - The theta middle layer is not merely an unnecessary detour. It acts
    as a curve-shape bottleneck / ODE decoder that improves long-horizon
    reconstruction from early observations.
  - The largest theta advantage appears on internal181 held-out polymer
    with formulation + early release (0.922 vs 0.801, +0.121).
  - The only near-tie is cross321 formulation-only held-out polymer
    (0.666 vs 0.664), where both methods are weak; this does not
    challenge the early-observation theta route.

#### 48 - NC wet external prediction check
- Input: `D:\最终版框架\dataset\Prediction.csv`
- Script: `scripts/48_external_wet_prediction_eval.py`
- Outputs:
  - `outputs/48_external_wet_prediction_eval/`
  - `outputs/48_external_wet_prediction_eval_d1_3/`
- Data shape:
  - 2 wet deployment curves, 15 total rows.
  - `Prediction_1`: OLA-PLGA, 10 observations, `t_max=27d`.
  - `Prediction_2`: SA-PLGA, 5 observations, `t_max=3d`.
  - Wet file lacks `Drug_NHA`, so the model uses the 12 shared
    descriptor columns plus early-release observations.
- Evaluation protocol:
  - Train on cleaned internal181 oracle-theta bank (`n=169`).
  - Do not mix wet curves into CV or training.
  - Evaluate theta->ODE and direct RF/ET->Q on wet curves.
  - Also run a quick full-curve oracle fit as a diagnostic only.
- `T=0.25/T=1.0` early-window result:

```text
quick oracle fit:
  Prediction_1 OLA-PLGA  R^2=1.000  RMSE=0.0005
  Prediction_2 SA-PLGA   R^2=0.829  RMSE=0.1485

selected theta->ODE:
  Prediction_1 OLA-PLGA  R^2=0.338  RMSE=0.314
  Prediction_2 SA-PLGA   R^2=0.609  RMSE=0.225

median across 2 wet curves:
  best theta family    ~0.48
  early_select theta   ~0.47
  best direct RF/ET->Q ~0.22
```

- `Q(1,3)` early-window result:

```text
selected theta->ODE:
  Prediction_1 OLA-PLGA  R^2=0.248  RMSE=0.335
  Prediction_2 SA-PLGA   R^2=0.657  RMSE=0.210

median across 2 wet curves:
  best theta family    ~0.46
  early_select theta   ~0.45
  best direct RF/ET->Q <= 0
```

- Interpretation:
  - This is not a successful external wet prediction benchmark yet.
  - The wet curves are a stress test: `Prediction_1` has a delayed sharp
    transition after day 15, while `Prediction_2` has an extreme early
    burst by day 1.
  - The quick oracle fit says the ODE can express `Prediction_1`
    almost perfectly if the full curve is given, so the failure is mainly
    external theta selection/generalization, not ODE expressivity.
  - `Prediction_2` is harder even for quick oracle fitting, suggesting
    the very fast-burst regime is under-covered by the current prior or
    training bank.
  - Direct RF/ET->Q remains worse than theta->ODE here, but the absolute
    wet performance is too low to use as a headline claim.

#### 49 - Regime-gated theta wet check
- Motivation: the legacy `D:\最终版框架\ME LGBM.py` script performs better
  on the two NC wet curves because it uses a fast/slow gate and
  branch-specific formula parameters. Script `49` tests the same idea
  inside the current theta->ODE stack.
- Script: `scripts/49_regime_gated_theta_wet_eval.py`
- Outputs:
  - `outputs/49_regime_gated_theta_wet_eval/`
  - `outputs/49_regime_gated_theta_wet_eval_d1_3/`
- Method:
  - Estimate `t50` on train curves only.
  - Define fast/slow train regimes by median train `t50` (`12.917d`).
  - Train an ExtraTrees gate from shared descriptors + early Q to
    `p_fast`.
  - Train branch-specific RF/ET theta mappers.
  - Decode through the existing PLGABiphasic ODE.
  - Optional local refinement uses only wet early observations, not the
    full wet curve.
- `T=0.25/T=1.0` result:

```text
gate:
  Prediction_1 OLA-PLGA  p_fast=0.239
  Prediction_2 SA-PLGA   p_fast=0.911

best fixed method:
  ET_regime_soft median R^2 = 0.557

per-curve diagnostic best:
  Prediction_1 OLA-PLGA  R^2=0.683
  Prediction_2 SA-PLGA   R^2=0.822
```

- `Q(1,3)` result:

```text
gate:
  Prediction_1 OLA-PLGA  p_fast=0.174
  Prediction_2 SA-PLGA   p_fast=0.926

fixed method: RF_ET_regime_hard_avg_early_refined
  Prediction_1 OLA-PLGA  R^2=0.812  RMSE=0.167
  Prediction_2 SA-PLGA   R^2=0.819  RMSE=0.153
  median R^2             0.816
```

- Interpretation:
  - Borrowing the fast/slow gate idea helps the external wet curves.
  - The biggest gain comes from combining three pieces:
    regime gate + branch-specific theta mapper + early-only local
    refinement.
  - The `Q(1,3)` result is not the same task as the NC paper's
    sub-day/day-1 prospective prediction; it is closer to our
    partial-observation trajectory reconstruction framing.
  - This is the strongest evidence so far that the next version should
    be regime-aware theta selection, not a larger black-box network.

### Integrated finding

For PLGA release prediction on the cross321 scale (218 curves, 14-d
input), standard tabular ML (Random Forest with default hyperparams) is:

- More accurate (median R^2 ~0.93-0.96 vs ~0.78-0.91 for the
  physics-informed MLP),
- More robust (catastrophic failure rate 2-3% vs 4-13%),
- Better out-of-distribution (smaller polymer-drop and drug-drop),
- And does not require the ODE in the loop.

The "physics-informed end-to-end" architectural narrative is therefore
**falsified for this prediction task at this data scale**. RF wins
because n=218, dim=14 is the textbook tabular regime where decision-tree
ensembles outperform deep networks.

The deeper middle-layer finding is stronger than "RF predicts theta":
RF->theta succeeds functionally even though per-parameter theta recovery
is weak. This supports the non-identifiability story from 29/30: many
theta vectors can decode to similar release curves. The next criterion
should optimize for decodable, stable, physically feasible theta families,
not for matching one arbitrary oracle theta vector.

The 40 selector validates this direction in first-pass form: it keeps
OOD median R^2 around 0.925 with only ~3 active parameters. Its weakness
is tail robustness; RF->theta remains the cleaner prediction baseline.

The 41 audit improves the prediction baseline itself: RF->theta is no
longer the absolute tree-ensemble ceiling. ExtraTrees reaches 0.963
random median, and a simple RF/ET or early-window selector improves the
held-out polymer median to ~0.94.

The 42 internal181 audit shows the same route is not cross321-specific.
On cleaned internal181, early-select tree ensemble -> theta reaches
0.971 random / 0.938 held-out drug / 0.922 held-out polymer.

The 43 mixed-source audit adds useful but nuanced generalization evidence.
Pooling internal181 and cross321 helps held-out drug OOD on both target
sources, and helps internal181 held-out polymer even after strict aux
filtering. The cross321 polymer gain does not survive strict filtering,
so it should not be over-claimed.

The 45 input-source audit clarifies what the model is really exploiting.
It is not a formulation-only predictor. One-week early release is the
dominant signal; formulation descriptors provide dataset-dependent
conditioning, strongest on internal181 and mixed-source settings.

The 46 direct-curve audit clarifies why the theta middle layer is still
valuable. Direct RF/ET can predict Q grids from the same inputs, but it
loses to the theta->ODE route across nearly every dataset/split/input
combination. The middle layer is therefore a useful inductive bottleneck:
early observations identify a kinetic fingerprint, the tree ensemble maps
that fingerprint into an ODE-decodable theta family, and the simulator
regularizes the full curve.

The 48 wet check should be treated as an external stress test, not a
paper headline. It confirms that theta->ODE remains better than direct
RF/ET on the two NC wet curves, but the absolute external R^2 is modest.
This exposes the next real gap: external wet deployment needs a stronger
candidate-theta feasibility/refinement step or better coverage of sharp
burst/delayed-transition regimes.

The 49 wet follow-up confirms the likely fix: regime-aware theta
selection. Fast/slow gating separates OLA-PLGA and SA-PLGA correctly,
and with day-1/day-3 early observations plus early-only local refinement,
both wet curves reach R^2 ~0.81. This does not replace the main CV
benchmark, but it gives a concrete path for external wet deployment.

The 50 CV audit tests whether that 49 wet fix generalizes. The honest
answer is nuanced. Regime gating alone is not a stable upgrade: with no
early refinement, global RF/ET theta models still usually win. The
actual upgrade is a light early-observation refinement step, and regime
gating helps mainly in OOD settings. With 5 least-squares evaluations,
cross321 held-out drug improves from global RF/ET ztheta median R^2
0.927 to regime-hard-refined 0.943, and cross321 held-out polymer
improves from 0.935 to 0.946. On internal181 held-out polymer, regime-
hard-refined improves from global 0.918 to 0.929. On internal181 held-
out drug, global early refinement is better than regime refinement.

Interpretation: do not claim "regime gating universally improves
prediction." Claim instead that the theta bottleneck can be upgraded
by **test-time early-Q refinement**, and that regime-aware branches help
selected OOD stress cases, especially held-out polymer and the external
wet curves.

### What this changes for the paper story

**Cannot claim** any of:
- Physics-informed end-to-end model is necessary for prediction.
- ODE in the loop helps accuracy.
- A novel architecture beats baselines.
- Formulation descriptors alone explain the high OOD prediction numbers.

**Can claim** all of:
- ~1 week early release observations plus formulation conditioning and
  standard tabular ML achieve R^2 ~0.91-0.96 (in CV including drug- and
  polymer-OOD), improving on classical fit-then-extrapolate (CFE) by
  0.10-0.15.
- Early release is the dominant predictive signal; formulation
  descriptors alone are not sufficient for OOD prediction.
- The theta middle layer improves over direct curve RF/ET, so it is a
  useful ODE-decodable bottleneck rather than an arbitrary parameter
  regression target.
- A light early-observation refinement of predicted theta can improve
  OOD performance without using the full test curve; regime-aware
  branches help in selected held-out-polymer / wet stress settings.
- Out-of-distribution generalization holds: cross-drug 0.91,
  cross-polymer 0.93 (RF -> theta).
- Catastrophic failure rate <3% across all CV schemes.
- The 9-parameter ODE remains useful as a **mechanism interpretation
  tool** (the 32-36 branch), not as a prediction tool.

### What is still unresolved

1. **Regime-conditional RF**: would training one RF per regime improve
   over the global RF? If yes, the mechanism branch (32-36) would
   connect mechanically to the prediction branch via regime-conditional
   prediction. Untested.
2. **Curve-loss MLP**: end-to-end backprop through the ODE was
   infeasibly slow on CPU. With GPU + adjoint solver, could MLP catch
   up to RF? Speculative.
3. **Better chemistry features**: replacing Drug MW + TPSA + LogP with
   Mordred / RDKit fingerprints could push RF higher. Not attempted.
4. **Active learning**: which time point would a researcher
   measure next to most reduce predictive uncertainty? RF gives ensemble
   variance for free; calibration and AL untested.
5. **Internal181 transfer**: internal181 has no formulation labels so
   we cannot test transfer of the cross321-trained model.
6. **Mechanistic feasibility criterion**: theta R^2 is the wrong target
   for the next method. We need a criterion that ranks candidate theta
   explanations by curve decoding quality, physical validity,
   descriptor consistency, stability under perturbation, and computation
   cost.
7. **40 as prediction replacement**: not yet. The full run shows 40 is
   a low-dimensional OOD-feasible explanation selector, not a universal
   R^2 upgrade over RF->theta.
8. **Default RF as final ceiling**: falsified by 41. ExtraTrees and
   target-aware tree variants can lift the 38d RF->theta baseline.
9. **321-only artifact**: falsified by 42. The same tree-ensemble theta
   bottleneck works on cleaned internal181.
10. **Dataset mixing is useless**: partly falsified by 43. Mixed-source
    training robustly improves held-out drug medians on both target
    sources and internal181 held-out polymer. It does not robustly improve
    cross321 held-out polymer under strict aux filtering.

### Recommended next session

Pick one of (1)-(4) above as the new differentiator, ideally (1) since
it ties the two completed branches together. Or stop the prediction
side here, write up RF as the prediction baseline + 32-36 as the
mechanism contribution, and submit.

If continuing method development, the strongest next move is not a
larger network. It is a candidate-theta feasibility model: generate many
theta candidates per curve/formulation, score them by a mechanistic
feasibility criterion, and train a fast model to recover the most stable,
low-cost, decodable parameter family.

After 40, the next cheap prediction-side question is whether the RF->theta
baseline itself can be raised by a classical ensemble audit (RF
hyperparameters, ExtraTrees, and RF/ET theta averaging) before trying a
larger neural model.

41 answered yes. The next model-development move should tune the
tree-ensemble family lightly, not jump straight to KAN.

42 answered the internal-transfer question: use 181-native descriptors
plus early Q, after 321-style curve cleaning and oracle-bank alignment.
Do not force the 321 descriptor schema onto 181 for the first comparison.

43 answered the mixing question with an important caveat: for a fair
shared-feature comparison, mixing 181 and 321 robustly helps held-out
drug OOD more than random CV. Polymer OOD needs strict reporting because
some polymer descriptor overlap exists across sources.

44 answered the early-window question: day 7 is not arbitrary. Shorter
windows already work, but the full 1/3/5/7-day window is best across all
dataset x split combinations, with the largest added value on polymer
OOD.

50 answered the regime-gated-CV question: gating alone is not enough.
The reliable improvement comes from early-only local theta refinement;
regime-gated refinement gives the strongest gains in cross321 held-out
drug/polymer and internal181 held-out polymer, but not every split.

### Files to inspect

- `scripts/37_formulation_to_curve_e2e.py`
- `scripts/37c_hybrid_formulation_plus_early_obs.py`
- `scripts/37d_cv_generalization.py`
- `scripts/38_prediction_baselines.py`
- `scripts/38d_baselines_groupkfold.py`
- `scripts/39_rf_theta_mapping_audit.py`
- `scripts/40_theta_candidate_feasibility.py`
- `scripts/41_theta_target_scaling_audit.py`
- `scripts/42_internal181_tree_theta_audit.py`
- `scripts/43_mixed_source_tree_theta_audit.py`
- `scripts/50_regime_gated_theta_cv.py`
- `outputs/37_formulation_to_curve_e2e/summary.txt`
- `outputs/37c_hybrid_formulation_plus_early_obs/summary.txt`
- `outputs/37d_cv_generalization/summary.txt`
- `outputs/38_prediction_baselines/summary.txt`
- `outputs/38d_baselines_groupkfold/summary.txt`
- `outputs/38d_baselines_groupkfold/comparison_distribution.png`
- `outputs/38d_baselines_groupkfold/degradation_table.png`
- `outputs/39_rf_theta_mapping_audit/summary.txt`
- `outputs/39_rf_theta_mapping_audit/importance_heatmap_random_5fold.png`
- `outputs/40_theta_candidate_feasibility/summary.txt`
- `outputs/40_theta_candidate_feasibility/scheme_summary.csv`
- `outputs/41_theta_target_scaling_audit/summary.txt`
- `outputs/41_theta_target_scaling_audit/scheme_method_summary.csv`
- `outputs/42_internal181_tree_theta_audit/summary.txt`
- `outputs/42_internal181_tree_theta_audit/scheme_method_summary.csv`
- `outputs/43_mixed_source_tree_theta_audit/summary.txt`
- `outputs/43_mixed_source_tree_theta_audit/summary.csv`
- `outputs/44_early_window_ablation/early_window_summary.csv`
- `outputs/50_regime_gated_theta_cv_refine5/summary.txt`
- `outputs/50_regime_gated_theta_cv_no_refine/summary.txt`

### Lessons (process)

- "Physics-informed" is not a free lunch. At n=218, dim=14, naive RF
  wins. Test the obvious baseline early; we left it to 38 and could
  have saved the entire MLP architectural narrative.
- The user's identifiability concern (theta is non-unique) was
  sharpened empirically by the RF benchmark and 39: RF can decode curves
  well while per-parameter theta R^2 remains weak. Matching the oracle
  theta is not the right objective; finding a feasible decodable theta
  family is.
- CV scheme choice matters more than model choice for honest numbers.
  Random 5-fold inflated MLP by leaking same-drug curves. group-by-
  drug / by-polymer revealed the real degradation pattern.
