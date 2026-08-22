# Release Transfer Baseline Comparison, 2026-06-11

This note summarizes the current baseline readout for the clean no-metadata
transfer probe:

- Run: `outputs/81_release_corpus_time_budget_mlp_none_e20`
- Report: `outputs/82_release_transfer_report_mlp_none_e20/report.md`
- Model: anchored continuous-time MLP decoder, ensemble size 3, CUDA, no metadata
- Claim scope: sparse-observation transfer signal only

Updated rerun with the stronger shape-increment baseline:

- Run: `outputs/81_release_corpus_time_budget_mlp_none_e20_shapeanchored`
- Report: `outputs/82_release_transfer_report_mlp_none_e20_shapeanchored/report.md`
- Added baselines: `ShapePriorIncrementAnchored:{weibull,biexponential,hill}`
- Reason: the previous run could silently miss shape priors when parquet support
  was unavailable; the script now falls back to `all_family_fits.csv`.

Failure and interval diagnostics:

- Script: `scripts/83_release_transfer_failure_calibration.py`
- Output: `outputs/83_release_transfer_failure_calibration_shapeanchored/summary.md`
- Scope: post-processing only; no new model training

Data-regime diagnostics:

- Script: `scripts/84_release_transfer_data_regime_audit.py`
- Output: `outputs/84_release_transfer_data_regime_audit_shapeanchored/summary.md`
- Purpose: separate real future-dynamics forecasting from short, flat, or
  saturated residual target windows

Shape-prior leakage audit:

- Script: `scripts/85_release_shape_prior_leakage_audit.py`
- Output: `outputs/85_release_shape_prior_leakage_audit_shapeanchored/summary.md`
- Result: `0` leakage rows and `0` baseline availability problems

Dynamic-future masked benchmark:

- Script: `scripts/86_release_dynamic_future_benchmark.py`
- Output: `outputs/86_release_dynamic_future_benchmark_shapeanchored/summary.md`
- Purpose: recompute model-vs-baseline only on curve/budget rows with
  `hard_dynamic_future == True`

Data expansion plan:

- Script: `scripts/87_release_transfer_data_expansion_plan.py`
- Output: `outputs/87_release_transfer_data_expansion_plan_shapeanchored/summary.md`
- Purpose: convert negative-case and dynamic-future diagnostics into a concrete
  data collection priority list

Claim package:

- Script: `scripts/88_release_transfer_claim_package.py`
- Output: `outputs/88_release_transfer_claim_package_shapeanchored/report.md`
- Purpose: consolidate scripts `81-87` into a conservative claim ladder for
  reporting

Do not use this as evidence for a foundation model, mechanism learning, LNN
superiority, or calibrated uncertainty.

## Main Read

The next research step is baseline comparison, but the first pass is already
informative:

1. The headline should be a time-window observation-budget result, not a
   fixed point-count result.
2. Aggregate LOSO performance still beats the closest aggregate baseline at
   every measured time budget after adding shape-increment baselines.
3. System-level results are mixed and now look harsher; this is not yet a
   universal transfer win.
4. LNN/CfC is not currently justified as the main story because the MLP decoder
   matches it within noise.
5. Raw ensemble intervals are not calibrated uncertainty. Leave-one-system
   conformal scaling needs very large width inflation and still transfers
   unevenly across systems.
6. The stronger shape-prior baselines are not leaking held-out curves: every
   split/family uses train-only source IDs even though the fit bank contains all
   curve fits.
7. When restricted to true dynamic-future rows, aggregate performance still
   improves over named baselines, but late-budget evidence is PLGA-only and
   non-PLGA dynamic subsets mostly lose to shape-adapted baselines.
8. The next data cycle should not prioritize more IID PLGA. It should fix
   observability in non-PLGA systems and run model ablations on liposome.
9. The current report entry point is the claim package in
   `outputs/88_release_transfer_claim_package_shapeanchored/report.md`.

## Aggregate Time-Budget Result

The strongest aggregate baseline by median RMSE remains `GlobalMeanAnchored` at
all time budgets. Negative delta means the transfer model is better.

| Budget | Model median RMSE | Best aggregate baseline | Baseline median RMSE | Median delta |
|---|---:|---|---:|---:|
| t<=0.25d | 0.203 | GlobalMeanAnchored | 0.256 | -0.019 |
| t<=1d | 0.158 | GlobalMeanAnchored | 0.278 | -0.060 |
| t<=3d | 0.110 | GlobalMeanAnchored | 0.242 | -0.065 |
| t<=7d | 0.078 | GlobalMeanAnchored | 0.152 | -0.033 |
| t<=14d | 0.059 | GlobalMeanAnchored | 0.068 | -0.006 |

The main effect is not metadata lookup. The cleanest headline remains:

> Early release observations carry transferable information across release
> systems; a no-metadata pretrained release model converts a short observation
> window into lower future-curve error than simple anchored priors.

## System-Level Caveat

Against each system's own best baseline, the result is not uniformly positive.

At `t<=14d`, after adding shape-increment baselines, the model clearly beats
the best available baseline for:

- `PLGA`: 0.094 vs 0.141 against the stronger shape baseline
- `golf ball-shaped microsphere`: 0.004 vs 0.011

It is essentially tied or only marginally different for:

- `liposome`: 0.022 vs 0.022
- `starch nanoparticle`: 0.009 vs 0.009
- `DegraPol mesh`: 0.057 vs 0.051

It loses to stronger classical/local baselines on:

- `Alginate microbead`: 0.274 vs 0.215
- `sodium caseinate film`: 0.124 vs 0.014
- `collagen-alginate hydrogel`: 0.032 vs 0.012

Interpretation: the transfer signal is real at aggregate level, but small-N
systems with flat, short, or very family-regular trajectories can be better
served by local early-observation or shape-increment baselines. This should be
reported as a limitation, not hidden.

Script `83` now separates two failure readings:

- `named_best_by_system_budget.csv`: compares against the single best named
  baseline per system/budget. This is the fairer headline diagnostic.
- `failure_by_system_budget.csv`: compares against an oracle envelope that can
  choose the best baseline separately for each curve. This is a harsh stress
  test, not a deployable competitor.

At `t<=14d`, the fair named-best table says the model beats:

- `PLGA`: 0.094 vs 0.141 (`ShapePrior:weibull`)
- `golf ball-shaped microsphere`: 0.004 vs 0.011 (`LastValueCarryForward`)

It is nearly tied with:

- `liposome`: 0.022 vs 0.022 (`GlobalMeanAnchored`)
- `starch nanoparticle`: 0.009 vs 0.009 (`LastValueCarryForward`)
- `DegraPol mesh`: 0.057 vs 0.051 (`ShapePrior:weibull`)

It loses clearly on:

- `sodium caseinate film`: 0.124 vs 0.014
- `Alginate microbead`: 0.274 vs 0.215
- `collagen-alginate hydrogel`: 0.032 vs 0.012

## Interval Calibration Caveat

Raw ensemble intervals should be treated as ranking/diagnostic bands only.
In the LOSO time-budget diagnostic, median raw cov90 is near zero at all
budgets. A leave-one-heldout-system conformal scale inflates interval width by
roughly `23x-123x`, and transferred coverage is still unstable:

| Budget | Median raw cov90 | Median conformal cov90 | Median width inflation |
|---|---:|---:|---:|
| t<=0.25d | 0.007 | 0.906 | 123x |
| t<=1d | 0.000 | 0.818 | 41x |
| t<=3d | 0.000 | 0.343 | 23x |
| t<=7d | 0.005 | 0.632 | 25x |
| t<=14d | 0.005 | 0.688 | 38x |

Interpretation: the current ensemble is useful for point prediction and
observation-budget curves, but not yet useful as calibrated uncertainty. A real
uncertainty claim needs either explicit calibration data, wider residual model,
or a different uncertainty head.

## Data-Regime Caveat

Script `84` shows that `t<=14d` is not equally meaningful across systems. At
the latest budget, every non-PLGA system has `frac_short_target = 1.0`: the
future target window often collapses to one point or a flat tail. This explains
why carry-forward and shape baselines become hard to beat there.

The more honest comparison is the per-system "informative budget" where the
held-out system still has at least three median target points and a majority of
dynamic futures. Under that stricter dynamic-future view:

| System | Informative budget | Best named baseline | Model delta |
|---|---|---|---:|
| PLGA | t<=14d | ShapePrior:weibull | -0.047 |
| liposome | t<=0.25d | GlobalMeanAnchored | +0.009 |
| DegraPol mesh | t<=1d | ShapePriorIncrementAnchored:hill | +0.121 |
| sodium caseinate film | t<=3d | ShapePriorIncrementAnchored:biexponential | +0.205 |
| golf ball-shaped microsphere | t<=3d | ShapePriorIncrementAnchored:hill | +0.183 |
| Alginate microbead | no dynamic-majority budget | ShapePriorIncrementAnchored:biexponential | +0.059 |
| collagen-alginate hydrogel | no dynamic-majority budget | LastValueCarryForward | +0.040 |
| starch nanoparticle | no dynamic-majority budget | LastValueCarryForward | +0.000 |

Negative delta means the transfer model is better. This is a sharper and more
sobering read: the current model's strongest real win is PLGA. Cross-system
transfer exists in aggregate, but dynamic-future transfer outside PLGA is not
yet beating simple shape-adapted baselines.

## Shape-Prior Leakage Audit

The shape-fit bank contains all 751 curve-level family fits, so the baseline
needed an explicit leakage check. Script `85` reconstructs each split from
`split_manifest.csv`, intersects train/test IDs with the fit bank, and checks
that emitted `ShapePrior:*` and `ShapePriorIncrementAnchored:*` rows have
train-only source curves.

Result:

- Leakage rows: `0`
- Shape baseline rows emitted without source curves: `0`
- PLGA LOSO shape source curves: `249` per family
- PLGA held-out fit-bank curves present but not used as source: `502`

This supports using shape baselines as fair train-only competitors in the
current reports.

## Dynamic-Future Masked Benchmark

Script `86` removes curve/budget rows where the post-budget target window is
short or flat. This is the cleanest test of true future-dynamics transfer.

Aggregate dynamic-future rows still favor the model:

| Budget | Dynamic curves | Systems | Model median RMSE | Best named baseline | Baseline RMSE | Delta |
|---|---:|---:|---:|---|---:|---:|
| t<=0.25d | 639 | 6 | 0.227 | GlobalMeanAnchored | 0.288 | -0.060 |
| t<=1d | 553 | 5 | 0.200 | GlobalMean | 0.263 | -0.063 |
| t<=3d | 508 | 5 | 0.167 | PayloadNameMean | 0.249 | -0.082 |
| t<=7d | 427 | 1 | 0.149 | PayloadNameMean | 0.233 | -0.084 |
| t<=14d | 315 | 1 | 0.142 | ShapePrior:weibull | 0.211 | -0.069 |

But the system-level dynamic-future view is more sobering:

- PLGA is the only robust late-budget dynamic win.
- Liposome has dynamic rows early but loses to `ShapePriorIncrementAnchored:hill`
  at the informative budget.
- DegraPol, sodium caseinate, and golf-ball microsphere all lose to
  shape-increment baselines under dynamic-future filtering.
- Alginate and starch have no dynamic rows at their selected informative budget.

So the paper-safe statement is:

> Dynamic-future transfer is proven for PLGA and visible in aggregate, but
> non-PLGA dynamic transfer is not yet beating simple shape-adapted baselines.

## Data Expansion Plan

Script `87` converts the diagnostics into collection priorities:

| Priority | Systems | Reason |
|---|---|---|
| P0 collect before claim | Alginate microbead, starch nanoparticle, collagen-alginate hydrogel | dynamic transfer is not evaluable or dynamic subset is too small |
| P1 high-value expansion | sodium caseinate film, golf ball-shaped microsphere, DegraPol mesh | dynamic rows exist but shape-increment baselines beat the model |
| P2 model ablation / targeted follow-up | liposome | enough dynamic rows exist; shape prior still wins, so this is a modeling testbed |
| P3 monitor / secondary | PLGA | model already wins; use for source-domain backbone, OOD validation, and calibration sandbox |

Minimum next data target:

- For weak non-PLGA systems: at least `30-40` total curves per system.
- At least `8` timepoints per curve.
- At least `3` post-context target points after the chosen early-observation
  budget.
- Avoid single-point tails; preserve dense early windows plus a late dynamic
  tail.

This makes the next experimental goal concrete: collect enough non-PLGA dynamic
future curves to test whether the transfer model can beat
`ShapePriorIncrementAnchored` without relying on PLGA dominance.

## Claim Package

Script `88` is the current one-file reporting package. Its claim ladder is:

| Status | Claim | Safe wording |
|---|---|---|
| Supported | Early observation budget contains transferable release information | Observation-budget transfer signal is present |
| Supported | PLGA dynamic-future transfer beats shape baselines | PLGA is the strongest validated source-domain win |
| Supported with caveat | Aggregate dynamic-future rows favor the model | Encouraging but source-dominated |
| Not yet supported | Non-PLGA dynamic transfer beats shape-adapted baselines | Needs more data or model changes |
| Not supported | Raw ensemble intervals are calibrated uncertainty | Intervals are diagnostic only |
| Audited | Shape-prior baselines are train-only | Fair train-only competitors |

Use `outputs/88_release_transfer_claim_package_shapeanchored/report.md` as the
starting point for slides or a manuscript result section.

## Liposome Ablation Follow-Up

Script `89` consolidates the first liposome-only ablation set:

- `mlp_none_e10`
- `mlp_safe_e10`
- `lnn_none_e10`

All three variants use script `81` with `--only-heldout-systems liposome` and
`--skip-within-large`, then pass through scripts `83`, `84`, and `86`.

Main dynamic early-budget result (`t<=0.25d`, 117 hard-dynamic curves):

| Variant | Model median RMSE | Best dynamic baseline | Baseline RMSE | Delta |
|---|---:|---|---:|---:|
| `mlp_safe_e10` | 0.240 | `ShapePriorIncrementAnchored:hill` | 0.169 | +0.072 |
| `lnn_none_e10` | 0.241 | `ShapePriorIncrementAnchored:hill` | 0.169 | +0.072 |
| `mlp_none_e10` | 0.241 | `ShapePriorIncrementAnchored:hill` | 0.169 | +0.072 |

So the first targeted test does **not** support "LNN fixes liposome" or
"safe metadata fixes liposome." Liposome remains useful precisely because it is
large enough to falsify model tweaks: the next useful modeling step should be a
shape-prior-aware adapter, better liposome process descriptors, or a loss that
targets the early dynamic regime directly. A larger LNN alone is not currently
justified by evidence.

Report entry point:
`outputs/89_release_liposome_ablation_report/report.md`.

## LNN Transfer Audit

Script `90` answers the follow-up audit questions directly:

1. `ShapePriorIncrementAnchored:hill` for liposome is leakage-free under the
   reconstructed split audit.
2. Liposome loses to the hill increment-anchored prior on the main dynamic early
   budget.
3. LNN is not meaningfully better than MLP in PLGA within-system.
4. Current evidence favors system-local or late-budget pattern learning, not
   robust cross-system transfer.
5. The neural model should not be globally demoted to a negative control because
   PLGA still has real wins, but the LNN-specific claim and non-PLGA transfer
   claim should be demoted.

Key numbers:

| Question | Result |
|---|---|
| Liposome hill leakage | 542 train source curves, 209 heldout fit-bank curves, 0 leaked source curves |
| Liposome `t<=0.25d` hard-dynamic vs hill | median per-curve delta model-hill = +0.106; model wins 33.3% |
| PLGA within `t<=14d` LNN vs MLP | LNN - MLP median RMSE = -0.0008 |
| Non-PLGA informative dynamic wins | 0/7 |

Interpretation: PLGA within-system late-budget performance is a system-local
neural pattern result, not evidence that LNN learned cross-system release
transfer. A larger LNN is therefore not justified by the current diagnostics.

Report entry point:
`outputs/90_release_lnn_transfer_audit/report.md`.

## What To Compare Next

Priority baselines for the next run:

1. A real calibration design, not just post-hoc LOSO diagnostic scaling.
2. Reframe headline plots around informative budgets or dynamic-future masks,
   not a single `t<=14d` horizon across all systems.
3. Expand non-PLGA dynamic-future data before claiming cross-mechanism transfer.
4. Use liposome as the first model-ablation target because it already has enough
   dynamic rows to test whether architecture changes beat shape priors.
5. Within-large-system PLGA and liposome baselines as secondary evidence, not
   the headline.

The methodological target is now clear: beat anchored classical priors under
LOSO time-window forecasting, then show uncertainty contraction is meaningful
after calibration.
