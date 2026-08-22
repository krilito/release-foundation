# Script Map

This directory is an experiment ledger, not a clean application API. The
numbered scripts preserve the evidence chain behind the broader release
research program. Do not start a new numbered script until checking this map.

## Numbering rule

- `01_` to `39_`: early SBI, simulator, and baseline history
- `40_` to `59_`: theta-route prediction benchmark and mechanism-layer defense
- `60_` to `67_`: posterior-family / UQ / prospective bridge work
- `68_` to `77_`: Active Observer, diagnostics, and post-bugfix audits
- `78_` onward: current liposome/export endpoints plus archived release-wave
  side routes; `80_*` and above are not automatically current just because
  their number is high

Higher script numbers are not automatically more canonical. The canonical
question is whether the script supports the current closeout framing and points
to a durable output the paper can honestly cite.

## Current Entry Points

Use these first when resuming the main research lines:

| Route | Entrypoint | Output | Status |
|---|---|---|---|
| Canonical split file | `make_canonical_split.py` | `data/canonical_split_v1.csv` | emits the fixed split file; defaults to seed 42 by design |
| PLGA tree-theta benchmark | `38d_baselines_groupkfold.py` | `outputs/38d_baselines_groupkfold/` | historical baseline |
| Theta-route vs direct-Q | `46_direct_curve_rf_input_ablation.py` | `outputs/46_direct_curve_rf_input_ablation/` | main-text eligible control |
| Middle-layer gain | `59_middle_layer_gain_audit.py` | `outputs/59_middle_layer_gain_audit/` | paper defense |
| LightGBM middle-layer check | `62_lgbm_middle_layer_gain.py` | `outputs/62_lgbm_middle_layer_gain/` | model-family defense |
| FIB-CASP cross-mechanism UQ | `62_fib_casp_benchmark.py` | `outputs/62_fib_casp_benchmark/` | few-shot UQ core, OOD caveat |
| Ensemble-CASP zero-shot UQ | `66_ensemble_casp.py` | `outputs/66_ensemble_casp/` | random-split zero-shot UQ, hard OOD weak |
| Chitosan prospective lock | `67_chitosan_prospective_predictions.py` | `outputs/67_chitosan_prospective/` | prospective pilot lock, wet reveal pending |
| Active kinetic observer | `69_active_observer_v3.py` | `outputs_active_observer_v3/` | supplementary UQ / decision-support route |
| Active observer GroupKFold | `68_active_kinetic_observer.py --groupkfold` | `outputs_active_observer/` | historical branch, not current champion |
| Corrected 4-way benchmark | `72_canonical_benchmark_v2.py` | `outputs/72_canonical_benchmark_v2/` | current corrected benchmark; roadmap still flags split-harness cleanup |
| Post-bugfix diagnostics | `73_diagnostics.py` | console + local figures/tables | supporting diagnostics, but currently user-modified in worktree |
| Canonical-split-safe diagnostics | `73b_diagnostics_canonical_split.py` | console + local figures/tables | safe A3 companion without overwriting `73` |
| Regime verification v2 | `74b_regime_verification_v2.py` | `outputs/74_regime_verification_v2/` | negative-result / limitation evidence |
| Chitosan reveal-time eval | `75_chitosan_prospective_eval.py` | `outputs/75_chitosan_prospective_eval/` | preregistered wet reveal evaluator |
| Internal NPE comparator scaffold | `27e_internal_canonical_npe_fixed_eval.py` | `outputs/27e_internal_canonical_npe_fixed_eval/` | first same-mouth B4 scaffold against `72_v2` |
| Curve-first diagnostic | `70_curve_first_kinetic_observer.py` | `outputs/70_curve_first_kinetic_observer_*` | future route, not champion |
| Synthetic hidden-state audit | `118_synthetic_hidden_state_identifiability_audit.py` | `outputs/118_synthetic_hidden_state_identifiability_audit/` | diagnostic simulation for static-X information gaps and early-Q state recovery |
| Real PLGA synthetic-state bridge | `119_plga_synthetic_state_real_projection.py` | `outputs/119_plga_synthetic_state_real_projection/` | projects 321 PLGA curves into H_like coordinates; bridge diagnostic with OOD coverage warning |
| PLGA missing-state residual map | `120_plga_missing_state_residual_map.py` | `outputs/120_plga_missing_state_residual_map/` | formal ReleaseStateSpace residual map; prerequisite for RSF/MSVS measurement scoring |
| PLGA candidate measurement scoring | `121_plga_candidate_measurement_value_scoring.py` | `outputs/121_plga_candidate_measurement_value_scoring/` | ranks candidate measurements from 120 strict OOF residuals; separates evidence-backed proxies from hypothesis-only priorities |
| Release dataset candidate audit gate | `122_release_dataset_candidate_audit_and_gate.py` | `outputs/122_release_dataset_candidate_audit_and_gate/` | audits candidate external release datasets and runs a minimal ExtraTrees gate only on eligible local standardized curve pools |

## Route Groups

### Archived 02-28: SBI and ODE feasibility history

Early PLGA posterior and simulator diagnostics. These scripts establish why the
ODE scaffold can fit release curves, but also why pure descriptor-to-parameter
deployment is hard.

Status: moved to `archive/2026-06-10_historical_sbi_active_set/`.

Keep as evidence. Do not extend this line unless debugging original SBI results.

### Archived 29-36: Active-set and regime mechanics

Mechanistic identifiability evidence:

- `29_minimal_active_set_audit.py`
- `30_local_effective_dimension_audit.py`
- `33_active_set_regimes.py`
- `36_r6_regime_prototype.py`

These support the claim that curves identify low-dimensional feasible theta
families, not unique 9-D mechanisms.

Status: moved to `archive/2026-06-10_historical_sbi_active_set/`.

### Archived 37-50: Prediction benchmark and theta bottleneck

Main PLGA prediction defense:

- formulation + early Q -> theta -> ODE
- direct Q controls
- input-source ablations
- internal181 transfer
- regime refinement guardrails

Do not replace this line with another model-zoo run unless a specific reviewer
or result gap requires it.

Status: most non-entry scripts moved to
`archive/2026-06-10_historical_prediction_routes/`. Current benchmark-facing
scripts remain at the top level when listed in "Current Entry Points".

### Archived 52-55: Liposome accelerated-IVR extension

Cross-mechanism evidence using a Weibull decoder. This is the first non-PLGA
proof that the framework is decoder-agnostic.

Main result: sparse early release calibrates a continuous Weibull state much
better than kinetic-class prototype curves.

Status: moved to `archive/2026-06-10_historical_prediction_routes/`.

### Archived 56-58, 64-65, 70: Curve-first / world-model diagnostics

These test whether the release curve itself can be the primary state. The
answer is useful but not yet dominant: curve-first models are diagnostics and
future-model scaffolds, but the current early-Q -> theta-family -> decoder
route is stronger.

Status: moved to `archive/2026-06-10_historical_prediction_routes/` or
`archive/2026-06-10_future_diagnostics/`.

### 59, 62, 63: Middle-layer gain

These scripts quantify the value of the kinetic-state middle layer against
direct Q prediction. Use them for reviewer defense and for the "theta is not
decorative" claim.

Current top-level scripts: `59_middle_layer_gain_audit.py`,
`62_lgbm_middle_layer_gain.py`, `62_fib_casp_benchmark.py`. Future or
side-route `63_*` scripts are archived under
`archive/2026-06-10_future_diagnostics/`.

### 60-66: CASP posterior family

Uncertainty and posterior framework:

- `60_*`: neural CASP v0, mostly falsified by small data.
- `61_fib_casp_v0.py`: single-mechanism FIB sanity.
- `62_fib_casp_benchmark.py`: current few-shot UQ benchmark; PLGA strongest,
  liposome useful but still small-N.
- `63_pcap_pullback_targets.py` / `64_pcap_train_phaseB.py`: future-mechanism
  pull-back target route, not a current winner.
- `65_fib_casp_zero_shot.py`: zero-shot FIB under-calibrated.
- `66_ensemble_casp.py`: zero-shot ensemble posterior; random split useful,
  hard OOD still weak and should stay supplementary.

Current top-level scripts: `62_fib_casp_benchmark.py`,
`66_ensemble_casp.py`. The falsified or future CASP/PCAP scripts are archived
under `archive/2026-06-10_future_diagnostics/`.

### 67: Chitosan prospective bridge

Locked predictions for the user's chitosan wet experiment. These files matter
because the predictions were generated before observed release curves were
compared. Treat this as a prospective pilot scaffold, not yet as proof of a
single unified learned cross-mechanism model.

### 68-69: Active Kinetic Observer

Particle observer:

```text
formulation -> theta particles -> ODE rollout -> active time selection
-> observation update -> future prediction + conformal uncertainty
```

Current interpretation after the 72/77 audits:

- selection is the part that survives most cleanly;
- the particle posterior does not currently beat matched Direct-Q point
  prediction;
- keep this route as supplementary decision-support / UQ evidence, not as the
  main predictive headline.

### 72-77: Post-bugfix correction layer

These scripts are the honesty layer added after the 2026-05-28 audit:

- `72_canonical_benchmark_v2.py`: corrected 4-way comparison
- `73_diagnostics.py`: supporting diagnostic summaries
- `73b_diagnostics_canonical_split.py`: safe fixed-split companion to `73`
- `74b_regime_verification_v2.py`: independent-space regime verification
- `75_chitosan_prospective_eval.py`: preregistered reveal evaluator
- `76_casp_conformal_recalibration.py`: posterior-family recalibration support
- `77b_regime_benefit_loocv_v2.py`: regime × benefit negative-result audit

If a note or older script conflicts with these corrected outputs, prefer the
durable outputs plus `docs/drug_release_closeout_2026-05-28.md`.

### 27e: Internal comparator scaffold

`27e_internal_canonical_npe_fixed_eval.py` is the first repo-local bridge from
the existing partial-curve NPE work into the corrected internal canonical
benchmark mouth.

Current scope:

- loads a trained `27c` posterior checkpoint
- evaluates on the fixed canonical test fold
- uses the same fixed 4-point early-observation mouth as `72_v2`
- writes per-curve metrics plus a pairwise comparison scaffold against
  `outputs/72_canonical_benchmark_v2/`

This is a B4 execution hook, not yet evidence that B4 is closed.

### 78-119: Release-wave export and manuscript-support assets

This wave contains liposome intake, posterior/export helpers, uncertainty
snapshots, paper figures, bridge panels, stack assets, and strategy-support
material.

Important:

- `80_*` through `119_*` have been moved to
  `archive/2026-06-10_release_wave_assets/`
- it is useful for manuscript assembly and internal synthesis
- it should not define the repo entry narrative
- anti-scope in `docs/top_journal_plan_2026-05-30.md` explicitly says not to
  expand the stack/positioning/deck side of this wave

`78_liposome_ivr_intake.py` and `79_liposome_prediction_export.py` remain at
the top level because they are closer to the current cross-mechanism evidence
line than the later figure/positioning/export wave.

### 120-122: Release-state forensics and candidate-data gates

The current top-level `120`, `121`, and `122` scripts are not part of the old
archived release-corpus wave.

- `120_plga_missing_state_residual_map.py`: maps same-space release-state
  residuals and source/method structure.
- `121_plga_candidate_measurement_value_scoring.py`: scores candidate
  measurements from strict OOF residuals, separating evidence-backed proxies
  from hypothesis-only priorities.
- `122_release_dataset_candidate_audit_and_gate.py`: builds a candidate
  release-dataset inventory and tests only eligible local standardized pools
  with a minimal static vs static-plus-early gate.

Use `122` to decide what data to collect next. Do not treat it as permission to
start a new architecture sprint.

### Archived 120-161: Release-corpus / KGSO / world-model assets

This wave contains release-corpus intake, external-source preprocessing,
layered-pool assembly, hard gates, theta freezing, hierarchical baselines, and
early world-model/KGSO support scripts.

Current status:

- moved to `archive/2026-06-10_release_corpus_world_model/`
- future-method / corpus-building workspace, not current release-paper truth
- revive only with a route document plus checked output manifest

Exception: the current top-level `120`, `121`, and `122` scripts are new
Release-State Inference / candidate-data experiments, not part of the old
archived release-corpus wave.

## Archive Scripts

These files are preserved as historical evidence:

| Path | Status |
|---|---|
| `archive/01_oracle_baseline.py` | historical oracle baseline |
| `archive/72_canonical_benchmark.py` | superseded by `72_canonical_benchmark_v2.py` |
| `archive/72b_canonical_benchmark_fixed.py` | historical bugfix checkpoint |
| `archive/74_regime_verification.py` | superseded by `74b_regime_verification_v2.py` |
| `archive/77_regime_benefit_loocv.py` | superseded by `77b_regime_benefit_loocv_v2.py` |
| `archive/2026-06-10_historical_sbi_active_set/` | 02-36 early SBI, posterior, deployment, and active-set/regime scripts |
| `archive/2026-06-10_historical_prediction_routes/` | 37-58 non-entry prediction, theta, wet-eval, liposome, and curve-hybrid scripts |
| `archive/2026-06-10_future_diagnostics/` | falsified/future CASP, PCAP, curve-world, active-observer, and green-claim diagnostics |
| `archive/2026-06-10_release_wave_assets/` | 80-119 release-wave export, figure, bridge, stack, and positioning scripts |
| `archive/2026-06-10_release_corpus_world_model/` | 120-161 corpus, KGSO, hard-gate, theta-freeze, and world-model scripts |

## Cleanup Policy

1. Keep numbered scripts as immutable evidence once their outputs are cited in
   docs or paper notes.
2. Prefer writing a new short route document before adding a new numbered
   script.
3. If a route is falsified, document that status here instead of deleting the
   script.
4. New reusable logic belongs in root modules or small packages such as
   `casp/`; numbered scripts should remain thin experiment runners.
5. Do not commit run outputs, real data, private handoff notes, or local
   `.claude/` files.
