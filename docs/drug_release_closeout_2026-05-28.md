# Drug Release Closeout — 2026-05-28

Scope:
- Release-only closeout.
- No new model routes.
- Goal: reduce the remaining open work to the chitosan wet-lab reveal plus any strictly necessary reveal-time bookkeeping.

## Bottom line

The current release paper should be closed around the following object:

```text
formulation + sparse early release
    -> feasible mechanism state / theta family
    -> simulator decoder
    -> full-trajectory prediction (+ optional posterior/UQ layer)
```

Do not close the paper around `world model`, `RSSM`, or `Active Observer`.
Those are side routes, diagnostics, or negative results.

## Surviving core algorithm

### Recommended name

**Early-release kinetic fingerprint model with a feasible mechanism-state family**

Short internal paraphrase:

```text
early Q is the fingerprint
theta-family is the bottleneck
ODE / decoder is the rollout engine
```

### Why this is the surviving line

1. `outputs/45_input_source_ablation/input_source_summary.csv`
   shows that early release is the dominant OOD signal, while formulation is
   conditioning context rather than the main predictor.
2. `outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv`
   shows that the theta-route beats matched direct-Q controls in every audited
   summary cell.
3. `outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv`
   shows a positive middle-layer gain overall, including OOD
   formulation-plus-early cells.
4. `outputs/38d_baselines_groupkfold/summary.txt`
   shows RF tree-theta beats the earlier MLP route under held-out drug and
   held-out polymer evaluation.

## Asset status

### Main-text eligible

| Asset | Role | Evidence status |
|---|---|---|
| `outputs/45_input_source_ablation/` | early-vs-formulation source attribution | Main-text usable |
| `outputs/46_direct_curve_rf_input_ablation/` | theta-route vs direct-Q control | Main-text usable |
| `outputs/38d_baselines_groupkfold/` | RF/MLP baseline closure | Main-text usable, but can be compressed |

### Supplementary but worth keeping

| Asset | Role | Evidence status |
|---|---|---|
| `outputs/59_middle_layer_gain_audit/` | reviewer-defense summary of theta middle layer | Supplementary |
| `outputs/62_fib_casp_benchmark/aggregate_table.csv` | few-shot UQ / posterior-family evidence | Supplementary; PLGA stronger than liposome |
| `outputs/66_ensemble_casp/aggregate_table.csv` | zero-shot random-split UQ evidence | Supplementary; OOD calibration weak |
| `outputs_active_observer_v3/metrics_summary.csv` | UQ / decision-support branch | Supplementary only |
| `outputs/74_regime_verification_v2/summary.json` | negative regime validation result | Supplementary / limitation |
| `outputs/77_regime_benefit_loocv_v2/summary.json` | negative regime × benefit result | Supplementary / limitation |

### Internal-only / do-not-cite

| Asset | Why not citable |
|---|---|
| `scripts/71_release_world_model.py` + `outputs/71_diagnostic_bugfix/summary.json` | useful negative result, but not a paper method |
| `scripts/74_regime_verification.py` | mixed-dataset `fid` mapping; superseded by `74b` |
| `scripts/77_regime_benefit_loocv.py` | mixed-dataset `fid` mapping; superseded by `77b` |
| `outputs/75_chitosan_prospective_eval/smoke*` | smoke tests only; not wet reveal evidence |
| `docs/decision_post_bugfix_2026-05-28.md` | intermediate note with non-final numbers |

## Claims allowed now

1. Sparse early release can be converted into a simulator-compatible mechanism
   state / theta family for long-horizon PLGA release prediction.
2. The theta-route is consistently stronger than matched direct-Q regression in
   the audited PLGA benchmarks.
3. Theta should be interpreted as a feasible mechanism family, not a unique
   recovered ground-truth parameter.
4. CASP-style posterior families are useful as a UQ layer, especially in the
   few-shot setting, but OOD calibration remains incomplete.

## Claims banned now

1. `world model`
2. `foundation model`
3. `Active Observer is the best predictive algorithm`
4. `regime explains Active Observer benefit`
5. `cross-mechanism validation is complete`
6. `zero-shot OOD calibration is solved`
7. `RSSM is competitive after bugfix`

## Chitosan status

### What is genuinely locked

The prospective lock is valid:

- `outputs/67_chitosan_prospective/lock_metadata.json`
- `outputs/67_chitosan_prospective/HASHES.txt`
- git tag `prospective-chitosan-batch1-2026-05-28`
- prediction commit `ceff62647ba9e43324e4889145f99c08dffd71b3`

### What it does and does not prove

What it proves if the wet reveal succeeds:
- a prospectively locked, mechanism-constrained chitosan pilot can achieve the
  preregistered coverage target.

What it does not prove:
- that the exact PLGA Active Observer v3 artifact transfers unchanged to
  chitosan.

Reason:
- `scripts/67_chitosan_prospective_predictions.py` uses a chitosan-specific
  simulator and a hand-specified low-rank theta family, not the same trained
  PLGA observer artifact.

## Release-time evaluator status

`scripts/75_chitosan_prospective_eval.py` has been aligned to the real
preregistration:

- primary endpoint = aggregate `cov90`
- success threshold = `cov90 >= 0.83`
- source doc = `docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md`

Secondary outputs now reported descriptively:

- per-curve hit rate (`cov90 >= 0.80`)
- pooled R²
- mean PI width
- empirical CRPS from `theta_targets.npz`

Important:
- the old curve-level `R2 / MAPE` gate should not be used as the primary
  prospective pass/fail rule.

## Active Observer final status

Demote `Active Observer` from headline method to supplementary
decision-support/UQ branch.

Reason:
- `outputs/72_canonical_benchmark_v2/pairwise_comparisons.csv` does not show a
  significant point-prediction win over matched Direct-Q.
- `outputs/77_regime_benefit_loocv_v2/summary.json` shows no significant regime
  × benefit interaction.
- `outputs_active_observer_v3/metrics_summary.csv` is still useful for UQ and
  measurement-scheduling discussion, but not as the main predictive winner.

## Remaining minimum work

### Before the release section is considered closed

1. Keep `docs/paper_requirements_locked_2026-05-28.md` path-stable but treat it
   as provisional only.
2. Do not cite `outputs/75_chitosan_prospective_eval/smoke*`.
3. At wet reveal time, run `scripts/75_chitosan_prospective_eval.py` on the real
   observed CSV and report the preregistered `cov90 >= 0.83` outcome exactly.

### After that

If chitosan `cov90 >= 0.83`:
- report it as prospective support for the mechanism-constrained transfer
  scaffold, with explicit scope limits.

If chitosan `cov90 < 0.83`:
- keep the PLGA paper.
- write the chitosan result honestly as a prospective transfer limitation.

## Single practical rule

If a future note conflicts with this file, trust the durable output files
first, then this closeout note, then older planning notes.
