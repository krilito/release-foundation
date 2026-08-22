# PLGA Conditional Feature-Relevance Protocol

Date: 2026-06-11

## Question

Do different PLGA release curves require different static feature subsets to be
predictable?

This protocol is designed before running the formal experiment. It separates a
diagnostic oracle from a valid predictive gate so that per-curve feature choice
does not become target leakage.

## Script

Protocol-only dry run:

`scripts/93_plga_conditional_feature_relevance_protocol.py`

Generated protocol:

`outputs/93_plga_conditional_feature_relevance_protocol/protocol.md`

The script currently writes the feature-combo plan and leak guard. It does not
train the formal experiment.

## Core Distinction

`test_oracle_combo`

- Try all static feature combinations on a heldout curve.
- Report the best combination after seeing heldout errors.
- Diagnostic only.
- Answers: does this curve appear representable by some static subset?

`gated_combo`

- Learn which combination to use from train-fold out-of-fold labels.
- At test time, choose the combination from static descriptors only.
- Valid predictive evidence.
- Answers: can we know before measurement which feature subset should matter?

## Algorithm

For each outer split:

1. Train expert models for each predefined static feature combo, shape family,
   and regressor using training curves only.
2. Inside the training fold, produce out-of-fold predictions for every expert.
3. For each training curve, assign:
   - `best_combo_label`
   - `predictability_score = best_oof_rmse / intercept_oof_rmse`
4. Train a gate from static descriptors to `best_combo_label` or expert weights.
5. On heldout curves, evaluate:
   - `global_train_best`
   - `gated_combo`
   - `test_oracle_combo`
6. Report:
   - per-curve oracle RMSE
   - gated RMSE
   - oracle gap: `gated_rmse - oracle_rmse`
   - feature-combo frequency by curve regime

## Interpretation

| Pattern | Meaning |
|---|---|
| oracle strong, gate weak | curve-specific feature relevance exists, but current static descriptors cannot identify it before measurement |
| oracle strong, gate close to oracle | conditional feature relevance is a real predictive mechanism |
| oracle weak, gate weak | static descriptors are insufficient for those curves |
| source-diagnostic combos dominate | result may be source/protocol memory rather than physical descriptor transfer |

## Feature Combo Families

Base groups:

- `polymer_core`
- `formulation_core`
- `drug_physchem`
- `environment_protocol`
- `source_diagnostic_only`

Default deployable combos exclude `source_diagnostic_only`. Source fields can be
used only as a diagnostic stress test.

## Leak Guard

Forbidden inputs:

- early `Q(t)`
- heldout curve fitted theta
- heldout curve shape-family fit
- columns derived from the full heldout release curve
- feature-combo selection based on heldout errors, except for the explicitly
  labeled `test_oracle_combo`

## Execution Gate

Do not run the formal experiment until these are true:

- dry-run feature-combo plan has no target-derived columns
- feature coverage is reported for every combo
- `test_oracle_combo` is labeled diagnostic-only in outputs
- `gated_combo` selects experts using only static descriptors
- source-group and source-dataset splits are included
- outputs distinguish deployable claims from diagnostic oracle claims

This is the 95% confidence gate before spending compute or interpreting the
result.

## First Gate Screening

Script:

`scripts/95_plga_static_combo_gate_screen.py`

Reports:

- `outputs/95_plga_static_combo_gate_screen/report.md`
- `outputs/95_plga_static_combo_gate_screen_with_source_diag/report.md`

This is a screening experiment, not the final nested gate. It uses existing
script `91` out-of-fold expert errors as meta-labels, then trains a static
meta-gate to predict each expert's error and select the lowest predicted-error
combo.

Best no-source gate results:

| Split | Gate | Gated RMSE | Global best RMSE | Oracle RMSE | Oracle feature-group hit |
|---|---|---:|---:|---:|---:|
| random-kfold | `extra_trees` | 0.108 | 0.109 | 0.059 | 21.7% |
| source-group-kfold | `extra_trees` | 0.235 | 0.236 | 0.114 | 14.9% |
| source-dataset-lodo | `extra_trees` | 0.220 | 0.232 | 0.118 | 19.3% |

Adding `source_dataset/source_group` to the gate barely changes the result:

- random-kfold best gated RMSE: `0.108`
- source-group-kfold best gated RMSE: `0.235`
- source-dataset-lodo best gated RMSE: `0.219`

Interpretation:

- The per-curve oracle remains strong, so curve-specific static feature
  relevance is plausible.
- The first static gate does not recover that oracle; it mostly ties the global
  best combo.
- The current static descriptors are not enough to reliably identify, before
  measurement, which feature subset should control a given curve.

This keeps the next decision clean: either build a stricter nested gate with
better regime descriptors, or treat the oracle as evidence of missing regime
variables rather than a deployable predictor.
