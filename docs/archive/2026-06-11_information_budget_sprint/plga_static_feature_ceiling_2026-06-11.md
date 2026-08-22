# PLGA Static-Feature Ceiling Probe

Date: 2026-06-11

## Question

Can PLGA release curves be predicted well from static, pre-release descriptors
alone?

This probe intentionally excludes early release observations. It asks whether
the descriptor bridge itself contains enough information before adding few-shot
context points.

## Method

Script:

`scripts/91_plga_static_feature_ceiling_probe.py`

Consolidated report:

`outputs/92_plga_static_feature_ceiling_report/report.md`

Modeling route:

`static features -> shape-family parameters -> Q(t)`

Training uses fitted shape-family parameters only for training curves. Heldout
curve fitted parameters are not used as inputs. Evaluation is against raw
heldout release observations.

Feature groups tested:

- `intercept_only`
- `polymer_core`
- `formulation_core`
- `drug_physchem`
- `intrinsic_no_source`
- `all_measured_no_source`
- `source_diagnostic`
- `all_with_source_diagnostic`

Shape families tested:

- `weibull`
- `biexponential`
- `hill`

Models tested:

- `extra_trees`
- `ridge`
- `knn`

## Main Results

| Split | Best config | Median RMSE | Median R2 | Interpretation |
|---|---|---:|---:|---|
| random-kfold | `intrinsic_no_source + hill + extra_trees` | 0.109 | 0.846 | strong within-distribution descriptor signal |
| source-group-kfold | `drug_physchem + hill + ridge` | 0.216 | 0.478 | collapses near intercept-only |
| source-dataset-lodo | `polymer_core + hill + extra_trees` | 0.215 | 0.475 | collapses near intercept-only |

Intercept-only references:

- source-group-kfold best intercept-only median RMSE: `0.216`
- source-dataset-lodo best intercept-only median RMSE: `0.221`

## Interpretation

Static PLGA descriptors clearly contain useful information under random
within-distribution splits. This supports the idea that the formulation table is
not empty signal.

However, the bridge largely collapses under source-group and source-dataset
holdout. The random-kfold result should therefore be treated as an information
ceiling, not deployment evidence.

The current best reading is:

> Static descriptors can predict PLGA release well inside the same data regime,
> but the present descriptor bridge is source-local and does not yet survive
> source shift.

## Per-Curve Static Combo Oracle

Script `94` asks a sharper diagnostic question:

> If each heldout curve is allowed to choose its best static feature combination
> after seeing the heldout error, can static descriptors predict well?

This is **not** a deployable predictor. The oracle selector sees heldout errors.
It is only an upper bound for whether a curve appears representable by some
static feature subset.

Report:

`outputs/94_plga_static_combo_oracle_report/report.md`

Main result:

| Split | Intercept median RMSE | Deployable oracle median RMSE | Ratio | Curves improved |
|---|---:|---:|---:|---:|
| random-kfold | 0.165 | 0.059 | 0.441 | 92.8% |
| source-group-kfold | 0.213 | 0.114 | 0.613 | 89.4% |
| source-dataset-lodo | 0.215 | 0.118 | 0.655 | 83.3% |

The oracle result changes the diagnosis:

- The static feature table likely contains enough information to explain many
  individual PLGA curves if the correct feature subset is chosen.
- The earlier global-model collapse under strict splits is therefore not simply
  "static descriptors are useless."
- The missing piece is whether the correct feature subset can be selected
  before seeing the release curve.

Next experiment:

`static descriptors -> gated feature-combo selector -> selected expert -> Q(t)`

Only the gated selector can become a predictive claim. The per-curve oracle
remains diagnostic.

## Next Step

The next experiment should not simply add a larger neural model. It should split
features into layered mechanisms:

`theta = theta_base(static) + delta_process(process) + delta_env(environment)`

Priority fields for strengthening the bridge:

- polymer grade details
- fabrication route
- solvent and surfactant system
- particle-size definition
- medium type
- pH
- temperature
- agitation or replacement protocol

Use random-kfold only as the upper-bound descriptor probe. Use source-group and
source-dataset splits as the stricter bridge tests.
