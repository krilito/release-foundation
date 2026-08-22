# Liposome Candidate Audit E0

Date: 2026-06-12

Status: completed E0 gate for the second drug-release system.

## Purpose

Start the second-system release route with a data audit, not a model. The goal
is to decide whether liposome IVR is suitable for the PLGA-style sequence:

```text
static whole-curve prediction -> bottleneck decomposition if static fails ->
early-observation budget -> uncertainty and stopping rules
```

## Reproduction

Script:

```text
scripts/108_liposome_candidate_audit.py
```

Command:

```powershell
python scripts\108_liposome_candidate_audit.py --pool outputs\149_release_caveat_augmented_cumulative_v1 --out outputs\108_liposome_candidate_audit --seed 0 --min-timepoints 6
```

Output directory:

```text
outputs/108_liposome_candidate_audit/
```

Required output files were generated:

| File | Role |
|---|---|
| `dataset_completeness.csv` | corpus and release-value integrity summary |
| `descriptor_completeness.csv` | static descriptor missingness and uniqueness |
| `split_candidate_summary.csv` | feasible heldout groups by split axis |
| `timepoint_coverage.csv` | curve-level timepoint and early-budget coverage |
| `candidate_decision_table.csv` | E0 proceed / reject gate |
| `data_checks.csv` | machine-readable verification checks |
| `lock_metadata.json` | git hash, input metadata, CLI args, row counts |
| `report.md` | concise E0 interpretation |

## Main Result

Liposome IVR passes the E0 gate and should proceed to E1 static whole-curve
prediction.

Key counts:

| Metric | Value |
|---|---:|
| liposome curves | 209 |
| curve IDs with release rows | 209 |
| release rows | 3332 |
| valid finite release rows | 3332 |
| curves with at least 6 valid timepoints | 209 |
| median valid timepoints per curve | 11 |
| median max time | 1.001 days |
| raw release fraction range | 0.000 to 1.152 |

Descriptor completeness:

| Descriptor | Nonmissing |
|---|---:|
| API name | 208 / 209 |
| API ID | 208 / 209 |
| structure type | 195 / 209 |
| drug molecular weight | 208 / 209 |
| particle size | 159 / 209 |
| PDI | 109 / 209 |
| zeta potential | 78 / 209 |
| pH | 163 / 209 |
| temperature | 197 / 209 |
| release method | 208 / 209 |

Feasible strict split axes:

```text
API_ID
API_name
structure_type
media_pH
media_temp_oC
release_method
```

Source split is not suitable as headline evidence because all liposome curves
come from one `source_dataset`.

## Verification

All E0 checks passed:

| Check | Status | Evidence |
|---|---|---|
| liposome IDs have curve rows | pass | formulations=209; curves_long=209 |
| finite time and release rows | pass | 3332 / 3332 |
| every analyzed curve passes min timepoints | pass | min_timepoints=6; pass=209 / 209 |
| raw release fraction preserved | pass | no model clipping performed |
| at least one strict split axis feasible | pass | API, structure, pH, temperature, release method |

Validation command:

```powershell
python -m py_compile scripts\108_liposome_candidate_audit.py
```

## Interpretation

This is a usable second retrospective system, but it is not a PLGA mechanism
transfer test. It is a test of whether the information-budget story has a
second-system analogue.

The immediate next question is:

```text
Can static liposome formulation and assay descriptors predict the whole release
curve under strict non-source splits?
```

If yes, liposome is a contrast case where descriptors are comparatively
sufficient. If no, then the project should decompose the failure and only then
test early release observations.

## Next Step

Build E1:

```text
scripts/109_liposome_static_whole_curve_prediction.py
```

E1 must stay static-only. Do not add early observations, neural sequence models,
LNN, KAN, MoE, PINN, or symbolic regression before the static descriptor ceiling
is measured.

