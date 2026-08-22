# Release Dataset Candidate Audit And Gate

Date: 2026-06-14

## Purpose

Experiment 122 keeps the project on the release-state inference route while expanding candidate datasets.

The goal is not to collect many curves blindly. A candidate dataset matters only if it can test:

```text
static X -> release endpoint / curve state
static X + early Q -> future release endpoint / curve state
residual -> missing material/process/state descriptor hypothesis
```

## Gate Policy

- Web candidates are audited for access, scale, descriptors, and fit to the story.
- Training runs only on local standardized curve pools with enough curves and timepoints.
- No timepoint-independent split is allowed.
- No new neural model or architecture is introduced.

## Current Highest-Priority Collection Targets

1. Alginate hydrogel active-learning release profiles, because it directly supports the chitosan/HA/PVP material-controllability story.
2. Chitosan nanoparticle release profile dataset, because it is the closest literature analogue to the planned chitosan direction.
3. Direct-compression tablet profiles, because it is a large outgroup if the raw table can be obtained.

## Local Training Gate

The local standardized pools were audited before training. A dataset is training-eligible only when it has enough curve-level samples, enough release timepoints, and usable descriptors. This keeps the project from drifting into timepoint leakage or model invention.

The only eligible local second-system pool in this pass is:

| Pool | Dataset | Curves | Group split | Role |
|---|---:|---:|---|---|
| `outputs/148_release_main_cumulative_v1/` | `liposome_ivr209` | 209 | `API_name` | second-system validation |
| `outputs/149_release_caveat_augmented_cumulative_v1/` | `liposome_ivr209` | 209 | `API_name` | sensitivity pool |

Small hydrogel, chitosan-bridge, starch, halloysite, Degrapol, and other local standardized sets remain useful as audit/background sources, but they are not yet large enough to support a headline training comparison.

## Minimal Training Result

The gate used only small-data friendly ExtraTrees baselines:

```text
train mean prior
static X
static X + earliest two release observations
```

Under both random curve folds and strict `group_by_API_name`, `static X + early Q` beat `static X` for all checked endpoint and AUC rows. The strict API split remains difficult: static descriptors alone are weak, and early observations carry most of the transferable release-state signal.

This supports the existing project thesis:

```text
cross-system release prediction should be treated as release-state inference
under limited descriptors and costly observations, not as a search for a
larger black-box curve predictor.
```

## Access Audit Boundary

- The alginate hydrogel and chitosan nanoparticle sources are high-priority collection targets, but their machine-readable raw curve tables are not yet local.
- The direct-compression tablet paper is open and its supplementary DOCX was downloaded, but that file contains supplier tables and release-profile figures rather than a machine-readable formulation-release matrix.
- PLGA nanoparticles Mendeley data are useful for descriptor/CQA context but are endpoint data, not a release-curve training corpus.

## Output Anchor

`../outputs/122_release_dataset_candidate_audit_and_gate/`
