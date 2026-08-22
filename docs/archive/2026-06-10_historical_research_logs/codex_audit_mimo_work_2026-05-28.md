# Codex audit of Mimo-generated work

Date: 2026-05-28
Auditor: Codex
Scope: Mimo/parallel-agent artifacts added around 2026-05-28, especially scripts 70/71/72/74/77, their outputs, and the locked paper requirement documents.

## Executive verdict

The new Mimo work contains useful prototypes and several good audit instincts, but it is not ready to serve as paper evidence. At least two central result streams are contaminated:

1. The v2 adaptive benchmark observes day 21 while day 21 is also part of the future evaluation target.
2. The v2 regime verification maps regime labels by `fid` across mixed datasets with duplicate IDs, so labels can be overwritten by the wrong dataset.

Until these are fixed and rerun, do not treat `docs/paper_requirements_locked_2026-05-28.md` as the single source of truth. Treat it as provisional working notes.

## P0 blockers

### P0-1: Adaptive benchmark leaks future labels

Affected files:

- `scripts/72_canonical_benchmark_v2.py`
- `scripts/77b_regime_benefit_loocv_v2.py`
- `outputs/72_canonical_benchmark_v2/*`
- `outputs/77_regime_benefit_loocv_v2/*`

Evidence:

- `scripts/72_canonical_benchmark_v2.py` defines `future_start = 14.0` and evaluates `time_grid > future_start`, so the future target includes day 21, 28, 42, 56, and 84.
- The same script allows adaptive late candidates `[10.0, 14.0, 21.0]`.
- The run artifact confirms every adaptive case used `7,21`:

```text
Active-adaptive, 7,21     38
DirectQ-adaptive, 7,21    38
```

- `outputs/72_canonical_benchmark_v2/lock_metadata.json` confirms evaluation times include `21.0`.
- `scripts/77b_regime_benefit_loocv_v2.py` has the same pattern: `future_mask = time_grid > 14.0` and `candidate_times` includes `21.0`.

Why this matters:

The adaptive methods are allowed to observe a value that is also scored as part of the prediction target. This contaminates the comparison between fixed and adaptive observation, and it especially inflates `DirectQ-adaptive` and `Active-adaptive`.

Current consequence:

- Do not use the 72_v2 adaptive RMSE numbers as evidence.
- Do not use the 77b LOOCV adaptive-vs-fixed conclusions as evidence.
- Do not use E1/E4/E9 conclusions in the locked requirement doc until this is rerun.

Required fix:

Use one of two clean protocols:

1. Restrict all observation candidates to `t <= 14.0`, with evaluation `t > 14.0`.
2. If day 21 observation is scientifically necessary, change evaluation to `t > 21.0` and explicitly report the different prediction horizon.

The first option is cleaner for the current manuscript.

### P0-2: Regime verification mixes dataset labels

Affected files:

- `scripts/74b_regime_verification_v2.py`
- `outputs/74_regime_verification_v2/summary.json`

Evidence:

- `outputs/33_active_set_regimes/regime_assignments.csv` contains both `internal181` and `cross321`.
- It has duplicate `fid` values across datasets: 116 duplicated IDs.
- `scripts/74b_regime_verification_v2.py` constructs `regime_map = dict(zip(regime_df["fid"], regime_df["regime"]))`.
- Because the map key is only `fid`, one dataset can overwrite the other.
- The script then applies that map to local `data/formulations.csv`, `data/theta_bank.csv`, and `data/curves_long.csv`, which are internal-style assets.

Why this matters:

The reported result:

```json
{
  "sil_theta": -0.0573,
  "sil_x": -0.1964,
  "sil_curve": -0.2545,
  "decision": "Theta orthogonal to regime; ADR-027 supplementary framing valid"
}
```

may be based on wrong regime labels. The conclusion may be true or false, but this run cannot decide it.

Required fix:

Filter `regime_assignments.csv` by dataset before mapping labels. For internal data, use only `dataset == "internal181"`. For cross-source data, build a dataset-aware key `(dataset, fid)` or run separate verification per dataset.

### P0-3: The locked requirement document is over-authoritative

Affected file:

- `docs/paper_requirements_locked_2026-05-28.md`

Why this matters:

The document claims to supersede prior evidence, but several of its strongest decisions rely on contaminated or not-yet-durable results:

- E1 / E4 / E9 depend on the 72_v2 and 77b adaptive benchmarks.
- E10 appears tied to the 74b regime verification run.
- E5 cites RSSM bugfix metrics, but I did not find a durable output artifact matching those metrics.

Required fix:

Rename its status mentally and in future documents from "locked" to "provisional after Mimo v2". It can become locked only after the P0 reruns pass.

## P1 risks

### P1-1: RSSM BUG4 patch may still be temporally off by one

Affected file:

- `scripts/71_release_world_model.py`

Observed patch direction:

The patch decodes before overwriting the posterior sample with the prior sample. That fixes the most obvious "overwrite before decode" bug.

Remaining concern:

Inside the future rollout, the code updates `h_sample`, decodes using the previous `z_sample`, then samples the prior for the next iteration. In standard RSSM semantics, for a future time step one normally updates deterministic state, samples or uses the prior latent for that same time step, then decodes `(h_t, z_t)`. The current patch may decode `(h_t, z_{t-1})`.

Required fix:

Create a durable diagnostic artifact under `outputs/71_diagnostic_bugfix/summary.json` with:

- reconstruction RMSE,
- future RMSE,
- KL per latent dimension,
- coverage if uncertainty is claimed,
- explicit note of the latent-timing convention.

Do not cite console-only metrics.

### P1-2: Feasible posterior family is a useful prototype, not evidence

Affected directory:

- `..release-foundation-worldmodel/`

Good idea:

The "Feasible Posterior Family via Sensitivity SVD" prototype points in the right conceptual direction: a family of release curves consistent with sparse observations and mechanistic sensitivity structure.

Current limitations:

- It is mainly synthetic/prior-driven rather than fitted to the real cross-source evidence.
- Intervals are very wide, often `width_90` around 0.5 to 0.6.
- `exp5_family_update` contains `all_log_liks_non_finite`, `ess = 0`, and unchanged initial/updated metrics.
- In one comparison, constrained and unconstrained behavior appears identical, so the constraint is not yet doing measurable work.

Required fix:

Before integration into the main paper route, it needs:

- real-data evaluation,
- finite likelihood update,
- ESS diagnostics,
- constrained-vs-unconstrained ablation where the constraint changes predictions,
- conformal calibration or another explicit calibration layer.

### P1-3: Separate worktree was created inside the repo

Affected path:

- `D:\release-foundation\..release-foundation-worldmodel`

Why this matters:

This is not a clean sibling worktree. It is an untracked child directory inside the main repository, so `git status` sees it as `?? ..release-foundation-worldmodel/`.

Required fix:

Move it outside the repo or explicitly keep it ignored. Do not commit it.

## Salvageable assets

### Asset A: The four-way factorial design is the right comparison

The conceptual design is good:

| Selection | Inference | Method |
|---|---|---|
| fixed | direct future model | DirectQ-fixed |
| adaptive | direct future model | DirectQ-adaptive |
| fixed | mechanistic posterior | Active-fixed |
| adaptive | mechanistic posterior | Active-adaptive |

This is exactly how to separate observation-selection value from inference-model value. The implementation needs the P0 leakage fix.

### Asset B: `scripts/72b_canonical_benchmark_fixed.py` is closer to a clean baseline

This script compares Direct-Q fixed 4-point and Active fixed 4-point under matched observations, and trains the GRU/direct model on train data only. It is currently not a citable artifact because I did not find a durable output directory/summary.

Required upgrade:

Make it write:

- `outputs/72b_canonical_benchmark_fixed/summary.csv`,
- `per_curve_results.csv`,
- `lock_metadata.json`,
- bootstrap CI and paired deltas.

### Asset C: Script 33 active-set clustering is still meaningful, but only as definition

The active-set regime clustering is not invalid. It is a useful mathematical taxonomy of identifiability regimes.

But it cannot be sold as independent evidence that "PLGA has natural six regimes" unless validated against independent features, curves, or external data. If the clustering input is active-set indicators, then active-set silhouette is definitional, not validation.

Better wording:

"We define six identifiability regimes from active-parameter signatures, then test whether these regimes are reflected in independent formulation, kinetic, or release-curve features."

### Asset D: The world-model direction document is useful as strategy

`docs/world_model_research_direction_2026-05-28.md` has a strong internal north star: truth-aware release modeling under partial observation. Keep it as research strategy, but do not let "world model" become the manuscript headline until the learned dynamics, planning, and cross-mechanism evidence are real.

## Immediate next actions

1. Patch 72_v2 and 77b so adaptive observations cannot overlap the evaluation target; rerun both.
2. Patch 74b to use dataset-aware regime labels; rerun.
3. Add durable outputs for 71 diagnostics; rerun before citing RSSM bugfix metrics.
4. Convert 72b into a real artifact-writing benchmark.
5. Mark `docs/paper_requirements_locked_2026-05-28.md` as provisional in subsequent planning until the reruns pass.
6. Keep `..release-foundation-worldmodel/` out of git and consider moving it outside the repository.

## Bottom line for Nature-level strategy

The route is still viable, but not with the current Mimo evidence as-is. The strongest honest route remains:

1. Mechanism-constrained Bayesian/particle inference as the core engine.
2. Matched fixed/adaptive observation benchmarks with no horizon leakage.
3. Dataset-aware regime taxonomy as a supplementary mechanistic lens.
4. Cross-source and cross-mechanism robustness as the main claim.
5. Chitosan wet-lab data as prospective external validation, not just another retrospective dataset.

The current priority is not adding more models. It is making the evidence impossible to poke holes in.
