# Plan 51 — Model zoo defense run

Date: `2026-05-26`. Supersedes the original `plan_40_model_zoo_benchmark.md`
(renumbered because script 40 is now `40_theta_candidate_feasibility.py`).

## Why narrowed scope

Codex pushed a large batch (scripts 40-50, paper_closeout_release_quality_paradigm,
plan_41_mechanistic_feasibility_criterion) between the original plan_40 draft and now.
That batch already covers about 70% of the original 8-cell matrix on RF/ExtraTrees
variants. The paper closeout explicitly frames the model zoo as **defense, not
creative contribution**: appendix-level evidence that RF/ET is not winning only
because XGBoost/LGBM/Linear/SVR were never tried.

This plan only fills the actual gaps.

## What codex already covered (do not redo)

| Cell from old plan | Existing coverage |
|---|---|
| S1 cross321 zero-shot CV | scripts 40, 41, 46 (RF/ET multi-variant) |
| S2 cross321 few-shot CV | scripts 40, 45, 46, 50 |
| S3 internal181 zero-shot CV | scripts 42, 46 |
| S4 internal181 few-shot CV | scripts 42, 45, 46 |
| Input-source ablation | script 45 |
| Direct-Q vs theta-bottleneck | script 46 (theta beats direct Q by +0.04 to +0.12) |
| Early-window length sweep | scripts 44, 47 |
| Regime gating x CV | script 50 |
| External wet curves | scripts 48, 49 |
| Mixed-source augmentation | script 43 |

## What this plan adds

Two real gaps:

1. **Model family breadth.** All of 40-50 stay inside RF and ExtraTrees with
   leaf/seed variations. The paper currently cannot answer "did you try
   XGBoost / LightGBM / Linear / SVR / KNN / MLP?" without hand-waving.
2. **Pure zero-shot source transfer (S5/S6).** Script 43 tests *augmentation*
   (train = target + other source, test = target). It does **not** test
   `train = internal181 only, test = cross321 only` or vice versa. The
   original "0.54 transfer story" anchor has no model-zoo backup.

## Scope

### Sub-run A — model breadth on already-covered cells

For each `(dataset in [cross321, internal181]) x (input in [early_only,
formulation_plus_early]) x (cv in [random_5fold, group_by_drug,
group_by_polymer])`, add these models that codex has not tested:

| Family | Models | Output route |
|---|---|---|
| Linear | Ridge, Lasso, ElasticNet | both Q and theta |
| Boosting | GradientBoosting (sklearn), XGBoost, LightGBM | both Q and theta |
| Kernel | SVR (RBF) | Q only |
| Neighbors | KNN | Q only |
| Neural | MLP-{32x2, 64x2, 128x3} dropout 0.1 | Q only |

Library defaults, no per-cell tuning. Seed = 0.

The theta variant runs only for linear and boosting families (since they
naturally output bounded vectors after sigmoid squash). SVR / KNN / MLP
direct-Q only.

Expected output: a 14-row x 4-CV-cell table, sorted by median R^2, showing
whether anything passes RF/ET. Predicted result: RF/ET keeps the crown on
random/by_drug but XGBoost/LGBM may close the gap on by_polymer. MLP loses.
If the prediction is wrong, paper framing shifts.

### Sub-run B — pure zero-shot transfer (S5, S6)

Single-split transfer with full 14-model + RF/ET roster. Feature alignment
uses the 7-column intersection that script 43 already locks in:

```text
LA/GA, Polymer MW, Initial drug/polymer ratio, DLC,
Drug MW, Drug TPSA, Drug LogP, [+ Q(1,3,5,7) for few-shot]
```

Two scenarios:

```text
S5  train = internal181 (cleaned, n=169)
    test  = cross321 held-out (n=218)
    input = early_only AND formulation_plus_early

S6  train = cross321 (n=218)
    test  = internal181 held-out (n=169)
    input = early_only AND formulation_plus_early
```

Reverse direction (cross321 -> internal181) replaces what was S7/S8 in the
old plan. The internal-PLGA-only subset variant is dropped from this plan;
parking it as an optional appendix figure if reviewers ask.

Both directions report median / p10 / frac>=0.9 / frac>=0 / catastrophic
rate, theta and direct-Q routes both.

## Implementation

Two scripts:

```text
scripts/51_model_zoo_within_domain.py  # sub-run A
scripts/52_model_zoo_transfer.py        # sub-run B
```

Both reuse existing data loaders from scripts 42, 43, 46. They do not
re-clean curves or re-fit oracle theta. They only swap the model class.

Outputs:

```text
outputs/51_model_zoo_within_domain/
  per_curve.csv
  scheme_method_summary.csv
  vs_codex_baseline.csv   # explicit delta vs RF_raw / ET_raw / early_select
outputs/52_model_zoo_transfer/
  per_curve.csv
  direction_method_summary.csv
```

Estimated wall-clock:
- 51: ~60-90 min (most of it MLP and SVR on internal181 by_drug)
- 52: ~20-30 min (single split, no CV)

## Decision rule after running

Update `docs/paper_closeout_release_quality_paradigm_2026-05-26.md`
Claim-evidence map row:

> "RF/ET theta mapping is better than MLP here." -- currently cites only
> 37c/38/41.

After 51 runs, this row's evidence should also cite 51 with the breadth
of model families covered. If RF/ET is still champion, the line stands.
If XGB/LGBM beats RF on by_polymer (the most realistic OOD), paper rewrites
that sentence to specify the boosting family.

For transfer (52), the question is harder. If pure zero-shot S5 stays in the
0.3-0.5 R^2 band the SBI track found, the transfer story is unchanged. If
some model family produces > 0.7 zero-shot, that becomes the new paper
contribution and reorders the narrative.

## Not in this plan (parked separately)

- Liposome IVR 271 cross-mechanism expansion. Tracked in
  `docs/data_collection_inventory_2026-05-26.md` Tier A3. Separate sprint.
- internal-PLGA-only subset transfer. Optional appendix if reviewers ask.
- Round 2 winner tuning. Deferred until round 1 says it would matter.
