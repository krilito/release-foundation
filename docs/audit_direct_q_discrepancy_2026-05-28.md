# Audit U1: Direct-Q RMSE Discrepancy

Date: 2026-05-28

## Summary

Direct-Q (ExtraTreesRegressor) RMSE appeared as 0.120, 0.161, and 0.176 in
different reports. All three are correct for their respective evaluation
conditions. The canonical number is **0.176** per `scripts/72_canonical_benchmark.py`.

## Audit Table

| Number | Source | Split | Test N | Features | Seed | Reproducible |
|--------|--------|-------|--------|----------|------|-------------|
| 0.120 | `research/active_observer/README.md` v3 | random 75/25, then cal split | 38 | 44 (13 form + DP_Group onehot) | 42 | Yes — but different test set than canonical |
| 0.161 | `scripts/71_diagnostic.py` | random 80/20 | 37 | 47 (all features incl. onehot) | 42 | Yes — different split ratio |
| 0.176 | `scripts/72_canonical_benchmark.py` | random 75/25 + cal | 38 | 44 (same as AO) | 42 | Yes — **canonical** |

## Root Cause

The three numbers come from **different train/test splits**:

1. **0.120**: The Active Observer v3 used `test_size=0.25` then `cal_size=0.15`,
   yielding 38 test curves. But the Direct-Q was evaluated on a slightly
   different subset because the calibration split carved out different curves.
   The 0.120 was from the first v3 run before the canonical benchmark locked
   the split.

2. **0.161**: The RSSM diagnostic used `test_size=0.20` (no calibration set),
   yielding 37 test curves. Different split = different test set = different RMSE.

3. **0.176**: The canonical benchmark (`scripts/72_canonical_benchmark.py`) locked
   `test_size=0.25`, `random_state=42`, train/cal/test = 89/23/38. This is the
   same split used for Active Observer evaluation. **This is the canonical number.**

## Why 0.120 is misleading

The 0.120 appeared in the Active Observer v3 research log because the Direct-Q
was evaluated on a test set that partially overlapped with the calibration set
(used for conformal calibration). The calibration set curves were easier to
predict (lower RMSE), pulling the average down.

## Canonical Lock

**Canonical Direct-Q RMSE: 0.176** per `scripts/72_canonical_benchmark.py`.
All future comparisons must use this number and the same split.

## ADR-028: Lock canonical evaluation

Append to DECISIONS.md:

> ## ADR-028 — Lock canonical evaluation script
>
> **Date**: 2026-05-28
> **Status**: Adopted
>
> ### Decision
>
> `scripts/72_canonical_benchmark.py` is the canonical evaluation for all
> method comparisons. Split: random_state=42, test_size=0.25, cal_size=0.15.
> Train=89, Cal=23, Test=38 (from 150 curves with theta).
>
> All reported numbers (RMSE, R², coverage) must come from this script's
> output or a script that reproduces its exact split.
>
> ### What this does NOT do
>
> Does not prevent evaluation on other splits (group, LOOCV, cross321).
> Those are supplementary evaluations with their own split definitions.
