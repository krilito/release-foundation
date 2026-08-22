# Release Uncertainty Snapshot

Date: `2026-05-29`

Purpose:

```text
Say exactly where the uncertainty layer stands in the unified
drug-release program, using current benchmark outputs rather than aspiration.
```

Key artifacts:

- [88_release_uncertainty_snapshot.py](D:/release-foundation/scripts/88_release_uncertainty_snapshot.py)
- [raw_uq_benchmark_snapshot.csv](D:/release-foundation/outputs/88_release_uncertainty_snapshot/raw_uq_benchmark_snapshot.csv)
- [calibrated_family_snapshot.csv](D:/release-foundation/outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv)
- [summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)

## 1. Raw benchmark reality

The current raw uncertainty picture is not uniformly strong.

From [raw_uq_benchmark_snapshot.csv](D:/release-foundation/outputs/88_release_uncertainty_snapshot/raw_uq_benchmark_snapshot.csv):

- raw `FIB` `cov90_mean` spans `0.720` to `0.913`
- raw `Ensemble` `cov90_mean` spans `0.430` to `0.750`

Interpretation:

- `FIB` is the stronger current uncertainty constructor
- even `FIB` is still under-nominal on harder OOD cells
- `Ensemble` is useful structurally, but not the current headline UQ route

## 2. What conformal repair now buys us

The current calibrated-family panel now includes:

- `cross321 / group_by_drug`
- `cross321 / group_by_polymer`
- `cross321 / random_5fold`
- `internal181 / group_by_drug`
- `internal181 / group_by_polymer`
- `internal181 / random_5fold`
- `liposome / group_by_drug`
- `liposome / group_by_polymer`
- `liposome / random_5fold`

From [summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt), the most diagnostic examples are:

- `cross321 / group_by_drug`
  - raw conformal cell `cov90_mean = 0.752`
  - global conformal `cov90_mean = 0.902`
  - width median grows from `0.241` to `0.485`

- `liposome / group_by_drug`
  - raw conformal cell `cov90_mean = 0.744`
  - global conformal `cov90_mean = 0.846`
  - width median grows from `0.307` to `0.454`

Across the current panel, the right reading is:

> conformal repair helps, but it is not free.

The price is interval inflation, and the lift is stronger on PLGA than on the
current liposome bridge cell.

## 3. What became newly true

The project now has a small `calibrated family panel`, not only:

- raw `FIB` / `Ensemble` tables
- per-curve conformal summaries
- low-rank family exports without calibration

This matters because downstream code can now consume one object that contains:

- latent feasible-state family
- decoder handle
- calibration context in release space

That is the real systems-level upgrade.

## 4. What we still cannot claim

We still cannot honestly claim:

1. benchmark-wide calibrated uncertainty is solved
2. cross-mechanism uncertainty is complete
3. latent family geometry itself was recalibrated
4. prospective calibrated-family validation exists

## 5. Bottom line

The uncertainty layer has crossed from:

> separate benchmark tables and UQ experiments

to:

> partial shared calibrated-family infrastructure across the current full
> PLGA + liposome panel

That is enough to strengthen the `unified drug-release intelligence` story.
It is not enough to close the UQ story completely.
