# Conformal Family Attachment Verdict

Date: `2026-05-29`

Purpose:

```text
Record what became true once split-conformal recalibration from script 76
was attached to the shared posterior-family object rather than left as a
separate summary table.
```

Key artifacts:

- [87_release_conformal_family_export.py](D:/release-foundation/scripts/87_release_conformal_family_export.py)
- [cross321 / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug)
- [cross321 / group_by_polymer](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_polymer)
- [cross321 / random_5fold](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/random_5fold)
- [internal181 / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/internal181/group_by_drug)
- [internal181 / group_by_polymer](D:/release-foundation/outputs/87_release_conformal_family_export/internal181/group_by_polymer)
- [internal181 / random_5fold](D:/release-foundation/outputs/87_release_conformal_family_export/internal181/random_5fold)
- [liposome / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_drug)
- [liposome / group_by_polymer](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_polymer)
- [liposome / random_5fold](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/random_5fold)
- [76 summary](D:/release-foundation/outputs/76_casp_conformal_recalibration/summary.txt)

## 1. What changed

Before this step, the project had:

1. raw low-rank family exports from `FIB` / `ensemble`
2. separate split-conformal coverage and width tables

After this step, it also has:

3. one shared family artifact that carries both:
   - latent feasible-state structure
   - decoder-output-space interval calibration context

That is a meaningful systems step. Deployment-side code no longer has to look
in one place for the family and another place for the interval scales.

## 2. What the object now carries

The new `calibration_context` field records:

- calibration method
- calibration domain
- nominal coverage
- global scale
- local time-bin scales
- empirical raw/global/local coverage
- empirical raw/global/local interval width

Example artifact:

- [cross321_2.json](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug/cross321_2.json)

## 3. What this does and does not mean

What it means:

- a real `PLGA + liposome` panel of `FIB` families can now be serialized as
  calibrated deployment artifacts
- the shared posterior-family schema is now richer than:
  - point estimate only
  - raw low-rank family only

What it does **not** mean:

- latent `z_center / z_basis / z_scale` were recalibrated
- conformal calibration is benchmark-wide
- cross-mechanism calibrated uncertainty is solved

The calibration still lives in release-trajectory space, which is the honest
place to keep it.

## 4. Current verified scope

Verified export scope today:

- `cross321 / group_by_drug`: `259` calibrated family JSONs
- `cross321 / group_by_polymer`: `259`
- `cross321 / random_5fold`: `259`
- `internal181 / group_by_drug`: `162`
- `internal181 / group_by_polymer`: `162`
- `internal181 / random_5fold`: `162`
- `liposome / group_by_drug`: `93`
- `liposome / group_by_polymer`: `93`
- `liposome / random_5fold`: `93`

Evidence:

- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_polymer/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/random_5fold/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/internal181/group_by_drug/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/internal181/group_by_polymer/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/internal181/random_5fold/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_drug/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_polymer/summary.csv)
- [summary.csv](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/random_5fold/summary.csv)

## 5. Bottom line

The strongest honest sentence that became true is:

> unified drug-release intelligence now has a real calibrated-family panel of
> posterior-family artifacts, not just shared benchmark tables and raw
> low-rank families.

That is still not the same as benchmark-wide solved calibrated uncertainty,
but it is a real step toward it.
