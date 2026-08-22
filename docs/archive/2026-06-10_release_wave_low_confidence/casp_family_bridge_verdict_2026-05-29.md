# CASP Family Bridge Verdict

Date: `2026-05-29`

Purpose:

```text
Record what became true once the shared posterior-family schema was connected
to existing FIB-CASP and Ensemble-CASP machinery.
```

Key artifacts:

- [86_release_casp_family_export.py](D:/release-foundation/scripts/86_release_casp_family_export.py)
- [87_release_conformal_family_export.py](D:/release-foundation/scripts/87_release_conformal_family_export.py)
- [FIB cross321 / group_by_drug](D:/release-foundation/outputs/86_release_casp_family_export/fib/cross321/group_by_drug)
- [FIB liposome / group_by_drug](D:/release-foundation/outputs/86_release_casp_family_export/fib/liposome/group_by_drug)
- [Ensemble cross321 / random_5fold](D:/release-foundation/outputs/86_release_casp_family_export/ensemble/cross321/random_5fold)
- [Conformal family: cross321 / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug)
- [Conformal family: liposome / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_drug)

## 1. What changed

Before this step, the project had:

1. a shared schema
2. example objects
3. benchmark-side point-estimate exports

After this step, it also has:

4. selected low-rank family exports from real uncertainty constructors
5. selected low-rank family artifacts with attached split-conformal scales

That is a real shift. The shared object is no longer just an abstract wrapper
around point predictors.

## 2. The three object levels are now distinct

### Level A — schema examples

These prove the object shape is mechanism-agnostic.

Source:

- [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)

### Level B — benchmark-side point families

These prove current mechanism routes can emit a shared object even when they
only retain a latent center and no honest low-rank spread.

Source:

- [outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2](D:/release-foundation/outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2)
- [outputs/85_liposome_group_by_api_our_et_refined](D:/release-foundation/outputs/85_liposome_group_by_api_our_et_refined)

### Level C — low-rank CASP families

These prove the existing UQ machinery can also land in the same object:

- `FIB` family:
  `mu + U + sigma + inactive-scale`
- `Ensemble` family:
  `mu + U + sigma + tree-spectrum`

Source:

- [outputs/86_release_casp_family_export/fib/cross321/group_by_drug](D:/release-foundation/outputs/86_release_casp_family_export/fib/cross321/group_by_drug)
- [outputs/86_release_casp_family_export/fib/liposome/group_by_drug](D:/release-foundation/outputs/86_release_casp_family_export/fib/liposome/group_by_drug)
- [outputs/86_release_casp_family_export/ensemble/cross321/random_5fold](D:/release-foundation/outputs/86_release_casp_family_export/ensemble/cross321/random_5fold)

### Level D — calibrated family artifacts

These prove that conformal interval calibration can now be serialized onto the
same shared family object without pretending the latent geometry itself was
retrained or recalibrated.

Source:

- [outputs/87_release_conformal_family_export/cross321/group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug)
- [outputs/87_release_conformal_family_export/liposome/group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_drug)

## 3. Why this matters scientifically

This means the project can now honestly say:

> the common object is not just a point latent state. It can already carry
> low-rank feasible release-state structure from multiple uncertainty
> constructions, plus post-hoc calibrated interval scales in release space.

That is a much stronger version of "unified release intelligence" than:

- one benchmark API
- one set of summary tables
- one schema demo

## 4. What still blocks a bigger claim

Four limits remain real.

1. `selected cells only`
   the low-rank family bridge is not yet emitted benchmark-wide

2. `selected cells only for conformal attachment`
   conformal write-back now exists, but only for selected `FIB` cells rather
   than the full benchmark/UQ stack

3. `constructor heterogeneity`
   FIB and ensemble disagree in what their spread means
   - local curve-conditioned identifiability
   - formulation-conditioned disagreement

4. `no prospective calibrated-family proof`
   chitosan is still pending reveal

## 5. Bottom line

The project has crossed an important line:

> unified release intelligence is no longer only a shared benchmark idea or a
> shared schema idea. It now has a shared low-rank family object that multiple
> existing uncertainty constructors can populate, and selected FIB families can
> already carry split-conformal interval calibration as attached object-level
> context.

That is real progress toward unifying the release problem at the object level.

But the honest stopping point is still:

> partial unification of posterior-family structure plus selected calibrated
> interval attachment

not:

> solved cross-mechanism calibrated uncertainty.
