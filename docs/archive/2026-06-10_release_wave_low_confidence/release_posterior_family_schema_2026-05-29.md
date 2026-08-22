# Release Posterior-Family Schema

Date: `2026-05-29`

Purpose:

```text
Define the shared executable object that makes
"unified drug-release intelligence" concrete at the object level.
```

Key code:

- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [84_release_posterior_family_examples.py](D:/release-foundation/scripts/84_release_posterior_family_examples.py)
- [85_release_posterior_family_export.py](D:/release-foundation/scripts/85_release_posterior_family_export.py)
- [86_release_casp_family_export.py](D:/release-foundation/scripts/86_release_casp_family_export.py)
- [outputs/84_release_posterior_family_examples](D:/release-foundation/outputs/84_release_posterior_family_examples)

## 1. Why this object matters

The project's boldest honest claim is not:

> one universal theta shared literally across every release mechanism

It is:

> one shared posterior-family object that can carry feasible release-state
> uncertainty across mechanisms, while delegating actual trajectory decoding to
> mechanism-specific simulators or kinetic adapters.

That is exactly what the code object now does.

## 2. Schema

The executable object is `ReleasePosteriorFamily`:

```python
ReleasePosteriorFamily(
    mechanism_id: str,
    decoder_handle: str,
    z_center: list[float],
    z_basis: list[list[float]],
    z_scale: list[float],
    assay_context: dict[str, Any],
    observation_context: dict[str, Any],
    calibration_context: dict[str, Any],
    metadata: dict[str, Any],
)
```

Meaning:

- `mechanism_id`
  identifies the mechanism family, such as `plga_biphasic` or
  `liposome_weibull`
- `decoder_handle`
  tells downstream code which decoder or simulator should turn latent state
  into release trajectories
- `z_center`
  is the center of the feasible release-state family
- `z_basis` and `z_scale`
  define a low-rank family around that center
- `assay_context`
  carries assay-specific context like time unit and early window
- `observation_context`
  records how the family was inferred under partial observation
- `calibration_context`
  records post-hoc interval calibration attached in decoder-output space
- `metadata`
  stores non-core context without hard-coding it into the main schema

## 3. What is shared and what is not

Shared across mechanisms:

- object shape
- serialization
- validation
- downstream contract for uncertainty / decoding / decision support

Not forced to be shared across mechanisms:

- physical meaning of latent coordinates
- parameter dimension
- simulator equation
- assay timescale

That boundary is the key reason this schema is honest.

## 4. Example mechanism instances

The example exporter now writes three concrete JSON objects:

- [plga_biphasic_family.json](D:/release-foundation/outputs/84_release_posterior_family_examples/plga_biphasic_family.json)
- [liposome_weibull_family.json](D:/release-foundation/outputs/84_release_posterior_family_examples/liposome_weibull_family.json)
- [chitosan_ritger_peppas_family.json](D:/release-foundation/outputs/84_release_posterior_family_examples/chitosan_ritger_peppas_family.json)

Summary:

- [summary.txt](D:/release-foundation/outputs/84_release_posterior_family_examples/summary.txt)
- [summary.csv](D:/release-foundation/outputs/84_release_posterior_family_examples/summary.csv)

These examples demonstrate:

1. `PLGA` can use a `theta-like kinetic state`
2. `Liposome` can use a `log_alpha / log_beta` latent state
3. `Chitosan` can use a mechanism-specific low-rank scaffold
4. all three still serialize into the same executable contract

## 5. What this now enables

This is enough to make the following sentence true in code, not just prose:

> drug release can be unified at the posterior-family object level while
> keeping mechanism-specific decoders.

It is now also true one step deeper in the stack:

- example objects exist for `PLGA`, `liposome`, and `chitosan`
- benchmark-side point-estimate exports now exist for mechanism routes:
  - [85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2](D:/release-foundation/outputs/85_plga_cross321_group_by_drug_theta_rf_ztheta_leaf2)
  - [85_plga_cross321_group_by_polymer_theta_early_select](D:/release-foundation/outputs/85_plga_cross321_group_by_polymer_theta_early_select)
  - [85_plga_internal181_group_by_drug_theta_early_select](D:/release-foundation/outputs/85_plga_internal181_group_by_drug_theta_early_select)
  - [85_plga_internal181_group_by_polymer_theta_early_select](D:/release-foundation/outputs/85_plga_internal181_group_by_polymer_theta_early_select)
  - [85_liposome_group_by_api_our_et_refined](D:/release-foundation/outputs/85_liposome_group_by_api_our_et_refined)
  - [85_liposome_group_by_method_our_et_refined](D:/release-foundation/outputs/85_liposome_group_by_method_our_et_refined)
- low-rank family exports now also exist directly from current UQ machinery:
  - [FIB cross321 / group_by_drug](D:/release-foundation/outputs/86_release_casp_family_export/fib/cross321/group_by_drug)
  - [FIB liposome / group_by_drug](D:/release-foundation/outputs/86_release_casp_family_export/fib/liposome/group_by_drug)
  - [Ensemble cross321 / random_5fold](D:/release-foundation/outputs/86_release_casp_family_export/ensemble/cross321/random_5fold)
- selected conformal-attached family exports now also exist:
  - [cross321 / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/cross321/group_by_drug)
  - [liposome / group_by_drug](D:/release-foundation/outputs/87_release_conformal_family_export/liposome/group_by_drug)

This means the shared object now spans three levels:

1. `schema examples`
2. `benchmark-side point estimates`
3. `selected low-rank family exports from FIB / ensemble CASP`
4. `selected low-rank families with attached split-conformal interval scales`

What it does **not** yet enable by itself:

- benchmark-wide calibrated family emission for every release cell
- latent-space recalibration of `z_center / z_basis / z_scale`
- proven cross-mechanism uncertainty calibration
- universal release-duration superiority
- automatic decoder interchangeability
- leave-one-mechanism-out success

## 6. Bottom line

This schema is the current best answer to:

> can we really try to unify the whole drug-release problem?

Answer:

> yes, at the object-contract level first.

That is a stronger and more executable kind of boldness than just claiming
the eventual existence of a universal model.
