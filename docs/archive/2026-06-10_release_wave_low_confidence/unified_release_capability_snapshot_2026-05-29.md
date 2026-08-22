# Unified Release Capability Snapshot

Date: `2026-05-29`

Purpose:

```text
Answer the real strategic question:
what is the field doing now, what have we actually built, how far can we
honestly push the "unified drug-release" story, and where the competitors and
borrowing targets sit.
```

This is the current-state verdict, not a future wish list.

## 1. What the field is mostly doing now

The current external field is still dominated by:

1. `tabular descriptors -> release prediction`
2. `descriptors + a few early observations -> later release`
3. single-mechanism kinetic surrogates
4. data / schema / workflow standardization
5. broader formulation-decision platforms

The relevant external map is already written in:

- [drug_release_landscape_and_unification_2026-05-28.md](D:/release-foundation/docs/drug_release_landscape_and_unification_2026-05-28.md)
- [drug_release_competitor_capability_matrix_2026-05-28.md](D:/release-foundation/docs/drug_release_competitor_capability_matrix_2026-05-28.md)
- [drug_release_external_field_ledger_2026-05-29.md](D:/release-foundation/docs/drug_release_external_field_ledger_2026-05-29.md)
- [drug_release_algorithm_taxonomy_and_unification_bets_2026-05-29.md](D:/release-foundation/docs/drug_release_algorithm_taxonomy_and_unification_bets_2026-05-29.md)
- [drug_release_competitive_action_map_2026-05-29.md](D:/release-foundation/docs/drug_release_competitive_action_map_2026-05-29.md)
- [drug_release_paper_positioning_playbook_2026-05-29.md](D:/release-foundation/docs/drug_release_paper_positioning_playbook_2026-05-29.md)
- [drug_release_paper_spine_and_figure_map_2026-05-29.md](D:/release-foundation/docs/drug_release_paper_spine_and_figure_map_2026-05-29.md)
- [drug_release_title_abstract_and_caption_pack_2026-05-29.md](D:/release-foundation/docs/drug_release_title_abstract_and_caption_pack_2026-05-29.md)
- [drug_release_results_and_discussion_prose_pack_2026-05-29.md](D:/release-foundation/docs/drug_release_results_and_discussion_prose_pack_2026-05-29.md)
- [drug_release_introduction_prose_pack_2026-05-29.md](D:/release-foundation/docs/drug_release_introduction_prose_pack_2026-05-29.md)
- [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)

The short version is:

> the field is not yet organized around a shared partially observed
> release-intelligence framework.

That slot is still open.

## 2. What our current evidence now proves

The strongest compact evidence is now in:

- [summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt)
- [plga_signal_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_signal_snapshot.csv)
- [plga_theta_gain_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_theta_gain_snapshot.csv)
- [plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv)
- [plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv)
- [liposome_bridge_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_bridge_snapshot.csv)
- [liposome_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv)
- [prospective_status_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/prospective_status_snapshot.csv)
- [release_posterior_family_schema_2026-05-29.md](D:/release-foundation/docs/release_posterior_family_schema_2026-05-29.md)
- [summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)

### 2.1 PLGA audited core

From [summary.txt](D:/release-foundation/outputs/82_release_capability_snapshot/summary.txt):

- formulation-only OOD median best-`R²` range: `0.525` to `0.672`
- early-only OOD median best-`R²` range: `0.907` to `0.949`
- theta-minus-direct OOD gain range: `+0.002` to `+0.121`

Meaning:

- `early release` is the dominant OOD signal
- `mechanism route` still beats matched `direct-Q` on current audited PLGA tasks
- pure formulation prediction remains substantially weaker in harder OOD splits

### 2.1b Canonical PLGA OOD timescale panel

From [plga_timescale_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv):

- cells:
  - `cross321_group_by_drug_formulation_plus_early`
  - `cross321_group_by_polymer_formulation_plus_early`
  - `internal181_group_by_drug_formulation_plus_early`
  - `internal181_group_by_polymer_formulation_plus_early`

Panel tallies:

- mechanism route wins median curve `R²` in `4/4` cells
- mechanism route wins pooled `R²` in `1/4` cells
- mechanism route wins future-only `t10` MAE in `2/4` cells
- mechanism route wins future-only `t50` MAE in `1/4` cells
- mechanism route wins future-only `t80` MAE in `0/4` cells

Meaning:

- the mechanism route remains the more consistent winner on median per-curve
  resemblance
- the direct route is often stronger on pooled fit and future threshold timing
- the auto-selected PLGA route is already cell-dependent:
  - `cross321/group_by_drug` resolves to `RF_ztheta_leaf2`
  - the harder `group_by_polymer` / `internal181` cells resolve to
    `early_select`

This is stronger evidence than a single example because it shows the same
route-ranking divergence across multiple canonical OOD PLGA cells, not just
one hand-picked benchmark.

### 2.1c PLGA release-shape panel

From [plga_shape_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/plga_shape_snapshot.csv):

- mechanism route wins `burst` MAE in `3/4` cells
- mechanism route wins `post-window release` MAE in `4/4` cells
- mechanism route wins `residual-tail` MAE in `4/4` cells
- mechanism route wins `tail-AUC` MAE in only `1/4` cells

Meaning:

- once the object is "how much release remains after the observed prefix",
  the mechanism route looks stronger inside PLGA
- but once the object is an integrated tail-shape quantity, the advantage is
  no longer stable

So even inside PLGA, route superiority depends on whether the target is:

- curve resemblance
- threshold timing
- post-window / tail shape

### 2.2 First non-PLGA bridge: liposome

From [liposome_bridge_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/liposome_bridge_snapshot.csv):

Under `group_by_API`:

- pooled `R²`: mechanism `0.903` vs direct `0.864`
- median curve `R²`: mechanism `0.889` vs direct `0.867`
- `t10` MAE: mechanism `0.55 h` vs direct `1.05 h`
- `t50` MAE: mechanism `5.45 h` vs direct `2.22 h`
- `t80` MAE: mechanism `18.70 h` vs direct `34.34 h`

Under `group_by_release_method`:

- pooled `R²`: mechanism `0.903` vs direct `0.457`
- median curve `R²`: mechanism `0.889` vs direct `0.584`
- `t10` MAE: mechanism `0.55 h` vs direct `2.85 h`
- `t50` MAE: mechanism `5.45 h` vs direct `2.55 h`
- `t80` MAE: mechanism `18.70 h` vs direct `65.49 h`

Meaning:

- the shared release interface is no longer PLGA-only
- the mechanism route is more stable than direct-Q across liposome split families
- but timescale superiority is mixed, especially around `t50`
- shape-descriptor superiority is also split-dependent:
  - under `group_by_API`, direct-Q is better on all four shape descriptors
  - under `group_by_release_method`, the mechanism route is better on
    `burst`, `residual-tail`, and `tail-AUC`, while direct-Q remains better on
    `post-window release`

Taken together with the expanded PLGA panel, the honest cross-mechanism
curve/timescale/shape conclusion is:

> no single route has yet proven itself the universal winner on
> release-duration-and-shape prediction.

### 2.3 Prospective stress-test status

From [prospective_status_snapshot.csv](D:/release-foundation/outputs/82_release_capability_snapshot/prospective_status_snapshot.csv):

- status: `locked_prediction_pending_reveal`
- `12` curves
- `6` formulations
- `2` drugs
- prereg primary endpoint: `aggregate cov90 >= 0.83`

Meaning:

- we already have prospective evidence discipline
- we do **not** yet have completed prospective cross-mechanism validation

### 2.4 Shared object-level unification now exists in code

From:

- [release_posterior_family.py](D:/release-foundation/release_posterior_family.py)
- [summary.txt](D:/release-foundation/outputs/84_release_posterior_family_examples/summary.txt)
- [85_release_posterior_family_export.py](D:/release-foundation/scripts/85_release_posterior_family_export.py)

Current facts:

- a single executable `ReleasePosteriorFamily` schema now exists
- example instances have been emitted for:
  - `plga_biphasic`
  - `liposome_weibull`
  - `chitosan_ritger_peppas`
- benchmark-side point-estimate family exports now exist for:
  - four canonical PLGA mechanism-route cells
  - two liposome mechanism-route cells
- selected low-rank family exports now exist from current UQ machinery:
  - `FIB` on `cross321 / group_by_drug`
  - `FIB` on `liposome / group_by_drug`
  - `Ensemble` on `cross321 / random_5fold`
- selected conformal-attached family artifacts now also exist:
  - `cross321 / group_by_drug`
  - `liposome / group_by_drug`
- the shared part is the object contract, not a forced shared raw parameter
  space

Meaning:

- the project is now more unified than just "several benchmark scripts"
- object-level unification is no longer only a program statement; it is an
  executable artifact
- low-rank family export is no longer hypothetical, and selected `FIB`
  families can now carry split-conformal calibration metadata on the same
  object
- what is still missing is benchmark-wide calibrated family emission from the
  full benchmark / UQ stack rather than only selected or point-estimate
  exports

### 2.5 Shared uncertainty layer is now partially unified

From [summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt):

- raw `FIB` `cov90_mean` across current benchmark cells ranges from
  `0.720` to `0.913`
- raw `Ensemble` `cov90_mean` ranges from `0.430` to `0.750`
- a real calibrated-family panel now exists across `9` cells:
  - `cross321 / group_by_drug`
  - `cross321 / group_by_polymer`
  - `cross321 / random_5fold`
  - `internal181 / group_by_drug`
  - `internal181 / group_by_polymer`
  - `internal181 / random_5fold`
  - `liposome / group_by_drug`
  - `liposome / group_by_polymer`
  - `liposome / random_5fold`
- representative calibrated coverage levels in that panel include:
  - `cross321 / group_by_drug`: `cov90_global_mean = 0.902`
  - `cross321 / random_5fold`: `cov90_global_mean = 0.925`
  - `internal181 / group_by_polymer`: `cov90_global_mean = 0.946`
  - `liposome / random_5fold`: `cov90_global_mean = 0.923`

Meaning:

- the uncertainty layer is no longer only raw benchmark tables
- the project now has:
  - raw benchmark UQ evidence
  - recalibration summaries
  - calibrated family artifacts for a real `PLGA + liposome` panel
- this is stronger than saying UQ is still mostly a point-prediction sidecar
- it is still weaker than benchmark-wide or prospective cross-mechanism
  calibrated uncertainty

## 3. So can we try to unify all drug release?

`Yes, but not as one magical universal model.`

The evidence now supports unification at this level:

```text
shared benchmark contract
+ shared partially observed forecasting problem
+ shared release-state / posterior-family object
+ mechanism-specific decoders
+ shared curve / timescale / uncertainty reporting
```

This is the right bold claim.

The evidence does **not** support:

```text
one universal neural net
one universal theta shared literally across all mechanisms
fully solved cross-mechanism release prediction
```

So the honest high-ambition framing is:

> unified drug-release intelligence

not:

> universal drug-release foundation model

## 4. How bold can we be right now?

### Safe boldness

These are fair now.

1. We are moving beyond narrow release regressors toward a `shared partially observed release-intelligence framework`.
2. Our core scientific object is better described as `formulation/material + sparse early observations -> feasible mechanism state family -> future release`.
3. The shared interface already spans:
   - audited PLGA retrospective evidence
   - explicit PLGA future-only timescale and release-shape evidence
   - liposome retrospective bridge
   - shared posterior-family schema with PLGA / liposome / chitosan examples
   - locked chitosan prospective stress test

### Overreach

These are still too big.

1. `unified entire drug release is solved`
2. `mechanism route wins every metric across mechanisms`
3. `timescale prediction is already solved`
4. `uncertainty is solved across mechanisms`
5. `prospective cross-mechanism validation is complete`

## 5. Where the competitors still sit

The cleanest competitor map is still:

- [drug_release_competitor_capability_matrix_2026-05-28.md](D:/release-foundation/docs/drug_release_competitor_capability_matrix_2026-05-28.md)
- [drug_release_external_field_ledger_2026-05-29.md](D:/release-foundation/docs/drug_release_external_field_ledger_2026-05-29.md)

The short version:

### NC / Bannigan

They occupy:

- simpler few-shot / zero-shot deployment story
- direct release benchmark identity

We occupy:

- explicit mechanism state
- audited direct-Q vs mechanism-route comparison
- stronger route toward uncertainty and prospective discipline

Borrow:

- few-shot / zero-shot language
- simpler deployment framing

### Liposome IVR / accelerated IVR

They occupy:

- non-PLGA data ecosystem
- dataset packaging
- kinetic-summary discipline

We now borrow from them successfully, and they are no longer only an external
reference; they are now part of our internal bridge evidence.

### AI-ready release data / standardization work

They occupy:

- schema discipline
- interoperability mindset

Borrow:

- metadata-first federation
- explicit intake criteria

### FormulationAI-like platforms

They occupy:

- decision-support framing
- broader workflow value

Borrow:

- output-as-decision rather than output-as-only-prediction

## 6. What our actual position is now

The current strongest one-paragraph self-description is:

> The project is no longer just a PLGA predictor. It now has an executable
> shared benchmark layer, a first non-PLGA bridge through liposome, a locked
> prospective chitosan stress test, and a consistent result that sparse early
> release observations are the main OOD signal while mechanism-routed
> prediction remains stronger than matched direct curve regression on the
> audited PLGA core. The evidence supports a bold move toward unified
> drug-release intelligence, but not yet a claim of universal cross-mechanism
> release modeling.

## 7. What is missing before the big story becomes truly hard to dismiss

Three things still matter most:

1. `uncertainty-aware duration / shape layer`
   because both PLGA and liposome now show that better curve fit does not
   automatically mean better release-duration or tail-shape prediction, and
   current shared evidence is still mostly point-estimate based

2. `cross-mechanism uncertainty layer`
   because calibrated-family unification now exists for a real but still
   incomplete panel, not yet benchmark-wide or prospectively

3. `chitosan reveal`
   because locked prospective evidence is the hardest credibility upgrade left

## Bottom line

`Yes, we can and should be bolder.`

But the boldness has to be at the right layer:

- bold about the problem definition
- bold about the shared release interface
- bold about partially observed release intelligence
- not yet bold about a universal solved model

That is the current maximum honest ambition supported by evidence.
