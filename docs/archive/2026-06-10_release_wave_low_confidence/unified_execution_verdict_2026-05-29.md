# Unified Execution Verdict — object as currency

Date: `2026-05-29`

Direction chosen by the user: **convergence point — make the shared
posterior-family object the benchmark's native currency.** Not application-only
unification, not algorithm-only world-model framing; the single move that feeds
both.

This document is a diagnostic and a build plan. No feature code was written for
it. It exists to be read before the next coding session.

## 1. Why current unification is still rhetorical

There are two parallel pipelines in the repo that never touch each other.

### Pipeline A — benchmark spine

[scripts/80_release_partial_observation_benchmark.py](D:/release-foundation/scripts/80_release_partial_observation_benchmark.py)

- consumes `curves_long.csv` + optional `predictions.csv`
  (long-form `Q_pred`, optional `Q_lo90/Q_hi90/Q_lo50/Q_hi50`)
- emits curve / timescale / coverage tables
- knows nothing about `ReleasePosteriorFamily`
- the uncertainty it scores is **pre-baked CSV columns**, not a family object

### Pipeline B — family export leaves

- [85_release_posterior_family_export.py](D:/release-foundation/scripts/85_release_posterior_family_export.py)
  reads retained latent (`theta_*` or `alpha/beta`) → rank-0 point families.
  No decode, no curve, no metric.
- [86_release_casp_family_export.py](D:/release-foundation/scripts/86_release_casp_family_export.py)
  reloads bench62 bundles, refits RF, runs `fib_posterior` / `ensemble_posterior`
  → low-rank families. **Hardwired to 3 datasets**; a parallel reimplementation,
  not benchmark-wide.
- [87_release_conformal_family_export.py](D:/release-foundation/scripts/87_release_conformal_family_export.py)
  attaches script-76 conformal scales to 86 families. Selected cells only.

### The missing edge

`decoder_handle` is a string (`"PLGABiphasic.simulate_numpy"`). **No code
resolves that string and runs it.** The family → trajectory link is nominal.
Therefore the family object is a terminal leaf: nothing consumes it, and it is
never the thing scored by the benchmark.

That is the precise reason the capability snapshot keeps saying
"benchmark-wide calibrated family emission is still missing." It is not a
strategy gap. It is one missing function.

## 2. The keystone

Close exactly one loop:

```text
ReleasePosteriorFamily
   --(resolve decoder_handle; sample z = z_center + Σ_k N(0,1)·z_scale[k]·z_basis[k];
      clip to prior box; simulate each draw; pointwise center + quantiles)-->
predictions.csv  (the contract script 80 already eats)
   --(feed into 80)-->
curve + timescale + coverage metrics, derived FROM the family object
```

Once this edge exists:

- the family becomes the currency: every mechanism emits a family, one decoder
  renders it, one runner scores it
- uncertainty stops being pre-baked CSV columns and becomes **derived from
  family geometry**, so benchmark-wide coverage falls out automatically
- it serves both identities at once:
  application unification (one object across PLGA / liposome / chitosan) **and**
  Paper A (the feasible posterior family is the evaluated object, not a sidecar)

## 3. Minimal change set (surgical, two files)

### New file 1 — `release_family_decoder.py` (the only genuinely new logic)

- `DECODER_REGISTRY: dict[str, Callable]` mapping `decoder_handle` strings to
  the real bound simulator (`PLGABiphasic.simulate_numpy`,
  `WeibullSimulator.simulate_numpy`, ...). One source of truth for the string →
  callable resolution.
- `decode_family(family, t_grid, n_samples, seed) -> dict[str, np.ndarray]`
  returning `Q_pred` (center decode) and `Q_lo90/Q_hi90/Q_lo50/Q_hi50`
  (sample quantiles). Rank-0 families decode to a point curve with no intervals.
- if `family.calibration_context` carries the 87 interval scales, apply them in
  output space so calibrated families render calibrated intervals. Keep 87's
  honesty note intact: scales live in decoder-output space, latent geometry is
  unchanged.

### New file 2 — `scripts/89_family_to_benchmark.py` (thin wiring)

- input: a directory of family JSONs (from 85 / 86 / 87) + observed
  `curves_long.csv`
- call `decode_family` per curve → write `predictions.csv` in 80's contract
- invoke 80's existing functions → emit the standard metrics / timescale / uq
  tables, now provably **from the family object**
- no new metrics, no new schema, no mechanism-specific branches in the runner
  (only the registry resolves the decoder)

### What this explicitly does NOT touch

- ❌ no universal theta (schema stays mechanism-local)
- ❌ no latent-space recalibration (output-space conformal only, for now)
- ❌ no retraining, no new model zoo, no new mechanism
- ❌ no change to `release_posterior_family.py`, no change to 80's contract
- ❌ no ODE form / prior bound / t_grid change

## 4. Decision-type tagging (per CLAUDE.md §1)

- The decoder is an **inference / evaluation** decision (how to render a
  posterior family to predictive intervals). It touches no physics. Reversal
  cost is low. It does **not** trip the AGENTS.md "evidence before configuration
  change" gate.
- The one modeling choice with real consequence is the **sampling scheme**
  (Gaussian in the z-basis, clip to prior box). It moves coverage numbers, so it
  needs a diagnostic before being trusted benchmark-wide — see §5 anchor 1.

## 5. Verification criteria (goal-driven; the real verifiers, not green pytest)

The keystone is verified when all three hold:

1. **Consistency anchor.** Decoding the 87 calibrated families for
   `cross321 / group_by_drug` and `liposome / group_by_drug` through the new
   decoder → 80 path reproduces the already-audited
   `cov90_global ≈ 0.902 / 0.846` (within tolerance). If not, the
   sampling/decoding is wrong — not the benchmark.
   Source numbers:
   [outputs/88_release_uncertainty_snapshot/summary.txt](D:/release-foundation/outputs/88_release_uncertainty_snapshot/summary.txt)
2. **Currency proof.** PLGA and liposome families both render through the *same*
   decoder + *same* runner into the *same* metrics schema, with no
   mechanism-specific branch in the runner.
3. **Point-estimate degeneracy.** A rank-0 family from 85 decodes to a curve
   with no intervals whose `Q_pred` matches the route's original benchmark
   `Q_pred`. The object round-trips the existing point prediction.

If all three hold, this sentence becomes executable rather than rhetorical:

> drug release is unified at the posterior-family object level, benchmark-wide,
> with calibrated uncertainty.

That directly advances the still-open gates in the claim map: C2.3
(cross-mechanism uncertainty) and G4 (posterior-family object shared in code).
See [drug_release_claim_map_and_build_queue_2026-05-28.md](D:/release-foundation/docs/drug_release_claim_map_and_build_queue_2026-05-28.md).

## 6. What stays out of scope this round

- benchmark-wide CASP family construction (generalizing 86 off its 3 hardwired
  datasets) — valuable, but the keystone proves the loop first on existing
  families
- latent-space recalibration
- chitosan reveal (separate locked-prereg task)
- active measurement (Queue 5)

## Bottom line

The unification blueprint is complete and the object exists. The gap is one
edge: family → decoder → prediction contract → benchmark. Build
`release_family_decoder.py` + `scripts/89_family_to_benchmark.py`, verify
against the two audited coverage anchors, and "object as currency" stops being a
program statement and becomes a runnable fact.
