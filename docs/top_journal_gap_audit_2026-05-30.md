# Top-Journal Gap Audit — 2026-05-30

Purpose:
- audit the current repo against the locked closeout + roadmap
- separate what is already evidenced from what is still missing
- give an execution order that improves publication readiness without reopening banned scope

Authority order for this audit:

1. durable output files in `outputs/` / `outputs_active_observer_v3/`
2. `docs/drug_release_closeout_2026-05-28.md`
3. `docs/top_journal_plan_2026-05-30.md`
4. `docs/paper_requirements_locked_2026-05-28.md`

This file is an audit, not a new source of truth.

## 1. Current headline object

The paper object is still the closeout object:

```text
formulation + sparse early release
    -> feasible mechanism state / theta family
    -> simulator decoder
    -> full-trajectory prediction
    (+ optional posterior/UQ layer)
```

Not headline:

- world model
- RSSM
- Active Observer as best predictive algorithm
- stack / positioning / unified-intelligence narrative

## 2. Four pillars: current status

| Pillar | Status | Strongest current evidence | What is still missing |
|---|---|---|---|
| P1 early release is the dominant OOD signal | `DONE` | `outputs/45_input_source_ablation/input_source_summary.csv` | same-mouth CI table if needed for submission polish |
| P2 theta middle layer beats matched direct-Q controls | `DONE, but needs same-mouth comparator closure` | `outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv`, `outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv` | canonical NPE/SNRE comparator and refreshed CI |
| P3 posterior family is calibrated, not just wide | `PARTIAL` | `outputs/62_fib_casp_benchmark/aggregate_table.csv`, `outputs/66_ensemble_casp/aggregate_table.csv`, `outputs/76_casp_conformal_recalibration/` | SBC on the actual canonical method + sharpness + naive-band comparison |
| P4 preregistered prospective cross-mechanism evidence | `LOCKED BUT BLOCKED ON WET REVEAL` | `outputs/67_chitosan_prospective/lock_metadata.json`, prereg doc at `docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md` | real wet-lab CSV and preregistered `75` evaluation |

Bottom line:

- P1 and the main P2 argument already exist.
- P3 is not strong enough yet for top-tier review.
- P4 is externally blocked by the wet-lab reveal.

## 3. Evidence-backed repo status by requirement

### A3 canonical split closure

Status: `FUNCTIONALLY CLOSED VIA 73b, NOT YET CONVERGED IN FILE CHOICE`

Evidence:

- `data/canonical_split_v1.csv` exists.
- `data/canonical_split_v1.csv` currently contains `train=89`, `cal=23`, `test=38`.
- `scripts/make_canonical_split.py` emits that file.
- `scripts/72_canonical_benchmark_v2.py` has been patched in the current worktree
  to read `data/canonical_split_v1.csv` instead of re-deriving the split.
- `scripts/73b_diagnostics_canonical_split.py` now exists as a safe split-fixed
  diagnostic companion that does not overwrite the user-modified `73`.
- `73b` has now been runtime-validated on CPU and reports:
  - bootstrap delta RMSE (Direct-Q - Active) = `0.0258`
  - 95 % CI = `[-0.0038, 0.0555]` → not significant
  - pooled R²: `Direct-Q = 0.5115`, `Active = 0.6204`
  - regime verdict: `GRU adds little (NO CLAIM)`
- `scripts/73_diagnostics.py` still re-derives splits with `train_test_split`.

Implication:

- the main corrected benchmark has moved onto the fixed split file.
- there is now a safe, actually executed path to run split-aligned diagnostics
  without touching the user-modified `73`.
- scientifically, the fixed-split benchmark + diagnostic mouth now exists and
  runs (`72_v2` + `73b`).
- the remaining A3 gap is governance / file convergence, not missing code:
  the repo still needs to decide whether `73b` is the accepted replacement or
  whether the user-modified `73` will be patched later.

### B4 NPE / SNRE same-mouth comparator

Status: `PARTIAL RUNTIME EVIDENCE, NOW INCLUDES FIRST SAME-MOUTH SNRE READOUT`

Evidence:

- scripts exist: `27_partial_curve_npe.py`, `27c_sbi_npe_partial.py`, `27d_residual_npe.py`
- output exists: `outputs/27_partial_curve_npe_v1/`
- those scripts are primarily framed around external-321 suffix prediction from
  partial real curves, not the internal canonical benchmark mouth used by `72_v2`
- `scripts/27e_internal_canonical_npe_fixed_eval.py` now exists as the first
  internal same-mouth evaluation scaffold for the trained `27c` posterior
- `27e` has now passed CPU runtime validation on the fixed canonical split
  and wrote smoke / medium-size outputs at
  `outputs/27e_internal_canonical_npe_fixed_eval_smoke/` and
  `outputs/27e_internal_canonical_npe_fixed_eval_cpu128/`
- current medium-size CPU result (`n_posterior_samples=128`) is:
  `rmse=0.2320`, `pooled_r2=0.1531`, with paired comparison against
  `72_v2` `DirectQ-fixed` showing `delta_rmse=-0.0589`
  (95 % CI `[-0.0948, -0.0265]`)
- the same harness now also reconstructs and evaluates `27d` on the internal
  canonical mouth; medium-size CPU output lives at
  `outputs/27e_internal_canonical_npe_fixed_eval_27d_cpu128/`
- current `27d` medium-size CPU result is:
  `rmse=0.2229`, `pooled_r2=0.2188`, with paired comparison against
  `72_v2` `DirectQ-fixed` showing `delta_rmse=-0.0503`
  (95 % CI `[-0.0810, -0.0167]`)
- `scripts/27g_internal_canonical_snre_eval.py` now exists and has produced the
  first same-mouth internal SNRE readout on the canonical split
- current quick directional SNRE result
  (`outputs/27g_internal_canonical_snre_eval/`, `n_sim=4000`,
  `n_posterior_samples=48`, rejection posterior sampling) is:
  `rmse=0.2165`, `r2_pooled=0.2628`
- paired bootstrap vs `72_v2` shows:
  - `SNRE-fixed7d` is clearly worse than `DirectQ-fixed`
    (`delta_rmse=+0.0435`, 95 % CI `[+0.0208, +0.0658]`)
  - clearly worse than `DirectQ-adaptive`
    (`delta_rmse=+0.0492`, 95 % CI `[+0.0268, +0.0709]`)
  - clearly worse than `Active-adaptive`
    (`delta_rmse=+0.0476`, 95 % CI `[+0.0173, +0.0704]`)
  - not clearly separated from `Active-fixed`
    (`delta_rmse=+0.0173`, 95 % CI `[-0.0058, +0.0396]`)
- switching the posterior sampler from rejection to importance does not change
  the directional conclusion:
  - quick importance run (`n_sim=4000`, `n_posterior_samples=48`) improves
    slightly to `rmse=0.2061`, `r2_pooled=0.3318`, but remains clearly worse
    than `DirectQ-fixed`, `DirectQ-adaptive`, and `Active-adaptive`
  - a larger importance run (`n_sim=8000`, `n_posterior_samples=64`) regresses
    to `rmse=0.2211`, `r2_pooled=0.2308`
- roadmap explicitly says these runs are not on the canonical split and not on the same evaluation mouth as the corrected benchmark

Implication:

- this is not from-zero work
- but it is still a real top-journal gap because the current evidence does not answer K4
- practically, B4 likely needs an adapter or a new benchmark harness rather than
  a trivial one-line split swap inside `27c` / `27d`
- `27e` is no longer just a scaffold; it now gives the first honest same-mouth
  readout for `27c`, and now for `27d` as well
- both same-mouth readouts currently point the same way: on the current fixed
  4-point canonical mouth, `DirectQ-fixed` still outperforms both `27c` and
  `27d`
- `27d` is directionally better than `27c`, so the residual does recover some
  of the gap, but not enough to overturn the comparator
- the remaining B4 question is therefore narrower and more honest:
  not "will SNRE magically rescue the story?", but whether a higher-budget
  SNRE run is still worth comparator time after the first directional readout
  already lands well below the current direct-Q / adaptive headline methods;
  current evidence suggests diminishing upside rather than a hidden win

### B5 HMC ground-truth posterior

Status: `ACTIVE POSTERIOR AUDIT; HMC FALLBACK EXISTS BUT CURRENT CHAINS ARE NOT YET MIXED`

Evidence:

- no `numpyro`, `pyro`, `pymc`, `blackjax`, `emcee`, `NUTS`, or `Hamiltonian`
  code references are present in the repo
- the current Python environment also fails to import those sampler stacks, so
  B5 cannot be executed yet without a new dependency route
- roadmap explicitly marks B5 as from-zero work
- `scripts/27f_export_hmc_subset.py` now exists and exports a locked HMC input
  package without choosing a sampler implementation
- `scripts/27h_b5_logposterior_backend.py` now exists and defines the
  sampler-agnostic fixed-four canonical-mouth log posterior on top of the
  locked 27f subset package
- `scripts/27i_b5_hmc_pyro_runner.py` now exists as the first concrete HMC/NUTS
  runner entrypoint, written to lazy-import `pyro-ppl` once the dependency is
  approved and installed
- `scripts/27j_b5_hmc_torch_runner.py` now exists as a dependency-free fallback
  HMC runner on top of the same unconstrained 9D torch potential
- primary claim-facing subset is now locked as all `38` canonical test curves:
  `outputs/27f_hmc_subset_export_test/`
- a deterministic 50-curve development package also now exists:
  `outputs/27f_hmc_subset_export_testpluscal12/`
  with `38` claim-test curves + `12` calibration extensions
- the extension rule is explicit and reproducible:
  prefer calibration `DP_Group`s not already represented in test, then fill
  the remaining slots in canonical `fold_index` order
- both export packages include:
  `subset_manifest.csv`, `formulations.csv`, `curves_long.csv`,
  `curve_grid_14pt.csv`, `fixed_four_mouth.csv`, `theta_reference.csv`,
  and `lock_metadata.json`
- the 27h backend has now passed a real optimization smoke test on
  `curve_id=11`:
  `outputs/27h_b5_logposterior_backend_smoke/map_smoke_summary.json`
  shows a finite MAP estimate and finite log posterior under the fixed-four
  observation model
- the same backend has now also passed a differentiable torch-potential smoke
  test on `curve_id=11`:
  `outputs/27h_b5_logposterior_backend_torch_smoke/torch_smoke_summary.json`
  shows finite potential and finite gradients for the unconstrained 9D
  parameterization that future HMC/NUTS samplers would use
- the dependency-free HMC fallback has now produced real posterior draws on
  locked claim-test curves:
  - `outputs/27j_b5_hmc_torch_runner_smoke/summary_curve_11.json`
  - `outputs/27j_b5_hmc_torch_runner_smoke_curve20/summary_curve_20.json`
- these are only smoke runs (`curve_id=11` with 40 post-warmup draws,
  `curve_id=20` with 30 post-warmup draws), but they establish that B5 is no
  longer blocked on dependency installation for first posterior evidence
- a first posterior-agreement comparison against the amortized `27c` posterior
  now also exists:
  `outputs/27k_b5_hmc_vs_amortized_compare/` and the tuned rerun at
  `outputs/27k_b5_hmc_vs_amortized_compare_tuned008/`
- using the more informative tuned HMC smoke runs (`step_size=0.08`), the
  mean 90 % interval-overlap fraction between HMC and amortized `27c`
  posteriors is still low on the two tested claim-test curves:
  - curve `11`: `0.257`
  - curve `20`: `0.168`
  - overall mean across the two curves: `0.213`
- the worst disagreements are not random small shifts; several parameters show
  zero interval overlap even after the tuned rerun, especially
  `log_kw`, `log_alpha`, and `log_tau_burst`
- this has now been extended into a deterministic four-curve small-panel audit
  via `scripts/27l_b5_small_panel_audit.py`, with outputs at
  `outputs/27l_b5_small_panel_audit/`
- current panel covers four distinct claim-test chemistries:
  `11 (DEX-PLGA)`, `17 (QRC-PCL)`, `20 (PTX-PVL-co-PAVL)`, `34 (TAA-PLGA)`
- panel-level posterior-agreement summary remains poor:
  - mean interval-overlap fraction by curve:
    - `11`: `0.332`
    - `17`: `0.131`
    - `20`: `0.105`
    - `34`: `0.298`
  - overall panel mean: `0.217`
  - worst parameter overlap fraction: `0.0`
- repeated zero-overlap failure modes now appear across the panel rather than
  only on one curve; examples include:
  - curve `11`: `log_alpha`, `log_tau_burst`
  - curve `17`: `log_kw`, `m_crit`, `q_burst`
  - curve `20`: `log_kw`, `log_alpha`, `log_kd`
  - curve `34`: `m_crit`, `q_burst`, `Q_max`
- however, longer hand-written HMC chains materially soften that picture:
  `outputs/27k_b5_hmc_vs_amortized_compare_long_panel_s008/`
  on the same four-curve panel gives:
  - curve `11`: mean overlap `0.478`
  - curve `17`: mean overlap `0.543`
  - curve `20`: mean overlap `0.624`
  - curve `34`: mean overlap `0.751`
  - overall panel mean: `0.599`
  - worst parameter overlap fraction: `0.110`
- so the strongest current B5 readout is no longer "posterior disagreement is
  obviously catastrophic", but the more careful statement:
  short chains exaggerated the mismatch, yet even longer chains still leave
  non-trivial parameter disagreements on some curves, especially around
  `log_kd`, `log_kw`, `m_crit`, and `log_tau_burst`
- `scripts/27m_b5_chain_diagnostics.py` now exists as the first multi-chain
  quality gate for the dependency-free HMC fallback
- `27m` now also writes `per_chain_diagnostics.csv`, so chain-level pathologies
  are no longer hidden behind one aggregate R-hat number
- first split-R-hat readouts on three representative claim-test curves are:
  - `outputs/27m_b5_chain_diagnostics_curve11_s008/per_curve_summary.csv`
    - curve `11` (`DEX-PLGA`): mean accept `0.9875`, max split-R-hat `1.852`
  - `outputs/27m_b5_chain_diagnostics_curve20_s008/per_curve_summary.csv`
    - curve `20` (`PTX-PVL-co-PAVL`): mean accept `0.9750`, max split-R-hat `3.641`
  - `outputs/27m_b5_chain_diagnostics_curve34_s008/per_curve_summary.csv`
    - curve `34` (`TAA-PLGA`): mean accept `1.0000`, max split-R-hat `1.745`
- parameter-level failures are not isolated to one nuisance dimension:
  - curve `11`: worst at `log_tau_burst = 1.852`
  - curve `20`: worst at `log_ke = 3.641`, with `log_tau_burst = 2.436`
  - curve `34`: worst at `log_tau_burst = 1.745`
- the new chain-level diagnostic table already clarifies one concrete failure
  mode on `curve 20`:
  `outputs/27m_b5_chain_diagnostics_curve20_chaincsv_s008/per_chain_diagnostics.csv`
  shows both chains ending with `final_step_size ~= 0.1059`,
  `accept_rate_post = 1.0`, and tiny post-warmup energy error, while the
  same run still has `max split-R-hat = 2.968`; this is consistent with
  sticky, under-exploratory chains rather than with a stable reference
  posterior
- a fourth curve (`17`, `QRC-PCL`) was started but not completed within the
  current runtime budget, so no honest multi-chain conclusion is recorded yet
- these R-hat values are not close to a credible convergence threshold, and the
  near-1.0 accept rates suggest the current hand-written HMC configuration is
  still too conservative / sticky to support strong posterior-geometry claims
- a small step-size sweep now confirms that chain quality is tunable, but not
  with one obvious global fixed step size:
  - curve `20` improves steadily as the initial step size increases
    (`0.08 -> 0.12 -> 0.16 -> 0.20` gives
    `max split-R-hat 3.641 -> 2.233 -> 1.441 -> 1.314`)
  - curve `34` improves from `0.08` to `0.16`
    (`1.745 -> 1.661 -> 1.402`) and then degrades slightly at `0.20`
    (`1.509`)
  - curve `11` improves at `0.12` (`1.604` vs `1.852` at `0.08`) but gets
    worse at `0.16` (`2.386`)
- this is useful because it argues against a simple "just raise the default
  step size" fix; the fallback HMC appears curve-sensitive, which in turn
  strengthens the case for a standard adaptive NUTS implementation if B5 needs
  submission-grade posterior auditing
- an attempted upgrade to a more aggressive automatic step-size initialization
  was not retained as the default because it did not deliver a robust
  improvement across representative curves; the current code keeps the simpler
  baseline adaptation and treats more advanced adaptation as future work

Implication:

- this is no longer "nothing started"
- the subset and evaluation mouth are now frozen independently of the future
  sampler choice, which removes one source of benchmark drift
- the log-prior / log-likelihood / log-posterior layer is also now frozen
  enough for future HMC/NUTS code to call directly
- a concrete runner path also now exists, so the remaining B5 blocker is not
  "design the benchmark" or "figure out the math"
- dependency approval is no longer the blocker for first B5 evidence; the
  remaining question is whether to scale the hand-written HMC fallback further
  or to approve `pyro-ppl` for a more standard NUTS implementation
- the first agreement readout is already scientifically informative, but the
  new chain-quality evidence shows those overlap numbers are still limited by
  poor mixing rather than being ready for strong posterior-geometry claims
- in either case, B5 has now moved from "missing comparator" to
  "active posterior-disagreement audit"
- the small-panel extension makes it harder to dismiss this as a single-curve
  accident, but the longer-chain follow-up also shows the first severe
  disagreement readout was overstated by chain quality
- the current honest position is:
  B5 already raises a real calibration / posterior-geometry question for the
  amortized posterior, but it does not yet prove a dramatic failure, and it
  also does not yet support a clean "HMC agrees" reassurance
- paper-facing claims should still be reported on the `38` canonical test
  curves; the 50-curve package is for development / runtime amortization only

### B6 / B7 fPCA + mean baseline in current table

Status: `CANONICAL RUNTIME EVIDENCE NOW EXISTS`

Evidence:

- `scripts/12_baseline_comparators.py` contains an fPCA comparator route
- `outputs/12_baseline_comparators/` exists
- current roadmap still asks to bring fPCA + mean baseline into the main benchmark table via `scripts/38_prediction_baselines.py`
- `scripts/38b_canonical_functional_baselines.py` now exists as a clean
  canonical-mouth baseline layer rather than mutating the older random-CV
  script history
- `38b` runs on the same fixed split and future evaluation times as
  `72_canonical_benchmark_v2.py`
- current canonical results are:
  - `GlobalMean`: `rmse=0.2451`, `r2_pooled=0.0548`
  - `fPCA-Ridge`: `rmse=0.1695`, `r2_pooled=0.5478`
- paired bootstrap comparisons against the current `72_v2` table are:
  - `GlobalMean` is clearly worse than every main benchmark method
  - `fPCA-Ridge` is statistically tied with `DirectQ-fixed`,
    `DirectQ-adaptive`, and `Active-adaptive`
  - `fPCA-Ridge` is also not clearly worse than `Active-fixed`

Implication:

- B7 is now cleanly closed as a true sanity floor on the canonical mouth
- B6 is also no longer missing; there is now an honest functional-regression
  comparator on the canonical mouth
- this is scientifically useful but strategically uncomfortable:
  the generic `fPCA-Ridge` baseline is not getting cleanly separated from the
  current headline methods on this benchmark, so future claims about the
  necessity of the mechanism layer must stay narrow and evidence-backed

### S6 SBC on canonical posterior

Status: `PARTIAL RUNTIME EVIDENCE`

Evidence:

- SBC scripts exist: `05_sbc_curve_posterior.py`, `27b_widened_prior_sbc.py`
- historical output folders exist: `outputs/05_sbc*`, `outputs/27b_widened_prior_sbc`
- roadmap explicitly says these are not yet running on the canonical paper posterior
- `scripts/69b_active_observer_v3_sbc.py` now exists and runs an honest
  empirical-particle SBC for the Active Observer v3 posterior on the canonical
  split test formulations
- medium CPU run output now exists at
  `outputs/69b_active_observer_v3_sbc_active2_cpu100/`
- current `active_two_point` SBC result on `n_sbc=100` passes `8 / 9` theta
  dimensions by KS `p > 0.05`
- the clear failure dimension is `Q_max`:
  `ks_stat = 0.5951`, `ks_pvalue = 2.5e-34`,
  `mean_normalized_rank = 0.2475`
- targeted ablations now exist via `69b` CLI overrides:
  - removing `q_max_correction` and `q_max_cap_margin` only partially helps
    `Q_max` under `active_two_point`
    (`ks_stat = 0.4037`, `p = 1.9e-3`, mean rank `0.2537`)
  - reducing particle count does not remove the `active_two_point` scar
    (`ks_stat = 0.3877`, `p = 3.2e-3`, mean rank `0.3123`)
  - switching to `fixed_four_point` with low-particle diagnostic runs
    materially improves `Q_max` calibration
    (`ks_stat = 0.2907`, `p = 0.0543`, mean rank `0.4019`)
  - a step-2-specific ablation is even more diagnostic: keeping the global
    default `q_max_cap_margin=0.05` but forcing the second update only to
    `0.0` moves `Q_max` from clear failure to approximate pass in the same
    30x80 diagnostic regime
    (`ks_stat: 0.3877 -> 0.1543`, `p: 1.4e-4 -> 0.429`,
     mean rank: `0.368 -> 0.514`, mean bias: `+0.026 -> -0.003`)
  - the larger like-for-like 60x100 rerun keeps the same direction but with a
    more conservative conclusion: step-2-no-cap materially reduces the
    `Q_max` scar without fully eliminating it
    (`ks_stat: 0.5401 -> 0.3068`, `p: 6.5e-17 -> 1.6e-5`,
     mean rank: `0.303 -> 0.392`, mean bias: `+0.038 -> +0.021`)
  - the high-budget 100x200 confirmation keeps the same conclusion:
    `step2-no-cap` is a substantial improvement over the base
    `active_two_point` posterior but still not a clean SBC pass
    for `Q_max`
    (`ks_stat: 0.5951 -> 0.3051`, `mean rank: 0.248 -> 0.408`,
     mean bias: `+0.038` on the 60x100 base vs `+0.015` on the 100x200
     step2-no-cap run)
  - an `active_one_point` 100x200 SBC run now sharpens the causal picture:
    `Q_max` is well calibrated after the first update alone
    (`ks_stat = 0.1078`, `p = 0.182`, mean rank `0.525`, mean bias `-0.002`)
    while `log_tau_burst` is already the main remaining one-point scar
  - a `prior_only` 100x200 control closes the causal loop:
    the formulation-conditioned empirical particle prior itself is well
    calibrated across all 9 dimensions (`9 / 9` pass), including
    `Q_max`, `q_burst`, and `log_tau_burst`
  - a production-side ablation path now exists in
    `scripts/69_active_observer_v3.py` via `--step2-q-max-cap-margin`
    without changing the default behavior
  - running the actual AO evaluation mouth with
    `--step2-q-max-cap-margin 0.0` produced
    `outputs_active_observer_v3_step2nocap/` and did not show a predictive
    regression for `active_two_point_conformal`; instead it improved:
    `rmse_mean 0.1380 -> 0.1250`, `crps 0.1118 -> 0.0988`,
    `coverage90 0.9298 -> 0.9035`, `width90 0.4996 -> 0.5045`

Implication:

- again, not from-zero anymore
- the posterior calibration story is now more concrete and more honest:
  most dimensions look approximately calibrated under the empirical particle
  prior, but `Q_max` is strongly miscalibrated
- this means S6 cannot yet be claimed as a clean PASS for Active Observer
- the ablation evidence now narrows the problem:
  - the q-max cap / correction heuristics worsen the issue, but are not the
    sole cause
  - the scar is substantially more tied to the `active_two_point` posterior
    path than to the fixed-four canonical mouth
  - replicate-level diagnostics from `69b` also show the current
    `active_two_point` path pushes posterior `Q_max` upward on average
    (mean posterior bias `+0.026` in a 30-replicate diagnostic run), with the
    largest failures appearing after high late second observations
  - more specifically, the second-step `q_max_cap_margin` now looks like a
    primary driver rather than a minor heuristic, but not the only one
  - importantly, the production-side ablation does not currently look like a
    “calibration win, prediction loss” tradeoff; on this split it improves the
    main active-two-point predictive metrics as well
  - however, the 100x200 SBC confirmation shows that this is still a partial
    repair, not yet a promotion-ready final default; new failures can surface
    in other dimensions (`q_burst`, `log_tau_burst`) once the dominant `Q_max`
    scar is reduced
  - combining the two strongest low-risk ablations gives the current best
    posterior candidate:
    `likelihood_beta=1.0` plus `step2_q_max_cap_margin=0.0`
    on `active_two_point`
    - SBC: `8 / 9` dims pass at 100x200; `q_burst` and `log_tau_burst`
      recover, leaving `Q_max` as the only remaining failure
    - production evaluation mouth: `active_two_point_conformal`
      remains strongly competitive (`rmse_mean = 0.1261`, `crps = 0.0995`)
      and still improves substantially over the original default
  - increasing `q_max_correction` from `0.10` to `0.15` on top of that combo
    improves the calibration profile further without hurting the production
    result:
    - SBC still has `8 / 9` pass at 100x200, but `Q_max` mean rank moves
      from `0.378 -> 0.418` and the other 8 dimensions stay clean
    - production evaluation gives the current strongest overall candidate:
      `active_two_point_conformal rmse_mean = 0.1243`,
      `coverage90 = 0.9123`, `crps = 0.1017`
  - `scripts/69c_active_observer_v3_paper_candidate.py` now exists as a
    stable rerun entrypoint for that best-current production branch, so the
    candidate no longer lives only as scattered CLI flags and output folders
- combining the one-point and two-point runs, the strongest current
  interpretation is:
  `Q_max` miscalibration is introduced mainly by the second sequential update,
  whereas `log_tau_burst` is a broader posterior-family weakness that already
  appears after one update
  - the `prior_only` control further strengthens this: those residual scars are
    not coming from the empirical particle prior itself; they are introduced by
    the posterior update mechanism
- next S6 work should therefore prioritize:
  - treating the combined
    `likelihood_beta=1.0 + step2_q_max_cap_margin=0.0 + q_max_correction=0.15`
    branch as the current best paper-facing candidate while continuing work on
    the remaining `Q_max` scar
  - only then deciding whether the paper should claim posterior calibration
    for `fixed_four_point` only, while treating `active_two_point` posterior
    calibration as a limitation

### S4 sharpness + naive-band comparison

Status: `PARTIAL RUNTIME EVIDENCE`

Evidence:

- `scripts/75_chitosan_prospective_eval.py` reports cov90, cov50-adjacent interval widths, pooled R², and CRPS for the prospective bridge
- `scripts/76_casp_conformal_recalibration.py` reports coverage and width summaries
- requirement doc still says coverage claims need width + naive-band comparison in the publication-facing table
- `scripts/69d_s4_publication_table.py` now exists and builds a
  publication-facing canonical-split coverage / sharpness table from the
  current Active Observer paper-candidate output plus an explicit naive
  empirical train-band baseline
- runtime output now exists at `outputs/69d_s4_publication_table/`
- `scripts/69e_active_observer_v3_cov50_eval.py` now exists and reruns the
  same AO paper-facing candidate while adding raw central `q25-q75` reporting
  (`coverage_50`, `width_50`) without editing the user-modified `69` script
- extended runtime output now exists at:
  - `outputs_active_observer_v3_beta1_step2nocap_qcorr015_cov50/`
  - `outputs/69d_s4_publication_table_cov50/`
- the current canonical table shows:
  - `active_two_point_conformal`:
    `cov90 mean = 0.912`, `cov90 median = 1.000`,
    `width90 mean = 0.563`, `crps = 0.1017`
  - `naive_train_band`:
    `cov90 mean = 0.877`, `cov90 median = 1.000`,
    `width90 mean = 0.613`
  - so the current best AO candidate is not merely wider-than-naive:
    it improves `cov90` by `+0.035` while using a narrower 90% band
    (`width90_vs_naive = 0.919`)
  - `zero_early_prior_conformal` is also slightly better than naive
    (`cov90 +0.018`) while still narrower (`width90_vs_naive = 0.963`)
  - `fixed_four_point_conformal` is narrower than naive
    (`width90_vs_naive = 0.930`) but currently slightly under naive in
    mean cov90 (`-0.009`)
- the extended 50% table now sharpens the remaining S4 caveat:
  - `naive_train_band` central 50% band:
    `cov50 mean = 0.544`, `cov50 median = 0.667`,
    `width50 mean = 0.300`
  - `active_two_point_conformal` raw central 50% posterior band:
    `cov50 mean = 0.254`, `cov50 median = 0.000`,
    `width50 mean = 0.047`, `width50_vs_naive = 0.158`
  - so the current AO candidate is much sharper than the naive 50% band,
    but also dramatically under-covered at that central-band level
- the same table also makes the uncalibrated reference honest:
  `zero_early_prior_only` has tiny width (`0.129`) but catastrophic
  under-coverage (`cov90 mean = 0.211`, median = `0.000`)

Implication:

- the naive-band comparison is no longer missing for the canonical AO
  candidate; there is now a concrete submission-facing 90/80 table
- this meaningfully strengthens P3 because the current best AO candidate is
  not buying coverage only by inflating bands beyond a trivial train-band
  baseline
- the new rerun closes part of that cov50 gap by providing a real central
  `q25-q75` band on the same canonical AO candidate, but it also shows that
  the current posterior is far too concentrated at the 50% level
- so S4 is now substantially more complete, but not a clean PASS:
  the 90% story is strong and beats naive width-for-width, while the 50%
  central interval currently reads as under-dispersed rather than calibrated

### T10 / K2 chitosan reveal

Status: `BLOCKED BY EXTERNAL DATA`

Evidence:

- `scripts/75_chitosan_prospective_eval.py` is ready and checks preregistered `cov90 >= 0.83`
- `outputs/75_chitosan_prospective_eval/summary.txt` is currently missing
- no real wet-lab observed CSV is present

Implication:

- nothing more truthful can be claimed here until the observed release file exists

## 4. Repo-cleanup status relevant to publication readiness

### Already improved in this pass

- root `README.md` now reflects the closeout framing
- `scripts/README.md` now distinguishes canonical entry points from archive/history
- `docs/INDEX.md` now gives a stable reading order
- `REPO_INVENTORY.md` now inventories root/docs/scripts status
- local Claude handoff exists in `.claude/handoffs/2026-05-30_1341_codex_repo-cleanup-docs.md`

### Still dirty in ways that matter

- many authority/support docs are still untracked
- `README.md`, `scripts/README.md`, `AGENTS.md`, `DECISIONS.md`, `.gitignore`, and key scripts are already modified
- `sbi-logs/` still has 58 tracked binaries
- the release-wave 78-119 material is still largely untracked and mixed into the main worktree

### Still dirty but lower priority than scientific gates

- old handoff files at root
- non-authority prose packs and verdict notes
- manuscript-support CSV placement under `docs/`

## 5. Highest-leverage execution order

If the goal is "maximize top-journal readiness per unit time", the order should be:

1. **A3 split closure**
   - decide whether `73b` should temporarily stand in for `73`, or whether
     `73` itself should be patched after approval
   - then verify train/cal/test counts match across the accepted diagnostic
     scripts that share the canonical split

2. **Rerun corrected benchmark + diagnostics on the fixed split**
   - regenerate the corrected benchmark and diagnostic outputs after A3 closure
   - lock the refreshed cited numbers

3. **Close the accepted A3 diagnostic mouth**
   - decide whether `73b` is the temporary accepted replacement for `73`
   - rerun the split-aligned diagnostics that the README / paper will cite

4. **Extend B4 only if there is still upside**
   - `27c` and `27d` now both have same-mouth internal readouts and neither is winning
   - the next honest B4 move is not to relitigate those two again, but to
     decide whether an SNRE route is still worth comparator time

5. **S6 SBC on the actual canonical posterior**
   - adapt `05` / `27b` to the method that the paper is actually claiming

6. **B5 HMC subset benchmark**
   - the mouth is now locked via `27f`, so future sampler work should consume
     the exported package rather than redefining the subset
   - the remaining work is the actual HMC / NUTS runner plus a justified
     dependency addition

7. **B6/B7 fPCA + mean baseline closure**
   - baseline layer now exists on the canonical mouth via `38b`
   - the remaining work is interpretive, not implementation:
     decide how to present the fact that `fPCA-Ridge` is statistically tied
     with the current headline methods

8. **S4 sharpness / naive-band publication table**
   - finish the UQ reporting mouth after the comparator and SBC mouths are fixed

9. **Wet reveal**
   - blocked on external data

## 6. Anti-scope reminders

Do not spend time here until the scientific gates above are closed:

- expanding `80-119` stack/export/deck material
- new world-model routes
- new Active Observer story inflation
- manuscript polishing beyond what is needed to keep docs readable
- moving lots of files around in git before the scientific baseline gaps are closed

## 7. Practical next implementation target

The next code task with the best leverage is:

```text
Decide whether `73b_diagnostics_canonical_split.py` is the accepted
temporary A3 diagnostic mouth, then rerun the corrected benchmark-adjacent
diagnostics on the fixed split and lock those refreshed outputs.
```

Why this first:

- it is not scientifically speculative
- it closes the main reproducibility scar before more comparator work
- it prevents citing mixed-mouth benchmark and diagnostic numbers
- `27c` and `27d` already have honest same-mouth readouts, so the highest
  uncertainty has shifted back to A3 diagnostic convergence and then to
  whether SNRE deserves any further comparator time
