# Project Archaeology And Method Inventory, 2026-06-12

This document freezes the project archaeology so future work does not drift
back into retired method branches. It is a guardrail, not a new experiment.

## Controlling Thesis

The current paper route is:

```text
Drug release prediction is release-state inference under limited
pre-experimental descriptors and costly observations.
```

Do not frame the current project as:

```text
We developed a stronger release-curve predictor.
```

The project separates four objects:

```text
X      = static formulation / drug / carrier / medium / condition descriptors
Q(t)   = one release function, not independent time-Q points
z      = latent release state / curve-language coefficients
Y_k    = k costly observed release points
```

and the active research structure is:

```text
Q(t) -> z
z -> Q(t)
X -> prior over z
X + Y_k -> posterior-like state estimate over z
z / Q_hat(t) -> design, risk, and stopping utility
```

## Claim Strength

| Level | Meaning |
|---|---|
| Supported | Reproducible in locked scripts or durable outputs under strict splits. |
| Partial | Holds in some systems/splits but weakens under stricter transfer tests. |
| Diagnostic | Useful for understanding the problem but not a deployable claim. |
| Retired | Tested and demoted; do not restart without a new preregistered diagnostic. |

## Supported Core Findings

| Finding | Status | Evidence anchor |
|---|---|---|
| PLGA static descriptors under-identify release. | Supported | `docs/plga_information_budget_claim_table_2026-06-11.md`, `outputs/102_plga_information_budget_claim_table/` |
| Measured early observations expose missing release state. | Supported | `outputs/97_plga_direct_early_blackbox_control/`, `outputs/101_lai_author_style_on_our_splits/` |
| Static-predicted early points do not replace measured early points. | Supported | `outputs/99_plga_static_to_early_proxy_probe/` |
| Early observations contract empirical future-error uncertainty. | Supported | `outputs/104_plga_empirical_uncertainty_contraction/` |
| Simple stopping rules can reduce observation cost with small error penalty. | Supported | `outputs/106_plga_stopping_rule_simulation/` |
| Release curves admit low-dimensional shape/latent languages. | Supported | `outputs/114_liposome_curve_dictionary_static_probe/`, `outputs/157_curve_world_hard_gate/`, `outputs/158_freeze_theta_targets/` |
| Liposome early observations close part of the static-to-oracle latent gap. | Partial | `outputs/115_liposome_latent_gap_observation_budget/` |
| Static priors can help candidate ranking even when exact curve RMSE is weak. | Partial | `outputs/116_liposome_design_utility_benchmark/` |
| PubChem/RDKit molecular descriptors do not close the strict liposome API gap. | Supported negative | `outputs/117_liposome_descriptor_enrichment_rdkit/` |
| Cross-system release transfer is not solved. | Supported negative | `outputs/88_release_transfer_claim_package_shapeanchored/`, `outputs/90_release_lnn_transfer_audit/` |

## Method Inventory

| Family | Methods tried | Current interpretation |
|---|---|---|
| Classical direct predictors | RandomForest, ExtraTrees, LightGBM, Ridge, KNN, direct `X,t -> Q` | Strong baselines; often beat neural models in small data. Use them as controls, not strawmen. |
| Middle-layer / shape routes | Weibull, biexponential, Hill, theta mappers, prototype curves, shape priors | Valuable curve-language and bottleneck diagnostics. Deployable only when selector does not see future error. |
| Curve dictionaries | PCA, NMF/FPCA-style curve bases, dictionary coefficients | Strong representation layer. Must be fit train-fold only. |
| SBI / mechanistic inference | PLGA ODE, SNPE/NPE, MAF/NSF, HMC/SNRE comparisons, SBC | Useful calibration and identifiability discipline. Not the current "better predictor" headline. |
| Early-observation routes | early-only, static+early, early residual selector, timepoint value, stopping rules | Core current evidence for release-state inference. |
| Uncertainty routes | ensemble, bootstrap, CASP/conformal, empirical fold-jackknife width | Useful for observation-budget and stopping claims; raw neural ensemble intervals are not calibrated posterior intervals. |
| Neural sequence/world routes | MLP, GRU, RSSM, CNP, LNN/CfC, partial-curve Transformer/NPE | Mostly diagnostic or negative under current data. Do not promote architecture novelty. |
| Symbolic routes | PySR, KAN-guided symbolic search, formula distillation | Retired from main training. Only allowed as offline distillation after a deployable state model exists. |
| Descriptor enrichment | RDKit, PubChem properties, molecular fingerprints, interaction features | Useful audit; current liposome result points to missing process/microstructure rather than simple molecular descriptors. |
| Author baselines | LAI author-style early baseline, Yanes liposome kinetic-class baseline | Important for task alignment. Many public baselines predict class/point targets, not continuous future curves. |
| Prospective bridge | Chitosan prospective lock and reveal-time evaluator | Valuable prospective scaffold. It does not prove one unified cross-mechanism learned model unless the preregistered wet reveal passes. |
| Exact posterior comparators | HMC/NUTS, SNRE, B5 log-posterior backend, partial-curve NPE | Useful calibration/comparator infrastructure. Not current manuscript headline evidence. |
| Older functional baselines | fPCA, NMF-increment, Direct LGBM, MEP-style baselines | Essential controls. Matched-protocol audits overturned earlier apples-to-oranges claims; keep them as comparators, not proof of a new route. |

## Retired Or Demoted Routes

### SRDS Cascade

Retired. The legacy cascade was:

```text
KAN + XGBoost teacher -> PySR symbolic scaffold -> per-curve oracle inversion
-> LightGBM x->theta mapper
```

Reason: pseudo-label identifiability, error compounding, fragile reproducibility,
and unclear deployable meaning. Do not rebuild it under a new name.

### PySR Or Symbolic Regression As Main Method

Retired from main training. The project already has physical/shape priors from
ODEs and curve families. Symbolic regression may be used only after a locked
deployable model exists, and only as offline explanation or distillation.

### Generic RSSM / World Model

Retired for current data scale. The historical RSSM run on sparse PLGA curves
showed posterior collapse and poor coverage. Deterministic temporal features had
some signal, but the stochastic latent world model did not become a usable
posterior. Do not restart generic RSSM training without a larger, matched,
multi-system corpus and a preregistered collapse diagnostic.

### LNN / CfC / CNP Cross-System Fantasy

Demoted. Release-corpus LNN/CNP probes show observation-budget signal, but the
durable evidence favors PLGA/system-local pattern rather than robust
cross-system transfer. Liposome and other non-PLGA dynamic tests do not beat
strong shape priors. LNN/CfC can remain a negative or ablation control, not a
claim engine.

### Active Observer As Predictive Headline

Demoted. After bug fixes and canonical benchmarking, the active-observer route
supports observation scheduling and uncertainty analysis, not a claim that it is
the best predictive algorithm.

### CASP / PCAP / Posterior-Family Expansion

Partial and supplementary. FIB-CASP and ensemble-CASP provide useful
few-shot/UQ evidence, especially for PLGA, but hard OOD calibration remains
weak. PCAP/pullback target scripts are future diagnostics, not current winners.
Use these routes to discuss uncertainty objects and posterior-family interfaces,
not to claim solved zero-shot cross-mechanism calibration.

### Chitosan Prospective Bridge

Prospective scaffold, not proof yet. The chitosan batch was hash-locked before
wet reveal, which is valuable if the reveal succeeds. However, the chitosan
route uses a chitosan-specific simulator and low-rank theta family, so it cannot
be used as evidence that the PLGA observer or a unified learned world model
transfers unchanged.

### fPCA / NMF / Direct LGBM / MEP Matched-Protocol Audits

Important controls. Older notes briefly suggested SBI was the clear internal
winner over fPCA and Direct LGBM, but matched-protocol audits corrected that:
on internal data SBI is competitive rather than dominant, while cross-DOI tail
robustness remains the cleaner SBI-era differentiator. Do not reuse the older
"SBI beats both bypass methods" language.

### B5 HMC / SNRE / Partial-Curve NPE

Comparator infrastructure. These scripts test whether exact or alternative
posterior inference changes the story. They are useful for calibration and
debugging amortized inference, but they do not supersede the current
information-budget route.

### Static-Only Exact Curve Prediction

Demoted as a sufficient solution. Static PLGA descriptors can perform well under
random splits, and per-curve static feature-combo oracles are strong diagnostic
upper bounds. Under source-group and source-dataset splits, however, the static
bridge is mostly source-local. The usable interpretation is "X provides a prior
over z," not "X determines Q(t)."

### New Neural Architecture Sprints

Frozen unless they answer a predefined release-state question. Do not open new
RL, foundation model, KAN, Transformer, LNN, MoE, PINN, or generic neural net
sprints merely because the current RMSE is imperfect.

### Green-Claim / Figure / Strategy Asset Waves

Archived as manuscript-support material. Release-wave figure, stack, bridge,
readiness, and green-claim assets may help assemble a paper or internal deck,
but they must not define the scientific claim. When they conflict with locked
outputs or closeout docs, trust the locked outputs.

## Known Methodological Traps

- The sample unit is a curve, never an independent timepoint.
- PCA/NMF/FPCA/prototypes/dictionaries/normalization are train-fold only.
- Oracle rows must be marked and excluded from deployable comparisons.
- Source, API, release-method, and dataset grouped splits are the meaningful
  stress tests; random splits are upper bounds.
- Strong classical baselines are real competitors, not weak strawmen.
- Raw ensemble intervals are not calibrated Bayesian posteriors.
- More data is useful only if it fixes observability: dynamic future windows,
  matched early/late sampling, process/microstructure variables, and repeated
  conditions.

## Consolidation-Only Next Work

The next research cycle should not invent a new model. It should consolidate:

1. one unified evidence table across PLGA and liposome,
2. one unified split taxonomy,
3. one claim-strength ladder,
4. one figure set separating `X -> z`, `X + Y_k -> z`, `z -> Q(t)`, and design
   utility,
5. one manuscript narrative around information budget and release-state
   inference.

Any proposed new algorithm must first state which gap it closes:

```text
representation gap: Q(t) -> z
static prior gap: X -> z
observation update gap: X + Y_k -> z
decision gap: z / Q_hat(t) -> design or stopping
```

If it does not close one of these gaps under strict splits, it is scope creep.

## Omission Audit

This freeze reviewed:

- `README.md`, `AGENTS.md`, `ARCHITECTURE.md`, and `DECISIONS.md`
- current docs in `docs/`
- archived notes under `docs/archive/`
- active scripts `27e-117`
- archived release-transfer/LNN scripts `80-90`
- historical SBI / Active Observer / release-corpus scripts in `scripts/archive/`
- durable outputs from PLGA `91-107`, liposome `108-117`, release-corpus
  transfer `81-90`, corpus grammar `157-161`, and older baseline/UQ lines
- script map entries for CASP/PCAP, chitosan prospective, B5 HMC/SNRE,
  fPCA/NMF/Direct LGBM controls, partial-curve NPE, residual hybrid, and
  release-wave manuscript assets

No additional model family was found that should override the current
information-budget / release-state inference route.
