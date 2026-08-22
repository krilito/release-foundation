# Nature-level generalization strategy

Date: 2026-05-28
Author: Codex
Purpose: define a Nature-level route that uses the current algorithmic assets, can absorb new drug-release datasets, and is not intellectually trapped inside drug release.

## Executive decision

The paper should not be framed as "better drug-release prediction" or "world model for drug release."

The stronger object is:

> **Calibrated posterior-family inference and active observation for partially observed scientific dynamical systems.**

Drug release is the first scientific micro-world. Chitosan is the prospective wet-lab validation. Liposome IVR is the first retrospective non-PLGA release mechanism. Battery degradation and engineered prognostics are the route out of drug release.

The core claim should not be "our model has the lowest RMSE." It should be:

> Sparse early observations can be converted into an uncertainty-calibrated feasible state family, and that family can decide what to measure next under limited experimental budget.

This is the only version that can plausibly scale to Nature Methods / Nature Machine Intelligence. A narrow controlled-release paper is not enough.

## The algorithmic object

The central object must be one thing, not a pile of modules:

```text
partial observations y_obs
        ->
feasible posterior family over latent state theta
        ->
calibrated predictive distribution over future trajectories
        ->
active measurement utility for the next observation
```

The implementation may use ExtraTrees, particles, ODE decoders, conformal calibration, or CASP. But the manuscript must be organized around the posterior-family object, not around those components.

The object is valid across domains if each domain supplies:

1. a trajectory or time-evolving quality state,
2. sparse early observations,
3. a latent mechanism or kinetic state,
4. a long-horizon target,
5. a meaningful measurement cost,
6. an uncertainty claim that can be tested.

## Drug-release datasets: what to use

### Tier A: main evidence

| Dataset | Use | Current status | Source |
|---|---|---|---|
| PLGA microparticles 321 | Main retrospective release benchmark; cross-source / group split stress test | Already in project | Scientific Data / Mendeley: https://www.nature.com/articles/s41597-025-04621-9 and https://data.mendeley.com/datasets/zzvtdrcy76/2 |
| LAI / Bannigan 181 | Second PLGA-style benchmark and internal-vs-external transfer check | Already in project | Zenodo: https://zenodo.org/records/7309021 |
| Liposome IVR 271 | First non-PLGA release-mechanism test | Downloaded / partially inspected | GitHub: https://github.com/danielyanes22/nanomed_IVR_data and Nottingham RDM DOI `10.17639/nott.7542` |
| Chitosan prospective batch 1 | Prospective wet-lab validation, not just another dataset | Predictions hashed and tagged | Local: `docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md` |

### Tier B: useful but not main proof yet

| Dataset | Use | Warning |
|---|---|---|
| PLGA nanoparticle 433 formulation dataset | Descriptor pretraining / formulation-property auxiliary task | Not a release-curve dataset; do not train Q(t) from it |
| PVL-co-PAVL corticosteroid microparticles | Small external polymer stress test | Needs manual curve audit |
| PLGA doxorubicin electrospraying | Wet-style case study | Likely too small for training |
| Raman spectra -> release from polysaccharide coatings | Future "cheap observation -> release quality" story | Need access to full curves and spectra |

### Dataset intake gate

Before any new drug-release dataset enters the main benchmark, create an intake table:

```text
dataset
n_curves
n_formulations
n_drugs
time units
release units
timepoint distribution
descriptor completeness
curve monotonicity / noise
early-window availability
mechanism family
can fit current decoder? yes/no
recommended role: train / test / stress / auxiliary / reject
```

No new dataset should be merged just because it exists. It must strengthen one of the paper's claims.

## Beyond drug release: best structural testbeds

### 1. Battery degradation and cycle-life prediction

This is the best non-drug-release expansion.

Why it matches:

- early cycles are sparse observations,
- latent state is degradation mode / capacity-fade state,
- long-horizon target is cycle life, knee point, or capacity trajectory,
- measurements are costly because full cycling takes time,
- the field already values early prediction and uncertainty.

Candidate sources:

- Severson et al. battery cycle-life data via MATR: https://data.matr.io/1/
- NASA Prognostics Center battery data: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- CALCE battery datasets: https://calce.umd.edu/battery-data

Role in the paper:

Battery should be the first non-biomedical testbed if the paper wants to escape "drug release." It is structurally homologous and scientifically important.

Minimum experiment:

```text
early cycles 1..N -> feasible degradation-state family -> capacity trajectory / EOL prediction
compare:
  point predictor
  posterior family
  posterior family + conformal calibration
  active next-cycle / next-diagnostic measurement utility
```

Primary metrics:

- EOL error,
- capacity-trajectory RMSE,
- cov90 / interval width,
- early-budget curve,
- OOD split by protocol / batch if metadata supports it.

### 2. NASA C-MAPSS turbofan degradation

Why it matches:

- sensor streams are partial observations,
- hidden health state evolves over time,
- target is remaining useful life,
- grouped OOD splits can be defined by operating condition and fault mode.

Source:

- NASA PCoE data repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Role:

Use as an engineering-prognostics sanity test, not the first headline. It proves the algorithm is not tied to chemistry.

Warning:

C-MAPSS is a heavily used benchmark. The paper should not pretend to beat the RUL literature. Its role is structural validation of posterior-family and calibration behavior.

### 3. Population pharmacokinetics sparse sampling

Why it matches:

- sparse blood draws,
- latent PK parameters,
- active sampling time design,
- calibrated concentration-time prediction.

Role:

This is scientifically close to drug release, so it does not fully escape the pharmaceutical domain. It is still valuable because the active-measurement logic is natural and clinically interpretable.

Warning:

Use only if there is an accessible, clean public dataset and if patient privacy/licensing is unambiguous.

### 4. Catalyst deactivation / materials aging

Why it matches:

- early activity decline predicts lifetime,
- latent state is active-site poisoning, sintering, phase change, or transport limitation,
- expensive observations include in-situ XRD/XAS or microscopy.

Role:

High Nature appeal, but only if a clean open dataset exists. Otherwise this is Phase 2, not immediate.

### 5. Bioreactor / microbial community dynamics

Why it matches:

- early pH, OD, substrate, and product signals predict later yield or collapse,
- latent state is kinetic or community composition,
- expensive observations include omics or high-frequency sampling.

Role:

Good future direction. Do not use immediately unless a clean benchmark already exists.

## What not to touch now

| Candidate | Reason to avoid |
|---|---|
| Toy ODEs only | Too synthetic for Nature-level claim |
| Weather / climate | State dimension and modeling assumptions are incompatible with current method |
| Finance | Mechanistic state is unstable and not scientific dynamics |
| Infectious disease SIR as main test | Policy interventions change the mechanism; easy to overclaim |
| Generic UCI time-series tables | They do not test mechanistic ambiguity or measurement selection |
| Any dataset without a real long-horizon target | It cannot support the posterior-family story |

## Paper architecture

### Main claim

> A calibrated feasible-state family enables truthful long-horizon prediction and active measurement selection in sparse scientific dynamical systems.

### Figure 1: Problem and object

Show the common structure:

```text
sparse observations -> feasible latent family -> calibrated future -> next measurement
```

Use PLGA/chitosan as the visual example, but label the abstraction above it.

### Figure 2: Retrospective drug-release proof

Use PLGA 321 + LAI 181 + liposome IVR.

Required:

- leakage-fixed benchmark,
- grouped OOD split,
- point prediction + coverage + interval width,
- fixed vs active observation budgets.

### Figure 3: Chitosan prospective validation

This is the wet-lab anchor.

Required:

- show hash/tag and pre-registration,
- report cov90, sharpness, R2 / RMSE,
- show failures honestly,
- do not post-hoc change thresholds.

### Figure 4: Non-release generalization

Use battery degradation as the preferred testbed.

Minimum acceptable evidence:

- same method object,
- no domain-specific rewrites beyond decoder/feature adapter,
- early-budget curve,
- calibrated uncertainty,
- OOD split if possible.

### Extended Data

- RSSM negative result,
- CASP / conformal ablation,
- regime analysis after dataset-label bug is fixed,
- dataset intake audit,
- failure cases.

## Two possible Nature-level routes

### Route A: Nature Methods / NMI, cross-domain method paper

Title style:

> Calibrated feasible-state inference for sparse scientific dynamics

Evidence required:

1. PLGA retrospective benchmark cleaned of leakage.
2. Liposome IVR as non-PLGA release mechanism.
3. Chitosan prospective batch 1.
4. Battery degradation as non-release testbed.
5. Same posterior-family object across all domains.

This is the ambitious route.

### Route B: Nature Communications / Nature Methods, biomedical-method paper

Title style:

> Uncertainty-calibrated active observation for controlled release experiments

Evidence required:

1. PLGA + LAI retrospective.
2. Liposome IVR.
3. Chitosan prospective.
4. Clear experimental-cost reduction.

This is safer but less aligned with the user's stated goal of not being trapped in drug release.

## Kill / keep decisions

### Keep

- Chitosan prospective validation.
- Conformal calibration / CASP recalibration.
- Feasible posterior family as the central object.
- Active observation selection, after leakage is fixed.
- Liposome IVR as first non-PLGA release test.
- Battery degradation as first non-release test.

### Demote

- Active Observer as "the main model" until clean benchmarks prove it.
- DirectQ-adaptive as "best method" until day-21 leakage is removed.
- Regime story until dataset-label mixing is fixed.
- RSSM as a negative-result baseline.

### Kill or freeze

- Foundation-model wording.
- World-model headline, unless learned dynamics + intervention planning are actually demonstrated.
- Any claim based only on point RMSE.
- Any benchmark where the model observes a target timepoint and is then scored on it.

## Immediate two-week plan

### Week 1: make drug-release evidence clean

1. Patch `72_canonical_benchmark_v2.py` and `77b_regime_benefit_loocv_v2.py` so observation times do not overlap evaluation targets.
2. Patch `74b_regime_verification_v2.py` to use dataset-aware regime labels.
3. Rerun fixed/adaptive benchmarks and report paired bootstrap CIs.
4. Audit liposome IVR ingestion and decide whether it is paper-grade or supplementary.
5. Verify chitosan hash/tag artifacts remain intact.

### Week 2: open the non-release route

1. Create a battery-degradation intake script and data audit.
2. Implement the minimum adapter:

```text
trajectory loader
early-window splitter
future-target evaluator
conformal calibration wrapper
posterior-family / ensemble uncertainty object
```

3. Run a smoke benchmark:

```text
early observations -> capacity trajectory or EOL prediction
random split + grouped split if metadata supports it
RMSE + cov90 + interval width
```

4. Decide whether battery enters the main paper or remains a Phase 2 bridge.

## Success criteria

The Nature-level route is alive only if all are true:

1. Chitosan prospective analysis is reported exactly per pre-registration.
2. Drug-release adaptive benchmarks are leakage-free.
3. At least one non-PLGA release dataset runs through the same object.
4. At least one non-release dynamical dataset runs through the same object.
5. The paper can state one algorithmic object without naming RF, ExtraTrees, PLGA, or chitosan.

If criterion 4 fails, the honest target becomes a strong drug-release method paper, not a cross-domain Nature-level algorithm paper.

## Current recommendation

Do not choose between "drug release paper" and "world model paper" yet.

Choose this:

> **A posterior-family method for sparse scientific dynamics, with drug release as the first real laboratory system and battery degradation as the first external dynamical-system test.**

That preserves the user's ambition while staying tied to evidence.
