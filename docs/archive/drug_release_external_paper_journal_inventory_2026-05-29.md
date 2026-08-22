# Drug Release External Paper and Journal Inventory

Date: `2026-05-29`

Purpose:

```text
Refresh the outside view with current-source evidence:
1. what ML + drug-release papers are actually appearing now,
2. which algorithm families those papers belong to,
3. which journals and venue bands currently host the space,
4. what this means for our fight / borrow / defend map.
```

This file is intentionally narrower than a full review.  
It is a source-backed inventory for strategic use.

## 1. Executive refresh

The current external field still clusters into four practical operating styles:

1. `descriptor or descriptor+time supervised release prediction`
2. `descriptor + sparse early-observation forecasting`
3. `mechanism-local workflow / release-test design / kinetic-class prediction`
4. `optimization / automation / formulation-platform systems`

What changed in the latest refresh is not the basic taxonomy, but the venue and
tooling pattern:

- `Journal of Controlled Release` and `International Journal of Pharmaceutics`
  remain the most natural homes for serious drug-delivery ML framing.
- `Scientific Reports`, `Nanoscale`, and `The AAPS Journal` are currently
  carrying more concrete model papers than many people in the field may expect.
- `Digital Discovery` and `Advanced Drug Delivery Reviews` matter more for the
  `workflow / optimization / platform` layer than for direct release benchmark
  ownership.

Short verdict:

`the open slot is still not "best single regressor"; it is a more complete shared release-intelligence stack.`

## 2. Representative paper inventory

| Work | Year | Venue | System / scope | Algorithm family | Input -> output | Strategic role |
| --- | --- | --- | --- | --- | --- | --- |
| [Bannigan et al.](https://www.nature.com/articles/s41467-022-35343-w) | 2023 | Nature Communications | polymeric long-acting injectables | tabular model zoo; few-shot / zero-shot | descriptors ± early points -> release | `fight` |
| [Machine learning in drug delivery](https://pubmed.ncbi.nlm.nih.gov/38909704/) | 2024 | Journal of Controlled Release | field-wide review | review / umbrella framing | drug-delivery complexity -> ML framing | `borrow` |
| [Machine learning integrated with in vitro experiments for study of drug release from PLGA nanoparticles](https://www.nature.com/articles/s41598-024-82728-6) | 2025 | Scientific Reports | PLGA micro/nanoparticles | linear / PCA / GPR / ANN | literature-extracted descriptors -> release guidance | `borrow` |
| [Utilizing machine learning for predicting drug release from polymeric drug delivery systems](https://www.sciencedirect.com/science/article/pii/S0010482525001064) | 2025 | Computer Methods and Programs in Biomedicine | release-specific review | review of ANN / RF / XGBoost / SVM / ensemble | formulation data -> release prediction families | `borrow` |
| [A machine learning workflow to accelerate the design of in vitro release tests from liposomes](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00112a) | 2025 | Digital Discovery | liposome IVR workflow | clustered release classes + classifier workflow | formulation + IVR method -> release class | `borrow` |
| [Genetic algorithm optimization of tree-based models to predict cargo- and carrier-related factors affecting drug release from liposomes](https://pubs.rsc.org/en/Content/ArticleLanding/2025/NR/D5NR00016E) | 2025 | Nanoscale | liposome release prediction | RF / XGBoost + GA + SHAP | cargo/carrier/media descriptors -> release profile | `fight-adjacent` |
| [Machine Learning Predicts Drug Release Profiles and Kinetic Parameters Based on Tablets' Formulations](https://link.springer.com/article/10.1208/s12248-025-01101-1) | 2025 | The AAPS Journal | oral tablets / dynamic dissolution | supervised ML on structured formulation table | tablet formulation -> full profile + kinetic params | `borrow-adjacent` |
| [Polymer microparticles in an evolving drug delivery landscape: challenges and the role of machine learning](https://www.sciencedirect.com/science/article/pii/S0378517325007434) | 2025 | International Journal of Pharmaceutics | LAI polymer microparticles | review / positioning | formulation complexity -> ML opportunity framing | `borrow` |
| [Artificial intelligence and machine learning guided optimization in drug delivery](https://pubmed.ncbi.nlm.nih.gov/41579967/) | 2026 | Advanced Drug Delivery Reviews | cross-modality drug delivery | review of surrogate modeling / BO / active learning | formulation/process space -> optimization workflow | `defend` |
| [Interpretable Two-Stage Machine Learning for Early and Full Drug Release Prediction in PLGA Microspheres](https://www.mdpi.com/1424-8247/19/5/767) | 2026 | Pharmaceuticals | PLGA 321-curve setting | staged interpretable supervised ML | formulation + early release -> early/full release | `fight` |

## 3. What these papers say about "how people are doing it now"

### 3.1 The default remains supervised tabular prediction

The 2025 release-specific review in
[Computer Methods and Programs in Biomedicine](https://www.sciencedirect.com/science/article/pii/S0010482525001064)
still centers the field around `ANN`, `RF`, `XGBoost`, `SVM`, and ensembles.

That means most reviewers will still map "ML for release" to:

- table inputs
- direct release outputs
- relatively modest model classes

### 3.2 Early-observation forecasting is real, but still narrow

`NC / Bannigan` remains the canonical outside reference for
`descriptors + sparse early points -> later release`.

The latest PLGA and tablet papers still look much closer to:

- `supervised predictor`
- `interpretable stagewise model`
- `kinetic-parameter prediction`

than to a shared posterior-family object.

### 3.3 Liposome work is currently stronger on workflow than on unified inference

The two 2025 liposome papers are especially informative:

- the `Digital Discovery` paper uses ML to design and classify IVR workflows
- the `Nanoscale` paper uses tree models and SHAP to predict release behavior

Together they show that non-PLGA work is active, but mostly at:

- `workflow / assay design`
- `descriptor-driven release prediction`

rather than at a unified partially observed inference layer.

### 3.4 The strongest platform pressure is moving toward optimization systems

The 2026 `Advanced Drug Delivery Reviews` optimization review makes this very
clear: the most dangerous adjacent systems are not simply better regressors,
but ML-guided optimization, active learning, and self-driving laboratory
framings.

That is why our most important weak layer is still `L5 reporting / decision`.

## 4. Journal and venue bands

The current journal picture supports a fairly sharp venue strategy.

### 4.1 Best-fit non-pure-OA core venues

These are still the strongest mainline venues for a release-intelligence paper.

| Journal | Why it matters | AbleSci 2025 IF / OA share |
| --- | --- | --- |
| [Journal of Controlled Release](https://www.ablesci.com/journal/detail?id=5JW48D) | strongest flagship for delivery science and formulation/release mechanisms | IF `11.5`; Gold OA `22.04%` |
| [International Journal of Pharmaceutics](https://www.ablesci.com/journal/detail?id=rwqKB5) | broad home for formulation, devices, delivery systems, evaluation | IF `5.2`; Gold OA `20.47%` |
| [Molecular Pharmaceutics](https://www.ablesci.com/journal/detail?id=rRQQwD) | best fit for mechanistic or object-level drug-delivery reasoning | IF `4.5`; Gold OA `12.79%` |
| [Drug Delivery and Translational Research](https://www.ablesci.com/journal/detail?id=5x8Ng5) | strongest translational/drug-delivery framing lane | IF `5.5`; Gold OA `27.38%` |
| [European Journal of Pharmaceutics and Biopharmaceutics](https://www.ablesci.com/journal/detail?id=DGL2vr) | strong pharmaceutics / biopharmaceutics / release-systems venue | IF `4.3`; Gold OA `28.61%` |
| [Journal of Drug Delivery Science and Technology](https://www.ablesci.com/journal/detail?id=DZ69XD) | practical delivery/process/technology venue, lower prestige but very on-topic | IF `4.9`; Gold OA `7.21%` |

### 4.2 Review / conceptual framing venues

- [Journal of Controlled Release review, 2024](https://pubmed.ncbi.nlm.nih.gov/38909704/)
- [Advanced Drug Delivery Reviews optimization review, 2026](https://pubmed.ncbi.nlm.nih.gov/41579967/)

These are the places shaping the language of the field, even when they are not
the most direct homes for our final original research article.

### 4.3 Venues that matter, but mostly for adjacent pressure

- `Scientific Reports`
- `Nanoscale`
- `The AAPS Journal`
- `Digital Discovery`
- `Pharmaceutics`

These are important because they show where concrete adjacent systems are
appearing. But they do not yet redefine the center of release-intelligence
positioning the way `JCR` or `IJP` do.

## 5. Updated fight / borrow / defend implications

### Fight

- `NC / Bannigan`
- the 2026 two-stage PLGA direct-release family
- any paper that frames itself as the clean few-shot release benchmark winner

### Borrow

- `JCR` / `IJP` / `Molecular Pharmaceutics` level scientific language
- liposome IVR workflow packaging
- structured formulation databases and data-schema discipline

### Defend

- optimization / active-learning / self-driving lab narratives
- formulation-platform systems that are stronger than us at `L5 utility`

## 6. Bottom line

The refreshed outside view reinforces rather than weakens the current internal
position:

`the field is still fragmented across direct predictors, workflow systems, and optimization platforms, so the most credible bold move remains to occupy the shared release-intelligence stack rather than to claim one universal solved model.`

## Quick links

- English master entry point:
  [drug_release_master_positioning_dossier_2026-05-29.md](D:/release-foundation/docs/drug_release_master_positioning_dossier_2026-05-29.md)
- Chinese strategy answer:
  [drug_release_strategy_answer_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_strategy_answer_zh_2026-05-29.md)
- Chinese battlecard:
  [drug_release_battlecard_zh_2026-05-29.md](D:/release-foundation/docs/drug_release_battlecard_zh_2026-05-29.md)
