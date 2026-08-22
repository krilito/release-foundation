# Data collection inventory

Date: `2026-05-26`

Goal:

```text
Expand beyond the current PLGA release benchmark and test whether the
same paradigm holds:

early release observation -> ODE-decodable mechanism state -> simulator
forecast -> release-quality prediction.
```

This is not a generic literature list. It is a triage list for datasets
that could enter the current release-foundation pipeline.

## Search state

Web of Science was opened in Chrome with institutional access visible
through Qingdao University. A first Smart Search query:

```text
drug release machine learning dataset PLGA
```

returned `41` Web of Science Core Collection records. The page is slow
under browser automation, so the first pass below combines WoS discovery
with direct source checks from Mendeley, Zenodo, Nature/Springer, GitHub,
and Nottingham RDM.

## Tier A - directly useful release-curve datasets

### A1. PLGA microparticles 321

Source:

- Mendeley Data: https://data.mendeley.com/datasets/zzvtdrcy76/2
- Scientific Data: https://www.nature.com/articles/s41597-025-04621-9
- DOI: `10.17632/zzvtdrcy76.2`

What it contains:

- Literature-mined PLGA microparticle dataset.
- Scientific Data reports `321` in vitro PLGA MP release experiments,
  `4913` release points, `89` drugs, and `113` source publications.
- Mendeley description says the dataset includes drug properties,
  formulation compositions, formulation parameters, and release profiles.

Fit for our pipeline:

- Already the core cross321 dataset.
- Best use now is not "new data", but manuscript framing and leakage
  checks: this is the main PLGA micro/microparticle source.

Status:

- Already locally available through the existing cross321 pipeline.

### A2. Bannigan / long-acting injectables 181

Source:

- Zenodo: https://zenodo.org/records/7309021
- DOI: `10.5281/zenodo.7309021`
- Related paper: Bannigan et al., Nature Communications 2023.

What it contains:

- `181` drug release profiles.
- `43` unique drug-polymer combinations.
- `3783` fractional release measurements.
- Includes formulation descriptors and early release values at `T=0.25`,
  `T=0.5`, and `T=1.0` days.

Fit for our pipeline:

- Already tested as cleaned internal181 / external wet-style transfer.
- Best near-term use is to keep it as the second PLGA/LAI benchmark,
  especially for direct-Q vs theta->ODE controls and early-window
  ablations.

Status:

- Downloaded to:

```text
data/external/bannigan_lai_181/Dataset_14_feat.tsv
data/external/bannigan_lai_181/Dataset_17_feat.tsv
```

These files are in `data/` and should not be committed.

### A3. Liposome IVR database 271

Source:

- Paper: https://www.sciencedirect.com/science/article/pii/S2590156725000787
- GitHub: https://github.com/danielyanes22/nanomed_IVR_data
- Nottingham RDM: https://rdmc.nottingham.ac.uk/handle/internal/11960
- DOI: `10.17639/nott.7542`
- Accelerated-IVR follow-up repo:
  https://github.com/danielyanes22/accelerated_IVR

What it contains:

- Open-access liposome IVR database.
- Article reports `271` distinct IVR profiles, `141` liposome
  formulations, and `22` drugs.
- Includes formulation parameters, IVR testing conditions, lipid
  composition features, and digitized release data.
- The `accelerated_IVR` repo additionally provides a cleaner modeling
  entry point: processed ML features, Weibull fit parameters,
  experimental release curves, kinetic clusters, and classifier
  benchmark outputs.

Fit for our pipeline:

- This is the best non-PLGA expansion candidate.
- It can test whether the paradigm transfers from PLGA to another
  nanomedicine release mechanism.
- It will require a new simulator or a generic curve surrogate first;
  do not force the PLGA ODE onto liposomes.
- Start from `accelerated_IVR`, not the broader raw database, because it
  already has aligned feature tables and fitted kinetic summaries.

Status:

- GitHub code/data zip downloaded to:

```text
data/external/nanomed_IVR_data/nanomed_IVR_data_main.zip
data/external/accelerated_IVR/accelerated_IVR_main.zip
```

- `accelerated_IVR` quick inspection:

```text
data/clean/ML_7_features_df.csv          77 x 8
data/clean/ML_9_features_df.csv          78 x 10
data/clean/weibull_params.csv           169 x 4
data/unprocessed/backend_data.csv       271 x 15
results/fitting/drug_release_exp.csv   3332 x 3
```

- Full Nottingham RDM zipped database is reported as about `269 MB`.
  Download not yet attempted in this pass.

## Tier B - useful but not immediately same-task

### B1. PLGA nanoparticles 433 formulation dataset

Source:

- Scientific Data: https://www.nature.com/articles/s41597-025-05520-9
- Mendeley Data: https://data.mendeley.com/datasets/sbjf5csrdm/1
- DOI: `10.17632/sbjf5csrdm.1`

What it contains:

- `433` PLGA nanoparticle formulations.
- `65` small molecules.
- `18` formulation / molecule / excipient features.
- Key performance metrics are particle size, encapsulation efficiency,
  and loading capacity.

Fit for our pipeline:

- Not a release-curve dataset.
- Useful for descriptor pretraining, formulation-space coverage, and
  future multi-task quality modeling.
- Do not mix it into release curve training as if it has Q(t).

Status:

- Candidate only; not downloaded yet.

### B2. PVL-co-PAVL corticosteroid microparticles

Source:

- Mendeley Data: https://data.mendeley.com/datasets/8yfwrx7krh/2
- DOI: `10.17632/8yfwrx7krh.2`

What it contains:

- Corticosteroid-loaded poly(delta-valerolactone-co-allyl-delta-
  valerolactone) microparticles.
- Includes drug loading and release of triamcinolone acetonide and
  triamcinolone hexacetonide.

Fit for our pipeline:

- Small external LAI/polymer test.
- Useful as an out-of-family stress test after PLGA/internal181 is
  stable.
- Needs manual inspection of file structure and curve availability.

Status:

- Candidate only; not downloaded yet.

### B3. PLGA doxorubicin nanoparticles by coaxial electrospraying

Source:

- Mendeley Data: https://data.mendeley.com/datasets/h4tt4433w9/1
- DOI: `10.17632/h4tt4433w9.1`

What it contains:

- Doxorubicin release profile and MTT assay data from PLGA nanoparticles
  prepared by coaxial electrospraying.

Fit for our pipeline:

- Probably too small for training.
- Useful as a wet-style external stress curve, similar to the two NC
  `Prediction.csv` curves.

Status:

- Candidate only; not downloaded yet.

## Tier C - papers to mine / benchmark against

### C1. PLGA NP/MP release ML + experiments

Source:

- Scientific Reports: https://www.nature.com/articles/s41598-024-82728-6

What it contains:

- Literature data from about `50` papers.
- Article reports `97` observations and focuses on release amount,
  pH, drug solubility, drug molecular weight, and particle size.
- Includes new in vitro experiments with gentamicin/penicillin PLGA
  particles measured at short early times.

Fit for our pipeline:

- Good conceptual comparison and possible supplemental mining target.
- It appears closer to point-level release prediction than full curve
  release-quality prediction.
- Need to inspect supplementary tables before treating it as a dataset.

Status:

- Candidate paper; data extraction not started.

### C2. Raman spectra predicting release from polysaccharide coatings

Source mentioned in the liposome IVR article references:

- Abdalla et al., Journal of Controlled Release 2024,
  "Machine learning of Raman spectra predicts drug release from
  polysaccharide coatings for targeted colonic delivery."

Fit for our pipeline:

- Not PLGA and not necessarily full release curves.
- Very relevant for future "auxiliary observation -> release quality"
  framing, because Raman is an early/cheap quality signal.

Status:

- Literature-control candidate; not a primary dataset yet.

## Immediate execution plan

1. **Keep current paper scope on PLGA release curves.**
   Use cross321 + internal181 + NC wet curves as the manuscript data.

2. **Use liposome IVR 271 as the first true expansion dataset.**
   It is the best test of whether this becomes a broader drug-release
   quality paradigm.

3. **Do not merge formulation-only datasets into release prediction.**
   PLGA NP 433 is valuable, but it lacks Q(t). Treat it as descriptor /
   formulation-quality data, not release dynamics.

4. **Next script should be a data ingestion audit, not a model.**
   For each candidate dataset, report:

```text
dataset
n_curves
n_drugs
n_formulations
time unit coverage
release unit coverage
descriptor completeness
early-window availability
curve length / t_max distribution
can fit current ODE? yes/no/not applicable
recommended next use
```

## Current judgment

The best expansion path is:

```text
PLGA 321 + LAI 181
        ->
paper-grade release-quality surrogate
        ->
liposome IVR 271 as cross-mechanism proof-of-concept
        ->
future multi-mechanism release-quality paradigm
```

Do not dilute the current PLGA paper by rushing every new dataset into
the same model. Collect them, audit them, then decide whether they are
release curves, formulation-property tables, or external stress tests.

Detailed execution plan for the first expansion dataset:

```text
docs/plan_52_accelerated_ivr_extension.md
```
