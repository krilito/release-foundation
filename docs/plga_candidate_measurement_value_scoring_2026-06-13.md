# PLGA Candidate Measurement Value Scoring

Date: 2026-06-13

## Purpose

Experiment 121 consumes the strict out-of-fold residual map from Experiment 120 and ranks candidate measurement variables.

It does not redefine the residual, train a new release predictor as the headline object, or output a formulation-level remeasurement plan.

## Core Distinction

- EvidenceScore: support from currently available table proxies.
- PriorityScore: future measurement priority.

No-proxy candidates can rank only as hypotheses. They receive no observed proxy support, no proxy prediction gain, and no EvidenceScore.

## Guardrails

- Primary residuals must be strict out-of-fold rows from Experiment 120.
- Proxy validation uses a max-statistic permutation null across proxy columns, residual targets, state spaces, and strict group splits.
- Prediction gain is conditional on all available reported X except the candidate proxy.
- Nonredundancy is computed as how poorly existing X predicts the candidate proxy, not as a duplicate of prediction gain.
- DOI is only a source/confound audit label and is never a candidate measurement.
- Multiple null candidates calibrate score competitiveness.

## Interpretation Boundary

121 ranks candidate measurements. 121 does not discover physical hidden variables.

A high hypothesis-only PriorityScore is not empirical evidence. It is a testable experimental hypothesis.

DOI/method structure may represent source, protocol, assay, batch, or process confounding.

No candidate may be described as causal without direct experimental validation.

## Output Anchor

`../outputs/121_plga_candidate_measurement_value_scoring/`
