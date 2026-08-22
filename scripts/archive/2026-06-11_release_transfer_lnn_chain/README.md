# 2026-06-11 Release-Transfer / LNN Chain Archive

This folder preserves scripts `80`-`90` from the release-corpus transfer sprint.

They are archived instead of kept in the active `scripts/` namespace because
the current PLGA paper route is:

```text
information-budgeted sparse-observation forecasting for PLGA release
```

not:

```text
release-corpus foundation model / cross-system transfer
```

## Contents

| Script | Role |
|---|---|
| `80_lnn_observation_budget_probe.py` | early LNN/CfC-style PLGA observation-budget probe |
| `81_release_corpus_cnp_lnn_probe.py` | release-corpus CNP/LNN transfer probe |
| `82_release_transfer_report.py` | transfer-run summary |
| `83_release_transfer_failure_calibration.py` | failure and interval calibration diagnostics |
| `84_release_transfer_data_regime_audit.py` | data-regime audit for negative transfer cases |
| `85_release_shape_prior_leakage_audit.py` | shape-prior leakage audit |
| `86_release_dynamic_future_benchmark.py` | dynamic-future masked benchmark |
| `87_release_transfer_data_expansion_plan.py` | data expansion plan |
| `88_release_transfer_claim_package.py` | transfer claim package |
| `89_release_liposome_ablation_report.py` | liposome ablation report |
| `90_release_lnn_transfer_audit.py` | LNN transfer audit |

## Current Interpretation

These scripts are useful historical evidence for the transfer/LNN branch:

- LNN/CNP transfer was explored on the 751-curve release corpus.
- The strongest durable PLGA conclusion did not come from claiming cross-system
  transfer success.
- Current project framing should not depend on these scripts unless a new
  release-corpus goal explicitly reopens the transfer line.

## Reuse Rule

If this line is reopened, move scripts back intentionally or import from this
archive with a new numbered active script. Do not mix them into the current
`91`-`103` PLGA information-budget chain by accident.
