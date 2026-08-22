# vendored/diagnostic_paper

Read-only reference copy of selected artifacts from the user's solo-authored
diagnostic paper (`D:/诊断论文/`, snapshot dated 2026-04-28). Imported into
release-foundation on 2026-05-22 to support ADR-023 (bridge collapse
framing) and the Sprint 0 comparator runs.

**Not production code.** Do not import from these files at runtime; the
release-foundation reimplementation lives at
`scripts/12_baseline_comparators.py`. These files exist solely so that:

1. The comparator script's numbers can be checked against the originals
   (`results/summary_with_locked_baselines.csv`, `results/paired_tests.csv`).
2. The fairness audit (`results/fpca_fairness_audit.md`) is preserved as
   the methodological reference for protocol choices.
3. Future readers can trace ADR-023's claims back to the source.

## Contents

- `scripts/run_fpca_nested_and_monotone_basis.py` — original nested fPCA
  pipeline (GroupKFold-5 outer, GroupKFold-3 inner k selection).
- `scripts/run_alternative_curve_routes.py` — shared helpers
  (FEATURES list, LGBM_PARAMS, fit_functional_pca_models,
  Direct LGBM run_direct, log-time grid, monotone clip).
- `results/summary_with_locked_baselines.csv` — locked numbers cited in
  ADR-023: nested fPCA LGBM R²=0.7518, Direct LGBM R²=0.7073, MEP R²=0.6940.
- `results/paired_tests.csv` — Wilcoxon paired tests; fPCA vs Direct
  p=0.6662 (not significant), fPCA vs MEP p=0.0325.
- `results/metrics_by_group.csv` — per-formulation RMSE used in the
  paired tests.
- `results/fpca_fairness_audit.md` — methodological audit + verdict text.

## Publication status of the source

The source paper is not under review and the user does not plan to submit
it to a journal. A chemrxiv preprint with a DOI anchor is the planned
fallback (Sprint 0 deliverable). The release-foundation repo absorbs the
bridge-collapse framing into ADR-023 and uses the numbers above as the
hypothesis-anchoring baseline for Phase 2 work.
