# Audit U2: Script 33 Clustering Input

Date: 2026-05-28

## Question

What features did script 33 cluster on? Does the "regime = identifiability
cluster" interpretation hold?

## Answer

**Yes.** Script 33 clusters on active-set indicator vectors — binary vectors
of length 9 (one bit per PLGA kinetic parameter) indicating which parameters
are in each curve's k=4 active set.

## Evidence

### File/line citations

`scripts/33_active_set_regimes.py`:

- **Lines 0-10** (docstring): "Take each curve's k=4 active parameter set from
  script 29 and represent it as a 9-bit binary vector."

- **Lines 81-92** (`_build_membership`): Builds binary matrix M of shape
  (n_curves, n_params). Each row is a curve, each column is a parameter.
  M[i,j] = 1 if parameter j is in curve i's active set.

- **Lines 114-135** (`_cluster_silhouette_curve`): Clusters using
  `pdist(M, metric="jaccard")` followed by `linkage(D_condensed, method="average")`.

- **Lines 138-165** (`_null_silhouette_distribution`): Generates null distribution
  by swap-preserving permutation of the binary matrix (preserves row/column
  marginals).

### Data flow

```
Script 29 outputs: per-curve active sets (k=4 parameters per curve)
  → Script 33 _build_membership(): active sets → 9-bit binary vectors
  → pdist(M, metric="jaccard"): Jaccard distance matrix
  → linkage(average): hierarchical clustering
  → fcluster(): regime labels
```

### Clustering input

- **Feature space**: Binary indicator vectors, length = 9 (one per PLGA parameter)
- **Distance metric**: Jaccard (fraction of bits that differ)
- **Linkage**: Average
- **NOT**: formulation features, theta values, curve shapes, or learned representations

## Verdict

The regime = identifiability cluster interpretation **holds**. Regimes are
clusters of curves that share similar patterns of which kinetic parameters
are recoverable from their release trajectories. This is a property of the
interaction between formulation and observation window, not of formulation
alone or curve shape alone.

## Implications

This explains the negative silhouette results in formulation space (sil=-0.18)
and theta space (sil=-0.06): regimes are defined by identifiability structure,
not by parameter values or formulation features. Two curves with very different
formulations and theta values can be in the same regime if they share the same
active set pattern.
