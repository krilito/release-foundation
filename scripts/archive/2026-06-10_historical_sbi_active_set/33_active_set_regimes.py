"""
33 - Regime discovery on per-curve active sets.

What this does:
    1. Take each curve's k=4 active parameter set from script 29 and
       represent it as a 9-bit binary vector.
    2. Cluster all curves (internal181 + cross321 combined) under Jaccard
       distance with average-linkage hierarchical clustering.
    3. Pick a "best" number of regimes by silhouette score over
       n_clusters in [2, 8].
    4. Compare the observed silhouette curve against a swap-preserving
       marginal-null distribution (same row/column marginals as observed).
       Without this, any apparent clustering could just be artifact of
       heterogeneous parameter marginals.
    5. Reverse-check each regime on external descriptors that were NOT
       used for clustering:
         - t_max, n_obs           (experimental design)
         - boundary_hits          (fit-difficulty)
         - fast_regime, short_window (kinetic flags, cross321 only)
         - Q_max_full             (recovered capacity, semi-circular)
       If a regime is a real kinetic mode, its members should be similar
       on at least one of these external axes. If members are dispersed
       on all of them, the "regime" is a statistical hallucination.

Why this exists:
    Script 32 found only one cross-dataset-consistent coupling pair
    (log_kw ~ Q_max). That rules out a dense pairwise coupling network.
    It does not rule out that curves fall into a small number of
    discrete regimes, each with its own active-set pattern. This script
    tests that latent-class story directly.

Anti-self-deception bar:
    A regime structure is "real" only if:
      (a) observed silhouette at best n_clusters exceeds the 95th
          percentile of the marginal-null silhouette distribution,
      (b) at least one external descriptor shows statistically
          significant variation across regimes (Kruskal-Wallis p < 0.01
          for continuous, chi-square p < 0.01 for categorical),
      (c) the same regime is non-trivially populated by both datasets
          (so it is not a dataset-specific artifact).
    Anything weaker is reported as "hint" or "noise", not as a regime
    catalogue we should build models on.

Outputs:
    outputs/33_active_set_regimes/silhouette_curve.png
    outputs/33_active_set_regimes/dendrogram.png
    outputs/33_active_set_regimes/regime_active_set_heatmap.png
    outputs/33_active_set_regimes/regime_descriptor_panels.png
    outputs/33_active_set_regimes/regime_assignments.csv
    outputs/33_active_set_regimes/regime_summary.csv
    outputs/33_active_set_regimes/null_silhouette_distribution.csv
    outputs/33_active_set_regimes/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency, kruskal
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402

DATASET_COL = "dataset"


def _parse_active_set(cell: object) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [token.strip() for token in cell.split(",") if token.strip()]


def _build_membership(
    active_sets: list[list[str]],
    param_names: list[str],
) -> np.ndarray:
    p_index = {name: i for i, name in enumerate(param_names)}
    M = np.zeros((len(active_sets), len(param_names)), dtype=np.int8)
    for r, aset in enumerate(active_sets):
        for name in aset:
            idx = p_index.get(name)
            if idx is not None:
                M[r, idx] = 1
    return M


def _attempt_swap(M: np.ndarray, rng: np.random.Generator) -> bool:
    n_rows = M.shape[0]
    i1, i2 = rng.choice(n_rows, size=2, replace=False)
    r1 = M[i1]
    r2 = M[i2]
    a = np.where((r1 == 1) & (r2 == 0))[0]
    b = np.where((r1 == 0) & (r2 == 1))[0]
    if a.size == 0 or b.size == 0:
        return False
    c1 = int(rng.choice(a))
    c2 = int(rng.choice(b))
    M[i1, c1] = 0
    M[i1, c2] = 1
    M[i2, c1] = 1
    M[i2, c2] = 0
    return True


def _cluster_silhouette_curve(
    M: np.ndarray,
    cluster_range: range,
) -> tuple[dict[int, float], np.ndarray, np.ndarray]:
    """Returns silhouette per n_clusters, full distance matrix, linkage."""
    # Jaccard distance via scipy
    D_condensed = pdist(M, metric="jaccard")
    D = squareform(D_condensed)
    Z = linkage(D_condensed, method="average")
    out: dict[int, float] = {}
    for n_clusters in cluster_range:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            out[n_clusters] = float("nan")
            continue
        # silhouette wants precomputed distance matrix
        try:
            sc = float(silhouette_score(D, labels, metric="precomputed"))
        except ValueError:
            sc = float("nan")
        out[n_clusters] = sc
    return out, D, Z


def _null_silhouette_distribution(
    M_obs: np.ndarray,
    cluster_range: range,
    n_null: int,
    swaps_per_sample: int,
    burn_in: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    M = M_obs.copy()
    for _ in range(burn_in):
        _attempt_swap(M, rng)
    rows: list[dict[str, object]] = []
    for b in range(n_null):
        for _ in range(swaps_per_sample):
            _attempt_swap(M, rng)
        D_c = pdist(M, metric="jaccard")
        Z_b = linkage(D_c, method="average")
        D_b = squareform(D_c)
        for n_clusters in cluster_range:
            labels = fcluster(Z_b, t=n_clusters, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                rows.append({"sample": b, "n_clusters": n_clusters, "silhouette": float("nan")})
                continue
            try:
                sc = float(silhouette_score(D_b, labels, metric="precomputed"))
            except ValueError:
                sc = float("nan")
            rows.append({"sample": b, "n_clusters": n_clusters, "silhouette": sc})
    return pd.DataFrame(rows)


def _regime_modal_active(M_subset: np.ndarray, param_names: list[str], top: int = 4) -> str:
    counts = M_subset.sum(axis=0)
    order = np.argsort(-counts)[:top]
    return ",".join(param_names[i] for i in order)


def _kruskal_safe(values_by_group: list[np.ndarray]) -> tuple[float, float]:
    groups = [g for g in values_by_group if g.size >= 2 and np.std(g) > 0]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    try:
        stat, p = kruskal(*groups)
        return (float(stat), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


def _chi2_safe(contingency: np.ndarray) -> tuple[float, float]:
    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return (float("nan"), float("nan"))
    if contingency.sum() == 0:
        return (float("nan"), float("nan"))
    try:
        chi2, p, *_ = chi2_contingency(contingency)
        return (float(chi2), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--active-set-csv",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/per_curve_active_set.csv"),
    )
    ap.add_argument(
        "--oracle-csv",
        type=Path,
        default=Path("outputs/27a_ode_expressivity_audit/per_curve.csv"),
        help="27a per-curve.csv for tail flags (cross321 only).",
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cluster-range", type=int, nargs=2, default=[2, 8])
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--swaps-per-sample", type=int, default=400)
    ap.add_argument("--burn-in", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--selection",
        choices=("best_excess", "best_silhouette", "lowest_p"),
        default="best_excess",
        help=(
            "How to pick the regime count. 'best_excess' = max(observed - null_mean), "
            "the strongest signal beyond marginals. 'best_silhouette' uses raw observed "
            "silhouette (vulnerable to outlier-isolation at small n). 'lowest_p' picks "
            "by empirical p-value vs null."
        ),
    )
    ap.add_argument(
        "--min-regime-size",
        type=int,
        default=10,
        help=(
            "Reject regime counts where any regime has fewer members than this; "
            "prevents degenerate outlier-isolation solutions."
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/33_active_set_regimes"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sim = PLGABiphasic()
    param_names = list(sim.param_names)

    df_in = pd.read_csv(args.active_set_csv)
    active_col = f"active_set_k{args.k}"
    if active_col not in df_in.columns:
        raise KeyError(f"missing column {active_col}")

    df_in = df_in.copy()
    df_in["_active"] = df_in[active_col].apply(_parse_active_set)
    df_in = df_in[df_in["_active"].apply(len) == args.k].reset_index(drop=True)

    active_sets = df_in["_active"].tolist()
    M = _build_membership(active_sets, param_names)
    cluster_range = range(args.cluster_range[0], args.cluster_range[1] + 1)

    # Observed silhouette curve
    obs_sil, D, Z = _cluster_silhouette_curve(M, cluster_range)

    # Null distribution
    null_df = _null_silhouette_distribution(
        M,
        cluster_range=cluster_range,
        n_null=args.n_null,
        swaps_per_sample=args.swaps_per_sample,
        burn_in=args.burn_in,
        rng=rng,
    )
    null_df.to_csv(args.out / "null_silhouette_distribution.csv", index=False)

    # Per-n_clusters: observed vs null 95th percentile
    null_stats: list[dict[str, object]] = []
    for n_clusters in cluster_range:
        sub = null_df[null_df["n_clusters"] == n_clusters]["silhouette"].dropna()
        obs = obs_sil[n_clusters]
        if sub.empty:
            null_stats.append({
                "n_clusters": n_clusters, "observed": obs,
                "null_mean": float("nan"), "null_std": float("nan"),
                "null_p95": float("nan"), "p_value": float("nan"),
            })
            continue
        null_p95 = float(np.percentile(sub, 95))
        p_value = float((sub >= obs).mean()) if not np.isnan(obs) else float("nan")
        null_stats.append({
            "n_clusters": n_clusters,
            "observed": float(obs),
            "null_mean": float(sub.mean()),
            "null_std": float(sub.std(ddof=1)),
            "null_p95": null_p95,
            "p_value": p_value,
        })
    null_stats_df = pd.DataFrame(null_stats)

    # Decide best n_clusters honestly. Filter out degenerate cuts where any
    # regime has fewer than args.min_regime_size members.
    cand_rows: list[dict[str, object]] = []
    for r in null_stats:
        n_c = int(r["n_clusters"])
        labs = fcluster(Z, t=n_c, criterion="maxclust")
        _, counts = np.unique(labs, return_counts=True)
        cand_rows.append({
            "n_clusters": n_c,
            "observed": r["observed"],
            "null_mean": r["null_mean"],
            "excess": r["observed"] - r["null_mean"] if not np.isnan(r["observed"]) else float("nan"),
            "p_value": r["p_value"],
            "min_regime_size": int(counts.min()),
        })
    cand_df = pd.DataFrame(cand_rows)
    eligible = cand_df[cand_df["min_regime_size"] >= args.min_regime_size]
    if eligible.empty:
        # fall back to all candidates
        eligible = cand_df

    if args.selection == "best_excess":
        best_n = int(eligible.loc[eligible["excess"].idxmax(), "n_clusters"])
    elif args.selection == "lowest_p":
        best_n = int(eligible.loc[eligible["p_value"].idxmin(), "n_clusters"])
    else:  # best_silhouette
        best_n = int(eligible.loc[eligible["observed"].idxmax(), "n_clusters"])
    best_sil = float(obs_sil[best_n])

    # Plot silhouette curve with null band
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ks = [r["n_clusters"] for r in null_stats]
    obs_vals = [r["observed"] for r in null_stats]
    null_mean = [r["null_mean"] for r in null_stats]
    null_p95 = [r["null_p95"] for r in null_stats]
    ax.plot(ks, obs_vals, "o-", color="tab:red", label="observed")
    ax.plot(ks, null_mean, "s--", color="tab:gray", label="null mean (marginal-preserving)")
    ax.plot(ks, null_p95, "x--", color="tab:gray", alpha=0.6, label="null 95th percentile")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("silhouette score (Jaccard, avg linkage)")
    ax.set_title("Silhouette curve: observed vs marginal-preserving null")
    ax.legend()
    fig.savefig(args.out / "silhouette_curve.png", dpi=150)
    plt.close(fig)

    # Assign labels at best n_clusters
    labels = fcluster(Z, t=best_n, criterion="maxclust")
    df_in = df_in.assign(regime=labels)
    df_in[[DATASET_COL, "fid", "regime", active_col]].to_csv(
        args.out / "regime_assignments.csv", index=False
    )

    # Per-regime summary
    regime_rows: list[dict[str, object]] = []
    regime_active_grid = np.zeros((best_n, len(param_names)), dtype=float)
    for c in range(1, best_n + 1):
        mask = labels == c
        M_c = M[mask]
        size = int(mask.sum())
        size_int = int(((labels == c) & (df_in[DATASET_COL] == "internal181")).sum())
        size_cross = int(((labels == c) & (df_in[DATASET_COL] == "cross321")).sum())
        modal_set = _regime_modal_active(M_c, param_names, top=args.k)
        # Within-regime active diversity = entropy of active-set strings
        sets_str = df_in.loc[mask, active_col].astype(str)
        vc = sets_str.value_counts(normalize=True)
        entropy = float(-(vc * np.log2(vc + 1e-12)).sum())
        regime_active_grid[c - 1] = M_c.mean(axis=0) if size > 0 else 0.0
        regime_rows.append({
            "regime": c,
            "size": size,
            "size_internal181": size_int,
            "size_cross321": size_cross,
            "frac_internal": size_int / max(size, 1),
            "modal_active_set": modal_set,
            "n_distinct_active_sets": int(sets_str.nunique()),
            "entropy_active_sets_bits": entropy,
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(args.out / "regime_summary.csv", index=False)

    # Heatmap: regime x param inclusion fraction
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * best_n + 1.6), constrained_layout=True)
    im = ax.imshow(regime_active_grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticks(range(best_n))
    ax.set_yticklabels([f"R{c+1} (n={regime_df.iloc[c]['size']})" for c in range(best_n)])
    for i in range(best_n):
        for j in range(len(param_names)):
            v = regime_active_grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("within-regime inclusion frequency")
    ax.set_title(f"Regime active-parameter profile (k={args.k}, n_regimes={best_n})")
    fig.savefig(args.out / "regime_active_set_heatmap.png", dpi=150)
    plt.close(fig)

    # Dendrogram (truncated for readability)
    fig, ax = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
    dendrogram(Z, no_labels=True, color_threshold=0.0, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram (Jaccard, average linkage)")
    ax.set_ylabel("Jaccard distance")
    fig.savefig(args.out / "dendrogram.png", dpi=150)
    plt.close(fig)

    # Reverse-check on external descriptors
    # Merge in 27a's tail flags for cross321
    df_oracle = pd.read_csv(args.oracle_csv)
    flag_cols = [c for c in ["fast_regime", "short_window", "boundary_hits"] if c in df_oracle.columns]
    df_in_x = df_in.merge(
        df_oracle[["fid"] + flag_cols].rename(columns={c: f"oracle_{c}" for c in flag_cols}),
        on="fid",
        how="left",
    )

    descriptor_results: list[dict[str, object]] = []
    # Continuous descriptors (Kruskal-Wallis)
    for col, label, restrict_dataset in [
        ("t_max", "t_max", None),
        ("n_obs", "n_obs", None),
        ("full_r2", "full_r2", None),
        ("oracle_boundary_hits", "boundary_hits", "cross321"),
        ("Q_max_full", "Q_max_full", None),
    ]:
        if col not in df_in_x.columns:
            continue
        sub = df_in_x if restrict_dataset is None else df_in_x[df_in_x[DATASET_COL] == restrict_dataset]
        groups = [sub.loc[sub["regime"] == c, col].dropna().to_numpy() for c in range(1, best_n + 1)]
        stat, p = _kruskal_safe(groups)
        descriptor_results.append({
            "descriptor": label,
            "type": "continuous",
            "restricted_to": restrict_dataset or "all",
            "stat_or_chi2": stat,
            "p_value": p,
            "n_used": int(sum(g.size for g in groups)),
        })
    # Categorical descriptors (chi-square)
    for col, label, restrict_dataset in [
        ("oracle_fast_regime", "fast_regime", "cross321"),
        ("oracle_short_window", "short_window", "cross321"),
        (DATASET_COL, "dataset", None),
    ]:
        if col not in df_in_x.columns:
            continue
        sub = df_in_x if restrict_dataset is None else df_in_x[df_in_x[DATASET_COL] == restrict_dataset]
        tab = pd.crosstab(sub["regime"], sub[col])
        stat, p = _chi2_safe(tab.to_numpy())
        descriptor_results.append({
            "descriptor": label,
            "type": "categorical",
            "restricted_to": restrict_dataset or "all",
            "stat_or_chi2": stat,
            "p_value": p,
            "n_used": int(sub.shape[0]),
        })
    descriptor_df = pd.DataFrame(descriptor_results)
    descriptor_df.to_csv(args.out / "regime_descriptor_tests.csv", index=False)

    # Descriptor panels
    panels = [
        ("t_max", "continuous", None),
        ("n_obs", "continuous", None),
        ("full_r2", "continuous", None),
        ("oracle_boundary_hits", "continuous", "cross321"),
        ("Q_max_full", "continuous", None),
        ("oracle_fast_regime", "categorical", "cross321"),
    ]
    panels = [p for p in panels if p[0] in df_in_x.columns]
    n_panels = len(panels)
    if n_panels > 0:
        ncols = 3
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for idx, (col, kind, restrict) in enumerate(panels):
            ax = axes[idx // ncols, idx % ncols]
            sub = df_in_x if restrict is None else df_in_x[df_in_x[DATASET_COL] == restrict]
            if kind == "continuous":
                data = [sub.loc[sub["regime"] == c, col].dropna().to_numpy()
                        for c in range(1, best_n + 1)]
                ax.boxplot(data, labels=[f"R{c}" for c in range(1, best_n + 1)], showfliers=False)
                ax.set_ylabel(col)
            else:
                tab = pd.crosstab(sub["regime"], sub[col], normalize="index")
                bottom = np.zeros(tab.shape[0])
                for cat in tab.columns:
                    vals = tab[cat].to_numpy()
                    ax.bar([f"R{c}" for c in tab.index], vals, bottom=bottom, label=str(cat))
                    bottom = bottom + vals
                ax.set_ylim(0, 1)
                ax.set_ylabel(col + " (fraction)")
                ax.legend(fontsize=8, loc="upper right")
            title = f"{col}" + (f" [{restrict}]" if restrict else "")
            ax.set_title(title)
            ax.tick_params(axis="x", labelsize=8)
        # Hide extra axes
        for idx in range(n_panels, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.suptitle("Reverse-check: external descriptors across regimes", fontsize=13)
        fig.savefig(args.out / "regime_descriptor_panels.png", dpi=150)
        plt.close(fig)

    # Summary text
    lines: list[str] = [
        "=== 33 -- active-set regime discovery ===",
        "",
        f"input               : {args.active_set_csv}",
        f"k (active set size) : {args.k}",
        f"n_curves used       : {len(df_in)}",
        f"cluster range       : {args.cluster_range[0]}..{args.cluster_range[1]}",
        f"n_null permutations : {args.n_null}",
        "",
        "--- silhouette curve (observed vs marginal-preserving null) ---",
        f"  {'n_clusters':>10}  {'observed':>10}  {'null_mean':>10}  {'null_p95':>10}  {'p_value':>10}",
    ]
    for r in null_stats:
        lines.append(
            f"  {int(r['n_clusters']):>10}  "
            f"{r['observed']:>10.4f}  "
            f"{r['null_mean']:>10.4f}  "
            f"{r['null_p95']:>10.4f}  "
            f"{r['p_value']:>10.4f}"
        )
    lines.append("")
    lines.append(f"best n_clusters by silhouette : {best_n}  (silhouette={best_sil:.4f})")
    obs_at_best = best_sil
    null_at_best_mean = null_stats_df.set_index("n_clusters").loc[best_n, "null_mean"]
    null_at_best_p95 = null_stats_df.set_index("n_clusters").loc[best_n, "null_p95"]
    p_at_best = null_stats_df.set_index("n_clusters").loc[best_n, "p_value"]
    lines.append(
        f"  vs null at best n   : observed={obs_at_best:.4f} | "
        f"null mean={null_at_best_mean:.4f} | "
        f"null 95% = {null_at_best_p95:.4f} | "
        f"p={p_at_best:.4f}"
    )
    pass_clusters = bool(obs_at_best > null_at_best_p95)
    lines.append(
        f"  CRITERION A (obs > null 95th pct) : {'PASS' if pass_clusters else 'FAIL'}"
    )
    lines.append("")

    lines.append("--- regime composition (at best n_clusters) ---")
    for _, r in regime_df.iterrows():
        lines.append(
            f"  R{int(r['regime'])}: n={int(r['size'])} "
            f"(internal={int(r['size_internal181'])}, cross={int(r['size_cross321'])}, "
            f"frac_internal={r['frac_internal']:.2f})"
        )
        lines.append(
            f"      modal active set      : {r['modal_active_set']}"
        )
        lines.append(
            f"      n distinct active sets: {int(r['n_distinct_active_sets'])}  "
            f"(entropy={r['entropy_active_sets_bits']:.2f} bits)"
        )
    lines.append("")

    # CRITERION C: both datasets non-trivially populate same regime
    cross_pop_ok = bool(
        ((regime_df["size_internal181"] >= 5) & (regime_df["size_cross321"] >= 5)).sum() >= 2
    )
    lines.append(
        f"  CRITERION C (>=2 regimes have >=5 curves from BOTH datasets) : "
        f"{'PASS' if cross_pop_ok else 'FAIL'}"
    )
    lines.append("")

    lines.append("--- external descriptor tests (reverse check) ---")
    lines.append(f"  {'descriptor':>16}  {'type':>11}  {'subset':>10}  {'stat':>10}  {'p':>10}  {'n':>6}")
    for _, r in descriptor_df.iterrows():
        lines.append(
            f"  {str(r['descriptor']):>16}  "
            f"{str(r['type']):>11}  "
            f"{str(r['restricted_to']):>10}  "
            f"{r['stat_or_chi2']:>10.3f}  "
            f"{r['p_value']:>10.4f}  "
            f"{int(r['n_used']):>6}"
        )
    any_descriptor_pass = bool((descriptor_df["p_value"] < 0.01).any())
    lines.append("")
    lines.append(
        f"  CRITERION B (at least one external descriptor with p < 0.01) : "
        f"{'PASS' if any_descriptor_pass else 'FAIL'}"
    )
    lines.append("")

    lines.append("--- overall verdict ---")
    verdict_pass = pass_clusters and any_descriptor_pass and cross_pop_ok
    if verdict_pass:
        lines.append("  REGIME STRUCTURE REAL (passes A + B + C)")
        lines.append("  Implication: discrete regime-conditional parameterization is justified")
        lines.append("  before a continuous latent-driver reparameterization.")
    elif pass_clusters and not any_descriptor_pass:
        lines.append("  STATISTICAL CLUSTERS EXIST BUT LACK EXTERNAL MEANING")
        lines.append("  Clusters look real numerically but no external descriptor separates them.")
        lines.append("  This is the self-deception trap: do NOT build a regime-conditional model")
        lines.append("  yet. The clusters may be greedy-selection artefacts of script 29.")
    elif not pass_clusters:
        lines.append("  NO REGIME STRUCTURE BEYOND MARGINALS")
        lines.append("  Observed silhouette does not exceed marginal-preserving null. The")
        lines.append("  apparent 'low-dimensional slices' from script 30 are NOT explained by")
        lines.append("  curves clustering into a small number of discrete active-set regimes.")
        lines.append("  Next move should be: revisit script 30 to test whether the low local")
        lines.append("  effective dimension is shared (continuous manifold) or per-curve")
        lines.append("  idiosyncratic.")
    else:
        lines.append("  PARTIAL: clusters and external meaning, but dataset-specific.")
        lines.append("  Regimes may be data-collection artefacts; not safe to model on yet.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
