"""
32 - Substitution-vs-complementarity test for active-set membership.

What this does:
    For each chosen active-set size k in {2, 3, 4}, take the per-curve
    k-subsets from script 29 and test, for each parameter pair, whether
    they co-occur more or less than expected under a null that preserves:
      - per-curve cardinality (exactly k active),
      - per-parameter marginal inclusion rate,
    using a row+column-marginal-preserving swap permutation of the binary
    curve x param membership matrix.

Why this exists:
    Naive lift = observed / (marginal * marginal) is biased because the
    fixed cardinality k forces negative dependence on average. The
    swap-preserving null removes that artifact, so any remaining signal
    means "this pair gets picked together more / less than the marginals
    alone would explain".

What we want to learn:
    - substitute pairs (z < 0): two parameters that play the same role
      and rarely co-appear; candidates for collapsing into one latent
      direction.
    - complementary pairs (z > 0): two parameters that habitually fire
      together; candidates for a joint driver / coupling layer.
    Cross-dataset consistency (same sign and |z| >= 2 on BOTH
    internal181 and cross321) is treated as the bar for taking a pair
    seriously, not single-dataset significance alone.

Caveats (read these before believing the heatmap):
    - The active sets come from a *greedy* fit in script 29. If two
      parameters are near-substitutes, greedy can pick whichever lowers
      residual first and then under-recruit the other -> this looks like
      substitution even when both could equally well fit. Cross-dataset
      consistency partly guards against this but does not eliminate it.
    - "Co-occurrence in active set" is a property of the fitting
      diagnostic, not a direct statement about the ODE Jacobian. Reading
      a pair as "shared mechanism direction" still needs script 30 to
      corroborate (a substitute pair should also show high alignment in
      the local Jacobian SVD basis).
    - With 36 pairs and B permutations, Bonferroni-corrected significance
      needs |z| roughly >= 3.4. We report |z| <= 2 as "noise", 2-3 as
      "hint", >= 3 as "claim", and >= 3.4 with cross-dataset agreement
      as "strong claim".

Outputs:
    outputs/32_active_set_substitution/pair_stats_k{k}.csv
    outputs/32_active_set_substitution/zscore_heatmap_k{k}.png
    outputs/32_active_set_substitution/cross_dataset_consistent.csv
    outputs/32_active_set_substitution/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402

DATASET_COL = "dataset"


def _parse_active_set(cell: object) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [token.strip() for token in cell.split(",") if token.strip()]


def _build_membership_matrix(
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
    """One swap step preserving row and column marginals.

    Picks two rows, finds a (1,0) / (0,1) column pair, swaps the bits.
    Returns True if a swap was performed.
    """
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


def _null_pair_samples(
    M0: np.ndarray,
    n_samples: int,
    swaps_per_sample: int,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n_samples null pair-count matrices (P x P)."""
    M = M0.copy()
    for _ in range(burn_in):
        _attempt_swap(M, rng)
    P = M.shape[1]
    stack = np.zeros((n_samples, P, P), dtype=np.float64)
    for b in range(n_samples):
        for _ in range(swaps_per_sample):
            _attempt_swap(M, rng)
        stack[b] = M.T @ M
    return stack


def _pair_stats(
    M: np.ndarray,
    null_stack: np.ndarray,
    param_names: list[str],
) -> pd.DataFrame:
    observed = (M.T @ M).astype(np.float64)
    mu = null_stack.mean(axis=0)
    sd = null_stack.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (observed - mu) / sd, 0.0)
    # two-sided empirical p with +1 smoothing
    obs_dev = np.abs(observed - mu)
    null_dev = np.abs(null_stack - mu[None, :, :])
    p_emp = (np.sum(null_dev >= obs_dev[None, :, :], axis=0) + 1.0) / (
        null_stack.shape[0] + 1.0
    )

    rows: list[dict[str, object]] = []
    P = len(param_names)
    n_curves = float(M.shape[0])
    for i in range(P):
        for j in range(i + 1, P):
            rows.append({
                "param_a": param_names[i],
                "param_b": param_names[j],
                "obs_count": float(observed[i, j]),
                "obs_freq": float(observed[i, j] / n_curves),
                "null_mean": float(mu[i, j]),
                "null_std": float(sd[i, j]),
                "z": float(z[i, j]),
                "p_emp_two_sided": float(p_emp[i, j]),
                "marg_a": float(observed[i, i] / n_curves),
                "marg_b": float(observed[j, j] / n_curves),
            })
    return pd.DataFrame(rows)


def _plot_z_heatmap(
    ax: plt.Axes,
    z_full: np.ndarray,
    param_names: list[str],
    title: str,
    vlim: float,
) -> "plt.cm.ScalarMappable":
    # Mask diagonal
    Z = z_full.copy()
    np.fill_diagonal(Z, np.nan)
    im = ax.imshow(Z, vmin=-vlim, vmax=vlim, cmap="RdBu_r")
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    ax.set_title(title)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            if i == j:
                continue
            val = z_full[i, j]
            ax.text(
                j,
                i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.6 * vlim else "white",
                fontsize=7,
            )
    return im


def _zscore_matrix_from_df(df: pd.DataFrame, param_names: list[str]) -> np.ndarray:
    P = len(param_names)
    idx = {n: i for i, n in enumerate(param_names)}
    Z = np.zeros((P, P), dtype=np.float64)
    for _, row in df.iterrows():
        i = idx[row["param_a"]]
        j = idx[row["param_b"]]
        Z[i, j] = row["z"]
        Z[j, i] = row["z"]
    return Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--active-set-csv",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/per_curve_active_set.csv"),
    )
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["internal181", "cross321"],
        choices=("internal181", "cross321"),
    )
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--swaps-per-sample", type=int, default=200)
    ap.add_argument("--burn-in", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/32_active_set_substitution")
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    P = len(param_names)

    df_in = pd.read_csv(args.active_set_csv)

    summary_lines: list[str] = [
        "=== 32 -- active-set substitution / complementarity ===",
        "",
        f"input             : {args.active_set_csv}",
        f"datasets          : {args.datasets}",
        f"k values          : {args.ks}",
        f"null permutations : {args.n_null}",
        f"swaps per sample  : {args.swaps_per_sample}",
        f"burn-in swaps     : {args.burn_in}",
        f"seed              : {args.seed}",
        "",
        "Reading guide:",
        "  z > 0  -> pair appears together MORE than marginals predict (complementary)",
        "  z < 0  -> pair appears together LESS than marginals predict (substitute)",
        "  |z| <= 2.0 : noise",
        "  2-3       : hint",
        "  >= 3      : claim",
        "  >= 3.4 AND consistent sign on both datasets : strong claim",
        "",
    ]

    all_z_by_k_dataset: dict[tuple[int, str], pd.DataFrame] = {}

    for k in args.ks:
        active_col = f"active_set_k{k}"
        if active_col not in df_in.columns:
            raise KeyError(f"missing column {active_col}")

        # one heatmap row per dataset
        fig, axes = plt.subplots(
            1,
            len(args.datasets),
            figsize=(6.5 * len(args.datasets), 5.5),
            constrained_layout=True,
        )
        if len(args.datasets) == 1:
            axes = np.array([axes])

        # Decide a common vlim across datasets for this k
        per_dataset_z: dict[str, np.ndarray] = {}
        per_dataset_df: dict[str, pd.DataFrame] = {}

        for dataset in args.datasets:
            sdf = df_in[df_in[DATASET_COL] == dataset]
            active_sets = [_parse_active_set(c) for c in sdf[active_col]]
            active_sets = [a for a in active_sets if len(a) == k]
            if not active_sets:
                summary_lines.append(f"[k={k}] {dataset}: no rows with cardinality {k}; skipped")
                summary_lines.append("")
                continue
            M = _build_membership_matrix(active_sets, param_names)
            # Skip degenerate cases
            col_sums = M.sum(axis=0)
            if (col_sums == 0).any() or (col_sums == M.shape[0]).any():
                # Some params never / always selected at this k. Still meaningful
                # but the null is degenerate for those columns. We continue; z
                # will be reported as 0 where std=0.
                pass
            null_stack = _null_pair_samples(
                M,
                n_samples=args.n_null,
                swaps_per_sample=args.swaps_per_sample,
                burn_in=args.burn_in,
                rng=rng,
            )
            df_pairs = _pair_stats(M, null_stack, param_names)
            df_pairs.insert(0, "dataset", dataset)
            df_pairs.insert(1, "k", k)
            per_dataset_df[dataset] = df_pairs
            per_dataset_z[dataset] = _zscore_matrix_from_df(df_pairs, param_names)
            all_z_by_k_dataset[(k, dataset)] = df_pairs

        # Save combined CSV for this k
        if per_dataset_df:
            df_k = pd.concat(per_dataset_df.values(), ignore_index=True)
            df_k.to_csv(args.out / f"pair_stats_k{k}.csv", index=False)

        # Compute shared color range
        all_z = np.concatenate([z.ravel() for z in per_dataset_z.values()]) if per_dataset_z else np.array([0.0])
        vlim = max(2.0, float(np.nanmax(np.abs(all_z))) if all_z.size else 2.0)

        im = None
        for ax, dataset in zip(axes, args.datasets):
            if dataset not in per_dataset_z:
                ax.set_visible(False)
                continue
            im = _plot_z_heatmap(
                ax,
                per_dataset_z[dataset],
                param_names,
                title=f"{dataset} pair z-score (k={k}, B={args.n_null})",
                vlim=vlim,
            )
        if im is not None:
            cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.85)
            cb.set_label("z-score: + complementary, - substitute")
        fig.suptitle(
            f"Active-set pair test, k={k}: marginal-preserving permutation null",
            fontsize=13,
        )
        fig.savefig(args.out / f"zscore_heatmap_k{k}.png", dpi=150)
        plt.close(fig)

        # Per-k textual summary
        summary_lines.append(f"--- k = {k} ---")
        for dataset, df_pairs in per_dataset_df.items():
            n = int((df_in[DATASET_COL] == dataset).sum())
            n_used = int(((df_in[DATASET_COL] == dataset) &
                          (df_in[active_col].apply(lambda c: len(_parse_active_set(c)) == k))
                          ).sum())
            df_sorted = df_pairs.sort_values("z")
            top_sub = df_sorted.head(5)
            top_comp = df_sorted.tail(5).iloc[::-1]
            summary_lines.append(
                f"[{dataset}] n_curves={n}, n_used_at_k={n_used}"
            )
            summary_lines.append("  most substitute (z most negative):")
            for _, r in top_sub.iterrows():
                summary_lines.append(
                    f"    {r['param_a']:15s} ~ {r['param_b']:15s}  z={r['z']:+.2f}  "
                    f"obs={r['obs_freq']:.3f}  null_mean={r['null_mean'] / n_used:.3f}  "
                    f"p={r['p_emp_two_sided']:.4f}"
                )
            summary_lines.append("  most complementary (z most positive):")
            for _, r in top_comp.iterrows():
                summary_lines.append(
                    f"    {r['param_a']:15s} ~ {r['param_b']:15s}  z={r['z']:+.2f}  "
                    f"obs={r['obs_freq']:.3f}  null_mean={r['null_mean'] / n_used:.3f}  "
                    f"p={r['p_emp_two_sided']:.4f}"
                )
            summary_lines.append("")

    # Cross-dataset consistency report (per k)
    consistent_rows: list[dict[str, object]] = []
    bonferroni_threshold_z = 3.4  # approx for 36 pairs, B=2000
    for k in args.ks:
        if (k, "internal181") not in all_z_by_k_dataset:
            continue
        if (k, "cross321") not in all_z_by_k_dataset:
            continue
        df_i = all_z_by_k_dataset[(k, "internal181")].set_index(["param_a", "param_b"])
        df_c = all_z_by_k_dataset[(k, "cross321")].set_index(["param_a", "param_b"])
        for key in df_i.index.intersection(df_c.index):
            zi = float(df_i.loc[key, "z"])
            zc = float(df_c.loc[key, "z"])
            if zi == 0.0 and zc == 0.0:
                continue
            same_sign = (zi > 0 and zc > 0) or (zi < 0 and zc < 0)
            min_abs = min(abs(zi), abs(zc))
            consistent_rows.append({
                "k": k,
                "param_a": key[0],
                "param_b": key[1],
                "z_internal181": zi,
                "z_cross321": zc,
                "same_sign": bool(same_sign),
                "min_abs_z": min_abs,
                "sign": "complement" if zi > 0 and zc > 0 else ("substitute" if zi < 0 and zc < 0 else "mixed"),
            })

    if consistent_rows:
        df_cons = pd.DataFrame(consistent_rows).sort_values(
            ["k", "same_sign", "min_abs_z"], ascending=[True, False, False]
        )
        df_cons.to_csv(args.out / "cross_dataset_consistent.csv", index=False)

        summary_lines.append("--- cross-dataset consistency (same sign on BOTH datasets) ---")
        for k in args.ks:
            df_k = df_cons[(df_cons["k"] == k) & (df_cons["same_sign"])]
            if df_k.empty:
                summary_lines.append(f"[k={k}] no pair has same-sign agreement on both datasets")
                summary_lines.append("")
                continue
            strong = df_k[df_k["min_abs_z"] >= bonferroni_threshold_z]
            claim = df_k[(df_k["min_abs_z"] >= 3.0) & (df_k["min_abs_z"] < bonferroni_threshold_z)]
            hint = df_k[(df_k["min_abs_z"] >= 2.0) & (df_k["min_abs_z"] < 3.0)]
            summary_lines.append(f"[k={k}] strong (min|z| >= {bonferroni_threshold_z}, both datasets):")
            for _, r in strong.iterrows():
                summary_lines.append(
                    f"    {r['param_a']:15s} ~ {r['param_b']:15s}  "
                    f"z_int={r['z_internal181']:+.2f}  z_cross={r['z_cross321']:+.2f}  [{r['sign']}]"
                )
            if strong.empty:
                summary_lines.append("    (none)")
            summary_lines.append(f"[k={k}] claim (3.0 <= min|z| < {bonferroni_threshold_z}):")
            for _, r in claim.iterrows():
                summary_lines.append(
                    f"    {r['param_a']:15s} ~ {r['param_b']:15s}  "
                    f"z_int={r['z_internal181']:+.2f}  z_cross={r['z_cross321']:+.2f}  [{r['sign']}]"
                )
            if claim.empty:
                summary_lines.append("    (none)")
            summary_lines.append(f"[k={k}] hint (2.0 <= min|z| < 3.0):")
            for _, r in hint.iterrows():
                summary_lines.append(
                    f"    {r['param_a']:15s} ~ {r['param_b']:15s}  "
                    f"z_int={r['z_internal181']:+.2f}  z_cross={r['z_cross321']:+.2f}  [{r['sign']}]"
                )
            if hint.empty:
                summary_lines.append("    (none)")
            summary_lines.append("")

    summary_lines.append("--- self-deception checklist ---")
    summary_lines.append("  1. greedy-selection bias: if a strong substitute pair is reported,")
    summary_lines.append("     verify by checking script 30 SVD: substitute params should share")
    summary_lines.append("     the same dominant local Jacobian direction.")
    summary_lines.append("  2. cardinality artifact: the null preserves k, so do NOT interpret")
    summary_lines.append("     the average negative z as 'everything substitutes for everything';")
    summary_lines.append("     by construction the per-pair null mean already accounts for it.")
    summary_lines.append("  3. low-marginal params (log_alpha, m_crit) have small obs counts, so")
    summary_lines.append("     their null std is large and z is hard to drive past 3.4 even if")
    summary_lines.append("     the underlying coupling is real. Treat their non-significance as")
    summary_lines.append("     uninformative, not as evidence of independence.")

    out_path = args.out / "summary.txt"
    out_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
