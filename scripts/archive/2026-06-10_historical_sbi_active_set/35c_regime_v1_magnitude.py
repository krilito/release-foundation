"""
35c - Follow-up to 35b: regime structure in |v1| magnitude space (not direction).

What this does:
    35b tested whether regimes separate in v1 DIRECTION (cosine distance,
    sign-aligned). Result: negative. But we noticed in 35b's per-regime
    mean |v1| table that every regime has log_kd as the dominant direction,
    which makes the direction-based test trivially fail.

    This script asks a different question that is conceptually closer to
    what the active-set diagnostic measures: do regimes have characteristically
    different patterns of WHICH PARAMETERS ARE SENSITIVE, regardless of
    which one happens to be largest?

    Two tests:

    (1) Magnitude silhouette: each curve -> 9-D vector |v1| (each entry
        l1-normalized so it sums to 1, removing magnitude scale).
        Euclidean distance, silhouette of regime labels, vs label-shuffle
        null. This is the "do regimes have different parameter-sensitivity
        profiles" test.

    (2) Per-curve top-k overlap: for each curve, compare the greedy
        active set (from 29, k=4) with the top-4 parameters by |v1|.
        Distribution of overlap counts (0..4) tells us whether greedy
        and local linear geometry agree at the per-curve level. If most
        curves have overlap >= 3, then 35b's regime-level alignment
        mismatch is an averaging artifact, not a per-curve disagreement.

Why this exists:
    35b's negative result could be either:
      (a) regimes really do not correspond to mechanism subspaces, or
      (b) v1 direction is a bad summary statistic for what regimes encode.
    These two tests distinguish (a) from (b). If (1) passes, the regime
    structure has geometric backing; only the direction-based view was
    wrong. If (2) shows high per-curve overlap, then 29's algebraic
    diagnostic and the geometric one agree per curve, and the discrepancy
    is purely at the regime aggregation level.

Outputs:
    outputs/35c_regime_v1_magnitude/magnitude_silhouette.txt
    outputs/35c_regime_v1_magnitude/per_curve_overlap.csv
    outputs/35c_regime_v1_magnitude/overlap_histogram.png
    outputs/35c_regime_v1_magnitude/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402

DATASET_COL = "dataset"
FID_COL = "fid"


def _parse_active_set(cell: object) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [token.strip() for token in cell.split(",") if token.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-curve-v1",
        type=Path,
        default=Path("outputs/35b_regime_jacobian_alignment/per_curve_v1.csv"),
    )
    ap.add_argument(
        "--active-set-csv",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/per_curve_active_set.csv"),
    )
    ap.add_argument("--k-active", type=int, default=4)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/35c_regime_v1_magnitude"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    v1_cols = [f"v1_{n}" for n in param_names]

    df = pd.read_csv(args.per_curve_v1)
    missing = [c for c in v1_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing v1 columns in per_curve_v1.csv: {missing}")

    V_abs = df[v1_cols].to_numpy(dtype=float)
    V_abs = np.abs(V_abs)
    # l1-normalize each row so silhouette tests the *shape* of the
    # sensitivity profile, not its overall magnitude.
    row_l1 = V_abs.sum(axis=1, keepdims=True)
    V_norm = V_abs / np.where(row_l1 > 1e-12, row_l1, 1.0)
    labels = df["regime"].astype(int).to_numpy()
    n_curves = len(df)
    n_regimes = len(np.unique(labels))

    # --- Test 1: silhouette on l1-normalized |v1| magnitude profiles ---
    D = squareform(pdist(V_norm, metric="euclidean"))
    obs_sil = float(silhouette_score(D, labels, metric="precomputed"))
    perm_labels = labels.copy()
    null_sils = np.empty(args.n_null)
    for b in range(args.n_null):
        rng.shuffle(perm_labels)
        try:
            null_sils[b] = silhouette_score(D, perm_labels, metric="precomputed")
        except ValueError:
            null_sils[b] = np.nan
    null_mean = float(np.nanmean(null_sils))
    null_p95 = float(np.nanpercentile(null_sils, 95))
    p_value = float((null_sils >= obs_sil).mean())
    test1_pass = bool(obs_sil > null_p95)

    test1_lines = [
        "Silhouette of regime labels in l1-normalized |v1| magnitude space:",
        f"  observed       : {obs_sil:.4f}",
        f"  null mean      : {null_mean:.4f}",
        f"  null 95th pct  : {null_p95:.4f}",
        f"  empirical p    : {p_value:.4f}",
        f"  n_curves       : {n_curves}, n_regimes : {n_regimes}",
        f"  n_null perms   : {args.n_null}",
        f"  PASS (obs > null p95) : {test1_pass}",
    ]
    (args.out / "magnitude_silhouette.txt").write_text("\n".join(test1_lines), encoding="utf-8")

    # --- Test 2: per-curve top-k overlap between greedy active set and |v1|-top-k ---
    df_as = pd.read_csv(args.active_set_csv)
    active_col = f"active_set_k{args.k_active}"
    df_as["_aset"] = df_as[active_col].apply(_parse_active_set)
    df_as_keep = df_as[[DATASET_COL, FID_COL, "_aset"]].merge(
        df[[DATASET_COL, FID_COL, "regime"] + v1_cols],
        on=[DATASET_COL, FID_COL],
        how="inner",
    )
    overlaps: list[dict[str, object]] = []
    p_idx = {n: i for i, n in enumerate(param_names)}
    k = args.k_active
    for _, r in df_as_keep.iterrows():
        v1 = np.abs(np.array([r[c] for c in v1_cols], dtype=float))
        top_v1 = set(np.argsort(-v1)[:k].tolist())
        greedy_idx = {p_idx[name] for name in r["_aset"] if name in p_idx}
        overlap = len(top_v1 & greedy_idx)
        overlaps.append({
            "dataset": r[DATASET_COL],
            "fid": int(r[FID_COL]),
            "regime": int(r["regime"]),
            "greedy_set": ",".join(sorted(r["_aset"])),
            "v1_top4": ",".join(sorted(param_names[i] for i in top_v1)),
            "overlap_count": int(overlap),
            "jaccard": float(len(top_v1 & greedy_idx) / max(len(top_v1 | greedy_idx), 1)),
        })
    ov_df = pd.DataFrame(overlaps)
    ov_df.to_csv(args.out / "per_curve_overlap.csv", index=False)

    overlap_hist = ov_df["overlap_count"].value_counts().sort_index()

    # Plot histogram
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    counts = [int(overlap_hist.get(i, 0)) for i in range(k + 1)]
    ax.bar(range(k + 1), counts, color="darkslateblue")
    ax.set_xticks(range(k + 1))
    ax.set_xlabel(f"overlap count between greedy active set and |v1|-top-{k}")
    ax.set_ylabel("number of curves")
    ax.set_title(f"Per-curve agreement between 29 greedy and 35b |v1| top-{k}")
    for i, c in enumerate(counts):
        ax.text(i, c + 1, str(c), ha="center", va="bottom", fontsize=9)
    fig.savefig(args.out / "overlap_histogram.png", dpi=150)
    plt.close(fig)

    # Per-regime overlap stats
    per_regime_overlap = (
        ov_df.groupby("regime")["overlap_count"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    # --- Summary ---
    lines = [
        "=== 35c -- regime structure in |v1| magnitude space (35b follow-up) ===",
        "",
        f"input    : {args.per_curve_v1}",
        f"n_curves : {n_curves}, n_regimes : {n_regimes}",
        "",
        "--- Test 1: magnitude-profile silhouette ---",
    ]
    lines.extend(test1_lines)
    lines.append("")

    lines.append(f"--- Test 2: greedy vs |v1|-top-{k} per-curve overlap ---")
    lines.append(f"  overlap distribution (count of {k} possible):")
    for i in range(k + 1):
        c = int(overlap_hist.get(i, 0))
        pct = c / n_curves * 100.0
        lines.append(f"    overlap = {i}: {c:>4}  ({pct:>5.1f}%)")
    mean_overlap = float(ov_df["overlap_count"].mean())
    median_overlap = float(ov_df["overlap_count"].median())
    chance_overlap_mean = k * k / 9.0  # E[|A∩B|] for random size-k subsets of 9
    lines.append(f"  mean overlap    : {mean_overlap:.2f}")
    lines.append(f"  median overlap  : {median_overlap:.1f}")
    lines.append(f"  chance baseline : {chance_overlap_mean:.2f}  "
                 f"(random size-{k} of 9)")
    lines.append("")
    lines.append("  per regime:")
    for _, r in per_regime_overlap.iterrows():
        lines.append(
            f"    R{int(r['regime'])}: n={int(r['count'])}, "
            f"mean_overlap={r['mean']:.2f}, median={r['median']:.1f}"
        )
    lines.append("")

    # Verdict
    high_overlap_frac = float((ov_df["overlap_count"] >= k - 1).mean())
    lines.append("--- verdict ---")
    lines.append(f"  Test 1 (magnitude silhouette PASS): {test1_pass}")
    lines.append(f"  Test 2 (fraction of curves with overlap >= {k-1}): "
                 f"{high_overlap_frac:.2%}")
    lines.append("")

    if test1_pass:
        lines.append("  35B WAS A METHOD ARTIFACT.")
        lines.append("  Regimes do separate in |v1| magnitude space, just not in v1 direction.")
        lines.append("  The shared dominant axis (log_kd in all regimes) masked the real")
        lines.append("  structure in cosine-based silhouette. The 'regime carries mechanism")
        lines.append("  info' story is restored at the geometric level.")
        lines.append("  Implication: regime-conditional parameterization is defensible after")
        lines.append("  all; the natural per-regime descriptor is the *sensitivity profile*")
        lines.append("  (which params matter), not the dominant direction.")
    elif high_overlap_frac >= 0.5:
        lines.append("  REGIMES STILL DO NOT SEPARATE GEOMETRICALLY,")
        lines.append("  but greedy active set and |v1|-top-k AGREE per curve in most cases.")
        lines.append("  Interpretation: 29's diagnostic IS capturing local geometry per curve,")
        lines.append("  but the regimes built from those active sets don't carve the curves")
        lines.append("  into geometrically distinct groups. The regime label remains useful")
        lines.append("  as a non-geometric latent class label (passes 33-35a) but does NOT")
        lines.append("  correspond to a parameter-space subspace partition.")
    else:
        lines.append("  STRONG NEGATIVE.")
        lines.append("  Neither the magnitude silhouette nor the per-curve overlap recovers the")
        lines.append("  regime structure geometrically. The active-set / regime story sits")
        lines.append("  entirely on top of nonlinear / non-geometric structure. Treat any")
        lines.append("  regime-conditional model with caution; do not rely on the local linear")
        lines.append("  picture for justification.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
