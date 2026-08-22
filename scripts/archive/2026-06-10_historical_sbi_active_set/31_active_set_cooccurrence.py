"""
31 - Active-set co-occurrence heatmap from script 29 results.

What this does:
    1. Read per-curve active-set diagnostics from script 29.
    2. Parse the chosen active set column (default: active_set_k4).
    3. Compute, for each dataset, how often each parameter appears in the
       active set and how often parameter pairs co-appear in the same curve.
    4. Plot side-by-side co-occurrence heatmaps and inclusion-frequency bars.

Why this exists:
    Script 29 showed that many curves only need 2-4 active parameters to
    recover near-full-fit quality. The next question is whether every curve
    uses the same small subset, or whether different curves live on different
    low-dimensional slices of the 9-D mechanism family.

Outputs:
    outputs/31_active_set_cooccurrence/inclusion_rates_k4.csv
    outputs/31_active_set_cooccurrence/top_active_sets_k4.csv
    outputs/31_active_set_cooccurrence/cooccurrence_internal181_k4.csv
    outputs/31_active_set_cooccurrence/cooccurrence_cross321_k4.csv
    outputs/31_active_set_cooccurrence/active_set_cooccurrence_k4.png
    outputs/31_active_set_cooccurrence/summary.txt

Expected runtime:
    Seconds. Pure post-processing on script 29 outputs.
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


def _cooccurrence_matrix(
    active_sets: list[list[str]],
    param_names: list[str],
) -> np.ndarray:
    mat = np.zeros((len(param_names), len(param_names)), dtype=float)
    if not active_sets:
        return mat
    for aset in active_sets:
        aset_set = set(aset)
        for i, pi in enumerate(param_names):
            if pi not in aset_set:
                continue
            for j, pj in enumerate(param_names):
                if pj in aset_set:
                    mat[i, j] += 1.0
    return mat / float(len(active_sets))


def _inclusion_rates(
    active_sets: list[list[str]],
    param_names: list[str],
) -> pd.Series:
    mat = _cooccurrence_matrix(active_sets, param_names)
    return pd.Series(np.diag(mat), index=param_names, dtype=float)


def _plot_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=7,
            )
    return im


def _plot_bars(
    ax: plt.Axes,
    rates: pd.Series,
    title: str,
) -> None:
    ordered = rates.sort_values(ascending=True)
    ax.barh(ordered.index, ordered.to_numpy(), color="tab:blue")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("inclusion frequency")
    ax.set_title(title)
    for y, val in enumerate(ordered.to_numpy()):
        ax.text(min(val + 0.02, 0.98), y, f"{val:.2f}", va="center", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--active-set-csv",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/per_curve_active_set.csv"),
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["internal181", "cross321"],
        choices=("internal181", "cross321"),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/31_active_set_cooccurrence"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    active_col = f"active_set_k{args.k}"

    df = pd.read_csv(args.active_set_csv)
    if active_col not in df.columns:
        raise KeyError(f"missing required column {active_col} in {args.active_set_csv}")

    inclusion_rows: list[dict[str, object]] = []
    top_set_rows: list[dict[str, object]] = []
    mats: dict[str, np.ndarray] = {}
    rates_by_dataset: dict[str, pd.Series] = {}

    for dataset in args.datasets:
        sdf = df[df[DATASET_COL] == dataset].copy()
        active_sets = [_parse_active_set(cell) for cell in sdf[active_col]]
        mat = _cooccurrence_matrix(active_sets, param_names)
        rates = _inclusion_rates(active_sets, param_names)
        mats[dataset] = mat
        rates_by_dataset[dataset] = rates

        for name in param_names:
            inclusion_rows.append({
                "dataset": dataset,
                "k": args.k,
                "param": name,
                "inclusion_rate": float(rates[name]),
            })

        vc = sdf[active_col].value_counts()
        for active_set, count in vc.items():
            top_set_rows.append({
                "dataset": dataset,
                "k": args.k,
                "active_set": str(active_set),
                "count": int(count),
                "fraction": float(count / len(sdf)),
            })

        pd.DataFrame(mat, index=param_names, columns=param_names).to_csv(
            args.out / f"cooccurrence_{dataset}_k{args.k}.csv"
        )

    inclusion_df = pd.DataFrame(inclusion_rows)
    top_sets_df = pd.DataFrame(top_set_rows)
    inclusion_df.to_csv(args.out / f"inclusion_rates_k{args.k}.csv", index=False)
    top_sets_df.to_csv(args.out / f"top_active_sets_k{args.k}.csv", index=False)

    fig, axes = plt.subplots(
        len(args.datasets),
        2,
        figsize=(14, 5 * len(args.datasets)),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )
    if len(args.datasets) == 1:
        axes = np.array([axes])

    image_handle = None
    for row_idx, dataset in enumerate(args.datasets):
        image_handle = _plot_heatmap(
            axes[row_idx, 0],
            mats[dataset],
            param_names,
            title=f"{dataset} active-set co-occurrence (k={args.k})",
        )
        _plot_bars(
            axes[row_idx, 1],
            rates_by_dataset[dataset],
            title=f"{dataset} inclusion frequency (k={args.k})",
        )
    assert image_handle is not None
    cb = fig.colorbar(image_handle, ax=axes[:, 0], shrink=0.85)
    cb.set_label("fraction of curves where param pair co-appears")
    fig.suptitle(
        "Per-curve active-parameter structure: shared core vs curve-specific slices",
        fontsize=14,
    )
    fig.savefig(args.out / f"active_set_cooccurrence_k{args.k}.png", dpi=150)
    plt.close(fig)

    lines = [
        f"=== 31 -- active-set co-occurrence (k={args.k}) ===",
        "",
        f"active_set_csv : {args.active_set_csv}",
        "",
    ]
    for dataset in args.datasets:
        sdf = top_sets_df[top_sets_df["dataset"] == dataset].copy()
        top1 = sdf.iloc[0]
        lines.append(f"{dataset}:")
        lines.append(f"  n_curves        : {int((df[DATASET_COL] == dataset).sum())}")
        lines.append(
            f"  top active set  : {top1['active_set']} "
            f"(count={int(top1['count'])}, frac={float(top1['fraction']):.3f})"
        )
        ordered = rates_by_dataset[dataset].sort_values(ascending=False)
        lines.append("  top inclusion rates:")
        for name, rate in ordered.head(5).items():
            lines.append(f"    {name:15s} {rate:.3f}")
        lines.append("  weakest inclusion rates:")
        for name, rate in ordered.tail(3).items():
            lines.append(f"    {name:15s} {rate:.3f}")
        lines.append("")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
