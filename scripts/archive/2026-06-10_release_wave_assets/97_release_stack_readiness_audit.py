"""
97 - Audit readiness for route-consistent uncertainty-aware duration/shape.

Purpose:
    Turn the current "UQ-to-timescale gap" into a concrete stack audit:
    identify which benchmark cells already have curve outputs, mechanism-route
    timescale panels, point-family objects, calibrated-family exports, and the
    final route-consistent uncertainty-aware duration/shape layer.

Consumes:
    outputs/80_*
    outputs/81_*
    outputs/85_*
    outputs/87_release_conformal_family_export/*/*/summary.csv
    outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv

Produces:
    outputs/97_release_stack_readiness_audit/readiness_matrix.csv
    outputs/97_release_stack_readiness_audit/component_tally.csv
    outputs/97_release_stack_readiness_audit/readiness_heatmap.png
    outputs/97_release_stack_readiness_audit/readiness_heatmap.pdf
    outputs/97_release_stack_readiness_audit/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(".")
OUTDIR = Path("outputs/97_release_stack_readiness_audit")

GAP_TABLE = Path("outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonical_label(dataset: str, scheme: str) -> str:
    scheme_map = {
        "group_by_drug": "group_by_drug",
        "group_by_polymer": "group_by_polymer",
        "random_5fold": "random_5fold",
        "group_by_api": "group_by_drug",
        "group_by_method": "group_by_polymer",
    }
    return f"{dataset}__{scheme_map[scheme]}"


def display_label(key: str) -> str:
    dataset, scheme = key.split("__")
    ds = {"cross321": "C321", "internal181": "I181", "liposome": "Lipo"}[dataset]
    sc = {
        "group_by_drug": "D",
        "group_by_polymer": "P",
        "random_5fold": "R",
    }[scheme]
    return f"{ds}-{sc}"


def discover_dirs(pattern: str) -> list[str]:
    return [p.name for p in Path("outputs").glob(pattern) if p.is_dir()]


def mechanism_curve_keys() -> set[str]:
    keys: set[str] = set()
    for name in discover_dirs("80_*"):
        if "direct_" in name or name.endswith("_split") or "smoke" in name:
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        if parts[1] == "liposome":
            dataset = "liposome"
            scheme = "group_by_api" if "group_by_api" in name else "group_by_method" if "group_by_method" in name else None
        else:
            dataset = parts[1]
            scheme = "_".join(parts[2:5])
        if scheme:
            keys.add(canonical_label(dataset, scheme))
    return keys


def mechanism_timescale_keys() -> set[str]:
    keys: set[str] = set()
    for name in discover_dirs("81_*"):
        if "direct_" in name:
            continue
        parts = name.split("_")
        if parts[1] == "liposome":
            dataset = "liposome"
            scheme = "group_by_api" if "group_by_api" in name else "group_by_method"
        else:
            dataset = parts[1]
            scheme = "_".join(parts[2:5])
        keys.add(canonical_label(dataset, scheme))
    return keys


def point_family_keys() -> set[str]:
    keys: set[str] = set()
    for name in discover_dirs("85_*"):
        if name.startswith("85_liposome_group_by_api"):
            keys.add(canonical_label("liposome", "group_by_api"))
        elif name.startswith("85_liposome_group_by_method"):
            keys.add(canonical_label("liposome", "group_by_method"))
        elif name.startswith("85_plga_cross321_group_by_drug"):
            keys.add(canonical_label("cross321", "group_by_drug"))
        elif name.startswith("85_plga_cross321_group_by_polymer"):
            keys.add(canonical_label("cross321", "group_by_polymer"))
        elif name.startswith("85_plga_internal181_group_by_drug"):
            keys.add(canonical_label("internal181", "group_by_drug"))
        elif name.startswith("85_plga_internal181_group_by_polymer"):
            keys.add(canonical_label("internal181", "group_by_polymer"))
    return keys


def calibrated_family_rows() -> pd.DataFrame:
    rows = []
    for path in Path("outputs/87_release_conformal_family_export").glob("*/*/summary.csv"):
        dataset = path.parts[-3]
        scheme = path.parts[-2]
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["dataset"] = dataset
        row["scheme"] = scheme
        row["cell_key"] = canonical_label(dataset, scheme)
        rows.append(row)
    return pd.DataFrame(rows)


def build_matrix() -> pd.DataFrame:
    calibrated = calibrated_family_rows()
    gap = pd.read_csv(GAP_TABLE)
    gap["cell_key"] = [canonical_label(d, s) for d, s in zip(gap["dataset"], gap["scheme"])]
    gap_map = gap.set_index("cell_key").to_dict(orient="index")

    curve = mechanism_curve_keys()
    timescale = mechanism_timescale_keys()
    point = point_family_keys()
    calibrated_keys = set(calibrated["cell_key"])

    all_keys = sorted(calibrated_keys | curve | timescale | point)
    rows = []
    for key in all_keys:
        dataset, scheme = key.split("__")
        gap_row = gap_map.get(key, {})
        rows.append(
            {
                "cell_key": key,
                "cell_label": display_label(key),
                "dataset": dataset,
                "scheme": scheme,
                "mechanism_curve_panel": key in curve,
                "mechanism_timescale_shape_panel": key in timescale,
                "point_family_object": key in point,
                "calibrated_family_panel": key in calibrated_keys,
                "route_consistent_duration_uq": bool(gap_row.get("route_consistent_duration_uq_available", False)),
                "current_gap": gap_row.get("current_gap", "not_audited_yet"),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "mechanism_curve_panel",
        "mechanism_timescale_shape_panel",
        "point_family_object",
        "calibrated_family_panel",
        "route_consistent_duration_uq",
    ]
    labels = ["Curve panel", "Timing/shape", "Point family", "Calibrated family", "Route-consistent\nUQ duration"]
    values = df[cols].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    cmap = ListedColormap(["#EFEFEF", "#2C7FB8"])
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["cell_label"], fontsize=10)
    ax.set_title("Current readiness for uncertainty-aware duration/shape", loc="left", fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, "yes" if values[i, j] else "no", ha="center", va="center", fontsize=8.5, color="white" if values[i, j] else "#333333")

    ax.text(
        1.02,
        0.98,
        "Gap labels",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.text(
            1.02,
            0.92 - idx * 0.085,
            f"{row['cell_label']}: {row['current_gap']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
        )

    out_png = OUTDIR / "readiness_heatmap.png"
    out_pdf = OUTDIR / "readiness_heatmap.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    yes_counts = {
        "curve panels": int(df["mechanism_curve_panel"].sum()),
        "timescale/shape panels": int(df["mechanism_timescale_shape_panel"].sum()),
        "point-family objects": int(df["point_family_object"].sum()),
        "calibrated-family panels": int(df["calibrated_family_panel"].sum()),
        "route-consistent duration UQ": int(df["route_consistent_duration_uq"].sum()),
    }
    lines = [
        "=== 97 -- release stack readiness audit ===",
        "",
        "Cell counts",
    ]
    lines.extend([f"  {k}: {v}" for k, v in yes_counts.items()])
    lines.extend(
        [
            "",
            "Interpretation",
            "  1. We already have broad coverage for curve panels, timing/shape panels, point-family objects, and calibrated-family exports.",
            "  2. The final route-consistent uncertainty-aware duration/shape layer is still absent across every currently audited cell.",
            "  3. The dominant failure mode is not 'missing UQ' or 'missing timing'; it is same-cell route mismatch between calibrated family artifacts and duration/shape outputs.",
        ]
    )
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    df = build_matrix().sort_values(["dataset", "scheme"]).reset_index(drop=True)
    df.to_csv(OUTDIR / "readiness_matrix.csv", index=False)

    tally = pd.DataFrame(
        {
            "component": [
                "mechanism_curve_panel",
                "mechanism_timescale_shape_panel",
                "point_family_object",
                "calibrated_family_panel",
                "route_consistent_duration_uq",
            ],
            "n_cells": [
                int(df["mechanism_curve_panel"].sum()),
                int(df["mechanism_timescale_shape_panel"].sum()),
                int(df["point_family_object"].sum()),
                int(df["calibrated_family_panel"].sum()),
                int(df["route_consistent_duration_uq"].sum()),
            ],
        }
    )
    tally.to_csv(OUTDIR / "component_tally.csv", index=False)
    plot_heatmap(df)
    write_summary(df)


if __name__ == "__main__":
    main()
