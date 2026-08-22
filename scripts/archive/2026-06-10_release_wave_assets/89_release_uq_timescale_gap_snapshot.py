"""
89 - Make the current uncertainty-to-timescale gap explicit.

Purpose:
    We now have:
      1. a calibrated-family panel in curve space
      2. a point-estimate timescale / shape panel

    What we still do NOT have is a route-consistent, uncertainty-aware
    duration / shape layer. This script makes that gap explicit by aligning the
    current uncertainty and timescale evidence per benchmark cell.

Produces:
    outputs/89_release_uq_timescale_gap_snapshot/gap_table.csv
    outputs/89_release_uq_timescale_gap_snapshot/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTDIR = Path("outputs/89_release_uq_timescale_gap_snapshot")
CAL_FAMILY = Path("outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv")
PLGA_TIME = Path("outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv")
PLGA_SHAPE = Path("outputs/82_release_capability_snapshot/plga_shape_snapshot.csv")
LIPO_TIME = Path("outputs/82_release_capability_snapshot/liposome_bridge_snapshot.csv")
LIPO_SHAPE = Path("outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv")


def _parse_plga_cell(label: str) -> tuple[str, str]:
    dataset, scheme, *_ = label.split("_")
    if scheme not in {"group", "random"}:
        raise ValueError(label)
    if scheme == "random":
        return dataset, "random_5fold"
    # label looks like dataset_group_by_drug_formulation_plus_early
    parts = label.split("_")
    return parts[0], "_".join(parts[1:4])


def build_gap_table() -> pd.DataFrame:
    cal = pd.read_csv(CAL_FAMILY).copy()
    cal["uq_family_route"] = "fib_split_conformal_global_family"

    rows: list[dict[str, object]] = []

    plga_t = pd.read_csv(PLGA_TIME)
    plga_s = pd.read_csv(PLGA_SHAPE)
    shape_map = {
        (row["benchmark_cell"].split("_", 1)[0], "_".join(row["benchmark_cell"].split("_")[1:4])): row
        for _, row in plga_s.iterrows()
    }
    for _, row in plga_t.iterrows():
        dataset, scheme = _parse_plga_cell(row["benchmark_cell"])
        shape = shape_map[(dataset, scheme)]
        rows.append(
            {
                "dataset": dataset,
                "scheme": scheme,
                "timescale_route_family": "point_estimate_mechanism_route",
                "timescale_route_name": row["mechanism_route"],
                "future_t10_mae": float(row["future_t10_mae_mechanism_d"]),
                "future_t50_mae": float(row["future_t50_mae_mechanism_d"]),
                "future_t80_mae": float(row["future_t80_mae_mechanism_d"]),
                "burst_mae": float(shape["burst_mae_mechanism"]),
                "post_window_mae": float(shape["post_window_release_mae_mechanism"]),
                "residual_tail_mae": float(shape["residual_tail_mae_mechanism"]),
                "tail_auc_mae": float(shape["tail_auc_mae_mechanism"]),
            }
        )

    lipo_t = pd.read_csv(LIPO_TIME)
    lipo_s = pd.read_csv(LIPO_SHAPE)
    lipo_shape_map = {row["scheme"]: row for _, row in lipo_s.iterrows()}
    lipo_scheme_map = {
        "group_by_API": "group_by_drug",
        "group_by_release_method": "group_by_polymer",
    }
    for _, row in lipo_t.iterrows():
        scheme = lipo_scheme_map[row["scheme"]]
        shape = lipo_shape_map[row["scheme"]]
        rows.append(
            {
                "dataset": "liposome",
                "scheme": scheme,
                "timescale_route_family": "point_estimate_mechanism_route",
                "timescale_route_name": row["mechanism_route"],
                "future_t10_mae": float(row["t10_mae_mechanism_h"]),
                "future_t50_mae": float(row["t50_mae_mechanism_h"]),
                "future_t80_mae": float(row["t80_mae_mechanism_h"]),
                "burst_mae": float(shape["burst_mae_mechanism"]),
                "post_window_mae": float(shape["post_window_release_mae_mechanism"]),
                "residual_tail_mae": float(shape["residual_tail_mae_mechanism"]),
                "tail_auc_mae": float(shape["tail_auc_mae_mechanism"]),
            }
        )

    ts = pd.DataFrame(rows)
    merged = cal.merge(ts, on=["dataset", "scheme"], how="outer", indicator=True)
    merged["route_consistent_duration_uq_available"] = False
    merged["current_gap"] = merged["_merge"].map(
        {
            "both": "same_cell_but_route_mismatch",
            "left_only": "uq_without_timescale_panel",
            "right_only": "timescale_without_calibrated_family",
        }
    )
    return merged.drop(columns=["_merge"]).sort_values(["dataset", "scheme"]).reset_index(drop=True)


def write_summary(df: pd.DataFrame) -> None:
    both = df[df["current_gap"] == "same_cell_but_route_mismatch"]
    uq_only = df[df["current_gap"] == "uq_without_timescale_panel"]
    ts_only = df[df["current_gap"] == "timescale_without_calibrated_family"]

    lines = [
        "=== 89 -- release UQ/timescale gap snapshot ===",
        "",
        f"cells with calibrated family + timescale evidence : {len(both)}",
        f"cells with calibrated family only                 : {len(uq_only)}",
        f"cells with timescale only                         : {len(ts_only)}",
        "",
        "Gap verdict",
        "  1. We now have the current full PLGA + liposome calibrated-family panel in curve space.",
        "  2. We also have the current PLGA + liposome mechanism-route timescale/shape panel.",
        "  3. But these are not yet the same route family, so uncertainty-aware duration is still missing.",
        "",
        "Current same-cell-but-route-mismatch panel",
    ]
    for row in both.itertuples(index=False):
        lines.extend(
            [
                f"  {row.dataset} / {row.scheme}:",
                f"    uq route          : {row.uq_family_route}",
                f"    timescale route   : {row.timescale_route_name}",
                f"    cov90_global_mean : {row.cov90_global_mean:.3f}",
                f"    width90_global    : {row.width90_global_median:.3f}",
                f"    future t10/t50/t80: {row.future_t10_mae:.3f} / {row.future_t50_mae:.3f} / {row.future_t80_mae:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "Bottom line",
            "  The next hard systems step is not another curve-space UQ table.",
            "  It is route-consistent uncertainty-aware duration / shape reporting.",
        ]
    )
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gap = build_gap_table()
    gap.to_csv(OUTDIR / "gap_table.csv", index=False)
    write_summary(gap)


if __name__ == "__main__":
    main()
