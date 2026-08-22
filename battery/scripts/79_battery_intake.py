from __future__ import annotations

"""
Battery trajectory intake for the cross-domain observer route.

Consumes:
    A CSV table with one row per (cell, cycle) measurement. Required columns
    are configurable, but must include cell id, cycle number, and capacity.
    An optional metadata CSV may provide one row per cell.

Produces:
    outputs/79_battery_intake/intake_summary.json
    outputs/79_battery_intake/battery_curves_long.csv
    outputs/79_battery_intake/battery_cell_metadata.csv
    outputs/79_battery_intake/README.md

Expected runtime:
    Seconds for small public datasets; I/O-bound for large raw exports.

This script is deliberately an intake/audit tool, not a model. It exists so
the battery trajectory task can enter the same partial-observation benchmark
interface as drug release without mixing in the separate D:\\battery material
screening project.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_CURVE_COLUMNS = [
    "cell_id",
    "cycle_number",
    "capacity",
    "normalized_capacity",
    "resistance",
    "coulombic_efficiency",
    "temperature_C",
    "protocol_id",
]

CANONICAL_METADATA_COLUMNS = [
    "cell_id",
    "dataset",
    "chemistry",
    "protocol_id",
    "charge_rate",
    "discharge_rate",
    "temperature_C",
    "nominal_capacity",
    "source",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: dict[str, str]) -> None:
    missing = [source for source in required.values() if source and source not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _optional_series(df: pd.DataFrame, column: str | None, default: Any = np.nan) -> pd.Series:
    if column and column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def standardize_curves(
    raw: pd.DataFrame,
    *,
    cell_col: str,
    cycle_col: str,
    capacity_col: str,
    normalized_capacity_col: str | None,
    resistance_col: str | None,
    coulombic_efficiency_col: str | None,
    temperature_col: str | None,
    protocol_col: str | None,
) -> pd.DataFrame:
    _require_columns(
        raw,
        {
            "cell_col": cell_col,
            "cycle_col": cycle_col,
            "capacity_col": capacity_col,
        },
    )

    curves = pd.DataFrame(
        {
            "cell_id": raw[cell_col].astype(str),
            "cycle_number": pd.to_numeric(raw[cycle_col], errors="coerce"),
            "capacity": pd.to_numeric(raw[capacity_col], errors="coerce"),
            "_normalized_capacity_source": pd.to_numeric(
                _optional_series(raw, normalized_capacity_col), errors="coerce"
            ),
            "resistance": pd.to_numeric(_optional_series(raw, resistance_col), errors="coerce"),
            "coulombic_efficiency": pd.to_numeric(
                _optional_series(raw, coulombic_efficiency_col), errors="coerce"
            ),
            "temperature_C": pd.to_numeric(_optional_series(raw, temperature_col), errors="coerce"),
            "protocol_id": _optional_series(raw, protocol_col, "unknown").astype(str),
        }
    )
    curves = curves.sort_values(["cell_id", "cycle_number"]).reset_index(drop=True)

    if normalized_capacity_col and normalized_capacity_col in raw.columns:
        curves["normalized_capacity"] = curves["_normalized_capacity_source"]
    else:
        first_capacity = curves.groupby("cell_id")["capacity"].transform(
            lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
        )
        curves["normalized_capacity"] = curves["capacity"] / first_capacity

    curves = curves.drop(columns=["_normalized_capacity_source"])
    return curves[CANONICAL_CURVE_COLUMNS]


def standardize_metadata(
    metadata: pd.DataFrame | None,
    curves: pd.DataFrame,
    *,
    dataset: str,
    metadata_cell_col: str,
) -> pd.DataFrame:
    if metadata is not None:
        if metadata_cell_col not in metadata.columns:
            raise ValueError(f"Metadata is missing cell id column: {metadata_cell_col}")
        out = metadata.copy()
        out = out.rename(columns={metadata_cell_col: "cell_id"})
        out["cell_id"] = out["cell_id"].astype(str)
    else:
        out = curves[["cell_id", "protocol_id", "temperature_C"]].drop_duplicates("cell_id")

    out["dataset"] = dataset
    for col in CANONICAL_METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[CANONICAL_METADATA_COLUMNS]


def audit_curves(curves: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    warnings: list[str] = []

    duplicate_pairs = int(curves.duplicated(["cell_id", "cycle_number"]).sum())
    if duplicate_pairs:
        warnings.append("duplicate_cell_cycle_pairs")

    missing_capacity = int(curves["capacity"].isna().sum())
    if missing_capacity:
        warnings.append("missing_capacity_values")

    nonpositive_capacity = int((curves["capacity"] <= 0).fillna(False).sum())
    if nonpositive_capacity:
        warnings.append("nonpositive_capacity_values")

    bad_cycles = 0
    for _, group in curves.groupby("cell_id"):
        diffs = group["cycle_number"].diff().dropna()
        if (diffs <= 0).any():
            bad_cycles += 1
    if bad_cycles:
        warnings.append("non_increasing_cycle_numbers")

    norm_high = int((curves["normalized_capacity"] > 1.2).fillna(False).sum())
    norm_low = int((curves["normalized_capacity"] < 0).fillna(False).sum())
    if norm_high or norm_low:
        warnings.append("normalized_capacity_outside_expected_range")

    counts = curves.groupby("cell_id").size()
    cycle_max = curves.groupby("cell_id")["cycle_number"].max()

    return {
        "n_entities": int(curves["cell_id"].nunique()),
        "n_measurements": int(len(curves)),
        "n_metadata_rows": int(len(metadata)),
        "measurements_per_cell": {
            "min": int(counts.min()) if len(counts) else 0,
            "median": float(counts.median()) if len(counts) else 0.0,
            "max": int(counts.max()) if len(counts) else 0,
        },
        "cycle_number_range": {
            "min": float(curves["cycle_number"].min()) if len(curves) else None,
            "max": float(curves["cycle_number"].max()) if len(curves) else None,
            "median_cell_max": float(cycle_max.median()) if len(cycle_max) else None,
        },
        "missing_capacity": missing_capacity,
        "duplicate_cell_cycle_pairs": duplicate_pairs,
        "cells_with_non_increasing_cycles": bad_cycles,
        "known_leakage_risks": warnings,
    }


def build_intake_summary(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    dataset: str,
    source_curves: str,
    source_metadata: str | None,
    normalized_capacity_note: str,
) -> dict[str, Any]:
    audit = audit_curves(curves, metadata)
    return {
        "dataset": dataset,
        "domain": "battery_cycling",
        "source_curves": source_curves,
        "source_metadata": source_metadata,
        "time_unit": "cycle_number",
        "target_unit": f"capacity_as_provided; normalized_capacity={normalized_capacity_note}",
        "descriptor_columns": [
            col for col in metadata.columns if col not in {"cell_id", "dataset", "source"}
        ],
        "trajectory_columns": CANONICAL_CURVE_COLUMNS,
        "id_columns": ["cell_id"],
        "group_columns": [
            col
            for col in ["chemistry", "protocol_id", "temperature_C"]
            if col in metadata.columns
        ],
        "early_windows": ["to_define: e.g. first 20/50/100 cycles"],
        "future_windows": ["to_define: cycles after the early window"],
        "benchmark_tier": "Tier 1 intake; promote after split/evaluation protocol is defined",
        "recommended_role": "battery trajectory intake for cross-domain smoke benchmark",
        **audit,
    }


def write_outputs(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "battery_curves_long.csv", index=False)
    metadata.to_csv(output_dir / "battery_cell_metadata.csv", index=False)
    (output_dir / "intake_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Battery intake output\n\n"
        "This directory was generated by `battery/scripts/79_battery_intake.py`.\n\n"
        "- `battery_curves_long.csv`: canonical long-form cycle trajectory table.\n"
        "- `battery_cell_metadata.csv`: canonical one-row-per-cell metadata table.\n"
        "- `intake_summary.json`: audit summary and benchmark-readiness metadata.\n",
        encoding="utf-8",
    )


def write_templates(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_CURVE_COLUMNS).to_csv(
        output_dir / "battery_curves_template.csv", index=False
    )
    pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS).to_csv(
        output_dir / "battery_metadata_template.csv", index=False
    )
    template_summary = {
        "dataset": "template",
        "domain": "battery_cycling",
        "required_next_step": "Fill templates or run with --curves to create an intake audit.",
        "curve_columns": CANONICAL_CURVE_COLUMNS,
        "metadata_columns": CANONICAL_METADATA_COLUMNS,
    }
    (output_dir / "intake_template_summary.json").write_text(
        json.dumps(template_summary, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize battery cycle trajectories.")
    parser.add_argument("--curves", type=Path, help="Raw cycle trajectory CSV.")
    parser.add_argument("--metadata", type=Path, help="Optional one-row-per-cell metadata CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/79_battery_intake"))
    parser.add_argument("--dataset", default="battery_dataset")
    parser.add_argument("--write-template", action="store_true")

    parser.add_argument("--cell-col", default="cell_id")
    parser.add_argument("--cycle-col", default="cycle_number")
    parser.add_argument("--capacity-col", default="capacity")
    parser.add_argument("--normalized-capacity-col")
    parser.add_argument("--resistance-col")
    parser.add_argument("--coulombic-efficiency-col")
    parser.add_argument("--temperature-col", default="temperature_C")
    parser.add_argument("--protocol-col", default="protocol_id")
    parser.add_argument("--metadata-cell-col", default="cell_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_template:
        write_templates(args.output_dir)
        print(f"Wrote battery intake templates to {args.output_dir}")
        return

    if args.curves is None:
        raise SystemExit("Provide --curves or use --write-template.")

    raw_curves = _read_csv(args.curves)
    raw_metadata = _read_csv(args.metadata) if args.metadata else None

    curves = standardize_curves(
        raw_curves,
        cell_col=args.cell_col,
        cycle_col=args.cycle_col,
        capacity_col=args.capacity_col,
        normalized_capacity_col=args.normalized_capacity_col,
        resistance_col=args.resistance_col,
        coulombic_efficiency_col=args.coulombic_efficiency_col,
        temperature_col=args.temperature_col,
        protocol_col=args.protocol_col,
    )
    metadata = standardize_metadata(
        raw_metadata,
        curves,
        dataset=args.dataset,
        metadata_cell_col=args.metadata_cell_col,
    )
    summary = build_intake_summary(
        curves,
        metadata,
        dataset=args.dataset,
        source_curves=str(args.curves),
        source_metadata=str(args.metadata) if args.metadata else None,
        normalized_capacity_note=(
            f"provided column {args.normalized_capacity_col}"
            if args.normalized_capacity_col
            else "capacity/first_valid_capacity"
        ),
    )
    write_outputs(curves, metadata, summary, args.output_dir)

    print(
        f"Wrote {summary['n_entities']} cells and {summary['n_measurements']} "
        f"measurements to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
