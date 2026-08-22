from __future__ import annotations

"""
NASA PCoE Battery Data Set adapter for the battery trajectory intake.

Consumes:
    data/external/nasa_pcoe_battery_5/expanded_nested/**/*.mat

Produces:
    outputs/79a_nasa_pcoe_battery5_adapter/nasa_pcoe_battery5_cycle_summary.csv
    outputs/79a_nasa_pcoe_battery5_adapter/nasa_pcoe_battery5_metadata.csv
    outputs/79a_nasa_pcoe_battery5_adapter/adapter_summary.json

Expected runtime:
    Seconds after the NASA zip has been downloaded and expanded locally.

This is a source adapter only. It extracts one row per battery discharge cycle
so `battery/scripts/79_battery_intake.py` can run the standard intake/audit.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io


DEFAULT_SOURCE_ROOT = Path("data/external/nasa_pcoe_battery_5/expanded_nested")
DEFAULT_OUTPUT_DIR = Path("outputs/79a_nasa_pcoe_battery5_adapter")
NOMINAL_CAPACITY_AH = 2.0


def _mat_cycles(path: Path) -> tuple[str, np.ndarray]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = path.stem
    if key not in mat:
        raise KeyError(f"Expected key {key!r} in {path}")
    obj = mat[key]
    cycles = obj.cycle
    if not isinstance(cycles, np.ndarray):
        cycles = np.asarray([cycles])
    return key, cycles


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _count_discharges(path: Path) -> int:
    _, cycles = _mat_cycles(path)
    return sum(1 for cycle in cycles if str(cycle.type).lower() == "discharge")


def choose_unique_cell_files(paths: list[Path]) -> tuple[list[Path], dict[str, list[str]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)

    selected: list[Path] = []
    duplicates: dict[str, list[str]] = {}
    for cell_id, cell_paths in sorted(grouped.items()):
        if len(cell_paths) == 1:
            selected.append(cell_paths[0])
            continue

        scored = []
        for path in cell_paths:
            try:
                scored.append((_count_discharges(path), path.stat().st_size, path))
            except Exception:
                scored.append((-1, path.stat().st_size, path))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected.append(scored[0][2])
        duplicates[cell_id] = [str(path) for _, _, path in scored]
    return selected, duplicates


def extract_discharge_rows(path: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell_id, cycles = _mat_cycles(path)
    rows: list[dict[str, Any]] = []
    ambient_values: list[float] = []
    invalid_capacity_rows = 0

    discharge_index = 0
    for raw_index, cycle in enumerate(cycles):
        if str(cycle.type).lower() != "discharge":
            continue
        ambient = _safe_float(getattr(cycle, "ambient_temperature", np.nan))
        ambient_values.append(ambient)
        data = cycle.data
        capacity = _safe_float(getattr(data, "Capacity", np.nan))
        if not np.isfinite(capacity) or capacity <= 0.0:
            invalid_capacity_rows += 1
            continue
        discharge_index += 1
        rows.append(
            {
                "cell_id": cell_id,
                "cycle_number": discharge_index,
                "raw_cycle_index": raw_index,
                "capacity": capacity,
                "temperature_C": ambient,
                "protocol_id": path.parent.name,
                "source_file": str(path),
            }
        )

    meta = {
        "cell_id": cell_id,
        "dataset": dataset,
        "chemistry": "Li-ion",
        "protocol_id": path.parent.name,
        "charge_rate": np.nan,
        "discharge_rate": np.nan,
        "temperature_C": float(np.nanmedian(ambient_values)) if ambient_values else np.nan,
        "nominal_capacity": NOMINAL_CAPACITY_AH,
        "source": str(path),
        "invalid_capacity_rows_removed": invalid_capacity_rows,
    }
    return rows, meta


def build_tables(source_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mat_paths = sorted(source_root.rglob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found under {source_root}")

    selected_paths, duplicate_candidates = choose_unique_cell_files(mat_paths)

    all_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    failed_files: list[str] = []
    for path in selected_paths:
        try:
            rows, meta = extract_discharge_rows(path, dataset)
        except Exception as exc:
            failed_files.append(f"{path}: {exc}")
            continue
        all_rows.extend(rows)
        metadata_rows.append(meta)

    curves = pd.DataFrame(all_rows)
    metadata = pd.DataFrame(metadata_rows)
    if curves.empty:
        raise ValueError("No discharge capacity rows were extracted.")

    # NASA's README defines EOL as 30% fade from a rated 2 Ah capacity.
    # Using the first valid discharge as the denominator is unsafe here
    # because some cells begin with anomalously low recorded Capacity values.
    curves["normalized_capacity"] = curves["capacity"] / NOMINAL_CAPACITY_AH
    curves["resistance"] = np.nan
    curves["coulombic_efficiency"] = np.nan

    summary = {
        "dataset": dataset,
        "source_root": str(source_root),
        "n_mat_files_found": len(mat_paths),
        "n_cell_files_selected": len(selected_paths),
        "n_cells_extracted": int(curves["cell_id"].nunique()),
        "n_discharge_rows": int(len(curves)),
        "duplicate_cell_file_candidates": duplicate_candidates,
        "failed_files": failed_files,
        "invalid_capacity_rows_removed": int(metadata["invalid_capacity_rows_removed"].sum())
        if "invalid_capacity_rows_removed" in metadata.columns
        else 0,
        "capacity_range": {
            "min": float(curves["capacity"].min()),
            "max": float(curves["capacity"].max()),
        },
        "cycle_number_range": {
            "min": int(curves["cycle_number"].min()),
            "max": int(curves["cycle_number"].max()),
            "median_cell_max": float(curves.groupby("cell_id")["cycle_number"].max().median()),
        },
    }
    return curves, metadata, summary


def write_outputs(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "nasa_pcoe_battery5_cycle_summary.csv", index=False)
    metadata.to_csv(output_dir / "nasa_pcoe_battery5_metadata.csv", index=False)
    (output_dir / "adapter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# NASA PCoE battery adapter output\n\n"
        "Generated by `battery/scripts/79a_nasa_pcoe_battery5_adapter.py`.\n\n"
        "- `nasa_pcoe_battery5_cycle_summary.csv`: one row per discharge cycle.\n"
        "- `nasa_pcoe_battery5_metadata.csv`: one row per selected cell.\n"
        "- `adapter_summary.json`: source-level extraction audit.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt NASA PCoE battery .mat files to intake CSV.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default="nasa_pcoe_battery_5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves, metadata, summary = build_tables(args.source_root, args.dataset)
    write_outputs(curves, metadata, summary, args.output_dir)
    print(
        f"Wrote {summary['n_cells_extracted']} cells and "
        f"{summary['n_discharge_rows']} discharge rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
