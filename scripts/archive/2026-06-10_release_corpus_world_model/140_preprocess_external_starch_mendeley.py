from __future__ import annotations

"""
140_preprocess_external_starch_mendeley.py

Consume:
- data/external/wtnjj6smjd_mendeley/Roser Posada Data.xlsx

Produce:
- outputs/140_external_starch_mendeley/curves_long.csv
- outputs/140_external_starch_mendeley/formulations.csv
- outputs/140_external_starch_mendeley/curve_quality_summary.csv
- outputs/140_external_starch_mendeley/dataset_summary.csv
- outputs/140_external_starch_mendeley/manifest.json
- outputs/140_external_starch_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_GROUPS = [
    {"system_label": "Cur-SNP", "group_suffix": "", "time_col": 1, "analytes": [{"name": "curcumin", "value_col": 2, "sd_col": 3}]},
    {"system_label": "Cur-ASNP", "group_suffix": "", "time_col": 5, "analytes": [{"name": "curcumin", "value_col": 6, "sd_col": 7}]},
    {"system_label": "Pip-SNP", "group_suffix": "", "time_col": 9, "analytes": [{"name": "piperine", "value_col": 10, "sd_col": 11}]},
    {"system_label": "Pip-ASNP", "group_suffix": "", "time_col": 13, "analytes": [{"name": "piperine", "value_col": 14, "sd_col": 15}]},
    {
        "system_label": "Cur/Pip-ASNP",
        "group_suffix": "_block_a",
        "time_col": 17,
        "analytes": [
            {"name": "curcumin", "value_col": 18, "sd_col": 19},
            {"name": "piperine", "value_col": 20, "sd_col": 21},
        ],
    },
    {
        "system_label": "Cur/Pip-ASNP",
        "group_suffix": "_block_b",
        "time_col": 23,
        "analytes": [
            {"name": "curcumin", "value_col": 24, "sd_col": 25},
            {"name": "piperine", "value_col": 26, "sd_col": 27},
        ],
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="repository root")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "140_external_starch_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )


def detect_phase_blocks(df: pd.DataFrame) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current_start = None
    current_label = None
    for row_idx in range(len(df)):
        cell0 = str(df.iat[row_idx, 0]) if pd.notna(df.iat[row_idx, 0]) else ""
        if cell0.startswith("Release in "):
            current_start = row_idx
            current_label = cell0
        elif current_start is not None and row_idx > current_start:
            next_cell0 = str(df.iat[row_idx, 0]) if pd.notna(df.iat[row_idx, 0]) else ""
            if next_cell0.startswith("Release in "):
                blocks.append({"label": current_label, "start_row": current_start, "end_row": row_idx - 1})
                current_start = row_idx
                current_label = next_cell0
    if current_start is not None:
        blocks.append({"label": current_label, "start_row": current_start, "end_row": len(df) - 1})
    return blocks


def parse_phase_label(label: str) -> tuple[str, str]:
    if "gastric" in label.lower():
        return "gastric_simulated_media", "SGF"
    if "intestinal" in label.lower():
        return "intestinal_simulated_media", "SIF"
    return slugify(label), "UNK"


def load_release_workbook(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name="Hoja1", header=None)
    phase_blocks = detect_phase_blocks(df)
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    emitted: set[str] = set()

    for group in SYSTEM_GROUPS:
        time_offset_min = 0.0
        for analyte in group["analytes"]:
            curve_key = f"{slugify(group['system_label'] + group['group_suffix'])}_{slugify(analyte['name'])}"
            unified_curve_id = f"starch_mendeley_wtnjj6smjd:{curve_key}"
            all_points: list[dict[str, object]] = []
            time_offset_min = 0.0
            for block in phase_blocks:
                phase_name, phase_short = parse_phase_label(str(block["label"]))
                start = int(block["start_row"])
                end = int(block["end_row"])
                phase_rows: list[dict[str, object]] = []
                for row_idx in range(start, end + 1):
                    time_value = pd.to_numeric(df.iat[row_idx, group["time_col"]], errors="coerce")
                    release_value = pd.to_numeric(df.iat[row_idx, analyte["value_col"]], errors="coerce")
                    release_sd = pd.to_numeric(df.iat[row_idx, analyte["sd_col"]], errors="coerce")
                    if pd.isna(time_value) or pd.isna(release_value):
                        continue
                    phase_rows.append(
                        {
                            "phase_name": phase_name,
                            "phase_short": phase_short,
                            "time_min_phase": float(time_value),
                            "release_percent": float(release_value),
                            "release_sd_percent": float(release_sd) if pd.notna(release_sd) else np.nan,
                        }
                    )
                if not phase_rows:
                    continue
                phase_df = pd.DataFrame(phase_rows).sort_values("time_min_phase").reset_index(drop=True)
                for _, row in phase_df.iterrows():
                    all_points.append(
                        {
                            "phase_name": row["phase_name"],
                            "phase_short": row["phase_short"],
                            "time_min_phase": float(row["time_min_phase"]),
                            "time_min_global": float(time_offset_min + row["time_min_phase"]),
                            "release_percent": float(row["release_percent"]),
                            "release_sd_percent": float(row["release_sd_percent"])
                            if pd.notna(row["release_sd_percent"])
                            else np.nan,
                        }
                    )
                time_offset_min += float(phase_df["time_min_phase"].max())

            if not all_points:
                continue

            curve_df = pd.DataFrame(all_points).sort_values("time_min_global").reset_index(drop=True)
            for _, row in curve_df.iterrows():
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{row['time_min_global']:.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "starch_mendeley_wtnjj6smjd",
                        "source_curve_id": curve_key,
                        "time_raw": float(row["time_min_global"]),
                        "time_unit": "minute",
                        "time_days": float(row["time_min_global"]) / (24.0 * 60.0),
                        "release_raw": float(row["release_percent"]),
                        "release_unit": "percent_released",
                        "release_fraction": float(row["release_percent"]) / 100.0,
                        "release_percent": float(row["release_percent"]),
                        "curve_series": row["phase_name"],
                        "curve_level": "system_analyte_mean",
                        "source_group": group["system_label"] + group["group_suffix"],
                        "payload_name": analyte["name"],
                        "phase_short": row["phase_short"],
                        "time_min_phase": float(row["time_min_phase"]),
                        "release_sd_percent": float(row["release_sd_percent"])
                        if pd.notna(row["release_sd_percent"])
                        else pd.NA,
                    }
                )

            if unified_curve_id in emitted:
                continue
            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "starch_mendeley_wtnjj6smjd",
                    "source_curve_id": curve_key,
                    "source_group": group["system_label"] + group["group_suffix"],
                    "source_has_explicit_group": True,
                    "polymer_family": "starch nanoparticle",
                    "payload_name": analyte["name"],
                    "experimental_panel": "SGF_to_SIF_release",
                    "curve_level": "system_analyte_mean",
                    "normalization_basis": "reported_percent_released",
                    "release_measure_type": "cumulative_percent_released",
                    "phase_sequence": "gastric_simulated_media -> intestinal_simulated_media",
                    "source_label_duplicated": bool(group["group_suffix"]),
                    "duplicate_label_resolution": (
                        "added workbook-column suffix to repeated Cur/Pip-ASNP label"
                        if group["group_suffix"]
                        else pd.NA
                    ),
                    "Polymer_MW": pd.NA,
                    "Polymer_MW_raw_unit": pd.NA,
                    "LA/GA": pd.NA,
                    "CL Ratio": pd.NA,
                    "Drug_Tm": pd.NA,
                    "Drug_Pka": pd.NA,
                    "Initial D/M ratio": pd.NA,
                    "DLC": pd.NA,
                    "DLC_percent": pd.NA,
                    "EE": pd.NA,
                    "Particle_Size": pd.NA,
                    "SA-V": pd.NA,
                    "SE": pd.NA,
                    "Drug_Mw": pd.NA,
                    "Drug_TPSA": pd.NA,
                    "Drug_NHA": pd.NA,
                    "Drug_LogP": pd.NA,
                }
            )
            emitted.add(unified_curve_id)

    return pd.DataFrame(curve_rows), pd.DataFrame(formulation_rows)


def build_curve_quality_summary(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    form_index = formulations.set_index("unified_curve_id")
    for curve_id, group in curves.groupby("unified_curve_id", sort=False):
        group = group.sort_values("time_days")
        release = group["release_fraction"].to_numpy(float)
        meta = form_index.loc[curve_id]
        rows.append(
            {
                "unified_curve_id": curve_id,
                "source_dataset": group["source_dataset"].iloc[0],
                "source_curve_id": group["source_curve_id"].iloc[0],
                "curve_series": meta["experimental_panel"],
                "curve_level": group["curve_level"].iloc[0],
                "source_group": group["source_group"].iloc[0],
                "payload_name": group["payload_name"].iloc[0],
                "n_points": int(len(group)),
                "time_min_days": float(group["time_days"].min()),
                "time_max_days": float(group["time_days"].max()),
                "duration_days": float(group["time_days"].max() - group["time_days"].min()),
                "final_release_fraction": float(group["release_fraction"].iloc[-1]),
                "max_release_fraction": float(group["release_fraction"].max()),
                "monotonicity_violations": monotonicity_violation_count(release),
                "is_monotone_non_decreasing": monotonicity_violation_count(release) == 0,
                "has_release_gt_1p0": bool((group["release_fraction"] > 1.0 + 1e-6).any()),
                "phase_count": int(group["phase_short"].nunique()),
                "release_measure_type": meta["release_measure_type"],
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "starch_mendeley_wtnjj6smjd",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "fraction_release_gt_1p0": float(curve_quality["has_release_gt_1p0"].mean()),
                "n_duplicate_label_resolutions": int(formulations["source_label_duplicated"].sum()),
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, formulations: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_starch_mendeley",
        "",
        "Standardized cumulative-release corpus from Mendeley dataset `wtnjj6smjd`, built from the",
        "single workbook `Roser Posada Data.xlsx`.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Important Note",
        "",
        "- The workbook reports two sequential media phases: gastric simulated media followed by intestinal simulated media.",
        "- This preprocessing stitches the second phase onto the end of the first using a cumulative time offset.",
        "- The workbook contains a repeated `Cur/Pip-ASNP` label in two distinct column groups; both are preserved with `_block_a` / `_block_b` suffixes.",
        "",
        "## Exported Curves",
        "",
    ]
    for _, row in formulations.sort_values(["source_group", "payload_name"]).iterrows():
        lines.append(f"- `{row['source_group']}` / `{row['payload_name']}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = args.repo_root / "data" / "external" / "wtnjj6smjd_mendeley" / "Roser Posada Data.xlsx"
    curves, formulations = load_release_workbook(xlsx_path)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_starch_mendeley",
                "source_dataset": "starch_mendeley_wtnjj6smjd",
                "mendeley_dataset_id": "wtnjj6smjd",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "release_measure_type": "cumulative_percent_released",
                "eligible_for_main_cumulative_pool": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, formulations, args.outdir / "summary.md")

    print(f"[external-starch-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-starch-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
