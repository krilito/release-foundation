from __future__ import annotations

"""
129_preprocess_bannigan_lai_181_raw.py

Consume:
- data/external/bannigan_lai_181/Dataset_14_feat.tsv
- data/external/bannigan_lai_181/Dataset_17_feat.tsv
- data/curves_long.csv

Produce:
- outputs/129_bannigan_lai_181_raw/curves_long.csv
- outputs/129_bannigan_lai_181_raw/formulations.csv
- outputs/129_bannigan_lai_181_raw/dataset_summary.csv
- outputs/129_bannigan_lai_181_raw/comparison_to_internal181.csv
- outputs/129_bannigan_lai_181_raw/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="repository root",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "129_bannigan_lai_181_raw",
        help="output directory",
    )
    return parser.parse_args()


def load_excel_renamed_tsv(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Sheet1")


def build_curves(df17: pd.DataFrame) -> pd.DataFrame:
    curves = df17.rename(
        columns={
            "Experimental_index": "source_curve_id",
            "Time": "time_days",
            "Release": "release_fraction",
        }
    ).copy()
    curves["source_dataset"] = "bannigan_lai_181_raw"
    curves["source_curve_id"] = curves["source_curve_id"].astype(str)
    curves["unified_curve_id"] = curves["source_dataset"] + ":" + curves["source_curve_id"]
    curves["record_id"] = (
        curves["source_dataset"]
        + ":"
        + curves["source_curve_id"]
        + ":"
        + curves["time_days"].round(8).astype(str)
    )
    curves["time_raw"] = curves["time_days"]
    curves["time_unit"] = "day"
    curves["release_raw"] = curves["release_fraction"]
    curves["release_unit"] = "fraction"
    curves["release_percent"] = curves["release_fraction"] * 100.0
    return curves[
        [
            "record_id",
            "unified_curve_id",
            "source_dataset",
            "source_curve_id",
            "time_raw",
            "time_unit",
            "time_days",
            "release_raw",
            "release_unit",
            "release_fraction",
            "release_percent",
            "T=0.25",
            "T=0.5",
            "T=1.0",
        ]
    ].copy()


def build_formulations(df17: pd.DataFrame) -> pd.DataFrame:
    meta = df17.drop_duplicates("Experimental_index").rename(
        columns={
            "Experimental_index": "source_curve_id",
            "DP_Group": "source_group",
        }
    ).copy()
    meta["source_dataset"] = "bannigan_lai_181_raw"
    meta["source_curve_id"] = meta["source_curve_id"].astype(str)
    meta["unified_curve_id"] = meta["source_dataset"] + ":" + meta["source_curve_id"]
    meta["source_has_explicit_group"] = True
    meta["polymer_family"] = "PLGA-like"
    meta["Particle_Size"] = pd.NA
    meta["EE"] = pd.NA
    meta["DLC_percent"] = meta["DLC"] * 100.0
    meta["Polymer_MW_raw_unit"] = "Da"
    return meta[
        [
            "unified_curve_id",
            "source_dataset",
            "source_curve_id",
            "source_group",
            "source_has_explicit_group",
            "polymer_family",
            "LA/GA",
            "Polymer_MW",
            "Polymer_MW_raw_unit",
            "CL Ratio",
            "Drug_Tm",
            "Drug_Pka",
            "Initial D/M ratio",
            "DLC",
            "DLC_percent",
            "EE",
            "Particle_Size",
            "SA-V",
            "SE",
            "Drug_Mw",
            "Drug_TPSA",
            "Drug_NHA",
            "Drug_LogP",
        ]
    ].copy()


def build_dataset_summary(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    per_curve = curves.groupby("unified_curve_id", as_index=False).agg(
        n_points=("record_id", "count"),
        duration_days=("time_days", lambda x: float(x.max() - x.min())),
        final_release=("release_fraction", "last"),
    )
    return pd.DataFrame(
        [
            {
                "source_dataset": "bannigan_lai_181_raw",
                "n_curves": int(curves["unified_curve_id"].nunique()),
                "n_points_total": int(len(curves)),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(per_curve["n_points"].median()),
                "median_duration_days": float(per_curve["duration_days"].median()),
                "median_final_release": float(per_curve["final_release"].median()),
            }
        ]
    )


def compare_to_internal181(repo_root: Path, raw_curves: pd.DataFrame) -> pd.DataFrame:
    internal = pd.read_csv(repo_root / "data" / "curves_long.csv").rename(columns={"curve_id": "source_curve_id"}).copy()
    internal["source_curve_id"] = internal["source_curve_id"].astype(str)
    raw_curve_ids = set(raw_curves["source_curve_id"].unique())
    internal_curve_ids = set(internal["source_curve_id"].unique())
    overlap_ids = raw_curve_ids & internal_curve_ids
    return pd.DataFrame(
        [
            {"metric": "raw_curve_count", "value": len(raw_curve_ids)},
            {"metric": "internal_curve_count", "value": len(internal_curve_ids)},
            {"metric": "overlap_curve_count", "value": len(overlap_ids)},
            {"metric": "raw_point_count", "value": int(len(raw_curves))},
            {"metric": "internal_point_count", "value": int(len(internal))},
            {
                "metric": "overlap_point_count_if_id_only",
                "value": int(internal[internal["source_curve_id"].isin(overlap_ids)].shape[0]),
            },
        ]
    )


def write_summary(
    dataset_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    out_path: Path,
) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# bannigan_lai_181_raw",
        "",
        "Standardized preprocessing output for the raw Bannigan/Lai 181-source spreadsheet snapshot.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.3f}`",
        "",
        "## Comparison to current internal181",
        "",
    ]
    for _, row in comparison.iterrows():
        lines.append(f"- `{row['metric']}`: `{int(row['value'])}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = args.repo_root / "data" / "external" / "bannigan_lai_181"
    df14 = load_excel_renamed_tsv(root / "Dataset_14_feat.tsv")
    df17 = load_excel_renamed_tsv(root / "Dataset_17_feat.tsv")
    curves = build_curves(df17)
    formulations = build_formulations(df17)
    dataset_summary = build_dataset_summary(curves, formulations)
    comparison = compare_to_internal181(args.repo_root, curves)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    comparison.to_csv(args.outdir / "comparison_to_internal181.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "bannigan_lai_181_raw",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "source_dataset": "bannigan_lai_181_raw",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, comparison, args.outdir / "summary.md")

    print(f"[bannigan-lai-181-raw] wrote outputs to {args.outdir}")
    print(
        f"[bannigan-lai-181-raw] curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
