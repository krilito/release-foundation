from __future__ import annotations

"""
144_preprocess_external_caseinate_gallic_mendeley.py

Consume:
- data/external/8d3973kgb3_mendeley/Antioxidant capacity and release kinetics of active film based on sodium caseinate and gallic acid/Data repository.xlsx

Produce:
- outputs/144_external_caseinate_gallic_mendeley/curves_long.csv
- outputs/144_external_caseinate_gallic_mendeley/formulations.csv
- outputs/144_external_caseinate_gallic_mendeley/curve_quality_summary.csv
- outputs/144_external_caseinate_gallic_mendeley/dataset_summary.csv
- outputs/144_external_caseinate_gallic_mendeley/manifest.json
- outputs/144_external_caseinate_gallic_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="repository root")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "144_external_caseinate_gallic_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def load_release_workbook(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    release_df = pd.read_excel(xlsx_path, sheet_name="Release kinetics")
    model_df = pd.read_excel(xlsx_path, sheet_name="Mathematical modelling")

    raw_rows: list[dict[str, object]] = []
    current_sample = None
    for _, row in model_df.iterrows():
        sample = row.get("Sample film ")
        if pd.notna(sample):
            current_sample = str(sample)
        if current_sample is None:
            continue
        replication = pd.to_numeric(row.get("Replication"), errors="coerce")
        time_h = pd.to_numeric(row.get("Time (h)"), errors="coerce")
        exp_release = pd.to_numeric(row.get("Fick's model (Experimental release values)"), errors="coerce")
        if pd.isna(replication) or pd.isna(time_h) or pd.isna(exp_release):
            continue

        conc_match = release_df[
            (release_df["Replication"] == replication)
            & (release_df["Time (h)"] == time_h)
        ]
        concentration = pd.to_numeric(conc_match["Concentration (ug/ml)"].iloc[0], errors="coerce") if len(conc_match) else np.nan

        raw_rows.append(
            {
                "sample_film": current_sample,
                "replication": int(replication),
                "time_h": float(time_h),
                "release_fraction": float(exp_release),
                "concentration_ug_ml": float(concentration) if pd.notna(concentration) else np.nan,
            }
        )

    long_df = pd.DataFrame(raw_rows).sort_values(["sample_film", "replication", "time_h"]).reset_index(drop=True)

    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    emitted: set[str] = set()

    for (sample_film, replication), group in long_df.groupby(["sample_film", "replication"], sort=False):
        curve_key = f"{sample_film.lower().replace(' ', '_').replace('-', '_')}_rep{replication}"
        unified_curve_id = f"caseinate_gallic_mendeley_8d3973kgb3:{curve_key}"
        group = group.sort_values("time_h")
        for _, row in group.iterrows():
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{row['time_h']:.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "caseinate_gallic_mendeley_8d3973kgb3",
                    "source_curve_id": curve_key,
                    "time_raw": float(row["time_h"]),
                    "time_unit": "hour",
                    "time_days": float(row["time_h"]) / 24.0,
                    "release_raw": float(row["release_fraction"]),
                    "release_unit": "fraction_released",
                    "release_fraction": float(row["release_fraction"]),
                    "release_percent": float(row["release_fraction"]) * 100.0,
                    "curve_series": "Fick_experimental_Mt_over_Minf",
                    "curve_level": "replicate",
                    "source_group": sample_film,
                    "payload_name": "gallic acid",
                    "replication": int(replication),
                    "concentration_ug_ml": float(row["concentration_ug_ml"]) if pd.notna(row["concentration_ug_ml"]) else pd.NA,
                }
            )

        if unified_curve_id not in emitted:
            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "caseinate_gallic_mendeley_8d3973kgb3",
                    "source_curve_id": curve_key,
                    "source_group": sample_film,
                    "source_has_explicit_group": True,
                    "polymer_family": "sodium caseinate film",
                    "payload_name": "gallic acid",
                    "experimental_panel": "release_kinetics",
                    "curve_level": "replicate",
                    "normalization_basis": "Mt_over_Minf_experimental_values",
                    "release_measure_type": "cumulative_fraction_released",
                    "measurement_assay": "ficks_model_table",
                    "replication": int(replication),
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
                "curve_series": group["curve_series"].iloc[0],
                "curve_level": group["curve_level"].iloc[0],
                "source_group": group["source_group"].iloc[0],
                "payload_name": group["payload_name"].iloc[0],
                "measurement_assay": meta["measurement_assay"],
                "n_points": int(len(group)),
                "time_min_days": float(group["time_days"].min()),
                "time_max_days": float(group["time_days"].max()),
                "duration_days": float(group["time_days"].max() - group["time_days"].min()),
                "final_release_fraction": float(group["release_fraction"].iloc[-1]),
                "max_release_fraction": float(group["release_fraction"].max()),
                "monotonicity_violations": monotonicity_violation_count(release),
                "is_monotone_non_decreasing": monotonicity_violation_count(release) == 0,
                "has_release_gt_1p0": bool((group["release_fraction"] > 1.0 + 1e-6).any()),
                "release_measure_type": meta["release_measure_type"],
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "caseinate_gallic_mendeley_8d3973kgb3",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "fraction_release_gt_1p0": float(curve_quality["has_release_gt_1p0"].mean()),
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, formulations: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_caseinate_gallic_mendeley",
        "",
        "Standardized cumulative-release corpus from Mendeley dataset `8d3973kgb3`, using",
        "the `Fick's model (Experimental release values)` column as the normalized `Mt/Minf` release target.",
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
        "- The workbook exposes raw concentration trajectories and a modeling sheet with experimental normalized `Mt/Minf` values.",
        "- This preprocessing uses the normalized experimental values as the release target and keeps the concentration values as auxiliary metadata in `curves_long.csv`.",
        "",
        "## Exported Curves",
        "",
    ]
    for _, row in formulations.sort_values(["source_group", "replication"]).iterrows():
        lines.append(f"- `{row['source_group']}` / `replicate {int(row['replication'])}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = (
        args.repo_root
        / "data"
        / "external"
        / "8d3973kgb3_mendeley"
        / "Antioxidant capacity and release kinetics of active film based on sodium caseinate and gallic acid"
        / "Data repository.xlsx"
    )
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
                "corpus_name": "external_caseinate_gallic_mendeley",
                "source_dataset": "caseinate_gallic_mendeley_8d3973kgb3",
                "mendeley_dataset_id": "8d3973kgb3",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "release_measure_type": "cumulative_fraction_released",
                "eligible_for_main_cumulative_pool": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, formulations, args.outdir / "summary.md")

    print(f"[external-caseinate-gallic-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-caseinate-gallic-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
