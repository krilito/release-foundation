from __future__ import annotations

"""
142_preprocess_external_nt3_hydrogel_mendeley.py

Consume:
- data/external/cs6x4f86f2_mendeley/Data on heparinised poloxamer hydrogel with hyaluronic acid can stabilise and sustain the release of bioactive NT-3/Data for manuscript.xlsx

Produce:
- outputs/142_external_nt3_hydrogel_mendeley/curves_long.csv
- outputs/142_external_nt3_hydrogel_mendeley/formulations.csv
- outputs/142_external_nt3_hydrogel_mendeley/curve_quality_summary.csv
- outputs/142_external_nt3_hydrogel_mendeley/dataset_summary.csv
- outputs/142_external_nt3_hydrogel_mendeley/manifest.json
- outputs/142_external_nt3_hydrogel_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FL_GROUPS = [
    {"label": "HA0", "mean_col": 1, "replicate_cols": [2, 3, 4, 5]},
    {"label": "HA0.5", "mean_col": 6, "replicate_cols": [7, 8, 9, 10]},
    {"label": "HA1", "mean_col": 11, "replicate_cols": [12, 13, 14, 15]},
    {"label": "HA1.5", "mean_col": 16, "replicate_cols": [17, 18, 19, 20]},
    {"label": "HA2", "mean_col": 21, "replicate_cols": [22, 23, 24, 25]},
    {"label": "HPHA1.5", "mean_col": 26, "replicate_cols": [27, 28, 29, 30]},
]

NT3_SECTIONS = [
    {
        "section_name": "elisa",
        "time_col": 14,
        "start_row": 3,
        "end_row": 14,
        "groups": [
            {"label": "NT-3", "value_cols": [15, 16, 17, 18]},
            {"label": "HPHA1.5", "value_cols": [19, 20, 21, 22]},
            {"label": "HA1.5", "value_cols": [23, 24, 25, 26]},
        ],
    },
    {
        "section_name": "cell_based_assay",
        "time_col": 14,
        "start_row": 20,
        "end_row": 31,
        "groups": [
            {"label": "NT-3", "value_cols": [15, 16, 17, 18]},
            {"label": "HPHA1.5", "value_cols": [19, 20, 21, 22]},
            {"label": "HA1.5", "value_cols": [23, 24, 25, 26]},
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
        default=repo_root / "outputs" / "142_external_nt3_hydrogel_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("__", "_")
        .strip("_")
    )


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def build_fl_curves(df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    time_days = pd.to_numeric(df.iloc[3:, 0], errors="coerce")
    for group in FL_GROUPS:
        values = pd.to_numeric(df.iloc[3:, group["mean_col"]], errors="coerce")
        replicate_df = df.iloc[3:, group["replicate_cols"]].apply(pd.to_numeric, errors="coerce")
        curve_key = slugify(f"fitc_lysozyme_{group['label']}")
        unified_curve_id = f"nt3_hydrogel_mendeley_cs6x4f86f2:{curve_key}"

        for idx in range(len(time_days)):
            t = time_days.iloc[idx]
            v = values.iloc[idx]
            if pd.isna(t) or pd.isna(v):
                continue
            reps = replicate_df.iloc[idx].dropna().to_numpy(float)
            sd = float(np.std(reps, ddof=1)) if len(reps) >= 2 else np.nan
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{float(t):.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "nt3_hydrogel_mendeley_cs6x4f86f2",
                    "source_curve_id": curve_key,
                    "time_raw": float(t),
                    "time_unit": "day",
                    "time_days": float(t),
                    "release_raw": float(v),
                    "release_unit": "percent_released",
                    "release_fraction": float(v) / 100.0,
                    "release_percent": float(v),
                    "curve_series": "FITC-Lysozyme release (%)",
                    "curve_level": "formulation_mean",
                    "source_group": group["label"],
                    "payload_name": "FITC-Lysozyme",
                    "measurement_assay": "fluorescence_tracer",
                    "release_sd_percent": sd if pd.notna(sd) else pd.NA,
                }
            )

        formulation_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "nt3_hydrogel_mendeley_cs6x4f86f2",
                "source_curve_id": curve_key,
                "source_group": group["label"],
                "source_has_explicit_group": True,
                "polymer_family": "heparinised poloxamer hydrogel with hyaluronic acid",
                "payload_name": "FITC-Lysozyme",
                "experimental_panel": "release_of_fl",
                "curve_level": "formulation_mean",
                "normalization_basis": "reported_percent_released",
                "release_measure_type": "cumulative_percent_released",
                "measurement_assay": "fluorescence_tracer",
                "P407_percent_wv": 20.0,
                "P188_percent_wv": 5.0,
                "HA_percent_wv": group["label"].replace("HP", ""),
                "NaCl_percent_wv": 0.6 if group["label"] == "HA0" else pd.NA,
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
    return curve_rows, formulation_rows


def build_nt3_curves(df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    for section in NT3_SECTIONS:
        times = pd.to_numeric(df.iloc[section["start_row"] : section["end_row"] + 1, section["time_col"]], errors="coerce")
        for group in section["groups"]:
            curve_key = slugify(f"nt3_{section['section_name']}_{group['label']}")
            unified_curve_id = f"nt3_hydrogel_mendeley_cs6x4f86f2:{curve_key}"
            block = df.iloc[section["start_row"] : section["end_row"] + 1, group["value_cols"]].apply(pd.to_numeric, errors="coerce")
            means = block.mean(axis=1, skipna=True)
            sds = block.std(axis=1, skipna=True, ddof=1)

            for idx in range(len(times)):
                t = times.iloc[idx]
                mean_val = means.iloc[idx]
                if pd.isna(t) or pd.isna(mean_val):
                    continue
                sd = sds.iloc[idx]
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{float(t):.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "nt3_hydrogel_mendeley_cs6x4f86f2",
                        "source_curve_id": curve_key,
                        "time_raw": float(t),
                        "time_unit": "day",
                        "time_days": float(t),
                        "release_raw": float(mean_val),
                        "release_unit": "percent_released",
                        "release_fraction": float(mean_val) / 100.0,
                        "release_percent": float(mean_val),
                        "curve_series": f"NT-3 {section['section_name']} (%)",
                        "curve_level": "formulation_mean",
                        "source_group": group["label"],
                        "payload_name": "NT-3",
                        "measurement_assay": section["section_name"],
                        "release_sd_percent": float(sd) if pd.notna(sd) else pd.NA,
                    }
                )

            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "nt3_hydrogel_mendeley_cs6x4f86f2",
                    "source_curve_id": curve_key,
                    "source_group": group["label"],
                    "source_has_explicit_group": True,
                    "polymer_family": "heparinised poloxamer hydrogel with hyaluronic acid",
                    "payload_name": "NT-3",
                    "experimental_panel": "release_of_nt3",
                    "curve_level": "formulation_mean",
                    "normalization_basis": "reported_percent_released",
                    "release_measure_type": "cumulative_percent_released",
                    "measurement_assay": section["section_name"],
                    "P407_percent_wv": 20.0,
                    "P188_percent_wv": 5.0,
                    "HA_percent_wv": 1.5 if "HA1.5" in group["label"] else pd.NA,
                    "NaCl_percent_wv": 0.6 if group["label"] == "NT-3" else pd.NA,
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
    return curve_rows, formulation_rows


def load_release_workbook(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fl_df = pd.read_excel(xlsx_path, sheet_name="Release of FL", header=None)
    nt3_df = pd.read_excel(xlsx_path, sheet_name="Release of NT-3", header=None)

    fl_curve_rows, fl_formulation_rows = build_fl_curves(fl_df)
    nt3_curve_rows, nt3_formulation_rows = build_nt3_curves(nt3_df)
    curves = pd.DataFrame(fl_curve_rows + nt3_curve_rows)
    formulations = pd.DataFrame(fl_formulation_rows + nt3_formulation_rows)
    return curves, formulations


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
                "measurement_assay": group["measurement_assay"].iloc[0],
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
                "source_dataset": "nt3_hydrogel_mendeley_cs6x4f86f2",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "fraction_release_gt_1p0": float(curve_quality["has_release_gt_1p0"].mean()),
                "n_measurement_assays": int(curve_quality["measurement_assay"].nunique()),
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, formulations: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_nt3_hydrogel_mendeley",
        "",
        "Standardized cumulative-release corpus from Mendeley dataset `cs6x4f86f2`, using the",
        "release sheets in `Data for manuscript.xlsx`.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        f"- assay families: `{int(ds['n_measurement_assays'])}`",
        "",
        "## Important Note",
        "",
        "- The workbook mixes tracer release (`FITC-Lysozyme`) and NT-3 release measured by two assay families (`elisa`, `cell_based_assay`).",
        "- All exported curves are reported as cumulative percent released, but assay identity is preserved in `measurement_assay`.",
        "",
        "## Exported Curves",
        "",
    ]
    for _, row in formulations.sort_values(["payload_name", "measurement_assay", "source_group"]).iterrows():
        lines.append(f"- `{row['payload_name']}` / `{row['measurement_assay']}` / `{row['source_group']}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = (
        args.repo_root
        / "data"
        / "external"
        / "cs6x4f86f2_mendeley"
        / "Data on heparinised poloxamer hydrogel with hyaluronic acid can stabilise and sustain the release of bioactive NT-3"
        / "Data for manuscript.xlsx"
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
                "corpus_name": "external_nt3_hydrogel_mendeley",
                "source_dataset": "nt3_hydrogel_mendeley_cs6x4f86f2",
                "mendeley_dataset_id": "cs6x4f86f2",
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

    print(f"[external-nt3-hydrogel-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-nt3-hydrogel-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
