from __future__ import annotations

"""
131_preprocess_external_bmp2_hydrogel_zenodo.py

Consume:
- data/external/bmp2_hydrogel_zenodo_18279506/F3_release_kinetics_ALL_LIGHT.csv
- data/external/bmp2_hydrogel_zenodo_18279506/FigureS2_PanelA_BMP2_Release.csv

Produce:
- outputs/131_external_bmp2_hydrogel_zenodo/curves_long.csv
- outputs/131_external_bmp2_hydrogel_zenodo/formulations.csv
- outputs/131_external_bmp2_hydrogel_zenodo/curve_quality_summary.csv
- outputs/131_external_bmp2_hydrogel_zenodo/dataset_summary.csv
- outputs/131_external_bmp2_hydrogel_zenodo/manifest.json
- outputs/131_external_bmp2_hydrogel_zenodo/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
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
        default=repo_root / "outputs" / "131_external_bmp2_hydrogel_zenodo",
        help="output directory",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def load_f3_series(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(root / "F3_release_kinetics_ALL_LIGHT.csv")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df["mean_release_pct"] = pd.to_numeric(df["mean_release_pct"], errors="coerce")
    df["sd_release_pct"] = pd.to_numeric(df["sd_release_pct"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df = df.dropna(subset=["time_h", "group", "mean_release_pct"]).copy()

    curve_rows: list[dict[str, object]] = []
    form_rows: list[dict[str, object]] = []
    for group_name, group in df.groupby("group", sort=False):
        curve_key = f"f3_{slugify(group_name)}"
        unified_curve_id = f"bmp2_hydrogel_zenodo18279506:{curve_key}"
        for _, row in group.sort_values("time_h").iterrows():
            time_days = float(row["time_h"]) / 24.0
            release_percent = float(row["mean_release_pct"])
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{time_days:.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "bmp2_hydrogel_zenodo18279506",
                    "source_curve_id": curve_key,
                    "time_raw": float(row["time_h"]),
                    "time_unit": "hour",
                    "time_days": time_days,
                    "release_raw": release_percent,
                    "release_unit": "percent",
                    "release_fraction": release_percent / 100.0,
                    "release_percent": release_percent,
                    "curve_series": "F3_ALL_LIGHT_group_mean",
                    "curve_level": "mean",
                    "source_group": group_name,
                    "n_replicates": int(row["n"]) if pd.notna(row["n"]) else pd.NA,
                    "sd_release_percent": float(row["sd_release_pct"]) if pd.notna(row["sd_release_pct"]) else pd.NA,
                }
            )
        form_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "bmp2_hydrogel_zenodo18279506",
                "source_curve_id": curve_key,
                "source_group": group_name,
                "source_has_explicit_group": True,
                "polymer_family": "collagen-alginate hydrogel",
                "payload_name": "BMP-2",
                "experimental_panel": "F3_ALL_LIGHT_group_mean",
                "curve_level": "mean",
                "light_condition": "Light",
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

    return pd.DataFrame(curve_rows), pd.DataFrame(form_rows)


def load_s2_series(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(root / "FigureS2_PanelA_BMP2_Release.csv")
    df.columns = [col.strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    numeric_cols = [
        "Time (min)",
        "BMP-2 Released (%) - Rep1",
        "BMP-2 Released (%) - Rep2",
        "BMP-2 Released (%) - Rep3",
        "Mean (%)",
        "SD (%)",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time (min)"]).copy()

    series_specs = [
        ("rep1", "BMP-2 Released (%) - Rep1", "replicate"),
        ("rep2", "BMP-2 Released (%) - Rep2", "replicate"),
        ("rep3", "BMP-2 Released (%) - Rep3", "replicate"),
        ("mean", "Mean (%)", "mean"),
    ]

    curve_rows: list[dict[str, object]] = []
    form_rows: list[dict[str, object]] = []
    for series_id, value_col, curve_level in series_specs:
        unified_curve_id = f"bmp2_hydrogel_zenodo18279506:s2_{series_id}"
        for _, row in df.sort_values("Time (min)").iterrows():
            release_percent = float(row[value_col])
            time_days = float(row["Time (min)"]) / (24.0 * 60.0)
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{time_days:.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "bmp2_hydrogel_zenodo18279506",
                    "source_curve_id": f"s2_{series_id}",
                    "time_raw": float(row["Time (min)"]),
                    "time_unit": "minute",
                    "time_days": time_days,
                    "release_raw": release_percent,
                    "release_unit": "percent",
                    "release_fraction": release_percent / 100.0,
                    "release_percent": release_percent,
                    "curve_series": "FigureS2_PanelA_BMP2_Release",
                    "curve_level": curve_level,
                    "source_group": "FigureS2_PanelA_BMP2_Release",
                    "n_replicates": 3 if series_id == "mean" else 1,
                    "sd_release_percent": float(row["SD (%)"]) if curve_level == "mean" else pd.NA,
                }
            )
        form_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "bmp2_hydrogel_zenodo18279506",
                "source_curve_id": f"s2_{series_id}",
                "source_group": "FigureS2_PanelA_BMP2_Release",
                "source_has_explicit_group": True,
                "polymer_family": "collagen-alginate hydrogel",
                "payload_name": "BMP-2",
                "experimental_panel": "FigureS2_PanelA_BMP2_Release",
                "curve_level": curve_level,
                "light_condition": "not_reported",
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

    return pd.DataFrame(curve_rows), pd.DataFrame(form_rows)


def build_curve_quality_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for curve_id, group in curves.groupby("unified_curve_id", sort=False):
        group = group.sort_values("time_days")
        release = group["release_fraction"].to_numpy(float)
        rows.append(
            {
                "unified_curve_id": curve_id,
                "source_dataset": group["source_dataset"].iloc[0],
                "source_curve_id": group["source_curve_id"].iloc[0],
                "curve_series": group["curve_series"].iloc[0],
                "curve_level": group["curve_level"].iloc[0],
                "source_group": group["source_group"].iloc[0],
                "n_points": int(len(group)),
                "time_min_days": float(group["time_days"].min()),
                "time_max_days": float(group["time_days"].max()),
                "duration_days": float(group["time_days"].max() - group["time_days"].min()),
                "final_release_fraction": float(group["release_fraction"].iloc[-1]),
                "max_release_fraction": float(group["release_fraction"].max()),
                "monotonicity_violations": monotonicity_violation_count(release),
                "is_monotone_non_decreasing": monotonicity_violation_count(release) == 0,
                "has_release_gt_1p0": bool((group["release_fraction"] > 1.0 + 1e-6).any()),
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "bmp2_hydrogel_zenodo18279506",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "median_final_release_fraction": float(curve_quality["final_release_fraction"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "fraction_release_gt_1p0": float(curve_quality["has_release_gt_1p0"].mean()),
            }
        ]
    )


def write_summary(
    dataset_summary: pd.DataFrame,
    curve_quality: pd.DataFrame,
    out_path: Path,
) -> None:
    ds = dataset_summary.iloc[0]
    panel_counts = (
        curve_quality.groupby(["curve_series", "curve_level"], as_index=False)
        .agg(n_curves=("unified_curve_id", "nunique"), n_points=("n_points", "sum"))
        .sort_values(["curve_series", "curve_level"])
    )
    lines = [
        "# external_bmp2_hydrogel_zenodo",
        "",
        "Standardized preprocessing output for Zenodo record `18279506`",
        "(Visible-Light–Triggered BMP-2 Release from Enzymatically Crosslinked Marine Collagen–Alginate Hydrogels).",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Panel breakdown",
        "",
    ]
    for _, row in panel_counts.iterrows():
        lines.append(
            f"- `{row['curve_series']}` / `{row['curve_level']}`: "
            f"`{int(row['n_curves'])}` curves, `{int(row['n_points'])}` points"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = args.repo_root / "data" / "external" / "bmp2_hydrogel_zenodo_18279506"
    f3_curves, f3_forms = load_f3_series(root)
    s2_curves, s2_forms = load_s2_series(root)

    curves = pd.concat([f3_curves, s2_curves], ignore_index=True)
    formulations = pd.concat([f3_forms, s2_forms], ignore_index=True)
    curve_quality = build_curve_quality_summary(curves)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_bmp2_hydrogel_zenodo",
                "source_dataset": "bmp2_hydrogel_zenodo18279506",
                "zenodo_record": 18279506,
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "panel_names": sorted(curves["curve_series"].unique().tolist()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, curve_quality, args.outdir / "summary.md")

    print(f"[external-bmp2-hydrogel-zenodo] wrote outputs to {args.outdir}")
    print(
        "[external-bmp2-hydrogel-zenodo] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
