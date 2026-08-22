from __future__ import annotations

"""
155_preprocess_external_plga_dox_mendeley.py

Consume:
- data/external/h4tt4433w9_mendeley/extracted/.../drug release profile.xlsx

Produce:
- outputs/155_external_plga_dox_mendeley/curves_long.csv
- outputs/155_external_plga_dox_mendeley/formulations.csv
- outputs/155_external_plga_dox_mendeley/curve_quality_summary.csv
- outputs/155_external_plga_dox_mendeley/dataset_summary.csv
- outputs/155_external_plga_dox_mendeley/manifest.json
- outputs/155_external_plga_dox_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_SPECS = [
    {
        "label": "Free DOX",
        "curve_key": "free_dox_control",
        "cols": [1, 2, 3],
        "is_control": True,
        "particle_architecture": "free_drug_control",
    },
    {
        "label": "Matrix DOX-loaded NPs",
        "curve_key": "matrix_dox_loaded_nps",
        "cols": [4, 5, 6],
        "is_control": False,
        "particle_architecture": "matrix_np",
    },
    {
        "label": "Core-shell DOX-loaded NPs",
        "curve_key": "core_shell_dox_loaded_nps",
        "cols": [7, 8, 9],
        "is_control": False,
        "particle_architecture": "core_shell_np",
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "155_external_plga_dox_mendeley",
    )
    return parser.parse_args()


def monotonicity_violation_count(values: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(values) < -tol))


def find_release_workbook(repo_root: Path) -> Path:
    short_path = repo_root / "data" / "external" / "h4tt4433w9_mendeley" / "drug_release_profile.xlsx"
    if short_path.exists():
        return short_path
    raise FileNotFoundError("drug_release_profile.xlsx not found under h4tt4433w9_mendeley")


def load_release_workbook(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(f"\\\\?\\{xlsx_path}", sheet_name="Sheet1", header=None)
    time_values = pd.to_numeric(df.iloc[1:, 0], errors="coerce")

    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    for spec in SYSTEM_SPECS:
        raw_block = df.iloc[1:, spec["cols"]].apply(pd.to_numeric, errors="coerce")
        valid = time_values.notna() & raw_block.notna().all(axis=1)
        time_h = time_values[valid].astype(float).to_numpy()
        block = raw_block.loc[valid].astype(float)
        release_mean = block.mean(axis=1).to_numpy()
        release_sd = block.std(axis=1, ddof=1).to_numpy()

        unified_curve_id = f"plga_dox_mendeley_h4tt4433w9:{spec['curve_key']}"
        for i in range(len(time_h)):
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{time_h[i]:.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "plga_dox_mendeley_h4tt4433w9",
                    "source_curve_id": spec["curve_key"],
                    "time_raw": float(time_h[i]),
                    "time_unit": "hour",
                    "time_days": float(time_h[i]) / 24.0,
                    "release_raw": float(release_mean[i]),
                    "release_unit": "percent_released",
                    "release_fraction": float(release_mean[i]) / 100.0,
                    "release_percent": float(release_mean[i]),
                    "curve_series": spec["label"],
                    "curve_level": "condition_mean_from_triplicates",
                    "source_group": spec["label"],
                    "payload_name": "doxorubicin",
                    "release_sd_percent": float(release_sd[i]) if pd.notna(release_sd[i]) else pd.NA,
                    "n_replicates": 3,
                    "is_free_drug_control": spec["is_control"],
                }
            )
        formulation_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "plga_dox_mendeley_h4tt4433w9",
                "source_curve_id": spec["curve_key"],
                "source_group": spec["label"],
                "source_has_explicit_group": True,
                "polymer_family": "PLGA nanoparticle",
                "payload_name": "doxorubicin",
                "experimental_panel": "drug release profile",
                "curve_level": "condition_mean_from_triplicates",
                "normalization_basis": "reported_percent_released",
                "release_measure_type": "cumulative_percent_released",
                "measurement_assay": "release profile worksheet",
                "particle_architecture": spec["particle_architecture"],
                "is_free_drug_control": spec["is_control"],
                "release_medium_condition": "PBS release profile",
                "source_sheet": "Sheet1",
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

    return pd.DataFrame(curve_rows), pd.DataFrame(formulation_rows)


def build_curve_quality_summary(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    form_index = formulations.set_index("unified_curve_id")
    rows: list[dict[str, object]] = []
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
                "is_free_drug_control": bool(meta["is_free_drug_control"]),
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "plga_dox_mendeley_h4tt4433w9",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "n_free_drug_controls": int(formulations["is_free_drug_control"].sum()),
                "release_measure_type": formulations["release_measure_type"].iloc[0],
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_plga_dox_mendeley",
        "",
        "Standardized external cumulative-release corpus from Mendeley dataset `h4tt4433w9`, using",
        "the `drug release profile.xlsx` worksheet.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        f"- free-drug controls: `{int(ds['n_free_drug_controls'])}`",
        "",
        "## Important Caveat",
        "",
        "- This corpus includes one `Free DOX` comparator alongside two PLGA nanoparticle release curves.",
        "- Because the free-drug control is not a carrier formulation, the corpus belongs in the caveat cumulative layer rather than the clean main pool.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = find_release_workbook(args.repo_root)
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
                "corpus_name": "external_plga_dox_mendeley",
                "source_dataset": "plga_dox_mendeley_h4tt4433w9",
                "mendeley_dataset_id": "h4tt4433w9",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "release_measure_type": "cumulative_percent_released",
                "eligible_for_main_cumulative_pool": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, args.outdir / "summary.md")

    print(f"[external-plga-dox-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-plga-dox-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
