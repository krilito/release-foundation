from __future__ import annotations

"""
137_preprocess_external_alginate_mendeley.py

Consume:
- data/external/alginate_microbeads_mendeley_hgtphykjnb/Release_Acid and Buffer.xlsx

Produce:
- outputs/137_external_alginate_mendeley/curves_long.csv
- outputs/137_external_alginate_mendeley/formulations.csv
- outputs/137_external_alginate_mendeley/curve_quality_summary.csv
- outputs/137_external_alginate_mendeley/dataset_summary.csv
- outputs/137_external_alginate_mendeley/manifest.json
- outputs/137_external_alginate_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SHEET_SPECS = {
    "pH1.2 figure": {
        "condition_label": "pH1.2",
        "data_start_row": 5,
        "mean_time_col": 0,
        "mean_cols": {"F1/7": 1, "F2/7": 2, "F3/7": 3},
        "sd_cols": {"F1/7": 12, "F2/7": 13, "F3/7": 14},
    },
    " pH6.8 figure": {
        "condition_label": "pH6.8",
        "data_start_row": 4,
        "mean_time_col": 0,
        "mean_cols": {"F1/7": 1, "F2/7": 2, "F3/7": 3},
        "sd_cols": {"F1/7": 13, "F2/7": 14, "F3/7": 15},
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="repository root")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "137_external_alginate_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def slugify_formulation(value: str) -> str:
    return value.lower().replace("/", "_")


def extract_curves_and_formulations(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    emitted_formulations: set[str] = set()

    for sheet_name, spec in SHEET_SPECS.items():
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
        for formulation_label, mean_col in spec["mean_cols"].items():
            curve_key = f"{slugify_formulation(formulation_label)}_{spec['condition_label'].lower().replace('.', '_')}"
            unified_curve_id = f"alginate_mendeley_hgtphykjnb:{curve_key}"
            sd_col = spec["sd_cols"][formulation_label]

            for row_idx in range(spec["data_start_row"], len(df)):
                time_days = pd.to_numeric(df.iat[row_idx, spec["mean_time_col"]], errors="coerce")
                release_percent = pd.to_numeric(df.iat[row_idx, mean_col], errors="coerce")
                sd_release_percent = pd.to_numeric(df.iat[row_idx, sd_col], errors="coerce")
                if pd.isna(time_days) or pd.isna(release_percent):
                    continue
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{float(time_days):.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "alginate_mendeley_hgtphykjnb",
                        "source_curve_id": curve_key,
                        "time_raw": float(time_days),
                        "time_unit": "minute",
                        "time_days": float(time_days) / (24.0 * 60.0),
                        "release_raw": float(release_percent),
                        "release_unit": "percent_released",
                        "release_fraction": float(release_percent) / 100.0,
                        "release_percent": float(release_percent),
                        "curve_series": spec["condition_label"],
                        "curve_level": "formulation_mean_from_figure",
                        "source_group": formulation_label,
                        "payload_name": "5-FU",
                        "sd_release_percent": float(sd_release_percent) if pd.notna(sd_release_percent) else pd.NA,
                    }
                )

            if unified_curve_id in emitted_formulations:
                continue
            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "alginate_mendeley_hgtphykjnb",
                    "source_curve_id": curve_key,
                    "source_group": formulation_label,
                    "source_has_explicit_group": True,
                    "polymer_family": "Alginate microbead",
                    "payload_name": "5-FU",
                    "experimental_panel": spec["condition_label"],
                    "curve_level": "formulation_mean_from_figure",
                    "normalization_basis": "reported_percent_released",
                    "release_medium_condition": spec["condition_label"],
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
            emitted_formulations.add(unified_curve_id)

    curves = pd.DataFrame(curve_rows).sort_values(["unified_curve_id", "time_days"]).reset_index(drop=True)
    formulations = pd.DataFrame(formulation_rows).sort_values("unified_curve_id").reset_index(drop=True)
    return curves, formulations


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
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "alginate_mendeley_hgtphykjnb",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "n_conditions": int(formulations["experimental_panel"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "median_final_release_fraction": float(curve_quality["final_release_fraction"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "fraction_release_gt_1p0": float(curve_quality["has_release_gt_1p0"].mean()),
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, formulations: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    condition_rows = formulations[["experimental_panel", "source_group"]].drop_duplicates()
    lines = [
        "# external_alginate_mendeley",
        "",
        "Standardized formulation-mean alginate microbead release corpus from Mendeley dataset `hgtphykjnb`",
        "using the cleaned figure-sheet means and SDs from `Release_Acid and Buffer.xlsx`.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- pH conditions: `{int(ds['n_conditions'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Exported curves",
        "",
    ]
    for _, row in condition_rows.sort_values(["experimental_panel", "source_group"]).iterrows():
        lines.append(f"- `{row['source_group']}` under `{row['experimental_panel']}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = (
        args.repo_root
        / "data"
        / "external"
        / "alginate_microbeads_mendeley_hgtphykjnb"
        / "Release_Acid and Buffer.xlsx"
    )
    curves, formulations = extract_curves_and_formulations(xlsx_path)
    curve_quality = build_curve_quality_summary(curves)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_alginate_mendeley",
                "source_dataset": "alginate_mendeley_hgtphykjnb",
                "mendeley_dataset_id": "hgtphykjnb",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "conditions": sorted(formulations["experimental_panel"].unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, formulations, args.outdir / "summary.md")

    print(f"[external-alginate-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-alginate-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
