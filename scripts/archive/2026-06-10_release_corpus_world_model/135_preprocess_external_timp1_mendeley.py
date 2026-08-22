from __future__ import annotations

"""
135_preprocess_external_timp1_mendeley.py

Consume:
- data/external/mendeley_degrapol_release/3cm7sytr3r__Release.xlsx

Produce:
- outputs/135_external_timp1_mendeley/curves_long.csv
- outputs/135_external_timp1_mendeley/formulations.csv
- outputs/135_external_timp1_mendeley/curve_quality_summary.csv
- outputs/135_external_timp1_mendeley/dataset_summary.csv
- outputs/135_external_timp1_mendeley/manifest.json
- outputs/135_external_timp1_mendeley/summary.md

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
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="repository root",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "135_external_timp1_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def load_anonymous_blocks(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path).copy()
    df["time_days"] = pd.to_numeric(
        df["Time (days)"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    df["release_percent"] = pd.to_numeric(df["[TIMP-1] % of maximum at day7"], errors="coerce")
    df = df.dropna(subset=["time_days"]).copy()
    df["block_id"] = df.index // 10 + 1
    return df


def build_curves_and_formulations(block_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []

    for block_id, group in block_df.groupby("block_id", sort=True):
        curve_key = f"anonymous_block_{int(block_id):02d}"
        unified_curve_id = f"timp1_mendeley_3cm7sytr3r:{curve_key}"
        grouped = (
            group.groupby("time_days", as_index=False)
            .agg(
                release_percent=("release_percent", "mean"),
                sd_release_percent=("release_percent", "std"),
                n_replicates=("release_percent", lambda x: int(x.notna().sum())),
            )
            .sort_values("time_days")
        )
        max_release = float(grouped["release_percent"].max()) if not grouped.empty else np.nan
        for _, row in grouped.iterrows():
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{float(row['time_days']):.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "timp1_mendeley_3cm7sytr3r",
                    "source_curve_id": curve_key,
                    "time_raw": float(row["time_days"]),
                    "time_unit": "day",
                    "time_days": float(row["time_days"]),
                    "release_raw": float(row["release_percent"]),
                    "release_unit": "percent_of_max_day7",
                    "release_fraction": (
                        float(row["release_percent"]) / max_release if max_release and max_release > 0 else np.nan
                    ),
                    "release_percent": float(row["release_percent"]),
                    "curve_series": "Release data",
                    "curve_level": "anonymous_block_mean",
                    "source_group": curve_key,
                    "payload_name": "TIMP-1",
                    "n_replicates": int(row["n_replicates"]),
                    "sd_release_percent": (
                        float(row["sd_release_percent"]) if pd.notna(row["sd_release_percent"]) else pd.NA
                    ),
                }
            )
        formulation_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "timp1_mendeley_3cm7sytr3r",
                "source_curve_id": curve_key,
                "source_group": curve_key,
                "source_has_explicit_group": False,
                "polymer_family": "DegraPol tube",
                "payload_name": "TIMP-1",
                "experimental_panel": "Release data",
                "curve_level": "anonymous_block_mean",
                "normalization_basis": "sheet_percent_of_max_day7",
                "anonymized_block_structure": "10 rows per block: two readings at days 0,1,2,4,7",
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
                "source_dataset": "timp1_mendeley_3cm7sytr3r",
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


def write_summary(dataset_summary: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_timp1_mendeley",
        "",
        "Cautious standardized preprocessing output for Mendeley dataset `3cm7sytr3r`",
        "using anonymous repeated 10-row release blocks from `Release.xlsx`.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.1f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Parsing rule",
        "",
        "- The sheet has 90 rows and decomposes into 9 consecutive 10-row blocks.",
        "- Each block follows `day0, day0, day1, day1, day2, day2, day4, day4, day7, day7`.",
        "- Each exported curve is the within-block mean across duplicate days.",
        "- Block identities are anonymous because the workbook does not expose reliable group labels.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = args.repo_root / "data" / "external" / "mendeley_degrapol_release" / "3cm7sytr3r__Release.xlsx"
    block_df = load_anonymous_blocks(xlsx_path)
    curves, formulations = build_curves_and_formulations(block_df)
    curve_quality = build_curve_quality_summary(curves)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_timp1_mendeley",
                "source_dataset": "timp1_mendeley_3cm7sytr3r",
                "mendeley_dataset_id": "3cm7sytr3r",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "parsing_mode": "anonymous_10_row_blocks_mean",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, args.outdir / "summary.md")

    print(f"[external-timp1-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-timp1-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
