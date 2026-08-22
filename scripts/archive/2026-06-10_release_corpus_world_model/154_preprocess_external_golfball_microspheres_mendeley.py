from __future__ import annotations

"""
154_preprocess_external_golfball_microspheres_mendeley.py

Consume:
- data/external/kbwcw7w4rn_mendeley/extracted/Research Data/Data2.xlsx

Produce:
- outputs/154_external_golfball_microspheres_mendeley/curves_long.csv
- outputs/154_external_golfball_microspheres_mendeley/formulations.csv
- outputs/154_external_golfball_microspheres_mendeley/curve_quality_summary.csv
- outputs/154_external_golfball_microspheres_mendeley/dataset_summary.csv
- outputs/154_external_golfball_microspheres_mendeley/manifest.json
- outputs/154_external_golfball_microspheres_mendeley/summary.md

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
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "154_external_golfball_microspheres_mendeley",
    )
    return parser.parse_args()


def monotonicity_violation_count(values: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(values) < -tol))


def load_fig7ab(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name="Fig. 7AB", header=None)
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []

    row_idx = 0
    while row_idx < len(df):
        title = df.iat[row_idx, 0]
        if not isinstance(title, str) or "microspheres" not in title.lower():
            row_idx += 1
            continue

        header_row = row_idx + 1
        sample_label = str(df.iat[header_row, 1]).split("-")[0]
        data_start = header_row + 1
        data_rows: list[dict[str, float]] = []
        cursor = data_start
        while cursor < len(df):
            time_value = pd.to_numeric(df.iat[cursor, 0], errors="coerce")
            if pd.isna(time_value):
                break
            reps = [pd.to_numeric(df.iat[cursor, j], errors="coerce") for j in [1, 2, 3]]
            mean_value = pd.to_numeric(df.iat[cursor, 4], errors="coerce")
            sd_value = pd.to_numeric(df.iat[cursor, 5], errors="coerce")
            if pd.isna(mean_value):
                break
            data_rows.append(
                {
                    "time_days": float(time_value),
                    "rep1": float(reps[0]),
                    "rep2": float(reps[1]),
                    "rep3": float(reps[2]),
                    "mean_percent": float(mean_value),
                    "sd_percent": float(sd_value) if pd.notna(sd_value) else np.nan,
                }
            )
            cursor += 1

        curve_key = sample_label.lower()
        unified_curve_id = f"golfball_microspheres_mendeley_kbwcw7w4rn:{curve_key}"
        for row in data_rows:
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{row['time_days']:.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "golfball_microspheres_mendeley_kbwcw7w4rn",
                    "source_curve_id": curve_key,
                    "time_raw": row["time_days"],
                    "time_unit": "day",
                    "time_days": row["time_days"],
                    "release_raw": row["mean_percent"],
                    "release_unit": "percent_released",
                    "release_fraction": row["mean_percent"] / 100.0,
                    "release_percent": row["mean_percent"],
                    "curve_series": "pH 7.4 medium",
                    "curve_level": "microsphere_mean_from_triplicates",
                    "source_group": sample_label,
                    "payload_name": "rotigotine",
                    "release_sd_percent": row["sd_percent"] if pd.notna(row["sd_percent"]) else pd.NA,
                    "n_replicates": 3,
                }
            )
        formulation_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "golfball_microspheres_mendeley_kbwcw7w4rn",
                "source_curve_id": curve_key,
                "source_group": sample_label,
                "source_has_explicit_group": True,
                "polymer_family": "golf ball-shaped microsphere",
                "payload_name": "rotigotine",
                "experimental_panel": "Fig. 7AB",
                "curve_level": "microsphere_mean_from_triplicates",
                "normalization_basis": "reported_percent_released",
                "release_measure_type": "cumulative_percent_released",
                "release_medium_condition": "pH 7.4 medium",
                "worksheet_title": title,
                "source_sheet": "Fig. 7AB",
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
        row_idx = cursor + 1

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
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "golfball_microspheres_mendeley_kbwcw7w4rn",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "release_measure_type": formulations["release_measure_type"].iloc[0],
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_golfball_microspheres_mendeley",
        "",
        "Standardized external cumulative-release corpus from Mendeley dataset `kbwcw7w4rn`, using",
        "the `Fig. 7AB` microsphere release table.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Important Caveat",
        "",
        "- This preprocessing uses the explicit microsphere release panel from `Fig. 7AB` only.",
        "- The workbook contains other figures and measurements, but they are not merged into this release corpus.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = (
        args.repo_root
        / "data"
        / "external"
        / "kbwcw7w4rn_mendeley"
        / "extracted"
        / "Research Data"
        / "Data2.xlsx"
    )
    curves, formulations = load_fig7ab(xlsx_path)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_golfball_microspheres_mendeley",
                "source_dataset": "golfball_microspheres_mendeley_kbwcw7w4rn",
                "mendeley_dataset_id": "kbwcw7w4rn",
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
    write_summary(dataset_summary, args.outdir / "summary.md")

    print(f"[external-golfball-microspheres-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-golfball-microspheres-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
