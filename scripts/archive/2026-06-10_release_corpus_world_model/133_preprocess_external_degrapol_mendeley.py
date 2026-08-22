from __future__ import annotations

"""
133_preprocess_external_degrapol_mendeley.py

Consume:
- data/external/mendeley_degrapol_release/4hgmtc5ph3__Release_kinetics.xlsx

Produce:
- outputs/133_external_degrapol_mendeley/curves_long.csv
- outputs/133_external_degrapol_mendeley/formulations.csv
- outputs/133_external_degrapol_mendeley/curve_quality_summary.csv
- outputs/133_external_degrapol_mendeley/dataset_summary.csv
- outputs/133_external_degrapol_mendeley/manifest.json
- outputs/133_external_degrapol_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SHEET_SPECS = {
    "PDGF-BB": {
        "payload_name": "PDGF-BB",
        "native_release_col": 13,
        "native_release_unit": "pg_per_ml_cumulative_mean",
        "polymer_family": "DegraPol mesh",
    },
    "IGF-1": {
        "payload_name": "IGF-1",
        "native_release_col": 13,
        "native_release_unit": "ng_per_ml_cumulative_mean",
        "polymer_family": "DegraPol mesh",
    },
    "Mix-IGF-1": {
        "payload_name": "IGF-1",
        "native_release_col": 13,
        "native_release_unit": "ng_per_ml_cumulative_mean",
        "polymer_family": "DegraPol mesh",
    },
    "Mix-PDGF-BB": {
        "payload_name": "PDGF-BB",
        "native_release_col": 13,
        "native_release_unit": "ng_per_ml_cumulative_mean",
        "polymer_family": "DegraPol mesh",
    },
}


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
        default=repo_root / "outputs" / "133_external_degrapol_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def detect_release_blocks(sheet_df: pd.DataFrame) -> list[tuple[int, str, int]]:
    blocks: list[tuple[int, str, int]] = []
    for row_idx in range(len(sheet_df)):
        row = sheet_df.iloc[row_idx]
        text_cells = row.dropna().astype(str)
        tube_cells = text_cells[text_cells.str.contains("Tube", case=False, na=False)]
        if tube_cells.empty:
            continue
        header_row = row
        day_cols = header_row.index[header_row.astype(str).eq("[d]")]
        if len(day_cols) == 0 and row_idx > 0:
            header_row = sheet_df.iloc[row_idx - 1]
            day_cols = header_row.index[header_row.astype(str).eq("[d]")]
        if len(day_cols) == 0 and row_idx + 1 < len(sheet_df):
            header_row = sheet_df.iloc[row_idx + 1]
            day_cols = header_row.index[header_row.astype(str).eq("[d]")]
        if len(day_cols) == 0:
            continue
        tube_label = str(tube_cells.iloc[0]).strip()
        time_col = int(day_cols[0])
        data_start_row = row_idx + 1
        first_candidate_is_empty = True
        if data_start_row < len(sheet_df):
            first_candidate_is_empty = pd.isna(sheet_df.iat[data_start_row, time_col])
        if first_candidate_is_empty:
            data_start_row = row_idx + 2
        blocks.append((data_start_row, tube_label, time_col))
    return blocks


def extract_block_curves(
    xlsx_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    form_rows: list[dict[str, object]] = []

    for sheet_name, spec in SHEET_SPECS.items():
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
        blocks = detect_release_blocks(df)
        for data_start_row, tube_label, time_col in blocks:
            tube_key = slugify(f"{sheet_name}_{tube_label}")
            unified_curve_id = f"degrapol_mendeley_4hgmtc5ph3:{tube_key}"
            data_rows: list[dict[str, object]] = []
            row_idx = data_start_row
            while row_idx < len(df):
                time_value = df.iat[row_idx, time_col]
                native_release = (
                    df.iat[row_idx, spec["native_release_col"]]
                    if spec["native_release_col"] < df.shape[1]
                    else np.nan
                )
                if pd.isna(time_value):
                    break
                time_days = pd.to_numeric(time_value, errors="coerce")
                native_release = pd.to_numeric(native_release, errors="coerce")
                if pd.isna(time_days) or pd.isna(native_release):
                    row_idx += 1
                    continue
                data_rows.append(
                    {
                        "time_days": float(time_days),
                        "release_native": float(native_release),
                    }
                )
                row_idx += 1

            if not data_rows:
                continue

            block_df = pd.DataFrame(data_rows).sort_values("time_days").reset_index(drop=True)
            max_native = float(block_df["release_native"].max())
            for _, row in block_df.iterrows():
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{row['time_days']:.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "degrapol_mendeley_4hgmtc5ph3",
                        "source_curve_id": tube_key,
                        "time_raw": float(row["time_days"]),
                        "time_unit": "day",
                        "time_days": float(row["time_days"]),
                        "release_raw": float(row["release_native"]),
                        "release_unit": spec["native_release_unit"],
                        "release_fraction": float(row["release_native"]) / max_native if max_native > 0 else np.nan,
                        "release_percent": (
                            float(row["release_native"]) / max_native * 100.0 if max_native > 0 else np.nan
                        ),
                        "curve_series": sheet_name,
                        "curve_level": "tube_mean_cumulative",
                        "source_group": tube_label,
                        "payload_name": spec["payload_name"],
                    }
                )
            form_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "degrapol_mendeley_4hgmtc5ph3",
                    "source_curve_id": tube_key,
                    "source_group": tube_label,
                    "source_has_explicit_group": True,
                    "polymer_family": spec["polymer_family"],
                    "payload_name": spec["payload_name"],
                    "experimental_panel": sheet_name,
                    "curve_level": "tube_mean_cumulative",
                    "normalization_basis": "curve_max_native_release",
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
                "source_dataset": "degrapol_mendeley_4hgmtc5ph3",
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


def write_summary(dataset_summary: pd.DataFrame, curve_quality: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    panel_counts = (
        curve_quality.groupby(["curve_series", "payload_name"], as_index=False)
        .agg(n_curves=("unified_curve_id", "nunique"), n_points=("n_points", "sum"))
        .sort_values(["curve_series", "payload_name"])
    )
    lines = [
        "# external_degrapol_mendeley",
        "",
        "Standardized preprocessing output for Mendeley dataset `4hgmtc5ph3`",
        "(DegraPol protein release kinetics).",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.3f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Panel breakdown",
        "",
    ]
    for _, row in panel_counts.iterrows():
        lines.append(
            f"- `{row['curve_series']}` / `{row['payload_name']}`: "
            f"`{int(row['n_curves'])}` curves, `{int(row['n_points'])}` points"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = args.repo_root / "data" / "external" / "mendeley_degrapol_release" / "4hgmtc5ph3__Release_kinetics.xlsx"
    curves, formulations = extract_block_curves(xlsx_path)
    curve_quality = build_curve_quality_summary(curves)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_degrapol_mendeley",
                "source_dataset": "degrapol_mendeley_4hgmtc5ph3",
                "mendeley_dataset_id": "4hgmtc5ph3",
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

    print(f"[external-degrapol-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-degrapol-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
