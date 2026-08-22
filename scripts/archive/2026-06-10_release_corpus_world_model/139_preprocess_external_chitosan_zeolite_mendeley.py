from __future__ import annotations

"""
139_preprocess_external_chitosan_zeolite_mendeley.py

Consume:
- data/external/kvv2jpjpz7_mendeley/Release raw data.xlsx
- data/external/kvv2jpjpz7_mendeley/Readme.txt

Produce:
- outputs/139_external_chitosan_zeolite_mendeley/curves_long.csv
- outputs/139_external_chitosan_zeolite_mendeley/formulations.csv
- outputs/139_external_chitosan_zeolite_mendeley/curve_quality_summary.csv
- outputs/139_external_chitosan_zeolite_mendeley/dataset_summary.csv
- outputs/139_external_chitosan_zeolite_mendeley/manifest.json
- outputs/139_external_chitosan_zeolite_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BLOCK_SPECS = [
    {"row_start": 0, "condition_label": "pH4.0", "buffer_label": "Citrate buffer"},
    {"row_start": 15, "condition_label": "pH5.5", "buffer_label": "Citrate buffer"},
    {"row_start": 30, "condition_label": "pH7.4", "buffer_label": "DPBS"},
]

MATERIAL_COLS = {
    "NaXCHRIS": (0, 1, 2),
    "CaXCHRIS": (4, 5, 6),
    "CaMgXCHRIS": (8, 9, 10),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="repository root")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "139_external_chitosan_zeolite_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_release_blocks(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name="Arkusz1", header=None)
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    emitted: set[str] = set()

    for block in BLOCK_SPECS:
        header_text = str(df.iat[block["row_start"], 0])
        calibration_text = str(df.iat[block["row_start"] + 1, 0])
        temperature_c = extract_float(r"T=(\d+(?:\.\d+)?)", header_text)
        withdrawal_ml = extract_float(r"withdrawal (\d+(?:\.\d+)?) ml", header_text)
        calibration_slope = extract_float(r"y=(\d+(?:\.\d+)?)x", calibration_text)
        calibration_intercept = extract_float(r"x\+(\d+(?:\.\d+)?)", calibration_text)

        for material_label, (time_col, abs1_col, abs2_col) in MATERIAL_COLS.items():
            curve_key = f"{slugify(material_label)}_{slugify(block['condition_label'])}"
            unified_curve_id = f"chitosan_zeolite_mendeley_kvv2jpjpz7:{curve_key}"
            data_rows: list[dict[str, object]] = []

            row_idx = block["row_start"] + 4
            while row_idx < len(df):
                time_value = pd.to_numeric(df.iat[row_idx, time_col], errors="coerce")
                if pd.isna(time_value):
                    break
                abs_values = [
                    pd.to_numeric(df.iat[row_idx, abs1_col], errors="coerce"),
                    pd.to_numeric(df.iat[row_idx, abs2_col], errors="coerce"),
                ]
                abs_values = [float(v) for v in abs_values if pd.notna(v)]
                signal_mean = float(np.mean(abs_values)) if abs_values else np.nan
                signal_std = float(np.std(abs_values, ddof=1)) if len(abs_values) > 1 else np.nan
                signal_fraction_of_max = np.nan
                data_rows.append(
                    {
                        "time_min": float(time_value),
                        "signal_mean": signal_mean,
                        "signal_std": signal_std,
                        "n_replicates": int(len(abs_values)),
                    }
                )
                row_idx += 1

            if not data_rows:
                continue

            block_df = pd.DataFrame(data_rows)
            max_signal = float(block_df["signal_mean"].max()) if block_df["signal_mean"].notna().any() else np.nan
            for _, row in block_df.iterrows():
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{row['time_min']:.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "chitosan_zeolite_mendeley_kvv2jpjpz7",
                        "source_curve_id": curve_key,
                        "time_raw": float(row["time_min"]),
                        "time_unit": "minute",
                        "time_days": float(row["time_min"]) / (24.0 * 60.0),
                        "release_raw": float(row["signal_mean"]) if pd.notna(row["signal_mean"]) else np.nan,
                        "release_unit": "absorbance_proxy",
                        "release_fraction": (
                            float(row["signal_mean"]) / max_signal
                            if pd.notna(row["signal_mean"]) and pd.notna(max_signal) and max_signal > 0
                            else np.nan
                        ),
                        "release_percent": (
                            float(row["signal_mean"]) / max_signal * 100.0
                            if pd.notna(row["signal_mean"]) and pd.notna(max_signal) and max_signal > 0
                            else np.nan
                        ),
                        "curve_series": block["condition_label"],
                        "curve_level": "time_resolved_absorbance_mean",
                        "source_group": material_label,
                        "payload_name": "risedronate",
                        "signal_std": float(row["signal_std"]) if pd.notna(row["signal_std"]) else pd.NA,
                        "n_replicates": int(row["n_replicates"]),
                    }
                )

            if unified_curve_id in emitted:
                continue
            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "chitosan_zeolite_mendeley_kvv2jpjpz7",
                    "source_curve_id": curve_key,
                    "source_group": material_label,
                    "source_has_explicit_group": True,
                    "polymer_family": "chitosan-zeolite composite powder",
                    "payload_name": "risedronate",
                    "experimental_panel": block["condition_label"],
                    "curve_level": "time_resolved_absorbance_mean",
                    "normalization_basis": "signal_mean_divided_by_curve_max",
                    "release_measure_type": "time_resolved_absorbance_proxy",
                    "release_medium_condition": header_text,
                    "buffer_label": block["buffer_label"],
                    "media_pH": float(block["condition_label"].replace("pH", "")),
                    "media_temp_oC": temperature_c,
                    "withdrawal_volume_mL": withdrawal_ml,
                    "calibration_equation": calibration_text,
                    "calibration_slope": calibration_slope,
                    "calibration_intercept": calibration_intercept,
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
    form_index = formulations.set_index("unified_curve_id")
    rows: list[dict[str, object]] = []
    for curve_id, group in curves.groupby("unified_curve_id", sort=False):
        group = group.sort_values("time_days")
        release = group["release_fraction"].to_numpy(float)
        signal = group["release_raw"].to_numpy(float)
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
                "signal_increase_violations": monotonicity_violation_count(signal),
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
                "source_dataset": "chitosan_zeolite_mendeley_kvv2jpjpz7",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "n_conditions": int(formulations["experimental_panel"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "release_measure_type": formulations["release_measure_type"].iloc[0],
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, formulations: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_chitosan_zeolite_mendeley",
        "",
        "Standardized external source from Mendeley dataset `kvv2jpjpz7`, using the raw release workbook",
        "as a time-resolved absorbance proxy corpus rather than as a cumulative-release corpus.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- pH conditions: `{int(ds['n_conditions'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        "",
        "## Important Caveat",
        "",
        "- The workbook reports time-resolved absorbance values under different pH conditions.",
        "- This output preserves those measurements as a release proxy and does not claim they are directly cumulative release percentages.",
        "- For that reason, this source is standardized and registered, but not automatically merged into the main cumulative-release candidate pool.",
        "",
        "## Exported Curves",
        "",
    ]
    for _, row in formulations.sort_values(["experimental_panel", "source_group"]).iterrows():
        lines.append(f"- `{row['source_group']}` under `{row['experimental_panel']}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xlsx_path = args.repo_root / "data" / "external" / "kvv2jpjpz7_mendeley" / "Release raw data.xlsx"
    curves, formulations = load_release_blocks(xlsx_path)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_chitosan_zeolite_mendeley",
                "source_dataset": "chitosan_zeolite_mendeley_kvv2jpjpz7",
                "mendeley_dataset_id": "kvv2jpjpz7",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "release_measure_type": "time_resolved_absorbance_proxy",
                "eligible_for_main_cumulative_pool": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, formulations, args.outdir / "summary.md")

    print(f"[external-chitosan-zeolite-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-chitosan-zeolite-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
