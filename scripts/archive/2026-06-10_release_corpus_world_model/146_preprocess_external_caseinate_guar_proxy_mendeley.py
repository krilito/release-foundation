from __future__ import annotations

"""
146_preprocess_external_caseinate_guar_proxy_mendeley.py

Consume:
- data/external/9md8g25gnx_mendeley/Active caseinateguar gum films incorporated with gallic acid physicochemical properties and release kinetics/Journal of Food Eng_Khan MR.xlsx

Produce:
- outputs/146_external_caseinate_guar_proxy_mendeley/curves_long.csv
- outputs/146_external_caseinate_guar_proxy_mendeley/formulations.csv
- outputs/146_external_caseinate_guar_proxy_mendeley/curve_quality_summary.csv
- outputs/146_external_caseinate_guar_proxy_mendeley/dataset_summary.csv
- outputs/146_external_caseinate_guar_proxy_mendeley/manifest.json
- outputs/146_external_caseinate_guar_proxy_mendeley/summary.md

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
        default=repo_root / "outputs" / "146_external_caseinate_guar_proxy_mendeley",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(values: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(values) < -tol))


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("+", "_")
        .replace("__", "_")
        .strip("_")
    )


def load_release_sheet(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name="Release kinetics")
    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []

    for (sample_film, replication), group in df.groupby(["Sample film ", "Replication"], sort=False):
        group = group.sort_values("Time (h)")
        max_signal = float(group["Concentration (ug/ml)"].max())
        curve_key = f"{slugify(str(sample_film))}_rep{int(replication)}"
        unified_curve_id = f"caseinate_guar_proxy_mendeley_9md8g25gnx:{curve_key}"

        for _, row in group.iterrows():
            signal = float(row["Concentration (ug/ml)"])
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{float(row['Time (h)']):.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "caseinate_guar_proxy_mendeley_9md8g25gnx",
                    "source_curve_id": curve_key,
                    "time_raw": float(row["Time (h)"]),
                    "time_unit": "hour",
                    "time_days": float(row["Time (h)"]) / 24.0,
                    "release_raw": signal,
                    "release_unit": "concentration_proxy_ug_ml",
                    "release_fraction": signal / max_signal if max_signal > 0 else np.nan,
                    "release_percent": signal / max_signal * 100.0 if max_signal > 0 else np.nan,
                    "curve_series": "release_kinetics_concentration_proxy",
                    "curve_level": "replicate",
                    "source_group": str(sample_film),
                    "payload_name": "gallic acid",
                    "replication": int(replication),
                }
            )

        formulation_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "caseinate_guar_proxy_mendeley_9md8g25gnx",
                "source_curve_id": curve_key,
                "source_group": str(sample_film),
                "source_has_explicit_group": True,
                "polymer_family": "caseinate-guar gum film",
                "payload_name": "gallic acid",
                "experimental_panel": "release_kinetics",
                "curve_level": "replicate",
                "normalization_basis": "concentration_divided_by_curve_max",
                "release_measure_type": "time_resolved_concentration_proxy",
                "measurement_assay": "concentration_ug_ml",
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

    return pd.DataFrame(curve_rows), pd.DataFrame(formulation_rows)


def build_curve_quality_summary(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    form_index = formulations.set_index("unified_curve_id")
    rows: list[dict[str, object]] = []
    for curve_id, group in curves.groupby("unified_curve_id", sort=False):
        group = group.sort_values("time_days")
        proxy = group["release_fraction"].to_numpy(float)
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
                "measurement_assay": meta["measurement_assay"],
                "n_points": int(len(group)),
                "time_min_days": float(group["time_days"].min()),
                "time_max_days": float(group["time_days"].max()),
                "duration_days": float(group["time_days"].max() - group["time_days"].min()),
                "final_release_fraction": float(group["release_fraction"].iloc[-1]),
                "max_release_fraction": float(group["release_fraction"].max()),
                "monotonicity_violations": monotonicity_violation_count(proxy),
                "signal_increase_violations": monotonicity_violation_count(signal),
                "is_monotone_non_decreasing": monotonicity_violation_count(proxy) == 0,
                "has_release_gt_1p0": bool((group["release_fraction"] > 1.0 + 1e-6).any()),
                "release_measure_type": meta["release_measure_type"],
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "caseinate_guar_proxy_mendeley_9md8g25gnx",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "n_sample_groups": int(formulations["source_group"].nunique()),
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
        "# external_caseinate_guar_proxy_mendeley",
        "",
        "Standardized external source from Mendeley dataset `9md8g25gnx`, using the release-kinetics sheet",
        "as a concentration-proxy corpus rather than as a cumulative-release corpus.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- sample groups: `{int(ds['n_sample_groups'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        "",
        "## Important Caveat",
        "",
        "- The workbook reports time-resolved concentration values (`ug/ml`), not an explicit cumulative release fraction.",
        "- This output preserves the trajectories as a release proxy using within-curve max normalization.",
        "- For that reason, the source is standardized and registered, but not merged into the main cumulative-release candidate pool.",
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
        / "9md8g25gnx_mendeley"
        / "Active caseinateguar gum films incorporated with gallic acid physicochemical properties and release kinetics"
        / "Journal of Food Eng_Khan MR.xlsx"
    )
    curves, formulations = load_release_sheet(xlsx_path)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_caseinate_guar_proxy_mendeley",
                "source_dataset": "caseinate_guar_proxy_mendeley_9md8g25gnx",
                "mendeley_dataset_id": "9md8g25gnx",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "release_measure_type": "time_resolved_concentration_proxy",
                "eligible_for_main_cumulative_pool": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, formulations, args.outdir / "summary.md")

    print(f"[external-caseinate-guar-proxy-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-caseinate-guar-proxy-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
