from __future__ import annotations

"""
151_preprocess_external_laponite_nh2_mendeley.py

Consume:
- data/external/mtds4ckns5_mendeley/Organic-inorganic hybrid based on Laponite as a pl/Release_pH5andpH7.opju

Produce:
- outputs/151_external_laponite_nh2_mendeley/curves_long.csv
- outputs/151_external_laponite_nh2_mendeley/formulations.csv
- outputs/151_external_laponite_nh2_mendeley/curve_quality_summary.csv
- outputs/151_external_laponite_nh2_mendeley/dataset_summary.csv
- outputs/151_external_laponite_nh2_mendeley/manifest.json
- outputs/151_external_laponite_nh2_mendeley/summary.md

Expected runtime:
- < 20 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import win32com.client


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "151_external_laponite_nh2_mendeley",
    )
    return parser.parse_args()


def monotonicity_violation_count(values: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(values) < -tol))


def _column_values(column) -> list[float]:
    values = list(column.GetData(0))
    out: list[float] = []
    for item in values:
        if isinstance(item, tuple):
            if len(item) == 0:
                continue
            item = item[0]
        if item is None or item == "":
            continue
        out.append(float(item))
    return out


def extract_book2(opju_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    app = win32com.client.Dispatch("Origin.Application")
    app.Visible = 0
    app.NewProject()
    try:
        app.Load(str(opju_path))
        ws = app.FindWorksheet("[Book2]Sheet1")
        cols = ws.Columns

        time_vals = _column_values(cols.Item(0))
        ph7_vals = _column_values(cols.Item(1))
        ph7_sd = _column_values(cols.Item(2))
        ph5_vals = _column_values(cols.Item(3))
        ph5_sd = _column_values(cols.Item(4))

        curve_specs = [
            {"condition_label": "pH 7", "release_vals": ph7_vals, "sd_vals": ph7_sd},
            {"condition_label": "pH 5", "release_vals": ph5_vals, "sd_vals": ph5_sd},
        ]

        curve_rows: list[dict[str, object]] = []
        formulation_rows: list[dict[str, object]] = []
        for spec in curve_specs:
            n = min(len(time_vals), len(spec["release_vals"]), len(spec["sd_vals"]))
            curve_key = f"laponita_nh2_2_{spec['condition_label'].replace(' ', '').lower()}"
            unified_curve_id = f"laponite_nh2_mendeley_mtds4ckns5:{curve_key}"
            for i in range(n):
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{time_vals[i]:.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "laponite_nh2_mendeley_mtds4ckns5",
                        "source_curve_id": curve_key,
                        "time_raw": float(time_vals[i]),
                        "time_unit": "hour",
                        "time_days": float(time_vals[i]) / 24.0,
                        "release_raw": float(spec["release_vals"][i]),
                        "release_unit": "percent_released",
                        "release_fraction": float(spec["release_vals"][i]) / 100.0,
                        "release_percent": float(spec["release_vals"][i]),
                        "curve_series": spec["condition_label"],
                        "curve_level": "formulation_mean",
                        "source_group": "laponita NH2 (2)",
                        "payload_name": "5-FU",
                        "release_sd_percent": float(spec["sd_vals"][i]),
                    }
                )
            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "laponite_nh2_mendeley_mtds4ckns5",
                    "source_curve_id": curve_key,
                    "source_group": "laponita NH2 (2)",
                    "source_has_explicit_group": True,
                    "polymer_family": "laponite NH2 hybrid",
                    "payload_name": "5-FU",
                    "experimental_panel": spec["condition_label"],
                    "curve_level": "formulation_mean",
                    "normalization_basis": "reported_percent_released",
                    "release_measure_type": "cumulative_percent_released",
                    "measurement_assay": "origin_project_extraction",
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
    finally:
        app.Exit()


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
                "source_dataset": "laponite_nh2_mendeley_mtds4ckns5",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_laponite_nh2_mendeley",
        "",
        "Partial cumulative-release extraction from Mendeley dataset `mtds4ckns5`, using direct COM access to",
        "the Origin project `Release_pH5andpH7.opju`.",
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
        "- This extraction currently includes only the confirmed `Book2` worksheet series:",
        "  `laponita NH2 (2)` under `pH 7` and `pH 5`.",
        "- A separate unresolved series remains tracked in `external_laponite_opju_registry` until `Book1` is fully labeled.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    opju_path = (
        args.repo_root
        / "data"
        / "external"
        / "mtds4ckns5_mendeley"
        / "Organic-inorganic hybrid based on Laponite as a pl"
        / "Release_pH5andpH7.opju"
    )
    curves, formulations = extract_book2(opju_path)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_laponite_nh2_mendeley",
                "source_dataset": "laponite_nh2_mendeley_mtds4ckns5",
                "mendeley_dataset_id": "mtds4ckns5",
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

    print(f"[external-laponite-nh2-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-laponite-nh2-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
