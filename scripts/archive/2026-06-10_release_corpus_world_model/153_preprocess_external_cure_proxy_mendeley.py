from __future__ import annotations

"""
153_preprocess_external_cure_proxy_mendeley.py

Consume:
- data/external/jk7j38n2rc_mendeley/extracted/Drug release kinetics analysis of CURE/Drug Release.xlsx

Produce:
- outputs/153_external_cure_proxy_mendeley/curves_long.csv
- outputs/153_external_cure_proxy_mendeley/formulations.csv
- outputs/153_external_cure_proxy_mendeley/curve_quality_summary.csv
- outputs/153_external_cure_proxy_mendeley/dataset_summary.csv
- outputs/153_external_cure_proxy_mendeley/manifest.json
- outputs/153_external_cure_proxy_mendeley/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_SPECS = [
    {"condition_label": "PBS", "value_cols": [2, 3, 4], "media_pH": 7.4, "gsh_mM": 0.0},
    {"condition_label": "0.5mM GSH", "value_cols": [5, 6, 7], "media_pH": 7.4, "gsh_mM": 0.5},
    {"condition_label": "pH 5.5", "value_cols": [8, 9, 10], "media_pH": 5.5, "gsh_mM": 0.0},
    {"condition_label": "pH 5.5 + 0.5mM GSH", "value_cols": [11, 12, 13], "media_pH": 5.5, "gsh_mM": 0.5},
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "153_external_cure_proxy_mendeley",
    )
    return parser.parse_args()


def monotonicity_violation_count(values: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(values) < -tol))


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("+", "_plus_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )


def load_release_workbook(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1", header=None)
    time_values = pd.to_numeric(df.iloc[1:, 1], errors="coerce")

    curve_rows: list[dict[str, object]] = []
    formulation_rows: list[dict[str, object]] = []
    for spec in CONDITION_SPECS:
        raw_block = df.iloc[1:, spec["value_cols"]].apply(pd.to_numeric, errors="coerce")
        valid = time_values.notna() & raw_block.notna().all(axis=1)
        time_h = time_values[valid].astype(float).to_numpy()
        block = raw_block.loc[valid].astype(float)
        signal_mean = block.mean(axis=1).to_numpy()
        signal_std = block.std(axis=1, ddof=1).to_numpy()

        baseline = float(signal_mean[0]) if len(signal_mean) else 0.0
        signal_corrected = np.clip(signal_mean - baseline, a_min=0.0, a_max=None)
        scale = float(signal_corrected.max()) if len(signal_corrected) else 0.0
        signal_fraction = signal_corrected / scale if scale > 0 else np.zeros_like(signal_corrected)

        curve_key = f"cure_{slugify(spec['condition_label'])}"
        unified_curve_id = f"cure_proxy_mendeley_jk7j38n2rc:{curve_key}"
        for i in range(len(time_h)):
            curve_rows.append(
                {
                    "record_id": f"{unified_curve_id}:{time_h[i]:.8f}",
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "cure_proxy_mendeley_jk7j38n2rc",
                    "source_curve_id": curve_key,
                    "time_raw": float(time_h[i]),
                    "time_unit": "hour",
                    "time_days": float(time_h[i]) / 24.0,
                    "release_raw": float(signal_mean[i]),
                    "release_unit": "hplc_release_level_proxy",
                    "release_fraction": float(signal_fraction[i]),
                    "release_percent": float(signal_fraction[i] * 100.0),
                    "curve_series": spec["condition_label"],
                    "curve_level": "condition_mean_from_triplicates",
                    "source_group": "NanoCURE",
                    "payload_name": "BTZ",
                    "signal_std": float(signal_std[i]),
                    "signal_baseline": baseline,
                    "signal_corrected": float(signal_corrected[i]),
                    "n_replicates": 3,
                }
            )
        formulation_rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": "cure_proxy_mendeley_jk7j38n2rc",
                "source_curve_id": curve_key,
                "source_group": "NanoCURE",
                "source_has_explicit_group": True,
                "polymer_family": "nanoparticle",
                "payload_name": "BTZ",
                "experimental_panel": spec["condition_label"],
                "curve_level": "condition_mean_from_triplicates",
                "normalization_basis": "baseline_subtracted_signal_divided_by_curve_max",
                "release_measure_type": "time_resolved_hplc_release_proxy",
                "media_pH": spec["media_pH"],
                "gsh_mM": spec["gsh_mM"],
                "measurement_assay": "HPLC released level",
                "dialysis_bag_mwco": 3000,
                "source_time_grid_hours": "0,0.5,1,2,4,8,12,24",
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
                "source_dataset": "cure_proxy_mendeley_jk7j38n2rc",
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


def write_summary(dataset_summary: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_cure_proxy_mendeley",
        "",
        "Standardized external source from Mendeley dataset `jk7j38n2rc`, treated as a time-resolved HPLC",
        "release proxy rather than as a cumulative percent-release corpus.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- conditions: `{int(ds['n_conditions'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        "",
        "## Important Caveat",
        "",
        "- The workbook reports released BTZ levels measured by HPLC, not a pre-normalized cumulative percent release.",
        "- This preprocessing baseline-subtracts each condition and rescales by that condition's observed maximum.",
        "- The resulting curves belong in the proxy layer, not in the main cumulative pool.",
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
        / "jk7j38n2rc_mendeley"
        / "extracted"
        / "Drug release kinetics analysis of CURE"
        / "Drug Release.xlsx"
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
                "corpus_name": "external_cure_proxy_mendeley",
                "source_dataset": "cure_proxy_mendeley_jk7j38n2rc",
                "mendeley_dataset_id": "jk7j38n2rc",
                "n_total_curves": int(curves["unified_curve_id"].nunique()),
                "n_total_points": int(len(curves)),
                "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
                "curve_levels": sorted(curves["curve_level"].unique().tolist()),
                "release_measure_type": "time_resolved_hplc_release_proxy",
                "eligible_for_main_cumulative_pool": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(dataset_summary, args.outdir / "summary.md")

    print(f"[external-cure-proxy-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-cure-proxy-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
