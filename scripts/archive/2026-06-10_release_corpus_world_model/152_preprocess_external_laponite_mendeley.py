from __future__ import annotations

"""
152_preprocess_external_laponite_mendeley.py

Consume:
- data/external/mtds4ckns5_mendeley/Organic-inorganic hybrid based on Laponite as a pl/Release_pH5andpH7.opju

Produce:
- outputs/152_external_laponite_mendeley/curves_long.csv
- outputs/152_external_laponite_mendeley/formulations.csv
- outputs/152_external_laponite_mendeley/curve_quality_summary.csv
- outputs/152_external_laponite_mendeley/dataset_summary.csv
- outputs/152_external_laponite_mendeley/manifest.json
- outputs/152_external_laponite_mendeley/summary.md

Expected runtime:
- < 20 s
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import win32com.client


SENTINEL_THRESHOLD = 1e-250


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "152_external_laponite_mendeley",
    )
    return parser.parse_args()


def monotonicity_violation_count(values: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(values) < -tol))


def _flatten_com_value(value: object) -> float | None:
    if isinstance(value, tuple):
        if len(value) == 0:
            return None
        value = value[0]
    if value in (None, ""):
        return None
    return float(value)


def _column_values(column) -> list[float]:
    out: list[float] = []
    for item in list(column.GetData(0)):
        value = _flatten_com_value(item)
        if value is None:
            continue
        out.append(value)
    return out


def _dataset_values(app, dataset_name: str) -> list[float]:
    values: list[float] = []
    for idx in range(1, 256):
        value = float(app.LTVar(f"{dataset_name}[{idx}]"))
        # Hidden Origin datasets terminate with a tiny nonzero sentinel, but real
        # release curves legitimately start at 0.0, so zero itself cannot stop parsing.
        if value != 0.0 and abs(value) < SENTINEL_THRESHOLD:
            break
        values.append(value)
    return values


def extract_laponite_corpus(opju_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    app = win32com.client.Dispatch("Origin.Application")
    app.Visible = 0
    app.NewProject()
    try:
        app.Load(str(opju_path))

        book2 = app.FindWorksheet("[Book2]Sheet1")
        cols = book2.Columns
        nh2_time = _column_values(cols.Item(0))
        nh2_ph7 = _column_values(cols.Item(1))
        nh2_ph7_sd = _column_values(cols.Item(2))
        nh2_ph5 = _column_values(cols.Item(3))
        nh2_ph5_sd = _column_values(cols.Item(4))

        laponite_time = _dataset_values(app, "Book2A_A")
        laponite_ph7 = _dataset_values(app, "Book2A_B")
        laponite_ph7_sd = _dataset_values(app, "Book2A_C")
        laponite_ph5 = _dataset_values(app, "Book2A_D")
        laponite_ph5_sd = _dataset_values(app, "Book2A_E")

        curve_specs = [
            {
                "curve_key": "laponita_nh2_2_ph7",
                "source_group": "laponita NH2 (2)",
                "condition_label": "pH 7",
                "time_vals": nh2_time,
                "release_vals": nh2_ph7,
                "sd_vals": nh2_ph7_sd,
                "label_evidence": "explicit_column_comments",
                "project_object": "Book2",
            },
            {
                "curve_key": "laponita_nh2_2_ph5",
                "source_group": "laponita NH2 (2)",
                "condition_label": "pH 5",
                "time_vals": nh2_time,
                "release_vals": nh2_ph5,
                "sd_vals": nh2_ph5_sd,
                "label_evidence": "explicit_column_comments",
                "project_object": "Book2",
            },
            {
                "curve_key": "laponita_ph7",
                "source_group": "laponita",
                "condition_label": "pH 7",
                "time_vals": laponite_time,
                "release_vals": laponite_ph7,
                "sd_vals": laponite_ph7_sd,
                "label_evidence": "dataset_name_plus_opju_visible_strings",
                "project_object": "Book2A",
            },
            {
                "curve_key": "laponita_ph5",
                "source_group": "laponita",
                "condition_label": "pH 5",
                "time_vals": laponite_time,
                "release_vals": laponite_ph5,
                "sd_vals": laponite_ph5_sd,
                "label_evidence": "dataset_name_plus_opju_visible_strings",
                "project_object": "Book2A",
            },
        ]

        curve_rows: list[dict[str, object]] = []
        formulation_rows: list[dict[str, object]] = []
        for spec in curve_specs:
            n = min(len(spec["time_vals"]), len(spec["release_vals"]), len(spec["sd_vals"]))
            unified_curve_id = f"laponite_mendeley_mtds4ckns5:{spec['curve_key']}"
            for i in range(n):
                curve_rows.append(
                    {
                        "record_id": f"{unified_curve_id}:{spec['time_vals'][i]:.8f}",
                        "unified_curve_id": unified_curve_id,
                        "source_dataset": "laponite_mendeley_mtds4ckns5",
                        "source_curve_id": spec["curve_key"],
                        "time_raw": float(spec["time_vals"][i]),
                        "time_unit": "hour",
                        "time_days": float(spec["time_vals"][i]) / 24.0,
                        "release_raw": float(spec["release_vals"][i]),
                        "release_unit": "percent_released",
                        "release_fraction": float(spec["release_vals"][i]) / 100.0,
                        "release_percent": float(spec["release_vals"][i]),
                        "curve_series": spec["condition_label"],
                        "curve_level": "formulation_mean",
                        "source_group": spec["source_group"],
                        "payload_name": "5-FU",
                        "release_sd_percent": float(spec["sd_vals"][i]),
                        "source_label_evidence": spec["label_evidence"],
                        "origin_project_object": spec["project_object"],
                    }
                )
            formulation_rows.append(
                {
                    "unified_curve_id": unified_curve_id,
                    "source_dataset": "laponite_mendeley_mtds4ckns5",
                    "source_curve_id": spec["curve_key"],
                    "source_group": spec["source_group"],
                    "source_has_explicit_group": spec["label_evidence"] == "explicit_column_comments",
                    "polymer_family": "laponite hybrid",
                    "payload_name": "5-FU",
                    "experimental_panel": spec["condition_label"],
                    "curve_level": "formulation_mean",
                    "normalization_basis": "reported_percent_released",
                    "release_measure_type": "cumulative_percent_released",
                    "measurement_assay": "origin_project_extraction",
                    "source_label_evidence": spec["label_evidence"],
                    "origin_project_object": spec["project_object"],
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
                "source_label_evidence": meta["source_label_evidence"],
                "origin_project_object": meta["origin_project_object"],
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    explicit_fraction = (
        curve_quality["source_label_evidence"].eq("explicit_column_comments").mean()
        if len(curve_quality)
        else math.nan
    )
    return pd.DataFrame(
        [
            {
                "source_dataset": "laponite_mendeley_mtds4ckns5",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "fraction_monotone_non_decreasing": float(curve_quality["is_monotone_non_decreasing"].mean()),
                "fraction_explicit_labels": float(explicit_fraction),
            }
        ]
    )


def write_summary(dataset_summary: pd.DataFrame, out_path: Path) -> None:
    ds = dataset_summary.iloc[0]
    lines = [
        "# external_laponite_mendeley",
        "",
        "Cumulative-release extraction from Mendeley dataset `mtds4ckns5`, using direct COM access to",
        "the Origin project `Release_pH5andpH7.opju`.",
        "",
        f"- curves: `{int(ds['n_curves'])}`",
        f"- total points: `{int(ds['n_points_total'])}`",
        f"- formulations: `{int(ds['n_formulations'])}`",
        f"- median points/curve: `{ds['median_points_per_curve']:.1f}`",
        f"- median duration days: `{ds['median_duration_days']:.4f}`",
        f"- monotone-clean fraction: `{ds['fraction_monotone_non_decreasing']:.3f}`",
        f"- explicit-label fraction: `{ds['fraction_explicit_labels']:.3f}`",
        "",
        "## Important Caveat",
        "",
        "- The `laponita NH2 (2)` series are explicitly labeled by worksheet column comments.",
        "- The plain `laponita` series are extracted from hidden `Book2A_*` datasets and labeled by",
        "  combining dataset names with visible-string evidence embedded in the Origin project.",
        "- This corpus is therefore usable as cumulative release data, but it stays in the caveat layer.",
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
    curves, formulations = extract_laponite_corpus(opju_path)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_laponite_mendeley",
                "source_dataset": "laponite_mendeley_mtds4ckns5",
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

    print(f"[external-laponite-mendeley] wrote outputs to {args.outdir}")
    print(
        "[external-laponite-mendeley] "
        f"curves={curves['unified_curve_id'].nunique()} "
        f"points={len(curves)} formulations={formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
