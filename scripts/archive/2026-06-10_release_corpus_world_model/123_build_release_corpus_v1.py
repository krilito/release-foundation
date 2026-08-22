from __future__ import annotations

"""
123_build_release_corpus_v1.py

Consume:
- data/curves_long.csv
- data/formulations.csv
- optional external 321-curve xlsx

Produce:
- outputs/123_release_corpus_v1/curves_long.csv
- outputs/123_release_corpus_v1/formulations.csv
- outputs/123_release_corpus_v1/curve_quality_summary.csv
- outputs/123_release_corpus_v1/dataset_summary.csv
- outputs/123_release_corpus_v1/manifest.json
- outputs/123_release_corpus_v1/summary.md

Expected runtime:
- < 30 s on the current local corpus
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CROSS_DOI_XLSX = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--cross-doi-xlsx",
        type=Path,
        default=DEFAULT_CROSS_DOI_XLSX,
        help="optional local 321-curve xlsx",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "123_release_corpus_v1",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


def load_internal_181(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(repo_root / "data" / "curves_long.csv")
    formulations = pd.read_csv(repo_root / "data" / "formulations.csv")

    curves = curves.rename(columns={"curve_id": "source_curve_id"}).copy()
    curves["source_dataset"] = "internal181"
    curves["source_curve_id"] = curves["source_curve_id"].astype(str)
    curves["time_raw"] = pd.to_numeric(curves["time"], errors="coerce")
    curves["release_raw"] = pd.to_numeric(curves["release"], errors="coerce")
    curves = curves.dropna(subset=["source_curve_id", "time_raw", "release_raw"]).copy()
    curves["time_days"] = curves["time_raw"]
    curves["release_fraction"] = curves["release_raw"].clip(lower=0.0)
    curves["release_percent"] = curves["release_fraction"] * 100.0
    curves["time_unit"] = "day"
    curves["release_unit"] = "fraction"
    curves["unified_curve_id"] = curves["source_dataset"] + ":" + curves["source_curve_id"]
    curves["record_id"] = (
        curves["source_dataset"]
        + ":"
        + curves["source_curve_id"]
        + ":"
        + curves["time_days"].round(8).astype(str)
    )
    curves = curves[
        [
            "record_id",
            "unified_curve_id",
            "source_dataset",
            "source_curve_id",
            "time_raw",
            "time_unit",
            "time_days",
            "release_raw",
            "release_unit",
            "release_fraction",
            "release_percent",
        ]
    ].copy()

    formulations = formulations.rename(columns={"curve_id": "source_curve_id", "DP_Group": "source_group"}).copy()
    formulations["source_dataset"] = "internal181"
    formulations["source_curve_id"] = formulations["source_curve_id"].astype(str)
    formulations["unified_curve_id"] = formulations["source_dataset"] + ":" + formulations["source_curve_id"]
    formulations["polymer_family"] = "PLGA-like"
    formulations["Particle_Size"] = np.nan
    formulations["EE"] = np.nan
    formulations["DLC_percent"] = formulations["DLC"] * 100.0
    formulations["Polymer_MW_raw_unit"] = "Da"
    formulations["source_has_explicit_group"] = True
    ordered_cols = [
        "unified_curve_id",
        "source_dataset",
        "source_curve_id",
        "source_group",
        "source_has_explicit_group",
        "polymer_family",
        "LA/GA",
        "Polymer_MW",
        "Polymer_MW_raw_unit",
        "CL Ratio",
        "Drug_Tm",
        "Drug_Pka",
        "Initial D/M ratio",
        "DLC",
        "DLC_percent",
        "EE",
        "Particle_Size",
        "SA-V",
        "SE",
        "Drug_Mw",
        "Drug_TPSA",
        "Drug_NHA",
        "Drug_LogP",
    ]
    return curves, formulations[ordered_cols].copy()


def load_cross_doi_321(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(xlsx_path).rename(
        columns={
            "Formulation Index": "source_curve_id",
            "Drug MW": "Drug_Mw",
            "Drug TPSA": "Drug_TPSA",
            "Drug LogP": "Drug_LogP",
            "Polymer MW": "Polymer_MW_kDa",
            "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
            "Particle Size": "Particle_Size",
            "Drug Loading Capacity": "DLC_percent",
            "Drug Encapsulation Efficiency": "EE",
            "Solubility Enhancer Concentration": "SE",
            "Time": "time",
            "Release": "release",
        }
    )
    raw["source_dataset"] = "cross321"
    raw["source_curve_id"] = raw["source_curve_id"].astype(str)
    raw["time_raw"] = pd.to_numeric(raw["time"], errors="coerce")
    raw["release_raw"] = pd.to_numeric(raw["release"], errors="coerce")
    raw = raw.dropna(subset=["source_curve_id", "time_raw", "release_raw"]).copy()

    curves = (
        raw.groupby(["source_dataset", "source_curve_id", "time_raw"], as_index=False, sort=False)["release_raw"]
        .mean()
        .copy()
    )
    curves["time_days"] = curves["time_raw"]
    curves["release_fraction"] = curves["release_raw"].clip(lower=0.0)
    curves["release_percent"] = curves["release_fraction"] * 100.0
    curves["time_unit"] = "day"
    curves["release_unit"] = "fraction"
    curves["unified_curve_id"] = curves["source_dataset"] + ":" + curves["source_curve_id"]
    curves["record_id"] = (
        curves["source_dataset"]
        + ":"
        + curves["source_curve_id"]
        + ":"
        + curves["time_days"].round(8).astype(str)
    )
    curves = curves[
        [
            "record_id",
            "unified_curve_id",
            "source_dataset",
            "source_curve_id",
            "time_raw",
            "time_unit",
            "time_days",
            "release_raw",
            "release_unit",
            "release_fraction",
            "release_percent",
        ]
    ].copy()

    formulations = raw.drop_duplicates("source_curve_id").copy()
    formulations["source_group"] = pd.NA
    formulations["source_has_explicit_group"] = False
    formulations["polymer_family"] = "PLGA microparticle"
    formulations["Polymer_MW"] = pd.to_numeric(formulations["Polymer_MW_kDa"], errors="coerce") * 1000.0
    formulations["Polymer_MW_raw_unit"] = "kDa"
    formulations["DLC"] = pd.to_numeric(formulations["DLC_percent"], errors="coerce") / 100.0
    formulations["CL Ratio"] = np.nan
    formulations["Drug_Tm"] = np.nan
    formulations["Drug_Pka"] = np.nan
    formulations["SA-V"] = np.nan
    formulations["Drug_NHA"] = np.nan
    ordered_cols = [
        "unified_curve_id",
        "source_dataset",
        "source_curve_id",
        "source_group",
        "source_has_explicit_group",
        "polymer_family",
        "LA/GA",
        "Polymer_MW",
        "Polymer_MW_raw_unit",
        "CL Ratio",
        "Drug_Tm",
        "Drug_Pka",
        "Initial D/M ratio",
        "DLC",
        "DLC_percent",
        "EE",
        "Particle_Size",
        "SA-V",
        "SE",
        "Drug_Mw",
        "Drug_TPSA",
        "Drug_NHA",
        "Drug_LogP",
    ]
    formulations["unified_curve_id"] = formulations["source_dataset"] + ":" + formulations["source_curve_id"]
    return curves, formulations[ordered_cols].copy()


def build_curve_quality_summary(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    form_index = formulations.set_index("unified_curve_id")
    rows: list[dict[str, object]] = []
    for unified_curve_id, sub in curves.groupby("unified_curve_id", sort=True):
        sub = sub.sort_values("time_days").reset_index(drop=True)
        release = sub["release_fraction"].to_numpy(dtype=float)
        time = sub["time_days"].to_numpy(dtype=float)
        source_dataset = str(sub["source_dataset"].iloc[0])
        source_curve_id = str(sub["source_curve_id"].iloc[0])
        row: dict[str, object] = {
            "unified_curve_id": unified_curve_id,
            "source_dataset": source_dataset,
            "source_curve_id": source_curve_id,
            "n_points": int(len(sub)),
            "time_start_days": float(time[0]),
            "time_end_days": float(time[-1]),
            "duration_days": float(time[-1] - time[0]),
            "release_final": float(release[-1]),
            "release_max": float(np.nanmax(release)),
            "release_min": float(np.nanmin(release)),
            "monotonicity_violations": monotonicity_violation_count(release),
            "has_duplicate_times": bool(sub["time_days"].duplicated().any()),
            "has_release_gt_1": bool(np.nanmax(release) > 1.0 + 1e-6),
            "has_release_gt_105pct": bool(np.nanmax(release) > 1.05 + 1e-6),
            "has_negative_release_raw": bool(np.nanmin(sub["release_raw"].to_numpy(dtype=float)) < -1e-6),
            "time_is_sorted": bool(np.all(np.diff(time) >= -1e-12)),
            "has_formulation_row": bool(unified_curve_id in form_index.index),
        }
        if unified_curve_id in form_index.index:
            meta = form_index.loc[unified_curve_id]
            row["source_group"] = meta.get("source_group", pd.NA)
            row["polymer_family"] = meta.get("polymer_family", pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    form_counts = formulations.groupby("source_dataset")["unified_curve_id"].nunique().rename("n_formulations")
    rows: list[dict[str, object]] = []
    for source_dataset, sub in curve_quality.groupby("source_dataset", sort=True):
        sub = sub.copy()
        rows.append(
            {
                "source_dataset": source_dataset,
                "n_curves": int(sub["unified_curve_id"].nunique()),
                "n_formulations": int(form_counts.get(source_dataset, 0)),
                "n_points_total": int(sub["n_points"].sum()),
                "median_points_per_curve": float(sub["n_points"].median()),
                "median_duration_days": float(sub["duration_days"].median()),
                "median_release_final": float(sub["release_final"].median()),
                "median_release_max": float(sub["release_max"].median()),
                "monotone_clean_fraction": float((sub["monotonicity_violations"] == 0).mean()),
                "release_gt_1_fraction": float(sub["has_release_gt_1"].mean()),
                "explicit_group_fraction": float(sub["source_group"].notna().mean()) if "source_group" in sub.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("source_dataset").reset_index(drop=True)


def build_manifest(curves: pd.DataFrame, formulations: pd.DataFrame, curve_quality: pd.DataFrame) -> dict[str, object]:
    return {
        "corpus_name": "release_corpus_v1",
        "time_unit": "day",
        "release_unit": "fraction",
        "n_total_curves": int(curve_quality["unified_curve_id"].nunique()),
        "n_total_points": int(len(curves)),
        "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
        "sources": sorted(curves["source_dataset"].unique().tolist()),
        "curve_columns": curves.columns.tolist(),
        "formulation_columns": formulations.columns.tolist(),
        "quality_columns": curve_quality.columns.tolist(),
    }


def write_summary_markdown(
    dataset_summary: pd.DataFrame,
    curve_quality: pd.DataFrame,
    manifest: dict[str, object],
    out_path: Path,
) -> None:
    lines = [
        "# release_corpus_v1",
        "",
        "Unified preprocessing output for the currently available PLGA release corpora.",
        "",
        f"- total curves: `{manifest['n_total_curves']}`",
        f"- total observations: `{manifest['n_total_points']}`",
        f"- total formulations: `{manifest['n_total_formulations']}`",
        f"- sources: `{', '.join(manifest['sources'])}`",
        "",
        "## Dataset Summary",
        "",
    ]
    for _, row in dataset_summary.iterrows():
        lines.extend(
            [
                f"### {row['source_dataset']}",
                "",
                f"- curves: `{int(row['n_curves'])}`",
                f"- formulations: `{int(row['n_formulations'])}`",
                f"- total points: `{int(row['n_points_total'])}`",
                f"- median points/curve: `{row['median_points_per_curve']:.1f}`",
                f"- median duration days: `{row['median_duration_days']:.2f}`",
                f"- monotone-clean fraction: `{row['monotone_clean_fraction']:.3f}`",
                f"- release > 1.0 fraction: `{row['release_gt_1_fraction']:.3f}`",
                f"- explicit-group fraction: `{row['explicit_group_fraction']:.3f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-Corpus Caveats",
            "",
            "- `internal181` includes an explicit `source_group`; `cross321` does not.",
            "- `Polymer_MW` is aligned to `Da`; the 321 source is converted from kDa.",
            "- `DLC` is aligned to fraction; the 321 source is converted from percent.",
            "- Release is preserved as fraction and percent; no upper clipping is applied.",
            "",
            "## Highest-Final-Release Curves",
            "",
        ]
    )
    top = curve_quality.sort_values("release_max", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['unified_curve_id']}`: max_release={row['release_max']:.4f}, "
            f"final_release={row['release_final']:.4f}, monotonicity_violations={int(row['monotonicity_violations'])}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    curves_parts: list[pd.DataFrame] = []
    form_parts: list[pd.DataFrame] = []

    internal_curves, internal_forms = load_internal_181(args.repo_root)
    curves_parts.append(internal_curves)
    form_parts.append(internal_forms)

    if args.cross_doi_xlsx.exists():
        cross_curves, cross_forms = load_cross_doi_321(args.cross_doi_xlsx)
        curves_parts.append(cross_curves)
        form_parts.append(cross_forms)
    else:
        print(f"[release-corpus-v1] cross-DOI xlsx not found, skipping: {args.cross_doi_xlsx}")

    curves = pd.concat(curves_parts, ignore_index=True).sort_values(
        ["source_dataset", "source_curve_id", "time_days"], kind="stable"
    ).reset_index(drop=True)
    formulations = pd.concat(form_parts, ignore_index=True).drop_duplicates("unified_curve_id").reset_index(drop=True)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)
    manifest = build_manifest(curves, formulations, curve_quality)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_markdown(dataset_summary, curve_quality, manifest, args.outdir / "summary.md")

    print(f"[release-corpus-v1] wrote outputs to {args.outdir}")
    print(f"[release-corpus-v1] total curves={manifest['n_total_curves']} total points={manifest['n_total_points']}")


if __name__ == "__main__":
    main()
