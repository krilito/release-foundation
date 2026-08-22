from __future__ import annotations

"""
126_preprocess_external_nanomed_registry.py

Consume:
- data/external/nanomed_IVR_data/repo/nanomed_IVR_data-main/data/unprocessed/backend_data.csv
- data/external/nanomed_IVR_data/repo/nanomed_IVR_data-main/data/unprocessed/media_components.csv
- data/external/nanomed_IVR_data/repo/nanomed_IVR_data-main/data/time_units.csv
- data/external/nanomed_IVR_data/repo/nanomed_IVR_data-main/data/quality_reporting.csv

Produce:
- outputs/126_external_nanomed_registry/ivr_registry.csv
- outputs/126_external_nanomed_registry/media_components_summary.csv
- outputs/126_external_nanomed_registry/manifest.json
- outputs/126_external_nanomed_registry/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "external"
    / "nanomed_IVR_data"
    / "repo"
    / "nanomed_IVR_data-main"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="root of the nanomed_IVR_data repository snapshot",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "126_external_nanomed_registry",
        help="output directory",
    )
    return parser.parse_args()


def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backend = pd.read_csv(root / "data" / "unprocessed" / "backend_data.csv").drop(columns=["Unnamed: 0"])
    media = pd.read_csv(root / "data" / "unprocessed" / "media_components.csv")
    time_units = pd.read_csv(root / "data" / "time_units.csv").rename(columns={"ID": "IVR_ID"})
    quality = pd.read_csv(root / "data" / "quality_reporting.csv").rename(
        columns={
            "Unnamed: 0": "Row_ID",
            "Unnamed: 1": "IVR_ID",
            "Reporting Quality ": "reported_units",
            "Unnamed: 3": "plot_resolution",
            "Performance bias ": "timepoint_count_sufficient",
            "Unnamed: 5": "shape_resembles_profile",
            "Detection bias ": "repeats_and_average_reported",
            "Comments ": "comments",
        }
    )
    quality = quality[pd.to_numeric(quality["IVR_ID"], errors="coerce").notna()].copy()
    quality["IVR_ID"] = quality["IVR_ID"].astype(int)
    return backend, media, time_units, quality


def build_registry(
    backend: pd.DataFrame,
    media: pd.DataFrame,
    time_units: pd.DataFrame,
    quality: pd.DataFrame,
    external_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    media_summary = (
        media.groupby("IVR_ID")["component_name"]
        .agg(lambda x: " | ".join(sorted(set(map(str, x)))))
        .rename("media_components")
        .reset_index()
    )
    media_counts = media.groupby("IVR_ID")["component_name"].nunique().rename("n_media_components").reset_index()
    quality = quality.drop_duplicates("IVR_ID")

    registry = backend.merge(time_units, on="IVR_ID", how="left")
    registry = registry.merge(quality, on="IVR_ID", how="left")
    registry = registry.merge(media_summary, on="IVR_ID", how="left")
    registry = registry.merge(media_counts, on="IVR_ID", how="left")

    registry["source_dataset"] = "nanomed_ivr_registry271"
    registry["unified_curve_id"] = registry["source_dataset"] + ":" + registry["IVR_ID"].astype(str)
    registry["time_unit_normalized"] = registry["Time_units"].astype(str).str.lower().str.strip()
    local_curve_dir = external_root / "data" / "drug_release"
    local_db_path = external_root / "data" / "liposome_IVR.db"
    registry["curve_files_present_locally"] = False
    registry["local_curve_dir_expected"] = str(external_root / "data" / "drug_release")
    registry["curve_file_stub"] = "data/drug_release/<IVR_ID>..csv"
    registry["local_curve_dir_exists"] = local_curve_dir.exists()
    registry["local_sqlite_db_expected"] = str(local_db_path)
    registry["local_sqlite_db_exists"] = local_db_path.exists()
    registry["has_quality_row"] = registry["reported_units"].notna()
    registry["has_media_components"] = registry["media_components"].notna()
    registry["has_smiles"] = registry["SMILES"].notna()
    registry["polymer_family"] = "liposome"
    registry["release_profile_available_now"] = False

    ordered = registry[
        [
            "unified_curve_id",
            "source_dataset",
            "IVR_ID",
            "formulation_ID",
            "release_profile_available_now",
            "curve_files_present_locally",
            "local_curve_dir_expected",
            "curve_file_stub",
            "local_curve_dir_exists",
            "local_sqlite_db_expected",
            "local_sqlite_db_exists",
            "release_method",
            "media_pH",
            "media_temp_oC",
            "media_volume_mL",
            "media_components",
            "n_media_components",
            "drug_loading",
            "structure_type",
            "particle_size_nm",
            "PDI",
            "zeta_potential",
            "API_ID",
            "API_name",
            "SMILES",
            "MolWt",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "MolLogP",
            "Time_units",
            "time_unit_normalized",
            "has_quality_row",
            "reported_units",
            "plot_resolution",
            "timepoint_count_sufficient",
            "shape_resembles_profile",
            "repeats_and_average_reported",
            "comments",
            "has_media_components",
            "has_smiles",
            "polymer_family",
        ]
    ].copy()

    return ordered, media_summary


def build_manifest(registry: pd.DataFrame) -> dict[str, object]:
    return {
        "corpus_name": "external_nanomed_registry",
        "source_dataset": "nanomed_ivr_registry271",
        "n_ivr_entries": int(len(registry)),
        "n_quality_rows": int(registry["has_quality_row"].sum()),
        "n_media_rows": int(registry["has_media_components"].sum()),
        "n_local_curve_files": int(registry["curve_files_present_locally"].sum()),
        "local_curve_dir_exists": bool(registry["local_curve_dir_exists"].iloc[0]),
        "local_sqlite_db_exists": bool(registry["local_sqlite_db_exists"].iloc[0]),
        "columns": registry.columns.tolist(),
    }


def write_summary(registry: pd.DataFrame, manifest: dict[str, object], out_path: Path) -> None:
    top_methods = registry["release_method"].fillna("UNK").value_counts().head(10)
    top_apis = registry["API_name"].fillna("UNK").value_counts().head(10)
    time_units = registry["time_unit_normalized"].fillna("UNK").value_counts()

    lines = [
        "# external_nanomed_registry",
        "",
        "Standardized metadata registry for the local nanomed_IVR_data snapshot.",
        "",
        f"- IVR entries: `{manifest['n_ivr_entries']}`",
        f"- local curve files present: `{manifest['n_local_curve_files']}`",
        f"- quality rows present: `{manifest['n_quality_rows']}`",
        f"- media component rows present: `{manifest['n_media_rows']}`",
        "",
        "## Important Caveat",
        "",
        f"- The local snapshot includes `data/drug_release/`: `{manifest['local_curve_dir_exists']}`",
        f"- The local snapshot includes `data/liposome_IVR.db`: `{manifest['local_sqlite_db_exists']}`",
        "- This means the current repository snapshot is metadata-only for now, even though the upstream project expects both the digitised release curves and the SQLite database.",
        "- It is ready for future curve attachment once the raw digitised release files and database are retrieved.",
        "",
        "## Time Units",
        "",
    ]
    for unit, count in time_units.items():
        lines.append(f"- `{unit}`: `{int(count)}`")
    lines.extend(["", "## Top Release Methods", ""])
    for name, count in top_methods.items():
        lines.append(f"- `{name}`: `{int(count)}`")
    lines.extend(["", "## Top APIs", ""])
    for name, count in top_apis.items():
        lines.append(f"- `{name}`: `{int(count)}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    backend, media, time_units, quality = load_tables(args.external_root)
    registry, media_summary = build_registry(backend, media, time_units, quality, args.external_root)
    manifest = build_manifest(registry)

    registry.to_csv(args.outdir / "ivr_registry.csv", index=False)
    media_summary.to_csv(args.outdir / "media_components_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(registry, manifest, args.outdir / "summary.md")

    print(f"[external-nanomed-registry] wrote outputs to {args.outdir}")
    print(
        f"[external-nanomed-registry] ivr_entries={manifest['n_ivr_entries']} "
        f"quality_rows={manifest['n_quality_rows']} local_curve_files={manifest['n_local_curve_files']}"
    )


if __name__ == "__main__":
    main()
