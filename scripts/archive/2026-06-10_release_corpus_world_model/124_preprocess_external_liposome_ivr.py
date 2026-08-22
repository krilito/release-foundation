from __future__ import annotations

"""
124_preprocess_external_liposome_ivr.py

Consume:
- data/external/accelerated_IVR/repo/accelerated_IVR-main/results/fitting/drug_release_exp.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/data/unprocessed/backend_data.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/results/clustering/3_PCA_KMC.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/data/quality_reporting.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/data/time_units.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/results/fitting/MAE_df.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/results/fitting/aic_df.csv
- data/external/accelerated_IVR/repo/accelerated_IVR-main/results/fitting/F2_df.csv

Produce:
- outputs/124_external_liposome_ivr/curves_long.csv
- outputs/124_external_liposome_ivr/formulations.csv
- outputs/124_external_liposome_ivr/curve_quality_summary.csv
- outputs/124_external_liposome_ivr/dataset_summary.csv
- outputs/124_external_liposome_ivr/manifest.json
- outputs/124_external_liposome_ivr/summary.md

Expected runtime:
- < 10 s on the bundled external corpus
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "external"
    / "accelerated_IVR"
    / "repo"
    / "accelerated_IVR-main"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="root of the accelerated_IVR external dataset",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "124_external_liposome_ivr",
        help="output directory",
    )
    return parser.parse_args()


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(np.diff(release) < -tol))


MODEL_NAME_MAP = {
    "Zero_orderAbsolute_Error": "Zero_order",
    "First_orderAbsolute_Error": "First_order",
    "HiguchiAbsolute_Error": "Higuchi",
    "Korsmeyer_PeppasAbsolute_Error": "Korsmeyer_Peppas",
    "WeibullAbsolute_Error": "Weibull",
    "ReciprocalAbsolute_Error": "Reciprocal",
    "Zero Order": "Zero_order",
    "First Order": "First_order",
    "Korsmeyer-Peppas": "Korsmeyer_Peppas",
}


def normalize_fit_table(df: pd.DataFrame, file_col: str) -> pd.DataFrame:
    keep = [col for col in df.columns if not col.startswith("Unnamed:")]
    out = df[keep].copy()
    out["IVR_ID"] = out[file_col].astype(str).str.extract(r"(\d+)").astype(int)
    renamed = {}
    for col in out.columns:
        if col == file_col or col == "IVR_ID":
            continue
        renamed[col] = MODEL_NAME_MAP.get(col, col).replace(" ", "_")
    return out.rename(columns=renamed)


def pick_best_model(row: pd.Series, mode: str) -> tuple[str | pd.NA, float | pd.NA]:
    candidates = {k: float(v) for k, v in row.items() if k != "IVR_ID" and pd.notna(v)}
    if not candidates:
        return pd.NA, pd.NA
    if mode == "min":
        key = min(candidates, key=candidates.get)
    else:
        key = max(candidates, key=candidates.get)
    return key, candidates[key]


def build_fit_summary(mae: pd.DataFrame, aic: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
    merged = mae.merge(aic, on="IVR_ID", how="outer", suffixes=("_mae", "_aic"))
    merged = merged.merge(f2, on="IVR_ID", how="outer")
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        mae_cols = {k.replace("_mae", ""): row[k] for k in merged.columns if k.endswith("_mae")}
        aic_cols = {k.replace("_aic", ""): row[k] for k in merged.columns if k.endswith("_aic")}
        f2_cols = {
            k: row[k]
            for k in ["Zero_order", "First_order", "Higuchi", "Korsmeyer_Peppas", "Weibull", "Reciprocal"]
            if k in merged.columns
        }
        best_mae_model, best_mae_value = pick_best_model(pd.Series(mae_cols), mode="min")
        best_aic_model, best_aic_value = pick_best_model(pd.Series(aic_cols), mode="min")
        best_f2_model, best_f2_value = pick_best_model(pd.Series(f2_cols), mode="max")
        rows.append(
            {
                "IVR_ID": int(row["IVR_ID"]),
                "best_model_mae": best_mae_model,
                "best_model_mae_value": best_mae_value,
                "best_model_aic": best_aic_model,
                "best_model_aic_value": best_aic_value,
                "best_model_f2": best_f2_model,
                "best_model_f2_value": best_f2_value,
            }
        )
    return pd.DataFrame(rows)


def load_external_tables(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp = pd.read_csv(root / "results" / "fitting" / "drug_release_exp.csv")
    backend = pd.read_csv(root / "data" / "unprocessed" / "backend_data.csv").drop(columns=["Unnamed: 0"])
    clustering = pd.read_csv(root / "results" / "clustering" / "3_PCA_KMC.csv").rename(columns={"file": "IVR_ID"})
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
    quality = quality.drop_duplicates("IVR_ID")
    mae = normalize_fit_table(pd.read_csv(root / "results" / "fitting" / "MAE_df.csv"), "file_name")
    aic = normalize_fit_table(pd.read_csv(root / "results" / "fitting" / "aic_df.csv"), "File Name")
    f2 = normalize_fit_table(pd.read_csv(root / "results" / "fitting" / "F2_df.csv"), "File Name")
    fit_summary = build_fit_summary(mae, aic, f2)
    return exp, backend, clustering, time_units, quality, fit_summary


def build_curves(exp: pd.DataFrame) -> pd.DataFrame:
    curves = exp.copy()
    curves["IVR_ID"] = curves["file_name"].str.extract(r"(\d+)").astype(int)
    curves["source_dataset"] = "liposome_ivr209"
    curves["source_curve_id"] = curves["IVR_ID"].astype(str)
    curves["unified_curve_id"] = curves["source_dataset"] + ":" + curves["source_curve_id"]
    curves["time_raw"] = pd.to_numeric(curves["time (Hrs)"], errors="coerce")
    curves["release_raw"] = pd.to_numeric(curves["release_percent"], errors="coerce")
    curves = curves.dropna(subset=["IVR_ID", "time_raw", "release_raw"]).copy()
    curves["time_unit"] = "hour"
    curves["time_days"] = curves["time_raw"] / 24.0
    curves["release_unit"] = "percent"
    curves["release_percent"] = curves["release_raw"].clip(lower=0.0)
    curves["release_fraction"] = curves["release_percent"] / 100.0
    curves["record_id"] = (
        curves["source_dataset"]
        + ":"
        + curves["source_curve_id"]
        + ":"
        + curves["time_raw"].round(8).astype(str)
    )
    return curves[
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
            "file_name",
        ]
    ].copy()


def build_formulations(
    curves: pd.DataFrame,
    backend: pd.DataFrame,
    clustering: pd.DataFrame,
    time_units: pd.DataFrame,
    quality: pd.DataFrame,
    fit_summary: pd.DataFrame,
) -> pd.DataFrame:
    unique_curves = curves[["unified_curve_id", "source_dataset", "source_curve_id"]].drop_duplicates().copy()
    unique_curves["IVR_ID"] = unique_curves["source_curve_id"].astype(int)
    meta = unique_curves.merge(backend, on="IVR_ID", how="left")
    meta = meta.merge(
        clustering[["IVR_ID", "cluster", "cluster_name", "alpha", "beta"]],
        on="IVR_ID",
        how="left",
    )
    meta = meta.merge(time_units, on="IVR_ID", how="left")
    meta = meta.merge(
        quality[
            [
                "IVR_ID",
                "reported_units",
                "plot_resolution",
                "timepoint_count_sufficient",
                "shape_resembles_profile",
                "repeats_and_average_reported",
                "comments",
            ]
        ],
        on="IVR_ID",
        how="left",
    )
    meta = meta.merge(fit_summary, on="IVR_ID", how="left")
    meta["source_group"] = pd.NA
    meta["source_has_explicit_group"] = False
    meta["polymer_family"] = "liposome"
    meta["LA/GA"] = np.nan
    meta["Polymer_MW"] = np.nan
    meta["Polymer_MW_raw_unit"] = pd.NA
    meta["CL Ratio"] = np.nan
    meta["Drug_Tm"] = meta["weighted_Tm"]
    meta["Drug_Pka"] = np.nan
    meta["Initial D/M ratio"] = meta["drug_loading"]
    meta["DLC"] = meta["drug_loading"]
    meta["DLC_percent"] = meta["drug_loading"] * 100.0
    meta["EE"] = np.nan
    meta["Particle_Size"] = meta["Z_average_nm"]
    meta["SA-V"] = np.nan
    meta["SE"] = np.nan
    meta["Drug_Mw"] = meta["weighted_Mw"]
    meta["Drug_TPSA"] = np.nan
    meta["Drug_NHA"] = np.nan
    meta["Drug_LogP"] = np.nan
    return meta[
        [
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
            "release_method",
            "media_pH",
            "media_temp_oC",
            "structure_type",
            "PDI",
            "zeta_potential",
            "API_ID",
            "API_name",
            "weighted_Mw",
            "weighted_Tm",
            "Time_units",
            "reported_units",
            "plot_resolution",
            "timepoint_count_sufficient",
            "shape_resembles_profile",
            "repeats_and_average_reported",
            "comments",
            "cluster",
            "cluster_name",
            "alpha",
            "beta",
            "best_model_mae",
            "best_model_mae_value",
            "best_model_aic",
            "best_model_aic_value",
            "best_model_f2",
            "best_model_f2_value",
            "formulation_ID",
            "IVR_ID",
        ]
    ].copy()


def build_curve_quality_summary(curves: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    form_index = formulations.set_index("unified_curve_id")
    rows: list[dict[str, object]] = []
    for unified_curve_id, sub in curves.groupby("unified_curve_id", sort=True):
        sub = sub.sort_values("time_days").reset_index(drop=True)
        release = sub["release_fraction"].to_numpy(dtype=float)
        time = sub["time_days"].to_numpy(dtype=float)
        meta = form_index.loc[unified_curve_id] if unified_curve_id in form_index.index else None
        rows.append(
            {
                "unified_curve_id": unified_curve_id,
                "source_dataset": str(sub["source_dataset"].iloc[0]),
                "source_curve_id": str(sub["source_curve_id"].iloc[0]),
                "n_points": int(len(sub)),
                "time_start_days": float(time[0]),
                "time_end_days": float(time[-1]),
                "duration_days": float(time[-1] - time[0]),
                "release_final": float(release[-1]),
                "release_max": float(np.nanmax(release)),
                "release_min": float(np.nanmin(release)),
                "monotonicity_violations": monotonicity_violation_count(release),
                "has_duplicate_times": bool(sub["time_raw"].duplicated().any()),
                "has_release_gt_1": bool(np.nanmax(release) > 1.0 + 1e-6),
                "has_release_gt_105pct": bool(np.nanmax(release) > 1.05 + 1e-6),
                "has_negative_release_raw": bool(np.nanmin(sub["release_raw"].to_numpy(dtype=float)) < -1e-6),
                "time_is_sorted": bool(np.all(np.diff(time) >= -1e-12)),
                "has_formulation_row": bool(meta is not None),
                "has_cluster_label": bool(pd.notna(meta["cluster_name"])) if meta is not None else False,
                "has_quality_row": bool(pd.notna(meta["reported_units"])) if meta is not None else False,
                "has_time_unit_row": bool(pd.notna(meta["Time_units"])) if meta is not None else False,
                "best_model_mae": meta["best_model_mae"] if meta is not None else pd.NA,
                "best_model_aic": meta["best_model_aic"] if meta is not None else pd.NA,
                "best_model_f2": meta["best_model_f2"] if meta is not None else pd.NA,
                "release_method": meta["release_method"] if meta is not None else pd.NA,
                "structure_type": meta["structure_type"] if meta is not None else pd.NA,
                "API_name": meta["API_name"] if meta is not None else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(curve_quality: pd.DataFrame, formulations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_dataset": "liposome_ivr209",
                "n_curves": int(curve_quality["unified_curve_id"].nunique()),
                "n_formulations": int(formulations["unified_curve_id"].nunique()),
                "n_points_total": int(curve_quality["n_points"].sum()),
                "median_points_per_curve": float(curve_quality["n_points"].median()),
                "median_duration_days": float(curve_quality["duration_days"].median()),
                "median_release_final": float(curve_quality["release_final"].median()),
                "median_release_max": float(curve_quality["release_max"].median()),
                "monotone_clean_fraction": float((curve_quality["monotonicity_violations"] == 0).mean()),
                "release_gt_1_fraction": float(curve_quality["has_release_gt_1"].mean()),
                "cluster_label_fraction": float(curve_quality["has_cluster_label"].mean()),
                "quality_row_fraction": float(curve_quality["has_quality_row"].mean()),
                "time_unit_row_fraction": float(curve_quality["has_time_unit_row"].mean()),
                "metadata_row_fraction": float(curve_quality["has_formulation_row"].mean()),
            }
        ]
    )


def build_manifest(curves: pd.DataFrame, formulations: pd.DataFrame, curve_quality: pd.DataFrame) -> dict[str, object]:
    return {
        "corpus_name": "external_liposome_ivr",
        "source_dataset": "liposome_ivr209",
        "time_raw_unit": "hour",
        "time_aligned_unit": "day",
        "release_raw_unit": "percent",
        "release_aligned_unit": "fraction",
        "n_total_curves": int(curve_quality["unified_curve_id"].nunique()),
        "n_total_points": int(len(curves)),
        "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
        "n_cluster_labeled_curves": int(curve_quality["has_cluster_label"].sum()),
        "curve_columns": curves.columns.tolist(),
        "formulation_columns": formulations.columns.tolist(),
        "quality_columns": curve_quality.columns.tolist(),
    }


def write_summary_markdown(
    dataset_summary: pd.DataFrame,
    curve_quality: pd.DataFrame,
    formulations: pd.DataFrame,
    manifest: dict[str, object],
    out_path: Path,
) -> None:
    summary = dataset_summary.iloc[0]
    top_api = formulations["API_name"].fillna("UNK").value_counts().head(10)
    top_methods = formulations["release_method"].fillna("UNK").value_counts().head(10)
    cluster_counts = formulations["cluster_name"].fillna("UNK").value_counts()

    lines = [
        "# external_liposome_ivr",
        "",
        "Standardized preprocessing output for the bundled external liposome IVR dataset.",
        "",
        f"- total curves: `{manifest['n_total_curves']}`",
        f"- total observations: `{manifest['n_total_points']}`",
        f"- total formulations: `{manifest['n_total_formulations']}`",
        f"- cluster-labeled curves: `{manifest['n_cluster_labeled_curves']}`",
        "",
        "## Dataset Summary",
        "",
        f"- median points/curve: `{summary['median_points_per_curve']:.1f}`",
        f"- median duration days: `{summary['median_duration_days']:.3f}`",
        f"- monotone-clean fraction: `{summary['monotone_clean_fraction']:.3f}`",
        f"- release > 1.0 fraction: `{summary['release_gt_1_fraction']:.3f}`",
        f"- metadata row fraction: `{summary['metadata_row_fraction']:.3f}`",
        f"- cluster label fraction: `{summary['cluster_label_fraction']:.3f}`",
        f"- quality row fraction: `{summary['quality_row_fraction']:.3f}`",
        f"- time-unit row fraction: `{summary['time_unit_row_fraction']:.3f}`",
        "",
        "## Top APIs",
        "",
    ]
    for name, count in top_api.items():
        lines.append(f"- `{name}`: `{int(count)}`")
    lines.extend(["", "## Top Release Methods", ""])
    for name, count in top_methods.items():
        lines.append(f"- `{name}`: `{int(count)}`")
    lines.extend(["", "## Cluster Labels", ""])
    for name, count in cluster_counts.items():
        lines.append(f"- `{name}`: `{int(count)}`")
    lines.extend(["", "## Best-Fit Model By MAE", ""])
    for name, count in formulations["best_model_mae"].fillna("UNK").value_counts().items():
        lines.append(f"- `{name}`: `{int(count)}`")
    lines.extend(["", "## Highest-Final-Release Curves", ""])
    top = curve_quality.sort_values("release_max", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['unified_curve_id']}`: max_release={row['release_max']:.4f}, "
            f"final_release={row['release_final']:.4f}, "
            f"release_method=`{row['release_method']}`, API=`{row['API_name']}`"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    exp, backend, clustering, time_units, quality, fit_summary = load_external_tables(args.external_root)
    curves = build_curves(exp)
    formulations = build_formulations(curves, backend, clustering, time_units, quality, fit_summary)
    curve_quality = build_curve_quality_summary(curves, formulations)
    dataset_summary = build_dataset_summary(curve_quality, formulations)
    manifest = build_manifest(curves, formulations, curve_quality)

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_markdown(dataset_summary, curve_quality, formulations, manifest, args.outdir / "summary.md")

    print(f"[external-liposome-ivr] wrote outputs to {args.outdir}")
    print(
        f"[external-liposome-ivr] total curves={manifest['n_total_curves']} "
        f"total points={manifest['n_total_points']} cluster_labeled={manifest['n_cluster_labeled_curves']}"
    )


if __name__ == "__main__":
    main()
