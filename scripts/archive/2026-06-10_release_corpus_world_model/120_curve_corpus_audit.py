from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CROSS_DOI_XLSX = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)


def load_release_tables(curves_csv: Path, formulations_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(curves_csv)
    formulations = pd.read_csv(formulations_csv)
    required_curve_cols = {"curve_id", "time", "release"}
    if not required_curve_cols.issubset(curves.columns):
        missing = required_curve_cols - set(curves.columns)
        raise ValueError(f"Missing curve columns in {curves_csv}: {sorted(missing)}")
    if "curve_id" not in formulations.columns:
        raise ValueError(f"Missing curve_id in {formulations_csv}")
    return curves.copy(), formulations.copy()


def load_cross_doi_321(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(xlsx_path).rename(
        columns={
            "Formulation Index": "curve_id",
            "Drug MW": "Drug_Mw",
            "Drug TPSA": "Drug_TPSA",
            "Drug LogP": "Drug_LogP",
            "Polymer MW": "Polymer_MW",
            "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
            "Drug Loading Capacity": "DLC",
            "Time": "time",
            "Release": "release",
        }
    )
    raw["curve_id"] = raw["curve_id"].astype(str)
    raw["Polymer_MW"] = raw["Polymer_MW"].astype(float) * 1000.0
    raw["DLC"] = raw["DLC"].astype(float) / 100.0
    raw["release"] = raw["release"].astype(float).clip(lower=0.0)
    raw["DP_Group"] = "UNK-PLGA"

    curves = (
        raw.groupby(["curve_id", "time"], as_index=False, sort=False)
        .agg(
            {
                **{c: "first" for c in raw.columns if c not in ("time", "release")},
                "release": "mean",
            }
        )[["curve_id", "time", "release"]]
        .copy()
    )

    formulations = (
        raw.drop_duplicates("curve_id")
        .loc[
            :,
            [
                "curve_id",
                "DP_Group",
                "LA/GA",
                "Polymer_MW",
                "Drug_Mw",
                "Drug_LogP",
                "Drug_TPSA",
                "Initial D/M ratio",
                "DLC",
            ],
        ]
        .copy()
    )
    return curves, formulations


def monotonicity_violation_count(release: np.ndarray, tol: float = 1e-6) -> int:
    diffs = np.diff(release)
    return int(np.sum(diffs < -tol))


def summarize_curves(curves: pd.DataFrame, formulations: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    meta = formulations.set_index("curve_id", drop=False)
    for curve_id, sub in curves.groupby("curve_id", sort=True):
        sub = sub.sort_values("time").reset_index(drop=True)
        time = sub["time"].to_numpy(dtype=float)
        release = sub["release"].to_numpy(dtype=float)
        max_release = float(np.nanmax(release))
        final_release = float(release[-1])
        duration = float(time[-1] - time[0])
        auc = float(np.trapz(release, time))
        first_positive_idx = np.where(release > 0)[0]
        t_first_positive = float(time[first_positive_idx[0]]) if len(first_positive_idx) else np.nan
        row: dict[str, float | int | str] = {
            "dataset": dataset_name,
            "curve_id": int(curve_id),
            "n_points": int(len(sub)),
            "time_start": float(time[0]),
            "time_end": float(time[-1]),
            "duration": duration,
            "t_first_positive": t_first_positive,
            "release_min": float(np.nanmin(release)),
            "release_max": max_release,
            "release_final": final_release,
            "auc": auc,
            "monotonicity_violations": monotonicity_violation_count(release),
        }
        if curve_id in meta.index:
            m = meta.loc[curve_id]
            row["DP_Group"] = str(m.get("DP_Group", ""))
            for col in ("LA/GA", "Polymer_MW", "Drug_Mw", "Drug_LogP"):
                if col in m.index:
                    row[col] = float(m[col]) if pd.notna(m[col]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def scalar_summary(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def build_corpus_summary(curve_stats: pd.DataFrame, curves: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    group_count = int(curve_stats["DP_Group"].nunique()) if "DP_Group" in curve_stats.columns else 0
    monotone_clean = int((curve_stats["monotonicity_violations"] == 0).sum())
    return {
        "dataset": dataset_name,
        "n_curves": int(len(curve_stats)),
        "n_observations": int(len(curves)),
        "n_groups": group_count,
        "points_per_curve": scalar_summary(curve_stats["n_points"]),
        "duration_days": scalar_summary(curve_stats["duration"]),
        "release_final": scalar_summary(curve_stats["release_final"]),
        "release_max": scalar_summary(curve_stats["release_max"]),
        "t_first_positive": scalar_summary(curve_stats["t_first_positive"]),
        "monotone_clean_curves": monotone_clean,
        "monotone_clean_fraction": float(monotone_clean / len(curve_stats)) if len(curve_stats) else np.nan,
    }


def write_markdown(
    summaries: list[dict[str, object]],
    group_summaries: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    lines = [
        "# Curve Corpus Audit",
        "",
        "## Dataset Headlines",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['dataset']}",
                "",
                f"- curves: `{summary['n_curves']}`",
                f"- observations: `{summary['n_observations']}`",
                f"- groups: `{summary['n_groups']}`",
                f"- monotone-clean fraction: `{summary['monotone_clean_fraction']:.3f}`",
                "",
                "Distribution snapshots:",
                "",
            ]
        )
        for key in ("points_per_curve", "duration_days", "release_final", "release_max", "t_first_positive"):
            stats = summary[key]
            if not isinstance(stats, dict) or stats.get("count", 0) == 0:
                continue
            lines.append(
                f"- `{key}`: min={stats['min']:.3f}, p25={stats['p25']:.3f}, "
                f"median={stats['median']:.3f}, mean={stats['mean']:.3f}, "
                f"p75={stats['p75']:.3f}, max={stats['max']:.3f}"
            )
        lines.append(
            ""
        )
        lines.extend(
            [
                "Top groups by curve count:",
                "",
            ]
        )
        group_summary = group_summaries[summary["dataset"]]
        top = group_summary.sort_values("n_curves", ascending=False).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['DP_Group']}`: curves={int(row['n_curves'])}, "
                f"median_points={row['median_points']:.1f}, median_duration={row['median_duration']:.2f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-doi-xlsx",
        type=Path,
        default=DEFAULT_CROSS_DOI_XLSX,
        help="optional local 321-curve xlsx; if absent, audit stays internal-only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    curves_csv = repo_root / "data" / "curves_long.csv"
    formulations_csv = repo_root / "data" / "formulations.csv"
    out_dir = repo_root / "outputs" / "120_curve_corpus_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    curves, formulations = load_release_tables(curves_csv, formulations_csv)
    datasets.append(("internal181", curves, formulations))

    if args.cross_doi_xlsx.exists():
        cross_curves, cross_formulations = load_cross_doi_321(args.cross_doi_xlsx)
        datasets.append(("cross321", cross_curves, cross_formulations))
    else:
        print(f"[curve-corpus-audit] cross-DOI xlsx not found, skipping: {args.cross_doi_xlsx}")

    curve_stats_all: list[pd.DataFrame] = []
    group_summaries: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for dataset_name, dataset_curves, dataset_formulations in datasets:
        curve_stats = summarize_curves(dataset_curves, dataset_formulations, dataset_name)
        curve_stats_all.append(curve_stats)
        group_summary = (
            curve_stats.groupby("DP_Group", dropna=False)
            .agg(
                n_curves=("curve_id", "count"),
                median_points=("n_points", "median"),
                median_duration=("duration", "median"),
                median_final_release=("release_final", "median"),
            )
            .reset_index()
        )
        group_summaries[dataset_name] = group_summary
        summaries.append(build_corpus_summary(curve_stats, dataset_curves, dataset_name))

    curve_stats_combined = pd.concat(curve_stats_all, ignore_index=True)
    curves_combined = pd.concat([x[1] for x in datasets], ignore_index=True)
    combined_summary = build_corpus_summary(curve_stats_combined, curves_combined, "combined")
    combined_group_summary = (
        curve_stats_combined.groupby(["dataset", "DP_Group"], dropna=False)
        .agg(
            n_curves=("curve_id", "count"),
            median_points=("n_points", "median"),
            median_duration=("duration", "median"),
            median_final_release=("release_final", "median"),
        )
        .reset_index()
    )
    group_summaries["combined"] = combined_group_summary.assign(
        DP_Group=lambda df: df["dataset"].astype(str) + "::" + df["DP_Group"].astype(str)
    )
    summaries.append(combined_summary)

    curve_stats_combined.to_csv(out_dir / "per_curve_summary.csv", index=False, encoding="utf-8-sig")
    for name, group_summary in group_summaries.items():
        group_summary.to_csv(out_dir / f"group_summary_{name}.csv", index=False, encoding="utf-8-sig")
    (out_dir / "corpus_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_markdown(summaries, group_summaries, out_dir / "summary.md")

    print(f"[curve-corpus-audit] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
