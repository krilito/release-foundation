from __future__ import annotations

"""
125_merge_standardized_release_corpora.py

Consume:
- outputs/123_release_corpus_v1/
- outputs/124_external_liposome_ivr/

Produce:
- outputs/125_release_corpus_candidates/curves_long.csv
- outputs/125_release_corpus_candidates/formulations.csv
- outputs/125_release_corpus_candidates/curve_quality_summary.csv
- outputs/125_release_corpus_candidates/dataset_summary.csv
- outputs/125_release_corpus_candidates/corpus_registry.csv
- outputs/125_release_corpus_candidates/manifest.json
- outputs/125_release_corpus_candidates/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus123",
        type=Path,
        default=repo_root / "outputs" / "123_release_corpus_v1",
        help="standardized PLGA corpus directory",
    )
    parser.add_argument(
        "--corpus124",
        type=Path,
        default=repo_root / "outputs" / "124_external_liposome_ivr",
        help="standardized external liposome corpus directory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "125_release_corpus_candidates",
        help="output directory",
    )
    return parser.parse_args()


def load_corpus(corpus_dir: Path, corpus_name: str) -> dict[str, pd.DataFrame]:
    return {
        "corpus_name": corpus_name,
        "curves": pd.read_csv(corpus_dir / "curves_long.csv"),
        "formulations": pd.read_csv(corpus_dir / "formulations.csv"),
        "quality": pd.read_csv(corpus_dir / "curve_quality_summary.csv"),
        "dataset_summary": pd.read_csv(corpus_dir / "dataset_summary.csv"),
        "manifest": json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8")),
    }


def concat_with_corpus_tag(frames: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    tagged: list[pd.DataFrame] = []
    for frame, name in zip(frames, names):
        part = frame.copy()
        part["corpus_name"] = name
        tagged.append(part)
    return pd.concat(tagged, ignore_index=True, sort=False)


def build_registry(corpora: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for corpus in corpora:
        manifest = corpus["manifest"]
        rows.append(
            {
                "corpus_name": corpus["corpus_name"],
                "source_dataset": ",".join(manifest.get("sources", [manifest.get("source_dataset", "UNK")])),
                "n_total_curves": int(manifest["n_total_curves"]),
                "n_total_points": int(manifest["n_total_points"]),
                "n_total_formulations": int(manifest["n_total_formulations"]),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    registry: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# release_corpus_candidates",
        "",
        "Merged registry of currently standardized release corpora.",
        "",
        "## Corpus Registry",
        "",
    ]
    for _, row in registry.iterrows():
        lines.extend(
            [
                f"### {row['corpus_name']}",
                "",
                f"- source datasets: `{row['source_dataset']}`",
                f"- curves: `{int(row['n_total_curves'])}`",
                f"- points: `{int(row['n_total_points'])}`",
                f"- formulations: `{int(row['n_total_formulations'])}`",
                "",
            ]
        )
    lines.extend(["## Dataset Summary", ""])
    for _, row in dataset_summary.sort_values(["corpus_name", "source_dataset"]).iterrows():
        lines.append(
            f"- `{row['corpus_name']} / {row['source_dataset']}`: "
            f"curves={int(row['n_curves'])}, points={int(row['n_points_total'])}, "
            f"median_points={row['median_points_per_curve']:.1f}, "
            f"median_duration_days={row['median_duration_days']:.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    corpora = [
        load_corpus(args.corpus123, "release_corpus_v1"),
        load_corpus(args.corpus124, "external_liposome_ivr"),
    ]
    corpus_names = [corpus["corpus_name"] for corpus in corpora]

    curves = concat_with_corpus_tag([corpus["curves"] for corpus in corpora], corpus_names)
    formulations = concat_with_corpus_tag([corpus["formulations"] for corpus in corpora], corpus_names)
    quality = concat_with_corpus_tag([corpus["quality"] for corpus in corpora], corpus_names)
    dataset_summary = concat_with_corpus_tag([corpus["dataset_summary"] for corpus in corpora], corpus_names)
    registry = build_registry(corpora)

    manifest = {
        "corpus_name": "release_corpus_candidates",
        "n_corpora": len(corpora),
        "corpora": registry["corpus_name"].tolist(),
        "n_total_curves": int(quality["unified_curve_id"].nunique()),
        "n_total_points": int(len(curves)),
        "n_total_formulations": int(formulations["unified_curve_id"].nunique()),
        "curve_columns": sorted(curves.columns.tolist()),
        "formulation_columns": sorted(formulations.columns.tolist()),
        "quality_columns": sorted(quality.columns.tolist()),
    }

    curves.to_csv(args.outdir / "curves_long.csv", index=False)
    formulations.to_csv(args.outdir / "formulations.csv", index=False)
    quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    dataset_summary.to_csv(args.outdir / "dataset_summary.csv", index=False)
    registry.to_csv(args.outdir / "corpus_registry.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(registry, dataset_summary, args.outdir / "summary.md")

    print(f"[release-corpus-candidates] wrote outputs to {args.outdir}")
    print(
        f"[release-corpus-candidates] corpora={manifest['n_corpora']} "
        f"curves={manifest['n_total_curves']} points={manifest['n_total_points']}"
    )


if __name__ == "__main__":
    main()
