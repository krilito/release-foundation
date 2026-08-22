from __future__ import annotations

"""
132_merge_release_corpus_candidates_v2.py

Consume:
- outputs/123_release_corpus_v1/*
- outputs/124_external_liposome_ivr/*
- outputs/131_external_bmp2_hydrogel_zenodo/*

Produce:
- outputs/132_release_corpus_candidates_v2/curves_long.csv
- outputs/132_release_corpus_candidates_v2/formulations.csv
- outputs/132_release_corpus_candidates_v2/curve_quality_summary.csv
- outputs/132_release_corpus_candidates_v2/dataset_summary.csv
- outputs/132_release_corpus_candidates_v2/corpus_registry.csv
- outputs/132_release_corpus_candidates_v2/manifest.json
- outputs/132_release_corpus_candidates_v2/summary.md

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
        "--repo-root",
        type=Path,
        default=repo_root,
        help="repository root",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "132_release_corpus_candidates_v2",
        help="output directory",
    )
    return parser.parse_args()


def load_corpus(root: Path, name: str) -> dict[str, pd.DataFrame]:
    base = root / "outputs" / name
    return {
        "curves": pd.read_csv(base / "curves_long.csv"),
        "formulations": pd.read_csv(base / "formulations.csv"),
        "curve_quality": pd.read_csv(base / "curve_quality_summary.csv"),
        "dataset_summary": pd.read_csv(base / "dataset_summary.csv"),
        "manifest": json.loads((base / "manifest.json").read_text(encoding="utf-8")),
    }


def write_summary(
    corpus_registry: pd.DataFrame,
    merged_curve_quality: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# release_corpus_candidates_v2",
        "",
        "Merged standardized release corpus across PLGA, liposome, and BMP-2 hydrogel sources.",
        "",
        f"- total curves: `{int(merged_curve_quality['unified_curve_id'].nunique())}`",
        f"- total points: `{int(merged_curve_quality['n_points'].sum())}`",
        "",
        "## Component corpora",
        "",
    ]
    for _, row in corpus_registry.iterrows():
        lines.append(
            f"- `{row['corpus_name']}`: `{int(row['n_curves'])}` curves, `{int(row['n_points_total'])}` points"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    corpus_names = [
        "123_release_corpus_v1",
        "124_external_liposome_ivr",
        "131_external_bmp2_hydrogel_zenodo",
    ]
    corpora = {name: load_corpus(args.repo_root, name) for name in corpus_names}

    merged_curves = pd.concat([corpora[name]["curves"] for name in corpus_names], ignore_index=True)
    merged_formulations = pd.concat([corpora[name]["formulations"] for name in corpus_names], ignore_index=True)
    merged_curve_quality = pd.concat([corpora[name]["curve_quality"] for name in corpus_names], ignore_index=True)
    corpus_registry = pd.DataFrame(
        [
            {
                "corpus_name": corpora[name]["manifest"].get("corpus_name", name),
                "source_dataset": corpora[name]["manifest"].get(
                    "source_dataset",
                    ",".join(corpora[name]["manifest"].get("sources", [])),
                ),
                "n_curves": int(corpora[name]["curve_quality"]["unified_curve_id"].nunique()),
                "n_points_total": int(len(corpora[name]["curves"])),
                "n_formulations": int(corpora[name]["formulations"]["unified_curve_id"].nunique()),
            }
            for name in corpus_names
        ]
    )

    merged_curves.to_csv(args.outdir / "curves_long.csv", index=False)
    merged_formulations.to_csv(args.outdir / "formulations.csv", index=False)
    merged_curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    corpus_registry.to_csv(args.outdir / "dataset_summary.csv", index=False)
    corpus_registry.to_csv(args.outdir / "corpus_registry.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "release_corpus_candidates_v2",
                "sources": [
                    "release_corpus_v1",
                    "external_liposome_ivr",
                    "external_bmp2_hydrogel_zenodo",
                ],
                "n_total_curves": int(merged_curve_quality["unified_curve_id"].nunique()),
                "n_total_points": int(len(merged_curves)),
                "n_total_formulations": int(merged_formulations["unified_curve_id"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(corpus_registry, merged_curve_quality, args.outdir / "summary.md")

    print(f"[release-corpus-candidates-v2] wrote outputs to {args.outdir}")
    print(
        "[release-corpus-candidates-v2] "
        f"curves={merged_curve_quality['unified_curve_id'].nunique()} "
        f"points={len(merged_curves)} formulations={merged_formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
