from __future__ import annotations

"""
136_merge_release_corpus_candidates_v4.py

Consume:
- outputs/123_release_corpus_v1/*
- outputs/124_external_liposome_ivr/*
- outputs/131_external_bmp2_hydrogel_zenodo/*
- outputs/133_external_degrapol_mendeley/*
- outputs/135_external_timp1_mendeley/*

Produce:
- outputs/136_release_corpus_candidates_v4/curves_long.csv
- outputs/136_release_corpus_candidates_v4/formulations.csv
- outputs/136_release_corpus_candidates_v4/curve_quality_summary.csv
- outputs/136_release_corpus_candidates_v4/corpus_registry.csv
- outputs/136_release_corpus_candidates_v4/dataset_summary.csv
- outputs/136_release_corpus_candidates_v4/manifest.json
- outputs/136_release_corpus_candidates_v4/summary.md

Expected runtime:
- < 10 s
"""

import argparse
import json
from pathlib import Path

import pandas as pd


CORPUS_DIRS = [
    "123_release_corpus_v1",
    "124_external_liposome_ivr",
    "131_external_bmp2_hydrogel_zenodo",
    "133_external_degrapol_mendeley",
    "135_external_timp1_mendeley",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "136_release_corpus_candidates_v4",
    )
    return parser.parse_args()


def load_corpus(root: Path, name: str) -> dict[str, pd.DataFrame | dict[str, object]]:
    base = root / "outputs" / name
    return {
        "curves": pd.read_csv(base / "curves_long.csv"),
        "formulations": pd.read_csv(base / "formulations.csv"),
        "curve_quality": pd.read_csv(base / "curve_quality_summary.csv"),
        "manifest": json.loads((base / "manifest.json").read_text(encoding="utf-8")),
    }


def write_summary(corpus_registry: pd.DataFrame, total_curves: int, total_points: int, out_path: Path) -> None:
    lines = [
        "# release_corpus_candidates_v4",
        "",
        "Merged standardized release corpus across PLGA, liposome, BMP-2 hydrogel, DegraPol, and TIMP-1 release sources.",
        "",
        f"- total curves: `{total_curves}`",
        f"- total points: `{total_points}`",
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

    corpora = {name: load_corpus(args.repo_root, name) for name in CORPUS_DIRS}
    merged_curves = pd.concat([corpora[name]["curves"] for name in CORPUS_DIRS], ignore_index=True)
    merged_formulations = pd.concat([corpora[name]["formulations"] for name in CORPUS_DIRS], ignore_index=True)
    merged_curve_quality = pd.concat([corpora[name]["curve_quality"] for name in CORPUS_DIRS], ignore_index=True)

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
            for name in CORPUS_DIRS
        ]
    )

    merged_curves.to_csv(args.outdir / "curves_long.csv", index=False)
    merged_formulations.to_csv(args.outdir / "formulations.csv", index=False)
    merged_curve_quality.to_csv(args.outdir / "curve_quality_summary.csv", index=False)
    corpus_registry.to_csv(args.outdir / "corpus_registry.csv", index=False)
    corpus_registry.to_csv(args.outdir / "dataset_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "release_corpus_candidates_v4",
                "sources": [
                    "release_corpus_v1",
                    "external_liposome_ivr",
                    "external_bmp2_hydrogel_zenodo",
                    "external_degrapol_mendeley",
                    "external_timp1_mendeley",
                ],
                "n_total_curves": int(merged_curve_quality["unified_curve_id"].nunique()),
                "n_total_points": int(len(merged_curves)),
                "n_total_formulations": int(merged_formulations["unified_curve_id"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(
        corpus_registry=corpus_registry,
        total_curves=int(merged_curve_quality["unified_curve_id"].nunique()),
        total_points=int(len(merged_curves)),
        out_path=args.outdir / "summary.md",
    )

    print(f"[release-corpus-candidates-v4] wrote outputs to {args.outdir}")
    print(
        "[release-corpus-candidates-v4] "
        f"curves={merged_curve_quality['unified_curve_id'].nunique()} "
        f"points={len(merged_curves)} formulations={merged_formulations['unified_curve_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
