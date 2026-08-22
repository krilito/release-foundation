from __future__ import annotations

"""
148_materialize_layered_release_pools.py

Consume:
- outputs/127_release_data_asset_registry/asset_registry.csv
- outputs/147_layered_release_corpus_registry/recommended_pools.json

Produce:
- outputs/148_release_main_cumulative_v1/*
- outputs/149_release_caveat_augmented_cumulative_v1/*

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
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    return parser.parse_args()


def load_asset_table(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(repo_root / "outputs" / "127_release_data_asset_registry" / "asset_registry.csv")


def load_recommended_pools(repo_root: Path) -> dict[str, object]:
    return json.loads(
        (repo_root / "outputs" / "147_layered_release_corpus_registry" / "recommended_pools.json").read_text(
            encoding="utf-8"
        )
    )


def load_corpus_from_manifest(manifest_path: Path) -> dict[str, pd.DataFrame | dict[str, object]]:
    base = manifest_path.parent
    return {
        "curves": pd.read_csv(base / "curves_long.csv"),
        "formulations": pd.read_csv(base / "formulations.csv"),
        "curve_quality": pd.read_csv(base / "curve_quality_summary.csv"),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
    }


def materialize_pool(
    repo_root: Path,
    asset_table: pd.DataFrame,
    asset_names: list[str],
    outdir: Path,
    corpus_name: str,
    description: str,
) -> None:
    selected = asset_table.set_index("asset_name").loc[asset_names].reset_index()
    corpora = {
        row["asset_name"]: load_corpus_from_manifest(Path(row["manifest_path"]))
        for _, row in selected.iterrows()
    }

    merged_curves = pd.concat([corpora[name]["curves"] for name in asset_names], ignore_index=True)
    merged_formulations = pd.concat([corpora[name]["formulations"] for name in asset_names], ignore_index=True)
    merged_curve_quality = pd.concat([corpora[name]["curve_quality"] for name in asset_names], ignore_index=True)

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
            for name in asset_names
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    merged_curves.to_csv(outdir / "curves_long.csv", index=False)
    merged_formulations.to_csv(outdir / "formulations.csv", index=False)
    merged_curve_quality.to_csv(outdir / "curve_quality_summary.csv", index=False)
    corpus_registry.to_csv(outdir / "corpus_registry.csv", index=False)
    corpus_registry.to_csv(outdir / "dataset_summary.csv", index=False)
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": corpus_name,
                "sources": asset_names,
                "n_total_curves": int(merged_curve_quality["unified_curve_id"].nunique()),
                "n_total_points": int(len(merged_curves)),
                "n_total_formulations": int(merged_formulations["unified_curve_id"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# {corpus_name}",
        "",
        description,
        "",
        f"- total curves: `{int(merged_curve_quality['unified_curve_id'].nunique())}`",
        f"- total points: `{int(len(merged_curves))}`",
        f"- total formulations: `{int(merged_formulations['unified_curve_id'].nunique())}`",
        "",
        "## Component corpora",
        "",
    ]
    for _, row in corpus_registry.iterrows():
        lines.append(
            f"- `{row['corpus_name']}`: `{int(row['n_curves'])}` curves, `{int(row['n_points_total'])}` points"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    asset_table = load_asset_table(args.repo_root)
    pools = load_recommended_pools(args.repo_root)

    main_assets = pools["main_cumulative_base_assets"]
    caveat_assets = pools["caveat_cumulative_assets"]

    materialize_pool(
        repo_root=args.repo_root,
        asset_table=asset_table,
        asset_names=main_assets,
        outdir=args.repo_root / "outputs" / "148_release_main_cumulative_v1",
        corpus_name="release_main_cumulative_v1",
        description="Materialized clean cumulative release pool built from the main cumulative base assets only.",
    )
    materialize_pool(
        repo_root=args.repo_root,
        asset_table=asset_table,
        asset_names=main_assets + caveat_assets,
        outdir=args.repo_root / "outputs" / "149_release_caveat_augmented_cumulative_v1",
        corpus_name="release_caveat_augmented_cumulative_v1",
        description="Materialized cumulative release pool built from the clean base assets plus caveat cumulative assets.",
    )

    print("[materialize-layered-release-pools] wrote outputs to:")
    print(f"  - {args.repo_root / 'outputs' / '148_release_main_cumulative_v1'}")
    print(f"  - {args.repo_root / 'outputs' / '149_release_caveat_augmented_cumulative_v1'}")


if __name__ == "__main__":
    main()
