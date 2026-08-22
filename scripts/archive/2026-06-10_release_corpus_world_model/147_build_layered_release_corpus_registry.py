from __future__ import annotations

"""
147_build_layered_release_corpus_registry.py

Consume:
- outputs/127_release_data_asset_registry/asset_registry.csv
- outputs/128_external_source_intake_registry/source_intake_registry.csv

Produce:
- outputs/147_layered_release_corpus_registry/layered_asset_registry.csv
- outputs/147_layered_release_corpus_registry/recommended_pools.json
- outputs/147_layered_release_corpus_registry/summary.md

Expected runtime:
- < 5 s
"""

import argparse
import json
from pathlib import Path

import pandas as pd


LAYER_OVERRIDES = {
    "release_corpus_v1": ("main_cumulative_base", "standardized PLGA base corpus"),
    "external_liposome_ivr": ("main_cumulative_base", "clean external liposome cumulative corpus"),
    "external_bmp2_hydrogel_zenodo": ("main_cumulative_base", "clean external hydrogel cumulative corpus"),
    "external_degrapol_mendeley": ("main_cumulative_base", "clean external DegraPol cumulative corpus"),
    "external_alginate_mendeley": ("main_cumulative_base", "clean external alginate cumulative corpus"),
    "external_starch_mendeley": ("main_cumulative_base", "clean external starch cumulative corpus"),
    "external_caseinate_gallic_mendeley": (
        "main_cumulative_base",
        "external cumulative corpus normalized via experimental Mt/Minf values",
    ),
    "external_golfball_microspheres_mendeley": (
        "main_cumulative_base",
        "external cumulative microsphere release corpus extracted from an explicit figure-sheet panel",
    ),
    "external_timp1_mendeley": (
        "caveat_cumulative",
        "cumulative curves are usable but workbook lacks reliable explicit group labels",
    ),
    "external_nt3_hydrogel_mendeley": (
        "caveat_cumulative",
        "cumulative curves are usable but assay identity is mixed and must be preserved",
    ),
    "external_chitosan_zeolite_mendeley": (
        "proxy_only",
        "time-resolved absorbance proxy, intentionally excluded from cumulative pool",
    ),
    "external_caseinate_guar_proxy_mendeley": (
        "proxy_only",
        "time-resolved concentration proxy, intentionally excluded from cumulative pool",
    ),
    "external_cure_proxy_mendeley": (
        "proxy_only",
        "time-resolved HPLC release proxy, intentionally excluded from cumulative pool",
    ),
    "external_nanomed_registry": (
        "metadata_only",
        "descriptor registry exists but raw release curves are missing locally",
    ),
    "bannigan_lai_181_raw": (
        "provenance_only",
        "raw-source provenance copy retained for transformation audits, not for merged training pools",
    ),
    "release_main_cumulative_v1": (
        "materialized_recommended_pool",
        "materialized clean cumulative pool derived from the main cumulative base layer",
    ),
    "release_caveat_augmented_cumulative_v1": (
        "materialized_recommended_pool",
        "materialized cumulative pool derived from the main cumulative base plus caveat cumulative layers",
    ),
    "external_laponite_opju_registry": (
        "provenance_only",
        "pre-extraction registry retained as provenance after the Origin project was recovered numerically",
    ),
    "external_laponite_nh2_mendeley": (
        "provenance_only",
        "partial NH2-only extraction retained for provenance; superseded by the full four-curve laponite corpus",
    ),
    "external_laponite_mendeley": (
        "caveat_cumulative",
        "cumulative curves are usable, but half of the labels come from hidden Book2A datasets resolved through dataset-name and visible-string evidence",
    ),
    "external_plga_dox_mendeley": (
        "caveat_cumulative",
        "cumulative curves are usable, but the corpus mixes a free-drug control with carrier formulations",
    ),
    "external_halloysite_origin_mendeley": (
        "caveat_cumulative",
        "cumulative curves are usable, but the worksheet only exposes compact condition labels (`0mM`, `2mM`, `4mM`) without full semantic expansion",
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "147_layered_release_corpus_registry",
    )
    return parser.parse_args()


def classify_asset(asset_name: str, asset_type: str) -> tuple[str, str]:
    if asset_name in LAYER_OVERRIDES:
        return LAYER_OVERRIDES[asset_name]
    if asset_type == "merged_curve_corpus":
        return ("merged_history", "historical merged cumulative pool snapshot")
    return ("unclassified", "no explicit layer assignment rule yet")


def write_summary(
    registry: pd.DataFrame,
    pool_totals: pd.DataFrame,
    recommended_pools: dict[str, object],
    out_path: Path,
) -> None:
    latest_main = recommended_pools["recommended_latest_main_cumulative_pool"]
    lines = [
        "# layered_release_corpus_registry",
        "",
        "Layered view of standardized release-data assets in the workspace.",
        "",
        f"- recommended latest main cumulative merged pool: `{latest_main}`",
        f"- main cumulative base corpora: `{len(recommended_pools['main_cumulative_base_assets'])}`",
        f"- caveat cumulative corpora: `{len(recommended_pools['caveat_cumulative_assets'])}`",
        f"- proxy-only corpora: `{len(recommended_pools['proxy_only_assets'])}`",
        f"- metadata-only registries: `{len(recommended_pools['metadata_only_assets'])}`",
        f"- provenance-only corpora: `{len(recommended_pools['provenance_only_assets'])}`",
        f"- materialized recommended pools: `{len(recommended_pools['materialized_recommended_pools'])}`",
        f"- pending numeric extraction registries: `{len(recommended_pools['pending_numeric_extraction_assets'])}`",
        "",
        "## Layer Totals",
        "",
    ]
    for _, row in pool_totals.iterrows():
        lines.append(
            f"- `{row['layer']}`: `{int(row['n_assets'])}` assets, "
            f"`{int(row['n_total_curves'])}` curves, `{int(row['n_total_points'])}` points"
        )
    lines.extend(
        [
            "",
            "## Recommended Pools",
            "",
            f"- `main_cumulative_base_assets`: {', '.join(recommended_pools['main_cumulative_base_assets'])}",
            f"- `caveat_cumulative_assets`: {', '.join(recommended_pools['caveat_cumulative_assets'])}",
            f"- `proxy_only_assets`: {', '.join(recommended_pools['proxy_only_assets'])}",
            f"- `metadata_only_assets`: {', '.join(recommended_pools['metadata_only_assets'])}",
            f"- `provenance_only_assets`: {', '.join(recommended_pools['provenance_only_assets'])}",
            f"- `materialized_recommended_pools`: {', '.join(recommended_pools['materialized_recommended_pools'])}",
            f"- `pending_numeric_extraction_assets`: {', '.join(recommended_pools['pending_numeric_extraction_assets'])}",
            "",
            "## Asset Notes",
            "",
        ]
    )
    for _, row in registry.iterrows():
        lines.extend(
            [
                f"### {row['asset_name']}",
                "",
                f"- layer: `{row['layer']}`",
                f"- n_total_curves: `{int(row['n_total_curves'])}`",
                f"- n_total_points: `{int(row['n_total_points'])}`",
                f"- rationale: {row['layer_rationale']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    asset_registry = pd.read_csv(args.repo_root / "outputs" / "127_release_data_asset_registry" / "asset_registry.csv")
    intake_registry = pd.read_csv(
        args.repo_root / "outputs" / "128_external_source_intake_registry" / "source_intake_registry.csv"
    )
    intake_map = intake_registry.set_index("source_name").to_dict(orient="index")

    rows: list[dict[str, object]] = []
    for _, row in asset_registry.iterrows():
        layer, rationale = classify_asset(str(row["asset_name"]), str(row["asset_type"]))
        intake = intake_map.get(str(row["asset_name"]), {})
        rows.append(
            {
                "asset_name": row["asset_name"],
                "asset_type": row["asset_type"],
                "layer": layer,
                "layer_rationale": rationale,
                "curve_complete": bool(row["curve_complete"]),
                "n_total_curves": int(row["n_total_curves"]),
                "n_total_points": int(row["n_total_points"]),
                "n_total_formulations": int(row["n_total_formulations"]),
                "source_dataset": row["source_dataset"],
                "source_status": intake.get("status", pd.NA),
                "manifest_path": row["manifest_path"],
                "notes": row["notes"],
            }
        )

    registry = pd.DataFrame(rows).sort_values(["layer", "asset_name"]).reset_index(drop=True)
    registry.to_csv(args.outdir / "layered_asset_registry.csv", index=False)

    pool_totals = (
        registry.groupby("layer", dropna=False)[["n_total_curves", "n_total_points"]]
        .sum()
        .reset_index()
        .merge(registry.groupby("layer", dropna=False).size().rename("n_assets").reset_index(), on="layer")
        .sort_values("layer")
        .reset_index(drop=True)
    )

    recommended_pools = {
        "recommended_latest_main_cumulative_pool": "release_main_cumulative_v1",
        "main_cumulative_base_assets": registry.loc[
            registry["layer"] == "main_cumulative_base", "asset_name"
        ].tolist(),
        "caveat_cumulative_assets": registry.loc[
            registry["layer"] == "caveat_cumulative", "asset_name"
        ].tolist(),
        "proxy_only_assets": registry.loc[registry["layer"] == "proxy_only", "asset_name"].tolist(),
        "metadata_only_assets": registry.loc[registry["layer"] == "metadata_only", "asset_name"].tolist(),
        "provenance_only_assets": registry.loc[registry["layer"] == "provenance_only", "asset_name"].tolist(),
        "materialized_recommended_pools": registry.loc[
            registry["layer"] == "materialized_recommended_pool", "asset_name"
        ].tolist(),
        "pending_numeric_extraction_assets": registry.loc[
            registry["layer"] == "pending_numeric_extraction", "asset_name"
        ].tolist(),
        "merged_history_assets": registry.loc[registry["layer"] == "merged_history", "asset_name"].tolist(),
    }
    (args.outdir / "recommended_pools.json").write_text(
        json.dumps(recommended_pools, indent=2),
        encoding="utf-8",
    )

    write_summary(
        registry=registry,
        pool_totals=pool_totals,
        recommended_pools=recommended_pools,
        out_path=args.outdir / "summary.md",
    )

    print(f"[layered-release-corpus-registry] wrote outputs to {args.outdir}")
    print(
        "[layered-release-corpus-registry] "
        f"assets={len(registry)} layers={registry['layer'].nunique()} "
        f"recommended_main={recommended_pools['recommended_latest_main_cumulative_pool']}"
    )


if __name__ == "__main__":
    main()
