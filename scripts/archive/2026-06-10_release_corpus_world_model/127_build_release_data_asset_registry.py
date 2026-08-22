from __future__ import annotations

"""
127_build_release_data_asset_registry.py

Consume:
- outputs/123_release_corpus_v1/manifest.json
- outputs/124_external_liposome_ivr/manifest.json
- outputs/125_release_corpus_candidates/manifest.json
- outputs/131_external_bmp2_hydrogel_zenodo/manifest.json
- outputs/132_release_corpus_candidates_v2/manifest.json
- outputs/133_external_degrapol_mendeley/manifest.json
- outputs/134_release_corpus_candidates_v3/manifest.json
- outputs/135_external_timp1_mendeley/manifest.json
- outputs/136_release_corpus_candidates_v4/manifest.json
- outputs/137_external_alginate_mendeley/manifest.json
- outputs/138_release_corpus_candidates_v5/manifest.json
- outputs/139_external_chitosan_zeolite_mendeley/manifest.json
- outputs/140_external_starch_mendeley/manifest.json
- outputs/141_release_corpus_candidates_v6/manifest.json
- outputs/142_external_nt3_hydrogel_mendeley/manifest.json
- outputs/143_release_corpus_candidates_v7/manifest.json
- outputs/144_external_caseinate_gallic_mendeley/manifest.json
- outputs/146_external_caseinate_guar_proxy_mendeley/manifest.json
- outputs/148_release_main_cumulative_v1/manifest.json
- outputs/149_release_caveat_augmented_cumulative_v1/manifest.json
- outputs/150_external_laponite_opju_registry/manifest.json
- outputs/151_external_laponite_nh2_mendeley/manifest.json
- outputs/152_external_laponite_mendeley/manifest.json
- outputs/153_external_cure_proxy_mendeley/manifest.json
- outputs/154_external_golfball_microspheres_mendeley/manifest.json
- outputs/155_external_plga_dox_mendeley/manifest.json
- outputs/156_external_halloysite_origin_mendeley/manifest.json
- outputs/145_release_corpus_candidates_v8/manifest.json
- outputs/126_external_nanomed_registry/manifest.json

Produce:
- outputs/127_release_data_asset_registry/asset_registry.csv
- outputs/127_release_data_asset_registry/summary.md

Expected runtime:
- < 5 s
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
        default=repo_root / "outputs" / "127_release_data_asset_registry",
        help="output directory",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "asset_name": "release_corpus_v1",
            "path": args.repo_root / "outputs" / "123_release_corpus_v1" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "internal181 + cross321 standardized PLGA corpus",
        },
        {
            "asset_name": "bannigan_lai_181_raw",
            "path": args.repo_root / "outputs" / "129_bannigan_lai_181_raw" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "raw-source 181 spreadsheet standardized for provenance comparison",
        },
        {
            "asset_name": "external_liposome_ivr",
            "path": args.repo_root / "outputs" / "124_external_liposome_ivr" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external liposome IVR corpus with 169 cluster-labeled curves",
        },
        {
            "asset_name": "external_bmp2_hydrogel_zenodo",
            "path": args.repo_root / "outputs" / "131_external_bmp2_hydrogel_zenodo" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external collagen-alginate BMP-2 hydrogel corpus from Zenodo record 18279506",
        },
        {
            "asset_name": "release_corpus_candidates",
            "path": args.repo_root / "outputs" / "125_release_corpus_candidates" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool for cross-system experiments",
        },
        {
            "asset_name": "release_corpus_candidates_v2",
            "path": args.repo_root / "outputs" / "132_release_corpus_candidates_v2" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, and BMP-2 hydrogel corpora",
        },
        {
            "asset_name": "external_degrapol_mendeley",
            "path": args.repo_root / "outputs" / "133_external_degrapol_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external DegraPol release kinetics corpus from Mendeley dataset 4hgmtc5ph3",
        },
        {
            "asset_name": "release_corpus_candidates_v3",
            "path": args.repo_root / "outputs" / "134_release_corpus_candidates_v3" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, hydrogel, and DegraPol release corpora",
        },
        {
            "asset_name": "external_timp1_mendeley",
            "path": args.repo_root / "outputs" / "135_external_timp1_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "cautiously standardized anonymous TIMP-1 release blocks from Mendeley dataset 3cm7sytr3r",
        },
        {
            "asset_name": "release_corpus_candidates_v4",
            "path": args.repo_root / "outputs" / "136_release_corpus_candidates_v4" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, hydrogel, DegraPol, and anonymous TIMP-1 release corpora",
        },
        {
            "asset_name": "external_alginate_mendeley",
            "path": args.repo_root / "outputs" / "137_external_alginate_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external alginate microbead 5-FU release corpus from Mendeley dataset hgtphykjnb",
        },
        {
            "asset_name": "release_corpus_candidates_v5",
            "path": args.repo_root / "outputs" / "138_release_corpus_candidates_v5" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, hydrogel, DegraPol, TIMP-1, and alginate microbead release corpora",
        },
        {
            "asset_name": "external_chitosan_zeolite_mendeley",
            "path": args.repo_root / "outputs" / "139_external_chitosan_zeolite_mendeley" / "manifest.json",
            "asset_type": "curve_corpus_proxy",
            "curve_complete": True,
            "notes": "external chitosan-zeolite release-proxy corpus from Mendeley dataset kvv2jpjpz7; standardized but kept out of the main cumulative-release pool",
        },
        {
            "asset_name": "external_starch_mendeley",
            "path": args.repo_root / "outputs" / "140_external_starch_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external starch nanoparticle release corpus from Mendeley dataset wtnjj6smjd",
        },
        {
            "asset_name": "release_corpus_candidates_v6",
            "path": args.repo_root / "outputs" / "141_release_corpus_candidates_v6" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, hydrogel, DegraPol, TIMP-1, alginate microbead, and starch nanoparticle release corpora",
        },
        {
            "asset_name": "external_nt3_hydrogel_mendeley",
            "path": args.repo_root / "outputs" / "142_external_nt3_hydrogel_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external NT-3 hydrogel release corpus from Mendeley dataset cs6x4f86f2, preserving assay identity",
        },
        {
            "asset_name": "release_corpus_candidates_v7",
            "path": args.repo_root / "outputs" / "143_release_corpus_candidates_v7" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, hydrogel, DegraPol, TIMP-1, alginate microbead, starch nanoparticle, and NT-3 hydrogel release corpora",
        },
        {
            "asset_name": "external_caseinate_gallic_mendeley",
            "path": args.repo_root / "outputs" / "144_external_caseinate_gallic_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external caseinate-gallic film release corpus from Mendeley dataset 8d3973kgb3, normalized via experimental Mt/Minf values",
        },
        {
            "asset_name": "external_golfball_microspheres_mendeley",
            "path": args.repo_root / "outputs" / "154_external_golfball_microspheres_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external cumulative microsphere release corpus from Mendeley dataset kbwcw7w4rn, standardized from the Fig. 7AB panel",
        },
        {
            "asset_name": "external_plga_dox_mendeley",
            "path": args.repo_root / "outputs" / "155_external_plga_dox_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external cumulative PLGA doxorubicin release corpus from Mendeley dataset h4tt4433w9, kept as caveat because it mixes a free-drug control with nanoparticle formulations",
        },
        {
            "asset_name": "external_halloysite_origin_mendeley",
            "path": args.repo_root / "outputs" / "156_external_halloysite_origin_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "external cumulative halloysite/chitosan release corpus from Mendeley dataset z7s49sgkc8, kept as caveat because the 0mM/2mM/4mM condition axis is not semantically expanded beyond explicit worksheet labels",
        },
        {
            "asset_name": "external_caseinate_guar_proxy_mendeley",
            "path": args.repo_root / "outputs" / "146_external_caseinate_guar_proxy_mendeley" / "manifest.json",
            "asset_type": "curve_corpus_proxy",
            "curve_complete": True,
            "notes": "external caseinate-guar film concentration-proxy corpus from Mendeley dataset 9md8g25gnx; standardized but kept out of the main cumulative-release pool",
        },
        {
            "asset_name": "release_main_cumulative_v1",
            "path": args.repo_root / "outputs" / "148_release_main_cumulative_v1" / "manifest.json",
            "asset_type": "materialized_recommended_pool",
            "curve_complete": True,
            "notes": "materialized clean cumulative pool built from the main cumulative base assets only",
        },
        {
            "asset_name": "release_caveat_augmented_cumulative_v1",
            "path": args.repo_root / "outputs" / "149_release_caveat_augmented_cumulative_v1" / "manifest.json",
            "asset_type": "materialized_recommended_pool",
            "curve_complete": True,
            "notes": "materialized cumulative pool built from the clean base assets plus caveat cumulative assets",
        },
        {
            "asset_name": "external_laponite_opju_registry",
            "path": args.repo_root / "outputs" / "150_external_laponite_opju_registry" / "manifest.json",
            "asset_type": "opju_pending_extraction_registry",
            "curve_complete": False,
            "notes": "registry retained for provenance after numeric extraction; visible strings documented the four release-series hypotheses before COM recovery",
        },
        {
            "asset_name": "external_laponite_nh2_mendeley",
            "path": args.repo_root / "outputs" / "151_external_laponite_nh2_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "partial cumulative release corpus retained for provenance; superseded by the full four-curve laponite extraction",
        },
        {
            "asset_name": "external_laponite_mendeley",
            "path": args.repo_root / "outputs" / "152_external_laponite_mendeley" / "manifest.json",
            "asset_type": "curve_corpus",
            "curve_complete": True,
            "notes": "full four-curve cumulative release corpus extracted from the Origin project, with hidden Book2A datasets recovered via LabTalk access",
        },
        {
            "asset_name": "external_cure_proxy_mendeley",
            "path": args.repo_root / "outputs" / "153_external_cure_proxy_mendeley" / "manifest.json",
            "asset_type": "curve_corpus_proxy",
            "curve_complete": True,
            "notes": "external NanoCURE HPLC release-level corpus from Mendeley dataset jk7j38n2rc; standardized as a proxy rather than a cumulative percent-release source",
        },
        {
            "asset_name": "release_corpus_candidates_v8",
            "path": args.repo_root / "outputs" / "145_release_corpus_candidates_v8" / "manifest.json",
            "asset_type": "merged_curve_corpus",
            "curve_complete": True,
            "notes": "merged candidate pool across PLGA, liposome, hydrogel, DegraPol, TIMP-1, alginate microbead, starch nanoparticle, NT-3 hydrogel, and caseinate-gallic film release corpora",
        },
        {
            "asset_name": "external_nanomed_registry",
            "path": args.repo_root / "outputs" / "126_external_nanomed_registry" / "manifest.json",
            "asset_type": "metadata_registry",
            "curve_complete": False,
            "notes": "metadata-ready, raw release curves missing from local snapshot",
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in specs:
        manifest = load_manifest(spec["path"])
        rows.append(
            {
                "asset_name": spec["asset_name"],
                "asset_type": spec["asset_type"],
                "curve_complete": spec["curve_complete"],
                "n_total_curves": manifest.get("n_total_curves", manifest.get("n_ivr_entries")),
                "n_total_points": manifest.get("n_total_points", 0),
                "n_total_formulations": manifest.get("n_total_formulations", 0),
                "source_dataset": manifest.get(
                    "source_dataset",
                    ",".join(manifest.get("sources", manifest.get("corpora", []))),
                ),
                "manifest_path": str(spec["path"]),
                "notes": spec["notes"],
            }
        )

    registry = pd.DataFrame(rows)
    registry.to_csv(args.outdir / "asset_registry.csv", index=False)

    lines = [
        "# release_data_asset_registry",
        "",
        "Current standardized release-data assets in the workspace.",
        "",
    ]
    for _, row in registry.iterrows():
        lines.extend(
            [
                f"## {row['asset_name']}",
                "",
                f"- asset_type: `{row['asset_type']}`",
                f"- curve_complete: `{row['curve_complete']}`",
                f"- n_total_curves: `{int(row['n_total_curves'])}`",
                f"- n_total_points: `{int(row['n_total_points'])}`",
                f"- n_total_formulations: `{int(row['n_total_formulations'])}`",
                f"- source_dataset: `{row['source_dataset']}`",
                f"- notes: {row['notes']}",
                "",
            ]
        )
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[release-data-asset-registry] wrote outputs to {args.outdir}")
    print(f"[release-data-asset-registry] assets={len(registry)}")


if __name__ == "__main__":
    main()
