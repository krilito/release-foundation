from __future__ import annotations

"""
128_external_source_intake_registry.py

Consume:
- data/external/methylcellulose_hydrogel_zenodo_17360516/*
- data/external/bmp2_hydrogel_zenodo_18279506/*
- data/external/phage_hydrogel_zenodo_17252016/*
- data/external/mendeley_degrapol_release/*
- data/external/alginate_microbeads_mendeley_hgtphykjnb/*
- data/external/kvv2jpjpz7_mendeley/*
- data/external/wtnjj6smjd_mendeley/*
- data/external/cs6x4f86f2_mendeley/*
- data/external/8d3973kgb3_mendeley/*
- data/external/9md8g25gnx_mendeley/*
- data/external/mtds4ckns5_mendeley/*
- data/external/jk7j38n2rc_mendeley/*
- data/external/kbwcw7w4rn_mendeley/*
- data/external/h4tt4433w9_mendeley/*
- data/external/z7s49sgkc8_mendeley/*
- data/external/hs5bgtstk3_mendeley/*
- outputs/123_release_corpus_v1/manifest.json
- outputs/124_external_liposome_ivr/manifest.json
- outputs/131_external_bmp2_hydrogel_zenodo/manifest.json
- outputs/133_external_degrapol_mendeley/manifest.json
- outputs/135_external_timp1_mendeley/manifest.json
- outputs/137_external_alginate_mendeley/manifest.json
- outputs/139_external_chitosan_zeolite_mendeley/manifest.json
- outputs/140_external_starch_mendeley/manifest.json
- outputs/142_external_nt3_hydrogel_mendeley/manifest.json
- outputs/144_external_caseinate_gallic_mendeley/manifest.json
- outputs/146_external_caseinate_guar_proxy_mendeley/manifest.json
- outputs/150_external_laponite_opju_registry/manifest.json
- outputs/151_external_laponite_nh2_mendeley/manifest.json
- outputs/152_external_laponite_mendeley/manifest.json
- outputs/153_external_cure_proxy_mendeley/manifest.json
- outputs/154_external_golfball_microspheres_mendeley/manifest.json
- outputs/155_external_plga_dox_mendeley/manifest.json
- outputs/156_external_halloysite_origin_mendeley/manifest.json
- outputs/126_external_nanomed_registry/manifest.json

Produce:
- outputs/128_external_source_intake_registry/source_intake_registry.csv
- outputs/128_external_source_intake_registry/summary.md

Expected runtime:
- < 5 s
"""

import argparse
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
        default=repo_root / "outputs" / "128_external_source_intake_registry",
        help="output directory",
    )
    return parser.parse_args()


def classify_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    if path.suffix.lower() != ".csv":
        return "non_csv"
    text = path.read_text(encoding="utf-8", errors="replace")
    head = text[:300].lower()
    if "<html" in head and "403 forbidden" in head:
        return "blocked_html"
    if "," in text[:300] or ";" in text[:300]:
        return "csv_like"
    return "unknown"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    methyl_dir = args.repo_root / "data" / "external" / "methylcellulose_hydrogel_zenodo_17360516"
    methyl_files = sorted(methyl_dir.glob("*.csv"))
    methyl_status = [classify_file(path) for path in methyl_files]
    bmp2_dir = args.repo_root / "data" / "external" / "bmp2_hydrogel_zenodo_18279506"
    bmp2_files = sorted(bmp2_dir.glob("*.csv"))
    bmp2_status = [classify_file(path) for path in bmp2_files]
    phage_dir = args.repo_root / "data" / "external" / "phage_hydrogel_zenodo_17252016"
    phage_files = sorted(phage_dir.glob("*.xlsx"))
    degrapol_dir = args.repo_root / "data" / "external" / "mendeley_degrapol_release"
    degrapol_files = sorted(degrapol_dir.glob("*.xlsx"))
    alginate_dir = args.repo_root / "data" / "external" / "alginate_microbeads_mendeley_hgtphykjnb"
    alginate_files = sorted(alginate_dir.glob("*.xlsx"))
    chitosan_dir = args.repo_root / "data" / "external" / "kvv2jpjpz7_mendeley"
    chitosan_files = sorted(chitosan_dir.glob("*.xlsx"))
    starch_dir = args.repo_root / "data" / "external" / "wtnjj6smjd_mendeley"
    starch_files = sorted(starch_dir.glob("*.xlsx"))
    nt3_dir = args.repo_root / "data" / "external" / "cs6x4f86f2_mendeley"
    nt3_files = sorted(nt3_dir.rglob("*.xlsx"))
    caseinate_dir = args.repo_root / "data" / "external" / "8d3973kgb3_mendeley"
    caseinate_files = sorted(caseinate_dir.rglob("*.xlsx"))
    guar_dir = args.repo_root / "data" / "external" / "9md8g25gnx_mendeley"
    guar_files = sorted(guar_dir.rglob("*.xlsx"))
    laponite_dir = args.repo_root / "data" / "external" / "mtds4ckns5_mendeley"
    laponite_files = sorted(laponite_dir.rglob("*"))
    cure_dir = args.repo_root / "data" / "external" / "jk7j38n2rc_mendeley"
    cure_files = sorted(cure_dir.rglob("*.xlsx"))
    golfball_dir = args.repo_root / "data" / "external" / "kbwcw7w4rn_mendeley"
    golfball_files = sorted(golfball_dir.rglob("*.xlsx"))
    plga_dox_dir = args.repo_root / "data" / "external" / "h4tt4433w9_mendeley"
    plga_dox_files = sorted(plga_dox_dir.glob("*.xlsx"))
    halloysite_dir = args.repo_root / "data" / "external" / "z7s49sgkc8_mendeley"
    halloysite_opj = sorted(halloysite_dir.rglob("Release.opj"))
    phb_docx_dir = args.repo_root / "data" / "external" / "hs5bgtstk3_mendeley"
    phb_docx_files = sorted(phb_docx_dir.rglob("*.docx"))

    rows = [
        {
            "source_name": "release_corpus_v1",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "123_release_corpus_v1"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized PLGA corpus already usable",
        },
        {
            "source_name": "bannigan_lai_181_raw",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "129_bannigan_lai_181_raw"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "raw-source 181 spreadsheet standardized and comparable to internal181",
        },
        {
            "source_name": "external_liposome_ivr",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "124_external_liposome_ivr"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized liposome IVR corpus already usable",
        },
        {
            "source_name": "external_bmp2_hydrogel_zenodo",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "131_external_bmp2_hydrogel_zenodo"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized BMP-2 hydrogel release corpus already usable",
        },
        {
            "source_name": "external_nanomed_registry",
            "source_type": "metadata_registry",
            "local_path": str(args.repo_root / "outputs" / "126_external_nanomed_registry"),
            "status": "metadata_only",
            "file_count": 4,
            "valid_curve_files": 0,
            "blocked_files": 0,
            "notes": "backend descriptors are present, but both data/drug_release and liposome_IVR.db are missing from the local snapshot",
        },
        {
            "source_name": "plos_figshare_28819167",
            "source_type": "public_curve_source_attempt",
            "local_path": str(args.repo_root / "data" / "external" / "plos_figshare_28819167"),
            "status": "blocked_or_empty",
            "file_count": 1,
            "valid_curve_files": 0,
            "blocked_files": 1,
            "notes": "download target resolved but current local xlsx is zero bytes",
        },
        {
            "source_name": "methylcellulose_hydrogel_zenodo_17360516",
            "source_type": "public_curve_source_attempt",
            "local_path": str(methyl_dir),
            "status": "blocked_by_provider",
            "file_count": len(methyl_files),
            "valid_curve_files": sum(status == "csv_like" for status in methyl_status),
            "blocked_files": sum(status == "blocked_html" for status in methyl_status),
            "notes": "Zenodo file requests returned 403 HTML payloads instead of CSV content",
        },
        {
            "source_name": "bmp2_hydrogel_zenodo_18279506",
            "source_type": "public_curve_source_attempt",
            "local_path": str(bmp2_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(bmp2_files),
            "valid_curve_files": sum(status == "csv_like" for status in bmp2_status),
            "blocked_files": sum(status == "blocked_html" for status in bmp2_status),
            "notes": "Zenodo API file downloads succeeded and were standardized into an external hydrogel corpus",
        },
        {
            "source_name": "phage_hydrogel_zenodo_17252016",
            "source_type": "public_source_audited_not_primary_release",
            "local_path": str(phage_dir),
            "status": "downloaded_but_not_release_curve_corpus",
            "file_count": len(phage_files),
            "valid_curve_files": len(phage_files),
            "blocked_files": 0,
            "notes": "downloaded Excel files mostly contain gelation, rheology, viability, and phage activity tables rather than a clean sustained-release curve corpus",
        },
        {
            "source_name": "mendeley_degrapol_release",
            "source_type": "public_curve_source_attempt",
            "local_path": str(degrapol_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(degrapol_files),
            "valid_curve_files": len(degrapol_files),
            "blocked_files": 0,
            "notes": "Mendeley public API downloads succeeded; both 4hgmtc5ph3 and 3cm7sytr3r were standardized, with the TIMP-1 workbook carried as ready_with_caveat",
        },
        {
            "source_name": "external_degrapol_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "133_external_degrapol_mendeley"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized DegraPol release corpus already usable",
        },
        {
            "source_name": "external_timp1_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "135_external_timp1_mendeley"),
            "status": "ready_with_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "anonymous 10-row release blocks standardized as mean curves; workbook lacks reliable explicit group labels",
        },
        {
            "source_name": "alginate_microbeads_mendeley_hgtphykjnb",
            "source_type": "public_curve_source_attempt",
            "local_path": str(alginate_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(alginate_files),
            "valid_curve_files": len(alginate_files),
            "blocked_files": 0,
            "notes": "Mendeley public API workbook download succeeded and was standardized from figure-sheet formulation means under pH1.2 and pH6.8 conditions",
        },
        {
            "source_name": "external_alginate_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "137_external_alginate_mendeley"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized alginate microbead 5-FU release corpus already usable",
        },
        {
            "source_name": "kvv2jpjpz7_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(chitosan_dir),
            "status": "downloaded_and_standardized_with_measurement_caveat",
            "file_count": len(chitosan_files),
            "valid_curve_files": len(chitosan_files),
            "blocked_files": 0,
            "notes": "downloaded via Mendeley zip endpoint and standardized as a time-resolved absorbance release-proxy source under pH 4.0, 5.5, and 7.4 conditions",
        },
        {
            "source_name": "external_chitosan_zeolite_mendeley",
            "source_type": "standardized_curve_corpus_proxy",
            "local_path": str(args.repo_root / "outputs" / "139_external_chitosan_zeolite_mendeley"),
            "status": "ready_with_measurement_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "time-resolved absorbance proxy curves standardized from the release workbook; intentionally not merged into the main cumulative-release pool",
        },
        {
            "source_name": "wtnjj6smjd_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(starch_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(starch_files),
            "valid_curve_files": len(starch_files),
            "blocked_files": 0,
            "notes": "downloaded via Mendeley zip endpoint and standardized as a cumulative-release corpus with stitched gastric-to-intestinal phase timing",
        },
        {
            "source_name": "external_starch_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "140_external_starch_mendeley"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized starch nanoparticle cumulative-release corpus already usable",
        },
        {
            "source_name": "cs6x4f86f2_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(nt3_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(nt3_files),
            "valid_curve_files": len(nt3_files),
            "blocked_files": 0,
            "notes": "downloaded via Mendeley zip endpoint and standardized as a cumulative-release hydrogel corpus with assay-preserved NT-3 and FITC-Lysozyme release curves",
        },
        {
            "source_name": "external_nt3_hydrogel_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "142_external_nt3_hydrogel_mendeley"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized NT-3 hydrogel cumulative-release corpus already usable; assay identity is preserved per curve",
        },
        {
            "source_name": "8d3973kgb3_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(caseinate_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(caseinate_files),
            "valid_curve_files": len(caseinate_files),
            "blocked_files": 0,
            "notes": "downloaded via Mendeley zip endpoint and standardized using the workbook's experimental Mt/Minf release values",
        },
        {
            "source_name": "external_caseinate_gallic_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "144_external_caseinate_gallic_mendeley"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "standardized caseinate-gallic film cumulative-release corpus already usable",
        },
        {
            "source_name": "9md8g25gnx_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(guar_dir),
            "status": "downloaded_and_standardized_with_measurement_caveat",
            "file_count": len(guar_files),
            "valid_curve_files": len(guar_files),
            "blocked_files": 0,
            "notes": "downloaded via Mendeley zip endpoint and standardized as a time-resolved concentration proxy source across multiple caseinate-guar-gallic formulations",
        },
        {
            "source_name": "external_caseinate_guar_proxy_mendeley",
            "source_type": "standardized_curve_corpus_proxy",
            "local_path": str(args.repo_root / "outputs" / "146_external_caseinate_guar_proxy_mendeley"),
            "status": "ready_with_measurement_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "time-resolved concentration proxy curves standardized from the release workbook; intentionally not merged into the main cumulative-release pool",
        },
        {
            "source_name": "mtds4ckns5_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(laponite_dir),
            "status": "downloaded_and_standardized_with_origin_caveat",
            "file_count": len([p for p in laponite_files if p.is_file()]),
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "Mendeley bundle downloaded; the Origin project was accessed numerically via COM/LabTalk and standardized into a four-curve cumulative corpus with hidden-dataset label caveats",
        },
        {
            "source_name": "external_laponite_opju_registry",
            "source_type": "pending_extraction_registry",
            "local_path": str(args.repo_root / "outputs" / "150_external_laponite_opju_registry"),
            "status": "registry_retained_after_extraction",
            "file_count": 3,
            "valid_curve_files": 0,
            "blocked_files": 0,
            "notes": "registry-only asset listing the pre-extraction four-series hypothesis from visible strings; retained after COM-based numeric recovery",
        },
        {
            "source_name": "external_laponite_nh2_mendeley",
            "source_type": "standardized_curve_corpus_partial",
            "local_path": str(args.repo_root / "outputs" / "151_external_laponite_nh2_mendeley"),
            "status": "superseded_partial_extraction",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "earlier NH2-only extraction retained for provenance; superseded by the full four-curve laponite corpus",
        },
        {
            "source_name": "external_laponite_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "152_external_laponite_mendeley"),
            "status": "ready_with_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "four-curve cumulative laponite corpus recovered from the Origin project; NH2 labels are explicit and plain-laponita labels come from hidden Book2A datasets matched through visible-string evidence",
        },
        {
            "source_name": "jk7j38n2rc_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(cure_dir),
            "status": "downloaded_and_standardized_with_measurement_caveat",
            "file_count": len(cure_files),
            "valid_curve_files": len(cure_files),
            "blocked_files": 0,
            "notes": "downloaded via the Mendeley public zip API and standardized as a time-resolved HPLC release-level proxy source across four media/GSH conditions",
        },
        {
            "source_name": "external_cure_proxy_mendeley",
            "source_type": "standardized_curve_corpus_proxy",
            "local_path": str(args.repo_root / "outputs" / "153_external_cure_proxy_mendeley"),
            "status": "ready_with_measurement_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "time-resolved HPLC release-proxy curves standardized from Drug Release.xlsx; intentionally not merged into the main cumulative-release pool",
        },
        {
            "source_name": "kbwcw7w4rn_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(golfball_dir),
            "status": "downloaded_and_standardized",
            "file_count": len(golfball_files),
            "valid_curve_files": len(golfball_files),
            "blocked_files": 0,
            "notes": "downloaded via the Mendeley public zip API and standardized as an external cumulative microsphere release corpus from the Fig. 7AB panel",
        },
        {
            "source_name": "external_golfball_microspheres_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "154_external_golfball_microspheres_mendeley"),
            "status": "ready",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "cumulative microsphere release curves standardized from Data2.xlsx; eligible for the main cumulative pool",
        },
        {
            "source_name": "h4tt4433w9_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(plga_dox_dir),
            "status": "downloaded_and_standardized_with_corpus_caveat",
            "file_count": len(plga_dox_files),
            "valid_curve_files": len(plga_dox_files),
            "blocked_files": 0,
            "notes": "downloaded via the Mendeley public zip API and standardized as a cumulative release corpus containing both PLGA nanoparticle formulations and a free-drug comparator",
        },
        {
            "source_name": "external_plga_dox_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "155_external_plga_dox_mendeley"),
            "status": "ready_with_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "cumulative PLGA doxorubicin release curves standardized from drug release profile.xlsx; kept in caveat cumulative because the corpus includes a free-drug control",
        },
        {
            "source_name": "z7s49sgkc8_mendeley",
            "source_type": "public_curve_source_attempt",
            "local_path": str(halloysite_dir),
            "status": "downloaded_and_standardized_with_origin_caveat",
            "file_count": len(halloysite_opj),
            "valid_curve_files": len(halloysite_opj),
            "blocked_files": 0,
            "notes": "downloaded as a nested zip bundle; the release data were recovered from Release.opj via Origin COM and standardized as a cumulative release corpus with compact worksheet condition labels",
        },
        {
            "source_name": "external_halloysite_origin_mendeley",
            "source_type": "standardized_curve_corpus",
            "local_path": str(args.repo_root / "outputs" / "156_external_halloysite_origin_mendeley"),
            "status": "ready_with_caveat",
            "file_count": 6,
            "valid_curve_files": 1,
            "blocked_files": 0,
            "notes": "cumulative halloysite/chitosan release curves standardized from Release.opj; kept in caveat cumulative because the project exposes only compact condition labels such as 0mM, 2mM, and 4mM",
        },
        {
            "source_name": "hs5bgtstk3_mendeley",
            "source_type": "public_source_document_only",
            "local_path": str(phb_docx_dir),
            "status": "downloaded_docx_without_numeric_tables",
            "file_count": len(phb_docx_files),
            "valid_curve_files": 0,
            "blocked_files": 0,
            "notes": "downloaded source contains a research-data docx describing the PHB dissolution experiment, but no tabulated numeric release curves were found",
        },
    ]

    registry = pd.DataFrame(rows)
    registry.to_csv(args.outdir / "source_intake_registry.csv", index=False)

    lines = [
        "# external_source_intake_registry",
        "",
        "Current intake status of supplemented release-data sources.",
        "",
    ]
    for _, row in registry.iterrows():
        lines.extend(
            [
                f"## {row['source_name']}",
                "",
                f"- source_type: `{row['source_type']}`",
                f"- status: `{row['status']}`",
                f"- local_path: `{row['local_path']}`",
                f"- file_count: `{int(row['file_count'])}`",
                f"- valid_curve_files: `{int(row['valid_curve_files'])}`",
                f"- blocked_files: `{int(row['blocked_files'])}`",
                f"- notes: {row['notes']}",
                "",
            ]
        )
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[external-source-intake-registry] wrote outputs to {args.outdir}")
    print(f"[external-source-intake-registry] sources={len(registry)}")


if __name__ == "__main__":
    main()
