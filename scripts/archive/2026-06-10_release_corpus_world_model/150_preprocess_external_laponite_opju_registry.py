from __future__ import annotations

"""
150_preprocess_external_laponite_opju_registry.py

Consume:
- data/external/mtds4ckns5_mendeley/Organic-inorganic hybrid based on Laponite as a pl/Release_pH5andpH7.opju

Produce:
- outputs/150_external_laponite_opju_registry/curve_registry.csv
- outputs/150_external_laponite_opju_registry/manifest.json
- outputs/150_external_laponite_opju_registry/summary.md

Expected runtime:
- < 5 s
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "150_external_laponite_opju_registry",
    )
    return parser.parse_args()


def extract_visible_strings(opju_path: Path) -> list[str]:
    data = opju_path.read_bytes()
    return [s.decode("latin1", errors="ignore") for s in re.findall(rb"[ -~]{3,}", data)]


def build_curve_registry(strings: list[str]) -> pd.DataFrame:
    joined = "\n".join(strings)
    candidates = [
        ("laponita pH 7", "pH 7", "laponita"),
        ("laponita pH 5", "pH 5", "laponita"),
        ("laponita NH2 (2)\npH 7", "pH 7", "laponita NH2 (2)"),
        ("laponita NH2 (2)\npH 5", "pH 5", "laponita NH2 (2)"),
    ]
    rows: list[dict[str, object]] = []
    for raw_label, ph_label, material_label in candidates:
        if raw_label in joined:
            rows.append(
                {
                    "source_dataset": "laponite_release_mendeley_mtds4ckns5",
                    "series_label": raw_label.replace("\n", " / "),
                    "material_label": material_label,
                    "condition_label": ph_label,
                    "inferred_release_measure": "drug released (%)",
                    "time_axis_label": "Time (h)",
                    "numeric_series_extracted": False,
                    "evidence_type": "opju_visible_strings",
                }
            )
    return pd.DataFrame(rows)


def write_summary(registry: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# external_laponite_opju_registry",
        "",
        "Registry-only preprocessing for Mendeley dataset `mtds4ckns5`.",
        "The available release source is an Origin project (`.opju`) that exposes release labels and axes",
        "through visible strings, but numeric worksheet extraction has not yet been completed.",
        "",
        f"- inferred release series: `{len(registry)}`",
        "",
        "## Important Caveat",
        "",
        "- This asset is not a curve corpus yet.",
        "- It is a pending extraction registry for an Origin project that appears to contain release curves with `Drug released (%)` vs `Time (h)`.",
        "",
        "## Inferred Series",
        "",
    ]
    for _, row in registry.iterrows():
        lines.append(
            f"- `{row['material_label']}` under `{row['condition_label']}` "
            f"(evidence: `{row['evidence_type']}`)"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    opju_path = (
        args.repo_root
        / "data"
        / "external"
        / "mtds4ckns5_mendeley"
        / "Organic-inorganic hybrid based on Laponite as a pl"
        / "Release_pH5andpH7.opju"
    )
    strings = extract_visible_strings(opju_path)
    registry = build_curve_registry(strings)
    registry.to_csv(args.outdir / "curve_registry.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_name": "external_laponite_opju_registry",
                "source_dataset": "laponite_release_mendeley_mtds4ckns5",
                "mendeley_dataset_id": "mtds4ckns5",
                "asset_kind": "opju_pending_extraction_registry",
                "n_total_curves": int(len(registry)),
                "n_total_points": 0,
                "n_total_formulations": 0,
                "curve_complete": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(registry, args.outdir / "summary.md")

    print(f"[external-laponite-opju-registry] wrote outputs to {args.outdir}")
    print(f"[external-laponite-opju-registry] inferred_series={len(registry)}")


if __name__ == "__main__":
    main()
