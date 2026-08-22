"""
87 - Attach split-conformal interval calibration to shared posterior-family objects.

Purpose:
    Script 86 exports raw low-rank family structure from FIB / ensemble UQ.
    Script 76 measures split-conformal interval inflation in release space.
    This bridge keeps those two ideas separate but serialized together:

        latent feasible family
            + calibrated interval scales in decoder-output space
            -> one shared deployable artifact

Important honesty rule:
    The conformal scales here do NOT recalibrate latent geometry. They are
    post-hoc interval multipliers defined in release-trajectory space. This
    script records them on the family object so benchmark outputs and
    deployment-side logic can consume one artifact without pretending the
    latent family itself was retrained.

Consumes:
    - outputs/86_release_casp_family_export/fib/<dataset>/<scheme>/*.json
    - outputs/76_casp_conformal_recalibration/<dataset>/<scheme>/per_curve.csv

Produces:
    outputs/87_release_conformal_family_export/<dataset>/<scheme>/
        *.json
        summary.csv
        summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_posterior_family import ReleasePosteriorFamily


DEFAULT_LOCAL_BINS = [0.0, 0.08, 0.20, 0.50, 1.000001]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--per-curve-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha-cov", type=float, default=0.10)
    parser.add_argument("--local-bins", type=float, nargs="+", default=DEFAULT_LOCAL_BINS)
    return parser.parse_args()


def _clean_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _calibration_payload(row: pd.Series, alpha_cov: float, local_bins: list[float]) -> dict[str, object]:
    return {
        "calibration_method": "split_conformal_multiplicative_interval_scale",
        "calibration_domain": "decoder_output_cumulative_fraction",
        "alpha_cov": float(alpha_cov),
        "nominal_coverage": float(1.0 - alpha_cov),
        "global_scale": float(row["q_global"]),
        "local_scale_bin_edges_t_norm": [float(x) for x in local_bins],
        "local_scale_values": [
            float(row["q_local_0_008"]),
            float(row["q_local_008_020"]),
            float(row["q_local_020_050"]),
            float(row["q_local_050_100"]),
        ],
        "empirical_curve_metrics": {
            "cov90_raw": float(row["cov90_raw"]),
            "cov90_global": float(row["cov90_global"]),
            "cov90_local": float(row["cov90_local"]),
            "width90_raw": float(row["width90_raw"]),
            "width90_global": float(row["width90_global"]),
            "width90_local": float(row["width90_local"]),
        },
        "source_script": "76_casp_conformal_recalibration.py",
        "calibration_note": (
            "Scales apply to predictive intervals in release space; "
            "latent z_center / z_basis / z_scale are unchanged."
        ),
    }


def _load_per_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "dataset",
        "scheme",
        "fold",
        "fid",
        "q_global",
        "q_local_0_008",
        "q_local_008_020",
        "q_local_020_050",
        "q_local_050_100",
        "cov90_raw",
        "cov90_global",
        "cov90_local",
        "width90_raw",
        "width90_global",
        "width90_local",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"per_curve csv missing required columns: {missing}")
    return df


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_curve = _load_per_curve(args.per_curve_csv)
    calibrations = {
        (int(row["fold"]), int(row["fid"])): row
        for _, row in per_curve.iterrows()
    }

    summary_rows: list[dict[str, object]] = []
    summary_lines = ["=== 87 -- conformal family export ===", ""]
    n_written = 0
    n_missing = 0

    for path in sorted(args.family_dir.glob("*.json")):
        family = ReleasePosteriorFamily.from_json(path)
        fid = family.metadata.get("fid")
        fold = family.observation_context.get("fold")
        if fid is None or fold is None:
            n_missing += 1
            continue
        key = (int(fold), int(fid))
        row = calibrations.get(key)
        if row is None:
            n_missing += 1
            continue

        calibrated = family.with_calibration(
            _calibration_payload(row=row, alpha_cov=args.alpha_cov, local_bins=args.local_bins)
        )
        outpath = args.outdir / path.name
        calibrated.to_json(outpath)
        n_written += 1

        item = calibrated.summary_dict()
        item.update(
            {
                "curve_id": _clean_scalar(calibrated.metadata.get("curve_id")),
                "fid": int(fid),
                "fold": int(fold),
                "global_scale": float(row["q_global"]),
                "cov90_raw": float(row["cov90_raw"]),
                "cov90_global": float(row["cov90_global"]),
                "cov90_local": float(row["cov90_local"]),
                "width90_raw": float(row["width90_raw"]),
                "width90_global": float(row["width90_global"]),
                "width90_local": float(row["width90_local"]),
            }
        )
        summary_rows.append(item)
        summary_lines.extend(
            [
                f"{path.stem}:",
                f"  mechanism_id    : {item['mechanism_id']}",
                f"  rank            : {item['rank']}",
                f"  global_scale    : {item['global_scale']:.3f}",
                f"  cov raw/g/l     : {item['cov90_raw']:.3f}/{item['cov90_global']:.3f}/{item['cov90_local']:.3f}",
                f"  width raw/g/l   : {item['width90_raw']:.3f}/{item['width90_global']:.3f}/{item['width90_local']:.3f}",
                "",
            ]
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.outdir / "summary.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "curve_id",
                "fid",
                "fold",
                "mechanism_id",
                "rank",
                "global_scale",
                "cov90_raw",
                "cov90_global",
                "cov90_local",
                "width90_raw",
                "width90_global",
                "width90_local",
            ]
        ).to_csv(args.outdir / "summary.csv", index=False)

    summary_lines.extend(
        [
            f"written_families: {n_written}",
            f"missing_matches : {n_missing}",
        ]
    )
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
