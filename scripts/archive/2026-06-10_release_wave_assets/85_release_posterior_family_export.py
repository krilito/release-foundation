"""
85 - Export benchmark-side posterior-family objects from retained latent states.

Purpose:
    Connect the shared `ReleasePosteriorFamily` schema to the existing
    benchmark/export stack for mechanism routes that actually retain a latent
    state per curve. This script is intentionally honest:

    - if a route only has a point latent state, emit a rank-0 point estimate
    - do not fabricate low-rank uncertainty that is not present in the source
    - do not export direct-Q routes as posterior-family objects

Consumes:
    fold_curve_metadata.csv from scripts 79 / 83
    optional metadata.csv with per-curve assay / formulation context

Produces:
    outputs/.../*.json
    outputs/.../summary.csv
    outputs/.../summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_posterior_family import ReleasePosteriorFamily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-meta-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mechanism-id", type=str, required=True)
    parser.add_argument("--decoder-handle", type=str, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--time-unit", type=str, default="")
    parser.add_argument("--early-window", type=float, default=np.nan)
    parser.add_argument("--latent-mode", choices=("auto", "plga_theta", "weibull_alpha_beta"), default="auto")
    return parser.parse_args()


def _infer_latent_mode(df: pd.DataFrame) -> str:
    theta_cols = sorted([c for c in df.columns if c.startswith("theta_")])
    if theta_cols:
        return "plga_theta"
    if {"alpha_pred", "beta_pred"}.issubset(df.columns):
        return "weibull_alpha_beta"
    raise ValueError("Could not infer latent mode from fold metadata columns")


def _latent_vector(row: pd.Series, latent_mode: str) -> tuple[list[float], dict[str, str]]:
    if latent_mode == "plga_theta":
        theta_cols = sorted([c for c in row.index if c.startswith("theta_")], key=lambda x: int(x.split("_")[1]))
        values = [float(row[c]) for c in theta_cols]
        if not np.all(np.isfinite(values)):
            raise ValueError("PLGA theta export requires finite theta columns")
        return values, {"latent_coordinates": "theta_like_kinetic_state"}
    if latent_mode == "weibull_alpha_beta":
        alpha = float(row["alpha_pred"])
        beta = float(row["beta_pred"])
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError("Weibull export requires finite alpha_pred and beta_pred")
        return [alpha, beta], {"latent_coordinates": "alpha_beta"}
    raise ValueError(f"Unsupported latent mode: {latent_mode}")


def _clean_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fold_meta = pd.read_csv(args.fold_meta_csv)
    metadata = pd.read_csv(args.metadata_csv) if args.metadata_csv is not None else None
    latent_mode = _infer_latent_mode(fold_meta) if args.latent_mode == "auto" else args.latent_mode

    meta_map: dict[str, pd.Series] = {}
    if metadata is not None and "curve_id" in metadata.columns:
        meta_map = {str(row["curve_id"]): row for _, row in metadata.iterrows()}

    summary_rows: list[dict[str, object]] = []
    summary_lines = ["=== 85 -- release posterior-family export ===", ""]

    for _, row in fold_meta.iterrows():
        curve_id = str(row["curve_id"])
        z_center, latent_meta = _latent_vector(row, latent_mode)
        meta_row = meta_map.get(curve_id)
        assay_context = {
            "time_unit": args.time_unit or (_clean_scalar(meta_row.get("time_unit_standard")) if meta_row is not None and "time_unit_standard" in meta_row else ""),
            "early_window": float(args.early_window) if np.isfinite(args.early_window) else (_clean_scalar(meta_row.get("early_window_days")) if meta_row is not None and "early_window_days" in meta_row else _clean_scalar(meta_row.get("early_window_h")) if meta_row is not None and "early_window_h" in meta_row else None),
            "release_object": "cumulative_fraction",
        }
        observation_context = {
            "fold": int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else None,
            "route_family": _clean_scalar(row.get("route_family")),
            "resolved_method": _clean_scalar(row.get("resolved_method") or row.get("method")),
            "selected_method": _clean_scalar(row.get("selected_method")),
            "family_kind": "point_estimate_from_benchmark_export",
        }
        metadata_payload = {
            "source_fold_meta": str(args.fold_meta_csv),
            "curve_id": curve_id,
            "r2": _clean_scalar(row.get("r2") or row.get("full_r2")),
            "rmse": _clean_scalar(row.get("rmse") or row.get("full_rmse")),
        }
        metadata_payload.update(latent_meta)
        if meta_row is not None:
            for key in ("fid", "source_id", "drug_id", "polymer_family", "API_name", "release_method", "dataset_name", "mechanism_id"):
                if key in meta_row.index:
                    metadata_payload[key] = _clean_scalar(meta_row[key])

        family = ReleasePosteriorFamily.point_estimate(
            mechanism_id=args.mechanism_id,
            decoder_handle=args.decoder_handle,
            z_center=z_center,
            assay_context=assay_context,
            observation_context=observation_context,
            metadata=metadata_payload,
        )
        outpath = args.outdir / f"{curve_id}.json"
        family.to_json(outpath)
        item = family.summary_dict()
        item["curve_id"] = curve_id
        summary_rows.append(item)
        summary_lines.extend(
            [
                f"{curve_id}:",
                f"  mechanism_id  : {item['mechanism_id']}",
                f"  decoder_handle: {item['decoder_handle']}",
                f"  latent_dim    : {item['latent_dim']}",
                f"  rank          : {item['rank']}",
                "",
            ]
        )

    pd.DataFrame(summary_rows).to_csv(args.outdir / "summary.csv", index=False)
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
