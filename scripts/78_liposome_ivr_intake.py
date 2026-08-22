"""
78 - Intake the accelerated_IVR liposome dataset into the shared release schema.

Purpose:
    Turn the Yanes accelerated_IVR repository into benchmark-ready assets that
    match the shared release benchmark contract. This is the first non-PLGA
    bridge in the unified release program.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/78_liposome_ivr_intake/curves_long.csv
    outputs/78_liposome_ivr_intake/metadata.csv
    outputs/78_liposome_ivr_intake/theta_targets.csv
    outputs/78_liposome_ivr_intake/mechanism_metadata.json
    outputs/78_liposome_ivr_intake/summary.txt

Expected runtime:
    Seconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weibull_simulator import WeibullSimulator


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUTDIR = Path("outputs/78_liposome_ivr_intake")
FEATURE_7 = [
    "media_pH",
    "media_temp_oC",
    "drug_loading",
    "Z_average_nm",
    "API_type",
    "weighted_Mw",
    "weighted_Tm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--early-window-h", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _curve_id(source_id: int) -> str:
    return f"liposome_{source_id:03d}"


def _detect_unit_multiplier(root: Path) -> dict[int, float]:
    df = pd.read_csv(root / "data/time_units.csv")
    mapping = {"seconds": 1.0 / 3600.0, "mins": 1.0 / 60.0, "minutes": 1.0 / 60.0, "hours": 1.0}
    out: dict[int, float] = {}
    for row in df.itertuples(index=False):
        unit = str(row.Time_units).strip().lower()
        if unit not in mapping:
            raise ValueError(f"Unsupported time unit {unit!r} in {root / 'data/time_units.csv'}")
        out[int(row.ID)] = float(mapping[unit])
    return out


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    loader_54 = _load_script(
        Path(__file__).resolve().parent / "54_accelerated_ivr_weibull_forecast.py",
        "_loader_54_78",
    )
    meta, curve_map = loader_54._load_dataset(args.root, early_window_h=args.early_window_h)
    time_scale = _detect_unit_multiplier(args.root)
    sim = WeibullSimulator()

    curve_rows: list[dict[str, object]] = []
    for row in meta.itertuples(index=False):
        curve_id = _curve_id(int(row.ID))
        sub = curve_map[int(row.ID)].sort_values("time_h").copy()
        raw_t_h = sub["time_h"].to_numpy(dtype=float)
        multiplier = float(time_scale.get(int(row.ID), 1.0))
        t_hours = raw_t_h * multiplier
        q_obs = np.clip(sub["release_pct"].to_numpy(dtype=float) / 100.0, 0.0, 1.2)
        n_early = int(np.sum(t_hours <= args.early_window_h))
        n_future = int(np.sum(t_hours > args.early_window_h))
        for t_h, q in zip(t_hours.tolist(), q_obs.tolist()):
            curve_rows.append(
                {
                    "curve_id": curve_id,
                    "source_id": int(row.ID),
                    "formulation": f"formulation_{int(row.formulation_ID)}",
                    "drug": str(row.API_name),
                    "t_hours": float(t_h),
                    "Q_observed": float(q),
                    "dataset_name": "liposome_accelerated_ivr",
                    "mechanism_id": "liposome_weibull",
                    "early_window_h": float(args.early_window_h),
                    "n_early_curve": n_early,
                    "n_future_curve": n_future,
                }
            )

    curves_long = pd.DataFrame(curve_rows).sort_values(["curve_id", "t_hours"]).reset_index(drop=True)
    curves_long.to_csv(args.outdir / "curves_long.csv", index=False)

    theta_raw = meta[["alpha", "beta"]].to_numpy(dtype=float)
    log_theta = np.log(np.clip(theta_raw, 1e-8, None))
    prior = sim.prior().base_dist
    lo = prior.low.numpy()
    hi = prior.high.numpy()
    log_theta = np.clip(log_theta, lo, hi)

    metadata = meta.copy()
    metadata.insert(0, "curve_id", [_curve_id(int(x)) for x in metadata["ID"].tolist()])
    metadata = metadata.rename(columns={"ID": "source_id", "IVR_ID": "source_id"})
    metadata["dataset_name"] = "liposome_accelerated_ivr"
    metadata["mechanism_id"] = "liposome_weibull"
    metadata["time_unit_standard"] = "hours"
    metadata["early_window_h"] = float(args.early_window_h)
    metadata["log_alpha"] = log_theta[:, 0]
    metadata["log_beta"] = log_theta[:, 1]
    metadata["recommended_group_api"] = metadata["API_name"].astype(str)
    metadata["recommended_group_method"] = metadata["release_method"].astype(str)
    metadata["curve_points"] = metadata["source_id"].map(curves_long.groupby("source_id").size().to_dict()).astype(int)
    metadata.to_csv(args.outdir / "metadata.csv", index=False)

    theta_targets = metadata[
        [
            "curve_id",
            "source_id",
            "alpha",
            "beta",
            "log_alpha",
            "log_beta",
            "cluster",
            "cluster_name",
            "API_name",
            "release_method",
        ]
    ].copy()
    theta_targets.to_csv(args.outdir / "theta_targets.csv", index=False)

    mechanism_metadata = {
        "dataset_name": "liposome_accelerated_ivr",
        "mechanism_id": "liposome_weibull",
        "simulator_class": "WeibullSimulator",
        "source_repository": str(args.root),
        "source_paper_doi": "10.1039/D5DD00112A",
        "time_unit_standard": "hours",
        "curve_count": int(metadata["curve_id"].nunique()),
        "point_count": int(len(curves_long)),
        "api_count": int(metadata["API_name"].nunique()),
        "release_method_count": int(metadata["release_method"].nunique()),
        "cluster_counts": {
            str(k): int(v) for k, v in metadata["cluster_name"].value_counts().sort_index().to_dict().items()
        },
        "feature_columns_7": FEATURE_7,
        "theta_parameterization": ["log_alpha", "log_beta"],
        "filters": {
            "early_window_h": float(args.early_window_h),
            "min_early_points": 2,
            "min_future_points": 2,
            "release_min_pct_ge": -5.0,
            "release_max_pct_le": 125.0,
        },
        "recommended_group_columns": {
            "group_by_API": "API_name",
            "group_by_release_method": "release_method",
        },
        "notes": [
            "Curves are filtered with the same loader used by scripts/54_accelerated_ivr_weibull_forecast.py.",
            "Observed release is stored as a fraction in [0, 1.2] after converting percent to fraction.",
            "Original time units are normalized to hours using data/time_units.csv.",
        ],
    }
    (args.outdir / "mechanism_metadata.json").write_text(
        json.dumps(mechanism_metadata, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "Liposome accelerated_IVR intake complete.",
        f"Root: {args.root}",
        f"Outdir: {args.outdir}",
        f"Curves retained: {int(metadata['curve_id'].nunique())}",
        f"Observed points: {int(len(curves_long))}",
        f"APIs: {int(metadata['API_name'].nunique())}",
        f"Release methods: {int(metadata['release_method'].nunique())}",
        f"Clusters: {metadata['cluster_name'].value_counts().sort_index().to_dict()}",
        f"Early window (h): {args.early_window_h}",
    ]
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
