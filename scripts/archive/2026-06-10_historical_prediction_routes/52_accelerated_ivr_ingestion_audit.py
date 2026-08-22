"""
52 - Accelerated IVR ingestion audit.

Consumes the downloaded danielyanes22/accelerated_IVR repository and
checks whether its liposome IVR curves can support our next benchmark:

    features + early release -> kinetic state -> full release curve

This script does not train a model. It only aligns curve IDs, descriptors,
Weibull parameters, and kinetic-class labels, then reports which curves are
usable for full-curve forecasting.

Outputs:
    outputs/52_accelerated_ivr_ingestion_audit/curve_inventory.csv
    outputs/52_accelerated_ivr_ingestion_audit/aligned_curve_table.csv
    outputs/52_accelerated_ivr_ingestion_audit/aligned_weibull_table.csv
    outputs/52_accelerated_ivr_ingestion_audit/dataset_summary.csv
    outputs/52_accelerated_ivr_ingestion_audit/summary.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_7 = [
    "media_pH",
    "media_temp_oC",
    "drug_loading",
    "Z_average_nm",
    "API_name",
    "weighted_Mw",
    "weighted_Tm",
]
FEATURE_9 = ["release_method", *FEATURE_7[:3], "structure_type", *FEATURE_7[3:]]
EARLY_WINDOWS = [6.0, 12.0, 24.0, 48.0, 72.0]


def _parse_id(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"cannot parse IVR ID from {value!r}")
    return int(match.group(1))


def _load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp = pd.read_csv(root / "results/fitting/drug_release_exp.csv")
    exp["ID"] = exp["file_name"].map(_parse_id).astype(int)
    exp = exp.rename(columns={"time (Hrs)": "time_h", "release_percent": "release_pct"})
    exp = exp[["ID", "time_h", "release_pct", "file_name"]].copy()

    backend = pd.read_csv(root / "data/unprocessed/backend_data.csv")
    backend = backend.rename(columns={"IVR_ID": "ID"}).drop(columns=["Unnamed: 0"], errors="ignore")
    backend["ID"] = backend["ID"].astype(int)

    weibull = pd.read_csv(root / "data/clean/weibull_params.csv")
    weibull["ID"] = weibull["ID"].astype(int)

    cluster = pd.read_csv(root / "results/clustering/3_PCA_KMC.csv")
    cluster = cluster.rename(columns={"file": "ID"})
    cluster["ID"] = cluster["ID"].astype(int)
    cluster = cluster[["ID", "alpha", "beta", "PC1", "PC2", "cluster", "cluster_name"]]
    return exp, backend, weibull, cluster


def _curve_summary(exp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for curve_id, sub in exp.groupby("ID"):
        sub = sub.sort_values("time_h")
        t = sub["time_h"].to_numpy(dtype=float)
        q = sub["release_pct"].to_numpy(dtype=float)
        row: dict[str, object] = {
            "ID": int(curve_id),
            "n_points": int(len(sub)),
            "t_min_h": float(np.min(t)),
            "t_max_h": float(np.max(t)),
            "release_min_pct": float(np.min(q)),
            "release_max_pct": float(np.max(q)),
            "release_out_of_range": bool((np.min(q) < -5.0) or (np.max(q) > 125.0)),
        }
        for window in EARLY_WINDOWS:
            row[f"n_early_le_{int(window)}h"] = int(np.sum(t <= window))
            row[f"n_late_gt_{int(window)}h"] = int(np.sum(t > window))
            row[f"covers_{int(window)}h"] = bool(np.max(t) >= window)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_inventory(
    exp: pd.DataFrame,
    backend: pd.DataFrame,
    weibull: pd.DataFrame,
    cluster: pd.DataFrame,
) -> pd.DataFrame:
    inventory = _curve_summary(exp)
    inventory = inventory.merge(backend, on="ID", how="left", indicator="backend_merge")
    inventory = inventory.merge(
        weibull[["ID", "alpha", "beta", "Time_units"]],
        on="ID",
        how="left",
        indicator="weibull_merge",
    )
    inventory = inventory.merge(
        cluster[["ID", "cluster", "cluster_name"]],
        on="ID",
        how="left",
        indicator="cluster_merge",
    )
    inventory["has_backend"] = inventory["backend_merge"].eq("both")
    inventory["has_weibull"] = inventory["weibull_merge"].eq("both")
    inventory["has_cluster"] = inventory["cluster_merge"].eq("both")
    inventory["complete_features_7"] = inventory[FEATURE_7].notna().all(axis=1)
    inventory["complete_features_9"] = inventory[FEATURE_9].notna().all(axis=1)
    inventory["valid_release_range"] = ~inventory["release_out_of_range"]

    for window in EARLY_WINDOWS:
        w = int(window)
        inventory[f"clean_forecast_{w}h"] = (
            inventory["has_backend"]
            & inventory["has_weibull"]
            & inventory["has_cluster"]
            & inventory["complete_features_7"]
            & inventory["valid_release_range"]
            & (inventory[f"n_early_le_{w}h"] >= 2)
            & (inventory[f"n_late_gt_{w}h"] >= 2)
        )
    inventory["clean_forecast_any"] = inventory[[f"clean_forecast_{int(w)}h" for w in EARLY_WINDOWS]].any(axis=1)
    return inventory.drop(columns=["backend_merge", "weibull_merge", "cluster_merge"])


def _dataset_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "n_release_curves", "value": int(len(inventory))},
        {"metric": "n_with_backend", "value": int(inventory["has_backend"].sum())},
        {"metric": "n_with_weibull", "value": int(inventory["has_weibull"].sum())},
        {"metric": "n_with_cluster", "value": int(inventory["has_cluster"].sum())},
        {"metric": "n_complete_features_7", "value": int(inventory["complete_features_7"].sum())},
        {"metric": "n_complete_features_9", "value": int(inventory["complete_features_9"].sum())},
        {"metric": "n_clean_forecast_any", "value": int(inventory["clean_forecast_any"].sum())},
    ]
    for window in EARLY_WINDOWS:
        rows.append({
            "metric": f"n_clean_forecast_{int(window)}h",
            "value": int(inventory[f"clean_forecast_{int(window)}h"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/accelerated_IVR/repo/accelerated_IVR-main"),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/52_accelerated_ivr_ingestion_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exp, backend, weibull, cluster = _load_tables(args.root)
    inventory = _build_inventory(exp, backend, weibull, cluster)
    summary = _dataset_summary(inventory)

    exp.to_csv(args.out / "aligned_curve_table.csv", index=False)
    inventory.to_csv(args.out / "curve_inventory.csv", index=False)
    summary.to_csv(args.out / "dataset_summary.csv", index=False)
    inventory[inventory["has_weibull"]].to_csv(args.out / "aligned_weibull_table.csv", index=False)

    lines = [
        "=== 52 -- accelerated IVR ingestion audit ===",
        "",
        f"root: {args.root}",
        "",
        "--- dataset summary ---",
    ]
    for _, row in summary.iterrows():
        lines.append(f"{row['metric']:<28} {int(row['value'])}")
    lines.extend([
        "",
        "--- verdict ---",
    ])
    n_best = int(summary.loc[summary["metric"].eq("n_clean_forecast_6h"), "value"].iloc[0])
    if n_best >= 60:
        lines.append("PASS: enough clean forecast curves for the first liposome benchmark.")
    else:
        lines.append("CAUTION: fewer than 60 clean forecast curves at the shortest early window.")
    lines.append("Use feature-complete Weibull-aligned curves for forecast experiments.")
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
