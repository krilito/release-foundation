"""
18 - Blend two cross-DOI SBI prediction directories and score the ensemble.

What this does:
    1. Load per-curve `.npz` predictions from two script-10 cross-DOI runs.
    2. Blend their pointwise median predictions with a fixed weight alpha.
    3. Recompute per-curve R^2 / MAE on the 259 high-quality 321 curves.
    4. Optionally summarize subgroup behavior using the failure-tail labels
       from script 14.

Why this exists:
    The missingness-aware masked SBI improved the median cross-DOI score
    but introduced a few catastrophic outliers, while the original full SBI
    was more stable on those curves. A simple fixed-weight ensemble tests
    whether the two models are complementary before touching the main code.

Outputs:
    outputs/18_cross_doi_sbi_ensemble/per_curve_metrics.csv
    outputs/18_cross_doi_sbi_ensemble/summary.txt
    outputs/18_cross_doi_sbi_ensemble/subgroup_summary.csv

Expected runtime:
    <5 s.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    return arr["Q_obs"], arr["Q_pred_median"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-dir-a",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_predictions"),
    )
    ap.add_argument(
        "--pred-dir-b",
        type=Path,
        default=Path("outputs/17_eval_masked/cross_doi_predictions"),
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="ensemble = alpha * pred_a + (1-alpha) * pred_b",
    )
    ap.add_argument(
        "--tail-flags-csv",
        type=Path,
        default=Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/18_cross_doi_sbi_ensemble"),
    )
    args = ap.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    args.out.mkdir(parents=True, exist_ok=True)

    paths_a = {p.name: p for p in args.pred_dir_a.glob("*.npz")}
    paths_b = {p.name: p for p in args.pred_dir_b.glob("*.npz")}
    common = sorted(set(paths_a) & set(paths_b), key=lambda s: int(Path(s).stem))
    if not common:
        raise RuntimeError("No common .npz prediction files found between the two dirs")

    rows = []
    for name in common:
        q_obs_a, q_pred_a = _load_curve(paths_a[name])
        q_obs_b, q_pred_b = _load_curve(paths_b[name])
        if q_obs_a.shape != q_obs_b.shape or not np.allclose(q_obs_a, q_obs_b):
            raise ValueError(f"Observed curve mismatch for {name}")
        q_pred = args.alpha * q_pred_a + (1.0 - args.alpha) * q_pred_b
        fid = int(Path(name).stem)
        rows.append({
            "Formulation_Index": fid,
            "n_obs": int(q_obs_a.shape[0]),
            "R2": _r2(q_obs_a, q_pred),
            "MAE": float(np.mean(np.abs(q_obs_a - q_pred))),
        })

    per_curve = pd.DataFrame(rows)
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)

    flags = pd.read_csv(args.tail_flags_csv)[
        ["Formulation_Index", "fast_regime", "short_window"]
    ].copy()
    merged = per_curve.merge(flags, on="Formulation_Index", how="left")
    subgroup_rows = []
    subsets = {
        "overall": merged,
        "fast": merged[merged["fast_regime"]],
        "short": merged[merged["short_window"]],
        "fast_short": merged[merged["fast_regime"] & merged["short_window"]],
        "neither": merged[~merged["fast_regime"] & ~merged["short_window"]],
    }
    for label, sub in subsets.items():
        subgroup_rows.append({
            "subset": label,
            "n_curves": int(len(sub)),
            "median_R2": float(sub["R2"].median()),
            "mean_R2": float(sub["R2"].mean()),
            "median_MAE": float(sub["MAE"].median()),
            "n_R2_gt_0": int((sub["R2"] > 0).sum()),
            "n_R2_gt_0p3": int((sub["R2"] > 0.3).sum()),
        })
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(args.out / "subgroup_summary.csv", index=False)

    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Cross-DOI SBI ensemble ===\n\n")
        f.write(f"pred_dir_a : {args.pred_dir_a}\n")
        f.write(f"pred_dir_b : {args.pred_dir_b}\n")
        f.write(f"alpha      : {args.alpha:.3f}\n")
        f.write(f"n_curves   : {len(per_curve)}\n\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"[18] wrote {args.out / 'summary.txt'}")
    print(subgroup.to_string(index=False))


if __name__ == "__main__":
    main()
