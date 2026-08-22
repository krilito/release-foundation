"""
59 - Middle-layer gain audit.

What this does:
    Scripts 45 and 46 already compare two routes under matched datasets,
    input modes, and split schemes:

        direct route:
            x / early Q -> Q grid

        kinetic-state route:
            x / early Q -> theta -> ODE -> Q(t)

    This script turns those results into one paper-facing statistic:

        MLG = median R^2(kinetic-state route) - median R^2(direct-Q route)

    The metric asks whether the middle layer contributes structure beyond
    the tabular regressor itself. It is intentionally computed from matched
    input/split cells, not from unrelated headline numbers.

Inputs:
    outputs/46_direct_curve_rf_input_ablation/scheme_method_summary.csv
    outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv
    outputs/45_input_source_ablation/input_source_method_summary.csv

Outputs:
    outputs/59_middle_layer_gain_audit/best_route_middle_layer_gain.csv
    outputs/59_middle_layer_gain_audit/family_middle_layer_gain.csv
    outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv
    outputs/59_middle_layer_gain_audit/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _optional_package_status() -> dict[str, bool]:
    return {
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "ngboost": importlib.util.find_spec("ngboost") is not None,
    }


def _direct_family(method: str) -> str:
    if method.startswith("RF_"):
        return "RF"
    if method.startswith("ET_"):
        return "ET"
    if method.startswith("LGB") or method.startswith("LightGBM"):
        return "LGBM"
    if method.startswith("XGB"):
        return "XGB"
    if method.startswith("MLP"):
        return "MLP"
    if method.startswith("KNN"):
        return "KNN"
    if method.startswith("SVR"):
        return "SVR"
    if method.startswith("Ridge") or method.startswith("Lasso") or method.startswith("Elastic"):
        return "Linear"
    return "Other"


def _theta_family(method: str) -> str:
    if method.startswith("RF_ET"):
        return "RF_ET"
    if method.startswith("RF_"):
        return "RF"
    if method.startswith("ET_"):
        return "ET"
    if method.startswith("early_select"):
        return "Selector"
    if method.startswith("LGB") or method.startswith("LightGBM"):
        return "LGBM"
    if method.startswith("XGB"):
        return "XGB"
    if method.startswith("MLP"):
        return "MLP"
    if method.startswith("Ridge") or method.startswith("Lasso") or method.startswith("Elastic"):
        return "Linear"
    return "Other"


def _best_by_cell(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    idx = df.groupby(group_cols, dropna=False)["median"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _load_best_route(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "dataset",
        "input_mode",
        "scheme",
        "direct_best_method",
        "direct_best_median",
        "theta_best_method",
        "theta_best_median",
        "theta_minus_direct",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out = out.rename(columns={"theta_minus_direct": "middle_layer_gain"})
    out["direct_family"] = out["direct_best_method"].map(_direct_family)
    out["theta_family"] = out["theta_best_method"].map(_theta_family)
    return out


def _load_family_gain(direct_path: Path, theta_path: Path) -> pd.DataFrame:
    direct = pd.read_csv(direct_path)
    theta = pd.read_csv(theta_path)
    for name, df in {"direct": direct, "theta": theta}.items():
        missing = {"dataset", "input_mode", "scheme", "method", "median"} - set(df.columns)
        if missing:
            raise KeyError(f"{name} summary missing columns: {sorted(missing)}")

    direct = direct.copy()
    theta = theta.copy()
    direct["family"] = direct["method"].map(_direct_family)
    theta["family"] = theta["method"].map(_theta_family)

    # Keep only true one-family comparisons. Selectors/ensembles are useful
    # headline routes, but they are not a same-backbone MLG control.
    direct = direct[direct["family"].isin({"RF", "ET", "LGBM", "XGB", "MLP", "Linear"})]
    theta = theta[theta["family"].isin({"RF", "ET", "LGBM", "XGB", "MLP", "Linear"})]

    group_cols = ["dataset", "input_mode", "scheme", "family"]
    direct_best = _best_by_cell(direct, group_cols).rename(columns={
        "method": "direct_method",
        "median": "direct_median",
        "frac_above_0": "direct_frac_above_0",
    })
    theta_best = _best_by_cell(theta, group_cols).rename(columns={
        "method": "theta_method",
        "median": "theta_median",
        "frac_above_0": "theta_frac_above_0",
    })

    keep_direct = group_cols + ["direct_method", "direct_median", "direct_frac_above_0"]
    keep_theta = group_cols + ["theta_method", "theta_median", "theta_frac_above_0"]
    merged = direct_best[keep_direct].merge(theta_best[keep_theta], on=group_cols, how="inner")
    merged["middle_layer_gain"] = merged["theta_median"] - merged["direct_median"]
    return merged.sort_values(["dataset", "input_mode", "scheme", "family"]).reset_index(drop=True)


def _aggregate_rows(best_route: pd.DataFrame, family_gain: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    groups: list[tuple[str, pd.DataFrame]] = [
        ("best_route/all", best_route),
        (
            "best_route/formulation_plus_early",
            best_route[best_route["input_mode"] == "formulation_plus_early"],
        ),
        ("best_route/early_only", best_route[best_route["input_mode"] == "early_only"]),
        (
            "best_route/OOD_formulation_plus_early",
            best_route[
                (best_route["input_mode"] == "formulation_plus_early")
                & (best_route["scheme"] != "random_5fold")
            ],
        ),
        ("family_matched/all", family_gain),
        (
            "family_matched/formulation_plus_early",
            family_gain[family_gain["input_mode"] == "formulation_plus_early"],
        ),
        (
            "family_matched/OOD_formulation_plus_early",
            family_gain[
                (family_gain["input_mode"] == "formulation_plus_early")
                & (family_gain["scheme"] != "random_5fold")
            ],
        ),
    ]

    for label, sub in groups:
        vals = sub["middle_layer_gain"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        rows.append({
            "group": label,
            "n_cells": int(len(vals)),
            "mean_gain": float(np.mean(vals)),
            "median_gain": float(np.median(vals)),
            "min_gain": float(np.min(vals)),
            "max_gain": float(np.max(vals)),
            "frac_positive": float(np.mean(vals > 0)),
        })

    for family, sub in family_gain.groupby("family", sort=True):
        vals = sub["middle_layer_gain"].dropna().to_numpy(dtype=float)
        rows.append({
            "group": f"family/{family}",
            "n_cells": int(len(vals)),
            "mean_gain": float(np.mean(vals)),
            "median_gain": float(np.median(vals)),
            "min_gain": float(np.min(vals)),
            "max_gain": float(np.max(vals)),
            "frac_positive": float(np.mean(vals > 0)),
        })

    return pd.DataFrame(rows)


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--direct-summary",
        type=Path,
        default=Path("outputs/46_direct_curve_rf_input_ablation/scheme_method_summary.csv"),
    )
    ap.add_argument(
        "--best-route-summary",
        type=Path,
        default=Path("outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv"),
    )
    ap.add_argument(
        "--theta-method-summary",
        type=Path,
        default=Path("outputs/45_input_source_ablation/input_source_method_summary.csv"),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/59_middle_layer_gain_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    best_route = _load_best_route(args.best_route_summary)
    family_gain = _load_family_gain(args.direct_summary, args.theta_method_summary)
    aggregate = _aggregate_rows(best_route, family_gain)
    package_status = _optional_package_status()

    best_route.to_csv(args.out / "best_route_middle_layer_gain.csv", index=False)
    family_gain.to_csv(args.out / "family_middle_layer_gain.csv", index=False)
    aggregate.to_csv(args.out / "aggregate_middle_layer_gain.csv", index=False)

    lines = [
        "=== 59 -- middle-layer gain audit ===",
        "",
        "Definition:",
        "  MLG = median R^2(kinetic-state route) - median R^2(direct-Q route)",
        "",
        "--- optional model-family availability ---",
    ]
    for name, ok in package_status.items():
        lines.append(f"  {name:<8}: {'available' if ok else 'not installed'}")

    lines.extend(["", "--- aggregate MLG ---"])
    for _, row in aggregate.iterrows():
        lines.append(
            f"  {row['group']:<46} n={int(row['n_cells']):2d} "
            f"mean={_fmt(row['mean_gain'])} median={_fmt(row['median_gain'])} "
            f"min={_fmt(row['min_gain'])} max={_fmt(row['max_gain'])} "
            f"frac>0={row['frac_positive']:.2f}"
        )

    best_oood = best_route[
        (best_route["input_mode"] == "formulation_plus_early")
        & (best_route["scheme"] != "random_5fold")
    ].sort_values("middle_layer_gain", ascending=False)
    if not best_oood.empty:
        lines.extend(["", "--- strongest OOD formulation+early cells ---"])
        for _, row in best_oood.head(6).iterrows():
            lines.append(
                f"  {row['dataset']:<11} {row['scheme']:<16} "
                f"direct={row['direct_best_median']:+.4f} ({row['direct_best_method']})  "
                f"theta={row['theta_best_median']:+.4f} ({row['theta_best_method']})  "
                f"MLG={row['middle_layer_gain']:+.4f}"
            )

    lines.extend([
        "",
        "--- interpretation guardrail ---",
        "  best_route compares the strongest direct route with the strongest",
        "  kinetic-state route in each matched cell. family_matched compares",
        "  RF-vs-RF and ET-vs-ET only, so it is the cleaner structural control.",
        "  LightGBM is available in this environment, but the current trusted",
        "  45/46 result files only contain RF/ET-family methods. XGBoost and",
        "  NGBoost are not installed and should not be added without a separate",
        "  dependency decision.",
    ])

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
