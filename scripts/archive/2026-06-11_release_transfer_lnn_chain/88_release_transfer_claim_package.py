"""88 - Claim package for the release transfer probe.

Purpose:
    Consolidate scripts 81-87 into a compact claim ladder for reporting:
    what is supported, what is not supported, and what the next experiment
    should do.

Consumes:
    outputs/82_release_transfer_report_mlp_none_e20_shapeanchored/
    outputs/83_release_transfer_failure_calibration_shapeanchored/
    outputs/84_release_transfer_data_regime_audit_shapeanchored/
    outputs/85_release_shape_prior_leakage_audit_shapeanchored/
    outputs/86_release_dynamic_future_benchmark_shapeanchored/
    outputs/87_release_transfer_data_expansion_plan_shapeanchored/

Produces:
    outputs/88_release_transfer_claim_package_shapeanchored/
      claim_ladder.csv
      key_numbers.csv
      report.md
      data_expansion_priority.png
      dynamic_delta_by_budget.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REPORT82 = Path("outputs/82_release_transfer_report_mlp_none_e20_shapeanchored")
DEFAULT_FAILURE83 = Path("outputs/83_release_transfer_failure_calibration_shapeanchored")
DEFAULT_REGIME84 = Path("outputs/84_release_transfer_data_regime_audit_shapeanchored")
DEFAULT_LEAK85 = Path("outputs/85_release_shape_prior_leakage_audit_shapeanchored")
DEFAULT_DYNAMIC86 = Path("outputs/86_release_dynamic_future_benchmark_shapeanchored")
DEFAULT_PLAN87 = Path("outputs/87_release_transfer_data_expansion_plan_shapeanchored")
DEFAULT_OUT = Path("outputs/88_release_transfer_claim_package_shapeanchored")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact claim package for release transfer.")
    parser.add_argument("--report82", type=Path, default=DEFAULT_REPORT82)
    parser.add_argument("--failure83", type=Path, default=DEFAULT_FAILURE83)
    parser.add_argument("--regime84", type=Path, default=DEFAULT_REGIME84)
    parser.add_argument("--leak85", type=Path, default=DEFAULT_LEAK85)
    parser.add_argument("--dynamic86", type=Path, default=DEFAULT_DYNAMIC86)
    parser.add_argument("--plan87", type=Path, default=DEFAULT_PLAN87)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(values.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def first_last_time_budget(time_summary: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    loso = time_summary[time_summary["split_kind"] == "loso-system"].sort_values("budget_value")
    first = loso.iloc[0]
    last = loso.iloc[-1]
    reduction = 1.0 - float(last["median_rmse"]) / max(float(first["median_rmse"]), 1e-12)
    return first, last, reduction


def build_key_numbers(
    time_summary: pd.DataFrame,
    dynamic_by_budget: pd.DataFrame,
    informative_dynamic: pd.DataFrame,
    conformal_by_budget: pd.DataFrame,
    leakage: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    first, last, reduction = first_last_time_budget(time_summary)
    dyn_last_multi = dynamic_by_budget[dynamic_by_budget["n_systems"] > 1].sort_values("budget_value").tail(1).iloc[0]
    dyn_last = dynamic_by_budget.sort_values("budget_value").tail(1).iloc[0]
    plga = informative_dynamic[informative_dynamic["heldout_system"] == "PLGA"].iloc[0]
    non_plga = informative_dynamic[informative_dynamic["heldout_system"] != "PLGA"].copy()
    non_plga_wins = int((non_plga["delta_model_minus_best_dynamic"] < 0).sum())
    no_dynamic = int((~non_plga["has_dynamic_rows_at_informative_budget"].astype(bool)).sum())
    worst_cal = conformal_by_budget.sort_values("median_width90_inflation", ascending=False).iloc[0]
    availability_problems = int(
        (
            availability["baseline_without_source"].astype(bool)
            | availability["baseline_with_leaked_source"].astype(bool)
        ).sum()
    )
    rows = [
        {
            "metric": "aggregate_time_rmse_first",
            "value": float(first["median_rmse"]),
            "context": str(first["budget_label"]),
        },
        {
            "metric": "aggregate_time_rmse_last",
            "value": float(last["median_rmse"]),
            "context": str(last["budget_label"]),
        },
        {
            "metric": "aggregate_time_relative_rmse_reduction",
            "value": float(reduction),
            "context": "LOSO all curves",
        },
        {
            "metric": "dynamic_last_multi_system_delta",
            "value": float(dyn_last_multi["delta_model_minus_best_dynamic"]),
            "context": f"{dyn_last_multi['budget_label']} with {int(dyn_last_multi['n_systems'])} systems",
        },
        {
            "metric": "dynamic_last_budget_delta",
            "value": float(dyn_last["delta_model_minus_best_dynamic"]),
            "context": f"{dyn_last['budget_label']} with {int(dyn_last['n_systems'])} system(s)",
        },
        {
            "metric": "plga_informative_dynamic_delta",
            "value": float(plga["delta_model_minus_best_dynamic"]),
            "context": str(plga["budget_label"]),
        },
        {
            "metric": "non_plga_informative_dynamic_wins",
            "value": float(non_plga_wins),
            "context": f"out of {len(non_plga)} non-PLGA systems",
        },
        {
            "metric": "non_plga_no_dynamic_rows",
            "value": float(no_dynamic),
            "context": "informative-budget dynamic review",
        },
        {
            "metric": "max_conformal_width_inflation",
            "value": float(worst_cal["median_width90_inflation"]),
            "context": str(worst_cal["budget_label"]),
        },
        {
            "metric": "shape_prior_leakage_rows",
            "value": float(len(leakage)),
            "context": "script 85",
        },
        {
            "metric": "shape_prior_availability_problems",
            "value": float(availability_problems),
            "context": "script 85",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_ladder(key: pd.DataFrame) -> pd.DataFrame:
    def val(metric: str) -> float:
        return float(key.loc[key["metric"] == metric, "value"].iloc[0])

    rows = [
        {
            "claim_level": "Supported",
            "claim": "Early observation budget contains transferable release information in the clean corpus.",
            "evidence": f"LOSO median RMSE reduction {val('aggregate_time_relative_rmse_reduction'):.1%} from earliest to latest time budget.",
            "safe_wording": "Observation-budget transfer signal is present.",
            "do_not_say": "Foundation model or mechanism learned.",
        },
        {
            "claim_level": "Supported",
            "claim": "PLGA dynamic-future transfer is a real win against shape baselines.",
            "evidence": f"PLGA informative dynamic delta {val('plga_informative_dynamic_delta'):.3f}.",
            "safe_wording": "PLGA remains the strongest validated source-domain win.",
            "do_not_say": "This proves all release systems transfer.",
        },
        {
            "claim_level": "Supported With Caveat",
            "claim": "Aggregate dynamic-future rows favor the model.",
            "evidence": "Dynamic-mask aggregate deltas are negative, but late budgets become PLGA-only.",
            "safe_wording": "Aggregate dynamic-future performance is encouraging but source-dominated.",
            "do_not_say": "Cross-mechanism dynamic transfer is solved.",
        },
        {
            "claim_level": "Not Yet Supported",
            "claim": "Non-PLGA dynamic transfer beats shape-adapted baselines.",
            "evidence": f"Non-PLGA informative dynamic wins: {int(val('non_plga_informative_dynamic_wins'))}.",
            "safe_wording": "Non-PLGA dynamic transfer needs more data or model changes.",
            "do_not_say": "The model generalizes across mechanisms.",
        },
        {
            "claim_level": "Not Supported",
            "claim": "Raw ensemble intervals are calibrated uncertainty.",
            "evidence": f"LOSO conformal diagnostic needs up to {val('max_conformal_width_inflation'):.0f}x width inflation.",
            "safe_wording": "Intervals are diagnostic only.",
            "do_not_say": "90% posterior or calibrated predictive interval.",
        },
        {
            "claim_level": "Audited",
            "claim": "Shape-prior baselines are train-only under the split.",
            "evidence": f"Leakage rows {int(val('shape_prior_leakage_rows'))}, availability problems {int(val('shape_prior_availability_problems'))}.",
            "safe_wording": "Shape baselines are fair train-only competitors in this run.",
            "do_not_say": "Shape baselines are weak or strawman baselines.",
        },
    ]
    return pd.DataFrame(rows)


def plot_priority(plan: pd.DataFrame, out: Path) -> Path:
    view = plan.sort_values("priority_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#b91c1c" if p.startswith("P0") else "#d97706" if p.startswith("P1") else "#2563eb" if p.startswith("P2") else "#64748b" for p in view["priority"]]
    ax.barh(view["heldout_system"], view["priority_score"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("priority score")
    ax.set_title("Data expansion priority by release system")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out / "data_expansion_priority.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_dynamic_delta(dynamic_by_budget: pd.DataFrame, out: Path) -> Path:
    view = dynamic_by_budget.sort_values("budget_value")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(view["budget_value"], view["delta_model_minus_best_dynamic"], marker="o", color="#1d4ed8")
    for _, row in view.iterrows():
        ax.annotate(f"{int(row['n_systems'])} sys", (row["budget_value"], row["delta_model_minus_best_dynamic"]), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("context horizon (days)")
    ax.set_ylabel("model minus best baseline RMSE")
    ax.set_title("Dynamic-future masked aggregate delta")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out / "dynamic_delta_by_budget.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def write_report(
    out: Path,
    claim_ladder: pd.DataFrame,
    key_numbers: pd.DataFrame,
    plan: pd.DataFrame,
    dynamic_by_budget: pd.DataFrame,
    plots: dict[str, Path],
    checks: list[dict[str, Any]],
) -> None:
    lines = [
        "# Release Transfer Claim Package",
        "",
        "This package consolidates the current release-corpus transfer probe. It is intentionally conservative.",
        "",
        "## Claim Ladder",
        "",
        markdown_table(claim_ladder, max_rows=12),
        "",
        "## Key Numbers",
        "",
        markdown_table(key_numbers, max_rows=20),
        "",
        "## Dynamic-Future Delta",
        "",
        f"![Dynamic delta]({plots['dynamic_delta'].as_posix()})",
        "",
        markdown_table(dynamic_by_budget, max_rows=10),
        "",
        "## Data Expansion Priority",
        "",
        f"![Data expansion priority]({plots['priority'].as_posix()})",
        "",
        markdown_table(
            plan[
                [
                    "heldout_system",
                    "priority",
                    "priority_score",
                    "purpose",
                    "recommended_min_total_curves",
                    "collection_target",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Recommended Next Move",
        "",
        "1. Use PLGA as the validated source-domain and calibration sandbox.",
        "2. Use liposome for model ablations because it has enough dynamic rows but still loses to shape priors.",
        "3. Expand non-PLGA systems with dynamic future windows before making cross-mechanism transfer claims.",
        "4. Keep raw ensemble intervals out of any calibrated-uncertainty claim.",
        "",
        "## Integrity Checks",
        "",
        markdown_table(pd.DataFrame(checks), max_rows=20),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    time_summary = read_csv(args.report82 / "headline_summary_by_time_budget.csv")
    conformal = read_csv(args.failure83 / "conformal_by_budget.csv")
    leakage = read_csv(args.leak85 / "leakage_rows.csv")
    availability = read_csv(args.leak85 / "baseline_family_availability.csv")
    dynamic_by_budget = read_csv(args.dynamic86 / "dynamic_by_budget.csv")
    informative_dynamic = read_csv(args.dynamic86 / "informative_dynamic_review.csv")
    plan = read_csv(args.plan87 / "data_expansion_priority.csv")

    key_numbers = build_key_numbers(time_summary, dynamic_by_budget, informative_dynamic, conformal, leakage, availability)
    claim_ladder = build_claim_ladder(key_numbers)
    priority_plot = plot_priority(plan, args.out)
    dynamic_plot = plot_dynamic_delta(dynamic_by_budget, args.out)

    checks = [
        finite_numeric_check("key_numbers", key_numbers),
        finite_numeric_check("claim_ladder", claim_ladder),
        finite_numeric_check("data_expansion_priority", plan),
        finite_numeric_check("dynamic_by_budget", dynamic_by_budget),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")

    key_numbers.to_csv(args.out / "key_numbers.csv", index=False)
    claim_ladder.to_csv(args.out / "claim_ladder.csv", index=False)
    metadata = {
        "script": "scripts/88_release_transfer_claim_package.py",
        "report82": str(args.report82),
        "failure83": str(args.failure83),
        "regime84": str(args.regime84),
        "leak85": str(args.leak85),
        "dynamic86": str(args.dynamic86),
        "plan87": str(args.plan87),
        "out": str(args.out),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        args.out,
        claim_ladder,
        key_numbers,
        plan,
        dynamic_by_budget,
        {"priority": priority_plot, "dynamic_delta": dynamic_plot},
        checks,
    )
    print(f"[88] wrote {args.out}")
    print(claim_ladder[["claim_level", "claim", "safe_wording"]].to_string(index=False))


if __name__ == "__main__":
    main()
