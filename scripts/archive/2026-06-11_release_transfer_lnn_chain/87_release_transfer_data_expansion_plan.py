"""87 - Data expansion plan for release transfer.

Purpose:
    Convert transfer diagnostics into a concrete data expansion priority list.
    The plan focuses on systems where dynamic-future transfer is either not
    evaluable or loses to simple shape-adapted baselines.

Consumes:
    outputs/84_release_transfer_data_regime_audit_shapeanchored/
    outputs/86_release_dynamic_future_benchmark_shapeanchored/

Produces:
    outputs/87_release_transfer_data_expansion_plan/
      data_expansion_priority.csv
      collection_targets.csv
      summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REGIME = Path("outputs/84_release_transfer_data_regime_audit_shapeanchored")
DEFAULT_DYNAMIC = Path("outputs/86_release_dynamic_future_benchmark_shapeanchored")
DEFAULT_OUT = Path("outputs/87_release_transfer_data_expansion_plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan data expansion for release transfer.")
    parser.add_argument("--regime-dir", type=Path, default=DEFAULT_REGIME)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC)
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


def classify_priority(row: pd.Series) -> tuple[str, str, float]:
    delta = float(row["delta_model_minus_best_dynamic"])
    n_dynamic = float(row["n_dynamic_curves"])
    has_dynamic = bool(row["has_dynamic_rows_at_informative_budget"])
    n_curves = float(row["n_total_curves"])
    selection_rule = str(row["selection_rule"])

    score = 0.0
    reasons: list[str] = []
    if not has_dynamic:
        score += 5.0
        reasons.append("not_evaluable_dynamic_future")
    if selection_rule == "no_budget_has_majority_dynamic_future":
        score += 3.0
        reasons.append("no_majority_dynamic_budget")
    if delta > 0:
        score += min(delta * 10.0, 3.0)
        reasons.append("model_loses_to_shape_or_local_baseline")
    if delta > 0.03 and n_dynamic >= 50:
        score += 2.0
        reasons.append("enough_dynamic_rows_for_model_ablation")
    if 0 < n_dynamic < 20:
        score += 2.0
        reasons.append("dynamic_subset_too_small")
    if n_curves < 20:
        score += 1.0
        reasons.append("system_low_n")
    if delta <= -0.02 and has_dynamic:
        score -= 2.0
        reasons.append("current_model_wins")

    if score >= 7.0:
        priority = "P0_collect_before_claim"
    elif score >= 4.0:
        priority = "P1_high_value_expansion"
    elif score >= 2.0:
        priority = "P2_model_ablation_or_targeted_followup"
    else:
        priority = "P3_monitor_or_secondary"
    return priority, ";".join(reasons), score


def collection_spec(row: pd.Series) -> dict[str, Any]:
    system = str(row["heldout_system"])
    has_dynamic = bool(row["has_dynamic_rows_at_informative_budget"])
    delta = float(row["delta_model_minus_best_dynamic"])
    n_dynamic = int(float(row["n_dynamic_curves"]))
    n_total = int(float(row["n_total_curves"]))
    budget_label = str(row["budget_label"])

    if not has_dynamic:
        target_curves = max(30, n_total + 20)
        target = "collect new curves with at least 3 post-context target points and non-flat release after early observations"
        purpose = "make_dynamic_transfer_evaluable"
    elif delta > 0.05:
        target_curves = max(40, n_total + 25)
        target = "collect matched dynamic curves with dense early window plus late tail; include repeats across payload/polymer conditions"
        purpose = "beat_shape_increment_baseline"
    elif delta > 0.0:
        target_curves = max(30, n_total + 15)
        target = "add dynamic curves around the current informative budget and preserve raw time grid"
        purpose = "turn_near_tie_into_win"
    else:
        target_curves = max(n_total, n_dynamic)
        target = "use as secondary validation; do not prioritize new collection before weaker systems"
        purpose = "validation_or_source_domain"

    if system == "liposome":
        target += "; stratify by payload and release method because early dynamic rows exist but shape prior still wins"
    if system == "PLGA":
        target = "use as source-domain backbone and calibration sandbox; prioritize external PLGA OOD conditions over more IID curves"
        purpose = "source_domain_and_ood_validation"
    if "Alginate" in system or "starch" in system:
        target += "; ensure measurements extend beyond the current ultra-short window"
    if "caseinate" in system or "hydrogel" in system:
        target += "; avoid single-point tails by requiring multiple late observations"

    return {
        "recommended_min_total_curves": int(target_curves),
        "recommended_timepoint_floor": 8,
        "recommended_post_context_target_points": 3,
        "anchor_budget_to_fix": budget_label,
        "collection_target": target,
        "purpose": purpose,
    }


def build_plan(informative_dynamic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for _, row in informative_dynamic.iterrows():
        priority, reason, score = classify_priority(row)
        spec = collection_spec(row)
        item = row.to_dict()
        item.update(
            {
                "priority": priority,
                "priority_score": float(score),
                "priority_reason": reason,
            }
        )
        item.update(spec)
        rows.append(item)
        targets.append(
            {
                "heldout_system": row["heldout_system"],
                "priority": priority,
                "recommended_min_total_curves": spec["recommended_min_total_curves"],
                "recommended_timepoint_floor": spec["recommended_timepoint_floor"],
                "recommended_post_context_target_points": spec["recommended_post_context_target_points"],
                "anchor_budget_to_fix": spec["anchor_budget_to_fix"],
                "purpose": spec["purpose"],
                "collection_target": spec["collection_target"],
            }
        )
    plan = pd.DataFrame(rows).sort_values(["priority_score", "delta_model_minus_best_dynamic"], ascending=[False, False])
    target_df = pd.DataFrame(targets).merge(
        plan[["heldout_system", "priority_score", "priority_reason", "delta_model_minus_best_dynamic", "n_dynamic_curves", "frac_curves_dynamic"]],
        on="heldout_system",
        how="left",
    )
    target_df = target_df.sort_values(["priority_score", "delta_model_minus_best_dynamic"], ascending=[False, False])
    return plan, target_df


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def write_summary(out: Path, regime_dir: Path, dynamic_dir: Path, plan: pd.DataFrame, targets: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    lines = [
        "# Release Transfer Data Expansion Plan",
        "",
        f"- Regime audit: `{regime_dir}`",
        f"- Dynamic benchmark: `{dynamic_dir}`",
        "- Goal: prioritize data that can make non-PLGA dynamic transfer genuinely testable.",
        "",
        "## Priority Ranking",
        "",
        markdown_table(
            plan[
                [
                    "heldout_system",
                    "priority",
                    "priority_score",
                    "budget_label",
                    "n_dynamic_curves",
                    "frac_curves_dynamic",
                    "delta_model_minus_best_dynamic",
                    "purpose",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Collection Targets",
        "",
        markdown_table(
            targets[
                [
                    "heldout_system",
                    "priority",
                    "recommended_min_total_curves",
                    "recommended_timepoint_floor",
                    "recommended_post_context_target_points",
                    "anchor_budget_to_fix",
                    "collection_target",
                ]
            ],
            max_rows=12,
        ),
        "",
        "## Interpretation",
        "",
        "- Do not spend the next data cycle on more IID PLGA unless it is OOD validation or calibration.",
        "- Prioritize non-PLGA systems where dynamic rows exist but shape-increment baselines still win.",
        "- For systems with no dynamic rows, first fix observability before using them as transfer benchmarks.",
        "",
        "## Integrity Checks",
        "",
        markdown_table(pd.DataFrame(checks), max_rows=20),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    informative = read_csv(args.dynamic_dir / "informative_dynamic_review.csv")
    plan, targets = build_plan(informative)

    checks = [
        finite_numeric_check("data_expansion_priority", plan),
        finite_numeric_check("collection_targets", targets),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")

    plan.to_csv(args.out / "data_expansion_priority.csv", index=False)
    targets.to_csv(args.out / "collection_targets.csv", index=False)
    metadata = {
        "script": "scripts/87_release_transfer_data_expansion_plan.py",
        "regime_dir": str(args.regime_dir),
        "dynamic_dir": str(args.dynamic_dir),
        "out": str(args.out),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out, args.regime_dir, args.dynamic_dir, plan, targets, checks)
    print(f"[87] wrote {args.out}")
    print(plan[["heldout_system", "priority", "priority_score", "purpose"]].to_string(index=False))


if __name__ == "__main__":
    main()
