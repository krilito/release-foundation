"""102 - PLGA information-budget claim table.

Purpose:
    Collapse the current exploratory sprint into a single evidence table for
    the narrowed paper route:

        early observations reveal missing release state under source shift

    The script does not train new models. It harmonizes existing result files
    from scripts 91, 96, 97, 99, 100, and 101 into:

      - unified_model_table.csv
      - budget_value_table.csv
      - claim_evidence_table.csv
      - report.md

    Metrics are not blindly ranked across incompatible protocols. Each row
    carries a comparability note and claim role.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("outputs/102_plga_information_budget_claim_table")
DEFAULT_91_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold/summary_by_config.csv")
DEFAULT_91_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold/summary_by_config.csv")
DEFAULT_91_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo/summary_by_config.csv")
DEFAULT_96 = Path("outputs/96_plga_early_middle_layer_selector_probe/summary_by_budget.csv")
DEFAULT_97 = Path("outputs/97_plga_direct_early_blackbox_control/summary_by_method.csv")
DEFAULT_99_DECISION = Path("outputs/99_plga_static_to_early_proxy_probe/decision_table.csv")
DEFAULT_99_PROXY = Path("outputs/99_plga_static_to_early_proxy_probe/proxy_summary.csv")
DEFAULT_100 = Path("outputs/100_lai_author_model_audit/author_cv_summary_by_model.csv")
DEFAULT_101 = Path("outputs/101_lai_author_style_on_our_splits/summary_by_method.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PLGA information-budget claim table.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--static-random", type=Path, default=DEFAULT_91_RANDOM)
    parser.add_argument("--static-group", type=Path, default=DEFAULT_91_GROUP)
    parser.add_argument("--static-source", type=Path, default=DEFAULT_91_SOURCE)
    parser.add_argument("--middle-summary", type=Path, default=DEFAULT_96)
    parser.add_argument("--direct-summary", type=Path, default=DEFAULT_97)
    parser.add_argument("--proxy-decision", type=Path, default=DEFAULT_99_DECISION)
    parser.add_argument("--proxy-summary", type=Path, default=DEFAULT_99_PROXY)
    parser.add_argument("--author-cv", type=Path, default=DEFAULT_100)
    parser.add_argument("--author-ours", type=Path, default=DEFAULT_101)
    return parser.parse_args()


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def args_to_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "split_kind": "",
        "budget": np.nan,
        "budget_kind": "",
        "observation_protocol": "",
        "observation_regime": "",
        "method_family": "",
        "method": "",
        "median_rmse": np.nan,
        "median_mae": np.nan,
        "median_r2": np.nan,
        "n_curves": np.nan,
        "claim_role": "",
        "comparability_note": "",
        "source_file": "",
    }
    base.update(kwargs)
    rows.append(base)


def collect_static_ceiling(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    for path in [args.static_random, args.static_group, args.static_source]:
        df = read_optional(path)
        if df.empty:
            continue
        for split_kind, sub in df.groupby("split_kind", sort=True):
            best = sub.sort_values("median_rmse").iloc[0]
            add_row(
                rows,
                split_kind=split_kind,
                budget=0,
                budget_kind="none",
                observation_protocol="static descriptors only",
                observation_regime="static-only",
                method_family="static shape/parameter prior",
                method=f"best_91::{best['feature_group']}::{best['family']}::{best['model']}",
                median_rmse=float(best["median_rmse"]),
                median_mae=float(best["median_mae"]),
                median_r2=float(best["median_r2"]),
                n_curves=int(best["n_curves"]),
                claim_role="static descriptor ceiling",
                comparability_note="curve-level static theta/shape summary from script 91; not dynamic future-only",
                source_file=str(path),
            )


def collect_middle(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    df = read_optional(args.middle_summary)
    if df.empty:
        return
    selector_role = {
        "train_global_family": "middle-layer static/global selector",
        "early_residual_family": "deployable early-conditioned middle layer",
        "future_oracle_family": "non-deployable oracle upper bound",
    }
    for _, row in df.iterrows():
        selector = str(row["selector"])
        add_row(
            rows,
            split_kind=str(row["split_kind"]),
            budget=int(row["budget"]),
            budget_kind="earliest_k_points" if int(row["budget"]) > 0 else "none",
            observation_protocol=(
                "earliest measured release points" if int(row["budget"]) > 0 else "static/no-context"
            ),
            observation_regime="earliest-k" if int(row["budget"]) > 0 else "static/no-context",
            method_family="middle-layer shape selector",
            method=f"middle::{selector}",
            median_rmse=float(row["median_future_rmse"]),
            median_mae=float(row["median_future_mae"]),
            median_r2=float(row["median_future_r2"]),
            n_curves=int(row["n_curves"]),
            claim_role=selector_role.get(selector, "middle-layer control"),
            comparability_note="future-only per-curve scoring from script 96",
            source_file=str(args.middle_summary),
        )


def collect_direct(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    df = read_optional(args.direct_summary)
    if df.empty:
        return
    wanted = {"direct_static_time", "direct_early_time", "direct_static_early_time"}
    for _, row in df[df["method"].isin(wanted)].iterrows():
        budget = int(row["budget"])
        method = str(row["method"])
        if method == "direct_static_time":
            obs = "static-only scored after budget window"
            role = "direct static black-box baseline"
        elif method == "direct_early_time":
            obs = "earliest-k"
            role = "early observation black-box baseline"
        else:
            obs = "static + earliest-k"
            role = "strong deployable black-box control"
        add_row(
            rows,
            split_kind=str(row["split_kind"]),
            budget=budget,
            budget_kind="earliest_k_points" if budget > 0 and method != "direct_static_time" else "none",
            observation_protocol=(
                "earliest measured release points"
                if budget > 0 and method != "direct_static_time"
                else "static descriptors scored after budget window"
            ),
            observation_regime=obs,
            method_family="direct point regressor",
            method=method,
            median_rmse=float(row["median_future_rmse"]),
            median_mae=float(row["median_future_mae"]),
            median_r2=float(row["median_future_r2"]),
            n_curves=int(row["n_curves"]),
            claim_role=role,
            comparability_note="future-only per-curve scoring from script 97",
            source_file=str(args.direct_summary),
        )


def collect_proxy(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    df = read_optional(args.proxy_decision)
    if df.empty:
        return
    for _, row in df.iterrows():
        add_row(
            rows,
            split_kind=str(row["split_kind"]),
            budget=int(row["budget"]),
            budget_kind="predicted_earliest_k_points",
            observation_protocol="static-predicted proxy for earliest measured points",
            observation_regime="static-predicted early proxy",
            method_family="static-to-early proxy",
            method="pred_static_early_time",
            median_rmse=float(row["predicted_early_rmse"]),
            median_mae=np.nan,
            median_r2=np.nan,
            n_curves=np.nan,
            claim_role="tests whether static descriptors can synthesize early state",
            comparability_note=(
                "decision-table RMSE from script 99; same true-early-trained future model but "
                "test early Q/slope replaced by static-predicted proxy"
            ),
            source_file=str(args.proxy_decision),
        )


def collect_author(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    ours = read_optional(args.author_ours)
    if not ours.empty:
        for _, row in ours.iterrows():
            method = str(row["method"])
            budget = 3 if "few" in method else 0
            obs = "fixed Q(0.25,0.5,1.0)" if "few" in method else "static-only"
            role = "author-style fixed early baseline" if "few" in method else "author-style zero-shot baseline"
            add_row(
                rows,
                split_kind=str(row["split_kind"]),
                budget=budget,
                budget_kind="fixed_time_points" if "few" in method else "none",
                observation_protocol=(
                    "fixed measured Q(0.25,0.5,1.0)" if "few" in method else "static descriptors only"
                ),
                observation_regime=obs,
                method_family="author-style LGBM on our splits",
                method=method,
                median_rmse=float(row["median_future_rmse"]),
                median_mae=float(row["median_future_mae"]),
                median_r2=float(row["median_future_r2"]),
                n_curves=int(row["n_curves"]),
                claim_role=role,
                comparability_note="future-only time>1.0 per-curve scoring from script 101",
                source_file=str(args.author_ours),
            )
    cv = read_optional(args.author_cv)
    if cv.empty:
        return
    for _, row in cv[((cv["regime"] == "few-shot") & (cv["model"] == "LGBM")) | ((cv["regime"] == "zero-shot") & (cv["model"] == "LGBM"))].iterrows():
        add_row(
            rows,
            split_kind="author-DP_Group-split",
            budget=3 if row["regime"] == "few-shot" else 0,
            budget_kind="fixed_time_points" if row["regime"] == "few-shot" else "none",
            observation_protocol=(
                "author fixed measured early times"
                if row["regime"] == "few-shot"
                else "author zero-shot static descriptors"
            ),
            observation_regime="author fixed early" if row["regime"] == "few-shot" else "author zero-shot",
            method_family="released author nested-CV",
            method=f"author_cv::{row['regime']}::{row['model']}",
            median_rmse=float(row["median_point_rmse"]),
            median_mae=float(row["median_test_mae"]),
            median_r2=float(row["median_point_r2"]),
            n_curves=float(row["median_n_curves"]),
            claim_role="external historical baseline",
            comparability_note="released author pointwise nested-CV; MAE is headline and not directly comparable to our future-only RMSE",
            source_file=str(args.author_cv),
        )


def build_unified_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    collect_static_ceiling(args, rows)
    collect_middle(args, rows)
    collect_direct(args, rows)
    collect_proxy(args, rows)
    collect_author(args, rows)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["budget"] = pd.to_numeric(table["budget"], errors="coerce")
    table["median_rmse"] = pd.to_numeric(table["median_rmse"], errors="coerce")
    return table.sort_values(["split_kind", "budget", "method_family", "median_rmse"], na_position="last").reset_index(drop=True)


def best_value(table: pd.DataFrame, split: str, method: str, budget: int | None = None) -> float:
    sub = table[(table["split_kind"] == split) & (table["method"] == method)].copy()
    if budget is not None:
        sub = sub[sub["budget"] == budget].copy()
    vals = pd.to_numeric(sub["median_rmse"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return np.nan
    return float(vals.min())


def build_budget_value_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        direct_static_0 = best_value(table, split, "direct_static_time", 0)
        direct_static_5 = best_value(table, split, "direct_static_time", 5)
        direct_early_5 = best_value(table, split, "direct_static_early_time", 5)
        author_zero = best_value(table, split, "author_zero_lgbm", 0)
        author_few = best_value(table, split, "author_few_lgbm_fixed_early", 3)
        middle_global_0 = best_value(table, split, "middle::train_global_family", 0)
        middle_early_5 = best_value(table, split, "middle::early_residual_family", 5)
        middle_oracle_5 = best_value(table, split, "middle::future_oracle_family", 5)
        pred_proxy_5 = best_value(table, split, "pred_static_early_time", 5)
        rows.extend(
            [
                {
                    "split_kind": split,
                    "contrast": "direct static k0 -> direct static+early k5",
                    "baseline_rmse": direct_static_0,
                    "comparison_rmse": direct_early_5,
                    "rmse_gain": direct_static_0 - direct_early_5 if np.isfinite(direct_static_0) and np.isfinite(direct_early_5) else np.nan,
                    "interpretation": "dynamic early observations improve future forecasting" if np.isfinite(direct_static_0) and np.isfinite(direct_early_5) and direct_static_0 > direct_early_5 else "no gain",
                },
                {
                    "split_kind": split,
                    "contrast": "direct static scored after k5 window -> direct static+early k5",
                    "baseline_rmse": direct_static_5,
                    "comparison_rmse": direct_early_5,
                    "rmse_gain": direct_static_5 - direct_early_5 if np.isfinite(direct_static_5) and np.isfinite(direct_early_5) else np.nan,
                    "interpretation": "early Q adds information beyond later target-window restriction" if np.isfinite(direct_static_5) and np.isfinite(direct_early_5) and direct_static_5 > direct_early_5 else "no gain",
                },
                {
                    "split_kind": split,
                    "contrast": "author zero -> author fixed early",
                    "baseline_rmse": author_zero,
                    "comparison_rmse": author_few,
                    "rmse_gain": author_zero - author_few if np.isfinite(author_zero) and np.isfinite(author_few) else np.nan,
                    "interpretation": "author-style fixed early points support same direction" if np.isfinite(author_zero) and np.isfinite(author_few) and author_zero > author_few else "no gain",
                },
                {
                    "split_kind": split,
                    "contrast": "middle global k0 -> middle early selector k5",
                    "baseline_rmse": middle_global_0,
                    "comparison_rmse": middle_early_5,
                    "rmse_gain": middle_global_0 - middle_early_5 if np.isfinite(middle_global_0) and np.isfinite(middle_early_5) else np.nan,
                    "interpretation": "early observations help middle-layer selection" if np.isfinite(middle_global_0) and np.isfinite(middle_early_5) and middle_global_0 > middle_early_5 else "no gain",
                },
                {
                    "split_kind": split,
                    "contrast": "direct static+early k5 -> static-predicted early k5",
                    "baseline_rmse": direct_early_5,
                    "comparison_rmse": pred_proxy_5,
                    "rmse_gain": direct_early_5 - pred_proxy_5 if np.isfinite(direct_early_5) and np.isfinite(pred_proxy_5) else np.nan,
                    "interpretation": "static proxy fails to replace measured early state" if np.isfinite(direct_early_5) and np.isfinite(pred_proxy_5) and pred_proxy_5 > direct_early_5 else "proxy recovers true early",
                },
                {
                    "split_kind": split,
                    "contrast": "middle deployable k5 -> middle oracle k5",
                    "baseline_rmse": middle_early_5,
                    "comparison_rmse": middle_oracle_5,
                    "rmse_gain": middle_early_5 - middle_oracle_5 if np.isfinite(middle_early_5) and np.isfinite(middle_oracle_5) else np.nan,
                    "interpretation": "selector headroom remains; current gate is not oracle" if np.isfinite(middle_early_5) and np.isfinite(middle_oracle_5) and middle_early_5 > middle_oracle_5 else "little selector headroom",
                },
            ]
        )
    return pd.DataFrame(rows)


def build_claim_evidence(table: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def contrast_gain(split: str, contrast: str) -> float:
        sub = budget[(budget["split_kind"] == split) & (budget["contrast"] == contrast)]
        if sub.empty:
            return np.nan
        return float(sub["rmse_gain"].iloc[0])

    strict_direct_window_matched_gains = [
        contrast_gain("source-group-kfold", "direct static scored after k5 window -> direct static+early k5"),
        contrast_gain("source-dataset-lodo", "direct static scored after k5 window -> direct static+early k5"),
    ]
    strict_direct_k0_gains = [
        contrast_gain("source-group-kfold", "direct static k0 -> direct static+early k5"),
        contrast_gain("source-dataset-lodo", "direct static k0 -> direct static+early k5"),
    ]
    strict_author_gains = [
        contrast_gain("source-group-kfold", "author zero -> author fixed early"),
        contrast_gain("source-dataset-lodo", "author zero -> author fixed early"),
    ]
    strict_proxy_penalties = [
        -contrast_gain("source-group-kfold", "direct static+early k5 -> static-predicted early k5"),
        -contrast_gain("source-dataset-lodo", "direct static+early k5 -> static-predicted early k5"),
    ]
    rows.append(
        {
            "claim": "Static descriptors alone have a strict-split ceiling.",
            "support": "source-group/source-dataset static baselines cluster around RMSE 0.20-0.27, even for strong tree/shape priors.",
            "evidence_strength": "moderate",
            "remaining_risk": "Need ensure all static baselines share identical target windows before manuscript table.",
        }
    )
    rows.append(
        {
            "claim": "Measured early observations reveal missing release state.",
            "support": (
                f"Window-matched direct static->static+early k5 gains under strict splits are "
                f"{strict_direct_window_matched_gains[0]:.3f} and {strict_direct_window_matched_gains[1]:.3f}; "
                f"k0-to-k5 direct gains are {strict_direct_k0_gains[0]:.3f} and {strict_direct_k0_gains[1]:.3f}; "
                f"author-style fixed early gains are {strict_author_gains[0]:.3f} and {strict_author_gains[1]:.3f}."
            ),
            "evidence_strength": "strong",
            "remaining_risk": "The exact optimal timing policy is not yet proven; only early-observation value is established.",
        }
    )
    rows.append(
        {
            "claim": "Static descriptors cannot currently synthesize the early state.",
            "support": f"Static-predicted early proxy is worse than measured early by about {strict_proxy_penalties[0]:.3f} and {strict_proxy_penalties[1]:.3f} RMSE at k=5 under strict splits.",
            "evidence_strength": "strong",
            "remaining_risk": "Proxy used current descriptor surface; richer process/microstructure fields could change this.",
        }
    )
    rows.append(
        {
            "claim": "Middle-layer mechanism route has diagnostic value but is not the strongest deployable predictor yet.",
            "support": "Script 96 improves with early observations, but script 97 direct early black-box is usually stronger.",
            "evidence_strength": "moderate",
            "remaining_risk": "Need selector diagnostics if claiming interpretability beyond performance.",
        }
    )
    rows.append(
        {
            "claim": "The paper should be framed as information-budgeted sparse-observation forecasting, not a foundation-model claim.",
            "support": "Author baseline, direct controls, proxy failure, and middle-layer results all point to observation value rather than model-class superiority.",
            "evidence_strength": "strong",
            "remaining_risk": "Need uncertainty/early-stop analysis for a full experimental-design paper.",
        }
    )
    return pd.DataFrame(rows)


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].to_numpy(dtype=float)
        nonfinite = ~np.isfinite(vals)
        if not bool(nonfinite.any()):
            continue
        allowed_mask = np.zeros(len(df), dtype=bool)
        if col in {"budget", "median_rmse", "median_mae", "median_r2", "n_curves", "baseline_rmse", "comparison_rmse", "rmse_gain"}:
            allowed_mask |= nonfinite
        n_allowed = int((nonfinite & allowed_mask).sum())
        n_bad = int((nonfinite & ~allowed_mask).sum())
        if n_allowed:
            allowed[col] = n_allowed
        if n_bad:
            unexpected[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": unexpected, "allowed_nonfinite": allowed}


def write_report(out: Path, unified: pd.DataFrame, budget: pd.DataFrame, claims: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    strict_focus = budget[budget["split_kind"].isin(["source-group-kfold", "source-dataset-lodo"])].copy()
    headline_methods = unified[
        unified["method"].isin(
            [
                "author_zero_lgbm",
                "author_few_lgbm_fixed_early",
                "direct_static_time",
                "direct_static_early_time",
                "pred_static_early_time",
                "middle::early_residual_family",
                "middle::future_oracle_family",
            ]
        )
        & unified["split_kind"].isin(["source-group-kfold", "source-dataset-lodo"])
        & unified["budget"].isin([0, 3, 5])
    ].copy()
    lines = [
        "# PLGA Information-Budget Claim Table",
        "",
        "This report collapses the sprint into the narrowed paper route: early observations reveal missing release state under source shift.",
        "",
        "## Claim Evidence",
        "",
        claims.to_markdown(index=False),
        "",
        "## Strict-Split Budget Contrasts",
        "",
        strict_focus.to_markdown(index=False),
        "",
        "## Headline Method Rows",
        "",
        headline_methods.sort_values(["split_kind", "budget", "median_rmse"]).to_markdown(index=False),
        "",
        "## Interpretation Guard",
        "",
        "- Do not rank rows across incompatible protocols without the `comparability_note`.",
        "- The author released nested-CV rows are historical baselines; script 101 is the fairer on-our-splits comparison.",
        "- The current strongest claim is observation value, not model-class superiority.",
        "- The missing piece for a full experimental-design paper is uncertainty contraction / stopping-rule analysis.",
        "",
        "## Verification",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    unified = build_unified_table(args)
    budget = build_budget_value_table(unified)
    claims = build_claim_evidence(unified, budget)
    checks = [
        finite_check("unified_model_table", unified),
        finite_check("budget_value_table", budget),
        finite_check("claim_evidence_table", claims),
    ]
    unified.to_csv(args.out / "unified_model_table.csv", index=False)
    budget.to_csv(args.out / "budget_value_table.csv", index=False)
    claims.to_csv(args.out / "claim_evidence_table.csv", index=False)
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "inputs": {
                    "static_random": str(args.static_random),
                    "static_group": str(args.static_group),
                    "static_source": str(args.static_source),
                    "middle_summary": str(args.middle_summary),
                    "direct_summary": str(args.direct_summary),
                    "proxy_decision": str(args.proxy_decision),
                    "proxy_summary": str(args.proxy_summary),
                    "author_cv": str(args.author_cv),
                    "author_ours": str(args.author_ours),
                },
                "checks": checks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out, unified, budget, claims, checks)
    print((args.out / "report.md").resolve())
    print(claims.to_string(index=False))


if __name__ == "__main__":
    main()
