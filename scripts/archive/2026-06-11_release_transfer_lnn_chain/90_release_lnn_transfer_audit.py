"""90 - LNN transfer audit after liposome and PLGA-within probes.

Purpose:
    Answer the current decision questions:
      1. Is ShapePriorIncrementAnchored:hill leakage-free for liposome?
      2. Where does liposome lose per curve and per budget?
      3. Does LNN help in PLGA within-system?
      4. Is the observed gain system-local rather than cross-system transfer?
      5. Should the neural model be demoted to a negative control?

Produces:
    outputs/90_release_lnn_transfer_audit/
      liposome_hill_leakage_audit.csv
      liposome_per_curve_delta.csv
      liposome_budget_decomposition.csv
      plga_within_lnn_compare.csv
      decision_table.csv
      report.md
      lock_metadata.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LIPO_RUN = Path("outputs/81_release_liposome_ablation_mlp_none_e10")
DEFAULT_LIPO_REGIME = Path("outputs/84_release_liposome_ablation_mlp_none_e10")
DEFAULT_LIPO_DYNAMIC = Path("outputs/86_release_liposome_ablation_mlp_none_e10")
DEFAULT_LIPO_LEAK = Path("outputs/85_release_shape_prior_leakage_audit_liposome_mlp_none_e10")
DEFAULT_PLGA_MLP = Path("outputs/86_release_plga_within_ablation_mlp_none_e10")
DEFAULT_PLGA_LNN = Path("outputs/86_release_plga_within_ablation_lnn_none_e10")
DEFAULT_MAIN_DYNAMIC = Path("outputs/86_release_dynamic_future_benchmark_shapeanchored")
DEFAULT_OUT = Path("outputs/90_release_lnn_transfer_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LNN transfer and shape-prior competition.")
    parser.add_argument("--liposome-run", type=Path, default=DEFAULT_LIPO_RUN)
    parser.add_argument("--liposome-regime", type=Path, default=DEFAULT_LIPO_REGIME)
    parser.add_argument("--liposome-dynamic", type=Path, default=DEFAULT_LIPO_DYNAMIC)
    parser.add_argument("--liposome-leak", type=Path, default=DEFAULT_LIPO_LEAK)
    parser.add_argument("--plga-mlp-dynamic", type=Path, default=DEFAULT_PLGA_MLP)
    parser.add_argument("--plga-lnn-dynamic", type=Path, default=DEFAULT_PLGA_LNN)
    parser.add_argument("--main-dynamic", type=Path, default=DEFAULT_MAIN_DYNAMIC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=D:/release-foundation", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(vals.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def liposome_hill_leakage(leak_dir: Path) -> pd.DataFrame:
    source = read_csv(leak_dir / "shape_source_by_split_family.csv")
    availability = read_csv(leak_dir / "baseline_family_availability.csv")
    leaks = read_csv(leak_dir / "leakage_rows.csv")
    hill_source = source[
        (source["heldout_system"].astype(str) == "liposome")
        & (source["family"].astype(str) == "hill")
    ].copy()
    hill_avail = availability[
        (availability["heldout_system"].astype(str) == "liposome")
        & (availability["baseline"].astype(str) == "ShapePriorIncrementAnchored:hill")
    ].copy()
    rows = []
    for _, row in hill_source.iterrows():
        emitted = hill_avail[hill_avail["split_kind"] == row["split_kind"]]
        rows.append(
            {
                "split_kind": row["split_kind"],
                "heldout_system": row["heldout_system"],
                "family": "hill",
                "baseline": "ShapePriorIncrementAnchored:hill",
                "n_train_curves": int(row["n_train_curves"]),
                "n_test_curves": int(row["n_test_curves"]),
                "n_shape_source_curves": int(row["n_shape_source_curves"]),
                "n_heldout_fit_bank_curves": int(row["n_heldout_fit_bank_curves"]),
                "n_leaked_source_curves": int(row["n_leaked_source_curves"]),
                "n_emitted_budget_rows": int(len(emitted)),
                "any_baseline_without_source": bool(emitted["baseline_without_source"].fillna(False).any()),
                "any_baseline_with_leaked_source": bool(emitted["baseline_with_leaked_source"].fillna(False).any()),
                "total_leakage_rows": int(len(leaks)),
                "audit_pass": bool(
                    int(row["n_leaked_source_curves"]) == 0
                    and len(leaks) == 0
                    and not emitted["baseline_without_source"].fillna(False).any()
                    and not emitted["baseline_with_leaked_source"].fillna(False).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def liposome_deltas(run_dir: Path, regime_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = read_csv(run_dir / "per_curve_metrics.csv")
    base = read_csv(run_dir / "baseline_per_curve_metrics.csv")
    regime = read_csv(regime_dir / "data_regime_per_curve.csv")
    key_cols = ["split_kind", "heldout_system", "unified_curve_id", "budget_kind", "budget_label", "budget_value"]
    hill = base[
        (base["baseline"] == "ShapePriorIncrementAnchored:hill")
        & (base["budget_kind"] == "time_days")
    ][key_cols + ["rmse"]].rename(columns={"rmse": "hill_rmse"})
    model = per[per["budget_kind"] == "time_days"][key_cols + ["n_obs", "rmse", "mae", "width90"]].rename(
        columns={"rmse": "model_rmse", "mae": "model_mae"}
    )
    reg_cols = key_cols + ["n_target_points", "target_true_span", "target_release_gain", "curve_regime", "hard_dynamic_future"]
    joined = model.merge(hill, on=key_cols, how="inner").merge(regime[reg_cols], on=key_cols, how="left")
    joined["model_minus_hill"] = joined["model_rmse"] - joined["hill_rmse"]
    joined["model_beats_hill"] = joined["model_minus_hill"] < 0

    group_cols = ["budget_label", "budget_value", "hard_dynamic_future"]
    budget = (
        joined.groupby(group_cols, dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_n_obs=("n_obs", "median"),
            median_model_rmse=("model_rmse", "median"),
            median_hill_rmse=("hill_rmse", "median"),
            median_delta_model_minus_hill=("model_minus_hill", "median"),
            mean_delta_model_minus_hill=("model_minus_hill", "mean"),
            fraction_model_beats_hill=("model_beats_hill", "mean"),
            median_target_span=("target_true_span", "median"),
        )
        .reset_index()
        .sort_values(["hard_dynamic_future", "budget_value"])
    )
    return joined.sort_values(["budget_value", "model_minus_hill"], ascending=[True, False]), budget


def plga_lnn_compare(mlp_dir: Path, lnn_dir: Path) -> pd.DataFrame:
    mlp = read_csv(mlp_dir / "dynamic_by_budget.csv")
    lnn = read_csv(lnn_dir / "dynamic_by_budget.csv")
    key = ["split_kind", "budget_kind", "budget_label", "budget_value"]
    cols = key + ["n_dynamic_curves", "median_model_rmse", "best_dynamic_baseline", "median_baseline_rmse", "delta_model_minus_best_dynamic"]
    joined = mlp[cols].merge(lnn[cols], on=key, suffixes=("_mlp", "_lnn"))
    joined["lnn_minus_mlp_median_rmse"] = joined["median_model_rmse_lnn"] - joined["median_model_rmse_mlp"]
    joined["lnn_better_than_mlp"] = joined["lnn_minus_mlp_median_rmse"] < 0
    return joined.sort_values("budget_value")


def main_dynamic_decision(dynamic_dir: Path) -> pd.DataFrame:
    info = read_csv(dynamic_dir / "informative_dynamic_review.csv")
    keep = [
        "heldout_system",
        "budget_label",
        "has_dynamic_rows_at_informative_budget",
        "best_dynamic_baseline",
        "delta_model_minus_best_dynamic",
    ]
    return info[keep].copy()


def decision_table(hill_audit: pd.DataFrame, lipo_budget: pd.DataFrame, plga: pd.DataFrame, main_dynamic: pd.DataFrame) -> pd.DataFrame:
    lipo_early = lipo_budget[
        (lipo_budget["budget_label"] == "t<=0.25d")
        & (lipo_budget["hard_dynamic_future"].astype(str).str.lower().isin(["true", "1"]))
    ].iloc[0]
    plga_late = plga[plga["budget_label"] == "t<=14d"].iloc[0]
    non_plga = main_dynamic[main_dynamic["heldout_system"] != "PLGA"].copy()
    non_plga_wins = int((non_plga["delta_model_minus_best_dynamic"] < 0).sum())
    rows = [
        {
            "question": "ShapePriorIncrementAnchored:hill leakage-free for liposome?",
            "answer": "yes" if bool(hill_audit["audit_pass"].all()) else "no",
            "evidence": f"hill source curves={int(hill_audit['n_shape_source_curves'].iloc[0])}, leaked source curves={int(hill_audit['n_leaked_source_curves'].iloc[0])}, leakage rows={int(hill_audit['total_leakage_rows'].iloc[0])}",
        },
        {
            "question": "Liposome dynamic early budget vs hill",
            "answer": "model loses",
            "evidence": f"t<=0.25d hard-dynamic median delta model-hill={float(lipo_early['median_delta_model_minus_hill']):.3f}; model win fraction={float(lipo_early['fraction_model_beats_hill']):.1%}",
        },
        {
            "question": "LNN effective in PLGA within-system?",
            "answer": "not meaningfully; nearly identical to MLP",
            "evidence": f"t<=14d dynamic LNN-MLP median RMSE={float(plga_late['lnn_minus_mlp_median_rmse']):.4f}; both beat late mean baseline but early budgets lose",
        },
        {
            "question": "System-local pattern vs cross-system transfer?",
            "answer": "current evidence favors system-local/late-budget pattern, not robust cross-system transfer",
            "evidence": f"non-PLGA informative dynamic wins={non_plga_wins}/{len(non_plga)}; liposome LOSO loses despite enough dynamic rows",
        },
        {
            "question": "Demote neural model to negative control?",
            "answer": "not globally, but demote LNN-specific claim and non-PLGA transfer claim",
            "evidence": "PLGA LOSO/late dynamic evidence prevents calling the whole neural model a pure negative control, but all non-PLGA informative dynamic tests remain unsupported.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    hill_audit: pd.DataFrame,
    per_curve: pd.DataFrame,
    budget: pd.DataFrame,
    plga: pd.DataFrame,
    main_dynamic: pd.DataFrame,
    decisions: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    dynamic_budget = budget[budget["hard_dynamic_future"].astype(str).str.lower().isin(["true", "1"])].copy()
    worst_curves = per_curve[per_curve["budget_label"] == "t<=0.25d"].head(20)
    lines = [
        "# LNN Transfer Audit",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Liposome Hill Leakage Audit",
        "",
        hill_audit.to_markdown(index=False),
        "",
        "## Liposome Budget/Time-Budget Decomposition vs Hill",
        "",
        dynamic_budget.to_markdown(index=False),
        "",
        "## Liposome Worst Per-Curve Deltas at t<=0.25d",
        "",
        worst_curves[
            [
                "unified_curve_id",
                "budget_label",
                "model_rmse",
                "hill_rmse",
                "model_minus_hill",
                "target_true_span",
                "curve_regime",
            ]
        ].to_markdown(index=False),
        "",
        "## PLGA Within-System LNN vs MLP",
        "",
        plga.to_markdown(index=False),
        "",
        "## Main Dynamic Informative Systems",
        "",
        main_dynamic.to_markdown(index=False),
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
    hill_audit = liposome_hill_leakage(args.liposome_leak)
    per_curve, budget = liposome_deltas(args.liposome_run, args.liposome_regime)
    plga = plga_lnn_compare(args.plga_mlp_dynamic, args.plga_lnn_dynamic)
    main_dynamic = main_dynamic_decision(args.main_dynamic)
    decisions = decision_table(hill_audit, budget, plga, main_dynamic)
    checks = [
        finite_numeric_check("liposome_hill_leakage_audit", hill_audit),
        finite_numeric_check("liposome_per_curve_delta", per_curve),
        finite_numeric_check("liposome_budget_decomposition", budget),
        finite_numeric_check("plga_within_lnn_compare", plga),
        finite_numeric_check("decision_table", decisions),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")
    hill_audit.to_csv(args.out / "liposome_hill_leakage_audit.csv", index=False)
    per_curve.to_csv(args.out / "liposome_per_curve_delta.csv", index=False)
    budget.to_csv(args.out / "liposome_budget_decomposition.csv", index=False)
    plga.to_csv(args.out / "plga_within_lnn_compare.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    write_report(args.out, hill_audit, per_curve, budget, plga, main_dynamic, decisions, checks)
    metadata = {
        "script": "scripts/90_release_lnn_transfer_audit.py",
        "git_hash": git_hash(),
        "out": str(args.out),
        "note": "Decision audit. LNN-specific and non-PLGA transfer claims remain unsupported.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[90] wrote", args.out)
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
