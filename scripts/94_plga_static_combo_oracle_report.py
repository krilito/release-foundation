"""94 - PLGA static feature-combo oracle report.

Purpose:
    Answer the first conditional-feature question:

        If each heldout curve is allowed to choose its best static feature
        combo after seeing the error, can static descriptors predict well?

This is explicitly a diagnostic upper bound, not a deployable predictor.

Consumes:
    outputs/91_plga_static_feature_ceiling_random_kfold/per_curve_metrics.csv
    outputs/91_plga_static_feature_ceiling_source_group_kfold/per_curve_metrics.csv
    outputs/91_plga_static_feature_ceiling_source_dataset_lodo/per_curve_metrics.csv

Produces:
    outputs/94_plga_static_combo_oracle_report/
      oracle_per_curve.csv
      oracle_summary.csv
      oracle_combo_frequency.csv
      oracle_gap_by_curve.csv
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


DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/94_plga_static_combo_oracle_report")
SOURCE_FEATURE_GROUPS = {"source_diagnostic", "all_with_source_diagnostic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PLGA static-combo test-oracle report.")
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


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


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(vals.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def load_per_curve(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for label, run in [
        ("random_kfold", args.random_run),
        ("source_group_kfold", args.source_group_run),
        ("source_dataset_lodo", args.source_dataset_run),
    ]:
        df = read_csv(run / "per_curve_metrics.csv").copy()
        df.insert(0, "run_label", label)
        df.insert(1, "run_dir", str(run))
        df["config_id"] = df["feature_group"].astype(str) + "|" + df["family"].astype(str) + "|" + df["model"].astype(str)
        df["uses_source_diagnostic"] = df["feature_group"].astype(str).isin(SOURCE_FEATURE_GROUPS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def best_rows(df: pd.DataFrame, label: str, mask: pd.Series) -> pd.DataFrame:
    key_cols = ["run_label", "split_kind", "fold", "unified_curve_id"]
    sub = df[mask].copy()
    idx = sub.sort_values("rmse").groupby(key_cols, dropna=False).head(1).index
    best = sub.loc[idx].copy()
    best["selector"] = label
    return best


def build_oracle_tables(per_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.append(best_rows(per_curve, "intercept_only", per_curve["feature_group"].astype(str) == "intercept_only"))
    rows.append(best_rows(per_curve, "global_deployable_oracle", ~per_curve["uses_source_diagnostic"].astype(bool)))
    rows.append(best_rows(per_curve, "all_config_oracle", pd.Series(True, index=per_curve.index)))
    oracle = pd.concat(rows, ignore_index=True)

    key_cols = ["run_label", "split_kind", "fold", "unified_curve_id"]
    wide = oracle.pivot_table(
        index=key_cols,
        columns="selector",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    wide["deployable_oracle_gain_vs_intercept"] = wide["intercept_only"] - wide["global_deployable_oracle"]
    wide["all_oracle_gain_vs_deployable"] = wide["global_deployable_oracle"] - wide["all_config_oracle"]
    wide["deployable_oracle_ratio_to_intercept"] = wide["global_deployable_oracle"] / wide["intercept_only"].clip(lower=1e-12)

    summary = (
        oracle.groupby(["run_label", "split_kind", "selector"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_mae=("mae", "median"),
            median_r2=("r2", "median"),
            median_release_span=("release_span", "median"),
            median_duration_days=("duration_days", "median"),
        )
        .reset_index()
        .sort_values(["split_kind", "selector"])
    )
    gap = (
        wide.groupby(["run_label", "split_kind"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_intercept_rmse=("intercept_only", "median"),
            median_deployable_oracle_rmse=("global_deployable_oracle", "median"),
            median_all_oracle_rmse=("all_config_oracle", "median"),
            median_gain_vs_intercept=("deployable_oracle_gain_vs_intercept", "median"),
            mean_gain_vs_intercept=("deployable_oracle_gain_vs_intercept", "mean"),
            median_ratio_to_intercept=("deployable_oracle_ratio_to_intercept", "median"),
            fraction_oracle_beats_intercept=("deployable_oracle_gain_vs_intercept", lambda s: float(np.mean(s > 0.0))),
            median_source_diagnostic_gain=("all_oracle_gain_vs_deployable", "median"),
        )
        .reset_index()
        .sort_values("split_kind")
    )
    freq = (
        oracle[oracle["selector"] == "global_deployable_oracle"]
        .groupby(["run_label", "split_kind", "feature_group", "family", "model"], dropna=False)
        .agg(n_curves=("unified_curve_id", "nunique"), median_rmse=("rmse", "median"))
        .reset_index()
        .sort_values(["split_kind", "n_curves", "median_rmse"], ascending=[True, False, True])
    )
    return oracle, summary, freq, gap


def build_decisions(summary: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        g = gap[gap["split_kind"] == split_kind].iloc[0]
        oracle_rmse = float(g["median_deployable_oracle_rmse"])
        ratio = float(g["median_ratio_to_intercept"])
        frac = float(g["fraction_oracle_beats_intercept"])
        if oracle_rmse <= 0.12:
            answer = "yes, static combos can be very good"
        elif ratio <= 0.75:
            answer = "partly, oracle improves meaningfully"
        else:
            answer = "weak, oracle barely improves over intercept"
        rows.append(
            {
                "split_kind": split_kind,
                "question": "Can per-curve static feature-combo oracle predict well?",
                "answer": answer,
                "evidence": f"median oracle RMSE={oracle_rmse:.3f}; ratio to intercept={ratio:.3f}; beats intercept on {frac:.1%} curves",
            }
        )
    random_oracle = gap[gap["split_kind"] == "random-kfold"].iloc[0]
    strict_group = gap[gap["split_kind"] == "source-group-kfold"].iloc[0]
    rows.append(
        {
            "split_kind": "overall",
            "question": "Does per-curve oracle change the earlier diagnosis?",
            "answer": "yes for within-distribution ceiling, no for source-shift deployment",
            "evidence": (
                f"random oracle RMSE={float(random_oracle['median_deployable_oracle_rmse']):.3f}; "
                f"source-group oracle RMSE={float(strict_group['median_deployable_oracle_rmse']):.3f}"
            ),
        }
    )
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    summary: pd.DataFrame,
    freq: pd.DataFrame,
    gap: pd.DataFrame,
    decisions: pd.DataFrame,
    oracle: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    random_freq = freq[freq["split_kind"] == "random-kfold"].head(20)
    strict_gap = gap[gap["split_kind"].isin(["source-group-kfold", "source-dataset-lodo"])]
    sample_oracle = (
        oracle[oracle["selector"] == "global_deployable_oracle"]
        .sort_values(["split_kind", "rmse"])
        .groupby("split_kind", as_index=False)
        .head(10)
    )
    lines = [
        "# PLGA Static Combo Oracle Report",
        "",
        "This is a diagnostic upper bound. The oracle selector sees heldout errors and is not a deployable prediction rule.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Oracle Gap Summary",
        "",
        gap.to_markdown(index=False),
        "",
        "## Selector Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Random-KFold Deployable Oracle Combo Frequency",
        "",
        random_freq.to_markdown(index=False),
        "",
        "## Strict Split Oracle Gap",
        "",
        strict_gap.to_markdown(index=False),
        "",
        "## Example Best-Fit Curves By Split",
        "",
        sample_oracle[
            [
                "split_kind",
                "unified_curve_id",
                "feature_group",
                "family",
                "model",
                "rmse",
                "r2",
                "source_dataset",
                "source_group",
            ]
        ].to_markdown(index=False),
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
    per_curve = load_per_curve(args)
    oracle, summary, freq, gap = build_oracle_tables(per_curve)
    decisions = build_decisions(summary, gap)
    checks = [
        finite_numeric_check("oracle_per_curve", oracle),
        finite_numeric_check("oracle_summary", summary),
        finite_numeric_check("oracle_combo_frequency", freq),
        finite_numeric_check("oracle_gap_by_curve", gap),
        finite_numeric_check("decision_table", decisions),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")
    oracle.to_csv(args.out / "oracle_per_curve.csv", index=False)
    summary.to_csv(args.out / "oracle_summary.csv", index=False)
    freq.to_csv(args.out / "oracle_combo_frequency.csv", index=False)
    gap.to_csv(args.out / "oracle_gap_by_curve.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    write_report(args.out, summary, freq, gap, decisions, oracle, checks)
    metadata = {
        "script": "scripts/94_plga_static_combo_oracle_report.py",
        "git_hash": git_hash(),
        "args": args_to_metadata(args),
        "out": str(args.out),
        "note": "Diagnostic oracle only. Selector sees heldout errors and must not be reported as deployable prediction.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[94] wrote", args.out)
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
