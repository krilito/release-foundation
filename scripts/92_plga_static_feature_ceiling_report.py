"""92 - Consolidate PLGA static-feature ceiling probe runs.

Consumes:
    outputs/91_plga_static_feature_ceiling_random_kfold/
    outputs/91_plga_static_feature_ceiling_source_group_kfold/
    outputs/91_plga_static_feature_ceiling_source_dataset_lodo/

Produces:
    outputs/92_plga_static_feature_ceiling_report/
      combined_summary.csv
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
DEFAULT_OUT = Path("outputs/92_plga_static_feature_ceiling_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate PLGA static-feature ceiling probe.")
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


def load_runs(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for label, run in [
        ("random_kfold", args.random_run),
        ("source_group_kfold", args.source_group_run),
        ("source_dataset_lodo", args.source_dataset_run),
    ]:
        df = read_csv(run / "summary_by_config.csv").copy()
        df.insert(0, "run_label", label)
        df.insert(1, "run_dir", str(run))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_decisions(summary: pd.DataFrame) -> pd.DataFrame:
    best = summary.sort_values(["split_kind", "median_rmse"]).groupby("split_kind", as_index=False).head(1)
    random_best = best[best["split_kind"] == "random-kfold"].iloc[0]
    group_best = best[best["split_kind"] == "source-group-kfold"].iloc[0]
    source_best = best[best["split_kind"] == "source-dataset-lodo"].iloc[0]
    intercept = summary[summary["feature_group"] == "intercept_only"].sort_values(["split_kind", "median_rmse"])
    rows = [
        {
            "question": "Can static PLGA descriptors predict curves within-distribution?",
            "answer": "yes, in random-kfold upper-bound setting",
            "evidence": f"best random median RMSE={float(random_best['median_rmse']):.3f}, median R2={float(random_best['median_r2']):.3f}; config={random_best['feature_group']}+{random_best['family']}+{random_best['model']}",
        },
        {
            "question": "Which feature combination wins random-kfold?",
            "answer": "intrinsic/static descriptors without source fields",
            "evidence": f"{random_best['feature_group']} with {random_best['family']} parameters and {random_best['model']}",
        },
        {
            "question": "Does the static bridge survive source-group holdout?",
            "answer": "mostly no",
            "evidence": f"best source-group median RMSE={float(group_best['median_rmse']):.3f}, close to intercept-only {float(intercept[intercept['split_kind']=='source-group-kfold']['median_rmse'].min()):.3f}",
        },
        {
            "question": "Does the static bridge survive source-dataset shift?",
            "answer": "mostly no",
            "evidence": f"best source-dataset median RMSE={float(source_best['median_rmse']):.3f}, close to intercept-only {float(intercept[intercept['split_kind']=='source-dataset-lodo']['median_rmse'].min()):.3f}",
        },
        {
            "question": "Immediate interpretation",
            "answer": "static descriptors contain real PLGA information, but the bridge is source-local",
            "evidence": "Random split success should be treated as descriptor ceiling, not deployment evidence. The next test needs layered static/process/environment features and stricter grouped splits.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(out: Path, summary: pd.DataFrame, decisions: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    top = summary.sort_values(["split_kind", "median_rmse"]).groupby("split_kind", as_index=False).head(10)
    random_top = summary[summary["split_kind"] == "random-kfold"].sort_values("median_rmse").head(20)
    strict_top = summary[summary["split_kind"].isin(["source-group-kfold", "source-dataset-lodo"])].sort_values(["split_kind", "median_rmse"]).groupby("split_kind", as_index=False).head(10)
    lines = [
        "# PLGA Static-Feature Ceiling Report",
        "",
        "This report is a descriptor-only probe. It uses no early release observations as inputs.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Best Configurations By Split",
        "",
        top.to_markdown(index=False),
        "",
        "## Random-KFold Upper-Bound Top 20",
        "",
        random_top.to_markdown(index=False),
        "",
        "## Strict Split Top Configurations",
        "",
        strict_top.to_markdown(index=False),
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
    summary = load_runs(args)
    decisions = build_decisions(summary)
    checks = [finite_numeric_check("combined_summary", summary), finite_numeric_check("decision_table", decisions)]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")
    summary.to_csv(args.out / "combined_summary.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    write_report(args.out, summary, decisions, checks)
    metadata = {
        "script": "scripts/92_plga_static_feature_ceiling_report.py",
        "git_hash": git_hash(),
        "args": args_to_metadata(args),
        "out": str(args.out),
        "runs": {
            "random": str(args.random_run),
            "source_group": str(args.source_group_run),
            "source_dataset": str(args.source_dataset_run),
        },
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[92] wrote", args.out)
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
