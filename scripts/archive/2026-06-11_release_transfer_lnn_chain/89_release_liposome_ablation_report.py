"""89 - Liposome ablation report for release transfer.

Purpose:
    Consolidate liposome-only ablations from script 81 and the 83/84/86
    diagnostics into one small decision report.

Consumes:
    outputs/83_release_liposome_ablation_*_e10/named_best_by_system_budget.csv
    outputs/84_release_liposome_ablation_*_e10/informative_budget_review.csv
    outputs/86_release_liposome_ablation_*_e10/dynamic_by_budget.csv

Produces:
    outputs/89_release_liposome_ablation_report/
      ablation_dynamic_summary.csv
      ablation_named_best_summary.csv
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


DEFAULT_OUT = Path("outputs/89_release_liposome_ablation_report")
DEFAULT_VARIANTS = {
    "mlp_none_e10": {
        "run81": Path("outputs/81_release_liposome_ablation_mlp_none_e10"),
        "failure83": Path("outputs/83_release_liposome_ablation_mlp_none_e10"),
        "regime84": Path("outputs/84_release_liposome_ablation_mlp_none_e10"),
        "dynamic86": Path("outputs/86_release_liposome_ablation_mlp_none_e10"),
    },
    "mlp_safe_e10": {
        "run81": Path("outputs/81_release_liposome_ablation_mlp_safe_e10"),
        "failure83": Path("outputs/83_release_liposome_ablation_mlp_safe_e10"),
        "regime84": Path("outputs/84_release_liposome_ablation_mlp_safe_e10"),
        "dynamic86": Path("outputs/86_release_liposome_ablation_mlp_safe_e10"),
    },
    "lnn_none_e10": {
        "run81": Path("outputs/81_release_liposome_ablation_lnn_none_e10"),
        "failure83": Path("outputs/83_release_liposome_ablation_lnn_none_e10"),
        "regime84": Path("outputs/84_release_liposome_ablation_lnn_none_e10"),
        "dynamic86": Path("outputs/86_release_liposome_ablation_lnn_none_e10"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build liposome ablation decision report.")
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
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(values.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    dynamic_rows = []
    named_rows = []
    informative_rows = []
    checks: list[dict[str, Any]] = []
    for variant, paths in DEFAULT_VARIANTS.items():
        dynamic = read_csv(paths["dynamic86"] / "dynamic_by_budget.csv")
        named = read_csv(paths["failure83"] / "named_best_by_system_budget.csv")
        informative = read_csv(paths["regime84"] / "informative_budget_review.csv")
        checks.extend(
            [
                finite_numeric_check(f"{variant}:dynamic_by_budget", dynamic),
                finite_numeric_check(f"{variant}:named_best_by_system_budget", named),
                finite_numeric_check(f"{variant}:informative_budget_review", informative),
            ]
        )

        dynamic = dynamic.copy()
        dynamic.insert(0, "variant", variant)
        dynamic_rows.append(dynamic)

        named = named.copy()
        named.insert(0, "variant", variant)
        named_rows.append(named)

        informative = informative.copy()
        informative.insert(0, "variant", variant)
        informative_rows.append(informative)

    return pd.concat(dynamic_rows, ignore_index=True), pd.concat(named_rows, ignore_index=True), pd.concat(informative_rows, ignore_index=True), checks


def write_report(out: Path, dynamic: pd.DataFrame, named: pd.DataFrame, informative: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    early = dynamic[dynamic["budget_label"] == "t<=0.25d"].sort_values("delta_model_minus_best_dynamic")
    later = dynamic[dynamic["budget_label"].isin(["t<=1d", "t<=3d"])].sort_values(["budget_value", "delta_model_minus_best_dynamic"])
    info = informative.sort_values(["variant", "budget_value"])
    named_early = named[named["budget_label"] == "t<=0.25d"].sort_values("delta_model_minus_best_named")

    best_early = early.iloc[0]
    lines = [
        "# Liposome Ablation Report",
        "",
        "Scope: liposome-only LOSO ablations from script `81`, evaluated with the same strong baseline and dynamic-future diagnostics as the main transfer probe.",
        "",
        "## Decision",
        "",
        (
            "The current liposome bottleneck is not solved by switching MLP to LNN or by adding safe metadata. "
            f"At the main dynamic early budget (`t<=0.25d`), the best variant is `{best_early['variant']}` "
            f"but it still trails `{best_early['best_dynamic_baseline']}` by "
            f"{float(best_early['delta_model_minus_best_dynamic']):.3f} median RMSE."
        ),
        "",
        "Interpretation: liposome remains a modeling/data target, not a claimed cross-system win. The next useful test is not larger LNN by default; it is a shape-prior-aware adapter or better liposome descriptors/process fields.",
        "",
        "## Dynamic-Future Summary",
        "",
        early.to_markdown(index=False),
        "",
        "## Later Dynamic Budgets",
        "",
        later.to_markdown(index=False),
        "",
        "## Named-Best Baseline Summary at t<=0.25d",
        "",
        named_early.to_markdown(index=False),
        "",
        "## Informative-Budget Review",
        "",
        info.to_markdown(index=False),
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
    dynamic, named, informative, checks = build_tables()
    dynamic.to_csv(args.out / "ablation_dynamic_summary.csv", index=False)
    named.to_csv(args.out / "ablation_named_best_summary.csv", index=False)
    informative.to_csv(args.out / "ablation_informative_review.csv", index=False)
    write_report(args.out, dynamic, named, informative, checks)
    metadata = {
        "script": "scripts/89_release_liposome_ablation_report.py",
        "git_hash": git_hash(),
        "out": str(args.out),
        "variants": {name: {key: str(value) for key, value in paths.items()} for name, paths in DEFAULT_VARIANTS.items()},
        "note": "Decision report only. Do not claim liposome transfer win from these ablations.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[89] wrote", args.out)
    print(dynamic[dynamic["budget_label"] == "t<=0.25d"].to_string(index=False))


if __name__ == "__main__":
    main()
