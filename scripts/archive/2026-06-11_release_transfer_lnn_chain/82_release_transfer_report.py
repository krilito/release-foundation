"""82 - Summarize release-corpus sparse-observation transfer runs.

Consumes selected outputs from script 81 and writes a compact, reproducible
report for the current transfer probe:

    outputs/82_release_transfer_report/

This is a reporting layer only. It does not train models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUT = Path("outputs/82_release_transfer_report")
DEFAULT_HEADLINE = Path("outputs/81_release_corpus_time_budget_lnn_e10")
DEFAULT_METADATA = {
    "none": Path("outputs/81_release_corpus_time_budget_lnn_meta_none_e5"),
    "safe": Path("outputs/81_release_corpus_time_budget_lnn_meta_safe_e5"),
    "all": Path("outputs/81_release_corpus_time_budget_lnn_meta_all_e5"),
}
DEFAULT_DECODER_COMPARE = Path("outputs/81_release_corpus_anchor_compare_e10.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize release transfer probe outputs.")
    parser.add_argument("--headline-run", type=Path, default=DEFAULT_HEADLINE)
    parser.add_argument("--decoder-compare", type=Path, default=DEFAULT_DECODER_COMPARE)
    parser.add_argument("--metadata-none", type=Path, default=DEFAULT_METADATA["none"])
    parser.add_argument("--metadata-safe", type=Path, default=DEFAULT_METADATA["safe"])
    parser.add_argument("--metadata-all", type=Path, default=DEFAULT_METADATA["all"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=D:/release-foundation", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def check_no_numeric_nan(name: str, df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=["number"])
    bad = numeric.replace([float("inf"), float("-inf")], pd.NA).isna().sum()
    bad = bad[bad > 0]
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad.to_dict()}


def load_headline(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = read_csv(run / "summary_by_budget.csv")
    system_summary = read_csv(run / "summary_by_system.csv")
    time_summary = read_csv(run / "summary_by_time_budget.csv")
    baseline_delta = read_csv(run / "baseline_delta_summary.csv")
    return summary, system_summary, time_summary, baseline_delta


def load_metadata_compare(paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for mode, path in paths.items():
        df = read_csv(path / "summary_by_budget.csv")
        df["metadata_mode"] = mode
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def normalize_decoder_compare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "budget_kind" not in out.columns:
        out["budget_kind"] = "point"
    if "budget_value" not in out.columns:
        out["budget_value"] = pd.to_numeric(out["n_obs"], errors="coerce")
    if "budget_label" not in out.columns:
        out["budget_label"] = "k=" + out["budget_value"].astype(int).astype(str)
    return out


def make_time_budget_plot(time_summary: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for split_kind, sub in time_summary.groupby("split_kind"):
        sub = sub.sort_values("budget_value")
        ax.plot(sub["budget_value"], sub["median_rmse"], marker="o", label=split_kind)
    ax.set_xscale("log")
    ax.set_xlabel("context horizon (days)")
    ax.set_ylabel("median RMSE")
    ax.set_title("Release transfer: time-budget curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out / "time_budget_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_metadata_plot(metadata: pd.DataFrame, out: Path) -> Path:
    sub = metadata[(metadata["split_kind"] == "loso-system") & (metadata["budget_kind"] == "time_days")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for mode, mode_df in sub.groupby("metadata_mode"):
        mode_df = mode_df.sort_values("budget_value")
        ax.plot(mode_df["budget_value"], mode_df["median_rmse"], marker="o", label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("context horizon (days)")
    ax.set_ylabel("median RMSE")
    ax.set_title("Metadata sensitivity under LOSO")
    ax.grid(True, alpha=0.25)
    ax.legend(title="metadata", frameon=False)
    fig.tight_layout()
    path = out / "metadata_sensitivity_loso.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_decoder_plot(decoder: pd.DataFrame, out: Path) -> Path:
    sub = decoder[(decoder["split_kind"] == "loso-system") & (decoder["budget_kind"] == "point")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(sub["budget_value"], sub["median_rmse_lnn"], marker="o", label="lnn")
    ax.plot(sub["budget_value"], sub["median_rmse_mlp"], marker="o", label="mlp")
    ax.set_xlabel("context points")
    ax.set_ylabel("median RMSE")
    ax.set_title("Decoder ablation under LOSO")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out / "decoder_ablation_loso.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def system_time_budget(system_summary: pd.DataFrame) -> pd.DataFrame:
    sub = system_summary[
        (system_summary["split_kind"] == "loso-system")
        & (system_summary["budget_kind"] == "time_days")
    ].copy()
    return sub.sort_values(["heldout_system", "budget_value"])


def hard_systems(system_time: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, sub in system_time.groupby("heldout_system"):
        sub = sub.sort_values("budget_value")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        reduction = 1.0 - float(last["median_rmse"]) / max(float(first["median_rmse"]), 1e-12)
        rows.append(
            {
                "heldout_system": system,
                "n_curves": int(last["n_curves"]),
                "first_budget": first["budget_label"],
                "first_median_rmse": float(first["median_rmse"]),
                "last_budget": last["budget_label"],
                "last_median_rmse": float(last["median_rmse"]),
                "relative_rmse_reduction": float(reduction),
                "last_mean_crps": float(last["mean_crps"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["last_median_rmse", "relative_rmse_reduction"], ascending=[False, True])


def make_system_time_plot(system_time: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for system, sub in system_time.groupby("heldout_system"):
        sub = sub.sort_values("budget_value")
        ax.plot(sub["budget_value"], sub["median_rmse"], marker="o", linewidth=1.2, alpha=0.75, label=system)
    ax.set_xscale("log")
    ax.set_xlabel("context horizon (days)")
    ax.set_ylabel("median RMSE")
    ax.set_title("LOSO time-budget curves by held-out system")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path = out / "system_time_budget_loso.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def top_baseline_delta(delta: pd.DataFrame) -> pd.DataFrame:
    sub = delta[(delta["split_kind"] == "loso-system") & (delta["budget_kind"] == "time_days")].copy()
    return sub.sort_values(["budget_value", "median_baseline", "median_delta_model_minus_baseline"])


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
    return view.to_markdown(index=False)


def write_report(
    out: Path,
    headline_run: Path,
    summary: pd.DataFrame,
    system_time: pd.DataFrame,
    hard: pd.DataFrame,
    time_summary: pd.DataFrame,
    baseline_delta: pd.DataFrame,
    metadata: pd.DataFrame,
    decoder: pd.DataFrame,
    checks: list[dict[str, Any]],
    plots: dict[str, Path],
) -> None:
    loso_time = time_summary[time_summary["split_kind"] == "loso-system"].copy()
    loso_time = loso_time.sort_values("budget_value")
    best = loso_time.iloc[-1]
    first = loso_time.iloc[0]
    reduction = 1.0 - float(best["median_rmse"]) / max(float(first["median_rmse"]), 1e-12)

    metadata_loso = metadata[(metadata["split_kind"] == "loso-system") & (metadata["budget_kind"] == "time_days")]
    metadata_wide = metadata_loso.pivot_table(
        index=["budget_label", "budget_value"],
        columns="metadata_mode",
        values="median_rmse",
        aggfunc="first",
    ).reset_index().sort_values("budget_value")

    decoder_loso = decoder[(decoder["split_kind"] == "loso-system") & (decoder["budget_kind"] == "point")].copy()

    lines = [
        "# Release-Corpus Sparse-Observation Transfer Report",
        "",
        "This report summarizes the current probe outputs. It supports a transfer/observation-budget claim, not a foundation-model, mechanism-learning, or calibrated-uncertainty claim.",
        "",
        "## Headline",
        "",
        f"- Headline run: `{headline_run}`",
        f"- LOSO time-budget median RMSE drops from `{first['median_rmse']:.3f}` at `{first['budget_label']}` to `{best['median_rmse']:.3f}` at `{best['budget_label']}`.",
        f"- Relative median RMSE reduction over the measured time horizon: `{reduction:.1%}`.",
        "- Raw ensemble intervals remain under-calibrated; coverage is diagnostic only.",
        "",
        "## Time-Budget Curve",
        "",
        f"![Time budget]({plots['time_budget'].as_posix()})",
        "",
        markdown_table(loso_time[["budget_label", "median_n_obs", "median_rmse", "mean_rmse", "mean_crps", "mean_width90"]]),
        "",
        "## Baseline Deltas",
        "",
        "`median_delta_model_minus_baseline < 0` means the transfer model is better.",
        "Rows are sorted by the baseline's own median RMSE within each budget, so the closest competitors appear first.",
        "",
        markdown_table(top_baseline_delta(baseline_delta), max_rows=30),
        "",
        "## System-Level Transfer",
        "",
        f"![System time budget]({plots['system_time'].as_posix()})",
        "",
        "Hardest systems at the latest measured horizon:",
        "",
        markdown_table(hard, max_rows=12),
        "",
        "## Metadata Sensitivity",
        "",
        f"![Metadata sensitivity]({plots['metadata'].as_posix()})",
        "",
        markdown_table(metadata_wide),
        "",
        "Interpretation: no abnormal gain appears from `metadata-mode all`; the budget signal is mainly from early observations rather than system/source labels.",
        "",
        "## Decoder Ablation",
        "",
        f"![Decoder ablation]({plots['decoder'].as_posix()})",
        "",
        markdown_table(decoder_loso[["budget_label", "median_rmse_lnn", "median_rmse_mlp", "delta_median_rmse_lnn_minus_mlp", "delta_mean_crps_lnn_minus_mlp"]]),
        "",
        "Interpretation: LNN/CfC does not yet show a material advantage over the monotone MLP decoder.",
        "",
        "## Integrity Checks",
        "",
        markdown_table(pd.DataFrame(checks), max_rows=50),
        "",
        "## Lock",
        "",
        f"- git hash: `{git_hash()}`",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary, system_summary, time_summary, baseline_delta = load_headline(args.headline_run)
    metadata = load_metadata_compare(
        {
            "none": args.metadata_none,
            "safe": args.metadata_safe,
            "all": args.metadata_all,
        }
    )
    decoder = normalize_decoder_compare(read_csv(args.decoder_compare))

    checks = [
        check_no_numeric_nan("headline_summary_by_budget", summary),
        check_no_numeric_nan("headline_summary_by_system", system_summary),
        check_no_numeric_nan("headline_summary_by_time_budget", time_summary),
        check_no_numeric_nan("headline_baseline_delta", baseline_delta),
        check_no_numeric_nan("metadata_compare", metadata),
        check_no_numeric_nan("decoder_compare", decoder),
    ]
    if any(check["bad_numeric"] for check in checks):
        raise RuntimeError(f"Numeric NaN/inf detected: {checks}")

    summary.to_csv(args.out / "headline_summary_by_budget.csv", index=False)
    system_summary.to_csv(args.out / "headline_summary_by_system.csv", index=False)
    time_summary.to_csv(args.out / "headline_summary_by_time_budget.csv", index=False)
    baseline_delta.to_csv(args.out / "headline_baseline_delta_summary.csv", index=False)
    metadata.to_csv(args.out / "metadata_compare_long.csv", index=False)
    decoder.to_csv(args.out / "decoder_compare.csv", index=False)

    system_time = system_time_budget(system_summary)
    hard = hard_systems(system_time)
    system_time.to_csv(args.out / "system_time_budget_loso.csv", index=False)
    hard.to_csv(args.out / "hard_systems_loso.csv", index=False)

    plots = {
        "time_budget": make_time_budget_plot(time_summary, args.out),
        "system_time": make_system_time_plot(system_time, args.out),
        "metadata": make_metadata_plot(metadata, args.out),
        "decoder": make_decoder_plot(decoder, args.out),
    }
    write_report(
        args.out,
        args.headline_run,
        summary,
        system_time,
        hard,
        time_summary,
        baseline_delta,
        metadata,
        decoder,
        checks,
        plots,
    )
    lock = {
        "script": "scripts/82_release_transfer_report.py",
        "git_hash": git_hash(),
        "headline_run": str(args.headline_run),
        "metadata_runs": {
            "none": str(args.metadata_none),
            "safe": str(args.metadata_safe),
            "all": str(args.metadata_all),
        },
        "decoder_compare": str(args.decoder_compare),
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"[82] wrote {args.out}")


if __name__ == "__main__":
    main()
