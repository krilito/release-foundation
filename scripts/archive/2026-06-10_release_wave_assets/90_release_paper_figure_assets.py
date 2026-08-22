"""
90 - Assemble figure-ready panel assets for the current drug-release paper.

Purpose:
    Convert the current audited release outputs into small figure-specific
    CSV assets so plotting no longer needs to read heterogeneous benchmark
    outputs directly.

Consumes:
    outputs/45_input_source_ablation/input_source_summary.csv
    outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv
    outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv
    outputs/82_release_capability_snapshot/*.csv
    outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv
    outputs/67_chitosan_prospective/lock_metadata.json
    docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md

Produces:
    outputs/90_release_paper_figure_assets/figure2/*.csv
    outputs/90_release_paper_figure_assets/figure3/*.csv
    outputs/90_release_paper_figure_assets/figure4/*.csv
    outputs/90_release_paper_figure_assets/figure5/*.csv
    outputs/90_release_paper_figure_assets/summary.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


OUTROOT = Path("outputs/90_release_paper_figure_assets")

INPUT_SOURCE = Path("outputs/45_input_source_ablation/input_source_summary.csv")
THETA_DIRECT = Path("outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv")
MIDDLE_GAIN = Path("outputs/59_middle_layer_gain_audit/aggregate_middle_layer_gain.csv")

SNAP82 = Path("outputs/82_release_capability_snapshot")
PLGA_TIMESCALE = SNAP82 / "plga_timescale_snapshot.csv"
PLGA_SHAPE = SNAP82 / "plga_shape_snapshot.csv"
LIPO_BRIDGE = SNAP82 / "liposome_bridge_snapshot.csv"
LIPO_SHAPE = SNAP82 / "liposome_shape_snapshot.csv"

SNAP88 = Path("outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv")
CHITOSAN_LOCK = Path("outputs/67_chitosan_prospective/lock_metadata.json")
CHITOSAN_PREREG = Path("docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_input_mode(value: str) -> str:
    return {
        "formulation_only": "Formulation only",
        "early_only": "Early only",
        "formulation_plus_early": "Formulation + early",
    }.get(value, value)


def normalize_scheme(value: str) -> str:
    return {
        "group_by_drug": "Group by drug",
        "group_by_polymer": "Group by polymer",
        "random_5fold": "Random 5-fold",
        "group_by_API": "Group by API",
        "group_by_release_method": "Group by release method",
    }.get(value, value)


def parse_primary_endpoint(prereg_text: str) -> str:
    match = re.search(
        r"(aggregate\s+)?cov90\s*(>=|≥)\s*0\.83",
        prereg_text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(0)
    raise ValueError("Could not recover preregistered primary endpoint from prereg text.")


def build_figure2(figdir: Path) -> list[Path]:
    outputs: list[Path] = []
    ensure_dir(figdir)

    inp = pd.read_csv(INPUT_SOURCE).copy()
    inp["input_mode_label"] = inp["input_mode"].map(normalize_input_mode)
    inp["scheme_label"] = inp["scheme"].map(normalize_scheme)
    inp = inp[
        [
            "dataset",
            "scheme",
            "scheme_label",
            "input_mode",
            "input_mode_label",
            "method",
            "median",
            "frac_above_0",
        ]
    ].sort_values(["dataset", "scheme", "input_mode"])
    out = figdir / "panelA_input_source_ablation.csv"
    inp.to_csv(out, index=False)
    outputs.append(out)

    theta = pd.read_csv(THETA_DIRECT).copy()
    theta["input_mode_label"] = theta["input_mode"].map(normalize_input_mode)
    theta["scheme_label"] = theta["scheme"].map(normalize_scheme)
    theta = theta[
        [
            "dataset",
            "scheme",
            "scheme_label",
            "input_mode",
            "input_mode_label",
            "direct_best_method",
            "direct_best_median",
            "theta_best_method",
            "theta_best_median",
            "theta_minus_direct",
        ]
    ].sort_values(["dataset", "scheme", "input_mode"])
    out = figdir / "panelB_theta_vs_direct.csv"
    theta.to_csv(out, index=False)
    outputs.append(out)

    middle = pd.read_csv(MIDDLE_GAIN).copy()
    middle["group_label"] = middle["group"].map(
        {
            "best_route/all": "Best route / all",
            "best_route/formulation_plus_early": "Best route / formulation + early",
            "best_route/OOD_formulation_plus_early": "Best route / OOD formulation + early",
            "family_matched/all": "Family matched / all",
            "family_matched/formulation_plus_early": "Family matched / formulation + early",
            "family_matched/OOD_formulation_plus_early": "Family matched / OOD formulation + early",
            "family/ET": "Family ET",
            "family/RF": "Family RF",
        }
    ).fillna(middle["group"])
    middle = middle[
        [
            "group",
            "group_label",
            "n_cells",
            "mean_gain",
            "median_gain",
            "min_gain",
            "max_gain",
            "frac_positive",
        ]
    ]
    out = figdir / "panelC_middle_layer_gain.csv"
    middle.to_csv(out, index=False)
    outputs.append(out)
    return outputs


def build_figure3(figdir: Path) -> list[Path]:
    outputs: list[Path] = []
    ensure_dir(figdir)

    time = pd.read_csv(PLGA_TIMESCALE).copy()
    time["scheme_label"] = time["benchmark_cell"].str.split("_").str[1:4].str.join("_").map(normalize_scheme)
    tallies = pd.DataFrame(
        [
            {"metric": "Median curve R2", "mechanism_wins": int((time["mechanism_median_curve_r2"] > time["direct_median_curve_r2"]).sum()), "n_cells": int(len(time))},
            {"metric": "Pooled R2", "mechanism_wins": int((time["mechanism_pooled_r2"] > time["direct_pooled_r2"]).sum()), "n_cells": int(len(time))},
            {"metric": "Future t10 MAE", "mechanism_wins": int((time["future_t10_mae_mechanism_d"] < time["future_t10_mae_direct_d"]).sum()), "n_cells": int(len(time))},
            {"metric": "Future t50 MAE", "mechanism_wins": int((time["future_t50_mae_mechanism_d"] < time["future_t50_mae_direct_d"]).sum()), "n_cells": int(len(time))},
            {"metric": "Future t80 MAE", "mechanism_wins": int((time["future_t80_mae_mechanism_d"] < time["future_t80_mae_direct_d"]).sum()), "n_cells": int(len(time))},
        ]
    )
    out = figdir / "panelA_plga_timescale_tallies.csv"
    tallies.to_csv(out, index=False)
    outputs.append(out)

    shape = pd.read_csv(PLGA_SHAPE).copy()
    shape_tallies = pd.DataFrame(
        [
            {"metric": "Burst MAE", "mechanism_wins": int((shape["burst_mae_mechanism"] < shape["burst_mae_direct"]).sum()), "n_cells": int(len(shape))},
            {"metric": "Post-window MAE", "mechanism_wins": int((shape["post_window_release_mae_mechanism"] < shape["post_window_release_mae_direct"]).sum()), "n_cells": int(len(shape))},
            {"metric": "Residual-tail MAE", "mechanism_wins": int((shape["residual_tail_mae_mechanism"] < shape["residual_tail_mae_direct"]).sum()), "n_cells": int(len(shape))},
            {"metric": "Tail-AUC MAE", "mechanism_wins": int((shape["tail_auc_mae_mechanism"] < shape["tail_auc_mae_direct"]).sum()), "n_cells": int(len(shape))},
        ]
    )
    out = figdir / "panelB_plga_shape_tallies.csv"
    shape_tallies.to_csv(out, index=False)
    outputs.append(out)

    rep = time.copy()
    rep["curve_win"] = rep["mechanism_median_curve_r2"] > rep["direct_median_curve_r2"]
    rep["duration_loss_count"] = (
        (rep["future_t10_mae_mechanism_d"] > rep["future_t10_mae_direct_d"]).astype(int)
        + (rep["future_t50_mae_mechanism_d"] > rep["future_t50_mae_direct_d"]).astype(int)
        + (rep["future_t80_mae_mechanism_d"] > rep["future_t80_mae_direct_d"]).astype(int)
    )
    rep["duration_penalty"] = (
        (rep["future_t10_mae_mechanism_d"] - rep["future_t10_mae_direct_d"]).clip(lower=0)
        + (rep["future_t50_mae_mechanism_d"] - rep["future_t50_mae_direct_d"]).clip(lower=0)
        + (rep["future_t80_mae_mechanism_d"] - rep["future_t80_mae_direct_d"]).clip(lower=0)
    )
    rep = rep.sort_values(["curve_win", "duration_loss_count", "duration_penalty"], ascending=[False, False, False])
    rep = rep.head(1).copy()
    out = figdir / "panelC_representative_divergence.csv"
    rep.to_csv(out, index=False)
    outputs.append(out)
    return outputs


def build_figure4(figdir: Path) -> list[Path]:
    outputs: list[Path] = []
    ensure_dir(figdir)

    bridge = pd.read_csv(LIPO_BRIDGE).copy()
    bridge["scheme_label"] = bridge["scheme"].map(normalize_scheme)
    out = figdir / "panelAB_liposome_bridge_metrics.csv"
    bridge.to_csv(out, index=False)
    outputs.append(out)

    shape = pd.read_csv(LIPO_SHAPE).copy()
    shape["scheme_label"] = shape["scheme"].map(normalize_scheme)
    out = figdir / "panelC_liposome_shape_metrics.csv"
    shape.to_csv(out, index=False)
    outputs.append(out)
    return outputs


def build_figure5(figdir: Path) -> list[Path]:
    outputs: list[Path] = []
    ensure_dir(figdir)

    cal = pd.read_csv(SNAP88).copy()
    cal["scheme_label"] = cal["scheme"].map(normalize_scheme)
    out = figdir / "panelA_calibrated_family_panel.csv"
    cal.to_csv(out, index=False)
    outputs.append(out)

    examples = cal[cal["scheme"] == "group_by_drug"].copy()
    examples = examples[examples["dataset"].isin(["cross321", "liposome"])].copy()
    out = figdir / "panelB_representative_conformal_examples.csv"
    examples.to_csv(out, index=False)
    outputs.append(out)

    lock = json.loads(CHITOSAN_LOCK.read_text(encoding="utf-8"))
    prereg_text = CHITOSAN_PREREG.read_text(encoding="utf-8")
    lock_row = pd.DataFrame(
        [
            {
                "status": "locked_prediction_pending_reveal",
                "git_tag": lock.get("git_tag"),
                "prediction_timestamp_utc": lock.get("timestamp_utc"),
                "curve_count": 12,
                "formulation_count": 6,
                "drug_count": 2,
                "primary_endpoint": parse_primary_endpoint(prereg_text),
            }
        ]
    )
    out = figdir / "panelC_chitosan_lock_summary.csv"
    lock_row.to_csv(out, index=False)
    outputs.append(out)
    return outputs


def write_summary(file_map: dict[str, list[Path]]) -> None:
    lines = [
        "=== 90 -- release paper figure assets ===",
        "",
        "Assembled figure-ready panel tables for the current release paper.",
        "",
    ]
    for figure_name, paths in file_map.items():
        lines.append(f"{figure_name}:")
        for path in paths:
            lines.append(f"  - {path.as_posix()}")
        lines.append("")
    lines.extend(
        [
            "Interpretation",
            "  1. Figure 2 is now data-assembled from the audited PLGA core.",
            "  2. Figure 3 now has explicit tally tables plus a representative divergence cell.",
            "  3. Figure 4 now has clean bridge metrics for both liposome split families.",
            "  4. Figure 5 now has a clean calibrated-family panel and a chitosan lock summary table.",
            "  5. Plotting no longer needs to read heterogeneous benchmark outputs directly.",
        ]
    )
    (OUTROOT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTROOT)
    file_map = {
        "figure2": build_figure2(OUTROOT / "figure2"),
        "figure3": build_figure3(OUTROOT / "figure3"),
        "figure4": build_figure4(OUTROOT / "figure4"),
        "figure5": build_figure5(OUTROOT / "figure5"),
    }
    write_summary(file_map)


if __name__ == "__main__":
    main()
