"""
82 - Aggregate the current cross-mechanism release capability snapshot.

Purpose:
    Gather the key audited outputs from PLGA, liposome, and locked chitosan
    assets into a compact current-state snapshot. This is the evidence anchor
    for "how far unified drug-release intelligence has actually progressed".

Consumes:
    outputs/45_input_source_ablation/input_source_summary.csv
    outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv
    outputs/67_chitosan_prospective/lock_metadata.json
    outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv
    outputs/80_liposome_group_by_api_direct_et_q/metrics_summary.csv
    outputs/80_liposome_group_by_api_our_et_refined/timescale_summary.csv
    outputs/80_liposome_group_by_api_direct_et_q/timescale_summary.csv
    outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv
    outputs/80_liposome_group_by_method_direct_et_q/metrics_summary.csv
    outputs/80_liposome_group_by_method_our_et_refined/timescale_summary.csv
    outputs/80_liposome_group_by_method_direct_et_q/timescale_summary.csv

Produces:
    outputs/82_release_capability_snapshot/plga_signal_snapshot.csv
    outputs/82_release_capability_snapshot/plga_theta_gain_snapshot.csv
    outputs/82_release_capability_snapshot/plga_timescale_snapshot.csv
    outputs/82_release_capability_snapshot/plga_shape_snapshot.csv
    outputs/82_release_capability_snapshot/liposome_bridge_snapshot.csv
    outputs/82_release_capability_snapshot/liposome_shape_snapshot.csv
    outputs/82_release_capability_snapshot/prospective_status_snapshot.csv
    outputs/82_release_capability_snapshot/summary.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("outputs/82_release_capability_snapshot")
PLGA_SIGNAL = Path("outputs/45_input_source_ablation/input_source_summary.csv")
PLGA_GAIN = Path("outputs/46_direct_curve_rf_input_ablation/theta_vs_direct_summary.csv")
CHITOSAN_LOCK = Path("outputs/67_chitosan_prospective/lock_metadata.json")
PLGA_TIMESCALE_PATHS = {
    "cross321_group_by_drug_formulation_plus_early": {
        "mechanism_metrics": Path("outputs/80_cross321_group_by_drug_theta_rf_ztheta_leaf2/metrics_summary.csv"),
        "mechanism_timescale": Path("outputs/81_cross321_group_by_drug_theta_rf_ztheta_leaf2/timescale_summary_future_only.csv"),
        "direct_metrics": Path("outputs/80_cross321_group_by_drug_direct_et_q/metrics_summary.csv"),
        "direct_timescale": Path("outputs/81_cross321_group_by_drug_direct_et_q/timescale_summary_future_only.csv"),
        "mechanism_route": "RF_ztheta_leaf2",
        "direct_route": "ET_direct_Q",
    },
    "cross321_group_by_polymer_formulation_plus_early": {
        "mechanism_metrics": Path("outputs/80_cross321_group_by_polymer_theta_early_select/metrics_summary.csv"),
        "mechanism_timescale": Path("outputs/81_cross321_group_by_polymer_theta_early_select/timescale_summary_future_only.csv"),
        "direct_metrics": Path("outputs/80_cross321_group_by_polymer_direct_et_q/metrics_summary.csv"),
        "direct_timescale": Path("outputs/81_cross321_group_by_polymer_direct_et_q/timescale_summary_future_only.csv"),
        "mechanism_route": "early_select",
        "direct_route": "ET_direct_Q",
    },
    "internal181_group_by_drug_formulation_plus_early": {
        "mechanism_metrics": Path("outputs/80_internal181_group_by_drug_theta_early_select/metrics_summary.csv"),
        "mechanism_timescale": Path("outputs/81_internal181_group_by_drug_theta_early_select/timescale_summary_future_only.csv"),
        "direct_metrics": Path("outputs/80_internal181_group_by_drug_direct_et_q/metrics_summary.csv"),
        "direct_timescale": Path("outputs/81_internal181_group_by_drug_direct_et_q/timescale_summary_future_only.csv"),
        "mechanism_route": "early_select",
        "direct_route": "ET_direct_Q",
    },
    "internal181_group_by_polymer_formulation_plus_early": {
        "mechanism_metrics": Path("outputs/80_internal181_group_by_polymer_theta_early_select/metrics_summary.csv"),
        "mechanism_timescale": Path("outputs/81_internal181_group_by_polymer_theta_early_select/timescale_summary_future_only.csv"),
        "direct_metrics": Path("outputs/80_internal181_group_by_polymer_direct_rf_q_leaf2/metrics_summary.csv"),
        "direct_timescale": Path("outputs/81_internal181_group_by_polymer_direct_rf_q_leaf2/timescale_summary_future_only.csv"),
        "mechanism_route": "early_select",
        "direct_route": "RF_direct_Q_leaf2",
    },
}

LIPOSOME_PATHS = {
    "group_by_API": {
        "mechanism_metrics": Path("outputs/80_liposome_group_by_api_our_et_refined/metrics_summary.csv"),
        "mechanism_timescale": Path("outputs/80_liposome_group_by_api_our_et_refined/timescale_summary.csv"),
        "direct_metrics": Path("outputs/80_liposome_group_by_api_direct_et_q/metrics_summary.csv"),
        "direct_timescale": Path("outputs/80_liposome_group_by_api_direct_et_q/timescale_summary.csv"),
    },
    "group_by_release_method": {
        "mechanism_metrics": Path("outputs/80_liposome_group_by_method_our_et_refined/metrics_summary.csv"),
        "mechanism_timescale": Path("outputs/80_liposome_group_by_method_our_et_refined/timescale_summary.csv"),
        "direct_metrics": Path("outputs/80_liposome_group_by_method_direct_et_q/metrics_summary.csv"),
        "direct_timescale": Path("outputs/80_liposome_group_by_method_direct_et_q/timescale_summary.csv"),
    },
}


def _read_single_row_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one-row csv at {path}, got {len(df)} rows")
    return df.iloc[0]


def build_plga_signal_snapshot() -> pd.DataFrame:
    df = pd.read_csv(PLGA_SIGNAL)
    keep = df[
        df["scheme"].isin(["group_by_drug", "group_by_polymer", "random_5fold"])
    ].copy()
    keep = keep.rename(columns={"median": "best_median_curve_r2", "frac_above_0": "best_frac_curve_r2_ge0"})
    return keep[
        ["dataset", "scheme", "input_mode", "method", "best_median_curve_r2", "best_frac_curve_r2_ge0"]
    ].sort_values(["dataset", "scheme", "input_mode"]).reset_index(drop=True)


def build_plga_theta_gain_snapshot() -> pd.DataFrame:
    df = pd.read_csv(PLGA_GAIN)
    keep = df[
        df["scheme"].isin(["group_by_drug", "group_by_polymer", "random_5fold"])
    ].copy()
    keep = keep.rename(
        columns={
            "direct_best_median": "direct_best_median_curve_r2",
            "theta_best_median": "theta_best_median_curve_r2",
            "theta_minus_direct": "theta_route_minus_direct_route",
        }
    )
    return keep[
        [
            "dataset",
            "scheme",
            "input_mode",
            "direct_best_method",
            "direct_best_median_curve_r2",
            "theta_best_method",
            "theta_best_median_curve_r2",
            "theta_route_minus_direct_route",
        ]
    ].sort_values(["dataset", "scheme", "input_mode"]).reset_index(drop=True)


def build_liposome_bridge_snapshot() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheme, bundle in LIPOSOME_PATHS.items():
        mech_metrics = _read_single_row_csv(bundle["mechanism_metrics"])
        direct_metrics = _read_single_row_csv(bundle["direct_metrics"])
        mech_t = pd.read_csv(bundle["mechanism_timescale"])
        direct_t = pd.read_csv(bundle["direct_timescale"])

        t_mech = {float(row["threshold"]): row for _, row in mech_t.iterrows()}
        t_direct = {float(row["threshold"]): row for _, row in direct_t.iterrows()}

        rows.append(
            {
                "scheme": scheme,
                "mechanism_route": "our_ET_weibull_theta_early_refined",
                "direct_route": "direct_ET_Q_grid",
                "mechanism_pooled_r2": float(mech_metrics["pooled_r2"]),
                "direct_pooled_r2": float(direct_metrics["pooled_r2"]),
                "mechanism_pooled_rmse": float(mech_metrics["pooled_rmse"]),
                "direct_pooled_rmse": float(direct_metrics["pooled_rmse"]),
                "mechanism_median_curve_r2": float(mech_metrics["median_curve_r2"]),
                "direct_median_curve_r2": float(direct_metrics["median_curve_r2"]),
                "mechanism_frac_curve_r2_ge0": float(mech_metrics["frac_curve_r2_ge0"]),
                "direct_frac_curve_r2_ge0": float(direct_metrics["frac_curve_r2_ge0"]),
                "mechanism_minus_direct_pooled_r2": float(mech_metrics["pooled_r2"]) - float(direct_metrics["pooled_r2"]),
                "mechanism_minus_direct_median_curve_r2": float(mech_metrics["median_curve_r2"]) - float(direct_metrics["median_curve_r2"]),
                "t10_mae_mechanism_h": float(t_mech[0.1]["mae_t"]),
                "t10_mae_direct_h": float(t_direct[0.1]["mae_t"]),
                "t50_mae_mechanism_h": float(t_mech[0.5]["mae_t"]) if 0.5 in t_mech else np.nan,
                "t50_mae_direct_h": float(t_direct[0.5]["mae_t"]) if 0.5 in t_direct else np.nan,
                "t80_mae_mechanism_h": float(t_mech[0.8]["mae_t"]) if 0.8 in t_mech else np.nan,
                "t80_mae_direct_h": float(t_direct[0.8]["mae_t"]) if 0.8 in t_direct else np.nan,
                "t10_n_valid_mechanism": int(t_mech[0.1]["n_valid"]),
                "t10_n_valid_direct": int(t_direct[0.1]["n_valid"]),
                "t50_n_valid_mechanism": int(t_mech[0.5]["n_valid"]) if 0.5 in t_mech else 0,
                "t50_n_valid_direct": int(t_direct[0.5]["n_valid"]) if 0.5 in t_direct else 0,
                "t80_n_valid_mechanism": int(t_mech[0.8]["n_valid"]) if 0.8 in t_mech else 0,
                "t80_n_valid_direct": int(t_direct[0.8]["n_valid"]) if 0.8 in t_direct else 0,
            }
        )
    return pd.DataFrame(rows)


def build_plga_timescale_snapshot() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, bundle in PLGA_TIMESCALE_PATHS.items():
        mech_metrics = _read_single_row_csv(bundle["mechanism_metrics"])
        direct_metrics = _read_single_row_csv(bundle["direct_metrics"])
        mech_t = pd.read_csv(bundle["mechanism_timescale"])
        direct_t = pd.read_csv(bundle["direct_timescale"])
        t_mech = {float(row["threshold"]): row for _, row in mech_t.iterrows()}
        t_direct = {float(row["threshold"]): row for _, row in direct_t.iterrows()}
        rows.append(
            {
                "benchmark_cell": label,
                "mechanism_route": bundle["mechanism_route"],
                "direct_route": bundle["direct_route"],
                "mechanism_pooled_r2": float(mech_metrics["pooled_r2"]),
                "direct_pooled_r2": float(direct_metrics["pooled_r2"]),
                "mechanism_median_curve_r2": float(mech_metrics["median_curve_r2"]),
                "direct_median_curve_r2": float(direct_metrics["median_curve_r2"]),
                "future_t10_mae_mechanism_d": float(t_mech[0.1]["mae_t"]),
                "future_t10_mae_direct_d": float(t_direct[0.1]["mae_t"]),
                "future_t50_mae_mechanism_d": float(t_mech[0.5]["mae_t"]),
                "future_t50_mae_direct_d": float(t_direct[0.5]["mae_t"]),
                "future_t80_mae_mechanism_d": float(t_mech[0.8]["mae_t"]),
                "future_t80_mae_direct_d": float(t_direct[0.8]["mae_t"]),
                "future_t10_n_mechanism": int(t_mech[0.1]["n_valid"]),
                "future_t10_n_direct": int(t_direct[0.1]["n_valid"]),
                "future_t50_n_mechanism": int(t_mech[0.5]["n_valid"]),
                "future_t50_n_direct": int(t_direct[0.5]["n_valid"]),
                "future_t80_n_mechanism": int(t_mech[0.8]["n_valid"]),
                "future_t80_n_direct": int(t_direct[0.8]["n_valid"]),
            }
        )
    return pd.DataFrame(rows)


def _descriptor_rows(label: str, mech_path: Path, direct_path: Path, mechanism_route: str, direct_route: str) -> dict[str, object]:
    mech = pd.read_csv(mech_path)
    direct = pd.read_csv(direct_path)
    mech_map = {str(row["descriptor"]): row for _, row in mech.iterrows()}
    direct_map = {str(row["descriptor"]): row for _, row in direct.iterrows()}
    return {
        "benchmark_cell": label,
        "mechanism_route": mechanism_route,
        "direct_route": direct_route,
        "burst_mae_mechanism": float(mech_map["burst_fraction_at_early_window"]["mae"]),
        "burst_mae_direct": float(direct_map["burst_fraction_at_early_window"]["mae"]),
        "post_window_release_mae_mechanism": float(mech_map["post_window_release"]["mae"]),
        "post_window_release_mae_direct": float(direct_map["post_window_release"]["mae"]),
        "residual_tail_mae_mechanism": float(mech_map["residual_tail_at_tmax"]["mae"]),
        "residual_tail_mae_direct": float(direct_map["residual_tail_at_tmax"]["mae"]),
        "tail_auc_mae_mechanism": float(mech_map["tail_auc_after_early_window"]["mae"]),
        "tail_auc_mae_direct": float(direct_map["tail_auc_after_early_window"]["mae"]),
    }


def build_plga_shape_snapshot() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, bundle in PLGA_TIMESCALE_PATHS.items():
        rows.append(
            _descriptor_rows(
                label=label,
                mech_path=bundle["mechanism_timescale"].parent / "descriptor_summary.csv",
                direct_path=bundle["direct_timescale"].parent / "descriptor_summary.csv",
                mechanism_route=bundle["mechanism_route"],
                direct_route=bundle["direct_route"],
            )
        )
    return pd.DataFrame(rows)


def build_liposome_shape_snapshot() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    descriptor_dirs = {
        "group_by_API": {
            "mechanism": Path("outputs/81_liposome_group_by_api_our_et_refined/descriptor_summary.csv"),
            "direct": Path("outputs/81_liposome_group_by_api_direct_et_q/descriptor_summary.csv"),
        },
        "group_by_release_method": {
            "mechanism": Path("outputs/81_liposome_group_by_method_our_et_refined/descriptor_summary.csv"),
            "direct": Path("outputs/81_liposome_group_by_method_direct_et_q/descriptor_summary.csv"),
        },
    }
    for label, bundle in LIPOSOME_PATHS.items():
        rows.append(
            _descriptor_rows(
                label=label,
                mech_path=descriptor_dirs[label]["mechanism"],
                direct_path=descriptor_dirs[label]["direct"],
                mechanism_route="our_ET_weibull_theta_early_refined",
                direct_route="direct_ET_Q_grid",
            )
        )
    out = pd.DataFrame(rows)
    return out.rename(columns={"benchmark_cell": "scheme"})


def build_prospective_status_snapshot() -> pd.DataFrame:
    lock = json.loads(CHITOSAN_LOCK.read_text(encoding="utf-8"))
    row = {
        "dataset": "chitosan_batch1_locked",
        "status": "locked_prediction_pending_reveal",
        "n_curves": int(lock["n_curves"]),
        "n_formulations": int(lock["n_formulations"]),
        "n_drugs": int(lock["n_drugs"]),
        "time_grid_points": int(len(lock["t_grid_hours"])),
        "simulator": str(lock["simulator"]),
        "primary_endpoint": "aggregate cov90 >= 0.83",
    }
    return pd.DataFrame([row])


def write_summary(
    plga_signal: pd.DataFrame,
    plga_gain: pd.DataFrame,
    plga_timescale: pd.DataFrame,
    plga_shape: pd.DataFrame,
    liposome: pd.DataFrame,
    liposome_shape: pd.DataFrame,
    prospective: pd.DataFrame,
) -> None:
    plga_formulation = plga_signal[
        (plga_signal["input_mode"] == "formulation_only")
        & (plga_signal["scheme"].isin(["group_by_drug", "group_by_polymer"]))
    ]
    plga_early = plga_signal[
        (plga_signal["input_mode"] == "early_only")
        & (plga_signal["scheme"].isin(["group_by_drug", "group_by_polymer"]))
    ]
    plga_gain_ood = plga_gain[
        plga_gain["scheme"].isin(["group_by_drug", "group_by_polymer"])
    ]

    mech_wins_median = int((plga_timescale["mechanism_median_curve_r2"] > plga_timescale["direct_median_curve_r2"]).sum())
    mech_wins_pooled = int((plga_timescale["mechanism_pooled_r2"] > plga_timescale["direct_pooled_r2"]).sum())
    t10_mech_wins = int((plga_timescale["future_t10_mae_mechanism_d"] < plga_timescale["future_t10_mae_direct_d"]).sum())
    t50_mech_wins = int((plga_timescale["future_t50_mae_mechanism_d"] < plga_timescale["future_t50_mae_direct_d"]).sum())
    t80_mech_wins = int((plga_timescale["future_t80_mae_mechanism_d"] < plga_timescale["future_t80_mae_direct_d"]).sum())
    plga_burst_wins = int((plga_shape["burst_mae_mechanism"] < plga_shape["burst_mae_direct"]).sum())
    plga_post_wins = int((plga_shape["post_window_release_mae_mechanism"] < plga_shape["post_window_release_mae_direct"]).sum())
    plga_residual_wins = int((plga_shape["residual_tail_mae_mechanism"] < plga_shape["residual_tail_mae_direct"]).sum())
    plga_tail_auc_wins = int((plga_shape["tail_auc_mae_mechanism"] < plga_shape["tail_auc_mae_direct"]).sum())

    lines = [
        "=== 82 -- release capability snapshot ===",
        "",
        "PLGA audited core",
        f"  formulation_only OOD median best-R2 range : "
        f"{plga_formulation['best_median_curve_r2'].min():.3f} to {plga_formulation['best_median_curve_r2'].max():.3f}",
        f"  early_only OOD median best-R2 range       : "
        f"{plga_early['best_median_curve_r2'].min():.3f} to {plga_early['best_median_curve_r2'].max():.3f}",
        f"  theta-minus-direct OOD gain range         : "
        f"{plga_gain_ood['theta_route_minus_direct_route'].min():+.3f} to "
        f"{plga_gain_ood['theta_route_minus_direct_route'].max():+.3f}",
        "",
        "PLGA canonical OOD timescale cells",
    ]
    for row in plga_timescale.itertuples(index=False):
        lines.extend(
            [
                f"  {row.benchmark_cell}:",
                f"    pooled R2 mechanism vs direct : {row.mechanism_pooled_r2:.3f} vs {row.direct_pooled_r2:.3f}",
                f"    median curve R2 mech vs dir   : {row.mechanism_median_curve_r2:.3f} vs {row.direct_median_curve_r2:.3f}",
                f"    future t10 MAE (d) mech vs dir: {row.future_t10_mae_mechanism_d:.2f} vs {row.future_t10_mae_direct_d:.2f}",
                f"    future t50 MAE (d) mech vs dir: {row.future_t50_mae_mechanism_d:.2f} vs {row.future_t50_mae_direct_d:.2f}",
                f"    future t80 MAE (d) mech vs dir: {row.future_t80_mae_mechanism_d:.2f} vs {row.future_t80_mae_direct_d:.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "PLGA timescale panel tallies",
            f"  mechanism wins median curve R2 in   : {mech_wins_median}/{len(plga_timescale)} cells",
            f"  mechanism wins pooled R2 in         : {mech_wins_pooled}/{len(plga_timescale)} cells",
            f"  mechanism wins future t10 MAE in    : {t10_mech_wins}/{len(plga_timescale)} cells",
            f"  mechanism wins future t50 MAE in    : {t50_mech_wins}/{len(plga_timescale)} cells",
            f"  mechanism wins future t80 MAE in    : {t80_mech_wins}/{len(plga_timescale)} cells",
            f"  mechanism wins burst MAE in         : {plga_burst_wins}/{len(plga_shape)} cells",
            f"  mechanism wins post-window MAE in   : {plga_post_wins}/{len(plga_shape)} cells",
            f"  mechanism wins residual-tail MAE in : {plga_residual_wins}/{len(plga_shape)} cells",
            f"  mechanism wins tail-AUC MAE in      : {plga_tail_auc_wins}/{len(plga_shape)} cells",
            "",
        "Liposome bridge",
        ]
    )
    for row in liposome.itertuples(index=False):
        lines.extend(
            [
                f"  {row.scheme}:",
                f"    pooled R2 mechanism vs direct : {row.mechanism_pooled_r2:.3f} vs {row.direct_pooled_r2:.3f}",
                f"    median curve R2 mech vs dir   : {row.mechanism_median_curve_r2:.3f} vs {row.direct_median_curve_r2:.3f}",
                f"    t10 MAE (h) mech vs dir       : {row.t10_mae_mechanism_h:.2f} vs {row.t10_mae_direct_h:.2f}",
                f"    t50 MAE (h) mech vs dir       : {row.t50_mae_mechanism_h:.2f} vs {row.t50_mae_direct_h:.2f}",
                f"    t80 MAE (h) mech vs dir       : {row.t80_mae_mechanism_h:.2f} vs {row.t80_mae_direct_h:.2f}",
            ]
        )
    lines.extend(["", "Liposome release-shape descriptors"])
    for row in liposome_shape.itertuples(index=False):
        lines.extend(
            [
                f"  {row.scheme}:",
                f"    burst MAE mech vs dir            : {row.burst_mae_mechanism:.3f} vs {row.burst_mae_direct:.3f}",
                f"    post-window MAE mech vs dir      : {row.post_window_release_mae_mechanism:.3f} vs {row.post_window_release_mae_direct:.3f}",
                f"    residual-tail MAE mech vs dir    : {row.residual_tail_mae_mechanism:.3f} vs {row.residual_tail_mae_direct:.3f}",
                f"    tail-AUC MAE mech vs dir         : {row.tail_auc_mae_mechanism:.3f} vs {row.tail_auc_mae_direct:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "Prospective status",
            f"  chitosan lock status            : {prospective.iloc[0]['status']}",
            f"  curves / formulations / drugs   : "
            f"{int(prospective.iloc[0]['n_curves'])} / "
            f"{int(prospective.iloc[0]['n_formulations'])} / "
            f"{int(prospective.iloc[0]['n_drugs'])}",
            f"  prereg primary endpoint         : {prospective.iloc[0]['primary_endpoint']}",
            "",
            "Interpretation",
            "  1. PLGA evidence still says early release is the dominant OOD signal.",
            "  2. PLGA evidence still says the mechanism route beats matched direct-Q on current audited tasks.",
            "  3. Expanded PLGA timescale evidence shows curve-level and timescale-level route rankings diverge across multiple canonical OOD cells.",
            "  4. PLGA release-shape descriptors show the mechanism route is usually better on burst, post-window release, and residual-tail descriptors, but not consistently on tail-AUC.",
            "  5. Liposome now proves the shared release interface is not PLGA-only.",
            "  6. Liposome does not justify a blanket claim that mechanism routing wins every timescale or shape metric.",
            "  7. Chitosan remains the locked prospective stress test, not a completed cross-mechanism validation.",
        ]
    )
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plga_signal = build_plga_signal_snapshot()
    plga_gain = build_plga_theta_gain_snapshot()
    plga_timescale = build_plga_timescale_snapshot()
    plga_shape = build_plga_shape_snapshot()
    liposome = build_liposome_bridge_snapshot()
    liposome_shape = build_liposome_shape_snapshot()
    prospective = build_prospective_status_snapshot()

    plga_signal.to_csv(OUTDIR / "plga_signal_snapshot.csv", index=False)
    plga_gain.to_csv(OUTDIR / "plga_theta_gain_snapshot.csv", index=False)
    plga_timescale.to_csv(OUTDIR / "plga_timescale_snapshot.csv", index=False)
    plga_shape.to_csv(OUTDIR / "plga_shape_snapshot.csv", index=False)
    liposome.to_csv(OUTDIR / "liposome_bridge_snapshot.csv", index=False)
    liposome_shape.to_csv(OUTDIR / "liposome_shape_snapshot.csv", index=False)
    prospective.to_csv(OUTDIR / "prospective_status_snapshot.csv", index=False)
    write_summary(
        plga_signal=plga_signal,
        plga_gain=plga_gain,
        plga_timescale=plga_timescale,
        plga_shape=plga_shape,
        liposome=liposome,
        liposome_shape=liposome_shape,
        prospective=prospective,
    )


if __name__ == "__main__":
    main()
