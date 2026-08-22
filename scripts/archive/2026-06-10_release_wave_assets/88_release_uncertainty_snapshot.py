"""
88 - Aggregate the current uncertainty layer for unified drug-release evidence.

Purpose:
    Collect the current state of release uncertainty from four evidence tiers:

        1. raw FIB-CASP benchmark tables
        2. raw Ensemble-CASP benchmark tables
        3. split-conformal recalibration summaries
        4. conformal-attached posterior-family artifacts

    The result is a compact evidence anchor for the question:

        how far has shared uncertainty / calibrated-family unification actually
        progressed across release mechanisms?

Produces:
    outputs/88_release_uncertainty_snapshot/raw_uq_benchmark_snapshot.csv
    outputs/88_release_uncertainty_snapshot/calibrated_family_snapshot.csv
    outputs/88_release_uncertainty_snapshot/calibrated_family_path_inventory.csv
    outputs/88_release_uncertainty_snapshot/summary.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTDIR = Path("outputs/88_release_uncertainty_snapshot")
FIB_AGG = Path("outputs/62_fib_casp_benchmark/aggregate_table.csv")
ENS_AGG = Path("outputs/66_ensemble_casp/aggregate_table.csv")
CONFORMAL_SUMMARY = Path("outputs/76_casp_conformal_recalibration/summary.csv")
CONFORMAL_FAMILY_ROOT = Path("outputs/87_release_conformal_family_export")


def discover_calibrated_family_summary_paths() -> list[Path]:
    paths = [
        p
        for p in CONFORMAL_FAMILY_ROOT.rglob("summary.csv")
        if len(p.relative_to(CONFORMAL_FAMILY_ROOT).parts) == 3
    ]
    return sorted(paths)


def build_raw_uq_benchmark_snapshot() -> pd.DataFrame:
    fib = pd.read_csv(FIB_AGG).rename(
        columns={
            "n": "n_curves",
            "cov90_mean": "cov90_mean",
            "cov90_median": "cov90_median",
            "pi_width": "pi_width_median",
        }
    )
    fib = fib.assign(
        evidence_tier="raw_benchmark",
        uncertainty_route="fib_casp_raw",
        r2_median=fib["r2_median"],
    )[
        [
            "evidence_tier",
            "uncertainty_route",
            "dataset",
            "scheme",
            "n_curves",
            "r2_median",
            "cov90_mean",
            "cov90_median",
            "pi_width_median",
        ]
    ]

    ens = pd.read_csv(ENS_AGG).rename(
        columns={
            "n_curves": "n_curves",
            "cov90_mean": "cov90_mean",
            "cov90_median": "cov90_median",
            "pi_width_90_median": "pi_width_median",
        }
    )
    ens = ens.assign(
        evidence_tier="raw_benchmark",
        uncertainty_route="ensemble_casp_raw",
        r2_median=ens["r2_median"],
    )[
        [
            "evidence_tier",
            "uncertainty_route",
            "dataset",
            "scheme",
            "n_curves",
            "r2_median",
            "cov90_mean",
            "cov90_median",
            "pi_width_median",
        ]
    ]

    conformal = pd.read_csv(CONFORMAL_SUMMARY)
    rows: list[dict[str, object]] = []
    route_specs = [
        ("fib_split_conformal_raw", "cov90_raw_mean", "width90_raw_median"),
        ("fib_split_conformal_global", "cov90_global_mean", "width90_global_median"),
        ("fib_split_conformal_local", "cov90_local_mean", "width90_local_median"),
    ]
    for row in conformal.itertuples(index=False):
        for route_name, cov_col, width_col in route_specs:
            rows.append(
                {
                    "evidence_tier": "recalibrated_summary",
                    "uncertainty_route": route_name,
                    "dataset": row.dataset,
                    "scheme": row.scheme,
                    "n_curves": int(row.n),
                    "r2_median": float(row.r2_median),
                    "cov90_mean": float(getattr(row, cov_col)),
                    "cov90_median": None,
                    "pi_width_median": float(getattr(row, width_col)),
                }
            )
    conf = pd.DataFrame(rows)
    conf["cov90_median"] = float("nan")
    return pd.concat([fib, ens, conf], ignore_index=True)


def build_calibrated_family_snapshot() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in discover_calibrated_family_summary_paths():
        dataset = path.parent.parent.name
        scheme = path.parent.name
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "scheme": scheme,
                "n_families": int(len(df)),
                "mechanism_id_mode": df["mechanism_id"].mode().iat[0],
                "latent_dim_median": float(df["latent_dim"].median()),
                "rank_median": float(df["rank"].median()),
                "global_scale_median": float(df["global_scale"].median()),
                "global_scale_max": float(df["global_scale"].max()),
                "cov90_raw_mean": float(df["cov90_raw"].mean()),
                "cov90_global_mean": float(df["cov90_global"].mean()),
                "cov90_local_mean": float(df["cov90_local"].mean()),
                "width90_raw_median": float(df["width90_raw"].median()),
                "width90_global_median": float(df["width90_global"].median()),
                "width90_local_median": float(df["width90_local"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_calibrated_family_path_inventory() -> pd.DataFrame:
    rows = []
    for path in discover_calibrated_family_summary_paths():
        rel = path.relative_to(CONFORMAL_FAMILY_ROOT)
        rows.append(
            {
                "dataset": rel.parts[0],
                "scheme": rel.parts[1],
                "summary_csv": str(path.as_posix()),
            }
        )
    return pd.DataFrame(rows)


def write_summary(raw_snapshot: pd.DataFrame, family_snapshot: pd.DataFrame) -> None:
    fib_raw = raw_snapshot[raw_snapshot["uncertainty_route"] == "fib_casp_raw"].copy()
    ens_raw = raw_snapshot[raw_snapshot["uncertainty_route"] == "ensemble_casp_raw"].copy()
    conformal_global = raw_snapshot[raw_snapshot["uncertainty_route"] == "fib_split_conformal_global"].copy()
    conformal_local = raw_snapshot[raw_snapshot["uncertainty_route"] == "fib_split_conformal_local"].copy()
    conformal_raw = raw_snapshot[raw_snapshot["uncertainty_route"] == "fib_split_conformal_raw"].copy()

    def _cell(df: pd.DataFrame, dataset: str, scheme: str) -> pd.Series:
        out = df[(df["dataset"] == dataset) & (df["scheme"] == scheme)]
        if len(out) != 1:
            raise ValueError(f"Expected one row for {dataset}/{scheme}, got {len(out)}")
        return out.iloc[0]

    cross_fib = _cell(fib_raw, "cross321", "group_by_drug")
    cross_conf_raw = _cell(conformal_raw, "cross321", "group_by_drug")
    cross_global = _cell(conformal_global, "cross321", "group_by_drug")
    cross_local = _cell(conformal_local, "cross321", "group_by_drug")
    lipo_fib = _cell(fib_raw, "liposome", "group_by_drug")
    lipo_conf_raw = _cell(conformal_raw, "liposome", "group_by_drug")
    lipo_global = _cell(conformal_global, "liposome", "group_by_drug")
    lipo_local = _cell(conformal_local, "liposome", "group_by_drug")

    lines = [
        "=== 88 -- release uncertainty snapshot ===",
        "",
        "Raw benchmark uncertainty layer",
        f"  FIB raw cov90 range across all cells      : {fib_raw['cov90_mean'].min():.3f} to {fib_raw['cov90_mean'].max():.3f}",
        f"  Ensemble raw cov90 range across all cells : {ens_raw['cov90_mean'].min():.3f} to {ens_raw['cov90_mean'].max():.3f}",
        f"  FIB raw width median range                : {fib_raw['pi_width_median'].min():.3f} to {fib_raw['pi_width_median'].max():.3f}",
        f"  Ensemble raw width median range           : {ens_raw['pi_width_median'].min():.3f} to {ens_raw['pi_width_median'].max():.3f}",
        f"  discovered calibrated-family cells        : {len(family_snapshot)}",
        "",
        "Representative split-conformal repair cells",
        "  cross321 / group_by_drug:",
        f"    fib raw -> global cov90 : {cross_global['cov90_mean'] - cross_fib['cov90_mean']:+.3f}",
        f"    conf raw -> global cov90: {cross_global['cov90_mean'] - cross_conf_raw['cov90_mean']:+.3f}",
        f"    global/local cov90 mean : {cross_global['cov90_mean']:.3f} / {cross_local['cov90_mean']:.3f}",
        f"    global/local width med  : {cross_global['pi_width_median']:.3f} / {cross_local['pi_width_median']:.3f}",
        "  liposome / group_by_drug:",
        f"    fib raw -> global cov90 : {lipo_global['cov90_mean'] - lipo_fib['cov90_mean']:+.3f}",
        f"    conf raw -> global cov90: {lipo_global['cov90_mean'] - lipo_conf_raw['cov90_mean']:+.3f}",
        f"    global/local cov90 mean : {lipo_global['cov90_mean']:.3f} / {lipo_local['cov90_mean']:.3f}",
        f"    global/local width med  : {lipo_global['pi_width_median']:.3f} / {lipo_local['pi_width_median']:.3f}",
        "",
        f"Calibrated family artifacts ({len(family_snapshot)} cells)",
    ]

    for row in family_snapshot.itertuples(index=False):
        lines.extend(
            [
                f"  {row.dataset} / {row.scheme}:",
                f"    families                 : {int(row.n_families)}",
                f"    mechanism / latent / rank: {row.mechanism_id_mode} / {row.latent_dim_median:.0f} / {row.rank_median:.0f}",
                f"    global scale median/max  : {row.global_scale_median:.3f} / {row.global_scale_max:.3f}",
                f"    cov90 raw/global/local   : {row.cov90_raw_mean:.3f} / {row.cov90_global_mean:.3f} / {row.cov90_local_mean:.3f}",
                f"    width raw/global/local   : {row.width90_raw_median:.3f} / {row.width90_global_median:.3f} / {row.width90_local_median:.3f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "  1. Raw FIB uncertainty is useful but under-nominal on the harder OOD release cells.",
            "  2. Raw ensemble uncertainty is generally weaker than FIB on the current audited release cells.",
            "  3. Split-conformal repair can move selected FIB cells closer to nominal coverage, but with real width inflation.",
            "  4. The project now has a real calibrated-family panel, not only raw family exports and standalone coverage tables.",
            "  5. Cross-mechanism uncertainty is now partially unified at the artifact level, not yet benchmark-wide or prospectively validated.",
        ]
    )
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw_snapshot = build_raw_uq_benchmark_snapshot()
    family_snapshot = build_calibrated_family_snapshot()
    path_inventory = build_calibrated_family_path_inventory()
    raw_snapshot.to_csv(OUTDIR / "raw_uq_benchmark_snapshot.csv", index=False)
    family_snapshot.to_csv(OUTDIR / "calibrated_family_snapshot.csv", index=False)
    path_inventory.to_csv(OUTDIR / "calibrated_family_path_inventory.csv", index=False)
    write_summary(raw_snapshot, family_snapshot)


if __name__ == "__main__":
    main()
