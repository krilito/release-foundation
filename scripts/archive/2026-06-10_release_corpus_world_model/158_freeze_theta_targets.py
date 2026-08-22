from __future__ import annotations

"""
158_freeze_theta_targets.py

Consume:
- outputs/148_release_main_cumulative_v1/curves_long.csv
- outputs/148_release_main_cumulative_v1/formulations.csv

Produce:
- outputs/158_freeze_theta_targets/theta_table_v1.parquet
- outputs/158_freeze_theta_targets/all_family_fits.csv
- outputs/158_freeze_theta_targets/family_summary.csv
- outputs/158_freeze_theta_targets/best_family_summary.csv
- outputs/158_freeze_theta_targets/manifest.json
- outputs/158_freeze_theta_targets/summary.md

Expected runtime:
- ~3-8 min on the current 751-curve main pool
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FamilySpec:
    name: str
    n_params: int
    bounds_lo: tuple[float, ...]
    bounds_hi: tuple[float, ...]


FAMILY_SPECS = {
    "weibull": FamilySpec("weibull", 3, (0.2, 1e-4, 0.1), (2.0, 1e3, 8.0)),
    "biexponential": FamilySpec("biexponential", 4, (0.2, 0.0, 1e-5, 1e-5), (2.0, 1.0, 50.0, 10.0)),
    "hill": FamilySpec("hill", 3, (0.2, 1e-4, 0.1), (2.0, 1e3, 8.0)),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=repo_root / "outputs" / "148_release_main_cumulative_v1",
        help="main cumulative pool directory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "outputs" / "158_freeze_theta_targets",
        help="output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="global random seed",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def map_system_id(source_dataset: str, polymer_family: str) -> str:
    if source_dataset in {"internal181", "cross321"}:
        return "PLGA"
    return polymer_family


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def weibull_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, tau, beta = p
    t_pos = np.clip(t, 0.0, None)
    return qmax * (1.0 - np.exp(-np.power(t_pos / max(tau, 1e-8), beta)))


def biexponential_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, w_fast, k_fast, k_slow = p
    t_pos = np.clip(t, 0.0, None)
    fast = 1.0 - np.exp(-k_fast * t_pos)
    slow = 1.0 - np.exp(-k_slow * t_pos)
    return qmax * (w_fast * fast + (1.0 - w_fast) * slow)


def hill_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, t50, alpha = p
    t_pos = np.clip(t, 0.0, None)
    num = np.power(t_pos, alpha)
    den = np.power(max(t50, 1e-8), alpha) + num + 1e-12
    return qmax * (num / den)


FAMILY_FUNCS = {
    "weibull": weibull_curve,
    "biexponential": biexponential_curve,
    "hill": hill_curve,
}


def initial_guesses(family: str, t: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    y_max = float(np.clip(np.nanmax(y), 0.3, 1.5))
    t_end = float(max(np.nanmax(t), 1e-3))
    if family == "weibull":
        return [
            np.array([y_max, max(t_end / 3.0, 1e-3), 1.0]),
            np.array([min(1.8, y_max + 0.1), max(t_end / 2.0, 1e-3), 0.7]),
            np.array([min(1.8, y_max + 0.2), max(t_end / 6.0, 1e-3), 1.5]),
        ]
    if family == "biexponential":
        return [
            np.array([y_max, 0.6, 1.0 / max(t_end / 5.0, 1e-3), 1.0 / max(t_end, 1e-3)]),
            np.array([min(1.8, y_max + 0.1), 0.3, 1.0 / max(t_end / 3.0, 1e-3), 1.0 / max(t_end * 2.0, 1e-3)]),
            np.array([min(1.8, y_max + 0.2), 0.8, 1.0 / max(t_end / 8.0, 1e-3), 1.0 / max(t_end / 2.0, 1e-3)]),
        ]
    if family == "hill":
        return [
            np.array([y_max, max(t_end / 3.0, 1e-3), 1.2]),
            np.array([min(1.8, y_max + 0.1), max(t_end / 2.0, 1e-3), 0.9]),
            np.array([min(1.8, y_max + 0.2), max(t_end / 5.0, 1e-3), 1.8]),
        ]
    raise KeyError(family)


def fit_family(family: str, t: np.ndarray, y: np.ndarray) -> dict[str, object]:
    spec = FAMILY_SPECS[family]
    fn = FAMILY_FUNCS[family]
    best: dict[str, object] | None = None

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = fn(t, params)
        return (pred - y) / np.sqrt(max(len(y), 1))

    for x0 in initial_guesses(family, t, y):
        try:
            result = least_squares(
                residuals,
                x0=np.clip(x0, spec.bounds_lo, spec.bounds_hi),
                bounds=(np.asarray(spec.bounds_lo), np.asarray(spec.bounds_hi)),
                max_nfev=4000,
            )
        except Exception:
            continue
        pred = fn(t, result.x)
        row = {
            "family": family,
            "params": result.x.tolist(),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))),
            "r2": safe_r2(y, pred),
            "pred": pred,
        }
        if best is None or float(row["rmse"]) < float(best["rmse"]):
            best = row

    if best is None:
        fallback = np.full_like(y, np.mean(y))
        return {
            "family": family,
            "params": [np.nan] * spec.n_params,
            "rmse": float(np.sqrt(np.mean((fallback - y) ** 2))),
            "mae": float(np.mean(np.abs(fallback - y))),
            "r2": safe_r2(y, fallback),
            "pred": fallback,
        }
    return best


def load_pool(pool_dir: Path) -> dict[str, pd.DataFrame]:
    curves = pd.read_csv(pool_dir / "curves_long.csv")
    formulations = pd.read_csv(pool_dir / "formulations.csv")
    formulations["system_id"] = [
        map_system_id(str(ds), str(fam))
        for ds, fam in zip(formulations["source_dataset"], formulations["polymer_family"])
    ]
    curve_index = formulations[
        ["unified_curve_id", "source_dataset", "system_id", "polymer_family", "payload_name", "source_group"]
    ].copy()
    curves = curves.merge(curve_index, on=["unified_curve_id", "source_dataset"], how="left")
    curves["time_days"] = pd.to_numeric(curves["time_days"], errors="coerce")
    curves["release_fraction"] = pd.to_numeric(curves["release_fraction"], errors="coerce")
    curves = curves.dropna(subset=["unified_curve_id", "time_days", "release_fraction", "system_id"]).copy()
    curves = curves.sort_values(["unified_curve_id", "time_days"], kind="stable").reset_index(drop=True)
    return {"curves": curves, "formulations": formulations}


def freeze_theta_table(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, object]] = []
    theta_rows: list[dict[str, object]] = []

    for curve_id, sub in curves.groupby("unified_curve_id", sort=True):
        sub = sub.sort_values("time_days").reset_index(drop=True)
        t = sub["time_days"].to_numpy(dtype=float)
        y = sub["release_fraction"].to_numpy(dtype=float)
        meta = sub.iloc[0]
        best: dict[str, object] | None = None

        for family in FAMILY_SPECS:
            fit = fit_family(family, t, y)
            params = list(fit["params"])
            fit_rows.append(
                {
                    "unified_curve_id": curve_id,
                    "system_id": meta["system_id"],
                    "source_dataset": meta["source_dataset"],
                    "polymer_family": meta["polymer_family"],
                    "payload_name": meta.get("payload_name"),
                    "source_group": meta.get("source_group"),
                    "family": family,
                    "n_points": int(len(sub)),
                    "duration_days": float(t[-1] - t[0]),
                    "release_final": float(y[-1]),
                    "release_max": float(np.max(y)),
                    "rmse": float(fit["rmse"]),
                    "mae": float(fit["mae"]),
                    "r2": float(fit["r2"]),
                    "param_1": float(params[0]) if len(params) > 0 else np.nan,
                    "param_2": float(params[1]) if len(params) > 1 else np.nan,
                    "param_3": float(params[2]) if len(params) > 2 else np.nan,
                    "param_4": float(params[3]) if len(params) > 3 else np.nan,
                    "params_json": json.dumps(params),
                }
            )
            if best is None or float(fit["rmse"]) < float(best["rmse"]):
                best = {
                    "family": family,
                    "params": params,
                    "rmse": float(fit["rmse"]),
                    "mae": float(fit["mae"]),
                    "r2": float(fit["r2"]),
                }

        assert best is not None
        theta_rows.append(
            {
                "unified_curve_id": curve_id,
                "system_id": meta["system_id"],
                "source_dataset": meta["source_dataset"],
                "polymer_family": meta["polymer_family"],
                "payload_name": meta.get("payload_name"),
                "source_group": meta.get("source_group"),
                "theta_family": best["family"],
                "theta_n_params": int(FAMILY_SPECS[str(best["family"])].n_params),
                "theta_fit_r2": float(best["r2"]),
                "theta_fit_rmse": float(best["rmse"]),
                "theta_fit_mae": float(best["mae"]),
                "theta_param_1": float(best["params"][0]) if len(best["params"]) > 0 else np.nan,
                "theta_param_2": float(best["params"][1]) if len(best["params"]) > 1 else np.nan,
                "theta_param_3": float(best["params"][2]) if len(best["params"]) > 2 else np.nan,
                "theta_param_4": float(best["params"][3]) if len(best["params"]) > 3 else np.nan,
                "theta_params_json": json.dumps(best["params"]),
                "n_points": int(len(sub)),
                "duration_days": float(t[-1] - t[0]),
                "release_final": float(y[-1]),
                "release_max": float(np.max(y)),
            }
        )

    return pd.DataFrame(theta_rows), pd.DataFrame(fit_rows)


def summarize_family_metrics(all_fits: pd.DataFrame) -> pd.DataFrame:
    return (
        all_fits.groupby(["system_id", "family"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "count"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            p25_r2=("r2", lambda x: float(np.nanpercentile(x, 25))),
            p75_r2=("r2", lambda x: float(np.nanpercentile(x, 75))),
            frac_r2_ge_095=("r2", lambda x: float(np.nanmean(np.asarray(x, dtype=float) >= 0.95))),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
        )
        .reset_index()
        .sort_values(["system_id", "median_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )


def summarize_best_family(theta_table: pd.DataFrame) -> pd.DataFrame:
    return (
        theta_table.groupby(["system_id", "theta_family"], dropna=False)
        .agg(
            n_best=("unified_curve_id", "count"),
            median_r2=("theta_fit_r2", "median"),
            mean_r2=("theta_fit_r2", "mean"),
            median_rmse=("theta_fit_rmse", "median"),
        )
        .reset_index()
        .sort_values(["system_id", "n_best"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_manifest(theta_table: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    return {
        "artifact_name": "theta_table_v1",
        "seed": int(args.seed),
        "source_pool": str(args.pool_dir),
        "n_curves": int(theta_table["unified_curve_id"].nunique()),
        "n_systems": int(theta_table["system_id"].nunique()),
        "systems": sorted(theta_table["system_id"].dropna().unique().tolist()),
        "family_specs": {
            name: {
                "n_params": int(spec.n_params),
                "bounds_lo": list(spec.bounds_lo),
                "bounds_hi": list(spec.bounds_hi),
            }
            for name, spec in FAMILY_SPECS.items()
        },
        "theta_columns": theta_table.columns.tolist(),
    }


def write_summary(
    out_path: Path,
    theta_table: pd.DataFrame,
    family_summary: pd.DataFrame,
    best_family_summary: pd.DataFrame,
) -> None:
    valid_r2 = theta_table["theta_fit_r2"].dropna().to_numpy(dtype=float)
    lines = [
        "# Freeze Theta Targets",
        "",
        "This artifact freezes one theta target per curve for the clean main cumulative pool.",
        "",
        f"- n_curves: `{int(theta_table['unified_curve_id'].nunique())}`",
        f"- n_systems: `{int(theta_table['system_id'].nunique())}`",
        f"- median best-family R2: `{float(np.median(valid_r2)):.4f}`",
        f"- fraction best-family R2 >= 0.95: `{float(np.mean(valid_r2 >= 0.95)):.4f}`",
        "",
        "## Best family counts",
        "",
    ]
    family_counts = theta_table["theta_family"].value_counts()
    for family, count in family_counts.items():
        lines.append(f"- `{family}`: `{int(count)}` curves")
    lines.extend(["", "## Best family by system", ""])
    for _, row in best_family_summary.iterrows():
        lines.append(
            f"- `{row['system_id']}` / `{row['theta_family']}`: "
            f"n_best={int(row['n_best'])}, median_R2={row['median_r2']:.4f}, median_RMSE={row['median_rmse']:.4f}"
        )
    lines.extend(["", "## Per-system family fit snapshots", ""])
    top = (
        family_summary.sort_values(["system_id", "median_r2"], ascending=[True, False])
        .groupby("system_id", sort=False)
        .head(1)
    )
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['system_id']}`: best family `{row['family']}` "
            f"(median_R2={row['median_r2']:.4f}, frac>=0.95={row['frac_r2_ge_095']:.4f})"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    pool = load_pool(args.pool_dir)
    theta_table, all_fits = freeze_theta_table(pool["curves"])
    family_summary = summarize_family_metrics(all_fits)
    best_family_summary = summarize_best_family(theta_table)
    manifest = build_manifest(theta_table, args)

    theta_table.to_parquet(args.outdir / "theta_table_v1.parquet", index=False)
    all_fits.to_csv(args.outdir / "all_family_fits.csv", index=False)
    family_summary.to_csv(args.outdir / "family_summary.csv", index=False)
    best_family_summary.to_csv(args.outdir / "best_family_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(args.outdir / "summary.md", theta_table, family_summary, best_family_summary)

    print(f"[freeze-theta-targets] wrote outputs to {args.outdir}")
    print(
        "[freeze-theta-targets] "
        f"curves={manifest['n_curves']} systems={manifest['n_systems']} "
        f"median_best_r2={float(np.median(theta_table['theta_fit_r2'].dropna().to_numpy(dtype=float))):.4f}"
    )


if __name__ == "__main__":
    main()
