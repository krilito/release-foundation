from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


DEFAULT_CROSS_DOI_XLSX = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)


@dataclass(frozen=True)
class FamilySpec:
    name: str
    n_params: int
    bounds_lo: tuple[float, ...]
    bounds_hi: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-doi-xlsx",
        type=Path,
        default=DEFAULT_CROSS_DOI_XLSX,
        help="optional local 321-curve xlsx; if absent, audit stays internal-only",
    )
    return parser.parse_args()


def load_internal_181(repo_root: Path) -> pd.DataFrame:
    df = pd.read_csv(repo_root / "data" / "Dataset_17_feat_augmented.csv")
    return (
        df.rename(
            columns={
                "Experimental_index": "curve_id",
                "Time": "time",
                "Release": "release",
            }
        )[["curve_id", "time", "release"]]
        .assign(dataset="internal181")
        .copy()
    )


def load_cross_doi_321(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path).rename(
        columns={
            "Formulation Index": "curve_id",
            "Time": "time",
            "Release": "release",
        }
    )
    curves = (
        raw.groupby(["curve_id", "time"], as_index=False, sort=False)["release"]
        .mean()
        .assign(dataset="cross321")
        .copy()
    )
    return curves


def load_combined_curves(repo_root: Path, cross_doi_xlsx: Path) -> pd.DataFrame:
    parts = [load_internal_181(repo_root)]
    if cross_doi_xlsx.exists():
        parts.append(load_cross_doi_321(cross_doi_xlsx))
    else:
        print(f"[shape-family-audit] cross-DOI xlsx not found, skipping: {cross_doi_xlsx}")
    curves = pd.concat(parts, ignore_index=True)
    curves["curve_id"] = curves["curve_id"].astype(str)
    curves["time"] = pd.to_numeric(curves["time"], errors="coerce")
    curves["release"] = pd.to_numeric(curves["release"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "time", "release"])
    curves["release"] = curves["release"].clip(lower=0.0)
    return curves


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((yt - yp) ** 2))
    return float(1.0 - ss_res / ss_tot)


def weibull_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, tau, beta = p
    t_pos = np.clip(t, 0.0, None)
    return qmax * (1.0 - np.exp(-np.power(np.clip(t_pos / tau, 0.0, None), beta)))


def biexponential_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, w_fast, k_fast, k_slow = p
    t_pos = np.clip(t, 0.0, None)
    fast = 1.0 - np.exp(-k_fast * t_pos)
    slow = 1.0 - np.exp(-k_slow * t_pos)
    return qmax * (w_fast * fast + (1.0 - w_fast) * slow)


def logistic_power_curve(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    qmax, t50, alpha = p
    t_pos = np.clip(t, 0.0, None)
    return qmax * (np.power(t_pos, alpha) / (np.power(t50, alpha) + np.power(t_pos, alpha) + 1e-12))


FAMILY_SPECS = {
    "weibull": FamilySpec("weibull", 3, (0.3, 0.05, 0.15), (1.2, 400.0, 6.0)),
    "biexponential": FamilySpec("biexponential", 4, (0.3, 0.0, 1e-4, 1e-4), (1.2, 1.0, 5.0, 1.0)),
    "logistic_power": FamilySpec("logistic_power", 3, (0.3, 0.05, 0.2), (1.2, 400.0, 6.0)),
}

FAMILY_FUNCS = {
    "weibull": weibull_curve,
    "biexponential": biexponential_curve,
    "logistic_power": logistic_power_curve,
}


def initial_guesses(family: str, t: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    y_max = float(np.clip(np.max(y), 0.3, 1.1))
    t_end = float(max(np.max(t), 1.0))
    if family == "weibull":
        return [
            np.array([y_max, max(t_end / 3.0, 0.3), 1.0]),
            np.array([min(1.05, y_max + 0.05), max(t_end / 2.0, 0.5), 0.7]),
            np.array([min(1.10, y_max + 0.1), max(t_end / 5.0, 0.2), 1.5]),
        ]
    if family == "biexponential":
        return [
            np.array([y_max, 0.5, 0.4, 0.03]),
            np.array([min(1.05, y_max + 0.05), 0.7, 0.8, 0.05]),
            np.array([min(1.10, y_max + 0.1), 0.3, 0.2, 0.01]),
        ]
    if family == "logistic_power":
        return [
            np.array([y_max, max(t_end / 3.0, 0.3), 1.2]),
            np.array([min(1.05, y_max + 0.05), max(t_end / 2.0, 0.5), 0.9]),
            np.array([min(1.10, y_max + 0.1), max(t_end / 5.0, 0.2), 1.8]),
        ]
    raise KeyError(family)


def fit_family(family: str, t: np.ndarray, y: np.ndarray) -> dict[str, object]:
    spec = FAMILY_SPECS[family]
    fn = FAMILY_FUNCS[family]
    best = None

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = fn(t, p)
        return pred - y

    for x0 in initial_guesses(family, t, y):
        x0 = np.clip(x0, spec.bounds_lo, spec.bounds_hi)
        try:
            res = least_squares(
                residuals,
                x0=x0,
                bounds=(np.array(spec.bounds_lo), np.array(spec.bounds_hi)),
                max_nfev=4000,
            )
        except Exception:
            continue
        pred = fn(t, res.x)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        mae = float(np.mean(np.abs(pred - y)))
        r2 = safe_r2(y, pred)
        row = {
            "family": family,
            "params": res.x.tolist(),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pred": pred,
            "cost": float(res.cost),
        }
        if best is None or row["rmse"] < best["rmse"]:
            best = row

    if best is None:
        pred = np.full_like(y, np.mean(y))
        return {
            "family": family,
            "params": [np.nan] * spec.n_params,
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))),
            "r2": safe_r2(y, pred),
            "pred": pred,
            "cost": np.nan,
        }
    return best


def per_curve_family_audit(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, curve_id), sub in curves.groupby(["dataset", "curve_id"], sort=True):
        sub = sub.sort_values("time").reset_index(drop=True)
        t = sub["time"].to_numpy(dtype=float)
        y = sub["release"].to_numpy(dtype=float)
        for family in FAMILY_SPECS:
            fit = fit_family(family, t, y)
            rows.append(
                {
                    "dataset": dataset,
                    "curve_id": curve_id,
                    "family": family,
                    "n_points": len(sub),
                    "duration": float(t[-1] - t[0]),
                    "release_final": float(y[-1]),
                    "rmse": fit["rmse"],
                    "mae": fit["mae"],
                    "r2": fit["r2"],
                    "params_json": json.dumps(fit["params"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_family_metrics(per_curve: pd.DataFrame) -> pd.DataFrame:
    return (
        per_curve.groupby(["dataset", "family"], dropna=False)
        .agg(
            n_curves=("curve_id", "count"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            p25_r2=("r2", lambda x: float(np.nanpercentile(x, 25))),
            p75_r2=("r2", lambda x: float(np.nanpercentile(x, 75))),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_mae=("mae", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "median_r2"], ascending=[True, False])
    )


def summarize_best_family(per_curve: pd.DataFrame) -> pd.DataFrame:
    idx = per_curve.groupby(["dataset", "curve_id"])["rmse"].idxmin()
    best = per_curve.loc[idx].copy()
    return (
        best.groupby(["dataset", "family"], dropna=False)
        .agg(
            n_best=("curve_id", "count"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            median_rmse=("rmse", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "n_best"], ascending=[True, False])
    )


def write_summary_markdown(
    family_summary: pd.DataFrame,
    best_family_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Shape-Family Audit",
        "",
        "Goal:",
        "",
        "- test whether small parametric curve families already explain most of the corpus",
        "- if yes, shrink the problem toward low-dimensional shape regression",
        "- if no, keep the larger curve-world route alive",
        "",
    ]
    for dataset in family_summary["dataset"].drop_duplicates().tolist():
        lines.extend([f"## {dataset}", ""])
        sub = family_summary[family_summary["dataset"] == dataset]
        lines.append("Family metrics:")
        lines.append("")
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['family']}`: median R²={row['median_r2']:.4f}, "
                f"mean R²={row['mean_r2']:.4f}, median RMSE={row['median_rmse']:.4f}"
            )
        lines.extend(["", "Best-family counts:", ""])
        best_sub = best_family_summary[best_family_summary["dataset"] == dataset]
        for _, row in best_sub.iterrows():
            lines.append(
                f"- `{row['family']}`: wins={int(row['n_best'])}, "
                f"median R² on wins={row['median_r2']:.4f}, median RMSE={row['median_rmse']:.4f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "121_shape_family_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_combined_curves(repo_root, args.cross_doi_xlsx)
    per_curve = per_curve_family_audit(curves)
    family_summary = summarize_family_metrics(per_curve)
    best_family_summary = summarize_best_family(per_curve)

    per_curve.to_csv(out_dir / "per_curve_family_metrics.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(out_dir / "family_summary.csv", index=False, encoding="utf-8-sig")
    best_family_summary.to_csv(out_dir / "best_family_summary.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(family_summary, best_family_summary, out_dir / "summary.md")

    print(f"[shape-family-audit] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
