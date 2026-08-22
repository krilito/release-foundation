"""
28 - Descriptor-conditioned feasible-prior diagnostic (kNN prototype).

What this does:
    1. Load full-curve oracle theta fits from script 27a on the matched
       321 PLGA curves.
    2. Re-link each fitted curve to its raw formulation descriptors from the
       same 321 xlsx used elsewhere in the repo.
    3. For each curve, find its k nearest neighbors in descriptor space
       (leave-one-out), then define a local feasible prior:

           theta | x  ~  Normal(mu_knn(x), sigma_knn(x)^2)

       where mu_knn / sigma_knn are estimated from neighbors' theta_hat.
    4. Report whether this local prior is materially narrower than the global
       theta spread, and whether the held-out curve's theta_hat typically lands
       inside the local 1σ / 2σ bands.

Why this exists:
    The project now has strong evidence that the PLGA ODE can express the
    observed curves (27a), but parameter identifiability remains weak. This
    script is the smallest safe prototype of a "mechanism feasibility filter":
    not every theta that fits a curve should be equally plausible for a given
    descriptor vector x. Before changing any training code, we first ask:

        do nearby formulations in descriptor space occupy a narrower theta
        family than the global population?

    If the answer is "no", integrating a descriptor-conditioned prior into SBI
    would likely be decorative. If the answer is "yes", this becomes a concrete
    candidate for a Stage-2 feasibility filter.

Consumes:
    outputs/27a_ode_expressivity_audit/per_curve.csv
    D:\chemical-world-model-v0\datset\321PLGA\A Dataset on Formulation Parameters
      and Characteristics of Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx

Produces:
    outputs/28_descriptor_feasible_prior/per_curve_local_prior.csv
    outputs/28_descriptor_feasible_prior/param_summary.csv
    outputs/28_descriptor_feasible_prior/summary.txt

Expected runtime:
    < 5 s.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402

FID_COL = "fid"
FEATURE_COLS = (
    "LA/GA",
    "Polymer_MW",
    "Initial D/M ratio",
    "DLC",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_LogP",
)
_321_RENAME = {
    "Formulation Index": FID_COL,
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",
}


def _load_cross_doi_descriptors(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).rename(columns=_321_RENAME)
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    keep = [FID_COL, *FEATURE_COLS]
    return df.drop_duplicates(FID_COL)[keep].copy()


def _zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std > 0.0, std, 1.0)
    return (x - mean) / std, mean, std


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    gram = x @ x.T
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * gram
    return np.maximum(d2, 0.0)


def _local_prior_table(
    merged: pd.DataFrame,
    theta_cols: list[str],
    k_neighbors: int,
    min_sigma_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = merged.loc[:, FEATURE_COLS].to_numpy(dtype=float)
    feat_z, _, _ = _zscore(feat)
    d2 = _pairwise_sq_dists(feat_z)
    np.fill_diagonal(d2, np.inf)

    theta = merged.loc[:, theta_cols].to_numpy(dtype=float)
    global_mean = theta.mean(axis=0)
    global_std = theta.std(axis=0, ddof=0)
    sigma_floor = np.maximum(global_std * min_sigma_frac, 1e-6)

    per_curve_rows: list[dict[str, float | int | str]] = []
    for i, row in merged.reset_index(drop=True).iterrows():
        nbr_idx = np.argsort(d2[i])[:k_neighbors]
        theta_nbr = theta[nbr_idx]
        mu = theta_nbr.mean(axis=0)
        sigma = np.maximum(theta_nbr.std(axis=0, ddof=0), sigma_floor)
        z = (theta[i] - mu) / sigma

        out: dict[str, float | int | str] = {
            FID_COL: int(row[FID_COL]),
            "r2": float(row["r2"]),
            "neighbor_fids": ",".join(str(int(v)) for v in merged.iloc[nbr_idx][FID_COL].tolist()),
        }
        for j, name in enumerate(theta_cols):
            out[f"{name}_true"] = float(theta[i, j])
            out[f"{name}_mu"] = float(mu[j])
            out[f"{name}_sigma"] = float(sigma[j])
            out[f"{name}_abs_z"] = float(abs(z[j]))
            out[f"{name}_inside_1sigma"] = bool(abs(z[j]) <= 1.0)
            out[f"{name}_inside_2sigma"] = bool(abs(z[j]) <= 2.0)
        per_curve_rows.append(out)

    per_curve = pd.DataFrame(per_curve_rows)

    param_rows: list[dict[str, float | str]] = []
    for name, gmu, gstd in zip(theta_cols, global_mean, global_std, strict=True):
        sigma_col = f"{name}_sigma"
        z_col = f"{name}_abs_z"
        param_rows.append({
            "param": name,
            "global_mean": float(gmu),
            "global_std": float(gstd),
            "median_local_sigma": float(per_curve[sigma_col].median()),
            "mean_local_sigma": float(per_curve[sigma_col].mean()),
            "median_shrink_ratio": float(per_curve[sigma_col].median() / max(gstd, 1e-12)),
            "mean_shrink_ratio": float(per_curve[sigma_col].mean() / max(gstd, 1e-12)),
            "coverage_1sigma": float(per_curve[f"{name}_inside_1sigma"].mean()),
            "coverage_2sigma": float(per_curve[f"{name}_inside_2sigma"].mean()),
            "median_abs_z": float(per_curve[z_col].median()),
            "p90_abs_z": float(per_curve[z_col].quantile(0.90)),
        })
    return per_curve, pd.DataFrame(param_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--theta-csv",
        type=Path,
        default=Path("outputs/27a_ode_expressivity_audit/per_curve.csv"),
    )
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument("--k-neighbors", type=int, default=15)
    ap.add_argument(
        "--min-r2",
        type=float,
        default=0.95,
        help="Only use oracle fits above this R^2 threshold as pseudo-ground-truth theta.",
    )
    ap.add_argument(
        "--min-sigma-frac",
        type=float,
        default=0.10,
        help="Local sigma floor as a fraction of global std, avoids fake zero-width priors.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/28_descriptor_feasible_prior"),
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    sim = PLGABiphasic()
    theta_cols = list(sim.param_names)

    theta_df = pd.read_csv(args.theta_csv)
    desc_df = _load_cross_doi_descriptors(args.cross_doi_data)
    merged = theta_df.merge(desc_df, on=FID_COL, how="inner")
    merged = merged[np.isfinite(merged["r2"]) & (merged["r2"] >= args.min_r2)].copy()
    merged = merged.dropna(subset=[*theta_cols, *FEATURE_COLS]).reset_index(drop=True)

    if len(merged) < args.k_neighbors + 1:
        raise SystemExit(
            f"Need at least k+1 rows after filtering, got {len(merged)} rows "
            f"for k={args.k_neighbors}."
        )

    per_curve, param_summary = _local_prior_table(
        merged=merged,
        theta_cols=theta_cols,
        k_neighbors=args.k_neighbors,
        min_sigma_frac=args.min_sigma_frac,
    )

    per_curve_path = args.out / "per_curve_local_prior.csv"
    param_path = args.out / "param_summary.csv"
    summary_path = args.out / "summary.txt"
    per_curve.to_csv(per_curve_path, index=False)
    param_summary.to_csv(param_path, index=False)

    median_shrink = float(param_summary["median_shrink_ratio"].median())
    strong_params = param_summary[param_summary["median_shrink_ratio"] < 0.80]["param"].tolist()
    moderate_params = param_summary[
        (param_summary["median_shrink_ratio"] >= 0.80)
        & (param_summary["median_shrink_ratio"] < 0.90)
    ]["param"].tolist()
    weak_params = param_summary[param_summary["median_shrink_ratio"] >= 0.95]["param"].tolist()

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== 28 -- descriptor-conditioned feasible prior (kNN prototype) ===\n\n")
        f.write(f"theta_csv           : {args.theta_csv}\n")
        f.write(f"cross_doi_data      : {args.cross_doi_data}\n")
        f.write(f"n_curves_used       : {len(merged)}\n")
        f.write(f"min_r2              : {args.min_r2:.3f}\n")
        f.write(f"k_neighbors         : {args.k_neighbors}\n")
        f.write(f"min_sigma_frac      : {args.min_sigma_frac:.3f}\n")
        f.write(f"feature_cols        : {list(FEATURE_COLS)}\n")
        f.write(f"theta_cols          : {theta_cols}\n\n")
        f.write("Per-parameter summary:\n")
        f.write(param_summary.to_string(index=False))
        f.write("\n\n")
        f.write(f"Median shrink ratio across params : {median_shrink:.3f}\n")
        f.write(f"Params with strong narrowing      : {strong_params}\n")
        f.write(f"Params with moderate narrowing    : {moderate_params}\n")
        f.write(f"Params with little/no narrowing   : {weak_params}\n\n")
        f.write("Interpretation:\n")
        f.write("  shrink ratio < 1.0  -> local descriptor neighborhoods occupy a narrower\n")
        f.write("                         theta family than the global population.\n")
        f.write("  coverage_1sigma     -> how often the held-out theta_hat lands inside the\n")
        f.write("                         local 1-sigma band.\n")
        f.write("  coverage_2sigma     -> same for 2-sigma; too low means local prior is too\n")
        f.write("                         sharp, too high means it is barely filtering.\n")
        f.write("\n")
        if strong_params:
            f.write(
                "Verdict: descriptor space carries usable signal for at least part of the\n"
                "theta family. This is enough to justify a next-step integration prototype.\n"
            )
        elif moderate_params:
            f.write(
                "Verdict: descriptor space carries weak-to-moderate narrowing signal, but\n"
                "not enough to hard-wire directly into training. The right next step is a\n"
                "better feasibility diagnostic, not immediate SBI integration.\n"
            )
        else:
            f.write(
                "Verdict: descriptor neighborhoods do not materially narrow theta. Stop here;\n"
                "do not integrate a descriptor-conditioned prior into training yet.\n"
            )

    print(f"[28] wrote {per_curve_path}")
    print(f"[28] wrote {param_path}")
    print(f"[28] wrote {summary_path}")
    print("\n[28] param summary:")
    print(param_summary.to_string(index=False))


if __name__ == "__main__":
    main()
