"""63 - PCA-P Phase A: build simulator pull-back targets for cross321.

For each curve in cross321, sample candidate theta from the prior +
local refinement, weight by likelihood against (t_obs, q_obs), and fit
a low-rank Gaussian (mu, U, sigma_active) — the "feasibility family".

Outputs:
  outputs/63_pcap_targets/per_curve_targets.npz   # numpy archive
  outputs/63_pcap_targets/per_curve_diagnostics.csv  # ESS / log_evidence
  outputs/63_pcap_targets/summary.txt

Run first on a small subset (--n-curves 20) to confirm ESS is sane;
then full run.

Why this script exists: the user's intuition was that the project should
NOT depend on early-Q observations at inference (which would block any
zero-shot SOTA claim). PCA-P pulls per-curve feasibility distributions
out of the simulator using real observations during TRAINING, then
trains a formulation-only encoder against those distributions, so
INFERENCE is true zero-shot. This script delivers Phase A: the
feasibility targets.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import pullback_target
from simulator import PLGABiphasic


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader_38d = _load_script(
    Path(__file__).resolve().parent / "38d_baselines_groupkfold.py",
    "_loader_38d_for_63",
)

CROSS_DOI_DATA = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)
MATCHED_FIDS_CSV = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
FULL_FIT_BANK = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/63_pcap_targets"))
    ap.add_argument("--n-curves", type=int, default=-1,
                    help="cap n curves (debug); -1 = all")
    ap.add_argument("--n-prior", type=int, default=2000)
    ap.add_argument("--n-local", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--sigma-likelihood", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sim = PLGABiphasic()
    FORMULATION_COL_MAP = _loader_38d.FORMULATION_COL_MAP
    FID_COL = _loader_38d.FID_COL

    df_meta = pd.read_excel(CROSS_DOI_DATA, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())
    df_bank = pd.read_csv(FULL_FIT_BANK)
    param_names = list(sim.param_names)
    df_bank_cross = df_bank[df_bank["dataset"] == "cross321"].copy()
    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )
    curves = _loader_38d._load_external_records(
        xlsx_path=CROSS_DOI_DATA, matched_fids_csv=MATCHED_FIDS_CSV,
        t_grid_max_days=90.0,
    )
    curve_map = {c.fid: c for c in curves}

    fids: list[int] = []
    curves_aligned: list[object] = []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        fids.append(int(row[FID_COL]))
        curves_aligned.append(c)
    print(f"[63] n_curves available: {len(fids)}", flush=True)

    if args.n_curves > 0:
        fids = fids[:args.n_curves]
        curves_aligned = curves_aligned[:args.n_curves]
        print(f"[63] limited to first {len(fids)} curves for debug", flush=True)

    # Run pull-back per curve
    mu_arr = np.zeros((len(fids), sim.n_params), dtype=np.float32)
    U_arr = np.zeros((len(fids), sim.n_params, args.rank), dtype=np.float32)
    sigma_arr = np.zeros((len(fids), args.rank), dtype=np.float32)
    diag_rows: list[dict] = []

    t0_all = time.time()
    for j, c in enumerate(curves_aligned):
        t0 = time.time()
        target = pullback_target(
            simulator=sim,
            t_obs=c.t_obs.astype(np.float64),
            q_obs=c.q_obs.astype(np.float64),
            n_prior_samples=args.n_prior, sigma_likelihood=args.sigma_likelihood,
            top_k=args.top_k, rank=args.rank,
            seed=args.seed + j,
            use_local_refine=True, n_local=args.n_local,
        )
        mu_arr[j] = target.mu
        U_arr[j] = target.U
        sigma_arr[j] = target.sigma_active
        elapsed = time.time() - t0
        diag_rows.append({
            "fid": fids[j],
            "ess": target.ess,
            "log_evidence": target.log_evidence,
            "n_kept": target.n_kept,
            "method": target.method,
            "sigma_max": float(target.sigma_active.max()),
            "sigma_min": float(target.sigma_active.min()),
            "sigma_mean": float(target.sigma_active.mean()),
            "elapsed_s": round(elapsed, 2),
        })
        if (j + 1) % 10 == 0 or j == 0:
            print(f"  [63] {j+1}/{len(fids)}  ESS={target.ess:.1f}  "
                  f"sigma_mean={target.sigma_active.mean():.3f}  "
                  f"({elapsed:.1f}s/curve)", flush=True)

    np.savez_compressed(
        args.out / "per_curve_targets.npz",
        fids=np.array(fids), mu=mu_arr, U=U_arr, sigma=sigma_arr,
    )
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(args.out / "per_curve_diagnostics.csv", index=False)

    total = time.time() - t0_all
    with open(args.out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== PCA-P Phase A — simulator pull-back targets ===\n\n")
        f.write(f"n_curves    : {len(fids)}\n")
        f.write(f"n_prior     : {args.n_prior}\n")
        f.write(f"n_local     : {args.n_local}\n")
        f.write(f"top_k       : {args.top_k}\n")
        f.write(f"rank        : {args.rank}\n")
        f.write(f"sigma_like  : {args.sigma_likelihood}\n")
        f.write(f"wall (min)  : {total/60:.1f}\n\n")
        f.write("ESS distribution (effective sample size):\n")
        f.write(f"  min   : {diag['ess'].min():.1f}\n")
        f.write(f"  p25   : {diag['ess'].quantile(0.25):.1f}\n")
        f.write(f"  median: {diag['ess'].median():.1f}\n")
        f.write(f"  p75   : {diag['ess'].quantile(0.75):.1f}\n")
        f.write(f"  max   : {diag['ess'].max():.1f}\n\n")
        f.write("Target sigma_active distribution (mean over rank):\n")
        f.write(f"  min   : {diag['sigma_mean'].min():.4f}\n")
        f.write(f"  p25   : {diag['sigma_mean'].quantile(0.25):.4f}\n")
        f.write(f"  median: {diag['sigma_mean'].median():.4f}\n")
        f.write(f"  p75   : {diag['sigma_mean'].quantile(0.75):.4f}\n")
        f.write(f"  max   : {diag['sigma_mean'].max():.4f}\n\n")
        f.write("Health check:\n")
        if diag['ess'].median() >= 20:
            f.write("  ESS median >= 20: GOOD (curve is informative about theta)\n")
        else:
            f.write("  ESS median < 20: WARN (top-k may be dominated by noise)\n")

    print(f"\n[63] done in {total/60:.1f} min", flush=True)
    with open(args.out / "summary.txt", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
