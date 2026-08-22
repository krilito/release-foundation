"""
30 - Local effective-dimension audit from full-fit Jacobians.

What this does:
    1. Load the full-fit oracle bank produced by script 29.
    2. Reload the corresponding raw internal 181 and cross-DOI 321 curves.
    3. For each strong-fit curve, compute the Jacobian of Q(t_obs) with
       respect to the 9 simulator parameters at the full-fit theta.
    4. Scale parameter directions by prior range and inspect the singular
       value spectrum of the local Jacobian.

Why this exists:
    Script 29 asks a global practical question ("how many active parameters
    recover the curve from a reference mechanism?"). Script 30 asks the local
    geometric question: around the best-fit theta, how many parameter
    directions actually matter?

Outputs:
    outputs/30_local_effective_dimension_audit/per_curve_dimension.csv
    outputs/30_local_effective_dimension_audit/dataset_summary.csv
    outputs/30_local_effective_dimension_audit/summary.txt
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_plga_181  # noqa: E402
from posterior import interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

DATASET_COL = "dataset"
FID_COL = "fid"
TIME_COL = "Time"
Y_COL = "Release"
FULL_R2_COL = "full_r2"
QUALITY_COL = "quality"

_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}
_321_RENAME = {
    "Formulation Index": FID_COL,
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",
}
_321_DROP = (
    "Particle Size",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
)


@dataclass
class CurveRecord:
    dataset: str
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray
    quality: str


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_internal_records(csv_path: Path) -> list[CurveRecord]:
    curves = load_plga_181(csv_path)
    out: list[CurveRecord] = []
    for c in curves:
        out.append(
            CurveRecord(
                dataset="internal181",
                fid=int(c.formulation_id),
                t_obs=c.t.numpy().astype(float),
                q_obs=np.clip(c.Q.numpy().astype(float), 0.0, 1.0),
                quality="na",
            )
        )
    return out


def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
    t_grid_max_days: float,
    min_quality: str,
) -> list[CurveRecord]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df = (
        df.groupby([FID_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in (TIME_COL, Y_COL)},
            Y_COL: "mean",
        })
    )
    df = df[df[FID_COL].astype(int).isin(matched_fids)].copy()
    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)
    t_grid = torch.linspace(0.0, t_grid_max_days, 64)
    min_rank = _QUALITY_RANK[min_quality]
    out: list[CurveRecord] = []
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = g[TIME_COL].to_numpy(dtype=float)
        q_obs = np.clip(g[Y_COL].to_numpy(dtype=float), 0.0, 1.0)
        if len(t_obs) < 3:
            continue
        try:
            _, quality, _ = interpolate_to_grid(
                torch.tensor(t_obs, dtype=torch.float32),
                torch.tensor(q_obs, dtype=torch.float32),
                t_grid,
                t_max_days=t_grid_max_days,
            )
        except ValueError:
            continue
        if _QUALITY_RANK.get(quality, -1) < min_rank:
            continue
        out.append(
            CurveRecord(
                dataset="cross321",
                fid=int(fid),
                t_obs=t_obs,
                q_obs=q_obs,
                quality=quality,
            )
        )
    return out


def _curve_jacobian(
    sim: PLGABiphasic,
    theta_np: np.ndarray,
    t_obs_np: np.ndarray,
    prior_range: np.ndarray,
) -> np.ndarray:
    theta = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32)
    q = sim.simulate(theta.unsqueeze(0), t_obs).squeeze(0)
    grads = []
    for i in range(len(t_obs_np)):
        grad = torch.autograd.grad(q[i], theta, retain_graph=True)[0]
        grads.append(grad.detach().cpu().numpy() * prior_range)
    return np.stack(grads, axis=0)


def _svd_metrics(jac: np.ndarray) -> dict[str, float]:
    s = np.linalg.svd(jac, compute_uv=False)
    s = np.asarray(s, dtype=float)
    energy = s ** 2
    total = float(np.sum(energy))
    if total <= 0.0 or not np.isfinite(total):
        return {
            "sv1": np.nan,
            "sv2": np.nan,
            "sv3": np.nan,
            "dim95": np.nan,
            "dim99": np.nan,
            "n_sv_rel_5pct": np.nan,
            "n_sv_rel_1pct": np.nan,
            "cond_nonzero": np.nan,
        }
    cum = np.cumsum(energy) / total
    sv1 = float(s[0]) if len(s) else np.nan
    rel = s / max(sv1, 1e-12)
    nz = s[s > 1e-12]
    return {
        "sv1": sv1,
        "sv2": float(s[1]) if len(s) > 1 else np.nan,
        "sv3": float(s[2]) if len(s) > 2 else np.nan,
        "dim95": float(np.searchsorted(cum, 0.95) + 1),
        "dim99": float(np.searchsorted(cum, 0.99) + 1),
        "n_sv_rel_5pct": float(np.sum(rel >= 0.05)),
        "n_sv_rel_1pct": float(np.sum(rel >= 0.01)),
        "cond_nonzero": float(nz[0] / nz[-1]) if len(nz) >= 2 else 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit/full_fit_bank.csv"),
    )
    ap.add_argument("--internal-data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument("--min-quality", type=str, default="high", choices=("high", "medium", "low"))
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("outputs/30_local_effective_dimension_audit"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    prior_range = (prior.high - prior.low).numpy()
    param_names = list(sim.param_names)

    curve_map = {
        (rec.dataset, rec.fid): rec
        for rec in [
            *_load_internal_records(args.internal_data),
            *_load_external_records(
                xlsx_path=args.cross_doi_data,
                matched_fids_csv=args.matched_fids_csv,
                t_grid_max_days=args.t_grid_max_days,
                min_quality=args.min_quality,
            ),
        ]
    }

    bank = pd.read_csv(args.full_fit_bank)
    bank = bank[np.isfinite(bank[FULL_R2_COL]) & (bank[FULL_R2_COL] >= args.fit_filter_r2)].copy()

    rows: list[dict[str, float | int | str]] = []
    for _i, row in bank.iterrows():
        dataset = str(row[DATASET_COL])
        fid = int(row[FID_COL])
        rec = curve_map[(dataset, fid)]
        jac = _curve_jacobian(
            sim=sim,
            theta_np=row[param_names].to_numpy(dtype=float),
            t_obs_np=rec.t_obs,
            prior_range=prior_range,
        )
        metrics = _svd_metrics(jac)
        out: dict[str, float | int | str] = {
            DATASET_COL: dataset,
            FID_COL: fid,
            QUALITY_COL: rec.quality,
            FULL_R2_COL: float(row[FULL_R2_COL]),
            "n_obs": len(rec.t_obs),
            "t_max": float(rec.t_obs.max()),
        }
        out.update(metrics)
        rows.append(out)
        if (len(rows) % 25) == 0:
            print(f"[30] jacobians {len(rows)}/{len(bank)}")

    per_curve = pd.DataFrame(rows)
    summary = (
        per_curve.groupby(DATASET_COL, as_index=False, sort=False)
        .agg({
            "dim95": ["median", "mean", "min", "max"],
            "dim99": ["median", "mean"],
            "n_sv_rel_5pct": ["median", "mean"],
            "n_sv_rel_1pct": ["median", "mean"],
            "cond_nonzero": ["median", "mean"],
            FID_COL: "count",
        })
    )
    summary.columns = [
        DATASET_COL,
        "dim95_median", "dim95_mean", "dim95_min", "dim95_max",
        "dim99_median", "dim99_mean",
        "n_sv_rel_5pct_median", "n_sv_rel_5pct_mean",
        "n_sv_rel_1pct_median", "n_sv_rel_1pct_mean",
        "cond_median", "cond_mean",
        "n_curves",
    ]

    per_curve_path = args.out / "per_curve_dimension.csv"
    summary_path = args.out / "dataset_summary.csv"
    txt_path = args.out / "summary.txt"
    per_curve.to_csv(per_curve_path, index=False)
    summary.to_csv(summary_path, index=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== 30 -- local effective-dimension audit ===\n\n")
        f.write(f"full_fit_bank       : {args.full_fit_bank}\n")
        f.write(f"fit_filter_r2       : {args.fit_filter_r2:.3f}\n")
        f.write(f"matched_fids_csv    : {args.matched_fids_csv}\n")
        f.write(f"cross_doi_data      : {args.cross_doi_data}\n\n")
        f.write("Dataset summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write("  dim95 is the smallest number of local singular directions that explain\n")
        f.write("  95% of sensitivity energy. Values around 3-4 support the claim that a\n")
        f.write("  curve may effectively live on a much lower-dimensional mechanism manifold\n")
        f.write("  than the global 9-parameter family.\n")

    print(f"[30] wrote {per_curve_path}")
    print(f"[30] wrote {summary_path}")
    print(f"[30] wrote {txt_path}")
    print("\n[30] dataset summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
