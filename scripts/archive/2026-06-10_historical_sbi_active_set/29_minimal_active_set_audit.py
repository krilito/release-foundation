"""
29 - Minimal active-set audit on internal 181 + cross-DOI 321.

What this does:
    1. Build a 9-D full-fit oracle bank for internal 181 (current ADR-026
       simulator) and load the existing 321 full-fit oracle bank from 27a.
    2. For each dataset, define a dataset-level reference mechanism
       theta_ref = median(theta_full) over curves with strong full-fit R^2.
    3. For each curve, greedily add 1, 2, 3, 4 active parameters that are
       allowed to deviate from theta_ref, while all other parameters are
       fixed at theta_ref.
    4. Report how much of the full-fit quality can be recovered with <=4
       active parameters.

Why this exists:
    If many curves recover near-full-fit quality with only 3-4 active
    parameters, then the per-curve effective active set is smaller than the
    global 9-parameter mechanism family.

Outputs:
    outputs/29_minimal_active_set_audit/full_fit_bank.csv
    outputs/29_minimal_active_set_audit/per_curve_active_set.csv
    outputs/29_minimal_active_set_audit/dataset_summary.csv
    outputs/29_minimal_active_set_audit/summary.txt

Expected runtime:
    internal full-fit bank (if missing) dominates; subset audit is moderate.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_plga_181  # noqa: E402
from posterior import interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

DATASET_COL = "dataset"
FID_COL = "fid"
TIME_COL = "Time"
Y_COL = "Release"
QUALITY_COL = "quality"
FULL_R2_COL = "full_r2"

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
    n_obs: int
    t_max: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _load_internal_records(csv_path: Path) -> list[CurveRecord]:
    curves = load_plga_181(csv_path)
    out: list[CurveRecord] = []
    for c in curves:
        t_obs = c.t.numpy().astype(float)
        q_obs = np.clip(c.Q.numpy().astype(float), 0.0, 1.0)
        out.append(
            CurveRecord(
                dataset="internal181",
                fid=int(c.formulation_id),
                t_obs=t_obs,
                q_obs=q_obs,
                quality="na",
                n_obs=len(t_obs),
                t_max=float(t_obs.max()),
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
                n_obs=len(t_obs),
                t_max=float(t_obs.max()),
            )
        )
    return out


def _full_fit_multistart(
    sim: PLGABiphasic,
    curve: CurveRecord,
    lows: np.ndarray,
    highs: np.ndarray,
    n_restarts: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lows + highs)
    best_theta: np.ndarray | None = None
    best_ss = np.inf

    def residual(theta_flat: np.ndarray) -> np.ndarray:
        return sim.simulate_numpy(theta_flat, curve.t_obs) - curve.q_obs

    for k in range(n_restarts):
        x0 = mid.copy() if k == 0 else rng.uniform(lows, highs)
        try:
            result = least_squares(
                residual,
                x0=x0,
                bounds=(lows, highs),
                method="trf",
                max_nfev=300,
            )
        except Exception:
            continue
        ss = float(np.sum(residual(result.x) ** 2))
        if np.isfinite(ss) and ss < best_ss:
            best_ss = ss
            best_theta = result.x.copy()

    if best_theta is None:
        row: dict[str, float | int] = {
            DATASET_COL: curve.dataset,
            FID_COL: curve.fid,
            "n_obs": curve.n_obs,
            "t_max": curve.t_max,
            QUALITY_COL: curve.quality,
            "rmse": np.nan,
            "mae": np.nan,
            FULL_R2_COL: np.nan,
        }
        for name in sim.param_names:
            row[name] = np.nan
        return row

    pred = sim.simulate_numpy(best_theta, curve.t_obs)
    res = curve.q_obs - pred
    row = {
        DATASET_COL: curve.dataset,
        FID_COL: curve.fid,
        "n_obs": curve.n_obs,
        "t_max": curve.t_max,
        QUALITY_COL: curve.quality,
        "rmse": float(np.sqrt(np.mean(res ** 2))),
        "mae": float(np.mean(np.abs(res))),
        FULL_R2_COL: _r2(curve.q_obs, pred),
    }
    for i, name in enumerate(sim.param_names):
        row[name] = float(best_theta[i])
    return row


def _fit_active_subset(
    sim: PLGABiphasic,
    curve: CurveRecord,
    theta_ref: np.ndarray,
    theta_oracle: np.ndarray,
    free_idx: list[int],
    lows: np.ndarray,
    highs: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    theta0 = theta_ref.copy()
    theta0[free_idx] = theta_oracle[free_idx]

    def residual(z: np.ndarray) -> np.ndarray:
        theta = theta_ref.copy()
        theta[free_idx] = z
        return sim.simulate_numpy(theta, curve.t_obs) - curve.q_obs

    z0 = theta0[free_idx]
    z_lo = lows[free_idx]
    z_hi = highs[free_idx]
    z0 = np.clip(z0, z_lo, z_hi)
    result = least_squares(
        residual,
        x0=z0,
        bounds=(z_lo, z_hi),
        method="trf",
        max_nfev=120,
    )
    theta = theta_ref.copy()
    theta[free_idx] = result.x
    pred = sim.simulate_numpy(theta, curve.t_obs)
    return theta, _r2(curve.q_obs, pred), float(np.sqrt(np.mean((curve.q_obs - pred) ** 2)))


def _greedy_active_path(
    sim: PLGABiphasic,
    curve: CurveRecord,
    theta_ref: np.ndarray,
    theta_oracle: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    param_names: list[str],
    k_max: int,
) -> dict[str, object]:
    path: dict[str, object] = {}
    ref_pred = sim.simulate_numpy(theta_ref, curve.t_obs)
    path["r2_k0"] = _r2(curve.q_obs, ref_pred)
    path["rmse_k0"] = float(np.sqrt(np.mean((curve.q_obs - ref_pred) ** 2)))

    active: list[int] = []
    theta_best = theta_ref.copy()
    remaining = list(range(len(param_names)))
    for k in range(1, k_max + 1):
        best_name = None
        best_idx: list[int] | None = None
        best_theta = None
        best_r2 = -np.inf
        best_rmse = np.inf
        for j in remaining:
            cand_idx = active + [j]
            theta_fit, r2_fit, rmse_fit = _fit_active_subset(
                sim=sim,
                curve=curve,
                theta_ref=theta_ref,
                theta_oracle=theta_oracle,
                free_idx=cand_idx,
                lows=lows,
                highs=highs,
            )
            if (r2_fit > best_r2) or (r2_fit == best_r2 and rmse_fit < best_rmse):
                best_idx = cand_idx
                best_name = param_names[j]
                best_theta = theta_fit
                best_r2 = r2_fit
                best_rmse = rmse_fit
        assert best_idx is not None and best_theta is not None and best_name is not None
        active = best_idx
        theta_best = best_theta
        remaining = [j for j in remaining if j not in active]
        path[f"added_param_k{k}"] = best_name
        path[f"active_set_k{k}"] = ",".join(param_names[j] for j in active)
        path[f"r2_k{k}"] = float(best_r2)
        path[f"rmse_k{k}"] = float(best_rmse)
        for j, name in enumerate(param_names):
            path[f"{name}_k{k}"] = float(theta_best[j])
    return path


def _k_close_to_full(row: pd.Series, k_max: int, delta_r2_tol: float) -> float:
    full_r2 = float(row[FULL_R2_COL])
    if not np.isfinite(full_r2):
        return np.nan
    for k in range(0, k_max + 1):
        r2k = float(row[f"r2_k{k}"])
        if np.isfinite(r2k) and (full_r2 - r2k) <= delta_r2_tol:
            return float(k)
    return np.nan


def _dataset_summary(
    per_curve: pd.DataFrame,
    param_names: list[str],
    k_max: int,
    delta_r2_tol: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, sub in per_curve.groupby(DATASET_COL, sort=False):
        row: dict[str, object] = {
            DATASET_COL: dataset,
            "n_curves": int(len(sub)),
            "median_full_r2": float(sub[FULL_R2_COL].median()),
            "median_r2_k0": float(sub["r2_k0"].median()),
        }
        for k in range(1, k_max + 1):
            row[f"median_r2_k{k}"] = float(sub[f"r2_k{k}"].median())
            row[f"frac_close_by_k{k}"] = float((sub["k_close_to_full"] <= k).mean())
        first_counts = sub["added_param_k1"].value_counts()
        row["most_common_first_param"] = first_counts.index[0] if len(first_counts) else "na"
        for name in param_names:
            row[f"selected_first_{name}"] = int((sub["added_param_k1"] == name).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument(
        "--external-oracle-csv",
        type=Path,
        default=Path("outputs/27a_ode_expressivity_audit/per_curve.csv"),
    )
    ap.add_argument("--min-quality", type=str, default="high", choices=("high", "medium", "low"))
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--internal-restarts", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--delta-r2-tol", type=float, default=0.02)
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--max-internal-curves", type=int, default=None)
    ap.add_argument("--max-external-curves", type=int, default=None)
    ap.add_argument(
        "--reuse-full-fit-bank",
        action="store_true",
        help="If out/full_fit_bank.csv already exists, reuse it instead of recomputing internal full fits.",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/29_minimal_active_set_audit"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy()
    highs = prior.high.numpy()
    param_names = list(sim.param_names)

    internal_records = _load_internal_records(args.internal_data)
    external_records = _load_external_records(
        xlsx_path=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid_max_days=args.t_grid_max_days,
        min_quality=args.min_quality,
    )
    if args.max_internal_curves is not None:
        internal_records = internal_records[: int(args.max_internal_curves)]
    if args.max_external_curves is not None:
        external_records = external_records[: int(args.max_external_curves)]
    curve_map = {
        (rec.dataset, rec.fid): rec
        for rec in [*internal_records, *external_records]
    }

    full_fit_path = args.out / "full_fit_bank.csv"
    full_fit_progress_path = args.out / "full_fit_bank_internal_progress.csv"
    if args.reuse_full_fit_bank and full_fit_path.exists():
        full_bank = pd.read_csv(full_fit_path)
        print(f"[29] reusing existing full-fit bank: {full_fit_path}")
    else:
        full_rows: list[dict[str, object]] = []
        if full_fit_progress_path.exists():
            progress_df = pd.read_csv(full_fit_progress_path)
            progress_df = progress_df[progress_df[DATASET_COL] == "internal181"].copy()
            progress_df = progress_df.drop_duplicates(subset=[DATASET_COL, FID_COL], keep="last")
            progress_df = progress_df[
                progress_df[FID_COL].astype(int).isin({rec.fid for rec in internal_records})
            ].copy()
            full_rows.extend(progress_df.to_dict(orient="records"))
            print(
                "[29] resuming internal full-fit progress: "
                f"{len(progress_df)}/{len(internal_records)} curves from {full_fit_progress_path}"
            )
        done_fids = {int(row[FID_COL]) for row in full_rows if row.get(DATASET_COL) == "internal181"}
        t0 = time.time()
        for i, rec in enumerate(internal_records, start=1):
            if rec.fid in done_fids:
                continue
            full_rows.append(
                _full_fit_multistart(
                    sim=sim,
                    curve=rec,
                    lows=lows,
                    highs=highs,
                    n_restarts=args.internal_restarts,
                    seed=args.seed + i,
                )
            )
            if i % 10 == 0:
                progress_df = pd.DataFrame(full_rows)
                progress_df = progress_df[progress_df[DATASET_COL] == "internal181"].copy()
                progress_df.to_csv(full_fit_progress_path, index=False)
            if i % 20 == 0:
                print(f"[29] internal full fits {i}/{len(internal_records)} elapsed {time.time() - t0:.1f}s")

        ext_df = pd.read_csv(args.external_oracle_csv).copy()
        ext_df[DATASET_COL] = "cross321"
        ext_df = ext_df.rename(columns={"r2": FULL_R2_COL})
        ext_keep = [DATASET_COL, FID_COL, "n_obs", "t_max", QUALITY_COL, "rmse", "mae", FULL_R2_COL, *param_names]
        ext_df = ext_df[ext_keep].copy()
        external_fids = {rec.fid for rec in external_records}
        ext_df = ext_df[ext_df[FID_COL].astype(int).isin(external_fids)].copy()
        pd.DataFrame(full_rows).to_csv(full_fit_progress_path, index=False)
        full_bank = pd.concat([pd.DataFrame(full_rows), ext_df], ignore_index=True)
        full_bank.to_csv(full_fit_path, index=False)

    ref_bank = full_bank[np.isfinite(full_bank[FULL_R2_COL]) & (full_bank[FULL_R2_COL] >= args.fit_filter_r2)].copy()
    theta_ref = {
        dataset: sub[param_names].median().to_numpy(dtype=float)
        for dataset, sub in ref_bank.groupby(DATASET_COL, sort=False)
    }

    per_curve_progress_path = args.out / "per_curve_active_set_progress.csv"
    per_curve_rows: list[dict[str, object]] = []
    all_curves = full_bank[np.isfinite(full_bank[FULL_R2_COL]) & (full_bank[FULL_R2_COL] >= args.fit_filter_r2)].copy()
    if per_curve_progress_path.exists():
        progress_df = pd.read_csv(per_curve_progress_path)
        progress_df = progress_df.drop_duplicates(subset=[DATASET_COL, FID_COL], keep="last")
        progress_df = progress_df[
            progress_df.apply(
                lambda r: (str(r[DATASET_COL]), int(r[FID_COL])) in curve_map,
                axis=1,
            )
        ].copy()
        per_curve_rows.extend(progress_df.to_dict(orient="records"))
        print(
            "[29] resuming active-set progress: "
            f"{len(progress_df)}/{len(all_curves)} curves from {per_curve_progress_path}"
        )
    done_curve_keys = {
        (str(row[DATASET_COL]), int(row[FID_COL]))
        for row in per_curve_rows
    }
    for _idx, row in all_curves.iterrows():
        dataset = str(row[DATASET_COL])
        fid = int(row[FID_COL])
        if (dataset, fid) in done_curve_keys:
            continue
        rec = curve_map[(dataset, fid)]
        theta_oracle = row[param_names].to_numpy(dtype=float)
        path = _greedy_active_path(
            sim=sim,
            curve=rec,
            theta_ref=theta_ref[dataset],
            theta_oracle=theta_oracle,
            lows=lows,
            highs=highs,
            param_names=param_names,
            k_max=args.k_max,
        )
        out = {
            DATASET_COL: dataset,
            FID_COL: fid,
            "n_obs": rec.n_obs,
            "t_max": rec.t_max,
            QUALITY_COL: rec.quality,
            FULL_R2_COL: float(row[FULL_R2_COL]),
            "full_rmse": float(row["rmse"]),
        }
        for j, name in enumerate(param_names):
            out[f"{name}_full"] = float(theta_oracle[j])
            out[f"{name}_ref"] = float(theta_ref[dataset][j])
        out.update(path)
        per_curve_rows.append(out)
        done_curve_keys.add((dataset, fid))
        if (len(per_curve_rows) % 25) == 0:
            print(f"[29] active-set curves {len(per_curve_rows)}/{len(all_curves)}")
            pd.DataFrame(per_curve_rows).to_csv(per_curve_progress_path, index=False)

    per_curve = pd.DataFrame(per_curve_rows)
    per_curve["k_close_to_full"] = per_curve.apply(
        _k_close_to_full, axis=1, k_max=args.k_max, delta_r2_tol=args.delta_r2_tol
    )

    summary = _dataset_summary(
        per_curve=per_curve,
        param_names=param_names,
        k_max=args.k_max,
        delta_r2_tol=args.delta_r2_tol,
    )

    per_curve_path = args.out / "per_curve_active_set.csv"
    summary_path = args.out / "dataset_summary.csv"
    txt_path = args.out / "summary.txt"
    per_curve.to_csv(per_curve_path, index=False)
    summary.to_csv(summary_path, index=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== 29 -- minimal active-set audit ===\n\n")
        f.write(f"internal_data       : {args.internal_data}\n")
        f.write(f"cross_doi_data      : {args.cross_doi_data}\n")
        f.write(f"matched_fids_csv    : {args.matched_fids_csv}\n")
        f.write(f"external_oracle_csv : {args.external_oracle_csv}\n")
        f.write(f"internal_restarts   : {args.internal_restarts}\n")
        f.write(f"k_max               : {args.k_max}\n")
        f.write(f"delta_r2_tol        : {args.delta_r2_tol:.3f}\n")
        f.write(f"fit_filter_r2       : {args.fit_filter_r2:.3f}\n\n")
        f.write("Dataset summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write("  k_close_to_full is the smallest active-set size whose R^2 is within\n")
        f.write("  delta_r2_tol of the full 9-D oracle fit. Low values imply that only a\n")
        f.write("  small number of active parameters are needed to recover the curve.\n")

    print(f"[29] wrote {full_fit_path}")
    print(f"[29] wrote {per_curve_path}")
    print(f"[29] wrote {summary_path}")
    print(f"[29] wrote {txt_path}")
    print("\n[29] dataset summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
