"""
27a -- ODE expressivity audit on external 321 PLGA cross-DOI curves.

What this does:
    For each matched 321 curve, fit PLGABiphasic by multi-start NLS on the
    raw observed (t, Q) points (NO grid interpolation -- we audit the ODE,
    not the interpolation). Report per-curve best RMSE / MAE / R^2 /
    theta_hat / boundary-hit flags, and aggregate by the project's standard
    subgroup splits (fast_regime / short_window).

Why this exists:
    Baseline 24 (curve-only SSL) wins all mechanism-aware partial-observation
    attempts (scripts 25, 26) on 321. Two incompatible explanations:
        (a) ODE class misspecification -- 9-D PLGABiphasic can't express
            a non-trivial fraction of 321, so any theta-conditional method
            inherits the gap.
        (b) Identifiability collapse -- prefix data doesn't pin down theta,
            so theta-prediction adds noise without adding signal.
    These two failure modes need different fixes. Inference engineering
    (better encoder, calibrated posterior, robust loss) helps (b) and does
    nothing for (a). Before committing to script 27 (amortized NPE on
    partial obs), we need to know which one is dominant.

    The internal-181 oracle (scripts/02_oracle_sweep.py) reports median R^2
    ~ 0.98, meaning the ODE class fits the internal training distribution
    well. That doesn't transfer automatically to 321 (different DOIs, drugs,
    polymer grades, manufacturing). This script runs the same diagnostic on
    321 specifically.

Pre-declared decision thresholds (these are judgment calls, not derived):
    - median R^2 >= 0.90 AND well_fit_frac >= 0.85
        -> GO 27 vanilla. ODE class is fine; bottleneck is posterior arch.
    - median R^2 in [0.70, 0.90) OR well_fit_frac in [0.60, 0.85)
        -> GO 27 with robust likelihood (NPE-RS / Tukey). Document which
        subgroups are out of model support.
    - otherwise
        -> STOP. Do not write 27. Extend model class first. Inference cannot
        recover what the model cannot express.

Outputs:
    outputs/27a_ode_expressivity_audit/per_curve.csv
    outputs/27a_ode_expressivity_audit/subgroup_summary.csv
    outputs/27a_ode_expressivity_audit/summary.txt
    outputs/27a_ode_expressivity_audit/r2_histogram.png
    outputs/27a_ode_expressivity_audit/rmse_histogram.png
    outputs/27a_ode_expressivity_audit/r2_by_subgroup.png
    outputs/27a_ode_expressivity_audit/boundary_hits.png
    outputs/27a_ode_expressivity_audit/theta_distributions.png
    outputs/27a_ode_expressivity_audit/worst_20_fits.png

Expected runtime:
    ~1.5-2 h single-core for ~250 curves * 4 restarts. Single-start (02
    convention) is ~30 min; multi-start guards against local minima which
    bias the expressivity estimate downward (a model that *can* fit the
    curve but the optimizer didn't find it would look misspecified).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

# 321 loader constants -- copied from scripts/26 (project convention: scripts
# self-contain helpers rather than depend on a utils module).
FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
DP_GROUP_COL = "DP_Group"
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
_TAIL_FLAGS_PATH = Path(
    "outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv"
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class CurveRecord:
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray
    quality: str
    t_max: float
    n_obs: int


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
    t_grid_max_days: float,
    min_quality: str,
) -> tuple[list[CurveRecord], pd.DataFrame]:
    """Load matched 321 curves at min_quality, returning raw (t_obs, q_obs).

    Quality classification uses the same interpolate_to_grid path as
    scripts/26 so subset membership is comparable across scripts. We do NOT
    use the interpolated grid for fitting -- expressivity is a property of
    the ODE on real observed points, not on a resampled approximation.
    """
    matched_fids = set(
        pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int)
    )
    df = pd.read_excel(xlsx_path).rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df[DP_GROUP_COL] = "UNK-PLGA"
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
    records: list[CurveRecord] = []
    min_rank = _QUALITY_RANK[min_quality]
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
        records.append(CurveRecord(
            fid=int(fid),
            t_obs=t_obs,
            q_obs=q_obs,
            quality=quality,
            t_max=float(t_obs.max()),
            n_obs=int(len(t_obs)),
        ))

    if _TAIL_FLAGS_PATH.exists():
        tail_flags = pd.read_csv(_TAIL_FLAGS_PATH)[
            ["Formulation_Index", "fast_regime", "short_window"]
        ]
    else:
        # graceful fallback: everything becomes "neither"
        tail_flags = pd.DataFrame(
            columns=["Formulation_Index", "fast_regime", "short_window"]
        )
    return records, tail_flags


def _fit_one_multistart(
    sim: PLGABiphasic,
    t_obs: np.ndarray,
    q_obs: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    n_restarts: int,
    seed: int,
) -> dict:
    """Multi-start NLS via scipy least_squares(method='trf').

    Restart 0 is mid-prior (matches scripts/02 single-start convention).
    Restarts 1..n-1 are uniform draws from prior support.

    Returns the best (lowest SSres) restart and a basin count (number of
    distinct optimizer endpoints, normalized in [0, 1] per dim with L_inf
    < 0.02 as the same-basin threshold). Basin count > 1 is a multimodality
    signal for that curve.
    """
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lows + highs)

    best_theta: np.ndarray | None = None
    best_ss = np.inf
    best_restart_idx = -1
    converged_thetas: list[np.ndarray] = []

    def residual(theta_flat: np.ndarray) -> np.ndarray:
        return sim.simulate_numpy(theta_flat, t_obs) - q_obs

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
            r = residual(result.x)
            ss = float(np.sum(r ** 2))
            if not np.isfinite(ss):
                continue
            converged_thetas.append(result.x.copy())
            if ss < best_ss:
                best_ss = ss
                best_theta = result.x.copy()
                best_restart_idx = k
        except Exception:
            continue

    if best_theta is None:
        return {
            "best_theta": np.full(len(lows), np.nan),
            "rmse": np.nan, "mae": np.nan, "r2": np.nan, "ss_res": np.nan,
            "n_converged": 0, "best_restart": -1, "n_basins": 0,
        }

    pred = sim.simulate_numpy(best_theta, t_obs)
    res = q_obs - pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    r2 = _r2(q_obs, pred)

    # basin clustering (cheap, K <= n_restarts)
    width = highs - lows
    normalized = [(t - lows) / width for t in converged_thetas]
    basins: list[np.ndarray] = []
    for nt in normalized:
        if not any(np.max(np.abs(nt - b)) < 0.02 for b in basins):
            basins.append(nt)

    return {
        "best_theta": best_theta,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ss_res": best_ss,
        "n_converged": len(converged_thetas),
        "best_restart": best_restart_idx,
        "n_basins": len(basins),
    }


def _boundary_hits(
    theta_hat: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    param_names: list[str],
    threshold: float = 0.01,
) -> tuple[int, list[str]]:
    """Count params pinned within `threshold` fraction of prior width.

    A boundary hit means the data wanted theta outside the prior; this is
    a clean misspec signature if it concentrates on a few params across
    many curves.
    """
    width = highs - lows
    hits: list[str] = []
    for k, name in enumerate(param_names):
        v = theta_hat[k]
        if not np.isfinite(v):
            continue
        if (v - lows[k]) / width[k] < threshold:
            hits.append(f"{name}_lo")
        elif (highs[k] - v) / width[k] < threshold:
            hits.append(f"{name}_hi")
    return len(hits), hits


def _classify_subgroups(per_curve: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall": per_curve,
        "neither": per_curve[~per_curve["fast_regime"] & ~per_curve["short_window"]],
        "fast": per_curve[per_curve["fast_regime"]],
        "short": per_curve[per_curve["short_window"]],
        "fast_short": per_curve[per_curve["fast_regime"] & per_curve["short_window"]],
    }


def _bin_rmse(rmse: float) -> str:
    if not np.isfinite(rmse):
        return "failed"
    if rmse < 0.05:
        return "<0.05"
    if rmse < 0.10:
        return "0.05-0.10"
    if rmse < 0.20:
        return "0.10-0.20"
    return ">0.20"


def _verdict(per_curve: pd.DataFrame) -> str:
    median_r2 = float(per_curve["r2"].median())
    well_fit = float((per_curve["r2"] > 0.9).mean())
    decent = float((per_curve["r2"] > 0.7).mean())
    broken = float((per_curve["r2"] < 0.5).mean())

    if well_fit >= 0.85 and median_r2 >= 0.90:
        decision = (
            "GO 27 vanilla NPE. ODE class expresses >=85% of 321 well. "
            "Residual gap is information-theoretic (identifiability) or "
            "observation noise. Bottleneck IS posterior architecture."
        )
    elif well_fit >= 0.60 and median_r2 >= 0.70:
        decision = (
            "GO 27 with robust likelihood. ODE class is partially adequate "
            "(60-85% well-fit). Use NPE-RS or Tukey-loss SBI. Document which "
            "subgroups (likely fast/short tails) are out of model support; "
            "exclude them from headline metrics."
        )
    else:
        decision = (
            "STOP. Do not write 27. ODE class is the bottleneck on 321 "
            "(<60% well-fit OR median R^2 < 0.70). Inference engineering "
            "cannot recover what the model cannot express. Next step: extend "
            "model class (triphasic? secondary-burst term? mechanism mixture?). "
            "Use worst_20_fits.png to decide which extension matters."
        )

    lines = [
        f"  overall median full-curve R^2 : {median_r2:.4f}",
        f"  frac well-fit (R^2 > 0.9)     : {well_fit:.3f}",
        f"  frac decent   (R^2 > 0.7)     : {decent:.3f}",
        f"  frac broken   (R^2 < 0.5)     : {broken:.3f}",
        "",
    ]
    for name, sdf in _classify_subgroups(per_curve).items():
        if len(sdf) == 0:
            continue
        lines.append(
            f"  {name:10s} (n={len(sdf):3d}) median R^2 = {sdf['r2'].median():+.4f}, "
            f"well-fit = {(sdf['r2'] > 0.9).mean():.3f}, "
            f"broken = {(sdf['r2'] < 0.5).mean():.3f}"
        )
    lines.append("")
    lines.append(f"DECISION: {decision}")
    return "\n".join(lines)


# -- plots ------------------------------------------------------------------

def _plot_r2_histogram(per_curve: pd.DataFrame, out_path: Path) -> None:
    r2 = per_curve["r2"].to_numpy()
    r2_clipped = np.clip(r2, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r2_clipped, bins=40, edgecolor="black")
    median = float(np.median(r2))
    ax.axvline(median, color="red", linestyle="--", label=f"full-curve median {median:.3f}")
    ax.axvline(0.349, color="orange", linestyle=":", label="baseline 24 suffix 3d = 0.349")
    ax.set_xlabel("full-curve oracle R^2  (clipped at [-1, 1])")
    ax.set_ylabel("count")
    ax.set_title(f"321 full-curve R^2 distribution (n={len(per_curve)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rmse_histogram(per_curve: pd.DataFrame, out_path: Path) -> None:
    rmse = per_curve["rmse"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(rmse, 0.0, 0.5), bins=40, edgecolor="black")
    for thr, color, lbl in [(0.05, "green", "0.05"), (0.10, "orange", "0.10"), (0.20, "red", "0.20")]:
        ax.axvline(thr, color=color, linestyle="--", label=lbl)
    ax.set_xlabel("full-curve fit RMSE (clipped at 0.5)")
    ax.set_ylabel("count")
    ax.set_title("321 full-curve RMSE distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_r2_by_subgroup(per_curve: pd.DataFrame, out_path: Path) -> None:
    groups = _classify_subgroups(per_curve)
    names = ["overall", "neither", "fast", "short", "fast_short"]
    data = [np.clip(groups[n]["r2"].to_numpy(), -1.0, 1.0) for n in names]
    labels = [f"{n}\n(n={len(groups[n])})" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("full-curve R^2  (clipped at [-1, 1])")
    ax.set_title("Full-curve R^2 by subgroup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_boundary_hits(boundary_counts: pd.Series, out_path: Path) -> None:
    if len(boundary_counts) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no boundary hits", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    boundary_counts.head(15).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("# curves pinned to this bound")
    ax.set_title("Boundary-hit frequency (top 15; suffix _lo/_hi)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_theta_distributions(
    per_curve: pd.DataFrame,
    param_names: list[str],
    lows: np.ndarray,
    highs: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for k, (name, ax) in enumerate(zip(param_names, axes.flat, strict=True)):
        vals = per_curve[name].dropna().to_numpy()
        ax.hist(vals, bins=30, edgecolor="black")
        ax.axvline(lows[k], color="red", linestyle="--", linewidth=1)
        ax.axvline(highs[k], color="red", linestyle="--", linewidth=1)
        ax.set_title(name)
    fig.suptitle(
        "theta_hat distributions over 321 (red = prior bounds; "
        "pile-up at edges = misspec)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_worst_20(
    records: list[CurveRecord],
    per_curve: pd.DataFrame,
    sim: PLGABiphasic,
    out_path: Path,
) -> None:
    rec_by_fid = {r.fid: r for r in records}
    sorted_df = (
        per_curve.dropna(subset=["rmse"])
        .sort_values("rmse", ascending=False)
        .head(20)
    )
    fig, axes = plt.subplots(4, 5, figsize=(17, 12), sharey=True)
    for ax, (_, row) in zip(axes.flat, sorted_df.iterrows(), strict=False):
        rec = rec_by_fid[int(row["fid"])]
        theta = np.array([row[n] for n in sim.param_names], dtype=float)
        t_dense = np.linspace(0.0, max(float(rec.t_obs.max()), 1.0), 200)
        try:
            q_dense = sim.simulate_numpy(theta, t_dense)
        except Exception:
            q_dense = np.full_like(t_dense, np.nan)
        ax.plot(rec.t_obs, rec.q_obs, "o", color="black", markersize=4, label="obs")
        ax.plot(t_dense, q_dense, "-", color="tab:red", linewidth=1.5, label="fit")
        subset = ("F" if row["fast_regime"] else "-") + ("S" if row["short_window"] else "-")
        ax.set_title(
            f"fid={int(row['fid'])} R^2={row['r2']:+.2f} RMSE={row['rmse']:.3f} [{subset}]",
            fontsize=9,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("t (days)")
    for ax in axes.flat[len(sorted_df):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Worst 20 full-curve fits by RMSE  (F=fast_regime, S=short_window)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
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
    ap.add_argument("--out", type=Path, default=Path("outputs/27a_ode_expressivity_audit"))
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None,
                    help="cap n curves (debug); default uses all matched")
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid_max = float(t_cfg["end"])

    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    prior = sim.prior()
    lows = prior.base_dist.low.numpy()
    highs = prior.base_dist.high.numpy()

    print(f"[27a] loading matched 321 from {args.cross_doi_data}")
    records, tail_flags = _load_external_records(
        xlsx_path=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid_max_days=t_grid_max,
        min_quality=args.min_quality,
    )
    print(f"[27a] kept {len(records)} curves at min_quality={args.min_quality}")

    if args.max_curves is not None:
        records = records[: args.max_curves]
        print(f"[27a] truncated to {len(records)} (debug)")

    print(f"[27a] multi-start NLS, n_restarts={args.n_restarts}")
    rows: list[dict] = []
    for i, rec in enumerate(records):
        result = _fit_one_multistart(
            sim=sim,
            t_obs=rec.t_obs,
            q_obs=rec.q_obs,
            lows=lows,
            highs=highs,
            n_restarts=args.n_restarts,
            seed=args.seed * 10_000 + rec.fid,
        )
        n_hits, hit_names = _boundary_hits(
            result["best_theta"], lows, highs, sim.param_names, threshold=0.01
        )
        row: dict = {
            "fid": rec.fid,
            "n_obs": rec.n_obs,
            "t_max": rec.t_max,
            "quality": rec.quality,
            "rmse": result["rmse"],
            "mae": result["mae"],
            "r2": result["r2"],
            "ss_res": result["ss_res"],
            "best_restart": result["best_restart"],
            "n_converged": result["n_converged"],
            "n_basins": result["n_basins"],
            "boundary_hits": n_hits,
            "boundary_params": ",".join(hit_names),
        }
        for k, name in enumerate(sim.param_names):
            row[name] = float(result["best_theta"][k])
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"[27a] {i+1}/{len(records)}  elapsed {time.time()-t0:.1f}s")
    print(f"[27a] fits done in {(time.time()-t0)/60:.1f} min")

    per_curve = pd.DataFrame(rows)
    flags = tail_flags.rename(columns={"Formulation_Index": "fid"})
    if len(flags):
        flags["fid"] = flags["fid"].astype(int)
    per_curve = per_curve.merge(flags, on="fid", how="left")
    per_curve["fast_regime"] = per_curve["fast_regime"].fillna(False).astype(bool)
    per_curve["short_window"] = per_curve["short_window"].fillna(False).astype(bool)
    per_curve["rmse_bin"] = per_curve["rmse"].apply(_bin_rmse)

    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    print(f"[27a] wrote {args.out / 'per_curve.csv'}")

    summary_rows: list[dict] = []
    for name, sdf in _classify_subgroups(per_curve).items():
        summary_rows.append({
            "subset": name,
            "n_curves": int(len(sdf)),
            "median_r2": float(sdf["r2"].median()) if len(sdf) else float("nan"),
            "mean_r2": float(sdf["r2"].mean()) if len(sdf) else float("nan"),
            "median_rmse": float(sdf["rmse"].median()) if len(sdf) else float("nan"),
            "median_mae": float(sdf["mae"].median()) if len(sdf) else float("nan"),
            "frac_well_fit": float((sdf["r2"] > 0.9).mean()) if len(sdf) else float("nan"),
            "frac_decent": float((sdf["r2"] > 0.7).mean()) if len(sdf) else float("nan"),
            "frac_broken": float((sdf["r2"] < 0.5).mean()) if len(sdf) else float("nan"),
        })
    subgroup_summary = pd.DataFrame(summary_rows)
    subgroup_summary.to_csv(args.out / "subgroup_summary.csv", index=False)

    bin_order = ["<0.05", "0.05-0.10", "0.10-0.20", ">0.20", "failed"]
    bin_counts = per_curve["rmse_bin"].value_counts().reindex(bin_order, fill_value=0)

    all_hits: list[str] = []
    for hits_str in per_curve["boundary_params"]:
        if isinstance(hits_str, str) and hits_str:
            all_hits.extend(hits_str.split(","))
    boundary_counts = pd.Series(all_hits).value_counts() if all_hits else pd.Series(dtype=int)

    _plot_r2_histogram(per_curve, args.out / "r2_histogram.png")
    _plot_rmse_histogram(per_curve, args.out / "rmse_histogram.png")
    _plot_r2_by_subgroup(per_curve, args.out / "r2_by_subgroup.png")
    _plot_boundary_hits(boundary_counts, args.out / "boundary_hits.png")
    _plot_theta_distributions(per_curve, sim.param_names, lows, highs,
                              args.out / "theta_distributions.png")
    _plot_worst_20(records, per_curve, sim, args.out / "worst_20_fits.png")

    n = len(per_curve)
    if len(boundary_counts):
        boundary_lines = [
            f"  {nm:20s} : {int(c):4d}  ({c/n*100:5.1f}%)"
            for nm, c in boundary_counts.head(10).items()
        ]
    else:
        boundary_lines = ["  (no boundary hits)"]

    summary_text = "\n".join([
        "=== 27a -- ODE expressivity audit on 321 PLGA cross-DOI curves ===",
        "",
        f"  data            : {args.cross_doi_data}",
        f"  matched_fids    : {args.matched_fids_csv}",
        f"  min_quality     : {args.min_quality}",
        f"  n_restarts      : {args.n_restarts}",
        f"  n_curves        : {n}",
        f"  wallclock       : {(time.time()-t0)/60:.1f} min",
        "",
        "RMSE bins:",
        *[f"  {b:12s} : {int(c):4d}  ({c/n*100:5.1f}%)" for b, c in bin_counts.items()],
        "",
        "Subgroup summary:",
        subgroup_summary.to_string(index=False),
        "",
        f"Top-10 boundary-hit params (out of {n} curves):",
        *boundary_lines,
        "",
        "=== VERDICT ===",
        _verdict(per_curve),
        "",
        "Plots:",
        "  r2_histogram.png       : R^2 distribution vs baseline 24 reference",
        "  rmse_histogram.png     : RMSE bins visualized",
        "  r2_by_subgroup.png     : box per subgroup (overall/neither/fast/short/fast_short)",
        "  boundary_hits.png      : which params get pinned to prior edges (misspec signature)",
        "  theta_distributions.png: per-param theta_hat with prior overlay",
        "  worst_20_fits.png      : 4x5 grid of worst RMSE -- inspect shapes for extension hypotheses",
    ])
    (args.out / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
