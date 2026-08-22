"""
23 - Prefix-observation world-model benchmark on external 321.

What this does:
    1. Load the strongest current Stage-2 prior, `weighted_mask1`
       (`r_psi(theta | x)`), plus the Stage-1 curve posterior `q_phi`.
    2. Load the matched 259 high-quality cross-DOI 321 curves.
    3. For each prefix horizon (default: 1, 3, 7 days), keep curves with
       at least 2 prefix points and at least 2 future points.
    4. Compare update routes on the suffix-only prediction task:
         - zero_shot_rpsi : rollout from `r_psi(theta | x)` only
         - prefix_qphi    : build a completed curve by taking the zero-shot
                            prior rollout on the full t_grid and overwriting
                            the observed prefix, then infer `q_phi(theta | Q̃)`
         - fused_world    : sample from `r_psi(theta | x)` and reweight
                            by `q_phi(theta | Q_prefix)` to approximate
                            p(theta | x, Q_prefix)` when requested

Why this exists:
    We now have a Stage-2 route that transfers better to 321 under descriptor
    missingness and fast-regime shift. The next question is whether the
    latent-state + simulator stack becomes genuinely useful once we allow a
    small amount of real early-release evidence. This is the minimal
    "observe -> update belief -> rollout" benchmark.

Important assumption:
    The simulator prior is a uniform box prior. Because
        q_phi(theta | Q_prefix) ∝ p(Q_prefix | theta) p0(theta)
    and p0 is constant inside the support, reweighting r_psi samples by
    q_phi(theta | Q_prefix) approximates
        p(theta | x, Q_prefix) ∝ p(Q_prefix | theta) r_psi(theta | x)
    up to a constant normalization factor.

Outputs:
    outputs/23_prefix_world_model_benchmark/per_curve_metrics.csv
    outputs/23_prefix_world_model_benchmark/summary.csv
    outputs/23_prefix_world_model_benchmark/subgroup_summary.csv
    outputs/23_prefix_world_model_benchmark/summary.txt

Expected runtime:
    ~5-12 min on GPU for prefix_days = 1,3,7 with K_eval = 64-256 for the
    `prefix_qphi` route. The stricter `fused_world` reweighting route can be
    slower on some out-of-distribution prefixes.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    if cdf[-1] <= 0:
        return float(np.median(values))
    threshold = q * cdf[-1]
    idx = int(np.searchsorted(cdf, threshold, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _weighted_curve_quantile(
    q_pred: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> np.ndarray:
    out = np.empty(q_pred.shape[1], dtype=float)
    for t_idx in range(q_pred.shape[1]):
        out[t_idx] = _weighted_quantile(q_pred[:, t_idx], weights, q)
    return out


def _chunked_log_prob(
    q_phi: CurvePosterior,
    theta: torch.Tensor,
    q_obs: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    if chunk_size <= 0 or theta.shape[0] <= chunk_size:
        with torch.no_grad():
            return q_phi.log_prob(theta, q_obs).detach().cpu().numpy().astype(float)
    chunks: list[np.ndarray] = []
    for start in range(0, theta.shape[0], chunk_size):
        stop = min(start + chunk_size, theta.shape[0])
        with torch.no_grad():
            lp = q_phi.log_prob(theta[start:stop], q_obs).detach().cpu().numpy().astype(float)
        chunks.append(lp)
    return np.concatenate(chunks, axis=0)


def _sample_curve_posterior_relaxed(
    q_phi: CurvePosterior,
    q_obs: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    if q_obs.ndim == 1:
        q_obs = q_obs.unsqueeze(0)
    q_obs = q_obs.to(q_phi._estimator_device())  # noqa: SLF001
    # Prefix-completed curves can still be mildly off the training manifold.
    # For this diagnostic benchmark we prefer a fast approximate update over
    # spending minutes in rejection sampling.
    return q_phi._posterior.sample(  # noqa: SLF001
        (n_samples,),
        x=q_obs.squeeze(0),
        show_progress_bars=False,
        reject_outside_prior=False,
        max_sampling_time=15.0,
        return_partial_on_timeout=True,
    )


def _load_cross_doi_xlsx(
    path: Path,
    rp: DescriptorPosterior,
    matched_fids: set[int] | None,
    t_grid: torch.Tensor,
    min_quality: str,
) -> tuple[pd.DataFrame, list[torch.Tensor], list[torch.Tensor], pd.DataFrame]:
    df = pd.read_excel(path).rename(columns=_321_RENAME)
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
    if matched_fids is not None:
        df = df[df[FID_COL].astype(int).isin(matched_fids)].copy()
    for col in rp.featurizer.continuous_cols:
        if col not in df.columns:
            idx = rp.featurizer.continuous_cols.index(col)
            df[col] = float(rp.featurizer.mean[idx])
    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)

    t_obs_list: list[torch.Tensor] = []
    q_obs_list: list[torch.Tensor] = []
    keep_rows = []
    min_rank = _QUALITY_RANK[min_quality]
    for row_idx, fid in enumerate(desc_df[FID_COL].tolist()):
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            _, quality, meta = interpolate_to_grid(
                t_obs, q_obs, t_grid, t_max_days=float(t_grid.max().item()),
            )
        except ValueError:
            continue
        if _QUALITY_RANK.get(quality, -1) < min_rank:
            continue
        t_obs_list.append(t_obs)
        q_obs_list.append(q_obs)
        keep_rows.append({
            "row_idx": row_idx,
            FID_COL: int(fid),
            "quality": quality,
            "tmax_obs_d": float(meta["tmax"]),
            "last_Q": float(meta["last_Q"]),
        })
    keep_idx = [row["row_idx"] for row in keep_rows]
    keep_desc = desc_df.iloc[keep_idx].reset_index(drop=True)
    keep_meta = pd.DataFrame(keep_rows).drop(columns=["row_idx"])
    return keep_desc, t_obs_list, q_obs_list, keep_meta


def _prefix_q_grid(
    t_obs: torch.Tensor,
    q_obs: torch.Tensor,
    prefix_days: float,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor | None, np.ndarray | None]:
    t_np = t_obs.cpu().numpy().astype(float)
    q_np = q_obs.cpu().numpy().astype(float)
    prefix_mask = t_np <= prefix_days
    future_mask = t_np > prefix_days
    if prefix_mask.sum() < 2 or future_mask.sum() < 2:
        return None, None
    t_prefix = torch.tensor(t_np[prefix_mask], dtype=torch.float32)
    q_prefix = torch.tensor(q_np[prefix_mask], dtype=torch.float32)
    q_grid, _, _ = interpolate_to_grid(
        t_prefix, q_prefix, t_grid, t_max_days=float(t_grid.max().item()),
    )
    return q_grid, future_mask


def _completed_curve_grid(
    prior_grid_median: np.ndarray,
    q_grid_prefix: torch.Tensor,
    prefix_days: float,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    q_full = np.asarray(prior_grid_median, dtype=float).copy()
    prefix_mask = t_grid.cpu().numpy().astype(float) <= float(prefix_days)
    q_full[prefix_mask] = q_grid_prefix.cpu().numpy().astype(float)[prefix_mask]
    return torch.tensor(q_full, dtype=torch.float32)


def _score_suffix(
    q_obs: torch.Tensor,
    q_pred_full: np.ndarray,
    future_mask: np.ndarray,
) -> tuple[float, float]:
    q_true = q_obs.cpu().numpy().astype(float)[future_mask]
    q_pred = np.asarray(q_pred_full, dtype=float)[future_mask]
    return _r2(q_true, q_pred), float(np.mean(np.abs(q_true - q_pred)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument("--posterior", type=Path, default=Path("outputs/04_curve_posterior/posterior.pt"))
    ap.add_argument(
        "--rpsi",
        type=Path,
        default=Path("outputs/21_weighted_plga_fast2x_mask1/posterior.pt"),
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
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument(
        "--tail-flags-csv",
        type=Path,
        default=Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/23_prefix_world_model_benchmark"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K-eval", type=int, default=256)
    ap.add_argument(
        "--logprob-chunk-size",
        type=int,
        default=8,
        help="chunk size for q_phi.log_prob over theta samples; <=0 disables chunking",
    )
    ap.add_argument(
        "--update-mode",
        choices=("prefix_qphi", "fused_world", "both"),
        default="prefix_qphi",
        help="world-model update route to evaluate alongside zero-shot r_psi",
    )
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument(
        "--prefix-days",
        type=str,
        default="1,3,7",
        help="comma-separated prefix horizons in days",
    )
    ap.add_argument(
        "--max-curves",
        type=int,
        default=None,
        help="optional cap on matched curves for a faster first-pass benchmark",
    )
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    prefix_days_list = tuple(float(x.strip()) for x in args.prefix_days.split(",") if x.strip())

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    print(f"[23] loading q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device=args.device)
    progress_path.write_text(f"loaded q_phi from {args.posterior}\n", encoding="utf-8")
    print(f"[23] loading r_psi from {args.rpsi}")
    rp = DescriptorPosterior.load(args.rpsi, simulator=sim, device=args.device)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(f"loaded r_psi from {args.rpsi}\n")

    matched_fids = set(pd.read_csv(args.matched_fids_csv)["Formulation_Index"].astype(int))
    print(f"[23] loading matched cross-DOI curves from {args.cross_doi_data}")
    keep_desc, t_obs_list, q_obs_list, meta_df = _load_cross_doi_xlsx(
        args.cross_doi_data,
        rp=rp,
        matched_fids=matched_fids,
        t_grid=t_grid,
        min_quality=args.min_quality,
    )
    print(f"[23] kept {len(keep_desc)} curves at min_quality={args.min_quality}")
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(f"kept_curves={len(keep_desc)} min_quality={args.min_quality}\n")
    if args.max_curves is not None and len(keep_desc) > args.max_curves:
        rng = np.random.default_rng(args.seed)
        keep_idx = np.sort(rng.choice(len(keep_desc), size=args.max_curves, replace=False))
        keep_desc = keep_desc.iloc[keep_idx].reset_index(drop=True)
        t_obs_list = [t_obs_list[i] for i in keep_idx]
        q_obs_list = [q_obs_list[i] for i in keep_idx]
        meta_df = meta_df.iloc[keep_idx].reset_index(drop=True)
        print(f"[23] subsampled to {len(keep_desc)} curves via --max-curves")
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(f"subsampled_curves={len(keep_desc)} seed={args.seed}\n")
    x_all = rp.featurizer.transform(keep_desc)
    tail_flags = pd.read_csv(args.tail_flags_csv)[["Formulation_Index", "fast_regime", "short_window"]]

    per_curve_rows: list[dict[str, object]] = []
    overall_t0 = time.time()
    for prefix_days in prefix_days_list:
        print(f"\n[23] prefix_days={prefix_days:g}")
        eligible = 0
        for row_idx, fid in enumerate(keep_desc[FID_COL].astype(int).tolist()):
            curve_t0 = time.time()
            q_grid_prefix, future_mask = _prefix_q_grid(
                t_obs_list[row_idx], q_obs_list[row_idx], prefix_days, t_grid,
            )
            if q_grid_prefix is None or future_mask is None:
                continue
            eligible += 1
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"prefix_days={prefix_days:g} row_idx={row_idx} fid={fid} "
                    f"stage=start n_prefix={int((~future_mask).sum())} "
                    f"n_future={int(future_mask.sum())}\n"
                )

            # zero-shot rollout from r_psi(theta | x)
            torch.manual_seed(args.seed + 10000 + row_idx)
            theta_r = rp.sample(x_all[row_idx], n_samples=args.K_eval, show_progress_bars=False)
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"prefix_days={prefix_days:g} row_idx={row_idx} fid={fid} "
                    f"stage=sample elapsed_s={time.time() - curve_t0:.3f}\n"
                )
            with torch.no_grad():
                q_pred_r = sim.simulate(theta_r, t_obs_list[row_idx]).cpu().numpy()
                q_pred_r_grid = sim.simulate(theta_r, t_grid).cpu().numpy()
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"prefix_days={prefix_days:g} row_idx={row_idx} fid={fid} "
                    f"stage=simulate elapsed_s={time.time() - curve_t0:.3f}\n"
                )
            pred_r = np.median(q_pred_r, axis=0)
            r2_r, mae_r = _score_suffix(q_obs_list[row_idx], pred_r, future_mask)
            per_curve_rows.append({
                "prefix_days": prefix_days,
                "method": "zero_shot_rpsi",
                "Formulation_Index": fid,
                "n_prefix_obs": int((~future_mask).sum()),
                "n_future_obs": int(future_mask.sum()),
                "R2": float(r2_r),
                "MAE": float(mae_r),
            })

            if args.update_mode in ("prefix_qphi", "both"):
                q_completed = _completed_curve_grid(
                    np.median(q_pred_r_grid, axis=0),
                    q_grid_prefix,
                    prefix_days,
                    t_grid,
                )
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"prefix_days={prefix_days:g} row_idx={row_idx} fid={fid} "
                        f"stage=completed_curve elapsed_s={time.time() - curve_t0:.3f}\n"
                    )
                theta_q = _sample_curve_posterior_relaxed(q_phi, q_completed, n_samples=args.K_eval)
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"prefix_days={prefix_days:g} row_idx={row_idx} fid={fid} "
                        f"stage=qphi_sample elapsed_s={time.time() - curve_t0:.3f}\n"
                    )
                with torch.no_grad():
                    q_pred_q = sim.simulate(theta_q, t_obs_list[row_idx]).cpu().numpy()
                pred_q = np.median(q_pred_q, axis=0)
                r2_q, mae_q = _score_suffix(q_obs_list[row_idx], pred_q, future_mask)
                per_curve_rows.append({
                    "prefix_days": prefix_days,
                    "method": "prefix_qphi",
                    "Formulation_Index": fid,
                    "n_prefix_obs": int((~future_mask).sum()),
                    "n_future_obs": int(future_mask.sum()),
                    "R2": float(r2_q),
                    "MAE": float(mae_q),
                })

            if args.update_mode in ("fused_world", "both"):
                # world-model fusion: reweight r_psi samples by q_phi(theta | Q_prefix)
                log_w = _chunked_log_prob(
                    q_phi,
                    theta_r,
                    q_grid_prefix,
                    chunk_size=args.logprob_chunk_size,
                )
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"prefix_days={prefix_days:g} row_idx={row_idx} fid={fid} "
                        f"stage=log_prob elapsed_s={time.time() - curve_t0:.3f}\n"
                    )
                log_w = log_w - float(np.nanmax(log_w))
                w = np.exp(log_w)
                if not np.isfinite(w).all() or float(w.sum()) <= 0.0:
                    w = np.ones_like(log_w, dtype=float)
                pred_f = _weighted_curve_quantile(q_pred_r, w, 0.50)
                r2_f, mae_f = _score_suffix(q_obs_list[row_idx], pred_f, future_mask)
                per_curve_rows.append({
                    "prefix_days": prefix_days,
                    "method": "fused_world",
                    "Formulation_Index": fid,
                    "n_prefix_obs": int((~future_mask).sum()),
                    "n_future_obs": int(future_mask.sum()),
                    "R2": float(r2_f),
                    "MAE": float(mae_f),
                })
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"prefix_days={prefix_days:g} row_idx={row_idx} "
                    f"fid={fid} elapsed_s={time.time() - curve_t0:.3f}\n"
                )

        print(f"[23]   eligible curves: {eligible}")

    per_curve = pd.DataFrame(per_curve_rows)
    per_curve = per_curve.merge(tail_flags, on="Formulation_Index", how="left")
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)

    summary = (
        per_curve.groupby(["prefix_days", "method"], as_index=False)
        .agg(
            n_curves=("Formulation_Index", "count"),
            median_R2=("R2", "median"),
            mean_R2=("R2", "mean"),
            median_MAE=("MAE", "median"),
            n_R2_gt_0=("R2", lambda s: int((s > 0).sum())),
            n_R2_gt_0p3=("R2", lambda s: int((s > 0.3).sum())),
        )
        .sort_values(["prefix_days", "median_R2"], ascending=[True, False])
    )
    summary.to_csv(args.out / "summary.csv", index=False)

    subgroup_rows = []
    for (prefix_days, method), mdf in per_curve.groupby(["prefix_days", "method"], sort=False):
        for subset, sdf in {
            "overall": mdf,
            "fast": mdf[mdf["fast_regime"]],
            "short": mdf[mdf["short_window"]],
            "fast_short": mdf[mdf["fast_regime"] & mdf["short_window"]],
            "neither": mdf[~mdf["fast_regime"] & ~mdf["short_window"]],
        }.items():
            subgroup_rows.append({
                "prefix_days": prefix_days,
                "method": method,
                "subset": subset,
                "n_curves": int(len(sdf)),
                "median_R2": float(sdf["R2"].median()),
                "mean_R2": float(sdf["R2"].mean()),
                "median_MAE": float(sdf["MAE"].median()),
            })
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(args.out / "subgroup_summary.csv", index=False)

    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Prefix world-model benchmark ===\n\n")
        f.write(f"q_phi              : {args.posterior}\n")
        f.write(f"r_psi              : {args.rpsi}\n")
        f.write(f"prefix_days        : {prefix_days_list}\n")
        f.write(f"K_eval             : {args.K_eval}\n")
        f.write(f"min_quality        : {args.min_quality}\n")
        f.write(f"matched_curves     : {len(keep_desc)}\n")
        f.write(f"wallclock_min      : {(time.time() - overall_t0) / 60:.1f}\n\n")
        f.write("Overall summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSubgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"\n[23] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
