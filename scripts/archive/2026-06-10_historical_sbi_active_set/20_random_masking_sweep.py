"""
20 - Sweep random block-masking rates for Stage-2 missingness-aware SBI.

What this does:
    1. Load q_phi and the internal PLGA 181 dataset.
    2. Keep the same 133 high-quality curves used in sprint1.
    3. Fit the full 13-descriptor featurizer once and sample K teacher
       targets once for all kept curves.
    4. For each mask probability p in a user-provided sweep:
         - duplicate each training curve once
         - on the duplicate, zero the six 321-absent descriptor dims with
           probability p (blockwise, not per-dimension)
         - train a fresh Stage-2 r_psi
         - evaluate directly on external 321 with the usual masked input
    5. Write per-probability cross-DOI summaries and subgroup metrics.

Why this exists:
    The deterministic masked-duplicate shadow model improved the external
    median R^2 but introduced a few catastrophic outliers. This sweep tests
    whether softer exposure to 321-style missingness can keep the median gain
    while recovering mean stability.

Outputs:
    outputs/20_random_masking_sweep/summary.csv
    outputs/20_random_masking_sweep/subgroup_summary.csv
    outputs/20_random_masking_sweep/p*/cross_doi_per_curve_metrics.csv
    outputs/20_random_masking_sweep/summary.txt

Expected runtime:
    ~25-45 min on GPU for the default 4-point sweep.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import PLGA_CONTINUOUS_COLS, FormulationFeaturizer, MLPFormulationEncoder  # noqa: E402
from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
DP_GROUP_COL = "DP_Group"
T_MAX_DAYS_CROSS_DOI = 90.0
_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}
_MASK_COLS_321_ABSENT = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
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


def _build_internal_keep(
    df: pd.DataFrame,
    t_grid: torch.Tensor,
    t_max_days: float,
    min_quality: str,
) -> tuple[pd.DataFrame, torch.Tensor]:
    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)
    q_grid_list: list[torch.Tensor | None] = []
    qualities: list[str] = []
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            q_grid_i, quality, _ = interpolate_to_grid(
                t_obs, q_obs, t_grid, t_max_days=t_max_days,
            )
        except ValueError:
            q_grid_list.append(None)
            qualities.append("skip")
            continue
        q_grid_list.append(q_grid_i)
        qualities.append(quality)

    min_rank = _QUALITY_RANK[min_quality]
    keep_mask = [
        (q in _QUALITY_RANK) and (_QUALITY_RANK[q] >= min_rank)
        for q in qualities
    ]
    keep_desc = desc_df.loc[keep_mask].reset_index(drop=True)
    q_grid = torch.stack(
        [q_grid_list[i] for i, keep in enumerate(keep_mask) if keep], dim=0
    )
    return keep_desc, q_grid


def _load_cross_doi_xlsx(
    path: Path,
    mean_by_col: dict[str, float],
    matched_fids: set[int] | None = None,
) -> tuple[pd.DataFrame, list[torch.Tensor], list[torch.Tensor]]:
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
    df = df[df[TIME_COL] <= T_MAX_DAYS_CROSS_DOI].copy()
    for col in _MASK_COLS_321_ABSENT:
        df[col] = float(mean_by_col[col])
    counts = df.groupby(FID_COL).size()
    df = df[df[FID_COL].isin(counts[counts >= 2].index)].copy()
    if matched_fids is not None:
        df = df[df[FID_COL].astype(int).isin(matched_fids)].copy()

    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)
    t_obs_list: list[torch.Tensor] = []
    q_obs_list: list[torch.Tensor] = []
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs_list.append(torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32))
        q_obs_list.append(torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0))
    return desc_df, t_obs_list, q_obs_list


def _make_rpsi(
    feat: FormulationFeaturizer,
    cfg: dict,
    sim: PLGABiphasic,
    device: str,
) -> DescriptorPosterior:
    encoder = MLPFormulationEncoder(
        input_dim=feat.input_dim,
        embed_dim=cfg["encoder"]["embed_dim"],
        hidden_dims=tuple(cfg["encoder"]["hidden_dims"]),
        dropout=cfg["encoder"]["dropout"],
    )
    return DescriptorPosterior(
        simulator=sim,
        featurizer=feat,
        encoder=encoder,
        flow=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        device=device,
    )


def _masked_feature_copy(
    x_full: torch.Tensor,
    featurizer: FormulationFeaturizer,
) -> tuple[torch.Tensor, list[int]]:
    idxs = [featurizer.continuous_cols.index(col) for col in _MASK_COLS_321_ABSENT]
    x_masked = x_full.clone()
    x_masked[:, idxs] = 0.0
    return x_masked, idxs


def _stochastic_augmented_x(
    x_full: torch.Tensor,
    mask_indices: list[int],
    mask_prob: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=x_full.device if x_full.is_cuda else "cpu")
    gen.manual_seed(seed)
    mask_flags = torch.rand(x_full.shape[0], generator=gen, device=x_full.device) < mask_prob
    x_twin = x_full.clone()
    if mask_flags.any():
        row_idx = torch.where(mask_flags)[0]
        x_twin[row_idx[:, None], torch.tensor(mask_indices, device=x_full.device)] = 0.0
    x_aug = torch.cat([x_full, x_twin], dim=0)
    return x_aug, mask_flags.cpu()


def _eval_curve(
    rp: DescriptorPosterior,
    sim: PLGABiphasic,
    x_i: torch.Tensor,
    t_obs: torch.Tensor,
    q_obs: torch.Tensor,
    k_eval: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    torch.manual_seed(seed)
    theta = rp.sample(x_i, n_samples=k_eval, show_progress_bars=False)
    with torch.no_grad():
        q_pred = sim.simulate(theta, t_obs)
    q_pred_np = q_pred.cpu().numpy()
    q_obs_np = q_obs.cpu().numpy()
    median = np.median(q_pred_np, axis=0)
    return {
        "median": median,
        "R2": _r2(q_obs_np, median),
        "MAE": float(np.mean(np.abs(q_obs_np - median))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
    )
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
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
        "--tail-flags-csv",
        type=Path,
        default=Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv"),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
        help="Restrict sweep scoring to the 259 high-quality external curves.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/20_random_masking_sweep"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K-train", type=int, default=64)
    ap.add_argument("--K-eval", type=int, default=256)
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--stop-after-epochs", type=int, default=20)
    ap.add_argument(
        "--mask-probs",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="comma-separated block-mask probabilities for the duplicated copy",
    )
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    mask_probs = tuple(float(x.strip()) for x in str(args.mask_probs).split(",") if x.strip())
    if any((p < 0.0 or p > 1.0) for p in mask_probs):
        raise ValueError("--mask-probs must be within [0, 1]")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    print(f"[20] loading q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device="cpu")

    print(f"[20] loading internal data from {args.data}")
    internal_df = pd.read_csv(args.data)
    keep_desc, q_grid = _build_internal_keep(
        df=internal_df,
        t_grid=t_grid,
        t_max_days=float(t_cfg["end"]),
        min_quality=args.min_quality,
    )
    print(f"[20] kept {len(keep_desc)} high-quality curves")

    _seed_everything(args.seed)
    feat = FormulationFeaturizer.fit(keep_desc, continuous_cols=PLGA_CONTINUOUS_COLS)
    mean_by_col = dict(zip(feat.continuous_cols, feat.mean.tolist(), strict=True))
    matched_fids = set(pd.read_csv(args.matched_fids_csv)["Formulation_Index"].astype(int))
    cross_desc, cross_t_obs, cross_q_obs = _load_cross_doi_xlsx(
        args.cross_doi_data,
        mean_by_col,
        matched_fids=matched_fids,
    )
    x_cross_full = feat.transform(cross_desc)
    x_cross_masked, mask_indices = _masked_feature_copy(x_cross_full, feat)
    print(f"[20] cross-DOI curves: {len(cross_desc)}  mask_indices={mask_indices}")

    print(f"[20] sampling teacher targets once (K={args.K_train})")
    _seed_everything(args.seed)
    temp_rp = _make_rpsi(feat, cfg, sim, device="cpu")
    t0 = time.time()
    theta_targets = temp_rp.build_teacher_targets(
        q_phi=q_phi,
        Q_grid=q_grid,
        K=args.K_train,
        seed=args.seed,
        show_progress_bars=False,
    )
    print(f"[20] teacher sampling finished in {time.time() - t0:.1f}s")

    x_full = feat.transform(keep_desc)
    tail_flags = pd.read_csv(args.tail_flags_csv)[["Formulation_Index", "fast_regime", "short_window"]].copy()

    summary_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    all_per_curve: list[pd.DataFrame] = []
    overall_t0 = time.time()
    for idx, mask_prob in enumerate(mask_probs):
        prob_label = f"p{int(round(mask_prob * 100)):03d}"
        out_dir = args.out / prob_label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[20] sweep {idx + 1}/{len(mask_probs)}  mask_prob={mask_prob:.2f}")

        x_aug, mask_flags = _stochastic_augmented_x(
            x_full=x_full.to(args.device),
            mask_indices=mask_indices,
            mask_prob=mask_prob,
            seed=args.seed + 1000 + idx,
        )
        theta_aug = torch.cat([theta_targets, theta_targets], dim=0)
        _seed_everything(args.seed)
        rp = _make_rpsi(feat, cfg, sim, args.device)
        t_train = time.time()
        log = rp.train(
            x=x_aug,
            theta_targets=theta_aug,
            training_batch_size=args.batch_size,
            max_num_epochs=args.max_epochs,
            learning_rate=cfg["train"]["lr"],
            stop_after_epochs=args.stop_after_epochs,
            seed=args.seed,
            verbose=False,
        )
        print(
            f"[20]   train {time.time() - t_train:.1f}s  "
            f"masked_twins={int(mask_flags.sum())}/{len(mask_flags)}"
        )
        ckpt_path = out_dir / "posterior.pt"
        rp.save(ckpt_path)
        train_log_payload = {
            "mask_prob": mask_prob,
            "seed": args.seed,
            "device": args.device,
            "mask_indices": mask_indices,
            "mask_cols": list(_MASK_COLS_321_ABSENT),
            "masked_twins": int(mask_flags.sum()),
            "n_curves_kept": int(len(keep_desc)),
            "teacher_checkpoint": str(args.posterior),
            "teacher_checkpoint_mtime": args.posterior.stat().st_mtime,
            "internal_data": str(args.data),
            "internal_data_mtime": args.data.stat().st_mtime,
            "cross_doi_data": str(args.cross_doi_data),
            "cross_doi_data_mtime": args.cross_doi_data.stat().st_mtime,
            "training_log": log,
        }
        (out_dir / "training_log.json").write_text(
            json.dumps(train_log_payload, indent=2),
            encoding="utf-8",
        )

        per_curve_rows = []
        pred_dir = out_dir / "cross_doi_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for j, fid in enumerate(cross_desc[FID_COL].astype(int).tolist()):
            res = _eval_curve(
                rp=rp,
                sim=sim,
                x_i=x_cross_masked[j],
                t_obs=cross_t_obs[j],
                q_obs=cross_q_obs[j],
                k_eval=args.K_eval,
                seed=args.seed + 20000 * (idx + 1) + j,
            )
            np.savez(
                pred_dir / f"{fid}.npz",
                t=cross_t_obs[j].cpu().numpy(),
                Q_obs=cross_q_obs[j].cpu().numpy(),
                Q_pred_median=np.asarray(res["median"], dtype=float),
            )
            per_curve_rows.append({
                "mask_prob": mask_prob,
                "Formulation_Index": fid,
                "n_obs": int(cross_t_obs[j].numel()),
                "R2": float(res["R2"]),
                "MAE": float(res["MAE"]),
            })

        per_curve = pd.DataFrame(per_curve_rows)
        per_curve.to_csv(out_dir / "cross_doi_per_curve_metrics.csv", index=False)
        merged = per_curve.merge(tail_flags, on="Formulation_Index", how="left")
        all_per_curve.append(merged.copy())

        summary_rows.append({
            "mask_prob": mask_prob,
            "n_curves": int(len(per_curve)),
            "masked_twins": int(mask_flags.sum()),
            "median_R2": float(per_curve["R2"].median()),
            "mean_R2": float(per_curve["R2"].mean()),
            "median_MAE": float(per_curve["MAE"].median()),
            "n_R2_gt_0": int((per_curve["R2"] > 0).sum()),
            "n_R2_gt_0p3": int((per_curve["R2"] > 0.3).sum()),
            "final_train_loss": log.get("final_train_loss"),
            "final_val_loss": log.get("final_val_loss"),
        })
        for subset, sdf in {
            "overall": merged,
            "fast": merged[merged["fast_regime"]],
            "short": merged[merged["short_window"]],
            "fast_short": merged[merged["fast_regime"] & merged["short_window"]],
            "neither": merged[~merged["fast_regime"] & ~merged["short_window"]],
        }.items():
            subgroup_rows.append({
                "mask_prob": mask_prob,
                "subset": subset,
                "n_curves": int(len(sdf)),
                "median_R2": float(sdf["R2"].median()),
                "mean_R2": float(sdf["R2"].mean()),
                "median_MAE": float(sdf["MAE"].median()),
            })

    summary = pd.DataFrame(summary_rows).sort_values("mask_prob")
    subgroup = pd.DataFrame(subgroup_rows).sort_values(["mask_prob", "subset"])
    summary.to_csv(args.out / "summary.csv", index=False)
    subgroup.to_csv(args.out / "subgroup_summary.csv", index=False)
    pd.concat(all_per_curve, ignore_index=True).to_csv(args.out / "per_curve_metrics_all.csv", index=False)

    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Random masking sweep ===\n\n")
        f.write(f"mask_probs         : {mask_probs}\n")
        f.write(f"n_curves_kept      : {len(keep_desc)}\n")
        f.write(f"cross_doi_curves   : {len(cross_desc)}\n")
        f.write(f"total_wallclock_min: {(time.time() - overall_t0) / 60:.1f}\n\n")
        f.write("Overall summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSubgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"\n[20] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
