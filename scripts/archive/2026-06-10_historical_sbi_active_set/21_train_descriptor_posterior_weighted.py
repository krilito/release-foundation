"""
21 - Weighted Stage-2 descriptor distillation via repeated teacher pairs.

What this does:
    1. Load the trained Stage-1 q_phi teacher and the internal PLGA 181 CSV.
    2. Keep the same high-quality internal curves used by script 09/17.
    3. Sample K teacher targets theta_k ~ q_phi(.|Q_i) per kept curve.
    4. Approximate a weighted KL objective by repeating selected curves more
       times in the Stage-2 training set.
    5. Optionally append masked copies that zero the six descriptor axes absent
       from the 321 external xlsx, so weighting can be tested with or without
       the missingness-aware shadow recipe.

Why this exists:
    The Stage-2 loss is already KL-style posterior distillation from the
    full observed 181 curve into r_psi(theta | x). This script tests whether
    emphasizing the internal regimes that matter for 321 deployment
    (pure-PLGA support, fast release, short windows) helps external transfer,
    without changing posterior.py or the simulator.

Outputs:
    outputs/21_descriptor_posterior_weighted_<scheme>/posterior.pt
    outputs/21_descriptor_posterior_weighted_<scheme>/training_log.txt
    outputs/21_descriptor_posterior_weighted_<scheme>/curve_weights.csv

Expected runtime:
    ~8-15 min per run on GPU.
"""
from __future__ import annotations

import argparse
import csv
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
_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}
_MASK_COLS_321_ABSENT = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_keep_data(
    df: pd.DataFrame,
    t_grid: torch.Tensor,
    t_max_days: float,
    min_quality: str,
) -> tuple[pd.DataFrame, torch.Tensor, list[dict], list[torch.Tensor], list[torch.Tensor]]:
    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)

    q_grid_list: list[torch.Tensor | None] = []
    qualities: list[str] = []
    metas: list[dict] = []
    t_obs_list: list[torch.Tensor] = []
    q_obs_list: list[torch.Tensor] = []
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            q_grid_i, quality, meta = interpolate_to_grid(
                t_obs, q_obs, t_grid, t_max_days=t_max_days,
            )
        except ValueError:
            q_grid_list.append(None)
            qualities.append("skip")
            metas.append({"tmax": 0.0, "last_Q": 0.0, "n_obs_kept": 0})
            t_obs_list.append(t_obs)
            q_obs_list.append(q_obs)
            continue
        q_grid_list.append(q_grid_i)
        qualities.append(quality)
        metas.append(meta)
        t_obs_list.append(t_obs)
        q_obs_list.append(q_obs)

    min_rank = _QUALITY_RANK[min_quality]
    keep_mask = [
        (q in _QUALITY_RANK) and (_QUALITY_RANK[q] >= min_rank)
        for q in qualities
    ]
    keep_desc = desc_df.loc[keep_mask].reset_index(drop=True)
    q_grid = torch.stack(
        [q_grid_list[i] for i, keep in enumerate(keep_mask) if keep], dim=0
    )
    kept_metas = [metas[i] for i, keep in enumerate(keep_mask) if keep]
    kept_t_obs = [t_obs_list[i] for i, keep in enumerate(keep_mask) if keep]
    kept_q_obs = [q_obs_list[i] for i, keep in enumerate(keep_mask) if keep]
    return keep_desc, q_grid, kept_metas, kept_t_obs, kept_q_obs


def _curve_flag_table(
    keep_desc: pd.DataFrame,
    t_obs_list: list[torch.Tensor],
    q_obs_list: list[torch.Tensor],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_idx, (_, row) in enumerate(keep_desc.iterrows()):
        t_np = t_obs_list[row_idx].cpu().numpy().astype(float)
        q_np = q_obs_list[row_idx].cpu().numpy().astype(float)
        q7 = float(np.interp(7.0, t_np, q_np, left=q_np[0], right=q_np[-1]))
        t50 = (
            float(np.interp(0.5, q_np, t_np))
            if (float(np.min(q_np)) <= 0.5 <= float(np.max(q_np)))
            else float("inf")
        )
        dp_group = str(row["DP_Group"])
        rows.append({
            FID_COL: int(row[FID_COL]),
            "DP_Group": dp_group,
            "pure_plga": bool(dp_group.endswith("-PLGA")),
            "fast_regime": bool((t50 <= 2.4) or (q7 >= 0.75)),
            "short_window": bool(float(np.max(t_np)) < 14.0),
            "t50": t50,
            "q7": q7,
            "tmax_obs_d": float(np.max(t_np)),
        })
    return pd.DataFrame(rows)


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


def _repeat_counts(flag_df: pd.DataFrame, scheme: str) -> np.ndarray:
    pure = flag_df["pure_plga"].to_numpy(dtype=bool)
    fast = flag_df["fast_regime"].to_numpy(dtype=bool)
    short = flag_df["short_window"].to_numpy(dtype=bool)
    counts = np.ones(len(flag_df), dtype=np.int64)
    if scheme == "none":
        return counts
    if scheme == "plga2x":
        counts += pure.astype(np.int64)
        return counts
    if scheme == "fast2x":
        counts += fast.astype(np.int64)
        return counts
    if scheme == "short2x":
        counts += short.astype(np.int64)
        return counts
    if scheme == "plga_fast2x":
        counts += pure.astype(np.int64)
        counts += fast.astype(np.int64)
        return counts
    raise ValueError(f"unknown scheme: {scheme}")


def _weighted_training_tensors(
    x_full: torch.Tensor,
    theta_targets: torch.Tensor,
    repeat_counts: np.ndarray,
    x_masked: torch.Tensor | None,
    mask_repeats: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_idx = np.repeat(np.arange(x_full.shape[0]), repeat_counts)
    x_blocks = [x_full[base_idx]]
    theta_blocks = [theta_targets[base_idx]]
    if x_masked is not None and mask_repeats > 0:
        for _ in range(mask_repeats):
            x_blocks.append(x_masked[base_idx].clone())
            theta_blocks.append(theta_targets[base_idx].clone())
    x_train = torch.cat(x_blocks, dim=0)
    theta_train = torch.cat(theta_blocks, dim=0)
    return x_train, theta_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/21_descriptor_posterior_weighted"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument(
        "--min-quality",
        choices=("high", "medium", "low"),
        default="high",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument(
        "--scheme",
        choices=("none", "plga2x", "fast2x", "short2x", "plga_fast2x"),
        default="plga_fast2x",
    )
    parser.add_argument(
        "--mask-repeats",
        type=int,
        default=0,
        help="number of masked 321-style copies per weighted original pair block",
    )
    args = parser.parse_args()

    if args.mask_repeats < 0:
        raise ValueError("--mask-repeats must be >= 0")

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    print(f"[21] loading teacher q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device="cpu")
    print(f"[21] loading internal data from {args.data}")
    df = pd.read_csv(args.data)
    keep_desc, q_grid, kept_metas, kept_t_obs, kept_q_obs = _build_keep_data(
        df=df,
        t_grid=t_grid,
        t_max_days=float(t_cfg["end"]),
        min_quality=args.min_quality,
    )
    print(f"[21] kept {len(keep_desc)} high-quality curves")

    featurizer = FormulationFeaturizer.fit(keep_desc, continuous_cols=PLGA_CONTINUOUS_COLS)
    rp = _make_rpsi(featurizer, cfg, sim, args.device)

    print(f"[21] sampling teacher targets once (K={args.K})")
    t0 = time.time()
    theta_targets = rp.build_teacher_targets(
        q_phi=q_phi,
        Q_grid=q_grid,
        K=args.K,
        seed=args.seed,
        show_progress_bars=False,
    )
    sample_elapsed = time.time() - t0
    print(f"[21] teacher sampling finished in {sample_elapsed:.1f}s")

    flag_df = _curve_flag_table(keep_desc, kept_t_obs, kept_q_obs)
    repeat_counts = _repeat_counts(flag_df, args.scheme)
    flag_df["repeat_count"] = repeat_counts

    x_full = featurizer.transform(keep_desc)
    x_masked = None
    mask_indices: list[int] = []
    if args.mask_repeats > 0:
        x_masked, mask_indices = _masked_feature_copy(x_full, featurizer)

    x_train, theta_train = _weighted_training_tensors(
        x_full=x_full,
        theta_targets=theta_targets,
        repeat_counts=repeat_counts,
        x_masked=x_masked,
        mask_repeats=args.mask_repeats,
    )

    print(
        f"[21] scheme={args.scheme} mask_repeats={args.mask_repeats} "
        f"-> weighted curves={int(repeat_counts.sum())}, "
        f"train_x={tuple(x_train.shape)}, train_theta={tuple(theta_train.shape)}"
    )

    t_train = time.time()
    log = rp.train(
        x=x_train,
        theta_targets=theta_train,
        training_batch_size=args.batch_size,
        max_num_epochs=args.max_epochs,
        learning_rate=cfg["train"]["lr"],
        stop_after_epochs=args.stop_after_epochs,
        seed=args.seed,
        verbose=True,
    )
    train_elapsed = time.time() - t_train
    print(f"\n[21] training finished in {train_elapsed:.1f}s")
    print(f"[21] log: {log}")

    ckpt_path = args.out / "posterior.pt"
    rp.save(ckpt_path)
    flag_df.to_csv(args.out / "curve_weights.csv", index=False)

    with (args.out / "training_log.txt").open("w", encoding="utf-8") as f:
        f.write(f"seed                  : {args.seed}\n")
        f.write(f"K                     : {args.K}\n")
        f.write(f"min_quality           : {args.min_quality}\n")
        f.write(f"scheme                : {args.scheme}\n")
        f.write(f"mask_repeats          : {args.mask_repeats}\n")
        f.write(f"mask_cols             : {_MASK_COLS_321_ABSENT}\n")
        f.write(f"mask_indices          : {mask_indices}\n")
        f.write(f"device                : {args.device}\n")
        f.write(f"n_curves_kept         : {len(keep_desc)}\n")
        f.write(f"weighted_curve_count  : {int(repeat_counts.sum())}\n")
        f.write(f"n_training_rows       : {x_train.shape[0]}\n")
        f.write(f"teacher_sample_seconds: {sample_elapsed:.1f}\n")
        f.write(f"train_seconds         : {train_elapsed:.1f}\n")
        f.write(f"repeat_count_min      : {int(repeat_counts.min())}\n")
        f.write(f"repeat_count_max      : {int(repeat_counts.max())}\n")
        f.write(f"pure_plga_curves      : {int(flag_df['pure_plga'].sum())}\n")
        f.write(f"fast_curves           : {int(flag_df['fast_regime'].sum())}\n")
        f.write(f"short_curves          : {int(flag_df['short_window'].sum())}\n")
        f.write(f"fast_short_curves     : {int((flag_df['fast_regime'] & flag_df['short_window']).sum())}\n")
        f.write("\nrepeat count distribution:\n")
        for rep, n in (
            flag_df["repeat_count"].value_counts().sort_index().items()
        ):
            f.write(f"  {int(rep):2d}x : {int(n)} curves\n")
        f.write("\ntraining log:\n")
        for key, value in log.items():
            f.write(f"  {key:20s}: {value}\n")

    with (args.out / "quality_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            FID_COL,
            "DP_Group",
            "tmax_days",
            "last_Q",
            "n_obs_kept",
            "pure_plga",
            "fast_regime",
            "short_window",
            "repeat_count",
        ])
        for row_idx, (_, row) in enumerate(keep_desc.iterrows()):
            meta = kept_metas[row_idx]
            flag_row = flag_df.iloc[row_idx]
            w.writerow([
                row[FID_COL],
                row["DP_Group"],
                f"{meta['tmax']:.2f}",
                f"{meta['last_Q']:.4f}",
                meta["n_obs_kept"],
                flag_row["pure_plga"],
                flag_row["fast_regime"],
                flag_row["short_window"],
                int(flag_row["repeat_count"]),
            ])

    print(f"[21] saved checkpoint to {ckpt_path}")
    print(f"[21] saved curve weights to {args.out / 'curve_weights.csv'}")
    print(f"[21] saved log to {args.out / 'training_log.txt'}")


if __name__ == "__main__":
    main()
