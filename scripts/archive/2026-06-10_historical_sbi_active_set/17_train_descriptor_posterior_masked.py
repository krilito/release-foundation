"""
17 - Train a missingness-aware Stage-2 r_psi shadow model.

What this does:
    1. Reuse the standard Stage-2 pipeline from script 09:
         - load q_phi teacher
         - keep high-quality internal curves
         - fit the full 13-descriptor featurizer
         - sample K teacher targets per curve
    2. Build a second "masked" copy of each formulation feature vector by
       zeroing the z-scored dimensions that are absent from the 321 xlsx:
         CL Ratio, Drug_Tm, Drug_Pka, Drug_NHA, SA-V, SE
    3. Concatenate original + masked copies and distill r_psi on both.

Why this exists:
    A plain 7-feature amputation underperformed the current full SBI on
    cross-DOI 321, but direct tabular models clearly suffer when they learn
    to depend on descriptors absent at deployment. This shadow experiment
    tests the minimal missingness-aware variant without rewriting the main
    training loop or changing the simulator.

Outputs:
    outputs/17_descriptor_posterior_masked/posterior.pt
    outputs/17_descriptor_posterior_masked/training_log.txt
    outputs/17_descriptor_posterior_masked/quality_distribution.csv

Expected runtime:
    ~6-10 min on GPU, dominated by teacher target sampling.
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
) -> tuple[pd.DataFrame, torch.Tensor, list[str], list[dict]]:
    fid_col, t_col, q_col = "Experimental_index", "Time", "Release"
    desc_df = (
        df.drop_duplicates(fid_col)
        .sort_values(fid_col)
        .reset_index(drop=True)
    )

    q_grid_list = []
    qualities = []
    metas = []
    for fid in desc_df[fid_col].tolist():
        g = df[df[fid_col] == fid].sort_values(t_col)
        t_obs = torch.tensor(g[t_col].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[q_col].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            q_grid_i, quality, meta = interpolate_to_grid(
                t_obs, q_obs, t_grid, t_max_days=t_max_days,
            )
        except ValueError:
            q_grid_list.append(None)
            qualities.append("skip")
            metas.append({"tmax": 0.0, "last_Q": 0.0, "n_obs_kept": 0})
            continue
        q_grid_list.append(q_grid_i)
        qualities.append(quality)
        metas.append(meta)

    min_rank = _QUALITY_RANK[min_quality]
    keep_mask = [
        (q in _QUALITY_RANK) and (_QUALITY_RANK[q] >= min_rank)
        for q in qualities
    ]
    keep_desc = desc_df.loc[keep_mask].reset_index(drop=True)
    q_grid = torch.stack(
        [q_grid_list[i] for i, keep in enumerate(keep_mask) if keep], dim=0
    )
    kept_qualities = [qualities[i] for i, keep in enumerate(keep_mask) if keep]
    kept_metas = [metas[i] for i, keep in enumerate(keep_mask) if keep]
    return keep_desc, q_grid, kept_qualities, kept_metas


def _masked_feature_copy(
    x_full: torch.Tensor,
    featurizer: FormulationFeaturizer,
    mask_cols: tuple[str, ...],
) -> tuple[torch.Tensor, list[int]]:
    idxs = [featurizer.continuous_cols.index(col) for col in mask_cols]
    x_masked = x_full.clone()
    x_masked[:, idxs] = 0.0
    return x_masked, idxs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
        help="path to Stage-1 q_phi checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/17_descriptor_posterior_masked"),
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
        "--mask-repeats",
        type=int,
        default=1,
        help="number of fully masked copies appended per curve",
    )
    args = parser.parse_args()

    if args.mask_repeats < 1:
        raise ValueError("--mask-repeats must be >= 1")

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

    print(f"[17] loading teacher q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device="cpu")
    print(f"[17] teacher training log: {q_phi._training_log}")

    print(f"[17] loading PLGA dataset from {args.data}")
    df = pd.read_csv(args.data)
    keep_desc, q_grid, _, kept_metas = _build_keep_data(
        df=df,
        t_grid=t_grid,
        t_max_days=float(t_cfg["end"]),
        min_quality=args.min_quality,
    )
    print(f"[17] keeping {len(keep_desc)}/{df['Experimental_index'].nunique()} formulations")

    featurizer = FormulationFeaturizer.fit(keep_desc, continuous_cols=PLGA_CONTINUOUS_COLS)
    encoder = MLPFormulationEncoder(
        input_dim=featurizer.input_dim,
        embed_dim=cfg["encoder"]["embed_dim"],
        hidden_dims=tuple(cfg["encoder"]["hidden_dims"]),
        dropout=cfg["encoder"]["dropout"],
    )
    rp = DescriptorPosterior(
        simulator=sim,
        featurizer=featurizer,
        encoder=encoder,
        flow=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        device=args.device,
    )

    print(f"[17] sampling K={args.K} teacher targets per curve (N={len(keep_desc)})")
    t_sample_start = time.time()
    theta_targets = rp.build_teacher_targets(
        q_phi=q_phi,
        Q_grid=q_grid,
        K=args.K,
        seed=args.seed,
        show_progress_bars=False,
    )
    sample_elapsed = time.time() - t_sample_start
    print(f"[17] teacher sampling finished in {sample_elapsed:.1f}s")

    x_full = featurizer.transform(keep_desc)
    x_masked, mask_indices = _masked_feature_copy(
        x_full=x_full,
        featurizer=featurizer,
        mask_cols=_MASK_COLS_321_ABSENT,
    )

    x_blocks = [x_full] + [x_masked.clone() for _ in range(args.mask_repeats)]
    theta_blocks = [theta_targets] + [theta_targets.clone() for _ in range(args.mask_repeats)]
    x_train = torch.cat(x_blocks, dim=0)
    theta_train = torch.cat(theta_blocks, dim=0)
    print(
        f"[17] training missingness-aware r_psi on x={tuple(x_train.shape)}, "
        f"theta_targets={tuple(theta_train.shape)}"
    )
    print(f"[17] masked cols: {_MASK_COLS_321_ABSENT}")
    print(f"[17] masked indices in x: {mask_indices}")

    t_train_start = time.time()
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
    train_elapsed = time.time() - t_train_start
    print(f"\n[17] training finished in {train_elapsed:.1f}s")
    print(f"[17] log: {log}")

    ckpt_path = args.out / "posterior.pt"
    rp.save(ckpt_path)
    print(f"[17] saved checkpoint to {ckpt_path}")

    log_path = args.out / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"seed                  : {args.seed}\n")
        f.write(f"K                     : {args.K}\n")
        f.write(f"min_quality           : {args.min_quality}\n")
        f.write(f"mask_repeats          : {args.mask_repeats}\n")
        f.write(f"mask_cols             : {_MASK_COLS_321_ABSENT}\n")
        f.write(f"mask_indices          : {mask_indices}\n")
        f.write(f"continuous_cols       : {featurizer.continuous_cols}\n")
        f.write(f"device                : {args.device}\n")
        f.write(f"n_curves_kept         : {len(keep_desc)}\n")
        f.write(f"n_curves_augmented    : {len(x_train)}\n")
        f.write(f"n_pairs               : {theta_train.shape[0] * theta_train.shape[1]}\n")
        f.write(f"featurizer.input_dim  : {featurizer.input_dim}\n")
        f.write(f"polymer_families      : {featurizer.polymer_families}\n")
        f.write(f"teacher_sample_seconds: {sample_elapsed:.1f}\n")
        f.write(f"train_seconds         : {train_elapsed:.1f}\n\n")
        f.write("training log:\n")
        for key, value in log.items():
            f.write(f"  {key:20s}: {value}\n")
    print(f"[17] saved log to {log_path}")

    qual_path = args.out / "quality_distribution.csv"
    with qual_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Experimental_index", "DP_Group", "quality", "tmax_days", "last_Q", "n_obs_kept"])
        for row_idx, (_, row) in enumerate(keep_desc.iterrows()):
            meta = kept_metas[row_idx]
            w.writerow([
                row["Experimental_index"],
                row["DP_Group"],
                args.min_quality,
                f"{meta['tmax']:.2f}",
                f"{meta['last_Q']:.4f}",
                meta["n_obs_kept"],
            ])
    print(f"[17] saved quality distribution to {qual_path}")


if __name__ == "__main__":
    main()
