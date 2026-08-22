"""
09 — Train MVP-B stage 2: `DescriptorPosterior` r_ψ(θ | x).

What this does:
    1. Load the trained CurvePosterior (q_φ) from script 04 as a frozen
       teacher.
    2. Load the PLGA 181 internal benchmark.
    3. For each formulation: interpolate (t_obs, Q_obs) onto the
       canonical t_grid (dropping points past t_max_days=90; see
       ADR-018), assign a teacher quality flag.
    4. Filter to formulations meeting the --min-quality threshold.
    5. Fit a FormulationFeaturizer on the kept formulations and build
       an MLPFormulationEncoder.
    6. Sample K teacher targets θ_k ~ q_φ(.|Q_grid_i) per curve.
    7. Train r_ψ via NPE-C on the (θ_k, x_i) pairs (KL distillation
       from q_φ; ADR-018).
    8. Save checkpoint + training_log + quality_distribution.csv.

Run:
    .\.venv\Scripts\python.exe scripts/09_train_descriptor_posterior.py

Outputs:
    outputs/09_descriptor_posterior/posterior.pt
    outputs/09_descriptor_posterior/training_log.txt
    outputs/09_descriptor_posterior/quality_distribution.csv

Expected runtime: 5–15 min (teacher target sampling dominates at K=64
over N=~155 high-quality curves).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import FormulationFeaturizer, MLPFormulationEncoder, PLGA_CONTINUOUS_COLS  # noqa: E402
from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}

# Continuous-column presets for the featurizer. `full` is the production
# 13-descriptor set (ADR-018). `cross-doi-7` is the subset present in the
# external 321 PLGA dataset; used by the ADR-022 ablation to isolate the
# imputation gap from the model gap in cross-DOI evaluation.
_COLS_PRESETS: dict[str, tuple[str, ...]] = {
    "full": PLGA_CONTINUOUS_COLS,
    "cross-doi-7": (
        "LA/GA",
        "Polymer_MW",
        "Initial D/M ratio",
        "DLC",
        "Drug_Mw",
        "Drug_TPSA",
        "Drug_LogP",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/plga_phase1.yaml")
    )
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
        "--out", type=Path, default=Path("outputs/09_descriptor_posterior")
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=64,
                        help="teacher samples per curve (ADR-018 default)")
    parser.add_argument(
        "--min-quality",
        choices=("high", "medium", "low"),
        default="high",
        help="exclude curves below this teacher-quality band (ADR-018)",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument(
        "--cols-preset",
        choices=tuple(_COLS_PRESETS.keys()),
        default="full",
        help="continuous-col set fed to the featurizer; 'cross-doi-7' "
             "is the ADR-022 ablation matching the 321 xlsx coverage",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    # ------------------------------------------------------------------
    # Load q_phi teacher (always on CPU for sampling; cheap and avoids
    # cross-device sample/transfer overhead given the small per-curve
    # cost.)
    # ------------------------------------------------------------------
    print(f"[09] loading teacher q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device="cpu")
    print(f"[09] teacher training log: {q_phi._training_log}")

    # ------------------------------------------------------------------
    # Load real data and build per-formulation tables.
    # ------------------------------------------------------------------
    print(f"[09] loading PLGA dataset from {args.data}")
    df = pd.read_csv(args.data)
    fid_col, t_col, q_col = "Experimental_index", "Time", "Release"
    desc_df = (
        df.drop_duplicates(fid_col)
        .sort_values(fid_col)
        .reset_index(drop=True)
    )
    print(f"[09] {len(desc_df)} unique formulations, "
          f"{len(df)} total (t, Q) rows")

    # ------------------------------------------------------------------
    # Per-curve interpolation + quality flag.
    # ------------------------------------------------------------------
    Q_grid_list = []
    qualities = []
    metas = []
    fids_used = []
    for fid in desc_df[fid_col].tolist():
        g = df[df[fid_col] == fid].sort_values(t_col)
        t_obs = torch.tensor(g[t_col].to_numpy(), dtype=torch.float32)
        Q_obs = torch.tensor(g[q_col].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            Q_grid_i, quality, meta = interpolate_to_grid(
                t_obs, Q_obs, t_grid, t_max_days=float(t_cfg["end"]),
            )
        except ValueError as e:
            print(f"[09] skipping formulation {fid}: {e}")
            qualities.append("skip")
            metas.append({"tmax": 0.0, "last_Q": 0.0, "n_obs_kept": 0})
            Q_grid_list.append(None)
            continue
        Q_grid_list.append(Q_grid_i)
        qualities.append(quality)
        metas.append(meta)
        fids_used.append(fid)

    # Quality distribution.
    dist = {k: 0 for k in ("high", "medium", "low", "skip")}
    for q in qualities:
        dist[q] += 1
    print(f"[09] quality distribution: {dist}")

    # ------------------------------------------------------------------
    # Filter to >= min-quality.
    # ------------------------------------------------------------------
    min_rank = _QUALITY_RANK[args.min_quality]
    keep_mask = [
        (q in _QUALITY_RANK) and (_QUALITY_RANK[q] >= min_rank)
        for q in qualities
    ]
    n_keep = sum(keep_mask)
    print(f"[09] keeping {n_keep}/{len(desc_df)} formulations at "
          f"min_quality={args.min_quality!r}")
    if n_keep == 0:
        raise RuntimeError("no curves pass the quality filter")

    keep_desc = desc_df.loc[keep_mask].reset_index(drop=True)
    Q_grid = torch.stack(
        [Q_grid_list[i] for i, k in enumerate(keep_mask) if k], dim=0
    )

    # ------------------------------------------------------------------
    # Featurizer + encoder.
    # ------------------------------------------------------------------
    cont_cols = _COLS_PRESETS[args.cols_preset]
    featurizer = FormulationFeaturizer.fit(keep_desc, continuous_cols=cont_cols)
    print(f"[09] cols_preset={args.cols_preset!r} -> "
          f"{len(cont_cols)} continuous cols: {cont_cols}")
    print(f"[09] featurizer input_dim={featurizer.input_dim} "
          f"({len(cont_cols)} continuous + "
          f"{len(featurizer.polymer_families)} polymer one-hot)")

    enc_cfg = cfg["encoder"]
    encoder = MLPFormulationEncoder(
        input_dim=featurizer.input_dim,
        embed_dim=enc_cfg["embed_dim"],
        hidden_dims=tuple(enc_cfg["hidden_dims"]),
        dropout=enc_cfg["dropout"],
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

    # ------------------------------------------------------------------
    # Teacher target sampling.
    # ------------------------------------------------------------------
    print(f"[09] sampling K={args.K} teacher targets per curve "
          f"(N={n_keep}, total={n_keep * args.K} pairs)")
    t_sample_start = time.time()
    theta_targets = rp.build_teacher_targets(
        q_phi=q_phi, Q_grid=Q_grid, K=args.K, seed=args.seed,
        show_progress_bars=False,
    )
    sample_elapsed = time.time() - t_sample_start
    print(f"[09] teacher sampling finished in {sample_elapsed:.1f}s, "
          f"targets shape={tuple(theta_targets.shape)}")

    # ------------------------------------------------------------------
    # Train r_psi.
    # ------------------------------------------------------------------
    x = featurizer.transform(keep_desc)
    print(f"[09] training r_psi on x={tuple(x.shape)}, "
          f"theta_targets={tuple(theta_targets.shape)}")
    t_train_start = time.time()
    log = rp.train(
        x=x,
        theta_targets=theta_targets,
        training_batch_size=args.batch_size,
        max_num_epochs=args.max_epochs,
        learning_rate=cfg["train"]["lr"],
        stop_after_epochs=args.stop_after_epochs,
        seed=args.seed,
        verbose=True,
    )
    train_elapsed = time.time() - t_train_start
    print(f"\n[09] training finished in {train_elapsed:.1f}s")
    print(f"[09] log: {log}")

    # ------------------------------------------------------------------
    # Persist.
    # ------------------------------------------------------------------
    ckpt_path = args.out / "posterior.pt"
    rp.save(ckpt_path)
    print(f"[09] saved checkpoint to {ckpt_path}")

    log_path = args.out / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"seed                  : {args.seed}\n")
        f.write(f"K                     : {args.K}\n")
        f.write(f"min_quality           : {args.min_quality}\n")
        f.write(f"cols_preset           : {args.cols_preset}\n")
        f.write(f"continuous_cols       : {cont_cols}\n")
        f.write(f"device                : {args.device}\n")
        f.write(f"n_curves_kept         : {n_keep}\n")
        f.write(f"n_pairs               : {n_keep * args.K}\n")
        f.write(f"featurizer.input_dim  : {featurizer.input_dim}\n")
        f.write(f"polymer_families      : {featurizer.polymer_families}\n")
        f.write(f"teacher_sample_seconds: {sample_elapsed:.1f}\n")
        f.write(f"train_seconds         : {train_elapsed:.1f}\n\n")
        f.write("quality distribution (all 181 curves):\n")
        for k, v in dist.items():
            f.write(f"  {k:8s}: {v}\n")
        f.write("\ntraining log:\n")
        for k, v in log.items():
            f.write(f"  {k:20s}: {v}\n")
    print(f"[09] saved log to {log_path}")

    qual_path = args.out / "quality_distribution.csv"
    with qual_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([fid_col, "DP_Group", "quality",
                    "tmax_days", "last_Q", "n_obs_kept", "used_in_training"])
        for i, fid in enumerate(desc_df[fid_col]):
            m = metas[i]
            w.writerow([
                fid, desc_df.loc[i, "DP_Group"], qualities[i],
                f"{m['tmax']:.2f}", f"{m['last_Q']:.4f}",
                m["n_obs_kept"], "yes" if keep_mask[i] else "no",
            ])
    print(f"[09] saved quality distribution to {qual_path}")


if __name__ == "__main__":
    main()
