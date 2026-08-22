"""
04 — Train MVP-B stage 1: `CurvePosterior` q_φ(θ | Q).

What this does:
    1. Build PLGABiphasic + canonical time grid (from config).
    2. Generate N synthetic (θ, Q) pairs by sampling prior + simulator.
    3. Train SNPE-C on the pairs.
    4. Save the posterior checkpoint to outputs/04_curve_posterior/.

Why it exists. This is the first stage of amortized hierarchical inference.
After training, q_φ(θ | Q) is a forward-pass posterior — given any release
curve on the canonical grid, we get a full distribution over θ in
milliseconds. Stage 2 (descriptor → posterior) trains on top of this in
script 06.

Run:
    .\.venv\Scripts\python.exe scripts/04_train_curve_posterior.py

Outputs:
    outputs/04_curve_posterior/posterior.pt
    outputs/04_curve_posterior/training_log.txt

Expected runtime: 30–60 min on a 4060 (4060 ~10 GB VRAM is plenty for
this scale).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import CurvePosterior  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/plga_phase1.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/04_curve_posterior"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None, help="overrides config")
    parser.add_argument("--n-sim", type=int, default=None, help="overrides config")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n_sim = args.n_sim or cfg["synthetic"]["n_pairs"]
    seed = args.seed if args.seed is not None else cfg["synthetic"]["seed"]
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    cp = CurvePosterior(
        simulator=sim,
        t_grid=t_grid,
        flow=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        embedding_output_dim=cfg["posterior"].get("embedding_output_dim", 16),
        embedding_hidden=cfg["posterior"].get("embedding_hidden", 64),
        embedding_layers=cfg["posterior"].get("embedding_layers", 3),
        noise_sigma=cfg["posterior"].get("noise_sigma", 0.03),
        device=args.device,
    )

    print(f"[04] device         : {args.device}")
    print(f"[04] n_simulations  : {n_sim}")
    print(f"[04] t_grid         : {t_cfg['start']} to {t_cfg['end']} days, {t_cfg['n']} pts")
    print(f"[04] flow           : {cfg['posterior']['flow_type']}, "
          f"hidden={cfg['posterior']['hidden_features']}, "
          f"transforms={cfg['posterior']['num_transforms']}")

    t0 = time.time()
    log = cp.train(
        n_simulations=n_sim,
        training_batch_size=cfg["train"]["batch_size"],
        max_num_epochs=cfg["train"]["max_epochs"],
        seed=seed,
        stop_after_epochs=cfg["train"]["early_stopping_patience"],
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[04] training finished in {elapsed:.1f}s")
    print(f"[04] log: {log}")

    ckpt_path = args.out / "posterior.pt"
    cp.save(ckpt_path)
    print(f"[04] saved checkpoint to {ckpt_path}")

    log_path = args.out / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"seed                : {seed}\n")
        f.write(f"n_simulations       : {n_sim}\n")
        f.write(f"device              : {args.device}\n")
        f.write(f"elapsed_seconds     : {elapsed:.1f}\n")
        for k, v in log.items():
            f.write(f"{k:20s}: {v}\n")
    print(f"[04] saved log to {log_path}")


if __name__ == "__main__":
    main()
