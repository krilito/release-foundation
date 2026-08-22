"""
03 — Identifiability replay (Task D).

What this does:
    1. Reads outputs/02_oracle_sweep/per_curve.csv to pick N curves spanning
       the R² distribution (default 5 curves at R² quantiles 0.1 / 0.3 / 0.5
       / 0.7 / 0.9).
    2. For each picked curve, runs N_INIT independent NLS fits with random
       prior-sampled initial points.
    3. Reports per-parameter std across fits, normalized by prior range.
    4. Plots a heatmap (curves x params) of normalized std.

Interpretation:
    normalized_std ≈ 0   → the parameter is identifiable from the curve
    normalized_std ≈ 0.3 → loose; the curve constrains it weakly
    normalized_std ≈ 0.6 → effectively unconstrained; this is a sloppy direction

Why it exists. Oracle R² says "the curve can be fit". Identifiability says
"the parameters are the unique fit". If θ wanders across random
initializations, the per-curve oracle θ we'd otherwise hand to a mapper is
a moving target — that is the actual mechanism behind the SRDS
oracle/deployment gap, and SBI's posterior width is the natural remedy.

Run:
    .\.venv\Scripts\python.exe scripts/03_identifiability_replay.py \\
        --data data/Dataset_17_feat_augmented.csv \\
        --sweep outputs/02_oracle_sweep/per_curve.csv

Outputs:
    outputs/03_identifiability_replay/replay_results.csv
    outputs/03_identifiability_replay/normalized_std_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_plga_181  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


def fit_with_init(
    sim: PLGABiphasic,
    t_obs: np.ndarray,
    Q_obs: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    x0: np.ndarray,
) -> tuple[np.ndarray, float]:
    def residual(theta_flat: np.ndarray) -> np.ndarray:
        return sim.simulate_numpy(theta_flat, t_obs) - Q_obs

    result = least_squares(residual, x0=x0, bounds=(lows, highs), method="trf", max_nfev=300)
    res = residual(result.x)
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((Q_obs - Q_obs.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return result.x, r2


def pick_curves(sweep_df: pd.DataFrame, n: int) -> list[str]:
    quantiles = np.linspace(0.1, 0.9, n)
    target_r2s = sweep_df["r2"].quantile(quantiles).to_numpy()
    picked: list[str] = []
    used: set[str] = set()
    for target in target_r2s:
        candidates = sweep_df[~sweep_df["formulation_id"].astype(str).isin(used)].copy()
        idx = (candidates["r2"] - target).abs().idxmin()
        fid = str(candidates.loc[idx, "formulation_id"])
        picked.append(fid)
        used.add(fid)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--n-curves", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/03_identifiability_replay"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sweep_df = pd.read_csv(args.sweep)
    picked = pick_curves(sweep_df, args.n_curves)
    print(f"[..] picked {len(picked)} curves spanning R2 quantiles: {picked}")

    curves = load_plga_181(args.data)
    curves_by_id = {c.formulation_id: c for c in curves}

    sim = PLGABiphasic()
    prior_dist = sim.prior()
    base = prior_dist.base_dist
    lows = base.low.numpy()
    highs = base.high.numpy()

    rows: list[dict] = []
    t0 = time.time()
    for fid in picked:
        if fid not in curves_by_id:
            print(f"[warn] {fid} not in dataset, skipping")
            continue
        curve = curves_by_id[fid]
        inits = prior_dist.sample((args.n_init,)).numpy()
        for k, x0 in enumerate(inits):
            theta_hat, r2 = fit_with_init(sim, curve.t.numpy(), curve.Q.numpy(), lows, highs, x0)
            row = {"formulation_id": fid, "init_idx": k, "r2": r2}
            for j, name in enumerate(sim.param_names):
                row[name] = float(theta_hat[j])
            rows.append(row)
        print(f"[ok] {fid}: {args.n_init} fits done  elapsed {time.time()-t0:.1f}s")

    csv_path = args.out / "replay_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    # Heatmap: rows = curves, cols = params, value = std normalized by prior range
    df = pd.DataFrame(rows)
    prior_range = highs - lows
    heatmap = np.zeros((len(picked), len(sim.param_names)))
    for i, fid in enumerate(picked):
        sub = df[df["formulation_id"] == fid]
        for j, name in enumerate(sim.param_names):
            heatmap[i, j] = sub[name].std() / prior_range[j]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=0.5)
    ax.set_xticks(range(len(sim.param_names)))
    ax.set_xticklabels(sim.param_names, rotation=45, ha="right")
    ax.set_yticks(range(len(picked)))
    ax.set_yticklabels(
        [f"{fid}\nR²={sweep_df.set_index('formulation_id').loc[int(fid) if str(fid).isdigit() else fid, 'r2']:.2f}"
         for fid in picked],
        fontsize=8,
    )
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center",
                    color="white" if heatmap[i, j] < 0.25 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("std(θ) / prior_range")
    ax.set_title(f"Identifiability heatmap ({args.n_init} initializations per curve)")
    fig.tight_layout()
    fig.savefig(args.out / "normalized_std_heatmap.png", dpi=150)
    plt.close(fig)
    print("[ok] wrote heatmap")

    # Per-param identifiability summary
    print("\n=== identifiability summary (mean normalized std across picked curves) ===")
    for j, name in enumerate(sim.param_names):
        m = heatmap[:, j].mean()
        flag = "identifiable" if m < 0.15 else "loose" if m < 0.35 else "SLOPPY"
        print(f"  {name:14s}  {m:.3f}   {flag}")


if __name__ == "__main__":
    main()
