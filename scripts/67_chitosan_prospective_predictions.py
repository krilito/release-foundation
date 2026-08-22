"""67 - Prospective predictions for YC chitosan experiments.

LOCKED BEFORE EXPERIMENTS. After the user runs the 12 release curves
(6 formulations x 2 drugs), the predictions in this script are compared
against the observed Q(t). cov90 hit rate becomes the pre-registered
prospective test result.

12 curves:
  6 formulations from orthogonal design (HA MW level x PVP MW level):
    F1: HA low MW  (20-40 MDa),  no PVP
    F2: HA low MW,                PVP 220 kDa
    F3: HA low MW,                PVP 1.3 MDa
    F4: HA high MW (40-80 MDa),  no PVP
    F5: HA high MW,               PVP 220 kDa
    F6: HA high MW,               PVP 1.3 MDa
  2 drugs:
    DEX (dexamethasone, MW 392, logP -1.8)  — inside chitosan NPs
    GCV (ganciclovir,    MW 255, logP -1.66) — in hydrogel matrix

Mechanism: Ritger-Peppas, theta = (log_k, n, Q_max), t in HOURS.

Important scope note:
  This is a prospective chitosan pilot scaffold with a chitosan-specific
  simulator and a hand-specified low-rank theta family. It is not the same
  trained PLGA Active Observer v3 artifact, and should be described that way
  in downstream closeout notes.

Prior conditioning (defensible physical adjustments):
  - DEX is in chitosan NPs, HA/PVP coatings act as physical barriers ->
    HA presence: log_k -= 0.25
    high-MW HA:  log_k -= 0.25 additional
    PVP present: log_k -= 0.30
    high-MW PVP: log_k -= 0.20 additional
  - GCV is in the hydrogel matrix, less affected by NP coatings:
    Apply 1/3 of the same adjustments (since some HA/PVP also ends up
    in the hydrogel, partial effect).
  - Drug-size effect on diffusivity (Stokes-Einstein, D ~ 1/r ~ 1/MW^{1/3}):
    GCV MW 255 vs DEX 392: GCV diffuses ~13% faster -> log_k for GCV
    +0.06 vs DEX in identical matrix.

Lock metadata:
  - prediction generated: BEFORE any chitosan experimental Q is observed
  - simulator: ChitosanRitgerPeppas(time_unit_hours=True)
  - prior: log_k in [-5, -0.5], n in [0.35, 1.0], Q_max in [0.3, 1.0]
  - preregistered primary endpoint: aggregate cov90 >= 0.83
  - eval time grid: [0.5, 1, 2, 6, 24, 72, 168, 336, 504, 672] hours
                    (0.5h - 28d; user picks their actual measurement times
                    later, we'll interpolate the predicted Q(t) curves to
                    whatever they pick. Predictions are saved on this
                    common grid + as (mu, U, sigma) so they can be
                    re-evaluated at any t)

Outputs:
  outputs/67_chitosan_prospective/predictions.csv  — per (formulation, drug) curve
  outputs/67_chitosan_prospective/theta_targets.npz — (mu, U, sigma) per curve
  outputs/67_chitosan_prospective/curves_plot.png   — visualization
  outputs/67_chitosan_prospective/lock_metadata.json — git hash + timestamp
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chitosan_simulator import ChitosanRitgerPeppas


# 12 curves: 6 formulations x 2 drugs
FORMULATIONS = [
    # (id, HA_present, HA_MW_band, PVP_present, PVP_MW_kDa)
    {"id": "F1", "HA_band": "low",  "HA_MW_MDa": 30.0,  "PVP": False, "PVP_MW_kDa": 0.0},
    {"id": "F2", "HA_band": "low",  "HA_MW_MDa": 30.0,  "PVP": True,  "PVP_MW_kDa": 220.0},
    {"id": "F3", "HA_band": "low",  "HA_MW_MDa": 30.0,  "PVP": True,  "PVP_MW_kDa": 1300.0},
    {"id": "F4", "HA_band": "high", "HA_MW_MDa": 60.0,  "PVP": False, "PVP_MW_kDa": 0.0},
    {"id": "F5", "HA_band": "high", "HA_MW_MDa": 60.0,  "PVP": True,  "PVP_MW_kDa": 220.0},
    {"id": "F6", "HA_band": "high", "HA_MW_MDa": 60.0,  "PVP": True,  "PVP_MW_kDa": 1300.0},
]

DRUGS = [
    {"name": "DEX", "MW": 392.46, "logP": -1.80, "location": "NP"},      # inside chitosan NPs
    {"name": "GCV", "MW": 255.23, "logP": -1.66, "location": "matrix"},   # in hydrogel matrix
]

T_GRID_HOURS = np.array([0.5, 1.0, 2.0, 6.0, 24.0, 72.0, 168.0, 336.0, 504.0, 672.0])


def derive_prior_centroid(form: dict, drug: dict) -> dict:
    """Per-curve prior centroid (mu) for Ritger-Peppas.

    Returns dict with keys log_k, n, Q_max. Based on:
      1. Default centroid in the middle of the prior
      2. Physical adjustments from drug + formulation
    """
    # Defaults: middle of prior
    log_k_center = -2.5     # k = 0.082 / h^n, gives Q ~ 50% at 24h for n=0.55
    n_center = 0.55          # mild anomalous transport
    Q_max_center = 0.80      # typical chitosan hydrogel releases 80% asymptote

    # Drug adjustments (Stokes-Einstein for diffusion)
    # Reference drug = DEX (MW 392). GCV is smaller -> faster.
    log_k = log_k_center
    log_k += -np.log10(drug["MW"] / 392.46) / 3.0   # 1/MW^{1/3} scaling

    # HA / PVP coating effect (DEX is shielded inside NPs; GCV is in
    # matrix but partially shielded by HA/PVP that also dissolves in hydrogel)
    coating_factor = 1.0 if drug["location"] == "NP" else 1.0 / 3.0

    if form["HA_band"] == "low":
        log_k += -0.25 * coating_factor              # HA present
    elif form["HA_band"] == "high":
        log_k += -0.50 * coating_factor              # higher MW HA = slower

    if form["PVP"]:
        log_k += -0.30 * coating_factor              # PVP barrier
        if form["PVP_MW_kDa"] >= 1000:
            log_k += -0.20 * coating_factor          # high-MW PVP slower

    # n: coatings tend to shift toward more anomalous (n closer to 0.5-0.7)
    n_val = n_center
    if form["PVP"]:
        n_val += 0.05                                # more diffusion-controlled

    # Q_max: drug location matters
    if drug["location"] == "NP":
        Q_max_val = 0.75                             # nano-encapsulation lowers Q_max
    else:
        Q_max_val = 0.85                             # matrix drug more accessible

    # Clamp to prior box
    log_k = float(np.clip(log_k, -5.0, -0.5))
    n_val = float(np.clip(n_val, 0.35, 1.0))
    Q_max_val = float(np.clip(Q_max_val, 0.30, 1.0))

    return {"log_k": log_k, "n": n_val, "Q_max": Q_max_val}


def derive_posterior_covariance(form: dict, drug: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-curve U and sigma_active. Captures our uncertainty about the
    prior conditioning above.

    Strategy: use a low-rank Gaussian with primary direction along log_k
    (largest physical uncertainty), secondary along n. Q_max gets a
    smaller uncertainty. This reflects what we actually don't know:
    rate constants are uncertain to ~1 order of magnitude (sigma ~0.5
    in log_k); n is uncertain to ~0.15; Q_max to ~0.1.
    """
    # Active directions in (log_k, n, Q_max) coordinates
    # Direction 1: pure log_k uncertainty
    # Direction 2: pure n uncertainty (orthogonal)
    # Direction 3: pure Q_max uncertainty
    U = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    sigma = np.array([0.50, 0.15, 0.08], dtype=np.float32)
    return U, sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/67_chitosan_prospective"))
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sim = ChitosanRitgerPeppas(time_unit_hours=True)
    prior = sim.prior().base_dist
    prior_low = prior.low.numpy(); prior_high = prior.high.numpy()
    t_t = torch.tensor(T_GRID_HOURS, dtype=torch.float32)

    rows: list[dict] = []
    mus: list[np.ndarray] = []
    Us: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    for form in FORMULATIONS:
        for drug in DRUGS:
            cent = derive_prior_centroid(form, drug)
            mu = np.array([cent["log_k"], cent["n"], cent["Q_max"]], dtype=np.float32)
            U, sigma = derive_posterior_covariance(form, drug)

            # Sample from this curve's posterior and compute PI
            S = args.n_samples
            eps = np.random.randn(S, U.shape[1])
            delta = (U @ (eps * sigma).T).T                     # (S, 3)
            theta_samples = mu[None, :] + delta
            theta_samples = np.clip(theta_samples, prior_low, prior_high)

            # Simulate
            theta_t = torch.tensor(theta_samples, dtype=torch.float32)
            with torch.no_grad():
                # Add t=0 prepending convention (SimulatorDecoder does this
                # but here we call simulate directly; do it manually).
                t_full = torch.cat([torch.zeros(1), t_t])
                Q_full = sim.simulate(theta_t, t_full).cpu().numpy()
                Q_samples = Q_full[:, 1:]                       # (S, T)
                Q_point = sim.simulate_numpy(mu, np.concatenate([[0.0], T_GRID_HOURS]))[1:]

            sample_mean = Q_samples.mean(axis=0)
            recenter = Q_point - sample_mean
            lo90 = np.percentile(Q_samples, 5.0, axis=0)  + recenter
            hi90 = np.percentile(Q_samples, 95.0, axis=0) + recenter
            lo50 = np.percentile(Q_samples, 25.0, axis=0) + recenter
            hi50 = np.percentile(Q_samples, 75.0, axis=0) + recenter

            mus.append(mu); Us.append(U); sigmas.append(sigma)
            for ti, t in enumerate(T_GRID_HOURS):
                rows.append({
                    "formulation": form["id"],
                    "drug": drug["name"],
                    "HA_band": form["HA_band"],
                    "HA_MW_MDa": form["HA_MW_MDa"],
                    "PVP": form["PVP"],
                    "PVP_MW_kDa": form["PVP_MW_kDa"],
                    "drug_MW": drug["MW"],
                    "drug_logP": drug["logP"],
                    "drug_loc": drug["location"],
                    "t_hours": float(t),
                    "t_days": float(t / 24.0),
                    "Q_predicted_point": float(Q_point[ti]),
                    "Q_lo90": float(lo90[ti]), "Q_hi90": float(hi90[ti]),
                    "Q_lo50": float(lo50[ti]), "Q_hi50": float(hi50[ti]),
                    "mu_log_k": float(mu[0]), "mu_n": float(mu[1]), "mu_Q_max": float(mu[2]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "predictions.csv", index=False)

    np.savez_compressed(
        args.out / "theta_targets.npz",
        formulations=[f"{f['id']}_{d['name']}" for f in FORMULATIONS for d in DRUGS],
        mu=np.stack(mus, axis=0),
        U=np.stack(Us, axis=0),
        sigma=np.stack(sigmas, axis=0),
        t_grid_hours=T_GRID_HOURS,
    )

    # Git hash for reproducibility lock
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
        ).decode().strip()
    except Exception:
        git_hash = "n/a"
    lock = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "script": "scripts/67_chitosan_prospective_predictions.py",
        "simulator": "ChitosanRitgerPeppas(time_unit_hours=True)",
        "n_curves": len(FORMULATIONS) * len(DRUGS),
        "n_formulations": len(FORMULATIONS),
        "n_drugs": len(DRUGS),
        "t_grid_hours": T_GRID_HOURS.tolist(),
        "prior_low": prior_low.tolist(),
        "prior_high": prior_high.tolist(),
        "method_family": "chitosan-specific hand-specified theta family + ChitosanRitgerPeppas simulator",
        "preregistration_doc": "docs/PROSPECTIVE_REGISTRATION_chitosan_batch1_2026-05-28.md",
        "preregistered_pass_rule": {
            "primary_endpoint": "aggregate_cov90",
            "threshold": 0.83,
            "unit": "fraction of all (curve,timepoint) observations inside 90% PI",
        },
        "note": ("Generated BEFORE any chitosan experimental Q is observed. "
                 "Predictions are prospective. Comparison to observed Q yields "
                 "the pre-registered prospective test cov90 statistic."),
    }
    with open(args.out / "lock_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)

    # Quick summary
    print(f"[67] Generated predictions for {len(FORMULATIONS)*len(DRUGS)} curves.", flush=True)
    print(f"[67] Wrote: {args.out}/predictions.csv", flush=True)
    print(f"[67] Wrote: {args.out}/theta_targets.npz", flush=True)
    print(f"[67] Wrote: {args.out}/lock_metadata.json (git {git_hash[:7]})", flush=True)

    # Print prediction summary at key timepoints
    print("\n[67] Predicted Q(t) at t=24h (1 day):")
    sub = df[df["t_hours"] == 24.0].copy()
    print(sub[["formulation", "drug", "Q_predicted_point",
               "Q_lo90", "Q_hi90"]].to_string(index=False))

    print("\n[67] Predicted Q(t) at t=168h (7 days):")
    sub = df[df["t_hours"] == 168.0].copy()
    print(sub[["formulation", "drug", "Q_predicted_point",
               "Q_lo90", "Q_hi90"]].to_string(index=False))

    print("\n[67] Predicted Q(t) at t=672h (28 days):")
    sub = df[df["t_hours"] == 672.0].copy()
    print(sub[["formulation", "drug", "Q_predicted_point",
               "Q_lo90", "Q_hi90"]].to_string(index=False))


if __name__ == "__main__":
    main()
