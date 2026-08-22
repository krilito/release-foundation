"""64 - PCA-P Phase B: train formulation-only encoder against pull-back targets.

Phase A (script 63) produced per-curve (mu_target, U_target, sigma_target)
from simulator pull-back. Phase B trains a CASPEncoder to predict those
targets from formulation features alone — NO early Q observations. This
is the true zero-shot regime.

Loss (factored, simpler than full KL between low-rank Gaussians):
    L = w_mu     * ||mu_p - mu_t||^2 / prior_std^2
      + w_U      * ||U_p U_p^T - U_t U_t^T||_F^2     (subspace projector distance)
      + w_sigma  * ||log sigma_p - log sigma_t||^2

Why factored: KL between two low-rank Gaussians involves inverses
that get ill-conditioned when target sigma is tiny. The factored
loss has the same minima (matching mu, subspace, scale) but cleaner
gradients.

Outputs:
  outputs/64_pcap_phaseB/per_curve_eval.csv
  outputs/64_pcap_phaseB/fold_summary.csv
  outputs/64_pcap_phaseB/summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import CASPEncoder, SimulatorDecoder
from casp.calibration import per_curve_r2
from simulator import PLGABiphasic


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader_38d = _load_script(
    Path(__file__).resolve().parent / "38d_baselines_groupkfold.py",
    "_loader_38d_for_64",
)

CROSS_DOI_DATA = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)
MATCHED_FIDS_CSV = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
FULL_FIT_BANK = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")
TARGETS_NPZ = Path("outputs/63_pcap_targets/per_curve_targets.npz")
TARGETS_DIAG = Path("outputs/63_pcap_targets/per_curve_diagnostics.csv")


def factored_loss(
    mu_p: Tensor, U_p: Tensor, log_sigma_p: Tensor,
    mu_t: Tensor, U_t: Tensor, log_sigma_t: Tensor,
    prior_std: Tensor,
    w_mu: float = 1.0, w_U: float = 1.0, w_sigma: float = 0.5,
) -> dict[str, Tensor]:
    """All inputs (B, ...). Returns scalar loss components."""
    # mu loss in prior-normalized coords
    mu_loss = ((mu_p - mu_t) / prior_std).pow(2).mean()
    # subspace projector loss: ||U_p U_p^T - U_t U_t^T||_F^2 per batch, then mean
    P_p = torch.einsum("bpr,bqr->bpq", U_p, U_p)
    P_t = torch.einsum("bpr,bqr->bpq", U_t, U_t)
    U_loss = (P_p - P_t).pow(2).sum(dim=(-1, -2)).mean()
    # sigma loss in log-scale
    sigma_loss = (log_sigma_p - log_sigma_t).pow(2).mean()
    total = w_mu * mu_loss + w_U * U_loss + w_sigma * sigma_loss
    return {"total": total, "mu": mu_loss, "U": U_loss, "sigma": sigma_loss}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/64_pcap_phaseB"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-samples-eval", type=int, default=64)
    ap.add_argument("--ess-threshold", type=float, default=10.0,
                    help="filter curves with effective sample size below this")
    ap.add_argument("--single-fold", action="store_true")
    ap.add_argument("--w-mu", type=float, default=1.0)
    ap.add_argument("--w-U", type=float, default=0.5)
    ap.add_argument("--w-sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-target", choices=["pullback", "oracle"], default="oracle",
                    help="pullback = weighted mean of feasible thetas; "
                         "oracle = single best NLS fit (script 29). oracle "
                         "preserves point-prediction sharpness; pullback "
                         "covariance is still used for (U, sigma).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Load targets ---
    targets = np.load(TARGETS_NPZ)
    fids_t = targets["fids"]                     # (N,)
    mu_t = targets["mu"]                         # (N, P)
    U_t = targets["U"]                           # (N, P, r)
    sigma_t = targets["sigma"]                   # (N, r)
    diag = pd.read_csv(TARGETS_DIAG)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    prior_low = prior.low.float(); prior_high = prior.high.float()
    prior_std = (prior_high - prior_low) / (12.0 ** 0.5)
    P = sim.n_params
    r = U_t.shape[-1]

    # --- Load formulation features (no early Q!) ---
    FORMULATION_COL_MAP = _loader_38d.FORMULATION_COL_MAP
    FID_COL = _loader_38d.FID_COL
    df_meta = pd.read_excel(CROSS_DOI_DATA, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())
    df_bank = pd.read_csv(FULL_FIT_BANK)
    param_names = list(sim.param_names)
    df_bank_cross = df_bank[df_bank["dataset"] == "cross321"].copy()
    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )
    # Restrict to fids that have both targets and features.
    fid_to_idx_target = {int(fid): i for i, fid in enumerate(fids_t)}
    keep = df[FID_COL].astype(int).isin(fid_to_idx_target).values
    df = df[keep].reset_index(drop=True)
    fids_keep = df[FID_COL].astype(int).to_numpy()

    # --- Filter by ESS ---
    diag_map = {int(row["fid"]): row["ess"] for _, row in diag.iterrows()}
    ess_vals = np.array([diag_map.get(int(fid), 0.0) for fid in fids_keep])
    ess_mask = ess_vals >= args.ess_threshold
    df = df[ess_mask].reset_index(drop=True)
    fids_keep = fids_keep[ess_mask]
    print(f"[64] curves: {len(df)} kept after ESS >= {args.ess_threshold} filter "
          f"(dropped {(~ess_mask).sum()})", flush=True)

    X = df[feature_cols].to_numpy(dtype=np.float32)            # (N, 10) — formulation only
    print(f"[64] X shape: {X.shape} (formulation only, NO early Q)", flush=True)

    # Get aligned targets in same order as df / X.
    idx_in_targets = np.array([fid_to_idx_target[int(fid)] for fid in fids_keep])
    mu_pull_aligned = mu_t[idx_in_targets].astype(np.float32)
    U_t_aligned = U_t[idx_in_targets].astype(np.float32)
    sigma_t_aligned = sigma_t[idx_in_targets].astype(np.float32)

    # Choose mu target: oracle (sharp point) vs pullback (mean over feasibility).
    # The default is oracle: in nonlinear simulators, the weighted mean of
    # feasible thetas can decode to a worse curve than any single sample,
    # which hurts point R^2. Oracle theta is the single best NLS fit and
    # gives the sharpest point prediction. The pullback covariance is kept
    # for (U, sigma), so the uncertainty story is unchanged.
    if args.mu_target == "oracle":
        mu_t_aligned = df[param_names].to_numpy(dtype=np.float32)
        print(f"[64] mu_target = ORACLE theta (from script 29 bank). "
              f"U/sigma still pull-back-derived.", flush=True)
    else:
        mu_t_aligned = mu_pull_aligned
        print(f"[64] mu_target = pullback weighted mean.", flush=True)

    # --- Load curves for evaluation (R^2 only) ---
    curves_full = _loader_38d._load_external_records(
        xlsx_path=CROSS_DOI_DATA, matched_fids_csv=MATCHED_FIDS_CSV,
        t_grid_max_days=90.0,
    )
    curve_map = {c.fid: c for c in curves_full}
    curves_aligned = [curve_map[int(fid)] for fid in fids_keep]

    # --- CV ---
    n = len(X)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(n)))
    if args.single_fold:
        folds = folds[:1]

    decoder = SimulatorDecoder(PLGABiphasic())
    sim_train_loose = PLGABiphasic(rtol=1e-3, atol=1e-4)  # for any per-curve sim if needed
    all_rows: list[dict] = []
    fold_summary: list[dict] = []

    for fi, (tr, te) in enumerate(folds):
        print(f"\n[64] === fold {fi+1}/{len(folds)} train={len(tr)} test={len(te)} ===",
              flush=True)
        # Standardize formulation features
        mu_f = X[tr].mean(axis=0, keepdims=True)
        sd_f = X[tr].std(axis=0, keepdims=True).clip(min=1e-6)
        X_s = (X - mu_f) / sd_f

        encoder = CASPEncoder(
            n_features=X.shape[1], n_params=P,
            prior_low=prior_low, prior_high=prior_high,
            rank=r, hidden=64, depth=3, dropout=0.1,
        )
        optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        X_tr_t = torch.tensor(X_s[tr], dtype=torch.float32)
        mu_t_tr = torch.tensor(mu_t_aligned[tr], dtype=torch.float32)
        U_t_tr = torch.tensor(U_t_aligned[tr], dtype=torch.float32)
        # log of target sigma, clipped to encoder's representable range
        log_sigma_t_tr = torch.log(torch.tensor(sigma_t_aligned[tr], dtype=torch.float32).clamp_min(1e-4))

        N_tr = len(tr)
        encoder.train()
        for ep in range(args.epochs):
            perm = torch.randperm(N_tr)
            ep_loss = 0.0; ep_mu = 0.0; ep_U = 0.0; ep_sigma = 0.0; nb = 0
            for s in range(0, N_tr, args.batch_size):
                sel = perm[s:s + args.batch_size]
                q = encoder(X_tr_t[sel])
                mu_p = q["mu"]                                  # (B, P)
                U_p = q["U"]                                    # (B, P, r)
                log_sigma_p = torch.log(q["sigma"].clamp_min(1e-4))
                out = factored_loss(
                    mu_p=mu_p, U_p=U_p, log_sigma_p=log_sigma_p,
                    mu_t=mu_t_tr[sel], U_t=U_t_tr[sel], log_sigma_t=log_sigma_t_tr[sel],
                    prior_std=prior_std,
                    w_mu=args.w_mu, w_U=args.w_U, w_sigma=args.w_sigma,
                )
                optim.zero_grad(); out["total"].backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optim.step()
                ep_loss += float(out["total"].detach()); ep_mu += float(out["mu"].detach())
                ep_U += float(out["U"].detach()); ep_sigma += float(out["sigma"].detach())
                nb += 1
            if (ep + 1) % 25 == 0 or ep == 0:
                print(f"  ep {ep+1}/{args.epochs}: total={ep_loss/nb:+.4f}  "
                      f"mu={ep_mu/nb:+.4f}  U={ep_U/nb:+.4f}  sigma={ep_sigma/nb:+.4f}",
                      flush=True)

        # --- Evaluate on test ---
        encoder.eval()
        X_te_t = torch.tensor(X_s[te], dtype=torch.float32)
        with torch.no_grad():
            q_te = encoder(X_te_t)
        # Per-curve: simulate at theta = mu (point prediction), sample S
        # draws for PI band.
        rows: list[dict] = []
        for j, idx in enumerate(te):
            c = curves_aligned[idx]
            t_obs = c.t_obs.astype(np.float64); q_obs = c.q_obs.astype(np.float64)
            mask = (t_obs > 0.0) & (t_obs <= 90.0)
            if mask.sum() < 2:
                continue
            t_eval = t_obs[mask]
            y_eval = q_obs[mask]
            t_eval_t = torch.tensor(t_eval, dtype=torch.float32)

            # Point prediction: simulate at mu
            mu_pred = q_te["mu"][j:j+1]
            with torch.no_grad():
                try:
                    Q_point = decoder(mu_pred, t_eval_t).cpu().numpy()[0]
                except Exception:
                    continue

            # Samples from encoder posterior for PI
            with torch.no_grad():
                q_single = {k: v[j:j+1] for k, v in q_te.items()}
                theta_samples = encoder.sample_theta(q_single, n_samples=args.n_samples_eval)
                try:
                    Q_samp = decoder(theta_samples, t_eval_t).cpu().numpy()
                    # theta_samples shape (S, 1, P) -> Q shape (S, 1, T)
                    Q_samp = Q_samp[:, 0, :]
                except Exception:
                    continue

            sample_mean = Q_samp.mean(axis=0)
            recenter = Q_point - sample_mean
            lo90 = np.percentile(Q_samp, 5.0, axis=0)  + recenter
            hi90 = np.percentile(Q_samp, 95.0, axis=0) + recenter
            lo50 = np.percentile(Q_samp, 25.0, axis=0) + recenter
            hi50 = np.percentile(Q_samp, 75.0, axis=0) + recenter
            cov90 = float(((y_eval >= lo90) & (y_eval <= hi90)).mean())
            cov50 = float(((y_eval >= lo50) & (y_eval <= hi50)).mean())
            pi_w = float((hi90 - lo90).mean())
            r2 = float(per_curve_r2(y_eval[None, :], Q_point[None, :])[0])
            rows.append({
                "fold": fi, "fid": int(fids_keep[idx]),
                "r2_point": r2, "cov90": cov90, "cov50": cov50,
                "pi_width_90": pi_w, "n_t_eval": int(len(t_eval)),
            })
        all_rows.extend(rows)
        if rows:
            df_fold = pd.DataFrame(rows)
            fold_summary.append({
                "fold": fi, "n": len(df_fold),
                "r2_median": float(df_fold["r2_point"].median()),
                "r2_mean": float(df_fold["r2_point"].mean()),
                "cov90_mean": float(df_fold["cov90"].mean()),
                "pi_width_90_median": float(df_fold["pi_width_90"].median()),
            })
            print(f"[64] fold {fi}: R^2 med={df_fold['r2_point'].median():+.4f}  "
                  f"cov90={df_fold['cov90'].mean():.3f}  "
                  f"PI90={df_fold['pi_width_90'].median():.3f}", flush=True)

    pd.DataFrame(all_rows).to_csv(args.out / "per_curve_eval.csv", index=False)
    pd.DataFrame(fold_summary).to_csv(args.out / "fold_summary.csv", index=False)

    df_all = pd.DataFrame(all_rows)
    with open(args.out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== PCA-P Phase B — formulation-only encoder (TRUE zero-shot) ===\n\n")
        f.write(f"n_curves used   : {len(X)} (ESS >= {args.ess_threshold} filter)\n")
        f.write(f"n_features      : {X.shape[1]} (formulation only — NO early Q)\n")
        f.write(f"epochs/fold     : {args.epochs}\n")
        f.write(f"loss weights    : mu={args.w_mu}  U={args.w_U}  sigma={args.w_sigma}\n\n")
        f.write("Per-curve aggregates:\n")
        f.write(f"  R^2 median           : {df_all['r2_point'].median():+.4f}\n")
        f.write(f"  R^2 mean             : {df_all['r2_point'].mean():+.4f}\n")
        f.write(f"  frac R^2 >= 0        : {(df_all['r2_point']>=0).mean():.3f}\n")
        f.write(f"  frac R^2 >= 0.5      : {(df_all['r2_point']>=0.5).mean():.3f}\n")
        f.write(f"  90% PI coverage mean : {df_all['cov90'].mean():.3f}\n")
        f.write(f"  90% PI coverage med  : {df_all['cov90'].median():.3f}\n")
        f.write(f"  PI90 width (median)  : {df_all['pi_width_90'].median():.3f}\n\n")
        f.write("Comparator baselines (ZERO-SHOT, formulation only):\n")
        f.write("  Toronto RF zero-shot (script 15)        : median R^2 ~0.33\n")
        f.write("  Script 45 formulation_only RF (cross321):\n")
        f.write("    random_5fold      median R^2 ~0.909\n")
        f.write("    group_by_drug     median R^2 ~0.672\n")
        f.write("    group_by_polymer  median R^2 ~0.666\n\n")
        f.write("Note: codex's 45 formulation_only also uses theta_oracle as the\n")
        f.write("regression target — it is RF -> theta_oracle (single point) -> ODE.\n")
        f.write("PCA-P uses pull-back feasibility distributions as targets instead;\n")
        f.write("matching or exceeding 0.91/0.67/0.67 would be the meaningful bar.\n")

    print(f"\n[64] wrote {args.out / 'summary.txt'}", flush=True)
    with open(args.out / "summary.txt", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
