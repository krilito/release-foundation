"""60 - CASP v0 forward-pass sanity check.

What this does:
    End-to-end forward of the CASP module:

        cross321 curves
            -> features [formulation + early Q(1,3,5,7)]
            -> CASPEncoder -> (mu, U, sigma_active)
            -> sample theta (S=4 draws per curve)
            -> SimulatorDecoder(PLGABiphasic) -> Q_hat(t)
            -> CASPLoss vs observed Q(t) (with warm-start from script 29)

    No training. No evaluation. Just a single forward + loss to confirm:
        - all shapes line up,
        - QR on U is stable, no NaN,
        - PLGABiphasic accepts shape (S*B, P) batched theta,
        - ELBO components are O(1) and not blowing up,
        - warm-start hook from outputs/29_minimal_active_set_audit_r5/
          full_fit_bank.csv works.

After this passes, scripts/60_casp_v0_train.py (next step) adds the
optimizer loop. After that passes a single fold, we go to Phase 2.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import CASPEncoder, CASPLoss, SimulatorDecoder
from simulator import PLGABiphasic


def _load_script(path: Path, name: str) -> ModuleType:
    """Codex-style importlib load. Used to reuse 38d's loader without
    making it a package. Keeps 38d unchanged."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclass / typing introspection
    # inside the module can find it during class construction.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader = _load_script(
    Path(__file__).resolve().parent / "38d_baselines_groupkfold.py",
    "_loader_38d",
)
FORMULATION_COL_MAP = _loader.FORMULATION_COL_MAP
FID_COL = _loader.FID_COL
load_external_records = _loader._load_external_records
interp_at = _loader._interp_at


CROSS_DOI_DATA = Path(
    r"D:\chemical-world-model-v0\datset\321PLGA"
    r"\A Dataset on Formulation Parameters and Characteristics of "
    r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
)
MATCHED_FIDS_CSV = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
FULL_FIT_BANK = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")
EARLY_TIMES = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)
LATE_GRID = np.array([10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0], dtype=float)
ALL_TIMES = np.concatenate([EARLY_TIMES, LATE_GRID])


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    prior_low = prior.low.float()
    prior_high = prior.high.float()
    param_names = list(sim.param_names)

    # --- load cross321 curves + oracle theta bank ---
    df_meta = pd.read_excel(CROSS_DOI_DATA, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())

    df_bank = pd.read_csv(FULL_FIT_BANK)
    df_bank_cross = df_bank[df_bank["dataset"] == "cross321"].copy()

    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )
    curves = load_external_records(
        xlsx_path=CROSS_DOI_DATA,
        matched_fids_csv=MATCHED_FIDS_CSV,
        t_grid_max_days=90.0,
    )
    curve_map = {c.fid: c for c in curves}

    early_Q_list: list[np.ndarray] = []
    full_Q_list: list[np.ndarray] = []
    keep_rows: list[int] = []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        early_Q_list.append(interp_at(c.t_obs, c.q_obs, EARLY_TIMES))
        full_Q_list.append(interp_at(c.t_obs, c.q_obs, ALL_TIMES))
        keep_rows.append(i)
    df = df.loc[keep_rows].reset_index(drop=True)
    early_Q = np.stack(early_Q_list, axis=0).astype(np.float32)
    full_Q = np.stack(full_Q_list, axis=0).astype(np.float32)

    X_form = df[feature_cols].to_numpy(dtype=np.float32)
    X = np.concatenate([X_form, early_Q], axis=1)
    theta_oracle = df[param_names].to_numpy(dtype=np.float32)
    n = len(df)
    print(f"[60] n_curves={n}, n_features={X.shape[1]}, "
          f"n_params={len(param_names)}", flush=True)

    # Standardize formulation features (oracle theta stays in raw units —
    # mu lives in prior box, not normalized).
    n_form = len(feature_cols)
    mu_f = X[:, :n_form].mean(axis=0, keepdims=True)
    sd_f = X[:, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
    X[:, :n_form] = (X[:, :n_form] - mu_f) / sd_f

    # --- small forward batch ---
    B = 8
    sel = np.arange(B)
    x_t = torch.tensor(X[sel], dtype=torch.float32)
    y_t = torch.tensor(full_Q[sel], dtype=torch.float32)
    oracle_t = torch.tensor(theta_oracle[sel], dtype=torch.float32)
    t_t = torch.tensor(ALL_TIMES, dtype=torch.float32)

    encoder = CASPEncoder(
        n_features=X.shape[1],
        n_params=sim.n_params,
        prior_low=prior_low,
        prior_high=prior_high,
        rank=4,
        hidden=64,
        depth=3,
        dropout=0.1,
    )
    decoder = SimulatorDecoder(sim)
    loss_fn = CASPLoss(
        prior_low=prior_low,
        prior_high=prior_high,
        sigma_obs=0.05,
        recon_w=1.0,
        kl_w=1.0,
        orth_w=1e-3,
        warm_w_init=1.0,
    )

    encoder.eval(); decoder.eval()    # no dropout during sanity
    with torch.no_grad():
        q = encoder(x_t)
        print("[60] forward shapes:",
              "mu", tuple(q["mu"].shape),
              "U", tuple(q["U"].shape),
              "sigma", tuple(q["sigma"].shape), flush=True)

        # Verify QR orthonormality
        UtU = torch.einsum("bpr,bps->brs", q["U"], q["U"])
        eye = torch.eye(q["U"].shape[-1]).unsqueeze(0)
        ortho_err = (UtU - eye).abs().max().item()
        print(f"[60] orthonormality max-err: {ortho_err:.2e}", flush=True)

        # Verify mu in prior box
        in_box = ((q["mu"] >= prior_low) & (q["mu"] <= prior_high)).all().item()
        print(f"[60] mu in prior box: {in_box}", flush=True)

        # Sample theta and decode
        theta = encoder.sample_theta(q, n_samples=4)
        print("[60] theta sample shape:", tuple(theta.shape), flush=True)

        # Some theta draws may step slightly outside the prior box due to
        # the active-direction perturbation. Clamp for decode sanity; the
        # real training loop will let those points contribute to the ELBO
        # and let the encoder learn to keep mu away from the boundary.
        theta_clamped = torch.clamp(theta, prior_low, prior_high)

        Q_hat = decoder(theta_clamped, t_t)
        print("[60] Q_hat shape:", tuple(Q_hat.shape), flush=True)
        print(f"[60] Q_hat range: [{Q_hat.min():.3f}, {Q_hat.max():.3f}]", flush=True)
        print(f"[60] y_obs range:  [{y_t.min():.3f}, {y_t.max():.3f}]", flush=True)

        # Loss with warm-start
        out = loss_fn(q, Q_hat, y_t, oracle_mu=oracle_t)
        print("[60] loss components:", flush=True)
        for k in ("total", "recon", "kl_active", "orth", "warm"):
            print(f"       {k:10s} = {float(out[k]):+.4f}", flush=True)

    # Sanity asserts
    assert not torch.isnan(out["total"]), "Loss is NaN"
    assert ortho_err < 1e-4, f"QR orthonormality bad: {ortho_err}"
    assert torch.isfinite(Q_hat).all(), "Q_hat has non-finite values"
    print("\n[60] PASS — forward + ELBO clean. Ready for v0 training loop.", flush=True)


if __name__ == "__main__":
    main()
