"""Zero-shot descriptor→curve optimization (streamlined).

Focus: direct θ regression with end-to-end curve loss through the
differentiable simulator. This avoids NPE's rejection sampling issues
and directly optimizes for curve quality.

Methods compared:
    A. ExtraTrees direct Q (strong baseline, no simulator)
    B. MLP θ regression + MSE loss only
    C. MLP θ regression + curve loss (end-to-end through simulator)
    D. MLP θ regression + curve loss + interaction features
    E. MLP θ regression + curve loss + interaction features + larger net

Usage:
    python optimize/zeroshot_v2.py --device cpu --quick
    python optimize/zeroshot_v2.py --device cuda
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from casp.calibration import per_curve_r2
from optimize.interaction_encoder import add_interaction_features
from simulator import PLGABiphasic


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader = _load_script(ROOT / "scripts" / "38d_baselines_groupkfold.py", "_zsv2")


def load_cross321() -> dict:
    MAP = _loader.FORMULATION_COL_MAP
    xlsx = Path(
        r"D:\chemical-world-model-v0\datset\321PLGA"
        r"\A Dataset on Formulation Parameters and Characteristics of "
        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
    )
    matched = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
    bank = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")

    df = pd.read_excel(xlsx, sheet_name=0)
    df1 = df.drop_duplicates("Formulation Index", keep="first")[
        ["Formulation Index"] + list(MAP.keys())
    ].rename(columns={"Formulation Index": "fid", **MAP})
    df1["fid"] = df1["fid"].astype(int)

    matched_fids = set(pd.read_csv(matched)[
        "Formulation_Index" if "Formulation_Index" in pd.read_csv(matched).columns else "fid"
    ].astype(int))
    df1 = df1[df1["fid"].isin(matched_fids)]

    feat_cols = list(MAP.values())
    X = df1[feat_cols].values.astype(np.float32)
    fids = df1["fid"].values

    # Oracle theta
    bank_df = pd.read_csv(bank)
    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    bank_map = {int(r["fid"]): r for _, r in bank_df.iterrows()}
    theta = np.full((len(fids), 9), np.nan, dtype=np.float32)
    for i, fid in enumerate(fids):
        if int(fid) in bank_map:
            theta[i] = [bank_map[int(fid)][c] for c in theta_cols]

    # Curves + Q matrix
    t_grid = np.linspace(0, 90, 64)
    curves = {}
    for _, row in df.iterrows():
        fid = int(row["Formulation Index"])
        if fid not in curves:
            curves[fid] = {"t": [], "q": []}
        curves[fid]["t"].append(float(row["Time"]))
        curves[fid]["q"].append(float(row["Release"]))
    for fid in curves:
        o = np.argsort(curves[fid]["t"])
        curves[fid]["t"] = np.array(curves[fid]["t"])[o]
        curves[fid]["q"] = np.array(curves[fid]["q"])[o]

    Q = np.zeros((len(fids), 64), dtype=np.float32)
    for i, fid in enumerate(fids):
        Q[i] = np.interp(t_grid, curves[int(fid)]["t"], curves[int(fid)]["q"])

    # Drug groups for CV
    drug_mw = df1["drug_mw"].values
    bins = np.quantile(drug_mw[~np.isnan(drug_mw)], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    groups = np.digitize(drug_mw, bins[1:-1]).astype(str)

    valid = ~np.isnan(theta).any(axis=1)
    return {
        "X": X[valid], "theta": theta[valid], "Q": Q[valid],
        "groups": groups[valid], "fids": fids[valid],
        "feat_cols": feat_cols,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ThetaMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int = 9, width: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(0.1)]
            d_in = width
        layers.append(nn.Linear(d_in, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_theta_mlp(
    X_tr: np.ndarray, Q_tr: np.ndarray, theta_tr: np.ndarray,
    sim: PLGABiphasic, t_grid_np: np.ndarray,
    use_curve_loss: bool = True, curve_weight: float = 1.0,
    width: int = 128, depth: int = 2,
    epochs: int = 200, lr: float = 1e-3, device: str = "cpu",
) -> tuple[ThetaMLP, np.ndarray, np.ndarray]:
    """Train θ MLP with optional curve loss.

    Returns (model, theta_mean, theta_std) for denormalization.
    """
    torch.manual_seed(0)

    # Normalize θ targets (critical for MSE loss)
    th_mean = theta_tr.mean(axis=0)
    th_std = theta_tr.std(axis=0) + 1e-8
    theta_norm = (theta_tr - th_mean) / th_std

    model = ThetaMLP(X_tr.shape[1], 9, width, depth).to(device)
    # Init last layer to output zeros (predict mean θ)
    nn.init.zeros_(model.net[-1].weight)
    nn.init.zeros_(model.net[-1].bias)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    x_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    th_t = torch.tensor(theta_norm, dtype=torch.float32).to(device)
    Q_t = torch.tensor(Q_tr, dtype=torch.float32).to(device)
    t_grid = torch.tensor(t_grid_np, dtype=torch.float32).to(device)

    prior = sim.prior().base_dist
    lo, hi = prior.low.to(device), prior.high.to(device)
    th_mean_t = torch.tensor(th_mean, dtype=torch.float32).to(device)
    th_std_t = torch.tensor(th_std, dtype=torch.float32).to(device)

    for ep in range(epochs):
        opt.zero_grad()
        th_norm_pred = model(x_t)

        # Normalized θ loss
        loss_th = nn.functional.mse_loss(th_norm_pred, th_t)
        loss = loss_th

        if use_curve_loss and ep > 5:
            # Denormalize, clip to prior, simulate
            # Skip first few epochs to let θ predictions stabilize
            th_raw = th_norm_pred * th_std_t + th_mean_t
            th_clip = lo + (hi - lo) * torch.sigmoid(th_raw)
            try:
                Q_pred = sim.simulate(th_clip, t_grid)
                if not torch.isnan(Q_pred).any():
                    loss_cur = nn.functional.mse_loss(Q_pred, Q_t)
                    loss = loss_th + curve_weight * loss_cur
                else:
                    loss_cur = torch.tensor(0.0)
            except Exception:
                loss_cur = torch.tensor(0.0)
        else:
            loss_cur = torch.tensor(0.0)

        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}: loss={loss.item():.4f} "
                  f"(θ={loss_th.item():.4f}, Q={loss_cur.item():.4f})")

    return model, th_mean, th_std


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_model(theta_pred: np.ndarray, Q_true: np.ndarray,
               sim: PLGABiphasic, t_grid: np.ndarray) -> dict:
    t_t = torch.tensor(t_grid, dtype=torch.float32)
    Q_pred = []
    prior = sim.prior().base_dist
    lo, hi = prior.low.numpy(), prior.high.numpy()
    with torch.no_grad():
        for i in range(len(theta_pred)):
            th = np.clip(theta_pred[i], lo, hi)
            try:
                q = sim.simulate(torch.tensor(th[np.newaxis], dtype=torch.float32), t_t)
                Q_pred.append(q.squeeze(0).numpy())
            except Exception:
                Q_pred.append(np.zeros(len(t_grid)))
    Q_pred = np.array(Q_pred)
    r2s = [per_curve_r2(Q_true[i], Q_pred[i]) for i in range(len(Q_true))]
    return {
        "r2_median": float(np.median(r2s)),
        "r2_mean": float(np.mean(r2s)),
        "frac_gte0": float(np.mean(np.array(r2s) >= 0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default="optimize/outputs/zeroshot_v2")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_folds = 2 if args.quick else 5
    epochs = min(args.epochs, 30) if args.quick else args.epochs

    print(f"{'='*60}")
    print(f"Zero-shot v2: direct θ regression + curve loss")
    print(f"{'='*60}")

    data = load_cross321()
    X, theta, Q = data["X"], data["theta"], data["Q"]
    groups = data["groups"]
    t_grid = np.linspace(0, 90, 64)
    sim = PLGABiphasic()
    print(f"  {len(X)} curves, {X.shape[1]} features")

    # Interaction features
    X_int = add_interaction_features(torch.tensor(X)).numpy()
    print(f"  With interactions: {X_int.shape[1]} features")

    gkf = GroupKFold(n_splits=n_folds)
    results = {}

    methods = [
        ("A_extratrees_Q",     "baseline"),
        ("B_mlp_theta_only",   "ablation"),
        ("C_mlp_curve_loss",   "proposed"),
        ("D_mlp_curve_int",    "proposed+feat"),
        ("E_mlp_curve_int_wide", "proposed+feat+wide"),
    ]

    for fold_idx, (tr, te) in enumerate(gkf.split(X, groups=groups)):
        print(f"\n=== Fold {fold_idx} (train={len(tr)}, test={len(te)}) ===")
        Xtr, Xte = X[tr], X[te]
        Xitr, Xite = X_int[tr], X_int[te]
        thtr, thte = theta[tr], theta[te]
        Qtr, Qte = Q[tr], Q[te]

        # A. ExtraTrees direct Q
        print("  [A] ExtraTrees direct Q")
        T = 64
        aug_tr = np.column_stack([np.repeat(Xtr, T, 0), np.tile(t_grid, len(Xtr))])
        aug_te = np.column_stack([np.repeat(Xte, T, 0), np.tile(t_grid, len(Xte))])
        et = ExtraTreesRegressor(500, min_samples_leaf=5, n_jobs=-1)
        et.fit(aug_tr, Qtr.flatten())
        Qet = et.predict(aug_te).reshape(-1, T)
        r2_et = [per_curve_r2(Qte[i], Qet[i]) for i in range(len(Qte))]
        r = {"r2_median": float(np.median(r2_et)), "r2_mean": float(np.mean(r2_et)),
             "frac_gte0": float(np.mean(np.array(r2_et) >= 0))}
        results.setdefault("A_extratrees_Q", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # B. MLP θ only (no curve loss)
        print("  [B] MLP θ regression (MSE only)")
        m, mu, sd = train_theta_mlp(Xtr, Qtr, thtr, sim, t_grid,
                                     use_curve_loss=False, epochs=epochs, device=args.device)
        m.eval()
        with torch.no_grad():
            th_norm = m(torch.tensor(Xte, dtype=torch.float32).to(args.device)).cpu().numpy()
        th_pred = th_norm * sd + mu
        r = eval_model(th_pred, Qte, sim, t_grid)
        results.setdefault("B_mlp_theta_only", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # C. MLP θ + curve loss
        print("  [C] MLP θ + curve loss")
        m, mu, sd = train_theta_mlp(Xtr, Qtr, thtr, sim, t_grid,
                                     use_curve_loss=True, curve_weight=0.5,
                                     epochs=epochs, device=args.device)
        m.eval()
        with torch.no_grad():
            th_norm = m(torch.tensor(Xte, dtype=torch.float32).to(args.device)).cpu().numpy()
        th_pred = th_norm * sd + mu
        r = eval_model(th_pred, Qte, sim, t_grid)
        results.setdefault("C_mlp_curve_loss", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # D. MLP θ + curve loss + interactions
        print("  [D] MLP θ + curve loss + interactions")
        m, mu, sd = train_theta_mlp(Xitr, Qtr, thtr, sim, t_grid,
                                     use_curve_loss=True, curve_weight=0.5,
                                     epochs=epochs, device=args.device)
        m.eval()
        with torch.no_grad():
            th_norm = m(torch.tensor(Xite, dtype=torch.float32).to(args.device)).cpu().numpy()
        th_pred = th_norm * sd + mu
        r = eval_model(th_pred, Qte, sim, t_grid)
        results.setdefault("D_mlp_curve_int", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # E. MLP θ + curve loss + interactions + wider
        print("  [E] MLP θ + curve + interactions + wide")
        m, mu, sd = train_theta_mlp(Xitr, Qtr, thtr, sim, t_grid,
                                     use_curve_loss=True, curve_weight=0.5,
                                     width=256, depth=3, epochs=epochs, device=args.device)
        m.eval()
        with torch.no_grad():
            th_norm = m(torch.tensor(Xite, dtype=torch.float32).to(args.device)).cpu().numpy()
        th_pred = th_norm * sd + mu
        r = eval_model(th_pred, Qte, sim, t_grid)
        results.setdefault("E_mlp_curve_int_wide", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")

    rows = []
    for name, _ in methods:
        folds = results.get(name, [])
        if not folds:
            continue
        r2s = [f["r2_median"] for f in folds]
        row = {"method": name, "r2_median": float(np.median(r2s)),
               "r2_mean": float(np.mean(r2s)),
               "frac_gte0": float(np.mean([f["frac_gte0"] for f in folds]))}
        rows.append(row)
        print(f"  {name:25s} R²={row['r2_median']:.4f}  frac≥0={row['frac_gte0']:.3f}")

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    with open(out / "summary.txt", "w") as f:
        f.write("=== Zero-shot v2 ===\n\n")
        for row in rows:
            f.write(f"  {row['method']:25s} R²={row['r2_median']:.4f}  frac≥0={row['frac_gte0']:.3f}\n")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
