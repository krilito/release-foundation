"""Early validation via gradient-based theta fitting.

The key insight: the simulator is differentiable. Given early Q
observations, we can gradient-descent on theta to find parameters
that reproduce the early curve, then simulate forward for the full
90-day prediction.

This is fundamentally different from direct regression (which ignores
the simulator) and from NPE (which needs a trained posterior). It's
test-time optimization using the physics model directly.

Pipeline:
    1. For each curve: gradient-fit theta to Q(1,3,5,7) via Adam
    2. Use fitted theta as teacher targets for a fast MLP
    3. At test time: early Q → MLP → theta → simulate (no optimization)

Usage:
    python optimize/early_gradfit.py --device cpu --quick
    python optimize/early_gradfit.py --device cuda
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


_loader = _load_script(ROOT / "scripts" / "38d_baselines_groupkfold.py", "_egf")


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

    df_m = pd.read_csv(matched)
    fid_col = "Formulation_Index" if "Formulation_Index" in df_m.columns else "fid"
    matched_fids = set(df_m[fid_col].astype(int))
    df1 = df1[df1["fid"].isin(matched_fids)]

    fids = df1["fid"].values

    # Curves
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

    # Oracle theta
    bank_df = pd.read_csv(bank)
    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    bank_map = {int(r["fid"]): r for _, r in bank_df.iterrows()}
    theta_oracle = np.full((len(fids), 9), np.nan, dtype=np.float32)
    for i, fid in enumerate(fids):
        if int(fid) in bank_map:
            theta_oracle[i] = [bank_map[int(fid)][c] for c in theta_cols]

    # Drug MW groups
    drug_mw = df1["drug_mw"].values
    bins = np.quantile(drug_mw[~np.isnan(drug_mw)], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    groups = np.digitize(drug_mw, bins[1:-1]).astype(str)

    valid = ~np.isnan(theta_oracle).any(axis=1)
    feat_cols = list(MAP.values())
    return {
        "X": df1[feat_cols].values.astype(np.float32)[valid],
        "theta_oracle": theta_oracle[valid],
        "fids": fids[valid], "groups": groups[valid],
        "curves": {k: v for k, v in curves.items() if k in set(fids[valid].astype(int))},
    }


# ---------------------------------------------------------------------------
# Gradient-based theta fitting
# ---------------------------------------------------------------------------

EARLY_TIMES = [1.0, 3.0, 5.0, 7.0]
FUTURE_TIMES = [10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0]
ALL_TIMES = EARLY_TIMES + FUTURE_TIMES


def gradient_fit_theta(
    Q_early_obs: np.ndarray,
    sim: PLGABiphasic,
    n_steps: int = 300,
    lr: float = 0.05,
    n_restarts: int = 3,
    device: str = "cpu",
) -> np.ndarray:
    """Fit theta to early Q observations via gradient descent.

    Uses the differentiable simulator to compute dQ/dθ and Adam to
    minimize ||Q_pred(early) - Q_obs(early)||².

    Multiple random restarts to avoid local minima.
    """
    prior = sim.prior().base_dist
    lo = prior.low.to(device)
    hi = prior.high.to(device)
    t_early = torch.tensor(EARLY_TIMES, dtype=torch.float32).to(device)
    Q_obs = torch.tensor(Q_early_obs, dtype=torch.float32).to(device)

    best_theta = None
    best_loss = float("inf")

    for restart in range(n_restarts):
        if restart == 0:
            # Start from prior center
            theta = ((lo + hi) / 2).clone()
        else:
            # Random start within prior
            theta = lo + (hi - lo) * torch.rand(9, device=device)

        # Map to unbounded space via inverse sigmoid
        theta_raw = torch.logit((theta - lo) / (hi - lo + 1e-8)).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta_raw], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            # Map back to prior bounds
            theta_bounded = lo + (hi - lo) * torch.sigmoid(theta_raw)
            Q_pred = sim.simulate(theta_bounded.unsqueeze(0), t_early).squeeze(0)
            loss = ((Q_pred - Q_obs) ** 2).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            theta_final = lo + (hi - lo) * torch.sigmoid(theta_raw)
            Q_final = sim.simulate(theta_final.unsqueeze(0), t_early).squeeze(0)
            final_loss = ((Q_final - Q_obs) ** 2).mean().item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = theta_final.detach().cpu().numpy()

    return best_theta


# ---------------------------------------------------------------------------
# MLP for fast theta prediction from early Q
# ---------------------------------------------------------------------------

class EarlyThetaMLP(nn.Module):
    def __init__(self, d_in: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 9),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_theta_mlp(
    Q_early_train: np.ndarray,
    theta_teacher: np.ndarray,
    sim: PLGABiphasic,
    epochs: int = 200, lr: float = 1e-3, device: str = "cpu",
) -> tuple[EarlyThetaMLP, np.ndarray, np.ndarray]:
    """Train early Q → theta MLP from gradient-fit teacher targets."""
    torch.manual_seed(0)
    th_mean = theta_teacher.mean(axis=0)
    th_std = theta_teacher.std(axis=0) + 1e-8
    theta_norm = (theta_teacher - th_mean) / th_std

    model = EarlyThetaMLP(Q_early_train.shape[1]).to(device)
    nn.init.zeros_(model.net[-1].weight)
    nn.init.zeros_(model.net[-1].bias)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    x_t = torch.tensor(Q_early_train, dtype=torch.float32).to(device)
    th_t = torch.tensor(theta_norm, dtype=torch.float32).to(device)

    for ep in range(epochs):
        opt.zero_grad()
        th_pred = model(x_t)
        loss = nn.functional.mse_loss(th_pred, th_t)
        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}: loss={loss.item():.4f}")

    return model, th_mean, th_std


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-10:
        return np.nan
    return 1.0 - ss_res / ss_tot


def eval_theta(theta_pred: np.ndarray, Q_future_true: np.ndarray,
               sim: PLGABiphasic) -> dict:
    Q_pred = []
    for i in range(len(theta_pred)):
        try:
            q = sim.simulate_numpy(theta_pred[i], np.array(FUTURE_TIMES))
        except Exception:
            q = np.zeros(len(FUTURE_TIMES))
        Q_pred.append(q)
    Q_pred = np.array(Q_pred)
    r2s = [safe_r2(Q_future_true[i], Q_pred[i]) for i in range(len(Q_future_true))]
    r2s = [r for r in r2s if not np.isnan(r)]
    return {
        "r2_median": float(np.median(r2s)) if r2s else float("nan"),
        "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
        "frac_gte0": float(np.mean(np.array(r2s) >= 0)) if r2s else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--grad-steps", type=int, default=300)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="optimize/outputs/early_gradfit")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_folds = 2 if args.quick else 5
    epochs = min(args.epochs, 30) if args.quick else args.epochs
    grad_steps = min(args.grad_steps, 100) if args.quick else args.grad_steps

    print(f"{'='*60}")
    print(f"Early Validation via Gradient Theta Fitting")
    print(f"{'='*60}")
    print(f"grad_steps={grad_steps}, restarts={args.n_restarts}, epochs={epochs}")

    data = load_cross321()
    fids, curves = data["fids"], data["curves"]
    theta_oracle = data["theta_oracle"]
    sim = PLGABiphasic()

    # Extract Q at early and future times
    Q_early = np.zeros((len(fids), len(EARLY_TIMES)), dtype=np.float32)
    Q_future = np.zeros((len(fids), len(FUTURE_TIMES)), dtype=np.float32)
    for i, fid in enumerate(fids):
        c = curves[int(fid)]
        Q_all = np.interp(ALL_TIMES, c["t"], c["q"])
        Q_early[i] = Q_all[:4]
        Q_future[i] = Q_all[4:]

    print(f"  {len(fids)} curves")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    results = {}

    for fold_idx, (tr, te) in enumerate(kf.split(Q_early)):
        print(f"\n=== Fold {fold_idx} (train={len(tr)}, test={len(te)}) ===")

        # A. Oracle theta (upper bound)
        print("  [A] Oracle theta → simulate")
        r = eval_theta(theta_oracle[te], Q_future[te], sim)
        results.setdefault("A_oracle", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

        # B. Gradient-fit theta on test set (direct optimization)
        print(f"  [B] Gradient-fit theta ({grad_steps} steps × {args.n_restarts} restarts)")
        t0 = time.time()
        theta_grad = []
        for j, i in enumerate(te):
            th = gradient_fit_theta(
                Q_early[i], sim,
                n_steps=grad_steps, n_restarts=args.n_restarts, device=args.device,
            )
            theta_grad.append(th)
            if (j + 1) % 50 == 0:
                print(f"    [{j+1}/{len(te)}] {time.time()-t0:.0f}s")
        theta_grad = np.array(theta_grad)
        wall = time.time() - t0
        r = eval_theta(theta_grad, Q_future[te], sim)
        results.setdefault("B_gradfit", []).append(r)
        print(f"    future R²={r['r2_median']:.4f} ({wall:.0f}s)")

        # C. Train MLP on gradient-fit theta (train set), evaluate on test
        print("  [C] MLP (gradient-fit teacher) → simulate")
        # Generate teacher targets for train set
        t0 = time.time()
        theta_teacher = []
        for j, i in enumerate(tr):
            th = gradient_fit_theta(
                Q_early[i], sim,
                n_steps=grad_steps, n_restarts=args.n_restarts, device=args.device,
            )
            theta_teacher.append(th)
            if (j + 1) % 50 == 0:
                print(f"    teacher [{j+1}/{len(tr)}] {time.time()-t0:.0f}s")
        theta_teacher = np.array(theta_teacher)
        print(f"    teacher gen: {time.time()-t0:.0f}s")

        # Train MLP
        m, mu, sd = train_theta_mlp(
            Q_early[tr], theta_teacher, sim,
            epochs=epochs, device=args.device,
        )
        m.eval()
        with torch.no_grad():
            th_norm = m(torch.tensor(Q_early[te], dtype=torch.float32).to(args.device)).cpu().numpy()
        th_pred = th_norm * sd + mu
        prior = sim.prior().base_dist
        th_pred = np.clip(th_pred, prior.low.numpy(), prior.high.numpy())
        r = eval_theta(th_pred, Q_future[te], sim)
        results.setdefault("C_mlp_gradfit", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

        # D. Direct ET baseline (for comparison)
        from sklearn.ensemble import ExtraTreesRegressor
        print("  [D] ExtraTrees direct Q (baseline)")
        T = len(ALL_TIMES)
        Q_all_tr = np.concatenate([Q_early[tr], Q_future[tr]], axis=1)
        X_aug_tr = np.column_stack([np.repeat(Q_early[tr], T, 0), np.tile(ALL_TIMES, len(tr))])
        X_aug_te = np.column_stack([np.repeat(Q_early[te], T, 0), np.tile(ALL_TIMES, len(te))])
        et = ExtraTreesRegressor(500, min_samples_leaf=5, n_jobs=-1)
        et.fit(X_aug_tr, Q_all_tr.flatten())
        Q_all_pred = et.predict(X_aug_te).reshape(-1, T)
        Qf_et = Q_all_pred[:, 4:]
        r2_et = [safe_r2(Q_future[te][i], Qf_et[i]) for i in range(len(te))]
        r2_et = [r for r in r2_et if not np.isnan(r)]
        r = {"r2_median": float(np.median(r2_et)), "r2_mean": float(np.mean(r2_et)),
             "frac_gte0": float(np.mean(np.array(r2_et) >= 0))}
        results.setdefault("D_et_baseline", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    rows = []
    for method, folds in results.items():
        r2s = [f["r2_median"] for f in folds if not np.isnan(f["r2_median"])]
        row = {"method": method, "r2_median": float(np.median(r2s)) if r2s else float("nan"),
               "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
               "frac_gte0": float(np.mean([f["frac_gte0"] for f in folds]))}
        rows.append(row)
        print(f"  {method:25s} future R²={row['r2_median']:.4f}  frac≥0={row['frac_gte0']:.3f}")

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    with open(out / "summary.txt", "w") as f:
        f.write("=== Early Validation via Gradient Theta Fitting ===\n\n")
        for row in rows:
            f.write(f"  {row['method']:25s} R²={row['r2_median']:.4f}  frac≥0={row['frac_gte0']:.3f}\n")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
