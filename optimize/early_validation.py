"""Early validation: predict full release curve from Q(1,3,5,7) days.

Given early release observations, extrapolate to the full 90-day curve.
This is the scenario where the mechanistic simulator SHOULD win: the
early observations constrain the kinetic parameters, and the simulator
provides physics-based extrapolation.

Current best: ET direct Q (early + descriptors) gives future R² = 0.758.
The particle posterior approach gives 0.579 because the theta posterior
is the bottleneck.

This script tests three approaches:

    A. Direct Q regression (ExtraTrees baseline)
    B. Early Q → theta (oracle fit to early points) → simulate
    C. Early Q → theta (MLP) → simulate (end-to-end)

The key insight: instead of trying to learn a good theta posterior from
descriptors (hard, 10 features → 9 params), use the early observations
to DIRECTLY constrain theta via least-squares fitting, then simulate.

Usage:
    python optimize/early_validation.py --device cpu --quick
    python optimize/early_validation.py --device cuda
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
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from casp.calibration import per_curve_r2
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


_loader = _load_script(ROOT / "scripts" / "38d_baselines_groupkfold.py", "_ev")


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
    theta_oracle = np.full((len(fids), 9), np.nan, dtype=np.float32)
    for i, fid in enumerate(fids):
        if int(fid) in bank_map:
            theta_oracle[i] = [bank_map[int(fid)][c] for c in theta_cols]

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

    # Drug groups
    drug_mw = df1["drug_mw"].values
    bins = np.quantile(drug_mw[~np.isnan(drug_mw)], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    groups = np.digitize(drug_mw, bins[1:-1]).astype(str)

    valid = ~np.isnan(theta_oracle).any(axis=1)
    return {
        "X": X[valid], "theta_oracle": theta_oracle[valid],
        "fids": fids[valid], "groups": groups[valid],
        "curves": {k: v for k, v in curves.items() if k in set(fids[valid].astype(int))},
        "feat_cols": feat_cols,
    }


# ---------------------------------------------------------------------------
# Feature engineering from early observations
# ---------------------------------------------------------------------------

EARLY_TIMES = [1.0, 3.0, 5.0, 7.0]
FUTURE_TIMES = [10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0]
ALL_TIMES = EARLY_TIMES + FUTURE_TIMES


def extract_early_features(curves: dict, fids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract early Q values and derived features.

    Returns:
        Q_early: (N, 4) Q at 1,3,5,7 days
        Q_future: (N, 7) Q at 10,14,21,28,45,60,90 days
        features: (N, F) derived features from early Q
    """
    Q_early_list = []
    Q_future_list = []

    for fid in fids:
        c = curves[int(fid)]
        t, q = c["t"], c["q"]
        Q_all = np.interp(ALL_TIMES, t, q)
        Q_early_list.append(Q_all[:4])
        Q_future_list.append(Q_all[4:])

    Q_early = np.array(Q_early_list, dtype=np.float32)
    Q_future = np.array(Q_future_list, dtype=np.float32)

    # Derived features
    slopes = np.diff(Q_early, axis=1) / np.diff(EARLY_TIMES)  # (N, 3)
    accel = np.diff(slopes, axis=1) / np.array([2.0, 2.0])    # (N, 2)
    auc_early_val = np.array([np.trapezoid(Q_early[i], EARLY_TIMES) for i in range(len(Q_early))])
    auc_early = auc_early_val.reshape(-1, 1)  # (N, 1)
    burst = (Q_early[:, 0:1] > 0.15).astype(np.float32)  # (N, 1) burst indicator

    features = Q_early  # just raw Q(1,3,5,7)
    return Q_early, Q_future, features


# ---------------------------------------------------------------------------
# Method B: Early Q → oracle theta → simulate
# ---------------------------------------------------------------------------

def fit_theta_to_early(
    Q_early: np.ndarray,
    sim: PLGABiphasic,
    t_early: np.ndarray = np.array(EARLY_TIMES),
    n_starts: int = 3,
) -> np.ndarray:
    """Fit theta to early observations via least-squares.

    Uses the simulator's numpy backend for speed. Multiple random
    starts to avoid local minima.
    """
    prior = sim.prior().base_dist
    lo = prior.low.numpy()
    hi = prior.high.numpy()

    theta_fits = []
    for i in range(len(Q_early)):
        q_obs = Q_early[i]

        def residual(th):
            Q_pred = sim.simulate_numpy(th, t_early)
            return Q_pred - q_obs

        best_cost = np.inf
        best_th = None
        for start in range(n_starts):
            if start == 0:
                # Start from prior center
                th0 = (lo + hi) / 2
            else:
                th0 = lo + np.random.rand(9) * (hi - lo)

            try:
                res = least_squares(residual, th0, bounds=(lo, hi), max_nfev=200)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_th = res.x
            except Exception:
                pass

        theta_fits.append(best_th if best_th is not None else (lo + hi) / 2)

    return np.array(theta_fits, dtype=np.float32)


# ---------------------------------------------------------------------------
# Method C: Early Q → theta MLP → simulate
# ---------------------------------------------------------------------------

class EarlyThetaMLP(nn.Module):
    def __init__(self, d_in: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 9),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_early_theta_mlp(
    feat_train: np.ndarray, theta_train: np.ndarray,
    Q_early_train: np.ndarray, Q_future_train: np.ndarray,
    sim: PLGABiphasic,
    epochs: int = 200, lr: float = 1e-3, device: str = "cpu",
) -> tuple[EarlyThetaMLP, np.ndarray, np.ndarray]:
    """Train early Q → theta MLP with normalized targets."""
    torch.manual_seed(0)
    th_mean = theta_train.mean(axis=0)
    th_std = theta_train.std(axis=0) + 1e-8
    theta_norm = (theta_train - th_mean) / th_std

    model = EarlyThetaMLP(feat_train.shape[1]).to(device)
    nn.init.zeros_(model.net[-1].weight)
    nn.init.zeros_(model.net[-1].bias)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    x_t = torch.tensor(feat_train, dtype=torch.float32).to(device)
    th_t = torch.tensor(theta_norm, dtype=torch.float32).to(device)
    Q_fut_t = torch.tensor(Q_future_train, dtype=torch.float32).to(device)

    prior = sim.prior().base_dist
    lo, hi = prior.low.to(device), prior.high.to(device)
    th_mean_t = torch.tensor(th_mean, dtype=torch.float32).to(device)
    th_std_t = torch.tensor(th_std, dtype=torch.float32).to(device)
    t_future = torch.tensor(FUTURE_TIMES, dtype=torch.float32).to(device)

    for ep in range(epochs):
        opt.zero_grad()
        th_norm_pred = model(x_t)
        loss_th = nn.functional.mse_loss(th_norm_pred, th_t)
        loss = loss_th

        if ep > 10:
            th_raw = th_norm_pred * th_std_t + th_mean_t
            th_clip = lo + (hi - lo) * torch.sigmoid(th_raw)
            try:
                Q_pred = sim.simulate(th_clip, t_future)
                if not torch.isnan(Q_pred).any():
                    loss_cur = nn.functional.mse_loss(Q_pred, Q_fut_t)
                    loss = loss_th + 0.5 * loss_cur
            except Exception:
                pass

        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}: loss={loss.item():.4f}")

    return model, th_mean, th_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default="optimize/outputs/early_validation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_folds = 2 if args.quick else 5
    epochs = min(args.epochs, 30) if args.quick else args.epochs

    print(f"{'='*60}")
    print(f"Early Validation: Q(1,3,5,7) → full curve")
    print(f"{'='*60}")

    data = load_cross321()
    X, theta_oracle = data["X"], data["theta_oracle"]
    fids, groups, curves = data["fids"], data["groups"], data["curves"]
    sim = PLGABiphasic()

    Q_early, Q_future, feat = extract_early_features(curves, fids)
    print(f"  {len(fids)} curves, early features: {feat.shape[1]}")
    print(f"  Early times: {EARLY_TIMES}")
    print(f"  Future times: {FUTURE_TIMES}")

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    results = {}

    for fold_idx, (tr, te) in enumerate(kf.split(feat)):
        print(f"\n=== Fold {fold_idx} (train={len(tr)}, test={len(te)}) ===")

        Xtr, Xte = X[tr], X[te]
        feat_tr, feat_te = feat[tr], feat[te]
        Qe_tr, Qe_te = Q_early[tr], Q_early[te]
        Qf_tr, Qf_te = Q_future[tr], Q_future[te]
        th_tr = theta_oracle[tr]

        # A. ExtraTrees direct Q (baseline)
        print("  [A] ExtraTrees direct Q from early")
        # Build (early_features, t) → Q(t) training pairs
        T_all = len(ALL_TIMES)
        X_aug_tr = np.column_stack([np.repeat(feat_tr, T_all, 0),
                                     np.tile(ALL_TIMES, len(feat_tr))])
        Q_all_tr = np.concatenate([Qe_tr, Qf_tr], axis=1)
        X_aug_te = np.column_stack([np.repeat(feat_te, T_all, 0),
                                     np.tile(ALL_TIMES, len(feat_te))])
        et = ExtraTreesRegressor(500, min_samples_leaf=5, n_jobs=-1)
        et.fit(X_aug_tr, Q_all_tr.flatten())
        Q_all_pred = et.predict(X_aug_te).reshape(-1, T_all)
        Qf_et = Q_all_pred[:, 4:]  # future portion
        def safe_r2(y_true, y_pred):
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            if ss_tot <= 1e-10:
                return np.nan
            return 1.0 - ss_res / ss_tot

        r2_et = [safe_r2(Qf_te[i], Qf_et[i]) for i in range(len(Qf_te))]
        r2_et = [r for r in r2_et if not np.isnan(r)]
        r = {"r2_median": float(np.median(r2_et)), "r2_mean": float(np.mean(r2_et)),
             "frac_gte0": float(np.mean(np.array(r2_et) >= 0))}
        results.setdefault("A_et_direct", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

        # B. Oracle theta fit → simulate
        print("  [B] Oracle theta → simulate (upper bound)")
        Qf_oracle = []
        for i in te:
            th = theta_oracle[i]
            q = sim.simulate_numpy(th, np.array(FUTURE_TIMES))
            Qf_oracle.append(q)
        Qf_oracle = np.array(Qf_oracle)
        r2_oracle = [per_curve_r2(Qf_te[i], Qf_oracle[i]) for i in range(len(Qf_te))]
        r = {"r2_median": float(np.median(r2_oracle)), "r2_mean": float(np.mean(r2_oracle)),
             "frac_gte0": float(np.mean(np.array(r2_oracle) >= 0))}
        results.setdefault("B_oracle_theta", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

        # C. Oracle theta → simulate (upper bound for mechanism approach)
        print("  [C] Oracle theta → simulate (upper bound)")
        Qf_oracle = []
        for i in te:
            th = theta_oracle[i]
            q = sim.simulate_numpy(th, np.array(FUTURE_TIMES))
            Qf_oracle.append(q)
        Qf_oracle = np.array(Qf_oracle)
        r2_oracle = [safe_r2(Qf_te[i], Qf_oracle[i]) for i in range(len(Qf_te))]
        r2_oracle = [r for r in r2_oracle if not np.isnan(r)]
        r = {"r2_median": float(np.median(r2_oracle)), "r2_mean": float(np.mean(r2_oracle)),
             "frac_gte0": float(np.mean(np.array(r2_oracle) >= 0))}
        results.setdefault("C_oracle_upper", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

        # E. ET with formulation + early Q
        print("  [E] ExtraTrees (formulation + early Q)")
        feat_full_tr = np.concatenate([Xtr, Qe_tr], axis=1)
        feat_full_te = np.concatenate([Xte, Qe_te], axis=1)
        X_aug_tr2 = np.column_stack([np.repeat(feat_full_tr, T_all, 0),
                                      np.tile(ALL_TIMES, len(feat_full_tr))])
        X_aug_te2 = np.column_stack([np.repeat(feat_full_te, T_all, 0),
                                      np.tile(ALL_TIMES, len(feat_full_te))])
        et2 = ExtraTreesRegressor(500, min_samples_leaf=5, n_jobs=-1)
        et2.fit(X_aug_tr2, Q_all_tr.flatten())
        Q_all_pred2 = et2.predict(X_aug_te2).reshape(-1, T_all)
        Qf_et2 = Q_all_pred2[:, 4:]
        r2_et2 = [safe_r2(Qf_te[i], Qf_et2[i]) for i in range(len(Qf_te))]
        r2_et2 = [r for r in r2_et2 if not np.isnan(r)]
        r = {"r2_median": float(np.median(r2_et2)), "r2_mean": float(np.mean(r2_et2)),
             "frac_gte0": float(np.mean(np.array(r2_et2) >= 0))}
        results.setdefault("E_et_formulation_early", []).append(r)
        print(f"    future R²={r['r2_median']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    rows = []
    for method, folds in results.items():
        r2s = [f["r2_median"] for f in folds]
        row = {"method": method, "r2_median": float(np.median(r2s)),
               "r2_mean": float(np.mean(r2s)),
               "frac_gte0": float(np.mean([f["frac_gte0"] for f in folds]))}
        rows.append(row)
        print(f"  {method:30s} future R²={row['r2_median']:.4f}  frac≥0={row['frac_gte0']:.3f}")

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    with open(out / "summary.txt", "w") as f:
        f.write("=== Early Validation ===\n\n")
        f.write(f"early_times: {EARLY_TIMES}\n")
        f.write(f"future_times: {FUTURE_TIMES}\n\n")
        for row in rows:
            f.write(f"  {row['method']:30s} R²={row['r2_median']:.4f}  frac≥0={row['frac_gte0']:.3f}\n")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
