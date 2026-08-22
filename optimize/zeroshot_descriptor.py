"""Zero-shot descriptor→curve optimization.

Target: from formulation descriptors alone (no early observations),
predict the full release curve on cross321.

Current best: weighted_mask1 NPE gives cross321 median R² = 0.540.
Direct Q regression (ExtraTrees): median R² ≈ 0.88.
Gap: ~0.34 R², entirely in the descriptor→theta mapping.

This script tests three improvements:

    A. Oracle θ teacher (no q_φ noise)
       Use per-curve oracle θ from the bank as teacher targets.
       Eliminates the q_φ distribution shift problem.

    B. Interaction features (drug × polymer)
       48 explicit cross features capture drug-polymer combos.

    C. End-to-end curve loss
       Backprop through the simulator: encoder → θ → simulate → Q
       with L_curve(Q_pred, Q_real) as primary loss.

Each improvement is tested independently and in combination.

Usage:
    python optimize/zeroshot_descriptor.py --device cuda
    python optimize/zeroshot_descriptor.py --device cpu --quick
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
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from casp.calibration import per_curve_r2
from optimize.interaction_encoder import add_interaction_features, interaction_input_dim
from optimize.temporal_embed import TemporalConvEmbedding
from encoder import FormulationFeaturizer, MLPFormulationEncoder, PLGA_CONTINUOUS_COLS
from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid
from simulator import PLGABiphasic


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_loader_38d = _load_script(
    ROOT / "scripts" / "38d_baselines_groupkfold.py", "_loader_38d_zs",
)


def load_internal181() -> dict:
    """Load internal 181 PLGA data with descriptors and curves."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "plga_phase1.yaml").read_text())
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    df = pd.read_csv(ROOT / "data" / "Dataset_17_feat_augmented.csv")
    fid_col, t_col, q_col = "Experimental_index", "Time", "Release"
    desc_df = df.drop_duplicates(fid_col).sort_values(fid_col).reset_index(drop=True)

    # Featurize
    featurizer = FormulationFeaturizer.fit(desc_df, continuous_cols=PLGA_CONTINUOUS_COLS)
    X = featurizer.transform(desc_df).numpy()
    fids = desc_df[fid_col].values

    # Curves
    curves = {}
    for fid in fids:
        g = df[df[fid_col] == fid].sort_values(t_col)
        curves[str(fid)] = {
            "t": g[t_col].values.astype(np.float32),
            "q": g[q_col].values.astype(np.float32),
        }

    # Drug groups (from DP_Group if available, else use first descriptor)
    if "DP_Group" in desc_df.columns:
        drug_groups = desc_df["DP_Group"].apply(lambda x: str(x).split("-")[0]).values
    else:
        drug_groups = np.array(["0"] * len(fids))

    return {
        "X": X,
        "fids": fids,
        "curves": curves,
        "drug_groups": drug_groups,
        "t_grid": t_grid,
        "featurizer": featurizer,
        "df": desc_df,
    }


def load_cross321() -> dict:
    """Load cross321 PLGA data with oracle theta."""
    FORMULATION_COL_MAP = _loader_38d.FORMULATION_COL_MAP
    xlsx = Path(
        r"D:\chemical-world-model-v0\datset\321PLGA"
        r"\A Dataset on Formulation Parameters and Characteristics of "
        r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
    )
    matched = Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv")
    bank_path = Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv")

    df_meta = pd.read_excel(xlsx, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())
    df_bank = pd.read_csv(bank_path)

    df_matched = pd.read_csv(matched)
    fid_col_m = "Formulation_Index" if "Formulation_Index" in df_matched.columns else "fid"
    matched_fids = set(df_matched[fid_col_m].astype(int))
    df_meta_first["fid"] = df_meta_first["fid"].astype(int)
    df_meta_first = df_meta_first[df_meta_first["fid"].isin(matched_fids)]

    X = df_meta_first[feature_cols].values.astype(np.float32)
    fids = df_meta_first["fid"].values.astype(int)

    # Oracle theta
    theta_cols = ["log_kw", "log_kh", "log_alpha", "log_kd", "log_ke",
                  "m_crit", "q_burst", "log_tau_burst", "Q_max"]
    theta_oracle = np.full((len(fids), 9), np.nan, dtype=np.float32)
    bank_map = {int(r["fid"]): r for _, r in df_bank.iterrows()}
    for i, fid in enumerate(fids):
        if fid in bank_map:
            theta_oracle[i] = [bank_map[fid][c] for c in theta_cols]

    # Curves
    curves = {}
    for _, row in df_meta.iterrows():
        fid = int(row["Formulation Index"])
        if fid not in curves:
            curves[fid] = {"t": [], "q": []}
        curves[fid]["t"].append(float(row["Time"]))
        curves[fid]["q"].append(float(row["Release"]))
    for fid in curves:
        order = np.argsort(curves[fid]["t"])
        curves[fid]["t"] = np.array(curves[fid]["t"])[order]
        curves[fid]["q"] = np.array(curves[fid]["q"])[order]

    # Drug groups
    drug_mw = df_meta_first["drug_mw"].values
    bins = np.quantile(drug_mw[~np.isnan(drug_mw)], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    drug_groups = np.digitize(drug_mw, bins[1:-1]).astype(str)

    return {
        "X": X,
        "fids": fids,
        "curves": curves,
        "theta_oracle": theta_oracle,
        "drug_groups": drug_groups,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Method A: Oracle θ teacher + NPE distillation
# ---------------------------------------------------------------------------

def train_npe_from_oracle(
    X_train: np.ndarray,
    theta_train: np.ndarray,
    simulator: PLGABiphasic,
    hidden_features: int = 64,
    num_transforms: int = 5,
    max_epochs: int = 200,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Train NPE r_ψ(θ|x) from oracle θ teacher (no q_φ noise)."""
    torch.manual_seed(seed)
    # Repeat each (x, θ) pair K times to give the flow enough signal
    K = 8
    N = len(X_train)
    x_rep = np.repeat(X_train, K, axis=0)
    theta_rep = np.repeat(theta_train, K, axis=0) + np.random.randn(N * K, 9).astype(np.float32) * 0.01

    x_tensor = torch.tensor(x_rep, dtype=torch.float32)
    theta_tensor = torch.tensor(theta_rep, dtype=torch.float32)

    embed_net = FCEmbedding(input_dim=X_train.shape[1], output_dim=16,
                            num_layers=3, num_hiddens=64)
    density_est = posterior_nn(
        model="maf", hidden_features=hidden_features,
        num_transforms=num_transforms, embedding_net=embed_net,
    )

    prior = simulator.prior()
    if device != "cpu":
        base = prior.base_dist
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(base.low.to(device), base.high.to(device)),
            prior.reinterpreted_batch_ndims,
        )

    inference = NPE(prior=prior, density_estimator=density_est, device=device)
    inference.append_simulations(theta_tensor, x_tensor)
    estimator = inference.train(
        training_batch_size=256, max_num_epochs=max_epochs,
        validation_fraction=0.1, stop_after_epochs=20,
        show_train_summary=True,
    )
    estimator.to(device)
    posterior = inference.build_posterior(estimator)
    return {"posterior": posterior, "estimator": estimator}


# ---------------------------------------------------------------------------
# Method B: Direct θ regression (MLP)
# ---------------------------------------------------------------------------

class DirectThetaRegressor(nn.Module):
    """MLP: x → θ_hat, trained with curve loss through simulator."""

    def __init__(self, input_dim: int, n_params: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_params),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_direct_theta(
    X_train: np.ndarray,
    Q_train: np.ndarray,
    theta_train: np.ndarray,
    simulator: PLGABiphasic,
    t_grid: Tensor,
    max_epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 0,
    use_curve_loss: bool = True,
    curve_loss_weight: float = 1.0,
) -> dict:
    """Train direct θ regressor with optional end-to-end curve loss.

    Loss = L_theta(θ_pred, θ_oracle) + w * L_curve(simulate(θ_pred), Q_real)
    """
    torch.manual_seed(seed)
    model = DirectThetaRegressor(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

    x_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    theta_t = torch.tensor(theta_train, dtype=torch.float32).to(device)
    Q_t = torch.tensor(Q_train, dtype=torch.float32).to(device)

    prior = simulator.prior().base_dist
    prior_low = prior.low.to(device)
    prior_high = prior.high.to(device)

    losses = []
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        theta_pred = model(x_t)

        # θ regression loss
        loss_theta = nn.functional.mse_loss(theta_pred, theta_t)

        # Curve loss (end-to-end through simulator)
        loss_curve = torch.tensor(0.0, device=device)
        if use_curve_loss:
            # Soft-clip θ to prior bounds
            theta_clipped = prior_low + (prior_high - prior_low) * torch.sigmoid(theta_pred)
            with torch.enable_grad():
                Q_pred = simulator.simulate(theta_clipped, t_grid.to(device))
            loss_curve = nn.functional.mse_loss(Q_pred, Q_t)

        loss = loss_theta + curve_loss_weight * loss_curve
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: loss={loss.item():.4f} "
                  f"(θ={loss_theta.item():.4f}, curve={loss_curve.item():.4f})")

    return {"model": model, "losses": losses}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_zeroshot(
    theta_pred: np.ndarray,
    Q_true: np.ndarray,
    simulator: PLGABiphasic,
    t_grid: Tensor,
) -> dict:
    """Evaluate zero-shot predictions: θ → simulate → Q vs Q_true."""
    Q_preds = []
    with torch.no_grad():
        for i in range(len(theta_pred)):
            th = torch.tensor(theta_pred[i:i+1], dtype=torch.float32)
            q = simulator.simulate(th, t_grid).squeeze(0).numpy()
            Q_preds.append(q)
    Q_preds = np.array(Q_preds)

    r2s = [per_curve_r2(Q_true[i], Q_preds[i]) for i in range(len(Q_true))]
    maes = [np.mean(np.abs(Q_true[i] - Q_preds[i])) for i in range(len(Q_true))]
    return {
        "r2_median": float(np.median(r2s)),
        "r2_mean": float(np.mean(r2s)),
        "frac_r2_gte_0": float(np.mean(np.array(r2s) >= 0)),
        "mae_median": float(np.median(maes)),
        "per_curve_r2": r2s,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot descriptor→curve optimization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="1 fold, fewer epochs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="optimize/outputs/zeroshot")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_folds = 2 if args.quick else 5
    epochs = min(args.epochs, 50) if args.quick else args.epochs

    print(f"{'='*70}")
    print(f"Zero-shot Descriptor→Curve Optimization")
    print(f"{'='*70}")
    print(f"device={args.device}, folds={n_folds}, epochs={epochs}")

    # Load data
    print("\n[1] Loading data...")
    c321 = load_cross321()
    X = c321["X"]
    theta_oracle = c321["theta_oracle"]
    curves = c321["curves"]
    fids = c321["fids"]
    drug_groups = c321["drug_groups"]
    t_grid = torch.linspace(0, 90, 64)
    sim = PLGABiphasic()

    # Build Q matrix for cross321
    Q_mat = np.zeros((len(fids), 64), dtype=np.float32)
    for i, fid in enumerate(fids):
        c = curves[int(fid)]
        Q_mat[i] = np.interp(t_grid.numpy(), c["t"], c["q"])

    # Filter valid
    valid = ~np.isnan(theta_oracle).any(axis=1)
    X, theta_oracle, Q_mat, drug_groups = X[valid], theta_oracle[valid], Q_mat[valid], drug_groups[valid]
    fids = fids[valid]
    print(f"  {len(fids)} valid formulations, {X.shape[1]} features")

    # Add interaction features
    X_enhanced = add_interaction_features(torch.tensor(X)).numpy()
    print(f"  With interactions: {X_enhanced.shape[1]} features")

    # Cross-validation
    print(f"\n[2] Running {n_folds}-fold group-by-drug CV...")
    gkf = GroupKFold(n_splits=n_folds)

    results = {}

    for fold_idx, (tr, te) in enumerate(gkf.split(X, groups=drug_groups)):
        print(f"\n--- Fold {fold_idx} (train={len(tr)}, test={len(te)}) ---")

        X_tr, X_te = X[tr], X[te]
        X_en_tr, X_en_te = X_enhanced[tr], X_enhanced[te]
        th_tr, th_te = theta_oracle[tr], theta_oracle[te]
        Q_tr, Q_te = Q_mat[tr], Q_mat[te]

        # A. Direct Q regression baseline (ExtraTrees)
        print("  [A] ExtraTrees direct Q...")
        T = 64
        X_aug_tr = np.column_stack([np.repeat(X_tr, T, 0), np.tile(t_grid.numpy(), len(X_tr))])
        X_aug_te = np.column_stack([np.repeat(X_te, T, 0), np.tile(t_grid.numpy(), len(X_te))])
        et = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1)
        et.fit(X_aug_tr, Q_tr.flatten())
        Q_et = et.predict(X_aug_te).reshape(-1, T)
        r2_et = [per_curve_r2(Q_te[i], Q_et[i]) for i in range(len(Q_te))]
        r = {"r2_median": float(np.median(r2_et)), "r2_mean": float(np.mean(r2_et)),
             "frac_r2_gte_0": float(np.mean(np.array(r2_et) >= 0))}
        results.setdefault("extratrees_Q", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # B. NPE with q_φ teacher (current baseline)
        print("  [B] NPE q_φ teacher (baseline)...")
        q_phi = CurvePosterior.load(
            ROOT / "outputs" / "sprint1_04_curve_posterior" / "posterior.pt",
            simulator=sim, device="cpu",
        )
        # Generate teacher targets
        K = 32
        theta_teacher = []
        for i in tr:
            q_i = torch.tensor(Q_mat[i], dtype=torch.float32)
            th_k = q_phi.sample(q_i, n_samples=K).cpu().numpy()
            theta_teacher.append(th_k)
        theta_teacher = np.array(theta_teacher)  # (N_tr, K, 9)

        # Flatten for NPE
        x_rep = np.repeat(X_tr, K, 0)
        th_flat = theta_teacher.reshape(-1, 9)
        x_t = torch.tensor(x_rep, dtype=torch.float32)
        th_t = torch.tensor(th_flat, dtype=torch.float32)

        embed = FCEmbedding(input_dim=X_tr.shape[1], output_dim=16, num_layers=3, num_hiddens=64)
        dens = posterior_nn(model="maf", hidden_features=64, num_transforms=5, embedding_net=embed)
        prior_dev = sim.prior()
        if args.device != "cpu":
            base = prior_dev.base_dist
            prior_dev = torch.distributions.Independent(
                torch.distributions.Uniform(base.low.to(args.device), base.high.to(args.device)),
                prior_dev.reinterpreted_batch_ndims,
            )
        inf = NPE(prior=prior_dev, density_estimator=dens, device=args.device)
        inf.append_simulations(th_t, x_t)
        est = inf.train(training_batch_size=256, max_num_epochs=epochs,
                        validation_fraction=0.1, stop_after_epochs=20, show_train_summary=False)
        est.to(args.device)
        post = inf.build_posterior(est)

        # Evaluate: sample θ from r_ψ(x_test), simulate
        theta_preds_npe = []
        for i in range(len(X_te)):
            x_i = torch.tensor(X_te[i:i+1], dtype=torch.float32).to(args.device)
            th_s = post.sample((256,), x=x_i.squeeze(0), show_progress_bars=False).cpu()
            theta_preds_npe.append(th_s.mean(dim=0).numpy())
        theta_preds_npe = np.array(theta_preds_npe)
        r = eval_zeroshot(theta_preds_npe, Q_te, sim, t_grid)
        results.setdefault("npe_qphi", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # C. NPE with oracle θ teacher
        print("  [C] NPE oracle θ teacher...")
        res_c = train_npe_from_oracle(
            X_tr, th_tr, sim, max_epochs=epochs, device=args.device, seed=args.seed + fold_idx,
        )
        theta_preds_oracle = []
        for i in range(len(X_te)):
            x_i = torch.tensor(X_te[i:i+1], dtype=torch.float32).to(args.device)
            th_s = res_c["posterior"].sample((256,), x=x_i.squeeze(0), show_progress_bars=False).cpu()
            theta_preds_oracle.append(th_s.mean(dim=0).numpy())
        theta_preds_oracle = np.array(theta_preds_oracle)
        r = eval_zeroshot(theta_preds_oracle, Q_te, sim, t_grid)
        results.setdefault("npe_oracle", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # D. Direct θ MLP + curve loss
        print("  [D] Direct θ MLP + curve loss...")
        res_d = train_direct_theta(
            X_en_tr, Q_tr, th_tr, sim, t_grid,
            max_epochs=epochs, device=args.device, seed=args.seed + fold_idx,
            use_curve_loss=True, curve_loss_weight=0.5,
        )
        res_d["model"].eval()
        with torch.no_grad():
            x_te_t = torch.tensor(X_en_te, dtype=torch.float32).to(args.device)
            theta_preds_mlp = res_d["model"](x_te_t).cpu().numpy()
        r = eval_zeroshot(theta_preds_mlp, Q_te, sim, t_grid)
        results.setdefault("mlp_curve_loss", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

        # E. NPE with oracle θ + interaction features
        print("  [E] NPE oracle θ + interactions...")
        res_e = train_npe_from_oracle(
            X_en_tr, th_tr, sim, max_epochs=epochs, device=args.device, seed=args.seed + fold_idx,
        )
        theta_preds_int = []
        for i in range(len(X_en_te)):
            x_i = torch.tensor(X_en_te[i:i+1], dtype=torch.float32).to(args.device)
            th_s = res_e["posterior"].sample((256,), x=x_i.squeeze(0), show_progress_bars=False).cpu()
            theta_preds_int.append(th_s.mean(dim=0).numpy())
        theta_preds_int = np.array(theta_preds_int)
        r = eval_zeroshot(theta_preds_int, Q_te, sim, t_grid)
        results.setdefault("npe_oracle_interact", []).append(r)
        print(f"    R²={r['r2_median']:.4f}")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*70}")

    summary_rows = []
    for method, folds in results.items():
        r2s = [f["r2_median"] for f in folds]
        fracs = [f["frac_r2_gte_0"] for f in folds]
        row = {
            "method": method,
            "r2_median": float(np.median(r2s)),
            "r2_mean": float(np.mean(r2s)),
            "frac_r2_gte_0": float(np.mean(fracs)),
        }
        summary_rows.append(row)
        print(f"  {method:30s} R²={row['r2_median']:.4f}  frac≥0={row['frac_r2_gte_0']:.3f}")

    # Save
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
    with open(out / "summary.txt", "w") as f:
        f.write("=== Zero-shot Descriptor→Curve Optimization ===\n\n")
        f.write(f"n_folds : {n_folds}\n")
        f.write(f"device  : {args.device}\n")
        f.write(f"epochs  : {epochs}\n\n")
        for row in summary_rows:
            f.write(f"  {row['method']:30s} R²={row['r2_median']:.4f}  frac≥0={row['frac_r2_gte_0']:.3f}\n")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
