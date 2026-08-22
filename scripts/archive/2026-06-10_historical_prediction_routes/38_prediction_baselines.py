"""
38 - Baselines for the formulation -> curve prediction task.

What this does:
    Runs three legitimate baselines on the same 5-fold cross-validation
    split as 37d, so 37c's end-to-end model can be compared fairly.

    B1 - Classical fit-then-extrapolate (CFE):
        For each test curve, fit the 9-parameter PLGA ODE by bounded
        least-squares on ONLY the early-window observations (t <= 7d),
        then simulate forward. Represents the standard PLGA modeling
        workflow in the literature: no ML, no formulation features.

    B2 - Random Forest on theta:
        Same inputs as 37c (formulation 10 + early Q 4 = 14), but
        replace the MLP with sklearn RandomForestRegressor (multi-output
        on theta). After predicting theta, simulate. Tests "is the MLP
        choice doing the work, or does any flexible regressor suffice?"

    B3 - Random Forest on Q (skip ODE):
        Same inputs as 37c. Output: Q values at fixed late time grid
        (linear interpolation back to evaluation grid). No simulator
        in the loop. Tests "does the physics-informed bottleneck
        actually help, or does a pure ML curve regressor match it?"

    Compared against 37c on the same random 5-fold CV.

Why this exists:
    The user pointed out we have not measured our end-to-end model
    against legitimate baselines. Without these, claims like "the model
    learned chemistry generalization" cannot be evaluated. If B1 is
    close to 37c, formulation features don't matter much. If B3
    matches B2 and 37c, the ODE-in-the-loop is window dressing. We
    need to know.

Anti-self-deception:
    - All four methods share the SAME 5-fold split (seed=0, shuffled
      random KFold, matching 37d's first scheme).
    - All eval on the SAME per-curve R^2 metric (full curve against
      actual observations).
    - For B1 (CFE), bounded multistart with 3 restarts. Early-window
      observations < 4 -> skip that curve (CFE has nothing to fit).
    - RF hyperparams (n_estimators=400, max_depth=None) are sklearn
      defaults that usually compete well on small tabular data.

Outputs:
    outputs/38_prediction_baselines/per_curve_per_method.csv
    outputs/38_prediction_baselines/comparison_distribution.png
    outputs/38_prediction_baselines/summary.txt
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402


DATASET_COL = "dataset"
FID_COL = "fid"

FORMULATION_COL_MAP = {
    "Drug MW": "drug_mw",
    "Drug TPSA": "drug_tpsa",
    "Drug LogP": "drug_logp",
    "Polymer MW": "polymer_mw",
    "LA/GA": "laga",
    "Initial Drug-to-Polymer Ratio": "drug_polymer_ratio",
    "Particle Size": "particle_size",
    "Drug Loading Capacity": "drug_loading",
    "Drug Encapsulation Efficiency": "drug_ee",
    "Solubility Enhancer Concentration": "solubility_enhancer",
}


@dataclass
class CurveRecord:
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray


def _load_external_records(
    xlsx_path: Path, matched_fids_csv: Path, t_grid_max_days: float
) -> list[CurveRecord]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns={"Formulation Index": "fid"})
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    df = df[df["fid"].astype(int).isin(matched_fids)].copy()
    out: list[CurveRecord] = []
    for fid in sorted(df["fid"].unique().tolist()):
        g = df[df["fid"] == fid].sort_values("Time")
        g = g.groupby("Time", as_index=False).agg({"Release": "mean"})
        t_obs = g["Time"].to_numpy(dtype=float)
        q_obs = np.clip(g["Release"].to_numpy(dtype=float), 0.0, 1.0)
        mask = t_obs <= t_grid_max_days
        t_obs = t_obs[mask]
        q_obs = q_obs[mask]
        if len(t_obs) < 3:
            continue
        out.append(CurveRecord(fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    return out


def _interp_at(t_obs: np.ndarray, q_obs: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.interp(times, t_obs, q_obs, left=q_obs[0], right=q_obs[-1])


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------
# 37c MLP (replicated for self-contained comparison)
# ---------------------------------------------------------------------

class HybridMLP(nn.Module):
    def __init__(self, n_in: int, n_params: int,
                 prior_low: torch.Tensor, prior_high: torch.Tensor,
                 hidden: int = 64, depth: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d_in = n_in
        for _ in range(depth - 1):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, n_params))
        self.net = nn.Sequential(*layers)
        self.register_buffer("prior_low", prior_low.clone())
        self.register_buffer("prior_high", prior_high.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        eps = 1e-3
        s = torch.sigmoid(z)
        s = eps + (1.0 - 2.0 * eps) * s
        return self.prior_low + (self.prior_high - self.prior_low) * s


def _train_37c_on_fold(
    X: np.ndarray, theta: np.ndarray, idx_tr: np.ndarray, idx_te: np.ndarray,
    n_form: int, prior_low: torch.Tensor, prior_high: torch.Tensor,
    epochs: int, lr: float, batch_size: int, hidden: int, depth: int,
    dropout: float, weight_decay: float, device: torch.device, seed: int,
) -> np.ndarray:
    torch.manual_seed(seed); np.random.seed(seed)
    Xf = X[:, :n_form]
    mu = Xf[idx_tr].mean(axis=0, keepdims=True)
    sd = Xf[idx_tr].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Xfs = (Xf - mu) / sd
    Xs = np.concatenate([Xfs, X[:, n_form:]], axis=1)

    X_tr = torch.tensor(Xs[idx_tr], dtype=torch.float32, device=device)
    X_te = torch.tensor(Xs[idx_te], dtype=torch.float32, device=device)
    Theta_tr = torch.tensor(theta[idx_tr], dtype=torch.float32, device=device)
    theta_mu = Theta_tr.mean(dim=0, keepdim=True)
    theta_sd = Theta_tr.std(dim=0, keepdim=True).clamp_min(1e-3)

    model = HybridMLP(n_in=X.shape[1], n_params=theta.shape[1],
                      prior_low=prior_low, prior_high=prior_high,
                      hidden=hidden, depth=depth, dropout=dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = 10
    n_train = len(idx_tr)
    for epoch in range(epochs):
        scale = (epoch + 1) / warmup if epoch < warmup else 1.0
        for g in optim.param_groups:
            g["lr"] = lr * scale
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            sel = perm[s:s + batch_size]
            tb = model(X_tr[sel])
            tgt = (Theta_tr[sel] - theta_mu) / theta_sd
            pred = (tb - theta_mu) / theta_sd
            loss = ((pred - tgt) ** 2).mean()
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
    model.eval()
    with torch.no_grad():
        return model(X_te).cpu().numpy()


def _train_rf_theta(
    X: np.ndarray, theta: np.ndarray, idx_tr: np.ndarray, idx_te: np.ndarray,
    seed: int, n_estimators: int = 400,
) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=None,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X[idx_tr], theta[idx_tr])
    return rf.predict(X[idx_te])


def _train_rf_q(
    X: np.ndarray, Q_grid: np.ndarray, idx_tr: np.ndarray, idx_te: np.ndarray,
    seed: int, n_estimators: int = 400,
) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=None,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X[idx_tr], Q_grid[idx_tr])
    return rf.predict(X[idx_te])


def _cfe_fit(
    sim: PLGABiphasic, t_early: np.ndarray, q_early: np.ndarray,
    lows: np.ndarray, highs: np.ndarray, n_restarts: int, seed: int,
) -> np.ndarray | None:
    """Bounded multistart fit of 9-param ODE on early-window observations.
    Returns the best theta or None if all restarts failed."""
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lows + highs)
    best_theta: np.ndarray | None = None
    best_ss = np.inf
    for k in range(n_restarts):
        x0 = mid if k == 0 else rng.uniform(lows, highs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = least_squares(
                    lambda th: sim.simulate_numpy(th, t_early) - q_early,
                    x0=x0, bounds=(lows, highs), method="trf", max_nfev=300,
                )
        except Exception:
            continue
        ss = float(np.sum(
            (sim.simulate_numpy(res.x, t_early) - q_early) ** 2
        ))
        if np.isfinite(ss) and ss < best_ss:
            best_ss = ss
            best_theta = res.x.copy()
    return best_theta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-csv",
        type=Path,
        default=Path("outputs/33_active_set_regimes/regime_assignments.csv"),
    )
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument("--cfe-early-cutoff", type=float, default=7.0,
                    help="Use observations at t<=this for CFE fit")
    ap.add_argument("--late-grid", nargs="+", type=float,
                    default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0],
                    help="Time grid that RF-on-Q learns to predict.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cfe-restarts", type=int, default=3)
    # MLP hyperparams matched to 37c
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/38_prediction_baselines"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[38] device: {device}", flush=True)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    prior_low = prior.low.float()
    prior_high = prior.high.float()
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    param_names = list(sim.param_names)

    df_meta = pd.read_excel(args.cross_doi_data, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())

    df_bank = pd.read_csv(args.full_fit_bank)
    df_bank_cross = df_bank[df_bank[DATASET_COL] == "cross321"].copy()
    df_reg = pd.read_csv(args.regime_csv)
    df_reg_cross = df_reg[df_reg[DATASET_COL] == "cross321"][[FID_COL, "regime"]]
    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .merge(df_reg_cross, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )
    curves = _load_external_records(
        xlsx_path=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid_max_days=90.0,
    )
    curve_map = {c.fid: c for c in curves}
    print(f"[38] dataset n_curves: {len(df)}", flush=True)

    # Build inputs: formulation + early Q
    early_times = np.array(args.early_times, dtype=float)
    late_times = np.array(args.late_grid, dtype=float)
    early_Q_list: list[np.ndarray] = []
    late_Q_list: list[np.ndarray] = []
    keep_rows: list[int] = []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        early_Q_list.append(_interp_at(c.t_obs, c.q_obs, early_times))
        late_Q_list.append(_interp_at(c.t_obs, c.q_obs, late_times))
        keep_rows.append(i)
    df = df.loc[keep_rows].reset_index(drop=True)
    early_Q = np.stack(early_Q_list, axis=0).astype(np.float32)
    late_Q = np.stack(late_Q_list, axis=0).astype(np.float32)

    X_form = df[feature_cols].to_numpy(dtype=np.float32)
    X_full = np.concatenate([X_form, early_Q], axis=1)
    theta_oracle = df[param_names].to_numpy(dtype=np.float32)
    fids = df[FID_COL].to_numpy()
    n_form = len(feature_cols)
    n = len(df)

    # Shuffled random 5-fold (deterministic seed=0 to match 37d).
    splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.arange(n)))

    all_rows: list[dict[str, object]] = []

    for fold_idx, (tr, te) in enumerate(splits):
        print(f"\n[38] === fold {fold_idx+1}/{args.n_folds}  train={len(tr)} test={len(te)} ===",
              flush=True)

        # 37c MLP
        print("[38]   training MLP (37c)...", flush=True)
        theta_mlp = _train_37c_on_fold(
            X_full, theta_oracle, tr, te, n_form,
            prior_low, prior_high, args.epochs, args.lr, args.batch_size,
            args.hidden, args.depth, args.dropout, args.weight_decay, device,
            args.seed + fold_idx,
        )

        # B2 RF on theta
        print("[38]   training RF on theta...", flush=True)
        theta_rf = _train_rf_theta(X_full, theta_oracle, tr, te, args.seed + fold_idx)

        # B3 RF on Q (late grid)
        print("[38]   training RF on Q late grid...", flush=True)
        q_late_rf = _train_rf_q(X_full, late_Q, tr, te, args.seed + fold_idx)

        # Train-set median theta for D baseline
        theta_med = np.median(theta_oracle[tr], axis=0)

        # Per-test-curve evaluation
        print(f"[38]   evaluating {len(te)} test curves (CFE inline) ...", flush=True)
        for i, j in enumerate(te):
            fid = int(fids[j])
            c = curve_map[fid]
            theta_o = theta_oracle[j].astype(np.float64)

            # 37c MLP prediction
            t_mlp = theta_mlp[i].astype(np.float64)
            r2_mlp = _r2(c.q_obs, sim.simulate_numpy(t_mlp, c.t_obs))

            # B1 CFE: fit 9-param on t<=cfe_early_cutoff observations
            mask_early = c.t_obs <= args.cfe_early_cutoff
            if int(mask_early.sum()) >= 3:
                cfe_theta = _cfe_fit(
                    sim, c.t_obs[mask_early], c.q_obs[mask_early],
                    lows, highs, args.cfe_restarts, args.seed + fid,
                )
                if cfe_theta is not None:
                    r2_cfe = _r2(c.q_obs, sim.simulate_numpy(cfe_theta, c.t_obs))
                else:
                    r2_cfe = float("nan")
            else:
                r2_cfe = float("nan")

            # B2 RF on theta -> simulate
            t_rf = theta_rf[i].astype(np.float64)
            # clip to prior bounds (RF doesn't know about them)
            t_rf = np.clip(t_rf, lows + 1e-4, highs - 1e-4)
            r2_rf_theta = _r2(c.q_obs, sim.simulate_numpy(t_rf, c.t_obs))

            # B3 RF on Q (late grid). Convert prediction to interp at observed times.
            # Combine with the observed early window (we know that part exactly).
            q_late_pred = q_late_rf[i]
            # Build a piecewise: use observed at t<=early_max, RF-predicted at later
            # Then interpolate to c.t_obs for R^2.
            t_combined = np.concatenate([early_times, late_times])
            q_combined = np.concatenate([early_Q[j], q_late_pred])
            # Sort
            order = np.argsort(t_combined)
            t_combined = t_combined[order]
            q_combined = q_combined[order]
            q_rf_q = np.interp(c.t_obs, t_combined, q_combined,
                               left=q_combined[0], right=q_combined[-1])
            q_rf_q = np.clip(q_rf_q, 0.0, 1.0)
            r2_rf_q = _r2(c.q_obs, q_rf_q)

            # D baseline
            r2_med = _r2(c.q_obs, sim.simulate_numpy(theta_med.astype(np.float64), c.t_obs))

            # E oracle reference
            r2_oracle = _r2(c.q_obs, sim.simulate_numpy(theta_o, c.t_obs))

            all_rows.append({
                "fold": fold_idx,
                "fid": fid,
                "r2_37c_MLP": r2_mlp,
                "r2_B1_CFE_early_only": r2_cfe,
                "r2_B2_RF_theta": r2_rf_theta,
                "r2_B3_RF_Q": r2_rf_q,
                "r2_D_train_median": r2_med,
                "r2_E_oracle_ref": r2_oracle,
            })

        # Per-fold summary
        f_rows = [r for r in all_rows if r["fold"] == fold_idx]
        for name in ["r2_37c_MLP", "r2_B1_CFE_early_only", "r2_B2_RF_theta",
                     "r2_B3_RF_Q", "r2_D_train_median"]:
            vals = np.array([r[name] for r in f_rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            print(f"[38]     {name:<25s} median={np.median(vals):.4f}  "
                  f">=0.9={(vals>=0.9).mean():.2f}  >=0={(vals>=0).mean():.2f}  "
                  f"(n={len(vals)})", flush=True)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(args.out / "per_curve_per_method.csv", index=False)

    methods = [
        ("r2_37c_MLP",          "37c MLP (ours)"),
        ("r2_B1_CFE_early_only","B1 CFE early-only"),
        ("r2_B2_RF_theta",      "B2 RF -> theta"),
        ("r2_B3_RF_Q",          "B3 RF -> Q (no ODE)"),
        ("r2_D_train_median",   "D train-median"),
        ("r2_E_oracle_ref",     "E oracle ref"),
    ]
    summary_rows: list[dict[str, object]] = []
    for col, name in methods:
        v = df_out[col].dropna().to_numpy()
        if len(v) == 0:
            continue
        summary_rows.append({
            "method": name,
            "n_eval": int(len(v)),
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "p25": float(np.percentile(v, 25)),
            "p10": float(np.percentile(v, 10)),
            "frac_above_0.9": float((v >= 0.9).mean()),
            "frac_above_0.5": float((v >= 0.5).mean()),
            "frac_above_0": float((v >= 0).mean()),
        })
    sdf = pd.DataFrame(summary_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    data = [df_out[c].dropna().to_numpy() for c, _ in methods]
    labels = [name for _, name in methods]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    colors = ["lightblue", "lightsalmon", "khaki", "plum", "lightcoral", "lightgreen"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.axhline(0.0, color="gray", lw=0.5, ls="-")
    ax.set_ylabel(f"Full-curve R^2 (pooled across {args.n_folds} folds)")
    ax.set_title(f"38: 37c vs three baselines on random {args.n_folds}-fold CV")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(min(-2.0, float(df_out[[c for c,_ in methods[:-1]]].min().min()) - 0.1), 1.05)
    fig.savefig(args.out / "comparison_distribution.png", dpi=150)
    plt.close(fig)

    # Summary text
    lines = [
        "=== 38 -- baselines for formulation -> curve prediction ===",
        "",
        f"n_curves            : {n}",
        f"n_folds (random)    : {args.n_folds}",
        f"early input times   : {args.early_times}",
        f"CFE early cutoff    : t <= {args.cfe_early_cutoff} d",
        f"late grid for RF-Q  : {args.late_grid}",
        "",
        "--- per-method aggregated R^2 (pooled across folds) ---",
        f"  {'method':<24}  {'n':>4}  {'median':>7}  {'mean':>7}  {'p25':>7}  "
        f"{'p10':>7}  {'>=.9':>5}  {'>=.5':>5}  {'>=0':>5}",
    ]
    for _, r in sdf.iterrows():
        lines.append(
            f"  {str(r['method']):<24}  {int(r['n_eval']):>4}  "
            f"{r['median']:>7.4f}  {r['mean']:>7.4f}  "
            f"{r['p25']:>7.4f}  {r['p10']:>7.4f}  "
            f"{r['frac_above_0.9']:>5.2f}  {r['frac_above_0.5']:>5.2f}  {r['frac_above_0']:>5.2f}"
        )
    lines.append("")

    # Direct comparisons
    def _med(name: str) -> float:
        v = df_out[name].dropna().to_numpy()
        return float(np.median(v)) if len(v) else float("nan")
    m_mlp = _med("r2_37c_MLP")
    m_cfe = _med("r2_B1_CFE_early_only")
    m_rft = _med("r2_B2_RF_theta")
    m_rfq = _med("r2_B3_RF_Q")
    m_d   = _med("r2_D_train_median")
    m_e   = _med("r2_E_oracle_ref")

    lines.append("--- head-to-head deltas (median R^2) ---")
    lines.append(f"  37c MLP - B1 CFE         : {m_mlp - m_cfe:+.4f}  "
                 f"(does using formulation help over classical fit-then-extrapolate?)")
    lines.append(f"  37c MLP - B2 RF -> theta : {m_mlp - m_rft:+.4f}  "
                 f"(does MLP beat RF given same inputs and ODE-in-loop?)")
    lines.append(f"  37c MLP - B3 RF -> Q     : {m_mlp - m_rfq:+.4f}  "
                 f"(does putting ODE in the loop beat pure black-box ML?)")
    lines.append(f"  37c MLP - D train-median : {m_mlp - m_d:+.4f}  "
                 f"(does the model do anything over constant prediction?)")
    lines.append(f"  E oracle - 37c MLP       : {m_e - m_mlp:+.4f}  "
                 f"(how far is our model from the parameter-recovery upper bound?)")
    lines.append("")

    # Verdict
    lines.append("--- verdict ---")
    beats_cfe = m_mlp - m_cfe > 0.03
    beats_rf_theta = m_mlp - m_rft > 0.02
    beats_rf_q = m_mlp - m_rfq > 0.02
    if beats_cfe and beats_rf_theta and beats_rf_q:
        lines.append("  37c BEATS ALL THREE BASELINES.")
        lines.append("  - Formulation info adds genuine value over classical CFE.")
        lines.append("  - The MLP architecture is doing real work (RF on theta is weaker).")
        lines.append("  - The ODE-in-the-loop helps (RF directly on Q is weaker).")
        lines.append("  This is the strongest possible story for the paper.")
    elif beats_cfe and not (beats_rf_theta and beats_rf_q):
        lines.append("  37c BEATS CFE BUT NOT ALL ML BASELINES.")
        lines.append("  Formulation info is the key ingredient; the specific MLP/ODE")
        lines.append("  combo is not uniquely good. Honest paper framing: this is about")
        lines.append("  formulation-conditioned prediction, the model architecture is")
        lines.append("  one of several viable choices.")
    elif not beats_cfe:
        lines.append("  37c DOES NOT BEAT CFE.")
        lines.append("  This means classical fit-then-extrapolate on just the first week")
        lines.append("  of data is essentially as good as our formulation-conditioned MLP.")
        lines.append("  The story has to be retold: the formulation features may not be")
        lines.append("  adding value beyond what early observations already convey.")
    else:
        lines.append("  MIXED. See deltas above.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
