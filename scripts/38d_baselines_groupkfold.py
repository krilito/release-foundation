"""
38d - Generalization audit for ALL prediction baselines.

What this does:
    Re-runs the four prediction methods from script 38 under two
    out-of-distribution CV schemes:

        - GroupKFold by drug  (Drug MW + TPSA + LogP)
        - GroupKFold by polymer (Polymer MW + LA/GA)

    Random 5-fold results are also re-run for the same seed and merged
    in, so the three-scheme degradation pattern is read off a single
    combined table.

    Methods (all matched to 38):
      37c MLP          : formulation + early Q -> MLP -> theta -> simulate
      B1 CFE early-only: 9-param ODE bounded LS fit on t <= 7d, extrapolate
      B2 RF -> theta   : Random Forest multi-output regression on theta
      B3 RF -> Q       : Random Forest directly on Q at late grid (no ODE)

Why this exists:
    38 found that RF beats the MLP in-distribution. Before we re-tell
    the paper story, we have to check whether RF still wins under
    out-of-distribution generalization tests (new drug, new polymer).
    If RF degrades less than MLP, the RF advantage holds for deployment.
    If RF degrades as much or more, both methods are equally limited
    by data and the architectural comparison is moot for real use.

Anti-self-deception:
    - All four methods see the SAME train/test indices in each fold
      (the only difference is the splitter).
    - Same hyperparameters as 38 (no method-specific tuning to favor
      one over another).
    - Random 5-fold uses the same seed=0 as 37d / 38 for reproducibility.
    - GroupKFold has no random_state; group order determines splits.
      We do not shuffle groups, so results are deterministic.
    - CFE is training-free, so its multistart seed depends only on fid.
      The same curve gets the same CFE fit under all CV schemes.

Outputs:
    outputs/38d_baselines_groupkfold/per_curve_all_schemes.csv
    outputs/38d_baselines_groupkfold/scheme_method_summary.csv
    outputs/38d_baselines_groupkfold/comparison_distribution.png
    outputs/38d_baselines_groupkfold/degradation_table.png
    outputs/38d_baselines_groupkfold/summary.txt
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
from sklearn.model_selection import GroupKFold, KFold

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
DRUG_KEY_COLS = ["drug_mw", "drug_tpsa", "drug_logp"]
POLYMER_KEY_COLS = ["polymer_mw", "laga"]


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


def _train_mlp_on_fold(
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


def _cfe_fit(
    sim: PLGABiphasic, t_early: np.ndarray, q_early: np.ndarray,
    lows: np.ndarray, highs: np.ndarray, n_restarts: int, seed: int,
) -> np.ndarray | None:
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
    ap.add_argument("--cfe-early-cutoff", type=float, default=7.0)
    ap.add_argument("--late-grid", nargs="+", type=float,
                    default=[10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cfe-restarts", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/38d_baselines_groupkfold"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[38d] device: {device}", flush=True)

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
    drug_keys = df[DRUG_KEY_COLS].apply(tuple, axis=1).to_numpy()
    polymer_keys = df[POLYMER_KEY_COLS].apply(tuple, axis=1).to_numpy()
    n_form = len(feature_cols)
    n = len(df)
    print(f"[38d] n_curves={n}, unique_drugs={pd.Series(drug_keys).nunique()}, "
          f"unique_polymers={pd.Series(polymer_keys).nunique()}", flush=True)

    schemes: list[tuple[str, object]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_keys),
        ("group_by_polymer", polymer_keys),
    ]

    all_rows: list[dict[str, object]] = []

    for scheme_name, groups in schemes:
        print(f"\n[38d] ====== scheme: {scheme_name} ======", flush=True)
        if scheme_name == "random_5fold":
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(n)))
        else:
            splitter = GroupKFold(n_splits=args.n_folds)
            splits = list(splitter.split(np.arange(n), groups=groups))

        for fold_idx, (tr, te) in enumerate(splits):
            print(f"[38d] {scheme_name} fold {fold_idx+1}/{args.n_folds}  "
                  f"train={len(tr)} test={len(te)}", flush=True)

            # MLP
            theta_mlp = _train_mlp_on_fold(
                X_full, theta_oracle, tr, te, n_form,
                prior_low, prior_high, args.epochs, args.lr, args.batch_size,
                args.hidden, args.depth, args.dropout, args.weight_decay, device,
                args.seed + fold_idx,
            )
            # RF -> theta
            rf_theta = RandomForestRegressor(
                n_estimators=400, max_depth=None,
                random_state=args.seed + fold_idx, n_jobs=-1,
            )
            rf_theta.fit(X_full[tr], theta_oracle[tr])
            theta_rf = rf_theta.predict(X_full[te])
            # RF -> Q
            rf_q = RandomForestRegressor(
                n_estimators=400, max_depth=None,
                random_state=args.seed + fold_idx, n_jobs=-1,
            )
            rf_q.fit(X_full[tr], late_Q[tr])
            q_late_rf = rf_q.predict(X_full[te])

            theta_med = np.median(theta_oracle[tr], axis=0)

            for i, j in enumerate(te):
                fid = int(fids[j])
                c = curve_map[fid]
                theta_o = theta_oracle[j].astype(np.float64)

                t_mlp = theta_mlp[i].astype(np.float64)
                r2_mlp = _r2(c.q_obs, sim.simulate_numpy(t_mlp, c.t_obs))

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

                t_rf = np.clip(theta_rf[i].astype(np.float64), lows + 1e-4, highs - 1e-4)
                r2_rf_theta = _r2(c.q_obs, sim.simulate_numpy(t_rf, c.t_obs))

                t_combined = np.concatenate([early_times, late_times])
                q_combined = np.concatenate([early_Q[j], q_late_rf[i]])
                order = np.argsort(t_combined)
                q_rf_q = np.interp(c.t_obs, t_combined[order], q_combined[order],
                                   left=q_combined[order[0]], right=q_combined[order[-1]])
                q_rf_q = np.clip(q_rf_q, 0.0, 1.0)
                r2_rf_q = _r2(c.q_obs, q_rf_q)

                r2_med = _r2(c.q_obs, sim.simulate_numpy(theta_med.astype(np.float64), c.t_obs))
                r2_oracle = _r2(c.q_obs, sim.simulate_numpy(theta_o, c.t_obs))

                all_rows.append({
                    "scheme": scheme_name, "fold": fold_idx, "fid": fid,
                    "r2_37c_MLP": r2_mlp,
                    "r2_B1_CFE": r2_cfe,
                    "r2_B2_RF_theta": r2_rf_theta,
                    "r2_B3_RF_Q": r2_rf_q,
                    "r2_D_median": r2_med,
                    "r2_E_oracle": r2_oracle,
                })

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(args.out / "per_curve_all_schemes.csv", index=False)

    # Per-scheme x method summary
    methods = ["r2_37c_MLP", "r2_B1_CFE", "r2_B2_RF_theta", "r2_B3_RF_Q",
               "r2_D_median", "r2_E_oracle"]
    method_labels = {
        "r2_37c_MLP": "37c MLP",
        "r2_B1_CFE": "B1 CFE",
        "r2_B2_RF_theta": "B2 RF->theta",
        "r2_B3_RF_Q": "B3 RF->Q",
        "r2_D_median": "D median",
        "r2_E_oracle": "E oracle",
    }

    rows: list[dict[str, object]] = []
    for scheme_name, _ in schemes:
        sub = df_out[df_out["scheme"] == scheme_name]
        for m in methods:
            v = sub[m].dropna().to_numpy()
            if len(v) == 0:
                continue
            rows.append({
                "scheme": scheme_name,
                "method": method_labels[m],
                "n": int(len(v)),
                "median": float(np.median(v)),
                "mean": float(np.mean(v)),
                "p25": float(np.percentile(v, 25)),
                "p10": float(np.percentile(v, 10)),
                "frac_above_0.9": float((v >= 0.9).mean()),
                "frac_above_0.5": float((v >= 0.5).mean()),
                "frac_above_0": float((v >= 0).mean()),
            })
    sdf = pd.DataFrame(rows)
    sdf.to_csv(args.out / "scheme_method_summary.csv", index=False)

    # ---- distribution plot: 3 schemes x methods ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True, sharey=True)
    for ax, (scheme_name, _) in zip(axes, schemes):
        sub = df_out[df_out["scheme"] == scheme_name]
        data = [sub[m].dropna().to_numpy() for m in methods]
        bp = ax.boxplot(data, tick_labels=[method_labels[m] for m in methods],
                        showfliers=True, patch_artist=True)
        colors = ["lightblue", "lightsalmon", "khaki", "plum", "lightcoral", "lightgreen"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.axhline(0.9, color="gray", lw=0.5, ls=":")
        ax.axhline(0, color="gray", lw=0.5, ls="-")
        ax.set_title(scheme_name)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(-2.5, 1.05)
    axes[0].set_ylabel("Full-curve R^2 (pooled across folds)")
    fig.suptitle("38d: how each prediction method degrades from in-distribution to out-of-distribution")
    fig.savefig(args.out / "comparison_distribution.png", dpi=150)
    plt.close(fig)

    # ---- degradation table plot ----
    pivot = sdf.pivot(index="method", columns="scheme", values="median")
    method_order = [method_labels[m] for m in methods]
    pivot = pivot.reindex(method_order)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(len(method_order))
    schemes_ordered = ["random_5fold", "group_by_drug", "group_by_polymer"]
    width = 0.27
    colors_s = {"random_5fold": "tab:blue",
                "group_by_drug": "tab:orange",
                "group_by_polymer": "tab:red"}
    for k, sch in enumerate(schemes_ordered):
        if sch not in pivot.columns:
            continue
        ax.bar(x + (k - 1) * width, pivot[sch].values, width,
               label=sch, color=colors_s[sch])
    ax.set_xticks(x)
    ax.set_xticklabels(method_order, rotation=20, ha="right")
    ax.set_ylabel("Median R^2")
    ax.set_title("Median R^2 across CV schemes per method")
    ax.legend()
    ax.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax.set_ylim(min(0.0, float(pivot.min().min()) - 0.05), 1.02)
    fig.savefig(args.out / "degradation_table.png", dpi=150)
    plt.close(fig)

    # ---- summary text ----
    lines = [
        "=== 38d -- baselines under group-by-drug / group-by-polymer CV ===",
        "",
        f"n_curves    : {n}",
        f"unique drugs    : {pd.Series(drug_keys).nunique()}",
        f"unique polymers : {pd.Series(polymer_keys).nunique()}",
        f"n_folds         : {args.n_folds}",
        "",
        "--- median R^2 per scheme x method ---",
        f"  {'method':<14}  {'random_5fold':>12}  {'by_drug':>10}  {'by_polymer':>11}  "
        f"{'dr_drop':>8}  {'pol_drop':>9}",
    ]
    for m in methods:
        label = method_labels[m]
        rrow = sdf[(sdf["method"] == label) & (sdf["scheme"] == "random_5fold")]
        drow = sdf[(sdf["method"] == label) & (sdf["scheme"] == "group_by_drug")]
        prow = sdf[(sdf["method"] == label) & (sdf["scheme"] == "group_by_polymer")]
        if rrow.empty or drow.empty or prow.empty:
            continue
        r_med = float(rrow["median"].iloc[0])
        d_med = float(drow["median"].iloc[0])
        p_med = float(prow["median"].iloc[0])
        lines.append(
            f"  {label:<14}  {r_med:>12.4f}  {d_med:>10.4f}  {p_med:>11.4f}  "
            f"{d_med - r_med:>+8.4f}  {p_med - r_med:>+9.4f}"
        )
    lines.append("")
    lines.append("--- frac_above_0 per scheme x method (robustness / no catastrophic failure) ---")
    lines.append(f"  {'method':<14}  {'random_5fold':>12}  {'by_drug':>10}  {'by_polymer':>11}")
    for m in methods:
        label = method_labels[m]
        def _frac(scheme_name: str) -> float:
            row = sdf[(sdf["method"] == label) & (sdf["scheme"] == scheme_name)]
            return float(row["frac_above_0"].iloc[0]) if not row.empty else float("nan")
        lines.append(
            f"  {label:<14}  {_frac('random_5fold'):>12.4f}  "
            f"{_frac('group_by_drug'):>10.4f}  {_frac('group_by_polymer'):>11.4f}"
        )
    lines.append("")

    # head-to-head deltas under each scheme
    lines.append("--- 37c MLP - B2 RF->theta head-to-head (median R^2) ---")
    for sch in ["random_5fold", "group_by_drug", "group_by_polymer"]:
        m_mlp = sdf[(sdf["method"] == "37c MLP") & (sdf["scheme"] == sch)]
        m_rf = sdf[(sdf["method"] == "B2 RF->theta") & (sdf["scheme"] == sch)]
        if not m_mlp.empty and not m_rf.empty:
            delta = float(m_mlp["median"].iloc[0] - m_rf["median"].iloc[0])
            who = "MLP" if delta > 0 else "RF"
            lines.append(f"  {sch:<20}  delta = {delta:+.4f}  ({who} wins)")
    lines.append("")

    lines.append("--- 37c MLP - B3 RF->Q (no ODE) head-to-head (median R^2) ---")
    for sch in ["random_5fold", "group_by_drug", "group_by_polymer"]:
        m_mlp = sdf[(sdf["method"] == "37c MLP") & (sdf["scheme"] == sch)]
        m_rfq = sdf[(sdf["method"] == "B3 RF->Q") & (sdf["scheme"] == sch)]
        if not m_mlp.empty and not m_rfq.empty:
            delta = float(m_mlp["median"].iloc[0] - m_rfq["median"].iloc[0])
            who = "MLP" if delta > 0 else "RF->Q"
            lines.append(f"  {sch:<20}  delta = {delta:+.4f}  ({who} wins)")
    lines.append("")

    # Verdict
    rand_rf = float(sdf[(sdf["method"] == "B2 RF->theta") & (sdf["scheme"] == "random_5fold")]["median"].iloc[0])
    drug_rf = float(sdf[(sdf["method"] == "B2 RF->theta") & (sdf["scheme"] == "group_by_drug")]["median"].iloc[0])
    poly_rf = float(sdf[(sdf["method"] == "B2 RF->theta") & (sdf["scheme"] == "group_by_polymer")]["median"].iloc[0])
    rand_mlp = float(sdf[(sdf["method"] == "37c MLP") & (sdf["scheme"] == "random_5fold")]["median"].iloc[0])
    drug_mlp = float(sdf[(sdf["method"] == "37c MLP") & (sdf["scheme"] == "group_by_drug")]["median"].iloc[0])
    poly_mlp = float(sdf[(sdf["method"] == "37c MLP") & (sdf["scheme"] == "group_by_polymer")]["median"].iloc[0])

    lines.append("--- verdict ---")
    lines.append(f"  RF->theta degradation : random {rand_rf:.3f} -> drug {drug_rf:.3f} "
                 f"({drug_rf - rand_rf:+.3f}) -> polymer {poly_rf:.3f} ({poly_rf - rand_rf:+.3f})")
    lines.append(f"  37c MLP   degradation : random {rand_mlp:.3f} -> drug {drug_mlp:.3f} "
                 f"({drug_mlp - rand_mlp:+.3f}) -> polymer {poly_mlp:.3f} ({poly_mlp - rand_mlp:+.3f})")
    lines.append("")
    if drug_rf > drug_mlp and poly_rf > poly_mlp:
        lines.append("  RF MAINTAINS ADVANTAGE OUT-OF-DISTRIBUTION.")
        lines.append("  RF beats MLP under both group-by-drug and group-by-polymer schemes,")
        lines.append("  not just on random splits. The 38 finding generalizes: RF is the")
        lines.append("  better predictive model in this regime regardless of held-out type.")
    elif drug_rf < drug_mlp or poly_rf < poly_mlp:
        lines.append("  MIXED. RF wins random but MLP catches up on at least one OOD scheme.")
        lines.append("  Possibly RF was over-relying on memorized chemistry, and the MLP's")
        lines.append("  ODE bottleneck constrains it to physically plausible regions that")
        lines.append("  generalize better. Look at the specific scheme results above.")
    else:
        lines.append("  See per-scheme deltas above.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
