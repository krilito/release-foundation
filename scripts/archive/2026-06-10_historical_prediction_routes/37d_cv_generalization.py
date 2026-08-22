"""
37d - Cross-validated generalization audit for 37c hybrid model.

What this does:
    Re-runs the 37c hybrid architecture (formulation + early Q at 1, 3, 5,
    7 days -> theta -> simulator -> curve) under three cross-validation
    schemes, to disambiguate:

    (1) Random 5-fold: tests stability of the 0.91 median R^2 number
        from 37c's single 75/25 split.

    (2) GroupKFold by drug: all curves with the same drug (Drug MW +
        TPSA + LogP triplet) are kept together. A drug appears either
        entirely in train or entirely in test. Tests "can the model
        predict a curve for a drug it has never seen?"

    (3) GroupKFold by polymer: same for polymer (Polymer MW + LA/GA).
        Tests "can the model predict a curve for a polymer it has
        never seen?"

    The audit of cross321 found 259 curves -> only 74 unique drugs / 57
    unique polymers. So random splits very likely leak: test curves share
    drug/polymer with train curves. Group-K-fold removes that leakage at
    the cost of harder generalization targets.

Why this exists:
    The user asked the right question: "is 0.91 R^2 truly our
    generalization capability?" The honest answer is "no, it is an
    in-distribution random-split estimate." This script measures how
    much that 0.91 degrades under stricter held-out schemes.

Anti-self-deception:
    - All folds share the SAME training-loop code as 37c. The only
      difference is the train/test index sets.
    - Same hyperparameters as 37c (hidden=64, depth=3, epochs=300).
    - We report median R^2 per fold AND the distribution across folds.
    - Pooling test predictions across folds gives one aggregated R^2
      curve to compare directly to 37c's single-split 0.91.

Outputs:
    outputs/37d_cv_generalization/per_fold_results.csv
    outputs/37d_cv_generalization/pooled_per_curve.csv
    outputs/37d_cv_generalization/comparison_distribution.png
    outputs/37d_cv_generalization/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class HybridFormulationModel(nn.Module):
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


def _train_and_eval_one_fold(
    X_full: np.ndarray, theta_oracle: np.ndarray,
    idx_tr: np.ndarray, idx_te: np.ndarray,
    feature_cols: list[str], param_names: list[str],
    prior_low: torch.Tensor, prior_high: torch.Tensor,
    sim: PLGABiphasic, curve_map: dict[int, CurveRecord], fids: np.ndarray,
    epochs: int, lr: float, batch_size: int, hidden: int, depth: int,
    dropout: float, weight_decay: float, device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize features on train fold only (formulation cols come
    # first; early Q values last and don't need standardization)
    n_form = len(feature_cols)
    Xf = X_full[:, :n_form]
    mu = Xf[idx_tr].mean(axis=0, keepdims=True)
    sd = Xf[idx_tr].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Xfs = (Xf - mu) / sd
    X_std = np.concatenate([Xfs, X_full[:, n_form:]], axis=1)

    X_tr = torch.tensor(X_std[idx_tr], dtype=torch.float32, device=device)
    X_te = torch.tensor(X_std[idx_te], dtype=torch.float32, device=device)
    Theta_tr = torch.tensor(theta_oracle[idx_tr], dtype=torch.float32, device=device)
    Theta_te = torch.tensor(theta_oracle[idx_te], dtype=torch.float32, device=device)
    theta_mu = Theta_tr.mean(dim=0, keepdim=True)
    theta_sd = Theta_tr.std(dim=0, keepdim=True).clamp_min(1e-3)

    model = HybridFormulationModel(
        n_in=X_full.shape[1], n_params=len(param_names),
        prior_low=prior_low, prior_high=prior_high,
        hidden=hidden, depth=depth, dropout=dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = 10
    n_train = len(idx_tr)
    for epoch in range(epochs):
        scale = (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        for g in optim.param_groups:
            g["lr"] = lr * scale
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            sel = perm[s:s + batch_size]
            theta_b = model(X_tr[sel])
            tgt = (Theta_tr[sel] - theta_mu) / theta_sd
            pred = (theta_b - theta_mu) / theta_sd
            loss = ((pred - tgt) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    model.eval()
    with torch.no_grad():
        theta_te_pred = model(X_te).cpu().numpy()
    theta_median_train = np.median(theta_oracle[idx_tr], axis=0)

    rows: list[dict[str, object]] = []
    for i, j in enumerate(idx_te):
        fid = int(fids[j])
        c = curve_map[fid]
        theta_pred = theta_te_pred[i].astype(np.float64)
        theta_med = theta_median_train.astype(np.float64)
        theta_o = theta_oracle[j].astype(np.float64)
        q_pred = sim.simulate_numpy(theta_pred, c.t_obs)
        q_med = sim.simulate_numpy(theta_med, c.t_obs)
        q_o = sim.simulate_numpy(theta_o, c.t_obs)
        rows.append({
            "fid": fid,
            "r2_A_full": _r2(c.q_obs, q_pred),
            "r2_D_full": _r2(c.q_obs, q_med),
            "r2_E_full": _r2(c.q_obs, q_o),
        })
    return rows


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
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/37d_cv_generalization"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[37d] device: {device}", flush=True)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    prior_low = prior.low.float()
    prior_high = prior.high.float()
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
        t_grid_max_days=args.t_grid_max_days,
    )
    curve_map = {c.fid: c for c in curves}

    # Build inputs / targets / groups
    early_times = np.array(args.early_times, dtype=float)
    keep_rows: list[int] = []
    early_Q_list: list[np.ndarray] = []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        early_Q_list.append(_interp_at(c.t_obs, c.q_obs, early_times))
        keep_rows.append(i)
    df = df.loc[keep_rows].reset_index(drop=True)
    early_Q = np.stack(early_Q_list, axis=0).astype(np.float32)

    X_form = df[feature_cols].to_numpy(dtype=np.float32)
    X_full = np.concatenate([X_form, early_Q], axis=1)
    theta_oracle = df[param_names].to_numpy(dtype=np.float32)
    fids = df[FID_COL].to_numpy()
    drug_keys = df[DRUG_KEY_COLS].apply(tuple, axis=1).to_numpy()
    polymer_keys = df[POLYMER_KEY_COLS].apply(tuple, axis=1).to_numpy()
    n = len(df)
    n_drugs = pd.Series(drug_keys).nunique()
    n_polymers = pd.Series(polymer_keys).nunique()
    print(f"[37d] n_curves={n}, n_unique_drugs={n_drugs}, n_unique_polymers={n_polymers}",
          flush=True)

    schemes: list[tuple[str, np.ndarray]] = [
        ("random_5fold", np.arange(n)),  # placeholder; KFold below ignores groups
        ("group_by_drug", drug_keys),
        ("group_by_polymer", polymer_keys),
    ]

    all_fold_rows: list[dict[str, object]] = []
    pooled_pred_rows: list[dict[str, object]] = []

    for scheme_name, groups in schemes:
        print(f"\n[37d] === scheme: {scheme_name} ===", flush=True)
        if scheme_name == "random_5fold":
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(n)))
        else:
            # GroupKFold doesn't accept random_state; group order is deterministic
            splitter = GroupKFold(n_splits=args.n_folds)
            splits = list(splitter.split(np.arange(n), groups=groups))

        scheme_per_curve: list[dict[str, object]] = []
        for fold_idx, (tr, te) in enumerate(splits):
            print(f"[37d]   fold {fold_idx+1}/{len(splits)}: train n={len(tr)}, test n={len(te)}",
                  flush=True)
            rows = _train_and_eval_one_fold(
                X_full=X_full, theta_oracle=theta_oracle,
                idx_tr=tr, idx_te=te,
                feature_cols=feature_cols, param_names=param_names,
                prior_low=prior_low, prior_high=prior_high,
                sim=sim, curve_map=curve_map, fids=fids,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                hidden=args.hidden, depth=args.depth, dropout=args.dropout,
                weight_decay=args.weight_decay, device=device,
                seed=args.seed + fold_idx,
            )
            for r in rows:
                r2_A = r["r2_A_full"]; r2_D = r["r2_D_full"]
                scheme_per_curve.append({"scheme": scheme_name, "fold": fold_idx,
                                         **r, "delta_A_D": r2_A - r2_D})
                pooled_pred_rows.append({"scheme": scheme_name, "fold": fold_idx, **r})

            # fold-level summary
            fold_arr_A = np.array([row["r2_A_full"] for row in rows])
            fold_arr_D = np.array([row["r2_D_full"] for row in rows])
            all_fold_rows.append({
                "scheme": scheme_name, "fold": fold_idx, "n_test": len(rows),
                "median_r2_A": float(np.median(fold_arr_A)),
                "mean_r2_A": float(np.mean(fold_arr_A)),
                "frac_A_above_0.9": float((fold_arr_A >= 0.9).mean()),
                "frac_A_above_0": float((fold_arr_A >= 0).mean()),
                "median_r2_D": float(np.median(fold_arr_D)),
                "delta_median_A_minus_D": float(np.median(fold_arr_A) - np.median(fold_arr_D)),
            })
            print(f"[37d]     median R^2(A)={np.median(fold_arr_A):.4f}, "
                  f"frac>=0.9={float((fold_arr_A >= 0.9).mean()):.2f}, "
                  f"frac>=0={float((fold_arr_A >= 0).mean()):.2f}", flush=True)

    fold_df = pd.DataFrame(all_fold_rows)
    pooled_df = pd.DataFrame(pooled_pred_rows)
    fold_df.to_csv(args.out / "per_fold_results.csv", index=False)
    pooled_df.to_csv(args.out / "pooled_per_curve.csv", index=False)

    # Per-scheme pooled summary
    summary_rows: list[dict[str, object]] = []
    for scheme_name, _ in schemes:
        sub = pooled_df[pooled_df["scheme"] == scheme_name]
        a = sub["r2_A_full"].to_numpy()
        d = sub["r2_D_full"].to_numpy()
        e = sub["r2_E_full"].to_numpy()
        summary_rows.append({
            "scheme": scheme_name,
            "n_curves_evaluated": len(a),
            "median_A": float(np.median(a)),
            "mean_A": float(np.mean(a)),
            "p25_A": float(np.percentile(a, 25)),
            "p10_A": float(np.percentile(a, 10)),
            "frac_A_above_0.9": float((a >= 0.9).mean()),
            "frac_A_above_0.5": float((a >= 0.5).mean()),
            "frac_A_above_0": float((a >= 0).mean()),
            "median_D": float(np.median(d)),
            "median_E": float(np.median(e)),
            "delta_A_D_median": float(np.median(a) - np.median(d)),
        })
    summary_df = pd.DataFrame(summary_rows)

    # Distribution plot: one box per scheme
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    data = []
    labels = []
    for scheme_name, _ in schemes:
        sub = pooled_df[pooled_df["scheme"] == scheme_name]
        data.append(sub["r2_A_full"].to_numpy())
        labels.append(f"{scheme_name}\n(n={len(sub)})")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    colors = ["lightblue", "lightsalmon", "khaki"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.axhline(0.91, color="tab:red", lw=1.0, ls="--", label="37c single-split = 0.91")
    ax.axhline(0.9, color="gray", lw=0.5, ls=":")
    ax.axhline(0.0, color="gray", lw=0.5, ls="-")
    ax.set_ylabel("full-curve R^2 (pooled across folds)")
    ax.set_title("37d: how does 37c's 0.91 hold up under harder splits?")
    ax.legend()
    ax.set_ylim(min(-1.5, float(pooled_df["r2_A_full"].min()) - 0.1), 1.05)
    fig.savefig(args.out / "comparison_distribution.png", dpi=150)
    plt.close(fig)

    # ---- summary text ----
    lines = [
        "=== 37d -- CV generalization audit ===",
        "",
        f"n_curves            : {n}",
        f"unique drugs        : {n_drugs}  (avg {n / n_drugs:.1f} curves per drug)",
        f"unique polymers     : {n_polymers}  (avg {n / n_polymers:.1f} curves per polymer)",
        f"early times         : {args.early_times}",
        f"hyperparams         : hidden={args.hidden}, depth={args.depth}, "
        f"epochs={args.epochs}, lr={args.lr}, dropout={args.dropout}",
        "",
        "--- per-scheme aggregated R^2 (pooled across all test folds) ---",
        f"  {'scheme':<22}  {'n_eval':>6}  {'median':>7}  {'mean':>7}  "
        f"{'p25':>7}  {'p10':>7}  {'>=.9':>5}  {'>=.5':>5}  {'>=0':>5}",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"  {str(r['scheme']):<22}  {int(r['n_curves_evaluated']):>6}  "
            f"{r['median_A']:>7.4f}  {r['mean_A']:>7.4f}  "
            f"{r['p25_A']:>7.4f}  {r['p10_A']:>7.4f}  "
            f"{r['frac_A_above_0.9']:>5.2f}  {r['frac_A_above_0.5']:>5.2f}  {r['frac_A_above_0']:>5.2f}"
        )
    lines.append("")
    lines.append("--- per-fold variability (median R^2 per fold) ---")
    for scheme_name, _ in schemes:
        sub = fold_df[fold_df["scheme"] == scheme_name]["median_r2_A"].to_numpy()
        lines.append(
            f"  {scheme_name:<22}  per-fold medians: {[f'{x:.3f}' for x in sub.tolist()]}  "
            f"mean of medians = {float(sub.mean()):.4f}, "
            f"range = [{float(sub.min()):.3f}, {float(sub.max()):.3f}]"
        )
    lines.append("")
    lines.append("--- comparison to 37c single-split (median R^2 = 0.914) ---")
    rand_pooled_median = float(summary_df.set_index("scheme").loc["random_5fold", "median_A"])
    drug_pooled_median = float(summary_df.set_index("scheme").loc["group_by_drug", "median_A"])
    poly_pooled_median = float(summary_df.set_index("scheme").loc["group_by_polymer", "median_A"])
    lines.append(f"  random 5-fold  median R^2 : {rand_pooled_median:.4f}  "
                 f"(delta vs 37c: {rand_pooled_median - 0.914:+.4f})")
    lines.append(f"  group-by-drug  median R^2 : {drug_pooled_median:.4f}  "
                 f"(delta vs 37c: {drug_pooled_median - 0.914:+.4f})")
    lines.append(f"  group-by-poly  median R^2 : {poly_pooled_median:.4f}  "
                 f"(delta vs 37c: {poly_pooled_median - 0.914:+.4f})")
    lines.append("")

    lines.append("--- interpretation ---")
    if rand_pooled_median >= 0.85 and drug_pooled_median < 0.6:
        lines.append("  IN-DISTRIBUTION GOOD, OUT-OF-DISTRIBUTION BAD.")
        lines.append("  Random 5-fold confirms 37c's 0.91 is robust within distribution. But")
        lines.append("  group-by-drug R^2 collapses, meaning the model cannot predict curves")
        lines.append("  for unseen drugs. 37c's number reflects 'interpolation within known")
        lines.append("  chemistries', not 'extrapolation to new chemistries'. For deployment")
        lines.append("  with NEW drugs, the model is much weaker than 0.91 suggests.")
    elif rand_pooled_median >= 0.85 and drug_pooled_median >= 0.6:
        lines.append("  REASONABLE GENERALIZATION.")
        lines.append("  Both random and group-by-drug medians stay above 0.6. The model has")
        lines.append("  some real chemistry generalization, not just memorization. 37c's 0.91")
        lines.append("  is an in-distribution upper bound; group-by-drug is a more")
        lines.append("  deployment-realistic estimate.")
    elif rand_pooled_median < 0.85:
        lines.append("  37c WAS A LUCKY SINGLE SPLIT.")
        lines.append("  Random 5-fold median falls below 0.85, so the 0.91 figure was a")
        lines.append("  favorable single-split estimate. Generalization story needs to be")
        lines.append("  retold with the CV number as the headline.")
    else:
        lines.append("  MIXED. See per-scheme stats above.")

    lines.append("")
    lines.append("--- caveats ---")
    lines.append("  - cross321 only. internal181 cannot be used (no formulation labels).")
    lines.append("  - With ~74 unique drugs / 57 polymers, group-by-X 5-fold means")
    lines.append("    ~15 drugs / 11 polymers per test fold. Reasonable but not large.")
    lines.append("  - All models use theta-loss training (the identifiability proxy). A")
    lines.append("    proper curve-loss end-to-end pass would tighten the numbers further")
    lines.append("    but is too slow on CPU at this scale.")
    lines.append("  - The simulator runs in float64 with simulate_numpy (DOP853); no")
    lines.append("    autograd is needed at evaluation, only at training, but here all")
    lines.append("    training is theta-loss so no ODE-autograd is invoked.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
