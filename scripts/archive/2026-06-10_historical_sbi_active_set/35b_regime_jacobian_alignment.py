"""
35b - Regime <-> local Jacobian principal-direction alignment.

What this does:
    For each curve that has a regime label from script 33 and a full-fit
    theta from script 29, compute the local Jacobian dQ/dtheta at the
    full-fit theta (scaled by prior range, as in script 30) and extract
    the right singular vectors. Then ask two questions:

    (1) ALIGNMENT: For each regime, does the average |v1| pattern across
        member curves match the regime's modal k=4 active set from
        script 33? If yes, the discrete "active set" diagnostic and the
        continuous "local Jacobian direction" diagnostic agree, and the
        modifier-set/backbone story has both algebraic and geometric
        backing.

    (2) SEPARATION: In the space of unit-norm v1 vectors (sign-aligned),
        do curves of the same regime sit closer to each other than to
        curves of other regimes? Measure silhouette under cosine
        distance (using 1 - |cos| so that v and -v are equivalent).
        Compare to a permutation null where regime labels are shuffled.
        If observed silhouette beats null 95th percentile, regime
        partitions the v1 space beyond chance.

    Also reports the (1) and (2) statistics computed on the top 3
    singular vectors aggregated (v1, v2, v3 stacked) so that low-rank
    structure is included, not only the top direction.

Why this exists:
    Scripts 32-34 + 35a established that regimes carry information
    beyond pair coupling, marginals, observation window, and formulation
    descriptors. This script tests whether the regime structure
    corresponds to actual mechanism subspaces in parameter space, or
    whether the regimes are merely greedy-selection labels with no
    geometric backing.

Anti-self-deception:
    - Jacobian is computed at the full-fit theta. If the fit is poor
      (e.g. boundary-hit dominated), the local direction may be
      artificial. We filter on full_r2 >= --fit-filter-r2 (default 0.95).
    - v1 is sign-aligned (largest |component| positive). Without this,
      cosine clusters would split each regime in two by sign flips.
    - The mean |v1| pattern per regime is compared to the modal active
      set using Spearman rank correlation (robust to scale). We also
      report which parameters are mismatched.
    - The silhouette null is a label permutation; it tests "are the
      regime labels informative of the v1 vectors", not "are there
      clusters in v1 space". Those are different questions and we only
      claim the first.

Outputs:
    outputs/35b_regime_jacobian_alignment/per_curve_v1.csv
    outputs/35b_regime_jacobian_alignment/regime_mean_abs_v1.csv
    outputs/35b_regime_jacobian_alignment/alignment_table.csv
    outputs/35b_regime_jacobian_alignment/silhouette_test.txt
    outputs/35b_regime_jacobian_alignment/regime_v1_heatmap.png
    outputs/35b_regime_jacobian_alignment/regime_alignment_scatter.png
    outputs/35b_regime_jacobian_alignment/summary.txt
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_plga_181  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


# Lightweight re-implementations of the helpers in script 30 that avoid
# importing posterior.py (which transitively requires sbi, not installed
# in this environment). Quality filtering is delegated to script 29's
# upstream filter -- any fid that has both a regime assignment and a
# full-fit theta is treated as valid.

@dataclass
class CurveRecord:
    dataset: str
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray


_321_RENAME = {
    "Formulation Index": "fid",
}


def _load_internal_records(csv_path: Path) -> list[CurveRecord]:
    curves = load_plga_181(csv_path)
    return [
        CurveRecord(
            dataset="internal181",
            fid=int(c.formulation_id),
            t_obs=c.t.numpy().astype(float),
            q_obs=np.clip(c.Q.numpy().astype(float), 0.0, 1.0),
        )
        for c in curves
    ]


def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
) -> list[CurveRecord]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns=_321_RENAME)
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    df = df[df["fid"].astype(int).isin(matched_fids)].copy()
    out: list[CurveRecord] = []
    for fid in sorted(df["fid"].unique().tolist()):
        g = df[df["fid"] == fid].sort_values("Time")
        # collapse duplicate time points (mean)
        g = g.groupby("Time", as_index=False).agg({"Release": "mean"})
        t_obs = g["Time"].to_numpy(dtype=float)
        q_obs = np.clip(g["Release"].to_numpy(dtype=float), 0.0, 1.0)
        if len(t_obs) < 3:
            continue
        out.append(CurveRecord(dataset="cross321", fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    return out


def _curve_jacobian(
    sim: PLGABiphasic,
    theta_np: np.ndarray,
    t_obs_np: np.ndarray,
    prior_range: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Batched forward-difference Jacobian dQ/dtheta_scaled at theta.

    Returns a (T, P) matrix in *prior-scaled* parameter units (each column
    multiplied by prior_range[p]) so the SVD directions are dimensionless
    and comparable across parameters.

    Implementation: stack the baseline theta and 9 forward-perturbed thetas
    into one (1+P, P) batch and run a single batched simulate(...) call.
    This avoids autograd through the ODE solver entirely (which was too
    slow per-time-point on long-window cross321 curves) and uses the
    simulator's native batch axis (~10x faster than serial perturbations).
    """
    theta = torch.tensor(theta_np, dtype=torch.float32)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32)
    P = len(theta_np)
    # Build (1+P, P) batch: row 0 = baseline; rows 1..P = perturbed
    batch = theta.unsqueeze(0).repeat(1 + P, 1).clone()
    for p in range(P):
        batch[1 + p, p] = batch[1 + p, p] + eps
    with torch.no_grad():
        Q = sim.simulate(batch, t_obs).cpu().numpy()  # (1+P, T)
    q0 = Q[0]
    jac = ((Q[1:] - q0[None, :]) / eps).T * prior_range[None, :]  # (T, P)
    return jac.astype(np.float64)


DATASET_COL = "dataset"
FID_COL = "fid"


def _parse_active_set(cell: object) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [token.strip() for token in cell.split(",") if token.strip()]


def _sign_align(v: np.ndarray) -> np.ndarray:
    """Flip sign so largest |component| is positive."""
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        return -v
    return v


def _abs_cosine_distance_matrix(V: np.ndarray) -> np.ndarray:
    """V: (n, 9) unit-norm rows. Returns (n,n) 1 - |cos| matrix."""
    # Cosine: V @ V.T (assuming unit-norm rows)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    Vn = V / np.where(norms > 1e-12, norms, 1.0)
    cos = Vn @ Vn.T
    np.clip(cos, -1.0, 1.0, out=cos)
    D = 1.0 - np.abs(cos)
    np.fill_diagonal(D, 0.0)
    return D


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
        "--active-set-csv",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/per_curve_active_set.csv"),
    )
    ap.add_argument(
        "--internal-data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
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
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-quality", type=str, default="high", choices=("high", "medium", "low"))
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--exclude-regimes", type=int, nargs="*", default=[1])
    ap.add_argument("--k-active", type=int, default=4)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/35b_regime_jacobian_alignment")
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    prior_range = (prior.high - prior.low).numpy()
    param_names = list(sim.param_names)

    # Load curves
    curves = [
        *_load_internal_records(args.internal_data),
        *_load_external_records(
            xlsx_path=args.cross_doi_data,
            matched_fids_csv=args.matched_fids_csv,
        ),
    ]
    curve_map = {(c.dataset, int(c.fid)): c for c in curves}

    # Load regime assignments and full-fit theta
    df_reg = pd.read_csv(args.regime_csv)
    df_reg = df_reg[~df_reg["regime"].isin(args.exclude_regimes)].copy()
    df_bank = pd.read_csv(args.full_fit_bank)
    df_bank = df_bank[
        np.isfinite(df_bank["full_r2"]) & (df_bank["full_r2"] >= args.fit_filter_r2)
    ].copy()
    df = df_reg.merge(
        df_bank[[DATASET_COL, FID_COL, "full_r2"] + param_names],
        on=[DATASET_COL, FID_COL],
        how="inner",
    )
    print(f"[35b] merged regime+bank: {len(df)} curves "
          f"(reg={len(df_reg)}, bank>={args.fit_filter_r2} = {len(df_bank)})", flush=True)
    print(f"[35b] curves loaded: internal={sum(1 for c in curves if c.dataset=='internal181')}, "
          f"cross={sum(1 for c in curves if c.dataset=='cross321')}", flush=True)

    # Compute Jacobian SVD per curve
    rows: list[dict[str, object]] = []
    V1_list: list[np.ndarray] = []
    V_top3_list: list[np.ndarray] = []
    keep_idx: list[int] = []
    n_total = len(df)
    print(f"[35b] computing Jacobians for {n_total} curves", flush=True)
    import time
    t_start = time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        if (i % 10) == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / max(i, 1)) * (n_total - i) if i > 0 else float("nan")
            print(f"[35b] {i}/{n_total}  elapsed={elapsed:.1f}s  eta={eta:.0f}s", flush=True)
        key = (str(r[DATASET_COL]), int(r[FID_COL]))
        rec = curve_map.get(key)
        if rec is None:
            continue
        theta = r[param_names].to_numpy(dtype=float)
        # Cap t_obs to avoid pathologically long ODE integrations on a few
        # cross321 curves with t_max up to 160 days. 90 days is the standard
        # horizon used elsewhere in this codebase (script 30 default).
        t_obs_use = rec.t_obs[rec.t_obs <= args.t_grid_max_days]
        if len(t_obs_use) < 3:
            continue
        try:
            jac = _curve_jacobian(sim, theta, t_obs_use, prior_range)
            U, S, Vt = np.linalg.svd(jac, full_matrices=False)
        except Exception as e:  # noqa: BLE001
            print(f"[35b] skip ({key}): {e}", flush=True)
            continue
        if Vt.shape[0] < 1:
            continue
        v1 = _sign_align(np.asarray(Vt[0], dtype=float))
        V1_list.append(v1)
        n_keep = min(3, Vt.shape[0])
        Vtop = np.stack([_sign_align(np.asarray(Vt[k], dtype=float)) for k in range(n_keep)], axis=0)
        # Pad to 3 if needed (very low rank)
        if Vtop.shape[0] < 3:
            pad = np.zeros((3 - Vtop.shape[0], len(param_names)))
            Vtop = np.concatenate([Vtop, pad], axis=0)
        V_top3_list.append(Vtop)
        keep_idx.append(int(i))
        row_out = {
            DATASET_COL: r[DATASET_COL],
            FID_COL: int(r[FID_COL]),
            "regime": int(r["regime"]),
            "full_r2": float(r["full_r2"]),
            "sv1": float(S[0]),
            "sv2": float(S[1]) if len(S) > 1 else float("nan"),
            "sv3": float(S[2]) if len(S) > 2 else float("nan"),
        }
        for j, name in enumerate(param_names):
            row_out[f"v1_{name}"] = float(v1[j])
        rows.append(row_out)

    if not rows:
        raise RuntimeError("no curves survived Jacobian computation")
    per_curve = pd.DataFrame(rows)
    per_curve.to_csv(args.out / "per_curve_v1.csv", index=False)
    V1 = np.stack(V1_list, axis=0)
    V_top3 = np.stack(V_top3_list, axis=0)  # (n, 3, 9)
    labels = per_curve["regime"].astype(int).to_numpy()

    # Sub-set: include the active-set modal pattern per regime from regime_csv
    # Use the per_curve_active_set.csv to compute regime modal at k=4
    df_as = pd.read_csv(args.active_set_csv)
    active_col = f"active_set_k{args.k_active}"
    df_as["_aset"] = df_as[active_col].apply(_parse_active_set)
    df_as_keep = df_as.merge(
        per_curve[[DATASET_COL, FID_COL, "regime"]],
        on=[DATASET_COL, FID_COL],
        how="inner",
    )
    p_index = {n: i for i, n in enumerate(param_names)}
    regime_active_grid: dict[int, np.ndarray] = {}
    for c, sub in df_as_keep.groupby("regime"):
        M = np.zeros((len(sub), len(param_names)), dtype=float)
        for i, a in enumerate(sub["_aset"]):
            for name in a:
                if name in p_index:
                    M[i, p_index[name]] = 1.0
        regime_active_grid[int(c)] = M.mean(axis=0)

    # --- Test 1: alignment between regime-mean |v1| and regime modal active set
    regime_mean_abs_v1: dict[int, np.ndarray] = {}
    alignment_rows: list[dict[str, object]] = []
    for c in sorted(np.unique(labels).tolist()):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_abs_v1 = np.mean(np.abs(V1[mask]), axis=0)
        regime_mean_abs_v1[c] = mean_abs_v1
        active_frac = regime_active_grid.get(c, np.zeros(len(param_names)))
        if active_frac.sum() == 0:
            sp_r, sp_p = (float("nan"), float("nan"))
            pe_r, pe_p = (float("nan"), float("nan"))
        else:
            sp_r, sp_p = spearmanr(mean_abs_v1, active_frac)
            pe_r = float(np.corrcoef(mean_abs_v1, active_frac)[0, 1])
            pe_p = float("nan")
        # Top-4 by |v1| vs modal active set (set overlap)
        top4_v1 = set(np.argsort(-mean_abs_v1)[:args.k_active].tolist())
        top4_active = set(np.argsort(-active_frac)[:args.k_active].tolist())
        jaccard = len(top4_v1 & top4_active) / max(len(top4_v1 | top4_active), 1)
        alignment_rows.append({
            "regime": c,
            "n_curves": int(mask.sum()),
            "spearman_r": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "topk_overlap_count": int(len(top4_v1 & top4_active)),
            "topk_jaccard": float(jaccard),
        })
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(args.out / "alignment_table.csv", index=False)
    pd.DataFrame(regime_mean_abs_v1, index=param_names).to_csv(
        args.out / "regime_mean_abs_v1.csv"
    )

    # --- Test 2: silhouette of regime labels in v1 space, vs label-shuffle null
    D = _abs_cosine_distance_matrix(V1)
    if len(np.unique(labels)) >= 2:
        obs_sil = float(silhouette_score(D, labels, metric="precomputed"))
    else:
        obs_sil = float("nan")
    null_sils = np.empty(args.n_null)
    perm_labels = labels.copy()
    for b in range(args.n_null):
        rng.shuffle(perm_labels)
        try:
            null_sils[b] = silhouette_score(D, perm_labels, metric="precomputed")
        except ValueError:
            null_sils[b] = np.nan
    null_mean = float(np.nanmean(null_sils))
    null_p95 = float(np.nanpercentile(null_sils, 95))
    p_value = float((null_sils >= obs_sil).mean()) if not np.isnan(obs_sil) else float("nan")
    sil_lines = [
        "Silhouette in v1 (abs cosine) space, regime labels:",
        f"  observed       : {obs_sil:.4f}",
        f"  null mean      : {null_mean:.4f}",
        f"  null 95th pct  : {null_p95:.4f}",
        f"  empirical p    : {p_value:.4f}",
        f"  n_curves       : {len(labels)}, n_regimes: {len(np.unique(labels))}",
        f"  n_null perms   : {args.n_null}",
    ]

    # --- Test 2b: same but on top-3 SVD vectors concatenated (3*9 = 27 dim)
    V_top3_flat = V_top3.reshape(V_top3.shape[0], -1)
    # Normalize per row
    norms = np.linalg.norm(V_top3_flat, axis=1, keepdims=True)
    Vn = V_top3_flat / np.where(norms > 1e-12, norms, 1.0)
    D3 = _abs_cosine_distance_matrix(Vn)
    obs_sil3 = float(silhouette_score(D3, labels, metric="precomputed"))
    null_sils3 = np.empty(args.n_null)
    for b in range(args.n_null):
        rng.shuffle(perm_labels)
        try:
            null_sils3[b] = silhouette_score(D3, perm_labels, metric="precomputed")
        except ValueError:
            null_sils3[b] = np.nan
    null_mean3 = float(np.nanmean(null_sils3))
    null_p95_3 = float(np.nanpercentile(null_sils3, 95))
    p_value3 = float((null_sils3 >= obs_sil3).mean()) if not np.isnan(obs_sil3) else float("nan")
    sil_lines.extend([
        "",
        "Silhouette in top-3 SVD-vector concat space (abs cosine):",
        f"  observed       : {obs_sil3:.4f}",
        f"  null mean      : {null_mean3:.4f}",
        f"  null 95th pct  : {null_p95_3:.4f}",
        f"  empirical p    : {p_value3:.4f}",
    ])
    (args.out / "silhouette_test.txt").write_text("\n".join(sil_lines), encoding="utf-8")

    # --- Plot: regime x param mean |v1| heatmap, with modal active set overlay
    regimes = sorted(regime_mean_abs_v1.keys())
    grid_v1 = np.stack([regime_mean_abs_v1[c] for c in regimes], axis=0)
    grid_act = np.stack([regime_active_grid.get(c, np.zeros(len(param_names))) for c in regimes], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(regimes) + 2.4),
                             constrained_layout=True, sharey=True)
    im0 = axes[0].imshow(grid_v1, cmap="magma", aspect="auto")
    axes[0].set_xticks(range(len(param_names)))
    axes[0].set_xticklabels(param_names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(regimes)))
    axes[0].set_yticklabels([f"R{c}" for c in regimes])
    axes[0].set_title("regime mean |v1| (local Jacobian direction)")
    for i in range(len(regimes)):
        for j in range(len(param_names)):
            axes[0].text(j, i, f"{grid_v1[i, j]:.2f}", ha="center", va="center",
                         color="white" if grid_v1[i, j] < 0.5 * grid_v1.max() else "black",
                         fontsize=7)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="mean |v1| component")

    im1 = axes[1].imshow(grid_act, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_xticks(range(len(param_names)))
    axes[1].set_xticklabels(param_names, rotation=45, ha="right")
    axes[1].set_yticks(range(len(regimes)))
    axes[1].set_yticklabels([f"R{c}" for c in regimes])
    axes[1].set_title(f"regime active-set inclusion (k={args.k_active})")
    for i in range(len(regimes)):
        for j in range(len(param_names)):
            axes[1].text(j, i, f"{grid_act[i, j]:.2f}", ha="center", va="center",
                         color="white" if grid_act[i, j] < 0.45 else "black",
                         fontsize=7)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="inclusion fraction")
    fig.suptitle("Geometric (Jacobian) vs algebraic (active set) per-regime parameter pattern",
                 fontsize=13)
    fig.savefig(args.out / "regime_v1_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Scatter: alignment correlation per regime
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    ax.bar([f"R{int(r['regime'])}" for _, r in alignment_df.iterrows()],
           alignment_df["spearman_r"].to_numpy(), color="teal")
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Spearman r between mean |v1| and active-set inclusion")
    ax.set_title("Regime alignment: geometry (Jacobian) vs algebra (active set)")
    for i, (_, r) in enumerate(alignment_df.iterrows()):
        ax.text(i, r["spearman_r"] + 0.03,
                f"jacc={r['topk_jaccard']:.2f}\nn={int(r['n_curves'])}",
                ha="center", va="bottom", fontsize=8)
    fig.savefig(args.out / "regime_alignment_scatter.png", dpi=150)
    plt.close(fig)

    # --- summary.txt
    lines = [
        "=== 35b -- regime <-> local Jacobian principal-direction alignment ===",
        "",
        f"n_curves with Jacobian + regime : {len(per_curve)}",
        f"fit_filter_r2                    : {args.fit_filter_r2}",
        f"excluded regimes                 : {args.exclude_regimes}",
        "",
        "--- per-regime alignment (mean |v1| vs modal active set) ---",
        f"  {'regime':>6}  {'n':>4}  {'spearman_r':>11}  {'spearman_p':>11}  "
        f"{'pearson_r':>10}  {'top-k overlap':>14}  {'jaccard':>8}",
    ]
    for _, r in alignment_df.iterrows():
        lines.append(
            f"  R{int(r['regime']):>4}  {int(r['n_curves']):>4}  "
            f"{r['spearman_r']:>11.3f}  {r['spearman_p']:>11.4f}  "
            f"{r['pearson_r']:>10.3f}  {int(r['topk_overlap_count']):>14}  "
            f"{r['topk_jaccard']:>8.2f}"
        )
    lines.append("")
    n_aligned = int((alignment_df["spearman_r"] > 0.5).sum())
    lines.append(f"  regimes with Spearman r > 0.5 : {n_aligned} / {len(alignment_df)}")
    lines.append("")

    lines.extend(sil_lines)
    lines.append("")

    # Verdict
    sil_pass = (not np.isnan(obs_sil)) and obs_sil > null_p95
    sil3_pass = (not np.isnan(obs_sil3)) and obs_sil3 > null_p95_3
    alignment_pass = n_aligned >= max(2, len(alignment_df) // 2)
    lines.append("--- verdict ---")
    lines.append(f"  silhouette v1 PASS (obs > null p95)        : {sil_pass}")
    lines.append(f"  silhouette top-3 PASS (obs > null p95)     : {sil3_pass}")
    lines.append(f"  >=half regimes Spearman r>0.5              : {alignment_pass}")
    lines.append("")

    if sil_pass and alignment_pass:
        lines.append("  GEOMETRY MATCHES ALGEBRA.")
        lines.append("  Regime labels separate curves in the local Jacobian principal-direction")
        lines.append("  space beyond chance, AND per-regime mean |v1| pattern correlates with")
        lines.append("  the regime's modal active set. The 'backbone + modifier-set' hypothesis")
        lines.append("  has both algebraic and geometric backing. Regime-conditional")
        lines.append("  parameterization (or a latent-driver model whose per-curve loading is")
        lines.append("  regime-aware) is now defensible.")
    elif sil_pass and not alignment_pass:
        lines.append("  GEOMETRY SEPARATES BUT DOES NOT MATCH ACTIVE-SET PATTERN.")
        lines.append("  Regimes carve up the v1 space but the within-regime mean |v1| does not")
        lines.append("  recover the modal active set. Likely: regimes encode mechanism")
        lines.append("  direction at the curve level but the greedy active-set diagnostic")
        lines.append("  picks a different set than the dominant direction would suggest.")
        lines.append("  Implication: trust the Jacobian story, not the active-set labels.")
    elif (not sil_pass) and alignment_pass:
        lines.append("  ACTIVE-SET PATTERN MATCHES MEAN |v1| BUT NO SEPARATION IN v1 SPACE.")
        lines.append("  Per-regime averages line up but individual curves are scattered. The")
        lines.append("  regime label is then an average descriptor without per-curve geometric")
        lines.append("  meaning. A regime-conditional model would learn the right average")
        lines.append("  pattern but not a per-curve mechanism subspace.")
    else:
        lines.append("  NEITHER TEST PASSES.")
        lines.append("  Regimes do not correspond to distinct dominant Jacobian directions.")
        lines.append("  Combined with 35a passing, this means: regime carries some information")
        lines.append("  but it is NOT captured by the local linear geometry around the full-fit")
        lines.append("  theta. Possible causes: (a) the relevant geometry is higher-rank than")
        lines.append("  the top SV, (b) greedy active-set selection is partly artifactual,")
        lines.append("  (c) regimes encode global rather than local structure. Revisit before")
        lines.append("  committing to a regime-conditional architecture.")
    lines.append("")
    lines.append("--- caveats ---")
    lines.append("  - Jacobian taken at full-fit theta; boundary-hit curves may have unreliable")
    lines.append("    local geometry. fit_filter_r2 reduces but does not eliminate this.")
    lines.append("  - Silhouette null is a label-shuffle on the same v1 vectors; tests whether")
    lines.append("    regime labels are informative for v1 clustering, not whether clusters")
    lines.append("    exist in v1 space.")
    lines.append("  - v1 is sign-ambiguous; we use abs cosine. This collapses physically")
    lines.append("    opposite mechanism directions; if a regime contains both +v and -v for")
    lines.append("    the same parameter pattern, they will be merged.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
