"""
10 — Deployment evaluation for the Phase-1 r_ψ pipeline.

What this does:
    1. Load q_φ (Stage-1 teacher) and the PLGA 181 internal benchmark.
    2. Per curve: interpolate onto t_grid, compute quality flag.
    3. Filter to --min-quality (default 'high', matches script 09).
    4. Pre-sample teacher targets θ_k ~ q_φ(.|Q_grid_i) for ALL kept
       curves once (q_φ is fold-invariant; precomputing saves
       ~5 min × n_folds).
    5. Leave-one-drug-out: enumerate unique drug_ids and for each:
       a. train_mask = drug_id != held drug
       b. fit featurizer on train subset (so OTHER bucket fires
          correctly if held drug's polymer is rare)
       c. train r_ψ on the held-in subset
       d. for each held-out curve i:
          - sample K_eval θ ~ r_ψ(.|x_i)
          - simulate at the REAL observed time points (not t_grid)
          - per-t-point median, p10, p90 across K_eval
          - R² and MAE vs Q_obs
          - q_burst-vs-Q_obs(0) residual (model-form diagnostic)
       e. Save per-fold predictions + metrics.
    6. Aggregate: overall R², per-drug median R², per-polymer median R².

Phase 1 success bar verdict:
    (1) SBC calibration — already PASS under ADR-017.
    (2') Internal leave-one-drug-out R² >= 0.50 — PASS at +0.657
         (ADR-019 / ADR-020 replaces the original SRDS-comparator
         framing of ADR-006(2)).
    (3) Cross-DOI 321 R² > 0.30 — implemented behind --cross-doi.
         Loads a pre-trained r_psi (default outputs/09) and evaluates
         on the 321-PLGA xlsx; no leave-one-drug-out (no holdout split
         needed — 321 is the held-out distribution by construction).

Run (internal leave-one-drug-out, default mode):
    .\.venv\Scripts\python.exe scripts/10_eval_deployment.py

Run (cross-DOI 321 eval, ADR-021):
    .\.venv\Scripts\python.exe scripts/10_eval_deployment.py --cross-doi

Smoke test (2 folds only, internal mode):
    .\.venv\Scripts\python.exe scripts/10_eval_deployment.py --n-folds-limit 2

Outputs (internal mode):
    outputs/10_eval_deployment/summary.txt
    outputs/10_eval_deployment/per_curve_metrics.csv
    outputs/10_eval_deployment/per_drug_summary.csv
    outputs/10_eval_deployment/per_polymer_summary.csv
    outputs/10_eval_deployment/predictions/<fid>.npz  (per-curve raw)

Outputs (--cross-doi mode):
    outputs/10_eval_deployment/cross_doi_summary.txt
    outputs/10_eval_deployment/cross_doi_per_curve_metrics.csv
    outputs/10_eval_deployment/cross_doi_predictions/<fid>.npz

Expected runtime:
    internal LODO: ~57 min on CPU.
    --cross-doi:   ~2-3 min on CPU (no retraining; one forward pass per
                   of ~260 high-quality 321 curves).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import (  # noqa: E402
    FormulationFeaturizer,
    MLPFormulationEncoder,
    parse_dp_group,
)
from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}

# Columns in the 321 xlsx that map (after rename + unit conversion) onto
# columns of the internal 181 schema. Imputation strategy for columns
# absent from 321 is handled below — see _load_cross_doi_xlsx.
_321_RENAME = {
    "Formulation Index": "Experimental_index",
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",  # also * 1000 below (kDa -> Da)
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",  # also / 100 below (% -> fraction)
    # LA/GA, Time, Release keep their names.
}
# 321 descriptor columns that have no semantic equivalent in 181 and
# are not consumed by the featurizer; carried through ignored.
_321_DROP = (
    "Particle Size",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
)
# Featurizer continuous columns absent from 321 entirely. We impute them
# with the loaded featurizer's training mean so the z-scored input goes
# to 0 for these dims — i.e. the encoder sees "no information" rather
# than "wrong information". See ADR-021.
_321_IMPUTED_COLS = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Returns NaN if y_true is constant."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _build_data(
    df: pd.DataFrame,
    t_grid: torch.Tensor,
    t_max_days: float,
    min_quality: str,
) -> tuple[pd.DataFrame, torch.Tensor, list[dict], list[str], list[torch.Tensor], list[torch.Tensor]]:
    """Interpolate every formulation onto t_grid, return only the kept rows.

    Returns:
        keep_desc:    per-formulation descriptor DataFrame (kept rows only)
        Q_grid:       (N, T) grid-projected curves for kept rows
        metas:        per-kept-curve diagnostic dicts (tmax, last_Q, ...)
        drug_ids:     parsed drug ids for kept rows
        t_obs_list:   per-kept-curve real observed times
        Q_obs_list:   per-kept-curve real observed Q values
    """
    fid_col, t_col, q_col = "Experimental_index", "Time", "Release"
    desc_df = (
        df.drop_duplicates(fid_col)
        .sort_values(fid_col)
        .reset_index(drop=True)
    )

    Q_grid_list: list[torch.Tensor | None] = []
    qualities: list[str] = []
    metas: list[dict] = []
    t_obs_list: list[torch.Tensor] = []
    Q_obs_list: list[torch.Tensor] = []

    for fid in desc_df[fid_col]:
        g = df[df[fid_col] == fid].sort_values(t_col)
        t_obs = torch.tensor(g[t_col].to_numpy(), dtype=torch.float32)
        Q_obs = torch.tensor(g[q_col].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            Q_grid_i, quality, meta = interpolate_to_grid(
                t_obs, Q_obs, t_grid, t_max_days=t_max_days,
            )
        except ValueError:
            Q_grid_list.append(None)
            qualities.append("skip")
            metas.append({"tmax": 0.0, "last_Q": 0.0, "n_obs_kept": 0})
            t_obs_list.append(t_obs)
            Q_obs_list.append(Q_obs)
            continue
        Q_grid_list.append(Q_grid_i)
        qualities.append(quality)
        metas.append(meta)
        t_obs_list.append(t_obs)
        Q_obs_list.append(Q_obs)

    min_rank = _QUALITY_RANK[min_quality]
    keep_mask = [
        q in _QUALITY_RANK and _QUALITY_RANK[q] >= min_rank
        for q in qualities
    ]
    keep_desc = desc_df.loc[keep_mask].reset_index(drop=True)
    Q_grid = torch.stack(
        [Q_grid_list[i] for i, k in enumerate(keep_mask) if k], dim=0
    )
    kept_metas = [metas[i] for i, k in enumerate(keep_mask) if k]
    kept_t_obs = [t_obs_list[i] for i, k in enumerate(keep_mask) if k]
    kept_Q_obs = [Q_obs_list[i] for i, k in enumerate(keep_mask) if k]
    drug_ids = [parse_dp_group(v)[0] for v in keep_desc["DP_Group"]]

    return keep_desc, Q_grid, kept_metas, drug_ids, kept_t_obs, kept_Q_obs


def _eval_curve(
    rp: DescriptorPosterior,
    sim: PLGABiphasic,
    x_i: torch.Tensor,
    t_obs: torch.Tensor,
    Q_obs: torch.Tensor,
    K_eval: int,
    seed: int,
) -> dict:
    """Predict at the curve's real observed times and score against Q_obs."""
    torch.manual_seed(seed)
    theta = rp.sample(x_i, n_samples=K_eval, show_progress_bars=False)
    with torch.no_grad():
        Q_pred = sim.simulate(theta, t_obs)  # (K_eval, T_real)
    Q_pred_np = Q_pred.cpu().numpy()
    Q_obs_np = Q_obs.cpu().numpy()

    median = np.median(Q_pred_np, axis=0)
    p10 = np.quantile(Q_pred_np, 0.10, axis=0)
    p90 = np.quantile(Q_pred_np, 0.90, axis=0)
    q_burst_mean = float(theta[:, sim.param_names.index("q_burst")].mean())

    return {
        "median": median,
        "p10": p10,
        "p90": p90,
        "R2": _r2(Q_obs_np, median),
        "MAE": float(np.abs(Q_obs_np - median).mean()),
        "q_burst_mean": q_burst_mean,
        # If t_obs[0] == 0, this is the t=0 model-form diagnostic
        "Q0_obs": float(Q_obs_np[0]),
        "Q0_pred_median": float(median[0]),
        "Q0_residual": float(median[0] - Q_obs_np[0]),
    }


def _load_cross_doi_xlsx(
    path: Path,
    featurizer_mean_by_col: dict[str, float],
) -> pd.DataFrame:
    """Load the 321-PLGA xlsx and reshape into the internal 181 schema.

    Unit conversions:
      Polymer MW  : kDa -> Da   (* 1000)
      DLC         : % -> frac   (/ 100)
      Release     : clamp to [0, 1]   (a few rows in the xlsx are > 1)

    Imputation: columns in _321_IMPUTED_COLS are filled with the loaded
    r_psi featurizer's training mean for that column. Combined with the
    featurizer's z-score in `transform()`, this puts those dims at z=0
    ("neutral" relative to the training distribution). The cross-DOI
    verdict (ADR-021) owns the bias this introduces.

    DP_Group: 321 has no drug identity column; synthesize "UNK-PLGA" so
    `parse_dp_group()` returns ("UNK", "PLGA") and the polymer-family
    one-hot fires correctly. 321 is PLGA-only by construction.
    """
    df = pd.read_excel(path)
    df = df.rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])

    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    df["DP_Group"] = "UNK-PLGA"

    # A few 321 formulations (e.g. 52, 136, 148, 305) have duplicate Time
    # rows that break torchdiffeq's strictly-monotonic-t requirement.
    # Average Release at duplicate (fid, t) pairs.
    df = (
        df.groupby(["Experimental_index", "Time"], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in ("Time", "Release")},
            "Release": "mean",
        })
    )

    for col in _321_IMPUTED_COLS:
        # Defensive fallback should not fire for the locked
        # PLGA_CONTINUOUS_COLS set.
        df[col] = featurizer_mean_by_col.get(col, 0.0)
    return df


def _run_cross_doi(
    args: argparse.Namespace,
    cfg: dict,
    sim: PLGABiphasic,
) -> None:
    """Cross-DOI 321 evaluation. ADR-021 Phase 1 success bar condition (3).

    No leave-one-out: r_psi is loaded as-is (the deployable model
    trained on the full internal 181 set, outputs/09 by default). 321
    is the held-out distribution by construction.
    """
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    print(f"[10] [cross-DOI] loading r_psi from {args.rpsi}")
    rp = DescriptorPosterior.load(args.rpsi, simulator=sim, device=args.device)
    feat = rp.featurizer
    mean_by_col = dict(zip(feat.continuous_cols, feat.mean.tolist(), strict=True))
    print(f"[10] [cross-DOI] featurizer input_dim={feat.input_dim}, "
          f"polymer_families={feat.polymer_families}")

    print(f"[10] [cross-DOI] loading 321 xlsx from {args.cross_doi_data}")
    df = _load_cross_doi_xlsx(args.cross_doi_data, mean_by_col)
    print(f"[10] [cross-DOI] {df['Experimental_index'].nunique()} formulations, "
          f"{len(df)} (fid,t) rows")

    keep_desc, Q_grid, metas, drug_ids, t_obs_list, Q_obs_list = _build_data(
        df=df, t_grid=t_grid,
        t_max_days=float(t_cfg["end"]),
        min_quality=args.min_quality,
    )
    n_kept = len(keep_desc)
    print(f"[10] [cross-DOI] {n_kept} curves at --min-quality {args.min_quality}")

    # Per-curve eval. Mirrors the internal-mode inner loop, but no fold
    # boundary because r_psi is fixed.
    pred_dir = args.out / "cross_doi_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    x_eval = feat.transform(keep_desc)
    per_curve_rows: list[dict] = []
    t0 = time.time()
    for j in range(n_kept):
        fid = int(keep_desc.loc[j, "Experimental_index"])
        res = _eval_curve(
            rp=rp, sim=sim,
            x_i=x_eval[j],
            t_obs=t_obs_list[j], Q_obs=Q_obs_list[j],
            K_eval=args.K_eval, seed=args.seed + j,
        )
        np.savez(
            pred_dir / f"{fid}.npz",
            t=t_obs_list[j].cpu().numpy(),
            Q_obs=Q_obs_list[j].cpu().numpy(),
            Q_pred_median=res["median"],
            Q_pred_p10=res["p10"],
            Q_pred_p90=res["p90"],
        )
        per_curve_rows.append({
            "Formulation_Index": fid,
            "n_obs": int(t_obs_list[j].numel()),
            "tmax_obs_d": float(t_obs_list[j].max()),
            "R2": res["R2"],
            "MAE": res["MAE"],
            "Q0_obs": res["Q0_obs"],
            "Q0_pred_median": res["Q0_pred_median"],
            "Q0_residual": res["Q0_residual"],
            "q_burst_mean": res["q_burst_mean"],
        })
        if (j + 1) % 25 == 0:
            print(f"[10] [cross-DOI]   {j + 1}/{n_kept} curves, "
                  f"elapsed {time.time() - t0:.1f}s")

    total_secs = time.time() - t0
    pcm_df = pd.DataFrame(per_curve_rows)
    pcm_df.to_csv(args.out / "cross_doi_per_curve_metrics.csv", index=False)
    print(f"[10] [cross-DOI] wrote cross_doi_per_curve_metrics.csv "
          f"({len(pcm_df)} rows) in {total_secs:.1f}s")

    median_r2 = float(np.nanmedian(pcm_df["R2"]))
    mean_r2 = float(np.nanmean(pcm_df["R2"]))
    median_mae = float(pcm_df["MAE"].median())
    median_q0 = float(pcm_df["Q0_residual"].median())
    n_r2_pos = int((pcm_df["R2"] > 0).sum())
    n_r2_above_030 = int((pcm_df["R2"] > 0.30).sum())

    summary_path = args.out / "cross_doi_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== Phase 1 cross-DOI evaluation (script 10 --cross-doi) ===\n\n")
        f.write(f"r_psi              : {args.rpsi}\n")
        f.write(f"xlsx               : {args.cross_doi_data}\n")
        f.write(f"min_quality        : {args.min_quality}\n")
        f.write(f"K_eval             : {args.K_eval}\n")
        f.write(f"n_curves_kept      : {n_kept} of {df['Experimental_index'].nunique()}\n")
        f.write(f"imputed_descriptors: {list(_321_IMPUTED_COLS)}\n")
        f.write(f"wallclock_seconds  : {total_secs:.1f}\n\n")
        f.write("Overall metrics (per-curve, no holdout — 321 is held-out by source):\n")
        f.write(f"  median R^2       : {median_r2:+.4f}\n")
        f.write(f"  mean R^2         : {mean_r2:+.4f}\n")
        f.write(f"  median MAE       : {median_mae:.4f}\n")
        f.write(f"  median Q0 resid  : {median_q0:+.4f}\n")
        f.write(f"  n curves R^2 > 0 : {n_r2_pos} / {n_kept}\n")
        f.write(f"  n curves R^2 > 0.30 : {n_r2_above_030} / {n_kept}\n\n")
        f.write("Phase 1 success bar condition (3): cross-DOI 321 R^2 > 0.30\n")
        verdict = "PASS" if median_r2 > 0.30 else "FAIL"
        f.write(f"  median R^2 = {median_r2:+.4f}  -> {verdict} (threshold 0.30)\n")
    print(f"[10] [cross-DOI] wrote {summary_path}")
    print(f"[10] [cross-DOI] median R^2 = {median_r2:+.4f}  ({verdict} vs 0.30)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/plga_phase1.yaml")
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
        help="path to Stage-1 q_phi checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/10_eval_deployment")
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K-train", type=int, default=64,
                        help="teacher samples per training curve (ADR-018)")
    parser.add_argument("--K-eval", type=int, default=256,
                        help="r_psi samples per evaluation curve (ADR-018)")
    parser.add_argument(
        "--min-quality",
        choices=("high", "medium", "low"),
        default="high",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument(
        "--n-folds-limit", type=int, default=None,
        help="run only the first N folds (for smoke testing)",
    )
    parser.add_argument(
        "--fold-by", choices=("drug", "curve"), default="drug",
        help="'drug' = leave-one-drug-out (ADR-019 default; outputs at "
             "<out>/). 'curve' = GroupKFold-5 by Experimental_index "
             "(ADR-023 comparator-aligned protocol; outputs at <out>/by_curve/)",
    )
    parser.add_argument(
        "--cross-doi", action="store_true",
        help="run cross-DOI 321 eval (ADR-021) instead of leave-one-drug-out",
    )
    parser.add_argument(
        "--cross-doi-data", type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
        help="path to the 321 PLGA xlsx (cross-DOI mode only)",
    )
    parser.add_argument(
        "--rpsi", type=Path,
        default=Path("outputs/09_descriptor_posterior/posterior.pt"),
        help="trained r_psi checkpoint for cross-DOI mode (no retraining)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    if args.cross_doi:
        _run_cross_doi(args=args, cfg=cfg, sim=sim)
        return

    (args.out / "predictions").mkdir(parents=True, exist_ok=True)

    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    print(f"[10] loading teacher q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device="cpu")

    print(f"[10] loading PLGA dataset from {args.data}")
    df = pd.read_csv(args.data)
    keep_desc, Q_grid, metas, drug_ids, t_obs_list, Q_obs_list = _build_data(
        df=df, t_grid=t_grid,
        t_max_days=float(t_cfg["end"]),
        min_quality=args.min_quality,
    )
    n_kept = len(keep_desc)
    print(f"[10] {n_kept} curves at --min-quality {args.min_quality}")
    print(f"[10] drug_ids: {sorted(set(drug_ids))}")

    # ------------------------------------------------------------------
    # Precompute teacher targets ONCE for all kept curves (q_phi is
    # fold-invariant). Use a throwaway DescriptorPosterior with a
    # featurizer fit on the full kept set just to access the
    # build_teacher_targets method.
    # ------------------------------------------------------------------
    full_feat = FormulationFeaturizer.fit(keep_desc)
    full_enc = MLPFormulationEncoder(
        input_dim=full_feat.input_dim,
        embed_dim=cfg["encoder"]["embed_dim"],
        hidden_dims=tuple(cfg["encoder"]["hidden_dims"]),
        dropout=cfg["encoder"]["dropout"],
    )
    sampler_rp = DescriptorPosterior(
        simulator=sim, featurizer=full_feat, encoder=full_enc,
        flow=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        device="cpu",  # sampling on CPU; q_phi already CPU-side
    )

    print(f"[10] sampling K_train={args.K_train} teacher targets for {n_kept} curves")
    t0 = time.time()
    theta_targets_all = sampler_rp.build_teacher_targets(
        q_phi=q_phi, Q_grid=Q_grid, K=args.K_train, seed=args.seed,
    )
    print(f"[10] teacher target precompute finished in {time.time() - t0:.1f}s, "
          f"shape={tuple(theta_targets_all.shape)}")

    # ------------------------------------------------------------------
    # Build the fold list. `fold_by = drug` is the ADR-019 default
    # (leave-one-drug-out). `fold_by = curve` is the ADR-023 comparator-
    # aligned protocol — GroupKFold-5 by Experimental_index, matching
    # fPCA / Direct LGBM in scripts/12_baseline_comparators.py.
    # ------------------------------------------------------------------
    fid_col = "Experimental_index"
    drug_idx_array = np.array(drug_ids)

    if args.fold_by == "drug":
        unique_drugs = sorted(set(drug_ids))
        fold_iter: list[tuple[str, np.ndarray]] = [
            (held_drug, drug_idx_array == held_drug) for held_drug in unique_drugs
        ]
    else:  # curve
        from sklearn.model_selection import GroupKFold  # noqa: PLC0415
        fids = np.array([str(f) for f in keep_desc[fid_col]])
        gkf = GroupKFold(n_splits=5)
        fold_iter = []
        for fold_no, (_, te) in enumerate(gkf.split(fids, groups=fids), start=1):
            m = np.zeros(len(fids), dtype=bool)
            m[te] = True
            fold_iter.append((f"fold{fold_no}", m))
        # Reroute outputs so we don't overwrite the LODO results.
        args.out = args.out / "by_curve"
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "predictions").mkdir(parents=True, exist_ok=True)
    if args.n_folds_limit is not None:
        fold_iter = fold_iter[: args.n_folds_limit]
    print(f"[10] fold_by={args.fold_by!r}; {len(fold_iter)} fold(s) to run")

    # Per-curve metrics rows, indexed by Experimental_index.
    per_curve_rows: list[dict] = []
    fold_summary_rows: list[dict] = []

    overall_t0 = time.time()
    for fold_idx, (fold_label, eval_mask) in enumerate(fold_iter):
        train_mask = ~eval_mask
        n_train = int(train_mask.sum())
        n_eval = int(eval_mask.sum())
        print(
            f"\n[10] fold {fold_idx + 1}/{len(fold_iter)}  "
            f"label={fold_label!r}  n_train={n_train}  n_eval={n_eval}"
        )

        train_desc = keep_desc.loc[train_mask].reset_index(drop=True)
        train_theta_targets = theta_targets_all[train_mask]

        feat = FormulationFeaturizer.fit(train_desc)
        encoder = MLPFormulationEncoder(
            input_dim=feat.input_dim,
            embed_dim=cfg["encoder"]["embed_dim"],
            hidden_dims=tuple(cfg["encoder"]["hidden_dims"]),
            dropout=cfg["encoder"]["dropout"],
        )
        rp = DescriptorPosterior(
            simulator=sim, featurizer=feat, encoder=encoder,
            flow=cfg["posterior"]["flow_type"],
            hidden_features=cfg["posterior"]["hidden_features"],
            num_transforms=cfg["posterior"]["num_transforms"],
            device=args.device,
        )

        x_train = feat.transform(train_desc)
        t_train_start = time.time()
        log = rp.train(
            x=x_train, theta_targets=train_theta_targets,
            training_batch_size=args.batch_size,
            max_num_epochs=args.max_epochs,
            learning_rate=cfg["train"]["lr"],
            stop_after_epochs=args.stop_after_epochs,
            seed=args.seed, verbose=False,
        )
        train_secs = time.time() - t_train_start

        # Evaluate held-out curves.
        eval_indices = np.where(eval_mask)[0]
        x_eval = feat.transform(keep_desc.loc[eval_mask].reset_index(drop=True))
        fold_r2: list[float] = []
        fold_mae: list[float] = []
        fold_q0_resid: list[float] = []
        for j, kept_idx in enumerate(eval_indices):
            fid = int(keep_desc.loc[kept_idx, fid_col])
            res = _eval_curve(
                rp=rp, sim=sim,
                x_i=x_eval[j], t_obs=t_obs_list[kept_idx],
                Q_obs=Q_obs_list[kept_idx],
                K_eval=args.K_eval, seed=args.seed + fold_idx,
            )
            # Save raw prediction for this curve.
            pred_path = args.out / "predictions" / f"{fid}.npz"
            np.savez(
                pred_path,
                t=t_obs_list[kept_idx].cpu().numpy(),
                Q_obs=Q_obs_list[kept_idx].cpu().numpy(),
                Q_pred_median=res["median"],
                Q_pred_p10=res["p10"],
                Q_pred_p90=res["p90"],
            )
            per_curve_rows.append({
                "Experimental_index": fid,
                "DP_Group": keep_desc.loc[kept_idx, "DP_Group"],
                "drug_id": drug_ids[kept_idx],
                "fold_label": fold_label,
                "polymer_family": parse_dp_group(keep_desc.loc[kept_idx, "DP_Group"])[1],
                "n_obs": int(t_obs_list[kept_idx].numel()),
                "R2": res["R2"],
                "MAE": res["MAE"],
                "Q0_obs": res["Q0_obs"],
                "Q0_pred_median": res["Q0_pred_median"],
                "Q0_residual": res["Q0_residual"],
                "q_burst_mean": res["q_burst_mean"],
            })
            fold_r2.append(res["R2"])
            fold_mae.append(res["MAE"])
            fold_q0_resid.append(res["Q0_residual"])

        fold_summary_rows.append({
            "fold_label": fold_label,
            "n_train": n_train,
            "n_eval": n_eval,
            "train_seconds": train_secs,
            "median_R2": float(np.nanmedian(fold_r2)) if fold_r2 else float("nan"),
            "mean_R2": float(np.nanmean(fold_r2)) if fold_r2 else float("nan"),
            "median_MAE": float(np.median(fold_mae)) if fold_mae else float("nan"),
            "median_Q0_residual": float(np.median(fold_q0_resid)) if fold_q0_resid else float("nan"),
            "final_train_loss": log.get("final_train_loss"),
            "final_val_loss": log.get("final_val_loss"),
        })
        print(
            f"[10] fold {fold_label}: train {train_secs:.1f}s; "
            f"median R^2 = {fold_summary_rows[-1]['median_R2']:.3f}, "
            f"median |Q0 resid| = {abs(fold_summary_rows[-1]['median_Q0_residual']):.3f}"
        )

    total_secs = time.time() - overall_t0
    print(f"\n[10] all folds finished in {total_secs / 60:.1f} min")

    # ------------------------------------------------------------------
    # Aggregate + write CSVs.
    # ------------------------------------------------------------------
    pcm_df = pd.DataFrame(per_curve_rows)
    pcm_df.to_csv(args.out / "per_curve_metrics.csv", index=False)
    print(f"[10] wrote per_curve_metrics.csv ({len(pcm_df)} rows)")

    per_drug = (
        pcm_df.groupby("drug_id")
        .agg(
            n_eval=("Experimental_index", "count"),
            median_R2=("R2", "median"),
            mean_R2=("R2", "mean"),
            median_MAE=("MAE", "median"),
            median_Q0_residual=("Q0_residual", "median"),
        )
        .reset_index()
        .sort_values("median_R2", ascending=False)
    )
    per_drug.to_csv(args.out / "per_drug_summary.csv", index=False)

    per_polymer = (
        pcm_df.groupby("polymer_family")
        .agg(
            n_eval=("Experimental_index", "count"),
            median_R2=("R2", "median"),
            mean_R2=("R2", "mean"),
            median_MAE=("MAE", "median"),
            median_Q0_residual=("Q0_residual", "median"),
        )
        .reset_index()
        .sort_values("median_R2", ascending=False)
    )
    per_polymer.to_csv(args.out / "per_polymer_summary.csv", index=False)

    overall_median_r2 = float(np.nanmedian(pcm_df["R2"]))
    overall_mean_r2 = float(np.nanmean(pcm_df["R2"]))
    overall_median_mae = float(pcm_df["MAE"].median())
    overall_median_q0 = float(pcm_df["Q0_residual"].median())

    summary_path = args.out / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=== Phase 1 deployment evaluation (script 10) ===\n\n")
        f.write(f"min_quality        : {args.min_quality}\n")
        f.write(f"K_train            : {args.K_train}\n")
        f.write(f"K_eval             : {args.K_eval}\n")
        f.write(f"n_curves_kept      : {n_kept}\n")
        f.write(f"fold_by            : {args.fold_by}\n")
        f.write(f"n_folds            : {len(fold_iter)}")
        if args.n_folds_limit is not None:
            f.write(f"  [LIMITED to first {args.n_folds_limit} folds]")
        f.write("\n")
        f.write(f"total_wallclock_min: {total_secs / 60:.1f}\n\n")
        f.write(f"Overall metrics (per-curve, fold_by={args.fold_by}):\n")
        f.write(f"  median R^2       : {overall_median_r2:+.4f}\n")
        f.write(f"  mean R^2         : {overall_mean_r2:+.4f}\n")
        f.write(f"  median MAE       : {overall_median_mae:.4f}\n")
        f.write(f"  median Q0 resid  : {overall_median_q0:+.4f}\n\n")
        f.write("Phase 1 success bar (ADR-006):\n")
        f.write("  (1) SBC calibration  : PASS (8/8 c2st_ranks <= 0.60, ADR-017)\n")
        f.write(f"  (2) grouped-holdout R^2 vs SRDS  : "
                f"absolute R^2 = {overall_median_r2:+.4f}; "
                f"SRDS baseline N/A (not present in repo)\n")
        f.write("  (3) cross-DOI R^2 > 0.30 on 321  : N/A (not run; --cross-doi v1 placeholder)\n\n")
        f.write("Per-drug summary (sorted by median R^2 desc):\n")
        f.write(per_drug.to_string(index=False))
        f.write("\n\nPer-polymer summary (sorted by median R^2 desc):\n")
        f.write(per_polymer.to_string(index=False))
        f.write("\n")
    print(f"[10] wrote {summary_path}")


if __name__ == "__main__":
    main()
