"""
19 - Internal pseudo-321 benchmark for missing-descriptor robustness.

What this does:
    1. Load q_phi and the internal PLGA 181 dataset.
    2. Keep the same 133 high-quality curves used in sprint1.
    3. Reuse the existing by-curve fold assignments from
       outputs/sprint1_10_eval_deployment/by_curve/per_curve_metrics.csv.
    4. On each fold, compare five Stage-2 routes:
         - full_nominal       : train full 13-feature SBI, eval with full input
         - full_masked_eval   : same model, but eval with the 6 pseudo-321-
                                missing descriptor dims zeroed after z-score
         - crossdoi7          : train/eval SBI on the 7 truly shared cols only
         - masked_aug         : train full 13-feature SBI with extra masked
                                copies of training rows, eval masked
         - ensemble_masked    : pointwise average of full_masked_eval and
                                masked_aug predictions
    5. Aggregate overall and subgroup metrics on the held-out internal curves.

Why this exists:
    The external 321 results suggest a missingness-aware Stage-2 route can
    improve median R^2 without changing the simulator. Before trusting that
    as more than a lucky external hit, we need an internal pseudo-321 check
    under a fixed held-out protocol.

Outputs:
    outputs/19_pseudo321_internal_benchmark/per_curve_metrics.csv
    outputs/19_pseudo321_internal_benchmark/summary.csv
    outputs/19_pseudo321_internal_benchmark/subgroup_summary.csv
    outputs/19_pseudo321_internal_benchmark/summary.txt

Expected runtime:
    ~20-40 min on GPU for all 5 by-curve folds.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import (  # noqa: E402
    PLGA_CONTINUOUS_COLS,
    FormulationFeaturizer,
    MLPFormulationEncoder,
    parse_dp_group,
)
from posterior import CurvePosterior, DescriptorPosterior, interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
DP_GROUP_COL = "DP_Group"
_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}
_MASK_COLS_321_ABSENT = (
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Drug_NHA",
    "SA-V",
    "SE",
)
_CROSS_DOI_7_COLS = (
    "LA/GA",
    "Polymer_MW",
    "Initial D/M ratio",
    "DLC",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_LogP",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def _build_data(
    df: pd.DataFrame,
    t_grid: torch.Tensor,
    t_max_days: float,
    min_quality: str,
) -> tuple[
    pd.DataFrame,
    torch.Tensor,
    list[str],
    list[torch.Tensor],
    list[torch.Tensor],
    pd.DataFrame,
]:
    desc_df = (
        df.drop_duplicates(FID_COL)
        .sort_values(FID_COL)
        .reset_index(drop=True)
    )

    q_grid_list: list[torch.Tensor | None] = []
    qualities: list[str] = []
    t_obs_list: list[torch.Tensor] = []
    q_obs_list: list[torch.Tensor] = []
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            q_grid_i, quality, _ = interpolate_to_grid(
                t_obs, q_obs, t_grid, t_max_days=t_max_days,
            )
        except ValueError:
            q_grid_list.append(None)
            qualities.append("skip")
            t_obs_list.append(t_obs)
            q_obs_list.append(q_obs)
            continue
        q_grid_list.append(q_grid_i)
        qualities.append(quality)
        t_obs_list.append(t_obs)
        q_obs_list.append(q_obs)

    min_rank = _QUALITY_RANK[min_quality]
    keep_mask = [
        (q in _QUALITY_RANK) and (_QUALITY_RANK[q] >= min_rank)
        for q in qualities
    ]
    keep_desc = desc_df.loc[keep_mask].reset_index(drop=True)
    q_grid = torch.stack(
        [q_grid_list[i] for i, keep in enumerate(keep_mask) if keep], dim=0
    )
    kept_t_obs = [t_obs_list[i] for i, keep in enumerate(keep_mask) if keep]
    kept_q_obs = [q_obs_list[i] for i, keep in enumerate(keep_mask) if keep]
    drug_ids = [parse_dp_group(v)[0] for v in keep_desc[DP_GROUP_COL]]
    flag_rows = []
    for row_idx, fid in enumerate(keep_desc[FID_COL].tolist()):
        t_np = kept_t_obs[row_idx].cpu().numpy().astype(float)
        q_np = kept_q_obs[row_idx].cpu().numpy().astype(float)
        q7 = float(np.interp(7.0, t_np, q_np, left=q_np[0], right=q_np[-1]))
        t50 = float(np.interp(0.5, q_np, t_np)) if (q_np.min() <= 0.5 <= q_np.max()) else float("inf")
        flag_rows.append({
            "Experimental_index": int(fid),
            "fast_regime": bool((t50 <= 2.4) or (q7 >= 0.75)),
            "short_window": bool(float(t_np.max()) < 14.0),
        })
    flags_df = pd.DataFrame(flag_rows)
    return keep_desc, q_grid, drug_ids, kept_t_obs, kept_q_obs, flags_df


def _load_fold_assignments(
    keep_desc: pd.DataFrame,
    fold_assignments_csv: Path | None,
) -> list[tuple[str, np.ndarray]]:
    if fold_assignments_csv is not None and fold_assignments_csv.exists():
        rdf = pd.read_csv(fold_assignments_csv)
        if "fold_label" not in rdf.columns or FID_COL not in rdf.columns:
            raise KeyError(
                f"{fold_assignments_csv} must contain {FID_COL!r} and 'fold_label'"
            )
        fid_to_fold = dict(zip(rdf[FID_COL].astype(int), rdf["fold_label"], strict=False))
        missing = [int(fid) for fid in keep_desc[FID_COL] if int(fid) not in fid_to_fold]
        if missing:
            raise KeyError(f"Fold assignment file missing kept fids: {missing[:10]}")
        labels = [fid_to_fold[int(fid)] for fid in keep_desc[FID_COL]]
        fold_iter: list[tuple[str, np.ndarray]] = []
        for fold_label in sorted(set(labels)):
            mask = np.array([lab == fold_label for lab in labels], dtype=bool)
            fold_iter.append((str(fold_label), mask))
        return fold_iter

    fids = np.array([str(fid) for fid in keep_desc[FID_COL]])
    gkf = GroupKFold(n_splits=5)
    fold_iter = []
    for fold_no, (_, te) in enumerate(gkf.split(fids, groups=fids), start=1):
        mask = np.zeros(len(fids), dtype=bool)
        mask[te] = True
        fold_iter.append((f"fold{fold_no}", mask))
    return fold_iter


def _make_rpsi(
    feat: FormulationFeaturizer,
    cfg: dict,
    sim: PLGABiphasic,
    device: str,
) -> DescriptorPosterior:
    encoder = MLPFormulationEncoder(
        input_dim=feat.input_dim,
        embed_dim=cfg["encoder"]["embed_dim"],
        hidden_dims=tuple(cfg["encoder"]["hidden_dims"]),
        dropout=cfg["encoder"]["dropout"],
    )
    return DescriptorPosterior(
        simulator=sim,
        featurizer=feat,
        encoder=encoder,
        flow=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        device=device,
    )


def _masked_feature_copy(
    x_full: torch.Tensor,
    featurizer: FormulationFeaturizer,
) -> tuple[torch.Tensor, list[int]]:
    idxs = [featurizer.continuous_cols.index(col) for col in _MASK_COLS_321_ABSENT]
    x_masked = x_full.clone()
    x_masked[:, idxs] = 0.0
    return x_masked, idxs


def _eval_pred_curve(
    rp: DescriptorPosterior,
    sim: PLGABiphasic,
    x_i: torch.Tensor,
    t_obs: torch.Tensor,
    q_obs: torch.Tensor,
    k_eval: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    torch.manual_seed(seed)
    theta = rp.sample(x_i, n_samples=k_eval, show_progress_bars=False)
    with torch.no_grad():
        q_pred = sim.simulate(theta, t_obs)
    q_pred_np = q_pred.cpu().numpy()
    q_obs_np = q_obs.cpu().numpy()
    median = np.median(q_pred_np, axis=0)
    return {
        "median": median,
        "R2": _r2(q_obs_np, median),
        "MAE": float(np.mean(np.abs(q_obs_np - median))),
    }


def _append_metric(
    rows: list[dict[str, object]],
    method: str,
    fold_label: str,
    fid: int,
    dp_group: str,
    drug_id: str,
    t_obs: torch.Tensor,
    metric: dict[str, np.ndarray | float],
) -> None:
    rows.append({
        "method": method,
        "Experimental_index": fid,
        "DP_Group": dp_group,
        "drug_id": drug_id,
        "polymer_family": parse_dp_group(dp_group)[1],
        "fold_label": fold_label,
        "n_obs": int(t_obs.numel()),
        "R2": float(metric["R2"]),
        "MAE": float(metric["MAE"]),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("data/Dataset_17_feat_augmented.csv"),
    )
    ap.add_argument(
        "--fold-assignments-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/by_curve/per_curve_metrics.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/19_pseudo321_internal_benchmark"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K-train", type=int, default=64)
    ap.add_argument("--K-eval", type=int, default=256)
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--stop-after-epochs", type=int, default=20)
    ap.add_argument("--mask-repeats", type=int, default=1)
    ap.add_argument("--n-folds-limit", type=int, default=None)
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )

    print(f"[19] loading q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device="cpu")

    print(f"[19] loading internal data from {args.data}")
    df = pd.read_csv(args.data)
    keep_desc, q_grid, drug_ids, t_obs_list, q_obs_list, flags_df = _build_data(
        df=df,
        t_grid=t_grid,
        t_max_days=float(t_cfg["end"]),
        min_quality=args.min_quality,
    )
    print(f"[19] kept {len(keep_desc)} high-quality curves")

    print(f"[19] sampling K={args.K_train} teacher targets once for all kept curves")
    t_sample_start = time.time()
    temp_feat = FormulationFeaturizer.fit(keep_desc, continuous_cols=PLGA_CONTINUOUS_COLS)
    theta_targets_all = DescriptorPosterior(
        simulator=sim,
        featurizer=temp_feat,
        encoder=MLPFormulationEncoder(
            input_dim=temp_feat.input_dim,
            embed_dim=cfg["encoder"]["embed_dim"],
            hidden_dims=tuple(cfg["encoder"]["hidden_dims"]),
            dropout=cfg["encoder"]["dropout"],
        ),
        device="cpu",
    ).build_teacher_targets(
        q_phi=q_phi,
        Q_grid=q_grid,
        K=args.K_train,
        seed=args.seed,
        show_progress_bars=False,
    )
    print(f"[19] teacher sampling finished in {time.time() - t_sample_start:.1f}s")

    fold_iter = _load_fold_assignments(keep_desc, args.fold_assignments_csv)
    if args.n_folds_limit is not None:
        fold_iter = fold_iter[: args.n_folds_limit]
    print(f"[19] running {len(fold_iter)} fold(s)")

    per_curve_rows: list[dict[str, object]] = []
    overall_t0 = time.time()
    for fold_idx, (fold_label, eval_mask) in enumerate(fold_iter, start=1):
        print(
            f"\n[19] fold {fold_idx}/{len(fold_iter)}  "
            f"label={fold_label!r}  n_train={int((~eval_mask).sum())}  n_eval={int(eval_mask.sum())}"
        )
        train_desc = keep_desc.loc[~eval_mask].reset_index(drop=True)
        eval_desc = keep_desc.loc[eval_mask].reset_index(drop=True)
        train_theta = theta_targets_all[~eval_mask]
        eval_indices = np.where(eval_mask)[0]

        # full_nominal + full_masked_eval share the same trained model
        feat_full = FormulationFeaturizer.fit(train_desc, continuous_cols=PLGA_CONTINUOUS_COLS)
        rp_full = _make_rpsi(feat_full, cfg, sim, args.device)
        x_train_full = feat_full.transform(train_desc)
        t0 = time.time()
        rp_full.train(
            x=x_train_full,
            theta_targets=train_theta,
            training_batch_size=args.batch_size,
            max_num_epochs=args.max_epochs,
            learning_rate=cfg["train"]["lr"],
            stop_after_epochs=args.stop_after_epochs,
            seed=args.seed,
            verbose=False,
        )
        print(f"[19]   full model train {time.time() - t0:.1f}s")
        x_eval_full_nominal = feat_full.transform(eval_desc)
        x_eval_full_masked, mask_indices = _masked_feature_copy(x_eval_full_nominal, feat_full)

        # crossdoi7 route
        feat_7 = FormulationFeaturizer.fit(train_desc, continuous_cols=_CROSS_DOI_7_COLS)
        rp_7 = _make_rpsi(feat_7, cfg, sim, args.device)
        x_train_7 = feat_7.transform(train_desc)
        t0 = time.time()
        rp_7.train(
            x=x_train_7,
            theta_targets=train_theta,
            training_batch_size=args.batch_size,
            max_num_epochs=args.max_epochs,
            learning_rate=cfg["train"]["lr"],
            stop_after_epochs=args.stop_after_epochs,
            seed=args.seed,
            verbose=False,
        )
        print(f"[19]   7f model train {time.time() - t0:.1f}s")
        x_eval_7 = feat_7.transform(eval_desc)

        # masked augmentation route
        rp_mask = _make_rpsi(feat_full, cfg, sim, args.device)
        x_train_masked, _ = _masked_feature_copy(x_train_full, feat_full)
        x_blocks = [x_train_full] + [x_train_masked.clone() for _ in range(args.mask_repeats)]
        theta_blocks = [train_theta] + [train_theta.clone() for _ in range(args.mask_repeats)]
        x_train_aug = torch.cat(x_blocks, dim=0)
        theta_train_aug = torch.cat(theta_blocks, dim=0)
        t0 = time.time()
        rp_mask.train(
            x=x_train_aug,
            theta_targets=theta_train_aug,
            training_batch_size=args.batch_size,
            max_num_epochs=args.max_epochs,
            learning_rate=cfg["train"]["lr"],
            stop_after_epochs=args.stop_after_epochs,
            seed=args.seed,
            verbose=False,
        )
        print(f"[19]   masked model train {time.time() - t0:.1f}s  mask_idxs={mask_indices}")

        for j, kept_idx in enumerate(eval_indices):
            fid = int(keep_desc.loc[kept_idx, FID_COL])
            dp_group = str(keep_desc.loc[kept_idx, DP_GROUP_COL])
            drug_id = drug_ids[kept_idx]
            t_obs = t_obs_list[kept_idx]
            q_obs = q_obs_list[kept_idx]

            res_full_nominal = _eval_pred_curve(
                rp_full, sim, x_eval_full_nominal[j], t_obs, q_obs, args.K_eval, args.seed + kept_idx
            )
            res_full_masked = _eval_pred_curve(
                rp_full, sim, x_eval_full_masked[j], t_obs, q_obs, args.K_eval, args.seed + 10000 + kept_idx
            )
            res_7 = _eval_pred_curve(
                rp_7, sim, x_eval_7[j], t_obs, q_obs, args.K_eval, args.seed + 20000 + kept_idx
            )
            res_mask = _eval_pred_curve(
                rp_mask, sim, x_eval_full_masked[j], t_obs, q_obs, args.K_eval, args.seed + 30000 + kept_idx
            )

            q_obs_np = q_obs.cpu().numpy()
            pred_ens = 0.5 * (
                np.asarray(res_full_masked["median"], dtype=float)
                + np.asarray(res_mask["median"], dtype=float)
            )
            res_ens = {
                "median": pred_ens,
                "R2": _r2(q_obs_np, pred_ens),
                "MAE": float(np.mean(np.abs(q_obs_np - pred_ens))),
            }

            _append_metric(per_curve_rows, "full_nominal", fold_label, fid, dp_group, drug_id, t_obs, res_full_nominal)
            _append_metric(per_curve_rows, "full_masked_eval", fold_label, fid, dp_group, drug_id, t_obs, res_full_masked)
            _append_metric(per_curve_rows, "crossdoi7", fold_label, fid, dp_group, drug_id, t_obs, res_7)
            _append_metric(per_curve_rows, "masked_aug", fold_label, fid, dp_group, drug_id, t_obs, res_mask)
            _append_metric(per_curve_rows, "ensemble_masked", fold_label, fid, dp_group, drug_id, t_obs, res_ens)

    total_secs = time.time() - overall_t0
    per_curve = pd.DataFrame(per_curve_rows)
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)

    summary = (
        per_curve.groupby("method", as_index=False)
        .agg(
            n_curves=("Experimental_index", "count"),
            median_R2=("R2", "median"),
            mean_R2=("R2", "mean"),
            median_MAE=("MAE", "median"),
            n_R2_gt_0=("R2", lambda s: int((s > 0).sum())),
            n_R2_gt_0p3=("R2", lambda s: int((s > 0.3).sum())),
        )
        .sort_values("median_R2", ascending=False)
    )
    summary.to_csv(args.out / "summary.csv", index=False)

    merged = per_curve.merge(flags_df, on="Experimental_index", how="left")
    subgroup_rows = []
    for method, mdf in merged.groupby("method", sort=False):
        subsets = {
            "overall": mdf,
            "fast": mdf[mdf["fast_regime"]],
            "short": mdf[mdf["short_window"]],
            "fast_short": mdf[mdf["fast_regime"] & mdf["short_window"]],
            "neither": mdf[~mdf["fast_regime"] & ~mdf["short_window"]],
        }
        for subset, sdf in subsets.items():
            subgroup_rows.append({
                "method": method,
                "subset": subset,
                "n_curves": int(len(sdf)),
                "median_R2": float(sdf["R2"].median()),
                "mean_R2": float(sdf["R2"].mean()),
                "median_MAE": float(sdf["MAE"].median()),
            })
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(args.out / "subgroup_summary.csv", index=False)

    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Internal pseudo-321 benchmark ===\n\n")
        f.write(f"min_quality        : {args.min_quality}\n")
        f.write(f"K_train            : {args.K_train}\n")
        f.write(f"K_eval             : {args.K_eval}\n")
        f.write(f"mask_repeats       : {args.mask_repeats}\n")
        f.write(f"n_curves_kept      : {len(keep_desc)}\n")
        f.write(f"n_folds            : {len(fold_iter)}\n")
        f.write(f"total_wallclock_min: {total_secs / 60:.1f}\n\n")
        f.write("Overall summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSubgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"\n[19] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
