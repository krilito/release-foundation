"""
25 - Prefix-aware synthetic Stage-1 theta regressor.

What this does:
    1. Sample synthetic `(theta, Q)` pairs from the PLGA simulator.
    2. Turn each full synthetic curve into prefix-observation tasks:
           partial curve on t_grid + observation mask -> theta
    3. Train a small MLP regressor on this synthetic partial-observation task.
    4. Evaluate on matched high-quality 321 curves by predicting theta from a
       real prefix, rolling out the simulator, and scoring suffix prediction.

Why this exists:
    Script 23 showed that the current `q_phi` was trained on full curves and
    becomes unstable when coerced into prefix-only updates. Script 24 showed
    that curve-only prefix self-supervision carries some transferable signal.
    This script tests the missing middle: a genuinely prefix-aware Stage-1
    latent inference route trained on synthetic data, without touching
    `posterior.py` or the user's in-progress core files.

Outputs:
    outputs/25_prefix_theta_regressor/model.pt
    outputs/25_prefix_theta_regressor/training_log.csv
    outputs/25_prefix_theta_regressor/per_curve_metrics.csv
    outputs/25_prefix_theta_regressor/summary.csv
    outputs/25_prefix_theta_regressor/subgroup_summary.csv
    outputs/25_prefix_theta_regressor/summary.txt

Expected runtime:
    ~3-10 min on a 4060 for the default smoke-scale settings.
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
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import interpolate_to_grid  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402

FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
DP_GROUP_COL = "DP_Group"
_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}
_321_RENAME = {
    "Formulation Index": FID_COL,
    "Drug MW": "Drug_Mw",
    "Drug TPSA": "Drug_TPSA",
    "Drug LogP": "Drug_LogP",
    "Polymer MW": "Polymer_MW",
    "Initial Drug-to-Polymer Ratio": "Initial D/M ratio",
    "Drug Loading Capacity": "DLC",
}
_321_DROP = (
    "Particle Size",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
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


class ThetaMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _prefix_task_from_curve(
    t_obs: torch.Tensor,
    q_obs: torch.Tensor,
    prefix_days: float,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor | None, np.ndarray | None]:
    t_np = t_obs.cpu().numpy().astype(float)
    q_np = q_obs.cpu().numpy().astype(float)
    prefix_obs_mask = t_np <= prefix_days
    future_obs_mask = t_np > prefix_days
    if prefix_obs_mask.sum() < 2 or future_obs_mask.sum() < 2:
        return None, None
    q_prefix_grid, _, _ = interpolate_to_grid(
        torch.tensor(t_np[prefix_obs_mask], dtype=torch.float32),
        torch.tensor(q_np[prefix_obs_mask], dtype=torch.float32),
        t_grid,
        t_max_days=float(t_grid.max().item()),
    )
    grid_prefix_mask = torch.tensor(
        (t_grid.cpu().numpy().astype(float) <= prefix_days).astype(np.float32),
        dtype=torch.float32,
    )
    q_in = q_prefix_grid * grid_prefix_mask
    x = torch.cat(
        [
            q_in,
            grid_prefix_mask,
            torch.tensor([prefix_days / float(t_grid.max().item())], dtype=torch.float32),
        ],
        dim=0,
    )
    return x, future_obs_mask


def _build_synthetic_tasks(
    sim: PLGABiphasic,
    t_grid: torch.Tensor,
    *,
    n_pairs: int,
    prefix_days_list: tuple[float, ...],
    noise_sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    torch.manual_seed(seed)
    theta = sim.sample_prior(n_pairs)
    with torch.no_grad():
        q_full = sim.simulate(theta, t_grid)
    if noise_sigma > 0:
        q_full = (q_full + torch.randn_like(q_full) * noise_sigma).clamp(0.0, 1.0)

    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    curve_ids: list[int] = []
    for i in range(n_pairs):
        q_i = q_full[i]
        for prefix_days in prefix_days_list:
            prefix_mask = (t_grid <= prefix_days).to(dtype=torch.float32)
            future_mask = t_grid > prefix_days
            if int(prefix_mask.sum().item()) < 2 or int(future_mask.sum().item()) < 2:
                continue
            q_in = q_i * prefix_mask
            x_i = torch.cat(
                [
                    q_in,
                    prefix_mask,
                    torch.tensor([prefix_days / float(t_grid.max().item())], dtype=torch.float32),
                ],
                dim=0,
            )
            x_rows.append(x_i)
            y_rows.append(theta[i])
            curve_ids.append(i)
    return (
        torch.stack(x_rows, dim=0),
        torch.stack(y_rows, dim=0),
        np.asarray(curve_ids, dtype=np.int64),
    )


def _fit_regressor(
    model: ThetaMLP,
    x_train: torch.Tensor,
    y_train_z: torch.Tensor,
    x_val: torch.Tensor,
    y_val_z: torch.Tensor,
    *,
    device: str,
    lr: float,
    batch_size: int,
    max_epochs: int,
    stop_after_epochs: int,
    out: Path,
    theta_mean: torch.Tensor,
    theta_std: torch.Tensor,
) -> list[dict[str, float]]:
    model.to(device)
    x_train = x_train.to(device)
    y_train_z = y_train_z.to(device)
    x_val = x_val.to(device)
    y_val_z = y_val_z.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start:start + batch_size]
            pred = model(x_train[idx])
            loss = ((pred - y_train_z[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = float(((model(x_train) - y_train_z) ** 2).mean().item())
            va_loss = float(((model(x_val) - y_val_z) ** 2).mean().item())
        log_rows.append({"epoch": float(epoch), "train_loss": tr_loss, "val_loss": va_loss})
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= stop_after_epochs:
            break

    if best_state is None:
        raise RuntimeError("no best checkpoint recorded")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "theta_mean": theta_mean.cpu(),
            "theta_std": theta_std.cpu(),
        },
        out / "model.pt",
    )
    pd.DataFrame(log_rows).to_csv(out / "training_log.csv", index=False)
    return log_rows


def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
    t_grid: torch.Tensor,
    min_quality: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns=_321_RENAME)
    df = df.drop(columns=[c for c in _321_DROP if c in df.columns])
    df["Polymer_MW"] = df["Polymer_MW"].astype(float) * 1000.0
    df["DLC"] = df["DLC"].astype(float) / 100.0
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    df[DP_GROUP_COL] = "UNK-PLGA"
    df = (
        df.groupby([FID_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{c: "first" for c in df.columns if c not in (TIME_COL, Y_COL)},
            Y_COL: "mean",
        })
    )
    df = df[df[FID_COL].astype(int).isin(matched_fids)].copy()

    min_rank = _QUALITY_RANK[min_quality]
    rows: list[dict[str, object]] = []
    for fid in sorted(df[FID_COL].astype(int).unique().tolist()):
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            _, quality, _ = interpolate_to_grid(
                t_obs,
                q_obs,
                t_grid,
                t_max_days=float(t_grid.max().item()),
            )
        except ValueError:
            continue
        if _QUALITY_RANK.get(quality, -1) < min_rank:
            continue
        rows.append({
            "Formulation_Index": fid,
            "t_obs": t_obs,
            "q_obs": q_obs,
        })
    tail_flags = pd.read_csv(
        Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv")
    )[["Formulation_Index", "fast_regime", "short_window"]]
    return rows, tail_flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
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
    ap.add_argument("--out", type=Path, default=Path("outputs/25_prefix_theta_regressor"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sim", type=int, default=20000)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--stop-after-epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--prefix-days", type=str, default="1,3,7")
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--max-external-curves", type=int, default=None)
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.time()

    prefix_days_list = tuple(float(x.strip()) for x in args.prefix_days.split(",") if x.strip())
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    noise_sigma = float(cfg["posterior"].get("noise_sigma", 0.03))

    print(f"[25] building synthetic prefix tasks, n_sim={args.n_sim}")
    x_all, theta_all, curve_ids = _build_synthetic_tasks(
        sim,
        t_grid=t_grid,
        n_pairs=args.n_sim,
        prefix_days_list=prefix_days_list,
        noise_sigma=noise_sigma,
        seed=args.seed,
    )
    print(f"[25] built {x_all.shape[0]} tasks from {args.n_sim} synthetic curves")

    rng = np.random.default_rng(args.seed)
    uniq_curve_ids = np.array(sorted(set(curve_ids.tolist())), dtype=np.int64)
    rng.shuffle(uniq_curve_ids)
    n_val_curves = max(1, int(round(len(uniq_curve_ids) * args.val_frac)))
    val_curve_ids = set(uniq_curve_ids[:n_val_curves].tolist())
    va_mask = np.array([cid in val_curve_ids for cid in curve_ids], dtype=bool)
    tr_mask = ~va_mask
    va_idx = np.flatnonzero(va_mask)
    tr_idx = np.flatnonzero(tr_mask)

    theta_mean = theta_all[tr_idx].mean(dim=0)
    theta_std = theta_all[tr_idx].std(dim=0).clamp_min(1e-6)
    theta_all_z = (theta_all - theta_mean) / theta_std

    model = ThetaMLP(
        input_dim=x_all.shape[1],
        output_dim=theta_all.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    log_rows = _fit_regressor(
        model,
        x_train=x_all[tr_idx],
        y_train_z=theta_all_z[tr_idx],
        x_val=x_all[va_idx],
        y_val_z=theta_all_z[va_idx],
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after_epochs,
        out=args.out,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )
    print(f"[25] trained for {len(log_rows)} epochs")

    external_rows, tail_flags = _load_external_records(
        args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid=t_grid,
        min_quality=args.min_quality,
    )
    if args.max_external_curves is not None and len(external_rows) > args.max_external_curves:
        keep_idx = np.sort(rng.choice(len(external_rows), size=args.max_external_curves, replace=False))
        external_rows = [external_rows[i] for i in keep_idx]
        print(f"[25] subsampled external curves to {len(external_rows)}")

    model.to(args.device)
    model.eval()
    per_curve_rows: list[dict[str, object]] = []
    for prefix_days in prefix_days_list:
        eligible = 0
        for row in external_rows:
            x_i, future_obs_mask = _prefix_task_from_curve(
                row["t_obs"],
                row["q_obs"],
                prefix_days=prefix_days,
                t_grid=t_grid,
            )
            if x_i is None or future_obs_mask is None:
                continue
            eligible += 1
            with torch.no_grad():
                theta_z = model(x_i.unsqueeze(0).to(args.device)).squeeze(0).cpu()
            theta_hat = theta_z * theta_std + theta_mean
            with torch.no_grad():
                q_pred = sim.simulate(theta_hat.unsqueeze(0), row["t_obs"]).squeeze(0).cpu().numpy()
            q_true = row["q_obs"].cpu().numpy().astype(float)[future_obs_mask]
            q_hat = q_pred.astype(float)[future_obs_mask]
            per_curve_rows.append({
                "prefix_days": prefix_days,
                "method": "prefix_theta_reg",
                "Formulation_Index": int(row["Formulation_Index"]),
                "n_prefix_obs": int((~future_obs_mask).sum()),
                "n_future_obs": int(future_obs_mask.sum()),
                "R2": float(_r2(q_true, q_hat)),
                "MAE": float(np.mean(np.abs(q_true - q_hat))),
            })
        print(f"[25] prefix_days={prefix_days:g} eligible={eligible}")

    per_curve = pd.DataFrame(per_curve_rows).merge(tail_flags, on="Formulation_Index", how="left")
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)

    summary = (
        per_curve.groupby(["prefix_days", "method"], as_index=False)
        .agg(
            n_curves=("Formulation_Index", "count"),
            median_R2=("R2", "median"),
            mean_R2=("R2", "mean"),
            median_MAE=("MAE", "median"),
            n_R2_gt_0=("R2", lambda s: int((s > 0).sum())),
            n_R2_gt_0p3=("R2", lambda s: int((s > 0.3).sum())),
        )
        .sort_values(["prefix_days", "median_R2"], ascending=[True, False])
    )
    summary.to_csv(args.out / "summary.csv", index=False)

    subgroup_rows: list[dict[str, object]] = []
    for (prefix_days, method), mdf in per_curve.groupby(["prefix_days", "method"], sort=False):
        for subset, sdf in {
            "overall": mdf,
            "fast": mdf[mdf["fast_regime"]],
            "short": mdf[mdf["short_window"]],
            "fast_short": mdf[mdf["fast_regime"] & mdf["short_window"]],
            "neither": mdf[~mdf["fast_regime"] & ~mdf["short_window"]],
        }.items():
            subgroup_rows.append({
                "prefix_days": prefix_days,
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
        f.write("=== Prefix-aware synthetic theta regressor ===\n\n")
        f.write(f"n_sim synthetic     : {args.n_sim}\n")
        f.write(f"prefix_days         : {prefix_days_list}\n")
        f.write(f"hidden_dim          : {args.hidden_dim}\n")
        f.write(f"dropout             : {args.dropout}\n")
        f.write(f"epochs_ran          : {len(log_rows)}\n")
        f.write(f"external_curves     : {len(external_rows)}\n")
        f.write(f"wallclock_min       : {(time.time() - overall_t0) / 60:.1f}\n\n")
        f.write("Overall summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSubgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"[25] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
