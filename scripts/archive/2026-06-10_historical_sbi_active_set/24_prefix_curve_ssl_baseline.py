"""
24 - Curve-only self-supervised prefix world-model baseline.

What this does:
    1. Load the internal 181 PLGA dataset and keep the same high-quality
       curves used in sprint1.
    2. Build self-supervised tasks of the form:
           observed prefix on t_grid + observation mask -> full release curve
       for prefix horizons such as 1, 3, and 7 days.
    3. Train a small curve-only MLP on 181 without descriptors or simulator.
    4. Evaluate on matched high-quality 321 curves using only their prefixes
       and scoring suffix prediction on the original observed time points.

Why this exists:
    This is the cleanest version of the user's idea: let the model watch many
    real release curves, summarize their dynamics, then predict the rest of a
    new curve after seeing only an early prefix. It is not RL; it is a
    self-supervised sequence-completion benchmark. The goal is to test whether
    curve dynamics alone carry transferable signal across 181 -> 321, separate
    from the weak descriptor-to-parameter bridge.

Outputs:
    outputs/24_prefix_curve_ssl_baseline/model.pt
    outputs/24_prefix_curve_ssl_baseline/training_log.csv
    outputs/24_prefix_curve_ssl_baseline/per_curve_metrics.csv
    outputs/24_prefix_curve_ssl_baseline/summary.csv
    outputs/24_prefix_curve_ssl_baseline/subgroup_summary.csv
    outputs/24_prefix_curve_ssl_baseline/summary.txt

Expected runtime:
    ~2-6 min on a 4060 for the default setup.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posterior import interpolate_to_grid  # noqa: E402

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


@dataclass
class CurveRecord:
    fid: int
    q_full_grid: torch.Tensor
    t_obs: torch.Tensor
    q_obs: torch.Tensor


class PrefixCurveMLP(nn.Module):
    def __init__(self, t_grid_len: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = 2 * t_grid_len + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, t_grid_len),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_keep_records(
    df: pd.DataFrame,
    t_grid: torch.Tensor,
    t_max_days: float,
    min_quality: str,
) -> list[CurveRecord]:
    desc_df = df.drop_duplicates(FID_COL).sort_values(FID_COL).reset_index(drop=True)
    records: list[CurveRecord] = []
    min_rank = _QUALITY_RANK[min_quality]
    for fid in desc_df[FID_COL].tolist():
        g = df[df[FID_COL] == fid].sort_values(TIME_COL)
        t_obs = torch.tensor(g[TIME_COL].to_numpy(), dtype=torch.float32)
        q_obs = torch.tensor(g[Y_COL].to_numpy(), dtype=torch.float32).clamp(0.0, 1.0)
        try:
            q_grid, quality, _ = interpolate_to_grid(
                t_obs,
                q_obs,
                t_grid,
                t_max_days=t_max_days,
            )
        except ValueError:
            continue
        if _QUALITY_RANK.get(quality, -1) < min_rank:
            continue
        records.append(
            CurveRecord(
                fid=int(fid),
                q_full_grid=q_grid,
                t_obs=t_obs,
                q_obs=q_obs,
            )
        )
    return records


def _load_internal_records(
    csv_path: Path,
    t_grid: torch.Tensor,
    min_quality: str,
) -> list[CurveRecord]:
    df = pd.read_csv(csv_path)
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
    return _build_keep_records(
        df,
        t_grid=t_grid,
        t_max_days=float(t_grid.max().item()),
        min_quality=min_quality,
    )


def _load_external_records(
    xlsx_path: Path,
    matched_fids_csv: Path,
    t_grid: torch.Tensor,
    min_quality: str,
) -> tuple[list[CurveRecord], pd.DataFrame]:
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
    records = _build_keep_records(
        df,
        t_grid=t_grid,
        t_max_days=float(t_grid.max().item()),
        min_quality=min_quality,
    )
    tail_flags = pd.read_csv(
        Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv")
    )[["Formulation_Index", "fast_regime", "short_window"]]
    return records, tail_flags


def _make_prefix_input(
    t_obs: torch.Tensor,
    q_obs: torch.Tensor,
    prefix_days: float,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, np.ndarray | None]:
    t_np = t_obs.cpu().numpy().astype(float)
    q_np = q_obs.cpu().numpy().astype(float)
    prefix_obs_mask = t_np <= prefix_days
    future_obs_mask = t_np > prefix_days
    if prefix_obs_mask.sum() < 2 or future_obs_mask.sum() < 2:
        return None, None, None
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
    return x, grid_prefix_mask, future_obs_mask


def _build_train_tasks(
    records: list[CurveRecord],
    prefix_days_list: tuple[float, ...],
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    w_rows: list[torch.Tensor] = []
    fid_rows: list[int] = []
    for rec in records:
        for prefix_days in prefix_days_list:
            x_i, prefix_grid_mask, future_obs_mask = _make_prefix_input(
                rec.t_obs,
                rec.q_obs,
                prefix_days=prefix_days,
                t_grid=t_grid,
            )
            if x_i is None or prefix_grid_mask is None or future_obs_mask is None:
                continue
            # Downweight prefix reconstruction so the model must care about
            # suffix dynamics rather than copying the observed prefix.
            weights = torch.where(
                prefix_grid_mask > 0.5,
                torch.full_like(prefix_grid_mask, 0.2),
                torch.ones_like(prefix_grid_mask),
            )
            x_rows.append(x_i)
            y_rows.append(rec.q_full_grid)
            w_rows.append(weights)
            fid_rows.append(rec.fid)
    return (
        torch.stack(x_rows, dim=0),
        torch.stack(y_rows, dim=0),
        torch.stack(w_rows, dim=0),
        np.asarray(fid_rows, dtype=np.int64),
    )


def _split_train_val(
    curve_fids: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.array(sorted(set(curve_fids.tolist())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_fids = set(uniq[:n_val].tolist())
    is_val = np.array([fid in val_fids for fid in curve_fids], dtype=bool)
    is_tr = ~is_val
    return is_tr, is_val


def _fit_model(
    model: PrefixCurveMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    w_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    w_val: torch.Tensor,
    *,
    device: str,
    lr: float,
    batch_size: int,
    max_epochs: int,
    stop_after_epochs: int,
    out: Path,
) -> list[dict[str, float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log_rows: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    w_train = w_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    w_val = w_val.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        batch_losses: list[float] = []
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start:start + batch_size]
            pred = model(x_train[idx])
            loss = (((pred - y_train[idx]) ** 2) * w_train[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train)
            train_loss = float((((train_pred - y_train) ** 2) * w_train).mean().item())
            val_pred = model(x_val)
            val_loss = float((((val_pred - y_val) ** 2) * w_val).mean().item())
        log_rows.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= stop_after_epochs:
            break

    if best_state is None:
        raise RuntimeError("training failed to produce a best checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict()}, out / "model.pt")
    pd.DataFrame(log_rows).to_csv(out / "training_log.csv", index=False)
    return log_rows


def _predict_full_grid(
    model: PrefixCurveMLP,
    x: torch.Tensor,
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return np.clip(pred.astype(float), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
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
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/24_prefix_curve_ssl_baseline"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--stop-after-epochs", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--prefix-days", type=str, default="1,3,7")
    ap.add_argument("--max-external-curves", type=int, default=None)
    args = ap.parse_args()

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.time()

    prefix_days_list = tuple(float(x.strip()) for x in args.prefix_days.split(",") if x.strip())
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])

    print(f"[24] loading internal 181 from {args.data}")
    internal_records = _load_internal_records(args.data, t_grid=t_grid, min_quality=args.min_quality)
    print(f"[24] kept {len(internal_records)} internal curves")

    x_all, y_all, w_all, task_fids = _build_train_tasks(
        internal_records,
        prefix_days_list=prefix_days_list,
        t_grid=t_grid,
    )
    tr_mask, va_mask = _split_train_val(task_fids, val_frac=args.val_frac, seed=args.seed)
    print(
        f"[24] built {x_all.shape[0]} self-supervised tasks "
        f"from {len(set(task_fids.tolist()))} internal curves"
    )

    model = PrefixCurveMLP(
        t_grid_len=len(t_grid),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    log_rows = _fit_model(
        model,
        x_train=x_all[tr_mask],
        y_train=y_all[tr_mask],
        w_train=w_all[tr_mask],
        x_val=x_all[va_mask],
        y_val=y_all[va_mask],
        w_val=w_all[va_mask],
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after_epochs,
        out=args.out,
    )
    print(f"[24] trained for {len(log_rows)} epochs; best checkpoint saved")

    print(f"[24] loading matched 321 from {args.cross_doi_data}")
    external_records, tail_flags = _load_external_records(
        args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid=t_grid,
        min_quality=args.min_quality,
    )
    if args.max_external_curves is not None and len(external_records) > args.max_external_curves:
        rng = np.random.default_rng(args.seed)
        keep_idx = np.sort(rng.choice(len(external_records), size=args.max_external_curves, replace=False))
        external_records = [external_records[i] for i in keep_idx]
        print(f"[24] subsampled external curves to {len(external_records)}")
    print(f"[24] evaluating on {len(external_records)} external curves")

    per_curve_rows: list[dict[str, object]] = []
    t_grid_np = t_grid.cpu().numpy().astype(float)
    for prefix_days in prefix_days_list:
        eligible = 0
        for rec in external_records:
            x_i, _, future_obs_mask = _make_prefix_input(
                rec.t_obs,
                rec.q_obs,
                prefix_days=prefix_days,
                t_grid=t_grid,
            )
            if x_i is None or future_obs_mask is None:
                continue
            eligible += 1
            pred_grid = _predict_full_grid(model, x_i, device=args.device)
            pred_obs = np.interp(
                rec.t_obs.cpu().numpy().astype(float),
                t_grid_np,
                pred_grid,
                left=pred_grid[0],
                right=pred_grid[-1],
            )
            q_true = rec.q_obs.cpu().numpy().astype(float)[future_obs_mask]
            q_pred = pred_obs[future_obs_mask]
            per_curve_rows.append({
                "prefix_days": prefix_days,
                "method": "curve_ssl",
                "Formulation_Index": rec.fid,
                "n_prefix_obs": int((~future_obs_mask).sum()),
                "n_future_obs": int(future_obs_mask.sum()),
                "R2": float(_r2(q_true, q_pred)),
                "MAE": float(np.mean(np.abs(q_true - q_pred))),
            })
        print(f"[24] prefix_days={prefix_days:g} eligible={eligible}")

    per_curve = pd.DataFrame(per_curve_rows).merge(
        tail_flags,
        on="Formulation_Index",
        how="left",
    )
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
        f.write("=== Curve-only prefix SSL benchmark ===\n\n")
        f.write(f"internal_data       : {args.data}\n")
        f.write(f"external_data       : {args.cross_doi_data}\n")
        f.write(f"prefix_days         : {prefix_days_list}\n")
        f.write(f"hidden_dim          : {args.hidden_dim}\n")
        f.write(f"dropout             : {args.dropout}\n")
        f.write(f"epochs_ran          : {len(log_rows)}\n")
        f.write(f"internal_curves     : {len(internal_records)}\n")
        f.write(f"external_curves     : {len(external_records)}\n")
        f.write(f"wallclock_min       : {(time.time() - overall_t0) / 60:.1f}\n\n")
        f.write("Overall summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSubgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"[24] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
