"""
26 - Iterative partial-observation world model on real PLGA curves.

What this does:
    1. Load the trusted Stage-1 curve posterior q_phi and the internal 181
       high-quality PLGA curves.
    2. For each full 181 curve, infer a teacher theta summary from q_phi.
    3. Build many partial-observation tasks by masking the full 181 curves in
       multiple ways (prefix, sparse, hybrid).
    4. Train a world model that:
         - encodes all observed time/value tokens with a Transformer,
         - predicts a mechanistic state theta within simulator bounds,
         - predicts a learned residual curve,
         - rolls out simulator(theta) + residual,
         - then performs one residual-guided refinement step and rolls out again.
    5. Evaluate on 321 cross-DOI curves using real partial observations and
       score the hidden suffix on original observed time points.

Why this exists:
    This is the first serious attempt at the user's intended world-model
    definition: give the model all observed points, let it learn what matters,
    form an internal state, simulate a curve, compare to observed evidence,
    refine once, and predict the rest. It keeps the simulator in the loop
    without forcing all information through theta alone.

Outputs:
    outputs/26_iterative_world_model/teacher_theta_cache.pt
    outputs/26_iterative_world_model/model.pt
    outputs/26_iterative_world_model/training_log.csv
    outputs/26_iterative_world_model/per_curve_metrics.csv
    outputs/26_iterative_world_model/summary.csv
    outputs/26_iterative_world_model/subgroup_summary.csv
    outputs/26_iterative_world_model/summary.txt

Expected runtime:
    ~10-30 min on a 4060 depending on teacher sampling and task count.
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

from posterior import CurvePosterior, interpolate_to_grid  # noqa: E402
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


@dataclass
class CurveRecord:
    fid: int
    q_full_grid: torch.Tensor
    t_obs: torch.Tensor
    q_obs: torch.Tensor


class IterativeWorldModel(nn.Module):
    def __init__(
        self,
        *,
        t_grid: torch.Tensor,
        theta_low: torch.Tensor,
        theta_high: torch.Tensor,
        model_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.register_buffer("t_grid", t_grid.clone().detach().float())
        self.register_buffer("theta_low", theta_low.clone().detach().float())
        self.register_buffer("theta_high", theta_high.clone().detach().float())
        self.residual_scale = residual_scale
        self.time_norm = float(max(t_grid.max().item(), 1.0))

        self.token_proj = nn.Linear(4, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.theta_head0 = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, theta_low.numel()),
        )
        self.theta_delta_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, theta_low.numel()),
        )
        self.resid_head0 = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.resid_delta_head = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def _encode(
        self,
        q_obs_grid: torch.Tensor,
        obs_mask: torch.Tensor,
        residual_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len = q_obs_grid.shape
        time = (self.t_grid / self.time_norm).view(1, t_len, 1).expand(bsz, -1, -1)
        feats = torch.cat(
            [
                time,
                q_obs_grid.unsqueeze(-1),
                obs_mask.unsqueeze(-1),
                residual_obs.unsqueeze(-1),
            ],
            dim=-1,
        )
        tokens = self.token_proj(feats)
        hidden = self.encoder(tokens)
        denom = obs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden * obs_mask.unsqueeze(-1)).sum(dim=1) / denom
        return hidden, pooled

    def _decode_theta0(self, pooled: torch.Tensor) -> torch.Tensor:
        raw = self.theta_head0(pooled)
        span = self.theta_high - self.theta_low
        return self.theta_low + torch.sigmoid(raw) * span

    def _decode_theta1(self, pooled: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
        span = self.theta_high - self.theta_low
        delta = 0.25 * span * torch.tanh(self.theta_delta_head(pooled))
        return torch.max(torch.min(theta0 + delta, self.theta_high), self.theta_low)

    def _decode_resid0(self, hidden: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        pooled_rep = pooled.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        raw = self.resid_head0(torch.cat([hidden, pooled_rep], dim=-1)).squeeze(-1)
        return self.residual_scale * torch.tanh(raw)

    def _decode_resid1(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        resid0: torch.Tensor,
    ) -> torch.Tensor:
        pooled_rep = pooled.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        delta = self.resid_delta_head(torch.cat([hidden, pooled_rep], dim=-1)).squeeze(-1)
        return resid0 + 0.5 * self.residual_scale * torch.tanh(delta)

    def forward(
        self,
        q_obs_grid: torch.Tensor,
        obs_mask: torch.Tensor,
        simulator: PLGABiphasic,
    ) -> dict[str, torch.Tensor]:
        zero_residual = torch.zeros_like(q_obs_grid)
        hidden0, pooled0 = self._encode(q_obs_grid, obs_mask, zero_residual)
        theta0 = self._decode_theta0(pooled0)
        resid0 = self._decode_resid0(hidden0, pooled0)
        with torch.no_grad():
            base0 = simulator.simulate(theta0, self.t_grid.to(theta0.device))
        pred0 = (base0 + resid0).clamp(0.0, 1.0)

        residual_obs = (q_obs_grid - pred0) * obs_mask
        hidden1, pooled1 = self._encode(q_obs_grid, obs_mask, residual_obs)
        theta1 = self._decode_theta1(pooled1, theta0)
        resid1 = self._decode_resid1(hidden1, pooled1, resid0)
        with torch.no_grad():
            base1 = simulator.simulate(theta1, self.t_grid.to(theta1.device))
        pred1 = (base1 + resid1).clamp(0.0, 1.0)

        return {
            "theta0": theta0,
            "theta1": theta1,
            "base0": base0,
            "base1": base1,
            "resid0": resid0,
            "resid1": resid1,
            "pred0": pred0,
            "pred1": pred1,
        }


def _load_internal_records(
    csv_path: Path,
    t_grid: torch.Tensor,
    min_quality: str,
) -> list[CurveRecord]:
    df = pd.read_csv(csv_path)
    df[Y_COL] = df[Y_COL].astype(float).clip(0.0, 1.0)
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
                t_max_days=float(t_grid.max().item()),
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
                t_max_days=float(t_grid.max().item()),
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
    tail_flags = pd.read_csv(
        Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv")
    )[["Formulation_Index", "fast_regime", "short_window"]]
    return records, tail_flags


def _teacher_theta_cache(
    *,
    records: list[CurveRecord],
    q_phi: CurvePosterior,
    out_path: Path,
    teacher_samples: int,
    seed: int,
) -> torch.Tensor:
    if out_path.exists():
        obj = torch.load(out_path, map_location="cpu")
        theta = obj["teacher_theta"]
        if theta.shape[0] == len(records):
            return theta.float()
    rows: list[torch.Tensor] = []
    for idx, rec in enumerate(records):
        torch.manual_seed(seed + idx)
        samples = q_phi.sample(rec.q_full_grid, n_samples=teacher_samples, show_progress_bars=False)
        rows.append(samples.mean(dim=0).cpu())
    teacher_theta = torch.stack(rows, dim=0)
    torch.save(
        {
            "teacher_theta": teacher_theta,
            "teacher_samples": teacher_samples,
            "fids": [rec.fid for rec in records],
        },
        out_path,
    )
    return teacher_theta.float()


def _sample_train_mask(
    *,
    t_grid: torch.Tensor,
    rng: np.random.Generator,
    prefix_days_list: tuple[float, ...],
) -> torch.Tensor:
    mode = rng.choice(["prefix", "random", "hybrid"], p=[0.40, 0.30, 0.30])
    t_np = t_grid.cpu().numpy().astype(float)
    mask = np.zeros_like(t_np, dtype=np.float32)
    if mode == "prefix":
        horizon = float(rng.choice(prefix_days_list))
        mask[t_np <= horizon] = 1.0
    elif mode == "random":
        keep_frac = float(rng.uniform(0.15, 0.45))
        n_keep = max(4, int(round(keep_frac * len(t_np))))
        idx = np.sort(rng.choice(len(t_np), size=n_keep, replace=False))
        mask[idx] = 1.0
        mask[0] = 1.0
    else:
        horizon = float(rng.choice(prefix_days_list))
        early = t_np <= horizon
        mask[early] = 1.0
        late_idx = np.where(~early)[0]
        if len(late_idx) > 0:
            n_keep_late = max(1, int(round(0.10 * len(late_idx))))
            late_keep = np.sort(rng.choice(late_idx, size=n_keep_late, replace=False))
            mask[late_keep] = 1.0
    if int(mask.sum()) < 4:
        first_idx = np.arange(min(4, len(mask)))
        mask[first_idx] = 1.0
    if int((1.0 - mask).sum()) < 2:
        drop_idx = np.where(mask > 0.5)[0][2:]
        if len(drop_idx) > 0:
            to_drop = rng.choice(drop_idx, size=min(2, len(drop_idx)), replace=False)
            mask[to_drop] = 0.0
    return torch.tensor(mask, dtype=torch.float32)


def _build_training_tasks(
    *,
    records: list[CurveRecord],
    teacher_theta: torch.Tensor,
    t_grid: torch.Tensor,
    prefix_days_list: tuple[float, ...],
    n_masks_per_curve: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_obs_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    q_full_rows: list[torch.Tensor] = []
    theta_rows: list[torch.Tensor] = []
    curve_ids: list[int] = []
    for curve_idx, rec in enumerate(records):
        for _ in range(n_masks_per_curve):
            mask = _sample_train_mask(t_grid=t_grid, rng=rng, prefix_days_list=prefix_days_list)
            q_obs_rows.append(rec.q_full_grid * mask)
            mask_rows.append(mask)
            q_full_rows.append(rec.q_full_grid)
            theta_rows.append(teacher_theta[curve_idx])
            curve_ids.append(curve_idx)
    return (
        torch.stack(q_obs_rows, dim=0),
        torch.stack(mask_rows, dim=0),
        torch.stack(q_full_rows, dim=0),
        torch.stack(theta_rows, dim=0),
        np.asarray(curve_ids, dtype=np.int64),
    )


def _split_train_val(
    curve_ids: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.array(sorted(set(curve_ids.tolist())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_ids = set(uniq[:n_val].tolist())
    is_val = np.array([cid in val_ids for cid in curve_ids], dtype=bool)
    return ~is_val, is_val


def _weighted_curve_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    weights = torch.where(obs_mask > 0.5, torch.full_like(obs_mask, 0.35), torch.ones_like(obs_mask))
    return (((pred - target) ** 2) * weights).mean()


def _resid_smoothness(resid: torch.Tensor) -> torch.Tensor:
    if resid.shape[1] < 2:
        return resid.new_tensor(0.0)
    return ((resid[:, 1:] - resid[:, :-1]) ** 2).mean()


def _fit_world_model(
    *,
    model: IterativeWorldModel,
    simulator: PLGABiphasic,
    q_obs_train: torch.Tensor,
    mask_train: torch.Tensor,
    q_full_train: torch.Tensor,
    theta_train: torch.Tensor,
    q_obs_val: torch.Tensor,
    mask_val: torch.Tensor,
    q_full_val: torch.Tensor,
    theta_val: torch.Tensor,
    batch_size: int,
    lr: float,
    max_epochs: int,
    stop_after_epochs: int,
    device: str,
    out: Path,
) -> list[dict[str, float]]:
    model.to(device)
    theta_span = (model.theta_high - model.theta_low).to(device).clamp_min(1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    q_obs_train = q_obs_train.to(device)
    mask_train = mask_train.to(device)
    q_full_train = q_full_train.to(device)
    theta_train = theta_train.to(device)
    q_obs_val = q_obs_val.to(device)
    mask_val = mask_val.to(device)
    q_full_val = q_full_val.to(device)
    theta_val = theta_val.to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    log_rows: list[dict[str, float]] = []

    def batch_loss(
        q_obs_b: torch.Tensor,
        mask_b: torch.Tensor,
        q_full_b: torch.Tensor,
        theta_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out_dict = model(q_obs_b, mask_b, simulator)
        curve0 = _weighted_curve_loss(out_dict["pred0"], q_full_b, mask_b)
        curve1 = _weighted_curve_loss(out_dict["pred1"], q_full_b, mask_b)
        obs1 = (((out_dict["pred1"] - q_obs_b) ** 2) * mask_b).sum() / mask_b.sum().clamp_min(1.0)
        theta0 = (((out_dict["theta0"] - theta_b) / theta_span) ** 2).mean()
        theta1 = (((out_dict["theta1"] - theta_b) / theta_span) ** 2).mean()
        smooth = _resid_smoothness(out_dict["resid1"])
        loss = (
            0.30 * curve0
            + 1.00 * curve1
            + 0.20 * obs1
            + 0.10 * theta0
            + 0.20 * theta1
            + 0.02 * smooth
        )
        stats = {
            "curve0": float(curve0.item()),
            "curve1": float(curve1.item()),
            "obs1": float(obs1.item()),
            "theta1": float(theta1.item()),
        }
        return loss, stats

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(q_obs_train.shape[0], device=device)
        for start in range(0, q_obs_train.shape[0], batch_size):
            idx = order[start:start + batch_size]
            loss, _ = batch_loss(
                q_obs_train[idx],
                mask_train[idx],
                q_full_train[idx],
                theta_train[idx],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            train_loss, train_stats = batch_loss(q_obs_train, mask_train, q_full_train, theta_train)
            val_loss, val_stats = batch_loss(q_obs_val, mask_val, q_full_val, theta_val)
        log_rows.append({
            "epoch": float(epoch),
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss.item()),
            "train_curve1": train_stats["curve1"],
            "val_curve1": val_stats["curve1"],
            "val_obs1": val_stats["obs1"],
            "val_theta1": val_stats["theta1"],
        })
        pd.DataFrame(log_rows).to_csv(out / "training_log.csv", index=False)
        print(
            f"[26] epoch {epoch:03d} "
            f"train={float(train_loss.item()):.4f} "
            f"val={float(val_loss.item()):.4f} "
            f"val_curve1={val_stats['curve1']:.4f} "
            f"val_obs1={val_stats['obs1']:.4f} "
            f"val_theta1={val_stats['theta1']:.4f}"
        )
        if float(val_loss.item()) < best_val:
            best_val = float(val_loss.item())
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= stop_after_epochs:
            break

    if best_state is None:
        raise RuntimeError("no best checkpoint recorded")
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict()}, out / "model.pt")
    return log_rows


def _prefix_obs_from_real_curve(
    rec: CurveRecord,
    prefix_days: float,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, np.ndarray | None]:
    t_np = rec.t_obs.cpu().numpy().astype(float)
    q_np = rec.q_obs.cpu().numpy().astype(float)
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
    obs_mask_grid = torch.tensor(
        (t_grid.cpu().numpy().astype(float) <= prefix_days).astype(np.float32),
        dtype=torch.float32,
    )
    return q_prefix_grid * obs_mask_grid, obs_mask_grid, future_obs_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument(
        "--posterior",
        type=Path,
        default=Path("outputs/04_curve_posterior/posterior.pt"),
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
    ap.add_argument("--out", type=Path, default=Path("outputs/26_iterative_world_model"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-samples", type=int, default=128)
    ap.add_argument("--n-masks-per-curve", type=int, default=16)
    ap.add_argument("--model-dim", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--residual-scale", type=float, default=0.35)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--stop-after-epochs", type=int, default=20)
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
    sim = PLGABiphasic(
        rtol=cfg["simulator"]["rtol"],
        atol=cfg["simulator"]["atol"],
        method=cfg["simulator"]["method"],
    )
    prior = sim.prior()
    theta_low = prior.base_dist.low.detach().cpu().float()
    theta_high = prior.base_dist.high.detach().cpu().float()

    print(f"[26] loading q_phi from {args.posterior}")
    q_phi = CurvePosterior.load(args.posterior, simulator=sim, device=args.device)
    print(f"[26] loading internal curves from {args.data}")
    internal_records = _load_internal_records(args.data, t_grid=t_grid, min_quality=args.min_quality)
    print(f"[26] kept {len(internal_records)} internal high-quality curves")

    teacher_cache = args.out / "teacher_theta_cache.pt"
    teacher_theta = _teacher_theta_cache(
        records=internal_records,
        q_phi=q_phi,
        out_path=teacher_cache,
        teacher_samples=args.teacher_samples,
        seed=args.seed,
    )
    print(f"[26] teacher theta cache ready: {teacher_cache}")

    q_obs_all, mask_all, q_full_all, theta_all, curve_ids = _build_training_tasks(
        records=internal_records,
        teacher_theta=teacher_theta,
        t_grid=t_grid,
        prefix_days_list=prefix_days_list,
        n_masks_per_curve=args.n_masks_per_curve,
        seed=args.seed,
    )
    print(f"[26] built {q_obs_all.shape[0]} masked tasks")
    tr_mask, va_mask = _split_train_val(curve_ids, val_frac=args.val_frac, seed=args.seed)

    model = IterativeWorldModel(
        t_grid=t_grid,
        theta_low=theta_low,
        theta_high=theta_high,
        model_dim=args.model_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    )
    log_rows = _fit_world_model(
        model=model,
        simulator=sim,
        q_obs_train=q_obs_all[tr_mask],
        mask_train=mask_all[tr_mask],
        q_full_train=q_full_all[tr_mask],
        theta_train=theta_all[tr_mask],
        q_obs_val=q_obs_all[va_mask],
        mask_val=mask_all[va_mask],
        q_full_val=q_full_all[va_mask],
        theta_val=theta_all[va_mask],
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after_epochs,
        device=args.device,
        out=args.out,
    )
    print(f"[26] trained for {len(log_rows)} epochs")

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
        print(f"[26] subsampled external curves to {len(external_records)}")

    model.to(args.device)
    model.eval()
    per_curve_rows: list[dict[str, object]] = []
    t_grid_np = t_grid.cpu().numpy().astype(float)
    for prefix_days in prefix_days_list:
        eligible = 0
        for rec in external_records:
            q_obs_grid, obs_mask_grid, future_obs_mask = _prefix_obs_from_real_curve(
                rec,
                prefix_days=prefix_days,
                t_grid=t_grid,
            )
            if q_obs_grid is None or obs_mask_grid is None or future_obs_mask is None:
                continue
            eligible += 1
            with torch.no_grad():
                out_dict = model(
                    q_obs_grid.unsqueeze(0).to(args.device),
                    obs_mask_grid.unsqueeze(0).to(args.device),
                    sim,
                )
            pred0_grid = out_dict["pred0"].squeeze(0).cpu().numpy().astype(float)
            pred1_grid = out_dict["pred1"].squeeze(0).cpu().numpy().astype(float)
            pred0_obs = np.interp(rec.t_obs.cpu().numpy().astype(float), t_grid_np, pred0_grid)
            pred1_obs = np.interp(rec.t_obs.cpu().numpy().astype(float), t_grid_np, pred1_grid)
            q_true = rec.q_obs.cpu().numpy().astype(float)[future_obs_mask]
            q_hat0 = pred0_obs[future_obs_mask]
            q_hat1 = pred1_obs[future_obs_mask]
            per_curve_rows.append({
                "prefix_days": prefix_days,
                "method": "iter_world_stage0",
                "Formulation_Index": rec.fid,
                "n_prefix_obs": int((~future_obs_mask).sum()),
                "n_future_obs": int(future_obs_mask.sum()),
                "R2": float(_r2(q_true, q_hat0)),
                "MAE": float(np.mean(np.abs(q_true - q_hat0))),
            })
            per_curve_rows.append({
                "prefix_days": prefix_days,
                "method": "iter_world_stage1",
                "Formulation_Index": rec.fid,
                "n_prefix_obs": int((~future_obs_mask).sum()),
                "n_future_obs": int(future_obs_mask.sum()),
                "R2": float(_r2(q_true, q_hat1)),
                "MAE": float(np.mean(np.abs(q_true - q_hat1))),
            })
        print(f"[26] prefix_days={prefix_days:g} eligible={eligible}")

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
        f.write("=== Iterative masked-time world model ===\n\n")
        f.write(f"posterior           : {args.posterior}\n")
        f.write(f"teacher_samples     : {args.teacher_samples}\n")
        f.write(f"n_masks_per_curve   : {args.n_masks_per_curve}\n")
        f.write(f"prefix_days         : {prefix_days_list}\n")
        f.write(f"model_dim           : {args.model_dim}\n")
        f.write(f"nhead               : {args.nhead}\n")
        f.write(f"num_layers          : {args.num_layers}\n")
        f.write(f"dropout             : {args.dropout}\n")
        f.write(f"residual_scale      : {args.residual_scale}\n")
        f.write(f"epochs_ran          : {len(log_rows)}\n")
        f.write(f"internal_curves     : {len(internal_records)}\n")
        f.write(f"external_curves     : {len(external_records)}\n")
        f.write(f"wallclock_min       : {(time.time() - overall_t0) / 60:.1f}\n\n")
        f.write("Overall summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nSubgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        f.write("\n")

    print(f"[26] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
