"""
27 - Partial-curve amortized SBI with diagonal Gaussian posterior.

What this does:
    1. Reuse the PLGA simulator prior and online synthetic sampling to train a
       prefix-aware posterior q(theta | partial curve).
    2. Represent a partial curve as gridded observed Q values plus an
       observation mask on the canonical t_grid.
    3. Encode (time, observed Q, mask) tokens with a Transformer and predict a
       diagonal Gaussian posterior over theta.
    4. Validate the posterior on held-out synthetic prefixes via PIT and
       68/95% coverage.
    5. Evaluate on external 321 curves by conditioning on true 1d/3d/7d
       prefixes, rolling the simulator with posterior point estimates, and
       scoring suffix-only R²/MAE on original observation times.

Why this exists:
    script26 mixed three confounded ideas: full-curve teacher means,
    unconstrained residuals, and a free refinement step. This script resets to
    the cleaner question: can amortized SBI on synthetic partial curves learn a
    calibrated posterior over theta that transfers to real 321 suffix
    forecasting?

Outputs:
    outputs/27_partial_curve_npe/model.pt
    outputs/27_partial_curve_npe/training_log.csv
    outputs/27_partial_curve_npe/calibration_summary.csv
    outputs/27_partial_curve_npe/calibration_by_dim.csv
    outputs/27_partial_curve_npe/pit_histograms.csv
    outputs/27_partial_curve_npe/prefix_sd_summary.csv
    outputs/27_partial_curve_npe/per_curve_metrics.csv
    outputs/27_partial_curve_npe/summary.csv
    outputs/27_partial_curve_npe/subgroup_summary.csv
    outputs/27_partial_curve_npe/summary.txt

Expected runtime:
    ~10-25 min on a 4060 for the default settings.
"""
from __future__ import annotations

import argparse
import math
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
    t_obs: torch.Tensor
    q_obs: torch.Tensor


class PartialCurvePosterior(nn.Module):
    def __init__(
        self,
        *,
        t_grid: torch.Tensor,
        theta_dim: int,
        model_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("t_grid", t_grid.clone().detach().float())
        self.time_norm = float(max(t_grid.max().item(), 1.0))
        self.token_proj = nn.Linear(3, model_dim)
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
        self.mu_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, theta_dim),
        )
        self.log_sd_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, theta_dim),
        )

    def forward(self, q_obs_grid: torch.Tensor, obs_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len = q_obs_grid.shape
        time = (self.t_grid / self.time_norm).view(1, t_len, 1).expand(bsz, -1, -1)
        tokens = torch.cat(
            [
                time,
                q_obs_grid.unsqueeze(-1),
                obs_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.encoder(self.token_proj(tokens))
        denom = obs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden * obs_mask.unsqueeze(-1)).sum(dim=1) / denom
        mu = self.mu_head(pooled)
        log_sd = torch.clamp(self.log_sd_head(pooled), min=-5.0, max=2.0)
        return mu, log_sd


def _normal_nll(theta_true: torch.Tensor, mu: torch.Tensor, log_sd: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-2.0 * log_sd)
    return 0.5 * (((theta_true - mu) ** 2) * inv_var + 2.0 * log_sd).sum(dim=1).mean()


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
        mask[: min(4, len(mask))] = 1.0
    if int((1.0 - mask).sum()) < 2:
        keep_idx = np.where(mask > 0.5)[0]
        if len(keep_idx) > 4:
            drop_idx = rng.choice(keep_idx[2:], size=min(2, len(keep_idx) - 2), replace=False)
            mask[drop_idx] = 0.0
    return torch.tensor(mask, dtype=torch.float32)


def _make_prefix_mask(prefix_days: float, t_grid: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        (t_grid.cpu().numpy().astype(float) <= prefix_days).astype(np.float32),
        dtype=torch.float32,
    )


def _read_noise_sigma(curve_posterior_ckpt: Path) -> float:
    ckpt = torch.load(curve_posterior_ckpt, map_location="cpu", weights_only=False)
    sigma = ckpt.get("noise_sigma")
    if sigma is None:
        return 0.02
    return float(sigma)


def _sample_batch(
    *,
    sim: PLGABiphasic,
    t_grid: torch.Tensor,
    batch_size: int,
    noise_sigma: float,
    prefix_days_list: tuple[float, ...],
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = sim.sample_prior(batch_size)
    with torch.no_grad():
        q_full = sim.simulate(theta.to(device), t_grid.to(device)).cpu()
    q_obs_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for i in range(batch_size):
        mask = _sample_train_mask(t_grid=t_grid, rng=rng, prefix_days_list=prefix_days_list)
        q_obs = q_full[i].clone()
        if noise_sigma > 0:
            q_obs = (q_obs + noise_sigma * torch.randn_like(q_obs) * mask).clamp(0.0, 1.0)
        q_obs_rows.append(q_obs * mask)
        mask_rows.append(mask)
    return (
        torch.stack(q_obs_rows, dim=0),
        torch.stack(mask_rows, dim=0),
        theta.cpu().float(),
    )


def _build_prefix_heldout(
    *,
    sim: PLGABiphasic,
    t_grid: torch.Tensor,
    n_total: int,
    noise_sigma: float,
    prefix_days_list: tuple[float, ...],
    seed: int,
    device: str,
) -> dict[float, dict[str, torch.Tensor]]:
    tasks_per_prefix = [n_total // len(prefix_days_list)] * len(prefix_days_list)
    for i in range(n_total % len(prefix_days_list)):
        tasks_per_prefix[i] += 1

    out: dict[float, dict[str, torch.Tensor]] = {}
    for prefix_days, n_prefix in zip(prefix_days_list, tasks_per_prefix, strict=True):
        theta = sim.sample_prior(n_prefix)
        with torch.no_grad():
            q_full = sim.simulate(theta.to(device), t_grid.to(device)).cpu()
        mask = _make_prefix_mask(prefix_days, t_grid).view(1, -1).repeat(n_prefix, 1)
        q_obs = q_full.clone()
        if noise_sigma > 0:
            q_obs = (q_obs + noise_sigma * torch.randn_like(q_obs) * mask).clamp(0.0, 1.0)
        out[float(prefix_days)] = {
            "q_obs_grid": q_obs * mask,
            "obs_mask": mask,
            "theta_true": theta.cpu().float(),
        }
    return out


def _fit_model(
    *,
    model: PartialCurvePosterior,
    sim: PLGABiphasic,
    t_grid: torch.Tensor,
    prefix_days_list: tuple[float, ...],
    noise_sigma: float,
    epoch_n_sim: int,
    batch_size: int,
    lr: float,
    max_epochs: int,
    stop_after_epochs: int,
    val_data: dict[float, dict[str, torch.Tensor]],
    device: str,
    seed: int,
    out: Path,
) -> list[dict[str, float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    log_rows: list[dict[str, float]] = []

    def eval_val() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for data in val_data.values():
                mu, log_sd = model(
                    data["q_obs_grid"].to(device),
                    data["obs_mask"].to(device),
                )
                loss = _normal_nll(data["theta_true"].to(device), mu, log_sd)
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    for epoch in range(1, max_epochs + 1):
        model.train()
        q_obs_epoch, mask_epoch, theta_epoch = _sample_batch(
            sim=sim,
            t_grid=t_grid,
            batch_size=epoch_n_sim,
            noise_sigma=noise_sigma,
            prefix_days_list=prefix_days_list,
            rng=rng,
            device=device,
        )
        q_obs_epoch = q_obs_epoch.to(device)
        mask_epoch = mask_epoch.to(device)
        theta_epoch = theta_epoch.to(device)
        order = torch.randperm(epoch_n_sim, device=device)
        batch_losses: list[float] = []
        for start in range(0, epoch_n_sim, batch_size):
            idx = order[start:start + batch_size]
            mu, log_sd = model(q_obs_epoch[idx], mask_epoch[idx])
            loss = _normal_nll(theta_epoch[idx], mu, log_sd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            batch_losses.append(float(loss.item()))

        val_nll = eval_val()
        mean_train = float(np.mean(batch_losses))
        log_rows.append({"epoch": float(epoch), "train_nll": mean_train, "val_nll": val_nll})
        pd.DataFrame(log_rows).to_csv(out / "training_log.csv", index=False)
        print(f"[27] epoch {epoch:03d} train_nll={mean_train:.4f} val_nll={val_nll:.4f}")

        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= stop_after_epochs:
            break

    if best_state is None:
        raise RuntimeError("no best checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict()}, out / "model.pt")
    return log_rows


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
        records.append(CurveRecord(fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    tail_flags = pd.read_csv(
        Path("outputs/sprint1_14_cross_doi_failure_tail/per_curve_enriched.csv")
    )[["Formulation_Index", "fast_regime", "short_window"]]
    return records, tail_flags


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
    obs_mask_grid = _make_prefix_mask(prefix_days, t_grid)
    return q_prefix_grid * obs_mask_grid, obs_mask_grid, future_obs_mask


def _clamp_theta(theta: torch.Tensor, theta_low: torch.Tensor, theta_high: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(theta, theta_high), theta_low)


def _std_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _evaluate_calibration(
    *,
    model: PartialCurvePosterior,
    heldout: dict[float, dict[str, torch.Tensor]],
    device: str,
    out: Path,
) -> pd.DataFrame:
    model.eval()
    calib_rows: list[dict[str, float]] = []
    dim_rows: list[dict[str, float]] = []
    hist_rows: list[dict[str, float]] = []
    z95 = 1.959963984540054
    with torch.no_grad():
        for prefix_days, data in heldout.items():
            mu, log_sd = model(
                data["q_obs_grid"].to(device),
                data["obs_mask"].to(device),
            )
            mu = mu.cpu()
            log_sd = log_sd.cpu()
            sd = torch.exp(log_sd)
            theta_true = data["theta_true"]
            z = (theta_true - mu) / sd
            pit = _std_normal_cdf(z).clamp(0.0, 1.0)
            cov68 = ((theta_true >= (mu - sd)) & (theta_true <= (mu + sd))).float().mean(dim=0)
            cov95 = ((theta_true >= (mu - z95 * sd)) & (theta_true <= (mu + z95 * sd))).float().mean(dim=0)
            mean_sd = sd.mean(dim=0)
            pit_mean = pit.mean(dim=0)
            pit_abs_dev = (pit_mean - 0.5).abs()
            for d in range(theta_true.shape[1]):
                dim_rows.append({
                    "prefix_days": float(prefix_days),
                    "theta_dim": int(d),
                    "coverage68": float(cov68[d].item()),
                    "coverage95": float(cov95[d].item()),
                    "mean_sd": float(mean_sd[d].item()),
                    "pit_mean": float(pit_mean[d].item()),
                    "pit_abs_dev_from_0p5": float(pit_abs_dev[d].item()),
                })
                hist, edges = np.histogram(pit[:, d].numpy(), bins=10, range=(0.0, 1.0))
                for b in range(len(hist)):
                    hist_rows.append({
                        "prefix_days": float(prefix_days),
                        "theta_dim": int(d),
                        "bin_left": float(edges[b]),
                        "bin_right": float(edges[b + 1]),
                        "count": int(hist[b]),
                    })
            calib_rows.append({
                "prefix_days": float(prefix_days),
                "n_tasks": float(theta_true.shape[0]),
                "coverage68_mean": float(cov68.mean().item()),
                "coverage95_mean": float(cov95.mean().item()),
                "mean_sd_over_dims": float(mean_sd.mean().item()),
                "pit_abs_dev_mean": float(pit_abs_dev.mean().item()),
            })

    calib_df = pd.DataFrame(calib_rows).sort_values("prefix_days")
    dim_df = pd.DataFrame(dim_rows).sort_values(["prefix_days", "theta_dim"])
    hist_df = pd.DataFrame(hist_rows).sort_values(["prefix_days", "theta_dim", "bin_left"])
    calib_df.to_csv(out / "calibration_summary.csv", index=False)
    dim_df.to_csv(out / "calibration_by_dim.csv", index=False)
    hist_df.to_csv(out / "pit_histograms.csv", index=False)
    return calib_df


def _read_baseline_table(summary_path: Path, method_col_value: str | None = None) -> pd.DataFrame | None:
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if method_col_value is not None and "method" in df.columns:
        df = df[df["method"] == method_col_value].copy()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument("--curve-posterior-ckpt", type=Path, default=Path("outputs/04_curve_posterior/posterior.pt"))
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
    ap.add_argument("--out", type=Path, default=Path("outputs/27_partial_curve_npe"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch-n-sim", type=int, default=4096)
    ap.add_argument("--heldout-n", type=int, default=10000)
    ap.add_argument("--val-n", type=int, default=2048)
    ap.add_argument("--model-dim", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--stop-after-epochs", type=int, default=10)
    ap.add_argument("--prefix-days", type=str, default="1,3,7")
    ap.add_argument("--min-quality", choices=("high", "medium", "low"), default="high")
    ap.add_argument("--mc-samples", type=int, default=100)
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
    theta_dim = theta_low.numel()
    noise_sigma = _read_noise_sigma(args.curve_posterior_ckpt)

    print(f"[27] theta_dim={theta_dim}")
    print(f"[27] using noise_sigma={noise_sigma:.4f} from {args.curve_posterior_ckpt}")

    heldout = _build_prefix_heldout(
        sim=sim,
        t_grid=t_grid,
        n_total=args.heldout_n,
        noise_sigma=noise_sigma,
        prefix_days_list=prefix_days_list,
        seed=args.seed + 17,
        device=args.device,
    )
    val_data = _build_prefix_heldout(
        sim=sim,
        t_grid=t_grid,
        n_total=args.val_n,
        noise_sigma=noise_sigma,
        prefix_days_list=prefix_days_list,
        seed=args.seed + 29,
        device=args.device,
    )

    model = PartialCurvePosterior(
        t_grid=t_grid,
        theta_dim=theta_dim,
        model_dim=args.model_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_log = _fit_model(
        model=model,
        sim=sim,
        t_grid=t_grid,
        prefix_days_list=prefix_days_list,
        noise_sigma=noise_sigma,
        epoch_n_sim=args.epoch_n_sim,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after_epochs,
        val_data=val_data,
        device=args.device,
        seed=args.seed,
        out=args.out,
    )
    print(f"[27] trained for {len(train_log)} epochs")

    calib_df = _evaluate_calibration(
        model=model,
        heldout=heldout,
        device=args.device,
        out=args.out,
    )

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
        print(f"[27] subsampled external curves to {len(external_records)}")

    model.to(args.device)
    model.eval()
    per_curve_rows: list[dict[str, object]] = []
    sd_rows: list[dict[str, float]] = []
    theta_low_dev = theta_low.to(args.device)
    theta_high_dev = theta_high.to(args.device)
    normal = torch.distributions.Normal(loc=torch.tensor(0.0, device=args.device), scale=torch.tensor(1.0, device=args.device))
    with torch.no_grad():
        for prefix_days in prefix_days_list:
            eligible = 0
            for rec in external_records:
                q_obs_grid, obs_mask_grid, future_obs_mask = _prefix_obs_from_real_curve(rec, prefix_days, t_grid)
                if q_obs_grid is None or obs_mask_grid is None or future_obs_mask is None:
                    continue
                eligible += 1
                mu, log_sd = model(
                    q_obs_grid.unsqueeze(0).to(args.device),
                    obs_mask_grid.unsqueeze(0).to(args.device),
                )
                mu = mu.squeeze(0)
                sd = torch.exp(log_sd.squeeze(0))
                sd_rows.append({
                    "prefix_days": float(prefix_days),
                    "Formulation_Index": int(rec.fid),
                    "mean_sd": float(sd.mean().item()),
                })

                theta_mean = mu
                theta_det = mu
                eps = normal.sample((args.mc_samples, theta_dim))
                theta_mcmean = (mu.unsqueeze(0) + sd.unsqueeze(0) * eps).mean(dim=0)
                theta_cases = {
                    "npe_theta_mean": theta_mean,
                    "npe_theta_det": theta_det,
                    "npe_theta_mcmean100": theta_mcmean,
                }
                for method, theta_hat in theta_cases.items():
                    theta_roll = _clamp_theta(theta_hat, theta_low_dev, theta_high_dev)
                    q_pred = sim.simulate(theta_roll.unsqueeze(0), rec.t_obs.to(args.device)).squeeze(0).cpu().numpy()
                    q_true = rec.q_obs.cpu().numpy().astype(float)[future_obs_mask]
                    q_hat = q_pred.astype(float)[future_obs_mask]
                    per_curve_rows.append({
                        "prefix_days": prefix_days,
                        "method": method,
                        "Formulation_Index": rec.fid,
                        "n_prefix_obs": int((~future_obs_mask).sum()),
                        "n_future_obs": int(future_obs_mask.sum()),
                        "R2": float(_r2(q_true, q_hat)),
                        "MAE": float(np.mean(np.abs(q_true - q_hat))),
                    })
            print(f"[27] prefix_days={prefix_days:g} eligible={eligible}")

    sd_df = pd.DataFrame(sd_rows)
    sd_summary = (
        sd_df.groupby("prefix_days", as_index=False)
        .agg(
            n_curves=("Formulation_Index", "count"),
            mean_sd=("mean_sd", "mean"),
            median_sd=("mean_sd", "median"),
        )
        .sort_values("prefix_days")
    )
    sd_summary.to_csv(args.out / "prefix_sd_summary.csv", index=False)

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

    baseline_24 = _read_baseline_table(Path("outputs/24_prefix_curve_ssl_baseline_full259/summary.csv"), "curve_ssl")
    baseline_25 = _read_baseline_table(Path("outputs/25_prefix_theta_regressor_full259/summary.csv"), "prefix_theta_reg")
    baseline_26 = _read_baseline_table(Path("outputs/26_iterative_world_model_v2/summary.csv"))

    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== Partial-curve diagonal-Gaussian NPE ===\n\n")
        f.write(f"curve_posterior_ckpt : {args.curve_posterior_ckpt}\n")
        f.write(f"theta_dim            : {theta_dim}\n")
        f.write(f"theta_low            : {theta_low.tolist()}\n")
        f.write(f"theta_high           : {theta_high.tolist()}\n")
        f.write(f"noise_sigma          : {noise_sigma:.4f}\n")
        f.write(f"epoch_n_sim          : {args.epoch_n_sim}\n")
        f.write(f"heldout_n            : {args.heldout_n}\n")
        f.write(f"val_n                : {args.val_n}\n")
        f.write(f"prefix_days          : {prefix_days_list}\n")
        f.write(f"model_dim            : {args.model_dim}\n")
        f.write(f"nhead                : {args.nhead}\n")
        f.write(f"num_layers           : {args.num_layers}\n")
        f.write(f"dropout              : {args.dropout}\n")
        f.write(f"batch_size           : {args.batch_size}\n")
        f.write(f"epochs_ran           : {len(train_log)}\n")
        f.write(f"external_curves      : {len(external_records)}\n")
        f.write(f"wallclock_min        : {(time.time() - overall_t0) / 60:.1f}\n\n")
        f.write("Calibration summary:\n")
        f.write(calib_df.to_string(index=False))
        f.write("\n\nPrefix posterior width summary:\n")
        f.write(sd_summary.to_string(index=False))
        f.write("\n\nExternal suffix summary:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nExternal subgroup summary:\n")
        f.write(subgroup.to_string(index=False))
        if baseline_24 is not None:
            f.write("\n\nBaseline 24 (curve_ssl):\n")
            f.write(baseline_24.to_string(index=False))
        if baseline_25 is not None:
            f.write("\n\nBaseline 25 (prefix_theta_reg):\n")
            f.write(baseline_25.to_string(index=False))
        if baseline_26 is not None:
            f.write("\n\nBaseline 26 (iterative world model):\n")
            f.write(baseline_26.to_string(index=False))
        f.write("\n")

    print(f"[27] wrote {args.out / 'summary.txt'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
