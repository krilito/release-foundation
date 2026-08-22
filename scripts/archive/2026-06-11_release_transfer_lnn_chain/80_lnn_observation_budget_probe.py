"""80 - LNN/CfC-style observation-budget probe on the canonical PLGA split.

Purpose:
    Test whether a small continuous-time neural backbone is worth pursuing for
    the new information-budget framing. This is a probe, not a paper claim.

    The model consumes formulation descriptors plus 0/1/2/4 early release
    observations and predicts the future curve on the same canonical future
    grid used by `72_canonical_benchmark_v2.py`.

    It intentionally does not claim physical states. The hidden state is a
    learned continuous-time release representation.

Produces:
    outputs/80_lnn_observation_budget_probe/summary.csv
    outputs/80_lnn_observation_budget_probe/per_curve_results.csv
    outputs/80_lnn_observation_budget_probe/pairwise_vs_known_baselines.csv
    outputs/80_lnn_observation_budget_probe/lock_metadata.json
    outputs/80_lnn_observation_budget_probe/summary.txt

Run:
    .\\.venv\\Scripts\\python.exe scripts\\80_lnn_observation_budget_probe.py --epochs 800
    .\\.venv\\Scripts\\python.exe scripts\\80_lnn_observation_budget_probe.py --smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_72 = Path("scripts/72_canonical_benchmark_v2.py")
OUTPUT_DIR = Path("outputs/80_lnn_observation_budget_probe")
OUTPUT_72_DIR = Path("outputs/72_canonical_benchmark_v2")
OUTPUT_38B_DIR = Path("outputs/38b_canonical_functional_baselines")
EARLY_SCHEDULES: dict[str, tuple[float, ...]] = {
    "obs0": (),
    "obs1": (3.0,),
    "obs2": (1.0, 7.0),
    "obs4": (1.0, 3.0, 5.0, 7.0),
}
FUTURE_START = 14.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=D:/release-foundation", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_benchmark72_module() -> Any:
    spec = importlib.util.spec_from_file_location("benchmark72_v2", SCRIPT_72)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_72}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiquidCell(nn.Module):
    """Small liquid-time-constant style recurrent cell.

    This is a lightweight CfC/LTC-inspired cell:

        h_next = h + alpha(dt, x, h) * (candidate(x, h) - h)

    where alpha is bounded in (0, 1). It gives the model an explicit
    continuous-time update without introducing a new dependency.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.log_tau = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=-1)
        cand = self.candidate(hx)
        tau = self.log_tau(hx) + 1e-3
        alpha = 1.0 - torch.exp(-torch.clamp(dt, min=0.0) / tau)
        return h + alpha * (cand - h)


class ReleaseLNN(nn.Module):
    """Condition on formulation + early observations, then decode Q(t)."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        time_grid: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("time_grid", torch.tensor(time_grid, dtype=torch.float32))
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.cell = LiquidCell(hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.qmax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_feat: torch.Tensor, curve_context: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        h = self.feature_encoder(x_feat)
        batch = x_feat.shape[0]
        prev_t = torch.zeros(batch, 1, device=x_feat.device, dtype=x_feat.dtype)

        for j, t_val in enumerate(self.time_grid):
            mask_j = obs_mask[:, j:j + 1]
            q_j = curve_context[:, j:j + 1]
            t_j = torch.full_like(q_j, float(t_val))
            obs_in = torch.cat([t_j / 84.0, q_j, mask_j], dim=-1)
            obs_x = self.obs_embed(obs_in * mask_j)
            dt = (t_j - prev_t).clamp_min(0.0)
            h_candidate = self.cell(h, obs_x, dt)
            h = torch.where(mask_j > 0.5, h_candidate, h)
            prev_t = torch.where(mask_j > 0.5, t_j, prev_t)

        q_steps: list[torch.Tensor] = []
        for t_val in self.time_grid:
            t_col = torch.full((batch, 1), float(t_val) / 84.0, device=x_feat.device, dtype=x_feat.dtype)
            raw = self.readout(torch.cat([h, t_col], dim=-1))
            q_steps.append(raw)
        raw_curve = torch.cat(q_steps, dim=1)

        # Shape constraint: cumulative positive increments, then scale into [0, 1.1].
        increments = F.softplus(raw_curve) + 1e-5
        cum = torch.cumsum(increments, dim=1)
        qmax = 0.35 + 0.85 * torch.sigmoid(self.qmax_head(h))
        q = qmax * cum / (cum[:, -1:] + 1e-6)
        return q


def make_obs_mask(time_grid: np.ndarray, obs_times: tuple[float, ...]) -> np.ndarray:
    mask = np.zeros(len(time_grid), dtype=np.float32)
    for t in obs_times:
        idx = int(np.argmin(np.abs(time_grid - t)))
        mask[idx] = 1.0
    return mask


def train_one_budget(
    *,
    x_train: np.ndarray,
    curves_train: np.ndarray,
    x_test: np.ndarray,
    curves_test: np.ndarray,
    time_grid: np.ndarray,
    obs_times: tuple[float, ...],
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    seed_all(seed)
    model = ReleaseLNN(x_train.shape[1], hidden_dim, time_grid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(curves_train, dtype=torch.float32, device=device)
    x_te = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(curves_test, dtype=torch.float32, device=device)
    obs_mask_np = make_obs_mask(time_grid, obs_times)
    obs_mask_tr = torch.tensor(obs_mask_np[None, :].repeat(len(x_train), axis=0), dtype=torch.float32, device=device)
    obs_mask_te = torch.tensor(obs_mask_np[None, :].repeat(len(x_test), axis=0), dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        pred = model(x_tr, y_tr * obs_mask_tr, obs_mask_tr)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.detach().cpu())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(x_te, y_te * obs_mask_te, obs_mask_te).cpu().numpy()
    return pred_test, {"train_mse": best_loss, "n_params": float(sum(p.numel() for p in model.parameters()))}


def load_known_baseline_rmse(test_curve_ids: list[int]) -> dict[str, np.ndarray]:
    baselines: dict[str, np.ndarray] = {}
    sources = [
        (OUTPUT_72_DIR / "per_curve_results.csv", "72"),
        (OUTPUT_38B_DIR / "per_curve_results.csv", "38b"),
    ]
    for path, prefix in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"curve_id", "method", "rmse"}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df["curve_id"] = df["curve_id"].astype(int)
        for method, sub in df.groupby("method"):
            ordered = sub.set_index("curve_id").loc[test_curve_ids]
            baselines[f"{prefix}:{method}"] = ordered["rmse"].to_numpy(dtype=float)
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 20)
        args.hidden_dim = min(args.hidden_dim, 24)
        args.out = Path(str(args.out) + "_smoke")

    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bench72 = load_benchmark72_module()
    split_df = bench72.load_canonical_split()
    formulations, curves, time_grid, _theta_matrix, common_ids = bench72.load_data(split_df=split_df)
    split_names = split_df["split_name"].to_numpy()
    train_idx = np.flatnonzero(split_names == "train")
    cal_idx = np.flatnonzero(split_names == "cal")
    test_idx = np.flatnonzero(split_names == "test")

    x_train, _x_cal, x_test, feature_cols, _preprocessor = bench72.prepare_features_split(
        formulations, train_idx, test_idx, cal_idx
    )
    curves_train, curves_test = curves[train_idx], curves[test_idx]
    test_curve_ids = [int(common_ids[i]) for i in test_idx]
    future_mask = time_grid > FUTURE_START
    y_test_future = curves_test[:, future_mask]

    summary_rows: list[dict[str, object]] = []
    per_curve_rows: list[dict[str, object]] = []
    schedule_rmse: dict[str, np.ndarray] = {}

    for schedule_name, obs_times in EARLY_SCHEDULES.items():
        pred, train_meta = train_one_budget(
            x_train=x_train,
            curves_train=curves_train,
            x_test=x_test,
            curves_test=curves_test,
            time_grid=time_grid,
            obs_times=obs_times,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + len(obs_times) * 97,
            device=device,
        )
        pred_future = np.clip(pred[:, future_mask], 0.0, 1.1)
        per_curve = bench72.per_curve_rmse(y_test_future, pred_future)
        schedule_rmse[schedule_name] = per_curve
        summary_rows.append(
            {
                "method": "LNN-probe",
                "schedule": schedule_name,
                "n_obs": len(obs_times),
                "obs_times": ",".join(str(t) for t in obs_times) if obs_times else "none",
                "n_test": int(len(test_idx)),
                "rmse": float(bench72.pooled_rmse(y_test_future, pred_future)),
                "r2_pooled": float(bench72.pooled_r2(y_test_future, pred_future)),
                "per_curve_rmse_mean": float(per_curve.mean()),
                "per_curve_rmse_median": float(np.median(per_curve)),
                "train_mse": float(train_meta["train_mse"]),
                "n_params": int(train_meta["n_params"]),
            }
        )
        for curve_id, rmse in zip(test_curve_ids, per_curve):
            per_curve_rows.append(
                {
                    "curve_id": int(curve_id),
                    "method": "LNN-probe",
                    "schedule": schedule_name,
                    "n_obs": len(obs_times),
                    "obs_times": ",".join(str(t) for t in obs_times) if obs_times else "none",
                    "rmse": float(rmse),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    per_curve_df = pd.DataFrame(per_curve_rows)
    summary_df.to_csv(args.out / "summary.csv", index=False)
    per_curve_df.to_csv(args.out / "per_curve_results.csv", index=False)

    known_baselines = load_known_baseline_rmse(test_curve_ids)
    pairwise_rows: list[dict[str, object]] = []
    for schedule_name, lnn_rmse in schedule_rmse.items():
        for baseline_name, baseline_rmse in known_baselines.items():
            delta, ci_low, ci_high = bench72.bootstrap_paired_ci(
                lnn_rmse,
                baseline_rmse,
                n_resamples=2000 if not args.smoke else 200,
                seed=args.seed,
            )
            pairwise_rows.append(
                {
                    "lnn_schedule": schedule_name,
                    "baseline": baseline_name,
                    "delta_rmse_lnn_minus_baseline": float(delta),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "lnn_better": bool(ci_high < 0.0),
                    "baseline_better": bool(ci_low > 0.0),
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(args.out / "pairwise_vs_known_baselines.csv", index=False)

    metadata = {
        "script": "scripts/80_lnn_observation_budget_probe.py",
        "git_hash": get_git_hash(),
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "early_schedules": {k: list(v) for k, v in EARLY_SCHEDULES.items()},
        "future_start": FUTURE_START,
        "feature_cols": feature_cols,
        "note": "Probe only. Hidden state is learned representation, not physical state.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "=== 80 -- LNN observation-budget probe ===",
        f"device: {device}",
        f"epochs: {args.epochs}",
        "",
        summary_df.to_string(index=False),
        "",
        "--- paired comparisons vs known baselines (delta = LNN - baseline) ---",
        pairwise_df.to_string(index=False) if len(pairwise_df) else "Known baseline outputs not found.",
        "",
        "Interpret as a backbone probe only; compare against locked black-box baselines before making claims.",
    ]
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
