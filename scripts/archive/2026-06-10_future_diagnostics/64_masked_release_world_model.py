"""
64 - Masked irregular release world model v0.

What this does:
    Train a small curve-first world model from partial release observations:

        observed tokens {(t_i, Q_i)} -> latent kinetic state -> future Q(t)

    The model is trained by randomly masking prefixes / irregular subsets of
    train-fold curves and reconstructing the unobserved grid points. At test
    time it sees Q at the requested early times and rolls out the future grid.

    This v0 is deliberately narrow:
        - PLGA cross321 and cleaned internal181 only.
        - Curve observations only; no formulation descriptors.
        - Monotone decoder on a fixed grid via positive increments.

    The goal is not to be final. It asks whether a learned masked-curve state
    can beat simple curve-only baselines on held-out future prediction.

Outputs:
    outputs/64_masked_release_world_model/per_curve.csv
    outputs/64_masked_release_world_model/scheme_method_summary.csv
    outputs/64_masked_release_world_model/summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import PLGABiphasic  # noqa: E402

DATASETS = ("cross321", "internal181")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _split_schemes(
    n: int,
    drug_groups: np.ndarray,
    polymer_groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    out: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out.append(("random_5fold", list(splitter.split(np.arange(n)))))
    for name, groups in [("group_by_drug", drug_groups), ("group_by_polymer", polymer_groups)]:
        n_splits = min(n_folds, int(pd.Series(groups).nunique()))
        if n_splits < 2:
            continue
        splitter = GroupKFold(n_splits=n_splits)
        out.append((name, list(splitter.split(np.arange(n), groups=groups))))
    return out


def _load_dataset(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, object], np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "cross321":
        fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = (
            mod46._load_cross321(args, mod41)
        )
    else:
        fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = (
            mod46._load_internal181(args, mod42)
        )
    return fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times


def _grid_targets(
    fids: np.ndarray,
    curve_map: dict[int, object],
    grid_times: np.ndarray,
    interp_fn: object,
) -> np.ndarray:
    return np.stack([
        interp_fn(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, grid_times)
        for fid in fids
    ]).astype(np.float32)


def _early_indices(grid_times: np.ndarray, early_times: np.ndarray) -> np.ndarray:
    idx: list[int] = []
    for t in early_times:
        hit = np.where(np.isclose(grid_times, t))[0]
        if len(hit) != 1:
            raise ValueError(f"early time {t} must appear exactly once in grid_times")
        idx.append(int(hit[0]))
    return np.asarray(idx, dtype=int)


def _reconstruct_at_obs(
    pred_grid_q: np.ndarray,
    grid_times: np.ndarray,
    early_q: np.ndarray,
    early_times: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    late_mask = grid_times > float(np.max(early_times))
    t = np.concatenate([early_times, grid_times[late_mask]])
    q = np.concatenate([early_q, pred_grid_q[late_mask]])
    order = np.argsort(t)
    q_hat = np.interp(t_obs, t[order], q[order], left=q[order][0], right=q[order][-1])
    return np.clip(q_hat, 0.0, 1.0)


def _future_metrics(
    t_obs: np.ndarray,
    q_obs: np.ndarray,
    q_hat: np.ndarray,
    early_max: float,
    min_future_obs: int,
) -> tuple[float, float]:
    mask = t_obs > early_max
    if int(np.sum(mask)) < min_future_obs:
        return float("nan"), float("nan")
    return _r2(q_obs[mask], q_hat[mask]), _rmse(q_obs[mask], q_hat[mask])


class MaskedReleaseWorldModel(nn.Module):
    def __init__(self, n_grid: int, context_dim: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.token = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        ) if context_dim else None
        self.state = nn.Sequential(
            nn.Linear(hidden * (2 if context_dim else 1), hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent),
            nn.SiLU(),
        )
        self.qmax = nn.Linear(latent, 1)
        self.increment = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_grid),
        )

    def forward(
        self,
        times: torch.Tensor,
        q_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.stack([times.expand_as(q_obs), q_obs, obs_mask], dim=-1)
        h = self.token(x)
        weighted = h * obs_mask.unsqueeze(-1)
        denom = obs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = weighted.sum(dim=1) / denom
        if self.context is not None:
            if context is None:
                raise ValueError("context tensor is required for context-conditioned model")
            pooled = torch.cat([pooled, self.context(context)], dim=1)
        z = self.state(pooled)
        qmax = torch.sigmoid(self.qmax(z))
        inc = F.softplus(self.increment(z)) + 1e-5
        cum = torch.cumsum(inc, dim=1)
        q = qmax * cum / cum[:, -1:].clamp_min(1e-6)
        return q.clamp(0.0, 1.0)


@dataclass
class TrainResult:
    model: MaskedReleaseWorldModel
    best_val_loss: float
    epochs_ran: int


def _make_random_masks(
    y: torch.Tensor,
    min_obs: int,
    prefix_prob: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_grid = y.shape
    obs = torch.zeros_like(y)
    target = torch.zeros_like(y)
    for i in range(batch):
        if rng.random() < prefix_prob:
            cutoff = int(rng.integers(min_obs, n_grid))
            candidates = np.arange(cutoff)
        else:
            cutoff = int(rng.integers(min_obs, n_grid + 1))
            candidates = np.arange(cutoff)
        n_obs = int(rng.integers(min_obs, max(min_obs + 1, len(candidates) + 1)))
        chosen = rng.choice(candidates, size=min(n_obs, len(candidates)), replace=False)
        obs[i, chosen] = 1.0
        target[i, :] = 1.0
        target[i, chosen] = 0.0
        if target[i].sum() < 1:
            target[i, -1] = 1.0
    q_obs = y * obs
    return q_obs, obs, target


def _synthetic_grid_curves(
    grid_times: np.ndarray,
    n: int,
    noise_sd: float,
    seed: int,
    device: str,
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, len(grid_times)), dtype=np.float32)
    _seed_everything(seed)
    sim = PLGABiphasic()
    t_grid = torch.tensor(grid_times, dtype=torch.float32, device=device)
    chunks: list[np.ndarray] = []
    done = 0
    with torch.no_grad():
        while done < n:
            batch = min(1024, n - done)
            theta = sim.sample_prior(batch).to(device)
            q = sim.simulate(theta, t_grid).detach().cpu().numpy().astype(np.float32)
            chunks.append(q)
            done += batch
    y = np.concatenate(chunks, axis=0)
    if noise_sd > 0:
        rng = np.random.default_rng(seed + 991)
        y = y + rng.normal(0.0, noise_sd, size=y.shape).astype(np.float32)
        y = np.maximum.accumulate(np.clip(y, 0.0, 1.0), axis=1)
    return y.astype(np.float32)


def _run_masked_epochs(
    model: MaskedReleaseWorldModel,
    train: torch.Tensor,
    context: torch.Tensor | None,
    times: torch.Tensor,
    weights: torch.Tensor,
    args: argparse.Namespace,
    rng: np.random.Generator,
    epochs: int,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = train[idx]
            q_obs, obs_mask, target_mask = _make_random_masks(
                batch,
                min_obs=args.min_mask_obs,
                prefix_prob=args.prefix_prob,
                rng=rng,
            )
            context_b = None if context is None else context[idx]
            pred = model(times, q_obs, obs_mask, context_b).clamp(0.0, 1.0)
            target_w = target_mask * weights
            anchor_w = obs_mask * weights
            target_loss = (((pred - batch) ** 2) * target_w).sum() / target_w.sum().clamp_min(1.0)
            anchor_loss = (((pred - batch) ** 2) * anchor_w).sum() / anchor_w.sum().clamp_min(1.0)
            loss = target_loss + args.anchor_weight * anchor_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if args.verbose and (epoch == 1 or epoch % 50 == 0):
            print(f"[64]       pretrain_epoch={epoch:04d} train={np.mean(losses):.5f}", flush=True)


def _train_world_model(
    y_train: np.ndarray,
    y_val: np.ndarray,
    x_train: np.ndarray | None,
    x_val: np.ndarray | None,
    grid_times: np.ndarray,
    y_pretrain: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
) -> TrainResult:
    device = torch.device(args.device)
    model = MaskedReleaseWorldModel(
        n_grid=y_train.shape[1],
        context_dim=0 if x_train is None else x_train.shape[1],
        hidden=args.hidden_dim,
        latent=args.latent_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    times = torch.tensor(
        grid_times / max(float(np.max(grid_times)), 1.0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    train = torch.tensor(y_train, dtype=torch.float32, device=device)
    val = torch.tensor(y_val, dtype=torch.float32, device=device)
    context_train = None if x_train is None else torch.tensor(x_train, dtype=torch.float32, device=device)
    context_val = None if x_val is None else torch.tensor(x_val, dtype=torch.float32, device=device)
    weights = torch.ones(y_train.shape[1], dtype=torch.float32, device=device).unsqueeze(0)
    future = torch.tensor(
        grid_times > float(np.max(args.early_times)),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    weights = torch.where(future, weights * args.future_weight, weights)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_left = args.patience

    if y_pretrain is not None and len(y_pretrain) and args.synthetic_pretrain_epochs > 0:
        pretrain = torch.tensor(y_pretrain, dtype=torch.float32, device=device)
        pretrain_context = None
        if context_train is not None:
            pretrain_context = torch.zeros((len(pretrain), context_train.shape[1]), dtype=torch.float32, device=device)
        _run_masked_epochs(
            model=model,
            train=pretrain,
            context=pretrain_context,
            times=times,
            weights=weights,
            args=args,
            rng=np.random.default_rng(seed + 779),
            epochs=args.synthetic_pretrain_epochs,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = train[idx]
            context_b = None if context_train is None else context_train[idx]
            q_obs, obs_mask, target_mask = _make_random_masks(
                batch,
                min_obs=args.min_mask_obs,
                prefix_prob=args.prefix_prob,
                rng=rng,
            )
            pred = model(times, q_obs, obs_mask, context_b)
            pred = pred.clamp_min(0.0).clamp_max(1.0)
            target_w = target_mask * weights
            anchor_w = obs_mask * weights
            target_loss = (((pred - batch) ** 2) * target_w).sum() / target_w.sum().clamp_min(1.0)
            anchor_loss = (((pred - batch) ** 2) * anchor_w).sum() / anchor_w.sum().clamp_min(1.0)
            loss = target_loss + args.anchor_weight * anchor_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_q_obs, val_obs, val_target = _make_random_masks(
                val,
                min_obs=args.min_mask_obs,
                prefix_prob=1.0,
                rng=rng,
            )
            val_pred = model(times, val_q_obs, val_obs, context_val)
            val_w = val_target * weights
            val_loss = float(((((val_pred - val) ** 2) * val_w).sum() / val_w.sum().clamp_min(1.0)).cpu())

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if args.verbose and (epoch == 1 or epoch % 50 == 0):
            print(
                f"[64]     epoch={epoch:04d} train={np.mean(losses):.5f} val={val_loss:.5f}",
                flush=True,
            )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_val_loss=best_val, epochs_ran=epoch)


def _predict_world_model(
    model: MaskedReleaseWorldModel,
    y_grid: np.ndarray,
    x_context: np.ndarray | None,
    grid_times: np.ndarray,
    early_idx: np.ndarray,
    device: str,
) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    times = torch.tensor(
        grid_times / max(float(np.max(grid_times)), 1.0),
        dtype=torch.float32,
        device=dev,
    ).unsqueeze(0)
    y = torch.tensor(y_grid, dtype=torch.float32, device=dev)
    obs = torch.zeros_like(y)
    obs[:, early_idx] = 1.0
    q_obs = y * obs
    context = None if x_context is None else torch.tensor(x_context, dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred = model(times, q_obs, obs, context).detach().cpu().numpy()
    return np.clip(pred, 0.0, 1.0)


def _eval_grid_predictions(
    dataset: str,
    scheme: str,
    fold: int,
    fids: np.ndarray,
    te: np.ndarray,
    curve_map: dict[int, object],
    early_q: np.ndarray,
    early_times: np.ndarray,
    grid_times: np.ndarray,
    pred_by_method: dict[str, np.ndarray],
    min_future_obs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    early_max = float(np.max(early_times))
    for local_i, j in enumerate(te):
        fid = int(fids[j])
        curve = curve_map[fid]
        for method, pred_grid in pred_by_method.items():
            q_hat = _reconstruct_at_obs(
                pred_grid_q=pred_grid[local_i],
                grid_times=grid_times,
                early_q=early_q[j].astype(float),
                early_times=early_times,
                t_obs=curve.t_obs,
            )
            future_r2, future_rmse = _future_metrics(
                curve.t_obs,
                curve.q_obs,
                q_hat,
                early_max=early_max,
                min_future_obs=min_future_obs,
            )
            rows.append({
                "dataset": dataset,
                "scheme": scheme,
                "fold": fold,
                "fid": fid,
                "method": method,
                "n_obs": int(len(curve.t_obs)),
                "n_future_obs": int(np.sum(curve.t_obs > early_max)),
                "full_r2": _r2(curve.q_obs, q_hat),
                "future_r2": future_r2,
                "future_rmse": future_rmse,
            })
    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, scheme, method), sub in per_curve.groupby(["dataset", "scheme", "method"], sort=False):
        full = sub["full_r2"].dropna().to_numpy(dtype=float)
        future = sub["future_r2"].dropna().to_numpy(dtype=float)
        frmse = sub["future_rmse"].dropna().to_numpy(dtype=float)
        rows.append({
            "dataset": dataset,
            "scheme": scheme,
            "method": method,
            "n": int(len(full)),
            "median_full_r2": float(np.median(full)),
            "p10_full_r2": float(np.percentile(full, 10)),
            "frac_full_r2_ge_0": float(np.mean(full >= 0.0)),
            "median_future_r2": float(np.median(future)) if len(future) else np.nan,
            "median_future_rmse": float(np.median(frmse)) if len(frmse) else np.nan,
            "frac_future_r2_ge_0": float(np.mean(future >= 0.0)) if len(future) else np.nan,
        })
    return pd.DataFrame(rows)


def _run_dataset(
    args: argparse.Namespace,
    dataset: str,
    mod41: ModuleType,
    mod42: ModuleType,
    mod46: ModuleType,
) -> list[dict[str, object]]:
    fids, x_form, early_q, curve_map, drug_groups, polymer_groups, early_times = _load_dataset(
        args, dataset, mod41, mod42, mod46
    )
    grid_times = np.unique(np.concatenate([
        np.asarray(args.early_times, dtype=float),
        np.asarray(args.grid_times, dtype=float),
    ]))
    grid_times = grid_times[grid_times <= args.t_grid_max_days]
    early_idx = _early_indices(grid_times, early_times)
    late_mask = grid_times > float(np.max(early_times))
    y_grid = _grid_targets(fids, curve_map, grid_times, mod41._interp_at)
    y_synth = _synthetic_grid_curves(
        grid_times=grid_times,
        n=args.synthetic_pretrain_n,
        noise_sd=args.synthetic_noise_sd,
        seed=args.seed + 64_000 + len(dataset),
        device=args.device,
    )

    print(
        f"[64] dataset={dataset} n={len(fids)} grid={len(grid_times)} "
        f"early={list(early_times)} future_grid={int(late_mask.sum())}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for scheme, splits in _split_schemes(len(fids), drug_groups, polymer_groups, args.n_folds, args.seed):
        print(f"[64]   scheme={scheme}", flush=True)
        for fold, (tr, te) in enumerate(splits):
            rng = np.random.default_rng(args.seed + 10_000 * fold + len(dataset))
            tr_perm = rng.permutation(tr)
            n_val = max(1, int(round(args.val_frac * len(tr_perm))))
            val = tr_perm[:n_val]
            tr_fit = tr_perm[n_val:]
            if len(tr_fit) < 4:
                tr_fit = tr
                val = tr[: min(len(tr), 4)]
            x_train: np.ndarray | None = None
            x_val: np.ndarray | None = None
            x_te: np.ndarray | None = None
            if args.context_mode == "formulation_plus_curve":
                x_mu = x_form[tr_fit].mean(axis=0, keepdims=True)
                x_sd = x_form[tr_fit].std(axis=0, keepdims=True)
                x_sd = np.where(x_sd < 1e-6, 1.0, x_sd)
                x_train = ((x_form[tr_fit] - x_mu) / x_sd).astype(np.float32)
                x_val = ((x_form[val] - x_mu) / x_sd).astype(np.float32)
                x_te = ((x_form[te] - x_mu) / x_sd).astype(np.float32)

            result = _train_world_model(
                y_train=y_grid[tr_fit],
                y_val=y_grid[val],
                x_train=x_train,
                x_val=x_val,
                grid_times=grid_times,
                y_pretrain=y_synth,
                args=args,
                seed=args.seed + 1000 * fold + len(dataset),
            )
            world_pred = _predict_world_model(
                result.model,
                y_grid[te],
                x_context=x_te,
                grid_times=grid_times,
                early_idx=early_idx,
                device=args.device,
            )

            train_median = np.median(y_grid[tr], axis=0)
            median_pred = np.tile(train_median, (len(te), 1))

            knn = KNeighborsRegressor(n_neighbors=min(args.n_neighbors, len(tr)), weights="distance")
            knn.fit(early_q[tr], y_grid[tr][:, late_mask])
            knn_late = np.clip(knn.predict(early_q[te]), 0.0, 1.0)
            knn_pred = np.tile(train_median, (len(te), 1))
            knn_pred[:, late_mask] = knn_late

            et = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                random_state=args.seed + 2000 + fold,
                n_jobs=-1,
            )
            et.fit(early_q[tr], y_grid[tr][:, late_mask])
            et_late = np.clip(et.predict(early_q[te]), 0.0, 1.0)
            et_pred = np.tile(train_median, (len(te), 1))
            et_pred[:, late_mask] = et_late

            pred_by_method = {
                "train_median_curve": median_pred,
                "KNN_earlyQ_to_lateQ": knn_pred,
                "ET_earlyQ_to_lateQ": et_pred,
                f"masked_world_model_{args.context_mode}": world_pred,
            }
            fold_rows = _eval_grid_predictions(
                dataset=dataset,
                scheme=scheme,
                fold=fold,
                fids=fids,
                te=te,
                curve_map=curve_map,
                early_q=early_q,
                early_times=early_times,
                grid_times=grid_times,
                pred_by_method=pred_by_method,
                min_future_obs=args.min_future_obs,
            )
            for row in fold_rows:
                row["world_best_val_loss"] = result.best_val_loss
                row["world_epochs_ran"] = result.epochs_ran
            rows.extend(fold_rows)
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
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--min-future-obs", type=int, default=3)
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument(
        "--grid-times",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 45.0, 60.0, 90.0],
    )
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument(
        "--context-mode",
        choices=("curve_only", "formulation_plus_curve"),
        default="curve_only",
    )
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=0.2)
    ap.add_argument("--future-weight", type=float, default=1.0)
    ap.add_argument("--min-mask-obs", type=int, default=1)
    ap.add_argument("--prefix-prob", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--synthetic-pretrain-n", type=int, default=0)
    ap.add_argument("--synthetic-pretrain-epochs", type=int, default=0)
    ap.add_argument("--synthetic-noise-sd", type=float, default=0.01)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/64_masked_release_world_model"))
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    _seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    mod41 = _load_script(scripts_dir / "41_theta_target_scaling_audit.py", "script41_theta_for64")
    mod42 = _load_script(scripts_dir / "42_internal181_tree_theta_audit.py", "script42_internal_for64")
    mod46 = _load_script(scripts_dir / "46_direct_curve_rf_input_ablation.py", "script46_direct_for64")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        rows.extend(_run_dataset(args, dataset, mod41, mod42, mod46))

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)

    best = summary.loc[summary.groupby(["dataset", "scheme"])["median_future_rmse"].idxmin()].copy()
    lines = [
        "=== 64 -- masked irregular release world model v0 ===",
        "",
        f"datasets     : {', '.join(datasets)}",
        f"early_times  : {args.early_times}",
        f"grid_times   : {args.grid_times}",
        f"device       : {args.device}",
        f"context_mode : {args.context_mode}",
        f"epochs       : {args.epochs}",
        f"future_weight: {args.future_weight}",
        f"synth_pretrain: n={args.synthetic_pretrain_n}, epochs={args.synthetic_pretrain_epochs}",
        "",
        "--- best method by future RMSE per dataset/scheme ---",
    ]
    for _, row in best.sort_values(["dataset", "scheme"]).iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} {row['method']:<22} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"future_RMSE={row['median_future_rmse']:.4f}"
        )
    lines.extend(["", "--- masked_world_model rows ---"])
    wm = summary[summary["method"].str.startswith("masked_world_model")].sort_values(["dataset", "scheme", "method"])
    for _, row in wm.iterrows():
        lines.append(
            f"  {row['dataset']:<11} {row['scheme']:<16} "
            f"full_R2={row['median_full_r2']:+.4f} "
            f"future_R2={row['median_future_r2']:+.4f} "
            f"future_RMSE={row['median_future_rmse']:.4f} "
            f"frac_future_R2>=0={row['frac_future_r2_ge_0']:.2f}"
        )
    lines.extend([
        "",
        "--- interpretation guardrail ---",
        f"  This is a masked-observer world model with context_mode={args.context_mode}.",
        "  Synthetic simulator pretraining is optional and should be treated as a domain-shift diagnostic.",
        "  It is not yet uncertainty-aware or mechanistically posterior-conditioned.",
    ])
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
