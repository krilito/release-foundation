"""76 - Proper split-conformal recalibration for FIB-CASP intervals.

The Phase-2 FIB-CASP benchmark (script 62) showed useful but under-nominal
interval coverage, especially for cross321 random splits. This script asks a
narrower question:

    If we keep the same RF -> theta -> FIB -> simulator pipeline, can a proper
    calibration split repair cov90 without exploding interval width?

For each outer split, the original training fold is split into fit/calibration
subsets. The model is fit only on the fit subset. Calibration curves choose a
single multiplicative interval scale (global) and optional time-local scales.
Those scales are then applied to the untouched outer test fold.

Important scope: cov90 here is empirical pointwise coverage over observed
timepoints, matching scripts 62/72. It is not a simultaneous whole-curve band.

Outputs:
  outputs/76_casp_conformal_recalibration/summary.csv
  outputs/76_casp_conformal_recalibration/{dataset}/{scheme}/per_curve.csv
  outputs/76_casp_conformal_recalibration/summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casp import SimulatorDecoder, fib_posterior, sample_theta_fib
from casp.calibration import per_curve_r2


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bench62 = _load_script(Path(__file__).resolve().parent / "62_fib_casp_benchmark.py", "_bench62_76")


@dataclass
class CurveInterval:
    fid: int
    fold: int
    r2_point: float
    t_norm: np.ndarray
    y: np.ndarray
    point: np.ndarray
    lo90: np.ndarray
    hi90: np.ndarray


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 1.0
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


def _score_curve(curve: CurveInterval, eps: float = 1e-6) -> np.ndarray:
    below = curve.y < curve.point
    denom_low = np.maximum(curve.point - curve.lo90, eps)
    denom_high = np.maximum(curve.hi90 - curve.point, eps)
    scores = np.where(
        below,
        (curve.point - curve.y) / denom_low,
        (curve.y - curve.point) / denom_high,
    )
    return np.maximum(scores, 0.0)


def _scaled_interval(curve: CurveInterval, scale: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    scale_arr = np.asarray(scale, dtype=float)
    lo = curve.point + scale_arr * (curve.lo90 - curve.point)
    hi = curve.point + scale_arr * (curve.hi90 - curve.point)
    return lo, hi


def _coverage_width(curve: CurveInterval, scale: np.ndarray | float) -> tuple[float, float]:
    lo, hi = _scaled_interval(curve, scale)
    cov = float(((curve.y >= lo) & (curve.y <= hi)).mean())
    width = float((hi - lo).mean())
    return cov, width


def _local_scales(cal_curves: list[CurveInterval], bins: np.ndarray, alpha: float) -> np.ndarray:
    all_t: list[np.ndarray] = []
    all_s: list[np.ndarray] = []
    for c in cal_curves:
        all_t.append(c.t_norm)
        all_s.append(_score_curve(c))
    t = np.concatenate(all_t) if all_t else np.array([], dtype=float)
    s = np.concatenate(all_s) if all_s else np.array([], dtype=float)
    global_q = _conformal_quantile(s, alpha)
    qs: list[float] = []
    for bi in range(len(bins) - 1):
        if bi == len(bins) - 2:
            mask = (t >= bins[bi]) & (t <= bins[bi + 1])
        else:
            mask = (t >= bins[bi]) & (t < bins[bi + 1])
        q = _conformal_quantile(s[mask], alpha) if mask.any() else global_q
        qs.append(max(q, global_q * 0.5))
    return np.asarray(qs, dtype=float)


def _scale_for_times(t_norm: np.ndarray, bins: np.ndarray, qs: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bins[1:-1], t_norm, side="right")
    idx = np.clip(idx, 0, len(qs) - 1)
    return qs[idx]


def _fit_cal_split(
    train_idx: np.ndarray,
    groups: np.ndarray,
    cal_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx)
    local_groups = groups[train_idx]
    n_unique = len(pd.unique(local_groups))
    if n_unique >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=cal_frac, random_state=seed)
        fit_local, cal_local = next(splitter.split(train_idx, groups=local_groups))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=cal_frac, random_state=seed)
        fit_local, cal_local = next(splitter.split(train_idx))
    return train_idx[fit_local], train_idx[cal_local]


def _evaluate_indices(
    bundle: object,
    rf: RandomForestRegressor,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    fold: int,
    n_samples: int,
    rank: int,
    alpha_fib: float,
    sigma0_frac: float,
    seed: int,
) -> list[CurveInterval]:
    n_form = bundle.n_form
    x_fit = bundle.X[fit_idx]
    x_eval = bundle.X[eval_idx]
    mu_f = x_fit[:, :n_form].mean(axis=0, keepdims=True)
    sd_f = x_fit[:, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
    x_eval_s = x_eval.copy()
    x_eval_s[:, :n_form] = (x_eval_s[:, :n_form] - mu_f) / sd_f
    theta_pred_raw = rf.predict(x_eval_s).astype(np.float32)

    sim_eval = bundle.simulator
    sim_train = bundle.simulator_train
    prior = sim_eval.prior().base_dist
    prior_low = prior.low.float()
    prior_high = prior.high.float()
    lo_np = prior_low.numpy()
    hi_np = prior_high.numpy()
    eps_box = 1e-4 * (hi_np - lo_np)
    theta_pred = np.clip(theta_pred_raw, lo_np + eps_box, hi_np - eps_box)
    decoder_eval = SimulatorDecoder(sim_eval)

    rows: list[CurveInterval] = []
    for j, idx in enumerate(eval_idx):
        c = bundle.curves[idx]
        t_obs = c.t_obs.astype(np.float64)
        q_obs = c.q_obs.astype(np.float64)
        mask = (t_obs > 0.0) & (t_obs <= bundle.t_max)
        if mask.sum() < 2:
            continue
        t_eval_np = t_obs[mask]
        y_eval = q_obs[mask]
        t_eval_t = torch.tensor(t_eval_np, dtype=torch.float32)
        theta_hat = torch.tensor(theta_pred[j], dtype=torch.float32)

        _seed_everything(seed + fold * 100_000 + int(j))
        try:
            with torch.no_grad():
                q_point = decoder_eval(theta_hat.unsqueeze(0), t_eval_t).cpu().numpy()[0]
            q_fib = fib_posterior(
                simulator=sim_train,
                theta_hat=theta_hat,
                t=t_eval_t,
                sigma_obs=bundle.sigma_obs,
                rank=rank,
                alpha=alpha_fib,
                sigma_0_frac=sigma0_frac,
                prior_low=prior_low,
                prior_high=prior_high,
            )
            theta_samples = sample_theta_fib(
                q_fib,
                n_samples=n_samples,
                prior_low=prior_low,
                prior_high=prior_high,
            )
            with torch.no_grad():
                q_samples = decoder_eval(theta_samples, t_eval_t).cpu().numpy()
        except Exception:
            continue

        sample_mean = q_samples.mean(axis=0)
        recenter = q_point - sample_mean
        lo90 = np.percentile(q_samples, 5.0, axis=0) + recenter
        hi90 = np.percentile(q_samples, 95.0, axis=0) + recenter
        r2 = float(per_curve_r2(y_eval[None, :], q_point[None, :])[0])
        rows.append(CurveInterval(
            fid=int(bundle.fids[idx]),
            fold=fold,
            r2_point=r2,
            t_norm=np.clip(t_eval_np / float(bundle.t_max), 0.0, 1.0),
            y=y_eval,
            point=q_point,
            lo90=lo90,
            hi90=hi90,
        ))
    return rows


def _fit_rf(bundle: object, fit_idx: np.ndarray, seed: int) -> RandomForestRegressor:
    n_form = bundle.n_form
    x_fit = bundle.X[fit_idx].copy()
    mu_f = x_fit[:, :n_form].mean(axis=0, keepdims=True)
    sd_f = x_fit[:, :n_form].std(axis=0, keepdims=True).clip(min=1e-6)
    x_fit[:, :n_form] = (x_fit[:, :n_form] - mu_f) / sd_f
    rf = RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed)
    rf.fit(x_fit, bundle.theta_oracle[fit_idx])
    return rf


def run_cell(
    bundle: object,
    scheme: str,
    out_dir: Path,
    n_samples: int,
    rank: int,
    alpha_fib: float,
    sigma0_frac: float,
    alpha_cov: float,
    cal_frac: float,
    seed: int,
    n_folds: int,
) -> dict[str, object]:
    n = len(bundle.X)
    splits = _bench62.make_splits(
        scheme,
        n,
        bundle.groups_drug,
        bundle.groups_polymer,
        n_folds=n_folds,
        seed=seed,
    )
    group_for_cal = bundle.groups_drug if scheme == "group_by_drug" else bundle.groups_polymer
    if scheme == "random_5fold":
        group_for_cal = np.arange(n)

    per_curve_rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        t0 = time.time()
        fit_idx, cal_idx = _fit_cal_split(train_idx, group_for_cal, cal_frac, seed + fold)
        rf = _fit_rf(bundle, fit_idx, seed + fold)
        cal_curves = _evaluate_indices(
            bundle, rf, fit_idx, cal_idx, fold, n_samples, rank, alpha_fib, sigma0_frac, seed,
        )
        test_curves = _evaluate_indices(
            bundle, rf, fit_idx, test_idx, fold, n_samples, rank, alpha_fib, sigma0_frac, seed + 999,
        )
        cal_scores = np.concatenate([_score_curve(c) for c in cal_curves]) if cal_curves else np.array([1.0])
        q_global = max(1.0, _conformal_quantile(cal_scores, alpha_cov))
        bins = np.array([0.0, 0.08, 0.20, 0.50, 1.000001], dtype=float)
        q_local = np.maximum(1.0, _local_scales(cal_curves, bins, alpha_cov))

        for c in test_curves:
            cov_raw, width_raw = _coverage_width(c, 1.0)
            cov_global, width_global = _coverage_width(c, q_global)
            local_scale = _scale_for_times(c.t_norm, bins, q_local)
            cov_local, width_local = _coverage_width(c, local_scale)
            per_curve_rows.append({
                "dataset": bundle.name,
                "scheme": scheme,
                "fold": fold,
                "fid": c.fid,
                "n_t_eval": int(len(c.y)),
                "r2_point": c.r2_point,
                "q_global": q_global,
                "q_local_0_008": q_local[0],
                "q_local_008_020": q_local[1],
                "q_local_020_050": q_local[2],
                "q_local_050_100": q_local[3],
                "cov90_raw": cov_raw,
                "cov90_global": cov_global,
                "cov90_local": cov_local,
                "width90_raw": width_raw,
                "width90_global": width_global,
                "width90_local": width_local,
            })
        print(
            f"[76] {bundle.name}/{scheme} fold {fold + 1}/{len(splits)} "
            f"fit={len(fit_idx)} cal={len(cal_idx)} test={len(test_curves)} "
            f"q={q_global:.2f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    cell_dir = out_dir / bundle.name / scheme
    cell_dir.mkdir(parents=True, exist_ok=True)
    per_curve = pd.DataFrame(per_curve_rows)
    per_curve.to_csv(cell_dir / "per_curve.csv", index=False)
    if per_curve.empty:
        return {"dataset": bundle.name, "scheme": scheme, "n": 0}

    return {
        "dataset": bundle.name,
        "scheme": scheme,
        "n": int(len(per_curve)),
        "r2_median": float(per_curve["r2_point"].median()),
        "cov90_raw_mean": float(per_curve["cov90_raw"].mean()),
        "cov90_global_mean": float(per_curve["cov90_global"].mean()),
        "cov90_local_mean": float(per_curve["cov90_local"].mean()),
        "width90_raw_median": float(per_curve["width90_raw"].median()),
        "width90_global_median": float(per_curve["width90_global"].median()),
        "width90_local_median": float(per_curve["width90_local"].median()),
        "global_width_inflation": float(
            per_curve["width90_global"].median() / max(per_curve["width90_raw"].median(), 1e-9)
        ),
        "local_width_inflation": float(
            per_curve["width90_local"].median() / max(per_curve["width90_raw"].median(), 1e-9)
        ),
        "q_global_median": float(per_curve["q_global"].median()),
    }


def _load_bundle(name: str) -> object:
    if name == "cross321":
        return _bench62.load_cross321()
    if name == "internal181":
        return _bench62.load_internal181()
    if name == "liposome":
        return _bench62.load_liposome()
    raise ValueError(name)


def _summarize_per_curve(per_curve: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": str(per_curve["dataset"].iloc[0]),
        "scheme": str(per_curve["scheme"].iloc[0]),
        "n": int(len(per_curve)),
        "r2_median": float(per_curve["r2_point"].median()),
        "cov90_raw_mean": float(per_curve["cov90_raw"].mean()),
        "cov90_global_mean": float(per_curve["cov90_global"].mean()),
        "cov90_local_mean": float(per_curve["cov90_local"].mean()),
        "width90_raw_median": float(per_curve["width90_raw"].median()),
        "width90_global_median": float(per_curve["width90_global"].median()),
        "width90_local_median": float(per_curve["width90_local"].median()),
        "global_width_inflation": float(
            per_curve["width90_global"].median() / max(per_curve["width90_raw"].median(), 1e-9)
        ),
        "local_width_inflation": float(
            per_curve["width90_local"].median() / max(per_curve["width90_raw"].median(), 1e-9)
        ),
        "q_global_median": float(per_curve["q_global"].median()),
    }


def summarize_existing(out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.glob("*/*/per_curve.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            rows.append(_summarize_per_curve(df))
    return pd.DataFrame(rows)


def write_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.sort_values(["dataset", "scheme"]).reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    lines = ["=== 76 -- FIB-CASP conformal recalibration ===", ""]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['dataset']:>11}/{r['scheme']:<16} n={int(r['n']):3d} "
            f"R2med={float(r['r2_median']):+.3f} "
            f"cov raw/global/local="
            f"{float(r['cov90_raw_mean']):.3f}/"
            f"{float(r['cov90_global_mean']):.3f}/"
            f"{float(r['cov90_local_mean']):.3f} "
            f"width inflation global/local="
            f"{float(r['global_width_inflation']):.2f}/"
            f"{float(r['local_width_inflation']):.2f}"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/76_casp_conformal_recalibration"))
    ap.add_argument("--datasets", nargs="+", default=["cross321"])
    ap.add_argument("--schemes", nargs="+", default=["random_5fold"])
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha-fib", type=float, default=1e-2)
    ap.add_argument("--sigma0-frac", type=float, default=0.05)
    ap.add_argument("--alpha-cov", type=float, default=0.10)
    ap.add_argument("--cal-frac", type=float, default=0.20)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-existing", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing:
        write_summary(summarize_existing(args.out), args.out)
        return

    summaries: list[dict[str, object]] = []
    for dataset in args.datasets:
        bundle = _load_bundle(dataset)
        for scheme in args.schemes:
            summaries.append(
                run_cell(
                    bundle=bundle,
                    scheme=scheme,
                    out_dir=args.out,
                    n_samples=args.n_samples,
                    rank=args.rank,
                    alpha_fib=args.alpha_fib,
                    sigma0_frac=args.sigma0_frac,
                    alpha_cov=args.alpha_cov,
                    cal_frac=args.cal_frac,
                    seed=args.seed,
                    n_folds=args.n_folds,
                )
            )

    write_summary(summarize_existing(args.out), args.out)


if __name__ == "__main__":
    main()
