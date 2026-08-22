"""27e - Internal canonical benchmark for the 27c partial-curve NPE posterior.

Purpose:
    Provide the first same-mouth internal comparator scaffold for B4.
    Unlike `27c_sbi_npe_partial.py`, which evaluates external 321 suffix
    prediction, this script evaluates the trained 27c posterior on the fixed
    internal canonical split used by `72_canonical_benchmark_v2.py`.

Comparator mouth:
    - test fold = `data/canonical_split_v1.csv`
    - conditioning input = 27c's own 64-point masked-prefix representation
      truncated at 7 days
    - prediction target = canonical future trajectory for times > 14 days on
      the same 14-point judge grid used by `72_canonical_benchmark_v2.py`
    - point prediction = simulator rollout from posterior-median curve

Outputs:
    outputs/27e_internal_canonical_npe_fixed_eval/summary.csv
    outputs/27e_internal_canonical_npe_fixed_eval/per_curve_metrics.csv
    outputs/27e_internal_canonical_npe_fixed_eval/pairwise_vs_72.csv
    outputs/27e_internal_canonical_npe_fixed_eval/summary.txt

Notes:
    - `27d_residual_npe.py` does not serialize all residual hyperparameters in
      its checkpoint, so residual evaluation must reconstruct them from the
      recorded script / output defaults (`n_modes`, `sigma_c`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posterior import interpolate_to_grid
from simulator import PLGABiphasic, PLGABiphasicResidual

CANONICAL_SPLIT_PATH = Path("data/canonical_split_v1.csv")
DEFAULT_POSTERIOR_PT = Path("outputs/27c_sbi_npe_partial/posterior.pt")
DEFAULT_OUT = Path("outputs/27e_internal_canonical_npe_fixed_eval")
DEFAULT_BENCH72 = Path("outputs/72_canonical_benchmark_v2")
OBS_TIMES = (1.0, 3.0, 5.0, 7.0)
FUTURE_START = 14.0


class _JointPrior(torch.distributions.Distribution):
    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        base_dist: torch.distributions.Distribution,
        resid_dist: torch.distributions.Distribution,
    ):
        self._base = base_dist
        self._resid = resid_dist
        self._d_base = base_dist.event_shape.numel()
        self._d_resid = resid_dist.event_shape.numel()
        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=torch.Size([self._d_base + self._d_resid]),
            validate_args=False,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.cat([self._base.sample(sample_shape), self._resid.sample(sample_shape)], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._base.log_prob(value[..., :self._d_base]) + self._resid.log_prob(value[..., self._d_base:])

    @property
    def support(self) -> torch.distributions.constraints.Constraint:
        return torch.distributions.constraints.real_vector


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_curve_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def bootstrap_paired_ci(rmse_a: np.ndarray, rmse_b: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(rmse_a)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        mse_a = (rmse_a[idx] ** 2).mean()
        mse_b = (rmse_b[idx] ** 2).mean()
        deltas.append(np.sqrt(mse_a) - np.sqrt(mse_b))
    deltas = np.asarray(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def load_canonical_split() -> pd.DataFrame:
    split_df = pd.read_csv(CANONICAL_SPLIT_PATH)
    required = {"curve_id", "split_name", "fold_index"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Canonical split missing columns: {sorted(missing)}")
    split_df = split_df.copy()
    split_df["curve_id"] = split_df["curve_id"].astype(int)
    split_df["split_name"] = split_df["split_name"].astype(str)
    split_df["fold_index"] = split_df["fold_index"].astype(int)
    return split_df.sort_values("fold_index").reset_index(drop=True)


def load_internal_curve_matrix(split_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int], pd.DataFrame]:
    formulations = pd.read_csv("data/formulations.csv")
    curves = pd.read_csv("data/curves_long.csv")
    theta_df = pd.read_csv("data/theta_bank.csv")
    curves["release"] = curves["release"].astype(float)
    if curves["release"].max() > 2.0:
        curves["release"] = curves["release"] / 100.0
    curves["release"] = curves["release"].clip(0.0, 1.2)

    time_grid = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 42.0, 56.0, 84.0])
    common_ids = sorted(set(formulations["curve_id"]) & set(curves["curve_id"]) & set(theta_df["curve_id"]))
    split_ids = split_df["curve_id"].tolist()
    missing = sorted(set(split_ids) - set(common_ids))
    if missing:
        raise ValueError(f"Canonical split contains curve_ids missing from internal tables: {missing[:10]}")

    curve_matrix = np.zeros((len(split_ids), len(time_grid)))
    for i, cid in enumerate(split_ids):
        cdf = curves[curves["curve_id"] == cid].sort_values("time")
        t, y = cdf["time"].to_numpy(), cdf["release"].to_numpy()
        curve_matrix[i, :] = np.interp(time_grid, t, y, left=y[0], right=y[-1])
    return curve_matrix, time_grid, split_ids, curves


def build_partial_npe_posterior(
    posterior_pt: Path,
    config_path: Path,
    device: str,
    model_kind: str,
    residual_n_modes: int,
    residual_sigma_c: float,
):
    ckpt = torch.load(posterior_pt, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    t_cfg = cfg["synthetic"]["t_grid_days"]
    t_grid = torch.linspace(t_cfg["start"], t_cfg["end"], t_cfg["n"])
    t_len = int(ckpt["config"]["T"])

    if t_len != len(t_grid):
        raise ValueError(
            f"Checkpoint T={t_len} but config t_grid length is {len(t_grid)}. "
            "This loader assumes the current plga_phase1 grid."
        )

    if model_kind == "27c":
        sim = PLGABiphasic(
            rtol=cfg["simulator"]["rtol"],
            atol=cfg["simulator"]["atol"],
            method=cfg["simulator"]["method"],
        )
        base = sim.prior().base_dist
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(base.low.to(device), base.high.to(device)),
            1,
        )
    elif model_kind == "27d":
        sim = PLGABiphasicResidual(
            n_modes=residual_n_modes,
            sigma_c=residual_sigma_c,
            t_max=float(t_cfg["end"]),
            rtol=cfg["simulator"]["rtol"],
            atol=cfg["simulator"]["atol"],
            method=cfg["simulator"]["method"],
        )
        base_uniform = PLGABiphasic().prior().base_dist
        base_prior = torch.distributions.Independent(
            torch.distributions.Uniform(base_uniform.low.to(device), base_uniform.high.to(device)),
            1,
        )
        resid_prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(residual_n_modes, device=device),
                residual_sigma_c * torch.ones(residual_n_modes, device=device),
            ),
            1,
        )
        prior = _JointPrior(base_prior, resid_prior)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    embedding = FCEmbedding(
        input_dim=2 * t_len,
        output_dim=cfg["posterior"].get("embedding_output_dim", 16),
        num_layers=cfg["posterior"].get("embedding_layers", 3),
        num_hiddens=cfg["posterior"].get("embedding_hidden", 64),
    )
    density_estimator_builder = posterior_nn(
        model=cfg["posterior"]["flow_type"],
        hidden_features=cfg["posterior"]["hidden_features"],
        num_transforms=cfg["posterior"]["num_transforms"],
        embedding_net=embedding,
    )
    inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)

    dummy_theta = sim.sample_prior(2).to(device)
    dummy_x = torch.zeros((2, 2 * t_len), dtype=torch.float32, device=device)
    density_estimator = density_estimator_builder(dummy_theta, dummy_x)
    density_estimator.load_state_dict(ckpt["density_estimator_state"])
    posterior_obj = inference.build_posterior(density_estimator)
    return sim, posterior_obj, t_grid


def build_obs_vector(curves_df: pd.DataFrame, curve_id: int, cond_grid: torch.Tensor) -> torch.Tensor:
    cdf = curves_df[curves_df["curve_id"] == curve_id].sort_values("time")
    t_np = cdf["time"].to_numpy(dtype=float)
    q_np = cdf["release"].to_numpy(dtype=float)
    prefix_mask = t_np <= max(OBS_TIMES)
    if prefix_mask.sum() < 2:
        raise ValueError(f"curve_id={curve_id} has fewer than 2 observed points in the fixed 7d prefix")
    q_prefix_grid, _, _ = interpolate_to_grid(
        torch.tensor(t_np[prefix_mask], dtype=torch.float32),
        torch.tensor(q_np[prefix_mask], dtype=torch.float32),
        cond_grid.cpu(),
        t_max_days=float(cond_grid.max().item()),
    )
    obs_mask = (cond_grid <= max(OBS_TIMES)).float()
    return torch.cat([q_prefix_grid * obs_mask, obs_mask], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior-pt", type=Path, default=DEFAULT_POSTERIOR_PT)
    ap.add_argument("--config", type=Path, default=Path("configs/plga_phase1.yaml"))
    ap.add_argument("--bench72-dir", type=Path, default=DEFAULT_BENCH72)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-kind", choices=("27c", "27d"), default="27c")
    ap.add_argument("--residual-n-modes", type=int, default=5)
    ap.add_argument("--residual-sigma-c", type=float, default=0.015)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-posterior-samples", type=int, default=1000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    split_df = load_canonical_split()
    curve_matrix, time_grid, curve_ids, curves_df = load_internal_curve_matrix(split_df)
    split_names = split_df["split_name"].to_numpy()
    test_idx = np.flatnonzero(split_names == "test")
    test_curve_ids = [int(curve_ids[i]) for i in test_idx]
    curves_test = curve_matrix[test_idx]
    future_mask = time_grid > FUTURE_START
    y_true = curves_test[:, future_mask]

    print(f"[27e] split source   : {CANONICAL_SPLIT_PATH}")
    print(f"[27e] posterior_pt   : {args.posterior_pt}")
    print(f"[27e] model kind     : {args.model_kind}")
    print(f"[27e] n_test curves  : {len(test_idx)}")
    print(f"[27e] obs times      : {list(OBS_TIMES)}")

    sim, posterior_obj, cond_grid = build_partial_npe_posterior(
        args.posterior_pt,
        args.config,
        args.device,
        args.model_kind,
        args.residual_n_modes,
        args.residual_sigma_c,
    )

    preds = []
    rows = []
    eval_grid_t = torch.tensor(time_grid, dtype=torch.float32, device=args.device)
    for curve_id, curve_row in zip(test_curve_ids, curves_test):
        x_obs = build_obs_vector(curves_df, curve_id, cond_grid).to(args.device)
        theta_samples = posterior_obj.sample(
            (args.n_posterior_samples,),
            x=x_obs,
            show_progress_bars=False,
        ).cpu()
        with torch.no_grad():
            q_samples = sim.simulate(theta_samples.to(args.device), eval_grid_t).cpu().numpy()
        q_median = np.median(q_samples, axis=0)
        y_pred_future = q_median[future_mask]
        preds.append(y_pred_future)
        rows.append(
            {
                "curve_id": int(curve_id),
                "rmse": float(np.sqrt(np.mean((curve_row[future_mask] - y_pred_future) ** 2))),
                "obs_times_used": ",".join(f"{t:g}" for t in OBS_TIMES),
                "posterior_mean_sd": float(theta_samples.std(dim=0).mean()),
            }
        )

    y_pred = np.asarray(preds)
    per_curve = pd.DataFrame(rows)
    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "method": "NPE-fixed-27c" if args.model_kind == "27c" else "NPE-fixed-27d",
                "n_test": int(len(test_idx)),
                "n_obs": len(OBS_TIMES),
                "rmse": pooled_rmse(y_true, y_pred),
                "r2_pooled": pooled_r2(y_true, y_pred),
                "per_curve_rmse_mean": float(per_curve["rmse"].mean()),
                "per_curve_rmse_median": float(per_curve["rmse"].median()),
            }
        ]
    )
    summary.to_csv(args.out / "summary.csv", index=False)

    pairwise_rows = []
    bench72_per_curve = args.bench72_dir / "per_curve_results.csv"
    if bench72_per_curve.exists():
        bench_df = pd.read_csv(bench72_per_curve)
        direct_fixed = bench_df[bench_df["method"] == "DirectQ-fixed"][["curve_id", "rmse"]].rename(columns={"rmse": "rmse_directq_fixed"})
        merged = per_curve.merge(direct_fixed, on="curve_id", how="inner")
        if len(merged) == len(per_curve):
            d_mean, d_lo, d_hi = bootstrap_paired_ci(
                merged["rmse_directq_fixed"].to_numpy(dtype=float),
                merged["rmse"].to_numpy(dtype=float),
                n_resamples=args.n_bootstrap,
                seed=args.seed,
            )
            pairwise_rows.append(
                {
                    "method_a": "DirectQ-fixed",
                    "method_b": "NPE-fixed-27c" if args.model_kind == "27c" else "NPE-fixed-27d",
                    "delta_rmse": d_mean,
                    "ci_low": d_lo,
                    "ci_high": d_hi,
                    "significant_95": bool(d_lo > 0 or d_hi < 0),
                    "note": "Same-mouth fixed 4-point comparison against 72_v2 test curves",
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(args.out / "pairwise_vs_72.csv", index=False)

    summary_lines = [
        "=== 27e internal canonical NPE fixed benchmark ===",
        f"posterior_pt      : {args.posterior_pt}",
        f"split_source      : {CANONICAL_SPLIT_PATH}",
        f"n_test            : {len(test_idx)}",
        f"obs_times         : {list(OBS_TIMES)}",
        f"future_start      : {FUTURE_START}",
        f"posterior_samples : {args.n_posterior_samples}",
        "",
        summary.to_string(index=False),
    ]
    if len(pairwise_df):
        summary_lines.extend(["", "Pairwise vs 72_v2 DirectQ-fixed:", pairwise_df.to_string(index=False)])
    (args.out / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    meta = {
        "script": "scripts/27e_internal_canonical_npe_fixed_eval.py",
        "model_kind": args.model_kind,
        "posterior_pt": str(args.posterior_pt),
        "split_source": str(CANONICAL_SPLIT_PATH),
        "bench72_dir": str(args.bench72_dir),
        "seed": args.seed,
        "n_posterior_samples": args.n_posterior_samples,
        "residual_n_modes": args.residual_n_modes,
        "residual_sigma_c": args.residual_sigma_c,
        "obs_times": list(OBS_TIMES),
        "future_start": FUTURE_START,
    }
    with open(args.out / "lock_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[27e] wrote {args.out / 'summary.csv'}")
    print(f"[27e] wrote {args.out / 'per_curve_metrics.csv'}")
    print(f"[27e] wrote {args.out / 'pairwise_vs_72.csv'}")


if __name__ == "__main__":
    main()
