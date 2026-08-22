"""38b - Canonical-split functional baselines for the paper benchmark.

Purpose:
    Close B6 / B7 on the same canonical mouth used by
    `72_canonical_benchmark_v2.py`, without mutating the older random-CV
    baseline history in `38_prediction_baselines.py`.

Baselines implemented here:
    B6 - fPCA-Ridge:
        A discrete functional baseline. We run PCA on the TRAIN future-curve
        matrix (times > 14d), then regress PCA scores from the same fixed
        inputs used by DirectQ-fixed: [x, Q(1), Q(3), Q(5), Q(7)].
        This is a pragmatic functional-regression baseline on the canonical
        mouth, not a spline-heavy FDA implementation.

    B7 - GlobalMean:
        Predict the TRAIN mean future trajectory for every TEST curve.
        This is the sanity floor requested in the locked requirements doc.

Outputs:
    outputs/38b_canonical_functional_baselines/summary.csv
    outputs/38b_canonical_functional_baselines/per_curve_results.csv
    outputs/38b_canonical_functional_baselines/pairwise_vs_72.csv
    outputs/38b_canonical_functional_baselines/summary.txt
    outputs/38b_canonical_functional_baselines/lock_metadata.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_72 = Path("scripts/72_canonical_benchmark_v2.py")
OUTPUT_DIR = Path("outputs/38b_canonical_functional_baselines")
OUTPUT_72_DIR = Path("outputs/72_canonical_benchmark_v2")
EARLY_TIMES = [1.0, 3.0, 5.0, 7.0]
FUTURE_START = 14.0


def load_benchmark72_module():
    spec = importlib.util.spec_from_file_location("benchmark72_v2", SCRIPT_72)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_72}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def run_global_mean(curves_train: np.ndarray, n_test: int, future_mask: np.ndarray) -> np.ndarray:
    mean_future = curves_train[:, future_mask].mean(axis=0, keepdims=True)
    return np.repeat(mean_future, repeats=n_test, axis=0)


def run_fpca_ridge(
    X_train: np.ndarray,
    curves_train: np.ndarray,
    X_test: np.ndarray,
    curves_test: np.ndarray,
    time_grid: np.ndarray,
    future_mask: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict[str, object]]:
    obs_idx = [int(np.argmin(np.abs(time_grid - t))) for t in EARLY_TIMES]
    X_aug_train = np.hstack([X_train, curves_train[:, obs_idx]])
    X_aug_test = np.hstack([X_test, curves_test[:, obs_idx]])
    y_train = curves_train[:, future_mask]

    max_components = min(n_components, y_train.shape[1], len(y_train) - 1)
    if max_components < 1:
        raise ValueError("Not enough training curves to fit PCA baseline")

    pca = PCA(n_components=max_components, svd_solver="full", random_state=0)
    score_train = pca.fit_transform(y_train)

    reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 17))),
        ]
    )
    reg.fit(X_aug_train, score_train)
    score_pred = reg.predict(X_aug_test)
    y_pred = pca.inverse_transform(score_pred)
    y_pred = np.clip(y_pred, 0.0, 1.1)
    meta = {
        "n_components": int(max_components),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "alphas": np.logspace(-4, 4, 17).tolist(),
        "chosen_alpha": float(reg.named_steps["ridge"].alpha_),
    }
    return y_pred, meta


def summarize_method(bench72, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_curve = bench72.per_curve_rmse(y_true, y_pred)
    return {
        "rmse": float(bench72.pooled_rmse(y_true, y_pred)),
        "r2_pooled": float(bench72.pooled_r2(y_true, y_pred)),
        "per_curve_rmse_mean": float(per_curve.mean()),
        "per_curve_rmse_median": float(np.median(per_curve)),
    }


def load_benchmark72_per_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected 72 canonical benchmark outputs at {path}. "
            "Run 72_v2 first before comparing B6/B7 against the main table."
        )
    df = pd.read_csv(path)
    required = {"curve_id", "method", "rmse"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"72 per_curve_results missing columns: {sorted(missing)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--fpca-components", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    bench72 = load_benchmark72_module()

    split_df = bench72.load_canonical_split()
    formulations, curves, time_grid, theta_matrix, common_ids = bench72.load_data(split_df=split_df)
    regime_labels = bench72.load_regime_labels(common_ids)

    split_names = split_df["split_name"].to_numpy()
    train_idx = np.flatnonzero(split_names == "train")
    cal_idx = np.flatnonzero(split_names == "cal")
    test_idx = np.flatnonzero(split_names == "test")

    X_train, X_cal, X_test, feature_cols, _ = bench72.prepare_features_split(
        formulations, train_idx, test_idx, cal_idx
    )
    curves_train, curves_test = curves[train_idx], curves[test_idx]
    regimes_test = regime_labels[test_idx]
    test_curve_ids = [int(common_ids[i]) for i in test_idx]

    future_mask = time_grid > FUTURE_START
    y_test = curves_test[:, future_mask]

    global_mean_pred = run_global_mean(curves_train, len(test_idx), future_mask)
    fpca_pred, fpca_meta = run_fpca_ridge(
        X_train,
        curves_train,
        X_test,
        curves_test,
        time_grid,
        future_mask,
        args.fpca_components,
    )

    methods = {
        "GlobalMean": global_mean_pred,
        "fPCA-Ridge": fpca_pred,
    }

    summary_rows: list[dict[str, object]] = []
    per_curve_rows: list[dict[str, object]] = []
    method_rmse: dict[str, np.ndarray] = {}
    obs_times_map = {
        "GlobalMean": "none",
        "fPCA-Ridge": ",".join(str(int(t)) for t in EARLY_TIMES),
    }

    for method, y_pred in methods.items():
        stats = summarize_method(bench72, y_test, y_pred)
        per_curve = bench72.per_curve_rmse(y_test, y_pred)
        method_rmse[method] = per_curve
        summary_rows.append(
            {
                "method": method,
                "n_test": int(len(test_idx)),
                "n_obs": 0 if method == "GlobalMean" else len(EARLY_TIMES),
                **stats,
            }
        )
        for curve_id, regime, rmse in zip(test_curve_ids, regimes_test, per_curve):
            per_curve_rows.append(
                {
                    "curve_id": int(curve_id),
                    "regime": float(regime) if np.isfinite(regime) else np.nan,
                    "method": method,
                    "rmse": float(rmse),
                    "obs_times_used": obs_times_map[method],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    per_curve_df = pd.DataFrame(per_curve_rows)
    summary_df.to_csv(args.out / "summary.csv", index=False)
    per_curve_df.to_csv(args.out / "per_curve_results.csv", index=False)

    df72 = load_benchmark72_per_curve(OUTPUT_72_DIR / "per_curve_results.csv")
    pairwise_rows: list[dict[str, object]] = []
    for baseline_name, baseline_rmse in method_rmse.items():
        for ref_method, ref_df in df72.groupby("method"):
            ref_ordered = (
                ref_df[["curve_id", "rmse"]]
                .assign(curve_id=lambda d: d["curve_id"].astype(int))
                .set_index("curve_id")
                .loc[test_curve_ids]
            )
            ref_rmse = ref_ordered["rmse"].to_numpy(dtype=float)
            delta_mean, ci_low, ci_high = bench72.bootstrap_paired_ci(
                baseline_rmse,
                ref_rmse,
                n_resamples=args.n_bootstrap,
                seed=args.seed,
            )
            pairwise_rows.append(
                {
                    "baseline_method": baseline_name,
                    "reference_method": ref_method,
                    "delta_rmse_baseline_minus_reference": float(delta_mean),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "baseline_better": bool(ci_high < 0.0),
                    "reference_better": bool(ci_low > 0.0),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(args.out / "pairwise_vs_72.csv", index=False)

    metadata = {
        "script": "scripts/38b_canonical_functional_baselines.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "git_hash": get_git_hash(),
        "seed": int(args.seed),
        "n_bootstrap": int(args.n_bootstrap),
        "fpca_components_requested": int(args.fpca_components),
        "fpca_meta": fpca_meta,
        "canonical_split_path": str(bench72.CANONICAL_SPLIT_PATH),
        "benchmark72_outputs": str(OUTPUT_72_DIR),
        "n_curves_total": int(len(common_ids)),
        "n_train": int(len(train_idx)),
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "early_times_fixed": EARLY_TIMES,
        "future_start": float(FUTURE_START),
        "eval_times": time_grid[future_mask].tolist(),
        "notes": [
            "GlobalMean uses only the TRAIN mean future trajectory and ignores observations.",
            "fPCA-Ridge uses the same fixed [x, Q(1), Q(3), Q(5), Q(7)] mouth as DirectQ-fixed.",
            "This is a discrete PCA-on-future-curves baseline, not a continuous spline FDA implementation.",
        ],
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "=== 38b -- canonical functional baselines ===",
        "",
        f"train/cal/test       : {len(train_idx)}/{len(cal_idx)}/{len(test_idx)}",
        f"future eval times    : {time_grid[future_mask].tolist()}",
        f"requested PCA comps  : {args.fpca_components}",
        f"actual PCA comps     : {fpca_meta['n_components']}",
        f"chosen ridge alpha   : {fpca_meta['chosen_alpha']:.6g}",
        "",
        "--- summary ---",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"  {row['method']:<12s} rmse={row['rmse']:.4f}  r2={row['r2_pooled']:.3f}  "
            f"median_curve_rmse={row['per_curve_rmse_median']:.4f}"
        )
    lines.append("")
    lines.append("--- pairwise vs 72 ---")
    for _, row in pairwise_df.iterrows():
        lines.append(
            f"  {row['baseline_method']} - {row['reference_method']}: "
            f"delta_rmse={row['delta_rmse_baseline_minus_reference']:+.4f}  "
            f"95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        )
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[38b] wrote outputs to {args.out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
