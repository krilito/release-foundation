"""99 - PLGA static-to-early proxy probe.

Purpose:
    Test whether static formulation descriptors can infer the early release
    context that made direct early-Q black-box controls strong.

    The diagnostic question is deliberately narrow:

        static descriptors -> early proxy features -> future release

    If predicted early features recover much of the true-early gain, the
    static descriptors contain an indirect early-state signal. If not, the
    early observations are likely carrying missing process/microstructure
    information that formula search cannot invent.

Consumes:
    outputs/148_release_main_cumulative_v1/{curves_long.csv,formulations.csv}
    outputs/91_plga_static_feature_ceiling_*/per_curve_metrics.csv

Produces:
    outputs/99_plga_static_to_early_proxy_probe/
      per_curve_metrics.csv
      proxy_fold_metrics.csv
      summary_by_method.csv
      decision_table.csv
      lock_metadata.json
      report.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT97 = ROOT / "scripts" / "97_plga_direct_early_blackbox_control.py"
DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/99_plga_static_to_early_proxy_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLGA static-to-early proxy probe.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default="1,2,3,5")
    parser.add_argument("--proxy-model", choices=["extra_trees", "random_forest", "ridge"], default="extra_trees")
    parser.add_argument("--direct-model", choices=["extra_trees", "random_forest", "ridge"], default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-curves", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_script97():
    spec = importlib.util.spec_from_file_location("script97", SCRIPT97)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT97}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_hash() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def args_to_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 1:
        return np.nan
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))


def parse_budgets(text: str) -> list[int]:
    budgets = [int(x) for x in str(text).split(",") if x.strip()]
    return [k for k in budgets if k > 0]


def proxy_target_names(k: int) -> list[str]:
    return [f"q_{i + 1}" for i in range(k)] + [f"slope_{i + 1}" for i in range(k)]


def split_proxy_parts(early_matrix: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([early_matrix[:, k : 2 * k], early_matrix[:, 2 * k : 3 * k]], axis=1)


def compose_predicted_early(
    *,
    true_early: np.ndarray,
    predicted_proxy: np.ndarray,
    train_proxy: np.ndarray,
    k: int,
) -> np.ndarray:
    out = true_early.copy()
    q_pred = np.clip(predicted_proxy[:, :k], 0.0, 1.0)
    slope_pred = predicted_proxy[:, k : 2 * k].copy()
    lo = np.nanpercentile(train_proxy[:, k : 2 * k], 1, axis=0)
    hi = np.nanpercentile(train_proxy[:, k : 2 * k], 99, axis=0)
    lo = np.where(np.isfinite(lo), lo, -1e6)
    hi = np.where(np.isfinite(hi), hi, 1e6)
    slope_pred = np.clip(slope_pred, lo, hi)
    out[:, k : 2 * k] = q_pred
    out[:, 2 * k : 3 * k] = slope_pred
    return out


def summarize_proxy_fold(
    *,
    split_kind: str,
    fold: int,
    budget: int,
    model_name: str,
    train_size: int,
    test_size: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    names = proxy_target_names(budget)
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        rows.append(
            {
                "split_kind": split_kind,
                "fold": int(fold),
                "budget": int(budget),
                "proxy_model": model_name,
                "target": name,
                "target_group": "q" if name.startswith("q_") else "slope",
                "rmse": rmse(y_true[:, i], y_pred[:, i]),
                "r2": safe_r2(y_true[:, i], y_pred[:, i]),
                "n_train_curves": int(train_size),
                "n_test_curves": int(test_size),
            }
        )
    return pd.DataFrame(rows)


def add_eval_rows(
    *,
    s97,
    rows: list[pd.DataFrame],
    split_kind: str,
    fold: int,
    budget: int,
    method: str,
    model_name: str,
    train_points: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    test_points: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    seed: int,
    n_estimators: int,
) -> None:
    x_train, y_train, _ = train_points
    x_test, y_test, meta_test = test_points
    if len(x_train) < 50 or len(x_test) < 1:
        return
    model = s97.make_model(model_name, seed, n_estimators)
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=float)
    eval_df = s97.evaluate_points(meta_test, y_test, pred)
    eval_df.insert(0, "split_kind", split_kind)
    eval_df.insert(1, "fold", int(fold))
    eval_df.insert(2, "budget", int(budget))
    eval_df.insert(3, "method", method)
    eval_df.insert(4, "model", model_name)
    eval_df["n_train_points"] = int(len(x_train))
    eval_df["n_test_points"] = int(len(x_test))
    rows.append(eval_df)


def run_probe(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    s97 = load_script97()
    s97.seed_all(args.seed)
    forms, curve_groups = s97.load_plga_data(args)
    splits = s97.load_split_assignments(args)
    forms = forms[forms["unified_curve_id"].isin(set(splits["unified_curve_id"]))].copy()
    forms = forms[forms["unified_curve_id"].isin(set(curve_groups))].copy()
    splits = splits[splits["unified_curve_id"].isin(set(forms["unified_curve_id"]))].copy()
    forms_indexed = forms.set_index("unified_curve_id", drop=False)

    budgets = parse_budgets(args.budgets)
    metric_rows: list[pd.DataFrame] = []
    proxy_rows: list[pd.DataFrame] = []

    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        split_df = splits[splits["split_kind"] == split_kind].copy()
        for fold in sorted(split_df["fold"].unique()):
            test_ids = sorted(split_df.loc[split_df["fold"] == fold, "unified_curve_id"].astype(str).unique())
            train_ids = sorted(set(split_df["unified_curve_id"].astype(str).unique()) - set(test_ids))
            train_forms_base = forms_indexed.loc[[cid for cid in train_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            test_forms_base = forms_indexed.loc[[cid for cid in test_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            if len(train_forms_base) < 20 or test_forms_base.empty:
                continue

            for budget in budgets:
                train_early, train_last = s97.early_features(
                    curve_groups,
                    train_forms_base["unified_curve_id"].astype(str).tolist(),
                    budget,
                )
                test_early, test_last = s97.early_features(
                    curve_groups,
                    test_forms_base["unified_curve_id"].astype(str).tolist(),
                    budget,
                )
                valid_train = np.isfinite(train_early).all(axis=1)
                valid_test = np.isfinite(test_early).all(axis=1)
                train_forms = train_forms_base.loc[valid_train].copy().reset_index(drop=True)
                test_forms = test_forms_base.loc[valid_test].copy().reset_index(drop=True)
                train_early_v = train_early[valid_train]
                test_early_v = test_early[valid_test]
                train_last_v = train_last[valid_train]
                test_last_v = test_last[valid_test]
                if len(train_forms) < 20 or test_forms.empty:
                    continue

                x_train_static, x_test_static = s97.build_static_design(train_forms, test_forms)
                y_proxy_train = split_proxy_parts(train_early_v, budget)
                y_proxy_test = split_proxy_parts(test_early_v, budget)
                proxy_seed = args.seed + int(fold) * 101 + budget * 997 + len(split_kind)
                proxy = s97.make_model(args.proxy_model, proxy_seed, args.n_estimators)
                proxy.fit(x_train_static.to_numpy(dtype=float), y_proxy_train)
                y_proxy_pred = np.asarray(proxy.predict(x_test_static.to_numpy(dtype=float)), dtype=float)
                proxy_rows.append(
                    summarize_proxy_fold(
                        split_kind=split_kind,
                        fold=int(fold),
                        budget=budget,
                        model_name=args.proxy_model,
                        train_size=len(train_forms),
                        test_size=len(test_forms),
                        y_true=y_proxy_test,
                        y_pred=y_proxy_pred,
                    )
                )
                pred_early_v = compose_predicted_early(
                    true_early=test_early_v,
                    predicted_proxy=y_proxy_pred,
                    train_proxy=y_proxy_train,
                    k=budget,
                )

                static_train = s97.build_point_table(
                    forms=train_forms,
                    static_design=x_train_static,
                    early_matrix=train_early_v,
                    last_times=train_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="static",
                )
                static_test = s97.build_point_table(
                    forms=test_forms,
                    static_design=x_test_static,
                    early_matrix=test_early_v,
                    last_times=test_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="static",
                )
                true_static_early_train = s97.build_point_table(
                    forms=train_forms,
                    static_design=x_train_static,
                    early_matrix=train_early_v,
                    last_times=train_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="static_early",
                )
                true_static_early_test = s97.build_point_table(
                    forms=test_forms,
                    static_design=x_test_static,
                    early_matrix=test_early_v,
                    last_times=test_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="static_early",
                )
                pred_static_early_test = s97.build_point_table(
                    forms=test_forms,
                    static_design=x_test_static,
                    early_matrix=pred_early_v,
                    last_times=test_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="static_early",
                )
                true_early_train = s97.build_point_table(
                    forms=train_forms,
                    static_design=x_train_static,
                    early_matrix=train_early_v,
                    last_times=train_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="early",
                )
                true_early_test = s97.build_point_table(
                    forms=test_forms,
                    static_design=x_test_static,
                    early_matrix=test_early_v,
                    last_times=test_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="early",
                )
                pred_early_test = s97.build_point_table(
                    forms=test_forms,
                    static_design=x_test_static,
                    early_matrix=pred_early_v,
                    last_times=test_last_v,
                    curve_groups=curve_groups,
                    budget=budget,
                    feature_mode="early",
                )

                base_seed = args.seed + int(fold) * 31 + budget * 131 + len(split_kind)
                add_eval_rows(
                    s97=s97,
                    rows=metric_rows,
                    split_kind=split_kind,
                    fold=int(fold),
                    budget=budget,
                    method="static_time",
                    model_name=args.direct_model,
                    train_points=static_train,
                    test_points=static_test,
                    seed=base_seed,
                    n_estimators=args.n_estimators,
                )
                add_eval_rows(
                    s97=s97,
                    rows=metric_rows,
                    split_kind=split_kind,
                    fold=int(fold),
                    budget=budget,
                    method="true_static_early_time",
                    model_name=args.direct_model,
                    train_points=true_static_early_train,
                    test_points=true_static_early_test,
                    seed=base_seed + 1,
                    n_estimators=args.n_estimators,
                )
                add_eval_rows(
                    s97=s97,
                    rows=metric_rows,
                    split_kind=split_kind,
                    fold=int(fold),
                    budget=budget,
                    method="pred_static_early_time",
                    model_name=args.direct_model,
                    train_points=true_static_early_train,
                    test_points=pred_static_early_test,
                    seed=base_seed + 1,
                    n_estimators=args.n_estimators,
                )
                add_eval_rows(
                    s97=s97,
                    rows=metric_rows,
                    split_kind=split_kind,
                    fold=int(fold),
                    budget=budget,
                    method="true_early_time",
                    model_name=args.direct_model,
                    train_points=true_early_train,
                    test_points=true_early_test,
                    seed=base_seed + 2,
                    n_estimators=args.n_estimators,
                )
                add_eval_rows(
                    s97=s97,
                    rows=metric_rows,
                    split_kind=split_kind,
                    fold=int(fold),
                    budget=budget,
                    method="pred_early_time",
                    model_name=args.direct_model,
                    train_points=true_early_train,
                    test_points=pred_early_test,
                    seed=base_seed + 2,
                    n_estimators=args.n_estimators,
                )

    per_curve = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    proxy_metrics = pd.concat(proxy_rows, ignore_index=True) if proxy_rows else pd.DataFrame()
    return per_curve, proxy_metrics


def summarize_methods(per_curve: pd.DataFrame) -> pd.DataFrame:
    if per_curve.empty:
        return pd.DataFrame()
    return (
        per_curve.groupby(["split_kind", "budget", "method"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_mae=("future_mae", "median"),
            median_future_r2=("future_r2", "median"),
            frac_r2_positive=("future_r2", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.0))),
            median_context_time=("last_context_time", "median"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget", "median_future_rmse"])
    )


def summarize_proxy(proxy_metrics: pd.DataFrame) -> pd.DataFrame:
    if proxy_metrics.empty:
        return pd.DataFrame()
    return (
        proxy_metrics.groupby(["split_kind", "budget", "target_group"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget", "target_group"])
    )


def build_decision_table(summary: pd.DataFrame, proxy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame()
    min_meaningful_gain = 0.005
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        for budget in sorted(summary.loc[summary["split_kind"] == split_kind, "budget"].unique()):
            sub = summary[(summary["split_kind"] == split_kind) & (summary["budget"] == budget)].copy()
            vals = {
                row["method"]: float(row["median_future_rmse"])
                for _, row in sub.iterrows()
                if np.isfinite(float(row["median_future_rmse"]))
            }
            need = {"static_time", "true_static_early_time", "pred_static_early_time"}
            if not need.issubset(vals):
                continue
            static_rmse = vals["static_time"]
            true_rmse = vals["true_static_early_time"]
            pred_rmse = vals["pred_static_early_time"]
            denom = static_rmse - true_rmse
            recovery = (static_rmse - pred_rmse) / denom if denom > min_meaningful_gain else np.nan
            ps = proxy_summary[
                (proxy_summary["split_kind"] == split_kind)
                & (proxy_summary["budget"] == budget)
                & (proxy_summary["target_group"] == "q")
            ]
            q_r2 = float(ps["median_r2"].iloc[0]) if not ps.empty else np.nan
            if denom <= min_meaningful_gain:
                answer = "measured early gain is negligible; proxy question is underpowered"
            elif np.isfinite(recovery) and recovery >= 0.5 and pred_rmse < static_rmse:
                answer = "static proxy recovers a meaningful share of true-early gain"
            elif pred_rmse < static_rmse:
                answer = "static proxy helps, but recovery is weak"
            else:
                answer = "static proxy does not replace measured early points"
            rows.append(
                {
                    "split_kind": split_kind,
                    "budget": int(budget),
                    "answer": answer,
                    "static_rmse": static_rmse,
                    "true_early_rmse": true_rmse,
                    "predicted_early_rmse": pred_rmse,
                    "recovery_fraction": recovery,
                    "median_q_proxy_r2": q_r2,
                }
            )
    return pd.DataFrame(rows)


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    unexpected: dict[str, int] = {}
    allowed: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].to_numpy(dtype=float)
        nonfinite = ~np.isfinite(vals)
        if not bool(nonfinite.any()):
            continue
        allowed_mask = np.zeros(len(df), dtype=bool)
        if col in {"future_r2", "r2", "median_r2", "mean_r2", "recovery_fraction", "median_q_proxy_r2"}:
            allowed_mask |= nonfinite
        n_allowed = int((nonfinite & allowed_mask).sum())
        n_bad = int((nonfinite & ~allowed_mask).sum())
        if n_allowed:
            allowed[col] = n_allowed
        if n_bad:
            unexpected[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": unexpected, "allowed_nonfinite": allowed}


def write_report(
    out: Path,
    summary: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    focus_methods = [
        "static_time",
        "true_static_early_time",
        "pred_static_early_time",
        "true_early_time",
        "pred_early_time",
    ]
    focus = summary[summary["method"].isin(focus_methods)].copy()
    best = (
        focus.sort_values(["split_kind", "budget", "median_future_rmse"])
        .groupby(["split_kind", "budget"], as_index=False)
        .head(5)
    )
    lines = [
        "# PLGA Static-to-Early Proxy Probe",
        "",
        "Question: can static descriptors infer the early release state well enough to replace measured early points?",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "_No decisions generated._",
        "",
        "## Best Rows By Split And Budget",
        "",
        best.to_markdown(index=False) if not best.empty else "_No method summary._",
        "",
        "## Proxy Feature Summary",
        "",
        proxy_summary.to_markdown(index=False) if not proxy_summary.empty else "_No proxy summary._",
        "",
        "## Interpretation Guard",
        "",
        "- `static_time`: static descriptors plus query time, scored only after the same early budget window.",
        "- `true_static_early_time`: static descriptors plus measured early Q/slope features.",
        "- `pred_static_early_time`: same trained model, but test early Q/slope features are predicted from static descriptors.",
        "- Early sampling times are treated as known; only Q and slope features are proxied.",
        "- This is a precondition check for symbolic regression or PINN distillation, not a mechanism claim.",
        "",
        "## Run Metadata",
        "",
        f"- proxy_model: `{args.proxy_model}`",
        f"- direct_model: `{args.direct_model}`",
        f"- budgets: `{args.budgets}`",
        f"- n_estimators: `{args.n_estimators}`",
        f"- max_curves: `{args.max_curves}`",
        "",
        "## Verification",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_curve, proxy_metrics = run_probe(args)
    summary = summarize_methods(per_curve)
    proxy_summary = summarize_proxy(proxy_metrics)
    decisions = build_decision_table(summary, proxy_summary)
    checks = [
        finite_check("per_curve_metrics", per_curve),
        finite_check("proxy_fold_metrics", proxy_metrics),
        finite_check("summary_by_method", summary),
        finite_check("proxy_summary", proxy_summary),
        finite_check("decision_table", decisions),
    ]

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    proxy_metrics.to_csv(args.out / "proxy_fold_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    proxy_summary.to_csv(args.out / "proxy_summary.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "seed": args.seed,
                "pool_dir": str(args.pool_dir),
                "budgets": args.budgets,
                "proxy_model": args.proxy_model,
                "direct_model": args.direct_model,
                "n_estimators": args.n_estimators,
                "max_curves": args.max_curves,
                "checks": checks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out, summary, proxy_summary, decisions, checks, args)
    print((args.out / "report.md").resolve())
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
