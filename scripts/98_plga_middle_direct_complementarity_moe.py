"""98 - PLGA middle-vs-direct complementarity and MoE screen.

Purpose:
    Check whether the middle-layer route and direct early-Q black-box are good
    on the same curves or on complementary curves.

    If they are complementary, screen a simple deployable MoE gate:

        static descriptors + early context -> choose middle or fixed direct

Consumes:
    outputs/97_plga_direct_early_blackbox_control/combined_per_curve_metrics.csv
    outputs/148_release_main_cumulative_v1/{formulations.csv,curves_long.csv}

Produces:
    outputs/98_plga_middle_direct_complementarity_moe/
      complementarity_by_budget.csv
      moe_gate_per_curve.csv
      moe_gate_summary.csv
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
SCRIPT97 = ROOT / "scripts" / "97_plga_direct_early_blackbox_control.py"
DEFAULT_CONTROL = Path("outputs/97_plga_direct_early_blackbox_control")
DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/98_plga_middle_direct_complementarity_moe")

DIRECT_METHODS = ["direct_static_time", "direct_early_time", "direct_static_early_time"]
MIDDLE_METHOD = "middle_early_residual_family"
MIDDLE_ORACLE = "middle_future_oracle_family"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Middle/direct complementarity and MoE screen.")
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default="0,1,2,3,5")
    parser.add_argument("--gate-models", default="extra_trees,ridge,knn")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-curves", type=int, default=0)
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def make_gate(name: str, seed: int, n_estimators: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 13))
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=21, weights="distance")
    raise KeyError(name)


def load_wide(control_run: Path) -> pd.DataFrame:
    combined = read_csv(control_run / "combined_per_curve_metrics.csv").copy()
    combined["future_rmse"] = pd.to_numeric(combined["future_rmse"], errors="coerce")
    combined = combined[np.isfinite(combined["future_rmse"])].copy()
    key_cols = ["split_kind", "fold", "budget", "unified_curve_id", "source_dataset", "source_group"]
    wide = combined.pivot_table(index=key_cols, columns="method", values="future_rmse", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide


def best_direct_from_train(wide: pd.DataFrame, split_kind: str, budget: int, test_fold: int | None) -> str | None:
    sub = wide[(wide["split_kind"] == split_kind) & (wide["budget"] == budget)].copy()
    if test_fold is not None:
        sub = sub[sub["fold"] != test_fold].copy()
    medians: dict[str, float] = {}
    for method in DIRECT_METHODS:
        if method in sub.columns:
            vals = finite_numeric(sub[method])
            vals = vals[np.isfinite(vals)]
            if len(vals):
                medians[method] = float(np.median(vals))
    if not medians:
        return None
    return min(medians, key=medians.get)


def compute_complementarity(wide: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        for budget in budgets:
            sub = wide[(wide["split_kind"] == split_kind) & (wide["budget"] == budget)].copy()
            if sub.empty or MIDDLE_METHOD not in sub.columns:
                continue
            fixed_direct = best_direct_from_train(wide, split_kind, budget, None)
            if fixed_direct is None or fixed_direct not in sub.columns:
                continue
            keep_cols = [MIDDLE_METHOD, fixed_direct]
            keep_cols.extend([m for m in DIRECT_METHODS if m in sub.columns and m not in keep_cols])
            if MIDDLE_ORACLE in sub.columns and MIDDLE_ORACLE not in keep_cols:
                keep_cols.append(MIDDLE_ORACLE)
            keep = sub[keep_cols].copy()
            keep = keep.replace([np.inf, -np.inf], np.nan).dropna(subset=[MIDDLE_METHOD, fixed_direct])
            if len(keep) < 20:
                continue
            middle = keep[MIDDLE_METHOD].to_numpy(dtype=float)
            direct = keep[fixed_direct].to_numpy(dtype=float)
            direct_oracle = keep[[m for m in DIRECT_METHODS if m in keep.columns]].min(axis=1).to_numpy(dtype=float)
            deployable_oracle = np.minimum(middle, direct)
            all_direct_middle_oracle = np.minimum(middle, direct_oracle)
            corr = float(pd.Series(middle).corr(pd.Series(direct), method="spearman"))
            middle_win = middle < direct
            q_middle_good = np.quantile(middle, 0.25)
            q_direct_good = np.quantile(direct, 0.25)
            q_middle_bad = np.quantile(middle, 0.75)
            q_direct_bad = np.quantile(direct, 0.75)
            both_good = (middle <= q_middle_good) & (direct <= q_direct_good)
            both_bad = (middle >= q_middle_bad) & (direct >= q_direct_bad)
            direct_bad = direct >= q_direct_bad
            middle_bad = middle >= q_middle_bad
            row = {
                "split_kind": split_kind,
                "budget": int(budget),
                "fixed_direct_method": fixed_direct,
                "n_curves": int(len(keep)),
                "spearman_middle_direct_rmse": corr,
                "middle_win_rate_vs_fixed_direct": float(np.mean(middle_win)),
                "median_middle_rmse": float(np.median(middle)),
                "median_fixed_direct_rmse": float(np.median(direct)),
                "median_direct_oracle_rmse": float(np.median(direct_oracle)),
                "median_middle_fixed_direct_oracle_rmse": float(np.median(deployable_oracle)),
                "median_middle_all_direct_oracle_rmse": float(np.median(all_direct_middle_oracle)),
                "oracle_gain_vs_best_single": float(min(np.median(middle), np.median(direct)) - np.median(deployable_oracle)),
                "all_oracle_gain_vs_best_direct_oracle": float(np.median(direct_oracle) - np.median(all_direct_middle_oracle)),
                "both_good_q25_overlap": float(np.mean(both_good)),
                "both_bad_q75_overlap": float(np.mean(both_bad)),
                "middle_win_rate_when_direct_bad_q75": float(np.mean(middle_win[direct_bad])) if bool(direct_bad.any()) else np.nan,
                "direct_win_rate_when_middle_bad_q75": float(np.mean((direct < middle)[middle_bad])) if bool(middle_bad.any()) else np.nan,
            }
            if MIDDLE_ORACLE in keep.columns:
                mo = keep[MIDDLE_ORACLE].to_numpy(dtype=float)
                row["median_middle_future_oracle_rmse"] = float(np.median(mo))
                row["future_oracle_gain_vs_deployable_middle"] = float(np.median(middle) - np.median(mo))
                row["future_oracle_win_rate_vs_fixed_direct"] = float(np.mean(mo < direct))
            rows.append(row)
    return pd.DataFrame(rows)


def build_gate_design(s97, args: argparse.Namespace, split_kind: str, budget: int, train_ids: list[str], test_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    forms, curve_groups = s97.load_plga_data(args)
    forms = forms.set_index("unified_curve_id", drop=False)
    train_ids = [cid for cid in train_ids if cid in forms.index and cid in curve_groups]
    test_ids = [cid for cid in test_ids if cid in forms.index and cid in curve_groups]
    train_forms = forms.loc[train_ids].copy().reset_index(drop=True)
    test_forms = forms.loc[test_ids].copy().reset_index(drop=True)
    x_train_static, x_test_static = s97.build_static_design(train_forms, test_forms)
    if budget <= 0:
        return x_train_static.to_numpy(dtype=float), x_test_static.to_numpy(dtype=float), train_forms["unified_curve_id"].astype(str).tolist(), test_forms["unified_curve_id"].astype(str).tolist()
    train_early, _ = s97.early_features(curve_groups, train_forms["unified_curve_id"].astype(str).tolist(), budget)
    test_early, _ = s97.early_features(curve_groups, test_forms["unified_curve_id"].astype(str).tolist(), budget)
    valid_train = np.isfinite(train_early).all(axis=1)
    valid_test = np.isfinite(test_early).all(axis=1)
    train_ids_v = train_forms.loc[valid_train, "unified_curve_id"].astype(str).tolist()
    test_ids_v = test_forms.loc[valid_test, "unified_curve_id"].astype(str).tolist()
    x_train = np.concatenate([x_train_static.loc[valid_train].to_numpy(dtype=float), train_early[valid_train]], axis=1)
    x_test = np.concatenate([x_test_static.loc[valid_test].to_numpy(dtype=float), test_early[valid_test]], axis=1)
    return x_train, x_test, train_ids_v, test_ids_v


def run_moe_gate(wide: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    s97 = load_script97()
    rows: list[dict[str, Any]] = []
    gate_models = [x.strip() for x in str(args.gate_models).split(",") if x.strip()]
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        for budget in budgets:
            sub_all = wide[(wide["split_kind"] == split_kind) & (wide["budget"] == budget)].copy()
            if sub_all.empty or MIDDLE_METHOD not in sub_all.columns:
                continue
            for fold in sorted(sub_all["fold"].dropna().unique()):
                fixed_direct = best_direct_from_train(wide, split_kind, budget, int(fold))
                if fixed_direct is None or fixed_direct not in sub_all.columns:
                    continue
                train = sub_all[sub_all["fold"] != fold].dropna(subset=[MIDDLE_METHOD, fixed_direct]).copy()
                test = sub_all[sub_all["fold"] == fold].dropna(subset=[MIDDLE_METHOD, fixed_direct]).copy()
                if len(train) < 30 or test.empty:
                    continue
                x_train, x_test, train_ids, test_ids = build_gate_design(
                    s97,
                    args,
                    split_kind,
                    budget,
                    train["unified_curve_id"].astype(str).tolist(),
                    test["unified_curve_id"].astype(str).tolist(),
                )
                train = train.set_index("unified_curve_id").loc[train_ids].reset_index()
                test = test.set_index("unified_curve_id").loc[test_ids].reset_index()
                if len(train) < 30 or test.empty:
                    continue
                y_delta = train[MIDDLE_METHOD].to_numpy(dtype=float) - train[fixed_direct].to_numpy(dtype=float)
                for gate_name in gate_models:
                    gate = make_gate(gate_name, args.seed + int(fold) * 17 + budget * 101 + len(gate_name), args.n_estimators)
                    gate.fit(x_train, y_delta)
                    pred_delta = np.asarray(gate.predict(x_test), dtype=float)
                    choose_middle = pred_delta < 0.0
                    middle = test[MIDDLE_METHOD].to_numpy(dtype=float)
                    direct = test[fixed_direct].to_numpy(dtype=float)
                    chosen = np.where(choose_middle, middle, direct)
                    oracle = np.minimum(middle, direct)
                    for i, curve_id in enumerate(test["unified_curve_id"].astype(str)):
                        rows.append(
                            {
                                "split_kind": split_kind,
                                "fold": int(fold),
                                "budget": int(budget),
                                "gate_model": gate_name,
                                "fixed_direct_method": fixed_direct,
                                "unified_curve_id": curve_id,
                                "source_dataset": str(test.iloc[i].get("source_dataset", "")),
                                "source_group": str(test.iloc[i].get("source_group", "")),
                                "middle_rmse": float(middle[i]),
                                "direct_rmse": float(direct[i]),
                                "moe_rmse": float(chosen[i]),
                                "oracle_pair_rmse": float(oracle[i]),
                                "choose_middle": bool(choose_middle[i]),
                                "middle_is_better": bool(middle[i] < direct[i]),
                                "gate_hit_better": bool((choose_middle[i] and middle[i] < direct[i]) or ((not choose_middle[i]) and direct[i] <= middle[i])),
                                "predicted_delta_middle_minus_direct": float(pred_delta[i]),
                            }
                        )
    return pd.DataFrame(rows)


def summarize_gate(gate_df: pd.DataFrame) -> pd.DataFrame:
    if gate_df.empty:
        return pd.DataFrame()
    return (
        gate_df.groupby(["split_kind", "budget", "gate_model"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_moe_rmse=("moe_rmse", "median"),
            median_middle_rmse=("middle_rmse", "median"),
            median_direct_rmse=("direct_rmse", "median"),
            median_oracle_pair_rmse=("oracle_pair_rmse", "median"),
            choose_middle_rate=("choose_middle", "mean"),
            middle_better_rate=("middle_is_better", "mean"),
            gate_hit_rate=("gate_hit_better", "mean"),
            fixed_direct_methods=("fixed_direct_method", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["split_kind", "budget", "median_moe_rmse"])
    )


def build_decisions(comp: pd.DataFrame, gate_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        sub = comp[comp["split_kind"] == split_kind].copy()
        if sub.empty:
            continue
        best_oracle = sub.sort_values("median_middle_fixed_direct_oracle_rmse").iloc[0]
        best_gate = gate_summary[gate_summary["split_kind"] == split_kind].sort_values("median_moe_rmse").head(1)
        oracle_gain = float(best_oracle["oracle_gain_vs_best_single"])
        middle_win_bad = float(best_oracle["middle_win_rate_when_direct_bad_q75"])
        if best_gate.empty:
            gate_text = "no gate result"
            answer = "diagnostic only"
        else:
            g = best_gate.iloc[0]
            best_single = min(float(g["median_middle_rmse"]), float(g["median_direct_rmse"]))
            gate_gain = best_single - float(g["median_moe_rmse"])
            gate_text = f"best gate RMSE={float(g['median_moe_rmse']):.3f} at k={int(g['budget'])}, gate_gain={gate_gain:+.3f}"
            answer = "try MoE carefully" if gate_gain > 0.005 else "oracle complementarity not deployable yet"
        rows.append(
            {
                "split_kind": split_kind,
                "question": "Are direct and middle complementary enough for MoE?",
                "answer": answer,
                "evidence": (
                    f"pair-oracle gain={oracle_gain:.3f} at k={int(best_oracle['budget'])}; "
                    f"middle wins {middle_win_bad:.1%} of direct-bad curves; {gate_text}"
                ),
            }
        )
    return pd.DataFrame(rows)


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].to_numpy(dtype=float)
        n_bad = int((~np.isfinite(vals)).sum())
        if n_bad:
            bad[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": bad}


def write_report(out: Path, comp: pd.DataFrame, gate_summary: pd.DataFrame, decisions: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    best_comp = (
        comp.sort_values(["split_kind", "median_middle_fixed_direct_oracle_rmse"])
        .groupby("split_kind", as_index=False)
        .head(6)
    )
    best_gate = (
        gate_summary.sort_values(["split_kind", "median_moe_rmse"])
        .groupby("split_kind", as_index=False)
        .head(8)
        if not gate_summary.empty
        else pd.DataFrame()
    )
    lines = [
        "# PLGA Middle/Direct Complementarity and MoE Screen",
        "",
        "This is a diagnostic plus first gate screen. Oracle rows are not deployable.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "_No decisions generated._",
        "",
        "## Best Pair-Oracles By Split",
        "",
        best_comp.to_markdown(index=False),
        "",
        "## Best MoE Gates By Split",
        "",
        best_gate.to_markdown(index=False) if not best_gate.empty else "_No gate results._",
        "",
        "## Interpretation Guard",
        "",
        "- Low Spearman correlation or large pair-oracle gain means complementarity exists.",
        "- A useful MoE must beat the best single deployable method without seeing future error.",
        "- If pair-oracle helps but gate does not, the missing object is a better selector, not a bigger expert.",
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
    wide = load_wide(args.control_run)
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    comp = compute_complementarity(wide, budgets)
    gate = run_moe_gate(wide, args)
    gate_summary = summarize_gate(gate)
    decisions = build_decisions(comp, gate_summary)

    comp.to_csv(args.out / "complementarity_by_budget.csv", index=False)
    gate.to_csv(args.out / "moe_gate_per_curve.csv", index=False)
    gate_summary.to_csv(args.out / "moe_gate_summary.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks = [
        finite_check("complementarity_by_budget", comp),
        finite_check("moe_gate_per_curve", gate),
        finite_check("moe_gate_summary", gate_summary),
        finite_check("decision_table", decisions),
    ]
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "control_run": str(args.control_run),
                "budgets": args.budgets,
                "gate_models": args.gate_models,
                "n_estimators": args.n_estimators,
                "seed": args.seed,
                "checks": checks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out, comp, gate_summary, decisions, checks)
    print((args.out / "report.md").resolve())
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
