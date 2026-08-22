"""97 - PLGA direct early-Q black-box matched control.

Purpose:
    Compare the early-conditioned middle-layer route from script 96 against
    matched direct black-box controls on the same splits, early budgets, and
    future-only target points.

Consumes:
    outputs/148_release_main_cumulative_v1/curves_long.csv
    outputs/148_release_main_cumulative_v1/formulations.csv
    outputs/91_plga_static_feature_ceiling_*/per_curve_metrics.csv
    outputs/96_plga_early_middle_layer_selector_probe/per_curve_selector_metrics.csv

Produces:
    outputs/97_plga_direct_early_blackbox_control/
      direct_per_curve_metrics.csv
      combined_per_curve_metrics.csv
      summary_by_method.csv
      decision_table.csv
      lock_metadata.json
      report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV


DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_MIDDLE = Path("outputs/96_plga_early_middle_layer_selector_probe")
DEFAULT_OUT = Path("outputs/97_plga_direct_early_blackbox_control")

PLGA_SOURCES = {"internal181", "cross321"}

NUMERIC_COLS = [
    "Polymer_MW",
    "LA/GA",
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Initial D/M ratio",
    "DLC",
    "DLC_percent",
    "EE",
    "Particle_Size",
    "SA-V",
    "SE",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_NHA",
    "Drug_LogP",
    "media_pH",
    "media_temp_oC",
    "PDI",
    "zeta_potential",
    "weighted_Mw",
    "weighted_Tm",
]
CATEGORICAL_COLS = [
    "polymer_family",
    "payload_name",
    "release_medium_condition",
    "release_method",
    "measurement_assay",
    "structure_type",
    "light_condition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct early-Q black-box matched control for PLGA.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--middle-run", type=Path, default=DEFAULT_MIDDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default="0,1,2,3,5")
    parser.add_argument("--model", choices=["extra_trees", "random_forest", "ridge"], default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-curves", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def clean_category(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "__MISSING__"


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_split_assignments(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for run_label, run in [
        ("random_kfold", args.random_run),
        ("source_group_kfold", args.source_group_run),
        ("source_dataset_lodo", args.source_dataset_run),
    ]:
        df = read_csv(run / "per_curve_metrics.csv")
        keep = ["split_kind", "fold", "unified_curve_id", "source_dataset", "source_group"]
        sub = df[keep].drop_duplicates().copy()
        sub.insert(0, "run_label", run_label)
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["unified_curve_id"] = out["unified_curve_id"].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce").astype(int)
    return out


def load_plga_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    forms = read_csv(args.pool_dir / "formulations.csv")
    forms = forms[forms["source_dataset"].astype(str).isin(PLGA_SOURCES)].drop_duplicates("unified_curve_id").copy()
    forms["unified_curve_id"] = forms["unified_curve_id"].astype(str)
    if args.max_curves and len(forms) > args.max_curves:
        forms = forms.sample(n=args.max_curves, random_state=args.seed).sort_values("unified_curve_id").copy()

    ids = set(forms["unified_curve_id"])
    curves = read_csv(args.pool_dir / "curves_long.csv")
    curves["unified_curve_id"] = curves["unified_curve_id"].astype(str)
    curves = curves[curves["unified_curve_id"].isin(ids)].copy()
    curves["time_days"] = pd.to_numeric(curves["time_days"], errors="coerce")
    curves["release_fraction_raw"] = pd.to_numeric(curves["release_fraction"], errors="coerce")
    curves = curves.dropna(subset=["time_days", "release_fraction_raw"]).copy()
    curves["release_fraction"] = curves["release_fraction_raw"].clip(0.0, 1.0)
    curve_groups = {
        cid: sub.groupby("time_days", as_index=False)
        .agg(release_fraction=("release_fraction", "mean"), release_fraction_raw=("release_fraction_raw", "mean"))
        .sort_values("time_days")
        .reset_index(drop=True)
        for cid, sub in curves.groupby("unified_curve_id", sort=True)
    }
    good_ids = {cid for cid, sub in curve_groups.items() if len(sub) >= 3}
    forms = forms[forms["unified_curve_id"].isin(good_ids)].copy()
    return forms.reset_index(drop=True), curve_groups


def build_static_design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = [col for col in NUMERIC_COLS if col in train.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in train.columns]
    train_parts = [pd.DataFrame({"bias": np.ones(len(train), dtype=float)}, index=train.index)]
    test_parts = [pd.DataFrame({"bias": np.ones(len(test), dtype=float)}, index=test.index)]

    if numeric_cols:
        tr_num = train[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        te_num = test[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = tr_num.median(axis=0).fillna(0.0)
        tr_missing = tr_num.isna().astype(float).rename(columns={col: f"{col}__missing" for col in numeric_cols})
        te_missing = te_num.isna().astype(float).rename(columns={col: f"{col}__missing" for col in numeric_cols})
        tr_num = tr_num.fillna(med)
        te_num = te_num.fillna(med)
        mean = tr_num.mean(axis=0)
        std = tr_num.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        train_parts.extend([(tr_num - mean) / std, tr_missing])
        test_parts.extend([(te_num - mean) / std, te_missing])

    if categorical_cols:
        tr_cat = train[categorical_cols].map(clean_category)
        te_cat = test[categorical_cols].map(clean_category)
        for col in categorical_cols:
            known = set(tr_cat[col].unique())
            te_cat[col] = te_cat[col].where(te_cat[col].isin(known), "__UNK__")
        combined = pd.concat([tr_cat, te_cat], axis=0)
        dummy = pd.get_dummies(combined, prefix=categorical_cols, dtype=float)
        train_parts.append(dummy.iloc[: len(train)].set_index(train.index))
        test_parts.append(dummy.iloc[len(train) :].set_index(test.index))

    x_train = pd.concat(train_parts, axis=1)
    x_test = pd.concat(test_parts, axis=1).reindex(columns=x_train.columns, fill_value=0.0)
    return x_train.reset_index(drop=True), x_test.reset_index(drop=True)


def early_features(curve_groups: dict[str, pd.DataFrame], curve_ids: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    if k == 0:
        return np.zeros((len(curve_ids), 0), dtype=float), np.full(len(curve_ids), np.nan)
    feats = np.full((len(curve_ids), k * 3), np.nan, dtype=float)
    last_t = np.full(len(curve_ids), np.nan, dtype=float)
    for i, cid in enumerate(curve_ids):
        c = curve_groups[cid]
        if len(c) < k + 1:
            continue
        early = c.iloc[:k]
        t = early["time_days"].to_numpy(dtype=float)
        q = early["release_fraction"].to_numpy(dtype=float)
        dt = np.diff(np.r_[0.0, t])
        dq = np.diff(np.r_[0.0, q])
        slope = dq / np.clip(dt, 1e-8, None)
        feats[i, 0:k] = np.log1p(np.clip(t, 0.0, None))
        feats[i, k : 2 * k] = q
        feats[i, 2 * k : 3 * k] = slope
        last_t[i] = float(t[-1])
    return feats, last_t


def query_features(t: float, last_t: float) -> np.ndarray:
    if np.isfinite(last_t):
        delta = max(float(t) - float(last_t), 0.0)
        ratio = float(t) / max(float(last_t), 1e-8)
    else:
        delta = float(t)
        ratio = 1.0
    return np.asarray([float(t), np.log1p(max(float(t), 0.0)), delta, np.log1p(delta), ratio], dtype=float)


def make_model(name: str, seed: int, n_estimators: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 13))
    raise KeyError(name)


def build_point_table(
    *,
    forms: pd.DataFrame,
    static_design: pd.DataFrame,
    early_matrix: np.ndarray,
    last_times: np.ndarray,
    curve_groups: dict[str, pd.DataFrame],
    budget: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    curve_ids = forms["unified_curve_id"].astype(str).tolist()

    for i, cid in enumerate(curve_ids):
        c = curve_groups[cid]
        if budget > 0 and (not np.isfinite(early_matrix[i]).all()):
            continue
        last_t = float(last_times[i]) if budget > 0 else np.nan
        t_all = c["time_days"].to_numpy(dtype=float)
        y_all = c["release_fraction"].to_numpy(dtype=float)
        mask = t_all > last_t + 1e-12 if np.isfinite(last_t) else np.ones_like(t_all, dtype=bool)
        if int(mask.sum()) < 1:
            continue
        static_part = static_design.iloc[i].to_numpy(dtype=float)
        early_part = early_matrix[i] if budget > 0 else np.zeros(0, dtype=float)
        for t, y in zip(t_all[mask], y_all[mask]):
            parts = []
            if "static" in feature_mode:
                parts.append(static_part)
            if "early" in feature_mode:
                parts.append(early_part)
            parts.append(query_features(float(t), last_t))
            x_rows.append(np.concatenate(parts))
            y_rows.append(float(y))
            meta_rows.append(
                {
                    "unified_curve_id": cid,
                    "time_days": float(t),
                    "last_context_time": last_t,
                    "source_dataset": str(forms.iloc[i].get("source_dataset", "")),
                    "source_group": clean_category(forms.iloc[i].get("source_group", "")),
                }
            )
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), pd.DataFrame(meta_rows)


def evaluate_points(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    work = meta.copy()
    work["y_true"] = y_true
    work["y_pred"] = np.clip(y_pred, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    for cid, sub in work.groupby("unified_curve_id", sort=True):
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "unified_curve_id": cid,
                "source_dataset": str(sub["source_dataset"].iloc[0]),
                "source_group": str(sub["source_group"].iloc[0]),
                "last_context_time": float(sub["last_context_time"].iloc[0]),
                "future_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
                "future_mae": float(np.mean(np.abs(yp - yt))),
                "future_r2": safe_r2(yt, yp),
                "n_future": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def run_direct_controls(args: argparse.Namespace) -> pd.DataFrame:
    forms, curve_groups = load_plga_data(args)
    splits = load_split_assignments(args)
    forms = forms[forms["unified_curve_id"].isin(set(splits["unified_curve_id"]))].copy()
    forms = forms[forms["unified_curve_id"].isin(set(curve_groups))].copy()
    split_ids = set(forms["unified_curve_id"])
    splits = splits[splits["unified_curve_id"].isin(split_ids)].copy()
    forms_indexed = forms.set_index("unified_curve_id", drop=False)
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]

    feature_modes = ["static", "early", "static_early"]
    rows: list[pd.DataFrame] = []

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
                train_early, train_last = early_features(curve_groups, train_forms_base["unified_curve_id"].astype(str).tolist(), budget)
                test_early, test_last = early_features(curve_groups, test_forms_base["unified_curve_id"].astype(str).tolist(), budget)
                valid_train = np.isfinite(train_early).all(axis=1) if budget > 0 else np.ones(len(train_forms_base), dtype=bool)
                valid_test = np.isfinite(test_early).all(axis=1) if budget > 0 else np.ones(len(test_forms_base), dtype=bool)
                train_forms = train_forms_base.loc[valid_train].copy().reset_index(drop=True)
                test_forms = test_forms_base.loc[valid_test].copy().reset_index(drop=True)
                train_early_v = train_early[valid_train]
                test_early_v = test_early[valid_test]
                train_last_v = train_last[valid_train]
                test_last_v = test_last[valid_test]
                if len(train_forms) < 20 or test_forms.empty:
                    continue

                x_train_static, x_test_static = build_static_design(train_forms, test_forms)
                for feature_mode in feature_modes:
                    if feature_mode in {"early", "static_early"} and budget == 0:
                        continue
                    x_train, y_train, _ = build_point_table(
                        forms=train_forms,
                        static_design=x_train_static,
                        early_matrix=train_early_v,
                        last_times=train_last_v,
                        curve_groups=curve_groups,
                        budget=budget,
                        feature_mode=feature_mode,
                    )
                    x_test, y_test, meta_test = build_point_table(
                        forms=test_forms,
                        static_design=x_test_static,
                        early_matrix=test_early_v,
                        last_times=test_last_v,
                        curve_groups=curve_groups,
                        budget=budget,
                        feature_mode=feature_mode,
                    )
                    if len(x_train) < 50 or len(x_test) < 1:
                        continue
                    model = make_model(args.model, args.seed + int(fold) * 31 + budget * 101 + len(feature_mode), args.n_estimators)
                    model.fit(x_train, y_train)
                    pred = np.asarray(model.predict(x_test), dtype=float)
                    eval_df = evaluate_points(meta_test, y_test, pred)
                    eval_df.insert(0, "split_kind", split_kind)
                    eval_df.insert(1, "fold", int(fold))
                    eval_df.insert(2, "budget", int(budget))
                    eval_df.insert(3, "method", f"direct_{feature_mode}_time")
                    eval_df.insert(4, "model", args.model)
                    eval_df["n_train_points"] = int(len(x_train))
                    eval_df["n_test_points"] = int(len(x_test))
                    rows.append(eval_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_middle_metrics(args: argparse.Namespace) -> pd.DataFrame:
    middle = read_csv(args.middle_run / "per_curve_selector_metrics.csv")
    keep_selectors = ["train_global_family", "early_residual_family", "future_oracle_family"]
    middle = middle[middle["selector"].astype(str).isin(keep_selectors)].copy()
    middle["method"] = "middle_" + middle["selector"].astype(str)
    cols = [
        "split_kind",
        "fold",
        "budget",
        "method",
        "unified_curve_id",
        "source_dataset",
        "source_group",
        "last_context_time",
        "future_rmse",
        "future_mae",
        "future_r2",
        "n_future",
    ]
    return middle[cols].copy()


def summarize(combined: pd.DataFrame) -> pd.DataFrame:
    return (
        combined.groupby(["split_kind", "budget", "method"], dropna=False)
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


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        sub = summary[summary["split_kind"] == split_kind].copy()
        if sub.empty:
            continue
        middle = sub[sub["method"] == "middle_early_residual_family"].sort_values("median_future_rmse")
        direct = sub[sub["method"].isin(["direct_static_time", "direct_early_time", "direct_static_early_time"])].sort_values("median_future_rmse")
        oracle = sub[sub["method"] == "middle_future_oracle_family"].sort_values("median_future_rmse")
        if middle.empty or direct.empty:
            continue
        m = middle.iloc[0]
        d = direct.iloc[0]
        answer = "middle wins matched direct control" if float(m["median_future_rmse"]) < float(d["median_future_rmse"]) else "direct control wins"
        evidence = (
            f"best middle={float(m['median_future_rmse']):.3f} at k={int(m['budget'])}; "
            f"best direct={float(d['median_future_rmse']):.3f} ({d['method']}) at k={int(d['budget'])}"
        )
        if not oracle.empty:
            o = oracle.iloc[0]
            evidence += f"; middle future-oracle={float(o['median_future_rmse']):.3f} at k={int(o['budget'])}"
        rows.append(
            {
                "split_kind": split_kind,
                "question": "Does early-middle beat matched direct early-Q black-box controls?",
                "answer": answer,
                "evidence": evidence,
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
        if "budget" in df.columns:
            budget0 = pd.to_numeric(df["budget"], errors="coerce").fillna(-1).to_numpy(dtype=float) == 0
            if col in {"last_context_time", "median_context_time"}:
                allowed_mask |= budget0
        if col == "future_r2":
            allowed_mask |= nonfinite
        if col in {"n_train_points", "n_test_points"} and "method" in df.columns:
            methods = df["method"].astype(str).to_numpy()
            allowed_mask |= np.asarray([m.startswith("middle_") for m in methods], dtype=bool)
        if col in {"future_rmse", "future_mae"} and "n_future" in df.columns:
            n_future = pd.to_numeric(df["n_future"], errors="coerce").fillna(0).to_numpy(dtype=float)
            allowed_mask |= n_future < 2
        n_allowed = int((nonfinite & allowed_mask).sum())
        n_bad = int((nonfinite & ~allowed_mask).sum())
        if n_allowed:
            allowed[col] = n_allowed
        if n_bad:
            unexpected[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": unexpected, "allowed_nonfinite": allowed}


def write_report(out: Path, summary: pd.DataFrame, decisions: pd.DataFrame, checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    focus_methods = [
        "middle_early_residual_family",
        "middle_train_global_family",
        "middle_future_oracle_family",
        "direct_static_time",
        "direct_early_time",
        "direct_static_early_time",
    ]
    focus = summary[summary["method"].isin(focus_methods)].copy()
    best_by_split = (
        summary[summary["method"].isin(["middle_early_residual_family", "direct_static_time", "direct_early_time", "direct_static_early_time"])]
        .sort_values(["split_kind", "median_future_rmse"])
        .groupby("split_kind", as_index=False)
        .head(8)
    )
    lines = [
        "# PLGA Direct Early-Q Black-Box Matched Control",
        "",
        "All methods use the same splits, early budgets, and future-only scoring.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "_No decisions generated._",
        "",
        "## Best Deployable Rows By Split",
        "",
        best_by_split.to_markdown(index=False),
        "",
        "## Focus Summary",
        "",
        focus.to_markdown(index=False),
        "",
        "## Interpretation Guard",
        "",
        "- `direct_static_time`: static descriptors plus query time only.",
        "- `direct_early_time`: early observations plus query time only.",
        "- `direct_static_early_time`: static descriptors, early observations, and query time.",
        "- `middle_early_residual_family`: script 96 deployable middle-layer selector.",
        "- `middle_future_oracle_family`: diagnostic upper bound, not deployable.",
        "",
        "## Run Metadata",
        "",
        f"- model: `{args.model}`",
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
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    direct = run_direct_controls(args)
    middle = load_middle_metrics(args)
    common_keys = ["split_kind", "fold", "budget", "unified_curve_id"]
    direct_keys = set(map(tuple, direct[common_keys].astype(str).to_numpy())) if not direct.empty else set()
    middle = middle[middle[common_keys].astype(str).apply(tuple, axis=1).isin(direct_keys)].copy()
    combined = pd.concat([direct, middle], ignore_index=True, sort=False)
    summary = summarize(combined)
    decisions = build_decision_table(summary)

    direct.to_csv(args.out / "direct_per_curve_metrics.csv", index=False)
    combined.to_csv(args.out / "combined_per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_method.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks = [
        finite_check("direct_per_curve_metrics", direct),
        finite_check("combined_per_curve_metrics", combined),
        finite_check("summary_by_method", summary),
        finite_check("decision_table", decisions),
    ]
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "seed": args.seed,
                "pool_dir": str(args.pool_dir),
                "middle_run": str(args.middle_run),
                "budgets": args.budgets,
                "model": args.model,
                "n_estimators": args.n_estimators,
                "max_curves": args.max_curves,
                "checks": checks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.out, summary, decisions, checks, args)
    print((args.out / "report.md").resolve())
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
