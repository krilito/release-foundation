"""96 - PLGA early-conditioned middle-layer selector probe.

Purpose:
    Test whether early release observations improve the middle-layer route:

        static descriptors + first k observed points -> shape theta -> Q_future

    and whether the same early points can select a better shape family by
    comparing early residuals across family-specific theta mappers.

Consumes:
    outputs/148_release_main_cumulative_v1/curves_long.csv
    outputs/148_release_main_cumulative_v1/formulations.csv
    outputs/158_freeze_theta_targets/all_family_fits.csv
    outputs/91_plga_static_feature_ceiling_*/per_curve_metrics.csv

Produces:
    outputs/96_plga_early_middle_layer_selector_probe/
      per_curve_family_metrics.csv
      per_curve_selector_metrics.csv
      summary_by_budget.csv
      lock_metadata.json
      report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_baseline_utils import FAMILY_BOUNDS, FAMILY_FUNCS, safe_r2  # noqa: E402


DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_THETA_DIR = Path("outputs/158_freeze_theta_targets")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/96_plga_early_middle_layer_selector_probe")

PLGA_SOURCES = {"internal181", "cross321"}
FAMILIES = ("weibull", "biexponential", "hill")

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
    parser = argparse.ArgumentParser(description="Probe early-conditioned PLGA middle-layer routing.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--theta-dir", type=Path, default=DEFAULT_THETA_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", default="0,1,2,3,5")
    parser.add_argument("--model", choices=["extra_trees", "random_forest", "ridge"], default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=250)
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


def load_plga_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
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

    fits = read_csv(args.theta_dir / "all_family_fits.csv")
    fits["unified_curve_id"] = fits["unified_curve_id"].astype(str)
    fits = fits[fits["unified_curve_id"].isin(set(forms["unified_curve_id"]))].copy()
    return forms.reset_index(drop=True), curve_groups, fits.reset_index(drop=True)


def transform_params(family: str, params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if family == "biexponential":
        qmax = np.log(np.clip(params[:, 0], 1e-8, None))
        w = np.clip(params[:, 1], 1e-6, 1.0 - 1e-6)
        logit_w = np.log(w / (1.0 - w))
        k_fast = np.log(np.clip(params[:, 2], 1e-8, None))
        k_slow = np.log(np.clip(params[:, 3], 1e-8, None))
        return np.column_stack([qmax, logit_w, k_fast, k_slow])
    qmax = np.log(np.clip(params[:, 0], 1e-8, None))
    scale = np.log(np.clip(params[:, 1], 1e-8, None))
    shape = np.log(np.clip(params[:, 2], 1e-8, None))
    return np.column_stack([qmax, scale, shape])


def inverse_params(family: str, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if family == "biexponential":
        params = np.column_stack(
            [
                np.exp(target[:, 0]),
                1.0 / (1.0 + np.exp(-np.clip(target[:, 1], -40.0, 40.0))),
                np.exp(target[:, 2]),
                np.exp(target[:, 3]),
            ]
        )
    else:
        params = np.column_stack([np.exp(target[:, 0]), np.exp(target[:, 1]), np.exp(target[:, 2])])
    lo, hi = FAMILY_BOUNDS[family]
    return np.clip(params, lo, hi)


def params_for_family(fits: pd.DataFrame, family: str) -> pd.DataFrame:
    n_params = 4 if family == "biexponential" else 3
    cols = [f"param_{i}" for i in range(1, n_params + 1)]
    sub = fits[fits["family"].astype(str) == family][["unified_curve_id", "rmse", "r2"] + cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=cols).copy()
    sub["params"] = sub[cols].apply(lambda r: np.asarray([float(v) for v in r], dtype=float), axis=1)
    return sub


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
    return x_train, x_test


def early_features(curve_groups: dict[str, pd.DataFrame], curve_ids: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    if k == 0:
        return np.zeros((len(curve_ids), 0), dtype=float), np.zeros(len(curve_ids), dtype=float)
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


def metric_on_future(curve: pd.DataFrame, family: str, params: np.ndarray, last_context_t: float) -> dict[str, float]:
    t = curve["time_days"].to_numpy(dtype=float)
    y = curve["release_fraction"].to_numpy(dtype=float)
    if np.isfinite(last_context_t):
        mask = t > last_context_t + 1e-12
    else:
        mask = np.ones_like(t, dtype=bool)
    if int(mask.sum()) < 2:
        return {"future_rmse": np.nan, "future_mae": np.nan, "future_r2": np.nan, "n_future": int(mask.sum())}
    pred = FAMILY_FUNCS[family](t[mask], np.asarray(params, dtype=float))
    return {
        "future_rmse": float(np.sqrt(np.mean((pred - y[mask]) ** 2))),
        "future_mae": float(np.mean(np.abs(pred - y[mask]))),
        "future_r2": safe_r2(y[mask], pred),
        "n_future": int(mask.sum()),
    }


def early_residual(curve: pd.DataFrame, family: str, params: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    early = curve.iloc[:k]
    t = early["time_days"].to_numpy(dtype=float)
    y = early["release_fraction"].to_numpy(dtype=float)
    pred = FAMILY_FUNCS[family](t, np.asarray(params, dtype=float))
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def finite_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    allowed: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].to_numpy(dtype=float)
        nonfinite = ~np.isfinite(vals)
        if not bool(nonfinite.any()):
            continue
        allowed_mask = np.zeros(len(df), dtype=bool)
        if "budget" in df.columns:
            budget0 = pd.to_numeric(df["budget"], errors="coerce").fillna(-1).to_numpy(dtype=float) == 0
            if col in {"last_context_time", "early_residual", "selected_early_residual", "median_context_time"}:
                allowed_mask |= budget0
        if col == "future_r2":
            allowed_mask |= nonfinite
        if col in {"future_rmse", "future_mae"} and "n_future" in df.columns:
            n_future = pd.to_numeric(df["n_future"], errors="coerce").fillna(0).to_numpy(dtype=float)
            allowed_mask |= n_future < 2
        n_allowed = int((nonfinite & allowed_mask).sum())
        n_bad = int((nonfinite & ~allowed_mask).sum())
        if n_allowed:
            allowed[col] = n_allowed
        if n_bad:
            bad[col] = n_bad
    return {"name": name, "rows": int(len(df)), "unexpected_nonfinite": bad, "allowed_nonfinite": allowed}


def run_probe(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forms, curve_groups, fits = load_plga_data(args)
    splits = load_split_assignments(args)
    forms = forms[forms["unified_curve_id"].isin(set(splits["unified_curve_id"]))].copy()
    forms = forms[forms["unified_curve_id"].isin(set(curve_groups))].copy()
    split_ids = set(forms["unified_curve_id"])
    splits = splits[splits["unified_curve_id"].isin(split_ids)].copy()

    budgets = [int(x) for x in str(args.budgets).split(",") if str(x).strip()]
    family_fits = {family: params_for_family(fits, family) for family in FAMILIES}
    forms_indexed = forms.set_index("unified_curve_id", drop=False)

    family_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []

    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        split_df = splits[splits["split_kind"] == split_kind].copy()
        if split_df.empty:
            continue
        for fold in sorted(split_df["fold"].unique()):
            test_ids = sorted(split_df.loc[split_df["fold"] == fold, "unified_curve_id"].astype(str).unique())
            train_ids = sorted(set(split_df["unified_curve_id"].astype(str).unique()) - set(test_ids))
            train_forms_base = forms_indexed.loc[[cid for cid in train_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            test_forms_base = forms_indexed.loc[[cid for cid in test_ids if cid in forms_indexed.index]].copy().reset_index(drop=True)
            if len(train_forms_base) < 20 or test_forms_base.empty:
                continue

            for budget in budgets:
                train_early, _ = early_features(curve_groups, train_forms_base["unified_curve_id"].tolist(), budget)
                test_early, last_t = early_features(curve_groups, test_forms_base["unified_curve_id"].tolist(), budget)
                valid_train = np.isfinite(train_early).all(axis=1) if budget > 0 else np.ones(len(train_forms_base), dtype=bool)
                valid_test = np.isfinite(test_early).all(axis=1) if budget > 0 else np.ones(len(test_forms_base), dtype=bool)
                train_forms = train_forms_base.loc[valid_train].copy()
                test_forms = test_forms_base.loc[valid_test].copy()
                train_early_valid = train_early[valid_train]
                test_early_valid = test_early[valid_test]
                last_t_valid = last_t[valid_test] if budget > 0 else np.full(len(test_forms), np.nan)
                if len(train_forms) < 20 or test_forms.empty:
                    continue

                x_train_static, x_test_static = build_static_design(train_forms, test_forms)
                if budget > 0:
                    early_cols = [f"early_{j}" for j in range(train_early_valid.shape[1])]
                    x_train = pd.concat(
                        [x_train_static.reset_index(drop=True), pd.DataFrame(train_early_valid, columns=early_cols)],
                        axis=1,
                    )
                    x_test = pd.concat(
                        [x_test_static.reset_index(drop=True), pd.DataFrame(test_early_valid, columns=early_cols)],
                        axis=1,
                    )
                else:
                    x_train = x_train_static.reset_index(drop=True)
                    x_test = x_test_static.reset_index(drop=True)

                fold_predictions: dict[str, dict[str, dict[str, Any]]] = {cid: {} for cid in test_forms["unified_curve_id"].astype(str)}
                train_family_fit_rmse: dict[str, float] = {}

                for family in FAMILIES:
                    ff = family_fits[family]
                    train_target = train_forms[["unified_curve_id"]].merge(ff, on="unified_curve_id", how="inner")
                    if len(train_target) < 20:
                        continue
                    train_family_fit_rmse[family] = float(pd.to_numeric(train_target["rmse"], errors="coerce").median())
                    id_to_pos = {cid: i for i, cid in enumerate(train_forms["unified_curve_id"].astype(str))}
                    train_pos = [id_to_pos[cid] for cid in train_target["unified_curve_id"].astype(str)]
                    params_train = np.vstack(train_target["params"].to_list())
                    y_train = transform_params(family, params_train)
                    model = make_model(args.model, args.seed + int(fold) * 17 + budget * 101 + len(family), args.n_estimators)
                    model.fit(x_train.iloc[train_pos].to_numpy(dtype=float), y_train)
                    y_pred = np.asarray(model.predict(x_test.to_numpy(dtype=float)), dtype=float)
                    params_pred = inverse_params(family, y_pred)

                    for i, cid in enumerate(test_forms["unified_curve_id"].astype(str)):
                        curve = curve_groups[cid]
                        metrics = metric_on_future(curve, family, params_pred[i], float(last_t_valid[i]))
                        er = early_residual(curve, family, params_pred[i], budget)
                        row = {
                            "split_kind": split_kind,
                            "fold": int(fold),
                            "budget": int(budget),
                            "model": args.model,
                            "family": family,
                            "unified_curve_id": cid,
                            "source_dataset": str(test_forms.iloc[i].get("source_dataset", "")),
                            "source_group": clean_category(test_forms.iloc[i].get("source_group", "")),
                            "last_context_time": float(last_t_valid[i]) if np.isfinite(last_t_valid[i]) else np.nan,
                            "early_residual": er,
                            **metrics,
                        }
                        family_rows.append(row)
                        fold_predictions[cid][family] = {"row": row, "params": params_pred[i]}

                if not train_family_fit_rmse:
                    continue
                train_global_family = min(train_family_fit_rmse, key=train_family_fit_rmse.get)
                for cid, preds in fold_predictions.items():
                    if not preds:
                        continue
                    available = [fam for fam, item in preds.items() if np.isfinite(item["row"]["future_rmse"])]
                    if not available:
                        continue
                    if budget > 0:
                        early_available = [
                            fam for fam in available if np.isfinite(preds[fam]["row"]["early_residual"])
                        ]
                        residual_family = min(early_available, key=lambda fam: preds[fam]["row"]["early_residual"]) if early_available else train_global_family
                    else:
                        residual_family = train_global_family if train_global_family in available else available[0]
                    future_oracle_family = min(available, key=lambda fam: preds[fam]["row"]["future_rmse"])
                    for selector, family in [
                        ("train_global_family", residual_family if budget == 0 else train_global_family),
                        ("early_residual_family", residual_family),
                        ("future_oracle_family", future_oracle_family),
                    ]:
                        if family not in preds:
                            continue
                        base = preds[family]["row"].copy()
                        selector_rows.append(
                            {
                                "split_kind": base["split_kind"],
                                "fold": base["fold"],
                                "budget": base["budget"],
                                "model": base["model"],
                                "selector": selector,
                                "selected_family": family,
                                "unified_curve_id": cid,
                                "source_dataset": base["source_dataset"],
                                "source_group": base["source_group"],
                                "last_context_time": base["last_context_time"],
                                "selected_early_residual": base["early_residual"],
                                "future_rmse": base["future_rmse"],
                                "future_mae": base["future_mae"],
                                "future_r2": base["future_r2"],
                                "n_future": base["n_future"],
                                "oracle_family": future_oracle_family,
                                "hit_oracle_family": bool(family == future_oracle_family),
                            }
                        )

    family_df = pd.DataFrame(family_rows)
    selector_df = pd.DataFrame(selector_rows)
    summary = (
        selector_df.groupby(["split_kind", "budget", "selector"], dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_future_rmse=("future_rmse", "median"),
            mean_future_rmse=("future_rmse", "mean"),
            median_future_mae=("future_mae", "median"),
            median_future_r2=("future_r2", "median"),
            frac_r2_positive=("future_r2", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.0))),
            median_context_time=("last_context_time", "median"),
            oracle_family_hit_rate=("hit_oracle_family", "mean"),
        )
        .reset_index()
        .sort_values(["split_kind", "budget", "selector"])
    )
    return family_df, selector_df, summary


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_kind in ["random-kfold", "source-group-kfold", "source-dataset-lodo"]:
        sub = summary[summary["split_kind"] == split_kind]
        if sub.empty:
            continue
        zero = sub[(sub["budget"] == 0) & (sub["selector"] == "train_global_family")]
        best_early = sub[(sub["budget"] > 0) & (sub["selector"] == "early_residual_family")].sort_values("median_future_rmse")
        oracle = sub[(sub["budget"] > 0) & (sub["selector"] == "future_oracle_family")].sort_values("median_future_rmse")
        if zero.empty or best_early.empty:
            continue
        z = float(zero["median_future_rmse"].iloc[0])
        b = float(best_early["median_future_rmse"].iloc[0])
        budget = int(best_early["budget"].iloc[0])
        if not oracle.empty:
            o = float(oracle["median_future_rmse"].iloc[0])
            oracle_text = f"; best future-oracle RMSE={o:.3f}"
        else:
            oracle_text = ""
        answer = "yes, early middle-layer improves" if b < z else "no, early selector did not improve"
        rows.append(
            {
                "split_kind": split_kind,
                "question": "Does early-conditioned middle-layer routing beat static middle-layer mapping?",
                "answer": answer,
                "evidence": f"k=0 RMSE={z:.3f}; best early-residual RMSE={b:.3f} at k={budget}{oracle_text}",
            }
        )
    return pd.DataFrame(rows)


def write_report(out: Path, summary: pd.DataFrame, decisions: pd.DataFrame, checks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    comparison = summary[
        summary["selector"].isin(["train_global_family", "early_residual_family", "future_oracle_family"])
    ].copy()
    lines = [
        "# PLGA Early Middle-Layer Selector Probe",
        "",
        "This probe is deployable except for `future_oracle_family`, which is a diagnostic upper bound.",
        "For budget `k`, metrics are computed only on points strictly after the kth observed point.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "_No decisions generated._",
        "",
        "## Summary By Budget",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interpretation Guard",
        "",
        "- `train_global_family`: static descriptors plus the train-best shape family; no early residual selector.",
        "- `early_residual_family`: uses early points to choose among family-specific theta mappers.",
        "- `future_oracle_family`: sees heldout future error and is not deployable.",
        "- A real claim requires comparison against direct early-Q black-box routes on the same future-only metric.",
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
    family_df, selector_df, summary = run_probe(args)
    decisions = build_decision_table(summary)

    family_df.to_csv(args.out / "per_curve_family_metrics.csv", index=False)
    selector_df.to_csv(args.out / "per_curve_selector_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_budget.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    checks = [
        finite_check("per_curve_family_metrics", family_df),
        finite_check("per_curve_selector_metrics", selector_df),
        finite_check("summary_by_budget", summary),
        finite_check("decision_table", decisions),
    ]
    (args.out / "lock_metadata.json").write_text(
        json.dumps(
            {
                "git_hash": git_hash(),
                "args": args_to_metadata(args),
                "seed": args.seed,
                "pool_dir": str(args.pool_dir),
                "theta_dir": str(args.theta_dir),
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
