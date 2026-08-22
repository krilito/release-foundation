"""91 - PLGA static-feature information ceiling probe.

Purpose:
    Test how far PLGA release curves can be predicted from static, pre-release
    descriptors only. No early Q(t), heldout curve fit, or test target-derived
    quantity is used as an input.

Design:
    static features -> predicted shape-family parameters -> Q(t)

    Training may use fitted shape parameters from training curves as supervised
    labels. Evaluation is always against raw heldout release observations.

Consumes:
    outputs/148_release_main_cumulative_v1/curves_long.csv
    outputs/148_release_main_cumulative_v1/formulations.csv
    outputs/158_freeze_theta_targets/all_family_fits.csv

Produces:
    outputs/91_plga_static_feature_ceiling_probe/
      per_curve_metrics.csv
      summary_by_config.csv
      best_by_split.csv
      feature_coverage.csv
      report.md
      lock_metadata.json
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_baseline_utils import FAMILY_FUNCS, safe_r2  # noqa: E402


DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_THETA_DIR = Path("outputs/158_freeze_theta_targets")
DEFAULT_OUT = Path("outputs/91_plga_static_feature_ceiling_probe")
PLGA_SOURCES = {"internal181", "cross321"}
FAMILIES = ("weibull", "biexponential", "hill")

FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "intercept_only": {"numeric": [], "categorical": []},
    "polymer_core": {
        "numeric": ["Polymer_MW", "LA/GA", "CL Ratio"],
        "categorical": ["polymer_family"],
    },
    "formulation_core": {
        "numeric": ["Initial D/M ratio", "DLC", "DLC_percent", "EE", "Particle_Size", "SA-V", "SE"],
        "categorical": [],
    },
    "drug_physchem": {
        "numeric": ["Drug_Mw", "Drug_TPSA", "Drug_NHA", "Drug_LogP", "Drug_Pka", "Drug_Tm"],
        "categorical": ["payload_name"],
    },
    "intrinsic_no_source": {
        "numeric": [
            "Polymer_MW",
            "LA/GA",
            "CL Ratio",
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
            "Drug_Pka",
            "Drug_Tm",
        ],
        "categorical": ["polymer_family", "payload_name"],
    },
    "all_measured_no_source": {
        "numeric": [
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
        ],
        "categorical": [
            "polymer_family",
            "payload_name",
            "release_medium_condition",
            "release_method",
            "measurement_assay",
            "structure_type",
            "light_condition",
        ],
    },
    "source_diagnostic": {
        "numeric": [],
        "categorical": ["source_dataset", "source_group", "polymer_family"],
    },
    "all_with_source_diagnostic": {
        "numeric": [
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
        ],
        "categorical": [
            "source_dataset",
            "source_group",
            "polymer_family",
            "payload_name",
            "release_medium_condition",
            "release_method",
            "measurement_assay",
            "structure_type",
            "light_condition",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLGA static-feature release prediction ceiling probe.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--theta-dir", type=Path, default=DEFAULT_THETA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=["random_kfold", "source_group_kfold", "source_dataset_lodo", "all"], default="random_kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--models", default="extra_trees,ridge,knn")
    parser.add_argument("--families", default="weibull,biexponential,hill")
    parser.add_argument("--feature-groups", default=",".join(FEATURE_GROUPS))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-curves", type=int, default=0)
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
    return text if text else "__MISSING__"


def parse_list(text: str, allowed: set[str], label: str) -> list[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={sorted(allowed)}")
    return values


def load_plga_tables(pool_dir: Path, theta_dir: Path, max_curves: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    forms = pd.read_csv(pool_dir / "formulations.csv")
    curves = pd.read_csv(pool_dir / "curves_long.csv")
    fits = pd.read_csv(theta_dir / "all_family_fits.csv")

    forms = forms[forms["source_dataset"].astype(str).isin(PLGA_SOURCES)].drop_duplicates("unified_curve_id").copy()
    forms["unified_curve_id"] = forms["unified_curve_id"].astype(str)
    if max_curves and len(forms) > max_curves:
        forms = forms.sample(n=max_curves, random_state=seed).sort_values("unified_curve_id").copy()

    ids = set(forms["unified_curve_id"])
    curves = curves[curves["unified_curve_id"].astype(str).isin(ids)].copy()
    curves["unified_curve_id"] = curves["unified_curve_id"].astype(str)
    curves["time_days"] = pd.to_numeric(curves["time_days"], errors="coerce")
    curves["release_fraction"] = pd.to_numeric(curves["release_fraction"], errors="coerce")
    curves = curves.dropna(subset=["time_days", "release_fraction"]).sort_values(["unified_curve_id", "time_days"]).copy()
    curve_groups = {
        cid: sub.groupby("time_days", as_index=False)["release_fraction"].mean().sort_values("time_days")
        for cid, sub in curves.groupby("unified_curve_id", sort=True)
    }
    keep_ids = {cid for cid, sub in curve_groups.items() if len(sub) >= 3}
    forms = forms[forms["unified_curve_id"].isin(keep_ids)].copy()
    fits = fits[fits["unified_curve_id"].astype(str).isin(set(forms["unified_curve_id"]))].copy()
    fits["unified_curve_id"] = fits["unified_curve_id"].astype(str)
    return forms.reset_index(drop=True), fits, curve_groups


def feature_coverage(forms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, spec in FEATURE_GROUPS.items():
        for kind in ("numeric", "categorical"):
            for col in spec[kind]:
                if col not in forms.columns:
                    rows.append({"feature_group": group, "kind": kind, "column": col, "present": False, "nonmissing_fraction": 0.0, "n_unique": 0})
                    continue
                series = forms[col]
                rows.append(
                    {
                        "feature_group": group,
                        "kind": kind,
                        "column": col,
                        "present": True,
                        "nonmissing_fraction": float(series.notna().mean()),
                        "n_unique": int(series.nunique(dropna=True)),
                    }
                )
    return pd.DataFrame(rows)


def build_design(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric_cols = [col for col in numeric_cols if col in train.columns]
    categorical_cols = [col for col in categorical_cols if col in train.columns]
    train_parts = [pd.DataFrame({"bias": np.ones(len(train), dtype=float)}, index=train.index)]
    test_parts = [pd.DataFrame({"bias": np.ones(len(test), dtype=float)}, index=test.index)]

    if numeric_cols:
        tr_num = train[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        te_num = test[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = tr_num.median(axis=0).fillna(0.0)
        missing_tr = tr_num.isna().astype(float).rename(columns={col: f"{col}__missing" for col in numeric_cols})
        missing_te = te_num.isna().astype(float).rename(columns={col: f"{col}__missing" for col in numeric_cols})
        tr_num = tr_num.fillna(med)
        te_num = te_num.fillna(med)
        mean = tr_num.mean(axis=0)
        std = tr_num.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        train_parts.extend([(tr_num - mean) / std, missing_tr])
        test_parts.extend([(te_num - mean) / std, missing_te])

    if categorical_cols:
        tr_cat = train[categorical_cols].map(clean_category)
        te_cat = test[categorical_cols].map(clean_category)
        for col in categorical_cols:
            known = set(tr_cat[col].unique())
            te_cat[col] = te_cat[col].where(te_cat[col].isin(known), "__UNK__")
        tr_cat["__row_marker__"] = "train"
        te_cat["__row_marker__"] = "test"
        combined = pd.concat([tr_cat, te_cat], axis=0)
        dummy = pd.get_dummies(combined.drop(columns="__row_marker__"), prefix=categorical_cols, dtype=float)
        train_parts.append(dummy.iloc[: len(train)].set_index(train.index))
        test_parts.append(dummy.iloc[len(train) :].set_index(test.index))

    train_design = pd.concat(train_parts, axis=1)
    test_design = pd.concat(test_parts, axis=1).reindex(columns=train_design.columns, fill_value=0.0)
    return train_design.to_numpy(dtype=float), test_design.to_numpy(dtype=float), train_design.columns.tolist()


def transform_params(family: str, frame: pd.DataFrame) -> np.ndarray:
    p1 = np.clip(frame["param_1"].to_numpy(dtype=float), 1e-8, None)
    p2 = np.clip(frame["param_2"].to_numpy(dtype=float), 1e-8, None)
    p3 = np.clip(frame["param_3"].to_numpy(dtype=float), 1e-8, None)
    if family in {"weibull", "hill"}:
        return np.column_stack([np.log(p1), np.log(p2), np.log(p3)])
    if family == "biexponential":
        p4 = np.clip(frame["param_4"].to_numpy(dtype=float), 1e-8, None)
        w = np.clip(p2, 1e-6, 1.0 - 1e-6)
        return np.column_stack([np.log(p1), np.log(w / (1.0 - w)), np.log(p3), np.log(p4)])
    raise KeyError(family)


def inverse_params(family: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if family in {"weibull", "hill"}:
        out = np.exp(arr[:, :3])
        out[:, 0] = np.clip(out[:, 0], 0.0, 2.0)
        out[:, 1] = np.clip(out[:, 1], 1e-4, 1e3)
        out[:, 2] = np.clip(out[:, 2], 0.05, 10.0)
        return out
    if family == "biexponential":
        qmax = np.clip(np.exp(arr[:, 0]), 0.0, 2.0)
        w = 1.0 / (1.0 + np.exp(-arr[:, 1]))
        k_fast = np.clip(np.exp(arr[:, 2]), 1e-6, 100.0)
        k_slow = np.clip(np.exp(arr[:, 3]), 1e-6, 100.0)
        return np.column_stack([qmax, w, k_fast, k_slow])
    raise KeyError(family)


def make_model(name: str, seed: int, n_estimators: int):
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 13))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=-1,
        )
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    raise KeyError(name)


def curve_metrics(curve: pd.DataFrame, family: str, params: np.ndarray) -> dict[str, float]:
    t = curve["time_days"].to_numpy(dtype=float)
    y = curve["release_fraction"].to_numpy(dtype=float)
    pred = np.clip(FAMILY_FUNCS[family](t, params), 0.0, 1.2)
    err = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": safe_r2(y, pred),
        "n_points": int(len(y)),
        "duration_days": float(np.max(t) - np.min(t)) if len(t) else 0.0,
        "release_span": float(np.max(y) - np.min(y)) if len(y) else 0.0,
    }


def build_splits(forms: pd.DataFrame, split_mode: str, n_splits: int, seed: int) -> list[dict[str, Any]]:
    ids = forms["unified_curve_id"].astype(str).to_numpy()
    splits: list[dict[str, Any]] = []
    if split_mode in {"random_kfold", "all"}:
        kf = KFold(n_splits=min(n_splits, len(forms)), shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(ids)):
            splits.append({"split_kind": "random-kfold", "fold": fold, "train_ids": set(ids[train_idx]), "test_ids": set(ids[test_idx])})
    if split_mode in {"source_group_kfold", "all"}:
        groups = forms["source_group"].map(clean_category)
        groups = groups.where(groups != "__MISSING__", forms["source_dataset"].map(clean_category))
        n_group_splits = min(n_splits, int(groups.nunique()))
        gkf = GroupKFold(n_splits=n_group_splits)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(ids, groups=groups)):
            splits.append({"split_kind": "source-group-kfold", "fold": fold, "train_ids": set(ids[train_idx]), "test_ids": set(ids[test_idx])})
    if split_mode in {"source_dataset_lodo", "all"}:
        for fold, source in enumerate(sorted(forms["source_dataset"].astype(str).unique())):
            test_ids = set(forms.loc[forms["source_dataset"].astype(str) == source, "unified_curve_id"].astype(str))
            train_ids = set(ids) - test_ids
            splits.append({"split_kind": "source-dataset-lodo", "fold": fold, "heldout_source_dataset": source, "train_ids": train_ids, "test_ids": test_ids})
    return splits


def evaluate_config(
    *,
    forms: pd.DataFrame,
    fits: pd.DataFrame,
    curve_groups: dict[str, pd.DataFrame],
    split: dict[str, Any],
    feature_group: str,
    family: str,
    model_name: str,
    seed: int,
    n_estimators: int,
) -> list[dict[str, Any]]:
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    train_forms = forms[forms["unified_curve_id"].isin(train_ids)].copy()
    test_forms = forms[forms["unified_curve_id"].isin(test_ids)].copy()
    train_fit = fits[(fits["family"] == family) & (fits["unified_curve_id"].isin(train_ids))].copy()
    train_forms = train_forms.merge(train_fit[["unified_curve_id", "param_1", "param_2", "param_3", "param_4"]], on="unified_curve_id", how="inner")
    test_forms = test_forms[test_forms["unified_curve_id"].isin(curve_groups)].copy()
    if len(train_forms) < 5 or test_forms.empty:
        return []

    spec = FEATURE_GROUPS[feature_group]
    x_train, x_test, design_cols = build_design(train_forms, test_forms, spec["numeric"], spec["categorical"])
    y_train = transform_params(family, train_forms)

    if feature_group == "intercept_only" or model_name == "train_param_median":
        pred_params = np.repeat(np.median(y_train, axis=0, keepdims=True), len(test_forms), axis=0)
    else:
        model = make_model(model_name, seed + int(split["fold"]), n_estimators)
        model.fit(x_train, y_train)
        pred_params = model.predict(x_test)
    pred_params = inverse_params(family, pred_params)

    rows = []
    for idx, (_, row) in enumerate(test_forms.iterrows()):
        curve_id = str(row["unified_curve_id"])
        metrics = curve_metrics(curve_groups[curve_id], family, pred_params[idx])
        rows.append(
            {
                "split_kind": split["split_kind"],
                "fold": int(split["fold"]),
                "heldout_source_dataset": split.get("heldout_source_dataset", "none"),
                "feature_group": feature_group,
                "family": family,
                "model": "train_param_median" if feature_group == "intercept_only" else model_name,
                "unified_curve_id": curve_id,
                "source_dataset": clean_category(row.get("source_dataset", "")),
                "source_group": clean_category(row.get("source_group", "")),
                "n_train": int(len(train_forms)),
                "n_test": int(len(test_forms)),
                "n_design_cols": int(len(design_cols)),
                **metrics,
            }
        )
    return rows


def summarize(per_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["split_kind", "feature_group", "family", "model"]
    summary = (
        per_curve.groupby(group_cols, dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            n_folds=("fold", "nunique"),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_mae=("mae", "median"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            median_release_span=("release_span", "median"),
            median_duration_days=("duration_days", "median"),
        )
        .reset_index()
        .sort_values(["split_kind", "median_rmse", "feature_group", "family", "model"])
    )
    best = summary.sort_values(["split_kind", "median_rmse"]).groupby("split_kind", as_index=False).head(10)
    return summary, best


def write_report(out: Path, summary: pd.DataFrame, best: pd.DataFrame, coverage: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# PLGA Static-Feature Ceiling Probe",
        "",
        "Static-only probe: no early release observations and no heldout curve-derived parameters are used as inputs.",
        "",
        "## Best Configurations By Split",
        "",
        best.to_markdown(index=False),
        "",
        "## Full Summary",
        "",
        summary.head(80).to_markdown(index=False),
        "",
        "## Low-Coverage Feature Columns",
        "",
        coverage[(coverage["present"]) & (coverage["nonmissing_fraction"] < 0.5)].head(40).to_markdown(index=False),
        "",
        "## Run Settings",
        "",
        f"- split_mode: `{args.split_mode}`",
        f"- n_splits: `{args.n_splits}`",
        f"- models: `{args.models}`",
        f"- families: `{args.families}`",
        f"- feature_groups: `{args.feature_groups}`",
        "",
        "Interpretation rule: random-kfold is an upper-bound descriptor probe. Source-group and source-dataset splits are stricter tests of whether the descriptor bridge survives source shift.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def finite_numeric_check(df: pd.DataFrame, name: str) -> None:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        n = int(vals.isna().sum())
        if n:
            bad[col] = n
    if bad:
        raise RuntimeError(f"NaN/inf detected in {name}: {bad}")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    models = parse_list(args.models, {"extra_trees", "ridge", "knn"}, "models")
    families = parse_list(args.families, set(FAMILIES), "families")
    feature_groups = parse_list(args.feature_groups, set(FEATURE_GROUPS), "feature_groups")

    forms, fits, curve_groups = load_plga_tables(args.pool_dir, args.theta_dir, args.max_curves, args.seed)
    coverage = feature_coverage(forms)
    splits = build_splits(forms, args.split_mode, args.n_splits, args.seed)

    rows: list[dict[str, Any]] = []
    total = len(splits) * len(feature_groups) * len(families) * len(models)
    done = 0
    for split in splits:
        for feature_group in feature_groups:
            for family in families:
                for model_name in models:
                    if feature_group == "intercept_only" and model_name != models[0]:
                        continue
                    rows.extend(
                        evaluate_config(
                            forms=forms,
                            fits=fits,
                            curve_groups=curve_groups,
                            split=split,
                            feature_group=feature_group,
                            family=family,
                            model_name=model_name,
                            seed=args.seed,
                            n_estimators=args.n_estimators,
                        )
                    )
                    done += 1
                    if done % 20 == 0:
                        print(f"[91] configs {done}/{total} rows={len(rows)}")

    per_curve = pd.DataFrame(rows)
    if per_curve.empty:
        raise RuntimeError("No rows produced")
    summary, best = summarize(per_curve)
    finite_numeric_check(per_curve, "per_curve_metrics")
    finite_numeric_check(summary, "summary_by_config")

    per_curve.to_csv(args.out / "per_curve_metrics.csv", index=False)
    summary.to_csv(args.out / "summary_by_config.csv", index=False)
    best.to_csv(args.out / "best_by_split.csv", index=False)
    coverage.to_csv(args.out / "feature_coverage.csv", index=False)
    metadata = {
        "script": "scripts/91_plga_static_feature_ceiling_probe.py",
        "git_hash": git_hash(),
        "args": args_to_metadata(args),
        "pool_dir": str(args.pool_dir),
        "theta_dir": str(args.theta_dir),
        "out": str(args.out),
        "seed": int(args.seed),
        "split_mode": str(args.split_mode),
        "n_splits": int(args.n_splits),
        "models": models,
        "families": families,
        "feature_groups": feature_groups,
        "n_curves": int(forms["unified_curve_id"].nunique()),
        "note": "Static-only descriptor ceiling probe. Random-kfold is upper bound; source-shift splits are stricter.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out, summary, best, coverage, args)
    print("[91] wrote", args.out)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
