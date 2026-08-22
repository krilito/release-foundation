"""95 - PLGA static combo gate screening.

Purpose:
    First executable test of conditional feature relevance:

        static descriptors -> predicted expert errors -> selected combo -> Q(t)

This is a screening experiment, not the final nested gate. It uses the existing
script 91 out-of-fold expert errors as meta-training labels. That is enough to
test whether a static gate has signal, but the final claim still needs a fully
nested implementation.

Consumes:
    outputs/91_plga_static_feature_ceiling_*/per_curve_metrics.csv
    outputs/148_release_main_cumulative_v1/formulations.csv

Produces:
    outputs/95_plga_static_combo_gate_screen/
      gated_per_curve.csv
      gated_summary.csv
      selection_frequency.csv
      decision_table.csv
      report.md
      lock_metadata.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor


DEFAULT_POOL_DIR = Path("outputs/148_release_main_cumulative_v1")
DEFAULT_RANDOM = Path("outputs/91_plga_static_feature_ceiling_random_kfold")
DEFAULT_GROUP = Path("outputs/91_plga_static_feature_ceiling_source_group_kfold")
DEFAULT_SOURCE = Path("outputs/91_plga_static_feature_ceiling_source_dataset_lodo")
DEFAULT_OUT = Path("outputs/95_plga_static_combo_gate_screen")
PLGA_SOURCES = {"internal181", "cross321"}
SOURCE_FEATURE_GROUPS = {"source_diagnostic", "all_with_source_diagnostic"}

GATE_NUMERIC_COLS = [
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
GATE_CATEGORICAL_COLS = [
    "polymer_family",
    "payload_name",
    "release_medium_condition",
    "release_method",
    "measurement_assay",
    "structure_type",
    "light_condition",
]
SOURCE_CATEGORICAL_COLS = ["source_dataset", "source_group"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen static combo gate for PLGA.")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--random-run", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--source-group-run", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--source-dataset-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gate-models", default="extra_trees,ridge,knn")
    parser.add_argument("--include-source-gate", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_plga_forms(pool_dir: Path) -> pd.DataFrame:
    forms = read_csv(pool_dir / "formulations.csv")
    forms = forms[forms["source_dataset"].astype(str).isin(PLGA_SOURCES)].drop_duplicates("unified_curve_id").copy()
    forms["unified_curve_id"] = forms["unified_curve_id"].astype(str)
    return forms


def build_design(train: pd.DataFrame, test: pd.DataFrame, include_source: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric_cols = [col for col in GATE_NUMERIC_COLS if col in train.columns]
    categorical_cols = [col for col in GATE_CATEGORICAL_COLS if col in train.columns]
    if include_source:
        categorical_cols += [col for col in SOURCE_CATEGORICAL_COLS if col in train.columns]

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

    x_train_df = pd.concat(train_parts, axis=1)
    x_test_df = pd.concat(test_parts, axis=1).reindex(columns=x_train_df.columns, fill_value=0.0)
    return x_train_df.to_numpy(dtype=float), x_test_df.to_numpy(dtype=float), x_train_df.columns.tolist()


def load_expert_rows(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for label, run in [
        ("random_kfold", args.random_run),
        ("source_group_kfold", args.source_group_run),
        ("source_dataset_lodo", args.source_dataset_run),
    ]:
        df = read_csv(run / "per_curve_metrics.csv").copy()
        df.insert(0, "run_label", label)
        df.insert(1, "run_dir", str(run))
        df["config_id"] = df["feature_group"].astype(str) + "|" + df["family"].astype(str) + "|" + df["model"].astype(str)
        df["uses_source_diagnostic"] = df["feature_group"].astype(str).isin(SOURCE_FEATURE_GROUPS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_gate(name: str, seed: int, n_estimators: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 13))
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=15, weights="distance")
    raise KeyError(name)


def matrix_for_split(rows: pd.DataFrame, config_ids: list[str]) -> pd.DataFrame:
    key_cols = ["unified_curve_id", "fold"]
    wide = rows.pivot_table(index=key_cols, columns="config_id", values="rmse", aggfunc="first")
    return wide.reindex(columns=config_ids)


def actual_for_config(rows: pd.DataFrame, curve_id: str, fold: int, config_id: str) -> pd.Series:
    sub = rows[(rows["unified_curve_id"] == curve_id) & (rows["fold"] == fold) & (rows["config_id"] == config_id)]
    if sub.empty:
        raise KeyError((curve_id, fold, config_id))
    return sub.iloc[0]


def evaluate_gate(
    *,
    expert_rows: pd.DataFrame,
    forms: pd.DataFrame,
    split_kind: str,
    gate_model: str,
    include_source_gate: bool,
    seed: int,
    n_estimators: int,
) -> list[dict[str, Any]]:
    rows = expert_rows[expert_rows["split_kind"] == split_kind].copy()
    config_ids = sorted(rows.loc[~rows["uses_source_diagnostic"], "config_id"].unique())
    out_rows: list[dict[str, Any]] = []
    form_cols = ["unified_curve_id"] + sorted(set(GATE_NUMERIC_COLS + GATE_CATEGORICAL_COLS + SOURCE_CATEGORICAL_COLS) & set(forms.columns))
    forms_small = forms[form_cols].copy()

    for fold in sorted(rows["fold"].unique()):
        train_rows = rows[rows["fold"] != fold].copy()
        test_rows = rows[rows["fold"] == fold].copy()
        train_ids = sorted(train_rows["unified_curve_id"].astype(str).unique())
        test_ids = sorted(test_rows["unified_curve_id"].astype(str).unique())
        train_forms = forms_small[forms_small["unified_curve_id"].isin(train_ids)].copy().sort_values("unified_curve_id")
        test_forms = forms_small[forms_small["unified_curve_id"].isin(test_ids)].copy().sort_values("unified_curve_id")
        if train_forms.empty or test_forms.empty:
            continue

        train_matrix = matrix_for_split(train_rows[~train_rows["uses_source_diagnostic"]], config_ids)
        test_matrix = matrix_for_split(test_rows[~test_rows["uses_source_diagnostic"]], config_ids)
        train_matrix = train_matrix.loc[(train_forms["unified_curve_id"].to_numpy(), [fold] * len(train_forms))] if False else train_matrix
        # Reindex by curve only after confirming each curve has one row in this fold set.
        train_y = train_matrix.reset_index().drop(columns="fold").set_index("unified_curve_id").reindex(train_forms["unified_curve_id"])[config_ids]
        test_y = test_matrix.reset_index().drop(columns="fold").set_index("unified_curve_id").reindex(test_forms["unified_curve_id"])[config_ids]
        valid_train = ~train_y.isna().any(axis=1)
        valid_test = ~test_y.isna().any(axis=1)
        train_forms = train_forms.loc[valid_train.to_numpy()].copy()
        test_forms = test_forms.loc[valid_test.to_numpy()].copy()
        train_y = train_y.loc[valid_train].to_numpy(dtype=float)
        test_y_df = test_y.loc[valid_test].copy()
        if len(train_forms) < 10 or test_forms.empty:
            continue

        x_train, x_test, design_cols = build_design(train_forms, test_forms, include_source_gate)
        gate = make_gate(gate_model, seed + int(fold), n_estimators)
        gate.fit(x_train, train_y)
        pred_error = np.asarray(gate.predict(x_test), dtype=float)
        if pred_error.ndim == 1:
            pred_error = pred_error.reshape(-1, len(config_ids))
        selected_idx = np.argmin(pred_error, axis=1)

        train_median = np.median(train_y, axis=0)
        global_idx = int(np.argmin(train_median))
        for row_idx, (_, form_row) in enumerate(test_forms.iterrows()):
            curve_id = str(form_row["unified_curve_id"])
            selected_config = config_ids[int(selected_idx[row_idx])]
            global_config = config_ids[global_idx]
            actual = test_y_df.loc[curve_id]
            oracle_config = str(actual.idxmin())
            intercept_candidates = [cid for cid in config_ids if cid.startswith("intercept_only|")]
            intercept_config = min(intercept_candidates, key=lambda cid: float(actual[cid])) if intercept_candidates else global_config
            selected_actual = actual_for_config(test_rows, curve_id, int(fold), selected_config)
            oracle_actual = actual_for_config(test_rows, curve_id, int(fold), oracle_config)
            global_actual = actual_for_config(test_rows, curve_id, int(fold), global_config)
            intercept_actual = actual_for_config(test_rows, curve_id, int(fold), intercept_config)
            out_rows.append(
                {
                    "split_kind": split_kind,
                    "fold": int(fold),
                    "unified_curve_id": curve_id,
                    "gate_model": gate_model,
                    "include_source_gate": bool(include_source_gate),
                    "n_train_curves": int(len(train_forms)),
                    "n_test_curves": int(len(test_forms)),
                    "n_configs": int(len(config_ids)),
                    "n_gate_design_cols": int(len(design_cols)),
                    "selected_config_id": selected_config,
                    "selected_feature_group": selected_actual["feature_group"],
                    "selected_family": selected_actual["family"],
                    "selected_model": selected_actual["model"],
                    "gated_rmse": float(selected_actual["rmse"]),
                    "global_config_id": global_config,
                    "global_rmse": float(global_actual["rmse"]),
                    "oracle_config_id": oracle_config,
                    "oracle_rmse": float(oracle_actual["rmse"]),
                    "intercept_config_id": intercept_config,
                    "intercept_rmse": float(intercept_actual["rmse"]),
                    "gated_minus_oracle": float(selected_actual["rmse"] - oracle_actual["rmse"]),
                    "gated_minus_global": float(selected_actual["rmse"] - global_actual["rmse"]),
                    "gated_minus_intercept": float(selected_actual["rmse"] - intercept_actual["rmse"]),
                    "gate_hit_oracle_config": bool(selected_config == oracle_config),
                    "gate_hit_oracle_feature_group": bool(str(selected_actual["feature_group"]) == str(oracle_actual["feature_group"])),
                    "source_dataset": selected_actual["source_dataset"],
                    "source_group": selected_actual["source_group"],
                }
            )
    return out_rows


def summarize(per_curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["split_kind", "gate_model", "include_source_gate"]
    summary = (
        per_curve.groupby(group_cols, dropna=False)
        .agg(
            n_curves=("unified_curve_id", "nunique"),
            median_gated_rmse=("gated_rmse", "median"),
            median_global_rmse=("global_rmse", "median"),
            median_oracle_rmse=("oracle_rmse", "median"),
            median_intercept_rmse=("intercept_rmse", "median"),
            median_gated_minus_oracle=("gated_minus_oracle", "median"),
            median_gated_minus_global=("gated_minus_global", "median"),
            median_gated_minus_intercept=("gated_minus_intercept", "median"),
            fraction_beats_global=("gated_minus_global", lambda s: float(np.mean(s < 0.0))),
            fraction_beats_intercept=("gated_minus_intercept", lambda s: float(np.mean(s < 0.0))),
            oracle_config_hit_rate=("gate_hit_oracle_config", "mean"),
            oracle_feature_group_hit_rate=("gate_hit_oracle_feature_group", "mean"),
        )
        .reset_index()
        .sort_values(["split_kind", "median_gated_rmse"])
    )
    freq = (
        per_curve.groupby(["split_kind", "gate_model", "include_source_gate", "selected_feature_group", "selected_family", "selected_model"], dropna=False)
        .agg(n_curves=("unified_curve_id", "nunique"), median_gated_rmse=("gated_rmse", "median"))
        .reset_index()
        .sort_values(["split_kind", "gate_model", "n_curves"], ascending=[True, True, False])
    )
    decisions = []
    for split_kind, sub in summary.groupby("split_kind", dropna=False):
        best = sub.iloc[0]
        if float(best["median_gated_minus_global"]) < -0.01:
            answer = "gate improves over global best"
        elif float(best["median_gated_minus_global"]) <= 0.01:
            answer = "gate roughly ties global best"
        else:
            answer = "gate underperforms global best"
        decisions.append(
            {
                "split_kind": split_kind,
                "answer": answer,
                "evidence": (
                    f"best gate={best['gate_model']}, source_gate={bool(best['include_source_gate'])}; "
                    f"gated RMSE={float(best['median_gated_rmse']):.3f}; "
                    f"global={float(best['median_global_rmse']):.3f}; "
                    f"oracle={float(best['median_oracle_rmse']):.3f}; "
                    f"hit feature group={float(best['oracle_feature_group_hit_rate']):.1%}"
                ),
            }
        )
    return summary, freq, pd.DataFrame(decisions)


def finite_numeric_check(name: str, df: pd.DataFrame) -> dict[str, Any]:
    bad: dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        count = int(vals.isna().sum())
        if count:
            bad[col] = count
    return {"name": name, "rows": int(len(df)), "bad_numeric": bad}


def write_report(out: Path, summary: pd.DataFrame, freq: pd.DataFrame, decisions: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    lines = [
        "# PLGA Static Combo Gate Screening",
        "",
        "Screening only. The gate uses existing script 91 out-of-fold expert errors as meta-labels. A final claim needs a fully nested gate.",
        "",
        "## Decision Table",
        "",
        decisions.to_markdown(index=False),
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Selection Frequency",
        "",
        freq.head(80).to_markdown(index=False),
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
    gate_models = [part.strip() for part in args.gate_models.split(",") if part.strip()]
    forms = load_plga_forms(args.pool_dir)
    expert_rows = load_expert_rows(args)
    all_rows: list[dict[str, Any]] = []
    for split_kind in sorted(expert_rows["split_kind"].unique()):
        for gate_model in gate_models:
            all_rows.extend(
                evaluate_gate(
                    expert_rows=expert_rows,
                    forms=forms,
                    split_kind=split_kind,
                    gate_model=gate_model,
                    include_source_gate=False,
                    seed=args.seed,
                    n_estimators=args.n_estimators,
                )
            )
            if args.include_source_gate:
                all_rows.extend(
                    evaluate_gate(
                        expert_rows=expert_rows,
                        forms=forms,
                        split_kind=split_kind,
                        gate_model=gate_model,
                        include_source_gate=True,
                        seed=args.seed,
                        n_estimators=args.n_estimators,
                    )
                )
    per_curve = pd.DataFrame(all_rows)
    if per_curve.empty:
        raise RuntimeError("No gated rows produced")
    summary, freq, decisions = summarize(per_curve)
    checks = [
        finite_numeric_check("gated_per_curve", per_curve),
        finite_numeric_check("gated_summary", summary),
        finite_numeric_check("selection_frequency", freq),
        finite_numeric_check("decision_table", decisions),
    ]
    bad = {check["name"]: check["bad_numeric"] for check in checks if check["bad_numeric"]}
    if bad:
        raise RuntimeError(f"NaN/inf detected: {bad}")
    per_curve.to_csv(args.out / "gated_per_curve.csv", index=False)
    summary.to_csv(args.out / "gated_summary.csv", index=False)
    freq.to_csv(args.out / "selection_frequency.csv", index=False)
    decisions.to_csv(args.out / "decision_table.csv", index=False)
    write_report(args.out, summary, freq, decisions, checks)
    metadata = {
        "script": "scripts/95_plga_static_combo_gate_screen.py",
        "git_hash": git_hash(),
        "args": args_to_metadata(args),
        "out": str(args.out),
        "gate_models": gate_models,
        "include_source_gate": bool(args.include_source_gate),
        "note": "Screening only. Uses existing OOF expert errors as meta-labels; final claim needs nested gate.",
    }
    (args.out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[95] wrote", args.out)
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
