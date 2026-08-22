"""
111 - Audit the original accelerated_IVR liposome author baseline.

Purpose:
    Record exactly what the public Yanes et al. accelerated_IVR code predicts,
    how it trains, and how it should be converted into a curve-level baseline
    for our liposome observation-budget route.

Consumes:
    data/external/accelerated_IVR/repo/accelerated_IVR-main/

Produces:
    outputs/111_liposome_author_baseline_audit/
        author_repo_audit.csv
        author_dataset_summary.csv
        author_classifier_cv_summary.csv
        author_classifier_fold_results.csv
        author_training_logic.csv
        author_curve_baseline_plan.csv
        data_checks.csv
        lock_metadata.json
        report.md
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data/external/accelerated_IVR/repo/accelerated_IVR-main")
DEFAULT_OUT = Path("outputs/111_liposome_author_baseline_audit")

GITHUB_URL = "https://github.com/danielyanes22/accelerated_IVR"
PAPER_URL = "https://doi.org/10.1039/D5DD00112A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=D:/release-foundation", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _requirements_versions(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _read_text(root / "requirements.txt").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            out[name.strip().lower()] = version.strip()
    return out


def _extract_classifier_names(script_text: str) -> list[str]:
    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "classifier_names":
                    if isinstance(node.value, ast.List):
                        for item in node.value.elts:
                            if isinstance(item, ast.Constant):
                                names.append(str(item.value))
                    return names
    return names


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _dataset_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rel in [
        "data/clean/ML_7_features_df.csv",
        "data/clean/ML_9_features_df.csv",
        "data/clean/weibull_params.csv",
        "results/clustering/3_PCA_KMC.csv",
        "results/fitting/drug_release_exp.csv",
    ]:
        path = root / rel
        df = _safe_read_csv(path)
        row: dict[str, Any] = {
            "file": rel,
            "exists": path.exists(),
            "n_rows": int(len(df)) if not df.empty else 0,
            "n_columns": int(len(df.columns)) if not df.empty else 0,
            "columns": "|".join(map(str, df.columns)) if not df.empty else "",
        }
        if "cluster" in df.columns:
            counts = df["cluster"].value_counts(dropna=False).sort_index()
            row["cluster_counts"] = json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def _classifier_summary(root: Path) -> pd.DataFrame:
    test = _safe_read_csv(root / "results/ML_classifiers/test_CV.csv")
    train = _safe_read_csv(root / "results/ML_classifiers/train_CV.csv")
    if test.empty:
        return pd.DataFrame()
    if "Unnamed: 0" in test.columns:
        test = test.rename(columns={"Unnamed: 0": "model"})
    if "Unnamed: 0" in train.columns:
        train = train.rename(columns={"Unnamed: 0": "model"})
    keep = [
        "model",
        "mean_balaccuracy",
        "sd_balaccuracy",
        "mean_f1",
        "sd_f1",
        "mean_mcc",
        "sd_mcc",
    ]
    test = test[[c for c in keep if c in test.columns]].copy()
    train = train[[c for c in keep if c in train.columns]].copy()
    out = test.rename(columns={c: f"test_{c}" for c in test.columns if c != "model"})
    if not train.empty:
        train = train.rename(columns={c: f"train_{c}" for c in train.columns if c != "model"})
        out = out.merge(train, on="model", how="left")
        if {"train_mean_balaccuracy", "test_mean_balaccuracy"}.issubset(out.columns):
            out["balaccuracy_train_minus_test"] = out["train_mean_balaccuracy"] - out["test_mean_balaccuracy"]
    return out.sort_values("test_mean_balaccuracy", ascending=False).reset_index(drop=True)


def _training_logic(root: Path) -> pd.DataFrame:
    script = _read_text(root / "experiments/testing_scores.py")
    classifiers = _extract_classifier_names(script)
    rows = [
        {
            "stage": "target",
            "author_logic": "Predict kinetic class labels, not continuous future Q(t).",
            "strictness_note": "Useful author baseline, but endpoint differs from our curve-level observation-budget task.",
        },
        {
            "stage": "features",
            "author_logic": "Use data/clean/ML_7_features_df.csv; X is all columns except cluster.",
            "strictness_note": "The local file has 7 numeric/static features and 77 labeled rows.",
        },
        {
            "stage": "preprocessing",
            "author_logic": "StandardScaler is fit to the full X before cross-validation; LabelEncoder encodes cluster.",
            "strictness_note": "Full-dataset scaling before CV is minor preprocessing leakage under strict ML practice.",
        },
        {
            "stage": "models",
            "author_logic": ", ".join(classifiers),
            "strictness_note": "Seven ordinary classifiers; no early observations and no continuous curve decoder.",
        },
        {
            "stage": "split",
            "author_logic": "StratifiedKFold(n_splits=5, shuffle=True, random_state=15).",
            "strictness_note": "Random stratified CV is easier than our group-by-API/release-method transfer splits.",
        },
        {
            "stage": "metrics",
            "author_logic": "Balanced accuracy, micro F1, and Matthews correlation coefficient on class labels.",
            "strictness_note": "These are classification metrics; they do not measure RMSE on release curves.",
        },
        {
            "stage": "exports",
            "author_logic": "Save model pickle files plus train_CV.csv, test_CV.csv, fold results, Wilcoxon tests.",
            "strictness_note": "We use the CSVs for audit; generated model files remain untracked external artifacts.",
        },
    ]
    return pd.DataFrame(rows)


def _curve_baseline_plan() -> pd.DataFrame:
    rows = [
        {
            "baseline": "author_stratified_class_proto",
            "input": "paper-like static 7 features",
            "training": "train author classifier on train fold; predict kinetic class on test fold",
            "curve_conversion": "map predicted class to train-fold median Weibull alpha/beta; generate Q(t)",
            "role": "paper-like diagnostic only",
        },
        {
            "baseline": "author_group_by_API_class_proto",
            "input": "same static 7 features",
            "training": "same classifier, but folds hold out API_name groups",
            "curve_conversion": "train-fold class prototype Weibull curve",
            "role": "strict transfer baseline against our liposome E1/E3 API split",
        },
        {
            "baseline": "author_group_by_release_method_class_proto",
            "input": "same static 7 features",
            "training": "same classifier, but folds hold out release_method groups",
            "curve_conversion": "train-fold class prototype Weibull curve",
            "role": "strict assay-shift baseline",
        },
        {
            "baseline": "author_oracle_class_proto",
            "input": "true kinetic class label at test time",
            "training": "no classifier error; use train-fold prototypes",
            "curve_conversion": "train-fold class prototype Weibull curve",
            "role": "upper bound for class-prototype representation, not a deployable model",
        },
    ]
    return pd.DataFrame(rows)


def _data_checks(root: Path, cv: pd.DataFrame, datasets: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check": "local_author_root_exists",
            "passed": root.exists(),
            "detail": str(root),
        },
        {
            "check": "testing_scores_script_exists",
            "passed": (root / "experiments/testing_scores.py").exists(),
            "detail": "original author classifier script",
        },
        {
            "check": "ml_7_dataset_has_cluster",
            "passed": "cluster"
            in _safe_read_csv(root / "data/clean/ML_7_features_df.csv").columns,
            "detail": "classification target is kinetic class",
        },
        {
            "check": "author_cv_results_exist",
            "passed": not cv.empty,
            "detail": "results/ML_classifiers/test_CV.csv was found after reproduction",
        },
        {
            "check": "all_expected_files_seen",
            "passed": bool(datasets["exists"].all()) if not datasets.empty else False,
            "detail": "clean ML data, Weibull params, clustering, and release observations",
        },
    ]
    return pd.DataFrame(checks)


def _write_report(
    out: Path,
    root: Path,
    repo_audit: pd.DataFrame,
    datasets: pd.DataFrame,
    cv: pd.DataFrame,
    logic: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    best = cv.iloc[0].to_dict() if not cv.empty else {}
    ml7 = datasets[datasets["file"].eq("data/clean/ML_7_features_df.csv")]
    n_ml7 = int(ml7["n_rows"].iloc[0]) if not ml7.empty else 0
    cols_ml7 = str(ml7["columns"].iloc[0]).replace("|", ", ") if not ml7.empty else ""
    lines = [
        "# Liposome Author Baseline Audit",
        "",
        f"Date: 2026-06-12",
        f"Author repo: {GITHUB_URL}",
        f"Paper DOI: {PAPER_URL}",
        f"Local root: `{root.as_posix()}`",
        "",
        "## What They Predict",
        "",
        "The public accelerated_IVR code predicts a discrete liposome release kinetic class from static formulation and IVR-test features. It does not directly predict a continuous future release curve and it does not use early measured Q(t) context.",
        "",
        "## Training Logic",
        "",
    ]
    for row in logic.itertuples(index=False):
        lines.append(f"- {row.stage}: {row.author_logic} Strict note: {row.strictness_note}")
    lines.extend(
        [
            "",
            "## Local Reproduction Record",
            "",
            f"- ML_7 dataset: {n_ml7} rows; columns: {cols_ml7}.",
            f"- Reproduced command: `python -m experiments.testing_scores` from the author root.",
        ]
    )
    if best:
        lines.append(
            "- Best local test balanced accuracy: "
            f"{best.get('model')} = {best.get('test_mean_balaccuracy'):.3f} "
            f"(F1 {best.get('test_mean_f1'):.3f}, MCC {best.get('test_mean_mcc'):.3f})."
        )
    lines.extend(
        [
            "",
            "## Why This Is Not Yet Our Headline Baseline",
            "",
            "Their endpoint is class prediction under stratified random CV. Our endpoint is curve-level RMSE/MAE under stricter transfer splits and early-observation budgets. Therefore the fair next step is to convert their predicted class into a train-fold Weibull prototype curve and score it with the same folds as our E1/E3 liposome experiments.",
            "",
            "## Strict Caveats",
            "",
            "- The original script scales all X before CV, which is a small preprocessing leak under strict ML standards.",
            "- Stratified random CV can mix related APIs/methods across folds, so it is easier than group-by-API or group-by-release-method transfer.",
            "- Class labels come from the authors' Weibull/PCA/KMeans pipeline; they are useful kinetic summaries, not measured mechanisms.",
            "",
            "## Outputs",
            "",
            "- `author_repo_audit.csv` records repo/license/dependency metadata.",
            "- `author_classifier_cv_summary.csv` records reproduced train/test classification metrics.",
            "- `author_training_logic.csv` records the prediction and training logic in auditable rows.",
            "- `author_curve_baseline_plan.csv` defines the exact curve-space baselines to implement next.",
            "- `data_checks.csv` records pass/fail checks.",
            "",
            "## Checks",
            "",
        ]
    )
    for row in checks.itertuples(index=False):
        status = "PASS" if bool(row.passed) else "FAIL"
        lines.append(f"- {status}: {row.check} - {row.detail}")
    lines.append("")
    out.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    readme = _read_text(root / "README.md")
    license_text = _read_text(root / "LICENSE.md")
    versions = _requirements_versions(root)
    repo_audit = pd.DataFrame(
        [
            {
                "local_root": str(root),
                "github_url": GITHUB_URL,
                "paper_url": PAPER_URL,
                "readme_title": _first_nonempty_line(readme).lstrip("#").strip(),
                "license_first_line": _first_nonempty_line(license_text),
                "requirements_xgboost": versions.get("xgboost", ""),
                "requirements_scikit_learn": versions.get("scikit-learn", ""),
                "runtime_python": sys.version.split()[0],
                "runtime_platform": platform.platform(),
            }
        ]
    )
    datasets = _dataset_rows(root)
    cv = _classifier_summary(root)
    folds = _safe_read_csv(root / "results/ML_classifiers/model_per_fold_results.csv")
    logic = _training_logic(root)
    plan = _curve_baseline_plan()
    checks = _data_checks(root, cv, datasets)

    repo_audit.to_csv(out / "author_repo_audit.csv", index=False)
    datasets.to_csv(out / "author_dataset_summary.csv", index=False)
    cv.to_csv(out / "author_classifier_cv_summary.csv", index=False)
    folds.to_csv(out / "author_classifier_fold_results.csv", index=False)
    logic.to_csv(out / "author_training_logic.csv", index=False)
    plan.to_csv(out / "author_curve_baseline_plan.csv", index=False)
    checks.to_csv(out / "data_checks.csv", index=False)

    _write_report(out, root, repo_audit, datasets, cv, logic, checks)

    metadata = {
        "script": Path(__file__).name,
        "date": "2026-06-12",
        "seed": args.seed,
        "root": str(root),
        "out": str(out),
        "git_hash": _git_hash(),
        "github_url": GITHUB_URL,
        "paper_url": PAPER_URL,
        "generated_files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "lock_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Author baseline audit failed checks: {failed}")


if __name__ == "__main__":
    main()
