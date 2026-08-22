"""
39 - Interpret the RF -> theta middle layer.

What this does:
    Script 38d showed that the strongest predictor is:

        formulation descriptors + early Q(1,3,5,7)
            -> RandomForestRegressor
            -> 9-param theta
            -> PLGABiphasic simulator
            -> full release curve

    This script pulls out the middle layer and asks: what is the RF using
    to predict each mechanism parameter?

    For each CV scheme (random, held-out drug, held-out polymer), and for
    each theta parameter, it trains a separate RF regressor and computes
    held-out permutation importance for the 14 inputs. The separate
    per-parameter RFs make the feature -> parameter map readable; this is
    an interpretation diagnostic, not a replacement for script 38d's
    multi-output RF benchmark.

Outputs:
    outputs/39_rf_theta_mapping_audit/per_parameter_cv.csv
    outputs/39_rf_theta_mapping_audit/permutation_importance.csv
    outputs/39_rf_theta_mapping_audit/top_features_by_parameter.csv
    outputs/39_rf_theta_mapping_audit/feature_group_summary.csv
    outputs/39_rf_theta_mapping_audit/importance_heatmap_random_5fold.png
    outputs/39_rf_theta_mapping_audit/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import PLGABiphasic  # noqa: E402


DATASET_COL = "dataset"
FID_COL = "fid"

FORMULATION_COL_MAP = {
    "Drug MW": "drug_mw",
    "Drug TPSA": "drug_tpsa",
    "Drug LogP": "drug_logp",
    "Polymer MW": "polymer_mw",
    "LA/GA": "laga",
    "Initial Drug-to-Polymer Ratio": "drug_polymer_ratio",
    "Particle Size": "particle_size",
    "Drug Loading Capacity": "drug_loading",
    "Drug Encapsulation Efficiency": "drug_ee",
    "Solubility Enhancer Concentration": "solubility_enhancer",
}
DRUG_KEY_COLS = ["drug_mw", "drug_tpsa", "drug_logp"]
POLYMER_KEY_COLS = ["polymer_mw", "laga"]


@dataclass
class CurveRecord:
    fid: int
    t_obs: np.ndarray
    q_obs: np.ndarray


def _load_external_records(
    xlsx_path: Path, matched_fids_csv: Path, t_grid_max_days: float
) -> list[CurveRecord]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    df = pd.read_excel(xlsx_path).rename(columns={"Formulation Index": "fid"})
    df["Release"] = df["Release"].astype(float).clip(0.0, 1.0)
    df = df[df["fid"].astype(int).isin(matched_fids)].copy()
    out: list[CurveRecord] = []
    for fid in sorted(df["fid"].unique().tolist()):
        g = df[df["fid"] == fid].sort_values("Time")
        g = g.groupby("Time", as_index=False).agg({"Release": "mean"})
        t_obs = g["Time"].to_numpy(dtype=float)
        q_obs = np.clip(g["Release"].to_numpy(dtype=float), 0.0, 1.0)
        mask = t_obs <= t_grid_max_days
        t_obs = t_obs[mask]
        q_obs = q_obs[mask]
        if len(t_obs) < 3:
            continue
        out.append(CurveRecord(fid=int(fid), t_obs=t_obs, q_obs=q_obs))
    return out


def _interp_at(t_obs: np.ndarray, q_obs: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.interp(times, t_obs, q_obs, left=q_obs[0], right=q_obs[-1])


def _feature_group(feature: str) -> str:
    if feature.startswith("Q_day"):
        return "early_Q"
    if feature.startswith("drug_"):
        return "drug_descriptor"
    if feature in {"polymer_mw", "laga"}:
        return "polymer_descriptor"
    return "formulation_process"


def _normalize_positive(values: pd.Series) -> pd.Series:
    positive = values.clip(lower=0.0)
    denom = float(positive.sum())
    if denom <= 0:
        return positive * 0.0
    return positive / denom


def _load_dataset(args: argparse.Namespace) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray
]:
    sim = PLGABiphasic()
    param_names = list(sim.param_names)

    df_meta = pd.read_excel(args.cross_doi_data, sheet_name=0)
    df_meta_first = (
        df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
            ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
        ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})
    )
    feature_cols = list(FORMULATION_COL_MAP.values())

    df_bank = pd.read_csv(args.full_fit_bank)
    df_bank_cross = df_bank[df_bank[DATASET_COL] == "cross321"].copy()
    df_reg = pd.read_csv(args.regime_csv)
    df_reg_cross = df_reg[df_reg[DATASET_COL] == "cross321"][[FID_COL, "regime"]]
    df = (
        df_bank_cross.merge(df_meta_first, on=FID_COL, how="inner")
        .merge(df_reg_cross, on=FID_COL, how="inner")
        .dropna(subset=feature_cols + param_names)
        .reset_index(drop=True)
    )

    curves = _load_external_records(
        xlsx_path=args.cross_doi_data,
        matched_fids_csv=args.matched_fids_csv,
        t_grid_max_days=90.0,
    )
    curve_map = {c.fid: c for c in curves}

    early_times = np.array(args.early_times, dtype=float)
    early_q_list: list[np.ndarray] = []
    keep_rows: list[int] = []
    for i, row in df.iterrows():
        c = curve_map.get(int(row[FID_COL]))
        if c is None:
            continue
        early_q_list.append(_interp_at(c.t_obs, c.q_obs, early_times))
        keep_rows.append(i)

    df = df.loc[keep_rows].reset_index(drop=True)
    early_q = np.stack(early_q_list, axis=0).astype(np.float32)
    early_cols = [f"Q_day{t:g}" for t in early_times]

    x_form = df[feature_cols].to_numpy(dtype=np.float32)
    x = np.concatenate([x_form, early_q], axis=1)
    x_cols = feature_cols + early_cols
    theta = df[param_names].to_numpy(dtype=np.float32)
    drug_keys = df[DRUG_KEY_COLS].apply(tuple, axis=1).to_numpy()
    polymer_keys = df[POLYMER_KEY_COLS].apply(tuple, axis=1).to_numpy()
    return df, x, theta, x_cols, drug_keys, polymer_keys


def _plot_heatmap(
    imp: pd.DataFrame, scheme: str, out_path: Path, max_features: int = 14
) -> None:
    sub = imp[imp["scheme"] == scheme].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index="param", columns="feature", values="importance_norm", aggfunc="mean"
    )
    feature_order = (
        sub.groupby("feature")["importance_norm"]
        .mean()
        .sort_values(ascending=False)
        .head(max_features)
        .index.tolist()
    )
    pivot = pivot.reindex(columns=feature_order).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"39: RF feature importance for theta parameters ({scheme})")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("ODE parameter")
    fig.colorbar(im, ax=ax, label="normalized permutation importance")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-csv",
        type=Path,
        default=Path("outputs/33_active_set_regimes/regime_assignments.csv"),
    )
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument(
        "--cross-doi-data",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA"
            r"\A Dataset on Formulation Parameters and Characteristics of "
            r"Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument(
        "--matched-fids-csv",
        type=Path,
        default=Path("outputs/sprint1_10_eval_deployment/cross_doi_per_curve_metrics.csv"),
    )
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=250)
    ap.add_argument("--perm-repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/39_rf_theta_mapping_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, x, theta, x_cols, drug_keys, polymer_keys = _load_dataset(args)
    param_names = list(PLGABiphasic.param_names)

    schemes: list[tuple[str, object | None]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_keys),
        ("group_by_polymer", polymer_keys),
    ]

    cv_rows: list[dict[str, object]] = []
    imp_rows: list[dict[str, object]] = []

    for scheme_name, groups in schemes:
        print(f"[39] scheme={scheme_name}", flush=True)
        if scheme_name == "random_5fold":
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(len(df))))
        else:
            splitter = GroupKFold(n_splits=args.n_folds)
            splits = list(splitter.split(np.arange(len(df)), groups=groups))

        for fold_idx, (tr, te) in enumerate(splits):
            print(f"[39]   fold {fold_idx + 1}/{args.n_folds}", flush=True)
            for param_idx, param in enumerate(param_names):
                y = theta[:, param_idx]
                rf = RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=None,
                    random_state=args.seed + 1000 * fold_idx + param_idx,
                    n_jobs=-1,
                )
                rf.fit(x[tr], y[tr])
                pred = rf.predict(x[te])
                theta_r2 = float(r2_score(y[te], pred))
                cv_rows.append({
                    "scheme": scheme_name,
                    "fold": fold_idx,
                    "param": param,
                    "theta_r2": theta_r2,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                })

                perm = permutation_importance(
                    rf,
                    x[te],
                    y[te],
                    scoring="neg_mean_squared_error",
                    n_repeats=args.perm_repeats,
                    random_state=args.seed + 2000 * fold_idx + param_idx,
                    n_jobs=-1,
                )
                for feature, mean, std in zip(
                    x_cols, perm.importances_mean, perm.importances_std
                ):
                    imp_rows.append({
                        "scheme": scheme_name,
                        "fold": fold_idx,
                        "param": param,
                        "feature": feature,
                        "feature_group": _feature_group(feature),
                        "importance_mean": float(mean),
                        "importance_std": float(std),
                    })

    cv_df = pd.DataFrame(cv_rows)
    imp_raw = pd.DataFrame(imp_rows)
    imp = (
        imp_raw.groupby(["scheme", "param", "feature", "feature_group"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
    )
    imp["importance_norm"] = (
        imp.groupby(["scheme", "param"])["importance_mean"]
        .transform(_normalize_positive)
    )

    top = (
        imp.sort_values(
            ["scheme", "param", "importance_norm"],
            ascending=[True, True, False],
        )
        .groupby(["scheme", "param"], as_index=False)
        .head(5)
        .reset_index(drop=True)
    )
    group_summary = (
        imp.groupby(["scheme", "param", "feature_group"], as_index=False)[
            "importance_norm"
        ]
        .sum()
        .sort_values(["scheme", "param", "importance_norm"], ascending=[True, True, False])
    )

    cv_df.to_csv(args.out / "per_parameter_cv.csv", index=False)
    imp.to_csv(args.out / "permutation_importance.csv", index=False)
    top.to_csv(args.out / "top_features_by_parameter.csv", index=False)
    group_summary.to_csv(args.out / "feature_group_summary.csv", index=False)

    for scheme_name, _ in schemes:
        _plot_heatmap(
            imp,
            scheme=scheme_name,
            out_path=args.out / f"importance_heatmap_{scheme_name}.png",
        )

    cv_summary = (
        cv_df.groupby(["scheme", "param"], as_index=False)["theta_r2"]
        .agg(["median", "mean"])
        .reset_index()
    )
    overall_group = (
        group_summary.groupby(["scheme", "feature_group"], as_index=False)[
            "importance_norm"
        ]
        .mean()
        .sort_values(["scheme", "importance_norm"], ascending=[True, False])
    )

    lines = [
        "=== 39 -- RF -> theta middle-layer mapping audit ===",
        "",
        f"n_curves          : {len(df)}",
        f"n_features        : {len(x_cols)} ({len(FORMULATION_COL_MAP)} formulation + {len(args.early_times)} early Q)",
        f"n_theta_params    : {len(param_names)}",
        f"n_estimators      : {args.n_estimators}",
        f"perm_repeats      : {args.perm_repeats}",
        "",
        "--- median held-out theta R^2 by scheme ---",
    ]
    for scheme_name, _ in schemes:
        sub = cv_summary[cv_summary["scheme"] == scheme_name]
        med = float(sub["median"].median())
        lines.append(f"  {scheme_name:<16}: median across params = {med:+.3f}")
    lines.append("")
    lines.append("--- feature-group importance averaged over theta params ---")
    for scheme_name, _ in schemes:
        lines.append(f"  {scheme_name}:")
        sub = overall_group[overall_group["scheme"] == scheme_name]
        for _, r in sub.iterrows():
            lines.append(f"    {r['feature_group']:<22} {r['importance_norm']:.3f}")
    lines.append("")
    lines.append("--- top features per theta parameter (random_5fold) ---")
    random_top = top[top["scheme"] == "random_5fold"]
    for param in param_names:
        sub = random_top[random_top["param"] == param].head(3)
        pieces = [
            f"{r.feature} ({r.importance_norm:.2f})"
            for r in sub.itertuples(index=False)
        ]
        lines.append(f"  {param:<14}: " + ", ".join(pieces))
    lines.append("")
    lines.append("--- interpretation ---")
    lines.append("  This is the RF middle layer: it learns X -> theta, where X is")
    lines.append("  formulation descriptors plus first-week release points. High")
    lines.append("  early_Q importance means the model is using partial trajectory shape")
    lines.append("  as a kinetic fingerprint; descriptor importance means it is using")
    lines.append("  formulation context to choose a feasible theta region.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
