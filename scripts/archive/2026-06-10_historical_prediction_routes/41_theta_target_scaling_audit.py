"""
41 - Can we improve RF -> theta by making the theta target friendlier?

What this does:
    Script 38d established RF -> theta -> ODE as the strongest prediction
    baseline. This audit tests a narrow, parameter-layer-specific idea:

        multi-output tree regressors split on summed target MSE,
        so theta parameters with larger variance can dominate splitting.

    We compare raw-theta forests against target-standardized forests:

        z_theta = (theta - train_mean) / train_std

    Predictions are inverse-transformed back to theta before ODE decoding.
    The evaluation data, folds, and curve R^2 calculation match 38d.

Outputs:
    outputs/41_theta_target_scaling_audit/per_curve.csv
    outputs/41_theta_target_scaling_audit/scheme_method_summary.csv
    outputs/41_theta_target_scaling_audit/summary.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _clip_theta(theta: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    eps = 1e-4 * (highs - lows)
    return np.clip(theta.astype(float), lows + eps, highs - eps)


def _load_dataset(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    dict[int, CurveRecord],
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
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
        t_grid_max_days=args.t_grid_max_days,
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
    x_form = df[feature_cols].to_numpy(dtype=np.float32)
    x = np.concatenate([x_form, early_q], axis=1)
    theta = df[param_names].to_numpy(dtype=np.float32)
    drug_keys = df[DRUG_KEY_COLS].apply(tuple, axis=1).to_numpy()
    polymer_keys = df[POLYMER_KEY_COLS].apply(tuple, axis=1).to_numpy()
    return df, curve_map, x, theta, param_names, drug_keys, polymer_keys


def _fit_predict_raw(model: object, x: np.ndarray, theta: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    model.fit(x[tr], theta[tr])
    return model.predict(x[te])


def _fit_predict_ztheta(model: object, x: np.ndarray, theta: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    mu = theta[tr].mean(axis=0, keepdims=True)
    sd = theta[tr].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x[tr], (theta[tr] - mu) / sd)
    return model.predict(x[te]) * sd + mu


def _make_rf(args: argparse.Namespace, seed: int, **kwargs: object) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        **kwargs,
    )


def _make_et(args: argparse.Namespace, seed: int, **kwargs: object) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        **kwargs,
    )


def _decode_r2_rows(
    sim: PLGABiphasic,
    scheme: str,
    fold: int,
    fids: np.ndarray,
    te: np.ndarray,
    theta_pred_by_method: dict[str, np.ndarray],
    curve_map: dict[int, CurveRecord],
    early_times: np.ndarray,
    early_q: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    method_names = list(theta_pred_by_method.keys())
    for local_i, j in enumerate(te):
        fid = int(fids[j])
        curve = curve_map[fid]
        row: dict[str, object] = {"scheme": scheme, "fold": fold, "fid": fid}
        early_rmse_by_method: dict[str, float] = {}
        for method in method_names:
            theta = _clip_theta(theta_pred_by_method[method][local_i], lows, highs)
            pred = sim.simulate_numpy(theta, curve.t_obs)
            row[f"r2_{method}"] = _r2(curve.q_obs, pred)
            pred_early = sim.simulate_numpy(theta, early_times)
            early_rmse_by_method[method] = _rmse(early_q[j].astype(float), pred_early)

        select_pool = [m for m in ["RF_ztheta", "ET_ztheta", "RF_ET_zavg"] if m in method_names]
        selected = min(select_pool, key=lambda m: early_rmse_by_method[m])
        row["early_select_method"] = selected
        row["r2_early_select"] = row[f"r2_{selected}"]
        row["early_select_rmse"] = early_rmse_by_method[selected]
        rows.append(row)
    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    method_cols = [c for c in per_curve.columns if c.startswith("r2_")]
    rows: list[dict[str, object]] = []
    for scheme, sub in per_curve.groupby("scheme", sort=False):
        for col in method_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append({
                "scheme": scheme,
                "method": col.removeprefix("r2_"),
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p10": float(np.percentile(vals, 10)),
                "frac_above_0.9": float(np.mean(vals >= 0.9)),
                "frac_above_0.5": float(np.mean(vals >= 0.5)),
                "frac_above_0": float(np.mean(vals >= 0.0)),
            })
    return pd.DataFrame(rows)


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
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=None)
    ap.add_argument(
        "--input-mode",
        choices=("formulation_plus_early", "formulation_only", "early_only"),
        default="formulation_plus_early",
        help="Which inputs feed the tree model. Early Q is always kept for early_select scoring.",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/41_theta_target_scaling_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)

    df, curve_map, x, theta, _param_names, drug_keys, polymer_keys = _load_dataset(args)
    if args.max_curves is not None:
        n_keep = min(int(args.max_curves), len(df))
        df = df.iloc[:n_keep].reset_index(drop=True)
        x = x[:n_keep]
        theta = theta[:n_keep]
        drug_keys = drug_keys[:n_keep]
        polymer_keys = polymer_keys[:n_keep]

    fids = df[FID_COL].to_numpy(dtype=int)
    early_times = np.array(args.early_times, dtype=float)
    early_q = x[:, -len(early_times):]
    if args.input_mode == "formulation_only":
        x_model = x[:, :-len(early_times)]
    elif args.input_mode == "early_only":
        x_model = early_q
    else:
        x_model = x
    n = len(df)
    print(
        f"[41] n_curves={n}, n_estimators={args.n_estimators}, input_mode={args.input_mode}",
        flush=True,
    )

    schemes: list[tuple[str, object | None]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_keys),
        ("group_by_polymer", polymer_keys),
    ]
    rows: list[dict[str, object]] = []

    for scheme_name, groups in schemes:
        print(f"\n[41] ====== scheme: {scheme_name} ======", flush=True)
        if scheme_name == "random_5fold":
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(n)))
        else:
            splitter = GroupKFold(n_splits=args.n_folds)
            splits = list(splitter.split(np.arange(n), groups=groups))

        for fold_idx, (tr, te) in enumerate(splits):
            print(f"[41] {scheme_name} fold {fold_idx + 1}/{len(splits)}", flush=True)
            theta_pred: dict[str, np.ndarray] = {}
            theta_pred["RF_raw"] = _fit_predict_raw(
                _make_rf(args, args.seed + fold_idx), x_model, theta, tr, te
            )
            theta_pred["RF_ztheta"] = _fit_predict_ztheta(
                _make_rf(args, args.seed + 100 + fold_idx), x_model, theta, tr, te
            )
            theta_pred["RF_ztheta_leaf2"] = _fit_predict_ztheta(
                _make_rf(args, args.seed + 200 + fold_idx, min_samples_leaf=2),
                x_model,
                theta,
                tr,
                te,
            )
            theta_pred["ET_raw"] = _fit_predict_raw(
                _make_et(args, args.seed + 300 + fold_idx), x_model, theta, tr, te
            )
            theta_pred["ET_ztheta"] = _fit_predict_ztheta(
                _make_et(args, args.seed + 400 + fold_idx), x_model, theta, tr, te
            )
            theta_pred["ET_ztheta_leaf2"] = _fit_predict_ztheta(
                _make_et(args, args.seed + 500 + fold_idx, min_samples_leaf=2),
                x_model,
                theta,
                tr,
                te,
            )
            theta_pred["RF_ET_zavg"] = 0.5 * (
                theta_pred["RF_ztheta"] + theta_pred["ET_ztheta"]
            )
            theta_pred["RF_raw_zavg"] = 0.5 * (
                theta_pred["RF_raw"] + theta_pred["RF_ztheta"]
            )

            rows.extend(
                _decode_r2_rows(
                    sim=sim,
                    scheme=scheme_name,
                    fold=fold_idx,
                    fids=fids,
                    te=te,
                    theta_pred_by_method=theta_pred,
                    curve_map=curve_map,
                    early_times=early_times,
                    early_q=early_q,
                    lows=lows,
                    highs=highs,
                )
            )

    per_curve = pd.DataFrame(rows)
    summary = _summary(per_curve)
    per_curve.to_csv(args.out / "per_curve.csv", index=False)
    summary.to_csv(args.out / "scheme_method_summary.csv", index=False)

    lines = [
        "=== 41 -- theta target-scaling / tree ensemble audit ===",
        "",
        f"n_curves     : {n}",
        f"n_estimators : {args.n_estimators}",
        f"input_mode   : {args.input_mode}",
        "",
        "--- median R^2 by scheme x method ---",
    ]
    for scheme in ["random_5fold", "group_by_drug", "group_by_polymer"]:
        sub = summary[summary["scheme"] == scheme].sort_values("median", ascending=False)
        if sub.empty:
            continue
        lines.append(f"  {scheme}:")
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['method']:<18} median={row['median']:+.4f} "
                f"p10={row['p10']:+.4f} frac_R2>=0={row['frac_above_0']:.3f}"
            )
    lines.append("")
    lines.append("--- interpretation ---")
    lines.append("  RF_ztheta tests whether standardizing the 9-D theta target helps")
    lines.append("  multi-output trees give each ODE parameter comparable influence.")
    lines.append("  early_select chooses among RF_ztheta, ET_ztheta, and their average")
    lines.append("  by early-window ODE RMSE only; full curves are evaluation-only.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
