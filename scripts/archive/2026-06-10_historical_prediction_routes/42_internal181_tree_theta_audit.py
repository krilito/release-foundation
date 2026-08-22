"""
42 - Apply the tree-ensemble theta route to cleaned internal181 curves.

What this does:
    Script 41 improved the cross321 RF -> theta baseline with tree
    ensemble variants. This script asks whether the same idea transfers
    to the internal 181 PLGA benchmark after cleaning curves in the same
    spirit as the 321 pipeline:

        - average duplicate (formulation, time) rows,
        - clip Release into [0, 1],
        - truncate observations to <= 90 days,
        - require at least 3 observations,
        - align with the 29 full-fit oracle theta bank.

    Prediction route:

        internal181 native descriptors + Q(1,3,5,7)
            -> tree ensemble
            -> theta_9
            -> PLGABiphasic simulator
            -> full observed curve

Outputs:
    outputs/42_internal181_tree_theta_audit/cleaned_curve_inventory.csv
    outputs/42_internal181_tree_theta_audit/per_curve.csv
    outputs/42_internal181_tree_theta_audit/scheme_method_summary.csv
    outputs/42_internal181_tree_theta_audit/summary.txt
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

from encoder import PLGA_CONTINUOUS_COLS, parse_dp_group  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


DATASET_COL = "dataset"
FID_COL = "fid"
INTERNAL_FID_COL = "Experimental_index"
TIME_COL = "Time"
Y_COL = "Release"
DP_GROUP_COL = "DP_Group"


@dataclass
class CurveRecord:
    fid: int
    dp_group: str
    drug_id: str
    polymer_family: str
    t_obs: np.ndarray
    q_obs: np.ndarray


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


def _load_clean_internal(
    data_csv: Path,
    full_fit_bank: Path,
    fit_filter_r2: float,
    t_grid_max_days: float,
    min_obs: int,
) -> tuple[pd.DataFrame, dict[int, CurveRecord], pd.DataFrame]:
    raw = pd.read_csv(data_csv)
    required = {INTERNAL_FID_COL, DP_GROUP_COL, TIME_COL, Y_COL, *PLGA_CONTINUOUS_COLS}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"missing internal181 columns: {sorted(missing)}")

    raw[Y_COL] = raw[Y_COL].astype(float).clip(0.0, 1.0)
    raw = raw[raw[TIME_COL].astype(float) <= t_grid_max_days].copy()
    cleaned = (
        raw.groupby([INTERNAL_FID_COL, TIME_COL], as_index=False, sort=False)
        .agg({
            **{
                c: "first"
                for c in raw.columns
                if c not in {TIME_COL, Y_COL}
            },
            Y_COL: "mean",
        })
        .sort_values([INTERNAL_FID_COL, TIME_COL])
        .reset_index(drop=True)
    )

    bank = pd.read_csv(full_fit_bank)
    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    bank = bank[
        (bank[DATASET_COL] == "internal181")
        & np.isfinite(bank["full_r2"])
        & (bank["full_r2"] >= fit_filter_r2)
    ].copy()
    bank = bank[[FID_COL, "full_r2", *param_names]].rename(columns={FID_COL: INTERNAL_FID_COL})

    desc = (
        cleaned.drop_duplicates(INTERNAL_FID_COL)
        [[INTERNAL_FID_COL, DP_GROUP_COL, *PLGA_CONTINUOUS_COLS]]
        .merge(bank, on=INTERNAL_FID_COL, how="inner")
        .sort_values(INTERNAL_FID_COL)
        .reset_index(drop=True)
    )

    curve_map: dict[int, CurveRecord] = {}
    inventory_rows: list[dict[str, object]] = []
    keep_fids: list[int] = []
    for fid in desc[INTERNAL_FID_COL].astype(int).tolist():
        g = cleaned[cleaned[INTERNAL_FID_COL].astype(int) == fid].sort_values(TIME_COL)
        t_obs = g[TIME_COL].to_numpy(dtype=float)
        q_obs = np.clip(g[Y_COL].to_numpy(dtype=float), 0.0, 1.0)
        if len(t_obs) < min_obs:
            continue
        dp_group = str(g[DP_GROUP_COL].iloc[0])
        drug_id, polymer_family = parse_dp_group(dp_group)
        curve_map[fid] = CurveRecord(
            fid=fid,
            dp_group=dp_group,
            drug_id=drug_id,
            polymer_family=polymer_family,
            t_obs=t_obs,
            q_obs=q_obs,
        )
        keep_fids.append(fid)
        inventory_rows.append({
            INTERNAL_FID_COL: fid,
            DP_GROUP_COL: dp_group,
            "drug_id": drug_id,
            "polymer_family": polymer_family,
            "n_obs": int(len(t_obs)),
            "t_min": float(t_obs.min()),
            "t_max": float(t_obs.max()),
            "q_min": float(q_obs.min()),
            "q_max": float(q_obs.max()),
        })

    desc = desc[desc[INTERNAL_FID_COL].astype(int).isin(keep_fids)].reset_index(drop=True)
    return desc, curve_map, pd.DataFrame(inventory_rows)


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


def _decode_rows(
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
        row: dict[str, object] = {
            "scheme": scheme,
            "fold": fold,
            "fid": fid,
            "dp_group": curve.dp_group,
            "drug_id": curve.drug_id,
            "polymer_family": curve.polymer_family,
            "n_obs": int(len(curve.t_obs)),
        }
        early_rmse: dict[str, float] = {}
        for method in method_names:
            theta = _clip_theta(theta_pred_by_method[method][local_i], lows, highs)
            row[f"r2_{method}"] = _r2(curve.q_obs, sim.simulate_numpy(theta, curve.t_obs))
            pred_early = sim.simulate_numpy(theta, early_times)
            early_rmse[method] = _rmse(early_q[j].astype(float), pred_early)

        select_pool = [m for m in ["RF_ztheta", "ET_ztheta", "RF_ET_zavg"] if m in method_names]
        selected = min(select_pool, key=lambda m: early_rmse[m])
        row["early_select_method"] = selected
        row["r2_early_select"] = row[f"r2_{selected}"]
        row["early_select_rmse"] = early_rmse[selected]
        rows.append(row)
    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    method_cols = [c for c in per_curve.columns if c.startswith("r2_")]
    rows: list[dict[str, object]] = []
    for scheme, sub in per_curve.groupby("scheme", sort=False):
        for col in method_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
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
    ap.add_argument("--data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
    ap.add_argument(
        "--full-fit-bank",
        type=Path,
        default=Path("outputs/29_minimal_active_set_audit_r5/full_fit_bank.csv"),
    )
    ap.add_argument("--fit-filter-r2", type=float, default=0.95)
    ap.add_argument("--t-grid-max-days", type=float, default=90.0)
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--early-times", nargs="+", type=float, default=[1.0, 3.0, 5.0, 7.0])
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
    ap.add_argument("--out", type=Path, default=Path("outputs/42_internal181_tree_theta_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    param_names = list(sim.param_names)

    desc, curve_map, inventory = _load_clean_internal(
        data_csv=args.data,
        full_fit_bank=args.full_fit_bank,
        fit_filter_r2=args.fit_filter_r2,
        t_grid_max_days=args.t_grid_max_days,
        min_obs=args.min_obs,
    )
    if args.max_curves is not None:
        desc = desc.iloc[: int(args.max_curves)].reset_index(drop=True)
        keep = set(desc[INTERNAL_FID_COL].astype(int))
        inventory = inventory[inventory[INTERNAL_FID_COL].astype(int).isin(keep)].reset_index(drop=True)

    fids = desc[INTERNAL_FID_COL].to_numpy(dtype=int)
    early_times = np.array(args.early_times, dtype=float)
    early_q = np.stack([
        _interp_at(curve_map[int(fid)].t_obs, curve_map[int(fid)].q_obs, early_times)
        for fid in fids
    ]).astype(np.float32)
    x_form = desc[list(PLGA_CONTINUOUS_COLS)].to_numpy(dtype=np.float32)
    if args.input_mode == "formulation_only":
        x = x_form
    elif args.input_mode == "early_only":
        x = early_q
    else:
        x = np.concatenate([x_form, early_q], axis=1)
    theta = desc[param_names].to_numpy(dtype=np.float32)
    drug_groups = np.array([curve_map[int(fid)].drug_id for fid in fids], dtype=object)
    polymer_groups = np.array([curve_map[int(fid)].polymer_family for fid in fids], dtype=object)

    inventory.to_csv(args.out / "cleaned_curve_inventory.csv", index=False)
    print(
        f"[42] n_curves={len(desc)}, unique_drugs={pd.Series(drug_groups).nunique()}, "
        f"unique_polymers={pd.Series(polymer_groups).nunique()}, input_mode={args.input_mode}",
        flush=True,
    )

    schemes: list[tuple[str, object | None]] = [
        ("random_5fold", None),
        ("group_by_drug", drug_groups),
        ("group_by_polymer", polymer_groups),
    ]
    rows: list[dict[str, object]] = []

    for scheme_name, groups in schemes:
        print(f"\n[42] ====== scheme: {scheme_name} ======", flush=True)
        if scheme_name == "random_5fold":
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.arange(len(desc))))
        else:
            n_unique = pd.Series(groups).nunique()
            n_splits = min(args.n_folds, int(n_unique))
            if n_splits < 2:
                print(f"[42] skipping {scheme_name}: only {n_unique} unique groups", flush=True)
                continue
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.arange(len(desc)), groups=groups))

        for fold_idx, (tr, te) in enumerate(splits):
            print(f"[42] {scheme_name} fold {fold_idx + 1}/{len(splits)}", flush=True)
            theta_pred: dict[str, np.ndarray] = {}
            theta_pred["RF_raw"] = _fit_predict_raw(
                _make_rf(args, args.seed + fold_idx), x, theta, tr, te
            )
            theta_pred["RF_ztheta"] = _fit_predict_ztheta(
                _make_rf(args, args.seed + 100 + fold_idx), x, theta, tr, te
            )
            theta_pred["RF_ztheta_leaf2"] = _fit_predict_ztheta(
                _make_rf(args, args.seed + 200 + fold_idx, min_samples_leaf=2),
                x,
                theta,
                tr,
                te,
            )
            theta_pred["ET_raw"] = _fit_predict_raw(
                _make_et(args, args.seed + 300 + fold_idx), x, theta, tr, te
            )
            theta_pred["ET_ztheta"] = _fit_predict_ztheta(
                _make_et(args, args.seed + 400 + fold_idx), x, theta, tr, te
            )
            theta_pred["ET_ztheta_leaf2"] = _fit_predict_ztheta(
                _make_et(args, args.seed + 500 + fold_idx, min_samples_leaf=2),
                x,
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
                _decode_rows(
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
        "=== 42 -- internal181 cleaned tree-ensemble theta audit ===",
        "",
        f"data              : {args.data}",
        f"full_fit_bank     : {args.full_fit_bank}",
        f"fit_filter_r2     : {args.fit_filter_r2:.3f}",
        f"n_curves          : {len(desc)}",
        f"n_estimators      : {args.n_estimators}",
        f"input_mode        : {args.input_mode}",
        f"features          : {x.shape[1]} model inputs; early_select scores on {len(early_times)} early Q",
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
    lines.append("--- cleaning rule ---")
    lines.append("  Curves were grouped by (Experimental_index, Time), Release was")
    lines.append("  averaged for duplicate times and clipped to [0, 1], observations")
    lines.append("  after 90 days were removed, and curves were aligned to the internal")
    lines.append("  full-fit theta bank from script 29.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
