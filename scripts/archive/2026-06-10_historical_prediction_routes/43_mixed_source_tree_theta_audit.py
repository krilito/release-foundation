"""
43 - Mixed-source audit: does pooling internal181 + cross321 help?

What this does:
    Scripts 41 and 42 showed that tree ensemble -> theta -> ODE works
    on cross321 and cleaned internal181 separately. This script tests
    whether mixing the two data sources improves target-source CV.

Protocol:
    Use only descriptor columns that both datasets share semantically:

        LA/GA, Polymer MW, Initial drug/polymer ratio, DLC,
        Drug MW, Drug TPSA, Drug LogP, plus Q(1,3,5,7).

    For each target source and CV scheme:

        target_only:
            train = target train fold
            test  = target test fold

        augmented:
            train = target train fold + all curves from the other source
            test  = target test fold

    The test set is always only the target source. This asks whether the
    other dataset is useful training augmentation, not whether pooled
    random CV looks good.

Outputs:
    outputs/43_mixed_source_tree_theta_audit/per_curve.csv
    outputs/43_mixed_source_tree_theta_audit/summary.csv
    outputs/43_mixed_source_tree_theta_audit/summary.txt
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

from encoder import parse_dp_group  # noqa: E402
from simulator import PLGABiphasic  # noqa: E402


DATASET_COL = "dataset"
FID_COL = "fid"
TIME_COL = "Time"
Y_COL = "Release"
INTERNAL_FID_COL = "Experimental_index"
DP_GROUP_COL = "DP_Group"

COMMON_FEATURES = [
    "laga",
    "polymer_mw_da",
    "drug_polymer_ratio",
    "drug_loading_frac",
    "drug_mw",
    "drug_tpsa",
    "drug_logp",
]
INTERNAL_FEATURE_MAP = {
    "LA/GA": "laga",
    "Polymer_MW": "polymer_mw_da",
    "Initial D/M ratio": "drug_polymer_ratio",
    "DLC": "drug_loading_frac",
    "Drug_Mw": "drug_mw",
    "Drug_TPSA": "drug_tpsa",
    "Drug_LogP": "drug_logp",
}
CROSS_FEATURE_MAP = {
    "LA/GA": "laga",
    "Polymer MW": "polymer_mw_da",
    "Initial Drug-to-Polymer Ratio": "drug_polymer_ratio",
    "Drug Loading Capacity": "drug_loading_frac",
    "Drug MW": "drug_mw",
    "Drug TPSA": "drug_tpsa",
    "Drug LogP": "drug_logp",
}


@dataclass
class CurveRecord:
    source: str
    fid: int
    drug_group: object
    polymer_group: object
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


def _load_internal(
    data_csv: Path,
    full_fit_bank: Path,
    fit_filter_r2: float,
    t_max: float,
    min_obs: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int], CurveRecord]]:
    raw = pd.read_csv(data_csv)
    raw[Y_COL] = raw[Y_COL].astype(float).clip(0.0, 1.0)
    raw = raw[raw[TIME_COL].astype(float) <= t_max].copy()
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

    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    bank = pd.read_csv(full_fit_bank)
    bank = bank[
        (bank[DATASET_COL] == "internal181")
        & np.isfinite(bank["full_r2"])
        & (bank["full_r2"] >= fit_filter_r2)
    ].copy()
    bank = bank[[FID_COL, *param_names]].rename(columns={FID_COL: "fid"})

    desc = cleaned.drop_duplicates(INTERNAL_FID_COL).rename(columns=INTERNAL_FEATURE_MAP)
    desc = desc.rename(columns={INTERNAL_FID_COL: "fid"})
    desc = desc[["fid", DP_GROUP_COL, *COMMON_FEATURES]].merge(bank, on="fid", how="inner")
    desc["source"] = "internal181"
    desc["uid"] = desc["source"] + ":" + desc["fid"].astype(str)

    curve_map: dict[tuple[str, int], CurveRecord] = {}
    keep: list[int] = []
    for fid in desc["fid"].astype(int).tolist():
        g = cleaned[cleaned[INTERNAL_FID_COL].astype(int) == fid].sort_values(TIME_COL)
        t_obs = g[TIME_COL].to_numpy(dtype=float)
        q_obs = g[Y_COL].to_numpy(dtype=float)
        if len(t_obs) < min_obs:
            continue
        drug, polymer = parse_dp_group(str(g[DP_GROUP_COL].iloc[0]))
        curve_map[("internal181", fid)] = CurveRecord(
            source="internal181",
            fid=fid,
            drug_group=drug,
            polymer_group=polymer,
            t_obs=t_obs,
            q_obs=q_obs,
        )
        keep.append(fid)
    desc = desc[desc["fid"].astype(int).isin(keep)].reset_index(drop=True)
    return desc, curve_map


def _load_cross(
    xlsx_path: Path,
    matched_fids_csv: Path,
    full_fit_bank: Path,
    fit_filter_r2: float,
    t_max: float,
    min_obs: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int], CurveRecord]]:
    matched_fids = set(pd.read_csv(matched_fids_csv)["Formulation_Index"].astype(int))
    raw = pd.read_excel(xlsx_path)
    raw = raw.rename(columns={"Formulation Index": "fid"})
    raw[Y_COL] = raw[Y_COL].astype(float).clip(0.0, 1.0)
    raw = raw[raw["fid"].astype(int).isin(matched_fids)].copy()
    raw = raw[raw[TIME_COL].astype(float) <= t_max].copy()
    cleaned = (
        raw.groupby(["fid", TIME_COL], as_index=False, sort=False)
        .agg({
            **{
                c: "first"
                for c in raw.columns
                if c not in {TIME_COL, Y_COL}
            },
            Y_COL: "mean",
        })
        .sort_values(["fid", TIME_COL])
        .reset_index(drop=True)
    )

    sim = PLGABiphasic()
    param_names = list(sim.param_names)
    bank = pd.read_csv(full_fit_bank)
    bank = bank[
        (bank[DATASET_COL] == "cross321")
        & np.isfinite(bank["full_r2"])
        & (bank["full_r2"] >= fit_filter_r2)
    ].copy()
    bank = bank[[FID_COL, *param_names]].rename(columns={FID_COL: "fid"})

    desc = cleaned.drop_duplicates("fid").rename(columns=CROSS_FEATURE_MAP)
    desc = desc[["fid", *COMMON_FEATURES]].merge(bank, on="fid", how="inner")
    desc["source"] = "cross321"
    desc["uid"] = desc["source"] + ":" + desc["fid"].astype(str)

    # Unit alignment to internal181 schema.
    desc["polymer_mw_da"] = desc["polymer_mw_da"].astype(float) * 1000.0
    desc["drug_loading_frac"] = desc["drug_loading_frac"].astype(float) / 100.0

    curve_map: dict[tuple[str, int], CurveRecord] = {}
    keep: list[int] = []
    for fid in desc["fid"].astype(int).tolist():
        g = cleaned[cleaned["fid"].astype(int) == fid].sort_values(TIME_COL)
        t_obs = g[TIME_COL].to_numpy(dtype=float)
        q_obs = g[Y_COL].to_numpy(dtype=float)
        if len(t_obs) < min_obs:
            continue
        first = g.iloc[0]
        drug_group = (
            float(first["Drug MW"]),
            float(first["Drug TPSA"]),
            float(first["Drug LogP"]),
        )
        polymer_group = (
            float(first["Polymer MW"]),
            float(first["LA/GA"]),
        )
        curve_map[("cross321", fid)] = CurveRecord(
            source="cross321",
            fid=fid,
            drug_group=drug_group,
            polymer_group=polymer_group,
            t_obs=t_obs,
            q_obs=q_obs,
        )
        keep.append(fid)
    desc = desc[desc["fid"].astype(int).isin(keep)].reset_index(drop=True)
    return desc, curve_map


def _fit_predict_raw(model: object, x: np.ndarray, theta: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    model.fit(x[tr], theta[tr])
    return model.predict(x[te])


def _fit_predict_ztheta(model: object, x: np.ndarray, theta: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    mu = theta[tr].mean(axis=0, keepdims=True)
    sd = theta[tr].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    model.fit(x[tr], (theta[tr] - mu) / sd)
    return model.predict(x[te]) * sd + mu


def _make_rf(args: argparse.Namespace, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )


def _make_et(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )


def _predict_methods(
    args: argparse.Namespace,
    x: np.ndarray,
    theta: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    fold_idx: int,
) -> dict[str, np.ndarray]:
    pred: dict[str, np.ndarray] = {}
    pred["RF_raw"] = _fit_predict_raw(_make_rf(args, args.seed + fold_idx), x, theta, tr, te)
    pred["RF_ztheta"] = _fit_predict_ztheta(_make_rf(args, args.seed + 100 + fold_idx), x, theta, tr, te)
    pred["ET_raw"] = _fit_predict_raw(_make_et(args, args.seed + 200 + fold_idx), x, theta, tr, te)
    pred["ET_ztheta"] = _fit_predict_ztheta(_make_et(args, args.seed + 300 + fold_idx), x, theta, tr, te)
    pred["RF_ET_zavg"] = 0.5 * (pred["RF_ztheta"] + pred["ET_ztheta"])
    return pred


def _decode_rows(
    sim: PLGABiphasic,
    target_source: str,
    train_mode: str,
    scheme: str,
    fold: int,
    df: pd.DataFrame,
    te: np.ndarray,
    pred_by_method: dict[str, np.ndarray],
    curve_map: dict[tuple[str, int], CurveRecord],
    early_times: np.ndarray,
    early_q: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    method_names = list(pred_by_method.keys())
    for local_i, j in enumerate(te):
        source = str(df.loc[j, "source"])
        fid = int(df.loc[j, "fid"])
        curve = curve_map[(source, fid)]
        row: dict[str, object] = {
            "target_source": target_source,
            "train_mode": train_mode,
            "scheme": scheme,
            "fold": fold,
            "source": source,
            "fid": fid,
        }
        early_rmse: dict[str, float] = {}
        for method in method_names:
            theta = _clip_theta(pred_by_method[method][local_i], lows, highs)
            row[f"r2_{method}"] = _r2(curve.q_obs, sim.simulate_numpy(theta, curve.t_obs))
            pred_early = sim.simulate_numpy(theta, early_times)
            early_rmse[method] = _rmse(early_q[j].astype(float), pred_early)

        pool = ["RF_ztheta", "ET_ztheta", "RF_ET_zavg"]
        selected = min(pool, key=lambda m: early_rmse[m])
        row["early_select_method"] = selected
        row["r2_early_select"] = row[f"r2_{selected}"]
        row["early_select_rmse"] = early_rmse[selected]
        rows.append(row)
    return rows


def _summary(per_curve: pd.DataFrame) -> pd.DataFrame:
    method_cols = [c for c in per_curve.columns if c.startswith("r2_")]
    rows: list[dict[str, object]] = []
    group_cols = ["target_source", "train_mode", "scheme"]
    for keys, sub in per_curve.groupby(group_cols, sort=False):
        target_source, train_mode, scheme = keys
        for col in method_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
            rows.append({
                "target_source": target_source,
                "train_mode": train_mode,
                "scheme": scheme,
                "method": col.removeprefix("r2_"),
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p10": float(np.percentile(vals, 10)),
                "frac_above_0.9": float(np.mean(vals >= 0.9)),
                "frac_above_0": float(np.mean(vals >= 0.0)),
            })
    return pd.DataFrame(rows)


def _add_strict_overlap_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drug_desc_key"] = list(zip(
        out["drug_mw"].round(6),
        out["drug_tpsa"].round(6),
        out["drug_logp"].round(6),
    ))
    out["poly_desc_key"] = list(zip(
        out["polymer_mw_da"].round(3),
        out["laga"].round(6),
    ))
    out["form_desc_key"] = list(zip(*[out[c].round(6) for c in COMMON_FEATURES]))
    return out


def _strict_aux_indices(
    df: pd.DataFrame,
    aux_idx: np.ndarray,
    target_te: np.ndarray,
    scheme_name: str,
) -> np.ndarray:
    """Remove aux rows that share the held-out OOD key with target test rows.

    This is stricter than the main augmented protocol. It asks whether the
    gain remains when the other source cannot contain the held-out drug or
    polymer descriptor used to define the target OOD split.
    """
    if scheme_name == "group_by_drug":
        held_keys = set(df.loc[target_te, "drug_desc_key"])
        keep = ~df.loc[aux_idx, "drug_desc_key"].isin(held_keys).to_numpy()
        return aux_idx[keep]
    if scheme_name == "group_by_polymer":
        held_keys = set(df.loc[target_te, "poly_desc_key"])
        keep = ~df.loc[aux_idx, "poly_desc_key"].isin(held_keys).to_numpy()
        return aux_idx[keep]
    # For random CV, remove exact shared formulation descriptors. This is
    # mostly a sanity guard; random CV is not the OOD claim.
    held_keys = set(df.loc[target_te, "form_desc_key"])
    keep = ~df.loc[aux_idx, "form_desc_key"].isin(held_keys).to_numpy()
    return aux_idx[keep]


def _write_leakage_audit(
    df: pd.DataFrame,
    out_path: Path,
    n_folds: int,
    seed: int,
) -> None:
    lines = [
        "=== 43 leakage / overlap audit ===",
        "",
        "Global cross-source exact overlaps:",
    ]
    for key in ["drug_desc_key", "poly_desc_key", "form_desc_key", "x_full_key"]:
        internal_keys = set(df[df["source"] == "internal181"][key])
        cross_keys = set(df[df["source"] == "cross321"][key])
        lines.append(f"  {key:<18}: {len(internal_keys & cross_keys)}")

    rows: list[dict[str, object]] = []
    for target_source in ["cross321", "internal181"]:
        target_idx = df.index[df["source"] == target_source].to_numpy()
        aux_idx = df.index[df["source"] != target_source].to_numpy()
        schemes: list[tuple[str, object | None]] = [
            ("random_5fold", None),
            ("group_by_drug", df.loc[target_idx, "drug_group"].to_numpy(dtype=object)),
            ("group_by_polymer", df.loc[target_idx, "polymer_group"].to_numpy(dtype=object)),
        ]
        for scheme_name, groups in schemes:
            if scheme_name == "random_5fold":
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = list(splitter.split(target_idx))
            else:
                n_unique = pd.Series(groups).nunique()
                splitter = GroupKFold(n_splits=min(n_folds, int(n_unique)))
                splits = list(splitter.split(target_idx, groups=groups))
            for fold_idx, (tr_local, te_local) in enumerate(splits):
                target_tr = target_idx[tr_local]
                target_te = target_idx[te_local]
                strict_aux = _strict_aux_indices(df, aux_idx, target_te, scheme_name)
                for mode, tr in [
                    ("target_only", target_tr),
                    ("augmented", np.concatenate([target_tr, aux_idx])),
                    ("augmented_strict", np.concatenate([target_tr, strict_aux])),
                ]:
                    train_uids = set(df.loc[tr, "uid"])
                    test_uids = set(df.loc[target_te, "uid"])
                    rows.append({
                        "target_source": target_source,
                        "scheme": scheme_name,
                        "fold": fold_idx,
                        "mode": mode,
                        "uid_overlap": len(train_uids & test_uids),
                        "x_full_overlap": len(
                            set(df.loc[tr, "x_full_key"])
                            & set(df.loc[target_te, "x_full_key"])
                        ),
                        "form_desc_overlap": len(
                            set(df.loc[tr, "form_desc_key"])
                            & set(df.loc[target_te, "form_desc_key"])
                        ),
                        "drug_desc_overlap": len(
                            set(df.loc[tr, "drug_desc_key"])
                            & set(df.loc[target_te, "drug_desc_key"])
                        ),
                        "poly_desc_overlap": len(
                            set(df.loc[tr, "poly_desc_key"])
                            & set(df.loc[target_te, "poly_desc_key"])
                        ),
                    })
    audit = pd.DataFrame(rows)
    lines.extend([
        "",
        "Max fold-level overlaps by target / scheme / mode:",
        audit.groupby(["target_source", "scheme", "mode"])[
            [
                "uid_overlap",
                "x_full_overlap",
                "form_desc_overlap",
                "drug_desc_overlap",
                "poly_desc_overlap",
            ]
        ].max().to_string(),
        "",
        "Interpretation:",
        "  uid_overlap and x_full_overlap must be zero for hard leakage checks.",
        "  descriptor overlaps are reported because they can weaken strict OOD claims.",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal-data", type=Path, default=Path("data/Dataset_17_feat_augmented.csv"))
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
    ap.add_argument("--out", type=Path, default=Path("outputs/43_mixed_source_tree_theta_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sim = PLGABiphasic()
    prior = sim.prior().base_dist
    lows = prior.low.numpy().astype(np.float64)
    highs = prior.high.numpy().astype(np.float64)
    param_names = list(sim.param_names)

    internal, internal_curves = _load_internal(
        args.internal_data,
        args.full_fit_bank,
        args.fit_filter_r2,
        args.t_grid_max_days,
        args.min_obs,
    )
    cross, cross_curves = _load_cross(
        args.cross_doi_data,
        args.matched_fids_csv,
        args.full_fit_bank,
        args.fit_filter_r2,
        args.t_grid_max_days,
        args.min_obs,
    )
    curve_map = {**internal_curves, **cross_curves}
    df = pd.concat([internal, cross], ignore_index=True)

    early_times = np.array(args.early_times, dtype=float)
    early_q = np.stack([
        _interp_at(
            curve_map[(str(row.source), int(row.fid))].t_obs,
            curve_map[(str(row.source), int(row.fid))].q_obs,
            early_times,
        )
        for row in df.itertuples(index=False)
    ]).astype(np.float32)
    df["early_key"] = [tuple(np.round(v, 6)) for v in early_q]
    x = np.concatenate([df[COMMON_FEATURES].to_numpy(dtype=np.float32), early_q], axis=1)
    theta = df[param_names].to_numpy(dtype=np.float32)
    df["drug_group"] = [
        curve_map[(str(row.source), int(row.fid))].drug_group
        for row in df.itertuples(index=False)
    ]
    df["polymer_group"] = [
        curve_map[(str(row.source), int(row.fid))].polymer_group
        for row in df.itertuples(index=False)
    ]
    df = _add_strict_overlap_keys(df)
    df["x_full_key"] = list(zip(df["form_desc_key"], df["early_key"]))
    _write_leakage_audit(df, args.out / "leakage_audit.txt", args.n_folds, args.seed)

    rows: list[dict[str, object]] = []
    for target_source in ["cross321", "internal181"]:
        target_idx = df.index[df["source"] == target_source].to_numpy()
        aux_idx = df.index[df["source"] != target_source].to_numpy()
        print(
            f"[43] target={target_source} n_target={len(target_idx)} n_aux={len(aux_idx)}",
            flush=True,
        )
        target_drugs = df.loc[target_idx, "drug_group"].to_numpy(dtype=object)
        target_polymers = df.loc[target_idx, "polymer_group"].to_numpy(dtype=object)
        schemes: list[tuple[str, object | None]] = [
            ("random_5fold", None),
            ("group_by_drug", target_drugs),
            ("group_by_polymer", target_polymers),
        ]
        for scheme_name, groups in schemes:
            print(f"\n[43] target={target_source} scheme={scheme_name}", flush=True)
            if scheme_name == "random_5fold":
                splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
                splits = list(splitter.split(target_idx))
            else:
                n_unique = pd.Series(groups).nunique()
                n_splits = min(args.n_folds, int(n_unique))
                if n_splits < 2:
                    print(f"[43] skipping {scheme_name}: only {n_unique} groups", flush=True)
                    continue
                splitter = GroupKFold(n_splits=n_splits)
                splits = list(splitter.split(target_idx, groups=groups))

            for fold_idx, (tr_local, te_local) in enumerate(splits):
                target_tr = target_idx[tr_local]
                target_te = target_idx[te_local]
                strict_aux = _strict_aux_indices(df, aux_idx, target_te, scheme_name)
                for train_mode, tr in [
                    ("target_only", target_tr),
                    ("augmented", np.concatenate([target_tr, aux_idx])),
                    ("augmented_strict", np.concatenate([target_tr, strict_aux])),
                ]:
                    pred = _predict_methods(args, x, theta, tr, target_te, fold_idx)
                    rows.extend(
                        _decode_rows(
                            sim=sim,
                            target_source=target_source,
                            train_mode=train_mode,
                            scheme=scheme_name,
                            fold=fold_idx,
                            df=df,
                            te=target_te,
                            pred_by_method=pred,
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
    summary.to_csv(args.out / "summary.csv", index=False)

    lines = [
        "=== 43 -- mixed-source tree theta audit ===",
        "",
        f"internal181 curves : {len(internal)}",
        f"cross321 curves    : {len(cross)}",
        f"features           : {len(COMMON_FEATURES)} shared descriptors + {len(early_times)} early Q",
        f"n_estimators       : {args.n_estimators}",
        "",
        "--- best median R^2 per target/source/scheme/train mode ---",
    ]
    for target_source in ["cross321", "internal181"]:
        lines.append(f"  target={target_source}:")
        sub_target = summary[summary["target_source"] == target_source]
        for scheme_name in ["random_5fold", "group_by_drug", "group_by_polymer"]:
            for train_mode in ["target_only", "augmented", "augmented_strict"]:
                sub = sub_target[
                    (sub_target["scheme"] == scheme_name)
                    & (sub_target["train_mode"] == train_mode)
                ].sort_values("median", ascending=False)
                if sub.empty:
                    continue
                row = sub.iloc[0]
                lines.append(
                    f"    {scheme_name:<16} {train_mode:<12} "
                    f"{row['method']:<14} median={row['median']:+.4f} "
                    f"p10={row['p10']:+.4f} frac_R2>=0={row['frac_above_0']:.3f}"
                )
    lines.append("")
    lines.append("--- interpretation guardrail ---")
    lines.append("  The test fold always comes from one target source. 'augmented'")
    lines.append("  means the other source is added to the training fold only; it is")
    lines.append("  not pooled random CV across both sources.")
    lines.append("  'augmented_strict' additionally removes aux-source rows sharing")
    lines.append("  the held-out drug descriptor or polymer descriptor with the target")
    lines.append("  test fold.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
