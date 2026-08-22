"""
35a - Formulation-confound partial-out for regime structure.

What this does:
    Script 34 showed that regime label predicts fast_regime beyond
    t_max + short_window (LR p=7e-4 on cross321). That ruled out
    observation-window-only confounding. This script extends the
    partial-out to formulation descriptors (drug + polymer + process
    metadata). If regime still adds incremental predictive power on top
    of t_max + short_window + 10 formulation columns, then regime is
    not purely a re-encoding of formulation either.

    Nested logistic-regression progression:
        M0:  fast ~ 1
        M1:  M0 + t_max + short_window         (the previous null)
        M2:  M1 + 10 formulation descriptors   (the strong null here)
        M3:  M2 + C(regime)                    (test whether regime adds)
    Three LR statistics:
        M1 vs M0: pure observation-window signal
        M2 vs M1: formulation signal net of window
        M3 vs M2: regime signal net of window AND formulation  <-- key

    Auxiliary: multinomial logistic regression of regime on formulation
    alone. If formulation perfectly classifies regime, regime is a
    re-labeling. If accuracy is modest, regime encodes something
    formulation does not.

Why this exists:
    The user's working hypothesis is that the 9 ODE parameters share a
    backbone plus regime-specific modifiers (like a chemistry skeleton
    plus functional groups). To defend regime-conditional modeling we
    have to rule out that the modifier-set is just driven by formulation
    descriptors that we should have used directly.

Anti-self-deception:
    - With ~216 cross321 curves, ~70 events, and ~16 predictors at M3,
      the LR test is in mild overfit territory. We report log-likelihood
      and AIC alongside p-values so the user can see whether M3 buys
      its complexity.
    - Formulation descriptors are standardized. We exclude near-constant
      columns (std < 1e-6) defensively.
    - The multinomial classifier uses 5-fold CV accuracy; chance level
      is reported explicitly (5 regimes, weighted by frequency).

Outputs:
    outputs/35a_regime_formulation_partial/coef_tables.txt
    outputs/35a_regime_formulation_partial/nested_lr_table.csv
    outputs/35a_regime_formulation_partial/regime_from_formulation_cv.txt
    outputs/35a_regime_formulation_partial/summary.txt
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2 as chi2_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FORMULATION_COL_MAP: dict[str, str] = {
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


def _build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    df_r = pd.read_csv(args.regime_csv)
    df_o = pd.read_csv(args.oracle_csv)
    df_meta = pd.read_excel(args.metadata_xlsx, sheet_name=0)

    # cross321 only
    df_r = df_r[df_r["dataset"] == "cross321"].copy()
    df_r = df_r[~df_r["regime"].isin(args.exclude_regimes)].copy()

    # 27a oracle: t_max, fast_regime, short_window
    df = df_r.merge(
        df_o[["fid", "t_max", "fast_regime", "short_window", "boundary_hits"]],
        on="fid",
        how="left",
    )

    # Metadata: one row per Formulation Index
    df_meta_first = df_meta.drop_duplicates(subset="Formulation Index", keep="first")[
        ["Formulation Index"] + list(FORMULATION_COL_MAP.keys())
    ].rename(columns={"Formulation Index": "fid", **FORMULATION_COL_MAP})

    df = df.merge(df_meta_first, on="fid", how="left")

    # Drop rows with missing essentials
    essential = ["t_max", "fast_regime", "short_window"] + list(FORMULATION_COL_MAP.values())
    n_before = len(df)
    df = df.dropna(subset=essential).copy()
    n_after = len(df)
    if n_after < n_before:
        warnings.warn(f"dropped {n_before - n_after} rows with missing essentials")

    df["fast_regime"] = df["fast_regime"].astype(int)
    df["short_window"] = df["short_window"].astype(int)
    df["regime_cat"] = df["regime"].astype("category")

    # Standardize formulation cols; drop near-constant
    form_cols_raw = list(FORMULATION_COL_MAP.values())
    keep_cols: list[str] = []
    scaler_cols: list[str] = []
    for c in form_cols_raw:
        std = float(df[c].std())
        if std < 1e-6:
            warnings.warn(f"dropping near-constant column {c} (std={std:.2e})")
            continue
        keep_cols.append(c)
    scaler = StandardScaler()
    df[keep_cols] = scaler.fit_transform(df[keep_cols])
    scaler_cols = keep_cols
    return df, scaler_cols


def _fit_logit(formula: str, df: pd.DataFrame) -> "smf.logit":
    return smf.logit(formula, data=df).fit(disp=False, maxiter=200)


def _lr_test(m_small, m_big) -> tuple[float, int, float]:
    lr = 2.0 * (m_big.llf - m_small.llf)
    df_ = int(m_big.df_model - m_small.df_model)
    if df_ <= 0:
        return (float("nan"), 0, float("nan"))
    p = float(chi2_dist.sf(lr, df_))
    return (float(lr), df_, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-csv",
        type=Path,
        default=Path("outputs/33_active_set_regimes/regime_assignments.csv"),
    )
    ap.add_argument(
        "--oracle-csv",
        type=Path,
        default=Path("outputs/27a_ode_expressivity_audit/per_curve.csv"),
    )
    ap.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=Path(
            r"D:\chemical-world-model-v0\datset\321PLGA\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\mp_dataset_processed.xlsx"
        ),
    )
    ap.add_argument("--exclude-regimes", type=int, nargs="*", default=[1])
    ap.add_argument("--out", type=Path, default=Path("outputs/35a_regime_formulation_partial"))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df, form_cols = _build_dataset(args)
    n = len(df)
    n_pos = int(df["fast_regime"].sum())
    n_regimes = int(df["regime"].nunique())

    form_terms = " + ".join(form_cols)

    # Nested models
    f0 = "fast_regime ~ 1"
    f1 = "fast_regime ~ t_max + short_window"
    f2 = f"fast_regime ~ t_max + short_window + {form_terms}"
    f3 = f"fast_regime ~ t_max + short_window + {form_terms} + C(regime_cat)"

    m0 = _fit_logit(f0, df)
    m1 = _fit_logit(f1, df)
    m2 = _fit_logit(f2, df)
    m3 = _fit_logit(f3, df)

    lr01 = _lr_test(m0, m1)
    lr12 = _lr_test(m1, m2)
    lr23 = _lr_test(m2, m3)

    nested = pd.DataFrame([
        {"transition": "M0 -> M1 (+ t_max + short_window)",
         "added_terms": "t_max, short_window",
         "lr_stat": lr01[0], "df": lr01[1], "p_value": lr01[2],
         "aic_small": m0.aic, "aic_big": m1.aic, "ll_small": m0.llf, "ll_big": m1.llf},
        {"transition": "M1 -> M2 (+ formulation)",
         "added_terms": ", ".join(form_cols),
         "lr_stat": lr12[0], "df": lr12[1], "p_value": lr12[2],
         "aic_small": m1.aic, "aic_big": m2.aic, "ll_small": m1.llf, "ll_big": m2.llf},
        {"transition": "M2 -> M3 (+ regime)",
         "added_terms": "C(regime_cat)",
         "lr_stat": lr23[0], "df": lr23[1], "p_value": lr23[2],
         "aic_small": m2.aic, "aic_big": m3.aic, "ll_small": m2.llf, "ll_big": m3.llf},
    ])
    nested.to_csv(args.out / "nested_lr_table.csv", index=False)

    # Coefficient tables
    coef_lines = [
        "==== Nested logistic regression coefficient tables ====",
        "",
        f"n={n}, n_fast_regime={n_pos} ({n_pos / n:.2%}), n_regimes={n_regimes}",
        f"formulation cols (standardized): {form_cols}",
        "",
        "-- M1 (window only) --",
        str(m1.summary()),
        "",
        "-- M2 (window + formulation) --",
        str(m2.summary()),
        "",
        "-- M3 (window + formulation + regime) --",
        str(m3.summary()),
        "",
    ]
    (args.out / "coef_tables.txt").write_text("\n".join(coef_lines), encoding="utf-8")

    # Auxiliary: multinomial classifier of regime from formulation alone
    X = df[form_cols].to_numpy()
    y = df["regime"].astype(int).to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    chance_top = float(counts.max() / counts.sum())
    chance_balanced = float(1.0 / len(classes))

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_acc: list[float] = []
    cv_bal_acc: list[float] = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        # sklearn >=1.5 dropped multi_class; lbfgs handles multinomial by default
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0, random_state=args.seed,
        )
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        cv_acc.append(accuracy_score(y[te], yp))
        cv_bal_acc.append(balanced_accuracy_score(y[te], yp))
    cv_acc_mean = float(np.mean(cv_acc))
    cv_acc_std = float(np.std(cv_acc, ddof=1))
    cv_bal_mean = float(np.mean(cv_bal_acc))
    cv_bal_std = float(np.std(cv_bal_acc, ddof=1))

    aux_lines = [
        "==== Multinomial regime ~ formulation (5-fold CV) ====",
        "",
        f"n={n}, regimes={classes.tolist()}",
        f"class counts             : {dict(zip(classes.tolist(), counts.tolist()))}",
        f"top-class baseline       : {chance_top:.3f}",
        f"uniform-class baseline   : {chance_balanced:.3f}",
        "",
        f"CV accuracy              : {cv_acc_mean:.3f} +/- {cv_acc_std:.3f}",
        f"CV balanced accuracy     : {cv_bal_mean:.3f} +/- {cv_bal_std:.3f}",
        "",
        "Reading guide:",
        "  CV acc near top-class baseline -> formulation does NOT separate regimes",
        "  CV acc much above top-class baseline -> formulation does separate regimes,",
        "     i.e. regime may be partly a relabeling of formulation. Whether regime",
        "     adds NEW info is then decided by the M2 -> M3 LR test, not this CV.",
    ]
    (args.out / "regime_from_formulation_cv.txt").write_text("\n".join(aux_lines), encoding="utf-8")

    # ---- summary.txt ----
    lines = [
        "=== 35a -- regime / formulation confound partial-out ===",
        "",
        f"n_curves (cross321, excluding regimes {args.exclude_regimes}) : {n}",
        f"n_fast_regime=True : {n_pos} ({n_pos/n:.2%})",
        f"n_regimes used     : {n_regimes}",
        f"formulation cols   : {form_cols}",
        "",
        "--- nested LR table (fast_regime as outcome) ---",
        f"  {'transition':<38}  {'LR':>8}  {'df':>3}  {'p':>10}  {'AIC_small':>10}  {'AIC_big':>10}",
    ]
    for _, r in nested.iterrows():
        lines.append(
            f"  {r['transition']:<38}  {r['lr_stat']:>8.3f}  {int(r['df']):>3}  "
            f"{r['p_value']:>10.6f}  {r['aic_small']:>10.2f}  {r['aic_big']:>10.2f}"
        )

    delta_aic_23 = float(nested.iloc[2]["aic_small"] - nested.iloc[2]["aic_big"])
    lines.append("")
    lines.append(f"M3 vs M2 delta AIC : {delta_aic_23:+.2f}  "
                 f"({'M3 preferred' if delta_aic_23 > 0 else 'M2 preferred'})")
    lines.append("")

    # Regime ~ formulation
    lines.append("--- regime ~ formulation (multinomial CV) ---")
    lines.append(f"  CV accuracy          : {cv_acc_mean:.3f} +/- {cv_acc_std:.3f}")
    lines.append(f"  CV balanced accuracy : {cv_bal_mean:.3f} +/- {cv_bal_std:.3f}")
    lines.append(f"  top-class baseline   : {chance_top:.3f}")
    lines.append(f"  uniform baseline     : {chance_balanced:.3f}")
    lines.append("")

    # Verdict
    p_m2_m3 = float(nested.iloc[2]["p_value"])
    aic_pass = delta_aic_23 > 2.0  # ~AIC convention for "preferred"
    lr_pass = p_m2_m3 < 0.01

    lines.append("--- verdict ---")
    if lr_pass and aic_pass:
        lines.append("  REGIME ADDS BEYOND OBSERVATION WINDOW AND FORMULATION.")
        lines.append(f"  M2->M3 LR p={p_m2_m3:.4g}, dAIC={delta_aic_23:+.2f}. The regime label")
        lines.append("  contributes predictive power on top of t_max + short_window + 10")
        lines.append("  formulation descriptors. This rules out the strongest remaining")
        lines.append("  confound. Regime-conditional parameterization is now well supported.")
    elif lr_pass and not aic_pass:
        lines.append("  REGIME PASSES LR BUT NOT AIC.")
        lines.append(f"  M2->M3 p={p_m2_m3:.4g} but dAIC={delta_aic_23:+.2f} is marginal.")
        lines.append("  Likely overfitting: the regime-dummies are absorbing noise. Treat as")
        lines.append("  weak evidence; consider re-running with a smaller regime count or a")
        lines.append("  larger sample before claiming the result.")
    elif (not lr_pass) and aic_pass:
        lines.append("  REGIME PASSES AIC BUT NOT LR.")
        lines.append("  This combination is unusual and suggests boundary-case results.")
        lines.append("  Treat as inconclusive.")
    else:
        lines.append("  REGIME DOES NOT ADD BEYOND FORMULATION + WINDOW.")
        lines.append(f"  M2->M3 p={p_m2_m3:.4g}, dAIC={delta_aic_23:+.2f}. After conditioning")
        lines.append("  on formulation descriptors, regime is not informative. Likely the")
        lines.append("  regime structure in script 33 is a coarser projection of formulation")
        lines.append("  variation. Modeling implication: condition directly on formulation,")
        lines.append("  not on regime; B (regime <-> Jacobian alignment) becomes a sanity")
        lines.append("  check rather than a path to a new architecture.")
    lines.append("")

    # Caveats
    lines.append("--- caveats ---")
    lines.append("  - cross321 only; no fast_regime / formulation alignment on internal181.")
    lines.append("  - With n=216 events <=80, M3 has ~16 predictors -> mild overfit risk.")
    lines.append("    LL and AIC are reported so the user can audit.")
    lines.append("  - Formulation descriptors are standardized at dataset level; this can")
    lines.append("    underweight rare classes (e.g. solubility enhancer > 0 only in some)")
    lines.append("    but does not change the LR test outcome.")
    lines.append("  - A positive verdict here is necessary, not sufficient, evidence for")
    lines.append("    mechanism heterogeneity. Route B (regime vs script-30 Jacobian basis)")
    lines.append("    is still needed to connect 'regime adds info' to 'regime corresponds")
    lines.append("    to a distinct mechanism direction'.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
