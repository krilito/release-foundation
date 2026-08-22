"""
34 - Partial-out test: regimes vs observation-window confounding.

What this does:
    Script 33 found regime structure that passes silhouette and external-
    descriptor checks. The strongest external separator is t_max
    (Kruskal-Wallis p << 1e-9). That is a confound: short-window curves
    can only identify early-time parameters, so 'regime' assignment may
    track *which params were identifiable*, not *which mechanism is
    active*. This script tests whether the regime->fast_regime signal
    survives after controlling for t_max and short_window.

    Two complementary tests:

    (A) t_max-stratified contingency: split cross321 curves into t_max
        terciles. Within each stratum, run regime x fast_regime
        chi-square (Fisher fallback for sparse cells). If the
        association vanishes within strata, t_max explains the regime ->
        kinetic-flag link. If it survives in at least one stratum, the
        regime carries independent mechanism information.

    (B) Logistic LR test: fit two nested logistic models with
        fast_regime as the outcome:
            null:  fast_regime ~ t_max + short_window
            full:  fast_regime ~ t_max + short_window + C(regime)
        The likelihood-ratio statistic on the regime dummies tests
        regime's incremental predictive power after the confound is
        partialled out. This is the cleanest inferential answer.

    Both are reported; we believe the result only when they agree.

Outputs:
    outputs/34_regime_confound_partial/stratum_contingency_R{r}.csv
    outputs/34_regime_confound_partial/stratum_tests.csv
    outputs/34_regime_confound_partial/logistic_lr.txt
    outputs/34_regime_confound_partial/regime_by_stratum_fast_regime.png
    outputs/34_regime_confound_partial/summary.txt

Anti-self-deception checklist:
    - Cross321 only; internal181 has no fast_regime flag.
    - R1 (n=2) excluded from chi-square; too small for any cell test.
    - Reported effects are conditional on t_max strata defined by terciles.
      Different bin choices may give slightly different p-values; the LR
      test does not depend on binning.
    - A positive result here only justifies regime-conditional modeling
      as a hypothesis. It does NOT prove mechanism heterogeneity; only
      that the regime label adds information beyond observation window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))


def _safe_chi2(tab: np.ndarray) -> tuple[float, float, int, str]:
    """Return (stat, p, dof, method)."""
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2 or tab.sum() == 0:
        return (float("nan"), float("nan"), 0, "skipped")
    # Fisher if 2x2 and any expected count < 5
    if tab.shape == (2, 2):
        try:
            chi2, p, dof, exp = chi2_contingency(tab)
            if (exp < 5).any():
                _, p_f = fisher_exact(tab)
                return (chi2, p_f, dof, "fisher_exact")
            return (chi2, p, dof, "chi2")
        except ValueError:
            return (float("nan"), float("nan"), 0, "skipped")
    # Generic chi-square; switch to monte-carlo if many small cells
    try:
        chi2, p, dof, exp = chi2_contingency(tab)
        n_small = int((exp < 5).sum())
        method = "chi2"
        if n_small > 0:
            # Monte-carlo p-value as a fallback diagnostic
            rng = np.random.default_rng(0)
            n_iter = 5000
            row_sums = tab.sum(axis=1)
            col_sums = tab.sum(axis=0)
            n_total = int(tab.sum())
            count = 0
            for _ in range(n_iter):
                sim = _random_contingency(row_sums, col_sums, n_total, rng)
                sim_chi2, _, _, _ = chi2_contingency(sim)
                if sim_chi2 >= chi2:
                    count += 1
            p_mc = (count + 1) / (n_iter + 1)
            method = f"chi2+mc(p_mc={p_mc:.4f})"
            # Use the larger p (more conservative) as headline
            p = max(p, p_mc)
        return (chi2, p, dof, method)
    except ValueError:
        return (float("nan"), float("nan"), 0, "skipped")


def _random_contingency(
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample contingency table preserving marginals (independent permutation)."""
    flat_rows = np.repeat(np.arange(len(row_sums)), row_sums)
    rng.shuffle(flat_rows)
    flat_cols = np.repeat(np.arange(len(col_sums)), col_sums)
    rng.shuffle(flat_cols)
    sim = np.zeros((len(row_sums), len(col_sums)), dtype=int)
    for r_i, c_i in zip(flat_rows, flat_cols):
        sim[r_i, c_i] += 1
    return sim


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
    ap.add_argument("--exclude-regimes", type=int, nargs="*", default=[1],
                    help="regimes too small to include (default: R1)")
    ap.add_argument("--n-strata", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("outputs/34_regime_confound_partial"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df_r = pd.read_csv(args.regime_csv)
    df_o = pd.read_csv(args.oracle_csv)

    # Keep cross321 only for fast_regime / short_window analyses
    df = df_r.merge(
        df_o[["fid", "t_max", "fast_regime", "short_window", "boundary_hits"]],
        on="fid",
        how="left",
        suffixes=("", "_oracle"),
    )
    df = df[df["dataset"] == "cross321"].copy()
    df = df[~df["regime"].isin(args.exclude_regimes)].copy()
    df = df.dropna(subset=["t_max", "fast_regime"]).copy()
    df["fast_regime"] = df["fast_regime"].astype(bool)
    df["short_window"] = df["short_window"].astype(bool)

    # t_max strata (quantile bins)
    df["t_stratum"] = pd.qcut(
        df["t_max"], q=args.n_strata,
        labels=[f"t{i+1}_of{args.n_strata}" for i in range(args.n_strata)],
    )

    # --- (A) Per-stratum chi-square -----------------------------------
    stratum_rows: list[dict[str, object]] = []
    regimes_used = sorted(df["regime"].unique().tolist())
    for stratum, sub in df.groupby("t_stratum", observed=True):
        tab = pd.crosstab(sub["regime"], sub["fast_regime"])
        # Drop columns/rows that are all zero (defensive)
        if tab.shape[1] < 2:
            stat, p, dof, method = (float("nan"), float("nan"), 0, "single_outcome_in_stratum")
            cv = float("nan")
        else:
            mat = tab.to_numpy()
            stat, p, dof, method = _safe_chi2(mat)
            n_s = int(mat.sum())
            cv = _cramers_v(stat, n_s, mat.shape[0], mat.shape[1]) if not np.isnan(stat) else float("nan")
        stratum_rows.append({
            "stratum": str(stratum),
            "n": int(sub.shape[0]),
            "t_max_min": float(sub["t_max"].min()),
            "t_max_max": float(sub["t_max"].max()),
            "n_regimes_present": int(tab.shape[0]),
            "frac_fast_regime": float(sub["fast_regime"].mean()),
            "chi2": float(stat) if not np.isnan(stat) else float("nan"),
            "dof": int(dof) if not np.isnan(stat) else 0,
            "p_value": float(p) if not np.isnan(p) else float("nan"),
            "method": str(method),
            "cramers_v": float(cv),
        })
        tab.to_csv(args.out / f"contingency_{stratum}_regime_x_fast.csv")
    stratum_df = pd.DataFrame(stratum_rows)
    stratum_df.to_csv(args.out / "stratum_tests.csv", index=False)

    # Stouffer combine across strata
    valid_p = stratum_df["p_value"].dropna().to_numpy()
    if len(valid_p) >= 2:
        # avoid log(0) and p=1
        valid_p_safe = np.clip(valid_p, 1e-12, 1 - 1e-12)
        z = -np.array([np.log(p) for p in valid_p_safe])
        # Stouffer requires Z scores; use one-sided z = norm.isf(p)
        from scipy.stats import norm
        z_scores = norm.isf(valid_p_safe)
        z_stouffer = float(z_scores.sum() / np.sqrt(len(z_scores)))
        p_stouffer = float(norm.sf(z_stouffer))
    else:
        z_stouffer = float("nan")
        p_stouffer = float("nan")

    # Unconditional contingency (control)
    tab_all = pd.crosstab(df["regime"], df["fast_regime"])
    chi2_all, p_all, dof_all, method_all = _safe_chi2(tab_all.to_numpy())
    cv_all = _cramers_v(chi2_all, int(tab_all.values.sum()), tab_all.shape[0], tab_all.shape[1])

    # --- (B) Logistic regression LR test -----------------------------
    # Force float for statsmodels formula handling
    df_logit = df.copy()
    df_logit["fast"] = df_logit["fast_regime"].astype(int)
    df_logit["short"] = df_logit["short_window"].astype(int)
    df_logit["regime_cat"] = df_logit["regime"].astype("category")

    # Null: only confounders
    try:
        m_null = smf.logit("fast ~ t_max + short", data=df_logit).fit(disp=False)
        m_full = smf.logit("fast ~ t_max + short + C(regime_cat)", data=df_logit).fit(disp=False)
        lr_stat = 2.0 * (m_full.llf - m_null.llf)
        lr_dof = int(m_full.df_model - m_null.df_model)
        lr_p = float(chi2_dist.sf(lr_stat, lr_dof))
        logit_lines = [
            "Logistic LR test: regime adds beyond t_max + short_window",
            "",
            "--- null model: fast ~ t_max + short ---",
            str(m_null.summary()),
            "",
            "--- full model: fast ~ t_max + short + C(regime) ---",
            str(m_full.summary()),
            "",
            f"LR statistic       : {lr_stat:.3f}",
            f"degrees of freedom : {lr_dof}",
            f"LR p-value         : {lr_p:.6f}",
        ]
        logit_ok = True
    except Exception as e:  # pragma: no cover - statsmodels can fail on separation
        lr_stat = float("nan")
        lr_dof = 0
        lr_p = float("nan")
        logit_lines = [f"Logistic LR test failed: {e}"]
        logit_ok = False

    (args.out / "logistic_lr.txt").write_text("\n".join(logit_lines), encoding="utf-8")

    # --- Visual: regime composition x stratum, colored by fast_regime fraction
    fig, axes = plt.subplots(1, args.n_strata, figsize=(4.2 * args.n_strata, 4.2),
                             constrained_layout=True, sharey=True)
    if args.n_strata == 1:
        axes = [axes]
    for ax, (stratum, sub) in zip(axes, df.groupby("t_stratum", observed=True)):
        # Per regime: fraction fast_regime, bar
        grp = sub.groupby("regime")["fast_regime"].agg(["mean", "count"]).sort_index()
        ax.bar([f"R{int(r)}" for r in grp.index], grp["mean"].to_numpy(),
               color="tab:orange")
        for i, (r, row) in enumerate(grp.iterrows()):
            ax.text(i, min(row["mean"] + 0.03, 0.97),
                    f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{stratum}  (n={int(sub.shape[0])}, t_max {sub['t_max'].min():.1f}-{sub['t_max'].max():.1f})")
        ax.set_ylabel("fraction fast_regime=True")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Fast-regime fraction per regime, within t_max strata (cross321)",
                 fontsize=12)
    fig.savefig(args.out / "regime_by_stratum_fast_regime.png", dpi=150)
    plt.close(fig)

    # --- summary.txt --------------------------------------------------
    lines = [
        "=== 34 -- regime / observation-window confound partial-out ===",
        "",
        f"regime_csv : {args.regime_csv}",
        f"oracle_csv : {args.oracle_csv}",
        f"n_curves used (cross321, excluding regimes {args.exclude_regimes}) : {len(df)}",
        f"n strata   : {args.n_strata} (quantile bins of t_max)",
        "",
        "--- (A) Per-stratum regime x fast_regime contingency ---",
        f"  {'stratum':<14}  {'n':>4}  {'t_min':>7}  {'t_max':>7}  {'regs':>5}  "
        f"{'chi2':>8}  {'dof':>3}  {'p':>10}  {'V':>5}  method",
    ]
    for _, r in stratum_df.iterrows():
        lines.append(
            f"  {r['stratum']:<14}  {int(r['n']):>4}  {r['t_max_min']:>7.2f}  "
            f"{r['t_max_max']:>7.2f}  {int(r['n_regimes_present']):>5}  "
            f"{r['chi2']:>8.3f}  {int(r['dof']):>3}  {r['p_value']:>10.4f}  "
            f"{r['cramers_v']:>5.2f}  {r['method']}"
        )
    lines.append("")
    lines.append(f"Unconditional (pooled): chi2={chi2_all:.3f}, dof={dof_all}, "
                 f"p={p_all:.6f}, Cramer V={cv_all:.2f}")
    lines.append(f"Stouffer-combined per-stratum p : {p_stouffer:.6f} (z={z_stouffer:.3f})")
    lines.append("")
    lines.append("--- (B) Logistic LR test: fast ~ t_max + short  vs  + C(regime) ---")
    if logit_ok:
        lines.append(f"  LR stat = {lr_stat:.3f}  (df={lr_dof}, p={lr_p:.6f})")
    else:
        lines.append("  failed; see logistic_lr.txt")
    lines.append("")

    # Verdict
    n_stratum_pass = int((stratum_df["p_value"] < 0.01).sum())
    n_stratum_hint = int(((stratum_df["p_value"] >= 0.01) & (stratum_df["p_value"] < 0.05)).sum())
    lr_pass = bool(logit_ok and (not np.isnan(lr_p)) and lr_p < 0.01)
    stouffer_pass = bool((not np.isnan(p_stouffer)) and p_stouffer < 0.01)

    lines.append("--- verdict ---")
    lines.append(f"  per-stratum p<0.01 count   : {n_stratum_pass} of {len(stratum_df)}")
    lines.append(f"  per-stratum p<0.05 count   : {n_stratum_hint + n_stratum_pass} of {len(stratum_df)}")
    lines.append(f"  Stouffer combined p<0.01   : {stouffer_pass}")
    lines.append(f"  Logistic LR p<0.01         : {lr_pass}")
    lines.append("")

    if lr_pass and (n_stratum_pass >= 1 or stouffer_pass):
        lines.append("  REGIME CARRIES SIGNAL BEYOND t_max+short_window.")
        lines.append("  Regime-conditional parameterization is defensible: regime label")
        lines.append("  adds predictive power for fast_regime over and above observation")
        lines.append("  window. This is necessary but not sufficient evidence for mechanism")
        lines.append("  heterogeneity (we have not yet ruled out other latent confounds, e.g.")
        lines.append("  formulation descriptors).")
    elif lr_pass and n_stratum_pass == 0:
        lines.append("  MIXED. Pooled LR significant but no single stratum is. Could be a")
        lines.append("  weak signal that only emerges in aggregate, or could be t_max")
        lines.append("  binning artifact. Re-run with finer or alternative strata before")
        lines.append("  drawing model-design conclusions.")
    elif (not lr_pass) and (n_stratum_pass >= 1):
        lines.append("  MIXED. A single stratum shows the effect but logistic LR does not.")
        lines.append("  Treat as a hint, not a claim. Investigate that stratum manually.")
    else:
        lines.append("  REGIME SIGNAL ABSORBED BY t_max+short_window.")
        lines.append("  After controlling for observation window, regime no longer predicts")
        lines.append("  fast_regime. The 'regime structure' in script 33 is most likely an")
        lines.append("  identifiability artifact, not a mechanism-heterogeneity signal.")
        lines.append("  Next step should NOT be regime-conditional parameterization. Instead,")
        lines.append("  the model design should condition on observation window directly, and")
        lines.append("  the mechanism story should be revisited (e.g., test whether script 30's")
        lines.append("  local Jacobian directions cluster meaningfully without invoking")
        lines.append("  discrete regimes).")

    lines.append("")
    lines.append("--- caveats ---")
    lines.append("  - cross321 only; no fast_regime flag on internal181.")
    lines.append("  - R1 (n=2) excluded. R2..R6 vary in size; small strata may have low power.")
    lines.append("  - 't_max' captures only one aspect of identifiability (window length).")
    lines.append("    A negative LR result does not prove regimes are pure artifact; it")
    lines.append("    proves the regime label is not informative beyond the variables tested.")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
