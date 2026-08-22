"""
73 - Manuscript readiness matrix for the Nature-family route.

What this does:
    Consumes gate outputs from scripts 71, 72, and optional script 76,
    then turns them into a
    paper-facing claim/figure routing matrix:

      - What can be a main-text claim now?
      - What must be written as a caveat or supplementary result?
      - What is still missing before a Nature-family submission?

Outputs:
    outputs/73_manuscript_readiness_matrix/readiness_matrix.csv
    outputs/73_manuscript_readiness_matrix/figure_routing.csv
    outputs/73_manuscript_readiness_matrix/summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _status_label(status: str) -> str:
    return {
        "green": "GREEN",
        "yellow": "YELLOW",
        "red": "RED",
        "pending": "PENDING",
    }[status]


def _bool_count(s: pd.Series) -> tuple[int, int]:
    vals = s.astype(bool)
    return int(vals.sum()), int(len(vals))


def build_readiness(args: argparse.Namespace) -> pd.DataFrame:
    g1 = pd.read_csv(args.g1_rf_et)
    g1_lgbm = pd.read_csv(args.g1_lgbm)
    g2 = pd.read_csv(args.g2_classical)
    g3 = pd.read_csv(args.g3_regime)
    casp = pd.read_csv(args.casp_gate)
    active = pd.read_csv(args.active_gate)
    c76 = pd.read_csv(args.casp_recalibration) if args.casp_recalibration.exists() else pd.DataFrame()

    rows: list[dict[str, object]] = []

    mlg_focus = g1[g1["input_mode"] == "formulation_plus_early"]
    mlg_pass, mlg_n = _bool_count(mlg_focus["gate_ci_low_gt_0"])
    lgbm_focus = g1_lgbm[g1_lgbm["input_mode"] == "formulation_plus_early"]
    lgbm_pass, lgbm_n = _bool_count(lgbm_focus["gate_ci_low_gt_0"])
    rows.append({
        "claim_id": "C1",
        "claim": "The kinetic middle layer adds predictive structure beyond direct Q(t) regression.",
        "status": "green" if mlg_pass == mlg_n else "yellow",
        "main_text_role": "central claim",
        "evidence": (
            f"RF/ET formulation+early bootstrap CI > 0 in {mlg_pass}/{mlg_n} cells; "
            f"LGBM passes {lgbm_pass}/{lgbm_n} cells."
        ),
        "limits": "Effect size is modest; LGBM cross321 by-polymer is not significant.",
        "next_action": "Report paired bootstrap CI, not only median MLG.",
    })

    two_point = g2[g2["window"].str.contains("two_point", regex=False)].copy()
    best_two = (
        two_point.sort_values("future_r2_median", ascending=False)
        .groupby(["dataset", "window"], as_index=False)
        .head(1)
    )
    max_classical = float(best_two["future_r2_median"].max())
    rows.append({
        "claim_id": "C2",
        "claim": "Sparse early observations are not solved by simple classical curve extrapolators.",
        "status": "green" if max_classical < 0.85 else "red",
        "main_text_role": "supporting baseline claim",
        "evidence": f"Best two-point classical future-R2 median is {max_classical:.3f}; red line was 0.85.",
        "limits": "This tests sparse curve-only extrapolation, not all possible Bayesian time-series models.",
        "next_action": "Keep GP/KP/Weibull details in methods or supplementary.",
    })

    r5 = g3[g3["regime"] == "R5"].iloc[0]
    r6 = g3[g3["regime"] == "R6"].iloc[0]
    regime_green = bool(r5["gate_median_ge_0.95"]) and bool(r6["gate_median_ge_0.95"])
    rows.append({
        "claim_id": "C3",
        "claim": "Release curves identify low-dimensional, regime-dependent feasible kinetic states.",
        "status": "green" if regime_green else "yellow",
        "main_text_role": "scientific mechanism claim",
        "evidence": (
            f"R5 cond4 median R2={r5['regime_cond4_median']:.4f}; "
            f"R6 cond4 median R2={r6['regime_cond4_median']:.4f}. "
            "R5 frees diffusion/burst/capacity-like terms; R6 frees hydrolysis/diffusion/hydration/capacity-like terms."
        ),
        "limits": "R6 is partial-pass versus greedy4; avoid hard taxonomy language.",
        "next_action": "Frame regimes as soft/conditional mechanism regions, not immutable classes.",
    })

    random_casp = casp[casp["split_type"] == "random"]
    if not c76.empty:
        random_76 = c76[c76["scheme"] == "random_5fold"].copy()
        random_recal_ok = bool((random_76["cov90_local_mean"] >= 0.90).all())
        width_note = ", ".join(
            f"{r.dataset} local cov90={r.cov90_local_mean:.3f} widthx={r.local_width_inflation:.2f}"
            for r in random_76.itertuples()
        )
        rows.append({
            "claim_id": "C4",
            "claim": "Raw FIB-CASP intervals under-cover, but split-conformal recalibration restores near-nominal pointwise random-split coverage.",
            "status": "green" if random_recal_ok else "yellow",
            "main_text_role": "main UQ claim with explicit width cost",
            "evidence": (
                "Raw random cov90: "
                + ", ".join(f"{r.dataset}={r.cov90_mean:.3f}" for r in random_casp.itertuples())
                + "; after local conformal: "
                + width_note
            ),
            "limits": "Coverage repair is achieved by widening intervals; cov90 is pointwise over observed timepoints, not a simultaneous whole-curve band.",
            "next_action": "Use recalibrated intervals for main UQ panels; keep raw intervals as an ablation.",
        })
    else:
        random_pass_like = int(random_casp["gate_status"].isin(["pass", "borderline"]).sum())
        rows.append({
            "claim_id": "C4",
            "claim": "FIB-CASP gives useful uncertainty under random splits, but nominal 90% coverage is not uniformly achieved.",
            "status": "yellow" if random_pass_like == len(random_casp) else "red",
            "main_text_role": "main claim with calibration caveat",
            "evidence": (
                "Random split gate statuses: "
                + ", ".join(f"{r.dataset}={r.gate_status} cov90={r.cov90_mean:.3f}" for r in random_casp.itertuples())
            ),
            "limits": "cross321 random cov90=0.805 is systematically below nominal 0.90; this is useful UQ, not perfect calibration.",
            "next_action": "Show calibration curves; do not imply perfect 90% coverage.",
        })

    ood_casp = casp[casp["split_type"] == "ood"]
    if not c76.empty:
        ood_76 = c76[c76["scheme"] != "random_5fold"].copy()
        plga_ood = ood_76[ood_76["dataset"].isin(["cross321", "internal181"])]
        lip_ood = ood_76[ood_76["dataset"] == "liposome"]
        plga_ok = bool((plga_ood["cov90_local_mean"] >= 0.88).all())
        lip_ok = bool((lip_ood["cov90_local_mean"] >= 0.88).all()) if len(lip_ood) else False
        rows.append({
            "claim_id": "C5",
            "claim": "Conformal recalibration substantially repairs PLGA OOD pointwise coverage, but cross-mechanism OOD calibration remains unsolved.",
            "status": "yellow" if plga_ok and not lip_ok else "red",
            "main_text_role": "limitation / stress test",
            "evidence": (
                "PLGA OOD local cov90: "
                + ", ".join(
                    f"{r.dataset}:{r.scheme}={r.cov90_local_mean:.3f}"
                    for r in plga_ood.itertuples()
                )
                + "; liposome OOD local cov90: "
                + ", ".join(
                    f"{r.scheme}={r.cov90_local_mean:.3f}, R2med={r.r2_median:.3f}"
                    for r in lip_ood.itertuples()
                )
            ),
            "limits": "PLGA OOD repair costs wider intervals; liposome OOD remains weak in both R2 and coverage.",
            "next_action": "Write PLGA OOD as repaired-with-cost and liposome OOD as a boundary of decoder extensibility.",
        })
    else:
        ood_warn = int((ood_casp["gate_status"] == "warn").sum())
        rows.append({
            "claim_id": "C5",
            "claim": "OOD calibrated uncertainty is partially retained but not solved.",
            "status": "yellow" if ood_warn < len(ood_casp) else "red",
            "main_text_role": "limitation / stress test",
            "evidence": (
                "OOD gate statuses: "
                + ", ".join(f"{r.dataset}:{r.scheme}={r.gate_status}" for r in ood_casp.itertuples())
            ),
            "limits": "internal181 OOD and liposome OOD are weak; coverage CI lower bounds often miss floor.",
            "next_action": "Either recalibrate OOD or write as a boundary of the method.",
        })

    active_fixed = active[
        active["comparator"].str.contains("fixed_", regex=False)
        & active["comparator"].str.contains("_conformal", regex=False)
        & ~active["comparator"].str.contains("directQ", regex=False)
    ]
    active_direct = active[active["comparator"].str.contains("directQ", regex=False)]
    active_cov_ok = bool(active_fixed.get("coverage90_gate_active_nominal", pd.Series([False])).fillna(False).all())
    active_fixed_noninf = bool(active_fixed["rmse_gate_active_not_worse_10pct"].all())
    direct_noninf = bool(active_direct["rmse_gate_active_not_worse_10pct"].all())
    rows.append({
        "claim_id": "C6",
        "claim": "Active observer maintains nominal coverage while supporting sparse experimental decision-making.",
        "status": "yellow" if active_fixed_noninf and active_cov_ok else "red",
        "main_text_role": "decision module, not point-prediction champion",
        "evidence": (
            f"Active is non-inferior to fixed conformal schedules={active_fixed_noninf}; "
            f"active coverage nominal={active_cov_ok}; directQ non-inferior={direct_noninf}."
        ),
        "limits": "Active-vs-directQ non-inferiority is not demonstrated at n=38; this is inconclusive, not proof of inferiority.",
        "next_action": "Present as observation scheduling/UQ; keep point-RMSE champion language out.",
    })

    lip = casp[casp["dataset"] == "liposome"]
    lip_random = lip[lip["scheme"] == "random_5fold"].iloc[0]
    rows.append({
        "claim_id": "C7",
        "claim": "Cross-mechanism decoder swap is feasible on liposome IVR.",
        "status": "yellow",
        "main_text_role": "proof-of-concept / supplementary support",
        "evidence": f"Liposome random R2 median={lip_random.r2_median:.3f}, cov90={lip_random.cov90_mean:.3f}.",
        "limits": "Only 93 curves and 2-parameter Weibull; OOD R2 is below 0.80.",
        "next_action": "Do not claim broad cross-mechanism foundation behavior from liposome alone.",
    })

    rows.append({
        "claim_id": "C8",
        "claim": "Chitosan prospective wet-lab validation closes the biological loop.",
        "status": "pending",
        "main_text_role": "prospective validation if wet curves pass",
        "evidence": "Locked predictions exist, but wet comparison is not in the current gate outputs.",
        "limits": "No verified R2/coverage against real chitosan curves yet.",
        "next_action": "Predefine pass as >=8/12 curves with acceptable error and cov90 hit rate >=0.80 before writing this as a result.",
    })

    rows.append({
        "claim_id": "C9",
        "claim": "World-model framing is useful for talks, not for the title/abstract.",
        "status": "yellow",
        "main_text_role": "discussion-only analogy at most",
        "evidence": "Current evidence supports mechanism-state observer, not learned dynamics/planning.",
        "limits": "No learned latent dynamics or inverse-design planner has passed gates.",
        "next_action": "Use 'mechanism-constrained release dynamics observer' in manuscript-facing text.",
    })

    out = pd.DataFrame(rows)
    out["status_label"] = out["status"].map(_status_label)
    return out[[
        "claim_id", "status_label", "claim", "main_text_role",
        "evidence", "limits", "next_action",
    ]]


def build_figure_routing(readiness: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "figure": "Fig. 1",
            "title": "Sparse-observation mechanism-state observer framework",
            "route": "main text",
            "status": "YELLOW",
            "evidence_source": "architecture + gates C1-C4",
            "message": "Use observer / mechanism-state language; avoid world-model title claim.",
        },
        {
            "figure": "Fig. 2",
            "title": "Low-dimensional regime-dependent kinetic states",
            "route": "main text",
            "status": readiness.loc[readiness["claim_id"] == "C3", "status_label"].iloc[0],
            "evidence_source": "scripts 33/36 + outputs 71 G3",
            "message": "Strong enough for a mechanism claim if written as soft regime regions.",
        },
        {
            "figure": "Fig. 3",
            "title": "Mechanistic middle-layer gain over direct Q(t) regression",
            "route": "main text",
            "status": readiness.loc[readiness["claim_id"] == "C1", "status_label"].iloc[0],
            "evidence_source": "outputs 71 G1",
            "message": "Main quantitative defense of the kinetic bottleneck.",
        },
        {
            "figure": "Fig. 4",
            "title": "Classical sparse extrapolators fail under early-point forecasting",
            "route": "main or supplementary",
            "status": readiness.loc[readiness["claim_id"] == "C2", "status_label"].iloc[0],
            "evidence_source": "outputs 71 G2",
            "message": "Useful baseline panel; not the central novelty.",
        },
        {
            "figure": "Fig. 5",
            "title": "FIB-CASP uncertainty and conformal recalibration",
            "route": "main text for random splits; OOD in Extended Data / limitation",
            "status": readiness.loc[readiness["claim_id"] == "C4", "status_label"].iloc[0],
            "evidence_source": "outputs 72 G4 + outputs 76",
            "message": "Main panel should show raw pointwise under-coverage and conformal coverage repair with width cost.",
        },
        {
            "figure": "Fig. 6",
            "title": "Active observer for sparse experimental decision-making",
            "route": "main text only if framed as decision/UQ",
            "status": readiness.loc[readiness["claim_id"] == "C6", "status_label"].iloc[0],
            "evidence_source": "outputs 72 G5",
            "message": "Do not call it the best point predictor.",
        },
        {
            "figure": "Extended Data",
            "title": "Liposome cross-mechanism proof of concept",
            "route": "extended data / supplementary",
            "status": readiness.loc[readiness["claim_id"] == "C7", "status_label"].iloc[0],
            "evidence_source": "outputs 72 G4 + outputs 76",
            "message": "Proof of decoder extensibility in random splits; liposome OOD remains a negative boundary.",
        },
        {
            "figure": "Future Fig.",
            "title": "Chitosan prospective validation",
            "route": "main text only after wet comparison",
            "status": readiness.loc[readiness["claim_id"] == "C8", "status_label"].iloc[0],
            "evidence_source": "outputs 67 pending",
            "message": "This is the prospective biology gate, currently incomplete.",
        },
    ])


def write_summary(out: Path, readiness: pd.DataFrame, figures: pd.DataFrame) -> None:
    counts = readiness["status_label"].value_counts().to_dict()
    lines: list[str] = [
        "# Nature-family manuscript readiness matrix",
        "",
        "## Executive verdict",
        "",
        (
            "The strongest current manuscript route is **mechanism-constrained sparse-observation "
            "release forecasting**, not a title-level world-model claim."
        ),
        "",
        (
            f"Claim status counts: GREEN={counts.get('GREEN', 0)}, "
            f"YELLOW={counts.get('YELLOW', 0)}, RED={counts.get('RED', 0)}, "
            f"PENDING={counts.get('PENDING', 0)}."
        ),
        "",
        "Recommended target posture: Nature Methods / Nature Communications presubmission, "
        "with Nature-main language reserved only if chitosan prospective validation is strong.",
        "",
        "## Claim routing",
        "",
    ]
    for r in readiness.itertuples(index=False):
        lines.extend([
            f"### {r.claim_id} [{r.status_label}] {r.claim}",
            f"- Role: {r.main_text_role}",
            f"- Evidence: {r.evidence}",
            f"- Limit: {r.limits}",
            f"- Next: {r.next_action}",
            "",
        ])
    lines.extend(["## Figure routing", ""])
    for r in figures.itertuples(index=False):
        lines.extend([
            f"- **{r.figure}: {r.title}**",
            f"  Route: {r.route}; status: {r.status}; evidence: {r.evidence_source}.",
            f"  Message: {r.message}",
        ])
    lines.extend([
        "",
        "## Current writing rule",
        "",
        "Use **mechanism-constrained release dynamics observer** in manuscript-facing text. "
        "Use world-model language only as an informal internal north-star or discussion analogy, "
        "not as a title, abstract, or headline claim.",
        "",
    ])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/73_manuscript_readiness_matrix"))
    ap.add_argument("--g1-rf-et", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g1_middle_layer_bootstrap.csv"))
    ap.add_argument("--g1-lgbm", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g1_lgbm_bootstrap.csv"))
    ap.add_argument("--g2-classical", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g2_classical_sparse_summary.csv"))
    ap.add_argument("--g3-regime", type=Path, default=Path("outputs/71_nature_gate_diagnostics/g3_regime_gate_summary.csv"))
    ap.add_argument("--casp-gate", type=Path, default=Path("outputs/72_casp_active_gate_diagnostics/casp_bootstrap_gate.csv"))
    ap.add_argument("--casp-recalibration", type=Path, default=Path("outputs/76_casp_conformal_recalibration/summary.csv"))
    ap.add_argument("--active-gate", type=Path, default=Path("outputs/72_casp_active_gate_diagnostics/active_pairwise_gate.csv"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    readiness = build_readiness(args)
    figures = build_figure_routing(readiness)
    readiness.to_csv(args.out / "readiness_matrix.csv", index=False)
    figures.to_csv(args.out / "figure_routing.csv", index=False)
    write_summary(args.out, readiness, figures)
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
