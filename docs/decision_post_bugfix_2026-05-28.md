# Decision Post-Bugfix — 2026-05-28

Status note:
- This is an intermediate bugfix note, not the final release closeout record.
- Some figures here reflect intermediate reruns and are not the durable citable numbers.
- Use `docs/drug_release_closeout_2026-05-28.md` plus the referenced output files for current release-only decisions.

## Summary of corrected results

### 4-way factorial canonical benchmark (N=38)

```
Method                RMSE    obs
DirectQ-fixed        0.173    4 fixed
DirectQ-adaptive     0.160    2 adaptive
Active-fixed         0.200    4 fixed
Active-adaptive      0.151    2 adaptive
```

Active-adaptive vs DirectQ-fixed: Δ=+0.023, CI [-0.004, +0.055], NOT significant.

### 4-way factorial LOOCV (N=259)

```
Method                RMSE    obs
DirectQ-fixed        0.084    4 fixed
DirectQ-adaptive     0.046    2 adaptive
Active-fixed         0.129    4 fixed
Active-adaptive      0.084    2 adaptive
```

Active-adaptive vs DirectQ-fixed: Δ=+0.0001, CI [-0.008, +0.008], NOT significant.

### Key findings

1. **Selection effect is real**: adaptive 2-point beats fixed 4-point for BOTH
   predictors (p<0.001 in both canonical and LOOCV).

2. **Inference effect is negative**: Active (particle filter) is WORSE than
   DirectQ (ExtraTrees) under both observation strategies.
   - Canonical: Active-fixed 0.200 vs DQ-fixed 0.173
   - LOOCV: Active-fixed 0.129 vs DQ-fixed 0.084

3. **Best method is DirectQ-adaptive** (RMSE 0.046 on LOOCV), not
   Active-adaptive.

4. **Regime × benefit**: LOOCV p=0.521. No significant interaction.

### RSSM diagnostic (BUG 4 fixed)

- KL/dim = 0.028 (collapsed)
- cov90 = 0.481 (0.30-0.50 range)
- Posterior collapse confirmed.

### Regime verification (BUG 3 fixed)

- Theta-space silhouette: -0.057 (p=0.007)
- x-space silhouette: -0.196 (p=0.652)
- Regimes are orthogonal to parameter values.

## Truth table outcome

Per ADR-029 pre-decided rule:

| Canonical | LOOCV | Action |
|-----------|-------|--------|
| Active wins ≥5%, CI>0 | Active wins ≥5%, CI>0 | Reverse ADR-027 |
| Active wins | Active wins but marginal | Reverse with caveats |
| Wins canonical, loses LOOCV | — | Investigate |
| **Loses or insignificant in both** | — | **ADR-030 needed** |

**Outcome: Active-adaptive is NOT significantly better than DirectQ-fixed in
either canonical (CI crosses 0) or LOOCV (Δ≈0). ADR-030 is needed.**

## Proposed ADR-030 (for project lead approval)

> Restructure paper around adaptive observation selection as main contribution.
> Active Observer (particle filter + ODE) demoted to "mechanism-constrained
> baseline we tried". DirectQ-adaptive promoted to recommended method.
> Paper headline: "Adaptive observation scheduling reduces drug release
> experiments from 4 to 2 timepoints with equivalent predictive accuracy."

### What survives

- Adaptive observation selection (the utility function) — proven effective
- Conformal calibration — proven effective (cov90=0.93)
- RSSM negative result — confirmed (posterior collapse at this data scale)
- Regime structure (supplementary) — orthogonal to theta, novel framing
- Chitosan prospective — still pending wet-lab data

### What dies

- Active Observer as main method (inference effect negative)
- "+27% RMSE improvement" headline (artifact of information asymmetry)
- "Regime drives Active Observer benefit" (p=0.52, not supported)
- Direct-Q + h-features improvement (BUG 1 leakage)

## Pending decision

Project lead must approve ADR-030 before paper restructuring proceeds.
