## PLGA sub-day observation-window audit

Date: `2026-05-26`

Question:

```text
If reviewers object that Q(1,3,5,7) is not "early enough", what happens
with PLGA 6h / 12h / 24h observations?
```

Here:

```text
6h  = 0.25 day
12h = 0.5 day
24h = 1.0 day
```

## Observation availability

Important caveat:

```text
cross321 has t=0 for all curves, but true 6h/12h sampling is sparse.
internal181 has true 0.25/0.5/1.0 day columns/observations.
```

Exact model-set coverage:

```text
cross321, n=218:
  <=0.25d: any=1.000, >=2 points=0.188, exact 0.25d=0.000
  <=0.5d : any=1.000, >=2 points=0.353, exact 0.5d =0.000
  <=1.0d : any=1.000, >=2 points=0.633, exact 1.0d =0.028

internal181, n=169:
  <=0.25d: any=1.000, >=2 points=1.000, exact 0.25d=1.000
  <=0.5d : any=1.000, >=2 points=1.000, exact 0.5d =1.000
  <=1.0d : any=1.000, >=2 points=1.000, exact 1.0d =1.000
```

Interpretation:

```text
The 0.25/0.5/1.0 day experiment is realistic for internal181, but partly
hypothetical for cross321 because 6h/12h are usually interpolated rather
than directly measured.
```

## Prediction result

Command:

```powershell
python scripts\50_regime_gated_theta_cv.py `
  --early-times 0.25 0.5 1.0 `
  --refine-nfev 5 `
  --out outputs\50_regime_gated_theta_cv_subday_0p25_0p5_1_refine5
```

Best median curve R2:

```text
dataset      split             6/12/24h R2   1/3/5/7d R2   delta
cross321     random_5fold        0.894         0.961       -0.068
cross321     group_by_drug       0.759         0.943       -0.184
cross321     group_by_polymer    0.777         0.946       -0.168

internal181  random_5fold        0.940         0.966       -0.026
internal181  group_by_drug       0.860         0.939       -0.079
internal181  group_by_polymer    0.836         0.929       -0.093
```

Best methods:

```text
cross321 random:        ET_regime_hard
cross321 held-out drug: ET_regime_hard
cross321 held-out poly: early_select

internal181 random:        early_select
internal181 held-out drug: RF_regime_hard
internal181 held-out poly: early_select_early_refined
```

## Interpretation

The sub-day window is useful but not enough to preserve the headline
0.93-0.96 PLGA performance.

The result says:

```text
6/12/24h can forecast PLGA release moderately well.
But the one-week observation supplies major kinetic information,
especially for cross321 held-out drug/polymer OOD.
```

The paper should therefore avoid claiming:

```text
The model predicts long-term PLGA release from only one day.
```

The honest wording is:

```text
First-week partial observation gives the strongest deployable PLGA
forecasting regime.

Sub-day/one-day observation is a stricter deployment setting. It remains
predictive, especially for internal181, but with a clear OOD penalty.
```

## Relation to the "7 days already saw the curve" concern

This audit should be read alongside the horizon audit:

```text
Combined PLGA model set, n=387:
  median t_max = 30.0 days
  median Q7 / Qfinal = 47.6%
  only 2.1% curves end by day 7
  only 11.9% curves have Q7 >= 90% of final release
  86.8% curves still release at least 10% after day 7
```

So day 7 is not usually the end of the PLGA curve. But it is still a
substantial first-week kinetic observation, not an ultra-early readout.

Recommended manuscript language:

```text
one-week partial-observation forecasting
```

not:

```text
very-early prediction
```
