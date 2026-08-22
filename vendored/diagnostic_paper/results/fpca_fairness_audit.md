# fPCA Fairness Audit

## Verdict

The nested fPCA route is acceptable as a representation-control comparator, provided the manuscript does not claim statistically significant superiority over the locked Direct LGBM baseline.

## Checks

- Split: `GroupKFold-5` by `Experimental_index`.
- Static descriptors: `Polymer_MW, Drug_Mw, Drug_LogP, SA_V, LA/GA, Drug_Pka, DLC, Initial D/M ratio, Drug_TPSA`.
- fPCA basis: fitted inside each outer fold on training curves only.
- Nested k selection: `GroupKFold-3` within each outer training fold.
- Selected k values by outer fold: `18, 10, 18, 10, 18`.
- Held-out evaluation: formulation groups are held out by `Experimental_index`; no time-point random split.
- Paired comparison: nested fPCA versus locked Direct uses the same 181 held-out formulation groups.

## Direct discrepancy

The locked Direct LGBM remains the manuscript baseline: groupwise R2=0.7073, RMSE=0.1055. The same-script OOF Direct sanity run is lower: groupwise R2=0.6762, RMSE=0.1146. This discrepancy should be described as an implementation/recipe difference between the locked production baseline and the fPCA audit script, not as evidence that fPCA was compared against a weakened Direct model. The conservative comparison is against the stronger locked Direct baseline.

## Manuscript-ready text if retained

As a representation control, we evaluated a nested fPCA route under the same formulation-group deployment protocol. The fPCA basis was fitted only on training curves within each outer fold, and the number of components was selected inside the outer training set. This route reached groupwise R2=0.7518 and RMSE=0.1027, close to the locked Direct LGBM baseline. The paired RMSE difference versus locked Direct was not significant (Wilcoxon p=0.6662), so fPCA is interpreted as evidence that curve-level signal can be preserved in a functional latent coordinate, not as a superior predictor.

## Downgrade text if fairness is challenged

If a reviewer does not accept the fPCA implementation as fully comparable, the fPCA result should be moved to the Supplement as a sensitivity analysis. The main claim can still rest on matched-oracle decoder adequacy, clean Weibull3 deployment collapse, and post-hoc kinetic-summary behavior, while avoiding any central dependence on fPCA.
