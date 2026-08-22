## Curve-latent observer diagnostic

Date: `2026-05-26`

Script:

```text
scripts/56_curve_latent_observer.py
```

Question:

```text
Can the model first observe many release curves, learn their shape
manifold, and then use only early Q to reconstruct the future?
```

This is the minimal non-neural version of that idea. It does not use
formulation descriptors, ODE theta, or regime labels.

## Method

Within each CV fold:

```text
train full curves -> curve prior
test early Q(1,3,5,7) -> reconstructed future curve
test full curve -> final evaluation only
```

Compared curve-prior methods:

```text
train_median_curve:
  train-fold median curve, anchored by observed test early Q.

knn_curve_analog:
  find train curves closest in early-Q space, average their future curves.

pca_curve_latent_k:
  learn a PCA basis from train full curves, infer latent coordinates from
  test early Q by ridge least squares, reconstruct future curve.

ET_earlyQ_to_lateQ:
  supervised ExtraTrees baseline from early Q to late-grid Q.
```

The important leakage guardrail:

```text
PCA basis, KNN bank, and ET direct model are all fit on train folds only.
Test full curves are used only for final R2.
```

## Full-run result

Command:

```powershell
python scripts\56_curve_latent_observer.py --out outputs\56_curve_latent_observer
```

Best full-curve median R2:

```text
dataset      split             best method            full R2   future R2
cross321     random_5fold      ET_earlyQ_to_lateQ      0.910     0.661
cross321     group_by_drug     knn_curve_analog        0.881     0.575
cross321     group_by_polymer  knn_curve_analog        0.886     0.569

internal181  random_5fold      knn_curve_analog        0.875     0.549
internal181  group_by_drug     knn_curve_analog        0.859     0.346
internal181  group_by_polymer  ET_earlyQ_to_lateQ      0.840     0.195
```

PCA latent result:

```text
The low-dimensional PCA curve manifold works, but is weaker than KNN/ET.
Best PCA is usually k=2 or k=4. Higher k overfits early anchors and hurts
future R2, especially internal181 OOD.
```

## Interpretation

The "observe curves and learn their shape" idea is real, but the first
minimal version does not beat the current theta route.

Compared to the current best theta/ODE route (`50`):

```text
cross321 OOD theta route:     ~0.943-0.946
cross321 OOD curve prior:     ~0.881-0.886

internal181 OOD theta route:  ~0.929-0.939
internal181 OOD curve prior:  ~0.840-0.859
```

So the current paper should not replace the theta story with this.

The useful conclusion is narrower:

```text
Release curves have a reusable empirical shape manifold.
Early Q can retrieve an analog future curve, but the theta/ODE bottleneck
is still the stronger shape regularizer.
```

## Next version, if pursued

Do not scale this by blindly adding a bigger neural net.

The next serious version should be:

```text
simulator-augmented masked-curve pretraining
irregular-time input encoding
future-curve prediction objective
uncertainty over future trajectories
then optional alignment to theta/regime
```

The success criterion should be explicit:

```text
Beat theta->ODE on held-out drug/polymer future R2, not only full R2.
```

Until that happens, the curve-latent route is a future foundation-model
direction, not the main manuscript claim.

## 57 small-MLP follow-up

Script:

```text
scripts/57_curve_mlp_baseline.py
```

Question:

```text
Maybe the curve-observer route was weak because PCA/KNN/ET were too
simple. Can a small MLP learn early-Q -> future-Q better?
```

Test:

```text
Input:  Q(1,3,5,7)
Output: Q(10,14,21,28,45,60,90)
Models: KNN, ExtraTrees, MLP(64,64), MLP(128,64), MLP ensemble
Splits: random_5fold, group_by_drug, group_by_polymer
```

Full-run command:

```powershell
python scripts\57_curve_mlp_baseline.py --out outputs\57_curve_mlp_baseline
```

Best full-curve median R2:

```text
dataset      split             best method       full R2   future R2
cross321     random_5fold      ET                0.910     0.661
cross321     group_by_drug     MLP ensemble      0.889     0.561
cross321     group_by_polymer  MLP ensemble      0.893     0.561

internal181  random_5fold      KNN               0.875     0.549
internal181  group_by_drug     MLP 64x64         0.868     0.438
internal181  group_by_polymer  ET                0.840     0.195
```

More detailed read:

```text
cross321 OOD:
  MLP slightly improves full R2 over KNN/ET, but future R2 is similar
  or lower than the best non-MLP method.

internal181 held-out drug:
  MLP improves both full and future R2 over KNN/ET.

internal181 held-out polymer:
  MLP does not beat ET on full R2. Some MLP variants improve future R2,
  but the absolute future score remains weak.
```

Verdict:

```text
Small MLP is not a universal upgrade. It is a useful baseline, not a new
main method.
```

This closes the narrow "maybe ordinary MLP is enough" question. If we
pursue neural curve models, the next step should not be another small
tabular MLP. It should be masked-curve pretraining with synthetic
simulator augmentation and irregular-time inputs.

## 58 tree + MLP residual hybrid

Script:

```text
scripts/58_curve_residual_hybrid.py
```

Question:

```text
Can RF/ET provide the stable coarse future curve, while MLP only learns
the residual correction?
```

Implementation:

```text
early Q -> RF/ET base -> late Q
early Q + base late Q -> MLP residual -> correction
final prediction = base + residual
```

Leakage guardrail:

```text
Residual targets are not computed from in-sample base predictions.
Inside each outer train fold, an inner KFold creates OOF base predictions.
The residual MLP trains on train-fold OOF residuals only.
```

Full-run command:

```powershell
python scripts\58_curve_residual_hybrid.py --out outputs\58_curve_residual_hybrid
```

Best full-curve median R2:

```text
dataset      split             best method             full R2   future R2
cross321     random_5fold      ET base                 0.909     0.657
cross321     group_by_drug     RF base                 0.886     0.594
cross321     group_by_polymer  ET + MLP residual       0.889     0.485

internal181  random_5fold      RF/ET residual avg      0.878     0.556
internal181  group_by_drug     RF + MLP residual       0.858     0.384
internal181  group_by_polymer  RF/ET residual avg      0.852     0.227
```

Compared with `57`:

```text
cross321:
  Residual hybrid does not clearly help. It can raise full R2 slightly
  in held-out polymer vs ET base, but usually lowers future R2.

internal181:
  Residual hybrid helps random and held-out polymer full R2, and modestly
  improves future R2 in held-out polymer. It does not beat the best
  standalone MLP on held-out drug.
```

Verdict:

```text
Tree + MLP residual is conditionally useful, especially for internal181,
but it is not a universal upgrade and still does not challenge the
theta->ODE route.
```

This keeps the model-zoo conclusion stable:

```text
For curve-only prediction, simple analog/tree/MLP/residual methods are
good diagnostics. They show a learnable curve manifold. But the current
best deployable method remains theta->ODE with early observation.
```
