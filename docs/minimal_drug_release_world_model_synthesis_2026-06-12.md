# Minimal Drug Release World Model Synthesis

Date: 2026-06-12

## Controlling Claim

```text
Drug release prediction is not just curve regression;
it is release-state inference under limited pre-experimental descriptors
and costly observations.
```

The project should not be narrated as a better generic release-curve regressor.
The defensible contribution is a minimal world-model prototype that separates:

```text
Q(t) -> z
z -> Q(t)
X -> prior over z
X + Y_k -> posterior over z
z / Q_hat(t) -> design, risk, and stopping decisions
```

## Evidence Summary

### PLGA

The frozen PLGA result is an information-budget result:

```text
formulation descriptors alone under-identify release;
early measured release observations expose missing kinetic state;
after early Q is known, point prediction approaches an information-saturated regime.
```

The valuable PLGA claim is not that a model wins an RMSE race against every
black box. The valuable claim is that observation budget matters, measured
early points carry state information that static descriptors cannot supply,
and stopping/design rules should be evaluated separately from curve RMSE.

### Liposome 112/113

The Yanes et al. author baseline predicts kinetic class, not continuous future
curves. Class prototypes and theta-prototype mixtures carry signal, but under
strict splits they do not consistently beat early-only or refined theta routes.

This showed that a middle layer is meaningful but that class labels are too
coarse to be the final release-state language.

### Liposome 114

The curve dictionary result established that a release-state language exists:

```text
Q(t) -> z -> Q(t)
```

PCA reconstruction with train-fold dictionaries is strong. Static descriptors
can predict dictionary coefficients partially, especially in less strict
settings, but the static `X -> z` map is not sufficient as a full curve model.

### Liposome 115

The latent-gap observation-budget result directly tested:

```text
X -> z
X + Y_k -> z
oracle Q(t) -> z
```

Primary PCA-8 / ExtraTrees results:

| Split | k=5 oracle gap closed | width contraction |
|---|---:|---:|
| stratified_5fold | 50.7% | yes |
| group_by_API | 19.8% | no |
| group_by_release_method | 2.5% | no |

Interpretation: early observations identify curve state, but the effect is much
stronger within distribution than across assay/release-method shifts. This is
release-state inference evidence, not mechanism-transfer evidence.

### Liposome 116

The design-utility benchmark separated exact prediction from candidate ranking:

```text
prediction utility != exact curve RMSE
```

Static priors can enrich pre-observation candidate selection for some targets.
The strongest and most stable target is avoiding burst/failure. `sustained_mid`
also shows useful enrichment in several splits. `anti_burst` under group-by-API
remains weak.

Interpretation: even when exact curve prediction is limited, static descriptors
can still have experimental design value. That value is target-dependent and
split-dependent.

### Liposome 117

RDKit/PubChem descriptor enrichment had full coverage for the valid liposome
API names, and main group-by-API rows excluded raw API identity.

Key result:

```text
Molecular descriptors did not close the strict group_by_API static X -> z gap.
```

In group-by-API, adding molecular descriptors worsened median `z_rmse` and
reduced design hit rate for the tested targets. Interaction features helped
only weakly and inconsistently.

Interpretation: the missing static information is unlikely to be solved by
simple molecular descriptors alone. The missing state is more likely process,
microstructure, morphology, batch, or hidden assay variables.

## Final Judgment

The current evidence supports:

```text
B. Partial support:
   z language exists,
   Y_k closes part of the gap,
   X prior/design utility exists but is weak and conditional.
```

This is stronger than a negative result because it gives a structured research
object:

```text
available X gives a prior;
measured Y_k performs state identification;
decision utility can be positive even when full-curve RMSE is not decisive;
descriptor enrichment shows where the missing information probably is not.
```

It is weaker than strong support because strict transfer is not solved. The
model does not yet prove robust cross-system or cross-assay generalization.

## Research Story

The clean story is:

```text
Drug release ML lacks a stable modeling unit.
Treating time-Q points as independent samples is wrong.
Treating whole curves as functions exposes a latent release-state language.
Static descriptors provide an imperfect prior over this state.
Early observations update the state and close part of the latent gap.
Experimental value must be measured by observation budget and design utility,
not only full-curve RMSE.
```

This story maps PLGA and liposome into the same framework without claiming that
one model has learned all release mechanisms.

## What To Do Next

1. Freeze 114-117 as the second-system liposome evidence chain.
2. Convert the current synthesis into manuscript figures:
   `curve language`, `latent gap closure`, `design utility`, and `descriptor
   enrichment negative control`.
3. For a third system, reuse the same gates:
   whole-curve static prediction, curve dictionary, observation-budget gap
   closure, design utility, descriptor/process enrichment.
4. Do not introduce neural architectures until they answer one of the locked
   release-state questions better than these small-data controls.

## Forbidden Narrative

```text
We developed a better release curve regression model.
```

## Allowed Narrative

```text
We establish a minimal drug release world model prototype
that separates curve representation, pre-experimental prior,
observation-based state inference, and experiment-aware decision utility.
```
