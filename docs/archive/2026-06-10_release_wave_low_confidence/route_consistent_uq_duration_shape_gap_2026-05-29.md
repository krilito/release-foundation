# Route-Consistent UQ Duration/Shape Gap

Date: `2026-05-29`

Purpose:

```text
Translate the current "UQ-to-timescale gap" into a concrete moat-building
problem:
1. what components already exist,
2. what the actual missing layer is,
3. why that layer matters strategically,
4. what the next technical move should target.
```

Primary audit outputs:

- [readiness_matrix.csv](D:/release-foundation/outputs/97_release_stack_readiness_audit/readiness_matrix.csv)
- [component_tally.csv](D:/release-foundation/outputs/97_release_stack_readiness_audit/component_tally.csv)
- [readiness_heatmap.png](D:/release-foundation/outputs/97_release_stack_readiness_audit/readiness_heatmap.png)
- [summary.txt](D:/release-foundation/outputs/97_release_stack_readiness_audit/summary.txt)

## 1. The main result

Current audited coverage is:

- `curve panels`: `6`
- `timing/shape panels`: `6`
- `point-family objects`: `6`
- `calibrated-family panels`: `9`
- `route-consistent duration UQ`: `0`

Source:

- [component_tally.csv](D:/release-foundation/outputs/97_release_stack_readiness_audit/component_tally.csv)

That is the cleanest possible statement of the missing layer.

## 2. What this means

The problem is **not**:

- "we do not have uncertainty"
- "we do not have timing/shape"
- "we do not have a posterior-family object"

The problem is:

> those layers are not yet traveling through the same route contract in the
> same benchmark cell.

## 3. Two current failure modes

### Failure mode A — same cell, route mismatch

For:

- `C321-D`
- `C321-P`
- `I181-D`
- `I181-P`
- `Lipo-D`
- `Lipo-P`

we already have:

- curve panel
- timing/shape panel
- point-family object
- calibrated-family panel

But the duration/shape outputs are still tied to a point-estimate mechanism
route, while the uncertainty object is tied to a separate calibrated-family
route.

Audit label:

- `same_cell_but_route_mismatch`

### Failure mode B — UQ without timing/shape panel

For:

- `C321-R`
- `I181-R`
- `Lipo-R`

we already have calibrated-family outputs, but we do not yet have the
matching mechanism-route timing/shape exports.

Audit label:

- `uq_without_timescale_panel`

## 4. Why this matters strategically

This is the most important local technical gap because it sits exactly
between:

- a strong forecasting paper
and
- a real release decision-support stack

Without this layer, we can honestly say:

- unified benchmark
- shared posterior-family object
- calibrated family
- timing/shape reporting

But we still cannot honestly say:

> uncertainty-aware release duration / shape is part of the same executable
> route contract.

That is why this gap is not cosmetic.

## 5. Why this gap is also a moat opportunity

External benchmark rivals mostly do not even have the layers that we are now
trying to join.

Reference:

- [competitor_landscape.png](D:/release-foundation/outputs/92_release_strategy_figures/competitor_landscape/competitor_landscape.png)

So solving this layer would not just add one more result; it would move the
project further away from:

- direct release regressors
- platform systems that skip explicit release-state uncertainty

and closer to a true:

`release-intelligence -> release decision-support`

stack.

## 6. Best next technical move

The next step should target one narrow question:

> can the calibrated-family route emit timing and shape uncertainty in the
> same cell without dropping back to a separate point-estimate route?

That is the next moat-building problem.

## Bottom line

The current unified-release stack is already broad.

Its sharpest remaining missing layer is now explicit and measured:

> route-consistent uncertainty-aware duration / shape

This is the most valuable next technical target because it is both a genuine
scientific gap and a strategic differentiator.
