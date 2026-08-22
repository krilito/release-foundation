# Release Stack Occupancy Verdict

Date: `2026-05-29`

Purpose:

```text
Compare our current executable stack occupancy against the main external
families on the same five-layer unification surface.
```

Primary outputs:

- [internal_vs_external_stack_occupancy.csv](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/internal_vs_external_stack_occupancy.csv)
- [stack_layer_advantage_summary.csv](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/stack_layer_advantage_summary.csv)
- [summary.txt](D:/release-foundation/outputs/109_release_stack_occupancy_comparison/summary.txt)

Code:

- [109_release_stack_occupancy_comparison.py](D:/release-foundation/scripts/109_release_stack_occupancy_comparison.py)

## 1. Core verdict

The current local system now occupies the unified stack more completely than
the external benchmark, platform, and infrastructure families on the same
five-layer surface.

The strongest advantage is not:

- raw benchmark score
- one special model

It is:

> combined occupancy of benchmark rules, shared family object, mechanism
> decoders, and target-layer reporting semantics.

## 2. Where the edge is strongest

The structural edge should be strongest at:

- `L2 partial-observation benchmark`
- `L3 shared posterior-family object`

because most external systems remain weak or absent there.

## 3. Where the edge is still qualified

The edge is still qualified at:

- `L5 reporting / decision layer`

because our current evidence is strongest for:

- curve-space objects
- shape-aware reporting

and weaker for:

- deep-threshold timing defaults

## 4. Bottom line

This comparison sharpens the outside-in story:

> the open slot is not "best release regressor."
> the open slot is the most complete executable occupancy of the shared
> release-intelligence stack.
