# Workspace domain boundaries

Date: 2026-05-28
Author: Codex

## Decision

Separate the workspaces by mathematical object:

| Workspace | Object | Use |
|---|---|---|
| `D:\release-foundation` root | drug-release trajectory inference | mature PLGA / liposome / chitosan code |
| `D:\release-foundation\battery` | battery trajectory inference | capacity-fade / cycle-life extension of the shared observer algorithm |
| `D:\battery` | battery materials screening | solid electrolyte conductivity, interface risk, professor-facing candidate cards |

Do not mix these because they answer different questions.

## The shared algorithm in `release-foundation`

The shared algorithm is:

```text
partial observations
        ->
feasible posterior family
        ->
calibrated future trajectory
        ->
active measurement utility
```

Drug release and battery cycling are both trajectory problems.

## What belongs in `D:\release-foundation\battery`

Put here:

- battery cycling schemas,
- battery public dataset intake notes,
- battery trajectory benchmark scripts,
- battery degradation decoder prototypes,
- battery active-measurement timing experiments.

Do not put here:

- solid electrolyte conductivity tables,
- cathode|SSE|coating interface-risk samples,
- professor candidate cards,
- material-screening ranking reports.

Those belong in `D:\battery`.

## What belongs in `D:\battery`

Put here:

- electrolyte samples,
- interface samples,
- material screening models,
- Mn-cathode candidate cards,
- professor collaboration deliverables,
- full-cell metric templates for lab return.

Do not put release trajectory code or PLGA/liposome/chitosan code there.

## Why this matters

If these are mixed, the paper claim becomes incoherent:

- electrolyte screening predicts a **material property**,
- battery cycling predicts a **time trajectory**,
- drug release predicts a **time trajectory**.

The Nature-level cross-domain route depends on showing that the trajectory
algorithm generalizes from release to battery cycling, not that a material
screening table happens to involve batteries.

## Immediate execution order

1. Keep drug-release work in existing root/scripts until a deliberate
   migration is planned.
2. Put all new battery-cycle-life work under `battery/`.
3. Keep professor material-screening work in `D:\battery`.
4. Build the first `battery/scripts/79_battery_intake.py` only after a
   public battery trajectory dataset is selected.
5. Cross-domain scripts may live in `scripts/` only if they explicitly
   compare at least two domains.
