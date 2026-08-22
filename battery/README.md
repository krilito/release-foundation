# Battery trajectory workspace

This folder is for the **battery trajectory** extension of
`release-foundation`.

It is not the same project as `D:\battery`.

## Boundary

`D:\release-foundation\battery` is for:

- cycle-life / capacity-fade trajectories,
- early-cycle observation adapters,
- long-horizon battery prediction benchmarks,
- posterior-family inference over degradation-state coordinates,
- active measurement timing for future cycles or diagnostics.

`D:\battery` is for:

- solid electrolyte material screening,
- ionic-conductivity prediction,
- cathode|electrolyte|coating interface risk filtering,
- professor-facing candidate cards for experimental synthesis.

Do not mix these two.

## Shared algorithmic object

The battery trajectory task should use the same abstract object as the
drug-release task:

```text
partial observations y_obs
        ->
feasible posterior family over latent state theta
        ->
calibrated predictive distribution over future trajectory
        ->
active measurement utility
```

For battery cycling:

```text
x = cell metadata / protocol descriptors
t = cycle number
y(t) = capacity, normalized capacity, resistance, or other health signal
theta = degradation-state coordinates
future = remaining capacity trajectory, EOL, or knee point
```

## First dataset candidates

Use public trajectory datasets before asking the professor for private data:

- Severson / MATR battery cycle-life data: `https://data.matr.io/1/`
- NASA PCoE battery aging datasets
- CALCE battery datasets

Raw battery data should not be committed. Put local downloads under a
gitignored data directory and write intake scripts that generate small
auditable summaries.

## First implementation target

The first useful script should be an intake/audit script, not a model:

```text
battery/scripts/79_battery_intake.py
```

It should produce:

```text
outputs/79_battery_intake/intake_summary.json
outputs/79_battery_intake/battery_curves_long.csv
outputs/79_battery_intake/battery_cell_metadata.csv
```

Only after intake passes should we add a battery degradation decoder.

## Do not claim yet

- Do not claim battery SOTA.
- Do not claim a battery world model.
- Do not claim transfer from drug release until the same benchmark protocol
  runs on both domains.
- Do not use `D:\battery` electrolyte-screening results as battery
  trajectory data.
