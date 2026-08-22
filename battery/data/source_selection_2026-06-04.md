---
date: 2026-06-04
scope: first_battery_trajectory_source
status: initial_selection
depends_on:
  - battery/BATTERY_TRAJECTORY_KICKOFF_2026-06-04.md
  - battery/scripts/79_battery_intake.py
  - docs/cross_domain_benchmark_spec_2026-05-28.md
---

# Battery Source Selection

Goal:

```text
pick the first public battery trajectory dataset for the cross-domain observer smoke benchmark
```

This is not a literature review. It is a practical intake decision.

## Required Interface

The first source must be convertible to:

```text
cell_id
cycle_number
capacity
normalized_capacity
optional resistance
optional coulombic_efficiency
temperature_C
protocol_id
```

and one-row-per-cell metadata:

```text
cell_id
dataset
chemistry
protocol_id
charge_rate
discharge_rate
temperature_C
nominal_capacity
source
```

## Selection

Use this order.

## 1. MATR / TRI cycle-life data

Decision:

```text
try first
```

Source:

- `https://data.matr.io/`
- `https://data.matr.io/1/`

Reason:

- most aligned with early-cycle prediction
- dataset listing indicates raw formation data plus structured aging cycling data
- structured BEEP summaries may reduce parser burden

Immediate check:

```text
Can we download or export a cycle-summary table without interactive manual work?
```

Promote if:

- cell-cycle capacity summaries are accessible
- cell/protocol IDs are stable
- license permits research use and can be cited

Reject or defer if:

- access requires login/API work that blocks the first smoke
- only high-frequency raw cycling files are available and no summary export is easy

## 2. CALCE battery data

Decision:

```text
first fallback
```

Source:

- `https://calce.umd.edu/battery-data`
- `https://web.calce.umd.edu/batteries/data/`

Reason:

- public page exposes many capacity/cycle files
- protocol axes are scientifically useful: temperature, SOC window, C-rate
- good for grouped-by-protocol or grouped-by-temperature evaluation

Immediate check:

```text
Can one CALCE subset be parsed into cell-cycle capacity rows in <1 day?
```

Promote if:

- capacity data are tabular enough for a small adapter
- protocol labels are recoverable from file names or page sections

Reject or defer if:

- file formats are too heterogeneous for a one-day intake
- cell IDs or cycle numbers are ambiguous

## 3. NASA PCoE battery aging / randomized usage

Decision:

```text
small smoke fallback
```

Sources:

- `https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/`
- classic Battery Data Set direct link on NASA PCoE page
- Randomized Battery Usage direct link on NASA PCoE page

Reason:

- official prognostics source
- direct download links exist for some classic battery datasets
- useful for proving the interface can parse non-release trajectories

Risk:

- some newer NASA datasets currently have unavailable direct downloads
- classic files may be MATLAB structs rather than simple CSV
- sample size may be too small for headline evidence

Promote if:

- `.mat` parsing produces one row per discharge cycle quickly
- EOL threshold and capacity units are clear

## Day-1 Outcome

Current decision:

```text
MATR first, CALCE fallback, NASA classic fallback.
```

No modeling until one source produces:

```text
outputs/79_battery_intake/intake_summary.json
outputs/79_battery_intake/battery_curves_long.csv
outputs/79_battery_intake/battery_cell_metadata.csv
```

## First Command Once Raw CSV Exists

```powershell
python battery\scripts\79_battery_intake.py `
  --curves <raw_or_converted_cycle_summary.csv> `
  --metadata <optional_cell_metadata.csv> `
  --dataset <dataset_name> `
  --output-dir outputs\79_battery_intake `
  --cell-col <cell_id_col> `
  --cycle-col <cycle_col> `
  --capacity-col <capacity_col> `
  --normalized-capacity-col <optional_norm_col> `
  --temperature-col <optional_temperature_col> `
  --protocol-col <optional_protocol_col>
```

## Review Question

Before writing a model, ask:

```text
Does this dataset test partial-observation state inference,
or only reproduce a battery-specific cycle-life benchmark?
```

If the answer is the second one, keep it as a smoke test only.
