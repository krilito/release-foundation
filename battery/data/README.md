# Battery data area

Do not commit raw battery datasets here.

Use this folder only for lightweight notes, schemas, and generated
non-sensitive examples. Real downloads belong in ignored local paths and
should be summarized into `outputs/<script_name>/`.

Required canonical columns for an aligned battery trajectory table:

```text
cell_id
cycle_number
capacity
normalized_capacity
optional_resistance
optional_coulombic_efficiency
temperature_C
protocol_id
```

Required canonical columns for battery metadata:

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
