# Battery scripts

Battery-specific scripts for `release-foundation` belong here until they
are stable enough to be promoted into a shared cross-domain evaluator.

Planned scripts:

```text
79_battery_intake.py
80_battery_smoke_benchmark.py
```

Rules:

- Battery scripts must not import `D:\battery` material-screening code.
- Battery scripts must write outputs under `outputs/<script_name>/`.
- Battery scripts must accept `--seed` when randomness is used.
- Battery scripts must state whether they evaluate capacity trajectory,
  EOL, knee point, or another target.
