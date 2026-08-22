# B5 Dependency Route — 2026-05-30

Purpose:
- choose the narrowest dependency route that can execute the B5 HMC/NUTS benchmark
- avoid reopening benchmark design, posterior math, or environment scope

Status:
- subset is locked by `scripts/27f_export_hmc_subset.py`
- log-posterior backend is locked by `scripts/27h_b5_logposterior_backend.py`
- differentiable torch potential is validated by
  `outputs/27h_b5_logposterior_backend_torch_smoke/torch_smoke_summary.json`
- runner entrypoint exists in `scripts/27i_b5_hmc_pyro_runner.py`
- the remaining blocker is sampler dependency installation

## Environment facts

Verified locally on 2026-05-30:

- `pyro` / `pyro-ppl`: not installed
- `numpyro`: not installed
- `jax`: not installed
- `uv`: installed and available

## Recommendation

Use `pyro-ppl`, not `numpyro`, as the first B5 dependency.

Why this is the narrowest route:

1. The current B5 backend already exposes a differentiable **PyTorch** potential.
2. `pyro-ppl` can consume that potential directly for HMC / NUTS.
3. `numpyro` would pull in `jax` and likely open a second environment/debugging lane.
4. The repo already depends on `torch`, so `pyro-ppl` extends the current stack
   instead of introducing a parallel autodiff runtime.

## Why not `numpyro` first

- no `jax` is installed
- no JAX backend exists in repo code
- the current B5 posterior backend is torch-native, not JAX-native
- converting `27h` to JAX before running the first real B5 posterior would add
  engineering work that does not improve benchmark validity

## Exact minimal action once approved

1. Append an ADR entry to `DECISIONS.md`
   - justify `pyro-ppl` specifically for B5 HMC/NUTS
   - explain why `numpyro` is rejected for the first implementation

2. Add the dependency with uv

```powershell
uv add pyro-ppl
```

3. Run the first real posterior on a single locked claim-test curve

```powershell
python scripts/27i_b5_hmc_pyro_runner.py --curve-id 11 --kernel nuts --num-samples 400 --warmup-steps 200
```

4. If that works, expand to a small panel before the full 38-curve claim subset

## What this does not authorize

- no claim that B5 is done
- no move to the 50-curve dev extension yet
- no new JAX / numpyro lane
- no edits to user-modified files without approval

## Success criterion for the next increment

After approval, the next real milestone is:

- one successful Pyro/NUTS posterior run on one locked canonical test curve
- posterior draws exported
- summary JSON exported
- no benchmark redefinition
