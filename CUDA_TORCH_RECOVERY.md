# CUDA PyTorch Recovery Note

Date checked: 2026-05-22

## Current verdict

The GPU and NVIDIA driver are OK. The broken layer was the project virtual
environment:

- Hardware check: `nvidia-smi` sees `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Driver check: NVIDIA driver `576.88`, reported CUDA runtime `12.9`.
- Broken Python state before repair: `.venv` had `torch 2.12.0+cpu`.
- Broken torch result before repair: `torch.cuda.is_available()` returned `False`.
- Repaired Python state: `.venv` now has `torch 2.11.0+cu128`.
- Repaired torch result: `torch.cuda.is_available()` returns `True`.

This was not a CUDA driver failure. It was a CPU-only PyTorch wheel replacing
the manually installed CUDA PyTorch wheel.

## Why it happened

`pyproject.toml` only says:

```toml
torch>=2.4
```

On Windows, resolving torch from normal PyPI can install a CPU-only wheel. In
this repo, `uv run`, `uv sync`, or dependency changes may re-resolve torch and
overwrite the CUDA wheel with a CPU wheel.

The repo's `HANDOFF.md` already records the operational rule: do not use
`uv run python ...` for training or SBC jobs. Use the venv Python directly.

## Symptoms observed

Training log showed CPU execution:

```text
outputs/04_curve_posterior/training_log.txt
device              : cpu
elapsed_seconds     : 699.1
```

The multi-seed SBC run stopped before producing a complete summary:

```text
outputs/sprint1_run_20260522_104249.log
[05b] seed 0 KS p: ... log_tau_burst=0.002 ...
[05b] seed 1 KS p: ... log_tau_burst=0.000 ...
[05b] seed 7 (3/10)
Drawing 1000 samples ... 95%
```

Because that run used a CPU-trained posterior, its SBC failures should not be
treated as final evidence about identifiability until GPU retraining and SBC
are rerun cleanly.

## Check commands

Run these from `D:\release-foundation`:

```powershell
nvidia-smi

.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected healthy output:

```text
2.11.0+cu128
True
12.8
1
NVIDIA GeForce RTX 4060 Laptop GPU
```

For a stronger smoke test:

```powershell
.\.venv\Scripts\python.exe -c "import torch; x=torch.ones(2,device='cuda'); print(torch.__version__, torch.cuda.is_available(), x.device, float(x.sum().item()))"
```

Expected:

```text
2.11.0+cu128 True cuda:0 2.0
```

## Repair command

If torch is back to `+cpu`, repair it with:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps --progress-bar off torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Why these flags:

- `--force-reinstall` actually replaces the CPU wheel.
- `--no-deps` avoids a broad dependency re-resolution.
- The PyTorch CUDA index is required for the `+cu128` wheel.

The simpler command may hang or fail to replace the CPU wheel:

```powershell
.\.venv\Scripts\pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

If it leaves `torch 2.12.0+cpu` in place, use the force-reinstall command
above.

## Commands to avoid

Avoid these for training/SBC workflow in this repo:

```powershell
uv run python scripts\...
uv sync
uv add ...
```

They can put the environment back onto PyPI's CPU torch. If dependency changes
are truly needed, re-check CUDA immediately afterward and reinstall the CUDA
wheel if necessary.

## Safe invocation pattern

Use PowerShell and call the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe scripts\04_train_curve_posterior.py ...
.\.venv\Scripts\python.exe scripts\05b_sbc_multi_seed.py ...
```

Do not use Bash-style paths for the Windows venv.

## Next scientific action

The CPU run is not a clean basis for judging `log_tau_burst` identifiability.
After confirming CUDA is available:

1. Archive or ignore the incomplete CPU SBC run as environment-contaminated.
2. Retrain script 04 on GPU.
3. Rerun script 05b multi-seed SBC on the GPU-trained posterior.
4. Only then interpret whether `log_tau_burst` fails SBC because of true
   identifiability collapse.

