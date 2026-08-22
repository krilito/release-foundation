"""65 - FIB-CASP zero-shot: formulation-only input, no early Q.

The simpler answer to the "we want zero-shot" requirement: take the
existing FIB-CASP machinery (script 62) and train the RF point predictor
on formulation features alone. Everything else stays the same:

  - RF -> theta_RF                              (formulation only)
  - Fisher info @theta_RF over c.t_obs           (per-curve closed-form)
  - low-rank Gaussian posterior + sample + recenter

This is a few-line change from 62. The point of running it: see how the
FIB UQ quality holds up when the underlying RF point estimate is the
weaker (formulation-only) one. Two comparators:

  Toronto RF zero-shot (script 15)               R^2 ~0.33
  Script 45 RF -> theta_oracle (form only)       R^2 ~0.91 (random)
                                                       ~0.67 (by_drug)
                                                       ~0.67 (by_polymer)

The point of script 45's 0.91 is that formulation alone CAN drive an
RF -> theta -> ODE pipeline to high R^2 on random splits, even without
early observations. Until now we haven't built that pipeline with
calibrated PI. This script does.

Outputs:
  outputs/65_fib_casp_zero_shot/{dataset}/{scheme}/per_curve.csv
  outputs/65_fib_casp_zero_shot/aggregate_table.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import script 62 as a module so we can reuse loaders + evaluate_fold.
m62 = _load_script(
    Path(__file__).resolve().parent / "62_fib_casp_benchmark.py", "_m62_for_65",
)


def slice_to_formulation_only(bundle):
    """Drop the early-Q features from the bundle to make it zero-shot."""
    X_form = bundle.X[:, :bundle.n_form].copy()    # keep only formulation cols
    bundle.X = X_form
    # n_form unchanged; total features now equals n_form
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/65_fib_casp_zero_shot"))
    ap.add_argument("--datasets", nargs="+",
                    default=["cross321", "internal181"])
    ap.add_argument("--schemes", nargs="+",
                    default=["random_5fold", "group_by_drug", "group_by_polymer"])
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--sigma0-frac", type=float, default=0.05)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loaders = {
        "cross321": m62.load_cross321,
        "internal181": m62.load_internal181,
        "liposome": m62.load_liposome,
    }

    summary_rows: list[dict] = []
    for ds_name in args.datasets:
        print(f"\n[65] === dataset: {ds_name} (FORMULATION ONLY) ===", flush=True)
        bundle = loaders[ds_name]()
        n_form_before = bundle.n_form
        f_before = bundle.X.shape[1]
        bundle = slice_to_formulation_only(bundle)
        print(f"  features: {f_before} -> {bundle.X.shape[1]} (dropped early Q)",
              flush=True)

        for scheme in args.schemes:
            if ds_name == "liposome" and scheme == "group_by_polymer":
                scheme_name_used = "group_by_method"
            else:
                scheme_name_used = scheme
            print(f"  scheme: {scheme_name_used}", flush=True)
            t0 = time.time()
            row = m62.run_dataset_scheme(
                bundle, scheme=scheme, out_dir=args.out,
                n_samples=args.n_samples, rank=args.rank, alpha=args.alpha,
                sigma_0_frac=args.sigma0_frac, n_folds=args.n_folds, seed=args.seed,
            )
            row["scheme"] = scheme_name_used
            row["wallclock_s"] = round(time.time() - t0, 1)
            summary_rows.append(row)
            print(f"  -> cell done in {row['wallclock_s']}s  "
                  f"r2_med={row.get('r2_median', float('nan')):+.4f}  "
                  f"cov90={row.get('cov90_mean', float('nan')):.3f}", flush=True)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.out / "aggregate_table.csv", index=False)
    print("\n[65] === Aggregate (zero-shot) ===", flush=True)
    print(df_summary.to_string(index=False), flush=True)

    with open(args.out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== FIB-CASP zero-shot benchmark (formulation only) ===\n\n")
        f.write(f"datasets : {args.datasets}\n")
        f.write(f"schemes  : {args.schemes}\n")
        f.write(f"n_folds  : {args.n_folds}\n\n")
        f.write(df_summary.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
