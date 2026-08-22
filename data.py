"""
Data loaders.

Real-data layer: the internal 181-formulation PLGA benchmark, plus future
external sets (Mendeley 321, literature-mined curves).

Synthetic-data layer: sampling from a `ReleaseSimulator` for amortized SBI
training. Synthetic data is regenerated from seeds and never committed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from simulator import ReleaseSimulator


@dataclass(frozen=True)
class ReleaseCurve:
    """One formulation: (descriptors, sampled times, observed release)."""

    formulation_id: str
    t: Tensor              # (T_i,) days
    Q: Tensor              # (T_i,) cumulative release in [0, 1]
    descriptors: dict[str, float]


def load_plga_181(
    csv_path: str | Path,
    formulation_col: str = "Experimental_index",
    time_col: str = "Time",
    release_col: str = "Release",
    descriptor_cols: list[str] | None = None,
) -> list[ReleaseCurve]:
    """Load the internal 181-formulation PLGA benchmark.

    Defaults match the internal `Dataset_17_feat_augmented.csv` schema.

    Returns one `ReleaseCurve` per formulation, sorted by time.
    """
    df = pd.read_csv(csv_path)

    required = {formulation_col, time_col, release_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns {missing}. "
            f"Available: {sorted(df.columns)}. "
            f"Pass column names explicitly to load_plga_181()."
        )

    if descriptor_cols is None:
        # Keep the descriptor contract numeric for the Phase 1 encoder.
        blocked = {formulation_col, time_col, release_col}
        descriptor_cols = [
            c for c in df.columns
            if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
        ]

    curves: list[ReleaseCurve] = []
    for fid, group in df.groupby(formulation_col):
        g = group.sort_values(time_col)
        descriptors = {c: float(g[c].iloc[0]) for c in descriptor_cols}
        curves.append(
            ReleaseCurve(
                formulation_id=str(fid),
                t=torch.tensor(g[time_col].to_numpy(), dtype=torch.float32),
                Q=torch.tensor(g[release_col].to_numpy(), dtype=torch.float32).clamp(0, 1),
                descriptors=descriptors,
            )
        )
    return curves


def sample_synthetic_pairs(
    simulator: ReleaseSimulator,
    n: int,
    t_grid: Tensor,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Sample (θ, Q) pairs from a simulator for amortized SBI training.

    Returns:
        theta: (n, n_params)
        Q:     (n, T)
    """
    g = torch.Generator().manual_seed(seed)
    # `sbi` priors do not consume a generator; we set the global seed for
    # this draw and restore afterwards.
    prior = simulator.prior()
    with torch.random.fork_rng():
        torch.manual_seed(int(g.initial_seed()) & 0xFFFFFFFF)
        theta = prior.sample((n,))
    Q = simulator.simulate(theta, t_grid)
    return theta, Q
