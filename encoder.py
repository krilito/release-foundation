"""
Formulation descriptor encoders.

Stage 2 maps formulation descriptors `x` to a latent embedding consumed by
`DescriptorPosterior`. Keep descriptor preprocessing here so downstream
posterior code sees one numeric tensor contract instead of CSV-specific
column handling.

`DP_Group` is a concatenated label like 'PTX-PVL-co-PAVL' or '5-FU-PLGA'
that mixes drug identity and polymer family. With 34 unique values across
181 formulations, raw one-hot would be sparse and easy to overfit. The
Phase 1 featurizer therefore parses DP_Group into drug_id + polymer_family
and only one-hots polymer_family (the mechanistically relevant axis;
drug chemistry is already captured by Drug_Mw / Drug_TPSA / Drug_LogP /
Drug_Pka / Drug_Tm / Drug_NHA). drug_id is exposed for stratification and
audit but not fed to the encoder in v0. See ADR-018.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


PLGA_CONTINUOUS_COLS = (
    "LA/GA",
    "Polymer_MW",
    "CL Ratio",
    "Drug_Tm",
    "Drug_Pka",
    "Initial D/M ratio",
    "DLC",
    "SA-V",
    "SE",
    "Drug_Mw",
    "Drug_TPSA",
    "Drug_NHA",
    "Drug_LogP",
)

# Longest-first matters: PLGA-co-PAVL must match before PLGA when scanning
# the end of a DP_Group string. Without this, 'PTX-PLGA-co-PAVL' would
# parse to (drug='PTX-PLGA-co', polymer='PAVL') if order were not enforced.
_KNOWN_POLYMER_FAMILIES = (
    "PLGA-co-PAVL",
    "PLGA-co-PALA",
    "PLA-co-PALA",
    "PVL-co-PAVL",
    "PLGA",
    "PLA",
    "PCL",
    "PEA",
)
OTHER_POLYMER = "OTHER"


def parse_dp_group(value: str) -> tuple[str, str]:
    """Split a DP_Group label into (drug_id, polymer_family).

    Uses longest-suffix matching against `_KNOWN_POLYMER_FAMILIES` so
    multi-hyphen drug names (e.g., '5-FU-PLGA' → ('5-FU', 'PLGA')) and
    co-polymer names (e.g., 'PTX-PLGA-co-PAVL' → ('PTX', 'PLGA-co-PAVL'))
    are handled correctly. Returns `(value, OTHER_POLYMER)` if no known
    polymer matches; this lets downstream one-hot encoding stay valid
    when a future row carries a polymer outside the Phase-1 set.
    """
    s = str(value)
    for poly in _KNOWN_POLYMER_FAMILIES:
        if s == poly:
            return "", poly
        suffix = "-" + poly
        if s.endswith(suffix):
            return s[: -len(suffix)], poly
    return s, OTHER_POLYMER


@dataclass(frozen=True)
class FormulationFeaturizer:
    """Convert formulation descriptor rows into model-ready tensors.

    Continuous columns are z-scored from the training distribution.
    The `DP_Group` column is parsed into polymer_family + drug_id; only
    polymer_family is one-hot encoded. The `OTHER` bucket is always
    included in the one-hot vocabulary so transform never crashes on a
    polymer name not seen at fit time — such rows map to OTHER instead.

    drug_id is parsed and exposed via `parse_drug_ids()` for grouped-
    holdout / audit code, but is NOT part of the feature tensor in v0.
    See ADR-018 for why.
    """

    continuous_cols: tuple[str, ...]
    polymer_families: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    DP_GROUP_COL = "DP_Group"

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        continuous_cols: Sequence[str] = PLGA_CONTINUOUS_COLS,
    ) -> FormulationFeaturizer:
        continuous_cols = tuple(continuous_cols)
        missing = [c for c in continuous_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing continuous columns: {missing}")
        if cls.DP_GROUP_COL not in df.columns:
            raise KeyError(f"Missing required column {cls.DP_GROUP_COL!r}")

        x_cont = df.loc[:, continuous_cols].astype(float).to_numpy()
        mean = x_cont.mean(axis=0)
        std = x_cont.std(axis=0, ddof=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)

        # Always include OTHER even if no training row uses it; the
        # featurizer must accept unseen polymer names at deploy time.
        seen = {parse_dp_group(v)[1] for v in df[cls.DP_GROUP_COL].dropna()}
        seen.add(OTHER_POLYMER)
        polymer_families = tuple(sorted(seen))

        return cls(
            continuous_cols=continuous_cols,
            polymer_families=polymer_families,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    @property
    def input_dim(self) -> int:
        return len(self.continuous_cols) + len(self.polymer_families)

    def transform(self, df: pd.DataFrame) -> Tensor:
        missing = [c for c in self.continuous_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing continuous columns: {missing}")
        if self.DP_GROUP_COL not in df.columns:
            raise KeyError(f"Missing required column {self.DP_GROUP_COL!r}")

        x_cont = df.loc[:, self.continuous_cols].astype(float).to_numpy(dtype=np.float32)
        x_cont = (x_cont - self.mean) / self.std

        idx_of = {p: i for i, p in enumerate(self.polymer_families)}
        block = np.zeros((len(df), len(self.polymer_families)), dtype=np.float32)
        for row_idx, raw in enumerate(df[self.DP_GROUP_COL]):
            _, poly = parse_dp_group(raw)
            i = idx_of.get(poly, idx_of[OTHER_POLYMER])
            block[row_idx, i] = 1.0

        x = np.concatenate([x_cont, block], axis=1)
        return torch.tensor(x, dtype=torch.float32)

    def parse_drug_ids(self, df: pd.DataFrame) -> list[str]:
        """Drug-identity column parsed from DP_Group, for audit / grouping."""
        if self.DP_GROUP_COL not in df.columns:
            raise KeyError(f"Missing required column {self.DP_GROUP_COL!r}")
        return [parse_dp_group(v)[0] for v in df[self.DP_GROUP_COL]]


class FormulationEncoder(nn.Module, ABC):
    """Abstract encoder from numeric formulation features to an embedding."""

    input_dim: int
    embed_dim: int

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of formulation features, shape `(B, input_dim)`."""


class MLPFormulationEncoder(FormulationEncoder):
    """Simple Phase 1 tabular encoder for PLGA descriptors."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)

        layers: list[nn.Module] = []
        prev = self.input_dim
        for hidden in hidden_dims:
            if hidden <= 0:
                raise ValueError(f"hidden_dims must be > 0, got {hidden}")
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = int(hidden)
        layers.append(nn.Linear(prev, self.embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x has dim {x.shape[1]}, expected {self.input_dim}")
        return self.net(x)
