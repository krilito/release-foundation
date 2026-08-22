"""Tests for the Stage 2 formulation featurizer + encoder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from encoder import (
    OTHER_POLYMER,
    PLGA_CONTINUOUS_COLS,
    FormulationFeaturizer,
    MLPFormulationEncoder,
    parse_dp_group,
)


# ----------------------------------------------------------------------
# parse_dp_group
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("DEX-PLGA", ("DEX", "PLGA")),
        ("PTX-PVL-co-PAVL", ("PTX", "PVL-co-PAVL")),
        ("PTX-PLGA-co-PAVL", ("PTX", "PLGA-co-PAVL")),
        ("TAA-PLA-co-PALA", ("TAA", "PLA-co-PALA")),
        ("CBD-PCL", ("CBD", "PCL")),
        # multi-hyphen drug name must parse as drug='5-FU', polymer='PLGA'
        ("5-FU-PLGA", ("5-FU", "PLGA")),
        # bare polymer with no drug prefix
        ("PLGA", ("", "PLGA")),
        # unknown polymer suffix falls back to OTHER
        ("FOO-POLYWEIRD", ("FOO-POLYWEIRD", OTHER_POLYMER)),
    ],
)
def test_parse_dp_group(raw: str, expected: tuple[str, str]) -> None:
    assert parse_dp_group(raw) == expected


def test_parse_dp_group_prefers_longest_match() -> None:
    # 'PLGA-co-PAVL' must not be parsed as polymer='PAVL' (which is not in
    # the known list anyway, but the principle: longest known suffix wins).
    drug, poly = parse_dp_group("PTX-PLGA-co-PAVL")
    assert poly == "PLGA-co-PAVL"
    assert drug == "PTX"


# ----------------------------------------------------------------------
# Featurizer
# ----------------------------------------------------------------------
@pytest.fixture
def sample_df() -> pd.DataFrame:
    rows = [
        {"DP_Group": "DEX-PLGA", "LA/GA": 0.0, "Polymer_MW": 49000.0,
         "CL Ratio": 0.0, "Drug_Tm": 262.0, "Drug_Pka": 12.4,
         "Initial D/M ratio": 0.2, "DLC": 0.15, "SA-V": 422.0,
         "SE": 0.0, "Drug_Mw": 392.0, "Drug_TPSA": 94.8,
         "Drug_NHA": 6.0, "Drug_LogP": 1.9},
        {"DP_Group": "PTX-PVL-co-PAVL", "LA/GA": 0.5, "Polymer_MW": 21000.0,
         "CL Ratio": 0.07, "Drug_Tm": 213.0, "Drug_Pka": 11.7,
         "Initial D/M ratio": 0.1, "DLC": 0.10, "SA-V": 380.0,
         "SE": 1.0, "Drug_Mw": 853.0, "Drug_TPSA": 221.0,
         "Drug_NHA": 14.0, "Drug_LogP": 3.0},
        {"DP_Group": "5-FU-PLGA", "LA/GA": 0.5, "Polymer_MW": 30000.0,
         "CL Ratio": 0.0, "Drug_Tm": 282.0, "Drug_Pka": 8.0,
         "Initial D/M ratio": 0.3, "DLC": 0.20, "SA-V": 410.0,
         "SE": 0.0, "Drug_Mw": 130.0, "Drug_TPSA": 65.7,
         "Drug_NHA": 4.0, "Drug_LogP": -0.9},
    ]
    return pd.DataFrame(rows)


def test_featurizer_fit_includes_other_bucket(sample_df: pd.DataFrame) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    assert OTHER_POLYMER in feat.polymer_families
    assert "PLGA" in feat.polymer_families
    assert "PVL-co-PAVL" in feat.polymer_families


def test_featurizer_input_dim(sample_df: pd.DataFrame) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    assert feat.input_dim == len(PLGA_CONTINUOUS_COLS) + len(feat.polymer_families)


def test_featurizer_transform_shape_and_one_hot(sample_df: pd.DataFrame) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    x = feat.transform(sample_df)
    assert x.shape == (3, feat.input_dim)
    # Continuous block is z-scored on training data: per-column mean ~ 0.
    cont = x[:, : len(PLGA_CONTINUOUS_COLS)]
    torch.testing.assert_close(
        cont.mean(dim=0), torch.zeros(len(PLGA_CONTINUOUS_COLS)),
        atol=1e-5, rtol=0,
    )
    # One-hot block: exactly one 1 per row, all other entries 0.
    oh = x[:, len(PLGA_CONTINUOUS_COLS):]
    assert torch.all(oh.sum(dim=1) == 1.0)
    assert torch.all((oh == 0.0) | (oh == 1.0))


def test_featurizer_transform_unseen_polymer_falls_back_to_OTHER(
    sample_df: pd.DataFrame,
) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    novel = sample_df.iloc[[0]].copy()
    novel.loc[novel.index[0], "DP_Group"] = "XYZ-POLYWEIRD"
    x = feat.transform(novel)
    other_idx = feat.polymer_families.index(OTHER_POLYMER)
    oh = x[:, len(PLGA_CONTINUOUS_COLS):]
    assert oh[0, other_idx] == 1.0
    assert oh.sum() == 1.0


def test_featurizer_drug_ids(sample_df: pd.DataFrame) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    drug_ids = feat.parse_drug_ids(sample_df)
    assert drug_ids == ["DEX", "PTX", "5-FU"]


def test_featurizer_missing_column_raises(sample_df: pd.DataFrame) -> None:
    bad = sample_df.drop(columns=["LA/GA"])
    with pytest.raises(KeyError):
        FormulationFeaturizer.fit(bad)


def test_featurizer_missing_dp_group_raises(sample_df: pd.DataFrame) -> None:
    bad = sample_df.drop(columns=["DP_Group"])
    with pytest.raises(KeyError):
        FormulationFeaturizer.fit(bad)


# ----------------------------------------------------------------------
# MLPFormulationEncoder
# ----------------------------------------------------------------------
def test_mlp_encoder_forward_shape(sample_df: pd.DataFrame) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    enc = MLPFormulationEncoder(input_dim=feat.input_dim, embed_dim=64)
    x = feat.transform(sample_df)
    h = enc(x)
    assert h.shape == (3, 64)


def test_mlp_encoder_validates_input_shape(sample_df: pd.DataFrame) -> None:
    feat = FormulationFeaturizer.fit(sample_df)
    enc = MLPFormulationEncoder(input_dim=feat.input_dim, embed_dim=64)
    with pytest.raises(ValueError):
        enc(torch.randn(2, feat.input_dim + 1))
    with pytest.raises(ValueError):
        enc(torch.randn(feat.input_dim))


def test_mlp_encoder_validates_construction_args() -> None:
    with pytest.raises(ValueError):
        MLPFormulationEncoder(input_dim=0, embed_dim=64)
    with pytest.raises(ValueError):
        MLPFormulationEncoder(input_dim=10, embed_dim=0)
    with pytest.raises(ValueError):
        MLPFormulationEncoder(input_dim=10, embed_dim=64, dropout=1.0)
