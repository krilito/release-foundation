"""Drug-polymer interaction features for formulation encoding.

The current MLPFormulationEncoder feeds 13 continuous + 8 one-hot
features into an MLP. Drug-polymer interactions (e.g., PTX-PLGA vs
PTX-PCL) are critical for release behavior but the MLP must learn
these from scratch with only ~133 training samples.

This module adds explicit interaction features before the MLP:

    drug_feat (6-d)  ×  polymer_onehot (8-d)  →  48 cross features

These are concatenated with the original features, giving the MLP
a 61-d input instead of 21-d. The cross features directly encode
"which drug + which polymer" combinations.

Why outer product, not element-wise: the interaction between Drug_Mw
and PLGA-vs-PCL is not decomposable into (Drug_Mw effect) × (PLGA
effect). The outer product lets the model learn each (drug_dim,
polymer_class) pair independently.

Why only drug × polymer, not all × all: the 13 continuous features
include both drug and polymer descriptors (LA/GA, Polymer_MW are
polymer-side). Mixing drug-side and polymer-side continuous features
into the interaction would be redundant with the MLP's capacity.
The one-hot polymer is the one axis where explicit crossing is needed
because it is categorical.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Drug-side continuous columns (descriptors of the drug molecule).
DRUG_COLS = ("Drug_Mw", "Drug_TPSA", "Drug_LogP", "Drug_Pka", "Drug_Tm", "Drug_NHA")
# These are the first 6 columns in PLGA_CONTINUOUS_COLS. The remaining
# 7 are polymer/process-side (LA/GA, Polymer_MW, CL Ratio, D/M, DLC, SA-V, SE).


def add_interaction_features(
    x: Tensor,
    n_continuous: int | None = None,
    n_drug: int | None = None,
    n_polymer: int = 0,
) -> Tensor:
    """Append self-interaction (quadratic) features.

    When n_polymer=0 (no one-hot columns), generates pairwise products
    of all continuous features: x_i * x_j for i < j. This captures
    drug-polymer and other cross-term interactions without requiring
    one-hot encoding.

    When n_polymer>0, uses the drug×polymer outer product instead.

    Args:
        x: (B, D) features.
        n_continuous: number of continuous columns (default: all of x).
        n_drug: number of drug-side columns (default: first half).
        n_polymer: number of polymer one-hot columns (0 = no one-hot).

    Returns:
        x_enhanced: (B, D + n_cross) with interaction features appended.
    """
    B, D = x.shape
    if n_continuous is None:
        n_continuous = D
    if n_polymer > 0:
        # Drug × polymer outer product
        if n_drug is None:
            n_drug = n_continuous // 2
        drug = x[:, :n_drug]
        polymer = x[:, n_continuous:n_continuous + n_polymer]
        cross = drug.unsqueeze(2) * polymer.unsqueeze(1)
        cross_flat = cross.reshape(B, n_drug * n_polymer)
        return torch.cat([x, cross_flat], dim=1)
    else:
        # Pairwise products of continuous features
        x_cont = x[:, :n_continuous]
        # Upper triangle: i < j
        products = []
        for i in range(n_continuous):
            for j in range(i + 1, n_continuous):
                products.append(x_cont[:, i] * x_cont[:, j])
        if products:
            cross = torch.stack(products, dim=1)  # (B, n_pairs)
            return torch.cat([x, cross], dim=1)
        return x


def interaction_input_dim(
    n_continuous: int = 13,
    n_polymer: int = 8,
    n_drug: int = 6,
) -> int:
    """Total feature dimension after adding interactions."""
    if n_polymer > 0:
        return n_continuous + n_polymer + n_drug * n_polymer
    else:
        n_pairs = n_continuous * (n_continuous - 1) // 2
        return n_continuous + n_pairs
