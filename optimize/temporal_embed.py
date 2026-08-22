"""Temporal convolution embedding for release curves.

Replaces FCEmbedding (flat MLP on 64-D vector) with a dilated causal
convolution network that respects the time-series structure of Q(t).

Architecture: 5 layers, kernel=3, dilation=[1,2,4,8,16]. Receptive
field = 63 ≈ T=64. Each layer: Conv1d → BatchNorm → GELU. Final
global average pool → Linear → embed_dim.

Why causal: Q(t) is cumulative release; future values must not leak
into the embedding of an early-time partial observation.

Why dilated: exponential dilation gives O(log T) depth for full
receptive field, vs O(T) for a plain conv stack.

Why not Transformer: at T=64, the quadratic attention is wasteful
for a 1-D signal with strong local structure. Conv is 10x cheaper
and the inductive bias (local + translation-equivariant) matches
the physics.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalConvEmbedding(nn.Module):
    """Dilated causal conv → global pool → Linear → embed_dim."""

    def __init__(
        self,
        output_dim: int = 16,
        n_channels: int = 32,
        n_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        # Input projection: 1-D signal → n_channels
        self.in_proj = nn.Conv1d(1, n_channels, 1)
        # Dilated causal conv stack
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            # Causal left-padding: (kernel_size - 1) * dilation
            pad = (kernel_size - 1) * dilation
            self.layers.append(
                nn.Conv1d(n_channels, n_channels, kernel_size,
                          dilation=dilation, padding=pad)
            )
            self.norms.append(nn.BatchNorm1d(n_channels))
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        # Output projection
        self.out_proj = nn.Linear(n_channels, output_dim)

    def forward(self, Q: Tensor) -> Tensor:
        """
        Args:
            Q: (B, T) release curves, values in [0, 1].
        Returns:
            z: (B, output_dim) embedding.
        """
        if Q.ndim == 1:
            Q = Q.unsqueeze(0)
        x = Q.unsqueeze(1)                      # (B, 1, T)
        x = self.in_proj(x)                      # (B, C, T)
        for conv, norm, drop in zip(self.layers, self.norms, self.dropouts):
            x = conv(x)                          # (B, C, T + pad)
            x = x[..., :Q.shape[1]]              # causal trim to T
            x = norm(x)
            x = torch.nn.functional.gelu(x)
            x = drop(x)
        x = x.mean(dim=-1)                       # global average pool (B, C)
        return self.out_proj(x)                   # (B, output_dim)

    @property
    def input_dim(self) -> int:
        """Compatibility with FCEmbedding interface."""
        return -1  # dynamic, inferred from Q length
