"""
Mechanism-agnostic posterior-family object for unified drug-release inference.

Purpose:
    Make the project's "shared posterior-family object" executable rather
    than rhetorical. This schema is intentionally minimal:

        sparse observations + formulation context
            -> feasible latent release-state family
            -> mechanism-specific decoder handle

    The latent state does NOT need to be the same physical theta across every
    mechanism. What is shared is the object shape used by downstream
    forecasting, uncertainty, and decision-support code.

Design:
    - `mechanism_id` identifies the mechanism family, not the exact decoder.
    - `decoder_handle` tells downstream code which simulator/decoder to call.
    - `z_center`, `z_basis`, `z_scale` define a low-rank feasible family in a
      mechanism-local latent coordinate system.
    - `assay_context`, `observation_context`, and `metadata` carry
      mechanism-specific and deployment-specific context without hard-coding it
      into the core schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_float_vector(values: list[float] | tuple[float, ...] | np.ndarray, name: str) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return [float(x) for x in arr.tolist()]


def _as_float_matrix(
    values: list[list[float]] | tuple[tuple[float, ...], ...] | np.ndarray,
    name: str,
) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return [[float(x) for x in row] for row in arr.tolist()]


@dataclass(slots=True)
class ReleasePosteriorFamily:
    """Low-rank feasible release-state family with mechanism-specific decoding."""

    mechanism_id: str
    decoder_handle: str
    z_center: list[float]
    z_basis: list[list[float]] = field(default_factory=list)
    z_scale: list[float] = field(default_factory=list)
    assay_context: dict[str, Any] = field(default_factory=dict)
    observation_context: dict[str, Any] = field(default_factory=dict)
    calibration_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mechanism_id = str(self.mechanism_id).strip()
        self.decoder_handle = str(self.decoder_handle).strip()
        if not self.mechanism_id:
            raise ValueError("mechanism_id must be non-empty")
        if not self.decoder_handle:
            raise ValueError("decoder_handle must be non-empty")

        self.z_center = _as_float_vector(self.z_center, "z_center")
        self.z_basis = _as_float_matrix(self.z_basis, "z_basis") if self.z_basis else []
        self.z_scale = _as_float_vector(self.z_scale, "z_scale") if self.z_scale else []
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        latent_dim = len(self.z_center)
        if latent_dim == 0:
            raise ValueError("z_center must have at least one element")
        if not self.z_basis and self.z_scale:
            raise ValueError("z_scale provided without z_basis")
        if self.z_basis and not self.z_scale:
            raise ValueError("z_basis provided without z_scale")
        if self.z_basis:
            rank = len(self.z_basis)
            if len(self.z_scale) != rank:
                raise ValueError(
                    f"z_scale length {len(self.z_scale)} must match z_basis rank {rank}"
                )
            for i, row in enumerate(self.z_basis):
                if len(row) != latent_dim:
                    raise ValueError(
                        f"z_basis row {i} has length {len(row)} but latent dim is {latent_dim}"
                    )

    @property
    def latent_dim(self) -> int:
        return len(self.z_center)

    @property
    def rank(self) -> int:
        return len(self.z_basis)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "decoder_handle": self.decoder_handle,
            "z_center": self.z_center,
            "z_basis": self.z_basis,
            "z_scale": self.z_scale,
            "assay_context": self.assay_context,
            "observation_context": self.observation_context,
            "calibration_context": self.calibration_context,
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReleasePosteriorFamily":
        return cls(
            mechanism_id=payload["mechanism_id"],
            decoder_handle=payload["decoder_handle"],
            z_center=payload["z_center"],
            z_basis=payload.get("z_basis", []),
            z_scale=payload.get("z_scale", []),
            assay_context=dict(payload.get("assay_context", {})),
            observation_context=dict(payload.get("observation_context", {})),
            calibration_context=dict(payload.get("calibration_context", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReleasePosteriorFamily":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("json payload must be an object")
        return cls.from_dict(payload)

    @classmethod
    def point_estimate(
        cls,
        mechanism_id: str,
        decoder_handle: str,
        z_center: list[float] | tuple[float, ...] | np.ndarray,
        assay_context: dict[str, Any] | None = None,
        observation_context: dict[str, Any] | None = None,
        calibration_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ReleasePosteriorFamily":
        return cls(
            mechanism_id=mechanism_id,
            decoder_handle=decoder_handle,
            z_center=_as_float_vector(z_center, "z_center"),
            z_basis=[],
            z_scale=[],
            assay_context=dict(assay_context or {}),
            observation_context=dict(observation_context or {}),
            calibration_context=dict(calibration_context or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def low_rank_family(
        cls,
        mechanism_id: str,
        decoder_handle: str,
        z_center: list[float] | tuple[float, ...] | np.ndarray,
        z_basis: list[list[float]] | tuple[tuple[float, ...], ...] | np.ndarray,
        z_scale: list[float] | tuple[float, ...] | np.ndarray,
        assay_context: dict[str, Any] | None = None,
        observation_context: dict[str, Any] | None = None,
        calibration_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ReleasePosteriorFamily":
        return cls(
            mechanism_id=mechanism_id,
            decoder_handle=decoder_handle,
            z_center=_as_float_vector(z_center, "z_center"),
            z_basis=_as_float_matrix(z_basis, "z_basis"),
            z_scale=_as_float_vector(z_scale, "z_scale"),
            assay_context=dict(assay_context or {}),
            observation_context=dict(observation_context or {}),
            calibration_context=dict(calibration_context or {}),
            metadata=dict(metadata or {}),
        )

    def with_calibration(self, calibration_context: dict[str, Any]) -> "ReleasePosteriorFamily":
        merged = dict(self.calibration_context)
        merged.update(dict(calibration_context))
        return ReleasePosteriorFamily(
            mechanism_id=self.mechanism_id,
            decoder_handle=self.decoder_handle,
            z_center=self.z_center,
            z_basis=self.z_basis,
            z_scale=self.z_scale,
            assay_context=self.assay_context,
            observation_context=self.observation_context,
            calibration_context=merged,
            metadata=self.metadata,
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "decoder_handle": self.decoder_handle,
            "latent_dim": self.latent_dim,
            "rank": self.rank,
            "assay_keys": sorted(self.assay_context.keys()),
            "observation_keys": sorted(self.observation_context.keys()),
            "calibration_keys": sorted(self.calibration_context.keys()),
            "metadata_keys": sorted(self.metadata.keys()),
        }
