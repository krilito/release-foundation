"""
Mechanism-aware release target registry for the unified reporting layer.

Purpose:
    Make the target-layer contract executable rather than rhetorical. The
    unified stack can share target classes and output schemas while still
    allowing mechanism-aware defaults and caveats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReleaseTargetSpec:
    """One target definition inside the shared release-intelligence layer."""

    target_id: str
    target_class: str
    unit: str
    recommendation: str
    caveat: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_id = str(self.target_id).strip()
        self.target_class = str(self.target_class).strip()
        self.unit = str(self.unit).strip()
        self.recommendation = str(self.recommendation).strip()
        self.caveat = str(self.caveat).strip()
        if not self.target_id:
            raise ValueError("target_id must be non-empty")
        if not self.target_class:
            raise ValueError("target_class must be non-empty")
        if not self.unit:
            raise ValueError("unit must be non-empty")
        if not self.recommendation:
            raise ValueError("recommendation must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_class": self.target_class,
            "unit": self.unit,
            "recommendation": self.recommendation,
            "caveat": self.caveat,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReleaseTargetSpec":
        return cls(
            target_id=payload["target_id"],
            target_class=payload["target_class"],
            unit=payload["unit"],
            recommendation=payload["recommendation"],
            caveat=payload.get("caveat", ""),
            evidence=dict(payload.get("evidence", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ReleaseMechanismTargetProfile:
    """Mechanism-level registry entry for shared reporting targets."""

    mechanism_id: str
    default_targets: list[str]
    target_specs: list[ReleaseTargetSpec]
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mechanism_id = str(self.mechanism_id).strip()
        self.default_targets = [str(x).strip() for x in self.default_targets]
        if not self.mechanism_id:
            raise ValueError("mechanism_id must be non-empty")
        spec_ids = {spec.target_id for spec in self.target_specs}
        for target_id in self.default_targets:
            if target_id not in spec_ids:
                raise ValueError(f"default target {target_id!r} missing from target_specs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "default_targets": self.default_targets,
            "target_specs": [spec.as_dict() for spec in self.target_specs],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReleaseMechanismTargetProfile":
        return cls(
            mechanism_id=payload["mechanism_id"],
            default_targets=list(payload.get("default_targets", [])),
            target_specs=[ReleaseTargetSpec.from_dict(x) for x in payload.get("target_specs", [])],
            notes=list(payload.get("notes", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ReleaseTargetRegistry:
    """Top-level registry object for the unified target layer."""

    version: str
    profiles: list[ReleaseMechanismTargetProfile]
    shared_target_classes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.version = str(self.version).strip()
        if not self.version:
            raise ValueError("version must be non-empty")
        mechanism_ids = [p.mechanism_id for p in self.profiles]
        if len(mechanism_ids) != len(set(mechanism_ids)):
            raise ValueError("mechanism_id values must be unique")
        self.shared_target_classes = [str(x).strip() for x in self.shared_target_classes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "profiles": [profile.as_dict() for profile in self.profiles],
            "shared_target_classes": self.shared_target_classes,
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReleaseTargetRegistry":
        return cls(
            version=payload["version"],
            profiles=[ReleaseMechanismTargetProfile.from_dict(x) for x in payload.get("profiles", [])],
            shared_target_classes=list(payload.get("shared_target_classes", [])),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReleaseTargetRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("json payload must be an object")
        return cls.from_dict(payload)

    def profile(self, mechanism_id: str) -> ReleaseMechanismTargetProfile:
        for profile in self.profiles:
            if profile.mechanism_id == mechanism_id:
                return profile
        raise KeyError(mechanism_id)
