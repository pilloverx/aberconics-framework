"""Kernel-oriented reusable config objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from ..schemas import MemoryKernelSpec


@dataclass(slots=True)
class KernelPreset:
    """Named kernel preset that can be reused across experiments."""

    name: str
    gamma: list[float]
    w: list[float]
    status: str = "draft"
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_kernel_spec(self) -> MemoryKernelSpec:
        return MemoryKernelSpec(gamma=list(self.gamma), w=list(self.w))

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "KernelPreset":
        return cls(
            name=str(data["name"]),
            gamma=[float(value) for value in data["gamma"]],
            w=[float(value) for value in data["w"]],
            status=str(data.get("status", "draft")),
            notes=[str(value) for value in data.get("notes", [])],
            tags=[str(value) for value in data.get("tags", [])],
            metadata=dict(data.get("metadata", {})),
        )
