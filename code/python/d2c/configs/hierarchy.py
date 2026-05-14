"""Hierarchy experiment config objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from ..schemas import ChainEdgeSpec, ChainLevelSpec
from .kernels import KernelPreset
from .training import RunConfig


@dataclass(slots=True)
class HierarchyLevelConfig:
    """Reusable level definition for a constrained chain experiment."""

    name: str
    kernel: KernelPreset
    u: list[float]
    chi: list[float]
    dt: float
    linear_decay: list[float]
    forcing_bias: list[float]
    form: int
    coupling_index: int = 0
    notes: list[str] = field(default_factory=list)

    def to_chain_level_spec(self) -> ChainLevelSpec:
        kernel = self.kernel.to_kernel_spec()
        return ChainLevelSpec(
            name=self.name,
            gamma=list(kernel.gamma),
            w=list(kernel.w),
            u=list(self.u),
            chi=list(self.chi),
            dt=self.dt,
            linear_decay=list(self.linear_decay),
            forcing_bias=list(self.forcing_bias),
            form=self.form,
            coupling_index=self.coupling_index,
        )

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["kernel"] = self.kernel.to_mapping()
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HierarchyLevelConfig":
        return cls(
            name=str(data["name"]),
            kernel=KernelPreset.from_mapping(data["kernel"]),
            u=[float(value) for value in data["u"]],
            chi=[float(value) for value in data["chi"]],
            dt=float(data["dt"]),
            linear_decay=[float(value) for value in data["linear_decay"]],
            forcing_bias=[float(value) for value in data["forcing_bias"]],
            form=int(data["form"]),
            coupling_index=int(data.get("coupling_index", 0)),
            notes=[str(value) for value in data.get("notes", [])],
        )


@dataclass(slots=True)
class HierarchyEdgeConfig:
    """Reusable edge definition for a constrained chain experiment."""

    source_level: int
    target_level: int
    relation: int
    gain: float
    normalize_weights: bool = True
    notes: list[str] = field(default_factory=list)

    def to_chain_edge_spec(self) -> ChainEdgeSpec:
        return ChainEdgeSpec(
            source_level=self.source_level,
            target_level=self.target_level,
            relation=self.relation,
            gain=self.gain,
            normalize_weights=self.normalize_weights,
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HierarchyEdgeConfig":
        return cls(
            source_level=int(data["source_level"]),
            target_level=int(data["target_level"]),
            relation=int(data["relation"]),
            gain=float(data["gain"]),
            normalize_weights=bool(data.get("normalize_weights", True)),
            notes=[str(value) for value in data.get("notes", [])],
        )


@dataclass(slots=True)
class HierarchyExperimentConfig:
    """Named hierarchy experiment definition with room for iteration history."""

    name: str
    levels: list[HierarchyLevelConfig]
    edges: list[HierarchyEdgeConfig]
    run: RunConfig = field(default_factory=RunConfig)
    status: str = "draft"
    hypothesis: str = ""
    notes: list[str] = field(default_factory=list)
    outcome_summary: str = ""
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_chain_level_specs(self) -> list[ChainLevelSpec]:
        return [level.to_chain_level_spec() for level in self.levels]

    def to_chain_edge_specs(self) -> list[ChainEdgeSpec]:
        return [edge.to_chain_edge_spec() for edge in self.edges]

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "levels": [level.to_mapping() for level in self.levels],
            "edges": [edge.to_mapping() for edge in self.edges],
            "run": self.run.to_mapping(),
            "status": self.status,
            "hypothesis": self.hypothesis,
            "notes": list(self.notes),
            "outcome_summary": self.outcome_summary,
            "next_actions": list(self.next_actions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HierarchyExperimentConfig":
        return cls(
            name=str(data["name"]),
            levels=[HierarchyLevelConfig.from_mapping(level) for level in data["levels"]],
            edges=[HierarchyEdgeConfig.from_mapping(edge) for edge in data["edges"]],
            run=RunConfig.from_mapping(data.get("run", {})),
            status=str(data.get("status", "draft")),
            hypothesis=str(data.get("hypothesis", "")),
            notes=[str(value) for value in data.get("notes", [])],
            outcome_summary=str(data.get("outcome_summary", "")),
            next_actions=[str(value) for value in data.get("next_actions", [])],
            metadata=dict(data.get("metadata", {})),
        )
