"""Training and experiment lifecycle config objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping

ExperimentStatus = Literal[
    "draft",
    "active",
    "baseline",
    "promising",
    "unsuccessful",
    "on_hold",
    "archived",
]


@dataclass(slots=True)
class RunConfig:
    """Shared run settings used across experiments and phases."""

    steps: int = 80
    sample_stride: int = 10
    strict_finite: bool = True

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RunConfig":
        return cls(
            steps=int(data.get("steps", 80)),
            sample_stride=int(data.get("sample_stride", 10)),
            strict_finite=bool(data.get("strict_finite", True)),
        )


@dataclass(slots=True)
class TrainingPhaseConfig:
    """Mutable training-phase definition for future Python-side learning loops."""

    name: str
    status: ExperimentStatus = "draft"
    max_epochs: int = 1
    eval_stride: int = 1
    early_stop_on_instability: bool = True
    notes: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "TrainingPhaseConfig":
        return cls(
            name=str(data["name"]),
            status=str(data.get("status", "draft")),
            max_epochs=int(data.get("max_epochs", 1)),
            eval_stride=int(data.get("eval_stride", 1)),
            early_stop_on_instability=bool(data.get("early_stop_on_instability", True)),
            notes=[str(value) for value in data.get("notes", [])],
            failure_modes=[str(value) for value in data.get("failure_modes", [])],
            next_actions=[str(value) for value in data.get("next_actions", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ExperimentRecord:
    """Top-level reusable experiment definition and outcome notebook."""

    name: str
    status: ExperimentStatus = "draft"
    hypothesis: str = ""
    run: RunConfig = field(default_factory=RunConfig)
    phases: list[TrainingPhaseConfig] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "hypothesis": self.hypothesis,
            "run": self.run.to_mapping(),
            "phases": [phase.to_mapping() for phase in self.phases],
            "notes": list(self.notes),
            "failure_modes": list(self.failure_modes),
            "next_actions": list(self.next_actions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ExperimentRecord":
        return cls(
            name=str(data["name"]),
            status=str(data.get("status", "draft")),
            hypothesis=str(data.get("hypothesis", "")),
            run=RunConfig.from_mapping(data.get("run", {})),
            phases=[TrainingPhaseConfig.from_mapping(phase) for phase in data.get("phases", [])],
            notes=[str(value) for value in data.get("notes", [])],
            failure_modes=[str(value) for value in data.get("failure_modes", [])],
            next_actions=[str(value) for value in data.get("next_actions", [])],
            metadata=dict(data.get("metadata", {})),
        )
