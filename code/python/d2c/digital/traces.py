"""Trace records for digital D2C episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(slots=True)
class DigitalTraceStep:
    step: int
    t: float
    token: str | None
    forcing: list[float]
    u: list[float]
    chi: list[list[float]]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DigitalTrace:
    experiment_name: str
    steps: list[DigitalTraceStep] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def add_step(self, step: DigitalTraceStep) -> None:
        self.steps.append(step)

    def to_mapping(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "steps": [step.to_mapping() for step in self.steps],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DigitalTrace":
        trace = cls(experiment_name=str(data["experiment_name"]), metadata=dict(data.get("metadata", {})))
        for row in data.get("steps", []):
            trace.add_step(
                DigitalTraceStep(
                    step=int(row["step"]),
                    t=float(row["t"]),
                    token=None if row.get("token") is None else str(row["token"]),
                    forcing=[float(value) for value in row["forcing"]],
                    u=[float(value) for value in row["u"]],
                    chi=[[float(value) for value in channel] for channel in row["chi"]],
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return trace
