"""Simple scheduling helpers for early Director-managed experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass(slots=True)
class LorenzBatchScheduleItem:
    """One scheduled Lorenz63 run item."""

    ic_index: int
    initial_state: list[float]
    label: str


@dataclass(slots=True)
class SequentialScheduler:
    """Minimal sequential scheduler for initial runtime orchestration."""

    name: str = "sequential_ic_scheduler"
    notes: list[str] = field(default_factory=lambda: ["Runs each initial condition in order."])

    def build_lorenz_batch(self, initial_conditions: Sequence[Sequence[float]]) -> list[LorenzBatchScheduleItem]:
        return [
            LorenzBatchScheduleItem(
                ic_index=index,
                initial_state=[float(value) for value in state],
                label=f"ic{index}",
            )
            for index, state in enumerate(initial_conditions)
        ]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

