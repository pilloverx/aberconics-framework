"""Timestamped token/event streams for digital D2C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(slots=True)
class DigitalEvent:
    token: str
    t: float
    duration: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DigitalEvent":
        return cls(
            token=str(data["token"]),
            t=float(data["t"]),
            duration=float(data.get("duration", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class DigitalStream:
    events: list[DigitalEvent]
    target_token: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(event.duration < 0.0 for event in self.events):
            raise ValueError("event durations must be non-negative")
        times = [event.t for event in self.events]
        if times != sorted(times):
            raise ValueError("events must be sorted by time")

    def tokens(self) -> list[str]:
        return [event.token for event in self.events]

    def to_mapping(self) -> dict[str, object]:
        return {
            "events": [event.to_mapping() for event in self.events],
            "target_token": self.target_token,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DigitalStream":
        return cls(
            events=[DigitalEvent.from_mapping(event) for event in data["events"]],
            target_token=None if data.get("target_token") is None else str(data["target_token"]),
            metadata=dict(data.get("metadata", {})),
        )


def regular_token_stream(
    tokens: list[str],
    *,
    dt: float = 1.0,
    duration: float = 1.0,
    target_token: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DigitalStream:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return DigitalStream(
        events=[
            DigitalEvent(token=token, t=idx * dt, duration=duration)
            for idx, token in enumerate(tokens)
        ],
        target_token=target_token,
        metadata=dict(metadata or {}),
    )
