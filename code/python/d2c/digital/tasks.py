"""Synthetic symbolic tasks for digital D2C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Mapping

from .streams import DigitalStream, regular_token_stream
from .tokens import Vocabulary


@dataclass(slots=True)
class SymbolicInductionExample:
    """Single-pair long-gap retrieval example."""

    key_token: str
    value_token: str
    filler_tokens: list[str]
    stream: DigitalStream

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["stream"] = self.stream.to_mapping()
        return data


@dataclass(slots=True)
class SymbolicInductionConfig:
    key_count: int = 4
    value_count: int = 4
    filler_count: int = 3
    gap: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.key_count <= 0 or self.value_count <= 0 or self.filler_count <= 0:
            raise ValueError("token counts must be positive")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SymbolicInductionConfig":
        return cls(
            key_count=int(data.get("key_count", 4)),
            value_count=int(data.get("value_count", 4)),
            filler_count=int(data.get("filler_count", 3)),
            gap=int(data.get("gap", 8)),
            seed=int(data.get("seed", 0)),
        )


def build_symbolic_induction_vocabulary(config: SymbolicInductionConfig) -> Vocabulary:
    tokens = (
        [f"K{i}" for i in range(config.key_count)]
        + [f"V{i}" for i in range(config.value_count)]
        + [f"F{i}" for i in range(config.filler_count)]
    )
    return Vocabulary(tokens=tokens)


def generate_symbolic_induction_example(
    config: SymbolicInductionConfig,
    *,
    example_index: int = 0,
) -> SymbolicInductionExample:
    rng = random.Random(config.seed + example_index)
    key_token = f"K{rng.randrange(config.key_count)}"
    value_token = f"V{rng.randrange(config.value_count)}"
    filler_tokens = [f"F{rng.randrange(config.filler_count)}" for _ in range(config.gap)]
    tokens = [key_token, value_token, *filler_tokens, key_token]
    stream = regular_token_stream(
        tokens,
        dt=1.0,
        target_token=value_token,
        metadata={
            "task": "symbolic_induction_single_pair",
            "key_token": key_token,
            "value_token": value_token,
            "gap": config.gap,
        },
    )
    return SymbolicInductionExample(
        key_token=key_token,
        value_token=value_token,
        filler_tokens=filler_tokens,
        stream=stream,
    )
