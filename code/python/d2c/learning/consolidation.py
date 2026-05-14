"""Fast/slow memory-weight consolidation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


def _coerce_float_list(values: Sequence[float], *, name: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


@dataclass(slots=True)
class ConsolidationConfig:
    """How quickly fast weights are absorbed into slow memory."""

    rate: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ConsolidationState:
    """Fast/slow decomposition for kernel weights."""

    fast_weights: list[float]
    slow_weights: list[float]

    def __post_init__(self) -> None:
        self.fast_weights = _coerce_float_list(self.fast_weights, name="fast_weights")
        self.slow_weights = _coerce_float_list(self.slow_weights, name="slow_weights")
        if len(self.fast_weights) != len(self.slow_weights):
            raise ValueError("fast_weights and slow_weights must have matching lengths")

    @property
    def total_weights(self) -> list[float]:
        return [fast + slow for fast, slow in zip(self.fast_weights, self.slow_weights)]

    def to_mapping(self) -> dict[str, object]:
        return {
            "fast_weights": list(self.fast_weights),
            "slow_weights": list(self.slow_weights),
            "total_weights": self.total_weights,
        }


@dataclass(slots=True)
class ConsolidationResult:
    """Updated consolidation state and movement diagnostics."""

    state: ConsolidationState
    slow_delta: list[float]
    fast_delta: list[float]
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return {
            "state": self.state.to_mapping(),
            "slow_delta": list(self.slow_delta),
            "fast_delta": list(self.fast_delta),
            "notes": list(self.notes),
        }


def consolidate_weights(
    state: ConsolidationState,
    config: ConsolidationConfig | None = None,
) -> ConsolidationResult:
    """Move slow weights toward fast weights and relax fast weights back."""

    cfg = config or ConsolidationConfig()
    new_slow: list[float] = []
    new_fast: list[float] = []
    slow_delta: list[float] = []
    fast_delta: list[float] = []

    for fast, slow in zip(state.fast_weights, state.slow_weights):
        slow_step = cfg.rate * (fast - slow)
        updated_slow = slow + slow_step
        updated_fast = (1.0 - cfg.rate) * fast + cfg.rate * updated_slow
        new_slow.append(updated_slow)
        new_fast.append(updated_fast)
        slow_delta.append(updated_slow - slow)
        fast_delta.append(updated_fast - fast)

    return ConsolidationResult(
        state=ConsolidationState(fast_weights=new_fast, slow_weights=new_slow),
        slow_delta=slow_delta,
        fast_delta=fast_delta,
        notes=["consolidated fast weights toward slow memory baseline"],
    )
