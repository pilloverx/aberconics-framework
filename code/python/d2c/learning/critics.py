"""Small critic and TD helpers for Phase 4 learning scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Sequence


@dataclass(slots=True)
class ValueEstimate:
    """Per-channel value estimates for one augmented state."""

    values: list[float]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TemporalDifferenceError:
    """One TD error per SOE memory channel."""

    gamma: list[float]
    discounts: list[float]
    errors: list[float]
    reward: float
    dt: float

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def _coerce_float_list(values: Sequence[float], *, name: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def channel_discounts(gamma: Sequence[float], dt: float) -> list[float]:
    """Convert SOE decay rates into per-channel discrete discounts."""

    rates = _coerce_float_list(gamma, name="gamma")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if any(rate <= 0.0 for rate in rates):
        raise ValueError("gamma values must be positive")
    return [math.exp(-rate * dt) for rate in rates]


def per_channel_td_errors(
    *,
    reward: float,
    current_values: Sequence[float],
    next_values: Sequence[float],
    gamma: Sequence[float],
    dt: float,
) -> TemporalDifferenceError:
    """Compute ``r + exp(-gamma_l dt) V_next - V_current`` for each channel."""

    current = _coerce_float_list(current_values, name="current_values")
    future = _coerce_float_list(next_values, name="next_values")
    rates = _coerce_float_list(gamma, name="gamma")
    if len(current) != len(future) or len(current) != len(rates):
        raise ValueError("current_values, next_values, and gamma must have matching lengths")

    discounts = channel_discounts(rates, dt)
    errors = [float(reward) + discount * next_value - current_value for discount, next_value, current_value in zip(discounts, future, current)]
    return TemporalDifferenceError(
        gamma=rates,
        discounts=discounts,
        errors=errors,
        reward=float(reward),
        dt=float(dt),
    )


def state_norm_reward(state_u: Sequence[float], *, target_norm: float = 0.0, scale: float = 1.0) -> float:
    """Simple boundedness reward: less state energy is better."""

    state = _coerce_float_list(state_u, name="state_u")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    norm = math.sqrt(sum(value * value for value in state))
    return -(norm - float(target_norm)) / float(scale)
