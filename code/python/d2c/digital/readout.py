"""Readout helpers for digital D2C states."""

from __future__ import annotations

from typing import Sequence

from .bridge import DigitalMemoryConfig, DigitalMemoryState
from .tokens import Vocabulary, nearest_token


def memory_readout_vector(state: DigitalMemoryState, config: DigitalMemoryConfig) -> list[float]:
    """Weighted memory-channel readout vector."""

    if not state.chi:
        return list(state.u)
    return [
        sum(weight * channel[i] for weight, channel in zip(config.w, state.chi))
        for i in range(len(state.u))
    ]


def combined_readout_vector(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    *,
    state_weight: float = 0.25,
    memory_weight: float = 1.0,
) -> list[float]:
    memory = memory_readout_vector(state, config)
    return [
        state_weight * u_value + memory_weight * memory_value
        for u_value, memory_value in zip(state.u, memory)
    ]


def decode_state(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    vocabulary: Vocabulary,
    *,
    candidates: Sequence[str] | None = None,
    seed: int = 0,
    code_mode: str = "dense",
) -> tuple[str, float]:
    vector = combined_readout_vector(state, config)
    return nearest_token(vector, vocabulary, candidates=candidates, dim=len(vector), seed=seed, code_mode=code_mode)
