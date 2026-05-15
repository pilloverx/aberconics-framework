"""Simple digital-task baselines."""

from __future__ import annotations

from .streams import DigitalStream


def window_limited_lookup_prediction(
    stream: DigitalStream,
    *,
    window: int,
    value_prefix: str = "V",
) -> str | None:
    """Return most recent value token in the final window, if any."""

    if window <= 0:
        raise ValueError("window must be positive")
    tokens = stream.tokens()
    for token in reversed(tokens[-window:]):
        if token.startswith(value_prefix):
            return token
    return None
