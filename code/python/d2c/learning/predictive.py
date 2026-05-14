"""Prediction-error helpers over contiguous D2C trace windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Protocol, Sequence

from ..runtime import LorenzTraceWindow


def _coerce_vector(values: Sequence[float], *, name: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _subtract(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    if len(lhs) != len(rhs):
        raise ValueError("vectors must have matching dimensions")
    return [float(a) - float(b) for a, b in zip(lhs, rhs)]


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


class NextStatePredictor(Protocol):
    """Small interface for one-step prediction over a trace window."""

    name: str

    def predict_next_u(self, window: LorenzTraceWindow) -> list[float]:
        """Predict the next latent state after ``window.sample_u``."""


@dataclass(slots=True)
class PredictionResult:
    """One-step prediction and optional error against a known target."""

    predictor_name: str
    predicted_u: list[float]
    target_u: list[float] | None = None
    error: list[float] = field(default_factory=list)
    error_l2: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PredictionErrorSummary:
    """Compact prediction-error summary usable by update rules."""

    error: list[float]
    error_l2: float
    mean_abs_error: float
    signed_mean_error: float

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LastValuePredictor:
    """Baseline predictor: next state equals the final state in the window."""

    name: str = "last_value"

    def predict_next_u(self, window: LorenzTraceWindow) -> list[float]:
        if not window.sample_u:
            raise ValueError("window.sample_u must not be empty")
        return _coerce_vector(window.sample_u[-1], name="last sample_u")


@dataclass(slots=True)
class LinearDeltaPredictor:
    """Extrapolate the last observed delta one step forward."""

    name: str = "linear_delta"

    def predict_next_u(self, window: LorenzTraceWindow) -> list[float]:
        if len(window.sample_u) < 2:
            raise ValueError("linear delta prediction requires at least two samples")
        previous = _coerce_vector(window.sample_u[-2], name="previous sample_u")
        current = _coerce_vector(window.sample_u[-1], name="current sample_u")
        delta = _subtract(current, previous)
        return [value + step for value, step in zip(current, delta)]


def prediction_error_summary(predicted_u: Sequence[float], target_u: Sequence[float]) -> PredictionErrorSummary:
    """Summarize one prediction error vector."""

    predicted = _coerce_vector(predicted_u, name="predicted_u")
    target = _coerce_vector(target_u, name="target_u")
    error = _subtract(target, predicted)
    mean_abs = sum(abs(value) for value in error) / len(error)
    signed_mean = sum(error) / len(error)
    return PredictionErrorSummary(
        error=error,
        error_l2=_l2(error),
        mean_abs_error=mean_abs,
        signed_mean_error=signed_mean,
    )


def evaluate_next_state_prediction(
    window: LorenzTraceWindow,
    predictor: NextStatePredictor | None = None,
) -> PredictionResult:
    """Run a one-step predictor against ``window.next_u`` when available."""

    active_predictor = predictor or LastValuePredictor()
    predicted = active_predictor.predict_next_u(window)
    if window.next_u is None:
        return PredictionResult(
            predictor_name=active_predictor.name,
            predicted_u=predicted,
            metadata={"has_target": False},
        )

    summary = prediction_error_summary(predicted, window.next_u)
    return PredictionResult(
        predictor_name=active_predictor.name,
        predicted_u=predicted,
        target_u=list(window.next_u),
        error=summary.error,
        error_l2=summary.error_l2,
        metadata={
            "has_target": True,
            "mean_abs_error": summary.mean_abs_error,
            "signed_mean_error": summary.signed_mean_error,
        },
    )
