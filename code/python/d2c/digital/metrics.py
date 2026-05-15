"""Metrics for digital D2C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DigitalPrediction:
    target: str
    predicted: str
    score: float

    @property
    def correct(self) -> bool:
        return self.target == self.predicted

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["correct"] = self.correct
        return data


@dataclass(slots=True)
class DigitalMetricSummary:
    count: int
    correct: int
    accuracy: float
    mean_score: float

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def summarize_predictions(predictions: list[DigitalPrediction]) -> DigitalMetricSummary:
    if not predictions:
        return DigitalMetricSummary(count=0, correct=0, accuracy=0.0, mean_score=0.0)
    correct = sum(1 for prediction in predictions if prediction.correct)
    return DigitalMetricSummary(
        count=len(predictions),
        correct=correct,
        accuracy=correct / len(predictions),
        mean_score=sum(prediction.score for prediction in predictions) / len(predictions),
    )
