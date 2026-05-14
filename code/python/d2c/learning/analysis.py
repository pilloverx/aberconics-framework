"""TraceStore learning-analysis helpers for Phase 4 offline passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..runtime import TraceStore
from .critics import TemporalDifferenceError, per_channel_td_errors, state_norm_reward
from .predictive import LinearDeltaPredictor, NextStatePredictor, PredictionResult, evaluate_next_state_prediction
from .updates import KernelUpdateProposal, ThreeFactorUpdateConfig, propose_three_factor_update


TraceSource = Literal["baseline", "memory"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(slots=True)
class LearningWindowAnalysis:
    """Learning signals and proposed kernel update for one contiguous trace window."""

    ic_index: int
    source: TraceSource
    start_idx: int
    end_idx: int
    prediction: PredictionResult
    reward: float
    td_error: TemporalDifferenceError | None
    update_proposal: KernelUpdateProposal | None
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return {
            "ic_index": self.ic_index,
            "source": self.source,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "prediction": self.prediction.to_mapping(),
            "reward": self.reward,
            "td_error": None if self.td_error is None else self.td_error.to_mapping(),
            "update_proposal": None if self.update_proposal is None else self.update_proposal.to_mapping(),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class TraceLearningAnalysis:
    """Aggregate offline learning analysis over TraceStore windows."""

    experiment_name: str
    source: TraceSource
    window_size: int
    stride: int
    windows: list[LearningWindowAnalysis]
    notes: list[str] = field(default_factory=list)

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def mean_prediction_error_l2(self) -> float:
        return _mean(
            [
                float(window.prediction.error_l2)
                for window in self.windows
                if window.prediction.error_l2 is not None
            ]
        )

    @property
    def stability_rescale_count(self) -> int:
        return sum(1 for window in self.windows if window.update_proposal and window.update_proposal.stability_rescaled)

    @property
    def mean_stability_ratio_after(self) -> float:
        return _mean(
            [
                window.update_proposal.stability_ratio_after
                for window in self.windows
                if window.update_proposal is not None
            ]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "source": self.source,
            "window_size": self.window_size,
            "stride": self.stride,
            "window_count": self.window_count,
            "mean_prediction_error_l2": self.mean_prediction_error_l2,
            "stability_rescale_count": self.stability_rescale_count,
            "mean_stability_ratio_after": self.mean_stability_ratio_after,
            "windows": [window.to_mapping() for window in self.windows],
            "notes": list(self.notes),
        }


def analyze_trace_store_windows(
    trace_store: TraceStore,
    *,
    source: TraceSource = "memory",
    window_size: int = 8,
    stride: int = 4,
    predictor: NextStatePredictor | None = None,
    update_config: ThreeFactorUpdateConfig | None = None,
    reward_scale: float = 10.0,
) -> TraceLearningAnalysis:
    """Build prediction, TD, and stable update-proposal signals from TraceStore windows."""

    active_predictor = predictor or LinearDeltaPredictor()
    analyses: list[LearningWindowAnalysis] = []
    notes = [
        "Offline Phase 4 analysis over contiguous TraceStore windows.",
        "Kernel updates are proposals only; no runtime state is mutated.",
    ]

    for record in trace_store.records:
        gamma = list(record.memory_kernel_gamma)
        weights = list(record.memory_kernel_weights)
        windows = trace_store.build_windows(
            record.ic_index,
            source=source,
            window_size=window_size,
            stride=stride,
            include_next_u=True,
        )
        for window in windows:
            prediction = evaluate_next_state_prediction(window, active_predictor)
            if window.next_u is None:
                analyses.append(
                    LearningWindowAnalysis(
                        ic_index=window.ic_index,
                        source=source,
                        start_idx=window.start_idx,
                        end_idx=window.end_idx,
                        prediction=prediction,
                        reward=0.0,
                        td_error=None,
                        update_proposal=None,
                        notes=["window has no next_u target; skipped update proposal"],
                    )
                )
                continue

            # Default to target_norm=28.0 (Lorenz center) if not otherwise specified
            reward = state_norm_reward(window.next_u, target_norm=28.0, scale=reward_scale)
            if source == "memory" and gamma and weights:
                current_values = [0.0 for _ in gamma]
                next_values = [0.0 for _ in gamma]
                td = per_channel_td_errors(
                    reward=reward,
                    current_values=current_values,
                    next_values=next_values,
                    gamma=gamma,
                    dt=window.sample_t[-1] - window.sample_t[-2] if len(window.sample_t) >= 2 else 1.0,
                )

                # Compute signed Hebbian signal (dot product of chi and prediction error)
                # For Lorenz63, coupling is on u[0] (x)
                err_x = prediction.error[0]
                hebbian_signal = [
                    (sum(row[i] for row in window.sample_chi) / len(window.sample_chi)) * err_x
                    for i in range(len(gamma))
                ]

                proposal = propose_three_factor_update(
                    weights=weights,
                    gamma=gamma,
                    channel_activity=[0.0] * len(gamma),
                    prediction_error=prediction.error,
                    td_error=td.errors,
                    hebbian_signal=hebbian_signal,
                    config=update_config or ThreeFactorUpdateConfig(leak_rate=20.0),
                )
                analyses.append(
                    LearningWindowAnalysis(
                        ic_index=window.ic_index,
                        source=source,
                        start_idx=window.start_idx,
                        end_idx=window.end_idx,
                        prediction=prediction,
                        reward=reward,
                        td_error=td,
                        update_proposal=proposal,
                    )
                )
            else:
                analyses.append(
                    LearningWindowAnalysis(
                        ic_index=window.ic_index,
                        source=source,
                        start_idx=window.start_idx,
                        end_idx=window.end_idx,
                        prediction=prediction,
                        reward=reward,
                        td_error=None,
                        update_proposal=None,
                        notes=["prediction-only analysis for non-memory source"],
                    )
                )

    return TraceLearningAnalysis(
        experiment_name=trace_store.experiment_name,
        source=source,
        window_size=window_size,
        stride=stride,
        windows=analyses,
        notes=notes,
    )
