"""Learning scaffolds for Python-side D2C runtime work."""

from .analysis import LearningWindowAnalysis, TraceLearningAnalysis, analyze_trace_store_windows
from .consolidation import ConsolidationConfig, ConsolidationResult, ConsolidationState, consolidate_weights
from .critics import (
    TemporalDifferenceError,
    ValueEstimate,
    channel_discounts,
    per_channel_td_errors,
    state_norm_reward,
)
from .predictive import (
    LastValuePredictor,
    LinearDeltaPredictor,
    PredictionErrorSummary,
    PredictionResult,
    evaluate_next_state_prediction,
    prediction_error_summary,
)
from .reports import format_learning_analysis_report
from .updates import KernelUpdateProposal, ThreeFactorUpdateConfig, propose_three_factor_update

__all__ = [
    "ConsolidationConfig",
    "ConsolidationResult",
    "ConsolidationState",
    "KernelUpdateProposal",
    "LastValuePredictor",
    "LinearDeltaPredictor",
    "LearningWindowAnalysis",
    "PredictionErrorSummary",
    "PredictionResult",
    "TemporalDifferenceError",
    "ThreeFactorUpdateConfig",
    "TraceLearningAnalysis",
    "ValueEstimate",
    "analyze_trace_store_windows",
    "channel_discounts",
    "consolidate_weights",
    "evaluate_next_state_prediction",
    "format_learning_analysis_report",
    "per_channel_td_errors",
    "prediction_error_summary",
    "propose_three_factor_update",
    "state_norm_reward",
]
