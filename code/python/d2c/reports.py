"""Normalization helpers for D2C report surfaces."""

from __future__ import annotations

from .schemas import CrossLevelReport, RenormAnalysis


def normalize_cross_level_report(report: dict[str, object]) -> CrossLevelReport:
    """Convert a raw wrapper cross-level report into a typed schema."""

    return CrossLevelReport.from_mapping(report)


def normalize_renorm_analyses(analyses: list[dict[str, object]]) -> list[RenormAnalysis]:
    """Convert raw renorm analysis rows into typed schemas."""

    return [RenormAnalysis.from_mapping(row) for row in analyses]


def actionable_warnings(report: CrossLevelReport) -> list:
    """Return only warnings marked actionable by the ABI layer."""

    return [warning for warning in report.warnings if warning.actionable]


def changed_kernel_levels(report: CrossLevelReport) -> list[int]:
    """Return level indices whose kernels shifted during the run."""

    return [row.level_idx for row in report.kernel_shifts if row.changed]


def passing_renorm_analyses(analyses: list[RenormAnalysis]) -> list[RenormAnalysis]:
    """Return analyses that passed all consistency checks."""

    return [
        analysis
        for analysis in analyses
        if analysis.consistency_pass and analysis.mcap_pass and analysis.mscale_pass and analysis.deff_pass
    ]
