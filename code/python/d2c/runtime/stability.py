"""Lightweight stability checks for early D2C runtime loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from ..experiments.lorenz63 import PythonLorenzBaselineRun
from ..schemas import AbersoeRunResult


@dataclass(slots=True)
class StabilityAssessment:
    """Compact runtime-side stability summary."""

    label: str
    passed: bool
    all_finite: bool
    final_u_l2: float
    max_abs_u: float
    final_chi_l2: float | None = None
    max_abs_chi: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def assess_python_baseline_run(run: PythonLorenzBaselineRun) -> StabilityAssessment:
    return StabilityAssessment(
        label="python_baseline",
        passed=bool(run.diagnostics.all_finite and math.isfinite(run.diagnostics.final_u_l2)),
        all_finite=bool(run.diagnostics.all_finite),
        final_u_l2=float(run.diagnostics.final_u_l2),
        max_abs_u=float(run.diagnostics.max_abs_u),
        notes=["Pure Python Lorenz63 control condition."],
    )


def assess_abersoe_run(run: AbersoeRunResult) -> StabilityAssessment:
    return StabilityAssessment(
        label="cpp_memory",
        passed=bool(
            run.diagnostics.all_finite
            and math.isfinite(run.diagnostics.final_u_l2)
            and math.isfinite(run.diagnostics.final_chi_l2)
        ),
        all_finite=bool(run.diagnostics.all_finite),
        final_u_l2=float(run.diagnostics.final_u_l2),
        max_abs_u=float(run.diagnostics.max_abs_u),
        final_chi_l2=float(run.diagnostics.final_chi_l2),
        max_abs_chi=float(run.diagnostics.max_abs_chi),
        notes=["Override-backed C++ Lorenz63 memory run."],
    )

