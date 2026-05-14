"""Trace storage helpers for early Python-side D2C runtime loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping


@dataclass(slots=True)
class LorenzTraceRecord:
    """Stored baseline/memory trace bundle for one initial condition."""

    ic_index: int
    initial_state: list[float]
    baseline_trace_t: list[float]
    baseline_trace_u: list[list[float]]
    memory_trace_t: list[float]
    memory_trace_u: list[list[float]]
    memory_trace_chi: list[list[float]]
    memory_kernel_gamma: list[float]
    memory_kernel_weights: list[float]
    baseline_stability: dict[str, object]
    memory_stability: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LorenzTraceRecord":
        return cls(
            ic_index=int(data["ic_index"]),
            initial_state=[float(value) for value in data["initial_state"]],
            baseline_trace_t=[float(value) for value in data["baseline_trace_t"]],
            baseline_trace_u=[[float(value) for value in row] for row in data["baseline_trace_u"]],
            memory_trace_t=[float(value) for value in data["memory_trace_t"]],
            memory_trace_u=[[float(value) for value in row] for row in data["memory_trace_u"]],
            memory_trace_chi=[[float(value) for value in row] for row in data["memory_trace_chi"]],
            memory_kernel_gamma=[float(value) for value in data["memory_kernel_gamma"]],
            memory_kernel_weights=[float(value) for value in data["memory_kernel_weights"]],
            baseline_stability=dict(data["baseline_stability"]),
            memory_stability=dict(data["memory_stability"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class LorenzTraceWindow:
    """Fixed-size trace window for later learning/runtime phases."""

    ic_index: int
    source: Literal["baseline", "memory"]
    start_idx: int
    end_idx: int
    sample_t: list[float]
    sample_u: list[list[float]]
    sample_chi: list[list[float]]
    next_u: list[float] | None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TraceStore:
    """In-memory store for Director-managed traces and reusable windows."""

    experiment_name: str
    records: list[LorenzTraceRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    window_notes: list[str] = field(default_factory=lambda: ["Window extraction is computed lazily from stored traces."])

    def add_record(self, record: LorenzTraceRecord) -> None:
        self.records.append(record)

    def get_record(self, ic_index: int) -> LorenzTraceRecord:
        for record in self.records:
            if record.ic_index == ic_index:
                return record
        raise KeyError(f"no trace record stored for ic_index={ic_index}")

    def build_windows(
        self,
        ic_index: int,
        *,
        source: Literal["baseline", "memory"],
        window_size: int,
        stride: int = 1,
        include_next_u: bool = True,
    ) -> list[LorenzTraceWindow]:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        record = self.get_record(ic_index)
        if source == "baseline":
            trace_t = record.baseline_trace_t
            trace_u = record.baseline_trace_u
            trace_chi: list[list[float]] = []
        else:
            trace_t = record.memory_trace_t
            trace_u = record.memory_trace_u
            trace_chi = record.memory_trace_chi

        if len(trace_t) != len(trace_u):
            raise ValueError(f"{source} trace length mismatch for ic_index={ic_index}")

        windows: list[LorenzTraceWindow] = []
        limit = len(trace_u) - window_size + 1
        for start_idx in range(0, max(0, limit), stride):
            end_idx = start_idx + window_size
            next_u = trace_u[end_idx] if include_next_u and end_idx < len(trace_u) else None
            sample_chi = trace_chi[start_idx:end_idx] if trace_chi else []
            windows.append(
                LorenzTraceWindow(
                    ic_index=ic_index,
                    source=source,
                    start_idx=start_idx,
                    end_idx=end_idx - 1,
                    sample_t=list(trace_t[start_idx:end_idx]),
                    sample_u=[list(row) for row in trace_u[start_idx:end_idx]],
                    sample_chi=[list(row) for row in sample_chi],
                    next_u=None if next_u is None else list(next_u),
                    metadata={
                        "window_size": window_size,
                        "stride": stride,
                        "has_memory_channels": bool(sample_chi),
                    },
                )
            )
        return windows

    def build_all_windows(
        self,
        *,
        source: Literal["baseline", "memory"],
        window_size: int,
        stride: int = 1,
        include_next_u: bool = True,
    ) -> list[LorenzTraceWindow]:
        windows: list[LorenzTraceWindow] = []
        for record in self.records:
            windows.extend(
                self.build_windows(
                    record.ic_index,
                    source=source,
                    window_size=window_size,
                    stride=stride,
                    include_next_u=include_next_u,
                )
            )
        return windows

    def to_mapping(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "records": [record.to_mapping() for record in self.records],
            "notes": list(self.notes),
            "window_notes": list(self.window_notes),
        }
