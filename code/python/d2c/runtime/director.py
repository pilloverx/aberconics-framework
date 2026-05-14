"""First usable Director shell for Python-side D2C runtime work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TextIO

from ..experiments.lorenz63 import (
    LorenzBatchComparison,
    LorenzCppBaselineConfig,
    LorenzExperimentScaffold,
    LorenzPerICSummary,
    PythonLorenzBaselineScaffold,
    build_python_lorenz63_baseline_scaffold,
    build_reference_lorenz63_scaffold,
    cpp_baseline_config_from_scaffold,
    estimate_cpp_lorenz63_memory_lyapunov,
    estimate_python_lorenz63_baseline_lyapunov,
    run_cpp_lorenz63_with_overrides,
    run_python_lorenz63_baseline,
    spectral_diagnosis_from_kernel,
)
from ..ffi import load_library, step_abersoe_raw
from ..schemas import AbersoeRunResult
from .scheduler import SequentialScheduler
from .stability import assess_abersoe_run, assess_python_baseline_run
from .trace_store import LorenzTraceRecord, LorenzTraceWindow, TraceStore


def _emit_progress(stream: TextIO | None, message: str) -> None:
    if stream is None:
        return
    stream.write(message + "\n")
    stream.flush()


class LearningPhase(Enum):
    """Phases of the AberconicsDirector online learning state machine."""

    WARMUP = auto()
    EXPLORE = auto()
    CONSOLIDATE = auto()
    EVALUATE = auto()


@dataclass(slots=True)
class LearningDirectorConfig:
    """Hyperparameters for the online learning state machine."""

    warmup_steps: int = 200
    explore_steps: int = 800
    consolidate_steps: int = 100
    window_size: int = 8
    stride: int = 4
    update_cfg: Any = None  # ThreeFactorUpdateConfig
    consolidation_cfg: Any = None  # ConsolidationConfig
    reward_scale: float = 10.0
    target_norm: float = 28.0
    error_threshold: float = 1.0
    leak_rate: float = 10.0


@dataclass(slots=True)
class DirectorState:
    """Current state of a running experiment managed by the Director."""

    u: list[float]
    chi: list[float]
    t: float
    gamma: list[float]
    w: list[float]
    ic_index: int
    phase: LearningPhase = LearningPhase.WARMUP
    step_count: int = 0
    smoothed_error: float = 0.0
    consolidation: Any = None  # ConsolidationState


@dataclass(slots=True)
class DirectorBatchResult:
    """Combined Director-managed runtime result."""

    batch: LorenzBatchComparison
    trace_store: TraceStore
    scheduler: SequentialScheduler
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return {
            "batch": self.batch.to_mapping(),
            "trace_store": self.trace_store.to_mapping(),
            "scheduler": self.scheduler.to_mapping(),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class DirectorComparisonResult:
    """Single-IC view over Director-managed Lorenz63 execution."""

    summary: LorenzPerICSummary
    trace_record: LorenzTraceRecord
    scheduler: SequentialScheduler
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return {
            "summary": self.summary.to_mapping(),
            "trace_record": self.trace_record.to_mapping(),
            "scheduler": self.scheduler.to_mapping(),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class AberconicsDirector:
    """Early Director that runs the current Lorenz63 workflow through runtime components."""

    lib: Any = None
    library_path: str | None = None
    baseline_scaffold: PythonLorenzBaselineScaffold | None = None
    memory_scaffold: LorenzExperimentScaffold | None = None
    cpp_config: LorenzCppBaselineConfig | None = None
    scheduler: SequentialScheduler = field(default_factory=SequentialScheduler)
    trace_store: TraceStore | None = None
    stream: TextIO | None = None
    memory_runner: Callable[[list[float]], dict[str, object]] | None = None
    memory_lyapunov_runner: Callable[[list[float]], Any] | None = None
    current_state: DirectorState | None = None
    learning_cfg: LearningDirectorConfig = field(default_factory=LearningDirectorConfig)

    def __post_init__(self) -> None:
        self.memory_scaffold = self.memory_scaffold or build_reference_lorenz63_scaffold(quick_mode=True)
        self.baseline_scaffold = self.baseline_scaffold or build_python_lorenz63_baseline_scaffold(
            self.memory_scaffold.runtime
        )
        self.cpp_config = self.cpp_config or cpp_baseline_config_from_scaffold(self.memory_scaffold)
        self.trace_store = self.trace_store or TraceStore(
            experiment_name=self.memory_scaffold.name,
            notes=["First runtime-managed Lorenz63 trace store."],
        )

    def load_runtime_library(self) -> Any:
        if self.lib is None:
            self.lib = load_library(self.library_path)
        return self.lib

    def step(self, *, scenario_id: int = 1, dt: float | None = None) -> DirectorState:
        """Perform a single stateful step through the runtime library."""
        if self.current_state is None:
            raise RuntimeError("director current_state is None; call seed_state() first")

        # For now, default to Lorenz63 (scenario_id=1) as per the current Director scope
        step_dt = dt or self.cpp_config.dt if self.cpp_config else 0.01

        res = step_abersoe_raw(
            lib=self.load_runtime_library(),
            scenario=scenario_id,
            dt=step_dt,
            form=self.cpp_config.form if self.cpp_config else 1,  # Default to Formulation B
            u=self.current_state.u,
            chi=self.current_state.chi,
            t=self.current_state.t,
            gamma=self.current_state.gamma,
            w=self.current_state.w,
        )

        self.current_state = DirectorState(
            u=list(res["u"]),
            chi=list(res["chi"]),
            t=float(res["t"]),
            gamma=self.current_state.gamma,
            w=list(self.current_state.w),
            ic_index=self.current_state.ic_index,
            phase=self.current_state.phase,
            step_count=self.current_state.step_count + 1,
            smoothed_error=self.current_state.smoothed_error,
            consolidation=self.current_state.consolidation,
        )
        return self.current_state

    def seed_state(self, initial_u: list[float], *, ic_index: int = 0) -> DirectorState:
        """Initialize the director's current_state for a new trajectory."""
        from ..learning import ConsolidationState

        gamma = list(self.memory_scaffold.memory.gamma) if self.memory_scaffold else []
        weights = list(self.memory_scaffold.memory.weights) if self.memory_scaffold else []
        chi = [0.0] * len(gamma)

        self.current_state = DirectorState(
            u=list(initial_u),
            chi=chi,
            t=0.0,
            gamma=gamma,
            w=list(weights),
            ic_index=ic_index,
            phase=LearningPhase.WARMUP,
            step_count=0,
            smoothed_error=0.0,
            consolidation=ConsolidationState(fast_weights=list(weights), slow_weights=[0.0] * len(weights)),
        )
        return self.current_state

    def run_learning_episode(self, initial_u: list[float], total_steps: int = 1000) -> list[DirectorState]:
        """Run a full online learning episode, transitioning through state machine phases."""
        from ..learning import (
            ConsolidationConfig,
            ConsolidationState,
            LinearDeltaPredictor,
            ThreeFactorUpdateConfig,
            consolidate_weights,
            evaluate_next_state_prediction,
            per_channel_td_errors,
            propose_three_factor_update,
            state_norm_reward,
        )

        self.seed_state(initial_u)
        # Deep copy initial state for history
        initial_history_state = DirectorState(
            u=list(self.current_state.u),
            chi=list(self.current_state.chi),
            t=self.current_state.t,
            gamma=self.current_state.gamma,
            w=list(self.current_state.w),
            ic_index=self.current_state.ic_index,
            phase=self.current_state.phase,
            step_count=self.current_state.step_count,
            smoothed_error=self.current_state.smoothed_error,
            consolidation=ConsolidationState(
                fast_weights=list(self.current_state.consolidation.fast_weights),
                slow_weights=list(self.current_state.consolidation.slow_weights),
            ) if self.current_state.consolidation else None,
        )
        history: list[DirectorState] = [initial_history_state]

        predictor = LinearDeltaPredictor()
        window_size = self.learning_cfg.window_size
        stride = self.learning_cfg.stride
        update_cfg = self.learning_cfg.update_cfg or ThreeFactorUpdateConfig(
            leak_rate=self.learning_cfg.leak_rate
        )
        consolidation_cfg = self.learning_cfg.consolidation_cfg or ConsolidationConfig()

        # Buffer for online window extraction
        u_history: list[list[float]] = [self.current_state.u]  # type: ignore
        chi_history: list[list[float]] = [self.current_state.chi]  # type: ignore
        t_history: list[float] = [self.current_state.t]  # type: ignore

        for _ in range(total_steps):
            state = self.step()
            u_history.append(state.u)
            chi_history.append(state.chi)
            t_history.append(state.t)

            # Update learning state machine
            if state.phase == LearningPhase.WARMUP and state.step_count >= self.learning_cfg.warmup_steps:
                state.phase = LearningPhase.EXPLORE
                _emit_progress(self.stream, f"[phase] WARMUP -> EXPLORE at t={state.t:.3f}")

            if state.phase == LearningPhase.EXPLORE:
                # If we have enough history for a window update
                if state.step_count % stride == 0 and len(u_history) >= window_size + 1:
                    # Build window [s, s+W) with next_u target
                    s = len(u_history) - window_size - 1
                    win = LorenzTraceWindow(
                        ic_index=state.ic_index,
                        source="memory",
                        start_idx=s,
                        end_idx=s + window_size - 1,
                        sample_t=t_history[s : s + window_size],
                        sample_u=u_history[s : s + window_size],
                        sample_chi=chi_history[s : s + window_size],
                        next_u=u_history[s + window_size],
                    )

                    # Online Update
                    prediction = evaluate_next_state_prediction(win, predictor)
                    
                    # Smooth the prediction error
                    if prediction.error_l2 is not None:
                        state.smoothed_error = 0.9 * state.smoothed_error + 0.1 * prediction.error_l2

                    reward = state_norm_reward(
                        win.next_u,  # type: ignore
                        target_norm=self.learning_cfg.target_norm,
                        scale=self.learning_cfg.reward_scale
                    )
                    
                    # Compute TD error (no value head yet, so assume V=0)
                    td = per_channel_td_errors(
                        reward=reward,
                        current_values=[0.0] * len(state.gamma),
                        next_values=[0.0] * len(state.gamma),
                        gamma=state.gamma,
                        dt=win.sample_t[-1] - win.sample_t[-2] if len(win.sample_t) >= 2 else 0.01,
                    )

                    # Compute signed Hebbian signal (dot product of chi and prediction error)
                    # For Lorenz63, coupling is on u[0] (x)
                    err_x = prediction.error[0]
                    hebbian_signal = [
                        (sum(row[i] for row in win.sample_chi) / len(win.sample_chi)) * err_x
                        for i in range(len(state.gamma))
                    ]

                    # Update consolidation fast weights
                    if state.consolidation:
                        proposal = propose_three_factor_update(
                            weights=state.consolidation.fast_weights,
                            gamma=state.gamma,
                            channel_activity=[0.0] * len(state.gamma),  # Unused when hebbian_signal provided
                            prediction_error=prediction.error,
                            td_error=td.errors,
                            hebbian_signal=hebbian_signal,
                            config=update_cfg,
                        )
                        state.consolidation.fast_weights = proposal.clipped_weights
                        state.w = state.consolidation.total_weights

                    if state.smoothed_error < self.learning_cfg.error_threshold:
                        # Success condition: low smoothed prediction error
                        if state.step_count >= self.learning_cfg.explore_steps:
                            state.phase = LearningPhase.CONSOLIDATE
                            _emit_progress(self.stream, f"[phase] EXPLORE -> CONSOLIDATE at t={state.t:.3f}")

            elif state.phase == LearningPhase.CONSOLIDATE:
                if state.step_count % 10 == 0 and state.consolidation:
                    res = consolidate_weights(state.consolidation, consolidation_cfg)
                    state.consolidation = res.state
                    state.w = state.consolidation.total_weights

            # Snapshot state for history to avoid shared mutation issues
            history_state = DirectorState(
                u=list(state.u),
                chi=list(state.chi),
                t=state.t,
                gamma=state.gamma,
                w=list(state.w),
                ic_index=state.ic_index,
                phase=state.phase,
                step_count=state.step_count,
                smoothed_error=state.smoothed_error,
                consolidation=ConsolidationState(
                    fast_weights=list(state.consolidation.fast_weights),
                    slow_weights=list(state.consolidation.slow_weights),
                ) if state.consolidation else None,
            )
            history.append(history_state)

        return history

    def _default_memory_runner(self, initial_state: list[float]) -> dict[str, object]:
        return run_cpp_lorenz63_with_overrides(
            lib=self.load_runtime_library(),
            library_path=self.library_path,
            config=self.cpp_config,
            scaffold=self.memory_scaffold,
            initial_state=initial_state,
        )

    def _default_memory_lyapunov_runner(self, initial_state: list[float]) -> Any:
        return estimate_cpp_lorenz63_memory_lyapunov(
            lib=self.load_runtime_library(),
            library_path=self.library_path,
            config=self.cpp_config,
            scaffold=self.memory_scaffold,
            initial_state=initial_state,
        )

    def run_lorenz63_batch(self) -> DirectorBatchResult:
        assert self.trace_store is not None
        assert self.baseline_scaffold is not None
        assert self.memory_scaffold is not None

        schedule = self.scheduler.build_lorenz_batch(self.baseline_scaffold.initial_conditions)
        per_ic: list[LorenzPerICSummary] = []
        memory_runner = self.memory_runner or self._default_memory_runner
        memory_lyapunov_runner = self.memory_lyapunov_runner or self._default_memory_lyapunov_runner

        _emit_progress(self.stream, f"[director] schedule_size={len(schedule)} experiment={self.memory_scaffold.name}")

        for item in schedule:
            _emit_progress(self.stream, f"[director] running {item.label}")
            baseline_run = run_python_lorenz63_baseline(
                self.baseline_scaffold.config,
                initial_state=item.initial_state,
                progress_label=f"python_baseline_{item.label}",
                progress_stream=self.stream,
            )
            baseline_lyap = estimate_python_lorenz63_baseline_lyapunov(
                self.baseline_scaffold.config,
                initial_state=item.initial_state,
            )

            memory_raw = memory_runner(item.initial_state)
            memory_run = AbersoeRunResult.from_mapping(memory_raw)
            memory_lyap = memory_lyapunov_runner(item.initial_state)
            spectral = spectral_diagnosis_from_kernel(
                memory_raw["active_kernel"]["gamma"],
                memory_raw["active_kernel"]["w"],
                source="director_active_kernel",
                note="Computed from the active kernel returned during the Director-managed Lorenz63 run.",
            )
            baseline_stability = assess_python_baseline_run(baseline_run)
            memory_stability = assess_abersoe_run(memory_run)

            self.trace_store.add_record(
                LorenzTraceRecord(
                    ic_index=item.ic_index,
                    initial_state=list(item.initial_state),
                    baseline_trace_t=[float(value) for value in baseline_run.sample_t],
                    baseline_trace_u=[[float(value) for value in state] for state in baseline_run.sample_u],
                    memory_trace_t=[float(value) for value in memory_raw["trajectory"]["t"]],
                    memory_trace_u=[[float(value) for value in state] for state in memory_raw["trajectory"]["u"]],
                    memory_trace_chi=[[float(value) for value in state] for state in memory_raw["trajectory"]["chi"]],
                    memory_kernel_gamma=[float(value) for value in memory_raw["active_kernel"]["gamma"]],
                    memory_kernel_weights=[float(value) for value in memory_raw["active_kernel"]["w"]],
                    baseline_stability=baseline_stability.to_mapping(),
                    memory_stability=memory_stability.to_mapping(),
                    metadata={
                        "schedule_label": item.label,
                        "baseline_lyapunov_method": baseline_lyap.method,
                        "memory_lyapunov_method": memory_lyap.method,
                    },
                )
            )

            per_ic.append(
                LorenzPerICSummary(
                    ic_index=item.ic_index,
                    initial_state=list(item.initial_state),
                    baseline_final_state=list(baseline_run.final_state),
                    baseline_final_u_l2=baseline_run.diagnostics.final_u_l2,
                    baseline_all_finite=baseline_run.diagnostics.all_finite,
                    baseline_lyapunov=baseline_lyap,
                    memory_status="completed",
                    memory_final_state=[float(value) for value in memory_raw["final_state"]["u"]],
                    memory_final_u_l2=float(memory_raw["diagnostics"]["final_u_l2"]),
                    memory_all_finite=bool(memory_raw["diagnostics"]["all_finite"]),
                    memory_lyapunov=memory_lyap,
                    spectral_diagnosis=spectral,
                    note=(
                        "Director-managed runtime path used the override-capable C++ Lorenz63 ABI "
                        "with trace storage and stability snapshots."
                    ),
                    baseline_trace_t=[float(value) for value in baseline_run.sample_t],
                    baseline_trace_u=[[float(value) for value in state] for state in baseline_run.sample_u],
                    memory_trace_t=[float(value) for value in memory_raw["trajectory"]["t"]],
                    memory_trace_u=[[float(value) for value in state] for state in memory_raw["trajectory"]["u"]],
                    memory_trace_chi=[[float(value) for value in state] for state in memory_raw["trajectory"]["chi"]],
                    memory_kernel_gamma=[float(value) for value in memory_raw["active_kernel"]["gamma"]],
                    memory_kernel_weights=[float(value) for value in memory_raw["active_kernel"]["w"]],
                )
            )

        batch = LorenzBatchComparison(
            per_ic=per_ic,
            notes=[
                "Batch executed through the first Python-side AberconicsDirector shell.",
                "TraceStore retained baseline/memory traces plus runtime-side stability snapshots.",
                "The underlying Lorenz63 science workflow still uses the current override-backed ABI path.",
            ],
        )
        return DirectorBatchResult(
            batch=batch,
            trace_store=self.trace_store,
            scheduler=self.scheduler,
            notes=["First runtime-managed Lorenz63 batch completed."],
        )

    def run_lorenz63_comparison(self, *, ic_index: int = 0) -> DirectorComparisonResult:
        batch_result = self.run_lorenz63_batch()
        for summary in batch_result.batch.per_ic:
            if summary.ic_index == ic_index:
                trace_record = batch_result.trace_store.get_record(ic_index)
                return DirectorComparisonResult(
                    summary=summary,
                    trace_record=trace_record,
                    scheduler=batch_result.scheduler,
                    notes=["Single-comparison view derived from the Director-managed batch run."],
                )
        raise KeyError(f"no Lorenz63 comparison summary found for ic_index={ic_index}")
