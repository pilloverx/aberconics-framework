from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c import (
    AberconicsDirector,
    run_terminal_lorenz_batch,
    run_terminal_lorenz_comparison,
    LorenzLyapunovEstimate,
    TraceStore,
    build_python_lorenz63_baseline_scaffold,
    build_reference_lorenz63_scaffold,
    save_lorenz_batch_bundle,
)


def _fake_memory_runner_factory(baseline_scaffold):
    def _runner(initial_state):
        baseline = baseline_scaffold.run_default() if initial_state == baseline_scaffold.initial_conditions[0] else None
        if baseline is None:
            from d2c import run_python_lorenz63_baseline

            baseline = run_python_lorenz63_baseline(baseline_scaffold.config, initial_state=initial_state)

        memory_trace_u = [[state[0] * 0.5, state[1] * 0.5, state[2] * 0.9] for state in baseline.sample_u]
        memory_trace_chi = [[state[0] * 0.1, state[0] * 0.05, -state[0] * 0.025] for state in baseline.sample_u]
        final_state = memory_trace_u[-1]
        final_chi = memory_trace_chi[-1]
        final_u_l2 = sum(value * value for value in final_state) ** 0.5
        final_chi_l2 = sum(value * value for value in final_chi) ** 0.5
        max_abs_u = max(max(abs(value) for value in state) for state in memory_trace_u)
        max_abs_chi = max(max(abs(value) for value in state) for state in memory_trace_chi)
        mean_abs_u = sum(abs(value) for state in memory_trace_u for value in state) / (len(memory_trace_u) * 3.0)
        mean_abs_chi = sum(abs(value) for state in memory_trace_chi for value in state) / (len(memory_trace_chi) * 3.0)
        return {
            "final_state": {
                "u": final_state,
                "chi": final_chi,
                "t": baseline.sample_t[-1],
            },
            "diagnostics": {
                "steps_executed": baseline.config.steps,
                "all_finite": True,
                "max_abs_u": max_abs_u,
                "max_abs_chi": max_abs_chi,
                "final_u_l2": final_u_l2,
                "final_chi_l2": final_chi_l2,
                "mean_abs_u": mean_abs_u,
                "mean_abs_chi": mean_abs_chi,
            },
            "config": {
                "memory_channels": 3,
                "dt": baseline.config.dt,
                "coupling_index": 0,
                "steps": baseline.config.steps,
                "sample_stride": baseline.config.sample_stride,
                "strict_finite": True,
                "stochastic_forcing": False,
                "forcing_noise_std": 0.0,
                "stochastic_channel": 0,
                "seed": 0,
                "fit_backend": 0,
                "hebbian_enabled": False,
                "hebbian_rule": 0,
                "hebbian_learning_rate": 0.0,
                "hebbian_decay": 0.0,
            },
            "samples": {
                "t": list(baseline.sample_t),
                "u0": [state[0] for state in memory_trace_u],
                "chi0": [state[0] for state in memory_trace_chi],
                "u_l2": [sum(value * value for value in state) ** 0.5 for state in memory_trace_u],
                "chi_l2": [sum(value * value for value in state) ** 0.5 for state in memory_trace_chi],
            },
            "trajectory": {
                "t": list(baseline.sample_t),
                "u": memory_trace_u,
                "chi": memory_trace_chi,
            },
            "active_kernel": {
                "gamma": [0.2, 0.05, 0.01],
                "w": [0.5, 0.2, 0.1],
            },
        }

    return _runner


def _fake_memory_lyapunov(_initial_state):
    return LorenzLyapunovEstimate(
        status="computed",
        value=-0.0125,
        method="fake_runtime_test",
        note="Synthetic memory Lyapunov estimate for runtime test coverage.",
    )


def test_director_runs_lorenz_batch_and_populates_trace_store(tmp_path):
    memory_scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    baseline_scaffold = build_python_lorenz63_baseline_scaffold(memory_scaffold.runtime)
    director = AberconicsDirector(
        baseline_scaffold=baseline_scaffold,
        memory_scaffold=memory_scaffold,
        trace_store=TraceStore(experiment_name="runtime_test"),
        memory_runner=_fake_memory_runner_factory(baseline_scaffold),
        memory_lyapunov_runner=_fake_memory_lyapunov,
    )

    result = director.run_lorenz63_batch()
    bundle_paths = save_lorenz_batch_bundle(
        result.batch,
        output_dir=tmp_path,
        bundle_name="director_batch",
    )

    assert len(result.batch.per_ic) == 3
    assert len(result.trace_store.records) == 3
    assert result.trace_store.get_record(1).memory_stability["passed"] is True
    assert result.batch.per_ic[0].memory_lyapunov.value == -0.0125
    assert Path(bundle_paths["report_txt"]).exists()
    assert Path(bundle_paths["summary_json"]).exists()


def test_trace_store_builds_reusable_windows():
    memory_scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    baseline_scaffold = build_python_lorenz63_baseline_scaffold(memory_scaffold.runtime)
    director = AberconicsDirector(
        baseline_scaffold=baseline_scaffold,
        memory_scaffold=memory_scaffold,
        trace_store=TraceStore(experiment_name="runtime_windows"),
        memory_runner=_fake_memory_runner_factory(baseline_scaffold),
        memory_lyapunov_runner=_fake_memory_lyapunov,
    )

    result = director.run_lorenz63_batch()
    baseline_windows = result.trace_store.build_windows(0, source="baseline", window_size=8, stride=4)
    memory_windows = result.trace_store.build_all_windows(source="memory", window_size=8, stride=4)

    assert baseline_windows
    assert baseline_windows[0].source == "baseline"
    assert len(baseline_windows[0].sample_u) == 8
    assert baseline_windows[0].next_u is not None
    assert memory_windows
    assert memory_windows[0].sample_chi


def test_terminal_entrypoints_can_use_director_path(tmp_path):
    memory_scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    baseline_scaffold = build_python_lorenz63_baseline_scaffold(memory_scaffold.runtime)
    batch_director = AberconicsDirector(
        baseline_scaffold=baseline_scaffold,
        memory_scaffold=memory_scaffold,
        trace_store=TraceStore(experiment_name="runtime_batch_terminal"),
        memory_runner=_fake_memory_runner_factory(baseline_scaffold),
        memory_lyapunov_runner=_fake_memory_lyapunov,
    )
    comparison_director = AberconicsDirector(
        baseline_scaffold=baseline_scaffold,
        memory_scaffold=memory_scaffold,
        trace_store=TraceStore(experiment_name="runtime_comparison_terminal"),
        memory_runner=_fake_memory_runner_factory(baseline_scaffold),
        memory_lyapunov_runner=_fake_memory_lyapunov,
    )

    batch_paths = run_terminal_lorenz_batch(
        output_dir=tmp_path,
        bundle_name="director_terminal_batch",
        use_director=True,
        director=batch_director,
    )
    comparison_paths = run_terminal_lorenz_comparison(
        output_dir=tmp_path,
        bundle_name="director_terminal_comparison",
        use_director=True,
        director=comparison_director,
    )

    assert Path(batch_paths["trace_store_json"]).exists()
    assert Path(batch_paths["trace_windows_json"]).exists()
    assert Path(batch_paths["learning_analysis_json"]).exists()
    assert Path(comparison_paths["trace_store_json"]).exists()


def test_director_can_step_statefully():
    memory_scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    director = AberconicsDirector(memory_scaffold=memory_scaffold)

    # Lorenz63 default IC
    initial_u = [1.0, 1.0, 1.0]
    state0 = director.seed_state(initial_u)

    assert state0.u == initial_u
    assert state0.t == 0.0
    assert len(state0.chi) == 3
    assert state0.chi == [0.0, 0.0, 0.0]

    state1 = director.step(dt=0.01)

    assert state1.t == 0.01
    assert state1.u != initial_u
    assert any(val != 0.0 for val in state1.chi)

    state2 = director.step(dt=0.01)

    assert state2.t == 0.02
    assert state2.u != state1.u


def test_director_can_run_learning_episode():
    from d2c.runtime.director import LearningPhase
    from d2c import ThreeFactorUpdateConfig

    memory_scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    
    director = AberconicsDirector(memory_scaffold=memory_scaffold)
    # Slow down leak rate to make stability constraint tighter for testing
    director.learning_cfg.update_cfg = ThreeFactorUpdateConfig(leak_rate=0.5)
    
    # Configure for a short test episode
    director.learning_cfg.warmup_steps = 10
    director.learning_cfg.explore_steps = 50
    director.learning_cfg.window_size = 5
    director.learning_cfg.stride = 2
    director.learning_cfg.error_threshold = 1000.0 # Force quick transition to CONSOLIDATE
    
    initial_u = [1.0, 1.0, 1.0]
    history = director.run_learning_episode(initial_u, total_steps=100)
    
    assert len(history) == 101
    assert any(s.smoothed_error > 0.0 for s in history)
    
    phases = [s.phase for s in history]
    assert LearningPhase.WARMUP in phases
    assert LearningPhase.EXPLORE in phases
    assert LearningPhase.CONSOLIDATE in phases
    
    final_w = history[-1].w
    initial_w = history[0].w
    
    # Weights should have moved from initial values
    assert final_w != initial_w
    # Verify they stayed non-negative (CM preservation)
    assert all(w >= 0.0 for w in final_w)
