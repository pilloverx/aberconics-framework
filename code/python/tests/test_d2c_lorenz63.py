from pathlib import Path
import sys
from io import StringIO

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c import (
    ascii_sparkline,
    build_python_lorenz63_baseline_scaffold,
    compare_python_baseline_to_cpp_memory,
    cpp_baseline_config_from_scaffold,
    default_cpp_lorenz63_kernel_metadata,
    LorenzMemoryParameters,
    build_reference_lorenz63_scaffold,
    default_lorenz_initial_conditions,
    default_python_lorenz_initial_conditions,
    estimate_cpp_lorenz63_memory_lyapunov,
    estimate_python_lorenz63_baseline_lyapunov,
    format_lorenz_batch_report,
    lorenz_memory_rhs,
    lorenz_memory_variational_rhs,
    pack_memory_params,
    python_lorenz63_rhs,
    rho_stats,
    rho_value,
    run_multi_ic_lorenz_batch,
    run_terminal_lorenz_batch,
    run_terminal_lorenz_comparison,
    run_cpp_lorenz63_baseline,
    run_cpp_lorenz63_with_overrides,
    run_python_lorenz63_baseline,
    save_lorenz_batch_bundle,
    save_lorenz_comparison_bundle,
    spectral_diagnosis_from_kernel,
    format_lorenz_comparison_report,
    summarize_cpp_lorenz63_baseline,
    unpack_memory_params,
)
from gfe_ctypes import load_gfe_library


def test_pack_unpack_memory_params_roundtrip():
    gamma = [0.2, 0.05, 0.01]
    weights = [0.5, 0.2, 0.1]

    theta = pack_memory_params(gamma, weights)
    gamma_out, weights_out = unpack_memory_params(theta)

    assert len(theta) == 6
    assert all(abs(a - b) < 1e-12 for a, b in zip(gamma_out, gamma, strict=True))
    assert all(abs(a - b) < 1e-12 for a, b in zip(weights_out, weights, strict=True))


def test_lorenz_rhs_and_variational_shapes():
    params = LorenzMemoryParameters()
    state = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    augmented = state + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    rhs = lorenz_memory_rhs(state, params)
    variational_rhs = lorenz_memory_variational_rhs(augmented, params)

    assert len(rhs) == params.state_dim
    assert len(variational_rhs) == 2 * params.state_dim
    assert rho_value(state, params) == params.rho0


def test_reference_lorenz63_scaffold_matches_quick_mode():
    scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    record = scaffold.to_experiment_record()
    cpp_cfg = cpp_baseline_config_from_scaffold(scaffold)

    assert scaffold.quick_mode is True
    assert scaffold.evaluation_limit == 200
    assert len(scaffold.training_initial_conditions) == 3
    assert scaffold.stages[0].training_initial_condition_indices == [0]
    assert scaffold.validated_periodic_initial_conditions == 2
    assert record.metadata["recommended_mode"] == "memory_only"
    assert cpp_cfg.steps == 20000
    assert cpp_cfg.sample_stride == 10


def test_lorenz_rho_stats_over_initial_conditions():
    params = LorenzMemoryParameters()
    states = default_lorenz_initial_conditions(params.memory_channels)
    stats = rho_stats(states, params)

    assert stats["mean"] == params.rho0
    assert stats["std"] == 0.0
    assert stats["min"] == params.rho0
    assert stats["max"] == params.rho0


def test_python_lorenz63_baseline_rhs_and_runner():
    scaffold = build_python_lorenz63_baseline_scaffold()
    rhs = python_lorenz63_rhs([1.0, 1.0, 1.0], scaffold.config)
    run = run_python_lorenz63_baseline(scaffold.config, initial_state=default_python_lorenz_initial_conditions()[0])

    assert len(rhs) == 3
    assert run.diagnostics.steps_executed == 20000
    assert run.diagnostics.all_finite is True
    assert len(run.final_state) == 3
    assert len(run.sample_t) == len(run.sample_u)
    assert len(run.sample_t) > 0


def test_ascii_sparkline_and_bundle_writer(tmp_path):
    baseline_scaffold = build_python_lorenz63_baseline_scaffold()
    baseline = run_python_lorenz63_baseline(
        baseline_scaffold.config,
        initial_state=default_python_lorenz_initial_conditions()[0],
    )
    spark = ascii_sparkline([point[0] for point in baseline.sample_u], width=24)

    assert len(spark) <= 24

    # Build a minimal comparison object using a small live-like baseline plus a fabricated memory payload.
    from d2c import AbersoeConfig, AbersoeDiagnostics, AbersoeFinalState, AbersoeRunResult, AbersoeSamples, LorenzBaselineMemoryComparison

    memory = AbersoeRunResult(
        final_state=AbersoeFinalState(u=[1.0, 2.0, 3.0], chi=[0.1, 0.2, 0.3], t=200.0),
        diagnostics=AbersoeDiagnostics(
            steps_executed=20000,
            all_finite=True,
            max_abs_u=3.0,
            max_abs_chi=0.3,
            final_u_l2=3.7416573867739413,
            final_chi_l2=0.37416573867739417,
            mean_abs_u=2.0,
            mean_abs_chi=0.2,
        ),
        config=AbersoeConfig(
            memory_channels=3,
            dt=0.01,
            coupling_index=0,
            steps=20000,
            sample_stride=10,
            strict_finite=True,
            stochastic_forcing=False,
            forcing_noise_std=0.0,
            stochastic_channel=0,
            seed=0,
            fit_backend=0,
            hebbian_enabled=False,
            hebbian_rule=0,
            hebbian_learning_rate=0.0,
            hebbian_decay=0.0,
        ),
        samples=AbersoeSamples(
            t=[0.0, 0.1, 0.2],
            u0=[1.0, 1.5, 2.0],
            chi0=[0.0, 0.1, 0.2],
            u_l2=[1.0, 1.8, 2.5],
            chi_l2=[0.0, 0.1, 0.3],
        ),
    )
    comparison = LorenzBaselineMemoryComparison(baseline=baseline, memory=memory)
    report = format_lorenz_comparison_report(comparison)
    paths = save_lorenz_comparison_bundle(comparison, output_dir=tmp_path, bundle_name="bundle", report_text=report)

    assert Path(paths["report_txt"]).exists()
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["baseline_csv"]).exists()
    assert Path(paths["memory_csv"]).exists()
    assert Path(paths["xyz_comparison_png"]).exists()
    assert Path(paths["phase_xz_png"]).exists()
    assert Path(paths["memory_contributions_png"]).exists()


def test_spectral_diagnosis_and_python_lyapunov_estimate(tmp_path):
    gamma, weights = default_cpp_lorenz63_kernel_metadata()
    diagnosis = spectral_diagnosis_from_kernel(
        gamma,
        weights,
        source="test",
    )
    scaffold = build_python_lorenz63_baseline_scaffold()
    lyap = estimate_python_lorenz63_baseline_lyapunov(
        scaffold.config,
        initial_state=default_python_lorenz_initial_conditions()[0],
    )

    assert diagnosis.status == "computed"
    assert diagnosis.mcap is not None and diagnosis.mcap > 0.0
    assert diagnosis.deff is not None and diagnosis.deff > 0.0
    assert lyap.status in {"computed", "placeholder"}

    # exercise batch bundle writer on a tiny synthetic batch mapping
    from d2c import LorenzBatchComparison, LorenzLyapunovEstimate, LorenzPerICSummary

    batch = LorenzBatchComparison(
        per_ic=[
            LorenzPerICSummary(
                ic_index=0,
                initial_state=[1.0, 1.0, 1.0],
                baseline_final_state=[0.0, 0.0, 0.0],
                baseline_final_u_l2=0.0,
                baseline_all_finite=True,
                baseline_lyapunov=LorenzLyapunovEstimate(status="computed", value=0.1, method="test"),
                memory_status="placeholder",
                memory_lyapunov=LorenzLyapunovEstimate(status="placeholder", method="test"),
                spectral_diagnosis=diagnosis,
            )
        ]
    )
    report = format_lorenz_batch_report(batch)
    paths = save_lorenz_batch_bundle(batch, output_dir=tmp_path, bundle_name="batch", report_text=report)
    assert "Lorenz63 Batch Summary" in report
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["per_ic_dir"]).exists()
    assert Path(paths["baseline_final_x_by_ic_png"]).exists()
    assert Path(paths["ic_00_spectral_png"]).exists()


@pytest.mark.smoke
def test_cpp_lorenz63_baseline_run():
    try:
        lib = load_gfe_library(None)
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    run = run_cpp_lorenz63_baseline(lib=lib, scaffold=scaffold)
    summary = summarize_cpp_lorenz63_baseline(run)

    assert len(run.final_state.u) == 3
    assert len(run.final_state.chi) == 3
    assert run.final_state.t > 0.0
    assert run.diagnostics.steps_executed == 20000
    assert run.diagnostics.all_finite is True
    assert summary["config"]["memory_channels"] == 3


@pytest.mark.smoke
def test_cpp_lorenz63_override_run_and_memory_lyapunov():
    try:
        lib = load_gfe_library(None)
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    initial_state = scaffold.training_initial_conditions[1]
    raw = run_cpp_lorenz63_with_overrides(
        lib=lib,
        scaffold=scaffold,
        initial_state=initial_state,
    )
    lyap = estimate_cpp_lorenz63_memory_lyapunov(
        lib=lib,
        scaffold=scaffold,
        initial_state=initial_state,
    )

    assert raw["trajectory"]["u"][0] == pytest.approx(initial_state[:3])
    assert raw["active_kernel"]["gamma"] == pytest.approx(scaffold.memory.gamma)
    assert raw["active_kernel"]["w"] == pytest.approx(scaffold.memory.weights)
    assert lyap.status in {"computed", "placeholder"}


@pytest.mark.smoke
def test_python_baseline_to_cpp_memory_comparison():
    try:
        lib = load_gfe_library(None)
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    comparison = compare_python_baseline_to_cpp_memory(
        lib=lib,
        baseline_scaffold=build_python_lorenz63_baseline_scaffold(),
        memory_scaffold=build_reference_lorenz63_scaffold(quick_mode=True),
    )
    summary = comparison.summary()

    assert summary["baseline"]["steps_executed"] == 20000
    assert summary["memory"]["steps_executed"] == 20000
    assert summary["paired_metrics"]["same_step_count"] is True
    assert summary["paired_metrics"]["memory_run_has_explicit_memory"] is True


@pytest.mark.smoke
def test_terminal_lorenz_comparison_runner(tmp_path):
    try:
        lib = load_gfe_library(None)
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    stream = StringIO()
    paths = run_terminal_lorenz_comparison(
        lib=lib,
        baseline_scaffold=build_python_lorenz63_baseline_scaffold(),
        memory_scaffold=build_reference_lorenz63_scaffold(quick_mode=True),
        output_dir=tmp_path,
        bundle_name="terminal_run",
        stream=stream,
        tick_seconds=0.001,
    )
    output = stream.getvalue()

    assert "[run] python_baseline started" in output
    assert "[run] cpp_memory started" in output
    assert "Lorenz63 Baseline vs Memory" in output
    assert "xyz_comparison_png=" in output
    assert "peak_abs_channel_contrib=" in output
    assert Path(paths["report_txt"]).exists()
    assert Path(paths["xyz_comparison_png"]).exists()


@pytest.mark.smoke
def test_multi_ic_batch_runner(tmp_path):
    try:
        lib = load_gfe_library(None)
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    batch = run_multi_ic_lorenz_batch(
        lib=lib,
        baseline_scaffold=build_python_lorenz63_baseline_scaffold(),
        memory_scaffold=build_reference_lorenz63_scaffold(quick_mode=True),
        stream=None,
        tick_seconds=0.001,
    )
    mapping = batch.to_mapping()
    assert len(mapping["per_ic"]) == 3
    assert mapping["completed_memory_runs"] == 3
    assert all(item["memory_status"] == "completed" for item in mapping["per_ic"])
    assert all(item["spectral_diagnosis"]["source"] == "override_backed_active_kernel" for item in mapping["per_ic"])

    stream = StringIO()
    paths = run_terminal_lorenz_batch(
        lib=lib,
        baseline_scaffold=build_python_lorenz63_baseline_scaffold(),
        memory_scaffold=build_reference_lorenz63_scaffold(quick_mode=True),
        output_dir=tmp_path,
        bundle_name="batch_terminal",
        stream=stream,
        tick_seconds=0.001,
    )
    output = stream.getvalue()
    assert "Lorenz63 Batch Summary" in output
    assert "baseline_final_x_by_ic_png=" in output
    assert "channel_peak_abs_contrib=" in output
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["baseline_final_x_by_ic_png"]).exists()
    assert Path(paths["ic_00_traces_xyz_png"]).exists()
    assert Path(paths["ic_00_memory_contributions_png"]).exists()
