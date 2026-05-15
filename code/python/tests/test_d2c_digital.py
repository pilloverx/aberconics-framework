from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c.digital import (
    DigitalMemoryConfig,
    DigitalTrace,
    SymbolicInductionConfig,
    TokenForcingBridge,
    Vocabulary,
    build_forcing_schedule,
    build_symbolic_induction_vocabulary,
    generate_symbolic_induction_example,
    regular_token_stream,
    seed_memory_state,
    step_memory,
)
from d2c.experiments.symbolic_induction import (
    SymbolicInductionRunConfig,
    run_symbolic_induction_experiment,
    save_symbolic_induction_bundle,
)


def test_vocabulary_stream_and_forcing_schedule_are_deterministic():
    vocab = Vocabulary(["A", "B", "F"])
    bridge = TokenForcingBridge(vocabulary=vocab, dim=4, seed=3)
    stream = regular_token_stream(["A", "B"], dt=0.5, target_token="B")
    schedule = build_forcing_schedule(stream, bridge, pulse_steps=2, silence_steps=1)

    assert vocab.token_id("B") == 1
    assert stream.tokens() == ["A", "B"]
    assert len(schedule.forcings) == 6
    assert schedule.tokens == ["A", "A", None, "B", "B", None]
    assert bridge.forcing_for_token("A") == bridge.forcing_for_token("A")


def test_digital_memory_step_and_trace_roundtrip():
    cfg = DigitalMemoryConfig(gamma=[1.0, 0.1], w=[0.2, 0.4], dt=0.05, leak_rate=2.0)
    state = seed_memory_state(state_dim=3, channel_count=2)
    next_state = step_memory(state, [1.0, 0.0, -1.0], cfg)

    assert next_state.t == 0.05
    assert next_state.u != state.u
    assert cfg.deff() > 1.0
    assert cfg.stability_ratio() < 3.0

    trace = DigitalTrace(experiment_name="roundtrip")
    data = trace.to_mapping()
    restored = DigitalTrace.from_mapping(data)
    assert restored.experiment_name == "roundtrip"


def test_symbolic_induction_task_generation():
    cfg = SymbolicInductionConfig(key_count=2, value_count=2, filler_count=1, gap=5, seed=11)
    vocab = build_symbolic_induction_vocabulary(cfg)
    example = generate_symbolic_induction_example(cfg)

    assert len(vocab.tokens) == 5
    assert example.stream.tokens()[0] == example.key_token
    assert example.stream.tokens()[1] == example.value_token
    assert example.stream.tokens()[-1] == example.key_token
    assert example.stream.target_token == example.value_token


def test_symbolic_induction_experiment_and_bundle(tmp_path):
    cfg = SymbolicInductionRunConfig(
        task=SymbolicInductionConfig(key_count=3, value_count=3, filler_count=2, gap=8, seed=5),
        variant="full",
        seed_count=4,
        baseline_window=3,
    )
    result = run_symbolic_induction_experiment(cfg)
    paths = save_symbolic_induction_bundle(result, output_dir=tmp_path, bundle_name="quick")

    assert result["summary"]["count"] == 4
    assert result["memory_diagnostics"]["deff"] > 1.0
    assert result["baseline"]["accuracy"] == 0.0
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["report_txt"]).exists()
