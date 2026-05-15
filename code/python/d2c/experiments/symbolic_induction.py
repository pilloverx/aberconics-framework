"""Milestone D symbolic induction experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from ..digital import (
    DigitalMemoryConfig,
    DigitalPrediction,
    DigitalTrace,
    DigitalTraceStep,
    SymbolicInductionConfig,
    TokenForcingBridge,
    build_forcing_schedule,
    build_symbolic_induction_vocabulary,
    decode_state,
    format_symbolic_induction_report,
    generate_symbolic_induction_example,
    seed_memory_state,
    step_memory,
    summarize_predictions,
    window_limited_lookup_prediction,
)


@dataclass(slots=True)
class SymbolicInductionRunConfig:
    """Run settings for the single-pair symbolic induction probe."""

    task: SymbolicInductionConfig
    variant: str = "full"
    seed_count: int = 8
    pulse_steps: int = 2
    silence_steps: int = 1
    bridge_dim: int = 8
    bridge_seed: int = 17
    code_mode: str = "one_hot"
    baseline_window: int = 4

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["task"] = self.task.to_mapping()
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SymbolicInductionRunConfig":
        return cls(
            task=SymbolicInductionConfig.from_mapping(data.get("task", {})),
            variant=str(data.get("variant", "full")),
            seed_count=int(data.get("seed_count", 8)),
            pulse_steps=int(data.get("pulse_steps", 2)),
            silence_steps=int(data.get("silence_steps", 1)),
            bridge_dim=int(data.get("bridge_dim", 8)),
            bridge_seed=int(data.get("bridge_seed", 17)),
            code_mode=str(data.get("code_mode", "one_hot")),
            baseline_window=int(data.get("baseline_window", 4)),
        )


def memory_config_for_variant(variant: str) -> DigitalMemoryConfig:
    """Return stable, hand-specified memory kernels for early ablations."""

    if variant == "full":
        return DigitalMemoryConfig(
            gamma=[2.0, 0.45, 0.035],
            w=[0.04, 0.08, 0.12],
            dt=0.05,
            leak_rate=2.0,
            form="input_driven",
        )
    if variant == "no_slow":
        return DigitalMemoryConfig(
            gamma=[2.0, 0.9],
            w=[0.25, 0.18],
            dt=0.05,
            leak_rate=2.0,
            form="input_driven",
        )
    if variant == "collapsed_gamma":
        return DigitalMemoryConfig(
            gamma=[0.6, 0.6, 0.6],
            w=[0.18, 0.18, 0.18],
            dt=0.05,
            leak_rate=2.0,
            form="input_driven",
        )
    raise ValueError(f"unknown symbolic induction variant: {variant}")


def run_symbolic_induction_trial(
    config: SymbolicInductionRunConfig,
    *,
    example_index: int,
) -> tuple[DigitalPrediction, DigitalTrace, dict[str, object]]:
    vocab = build_symbolic_induction_vocabulary(config.task)
    bridge = TokenForcingBridge(
        vocabulary=vocab,
        dim=None if config.code_mode == "one_hot" else config.bridge_dim,
        seed=config.bridge_seed,
        code_mode=config.code_mode,
    )
    memory_cfg = memory_config_for_variant(config.variant)
    example = generate_symbolic_induction_example(config.task, example_index=example_index)
    schedule = build_forcing_schedule(
        example.stream,
        bridge,
        pulse_steps=config.pulse_steps,
        silence_steps=config.silence_steps,
    )
    state = seed_memory_state(state_dim=bridge.state_dim, channel_count=memory_cfg.channel_count)
    trace = DigitalTrace(
        experiment_name="symbolic_induction",
        metadata={
            "variant": config.variant,
            "example": example.to_mapping(),
            "memory_config": memory_cfg.to_mapping(),
        },
    )

    for step_idx, (forcing, token) in enumerate(zip(schedule.forcings, schedule.tokens)):
        state = step_memory(state, forcing, memory_cfg)
        trace.add_step(
            DigitalTraceStep(
                step=step_idx,
                t=state.t,
                token=token,
                forcing=list(forcing),
                u=list(state.u),
                chi=[list(channel) for channel in state.chi],
            )
        )

    value_candidates = [token for token in vocab.tokens if token.startswith("V")]
    predicted, score = decode_state(
        state,
        memory_cfg,
        vocab,
        candidates=value_candidates,
        seed=config.bridge_seed,
        code_mode=config.code_mode,
    )
    prediction = DigitalPrediction(
        target=example.value_token,
        predicted=predicted,
        score=score,
    )
    diagnostics = {
        "deff": memory_cfg.deff(),
        "stability_ratio": memory_cfg.stability_ratio(),
        "final_t": state.t,
        "step_count": len(schedule.forcings),
    }
    return prediction, trace, diagnostics


def run_symbolic_induction_experiment(config: SymbolicInductionRunConfig) -> dict[str, object]:
    predictions: list[DigitalPrediction] = []
    traces: list[DigitalTrace] = []
    diagnostics: list[dict[str, object]] = []
    baseline_correct = 0

    for idx in range(config.seed_count):
        prediction, trace, diag = run_symbolic_induction_trial(config, example_index=idx)
        predictions.append(prediction)
        traces.append(trace)
        diagnostics.append(diag)
        example = generate_symbolic_induction_example(config.task, example_index=idx)
        baseline = window_limited_lookup_prediction(example.stream, window=config.baseline_window)
        baseline_correct += int(baseline == example.value_token)

    summary = summarize_predictions(predictions)
    memory_cfg = memory_config_for_variant(config.variant)
    return {
        "experiment_name": "symbolic_induction",
        "variant": config.variant,
        "gap": config.task.gap,
        "seed_count": config.seed_count,
        "config": config.to_mapping(),
        "summary": summary.to_mapping(),
        "predictions": [prediction.to_mapping() for prediction in predictions],
        "memory_config": memory_cfg.to_mapping(),
        "memory_diagnostics": {
            "deff": memory_cfg.deff(),
            "stability_ratio": memory_cfg.stability_ratio(),
            "trial_diagnostics": diagnostics,
        },
        "baseline": {
            "name": "window_limited_lookup",
            "window": config.baseline_window,
            "count": config.seed_count,
            "correct": baseline_correct,
            "accuracy": baseline_correct / config.seed_count if config.seed_count else 0.0,
        },
        "trace_count": len(traces),
        "traces": [trace.to_mapping() for trace in traces],
    }


def save_symbolic_induction_bundle(
    result: Mapping[str, object],
    *,
    output_dir: str | Path,
    bundle_name: str | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    run_id = bundle_name or f"symbolic_induction_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = root / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary_path = bundle_dir / "summary.json"
    report_path = bundle_dir / "report.txt"
    trace_path = bundle_dir / "traces.json"

    summary = dict(result)
    traces = summary.pop("traces", [])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    report_path.write_text(format_symbolic_induction_report(result), encoding="utf-8")

    return {
        "bundle_dir": str(bundle_dir),
        "summary_json": str(summary_path),
        "traces_json": str(trace_path),
        "report_txt": str(report_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Milestone D symbolic induction probe.")
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("code/python/d2c/progress/symbolic_induction"))
    parser.add_argument("--bundle-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)
    config = SymbolicInductionRunConfig(
        task=SymbolicInductionConfig(gap=args.gap),
        variant=args.variant,
        seed_count=args.seeds,
    )
    result = run_symbolic_induction_experiment(config)
    paths = save_symbolic_induction_bundle(result, output_dir=args.output_dir, bundle_name=args.bundle_name)
    print(format_symbolic_induction_report(result))
    return paths


if __name__ == "__main__":
    main()
