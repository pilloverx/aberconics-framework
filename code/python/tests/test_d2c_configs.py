from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c import (
    ExperimentRecord,
    HierarchyEdgeConfig,
    HierarchyExperimentConfig,
    HierarchyLevelConfig,
    KernelPreset,
    RunConfig,
    TrainingPhaseConfig,
)
from gfe_ctypes import GFE_C_COUPLING_FORM_B, GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP


def test_hierarchy_experiment_config_roundtrip():
    kernel = KernelPreset(
        name="fast_memory",
        gamma=[1.0, 0.4],
        w=[0.65, 0.35],
        status="promising",
        notes=["good fit on short OU traces"],
        tags=["ou", "baseline"],
    )
    level = HierarchyLevelConfig(
        name="fast_level",
        kernel=kernel,
        u=[1.0],
        chi=[0.0, 0.0],
        dt=0.01,
        linear_decay=[0.25],
        forcing_bias=[0.0],
        form=GFE_C_COUPLING_FORM_B,
    )
    edge = HierarchyEdgeConfig(
        source_level=0,
        target_level=0,
        relation=GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
        gain=0.25,
    )
    config = HierarchyExperimentConfig(
        name="ou_probe_a",
        levels=[level],
        edges=[edge],
        run=RunConfig(steps=32, sample_stride=8, strict_finite=True),
        status="unsuccessful",
        hypothesis="A single fast level may hold enough short-horizon structure.",
        notes=["Diverged once top-down edge was introduced in an earlier variant."],
        outcome_summary="Stable numerically, but memory horizon was too short.",
        next_actions=["Add a slower second level."],
        metadata={"owner": "python-d2c"},
    )

    restored = HierarchyExperimentConfig.from_mapping(config.to_mapping())
    chain_levels = restored.to_chain_level_specs()
    chain_edges = restored.to_chain_edge_specs()

    assert restored.status == "unsuccessful"
    assert restored.next_actions == ["Add a slower second level."]
    assert restored.levels[0].kernel.status == "promising"
    assert chain_levels[0].gamma == [1.0, 0.4]
    assert chain_edges[0].gain == 0.25


def test_experiment_record_retains_unsuccessful_notes():
    record = ExperimentRecord(
        name="multi_sine_attempt_01",
        status="unsuccessful",
        hypothesis="A short warmup phase may be enough for stable adaptation.",
        run=RunConfig(steps=120, sample_stride=12, strict_finite=True),
        phases=[
            TrainingPhaseConfig(
                name="warmup",
                status="unsuccessful",
                max_epochs=3,
                notes=["Prediction loss plateaued early."],
                failure_modes=["Kernel specialization collapsed to one channel."],
                next_actions=["Increase channel count before re-running."],
            )
        ],
        notes=["Keep for reference so we do not retry the same schedule blindly."],
        failure_modes=["Underfit after epoch 2."],
        next_actions=["Try a two-phase schedule with longer warmup."],
    )

    restored = ExperimentRecord.from_mapping(record.to_mapping())

    assert restored.status == "unsuccessful"
    assert restored.phases[0].failure_modes == ["Kernel specialization collapsed to one channel."]
    assert restored.next_actions == ["Try a two-phase schedule with longer warmup."]
