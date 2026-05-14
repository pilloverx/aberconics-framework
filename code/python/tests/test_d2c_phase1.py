from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c import (
    ChainEdgeSpec,
    ChainLevelSpec,
    get_chain_cross_level_report,
    get_chain_renorm_report,
    run_abersoe,
    run_hierarchy_chain,
    validate_chain,
)
from gfe_ctypes import (
    GFE_C_ABERSOE_SCENARIO_LINEAR,
    GFE_C_COUPLING_FORM_B,
    GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
    GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
    load_gfe_library,
)


def _lib_path() -> str | None:
    return None


def _example_levels() -> list[ChainLevelSpec]:
    return [
        ChainLevelSpec(
            name="custom_fast",
            gamma=[1.0, 0.4],
            w=[0.65, 0.35],
            u=[1.0],
            chi=[0.0, 0.0],
            dt=0.01,
            linear_decay=[0.25],
            forcing_bias=[0.0],
            form=GFE_C_COUPLING_FORM_B,
            coupling_index=0,
        ),
        ChainLevelSpec(
            name="custom_slow",
            gamma=[0.7, 0.15],
            w=[0.6, 0.4],
            u=[0.1],
            chi=[0.0, 0.0],
            dt=0.01,
            linear_decay=[0.08],
            forcing_bias=[0.01],
            form=GFE_C_COUPLING_FORM_B,
            coupling_index=0,
        ),
    ]


def _example_edges() -> list[ChainEdgeSpec]:
    return [
        ChainEdgeSpec(
            source_level=0,
            target_level=1,
            relation=GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
            gain=0.35,
            normalize_weights=True,
        ),
        ChainEdgeSpec(
            source_level=1,
            target_level=0,
            relation=GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
            gain=0.18,
            normalize_weights=True,
        ),
    ]


@pytest.mark.smoke
def test_d2c_abersoe_typed_run():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    out = run_abersoe(
        GFE_C_ABERSOE_SCENARIO_LINEAR,
        lib=lib,
        dt=0.01,
        steps=20,
        sample_stride=5,
        strict_finite=True,
    )

    assert out.diagnostics.steps_executed == 20
    assert out.diagnostics.all_finite is True
    assert len(out.final_state.u) == 1
    assert len(out.samples.t) > 0


@pytest.mark.smoke
def test_d2c_chain_typed_reports():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    levels = _example_levels()
    edges = _example_edges()

    validate_chain(levels, edges, lib=lib)
    run = run_hierarchy_chain(levels, edges, lib=lib, steps=18, sample_stride=6, strict_finite=True)
    cross = get_chain_cross_level_report(levels, edges, lib=lib, steps=18, sample_stride=6, strict_finite=True)
    renorm = get_chain_renorm_report(levels, edges, lib=lib)

    assert run.diagnostics.level_count == 2
    assert run.levels[0].deff > 0.0
    assert cross.contract.all_finite_pass is True
    assert len(cross.relations) == 2
    assert len(renorm) == 1
    assert 0.0 <= renorm[0].retained_weight_fraction <= 1.0
