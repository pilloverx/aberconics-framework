"""Canonical Python entrypoints for D2C scenario and chain workflows."""

from __future__ import annotations

import ctypes
from typing import Mapping

import gfe_ctypes as gfe

from . import ffi, reports
from .schemas import (
    AbersoeRunResult,
    ChainEdgeSpec,
    ChainLevelSpec,
    CrossLevelReport,
    HierarchicalRunResult,
    RenormAnalysis,
    as_mapping,
)


def _coerce_levels(levels: list[ChainLevelSpec | Mapping[str, object]]) -> list[dict[str, object]]:
    return [as_mapping(ChainLevelSpec.from_mapping(level) if isinstance(level, Mapping) else level) for level in levels]


def _coerce_edges(edges: list[ChainEdgeSpec | Mapping[str, object]]) -> list[dict[str, object]]:
    return [as_mapping(ChainEdgeSpec.from_mapping(edge) if isinstance(edge, Mapping) else edge) for edge in edges]


def _resolve_library(lib: ctypes.CDLL | None, library_path: str | None) -> ctypes.CDLL:
    if lib is not None:
        return lib
    return ffi.load_library(library_path)


def list_abersoe_scenarios(
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
) -> list[str]:
    library = _resolve_library(lib, library_path)
    return ffi.available_abersoe_scenarios(library)


def list_hierarchical_scenarios(
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
) -> list[str]:
    library = _resolve_library(lib, library_path)
    return ffi.available_hierarchical_scenarios(library)


def run_abersoe(
    scenario: int,
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
    dt: float = 0.01,
    form: int = gfe.GFE_C_COUPLING_FORM_B,
    steps: int = 100,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> AbersoeRunResult:
    library = _resolve_library(lib, library_path)
    raw = ffi.run_abersoe_raw(
        library,
        scenario,
        dt=dt,
        form=form,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    return AbersoeRunResult.from_mapping(raw)


def run_hierarchy(
    scenario: int,
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> HierarchicalRunResult:
    library = _resolve_library(lib, library_path)
    raw = ffi.run_hierarchical_raw(
        library,
        scenario,
        enable_bottom_up=enable_bottom_up,
        enable_top_down=enable_top_down,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    return HierarchicalRunResult.from_mapping(raw)


def validate_chain(
    levels: list[ChainLevelSpec | Mapping[str, object]],
    edges: list[ChainEdgeSpec | Mapping[str, object]],
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
) -> None:
    library = _resolve_library(lib, library_path)
    ffi.validate_chain_raw(library, _coerce_levels(levels), _coerce_edges(edges))


def run_hierarchy_chain(
    levels: list[ChainLevelSpec | Mapping[str, object]],
    edges: list[ChainEdgeSpec | Mapping[str, object]],
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> HierarchicalRunResult:
    library = _resolve_library(lib, library_path)
    raw = ffi.run_hierarchical_chain_raw(
        library,
        _coerce_levels(levels),
        _coerce_edges(edges),
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    return HierarchicalRunResult.from_mapping(raw)


def get_chain_cross_level_report(
    levels: list[ChainLevelSpec | Mapping[str, object]],
    edges: list[ChainEdgeSpec | Mapping[str, object]],
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> CrossLevelReport:
    library = _resolve_library(lib, library_path)
    raw = ffi.get_chain_cross_level_report_raw(
        library,
        _coerce_levels(levels),
        _coerce_edges(edges),
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    return reports.normalize_cross_level_report(raw)


def get_chain_renorm_report(
    levels: list[ChainLevelSpec | Mapping[str, object]],
    edges: list[ChainEdgeSpec | Mapping[str, object]],
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
) -> list[RenormAnalysis]:
    library = _resolve_library(lib, library_path)
    raw = ffi.get_chain_renorm_report_raw(library, _coerce_levels(levels), _coerce_edges(edges))
    return reports.normalize_renorm_analyses(raw)


def get_scenario_cross_level_report(
    scenario: int,
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> CrossLevelReport:
    library = _resolve_library(lib, library_path)
    raw = ffi.get_scenario_cross_level_report_raw(
        library,
        scenario,
        enable_bottom_up=enable_bottom_up,
        enable_top_down=enable_top_down,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    return reports.normalize_cross_level_report(raw)


def get_scenario_renorm_report(
    scenario: int,
    *,
    lib: ctypes.CDLL | None = None,
    library_path: str | None = None,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
) -> list[RenormAnalysis]:
    library = _resolve_library(lib, library_path)
    raw = ffi.get_scenario_renorm_report_raw(
        library,
        scenario,
        enable_bottom_up=enable_bottom_up,
        enable_top_down=enable_top_down,
    )
    return reports.normalize_renorm_analyses(raw)
