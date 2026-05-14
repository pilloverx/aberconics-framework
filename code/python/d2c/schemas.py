"""Typed schemas for the Python D2C layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class MemoryKernelSpec:
    gamma: list[float]
    w: list[float]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ChainLevelSpec:
    name: str
    gamma: list[float]
    w: list[float]
    u: list[float]
    chi: list[float]
    dt: float
    linear_decay: list[float]
    forcing_bias: list[float]
    form: int
    coupling_index: int = 0

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ChainLevelSpec":
        return cls(
            name=str(data["name"]),
            gamma=[float(value) for value in data["gamma"]],
            w=[float(value) for value in data["w"]],
            u=[float(value) for value in data["u"]],
            chi=[float(value) for value in data["chi"]],
            dt=float(data["dt"]),
            linear_decay=[float(value) for value in data["linear_decay"]],
            forcing_bias=[float(value) for value in data["forcing_bias"]],
            form=int(data["form"]),
            coupling_index=int(data.get("coupling_index", 0)),
        )


@dataclass(slots=True)
class ChainEdgeSpec:
    source_level: int
    target_level: int
    relation: int
    gain: float
    normalize_weights: bool = True

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ChainEdgeSpec":
        return cls(
            source_level=int(data["source_level"]),
            target_level=int(data["target_level"]),
            relation=int(data["relation"]),
            gain=float(data["gain"]),
            normalize_weights=bool(data.get("normalize_weights", True)),
        )


@dataclass(slots=True)
class AbersoeFinalState:
    u: list[float]
    chi: list[float]
    t: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AbersoeFinalState":
        return cls(
            u=[float(value) for value in data["u"]],
            chi=[float(value) for value in data["chi"]],
            t=float(data["t"]),
        )


@dataclass(slots=True)
class AbersoeDiagnostics:
    steps_executed: int
    all_finite: bool
    max_abs_u: float
    max_abs_chi: float
    final_u_l2: float
    final_chi_l2: float
    mean_abs_u: float
    mean_abs_chi: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AbersoeDiagnostics":
        return cls(
            steps_executed=int(data["steps_executed"]),
            all_finite=bool(data["all_finite"]),
            max_abs_u=float(data["max_abs_u"]),
            max_abs_chi=float(data["max_abs_chi"]),
            final_u_l2=float(data["final_u_l2"]),
            final_chi_l2=float(data["final_chi_l2"]),
            mean_abs_u=float(data["mean_abs_u"]),
            mean_abs_chi=float(data["mean_abs_chi"]),
        )


@dataclass(slots=True)
class AbersoeConfig:
    memory_channels: int
    dt: float
    coupling_index: int
    steps: int
    sample_stride: int
    strict_finite: bool
    stochastic_forcing: bool
    forcing_noise_std: float
    stochastic_channel: int
    seed: int
    fit_backend: int
    hebbian_enabled: bool
    hebbian_rule: int
    hebbian_learning_rate: float
    hebbian_decay: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AbersoeConfig":
        return cls(
            memory_channels=int(data["memory_channels"]),
            dt=float(data["dt"]),
            coupling_index=int(data["coupling_index"]),
            steps=int(data["steps"]),
            sample_stride=int(data["sample_stride"]),
            strict_finite=bool(data["strict_finite"]),
            stochastic_forcing=bool(data["stochastic_forcing"]),
            forcing_noise_std=float(data["forcing_noise_std"]),
            stochastic_channel=int(data["stochastic_channel"]),
            seed=int(data["seed"]),
            fit_backend=int(data["fit_backend"]),
            hebbian_enabled=bool(data["hebbian_enabled"]),
            hebbian_rule=int(data["hebbian_rule"]),
            hebbian_learning_rate=float(data["hebbian_learning_rate"]),
            hebbian_decay=float(data["hebbian_decay"]),
        )


@dataclass(slots=True)
class AbersoeSamples:
    t: list[float]
    u0: list[float]
    chi0: list[float]
    u_l2: list[float]
    chi_l2: list[float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AbersoeSamples":
        return cls(
            t=[float(value) for value in data["t"]],
            u0=[float(value) for value in data["u0"]],
            chi0=[float(value) for value in data["chi0"]],
            u_l2=[float(value) for value in data["u_l2"]],
            chi_l2=[float(value) for value in data["chi_l2"]],
        )


@dataclass(slots=True)
class AbersoeRunResult:
    final_state: AbersoeFinalState
    diagnostics: AbersoeDiagnostics
    config: AbersoeConfig
    samples: AbersoeSamples

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AbersoeRunResult":
        return cls(
            final_state=AbersoeFinalState.from_mapping(data["final_state"]),
            diagnostics=AbersoeDiagnostics.from_mapping(data["diagnostics"]),
            config=AbersoeConfig.from_mapping(data["config"]),
            samples=AbersoeSamples.from_mapping(data["samples"]),
        )


@dataclass(slots=True)
class HierarchicalDiagnostics:
    steps_executed: int
    all_finite: bool
    level_count: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HierarchicalDiagnostics":
        return cls(
            steps_executed=int(data["steps_executed"]),
            all_finite=bool(data["all_finite"]),
            level_count=int(data["level_count"]),
        )


@dataclass(slots=True)
class HierarchicalLevelSummary:
    t: float
    state_dim: int
    memory_channels: int
    u0: float
    chi0: float
    w0: float
    gamma0: float
    mcap: float
    mscale: float
    hmem: float
    deff: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HierarchicalLevelSummary":
        return cls(
            t=float(data["t"]),
            state_dim=int(data["state_dim"]),
            memory_channels=int(data["memory_channels"]),
            u0=float(data["u0"]),
            chi0=float(data["chi0"]),
            w0=float(data["w0"]),
            gamma0=float(data["gamma0"]),
            mcap=float(data["mcap"]),
            mscale=float(data["mscale"]),
            hmem=float(data["hmem"]),
            deff=float(data["deff"]),
        )


@dataclass(slots=True)
class HierarchicalRunResult:
    diagnostics: HierarchicalDiagnostics
    levels: list[HierarchicalLevelSummary]

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HierarchicalRunResult":
        return cls(
            diagnostics=HierarchicalDiagnostics.from_mapping(data["diagnostics"]),
            levels=[HierarchicalLevelSummary.from_mapping(level) for level in data["levels"]],
        )


@dataclass(slots=True)
class CrossLevelContract:
    passed: bool
    all_finite_pass: bool
    bottom_up_mcap_pass: bool
    bottom_up_mscale_pass: bool
    actionable_warning_count: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CrossLevelContract":
        return cls(
            passed=bool(data["pass"]),
            all_finite_pass=bool(data["all_finite_pass"]),
            bottom_up_mcap_pass=bool(data["bottom_up_mcap_pass"]),
            bottom_up_mscale_pass=bool(data["bottom_up_mscale_pass"]),
            actionable_warning_count=int(data["actionable_warning_count"]),
        )


@dataclass(slots=True)
class KernelShiftRow:
    level_idx: int
    delta_mcap: float
    delta_mscale: float
    delta_hmem: float
    delta_deff: float
    changed: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "KernelShiftRow":
        return cls(
            level_idx=int(data["level_idx"]),
            delta_mcap=float(data["delta_mcap"]),
            delta_mscale=float(data["delta_mscale"]),
            delta_hmem=float(data["delta_hmem"]),
            delta_deff=float(data["delta_deff"]),
            changed=bool(data["changed"]),
        )


@dataclass(slots=True)
class RelationRow:
    relation_type: int
    source_level: int
    target_level: int
    delta_mcap: float
    delta_mscale: float
    delta_hmem: float
    delta_deff: float
    target_mcap_not_lower: bool
    target_mscale_not_lower: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RelationRow":
        return cls(
            relation_type=int(data["relation_type"]),
            source_level=int(data["source_level"]),
            target_level=int(data["target_level"]),
            delta_mcap=float(data["delta_mcap"]),
            delta_mscale=float(data["delta_mscale"]),
            delta_hmem=float(data["delta_hmem"]),
            delta_deff=float(data["delta_deff"]),
            target_mcap_not_lower=bool(data["target_mcap_not_lower"]),
            target_mscale_not_lower=bool(data["target_mscale_not_lower"]),
        )


@dataclass(slots=True)
class WarningRow:
    severity: int
    code: int
    metric: int
    observed: float
    threshold: float
    source_level: int
    target_level: int
    actionable: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "WarningRow":
        return cls(
            severity=int(data["severity"]),
            code=int(data["code"]),
            metric=int(data["metric"]),
            observed=float(data["observed"]),
            threshold=float(data["threshold"]),
            source_level=int(data["source_level"]),
            target_level=int(data["target_level"]),
            actionable=bool(data["actionable"]),
        )


@dataclass(slots=True)
class CrossLevelReport:
    contract: CrossLevelContract
    kernel_shifts: list[KernelShiftRow]
    relations: list[RelationRow]
    warnings: list[WarningRow]

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CrossLevelReport":
        return cls(
            contract=CrossLevelContract.from_mapping(data["contract"]),
            kernel_shifts=[KernelShiftRow.from_mapping(row) for row in data["kernel_shifts"]],
            relations=[RelationRow.from_mapping(row) for row in data["relations"]],
            warnings=[WarningRow.from_mapping(row) for row in data["warnings"]],
        )


@dataclass(slots=True)
class RenormAnalysis:
    input_level: int
    bath_level: int
    direct_level: int
    primary_diagnostic: int
    raw_output_channels: int
    output_channels: int
    pruned_channels: int
    retained_weight_fraction: float
    fit_l2_error: float
    pruned_fit_l2_error: float
    fit_linf_error: float
    pruned_fit_linf_error: float
    delta_mcap: float
    delta_mscale: float
    delta_deff: float
    consistency_pass: bool
    mcap_pass: bool
    mscale_pass: bool
    deff_pass: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RenormAnalysis":
        return cls(
            input_level=int(data["input_level"]),
            bath_level=int(data["bath_level"]),
            direct_level=int(data["direct_level"]),
            primary_diagnostic=int(data["primary_diagnostic"]),
            raw_output_channels=int(data["raw_output_channels"]),
            output_channels=int(data["output_channels"]),
            pruned_channels=int(data["pruned_channels"]),
            retained_weight_fraction=float(data["retained_weight_fraction"]),
            fit_l2_error=float(data["fit_l2_error"]),
            pruned_fit_l2_error=float(data["pruned_fit_l2_error"]),
            fit_linf_error=float(data["fit_linf_error"]),
            pruned_fit_linf_error=float(data["pruned_fit_linf_error"]),
            delta_mcap=float(data["delta_mcap"]),
            delta_mscale=float(data["delta_mscale"]),
            delta_deff=float(data["delta_deff"]),
            consistency_pass=bool(data["consistency_pass"]),
            mcap_pass=bool(data["mcap_pass"]),
            mscale_pass=bool(data["mscale_pass"]),
            deff_pass=bool(data["deff_pass"]),
        )


def as_mapping(value: Any) -> Any:
    """Convert dataclass-backed schemas into plain dictionaries/lists."""

    if hasattr(value, "to_mapping"):
        return value.to_mapping()
    if isinstance(value, list):
        return [as_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [as_mapping(item) for item in value]
    return value
