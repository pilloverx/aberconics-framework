# ABERSOE Module Plan

## Purpose
`abersoe` is the Aberconics-facing module built on top of `gfe_core` and `gfe_dynamics`.

It should provide:
- a stable, reusable augmented-memory dynamics API
- explicit SOE memory channels as first-class model components
- clear coupling semantics for Aberconics formulations (A/B/C path)
- experiment-ready runners for ODE/PDE-style workloads without GUI dependency

This module is the practical bridge from generic GFE numerics to Aberconics system behavior.

## Scope Boundary
- `gfe_core`: fitting/statistics/kernel numerics
- `gfe_dynamics`: low-level augmented stepping primitives
- `abersoe`: model assembly, runtime orchestration, stability guardrails, diagnostics

## Validation Position (OU Test)
OU colored-noise reproduction is a valid **integration check** for `abersoe` because it exercises:
- SOE fit pipeline
- memory channel realization
- spectral metrics reporting

But OU alone is not enough; `abersoe` also needs:
- deterministic unit checks for stepping/coupling semantics
- regression checks on nonlinear dynamics (Lorenz-like)
- parameter-pack roundtrip and fitter backend consistency checks

## Proposed Module Layout

```text
code/c++ core/
  include/abersoe/
    abersoe_model.hpp
    abersoe_runtime.hpp
    abersoe_config.hpp
    abersoe_diagnostics.hpp
  src/
    abersoe_model.cpp
    abersoe_runtime.cpp
    abersoe_diagnostics.cpp
  apps/
    abersoe_cli.cpp
  tests/
    abersoe_smoke_test.cpp
```

## Core Abstractions

### 1) Model specification
- `AberSOEModelSpec`
  - `gfe::MemoryKernelParams kernel`
  - `gfe::DynamicsConfig dynamics`
  - operator callbacks (`apply_L`, `apply_N`, `forcing`)
  - optional GST adapter
  - optional Hebbian memory-weight learning config (default rule: Oja-normalized, opt-in)

### 2) Runtime state
- `AberSOEState`
  - `gfe::State u`
  - `gfe::Vec chi`
  - `double t`

### 3) Execution API
- `step(model, state) -> state_next`
- `run(model, state0, n_steps, observer?) -> trajectory/summary`
- `simulate_until(model, state0, t_end, observer?)`

### 4) Diagnostics
- `fit quality`: L1/L2/kernel reconstruction stats
- `memory metrics`: Mcap/Mscale/Mres/Hmem/Hnorm/Deff
- `stability indicators`: norm growth, boundedness flags, NaN/Inf checks

## Coupling Semantics (Initial)
- Formulation A: input-driven memory (`chi` driven by forcing; memory term added)
- Formulation B: negative-feedback memory (`chi` driven by state; memory term subtracted)
- Formulation C: resonant second-order path over paired state layout `[q..., v...]`

Default policy:
- Use Formulation B as default in `abersoe` runtime presets.

## Fitter Integration
`abersoe` should consume the backend-driven fitter layer from `gfe_core`:
- default: `FitBackend::NnlsProjectedGradient`
- optional: `FitBackend::Prony`

Required design rule:
- `abersoe` stores fitter config in runtime/model spec so experiments can switch methods without API rewrites.

## Phase Plan

### Phase 1: Module skeleton + stable stepping wrapper
Deliver:
- `abersoe_model` and `abersoe_runtime` headers/sources
- wrappers over `gfe::step_augmented(...)`
- CLI app for minimal no-GUI run and CSV summary
- smoke test for A/B stepping and boundedness sanity

### Phase 2: Diagnostics and reproducibility
Deliver:
- deterministic seed/config support
- diagnostics output structs + CSV emitters
- backend-agnostic fit report support (NNLS/Prony)

Status:
- Implemented in current C++ core:
  - `AberSOERuntimeConfig` now supports seeded stochastic forcing controls.
  - diagnostics/config CSV writers added.
  - backend-agnostic fit report API added (`build_fit_report`) over `gfe::SoeFitOptions`.

### Phase 3: Nonlinear benchmark integration
Deliver:
- Lorenz-style reference benchmark wired through `abersoe`
- CLI scenario registry (ou, lorenz, custom)
- regression snapshots for trajectory/statistics consistency

Status:
- Implemented in current C++ core:
  - Lorenz63 reference scenario wired through `abersoe`.
  - CLI scenario registry added (`linear`, `lorenz63`, `resonant1d`).
  - Regression snapshot tooling implemented:
    - snapshot extraction from runtime outputs
    - default scenario baselines (linear/lorenz63/resonant1d)
    - tolerance-based drift checks (CLI + smoke tests)

### Phase 4: Advanced runtime hooks
Deliver:
- non-blocking theorem-assumption hook passthrough
- energy/Hamiltonian observer passthrough
- Formulation C runtime path (if/when second-order state is finalized)

Status:
- Implemented pass-through and observability:
  - assumption-check passthrough summarized in run results
  - energy observer passthrough counted/summarized in run results
  - recording energy observer utility + CSV export path
- Formulation C runtime path implemented in `gfe_dynamics` and exposed as `resonant1d` scenario in `abersoe`.

## Testing Matrix
- Unit:
  - config validation (sizes/ranges/coupling index)
  - step semantics for A/B signs
  - fitter-switch invariants (shape/order, finite outputs)
- Integration:
  - OU scenario smoke (core pipeline)
  - Lorenz scenario smoke (nonlinear regime)
- Regression:
  - CSV metric snapshots with tolerance bands

## Implementation Order (Immediate)
1. Create `include/abersoe` + `src/abersoe_*` skeleton.
2. Add `abersoe_smoke_test` target and `abersoe_cli`.
3. Implement `step`/`run` wrappers around `gfe::step_augmented`.
4. Add diagnostics summary struct and CSV writer.
5. Wire fitter options into model spec for NNLS/Prony switching.

## Non-Goals (for now)
- Sparse/PDE-scale GST backends and adaptive basis policy tuning.
- Theorem-specific high-rigor proofs beyond reference computable proxies.
- Domain-specific Hamiltonian functionals and conservative/symplectic integrator coupling.

## Foundation Audit For Current Non-Goals

This section answers: what must be furnished in `gfe_*` before each non-goal should move into `abersoe`.

### 1) Full GST eigensolver implementation
- Current foundation in `gfe_dynamics` + `gfe_gst`: Implemented (reference backend).
  - Available:
    - `GSTAdapter` interface + optional pass-through path.
    - `gfe_gst` concrete backend: `DenseLinearGSTAdapter` (dense symmetric Jacobi eigensolver).
    - modal metadata, basis lifecycle controls, mode selection, reconstruction report + failure policy.
  - Remaining: additional solver backends (sparse/PDE-scale) and adaptive basis policy tuning for large systems.
- Required `gfe` foundation before `abersoe` implementation:
  - `gfe_gst` module with concrete solver backends (at least one reference backend).
  - explicit modal metadata type (eigenvalues, basis validity window, conditioning diagnostics).
  - projection/reconstruction error metrics and failure policies.
- Readiness gate for `abersoe`:
  - deterministic GST benchmark with reconstruction tolerance tests. [Implemented: `gst_smoke_test`]
  - API stability for adaptive basis updates across timesteps.

### 2) Theorem-assumption automatic proof tooling
- Current foundation in `gfe_dynamics` + `gfe_assumptions`: Implemented (reference evaluator layer).
  - Available:
    - typed assumption evaluator API (`AssumptionEvaluation`, `AssumptionEvidence`)
    - computable criteria for kernel approximability / regularity proxies / boundedness proxy
    - deterministic corpus for known-valid and known-invalid cases
    - pass-through into `gfe::AssumptionChecks` inside `step_augmented`
  - Remaining:
    - richer theorem-specific criteria per PDE/operator class
    - stronger operator norms and uncertainty-aware bounds
- Required `gfe` foundation before `abersoe` implementation:
  - assumption-evaluator API in `gfe` (kernel approximability, regularity proxies, operator boundedness checks).
  - machine-readable result schema (pass/fail + margins + evidence metrics).
  - deterministic test corpus for known-valid and known-invalid cases.
- Readiness gate for `abersoe`:
  - `abersoe` only consumes typed evaluator output; no theorem logic embedded in `abersoe`. [Implemented]

### 3) Full energy/Hamiltonian conservation enforcement
- Current foundation in `gfe_dynamics` + `gfe_energy`: Implemented (reference bookkeeping layer).
  - Available:
    - standardized quadratic energy bookkeeping API (`QuadraticEnergyMonitor`)
    - transition-level residual computation (`dE/dt` vs `power-dissipation`)
    - tolerance enforcement path with runtime exception option
    - `EnergyObserver::on_transition(...)` hook integrated in `step_augmented`
  - Remaining:
    - domain-specific Hamiltonian functionals
    - conservative/symplectic integrator coupling for strict invariants
- Required `gfe` foundation before `abersoe` implementation:
  - standardized energy bookkeeping API in `gfe` (energy terms, dissipation terms, residuals).
  - invariant residual computation per step with tolerances.
  - integrator compatibility layer for conservative/stable stepping where required.
- Readiness gate for `abersoe`:
  - baseline scenarios with expected energy trend envelopes (conservative/dissipative classes). [Partial]
  - CI checks on energy residual tolerances. [Implemented in `gfe_foundation_smoke_test` at reference level]

## Practical Decision
- Keep advanced solver/proof/Hamiltonian extensions as non-goals until domain demands require them.
- Use current `gfe` foundations as the standard typed interface consumed by `abersoe`.
