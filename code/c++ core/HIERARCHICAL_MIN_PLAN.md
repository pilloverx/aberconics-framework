# Hierarchical MIN Plan

## Purpose
`hierarchical_min` is the proposed multi-scale orchestration layer above `abersoe`.

It should provide:
- a stable representation of multiple memory-bearing levels
- explicit bottom-up and top-down coupling between levels
- CM-preserving modulation of lower-level kernels
- renormalisation-flow utilities for moving between scales
- diagnostics and tests that make hierarchy behavior measurable rather than decorative

This module is the practical bridge from single-system Aberconics dynamics to explicit multi-scale memory architectures.

## Scope Boundary
- `gfe_core`: kernel numerics, SOE fitters, spectral units, backend dispatch
- `gfe_dynamics`: local auxiliary-variable stepping
- `abersoe`: single-level model assembly and runtime execution
- `hierarchical_min`: multi-level composition, coupling assembly, renormalisation checks, cross-level diagnostics

## Why This Layer Exists
The current repository can already represent explicit memory inside one reduced system. `MIN_SOE.md` extends that picture:
- each level of organization carries its own CM memory kernel
- each level admits SOE embedding
- levels are coupled in both directions
- coarse-graining induces a renormalisation flow on kernels and spectral units

The implementation target is therefore not "more scenarios" but a new orchestration layer with explicit scale-to-scale contracts.

## Design Principles
- Keep the hierarchy layer above `abersoe`; do not pollute `gfe_core` with multi-level policy.
- Preserve CM by construction under top-down modulation.
- Represent cross-level coupling explicitly as typed edges, not ad hoc callbacks hidden inside one monolithic model.
- Make renormalisation utilities deterministic and inspectable.
- Require diagnostics at each phase so hierarchy claims can be verified numerically.
- Start with trees and chains; leave general graphs and cyclic scheduling for later unless demanded by a concrete use case.

## Proposed Module Layout

```text
code/c++ core/
  include/abersoe/
    hierarchical_min.hpp
    hierarchical_min_coupling.hpp
    hierarchical_min_diagnostics.hpp
    hierarchical_min_renorm.hpp
  src/
    hierarchical_min.cpp
    hierarchical_min_coupling.cpp
    hierarchical_min_diagnostics.cpp
    hierarchical_min_renorm.cpp
  apps/
    hierarchical_min_cli.cpp
  tests/
    hierarchical_min_smoke_test.cpp
    hierarchical_min_renorm_test.cpp
    hierarchical_min_integration_test.cpp
```

## Core Abstractions

### 1) Level specification
- `HierarchicalLevelSpec`
  - `std::string name`
  - `abersoe::AberSOEModelSpec model`
  - optional level metadata: scale label, observable name, expected timescale band

### 2) Level state
- `HierarchicalLevelState`
  - `abersoe::AberSOEState state`
  - cached spectral summary for current kernel

### 3) Coupling edge
- `HierarchicalCouplingEdge`
  - `std::size_t source_level`
  - `std::size_t target_level`
  - `CouplingDirection direction`:
    - `BottomUpForcing`
    - `TopDownKernelModulation`
  - typed modulation/forcing callbacks

### 4) CM-safe kernel modulation
- `KernelModulationRule`
  - maps slow state to unconstrained `alpha`, `beta`
  - materializes:
    - `w_l(X_n) = exp(alpha_l(X_n))`
    - `gamma_l(X_n) = exp(beta_l(X_n))`
  - optional normalization / pruning policy

### 5) Hierarchical model
- `HierarchicalMINModel`
  - ordered level list
  - edge list
  - schedule policy
  - renormalisation options
  - diagnostics options

### 6) Execution API
- `step(model, hierarchy_state) -> hierarchy_state_next`
- `run(model, hierarchy_state0, runtime_cfg) -> HierarchicalRunResult`
- `build_renormalized_kernel(level_n, bath_n, opts) -> FitResult / MemoryKernelParams`
- `summarize_cross_level_spectral_units(...)`

### 7) Diagnostics
- `HierarchicalDiagnostics`
  - per-level boundedness / finiteness / energy summaries
  - cross-level `D_eff`, `M_cap`, `M_scale`, `H_mem`
  - channel-count growth / pruning stats
  - CM-preservation flags
  - renormalisation consistency checks

## Coupling Semantics

### Bottom-up coupling
At level `n`, the forcing term may depend on output from level `n-1`:

```text
F_n(t) = h(X_{n-1}(t))
```

Implementation rule:
- use source-level observable projections, not raw state concatenation by default
- preserve deterministic execution order within a step

### Top-down coupling
At level `n-1`, the kernel parameters may depend on the slow state of level `n`:

```text
K_{n-1}(t; X_n) = Σ_l w_l(X_n) exp(-gamma_l(X_n) t)
```

Implementation rule:
- top-down modulation updates `w` and `gamma` before stepping the fast level
- positivity of `w` and `gamma` is enforced structurally
- optional renormalization of total weight is explicit policy, not hidden behavior

### Coupling Assembly Rule
The runtime contract should make all three relationships explicit:
- level `n-1 -> n`: observable output drives slow forcing
- level `n -> n-1`: slow state modulates fast kernel parameters
- each fixed fast-level kernel instance must remain CM

## Renormalisation Utilities

### Required functionality
- SOE convolution:
  - `K_{n+1} = K_n * K_bath_n + K_direct_n`
- pruning / compression after convolution
- spectral summary before and after renormalisation
- optional channel-merging heuristics for nearly degenerate rates

### Required invariants
- CM preserved when inputs are CM
- SOE structure preserved under convolution + pruning
- rates remain positive after compression
- diagnostics expose when approximation, merging, or pruning changes `D_eff`

### First implementation target
Start with deterministic SOE-to-SOE renormalisation in the finite-dimensional case:
- no branch-cut / power-law inference yet
- no inverse renormalisation yet
- direct term optional but typed

## Phase Plan

### Phase 1: Hierarchy skeleton + 2-level runtime
Deliver:
- new hierarchy headers/sources
- ordered level container and state container
- explicit edge definitions for bottom-up forcing and top-down kernel modulation
- 2-level step/run wrapper over existing `abersoe::step` / `abersoe::run`
- CLI app for inspecting per-level trajectories and kernels

Tests required before phase completion:
- unit:
  - level-index validation
  - edge validation (no invalid endpoints, no unsupported self-edge policy)
  - CM-safe modulation returns positive `w` and `gamma`
  - top-down modulation actually changes lower-level parameters when configured
- integration:
  - 2-level toy system runs for fixed steps with all-finite outputs
  - bottom-up forcing measurably changes upper-level trajectory
  - top-down modulation measurably changes lower-level kernel summary
- regression:
  - deterministic seeded hierarchy run reproduces trajectory summary within tolerance

Definition of done:
- a 2-level demo can run end to end
- each level remains individually valid as an `abersoe` model
- coupling is numerically visible, not just syntactically present

### Phase 2: Cross-level diagnostics + spectral reporting
Deliver:
- per-level and cross-level spectral reports
- current-kernel summaries on every sampled level
- CSV export for hierarchy metrics
- warnings for CM violation, NaN/Inf drift, or invalid modulation output
- stable schema headers for hierarchy CSV/report surfaces
- regression snapshots for cross-level summaries, not only final-state summaries

Tests required before phase completion:
- unit:
  - spectral-summary aggregation shape and field completeness
  - CSV/report formatting stability
  - diagnostics remain finite for stable toy examples
- integration:
  - 2-level and 3-level toy examples report per-level `D_eff`, `M_cap`, `M_scale`
  - increasing slow timescale in upper level shifts lower-level reported summaries in expected direction
- regression:
  - golden metrics for simple hierarchy examples
  - cross-level regression snapshots for canonical scenarios
  - warning-code list stability for canonical scenarios where warnings are part of the intended diagnostic surface

Definition of done:
- hierarchy runs can be compared using spectral units rather than only raw state plots
- reporting headers and cross-level summary fields are stable enough to treat as a contract surface

### Phase 3: Renormalisation-flow utilities
Deliver:
- SOE convolution utility
- optional direct-memory term
- compression/pruning policy after convolution
- renormalised kernel report:
  - input channels
  - output channels
  - kept/pruned counts
  - spectral units before/after
- stable renorm CSV/report schema
- regression snapshots for canonical renorm analyses, not only one-off drift assertions

Tests required before phase completion:
- unit:
  - SOE convolution against analytically constructed small examples
  - positivity / finiteness of renormalized rates and weights
  - degenerate-rate handling path
  - pruning leaves output sorted by descending rate
- integration:
  - renormalized kernel can be plugged back into a level runtime and stepped
  - `M_scale` and `M_cap` shift consistently for known slow-bath toy examples
- property checks:
  - CM-preservation for CM input kernels
  - output fit remains within tolerance after prune/compress
- regression:
  - renorm report snapshots for canonical scenario flows
  - renorm CSV schema-header stability

Definition of done:
- we can compute and inspect a finite-dimensional renormalisation step between two levels
- renorm report fields are stable enough to be treated as a contract surface

### Phase 4: Renormalisation consistency checks
Deliver:
- automated checks for:
  - sub-multiplicative `D_eff` bound
  - monotone or non-decreasing `M_scale` across designated hierarchies
  - expected channel suppression / accumulation patterns
- diagnostics that distinguish:
  - physical channel merging
  - over-pruning
  - fit instability

### Pre-Phase E: Core shaping for constrained custom chain assembly
Deliver:
- a plain-data internal spec layer for constrained non-canonical chain hierarchies
- stable validation entrypoints for custom chain specs before any C ABI freezing
- one generic model-construction path that canonical scenarios and future custom specs can both lean on where practical

Recommended core-side spec surface:
- `HierarchicalLevelSpec`
  - plain-data per-level fields only
  - explicit kernel data (`gamma`, `w`)
  - explicit initial state data (`u`, `chi`)
  - optional lightweight metadata such as a level name or label
- `HierarchicalEdgeSpec`
  - `source_level`
  - `target_level`
  - adjacent relation type only
  - plain-data coupling parameters only
- `HierarchicalChainSpec`
  - ordered level list
  - adjacent edge list
  - optional runtime-facing defaults that are safe to keep data-only
- `validate_hierarchical_chain_spec(...)`
- `make_hierarchical_model_from_spec(...)`

Important constraint:
- this prep phase should target custom chain-style construction only
- it should not silently widen into arbitrary graph assembly
- it should not introduce callback-shaped ABI commitments through the back door

Testing rule for the first custom non-canonical integration test:
- the non-canonical example must remain only a test fixture, not a privileged runtime path
- do not hardcode scenario-specific branches in the production constructor just to make the test pass
- the test must exercise the same generic spec validation and construction path intended for future custom assembly
- the test should prove that the spec surface is genuinely reusable, not that a one-off special case was wired in

Definition of done:
- the core has a stable, data-oriented custom chain spec shape that feels natural in C++ first
- one non-canonical custom-chain integration test passes through the generic construction path
- we have enough confidence to mirror that shape into a future Phase E C ABI without overfreezing callback-heavy internals
- structured contract surfaces for:
  - actionable cross-level warnings
  - scenario-level pass/fail evaluation
  - renorm structural validity beyond raw theory metrics
- scenario-specific reference expectations for canonical hierarchies so contract results can be regression-tested, not only inspected interactively

Tests required before phase completion:
- unit:
  - bound-check logic on hand-constructed examples
  - failure messaging for each broken invariant
- integration:
  - reference hierarchy where `D_eff^(n+1) <= D_eff^n * D_eff(bath^n)` passes
  - adversarial example where the check fails and emits the correct diagnostic path
- regression:
  - stored cross-level consistency reports for reference models
  - scenario-specific contract expectations for:
    - `two_level_bidirectional`
    - `three_level_homeostatic`
  - renorm contract expectations for the benchmark-derived plug-back flow

Definition of done:
- hierarchy behavior is evaluated against theory claims, not just runtime stability
- benchmark/report warnings are machine-actionable rather than prose-only
- reference scenarios have documented contract expectations that are enforced in tests

### Phase 5: 3-level benchmark and scenario registry
Deliver:
- hierarchy scenario registry
- at least one canonical 3-level benchmark:
  - fast oscillator / channel-like process
  - intermediate adaptive regulator
  - slow homeostatic or plasticity-like modulator
- CLI scenario runner with CSV export

Tests required before phase completion:
- integration:
  - benchmark hierarchy runs in strict finite mode
  - per-level spectral summaries are emitted
  - top-down and bottom-up couplings can be toggled independently for ablation checks
- regression:
  - benchmark summary snapshots
  - ablation comparisons produce expected directional changes

Definition of done:
- there is a reusable hierarchy benchmark, not only toy unit tests

### Phase 6: Infinite-dimensional / fractal diagnostics
Deliver:
- power-law tail detector
- provisional fractal exponent estimate `d_s`
- truncation error estimator for finite SOE approximation
- explicit labeling of finite-truncation vs effectively infinite-memory regime

Tests required before phase completion:
- unit:
  - tail-fit estimation on synthetic power-law kernels
  - truncation error bound monotonicity with increasing channel count
- integration:
  - synthetic fractal-memory example triggers the infinite-memory diagnostic path
  - finite SOE examples do not falsely trigger it
- regression:
  - fixed synthetic examples keep `d_s` estimates within tolerance bands

Definition of done:
- the library can say when finite SOE is enough and when it is only an approximation

Status note:
- Phase 6 is currently paused.
- Reason:
  - the current priority is to stabilize the finite-dimensional engine surface first:
    - hierarchy runtime
    - cross-level diagnostics
    - renormalisation reports
    - consistency contracts
    - C ABI / wrapper access to the already-implemented core
- Practical implication:
  - finite-dimensional SOE hierarchies remain the supported and actively hardened path
  - infinite-dimensional / fractal diagnostics should not be treated as part of the current application-facing contract

## Testing Matrix

### Unit tests
- hierarchy config validation
- edge validation and schedule validation
- CM-safe parameterization of top-down modulation
- per-level observable projection helpers
- SOE convolution and pruning
- spectral aggregation and bound-check helpers
- fractal / truncation estimators

### Integration tests
- 2-level stable toy hierarchy
- 2-level hierarchy with only bottom-up coupling
- 2-level hierarchy with only top-down modulation
- 3-level benchmark hierarchy
- renormalized kernel plugged back into runtime
- ablation tests comparing no-coupling vs one-way vs bidirectional coupling

### Regression tests
- seeded trajectory summaries for reference hierarchies
- cross-level spectral metrics snapshots
- renormalisation summaries after prune/compress
- benchmark ablation outputs
- scenario-specific contract pass/fail expectations and actionable-warning metadata checks
- CSV schema-header stability checks for hierarchy report outputs
- scenario-specific renorm regression snapshots for canonical hierarchy flows

### Property / invariant tests
- CM-preservation under allowed modulation
- positivity of `w`, `gamma`
- sorted decay ordering after renormalisation
- `D_eff` bound checks where assumptions apply
- all-finite outputs in strict mode

## Initial File and Target Order
1. Add hierarchy headers with minimal typed structs.
2. Add validation helpers and unit tests for model/edge config.
3. Add 2-level runtime wrapper and smoke test.
4. Add cross-level diagnostics and CSV reporting.
5. Add renormalisation utilities and analytical tests.
6. Add hierarchy CLI and benchmark scenarios.
7. Add fractal-memory diagnostics after finite-dimensional path is stable.

## Non-Goals (for now)
- arbitrary cyclic graph scheduling with implicit fixed-point solves
- inverse renormalisation / inverse design at first pass
- auto-learning of all cross-level functions from raw data
- high-performance distributed execution before semantics are stable
- domain-specific biological pipelines baked directly into core C++ APIs

## Readiness Gates

### Before Phase 1
- existing `abersoe` runtime remains the single-level source of truth
- hierarchy layer must wrap, not duplicate, single-level stepping logic

### Before Phase 3
- per-level diagnostics from Phase 2 must be stable enough to evaluate renormalisation effects

### Before Phase 5
- at least one 2-level example must already satisfy the coupling and diagnostics invariants

### Before Phase 6
- finite-dimensional renormalisation tests must be passing and numerically trustworthy

## Practical Decision
- Treat `hierarchical_min` as an orchestration and diagnostics layer above `abersoe`.
- Keep the first milestone intentionally narrow: a 2-level model with visible bidirectional coupling and trustworthy diagnostics.
- Do not claim multi-scale support until renormalisation utilities and cross-level tests are in place.

## Current Phase-4 Boundary Note
- The implementation now has an initial real contract layer, not only report-generation:
  - cross-level contract evaluation
  - structured actionable warnings
  - expanded renorm consistency evaluation
- This is enough to start locking scenario expectations into regression tests.
- It is not yet the end of Phase 4:
  - additional designated hierarchies may still need their own explicit consistency expectations
  - future threshold changes must be documented in `Context.md` when they alter contract meaning
