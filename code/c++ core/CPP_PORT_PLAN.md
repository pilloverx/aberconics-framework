# GFE C++ Core Port Plan

## Scope
Port the numerical core from `code/julia/src/GFE.jl` to a reusable C++ library for Aberconics and downstream applications.

Julia source of truth:
- `exponential_basis`
- `design_matrix`
- `nnls_pg`
- `fit_soe_kernel`
- `spectral_units`
- `memory_capacity`
- `spectral_entropy`
- `effective_dimension`
- `pack_memory_params`
- `unpack_memory_params`
- `validate_decay_ordering`
- `create_memory_channels`

## Confirmed Baseline
`code/julia/src/test/runtests.jl` now passes (69/69) after include-path fix in test files.

## Frozen Standard (v0.1)
- Baseline date: 2026-02-25
- This baseline is the active core standard for new feature development.
- Default spectral semantics now target paper definitions:
  - `Mcap`: `sum(w/gamma^2) / sum(w/gamma)`
  - `Deff`: `L * exp(Hmem)`
- Legacy behavior is preserved via policy switches (`McapPolicy`, `DeffPolicy`).
- `Mscale == 0` handling is explicit via `MresPolicy`.
- Experiment-level OU/report parity is tracked separately and does not block feature implementation.

## Function Contracts To Preserve

### 1) Basis construction
- Input: `gamma` (size L), `t` (size N)
- Output: `Phi` with shape `(N, L)`, `Phi[n][i] = exp(-gamma[i] * t[n])`

### 2) NNLS projected gradient
- Minimize `||A w - b||^2` with `w >= 0`
- Julia behavior:
  - precompute `ATA = A^T A`, `ATb = A^T b`
  - iterate `w_new = max(w - lr * 2*(ATA*w - ATb), 0)`
  - early stop on `||w_new - w||_2 < 1e-10`

### 3) SOE fitting
- Log-spaced basis: `gamma_basis = 10^[linspace(log10(gamma_min), log10(gamma_max), n_basis)]`
- Fit with NNLS, prune by `w_i > threshold * max(w)`
- Return `gamma_fit`, `w_fit` sorted by descending `gamma`
- Return full reconstruction `fit = A * w` (unpruned basis weights)

### 4) Spectral units
- `W = sum(w)`
- `Mcap = sum(w_i / gamma_i^2) / sum(w_i / gamma_i)`
- `Mscale = log10(max(gamma) / min(gamma))`
- `Mres = L / Mscale`
- `p_i = w_i / W`
- `Hmem = -sum(p_i * log(p_i + eps))`
- `Hnorm = Hmem / log(max(L, 2))`
- `Deff = L * exp(Hmem)` (default)

### 5) Parameter packing
- Preconditions: `gamma` descending
- `dgamma[L-1] = max(gamma[L-1], 1e-12)`
- `dgamma[i] = max(gamma[i] - gamma[i+1], 1e-12)` for reverse pass
- `theta = [log(dgamma), log(w)]`

### 6) Parameter unpacking
- `L = theta.size()/2`, require even length
- `dgamma = exp(theta[0:L])`
- reverse cumulative sum to recover descending `gamma`
- `w = exp(theta[L:2L])`

### 7) Channel initialization
- Log-space `gamma` descending over `[gamma_max .. gamma_min]`
- Uniform `w` if not provided

## C++ Library Shape

### Namespaces
- `gfe::`

### Types
- `using Vec = std::vector<double>;`
- `using Mat = std::vector<Vec>;` (row-major for now)
- `struct SpectralUnits { double Mcap, Mscale, Mres, Hmem, Hnorm, Deff; };`
- `struct FitResult { Vec gamma_fit, w_fit, fit; };`

### Files
- `include/gfe/gfe_core.hpp`
- `src/gfe_core.cpp`
- `tests/` (phase 2): parity tests copied from Julia semantics

## Numerical/Design Notes
- Keep pure C++17 first (no Eigen/BLAS dependency) for easy embedding.
- Add optional accelerated backend later (Eigen/OpenBLAS).
- Treat `eps` as `std::numeric_limits<double>::epsilon()`.
- Validate dimensions and throw `std::invalid_argument` on mismatch.

## Compatibility Note
Policy switches remain in the API for legacy runs:
- `DeffPolicy::EffectiveRankOnly` -> `exp(Hmem)`
- `McapPolicy::LegacyWeightedTimescale` -> `sum(w/gamma)/sum(w)`

## Suggested Phased Delivery
1. Implement API parity in C++.
2. Add deterministic unit tests mirroring Julia tests.
3. Add golden-data cross-checks by exporting vectors from Julia.
4. Add benchmark harness (NNLS + fit_soe_kernel hot paths).
5. Add C ABI shim for application integrations.

## Next Architecture Track (Post-Parity)
- See `code/c++ core/GST_OPERATOR_DYNAMICS_PLAN.md` for implementation blueprint of:
  - GST abstraction layer
  - operator-coupled augmented dynamics
  - stable coupling forms (A/B/C-ready)
  - deferred placeholders for theorem-assumption tooling and energy bookkeeping

## Phase 1 Status (Implemented)
- Added `gfe_dynamics` API:
  - `include/gfe/gfe_dynamics.hpp`
  - `src/gfe_dynamics.cpp`
- Implemented `step_augmented(...)` for:
  - `CouplingForm::InputDriven` (Formulation A)
  - `CouplingForm::NegativeFeedback` (Formulation B)
- `CouplingForm::ResonantSecondOrder` currently explicit placeholder (throws).
- GST remains optional:
  - `GSTAdapter*` defaults to `nullptr` (operate without GST)
  - `IdentityGSTAdapter` provided
- Placeholder hooks compiled in:
  - `AssumptionChecks` (non-blocking)
  - `EnergyObserver` (no-op default)

## Build/Distribution Status (Implemented)
- `CMakeLists.txt` now supports:
  - install/export targets
  - generated package config (`gfe_coreConfig.cmake`)
  - CTest smoke registration
  - `space_clean` target for cache/temp artifact purge
- External C++ consumer path validated:
  - `find_package(gfe_core REQUIRED)`
  - `target_link_libraries(... PRIVATE gfe::gfe_core)`

## CLI-First Verification Status (Implemented)
- Added shared verification registry:
  - `include/gfe/gfe_verification.hpp`
  - `src/gfe_verification.cpp`
- Added CLI runner:
  - `apps/gfe_cli.cpp`
- Default CLI behavior runs key GFE operations:
  - `basis`, `soe_fit`, `spectral_units`, `pack_roundtrip`, `dynamics_a`, `dynamics_b`
- Current tests are linked into this path:
  - `gfe_smoke_test` executes `default_verification_operations()`
- Extension path for future experiments:
  - add a new operation in `gfe_verification.cpp`
  - it appears automatically in `gfe_cli --list` and can be included in tests

## Experiment Runner Status (Implemented)
- Added `gfe_ou_experiment` app (`apps/gfe_ou_experiment.cpp`)
- Purpose: reproduce OU-noise experiment workflow in pure C++ while reusing GFE components.
- Current GFE components used:
  - `create_memory_channels`
  - `design_matrix`
  - `nnls_pg`
  - `spectral_units`
- Output:
  - structured CLI summary
  - optional metrics CSV (`--csv <path>`) for later Python/Julia visualization

## Fitter Backends Status (Implemented)
- Added pluggable SOE fitter backend layer in `gfe_core`:
  - `FitBackend` enum
  - `SoeFitOptions` for unified fit configuration
  - `fit_soe_kernel(..., SoeFitOptions)` as the extensible front door
- Current backends:
  - `NnlsProjectedGradient` (default)
  - `Prony` (side-line experimental backend)
- Added reusable decomposition points for future fusion:
  - `logspace_gamma_basis(...)`
  - `fit_soe_on_basis(...)`
- `tests/smoke_test.cpp` now includes a Prony clean-signal regression check.

## Compute Backend Dispatch Status (Step 2 Scaffolded)
- Added domain-agnostic compute backend API:
  - `include/gfe/gfe_backend.hpp`
  - `src/gfe_backend.cpp`
- Added backend-aware SOE dispatch overloads:
  - `fit_soe_kernel(..., SoeFitOptions, BackendConfig)`
  - `fit_soe_kernel(..., n_basis, ..., BackendConfig)`
- Current implementation policy:
  - only `ComputeBackend::CpuReference` executes kernels
  - unavailable backends are explicit scaffold placeholders
  - fallback-to-CPU behavior is supported and tested
- This keeps existing behavior unchanged while preparing a stable API seam for:
  - `cpu_blas` implementation
  - GPU backends (`cuda`/`hip`/`sycl`)
