# gfe_core C++ Library

This directory builds the C++ `gfe_core` library for reuse in other C++ projects.

## Core Standard Status

Current baseline: `GFE Core Standard v0.1` (frozen on 2026-02-25).

- This baseline is stable for continued feature work.
- Default spectral metrics follow paper-oriented settings:
  - `McapPolicy::PaperMeanDepth`
  - `DeffPolicy::PaperEffectiveDimension`
- Legacy semantics remain available through explicit API policies.
- OU experiment/report numerical parity is tracked separately and is not a blocker for core feature development.

## Fitter Backends

SOE fitting is now backend-driven:
- default: NNLS projected gradient
- optional: Prony side-line backend

The API is structured for later fusion of additional fitters:
- `logspace_gamma_basis(...)`
- `fit_soe_on_basis(...)`
- `fit_soe_kernel(..., SoeFitOptions)`

## Compute Backend Scaffold (Step 2)

Runtime compute backend selection is now plumbed into SOE fit entry points:
- `fit_soe_kernel(..., SoeFitOptions, BackendConfig)`
- `fit_soe_kernel(..., n_basis, ..., BackendConfig)`

Current behavior remains CPU-reference by default:
- available now: `cpu_ref`
- scaffolded placeholders: `cpu_blas`, `gpu_cuda`, `gpu_hip`, `gpu_sycl`
- requesting an unavailable backend with fallback enabled resolves to `cpu_ref`

Example (GPU request with CPU fallback):

```cpp
gfe::BackendConfig cfg;
cfg.preferred = gfe::ComputeBackend::GpuCuda;
cfg.allow_fallback = true;
cfg.fallback = gfe::ComputeBackend::CpuReference;
auto fit = gfe::fit_soe_kernel(t, y, opts, cfg);
```

## C ABI Scaffold (Step A for Bindings)

Added stable C-callable entry points:
- header: `include/gfe/gfe_c_api.h`
- implementation: `src/gfe_c_api.cpp`

Current C ABI coverage:
- `gfe_c_default_soe_fit_options()`
- `gfe_c_default_backend_config()`
- `gfe_c_fit_soe_kernel(...)`

Design notes:
- plain C structs/enums only (wrapper-friendly for Python/Julia)
- explicit output buffer capacities + required-size reporting
- status codes + optional error message buffer
- routes through existing backend selection (CPU fallback behavior preserved)

## Minimal Wrappers (Python + Julia)

Build shared library for foreign-function wrappers:

```bash
cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
cmake --build "code/c++ core/build-shared" -j
```

Python `ctypes` proof-of-call:

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
python3 code/python/ctypes_smoke.py
```

Julia `ccall` proof-of-call:

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
julia code/julia/examples/05_capi_ccall_smoke.jl
```

## Build

```bash
cmake -S "code/c++ core" -B "code/c++ core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "code/c++ core/build" -j
```

## Install (local prefix)

```bash
cmake --install "code/c++ core/build" --prefix "$HOME/.local"
```

Installed package exports CMake config files under:
- `$prefix/lib/cmake/gfe_core`

## Use From Another C++ Project

```cmake
find_package(gfe_core REQUIRED)
target_link_libraries(my_target PRIVATE gfe::gfe_core)
```

## Smoke Test

```bash
cmake --build "code/c++ core/build" --target gfe_smoke_test
"code/c++ core/build"/gfe_smoke_test
```

## CLI Verification (No GUI)

Run all important GFE operations (default):

```bash
"code/c++ core/build"/gfe_cli
```

List operations:

```bash
"code/c++ core/build"/gfe_cli --list
```

Run selected operations and export CSV:

```bash
"code/c++ core/build"/gfe_cli --op soe_fit --op dynamics_b --csv /tmp/gfe_report.csv
```

## OU Experiment (CLI)

Build and run the C++ OU reproduction:

```bash
cmake --build "code/c++ core/build" --target gfe_ou_experiment
"code/c++ core/build"/gfe_ou_experiment
```

Optional CSV export:

```bash
"code/c++ core/build"/gfe_ou_experiment --csv /tmp/ou_metrics.csv
```

## ABERSOE Phase-1 (CLI)

Build and run the ABERSOE wrapper demo:

```bash
cmake --build "code/c++ core/build" --target abersoe_cli
"code/c++ core/build"/abersoe_cli --steps 500 --dt 0.01 --form B
```

List built-in scenarios:

```bash
"code/c++ core/build"/abersoe_cli --list-scenarios
```

Run deterministic nonlinear Lorenz63 scenario:

```bash
"code/c++ core/build"/abersoe_cli --scenario lorenz63 --steps 3000 --dt 0.005 --form B
```

Run resonant second-order Formulation C scenario:

```bash
"code/c++ core/build"/abersoe_cli --scenario resonant1d --steps 2000 --dt 0.01 --form C
```

Optional trajectory CSV:

```bash
"code/c++ core/build"/abersoe_cli --scenario linear --csv /tmp/abersoe_run.csv
```

Phase-4 hook passthrough visibility + energy trace:

```bash
"code/c++ core/build"/abersoe_cli --scenario linear --steps 200 --energy-csv /tmp/abersoe_energy.csv
```

ABERSOE smoke test:

```bash
cmake --build "code/c++ core/build" --target abersoe_smoke_test
"code/c++ core/build"/abersoe_smoke_test
```

## ABERSOE Phase-2 (Diagnostics + Reproducibility)

Deterministic stochastic forcing (seeded), plus diagnostics/config and backend-agnostic fit reports:

```bash
"code/c++ core/build"/abersoe_cli \
  --scenario linear \
  --steps 1000 \
  --dt 0.01 \
  --stochastic-noise-std 0.05 \
  --seed 42 \
  --diagnostics-csv /tmp/abersoe_diag.csv \
  --config-csv /tmp/abersoe_cfg.csv \
  --fit-report-csv /tmp/abersoe_fit.csv \
  --fit-backend nnls
```

Use Prony backend in the same report path:

```bash
"code/c++ core/build"/abersoe_cli --scenario linear --fit-report-csv /tmp/abersoe_fit_prony.csv --fit-backend prony
```

Hebbian learning (default rule available: Oja-normalized, opt-in):

```bash
"code/c++ core/build"/abersoe_cli \
  --scenario linear \
  --hebbian oja \
  --hebbian-lr 0.05 \
  --hebbian-decay 0.001
```

Phase-3 regression snapshot generation and tolerance check:

```bash
"code/c++ core/build"/abersoe_cli \
  --scenario lorenz63 \
  --steps 200 \
  --dt 0.005 \
  --form B \
  --snapshot-csv /tmp/abersoe_snapshot_lorenz.csv \
  --snapshot-check
```

## GFE GST Module (Reference Backend)

`gfe_gst` now includes a concrete dense symmetric eigensolver adapter:
- `gfe::DenseLinearGSTAdapter`
- modal metadata (`GSTModalMetadata`)
- basis lifecycle controls (`update_interval`, `basis_validity_horizon`)
- projection/reconstruction quality reporting (`GSTProjectionReport`)

Run the deterministic GST benchmark:

```bash
cmake --build "code/c++ core/build" --target gst_smoke_test
"code/c++ core/build"/gst_smoke_test
```

## GFE Assumptions + Energy Foundations

New domain-agnostic foundation modules:
- `gfe_assumptions`:
  - typed theorem-assumption evaluation (`AssumptionEvaluation`)
  - machine-readable evidence metrics (`AssumptionEvidence`)
  - deterministic known-valid/known-invalid corpus
- `gfe_energy`:
  - standardized quadratic energy bookkeeping (`QuadraticEnergyMonitor`)
  - per-step residual computation and optional tolerance enforcement
  - CSV export of energy residuals

Run the foundation smoke benchmark:

```bash
cmake --build "code/c++ core/build" --target gfe_foundation_smoke_test
"code/c++ core/build"/gfe_foundation_smoke_test
```

## Space-Conscious Cleanup

Clean compiled targets:

```bash
cmake --build "code/c++ core/build" --target clean
```

Purge CMake cache + temp artifacts (custom target):

```bash
cmake --build "code/c++ core/build" --target space_clean
```

Remove the entire build tree when needed:

```bash
rm -rf "code/c++ core/build"
```

## Notes
- GST support is optional in dynamics APIs.
- `gfe_dynamics` implements Formulations A/B/C stepping.
- `gfe_assumptions` and `gfe_energy` provide reference foundation layers (typed evaluator + energy residual bookkeeping).
