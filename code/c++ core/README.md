# gfe_core C++ Library

This directory is the main implementation surface for the repository.

It currently contains:
- the reusable `gfe_core` C++ library
- the `abersoe` single-level runtime layer
- the `hierarchical_min` hierarchy, diagnostics, and renormalization layer
- the C ABI in `include/gfe/gfe_c_api.h`
- CLI tools and local test targets

## What This Layer Owns

### `gfe_*`
- SOE fitting
- spectral units and kernel utilities
- augmented stepping primitives
- backend selection
- assumption and energy bookkeeping foundations

### `abersoe`
- single-level model assembly
- scenario execution
- runtime diagnostics
- optional Hebbian memory-weight adaptation

### `hierarchical_min`
- multi-level composition
- bottom-up and top-down coupling
- cross-level diagnostics
- renormalization analysis
- constrained custom chain construction

## Current Status

This C++ layer is no longer just a fitter plus a few demos.

The current codebase already includes:
- installable `gfe_core` library packaging
- shared-library builds for wrapper use
- canonical `abersoe` and hierarchy scenarios
- typed C ABI access to runtime and report surfaces
- renormalization consistency reporting
- passing local tests for core, ABI, runtime, hierarchy, and renorm paths

What is still intentionally incomplete relative to the broader D2C reference:
- predictive-coding learning heads
- per-channel value learning runtime
- Python-side Director/TraceStore orchestration
- language/token bridge modules

So this directory should be treated as the stable numerical and runtime substrate, not yet the full end-to-end D2C training stack.

## Build

### Static/library build

```bash
cmake -S "code/c++ core" -B "code/c++ core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "code/c++ core/build" -j
```

### Shared-library build for wrappers

```bash
cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
cmake --build "code/c++ core/build-shared" -j
```

## Install

```bash
cmake --install "code/c++ core/build" --prefix "$HOME/.local"
```

Installed CMake package files land under:
- `$prefix/lib/cmake/gfe_core`

Consumer pattern:

```cmake
find_package(gfe_core REQUIRED)
target_link_libraries(my_target PRIVATE gfe::gfe_core)
```

## Main Surfaces

### Core fitting and metrics
- backend-driven SOE fitting
- spectral metrics with policy control
- CPU reference path with scaffolded backend selection

Key families:
- `fit_soe_kernel(...)`
- `fit_soe_on_basis(...)`
- `logspace_gamma_basis(...)`

### ABERSOE runtime
- formulations A, B, and C
- deterministic and seeded-stochastic execution
- scenario registry
- diagnostics/config snapshots
- optional Hebbian updates

CLI target:
- `abersoe_cli`

### Hierarchy runtime
- canonical hierarchy scenarios
- constrained custom chain specs
- typed level summaries
- cross-level contract/warning reporting
- renormalization analyses and CSV export

CLI target:
- `hierarchical_min_cli`

### C ABI
- header: `include/gfe/gfe_c_api.h`
- implementation: `src/gfe_c_api.cpp`

The ABI already covers more than fitting:
- SOE fit calls
- `abersoe` scenario runs
- `abersoe` scenario runs with explicit state/kernel overrides and trajectory export
- hierarchy scenario runs
- cross-level report retrieval
- renorm report retrieval
- constrained custom chain validation and execution

Design rule:
- plain C structs/enums only
- explicit output capacities and required-size reporting
- typed reports for wrappers instead of CLI parsing

Worth noting for current experiment work:
- the richer single-level ABERSOE ABI entrypoint is `gfe_c_abersoe_run_scenario_with_overrides(...)`
- it is the intended surface when a caller needs:
  - custom Lorenz63 initial conditions
  - custom single-level memory kernels (`gamma` / `w`)
  - the active runtime kernel returned directly
  - sampled `u` / `chi` trajectories for memory-side analysis
- trajectory output follows the normal runtime `sample_stride`; use `sample_stride = 1` for dense analysis

## Quick Commands

### Run all local tests

```bash
ctest --test-dir "code/c++ core/build" --output-on-failure
```

### Run the basic CLI

```bash
"code/c++ core/build"/gfe_cli
```

List operations:

```bash
"code/c++ core/build"/gfe_cli --list
```

### Run the OU experiment

```bash
cmake --build "code/c++ core/build" --target gfe_ou_experiment
"code/c++ core/build"/gfe_ou_experiment
```

Optional CSV export:

```bash
"code/c++ core/build"/gfe_ou_experiment --csv /tmp/ou_metrics.csv
```

### Run ABERSOE scenarios

```bash
cmake --build "code/c++ core/build" --target abersoe_cli
"code/c++ core/build"/abersoe_cli --list-scenarios
```

Linear run:

```bash
"code/c++ core/build"/abersoe_cli --scenario linear --steps 500 --dt 0.01 --form B
```

Lorenz63 run:

```bash
"code/c++ core/build"/abersoe_cli --scenario lorenz63 --steps 3000 --dt 0.005 --form B
```

Resonant Formulation C run:

```bash
"code/c++ core/build"/abersoe_cli --scenario resonant1d --steps 2000 --dt 0.01 --form C
```

Diagnostics / reproducibility example:

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

Hebbian update example:

```bash
"code/c++ core/build"/abersoe_cli \
  --scenario linear \
  --hebbian oja \
  --hebbian-lr 0.05 \
  --hebbian-decay 0.001
```

### Run hierarchy surfaces

```bash
cmake --build "code/c++ core/build" --target hierarchical_min_cli
"code/c++ core/build"/hierarchical_min_cli
```

This is the right entrypoint when you want to inspect:
- canonical hierarchy scenarios
- per-level summaries
- cross-level report output
- renormalization report output

## Test Inventory

Notable local targets include:
- `gfe_smoke_test`
- `gst_smoke_test`
- `gfe_foundation_smoke_test`
- `gfe_backend_smoke_test`
- `gfe_c_api_smoke_test`
- `abersoe_smoke_test`
- `hierarchical_min_smoke_test`
- `hierarchical_min_integration_test`
- `hierarchical_min_unit_test`
- `hierarchical_min_renorm_test`
- `hierarchical_min_renorm_integration_test`
- `hierarchical_min_consistency_test`
- `hierarchical_min_benchmark_test`

These are wired in the local CMake build and can be run together through `ctest`.

## Related Docs

- Root overview: [`/README.md`](/workspaces/aberconics-framework/README.md)
- Python wrapper guide: [`/code/python/README.md`](/workspaces/aberconics-framework/code/python/README.md)
- Julia guide: [`/code/julia/README.md`](/workspaces/aberconics-framework/code/julia/README.md)
- Technical reference extract: [`/D2C.md`](/workspaces/aberconics-framework/D2C.md)
- Project notebook: [`/Context.md`](/workspaces/aberconics-framework/Context.md)

## Cleanup

Clean compiled targets:

```bash
cmake --build "code/c++ core/build" --target clean
```

Purge CMake cache and temp artifacts:

```bash
cmake --build "code/c++ core/build" --target space_clean
```
