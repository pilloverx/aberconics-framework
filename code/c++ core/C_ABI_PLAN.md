# C ABI Plan

## Purpose
This document defines the staged plan for bringing the C ABI and language wrappers up to the current C++ core surface.

Current gap:
- the C++ core already exposes substantial `abersoe` and `hierarchical_min` functionality
- the C ABI still only exposes SOE fitting
- Python and Julia wrappers therefore lag behind the actual engine

The goal is to expose the stabilized, regression-backed engine surface first, and defer the most callback-heavy or policy-heavy APIs until later.

## Scope Boundary
- `gfe_c_api` should expose stable, wrapper-friendly, plain-C structs and functions
- Python/Julia wrappers should bind to the C ABI, not parse CLI output
- canonical scenarios and report surfaces should be accessible before arbitrary custom hierarchy assembly

## Design Principles
- prefer plain C structs/enums over opaque C++-shaped objects
- prefer scenario-runner APIs before arbitrary callback-based model assembly
- expose report/diagnostic data as typed output buffers, not as CLI text
- expose only surfaces that are already stabilized by C++ tests and regression baselines
- keep buffer-capacity + required-size reporting as the standard contract pattern
- version schema/header changes intentionally and document them in `Context.md`

## Current State

### Already exposed
- `gfe_c_default_soe_fit_options()`
- `gfe_c_default_backend_config()`
- `gfe_c_fit_soe_kernel(...)`

### Not yet exposed
- `abersoe` runtime/model/config surfaces
- `abersoe` diagnostics and scenario registry
- `hierarchical_min` scenario registry
- `hierarchical_min` run results
- cross-level diagnostics
- renormalisation reports
- regression/contract report surfaces

Update as of `2026-05-01`:
- `abersoe` scenario runner is exposed
- canonical `hierarchical_min` scenario runner is exposed
- typed cross-level report ABI is exposed
- typed adjacent renorm report ABI is exposed
- Python `ctypes` bindings now cover all of the above
- arbitrary custom hierarchy assembly is still deferred

## Rollout Plan

### Phase A: Shared ABI building blocks
Deliver:
- common status/error conventions
- shared vector/buffer view structs
- memory-kernel C structs
- spectral-units C structs
- diagnostics-summary C structs
- runtime-config C structs
- schema/header query functions where useful

Why first:
- later ABI phases should reuse one common vocabulary for buffers, metrics, and diagnostics

Definition of done:
- all later planned ABI calls can be expressed using shared plain-C types without inventing new ad hoc conventions

#### Phase A draft header target
Recommended home:
- extend `include/gfe/gfe_c_api.h` first
- if the file becomes too large, split later into:
  - `gfe_c_types.h`
  - `gfe_c_abersoe.h`
  - `gfe_c_hierarchical.h`

#### Phase A naming conventions
- enums:
  - `gfe_c_*`
- structs:
  - `gfe_c_*`
- exported functions:
  - `gfe_c_*`
- "default config" functions return structs by value
- "run/report" functions:
  - write into caller-owned buffers
  - report required sizes through `size_t*`
  - return `GFE_C_STATUS_BUFFER_TOO_SMALL` when capacities are insufficient

#### Phase A shared C enums

```c
typedef enum gfe_c_bool {
    GFE_C_FALSE = 0,
    GFE_C_TRUE = 1
} gfe_c_bool;

typedef enum gfe_c_deff_policy {
    GFE_C_DEFF_EFFECTIVE_RANK_ONLY = 0,
    GFE_C_DEFF_PAPER_EFFECTIVE_DIMENSION = 1
} gfe_c_deff_policy;

typedef enum gfe_c_mcap_policy {
    GFE_C_MCAP_LEGACY_WEIGHTED_TIMESCALE = 0,
    GFE_C_MCAP_PAPER_MEAN_DEPTH = 1
} gfe_c_mcap_policy;

typedef enum gfe_c_mres_policy {
    GFE_C_MRES_RETURN_NAN = 0,
    GFE_C_MRES_RETURN_INFINITY = 1,
    GFE_C_MRES_THROW_ON_ZERO_SCALE = 2
} gfe_c_mres_policy;
```

Notes:
- we should keep `int` fields in structs for ABI simplicity and C compatibility, even when they logically represent enums
- wrapper layers can map them to richer enums

#### Phase A shared C structs

##### 1) Array/view helpers

```c
typedef struct gfe_c_array_view {
    const double* data;
    size_t size;
} gfe_c_array_view;

typedef struct gfe_c_mut_array_view {
    double* data;
    size_t capacity;
    size_t* size;
} gfe_c_mut_array_view;

typedef struct gfe_c_string_array_view {
    const char* const* data;
    size_t size;
} gfe_c_string_array_view;

typedef struct gfe_c_mut_string_buffer {
    char* data;
    size_t capacity;
    size_t* size;
} gfe_c_mut_string_buffer;
```

Purpose:
- unify the "caller owns memory, callee reports required size" pattern
- reduce repetition in later ABI functions

##### 2) Spectral-unit configuration and outputs

```c
typedef struct gfe_c_spectral_unit_options {
    int deff_policy;
    int mcap_policy;
    int mres_policy;
    double mscale_zero_tol;
} gfe_c_spectral_unit_options;

typedef struct gfe_c_spectral_units {
    double mcap;
    double mscale;
    double mres;
    double hmem;
    double hnorm;
    double deff;
} gfe_c_spectral_units;
```

##### 3) Memory-kernel representation

```c
typedef struct gfe_c_memory_kernel_view {
    const double* gamma;
    size_t gamma_size;
    const double* w;
    size_t w_size;
} gfe_c_memory_kernel_view;

typedef struct gfe_c_memory_kernel_mut_view {
    double* gamma;
    size_t gamma_capacity;
    size_t* gamma_size;
    double* w;
    size_t w_capacity;
    size_t* w_size;
} gfe_c_memory_kernel_mut_view;
```

Rules:
- `gamma_size` and `w_size` must match for valid kernels
- later ABI functions should reject mismatches as `GFE_C_STATUS_INVALID_ARGUMENT`

##### 4) State and sampled-series views

```c
typedef struct gfe_c_state_view {
    const double* u;
    size_t u_size;
    const double* chi;
    size_t chi_size;
    double t;
} gfe_c_state_view;

typedef struct gfe_c_state_mut_view {
    double* u;
    size_t u_capacity;
    size_t* u_size;
    double* chi;
    size_t chi_capacity;
    size_t* chi_size;
    double* t;
} gfe_c_state_mut_view;
```

##### 5) Core diagnostics summaries

These are intended as ABI mirrors of already-stable C++ summary structs, not full internal object mirrors.

```c
typedef struct gfe_c_abersoe_diagnostics {
    size_t steps_executed;
    int all_finite;
    double max_abs_u;
    double max_abs_chi;
    double final_u_l2;
    double final_chi_l2;
    double mean_abs_u;
    double mean_abs_chi;
} gfe_c_abersoe_diagnostics;

typedef struct gfe_c_abersoe_config_record {
    size_t memory_channels;
    double dt;
    size_t coupling_index;
    size_t steps;
    size_t sample_stride;
    int strict_finite;
    int stochastic_forcing;
    double forcing_noise_std;
    size_t stochastic_channel;
    unsigned long long seed;
    int fit_backend;
    int hebbian_enabled;
    int hebbian_rule;
    double hebbian_learning_rate;
    double hebbian_decay;
} gfe_c_abersoe_config_record;
```

Note:
- text-like fields such as `fit_backend` / `hebbian_rule` should prefer enums in the ABI, with optional helper functions that map enum to name

##### 6) Runtime-config structs

```c
typedef struct gfe_c_abersoe_runtime_config {
    size_t steps;
    size_t sample_stride;
    int strict_finite;
    int stochastic_forcing;
    double forcing_noise_std;
    size_t stochastic_channel;
    unsigned long long seed;
} gfe_c_abersoe_runtime_config;

typedef struct gfe_c_hierarchical_runtime_config {
    size_t steps;
    size_t sample_stride;
    int strict_finite;
} gfe_c_hierarchical_runtime_config;
```

##### 7) Report schema/header query result

```c
typedef struct gfe_c_string_result_view {
    char* data;
    size_t capacity;
    size_t* size;
} gfe_c_string_result_view;
```

Use:
- CSV header/schema strings
- scenario-name listing helpers when single joined strings are acceptable

#### Phase A draft function signatures

##### Shared defaults / schema helpers

```c
GFE_C_API gfe_c_spectral_unit_options gfe_c_default_spectral_unit_options(void);
GFE_C_API gfe_c_abersoe_runtime_config gfe_c_abersoe_default_runtime_config(void);
GFE_C_API gfe_c_hierarchical_runtime_config gfe_c_hierarchical_default_runtime_config(void);
```

##### Shared utility validation/conversion helpers

```c
GFE_C_API int gfe_c_validate_memory_kernel(const gfe_c_memory_kernel_view* kernel,
                                           char* error_msg,
                                           size_t error_msg_capacity);

GFE_C_API int gfe_c_spectral_units_from_kernel(const gfe_c_memory_kernel_view* kernel,
                                               const gfe_c_spectral_unit_options* options,
                                               gfe_c_spectral_units* out,
                                               char* error_msg,
                                               size_t error_msg_capacity);
```

##### Shared header/schema query functions

These should avoid requiring wrappers to hardcode CSV field order.

```c
GFE_C_API int gfe_c_hierarchy_run_csv_header(gfe_c_string_result_view* out,
                                             char* error_msg,
                                             size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_summary_csv_header(gfe_c_string_result_view* out,
                                                 char* error_msg,
                                                 size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_diagnostics_csv_header(gfe_c_string_result_view* out,
                                                     char* error_msg,
                                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_cross_level_csv_header(gfe_c_string_result_view* out,
                                                     char* error_msg,
                                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_renorm_csv_header(gfe_c_string_result_view* out,
                                                char* error_msg,
                                                size_t error_msg_capacity);
```

##### Enum-name helpers

Useful for wrappers that want display strings without duplicating lookup tables.

```c
GFE_C_API int gfe_c_fit_backend_name(int backend,
                                     gfe_c_string_result_view* out,
                                     char* error_msg,
                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_compute_backend_name(int backend,
                                         gfe_c_string_result_view* out,
                                         char* error_msg,
                                         size_t error_msg_capacity);
```

#### Phase A implementation notes
- do not expose scenario runner functions yet in Phase A
- do not expose opaque handles yet unless Phase E requires them
- keep the shared structs intentionally boring and copy-friendly
- every struct added in Phase A should be justifiable as reusable by Phase B/C/D, not tailored to one call site

#### Phase A acceptance checklist
- a wrapper author can understand all shared scalar/vector/kernel/config/report shapes from one header area
- later ABI phases can reuse these structs without inventing new buffer conventions
- CSV schema/header strings are queryable from the ABI instead of silently duplicated in wrappers

### Phase B: `abersoe` single-level runtime ABI
Deliver:
- default `abersoe` runtime config
- scenario listing/parsing for canonical `abersoe` scenarios
- single-level scenario runner
- final-state summary + diagnostics output
- sampled trajectory export through output buffers

Recommended first functions:
- `gfe_c_abersoe_default_runtime_config()`
- `gfe_c_abersoe_list_scenarios(...)`
- `gfe_c_abersoe_run_scenario(...)`

Why this phase matters:
- it gives wrappers a meaningful engine surface quickly
- it avoids arbitrary callback injection while still exposing real runtime behavior

Definition of done:
- Python/Julia can run canonical `abersoe` scenarios and inspect typed outputs without shelling out to CLI tools

Current status:
- Started in implementation.
- Present C ABI surface now includes:
  - `gfe_c_abersoe_scenario_count()`
  - `gfe_c_abersoe_scenario_name(...)`
  - `gfe_c_abersoe_run_scenario(...)`
- The current runner is intentionally narrow:
  - canonical scenarios only
  - caller-provided runtime config
  - final-state buffers
  - diagnostics/config outputs
  - sampled summary outputs
- Not yet exposed in Phase B:
  - custom `abersoe` model assembly
  - hook attachment through the ABI
  - fit-report and regression-snapshot ABI surfaces

### Phase C: Hierarchy scenario-runner ABI
Deliver:
- hierarchy scenario listing/parsing
- hierarchy runtime config
- run canonical hierarchy scenarios:
  - `two_level_bidirectional`
  - `three_level_homeostatic`
- bottom-up/top-down enable flags
- typed final-level summaries

Recommended first functions:
- `gfe_c_hierarchical_default_runtime_config()`
- `gfe_c_hierarchical_list_scenarios(...)`
- `gfe_c_hierarchical_run_scenario(...)`

Why scenario-first:
- the canonical hierarchy surfaces are already the most regression-backed
- it gives wrappers useful hierarchy support without prematurely freezing a complex custom-assembly ABI

Definition of done:
- wrappers can run canonical hierarchy scenarios and get typed final-level results

Current status:
- Started in implementation.
- Present C ABI surface now includes:
  - `gfe_c_hierarchical_scenario_count()`
  - `gfe_c_hierarchical_scenario_name(...)`
  - `gfe_c_hierarchical_run_scenario(...)`
- The current hierarchy runner is intentionally narrow:
  - canonical scenarios only
  - bottom-up/top-down enable flags
  - caller-provided runtime config
  - typed final-level summaries
  - run diagnostics
- Bound in Python `ctypes` as:
  - `list_hierarchical_scenarios(...)`
  - `run_hierarchical_scenario(...)`
- Not yet exposed in Phase C:
  - cross-level report ABI
  - renorm report ABI
  - sampled hierarchy trajectory ABI
  - custom hierarchy construction

### Phase D: Diagnostics and renorm report ABI
Deliver:
- cross-level report structs and retrieval functions
- warning record structs and retrieval functions
- renorm analysis structs and retrieval functions
- contract/regression snapshot structs where they are stable enough
- schema/header query functions for report tables

Recommended first functions:
- `gfe_c_hierarchical_get_cross_level_report(...)`
- `gfe_c_hierarchical_get_renorm_report(...)`
- `gfe_c_hierarchical_cross_level_csv_header(...)`
- `gfe_c_hierarchical_renorm_csv_header(...)`

Important note:
- wrappers should consume typed reports directly
- CSV and CLI remain secondary interoperability/export paths, not primary wrapper APIs

Definition of done:
- wrappers can inspect cross-level warnings, contract booleans, renorm diagnostics, and drift metrics directly from ABI calls

### Phase E: Constrained custom hierarchy assembly ABI

This phase must explicitly respect the architecture boundary recorded in `Context.md` under:
- `2026-04-28 - Architecture boundary note for arbitrary N`

That boundary still applies here:
- the core may expose more custom hierarchy assembly through the ABI
- but arbitrary hierarchy depth `N`, application-specific smoothness, and practical stability policy remain the responsibility of the embedding application
- Phase E must not imply that arbitrary custom hierarchies are as mature or as guaranteed as the canonical scenario surfaces

Deliver, in the narrow first sense:
- stable plain-data custom kernel/state/config assembly where the representation is mature enough
- stable plain-data custom level arrays for chain-style hierarchy construction
- stable plain-data adjacent coupling-edge arrays
- custom hierarchy construction without relying only on canonical scenarios, but still within a constrained and well-documented subset

Before Phase E can be considered ready to freeze, we need a stable plain-data representation for:
- arbitrary level specs
- arbitrary edge/coupling specs
- custom kernel injection per level
- custom initial state assembly

Important maturity note:
- the C++ core can do more than the ABI currently exposes
- but not every internal construction pattern is mature enough to become a long-lived C contract
- Phase E should freeze only the subset that is stable, testable, and explainable without leaking callback-heavy C++ internals

This phase is intentionally later because:
- `abersoe::AberSOEModelSpec` and `HierarchicalMINModel` currently depend on C++ callbacks
- arbitrary callback bridging from Python/Julia into C++ is the highest-risk ABI surface
- a rushed design here would freeze the most brittle part of the current engine
- the architecture boundary for arbitrary `N` would be easy to over-claim if the ABI looked broader than the tested guarantees really are

Possible strategies:
- configuration-only assembly for restricted linear/operator forms
- flat spec structs for constrained chain-style hierarchies
- opaque handle-based builders if plain-data assembly proves too awkward
- callback-free canonical operator families first

Recommended first target:
- custom chain-style hierarchy assembly ABI
- adjacent bottom-up/top-down edge specification only
- explicit per-level kernels and initial states
- no arbitrary callback injection

Phase-E first-slice implementation target:
- `gfe_c_hierarchical_chain_level_spec`
- `gfe_c_hierarchical_chain_edge_spec`
- `gfe_c_hierarchical_chain_spec_view`
- `gfe_c_validate_hierarchical_chain_spec(...)`
- `gfe_c_hierarchical_run_chain_spec(...)`
- `gfe_c_hierarchical_get_cross_level_report_for_chain_spec(...)`
- `gfe_c_hierarchical_get_renorm_report_for_chain_spec(...)`

The purpose of this first slice is:
- validate that the constrained custom-chain surface is ergonomic in plain C
- prove that `spec -> model + initial state -> run` can be exercised through the ABI
- give constrained custom chains the same typed observability path as canonical scenarios
- delay richer features such as custom string tables, explicit custom report retrieval, or non-chain topology until this narrow surface proves stable

Definition of done:
- applications can assemble non-canonical but still supported chain-style models through the ABI without relying on CLI wrappers
- the ABI remains consistent with the `Context.md` boundary that arbitrary-depth practical stability remains an application-level responsibility

## Wrapper Plan

### Python
Priority: first

Why:
- quickest feedback loop
- current `ctypes` wrapper is already present
- easiest place to validate shape/ergonomics of the ABI additions

Deliver:
- typed Python structs/classes over new ABI data
- helpers for:
  - `abersoe` scenario runs
  - hierarchy scenario runs
  - cross-level report decoding
  - renorm report decoding

### Julia
Priority: second

Deliver:
- mirror the same scenario-first ABI calls
- keep the wrapper thin and close to the ABI

## Recommended ABI Sequence
1. Shared structs and conventions
2. `abersoe` scenario runner
3. hierarchy scenario runner
4. diagnostics/report ABI
5. custom assembly ABI

## What Not To Do Yet
- do not expose arbitrary C++ callbacks directly in the first ABI
- do not design wrappers around CLI parsing
- do not freeze a custom hierarchy-construction ABI before the scenario/report ABI proves out
- do not bind paused Phase 6 surfaces yet

## Readiness Gates

### Before Phase B
- `abersoe` scenario outputs and diagnostics should be considered stable enough for wrappers

### Before Phase C
- hierarchy scenario baselines should remain green in C++ tests
- cross-level report headers and semantics should remain stable

### Before Phase D
- renorm report headers and regression baselines should remain stable
- contract surfaces should be meaningful enough that wrappers can surface them directly

### Before Phase E
- we should agree what subset of custom assembly is truly stable enough to freeze into a C ABI
- we should explicitly document how that subset preserves the `Context.md` architecture boundary for arbitrary `N`
- we should avoid freezing any representation that depends on callback-heavy internal construction patterns we may still want to revise

## Practical Decision
- expose the strongest tested scenario-based engine first
- let wrappers reach useful application workflows early
- postpone arbitrary callback-driven assembly until the ABI has proven stable on the canonical paths
