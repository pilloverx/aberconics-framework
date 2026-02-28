#include "abersoe/abersoe_config.hpp"

namespace abersoe {

bool validate_model_spec(const AberSOEModelSpec& model, std::string* error) {
    if (model.kernel.gamma.empty() || model.kernel.w.empty()) {
        if (error) {
            *error = "kernel.gamma and kernel.w must be non-empty";
        }
        return false;
    }
    if (model.kernel.gamma.size() != model.kernel.w.size()) {
        if (error) {
            *error = "kernel.gamma and kernel.w must have the same size";
        }
        return false;
    }
    if (model.dynamics.dt <= 0.0) {
        if (error) {
            *error = "dynamics.dt must be positive";
        }
        return false;
    }
    return true;
}

bool validate_runtime_config(const AberSOERuntimeConfig& cfg, std::string* error) {
    if (cfg.sample_stride == 0) {
        if (error) {
            *error = "sample_stride must be >= 1";
        }
        return false;
    }
    if (cfg.forcing_noise_std < 0.0) {
        if (error) {
            *error = "forcing_noise_std must be >= 0";
        }
        return false;
    }
    return true;
}

} // namespace abersoe



// Best path is: create a stable **C ABI layer** once, then bind both Python and Julia to that.

// Why this is best:
// - C++ ABI is compiler-sensitive.
// - C ABI is stable across Python/Julia and easier to package.
// - Keeps `gfe`/`abersoe` internals reusable without duplicating logic.

// Recommended architecture:

// 1. `libgfe_core` (already exists)
// - Your current C++ engine (`gfe_*`, `abersoe_*`).

// 2. `libgfe_capi` (new)
// - `extern "C"` wrappers for key APIs:
//   - create/destroy model/runtime handles
//   - run scenario
//   - get diagnostics/snapshots
//   - run fit report (NNLS/Prony)
//   - GST metadata/report access
// - Plain structs + arrays only.

// 3. Python binding
// - Fastest: start with `ctypes`/`cffi` over `libgfe_capi`.
// - Better UX later: add `pybind11` package (`gfe_py`) wrapping the same C API.
// - Expose NumPy arrays and dataclasses.

// 4. Julia binding
// - Start with `ccall` directly to `libgfe_capi`.
// - Optional later: `CxxWrap.jl` for higher-level Julia types, still backed by the same engine.

// What to export first (minimum useful surface):
// 1. `gfe_fit_soe(...)` with backend option.
// 2. `gfe_spectral_units(...)`.
// 3. `abersoe_run_scenario(...)` + diagnostics.
// 4. `abersoe_get_regression_snapshot(...)` + check function.
// 5. `abersoe_get_hook_summary(...)` and energy residual CSV function.

// Packaging:
// - Build shared libs in CMake (`BUILD_SHARED_LIBS=ON`).
// - Install headers + `libgfe_capi.so`.
// - Python: wheel via `scikit-build-core`.
// - Julia: small wrapper package with `Artifacts.toml` for library location.

// If you want, I can implement Phase 1 now:
// - create `include/gfe/gfe_capi.h` + `src/gfe_capi.cpp`
// - expose 3–5 core calls
// - add a minimal Python `ctypes` wrapper and Julia `ccall` example.