#pragma once

#include <string>
#include <vector>

namespace gfe {

enum class ComputeBackend {
    CpuReference,
    CpuBlas,
    GpuCuda,
    GpuHip,
    GpuSycl
};

struct BackendCapabilities {
    bool available = false;
    bool gpu = false;
    bool dense_linear_algebra = false;
    bool batched_kernels = false;
    bool mixed_precision = false;
    std::string note;
};

struct BackendConfig {
    ComputeBackend preferred = ComputeBackend::CpuReference;
    bool allow_fallback = true;
    ComputeBackend fallback = ComputeBackend::CpuReference;
};

struct BackendSelection {
    ComputeBackend requested = ComputeBackend::CpuReference;
    ComputeBackend resolved = ComputeBackend::CpuReference;
    bool fallback_used = false;
    std::string note;
};

const char* backend_name(ComputeBackend backend);
ComputeBackend parse_backend_name(const std::string& name);

std::vector<ComputeBackend> available_backends();
bool backend_available(ComputeBackend backend);
BackendCapabilities backend_capabilities(ComputeBackend backend);

BackendSelection select_backend(const BackendConfig& cfg);
void set_default_backend(const BackendConfig& cfg);
ComputeBackend get_default_backend();

} // namespace gfe

