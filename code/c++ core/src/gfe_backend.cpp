#include "gfe/gfe_backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace gfe {
namespace {

ComputeBackend g_default_backend = ComputeBackend::CpuReference;

BackendCapabilities cpu_ref_caps() {
    BackendCapabilities c;
    c.available = true;
    c.gpu = false;
    c.dense_linear_algebra = true;
    c.batched_kernels = false;
    c.mixed_precision = false;
    c.note = "reference scalar/vector CPU path";
    return c;
}

BackendCapabilities unavailable_caps(const char* note, bool gpu) {
    BackendCapabilities c;
    c.available = false;
    c.gpu = gpu;
    c.dense_linear_algebra = false;
    c.batched_kernels = false;
    c.mixed_precision = false;
    c.note = note;
    return c;
}

} // namespace

const char* backend_name(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::CpuReference:
            return "cpu_ref";
        case ComputeBackend::CpuBlas:
            return "cpu_blas";
        case ComputeBackend::GpuCuda:
            return "gpu_cuda";
        case ComputeBackend::GpuHip:
            return "gpu_hip";
        case ComputeBackend::GpuSycl:
            return "gpu_sycl";
    }
    return "unknown";
}

ComputeBackend parse_backend_name(const std::string& name) {
    if (name == "cpu_ref") {
        return ComputeBackend::CpuReference;
    }
    if (name == "cpu_blas") {
        return ComputeBackend::CpuBlas;
    }
    if (name == "gpu_cuda") {
        return ComputeBackend::GpuCuda;
    }
    if (name == "gpu_hip") {
        return ComputeBackend::GpuHip;
    }
    if (name == "gpu_sycl") {
        return ComputeBackend::GpuSycl;
    }
    throw std::invalid_argument("Unknown backend name: " + name);
}

BackendCapabilities backend_capabilities(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::CpuReference:
            return cpu_ref_caps();
        case ComputeBackend::CpuBlas:
            return unavailable_caps("BLAS backend scaffolded but not implemented", false);
        case ComputeBackend::GpuCuda:
            return unavailable_caps("CUDA backend scaffolded but not implemented", true);
        case ComputeBackend::GpuHip:
            return unavailable_caps("HIP backend scaffolded but not implemented", true);
        case ComputeBackend::GpuSycl:
            return unavailable_caps("SYCL backend scaffolded but not implemented", true);
    }
    return unavailable_caps("Unknown backend", false);
}

bool backend_available(ComputeBackend backend) {
    return backend_capabilities(backend).available;
}

std::vector<ComputeBackend> available_backends() {
    std::vector<ComputeBackend> out;
    const ComputeBackend all[] = {
        ComputeBackend::CpuReference,
        ComputeBackend::CpuBlas,
        ComputeBackend::GpuCuda,
        ComputeBackend::GpuHip,
        ComputeBackend::GpuSycl};
    for (ComputeBackend b : all) {
        if (backend_available(b)) {
            out.push_back(b);
        }
    }
    return out;
}

BackendSelection select_backend(const BackendConfig& cfg) {
    BackendSelection out;
    out.requested = cfg.preferred;

    if (backend_available(cfg.preferred)) {
        out.resolved = cfg.preferred;
        out.fallback_used = false;
        out.note = "preferred backend available";
        return out;
    }

    if (cfg.allow_fallback && backend_available(cfg.fallback)) {
        out.resolved = cfg.fallback;
        out.fallback_used = true;
        out.note = std::string("fallback selected from unavailable backend ") + backend_name(cfg.preferred);
        return out;
    }

    throw std::invalid_argument(std::string("Requested backend unavailable: ") + backend_name(cfg.preferred));
}

void set_default_backend(const BackendConfig& cfg) {
    const auto sel = select_backend(cfg);
    g_default_backend = sel.resolved;
}

ComputeBackend get_default_backend() {
    return g_default_backend;
}

} // namespace gfe

