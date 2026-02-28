#include "gfe/gfe_backend.hpp"

#include <iostream>

int main() {
    using namespace gfe;

    try {
        if (std::string(backend_name(ComputeBackend::CpuReference)) != "cpu_ref") {
            std::cerr << "backend_name cpu_ref mismatch\n";
            return 1;
        }

        if (parse_backend_name("cpu_ref") != ComputeBackend::CpuReference) {
            std::cerr << "parse_backend_name cpu_ref mismatch\n";
            return 1;
        }

        const auto caps_ref = backend_capabilities(ComputeBackend::CpuReference);
        if (!caps_ref.available || caps_ref.gpu) {
            std::cerr << "cpu_ref capabilities mismatch\n";
            return 1;
        }

        const auto caps_cuda = backend_capabilities(ComputeBackend::GpuCuda);
        if (caps_cuda.available || !caps_cuda.gpu) {
            std::cerr << "gpu_cuda scaffold capabilities mismatch\n";
            return 1;
        }

        const auto avail = available_backends();
        if (avail.empty() || avail.front() != ComputeBackend::CpuReference) {
            std::cerr << "available_backends should include cpu_ref\n";
            return 1;
        }

        BackendConfig cfg_ok;
        cfg_ok.preferred = ComputeBackend::CpuReference;
        const auto sel_ok = select_backend(cfg_ok);
        if (sel_ok.resolved != ComputeBackend::CpuReference || sel_ok.fallback_used) {
            std::cerr << "select_backend cpu_ref mismatch\n";
            return 1;
        }

        BackendConfig cfg_fallback;
        cfg_fallback.preferred = ComputeBackend::GpuCuda;
        cfg_fallback.allow_fallback = true;
        cfg_fallback.fallback = ComputeBackend::CpuReference;
        const auto sel_fb = select_backend(cfg_fallback);
        if (sel_fb.resolved != ComputeBackend::CpuReference || !sel_fb.fallback_used) {
            std::cerr << "select_backend fallback mismatch\n";
            return 1;
        }

        bool threw = false;
        try {
            BackendConfig cfg_fail;
            cfg_fail.preferred = ComputeBackend::GpuCuda;
            cfg_fail.allow_fallback = false;
            (void)select_backend(cfg_fail);
        } catch (...) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "select_backend should throw without fallback for unavailable backend\n";
            return 1;
        }

        set_default_backend(BackendConfig{});
        if (get_default_backend() != ComputeBackend::CpuReference) {
            std::cerr << "default backend should be cpu_ref\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "gfe_backend_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "gfe_backend_smoke_test passed\n";
    return 0;
}

