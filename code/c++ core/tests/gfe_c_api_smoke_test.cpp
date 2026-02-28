#include "gfe/gfe_c_api.h"
#include "gfe/gfe_core.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool almost_equal(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

} // namespace

int main() {
    try {
        std::vector<double> t;
        for (int i = 0; i <= 80; ++i) {
            t.push_back(0.1 * static_cast<double>(i));
        }

        const gfe::Vec gamma_true{0.8, 0.12};
        const gfe::Vec w_true{0.65, 0.35};
        const auto A = gfe::design_matrix(gamma_true, t);

        std::vector<double> y(t.size(), 0.0);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = A[i][0] * w_true[0] + A[i][1] * w_true[1];
        }

        gfe_c_soe_fit_options c_opts = gfe_c_default_soe_fit_options();
        c_opts.backend = GFE_C_FIT_BACKEND_NNLS_PG;
        c_opts.n_basis = 14;
        c_opts.gamma_min = 1e-2;
        c_opts.gamma_max = 10.0;
        c_opts.threshold = 0.0;
        c_opts.lr = 1e-3;
        c_opts.iters = 5000;

        gfe_c_backend_config c_backend = gfe_c_default_backend_config();
        c_backend.preferred = GFE_C_COMPUTE_BACKEND_GPU_CUDA;
        c_backend.allow_fallback = 1;
        c_backend.fallback = GFE_C_COMPUTE_BACKEND_CPU_REF;

        std::vector<double> gamma_out(64, 0.0);
        std::vector<double> w_out(64, 0.0);
        std::vector<double> fit_out(t.size(), 0.0);
        std::size_t gamma_size = 0;
        std::size_t w_size = 0;
        std::size_t fit_size = 0;
        char errbuf[256] = {0};

        gfe_c_fit_result_view out{};
        out.gamma_fit = gamma_out.data();
        out.gamma_fit_capacity = gamma_out.size();
        out.gamma_fit_size = &gamma_size;
        out.w_fit = w_out.data();
        out.w_fit_capacity = w_out.size();
        out.w_fit_size = &w_size;
        out.fit = fit_out.data();
        out.fit_capacity = fit_out.size();
        out.fit_size = &fit_size;

        const int rc = gfe_c_fit_soe_kernel(t.data(),
                                            t.size(),
                                            y.data(),
                                            y.size(),
                                            &c_opts,
                                            &c_backend,
                                            &out,
                                            errbuf,
                                            sizeof(errbuf));
        if (rc != GFE_C_STATUS_OK) {
            std::cerr << "gfe_c_fit_soe_kernel failed: " << errbuf << "\n";
            return 1;
        }

        gfe::SoeFitOptions cpp_opts;
        cpp_opts.backend = gfe::FitBackend::NnlsProjectedGradient;
        cpp_opts.n_basis = c_opts.n_basis;
        cpp_opts.gamma_min = c_opts.gamma_min;
        cpp_opts.gamma_max = c_opts.gamma_max;
        cpp_opts.threshold = c_opts.threshold;
        cpp_opts.lr = c_opts.lr;
        cpp_opts.iters = c_opts.iters;

        gfe::BackendConfig cpp_backend;
        cpp_backend.preferred = gfe::ComputeBackend::GpuCuda;
        cpp_backend.allow_fallback = true;
        cpp_backend.fallback = gfe::ComputeBackend::CpuReference;

        const auto cpp_fit = gfe::fit_soe_kernel(t, y, cpp_opts, cpp_backend);
        if (gamma_size != cpp_fit.gamma_fit.size() || w_size != cpp_fit.w_fit.size() || fit_size != cpp_fit.fit.size()) {
            std::cerr << "C ABI/C++ result sizes mismatch\n";
            return 1;
        }
        for (std::size_t i = 0; i < fit_size; ++i) {
            if (!almost_equal(fit_out[i], cpp_fit.fit[i], 1e-9)) {
                std::cerr << "C ABI/C++ fit mismatch\n";
                return 1;
            }
        }

        // Buffer-too-small contract check.
        std::vector<double> tiny_fit(4, 0.0);
        std::size_t tiny_fit_size = 0;
        out.fit = tiny_fit.data();
        out.fit_capacity = tiny_fit.size();
        out.fit_size = &tiny_fit_size;
        const int rc_small = gfe_c_fit_soe_kernel(t.data(),
                                                  t.size(),
                                                  y.data(),
                                                  y.size(),
                                                  &c_opts,
                                                  &c_backend,
                                                  &out,
                                                  errbuf,
                                                  sizeof(errbuf));
        if (rc_small != GFE_C_STATUS_BUFFER_TOO_SMALL) {
            std::cerr << "Expected BUFFER_TOO_SMALL\n";
            return 1;
        }
        if (tiny_fit_size != cpp_fit.fit.size()) {
            std::cerr << "Expected fit_size to report required size\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "gfe_c_api_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "gfe_c_api_smoke_test passed\n";
    return 0;
}
