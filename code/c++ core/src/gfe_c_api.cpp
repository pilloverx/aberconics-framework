#include "gfe/gfe_c_api.h"

#include "gfe/gfe_backend.hpp"
#include "gfe/gfe_core.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

void write_error(char* error_msg, std::size_t error_msg_capacity, const std::string& msg) {
    if (!error_msg || error_msg_capacity == 0) {
        return;
    }
    const std::size_t n = std::min(error_msg_capacity - 1, msg.size());
    std::memcpy(error_msg, msg.data(), n);
    error_msg[n] = '\0';
}

gfe::FitBackend to_fit_backend(int value) {
    switch (value) {
        case GFE_C_FIT_BACKEND_NNLS_PG:
            return gfe::FitBackend::NnlsProjectedGradient;
        case GFE_C_FIT_BACKEND_PRONY:
            return gfe::FitBackend::Prony;
        default:
            throw std::invalid_argument("Unknown C fit backend");
    }
}

gfe::ComputeBackend to_compute_backend(int value) {
    switch (value) {
        case GFE_C_COMPUTE_BACKEND_CPU_REF:
            return gfe::ComputeBackend::CpuReference;
        case GFE_C_COMPUTE_BACKEND_CPU_BLAS:
            return gfe::ComputeBackend::CpuBlas;
        case GFE_C_COMPUTE_BACKEND_GPU_CUDA:
            return gfe::ComputeBackend::GpuCuda;
        case GFE_C_COMPUTE_BACKEND_GPU_HIP:
            return gfe::ComputeBackend::GpuHip;
        case GFE_C_COMPUTE_BACKEND_GPU_SYCL:
            return gfe::ComputeBackend::GpuSycl;
        default:
            throw std::invalid_argument("Unknown C compute backend");
    }
}

gfe::SoeFitOptions to_cpp_options(const gfe_c_soe_fit_options& c) {
    gfe::SoeFitOptions out;
    out.backend = to_fit_backend(c.backend);
    out.n_basis = c.n_basis;
    out.gamma_min = c.gamma_min;
    out.gamma_max = c.gamma_max;
    out.threshold = c.threshold;
    out.lr = c.lr;
    out.iters = c.iters;
    out.prony.order = c.prony_order;
    out.prony.ridge = c.prony_ridge;
    out.prony.root_imag_tol = c.prony_root_imag_tol;
    out.prony.root_min = c.prony_root_min;
    out.prony.root_max = c.prony_root_max;
    out.prony.enforce_nonnegative_weights = (c.prony_enforce_nonnegative_weights != 0);
    out.prony.max_root_iters = c.prony_max_root_iters;
    out.prony.root_tol = c.prony_root_tol;
    return out;
}

gfe::BackendConfig to_cpp_backend(const gfe_c_backend_config& c) {
    gfe::BackendConfig out;
    out.preferred = to_compute_backend(c.preferred);
    out.allow_fallback = (c.allow_fallback != 0);
    out.fallback = to_compute_backend(c.fallback);
    return out;
}

bool can_copy(const gfe_c_fit_result_view& out, const gfe::FitResult& fit) {
    return out.gamma_fit_capacity >= fit.gamma_fit.size() && out.w_fit_capacity >= fit.w_fit.size() &&
           out.fit_capacity >= fit.fit.size();
}

void copy_fit(gfe_c_fit_result_view& out, const gfe::FitResult& fit) {
    std::copy(fit.gamma_fit.begin(), fit.gamma_fit.end(), out.gamma_fit);
    std::copy(fit.w_fit.begin(), fit.w_fit.end(), out.w_fit);
    std::copy(fit.fit.begin(), fit.fit.end(), out.fit);
}

void write_sizes(gfe_c_fit_result_view& out, const gfe::FitResult& fit) {
    if (out.gamma_fit_size) {
        *out.gamma_fit_size = fit.gamma_fit.size();
    }
    if (out.w_fit_size) {
        *out.w_fit_size = fit.w_fit.size();
    }
    if (out.fit_size) {
        *out.fit_size = fit.fit.size();
    }
}

} // namespace

extern "C" {

gfe_c_soe_fit_options gfe_c_default_soe_fit_options(void) {
    const gfe::SoeFitOptions base;
    gfe_c_soe_fit_options out{};
    out.backend = GFE_C_FIT_BACKEND_NNLS_PG;
    out.n_basis = base.n_basis;
    out.gamma_min = base.gamma_min;
    out.gamma_max = base.gamma_max;
    out.threshold = base.threshold;
    out.lr = base.lr;
    out.iters = base.iters;
    out.prony_order = base.prony.order;
    out.prony_ridge = base.prony.ridge;
    out.prony_root_imag_tol = base.prony.root_imag_tol;
    out.prony_root_min = base.prony.root_min;
    out.prony_root_max = base.prony.root_max;
    out.prony_enforce_nonnegative_weights = base.prony.enforce_nonnegative_weights ? 1 : 0;
    out.prony_max_root_iters = base.prony.max_root_iters;
    out.prony_root_tol = base.prony.root_tol;
    return out;
}

gfe_c_backend_config gfe_c_default_backend_config(void) {
    const gfe::BackendConfig base;
    gfe_c_backend_config out{};
    out.preferred = static_cast<int>(base.preferred);
    out.allow_fallback = base.allow_fallback ? 1 : 0;
    out.fallback = static_cast<int>(base.fallback);
    return out;
}

int gfe_c_fit_soe_kernel(const double* t,
                         size_t t_len,
                         const double* data,
                         size_t data_len,
                         const gfe_c_soe_fit_options* options,
                         const gfe_c_backend_config* backend_cfg,
                         gfe_c_fit_result_view* out,
                         char* error_msg,
                         size_t error_msg_capacity) {
    try {
        if (!t || !data || !options || !backend_cfg || !out) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        if (t_len == 0 || data_len == 0 || t_len != data_len) {
            write_error(error_msg, error_msg_capacity, "t and data must have same non-zero length");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        if (!out->gamma_fit || !out->w_fit || !out->fit) {
            write_error(error_msg, error_msg_capacity, "Output buffers must not be null");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }

        const gfe::Vec t_vec(t, t + t_len);
        const gfe::Vec d_vec(data, data + data_len);
        const gfe::SoeFitOptions cpp_opt = to_cpp_options(*options);
        const gfe::BackendConfig cpp_backend = to_cpp_backend(*backend_cfg);

        const auto fit = gfe::fit_soe_kernel(t_vec, d_vec, cpp_opt, cpp_backend);
        write_sizes(*out, fit);

        if (!can_copy(*out, fit)) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_fit(*out, fit);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

} // extern "C"
