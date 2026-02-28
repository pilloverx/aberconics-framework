#pragma once

#include <stddef.h>

#if defined(_WIN32)
#  if defined(GFE_CORE_EXPORTS)
#    define GFE_C_API __declspec(dllexport)
#  else
#    define GFE_C_API __declspec(dllimport)
#  endif
#else
#  define GFE_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gfe_c_status {
    GFE_C_STATUS_OK = 0,
    GFE_C_STATUS_INVALID_ARGUMENT = 1,
    GFE_C_STATUS_RUNTIME_ERROR = 2,
    GFE_C_STATUS_BUFFER_TOO_SMALL = 3
} gfe_c_status;

typedef enum gfe_c_fit_backend {
    GFE_C_FIT_BACKEND_NNLS_PG = 0,
    GFE_C_FIT_BACKEND_PRONY = 1
} gfe_c_fit_backend;

typedef enum gfe_c_compute_backend {
    GFE_C_COMPUTE_BACKEND_CPU_REF = 0,
    GFE_C_COMPUTE_BACKEND_CPU_BLAS = 1,
    GFE_C_COMPUTE_BACKEND_GPU_CUDA = 2,
    GFE_C_COMPUTE_BACKEND_GPU_HIP = 3,
    GFE_C_COMPUTE_BACKEND_GPU_SYCL = 4
} gfe_c_compute_backend;

typedef struct gfe_c_soe_fit_options {
    int backend;
    int n_basis;
    double gamma_min;
    double gamma_max;
    double threshold;
    double lr;
    int iters;

    int prony_order;
    double prony_ridge;
    double prony_root_imag_tol;
    double prony_root_min;
    double prony_root_max;
    int prony_enforce_nonnegative_weights;
    int prony_max_root_iters;
    double prony_root_tol;
} gfe_c_soe_fit_options;

typedef struct gfe_c_backend_config {
    int preferred;
    int allow_fallback;
    int fallback;
} gfe_c_backend_config;

typedef struct gfe_c_fit_result_view {
    double* gamma_fit;
    size_t gamma_fit_capacity;
    size_t* gamma_fit_size;

    double* w_fit;
    size_t w_fit_capacity;
    size_t* w_fit_size;

    double* fit;
    size_t fit_capacity;
    size_t* fit_size;
} gfe_c_fit_result_view;

GFE_C_API gfe_c_soe_fit_options gfe_c_default_soe_fit_options(void);
GFE_C_API gfe_c_backend_config gfe_c_default_backend_config(void);

GFE_C_API int gfe_c_fit_soe_kernel(const double* t,
                                   size_t t_len,
                                   const double* data,
                                   size_t data_len,
                                   const gfe_c_soe_fit_options* options,
                                   const gfe_c_backend_config* backend_cfg,
                                   gfe_c_fit_result_view* out,
                                   char* error_msg,
                                   size_t error_msg_capacity);

#ifdef __cplusplus
} // extern "C"
#endif
