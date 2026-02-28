module GFE_CAPI

using Libdl

export
    GFE_C_STATUS_OK,
    GFE_C_FIT_BACKEND_NNLS_PG,
    GFE_C_COMPUTE_BACKEND_CPU_REF,
    GFE_C_COMPUTE_BACKEND_GPU_CUDA,
    GfeCSoeFitOptions,
    GfeCBackendConfig,
    load_gfe_library,
    default_soe_fit_options,
    default_backend_config,
    fit_soe_kernel

const GFE_C_STATUS_OK = 0
const GFE_C_STATUS_INVALID_ARGUMENT = 1
const GFE_C_STATUS_RUNTIME_ERROR = 2
const GFE_C_STATUS_BUFFER_TOO_SMALL = 3

const GFE_C_FIT_BACKEND_NNLS_PG = 0
const GFE_C_FIT_BACKEND_PRONY = 1

const GFE_C_COMPUTE_BACKEND_CPU_REF = 0
const GFE_C_COMPUTE_BACKEND_CPU_BLAS = 1
const GFE_C_COMPUTE_BACKEND_GPU_CUDA = 2
const GFE_C_COMPUTE_BACKEND_GPU_HIP = 3
const GFE_C_COMPUTE_BACKEND_GPU_SYCL = 4

Base.@kwdef mutable struct GfeCSoeFitOptions
    backend::Cint = GFE_C_FIT_BACKEND_NNLS_PG
    n_basis::Cint = 15
    gamma_min::Cdouble = 1e-2
    gamma_max::Cdouble = 10.0
    threshold::Cdouble = 0.01
    lr::Cdouble = 1e-3
    iters::Cint = 4000
    prony_order::Cint = 6
    prony_ridge::Cdouble = 1e-10
    prony_root_imag_tol::Cdouble = 1e-6
    prony_root_min::Cdouble = 1e-9
    prony_root_max::Cdouble = 1.0 - 1e-9
    prony_enforce_nonnegative_weights::Cint = 1
    prony_max_root_iters::Cint = 300
    prony_root_tol::Cdouble = 1e-10
end

Base.@kwdef mutable struct GfeCBackendConfig
    preferred::Cint = GFE_C_COMPUTE_BACKEND_CPU_REF
    allow_fallback::Cint = 1
    fallback::Cint = GFE_C_COMPUTE_BACKEND_CPU_REF
end

struct GfeCFitResultView
    gamma_fit::Ptr{Cdouble}
    gamma_fit_capacity::Csize_t
    gamma_fit_size::Ptr{Csize_t}
    w_fit::Ptr{Cdouble}
    w_fit_capacity::Csize_t
    w_fit_size::Ptr{Csize_t}
    fit::Ptr{Cdouble}
    fit_capacity::Csize_t
    fit_size::Ptr{Csize_t}
end

function _default_library_path()
    repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    return joinpath(repo_root, "code", "c++ core", "build-shared", "libgfe_core." * Libdl.dlext)
end

function load_gfe_library(; path::Union{Nothing, String} = nothing)
    candidate = isnothing(path) ? get(ENV, "GFE_CORE_LIB", _default_library_path()) : path
    return Libdl.dlopen(candidate)
end

function default_soe_fit_options(lib)
    return ccall((:gfe_c_default_soe_fit_options, lib), GfeCSoeFitOptions, ())
end

function default_backend_config(lib)
    return ccall((:gfe_c_default_backend_config, lib), GfeCBackendConfig, ())
end

function fit_soe_kernel(lib, t::Vector{Float64}, data::Vector{Float64};
                        options::Union{Nothing, GfeCSoeFitOptions} = nothing,
                        backend::Union{Nothing, GfeCBackendConfig} = nothing)
    length(t) == length(data) || throw(ArgumentError("t and data must have same length"))
    isempty(t) && throw(ArgumentError("t and data must be non-empty"))

    opts = isnothing(options) ? default_soe_fit_options(lib) : options
    cfg = isnothing(backend) ? default_backend_config(lib) : backend

    cap = max(64, length(t))
    gamma = zeros(Float64, cap)
    w = zeros(Float64, cap)
    fit = zeros(Float64, length(t))
    gamma_size = Ref{Csize_t}(0)
    w_size = Ref{Csize_t}(0)
    fit_size = Ref{Csize_t}(0)
    err = zeros(UInt8, 512)

    out = GfeCFitResultView(pointer(gamma),
                            Csize_t(length(gamma)),
                            Base.unsafe_convert(Ptr{Csize_t}, gamma_size),
                            pointer(w),
                            Csize_t(length(w)),
                            Base.unsafe_convert(Ptr{Csize_t}, w_size),
                            pointer(fit),
                            Csize_t(length(fit)),
                            Base.unsafe_convert(Ptr{Csize_t}, fit_size))

    status = ccall((:gfe_c_fit_soe_kernel, lib),
                   Cint,
                   (Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t,
                    Ref{GfeCSoeFitOptions}, Ref{GfeCBackendConfig},
                    Ref{GfeCFitResultView}, Ptr{Cchar}, Csize_t),
                   pointer(t), Csize_t(length(t)),
                   pointer(data), Csize_t(length(data)),
                   opts, cfg, out, pointer(err), Csize_t(length(err)))

    if status != GFE_C_STATUS_OK
        nul = findfirst(==(0x00), err)
        msg = isnothing(nul) ? String(err) : String(err[1:(nul - 1)])
        throw(ErrorException("gfe_c_fit_soe_kernel failed ($status): $msg"))
    end

    return gamma[1:Int(gamma_size[])], w[1:Int(w_size[])], fit[1:Int(fit_size[])]
end

end # module
