using Test

include("../GFE_CAPI.jl")
using .GFE_CAPI

@testset "GFE C API Wrapper" begin
    lib_path = get(ENV, "GFE_CORE_LIB", "")
    if isempty(lib_path) || !isfile(lib_path)
        @info "Skipping GFE C API wrapper tests; set GFE_CORE_LIB to shared library path"
        @test true
    else
        lib = load_gfe_library(path=lib_path)
        t = collect(0.0:0.1:8.0)
        data = 0.65 .* exp.(-0.8 .* t) .+ 0.35 .* exp.(-0.12 .* t)

        opts = default_soe_fit_options(lib)
        opts.backend = GFE_C_FIT_BACKEND_NNLS_PG
        opts.n_basis = 16
        opts.threshold = 0.0
        opts.iters = 5000

        cfg = default_backend_config(lib)
        cfg.preferred = GFE_C_COMPUTE_BACKEND_GPU_CUDA
        cfg.allow_fallback = 1
        cfg.fallback = GFE_C_COMPUTE_BACKEND_CPU_REF

        gamma_fit, w_fit, fit = fit_soe_kernel(lib, t, data; options=opts, backend=cfg)

        @test length(fit) == length(t)
        @test length(gamma_fit) > 0
        @test length(w_fit) == length(gamma_fit)
        @test all(isfinite, fit)
    end
end
