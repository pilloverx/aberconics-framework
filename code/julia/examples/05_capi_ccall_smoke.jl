include(joinpath(@__DIR__, "..", "src", "GFE_CAPI.jl"))
using .GFE_CAPI

lib = load_gfe_library()

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

println("julia ccall smoke ok")
println("modes=$(length(gamma_fit)) fit_len=$(length(fit))")
