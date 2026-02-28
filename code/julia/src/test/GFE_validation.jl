"""
Practical GFE Module Test - Validation for Real Workflows

This test demonstrates GFE working with realistic Aberconics use cases.
Run with: julia GFE_validation.jl
"""

using Random, Statistics, LinearAlgebra

include("../GFE.jl")
using .GFE

println("\n" * "="^80)
println("GFE Module - Practical Validation Test")
println("="^80)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Exponential Basis & Design Matrix
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 1] Exponential Basis Functions")
println("─" * "─"^78)

γ = [1.0, 0.1, 0.01]
t_range = 0:0.1:10
t_vec = collect(t_range)

Φ = exponential_basis(γ, t_range)  # Test with range input
@assert size(Φ) == (101, 3) "Basis shape correct"
@assert Φ[1, :] ≈ [1.0, 1.0, 1.0] "Basis at t=0 is [1,1,1]"
@assert all(Φ[2, :] .< Φ[1, :]) "Exponentials decay"
println("  ✓ exponential_basis works with range input")
println("  ✓ Shape: $(size(Φ)) (101 samples × 3 bases)")
println("  ✓ Basis at t=0: $(round.(Φ[1, :], digits=1))")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: NNLS Fitting
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 2] NNLS Kernel Fitting")
println("─" * "─"^78)

Random.seed!(42)
γ_true = [1.0, 0.1]
w_true = [0.6, 0.4]
t = LinRange(0, 10, 101) |> collect

A = design_matrix(γ_true, t)
b = A * w_true  # Noiseless data

w_fit = nnls_pg(A, b, lr=1e-2, iters=3000)
residual = norm(A * w_fit - b)

@assert residual < 1e-5 "NNLS converges"
@assert all(w_fit .≥ 0) "Weights non-negative"
println("  ✓ NNLS converged: ||residual|| = $(round(residual, sigdigits=2))")
println("  ✓ True weights:     $(round.(w_true, digits=3))")
println("  ✓ Fitted weights:   $(round.(w_fit, digits=3))")
println("  ✓ Recovery error:   $(round(norm(w_fit - w_true), sigdigits=2))")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Kernel Fitting (High-Level)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 3] Kernel Fitting (fit_soe_kernel)")
println("─" * "─"^78)

Random.seed!(42)
γ_true = [1.0, 0.1, 0.01]
w_true = [0.5, 0.3, 0.2]
t_acf = 0:0.1:20 |> collect

A_true = design_matrix(γ_true, t_acf)
acf_true = A_true * w_true
acf_noisy = acf_true .+ 0.01 * randn(length(t_acf))

γ_fit, w_fit, acf_fit = fit_soe_kernel(
    t_acf, acf_noisy,
    n_basis=12, threshold=0.005
)

@assert length(γ_fit) >= 2 "Multiple modes recovered"
@assert all(w_fit .> 0) "Weights positive"
@assert issorted(γ_fit, rev=true) || length(γ_fit) <= 1 "Decay rates sorted"

fit_error = norm(acf_fit - acf_noisy, 1) / length(t_acf)
println("  ✓ Recovered $(length(γ_fit)) modes from 12 basis functions")
println("  ✓ Decay rates:  $(round.(γ_fit, digits=4))")
println("  ✓ Weights:      $(round.(w_fit, digits=4))")
println("  ✓ L1 fit error: $(round(fit_error, sigdigits=3))")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Spectral Memory Metrics
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 4] Spectral Memory Metrics")
println("─" * "─"^78)

su = spectral_units(w_fit, γ_fit)

println("  Spectral Units:")
println("    · Memory capacity (Mcap):      $(round(su.Mcap, digits=2)) seconds")
println("    · Spectral span (Mscale):     $(round(su.Mscale, digits=2)) decades")
println("    · Resolution (Mres):           $(round(su.Mres, digits=2)) modes/decade")
println("    · Entropy (Hmem):              $(round(su.Hmem, digits=3)) nats")
println("    · Normalized entropy (Hnorm):  $(round(su.Hnorm, digits=3))")
println("    · Effective dimension (Deff):  $(round(su.Deff, digits=2)) modes")

@assert su.Mcap > 0 "Memory capacity positive"
@assert 0 <= su.Hnorm <= 1 "Entropy normalized"
@assert su.Deff > 0 "Effective dimension positive"
println("  ✓ All metrics computed and valid")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Parameter Packing/Unpacking
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 5] Parameter Packing for Optimization")
println("─" * "─"^78)

γ_init = [1.0, 0.1, 0.01]
w_init = [0.3, 0.3, 0.4]

θ = pack_memory_params(γ_init, w_init)
γ_recovered, w_recovered = unpack_memory_params(θ)

@assert γ_init ≈ γ_recovered atol=1e-10 "γ recovered"
@assert w_init ≈ w_recovered atol=1e-10 "w recovered"
@assert issorted(γ_recovered, rev=true) "γ stays sorted"
@assert all(w_recovered .> 0) "w stays positive"

println("  ✓ Pack/unpack round-trip successful")
println("  ✓ Packed vector size: $(length(θ)) (2×L parameters)")
println("  ✓ Original γ:   $(round.(γ_init, digits=3))")
println("  ✓ Recovered γ:  $(round.(γ_recovered, digits=3))")
println("  ✓ Match error:  γ=$(round(norm(γ_init - γ_recovered), sigdigits=2)), w=$(round(norm(w_init - w_recovered), sigdigits=2))")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Memory Channel Initialization
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 6] Memory Channel Initialization")
println("─" * "─"^78)

γ_ch, w_ch = create_memory_channels(3)

@assert length(γ_ch) == 3 "Correct channel count"
@assert sum(w_ch) ≈ 1.0 "Weights normalized"
@assert issorted(γ_ch, rev=true) "γ descending"
@assert all(w_ch .> 0) "w positive"

println("  ✓ Created 3-channel system")
println("  ✓ Decay rates (γ):  $(round.(γ_ch, digits=3))")
println("  ✓ Weights (w):      $(round.(w_ch, digits=3))")
println("  ✓ Sum of weights:   $(round(sum(w_ch), digits=4))")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Shorthand Functions
# ═══════════════════════════════════════════════════════════════════════════════

println("\n[TEST 7] Shorthand Metric Functions")
println("─" * "─"^78)

cap = memory_capacity(w_fit, γ_fit)
ent = spectral_entropy(w_fit)
deff = effective_dimension(w_fit)

@assert cap ≈ su.Mcap "Shorthand matches full"
@assert ent ≈ su.Hnorm "Shorthand matches full"
@assert deff ≈ su.Deff "Shorthand matches full"

println("  ✓ memory_capacity():    $(round(cap, digits=2)) s")
println("  ✓ spectral_entropy():   $(round(ent, digits=3))")
println("  ✓ effective_dimension(): $(round(deff, digits=2))")

# ═════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("✅ GFE MODULE VALIDATION COMPLETE")
println("="^80)
println("\nAll core functionality working:")
println("  [✓] Exponential basis functions (handles ranges + vectors)")
println("  [✓] NNLS fitting with projected gradient descent")
println("  [✓] High-level kernel fitting with pruning & sorting")
println("  [✓] Comprehensive spectral memory metrics")
println("  [✓] Parameter packing/unpacking for optimization")
println("  [✓] Memory channel initialization")
println("  [✓] Shorthand metric functions")
println("\nReady for use in experiments:")
println("  • examples/01_ou_noise_demo.jl")
println("  • examples/02_lorenz_chaos_suppression.jl")
println("  • examples/03_gray_scott_patterns.jl")
println("  • examples/04_echo_task_learning.jl")
println()
