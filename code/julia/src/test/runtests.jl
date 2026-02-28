"""
Test suite for GFE.jl module.

Run with: julia test/runtests.jl
"""

using Test
using LinearAlgebra
using Random
include("../GFE.jl")
using .GFE

@testset "GFE Module Tests" begin
    
    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1: Basis Functions
    # ═════════════════════════════════════════════════════════════════════════
    
    @testset "Exponential Basis" begin
        γ = [1.0, 0.1, 0.01]
        t = 0:1.0:10
        
        Φ = exponential_basis(γ, t)
        
        # Check shape
        @test size(Φ) == (11, 3)
        
        # Check values at t=0
        @test Φ[1, :] ≈ [1.0, 1.0, 1.0]  # e^(-γ*0) = 1
        
        # Check decay property: Φ(t+1) < Φ(t) for γ > 0
        @test all(Φ[2, :] .< Φ[1, :])
        
        # Check design_matrix is alias
        Φ2 = design_matrix(γ, t)
        @test Φ ≈ Φ2
    end
    
    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2: NNLS Fitting
    # ═════════════════════════════════════════════════════════════════════════
    
    @testset "NNLS via Projected Gradient" begin
        # Create synthetic problem: fit mixture of exponentials
        γ_true = [1.0, 0.1]
        w_true = [0.6, 0.4]
        t = LinRange(0, 10, 101)
        
        A = design_matrix(γ_true, collect(t))
        b = A * w_true  # Noiseless observation
        
        # Solve with NNLS
        w_fit = nnls_pg(A, b, lr=1e-2, iters=5000)
        
        # Check solution is accurate (allowing for some numerical error)
        @test norm(w_fit - w_true, 2) < 0.01
        
        # Check non-negativity
        @test all(w_fit .≥ 0)
        
        # Check residual
        residual = norm(A * w_fit - b)
        @test residual < 1e-6
    end
    
    @testset "fit_soe_kernel integration" begin
        # Create synthetic ACF from known kernel
        γ_true = [1.0, 0.1, 0.01]
        w_true = [0.5, 0.3, 0.2]
        t_acf = 0:0.1:20
        
        A = design_matrix(γ_true, collect(t_acf))
        acf_true = A * w_true
        
        # Add small noise
        Random.seed!(42)
        acf_noisy = acf_true .+ 0.01 * randn(length(t_acf))
        
        # Fit
        γ_fit, w_fit, acf_fit = fit_soe_kernel(
            collect(t_acf), acf_noisy,
            n_basis=10, threshold=0.01
        )
        
        # Check that we recovered some of the true structure
        @test length(γ_fit) >= 2
        @test all(w_fit .> 0)
        # After fitting and pruning, γ should be sorted descending
        @test issorted(γ_fit, rev=true) || length(γ_fit) <= 1  # Single element trivially sorted
        
        # Check fit quality
        fit_error = norm(acf_fit - acf_noisy, 1) / length(t_acf)
        @test fit_error < 0.05  # Should be reasonable
    end
    
    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3: Spectral Units
    # ═════════════════════════════════════════════════════════════════════════
    
    @testset "Spectral Memory Metrics" begin
        γ = [1.0, 0.1, 0.01]
        w = [0.33, 0.33, 0.34]
        
        su = spectral_units(w, γ)
        
        # Check all metrics are computed
        @test su.Mcap > 0
        @test su.Mscale > 0
        @test su.Mres > 0
        @test su.Hmem > 0
        @test 0 <= su.Hnorm <= 1
        @test su.Deff > 0
        
        # Paper memory capacity: (Σ wᵢ/γᵢ²)/(Σ wᵢ/γᵢ)
        Mcap_manual = sum(w ./ (γ .^ 2)) / sum(w ./ γ)
        @test su.Mcap ≈ Mcap_manual
        
        # For nearly uniform w, entropy should be high
        @test su.Hnorm > 0.8  # Near log(3) ≈ 1.1, so normalized ≈ 0.99
    end
    
    @testset "Spectral Units: Single Channel" begin
        γ = [0.5]
        w = [1.0]
        
        su = spectral_units(w, γ)
        
        # Single channel (paper): Deff = L * exp(H) = 1
        @test su.Deff < 1.1
        # Mcap_paper = (1/γ²)/(1/γ) = 1/γ
        @test su.Mcap ≈ 2.0
    end
    
    @testset "Shorthand Functions" begin
        γ = [1.0, 0.1, 0.01]
        w = [0.4, 0.3, 0.3]
        
        cap = memory_capacity(w, γ)
        ent = spectral_entropy(w)
        deff = effective_dimension(w)
        
        @test cap > 0
        @test ent > 0
        @test deff > 0
        
        # Check consistency with spectral_units
        su = spectral_units(w, γ)
        @test cap ≈ su.Mcap
        @test ent ≈ su.Hnorm
        @test deff ≈ su.Deff
    end
    
    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4: Parameter Packing/Unpacking
    # ═════════════════════════════════════════════════════════════════════════
    
    @testset "Parameter Packing" begin
        γ_orig = [1.0, 0.1, 0.01]
        w_orig = [0.5, 0.3, 0.2]
        
        # Pack
        θ = pack_memory_params(γ_orig, w_orig)
        
        @test length(θ) == 6  # [log(Δγ₁), log(Δγ₂), log(Δγ₃), log(w₁), log(w₂), log(w₃)]
        @test all(isfinite.(θ))
    end
    
    @testset "Parameter Unpacking" begin
        γ_orig = [1.0, 0.1, 0.01]
        w_orig = [0.5, 0.3, 0.2]
        
        # Pack and unpack
        θ = pack_memory_params(γ_orig, w_orig)
        γ_recovered, w_recovered = unpack_memory_params(θ)
        
        # Check recovery
        @test γ_recovered ≈ γ_orig  atol=1e-10
        @test w_recovered ≈ w_orig  atol=1e-10
        
        # Check properties
        @test issorted(γ_recovered, rev=true)
        @test all(w_recovered .> 0)
    end
    
    @testset "Pack/Unpack Round Trip" begin
        # Many random round-trips
        Random.seed!(42)
        for trial in 1:10
            L = rand(2:5)
            # Create DISTINCT decay rates to avoid numerical precision issues
            γ = sort!(exp10.(rand(L) * 2 .- 1), rev=true)  # Distinct values in [0.1, 10]
            w = rand(L)
            
            θ = pack_memory_params(γ, w)
            γ_r, w_r = unpack_memory_params(θ)
            
            @test γ ≈ γ_r  atol=1e-10
            @test w ≈ w_r  atol=1e-10
        end
    end
    
    @testset "Decay Rate Ordering Validation" begin
        # Valid: descending
        γ_valid = [1.0, 0.5, 0.1]
        @test validate_decay_ordering(γ_valid) == nothing
        
        # Invalid: ascending
        γ_invalid = [0.1, 0.5, 1.0]
        @test_throws AssertionError validate_decay_ordering(γ_invalid)
        
        # Invalid: unordered
        γ_unordered = [0.5, 1.0, 0.1]
        @test_throws AssertionError validate_decay_ordering(γ_unordered)
    end
    
    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5: Utilities
    # ═════════════════════════════════════════════════════════════════════════
    
    @testset "Memory Channel Initialization" begin
        # Default initialization
        γ, w = create_memory_channels(3)
        
        @test length(γ) == 3
        @test length(w) == 3
        @test issorted(γ, rev=true)
        @test sum(w) ≈ 1.0
        @test all(w .> 0)
        
        # Custom range
        γ2, w2 = create_memory_channels(4, γ_range=(0.001, 100))
        @test length(γ2) == 4
        @test minimum(γ2) >= 0.0005  # Some slack for log discretization
        @test maximum(γ2) <= 101
    end
    
    # ═════════════════════════════════════════════════════════════════════════
    # Integration Test
    # ═════════════════════════════════════════════════════════════════════════
    
    @testset "Integration: Full Workflow" begin
        # Simulate: fit kernel, optimize parameters, measure quality
        
        # 1. Create true kernel
        γ_true = [1.0, 0.1, 0.01]
        w_true = [0.5, 0.3, 0.2]
        t = 0:0.05:20
        
        A_true = design_matrix(γ_true, collect(t))
        acf_true = A_true * w_true
        
        # 2. Fit to synthetic data
        γ_fit, w_fit, acf_fit = fit_soe_kernel(
            collect(t), acf_true,
            n_basis=12, threshold=0.001
        )
        
        # 3. Compute metrics
        su = spectral_units(w_fit, γ_fit)
        
        @test su.Mcap > 0
        @test su.Deff >= 3.0  # Paper Deff = L*exp(H), should exceed 3 for multi-mode fits
        
        # 4. Pack for optimization
        θ = pack_memory_params(γ_fit, w_fit)
        @test length(θ) == 2 * length(γ_fit)
        
        # 5. Simulate optimization (perturb and recover)
        θ_perturbed = θ .+ 0.01 * randn(length(θ))
        γ_opt, w_opt = unpack_memory_params(θ_perturbed)
        
        @test issorted(γ_opt, rev=true)
        @test all(w_opt .> 0)
    end

end  # testset

println("\n✓ All GFE tests passed!")

# Optional C-ABI wrapper test (runs when GFE_CORE_LIB is configured).
include("GFE_CAPI_validation.jl")
