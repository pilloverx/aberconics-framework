"""
    GFE (Geometric Fourier Extension) Module

Core spectral engine for Aberconics framework. Provides:
- Exponential basis function generation and construction
- Non-negative least squares (NNLS) fitting of kernels
- Spectral memory metrics (capacity, entropy, effective dimension)
- Memory parameter packing/unpacking for optimization

Theory: Geometric Fourier Extension is a spectral method for smooth kernel approximation.
For memory systems, we use SOE (Sum-of-Exponentials) representation:
    K(t) = Σᵢ wᵢ exp(-γᵢ t)

where γᵢ are decay rates and wᵢ are weights.

Reference: papers/Aberconics_Framework_V2.0.pdf, papers/GFE_Theoretical_Foundations.pdf
"""

module GFE

using LinearAlgebra, Statistics

export
    # Basis functions
    exponential_basis,
    design_matrix,
    
    # NNLS fitting
    nnls_pg,
    fit_soe_kernel,
    
    # Spectral metrics
    SpectralUnits,
    spectral_units,
    memory_capacity,
    spectral_entropy,
    effective_dimension,
    
    # Parameter management
    pack_memory_params,
    unpack_memory_params,
    validate_decay_ordering,
    
    # Utilities
    create_memory_channels


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Exponential Basis Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    exponential_basis(γ::Vector, t) -> Matrix

Create exponential basis functions [exp(-γᵢ t) for i in eachindex(γ)].

Arguments:
- γ: decay rates (vector of length n_basis)
- t: time points (vector-like, will be converted to vector)

Returns:
- Matrix of shape (n_samples, n_basis) where each column i contains exp(-γᵢ * t)

Example:
    γ = [0.1, 0.5, 1.0]
    t = 0:0.1:10
    Φ = exponential_basis(γ, t)
"""
function exponential_basis(γ::Vector, t)
    n_basis = length(γ)
    t_vec = collect(t)  # Convert any iterable to vector
    n_samples = length(t_vec)
    Φ = zeros(n_samples, n_basis)
    for i in 1:n_basis
        @views Φ[:, i] .= exp.(-γ[i] .* t_vec)
    end
    return Φ
end

"""
    design_matrix(γ::Vector, t) -> Matrix

Alias for exponential_basis. Creates the design matrix A for NNLS fitting.
Most common usage: A = design_matrix(γ_fit, t_vals)
"""
design_matrix(γ::Vector, t) = exponential_basis(γ, t)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Non-Negative Least Squares Fitting
# ═══════════════════════════════════════════════════════════════════════════════

"""
    nnls_pg(A::Matrix, b::Vector; lr=1e-3, iters=4000, w0=nothing) -> Vector

Solve non-negative least squares via projected gradient descent:
    minimize ||Aw - b||²  subject to w ≥ 0

Uses iterative projected gradient with adaptive step size.

Arguments:
- A: design matrix (samples × basis functions)
- b: measurement vector (samples)
- lr: learning rate [default: 1e-3]
- iters: max iterations [default: 4000]
- w0: initial weights [default: zeros]

Returns:
- Fitted weight vector w

Example:
    γ_basis = exp10.(range(-2, stop=1, length=15))
    A = design_matrix(γ_basis, t_vals)
    w = nnls_pg(A, acf_measurement)
"""
function nnls_pg(A::Matrix, b::Vector; lr=1e-3, iters=4000, w0=nothing)
    (n_samples, n_basis) = size(A)
    length(b) == n_samples || throw(DimensionMismatch("A and b must have same # rows"))
    
    # Initialize
    w = isnothing(w0) ? zeros(n_basis) : copy(w0)
    
    # Precompute AT * A and AT * b for efficiency
    ATA = A' * A
    ATb = A' * b
    
    for iter in 1:iters
        # Gradient of ||Aw - b||²
        grad = 2 .* (ATA * w .- ATb)
        
        # Projected gradient step: w_new = max(w - lr * grad, 0)
        w_new = max.(w .- lr .* grad, 0.0)
        
        # Early stopping if converged
        if norm(w_new - w, 2) < 1e-10
            w = w_new
            break
        end
        w = w_new
    end
    
    return w
end

"""
    fit_soe_kernel(t::Vector, data::Vector; n_basis=15, γ_min=1e-2, γ_max=10, 
                   threshold=0.01, kwargs...) -> (γ::Vector, w::Vector, fit::Vector)

High-level kernel fitting function.

Arguments:
- t: time grid (lag times for ACF, or full timeline)
- data: measurement to fit (typically ACF)
- n_basis: number of exponential basis functions [default: 15]
- γ_min, γ_max: range of decay rates [default: 1e-2 to 10]
- threshold: pruning threshold as fraction of max weight [default: 0.01]
- kwargs: passed to nnls_pg (lr, iters, etc.)

Returns:
- (γ_fit, w_fit, fit): fitted decay rates, weights, and reconstructed data

Example:
    γ_fit, w_fit, acf_fit = fit_soe_kernel(t_vals, acf_white_ou, n_basis=15)
"""
function fit_soe_kernel(t::Vector, data::Vector; n_basis=15, γ_min=1e-2, γ_max=10,
                        threshold=0.01, kwargs...)
    # Create log-spaced basis
    γ_basis = exp10.(range(log10(γ_min), log10(γ_max), length=n_basis))
    
    # Build design matrix
    A = design_matrix(γ_basis, t)
    
    # Fit weights
    w = nnls_pg(A, data; kwargs...)
    
    # Prune small weights
    idx = w .> threshold * maximum(w)
    γ_fit = γ_basis[idx]
    w_fit = w[idx]
    
    # Sort in descending order (required for parameter packing)
    sort_idx = sortperm(γ_fit, rev=true)
    γ_fit = γ_fit[sort_idx]
    w_fit = w_fit[sort_idx]
    
    # Reconstruct fit
    fit = A * w
    
    return γ_fit, w_fit, fit
end


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Spectral Memory Units
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SpectralUnits

Container for memory metrics quantifying temporal reach and structure of a kernel.

Fields:
- Mcap: mean memory capacity (timescale in same units as input)
- Mscale: spectral span in decades (log₁₀ of γ range)
- Mres: memory resolution (# modes per decade)
- Hmem: memory entropy in nats (Shannon entropy of normalized weights)
- Hnorm: normalized entropy (Hmem / log(n_modes))
- Deff: effective dimension (e^Hmem / n_modes)
"""
struct SpectralUnits
    Mcap::Float64          # Mean memory capacity
    Mscale::Float64        # Spectral span (log₁₀)
    Mres::Float64          # Resolution (modes/decade)
    Hmem::Float64          # Entropy (nats)
    Hnorm::Float64         # Normalized entropy
    Deff::Float64          # Effective dimension
end

"""
    spectral_units(w::Vector, γ::Vector) -> SpectralUnits

Compute comprehensive memory metrics from SOE weights and decay rates.

Arguments:
- w: weights (sum should be ~1 for probability normalization)
- γ: decay rates (already sorted in descending order)

Returns:
- SpectralUnits struct with all metrics

Theory:
- Mcap = (Σᵢ wᵢ/γᵢ²) / (Σᵢ wᵢ/γᵢ) (mean temporal depth)
- Mscale = log₁₀(γ_max / γ_min)
- Hmem = -Σᵢ (wᵢ/W) log(wᵢ/W) where W = Σ wᵢ
- Deff = L * exp(Hmem) (paper effective memory dimension)

Example:
    su = spectral_units(w_fit, γ_fit)
    println("Memory capacity: \$(su.Mcap) seconds")
    println("Effective dimension: \$(su.Deff) modes")
"""
function spectral_units(w::Vector, γ::Vector)
    length(w) == length(γ) || throw(DimensionMismatch("w and γ must have same length"))
    
    # Mean temporal depth (paper): (Σ wᵢ/γᵢ²) / (Σ wᵢ/γᵢ)
    W = sum(w)
    Mcap = sum(w ./ (γ .^ 2)) / sum(w ./ γ)
    
    # Spectral span
    γ_min, γ_max = minimum(γ), maximum(γ)
    Mscale = log10(γ_max / γ_min)
    
    # Resolution (modes per decade)
    Mres = length(w) / Mscale
    
    # Entropy: -Σᵢ pᵢ log(pᵢ) with pᵢ = wᵢ / Σ wⱼ
    p = w / W
    Hmem = -sum(p .* log.(p .+ eps()))  # +eps() for numerical stability
    
    # Normalized entropy: Hmem / log(n_modes)
    Hnorm = Hmem / log(max(length(w), 2))
    
    # Paper effective dimension: L * e^Hmem
    Deff = length(w) * exp(Hmem)
    
    return SpectralUnits(Mcap, Mscale, Mres, Hmem, Hnorm, Deff)
end

"""
    memory_capacity(w::Vector, γ::Vector) -> Float64

Shorthand (paper): compute mean temporal depth = (Σ wᵢ/γᵢ²)/(Σ wᵢ/γᵢ).
"""
function memory_capacity(w::Vector, γ::Vector)
    return sum(w ./ (γ .^ 2)) / sum(w ./ γ)
end

"""
    spectral_entropy(w::Vector) -> Float64

Shorthand: compute normalized entropy = Hmem / log(n).
"""
function spectral_entropy(w::Vector)
    p = w / sum(w)
    H = -sum(p .* log.(p .+ eps()))
    return H / log(max(length(w), 2))
end

"""
    effective_dimension(w::Vector) -> Float64

Shorthand (paper): compute effective memory dimension = L * exp(Hmem).
"""
function effective_dimension(w::Vector)
    p = w / sum(w)
    H = -sum(p .* log.(p .+ eps()))
    return length(w) * exp(H)
end


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Parameter Packing & Unpacking
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pack_memory_params(γs::Vector, ws::Vector) -> Vector

Convert memory parameters (γ, w) into unconstrained optimization variables θ.

Strategy:
- γs must be in descending order (enforced)
- γ parameterization: store differences Δγ = γᵢ - γᵢ₊₁ in log space
- w parameterization: store weights in log space
- Full parameter: θ = [log(Δγ₁), ..., log(Δγₙ), log(w₁), ..., log(wₙ)]

Arguments:
- γs: decay rates, must be sorted γ₁ ≥ γ₂ ≥ ... ≥ γₙ
- ws: weights (positive)

Returns:
- θ: unconstrained parameter vector for optimization

Example:
    θ0 = pack_memory_params([1.0, 0.1, 0.01], [0.5, 0.3, 0.2])
"""
function pack_memory_params(γs::Vector, ws::Vector)
    validate_decay_ordering(γs)
    L = length(γs)
    length(ws) == L || throw(DimensionMismatch("γs and ws must have same length"))
    
    # Store Δγ differences in log space
    dγ = zeros(L)
    dγ[L] = max(γs[L], 1e-12)  # Safeguard against numerical noise
    for i in (L-1):-1:1
        dγ[i] = max(γs[i] - γs[i+1], 1e-12)
    end
    
    # Return [log(Δγ), log(w)]
    return vcat(log.(dγ), log.(ws))
end

"""
    unpack_memory_params(θ::Vector) -> (γs::Vector, ws::Vector)

Inverse of pack_memory_params. Recover memory parameters from optimization variables.

Arguments:
- θ: unconstrained parameter vector [log(Δγ₁), ..., log(Δγₙ), log(w₁), ..., log(wₙ)]

Returns:
- (γs, ws): decay rates (sorted descending) and weights (positive)

Example:
    γ, w = unpack_memory_params(θ)
    @assert issorted(γ, rev=true)
    @assert all(w .> 0)
"""
function unpack_memory_params(θ::Vector)
    L = length(θ) ÷ 2
    length(θ) == 2L || throw(DimensionMismatch("θ must have even length"))
    
    # Recover Δγ from log space
    dγ = exp.(θ[1:L])
    
    # Reconstruct γ from differences (backwards accumulation)
    γs = similar(dγ)
    γs[L] = dγ[L]
    for i in (L-1):-1:1
        γs[i] = γs[i+1] + dγ[i]
    end
    
    # Recover w from log space
    ws = exp.(θ[L+1:2L])
    
    return γs, ws
end

"""
    validate_decay_ordering(γs::Vector) -> nothing

Check that decay rates are in descending order (numerically required for parameter packing).

Throws AssertionError if violated.
"""
function validate_decay_ordering(γs::Vector)
    if !issorted(γs, rev=true)
        throw(AssertionError("Decay rates γs must be sorted in descending order: $(γs)"))
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Utilities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_memory_channels(L::Int; γ_range=(1e-2, 1), w_init=nothing) -> (γs, ws)

Convenience function to initialize memory channels for experiments.

Arguments:
- L: number of memory channels
- γ_range: tuple (γ_min, γ_max) for log-spaced decay rates [default: (1e-2, 1)]
- w_init: initial weights [default: uniform 1/L]

Returns:
- (γs, ws): initialized decay rates and weights

Example:
    γs, ws = create_memory_channels(3)  # γ ≈ [1.0, 0.1, 0.01], w ≈ [0.333, 0.333, 0.333]
"""
function create_memory_channels(L::Int; γ_range=(1e-2, 1), w_init=nothing)
    γ_min, γ_max = γ_range
    γs = exp10.(range(log10(γ_max), log10(γ_min), length=L))
    ws = isnothing(w_init) ? fill(1.0 / L, L) : copy(w_init)
    return γs, ws
end

end  # module GFE
