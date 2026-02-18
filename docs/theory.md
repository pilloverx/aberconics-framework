# Theory Notes & Framework Overview

This file captures the key theoretical context for the Aberconics framework and maps it to implementation in the `code/julia/` directory.

## The GFE Engine

**Geometric Fourier Extension (GFE)** is the spectral mathematical engine underlying all experiments. It provides:

- **Exponential basis functions** for smooth kernel approximation
- **Non-negative least squares (NNLS)** fitting of observation data to SOE representations
- **Spectral memory metrics** (capacity, entropy, effective dimension)
- **Parameter optimization** infrastructure via packing/unpacking

The **GFE.jl module** (`code/julia/src/GFE.jl`) centralizes these operations for reuse across all experiments.

### GFE in Action

```julia
using Aberconics.GFE

# 1. Create exponential basis (e.g., for kernel fitting)
γ = exp10.(range(-2, 1, length=15))  # 15 decay rates
A = design_matrix(γ, t_data)         # Design matrix

# 2. Fit kernel to observed ACF
γ_fit, w_fit, fit = fit_soe_kernel(t_acf, acf_observed, n_basis=15)

# 3. Analyze memory structure
su = spectral_units(w_fit, γ_fit)
println("Memory capacity: $(su.Mcap) seconds")
println("Effective dimension: $(su.Deff) modes")

# 4. Optimize memory parameters (for chaos suppression, etc.)
θ = pack_memory_params(γ_initial, w_initial)
# ... optimize θ ...
γ_opt, w_opt = unpack_memory_params(θ_result)
```

## Primary Paper Priority

### Aberconics V2 (`papers/Aberconics_Framework_V2.0.pdf`, `papers/Aberconics_V2_0-1.pdf`)

This is the main reference for the current code direction. The most important items to carry into implementation and interpretation are:

- **AE-3 structure**: reactive state + memory kernel + internal model + optional environment coupling.
- **Non-Markovian dynamics as first-class**: memory is explicit through convolution kernels, not hidden in a black-box latent state.
- **SOE embedding**: kernels represented as sums of exponentials, enabling local ODE realizations.
- **Interpretability requirement**: learned timescales and gains should map to meaningful temporal mechanisms.
- **Tier framing**: stable/passive memory constructions are preferred for reliability before moving to more expressive nonlinear updates.
- **Validation philosophy**: test whether memory changes dynamical regime (not only point prediction error).

## How Current Experiments Map to V2

Each experiment validates a key claim of the Aberconics framework using GFE as the spectral engine.

### 1. **OU_Noise Demo** (`code/julia/examples/01_ou_noise_demo.jl`)
- **GFE Usage**: `fit_soe_kernel()`, `spectral_units()`
- **Validates**: SOE can flexibly represent colored noise memory (no fixed kernel required)
- **Key Metrics**: L1 fit error < 0.005, D_eff ≈ 6.86, Mcap ≈ 100 seconds
- **Interpretation**: Confirms SOE kernel representation is expressive and accurate across timescale decades

### 2. **Lorenz Chaos Suppression** (`code/julia/examples/02_lorenz_chaos_suppression.jl`)
- **GFE Usage**: `pack_memory_params()`, `unpack_memory_params()` for optimization
- **Validates**: Optimized memory channels can shift dynamics (Lyapunov suppression by 25%)
- **Key Result**: 2/3 initial conditions → periodic orbits (regime change via memory)
- **Interpretation**: Memory is not just decorative; it reshapes basin structure and stability

### 3. **Gray-Scott Patterns** (`code/julia/examples/03_gray_scott_patterns.jl`)
- **GFE Usage**: Memory channel initialization, feedback coupling
- **Validates**: Framework scales to spatial PDEs (not just ODEs)
- **Current State**: Baseline patterns demonstrated; Aberconics integration pending
- **Caveat**: Full parameter sweep expensive (N=32 grid) — demonstrates scaling challenge

### 4. **Echo Task** (`code/julia/examples/04_echo_task_learning.jl`)
- **GFE Usage**: Initialization, tuning of memory channel parameters
- **Validates**: Hebbian-Volterra learning discovers echo behavior autonomously
- **Key Insight**: Adaptive learning rules (error-modulated) can tune memory without explicit supervision

## Secondary References

- `papers/Aberconic_1_0.pdf`: Earlier formulation history; shows evolution of framework
- `papers/Spectral_Memory_Units_Analysis.pdf`: Deep dive on memory capacity (Mcap), entropy (Hmem), effective dimension (Deff)
- `papers/geometric_fourier_extension_GFE_v1_0.pdf`: Pure spectral theory; mathematical foundations of exponential basis and GFE convergence

## GFE Module Architecture

The **GFE.jl** module (`code/julia/src/GFE.jl`) organizes the mathematical machinery into five sections:

| Section | Purpose | Key Functions |
|---------|---------|---|
| **1. Basis Functions** | Exponential bases for kernel approximation | `exponential_basis()`, `design_matrix()` |
| **2. NNLS Fitting** | Solve for kernel weights given measurements | `nnls_pg()`, `fit_soe_kernel()` |
| **3. Spectral Units** | Memory metrics and analysis | `spectral_units()`, `memory_capacity()`, `effective_dimension()` |
| **4. Parameter Packing** | Convert between (γ, w) and unconstrained θ for optimization | `pack_memory_params()`, `unpack_memory_params()` |
| **5. Utilities** | Convenience functions for initialization | `create_memory_channels()` |

### Why Modularize?

1. **Reusability**: All experiments call the same GFE functions (no duplicate code)
2. **Correctness**: Single tested implementation of NNLS, spectral metrics, packing logic
3. **Clarity**: Theory-to-code mapping is explicit (each function documents its math)
4. **Maintainability**: Bug fixes/improvements propagate to all experiments

## Repository Policy for Papers

- **PDFs in `papers/`**: Stored locally; kept outside tracked code by default (size/versioning reasons)
- **Distilled theory in Markdown**: This file + `code/julia/README.md` + example docstring comments
- **Explicit mappings**: Each experiment has inline comments linking to sections of papers where needed
- **Git LFS option** (future): If papers need full version control, use Git Large File Storage
