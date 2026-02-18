### A Pluto.jl notebook ###
# v0.20.21
# ==========================================
# Example 03: Gray-Scott Pattern Formation
# ==========================================
# Goal: Baseline Gray-Scott + Aberconics memory coupling comparisons.
# Runtime: High for full sweeps (minutes to hours depending on N and sweep size).
# Outputs: `results/gray_scott/` HTML pattern/sweep artifacts.
# Status: Included with performance caveat (architecture validated, heavy runtime).
# Note: Prefer coarse-to-fine sweeps to keep turnaround practical.

using Markdown
using InteractiveUtils

# ╔═╡ 6062b38e-068e-11f1-9f38-e378102b449a
### Gray-Scott Baseline (Production Version)
begin
    using DifferentialEquations, PlotlyLight, LinearAlgebra, FFTW, Statistics, Random
    
    # ========== CONFIGURATION ==========
    struct GSConfig
        N::Int              # Grid size (N×N)
        Lx::Float64         # Domain size
        Du::Float64         # Diffusion coefficient U
        Dv::Float64         # Diffusion coefficient V
        F::Float64          # Feed rate
        k::Float64          # Kill rate
        seed_radius::Int    # Initial perturbation radius
        tspan::Tuple{Float64,Float64}
    end
    
    # Classic Gray-Scott parameters for patterns
    function pattern_params(pattern_name::Symbol)
        if pattern_name == :spots
            return (Du=0.16, Dv=0.08, F=0.06, k=0.062)  # Classic spots
        elseif pattern_name == :stripes
            return (Du=0.14, Dv=0.06, F=0.035, k=0.065)  # Stripes
        elseif pattern_name == :waves
            return (Du=0.19, Dv=0.05, F=0.025, k=0.06)   # Traveling waves
        else
            return (Du=0.16, Dv=0.08, F=0.06, k=0.062)
        end
    end
    
    # ========== SPATIAL OPERATORS ==========
    struct SpatialOperators
        N::Int
        dx::Float64
        kx::Array{Float64,2}
        ky::Array{Float64,2}
        k2::Array{Float64,2}  # |k|² for spectral methods
    end
    
    function SpatialOperators(N, Lx=1.0)
        dx = Lx / N
        
        # Wavenumbers for spectral differentiation
        kx = fftfreq(N, 2π/dx)
        ky = fftfreq(N, 2π/dx)
        kx_grid = [kx[i] for i=1:N, j=1:N]
        ky_grid = [ky[j] for i=1:N, j=1:N]
        k2 = kx_grid.^2 + ky_grid.^2
        
        return SpatialOperators(N, dx, kx_grid, ky_grid, k2)
    end
    
    # Spectral Laplacian
    function spectral_laplacian(ops::SpatialOperators, u)
        u_hat = fft(u)
        lapl_hat = -ops.k2 .* u_hat
        return real(ifft(lapl_hat))
    end
    
    # Finite difference Laplacian (simpler, for reference)
    function fd_laplacian(ops::SpatialOperators, u)
        N = ops.N
        dx2 = ops.dx^2
        lapl = similar(u)
        
        for i=1:N, j=1:N
            ip1 = i == N ? 1 : i+1
            im1 = i == 1 ? N : i-1
            jp1 = j == N ? 1 : j+1
            jm1 = j == 1 ? N : j-1
            
            lapl[i,j] = (u[ip1,j] + u[im1,j] + u[i,jp1] + u[i,jm1] - 4*u[i,j]) / dx2
        end
        return lapl
    end

    function compute_laplacian(ops::SpatialOperators, u, method::Symbol)
        if method == :fd
            return fd_laplacian(ops, u)
        else
            return spectral_laplacian(ops, u)
        end
    end
    
    # ========== INITIAL CONDITIONS ==========
    function initial_conditions(config::GSConfig)
        N = config.N
        u0 = ones(N, N)
        v0 = zeros(N, N)
        
        # Add perturbation in center
        c = N ÷ 2
        r = config.seed_radius
        
        for i=max(1, c-r):min(N, c+r)
            for j=max(1, c-r):min(N, c+r)
                if (i-c)^2 + (j-c)^2 ≤ r^2
                    u0[i,j] = 0.5 + 0.1*randn()
                    v0[i,j] = 0.25 + 0.05*randn()
                end
            end
        end
        
        return u0, v0
    end
    
    # ========== GRAY-SCOTT DYNAMICS ==========
    function gray_scott_rhs!(du, u, p, t, laplacian::Symbol)
        config, ops = p
        N = config.N
        
        # Reshape state
        U = reshape(@view(u[1:N^2]), N, N)
        V = reshape(@view(u[N^2+1:2N^2]), N, N)
        
        # Laplacians (choose method)
        lap_U = compute_laplacian(ops, U, laplacian)
        lap_V = compute_laplacian(ops, V, laplacian)
        
        # Gray-Scott reactions
        UV2 = U .* V .^ 2
        
        # Dynamics
        dU = config.Du .* lap_U .- UV2 .+ config.F .* (1 .- U)
        dV = config.Dv .* lap_V .+ UV2 .- (config.F + config.k) .* V
        
        # Store derivatives
        du[1:N^2] .= vec(dU)
        du[N^2+1:2N^2] .= vec(dV)
    end
    
    # ========== SIMULATION RUNNER ==========
    function run_gray_scott(pattern=:spots; N=128, tspan=(0.0, 5000.0), saveat=100.0, laplacian=:spectral, progress=false, fast=false, progress_interval=50.0, alg=Tsit5())
        params = pattern_params(pattern)
        config = GSConfig(N, 1.0, params.Du, params.Dv, params.F, params.k, 5, tspan)
        ops = SpatialOperators(N)
        
        u0, v0 = initial_conditions(config)
        u_init = vcat(vec(u0), vec(v0))
        
        rhs = (du,u,p,t) -> gray_scott_rhs!(du, u, p, t, laplacian)
        prob = ODEProblem(rhs, u_init, tspan, (config, ops))
        cb = progress ? make_progress_callback("GS", N, progress_interval) : nothing
        if progress
            println("  Running GS: N=$(N), tspan=$(tspan), saveat=$(saveat), laplacian=$(laplacian), fast=$(fast)")
        end
        if fast
            sol = solve(prob, alg, saveat=saveat, adaptive=true, dt=0.5, save_everystep=false, maxiters=1_000_000, callback=cb)
        else
            sol = solve(prob, alg, saveat=saveat, reltol=1e-6, abstol=1e-8, callback=cb)
        end
        
        return sol, config, ops
    end

    function run_gray_scott_custom(; N=128, Du=0.16, Dv=0.08, F=0.06, k=0.062, seed_radius=5, tspan=(0.0, 5000.0), saveat=100.0, laplacian=:spectral, progress=false, fast=false, progress_interval=50.0, alg=Tsit5())
        config = GSConfig(N, 1.0, Du, Dv, F, k, seed_radius, tspan)
        ops = SpatialOperators(N)

        u0, v0 = initial_conditions(config)
        u_init = vcat(vec(u0), vec(v0))

        rhs = (du,u,p,t) -> gray_scott_rhs!(du, u, p, t, laplacian)
        prob = ODEProblem(rhs, u_init, tspan, (config, ops))
        cb = progress ? make_progress_callback("GS-custom", N, progress_interval) : nothing
        if progress
            println("  Running GS-custom: N=$(N), F=$(F), k=$(k), tspan=$(tspan), saveat=$(saveat), laplacian=$(laplacian), fast=$(fast)")
        end
        if fast
            sol = solve(prob, alg, saveat=saveat, adaptive=true, dt=0.5, save_everystep=false, maxiters=1_000_000, callback=cb)
        else
            sol = solve(prob, alg, saveat=saveat, reltol=1e-6, abstol=1e-8, callback=cb)
        end

        return sol, config, ops
    end

    # ========== OPTIMIZED SOLVERS FOR FAST EVALUATION ==========
    # Direct Rodas4P solver for coarse stage (no fallback loop, fails gracefully)
    function run_gray_scott_direct(; N=128, Du=0.16, Dv=0.08, F=0.06, k=0.062, seed_radius=5, tspan=(0.0, 5000.0), saveat=5000.0, laplacian=:fd)
        try
            config = GSConfig(N, 1.0, Du, Dv, F, k, seed_radius, tspan)
            ops = SpatialOperators(N)
            u0, v0 = initial_conditions(config)
            u_init = vcat(vec(u0), vec(v0))
            
            rhs = (du,u,p,t) -> gray_scott_rhs!(du, u, p, t, laplacian)
            prob = ODEProblem(rhs, u_init, tspan, (config, ops))
            
            # Direct Rodas4P with loose tolerances for coarse stage speed
            sol = solve(prob, Rodas4P(); 
                       saveat=saveat, 
                       reltol=1e-3,           # Loose for coarse screening
                       abstol=1e-5,
                       save_everystep=false,
                       maxiters=300_000)
            
            if SciMLBase.successful_retcode(sol.retcode)
                return sol, config, ops
            else
                return nothing, nothing, nothing
            end
        catch
            return nothing, nothing, nothing
        end
    end

    # ========== OPTIMIZED SWEEP WITH THREADING & BUDGET CONTROL ==========
    function sweep_fk_baseline_fast(; N=32, Du=0.16, Dv=0.08, F_vals=[0.022, 0.026, 0.03, 0.034, 0.038, 0.042], k_vals=[0.048, 0.052, 0.056, 0.06, 0.064], seed_radius=6, tspan=(0.0, 2000.0), saveat=2000.0, laplacian=:fd, var_threshold=1e-8, amp_threshold=1e-4, seed=1234, top_n=10, two_stage=true, coarse_tspan=(0.0, 800.0), coarse_saveat=800.0, coarse_var_threshold=5e-8, coarse_amp_threshold=5e-4, min_refine=8, use_threading=true, eval_budget_coarse=nothing, eval_budget_refine=nothing, progress=true)
        println("=== Baseline (F,k) sweep [OPTIMIZED] ===")
        println("  N=$(N), laplacian=$(laplacian), two_stage=$(two_stage), threading=$(use_threading)")
        println("  Nthreads=$(Threads.nthreads())")
        
        combos = [(F=F, k=k) for F in F_vals for k in k_vals]
        eval_counter_coarse = Ref(0)
        eval_counter_refine = Ref(0)

        # Shared function for fast evaluation
        function eval_case_fast(F::Float64, k::Float64, run_tspan::Tuple, run_saveat::Float64, case_seed::Int, use_fast::Bool=true)::NamedTuple
            try
                Random.seed!(case_seed)
                sol = refine = config = ops = nothing
                
                if use_fast
                    # Coarse stage: use fast direct solver
                    sol, config, ops = run_gray_scott_direct(
                        N=N, Du=Du, Dv=Dv, F=F, k=k, seed_radius=seed_radius,
                        tspan=run_tspan, saveat=run_saveat, laplacian=laplacian
                    )
                else
                    # Refine stage: use robust solver
                    sol, config, ops = run_gray_scott_custom(
                        N=N, Du=Du, Dv=Dv, F=F, k=k, seed_radius=seed_radius,
                        tspan=run_tspan, saveat=run_saveat, laplacian=laplacian,
                        progress=false, fast=false, alg=Tsit5()
                    )
                end
                
                if sol === nothing
                    # Solver failed; return penalty values
                    return (varU=NaN, ampU=NaN, meanU=NaN, meanV=NaN, success=false)
                end
                
                U_final = reshape(@view(sol[end][1:N^2]), N, N)
                V_final = reshape(@view(sol[end][N^2+1:2N^2]), N, N)
                varU = var(U_final)
                ampU = maximum(U_final) - minimum(U_final)
                meanU = mean(U_final)
                meanV = mean(V_final)
                return (varU=varU, ampU=ampU, meanU=meanU, meanV=meanV, success=true)
            catch err
                # Any error → penalty values
                return (varU=NaN, ampU=NaN, meanU=NaN, meanV=NaN, success=false)
            end
        end

        if two_stage
            println("  Stage-1 coarse: tspan=$(coarse_tspan), var_threshold=$(coarse_var_threshold), amp_threshold=$(coarse_amp_threshold)")
            coarse_results = NamedTuple[]
            coarse_total = length(combos)
            eval_budget_coarse_actual = eval_budget_coarse === nothing ? coarse_total : eval_budget_coarse
            coarse_start = time()
            
            # THREADING: Parallel evaluation in coarse stage
            if use_threading
                println("  [coarse] Starting $(coarse_total) evaluations using $(Threads.nthreads()) threads")
                coarse_results_tmp = Vector{NamedTuple}(undef, min(eval_budget_coarse_actual, coarse_total))
                coarse_times = Vector{Float64}(undef, min(eval_budget_coarse_actual, coarse_total))
                
                for batch_i in 1:min(eval_budget_coarse_actual, coarse_total)
                    c = combos[batch_i]
                    t_case = time()
                    println("    [T$(batch_i)] Starting F=$(c.F), k=$(c.k)...")
                    flush(stdout)
                    stats = eval_case_fast(c.F, c.k, coarse_tspan, coarse_saveat, seed + batch_i, true)
                    case_time = time() - t_case
                    coarse_times[batch_i] = case_time
                    label = (stats.varU > coarse_var_threshold && stats.ampU > coarse_amp_threshold && stats.success) ? "pattern-candidate" : "uniform/noise"
                    coarse_results_tmp[batch_i] = (F=c.F, k=c.k, stats..., label=label)
                    elapsed = time() - coarse_start
                    println("    [T$(batch_i)] Done: F=$(c.F), k=$(c.k), var=$(isnan(stats.varU) ? "NaN" : round(stats.varU, digits=3)), class=$(label), case_time=$(round(case_time, digits=2))s, elapsed=$(round(elapsed, digits=1))s")
                    flush(stdout)
                end
                
                push!(coarse_results, coarse_results_tmp...)
            else
                # Serial evaluation (fallback)
                println("  [coarse] Using serial evaluation")
                for (i, c) in enumerate(combos)
                    if i > eval_budget_coarse_actual
                        println("  [coarse] Budget reached ($eval_budget_coarse_actual/$coarse_total)")
                        break
                    end
                    t_case = time()
                    println("    [coarse $i/$eval_budget_coarse_actual] Starting F=$(c.F), k=$(c.k)...")
                    flush(stdout)
                    stats = eval_case_fast(c.F, c.k, coarse_tspan, coarse_saveat, seed + i, true)
                    case_time = time() - t_case
                    coarse_label = (stats.varU > coarse_var_threshold && stats.ampU > coarse_amp_threshold && stats.success) ? "pattern-candidate" : "uniform/noise"
                    push!(coarse_results, (F=c.F, k=c.k, varU=stats.varU, ampU=stats.ampU, meanU=stats.meanU, meanV=stats.meanV, label=coarse_label, success=stats.success))
                    elapsed = time() - coarse_start
                    println("    [coarse $i/$eval_budget_coarse_actual] Done: F=$(c.F), k=$(c.k), var=$(isnan(stats.varU) ? "NaN" : round(stats.varU, digits=3)), class=$(coarse_label), case_time=$(round(case_time, digits=2))s, elapsed=$(round(elapsed, digits=1))s")
                    flush(stdout)
                end
            end
            
            coarse_elapsed = time() - coarse_start
            println("  Stage-1 completed in $(round(coarse_elapsed, digits=2))s ($(length(coarse_results)) cases evaluated)")

            coarse_sorted = sort(coarse_results, by=x -> (x.ampU, x.varU), rev=true)
            shortlist = filter(x -> x.varU > coarse_var_threshold && x.ampU > coarse_amp_threshold && x.success, coarse_sorted)
            if length(shortlist) < min_refine
                shortlist = coarse_sorted[1:min(min_refine, length(coarse_sorted))]
                println("  Stage-1 produced few strong candidates. Refining top $(length(shortlist)) by coarse rank.")
            else
                println("  Stage-1 candidates above thresholds: $(length(shortlist))")
            end

            println("  Stage-2 refine: tspan=$(tspan), var_threshold=$(var_threshold), amp_threshold=$(amp_threshold)")
            results = NamedTuple[]
            refine_total = min(length(shortlist), eval_budget_refine === nothing ? length(shortlist) : eval_budget_refine)
            refine_start = time()
            
            # Refine stage - serial for now to avoid complexity
            println("  [refine] Evaluating $(refine_total) shortlisted cases...")
            for (i, c) in enumerate(shortlist[1:refine_total])
                t_case = time()
                println("    [refine $i/$refine_total] Starting F=$(c.F), k=$(c.k)...")
                flush(stdout)
                stats = eval_case_fast(c.F, c.k, tspan, saveat, seed + 10_000 + i, false)
                case_time = time() - t_case
                label = (stats.varU > var_threshold && stats.ampU > amp_threshold && stats.success) ? "pattern-candidate" : "uniform/noise"
                push!(results, (F=c.F, k=c.k, varU=stats.varU, ampU=stats.ampU, meanU=stats.meanU, meanV=stats.meanV, label=label, success=stats.success))
                elapsed = time() - refine_start
                println("    [refine $i/$refine_total] Done: F=$(c.F), k=$(c.k), var=$(isnan(stats.varU) ? "NaN" : round(stats.varU, digits=4)), class=$(label), case_time=$(round(case_time, digits=2))s, elapsed=$(round(elapsed, digits=1))s")
                flush(stdout)
            end
            
            refine_elapsed = time() - refine_start
            println("  Stage-2 completed in $(round(refine_elapsed, digits=2))s ($(length(results)) cases refined)")

            results_sorted = sort(results, by=x -> x.varU, rev=true)
            strong = filter(x -> x.varU > var_threshold && x.ampU > amp_threshold && x.success, results_sorted)
            println("\nSweep Summary:")
            println("  Stage-1 (coarse): $(round(coarse_elapsed, digits=2))s")
            println("  Stage-2 (refine): $(round(refine_elapsed, digits=2))s")
            println("  Total time: $(round(coarse_elapsed + refine_elapsed, digits=2))s")
            println("  Nonzero-variance regimes found: $(length(strong)) / $(length(results_sorted))")
            if isempty(strong)
                println("  No regimes above threshold. Try larger tspan, stronger seed, or adjust ranges.")
            else
                println("  Top regimes by final var(U):")
                for (i, row) in enumerate(strong[1:min(top_n, length(strong))])
                    println("    #$(i): F=$(row.F), k=$(row.k), var(U)=$(row.varU), amp(U)=$(row.ampU)")
                end
            end
            return results_sorted, strong
        end

        # One-stage mode (not optimized in this version)
        println("  One-stage mode: tspan=$(tspan), var_threshold=$(var_threshold), amp_threshold=$(amp_threshold)")
        results = NamedTuple[]
        total = length(combos)
        t_start = time()
        for (i, c) in enumerate(combos)
            t_case = time()
            stats = eval_case_fast(c.F, c.k, tspan, saveat, seed + i, false)
            elapsed = time() - t_start
            eta = (elapsed / i) * (total - i)
            label = (stats.varU > var_threshold && stats.ampU > amp_threshold && stats.success) ? "pattern-candidate" : "uniform/noise"
            push!(results, (F=c.F, k=c.k, varU=stats.varU, ampU=stats.ampU, meanU=stats.meanU, meanV=stats.meanV, label=label, success=stats.success))
            if progress && i % max(1, total ÷ 5) == 0
                println("  [$i/$total] F=$(c.F), k=$(c.k), var=$(isnan(stats.varU) ? "NaN" : round(stats.varU, digits=4)), class=$(label), case_s=$(round(time() - t_case, digits=2)), eta_s=$(round(eta, digits=1))")
            end
            eval_counter_refine[] += 1
        end

        results_sorted = sort(results, by=x -> x.varU, rev=true)
        strong = filter(x -> x.varU > var_threshold && x.ampU > amp_threshold && x.success, results_sorted)
        println("Sweep done. Total evals: $(eval_counter_refine[])")
        println("  Nonzero-variance regimes: $(length(strong)) / $(length(results_sorted))")
        if isempty(strong)
            println("  No regimes above threshold. Try larger tspan, stronger seed, or adjust ranges.")
        else
            println("Top regimes by final var(U):")
            for (i, row) in enumerate(strong[1:min(top_n, length(strong))])
                println("  #$(i): F=$(row.F), k=$(row.k), var(U)=$(row.varU), amp(U)=$(row.ampU)")
            end
        end
        return results_sorted, strong
    end

    function sweep_fk_baseline(; N=32, Du=0.16, Dv=0.08, F_vals=[0.022, 0.026, 0.03, 0.034, 0.038, 0.042], k_vals=[0.048, 0.052, 0.056, 0.06, 0.064], seed_radius=6, tspan=(0.0, 2000.0), saveat=2000.0, laplacian=:fd, fast=true, alg=Tsit5(), var_threshold=1e-8, amp_threshold=1e-4, seed=1234, top_n=10, two_stage=true, coarse_tspan=(0.0, 800.0), coarse_saveat=800.0, coarse_var_threshold=5e-8, coarse_amp_threshold=5e-4, min_refine=8)
        println("=== Baseline (F,k) sweep ===")
        println("  N=$(N), laplacian=$(laplacian), fast=$(fast), two_stage=$(two_stage)")

        combos = [(F=F, k=k) for F in F_vals for k in k_vals]

        function eval_case(F, k, run_tspan, run_saveat, case_seed)
            Random.seed!(case_seed)
            sol, _, _ = run_gray_scott_custom(
                N=N, Du=Du, Dv=Dv, F=F, k=k, seed_radius=seed_radius,
                tspan=run_tspan, saveat=run_saveat, laplacian=laplacian,
                progress=false, fast=fast, alg=alg
            )
            U_final = reshape(@view(sol[end][1:N^2]), N, N)
            V_final = reshape(@view(sol[end][N^2+1:2N^2]), N, N)
            varU = var(U_final)
            ampU = maximum(U_final) - minimum(U_final)
            meanU = mean(U_final)
            meanV = mean(V_final)
            return (varU=varU, ampU=ampU, meanU=meanU, meanV=meanV)
        end

        if two_stage
            println("  Stage-1 coarse: tspan=$(coarse_tspan), var_threshold=$(coarse_var_threshold), amp_threshold=$(coarse_amp_threshold)")
            coarse_results = NamedTuple[]
            coarse_total = length(combos)
            coarse_start = time()
            for (i, c) in enumerate(combos)
                t_case = time()
                stats = eval_case(c.F, c.k, coarse_tspan, coarse_saveat, seed + i)
                coarse_label = (stats.varU > coarse_var_threshold && stats.ampU > coarse_amp_threshold) ? "pattern-candidate" : "uniform/noise"
                push!(coarse_results, (F=c.F, k=c.k, varU=stats.varU, ampU=stats.ampU, meanU=stats.meanU, meanV=stats.meanV, label=coarse_label))
                elapsed = time() - coarse_start
                eta = (elapsed / i) * (coarse_total - i)
                println("  [coarse $i/$coarse_total] F=$(c.F), k=$(c.k), var(U)=$(stats.varU), amp(U)=$(stats.ampU), class=$(coarse_label), case_s=$(round(time() - t_case, digits=2)), eta_s=$(round(eta, digits=1))")
            end

            coarse_sorted = sort(coarse_results, by=x -> (x.ampU, x.varU), rev=true)
            shortlist = filter(x -> x.varU > coarse_var_threshold && x.ampU > coarse_amp_threshold, coarse_sorted)
            if length(shortlist) < min_refine
                shortlist = coarse_sorted[1:min(min_refine, length(coarse_sorted))]
                println("  Stage-1 produced few strong candidates. Refining top $(length(shortlist)) by coarse rank.")
            else
                println("  Stage-1 candidates above thresholds: $(length(shortlist))")
            end

            println("  Stage-2 refine: tspan=$(tspan), var_threshold=$(var_threshold), amp_threshold=$(amp_threshold)")
            results = NamedTuple[]
            refine_total = length(shortlist)
            refine_start = time()
            for (i, c) in enumerate(shortlist)
                t_case = time()
                stats = eval_case(c.F, c.k, tspan, saveat, seed + 10_000 + i)
                label = (stats.varU > var_threshold && stats.ampU > amp_threshold) ? "pattern-candidate" : "uniform/noise"
                push!(results, (F=c.F, k=c.k, varU=stats.varU, ampU=stats.ampU, meanU=stats.meanU, meanV=stats.meanV, label=label))
                elapsed = time() - refine_start
                eta = (elapsed / i) * (refine_total - i)
                println("  [refine $i/$refine_total] F=$(c.F), k=$(c.k), var(U)=$(stats.varU), amp(U)=$(stats.ampU), mean(U)=$(round(stats.meanU, digits=6)), mean(V)=$(round(stats.meanV, digits=6)), class=$(label), case_s=$(round(time() - t_case, digits=2)), eta_s=$(round(eta, digits=1))")
            end

            results_sorted = sort(results, by=x -> x.varU, rev=true)
            strong = filter(x -> x.varU > var_threshold && x.ampU > amp_threshold, results_sorted)
            println("Sweep done. Nonzero-variance regimes: $(length(strong)) / $(length(results_sorted))")
            if isempty(strong)
                println("  No regimes above threshold. Try larger tspan, stronger seed, or adjust ranges.")
            else
                println("Top regimes by final var(U):")
                for (i, row) in enumerate(strong[1:min(top_n, length(strong))])
                    println("  #$(i): F=$(row.F), k=$(row.k), var(U)=$(row.varU), amp(U)=$(row.ampU), mean(U)=$(round(row.meanU, digits=6)), mean(V)=$(round(row.meanV, digits=6))")
                end
            end
            return results_sorted, strong
        end

        println("  One-stage mode: tspan=$(tspan), var_threshold=$(var_threshold), amp_threshold=$(amp_threshold)")
        results = NamedTuple[]
        total = length(combos)
        t_start = time()
        for (i, c) in enumerate(combos)
            t_case = time()
            stats = eval_case(c.F, c.k, tspan, saveat, seed + i)
            elapsed = time() - t_start
            eta = (elapsed / i) * (total - i)
            label = (stats.varU > var_threshold && stats.ampU > amp_threshold) ? "pattern-candidate" : "uniform/noise"
            push!(results, (F=c.F, k=c.k, varU=stats.varU, ampU=stats.ampU, meanU=stats.meanU, meanV=stats.meanV, label=label))
            println("  [$i/$total] F=$(c.F), k=$(c.k), var(U)=$(stats.varU), amp(U)=$(stats.ampU), mean(U)=$(round(stats.meanU, digits=6)), mean(V)=$(round(stats.meanV, digits=6)), class=$(label), case_s=$(round(time() - t_case, digits=2)), eta_s=$(round(eta, digits=1))")
        end

        results_sorted = sort(results, by=x -> x.varU, rev=true)
        strong = filter(x -> x.varU > var_threshold && x.ampU > amp_threshold, results_sorted)
        println("Sweep done. Nonzero-variance regimes: $(length(strong)) / $(length(results_sorted))")
        if isempty(strong)
            println("  No regimes above threshold. Try larger tspan, stronger seed, or adjust ranges.")
        else
            println("Top regimes by final var(U):")
            for (i, row) in enumerate(strong[1:min(top_n, length(strong))])
                println("  #$(i): F=$(row.F), k=$(row.k), var(U)=$(row.varU), amp(U)=$(row.ampU), mean(U)=$(round(row.meanU, digits=6)), mean(V)=$(round(row.meanV, digits=6))")
            end
        end
        return results_sorted, strong
    end
    
    # ========== VISUALIZATION ==========
    function pl_heatmap(z; title="", colorscale="Viridis")
        plt = PlotlyLight.Plot()
        trace = PlotlyLight.Config(
            z = z,
            type = "heatmap",
            colorscale = colorscale,
            showscale = false
        )
        push!(plt.data, trace)
        plt.layout.title = title
        plt.layout.xaxis.showgrid = false
        plt.layout.yaxis.showgrid = false
        plt.layout.xaxis.zeroline = false
        plt.layout.yaxis.zeroline = false
        return plt
    end
    
    function pl_lines(x, ys, labels; title="", xaxis_title="", yaxis_title="")
        plt = PlotlyLight.Plot()
        for (i, y) in enumerate(ys)
            trace = PlotlyLight.Config(
                x = x,
                y = y,
                type = "scatter",
                mode = "lines",
                name = labels[i]
            )
            push!(plt.data, trace)
        end
        plt.layout.title = title
        plt.layout.xaxis.title = xaxis_title
        plt.layout.yaxis.title = yaxis_title
        return plt
    end

    function ensure_plot_dir(dir::AbstractString)
        mkpath(dir)
        return dir
    end

    function save_plot_html(p, dir::AbstractString, name::AbstractString)
        PlotlyLight.preset.source.standalone!()
        path = joinpath(dir, string(name, ".html"))
        PlotlyLight.save(p, path)
        return path
    end

    function save_plot_series(plots, dir::AbstractString, prefix::AbstractString)
        paths = String[]
        for (i, p) in enumerate(plots)
            name = string(prefix, "_", lpad(string(i), 2, '0'))
            push!(paths, save_plot_html(p, dir, name))
        end
        return paths
    end

    function write_plot_index(dir::AbstractString, name::AbstractString, files::Vector{String}; title::AbstractString="Plot Index")
        path = joinpath(dir, string(name, ".html"))
        open(path, "w") do io
            write(io, "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>")
            write(io, title)
            write(io, "</title></head><body>\n<h1>")
            write(io, title)
            write(io, "</h1>\n<ul>\n")
            for f in files
                fname = basename(f)
                write(io, "<li><a href=\"")
                write(io, fname)
                write(io, "\">")
                write(io, fname)
                write(io, "</a></li>\n")
            end
            write(io, "</ul>\n</body></html>\n")
        end
        return path
    end

    function make_progress_callback(label, N, interval)
        return PeriodicCallback(integrator -> begin
            U = @view integrator.u[1:N^2]
            V = @view integrator.u[N^2+1:2N^2]
            println("  [$label] t=$(round(integrator.t, digits=1)) mean(U)=$(round(mean(U), digits=4)) var(U)=$(round(var(U), digits=6)) mean(V)=$(round(mean(V), digits=5))")
        end, interval)
    end
    
    function plot_pattern_evolution(sol, config; n_frames=5)
        N = config.N
        times = range(sol.t[1], sol.t[end], length=n_frames+1)
        
        plots = []
        for (i, t) in enumerate(times[2:end])
            state = sol(t)
            U = reshape(@view(state[1:N^2]), N, N)

            p = pl_heatmap(U; title="t=$(round(t, digits=1))", colorscale="Viridis")
            push!(plots, p)
        end
        
        return plots
    end
    
    function plot_pattern_spectrum(sol, config)
        N = config.N
        times = [sol.t[1], sol.t[end]÷2, sol.t[end]]
        
        plots = []
        for (i, t) in enumerate(times)
            state = sol(t)
            U = reshape(@view(state[1:N^2]), N, N)
            
            # 2D FFT
            U_hat = fft(U)
            power = log10.(abs.(fftshift(U_hat)) .+ 1e-10)

            p = pl_heatmap(power; title="Spectrum t=$(round(t, digits=1))", colorscale="Inferno")
            push!(plots, p)
        end
        
        return plots
    end
    
    # ========== DIAGNOSTICS ==========
    function compute_diagnostics(sol, config)
        N = config.N
        diagnostics = Dict()
        
        # Energy over time
        energies = Float64[]
        for t in sol.t
            state = sol(t)
            U = reshape(@view(state[1:N^2]), N, N)
            V = reshape(@view(state[N^2+1:end]), N, N)
            
            E = 0.5 * (sum(U.^2) + sum(V.^2))
            push!(energies, E)
        end
        diagnostics[:energy] = energies
        
        # Pattern statistics at final time
        U_final = reshape(@view(sol[end][1:N^2]), N, N)
        diagnostics[:final_mean] = mean(U_final)
        diagnostics[:final_var] = var(U_final)
        diagnostics[:min_max] = (minimum(U_final), maximum(U_final))
        
        # Pattern scale (average blob size)
        U_hat = fft(U_final)
        k_max = argmax(abs.(U_hat))
        kx_max, ky_max = fftfreq(N, 2π/config.Lx)[k_max[1]], fftfreq(N, 2π/config.Lx)[k_max[2]]
        diagnostics[:dominant_wavelength] = 2π/sqrt(kx_max^2 + ky_max^2)
        
        return diagnostics
    end
end

# ╔═╡ c5822526-b6fa-4c71-bb6f-a8a6e00e6a68
### Aberconics Memory Extension (GFE-Constrained)
begin
    
    # ========== GFE MEMORY SYSTEM ==========
    struct GFEMemoryConfig
        L::Int               # Number of memory channels
        Φ_type::Symbol       # :classical, :fractional, :multiscale
        Φ_params::Dict{Symbol,Float64}
        w::Vector{Float64}   # SOE weights (positive)
        γ::Vector{Float64}   # Decay rates (positive)
        ℓ::Float64           # Leak (for stability: ℓ > 0.5*sum(w))
        L_op::Symbol         # :identity, :laplacian, :lowpass
    end
    
    function create_memory_config(pattern_timescales::Vector{Float64})
        # Create memory channels aligned with pattern timescales
        L = length(pattern_timescales)
        γ = 1 ./ pattern_timescales  # Convert to decay rates
        
        # Assign weights with slow-memory bias compensation
        w = zeros(L)
        for i in 1:L
            # Compensate for slow memory bias: give more weight to fast channels
            # Fast channels (large γ) get larger relative weights
            w[i] = γ[i] / sum(γ) * L
        end
        
        # Normalize and ensure stability
        w = w ./ maximum(w) * 0.1  # Keep weights small
        ℓ = 0.5 * sum(w) + 0.05    # Stability condition
        
        return GFEMemoryConfig(L, :multiscale, 
                               Dict(:α=>0.5, :β=>0.1), w, γ, ℓ, :laplacian)
    end
    
    # ========== MEMORY OPERATORS ==========
    function apply_L_op(u, ops::SpatialOperators, op_type::Symbol, laplacian::Symbol)
        if op_type == :identity
            return u
        elseif op_type == :laplacian
            return compute_laplacian(ops, u, laplacian)
        elseif op_type == :lowpass
            # Low-pass filter in spectral domain
            u_hat = fft(u)
            k_cutoff = 0.3 * maximum(ops.k2)
            filter = exp.(-ops.k2 ./ (2*k_cutoff^2))
            return real(ifft(u_hat .* filter))
        else
            return u
        end
    end
    
    # ========== ABERCONICS GRAY-SCOTT ==========
    struct AberconicsGrayScott
        base_config::GSConfig
        mem_config::GFEMemoryConfig
        ops::SpatialOperators
        laplacian::Symbol
    end
    
    function initial_conditions_ab(sys::AberconicsGrayScott)
        N = sys.base_config.N
        L = sys.mem_config.L
        
        u0, v0 = initial_conditions(sys.base_config)
        
        # Initialize memory channels to zero
        χ0 = [zeros(N, N) for _ in 1:L]
        
        # Flatten state: [U; V; χ1; χ2; ...; χL]
        state_vec = vcat(vec(u0), vec(v0))
        for χ in χ0
            append!(state_vec, vec(χ))
        end
        
        return state_vec
    end
    
    function aberconics_gs_rhs!(du, u, sys::AberconicsGrayScott, t)
        config = sys.base_config
        mem = sys.mem_config
        ops = sys.ops
        N = config.N
        L = mem.L
        
        # Extract fields
        idx_U = 1:N^2
        idx_V = N^2+1:2N^2
        idx_χ_start = 2N^2 + 1
        
        U = reshape(@view(u[idx_U]), N, N)
        V = reshape(@view(u[idx_V]), N, N)
        
        # Laplacians
        lap_U = compute_laplacian(ops, U, sys.laplacian)
        lap_V = compute_laplacian(ops, V, sys.laplacian)
        
        # Gray-Scott reactions
        UV2 = U .* V .^ 2
        
        # Base dynamics
        dU = config.Du .* lap_U .- UV2 .+ config.F .* (1 .- U)
        dV = config.Dv .* lap_V .+ UV2 .- (config.F + config.k) .* V
        
        # Memory feedback (sum over channels)
        mem_feedback = zeros(eltype(u), N, N)
        for l in 1:L
            χ_l = reshape(@view(u[idx_χ_start+(l-1)*N^2:idx_χ_start+l*N^2-1]), N, N)
            mem_feedback .+= mem.w[l] .* χ_l
        end
        
        # Add memory feedback to U dynamics
        dU .+= mem_feedback
        dU .-= mem.ℓ .* U  # Leak term for stability
        
        # Store U derivatives
        du[idx_U] .= vec(dU)
        du[idx_V] .= vec(dV)
        
        # Memory channel dynamics
        L_U = mem.L_op == :laplacian ? lap_U : apply_L_op(U, ops, mem.L_op, sys.laplacian)
        for l in 1:L
            χ_l = reshape(@view(u[idx_χ_start+(l-1)*N^2:idx_χ_start+l*N^2-1]), N, N)
            dχ = -mem.γ[l] .* χ_l .+ L_U
            du[idx_χ_start+(l-1)*N^2:idx_χ_start+l*N^2-1] .= vec(dχ)
        end
    end
    
    # ========== SIMULATION ==========
    function run_ab_gray_scott_custom(; N=64, L=4, Du=0.16, Dv=0.08, F=0.06, k=0.062, seed_radius=5, tspan=(0.0, 2000.0), saveat=50.0, laplacian=:spectral, progress=false, fast=false, progress_interval=50.0, alg=Tsit5())
        base_config = GSConfig(N, 1.0, Du, Dv, F, k, seed_radius, tspan)

        # Memory timescales aligned with pattern evolution
        base_times = [100.0, 500.0, 1000.0, 2000.0]
        if L <= length(base_times)
            pattern_times = base_times[1:L]
        else
            extra = [base_times[end] * 2.0^i for i in 1:(L - length(base_times))]
            pattern_times = vcat(base_times, extra)
        end
        mem_config = create_memory_config(pattern_times)
        
        ops = SpatialOperators(N)
        system = AberconicsGrayScott(base_config, mem_config, ops, laplacian)
        
        # Initial conditions
        u0 = initial_conditions_ab(system)
        
        # Create ODE problem
        prob = ODEProblem((du,u,p,t) -> aberconics_gs_rhs!(du, u, system, t), 
                         u0, tspan, system)
        
        # Solve
        cb = progress ? make_progress_callback("Aberconics", N, progress_interval) : nothing
        if progress
            println("  Running Aberconics GS: N=$(N), L=$(L), F=$(F), k=$(k), tspan=$(tspan), saveat=$(saveat), fast=$(fast)")
        end
        if fast
            sol = solve(prob, alg, saveat=saveat, adaptive=true, dt=0.5, save_everystep=false, maxiters=1_000_000, callback=cb)
        else
            sol = solve(prob, alg, saveat=saveat, reltol=1e-6, abstol=1e-8, callback=cb)
        end
        
        return sol, system
    end

    function run_ab_gray_scott(pattern=:spots; N=64, L=4, tspan=(0.0, 2000.0), saveat=50.0, laplacian=:spectral, progress=false, fast=false, progress_interval=50.0, alg=Tsit5())
        params = pattern_params(pattern)
        return run_ab_gray_scott_custom(
            N=N, L=L, Du=params.Du, Dv=params.Dv, F=params.F, k=params.k, seed_radius=5,
            tspan=tspan, saveat=saveat, laplacian=laplacian,
            progress=progress, fast=fast, progress_interval=progress_interval, alg=alg
        )
    end
    
    # ========== MEMORY DIAGNOSTICS ==========
    function analyze_memory_contributions(sol, system::AberconicsGrayScott)
        N = system.base_config.N
        L = system.mem_config.L
        mem = system.mem_config
        
        times = sol.t
        n_steps = length(times)
        
        # Memory channel contributions over time
        channel_contributions = zeros(n_steps, L)
        memory_energy = zeros(n_steps)
        
        for (step, t) in enumerate(times)
            state = sol(t)
            idx_χ_start = 2N^2 + 1
            
            for l in 1:L
                χ_l = reshape(@view(state[idx_χ_start+(l-1)*N^2:idx_χ_start+l*N^2-1]), N, N)
                channel_contributions[step, l] = sum(abs.(χ_l)) / N^2
            end
            
            # Memory energy (from GFE Proposition 5.1)
            E_mem = 0.0
            for l in 1:L
                χ_l = reshape(@view(state[idx_χ_start+(l-1)*N^2:idx_χ_start+l*N^2-1]), N, N)
                E_mem += mem.w[l]/(2*mem.γ[l]) * sum(χ_l.^2)
            end
            memory_energy[step] = E_mem
        end
        
        return channel_contributions, memory_energy
    end
    
    function plot_memory_dynamics(sol, system::AberconicsGrayScott)
        N = system.base_config.N
        L = system.mem_config.L
        
        # Get final state
        state_end = sol[end]
        
        # Plot U field
        p1 = pl_heatmap(reshape(@view(state_end[1:N^2]), N, N); title="U field (final)", colorscale="Viridis")
        
        # Plot memory channels
        memory_plots = []
        idx_χ_start = 2N^2 + 1
        
        for l in 1:min(L, 4)  # Show first 4 channels
            χ_l = reshape(@view(state_end[idx_χ_start+(l-1)*N^2:idx_χ_start+l*N^2-1]), N, N)
            p = pl_heatmap(χ_l; title="χ$(l) (τ=$(round(1/system.mem_config.γ[l], digits=0))s)", colorscale="Plasma")
            push!(memory_plots, p)
        end
        return (p1, memory_plots)
    end
end

# ╔═╡ b39d4bce-5be9-488e-9f9a-bbdb1cb176ff
### Run Comparative Experiment
begin
    sweep_N = 32
    sweep_tspan = (0.0, 2000.0)
    sweep_seed_radius = 6
    sweep_Du = 0.16
    sweep_Dv = 0.08

    # Find a baseline regime with nonzero final var(U) and amp(U), then use it for both runs.
    fk_results, fk_nonzero = sweep_fk_baseline_fast(
        N=sweep_N, Du=sweep_Du, Dv=sweep_Dv, seed_radius=sweep_seed_radius,
        tspan=sweep_tspan, saveat=sweep_tspan[2], laplacian=:fd,
        var_threshold=1e-8, amp_threshold=1e-4, two_stage=true, use_threading=false, progress=true,
        coarse_tspan=(0.0, 400.0), coarse_saveat=400.0, eval_budget_coarse=3, eval_budget_refine=2
    )
    if isempty(fk_nonzero)
        error("No valid (F,k) regime found in sweep. Adjust sweep ranges or thresholds.")
    end
    best_fk = fk_nonzero[1]
    selected_F = best_fk.F
    selected_k = best_fk.k
    compare_tspan = sweep_tspan
    compare_saveat = 200.0
    println("Selected sweep regime: F=$(selected_F), k=$(selected_k), var(U)=$(best_fk.varU), amp(U)=$(best_fk.ampU)")

    save_plots = true
    plots_dir = save_plots ? ensure_plot_dir(joinpath(pwd(), "plots_gray_scott")) : ""
    plot_files = String[]

    # Run baseline (no memory)
    t_total = time()
    println("=== Running Baseline Gray-Scott (No Memory) ===")
    t0 = time()
    sol_baseline, config_baseline = run_gray_scott_custom(
        N=sweep_N, Du=sweep_Du, Dv=sweep_Dv, F=selected_F, k=selected_k, seed_radius=sweep_seed_radius,
        tspan=compare_tspan, saveat=compare_saveat, laplacian=:fd,
        progress=true, fast=true, progress_interval=25.0, alg=Rodas5()
    )
    t_baseline = time() - t0
    println("Baseline runtime: $(round(t_baseline, digits=2))s")
    
    # Run with Aberconics memory
    println("=== Running Aberconics Gray-Scott (With Memory) ===")
    t0 = time()
    sol_ab, system_ab = run_ab_gray_scott_custom(
        N=sweep_N, L=4, Du=sweep_Du, Dv=sweep_Dv, F=selected_F, k=selected_k, seed_radius=sweep_seed_radius,
        tspan=compare_tspan, saveat=compare_saveat, laplacian=:fd,
        progress=true, fast=true, progress_interval=25.0, alg=Rodas5()
    )
    t_ab = time() - t0
    println("Aberconics runtime: $(round(t_ab, digits=2))s")
    
    # Compute diagnostics
    diag_baseline = compute_diagnostics(sol_baseline, config_baseline)
    println("\nBaseline Diagnostics:")
    println("  Final mean U: ", round(diag_baseline[:final_mean], digits=4))
    println("  Final variance U: ", round(diag_baseline[:final_var], digits=6))
    println("  Dominant wavelength: ", round(diag_baseline[:dominant_wavelength], digits=2))
    
    # Memory analysis
    channel_contribs, mem_energy = analyze_memory_contributions(sol_ab, system_ab)
    println("\nAberconics Memory Diagnostics:")
    println("  Memory channels: ", system_ab.mem_config.L)
    println("  Memory timescales (τ): ", round.(1 ./ system_ab.mem_config.γ, digits=0))
    println("  Memory weights: ", round.(system_ab.mem_config.w, digits=4))
    println("  Final memory energy: ", round(mem_energy[end], digits=4))
    println("  Channel contributions (final): ", round.(channel_contribs[end,:], digits=4))
    
    # Visual comparison
    p1_frames = plot_pattern_evolution(sol_baseline, config_baseline, n_frames=4)
    println("Baseline Gray-Scott (No Memory) frames:")
    for p in p1_frames
        display(p)
    end
    if save_plots
        append!(plot_files, save_plot_series(p1_frames, plots_dir, "baseline_frame"))
    end
    
    p2_u, p2_mem = plot_memory_dynamics(sol_ab, system_ab)
    display(p2_u)
    for p in p2_mem
        display(p)
    end
    if save_plots
        push!(plot_files, save_plot_html(p2_u, plots_dir, "abercs_u_final"))
        append!(plot_files, save_plot_series(p2_mem, plots_dir, "abercs_mem"))
    end
    
    # Plot energy comparison
    energy_baseline = diag_baseline[:energy]
    times_baseline = sol_baseline.t
    times_ab = sol_ab.t
    
    p_energy = pl_lines(
        times_baseline,
        [energy_baseline, mem_energy],
        ["Baseline (No Memory)", "Memory Energy"];
        title="Energy vs Time",
        xaxis_title="Time",
        yaxis_title="Energy"
    )
    
    # Plot channel contributions
    channel_labels = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"][1:system_ab.mem_config.L]
    channel_series = [channel_contribs[:, i] for i in 1:system_ab.mem_config.L]
    p_channels = pl_lines(
        sol_ab.t,
        channel_series,
        channel_labels;
        title="Memory Channel Contributions",
        xaxis_title="Time",
        yaxis_title="Average |χ|"
    )
    
    display(p_energy)
    if save_plots
        push!(plot_files, save_plot_html(p_energy, plots_dir, "energy_vs_time"))
    end
    display(p_channels)
    if save_plots
        push!(plot_files, save_plot_html(p_channels, plots_dir, "memory_channels"))
    end

    if save_plots
        index_path = write_plot_index(plots_dir, "index", plot_files; title="Gray-Scott Plots")
        println("Plots saved to: $(plots_dir) (index: $(index_path))")
    end
    
    println("Total runtime: $(round(time() - t_total, digits=2))s")
    (sol_baseline, config_baseline, sol_ab, system_ab)
end

# ╔═╡ 21d97a9e-bbcd-4901-b700-657076443e90
### Experiment Suite: Memory Effects on Pattern Formation
begin
    function run_experiment_suite(N=64; tspan=(0.0, 3000.0), save_plots=false, plots_dir=joinpath(pwd(), "plots_gray_scott_suite"))
        patterns = [:spots, :stripes, :waves]
        results = Dict()
        if save_plots
            ensure_plot_dir(plots_dir)
        end
        
        for pattern in patterns
            println("\n=== Testing Pattern: $pattern ===")
            
            # Baseline
            sol_baseline, config_baseline = run_gray_scott(pattern, N=N, tspan=tspan, saveat=200.0)
            diag_baseline = compute_diagnostics(sol_baseline, config_baseline)
            
            # With memory (choose timescales based on pattern)
            if pattern == :spots
                pattern_times = [200.0, 500.0, 1000.0, 1500.0]
            elseif pattern == :stripes
                pattern_times = [100.0, 300.0, 800.0, 1200.0]
            else  # waves
                pattern_times = [50.0, 150.0, 400.0, 800.0]
            end
            
            sol_ab, system_ab = run_ab_gray_scott(pattern, N=N, L=4, tspan=tspan, saveat=200.0)
            channel_contribs, mem_energy = analyze_memory_contributions(sol_ab, system_ab)
            
            # Store results
            results[pattern] = Dict(
                :baseline => (sol=sol_baseline, config=config_baseline, diag=diag_baseline),
                :abercs => (sol=sol_ab, system=system_ab, 
                           channel_contribs=channel_contribs, mem_energy=mem_energy)
            )
            
            # Quick visualization
            p1_frames = plot_pattern_evolution(sol_baseline, config_baseline, n_frames=3)
            println("$pattern Baseline frames:")
            for p in p1_frames
                display(p)
            end
            if save_plots
                pattern_dir = ensure_plot_dir(joinpath(plots_dir, string(pattern)))
                files = save_plot_series(p1_frames, pattern_dir, "baseline_frame")
                write_plot_index(pattern_dir, "index", files; title=string("Gray-Scott ", pattern, " Baseline"))
            end
            
            U_ab = reshape(@view(sol_ab[end][1:N^2]), N, N)
            p2 = pl_heatmap(U_ab; title="$pattern with Memory", colorscale="Viridis")
            display(p2)
            if save_plots
                pattern_dir = ensure_plot_dir(joinpath(plots_dir, string(pattern)))
                save_plot_html(p2, pattern_dir, "memory_final")
            end
        end
        
        return results
    end
    
    # Run the experiment suite (toggle on when needed)
    # results = run_experiment_suite(64, tspan=(0.0, 2000.0))
end

# ╔═╡ 2ea3e736-86c7-4e33-a7d9-03d276377eb9
##Key Insights & What to Measure

##Based on your previous findings, track these metrics:

begin
	function fftshift(A)
	    n1, n2 = size(A)
	    return circshift(A, (n1 ÷ 2, n2 ÷ 2))
	end
	
	function autocorrelate_2d(U)
	    U0 = U .- mean(U)
	    F = fft(U0)
	    ac = real(ifft(abs.(F).^2))
	    return fftshift(ac)
	end
	
	function radial_profile(Z)
	    n1, n2 = size(Z)
	    c1 = (n1 + 1) / 2
	    c2 = (n2 + 1) / 2
	    rmax = floor(Int, sqrt((n1/2)^2 + (n2/2)^2))
	    sums = zeros(rmax + 1)
	    counts = zeros(Int, rmax + 1)
	    for i in 1:n1, j in 1:n2
	        r = round(Int, sqrt((i - c1)^2 + (j - c2)^2))
	        if r <= rmax
	            sums[r + 1] += Z[i, j]
	            counts[r + 1] += 1
	        end
	    end
	    profile = sums ./ max.(counts, 1)
	    return collect(0:rmax), profile
	end
	
	function compute_correlation_length(autocorr)
	    r, prof = radial_profile(autocorr)
	    if isempty(prof)
	        return 0.0
	    end
	    thresh = prof[1] / exp(1)
	    idx = findfirst(x -> x <= thresh, prof)
	    return idx === nothing ? r[end] : r[idx]
	end
	
	function compute_angular_variance(power_spectrum)
	    r, prof = radial_profile(power_spectrum)
	    if length(prof) < 2
	        return 0.0
	    end
	    r_peak = r[argmax(prof[2:end]) + 1]
	    n1, n2 = size(power_spectrum)
	    c1 = (n1 + 1) / 2
	    c2 = (n2 + 1) / 2
	    ring = Float64[]
	    for i in 1:n1, j in 1:n2
	        rr = round(Int, sqrt((i - c1)^2 + (j - c2)^2))
	        if abs(rr - r_peak) <= 1
	            push!(ring, power_spectrum[i, j])
	        end
	    end
	    return isempty(ring) ? 0.0 : var(ring)
	end
	
	function find_blobs(U; threshold=0.5)
	    N, M = size(U)
	    mask = U .> threshold
	    visited = falses(N, M)
	    areas = Int[]
	    for i in 1:N, j in 1:M
	        if mask[i, j] && !visited[i, j]
	            stack = [(i, j)]
	            visited[i, j] = true
	            area = 0
	            while !isempty(stack)
	                (x, y) = pop!(stack)
	                area += 1
	                for (dx, dy) in ((1,0),(-1,0),(0,1),(0,-1))
	                    nx, ny = x + dx, y + dy
	                    if 1 <= nx <= N && 1 <= ny <= M && mask[nx, ny] && !visited[nx, ny]
	                        visited[nx, ny] = true
	                        push!(stack, (nx, ny))
	                    end
	                end
	            end
	            push!(areas, area)
	        end
	    end
	    return areas
	end
	
	function compute_pattern_metrics(U)
	    autocorr = autocorrelate_2d(U)
	    correlation_length = compute_correlation_length(autocorr)
	    
	    U_hat = fft(U)
	    power_spectrum = abs.(fftshift(U_hat))
	    angular_variance = compute_angular_variance(power_spectrum)
	    
	    areas = find_blobs(U; threshold=0.5)
	    blob_count = length(areas)
	    avg_blob_size = blob_count == 0 ? 0.0 : mean(areas)
	    
	    return (correlation_length, angular_variance, blob_count, avg_blob_size)
	end
	
	function analyze_channel_specialization(channel_contribs, pattern_metrics)
	    # pattern_metrics: matrix of size (T x M) or vector (T)
	    if ndims(pattern_metrics) == 1
	        pattern_metrics = reshape(pattern_metrics, :, 1)
	    end
	    L = size(channel_contribs, 2)
	    M = size(pattern_metrics, 2)
	    correlations = zeros(L, M)
	    for l in 1:L, m in 1:M
	        correlations[l, m] = cor(channel_contribs[:, l], pattern_metrics[:, m])
	    end
	    return correlations
	end
	
	function estimate_pattern_timescales(series; dt=1.0)
	    if length(series) < 2
	        return [dt]
	    end
	    s = series .- mean(series)
	    F = fft(s)
	    ac = real(ifft(abs.(F).^2))
	    ac = ac ./ ac[1]
	    idx = findfirst(x -> x <= exp(-1), ac[2:end])
	    τ = idx === nothing ? length(series) * dt : (idx + 1) * dt
	    return [τ]
	end
	
	function test_timescale_matching(system, pattern_series)
	    memory_timescales = 1 ./ system.mem_config.γ
	    pattern_timescales = estimate_pattern_timescales(pattern_series)
	    alignment = []
	    for mem_τ in memory_timescales
	        best_match = argmin(abs.(pattern_timescales .- mem_τ))
	        alignment_score = exp(-abs(pattern_timescales[best_match] - mem_τ)/mem_τ)
	        push!(alignment, alignment_score)
	    end
	    return mean(alignment)
	end
end

	

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "71853c6197a6a7f222db0f1978c7cb232b87c5ee"

[deps]
"""

# ╔═╡ Cell order:
# ╠═6062b38e-068e-11f1-9f38-e378102b449a
# ╠═c5822526-b6fa-4c71-bb6f-a8a6e00e6a68
# ╠═b39d4bce-5be9-488e-9f9a-bbdb1cb176ff
# ╠═21d97a9e-bbcd-4901-b700-657076443e90
# ╠═2ea3e736-86c7-4e33-a7d9-03d276377eb9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
