#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <iostream>

namespace {

abersoe::AberSOEModelSpec make_fast_level() {
    abersoe::AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.0, 0.25}, gfe::Vec{0.6, 0.4}};
    model.dynamics.dt = 0.02;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.9 * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [](double) { return gfe::State{0.0}; };
    return model;
}

abersoe::AberSOEModelSpec make_slow_level() {
    abersoe::AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{0.5, 0.1}, gfe::Vec{0.55, 0.45}};
    model.dynamics.dt = 0.02;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.15 * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [](double) { return gfe::State{0.05}; };
    return model;
}

abersoe::HierarchicalMINModel make_model(bool coupled) {
    abersoe::HierarchicalMINModel model;
    model.levels.push_back({"fast", make_fast_level(), {}});
    model.levels.push_back({"slow", make_slow_level(), {}});

    if (coupled) {
        abersoe::BottomUpCoupling bottom_up;
        bottom_up.source_level = 0;
        bottom_up.target_level = 1;
        bottom_up.forcing = [](const gfe::State& source_obs, double) {
            return gfe::State{0.25 * source_obs[0]};
        };
        model.bottom_up_couplings.push_back(bottom_up);

        abersoe::TopDownCoupling top_down;
        top_down.source_level = 1;
        top_down.target_level = 0;
        top_down.modulation = [](const gfe::State& source_obs) {
            const double s = source_obs[0];
            abersoe::PositiveKernelModulation out;
            out.alpha = gfe::Vec{0.25 * s, -0.15 * s};
            out.beta = gfe::Vec{0.1 + 0.35 * s, -1.4 - 0.2 * s};
            out.normalize_weights = true;
            return out;
        };
        model.top_down_couplings.push_back(top_down);
    }

    return model;
}

} // namespace

int main() {
    try {
        {
            abersoe::PositiveKernelModulation modulation;
            modulation.alpha = gfe::Vec{0.0, std::log(2.0)};
            modulation.beta = gfe::Vec{-0.3, 0.2};
            const auto kernel = abersoe::materialize_positive_kernel(modulation);
            if (kernel.gamma.size() != 2 || kernel.w.size() != 2) {
                std::cerr << "materialized kernel shape mismatch\n";
                return 1;
            }
            if (!(kernel.gamma[0] > kernel.gamma[1] && kernel.w[0] > 0.0 && kernel.w[1] > 0.0)) {
                std::cerr << "materialized kernel should be positive and sorted\n";
                return 1;
            }
            if (std::fabs(kernel.w[0] + kernel.w[1] - 1.0) > 1e-12) {
                std::cerr << "materialized kernel weights should normalize to 1\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalMINModel bad = make_model(false);
            bad.top_down_couplings.push_back(abersoe::TopDownCoupling{0, 0, {}, {}});
            std::string err;
            if (abersoe::validate_hierarchical_model(bad, &err)) {
                std::cerr << "invalid hierarchy should fail validation\n";
                return 1;
            }
        }

        const auto uncoupled_model = make_model(false);
        const auto coupled_model = make_model(true);

        std::vector<abersoe::AberSOEState> initial_states(2);
        initial_states[0].u = gfe::State{1.0};
        initial_states[0].chi = gfe::Vec{0.0, 0.0};
        initial_states[1].u = gfe::State{0.2};
        initial_states[1].chi = gfe::Vec{0.0, 0.0};

        const auto uncoupled_state0 = abersoe::initialize_hierarchy_state(uncoupled_model, initial_states);
        const auto coupled_state0 = abersoe::initialize_hierarchy_state(coupled_model, initial_states);

        abersoe::HierarchicalRuntimeConfig cfg;
        cfg.steps = 80;
        cfg.sample_stride = 10;
        cfg.strict_finite = true;

        const auto uncoupled_run = abersoe::run(uncoupled_model, uncoupled_state0, cfg);
        const auto coupled_run = abersoe::run(coupled_model, coupled_state0, cfg);

        if (uncoupled_run.diagnostics.steps_executed != cfg.steps ||
            coupled_run.diagnostics.steps_executed != cfg.steps) {
            std::cerr << "hierarchy step count mismatch\n";
            return 1;
        }
        if (!uncoupled_run.diagnostics.all_finite || !coupled_run.diagnostics.all_finite) {
            std::cerr << "hierarchy run should remain finite\n";
            return 1;
        }
        if (coupled_run.samples.empty() || coupled_run.diagnostics.per_level.size() != 2) {
            std::cerr << "hierarchy run should emit samples and per-level diagnostics\n";
            return 1;
        }

        const double uncoupled_slow = uncoupled_run.final_levels[1].state.u[0];
        const double coupled_slow = coupled_run.final_levels[1].state.u[0];
        if (std::fabs(coupled_slow - uncoupled_slow) < 1e-4) {
            std::cerr << "bottom-up coupling should change upper-level trajectory\n";
            return 1;
        }

        const auto& coupled_fast_kernel = coupled_run.final_levels[0].active_kernel;
        const auto& base_fast_kernel = coupled_model.levels[0].model.kernel;
        const bool kernel_changed =
            std::fabs(coupled_fast_kernel.w[0] - base_fast_kernel.w[0]) > 1e-6 ||
            std::fabs(coupled_fast_kernel.w[1] - base_fast_kernel.w[1]) > 1e-6 ||
            std::fabs(coupled_fast_kernel.gamma[0] - base_fast_kernel.gamma[0]) > 1e-6 ||
            std::fabs(coupled_fast_kernel.gamma[1] - base_fast_kernel.gamma[1]) > 1e-6;
        if (!kernel_changed) {
            std::cerr << "top-down coupling should change lower-level kernel\n";
            return 1;
        }

        if (!(coupled_run.final_levels[0].spectral.Deff > 0.0 && coupled_run.final_levels[1].spectral.Deff > 0.0)) {
            std::cerr << "final spectral summaries should be populated\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_smoke_test passed\n";
    return 0;
}
