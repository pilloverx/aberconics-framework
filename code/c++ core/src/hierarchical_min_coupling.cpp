#include "abersoe/hierarchical_min_coupling.hpp"

#include "abersoe/hierarchical_min.hpp"

#include <stdexcept>

namespace abersoe {
namespace {

AberSOEModelSpec make_fast_level() {
    AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.0, 0.25}, gfe::Vec{0.6, 0.4}};
    model.dynamics.dt = 0.02;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.9 * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [](double) { return gfe::State{0.0}; };
    return model;
}

AberSOEModelSpec make_slow_level() {
    AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{0.5, 0.1}, gfe::Vec{0.55, 0.45}};
    model.dynamics.dt = 0.02;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.15 * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [](double) { return gfe::State{0.05}; };
    return model;
}

AberSOEModelSpec make_intermediate_level() {
    AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{0.35, 0.08}, gfe::Vec{0.52, 0.48}};
    model.dynamics.dt = 0.02;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.22 * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [](double) { return gfe::State{0.02}; };
    return model;
}

AberSOEModelSpec make_homeostatic_level() {
    AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{0.12, 0.03}, gfe::Vec{0.58, 0.42}};
    model.dynamics.dt = 0.02;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.07 * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [](double) { return gfe::State{0.01}; };
    return model;
}

} // namespace

std::vector<std::string> list_hierarchical_scenarios() {
    return {"two_level_bidirectional", "three_level_homeostatic"};
}

std::string hierarchical_scenario_name(HierarchicalScenario scenario) {
    switch (scenario) {
        case HierarchicalScenario::TwoLevelBidirectional:
            return "two_level_bidirectional";
        case HierarchicalScenario::ThreeLevelHomeostatic:
            return "three_level_homeostatic";
    }
    throw std::invalid_argument("Unknown hierarchical scenario");
}

HierarchicalScenario parse_hierarchical_scenario(const std::string& name) {
    if (name == "two_level_bidirectional" || name == "two_level" || name == "toy_bidirectional") {
        return HierarchicalScenario::TwoLevelBidirectional;
    }
    if (name == "three_level_homeostatic" || name == "three_level" || name == "benchmark3") {
        return HierarchicalScenario::ThreeLevelHomeostatic;
    }
    throw std::invalid_argument("Unknown hierarchical scenario: " + name);
}

HierarchicalMINModel make_hierarchical_scenario_model(
    HierarchicalScenario scenario,
    bool enable_bottom_up,
    bool enable_top_down) {
    HierarchicalMINModel model;
    if (scenario == HierarchicalScenario::TwoLevelBidirectional) {
        model.levels.push_back({"fast", make_fast_level(), {}});
        model.levels.push_back({"slow", make_slow_level(), {}});

        if (enable_bottom_up) {
            BottomUpCoupling bottom_up;
            bottom_up.source_level = 0;
            bottom_up.target_level = 1;
            bottom_up.forcing = [](const gfe::State& source_obs, double) {
                return gfe::State{0.25 * source_obs[0]};
            };
            model.bottom_up_couplings.push_back(bottom_up);
        }

        if (enable_top_down) {
            TopDownCoupling top_down;
            top_down.source_level = 1;
            top_down.target_level = 0;
            top_down.modulation = [](const gfe::State& source_obs) {
                const double s = source_obs[0];
                PositiveKernelModulation out;
                out.alpha = gfe::Vec{0.25 * s, -0.15 * s};
                out.beta = gfe::Vec{0.1 + 0.35 * s, -1.4 - 0.2 * s};
                out.normalize_weights = true;
                return out;
            };
            model.top_down_couplings.push_back(top_down);
        }
        return model;
    }

    if (scenario == HierarchicalScenario::ThreeLevelHomeostatic) {
        model.levels.push_back({"fast", make_fast_level(), {}});
        model.levels.push_back({"adaptive", make_intermediate_level(), {}});
        model.levels.push_back({"homeostatic", make_homeostatic_level(), {}});

        if (enable_bottom_up) {
            BottomUpCoupling fast_to_mid;
            fast_to_mid.source_level = 0;
            fast_to_mid.target_level = 1;
            fast_to_mid.forcing = [](const gfe::State& source_obs, double) {
                return gfe::State{0.18 * source_obs[0]};
            };
            model.bottom_up_couplings.push_back(fast_to_mid);

            BottomUpCoupling mid_to_slow;
            mid_to_slow.source_level = 1;
            mid_to_slow.target_level = 2;
            mid_to_slow.forcing = [](const gfe::State& source_obs, double) {
                return gfe::State{0.14 * source_obs[0]};
            };
            model.bottom_up_couplings.push_back(mid_to_slow);
        }

        if (enable_top_down) {
            TopDownCoupling mid_to_fast;
            mid_to_fast.source_level = 1;
            mid_to_fast.target_level = 0;
            mid_to_fast.modulation = [](const gfe::State& source_obs) {
                const double s = source_obs[0];
                PositiveKernelModulation out;
                out.alpha = gfe::Vec{0.18 * s, -0.10 * s};
                out.beta = gfe::Vec{-0.05 + 0.22 * s, -1.55 - 0.15 * s};
                out.normalize_weights = true;
                return out;
            };
            model.top_down_couplings.push_back(mid_to_fast);

            TopDownCoupling slow_to_mid;
            slow_to_mid.source_level = 2;
            slow_to_mid.target_level = 1;
            slow_to_mid.modulation = [](const gfe::State& source_obs) {
                const double s = source_obs[0];
                PositiveKernelModulation out;
                out.alpha = gfe::Vec{0.12 * s, -0.08 * s};
                out.beta = gfe::Vec{-0.55 + 0.18 * s, -2.10 - 0.10 * s};
                out.normalize_weights = true;
                return out;
            };
            model.top_down_couplings.push_back(slow_to_mid);
        }
        return model;
    }

    throw std::invalid_argument("Unhandled hierarchical scenario");
}

std::vector<AberSOEState> make_hierarchical_scenario_initial_states(HierarchicalScenario scenario) {
    if (scenario == HierarchicalScenario::TwoLevelBidirectional) {
        std::vector<AberSOEState> initial_states(2);
        initial_states[0].u = gfe::State{1.0};
        initial_states[0].chi = gfe::Vec{0.0, 0.0};
        initial_states[1].u = gfe::State{0.2};
        initial_states[1].chi = gfe::Vec{0.0, 0.0};
        return initial_states;
    }
    if (scenario == HierarchicalScenario::ThreeLevelHomeostatic) {
        std::vector<AberSOEState> initial_states(3);
        initial_states[0].u = gfe::State{1.0};
        initial_states[0].chi = gfe::Vec{0.0, 0.0};
        initial_states[1].u = gfe::State{0.12};
        initial_states[1].chi = gfe::Vec{0.0, 0.0};
        initial_states[2].u = gfe::State{0.04};
        initial_states[2].chi = gfe::Vec{0.0, 0.0};
        return initial_states;
    }
    throw std::invalid_argument("Unhandled hierarchical scenario");
}

} // namespace abersoe
