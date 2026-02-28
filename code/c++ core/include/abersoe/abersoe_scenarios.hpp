#pragma once

#include <string>
#include <vector>

#include "abersoe/abersoe_runtime.hpp"

namespace abersoe {

enum class AberSOEScenario {
    LinearScalar,
    Lorenz63,
    Resonant1D
};

std::vector<std::string> list_scenarios();
std::string scenario_name(AberSOEScenario scenario);
AberSOEScenario parse_scenario(const std::string& name);

// Build a ready-to-run model for common deterministic/nonlinear baselines.
AberSOEModelSpec make_scenario_model(AberSOEScenario scenario, double dt, gfe::CouplingForm form);
AberSOEState make_scenario_initial_state(AberSOEScenario scenario);

} // namespace abersoe
