#pragma once

#include <string>
#include <vector>

#include "abersoe/abersoe_runtime.hpp"

namespace abersoe {

enum class HierarchicalScenario : int;
struct HierarchicalMINModel;

std::vector<std::string> list_hierarchical_scenarios();
std::string hierarchical_scenario_name(HierarchicalScenario scenario);
HierarchicalScenario parse_hierarchical_scenario(const std::string& name);
HierarchicalMINModel make_hierarchical_scenario_model(
    HierarchicalScenario scenario,
    bool enable_bottom_up,
    bool enable_top_down);
std::vector<AberSOEState> make_hierarchical_scenario_initial_states(HierarchicalScenario scenario);

} // namespace abersoe
