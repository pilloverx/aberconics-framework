#pragma once

#include <string>
#include <vector>

#include "abersoe/abersoe_runtime.hpp"
#include "abersoe/abersoe_scenarios.hpp"

namespace abersoe {

struct RegressionSnapshot {
    std::size_t state_dim = 0;
    std::size_t memory_channels = 0;
    std::size_t steps_executed = 0;
    bool all_finite = true;
    double max_abs_u = 0.0;
    double max_abs_chi = 0.0;
    double final_u_l2 = 0.0;
    double final_chi_l2 = 0.0;
    double mean_abs_u = 0.0;
    double mean_abs_chi = 0.0;
};

struct RegressionTolerance {
    double max_abs_u = 0.0;
    double max_abs_chi = 0.0;
    double final_u_l2 = 0.0;
    double final_chi_l2 = 0.0;
    double mean_abs_u = 0.0;
    double mean_abs_chi = 0.0;
};

struct RegressionCheckResult {
    bool pass = true;
    std::vector<std::string> failures;
};

struct ScenarioRegressionReference {
    AberSOEScenario scenario = AberSOEScenario::LinearScalar;
    double dt = 0.0;
    std::size_t steps = 0;
    gfe::CouplingForm form = gfe::CouplingForm::NegativeFeedback;
    RegressionSnapshot baseline;
    RegressionTolerance tolerance;
};

RegressionSnapshot make_regression_snapshot(const AberSOERunResult& run);
RegressionCheckResult check_regression_snapshot(
    const RegressionSnapshot& observed,
    const RegressionSnapshot& baseline,
    const RegressionTolerance& tolerance);

ScenarioRegressionReference default_regression_reference(AberSOEScenario scenario);

void write_regression_snapshot_csv(const std::string& path, const RegressionSnapshot& snapshot);

} // namespace abersoe
