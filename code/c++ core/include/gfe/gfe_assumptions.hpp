#pragma once

#include <functional>
#include <string>
#include <vector>

#include "gfe/gfe_core.hpp"

namespace gfe {

using AssumptionState = std::vector<double>;

struct AssumptionOperatorCallbacks {
    std::function<AssumptionState(const AssumptionState&, double)> apply_L;
    std::function<AssumptionState(const AssumptionState&, double)> apply_N;
};

struct AssumptionEvalConfig {
    double min_gamma = 1e-10;
    double min_weight_sum = 1e-12;
    double min_spectral_span = 1e-9;
    double max_spectral_span = 12.0;
    double max_lipschitz_proxy = 1e6;
    double boundedness_sample_scale = 2.0;
};

struct AssumptionEvidence {
    std::string name;
    bool pass = true;
    double margin = 0.0;
    double value = 0.0;
    double threshold = 0.0;
    std::string note;
};

struct AssumptionEvaluation {
    bool pass = true;
    bool kernel_approximable = true;
    bool regularity_ok = true;
    bool operator_boundedness_ok = true;
    double kernel_margin = 0.0;
    double regularity_margin = 0.0;
    double operator_boundedness_margin = 0.0;
    std::vector<AssumptionEvidence> evidence;
    std::string note;
};

AssumptionEvaluation evaluate_assumptions(
    const Vec& gamma,
    const Vec& w,
    const AssumptionState& sample_state,
    double t,
    const AssumptionOperatorCallbacks& ops,
    const AssumptionEvalConfig& cfg = {});

std::vector<std::string> assumption_test_corpus_names();
AssumptionEvaluation run_assumption_test_case(const std::string& name);

} // namespace gfe

