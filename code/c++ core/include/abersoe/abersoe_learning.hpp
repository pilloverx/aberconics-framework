#pragma once

#include <cstddef>

#include "gfe/gfe_dynamics.hpp"

namespace abersoe {

enum class HebbianRule {
    OjaNormalized,
    PlainHebbian
};

struct HebbianLearningConfig {
    bool enabled = false; // opt-in for backward-compatible defaults
    HebbianRule rule = HebbianRule::OjaNormalized;
    double learning_rate = 1e-3;
    double decay = 1e-4;
    double min_weight = 1e-12;
    double max_weight = 1e3;
    bool renormalize = true;
    double target_weight_sum = 1.0;
};

const char* hebbian_rule_name(HebbianRule rule);

void apply_hebbian_weight_update(
    gfe::Vec& w,
    const gfe::Vec& chi,
    const gfe::State& u,
    std::size_t coupling_index,
    const HebbianLearningConfig& cfg,
    double dt);

} // namespace abersoe

