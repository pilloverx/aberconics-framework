#pragma once

#include <string>
#include <vector>

#include "abersoe/abersoe_config.hpp"
#include "abersoe/abersoe_diagnostics.hpp"
#include "abersoe/abersoe_hooks.hpp"
#include "abersoe/abersoe_model.hpp"

namespace abersoe {

struct AberSOEState {
    gfe::State u;
    gfe::Vec chi;
    double t = 0.0;
};

struct AberSOERunResult {
    AberSOEState final_state;
    AberSOEDiagnostics diagnostics;
    std::vector<AberSOEState> samples;
    AberSOEConfigRecord config;
    HookPassThroughSummary hooks;
    bool stochastic_forcing_applied = false;
    unsigned long long seed_used = 0ULL;
    gfe::Vec final_kernel_w;
    std::size_t learning_updates = 0;
};

// Single-step wrapper over gfe::step_augmented.
AberSOEState step(const AberSOEModelSpec& model, const AberSOEState& state);

// Time-marching runtime for deterministic or seeded-stochastic runs.
AberSOERunResult run(
    const AberSOEModelSpec& model,
    const AberSOEState& state0,
    const AberSOERuntimeConfig& cfg = {});

void write_run_csv(const std::string& path, const AberSOERunResult& result);

} // namespace abersoe
