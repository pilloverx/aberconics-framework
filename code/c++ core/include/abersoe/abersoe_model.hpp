#pragma once

#include "gfe/gfe_dynamics.hpp"
#include "gfe/gfe_core.hpp"
#include "abersoe/abersoe_learning.hpp"

namespace abersoe {

// Model specification for deterministic or stochastic, linear or nonlinear systems.
// - Deterministic/non-stochastic runs: keep forcing deterministic and runtime stochastic options disabled.
// - Stochastic runs: enable runtime stochastic forcing controls in AberSOERuntimeConfig.
// - Nonlinear systems: supply nonlinear dynamics through operators.apply_N.
struct AberSOEModelSpec {
    gfe::MemoryKernelParams kernel;
    gfe::DynamicsConfig dynamics;
    gfe::OperatorCallbacks operators;

    // Optional hooks/adapters propagated to gfe::step_augmented.
    const gfe::GSTAdapter* gst = nullptr;
    gfe::AssumptionChecks* assumption_checks = nullptr;
    gfe::EnergyObserver* energy_observer = nullptr;

    // Fitter configuration for experiment pipelines that estimate kernels.
    gfe::SoeFitOptions fitter;

    // Optional adaptive memory-weight learning.
    // Default rule is Oja-normalized; disabled by default for backward-compatible trajectories.
    HebbianLearningConfig learning;
};

} // namespace abersoe
