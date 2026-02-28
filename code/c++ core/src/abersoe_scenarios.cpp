#include "abersoe/abersoe_scenarios.hpp"

#include <stdexcept>

namespace abersoe {

std::vector<std::string> list_scenarios() {
    return {"linear", "lorenz63", "resonant1d"};
}

std::string scenario_name(AberSOEScenario scenario) {
    switch (scenario) {
        case AberSOEScenario::LinearScalar:
            return "linear";
        case AberSOEScenario::Lorenz63:
            return "lorenz63";
        case AberSOEScenario::Resonant1D:
            return "resonant1d";
    }
    throw std::invalid_argument("Unknown AberSOEScenario");
}

AberSOEScenario parse_scenario(const std::string& name) {
    if (name == "linear") {
        return AberSOEScenario::LinearScalar;
    }
    if (name == "lorenz63" || name == "lorenz") {
        return AberSOEScenario::Lorenz63;
    }
    if (name == "resonant1d" || name == "resonant") {
        return AberSOEScenario::Resonant1D;
    }
    throw std::invalid_argument("Unknown scenario: " + name);
}

AberSOEModelSpec make_scenario_model(AberSOEScenario scenario, double dt, gfe::CouplingForm form) {
    AberSOEModelSpec model;
    model.dynamics.dt = dt;
    model.dynamics.form = form;
    model.dynamics.coupling_index = 0;
    model.fitter.backend = gfe::FitBackend::NnlsProjectedGradient;
    // Learning defaults: available and configurable, but disabled unless experiment opts in.
    model.learning.enabled = false;
    model.learning.rule = abersoe::HebbianRule::OjaNormalized;

    if (scenario == AberSOEScenario::LinearScalar) {
        model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.0, 0.5, 0.2}, gfe::Vec{0.5, 0.3, 0.2}};
        model.operators.apply_L = [](const gfe::State& u, double /*t*/) { return gfe::State{-0.2 * u[0]}; };
        model.operators.apply_N = [](const gfe::State& /*u*/, double /*t*/) { return gfe::State{0.0}; };
        model.operators.forcing = [](double /*t*/) { return gfe::State{1.0}; };
        return model;
    }

    if (scenario == AberSOEScenario::Lorenz63) {
        // Lorenz63 split into linear + nonlinear operators.
        const double sigma = 10.0;
        const double rho = 28.0;
        const double beta = 8.0 / 3.0;

        model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.2, 0.4, 0.1}, gfe::Vec{0.5, 0.3, 0.2}};
        model.operators.apply_L = [sigma, rho, beta](const gfe::State& u, double /*t*/) {
            return gfe::State{
                -sigma * u[0] + sigma * u[1],
                rho * u[0] - u[1],
                -beta * u[2]};
        };
        model.operators.apply_N = [](const gfe::State& u, double /*t*/) {
            return gfe::State{
                0.0,
                -u[0] * u[2],
                u[0] * u[1]};
        };
        model.operators.forcing = [](double /*t*/) { return gfe::State{0.0, 0.0, 0.0}; };
        return model;
    }

    if (scenario == AberSOEScenario::Resonant1D) {
        model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.0, 0.3, 0.1}, gfe::Vec{0.5, 0.3, 0.2}};
        model.dynamics.form = gfe::CouplingForm::ResonantSecondOrder;
        model.dynamics.coupling_index = 0;
        model.dynamics.resonant_stiffness = 1.0;
        model.dynamics.resonant_damping = 0.15;
        model.dynamics.resonant_feedback_gain = 1.0;
        model.operators.apply_L = [](const gfe::State& /*u*/, double /*t*/) { return gfe::State{0.0, 0.0}; };
        model.operators.apply_N = [](const gfe::State& /*u*/, double /*t*/) { return gfe::State{0.0, 0.0}; };
        model.operators.forcing = [](double /*t*/) { return gfe::State{0.0, 0.0}; };
        return model;
    }

    throw std::invalid_argument("Unhandled scenario");
}

AberSOEState make_scenario_initial_state(AberSOEScenario scenario) {
    if (scenario == AberSOEScenario::LinearScalar) {
        return AberSOEState{gfe::State{0.0}, gfe::Vec{0.0, 0.0, 0.0}, 0.0};
    }
    if (scenario == AberSOEScenario::Lorenz63) {
        return AberSOEState{gfe::State{1.0, 1.0, 1.0}, gfe::Vec{0.0, 0.0, 0.0}, 0.0};
    }
    if (scenario == AberSOEScenario::Resonant1D) {
        return AberSOEState{gfe::State{1.0, 0.0}, gfe::Vec{0.0, 0.0, 0.0}, 0.0};
    }
    throw std::invalid_argument("Unhandled scenario");
}

} // namespace abersoe
