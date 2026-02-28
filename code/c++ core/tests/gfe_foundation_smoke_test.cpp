#include "gfe/gfe_assumptions.hpp"
#include "gfe/gfe_energy.hpp"
#include "gfe/gfe_dynamics.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace gfe;

    try {
        // Deterministic assumption test corpus checks.
        const auto names = assumption_test_corpus_names();
        if (names.size() < 4) {
            std::cerr << "assumption corpus unexpectedly small\n";
            return 1;
        }

        const auto valid = run_assumption_test_case("valid_reference");
        if (!valid.pass) {
            std::cerr << "valid_reference should pass\n";
            return 1;
        }

        const auto bad_gamma = run_assumption_test_case("invalid_gamma_negative");
        if (bad_gamma.pass || bad_gamma.kernel_approximable) {
            std::cerr << "invalid_gamma_negative should fail kernel approximability\n";
            return 1;
        }

        const auto bad_weight = run_assumption_test_case("invalid_weight_sum_zero");
        if (bad_weight.pass || bad_weight.kernel_approximable) {
            std::cerr << "invalid_weight_sum_zero should fail kernel approximability\n";
            return 1;
        }

        const auto bad_op = run_assumption_test_case("invalid_operator_unbounded");
        if (bad_op.pass || bad_op.operator_boundedness_ok) {
            std::cerr << "invalid_operator_unbounded should fail operator boundedness\n";
            return 1;
        }

        // Energy bookkeeping via dynamics transition hook.
        State u0{1.0};
        Vec chi0{0.1, 0.0};
        MemoryKernelParams mk{Vec{1.0, 0.5}, Vec{0.4, 0.2}};
        DynamicsConfig cfg;
        cfg.dt = 0.01;
        cfg.form = CouplingForm::NegativeFeedback;
        cfg.coupling_index = 0;

        OperatorCallbacks ops;
        ops.apply_L = [](const State& u, double /*t*/) { return State{-0.2 * u[0]}; };
        ops.apply_N = [](const State& /*u*/, double /*t*/) { return State{0.0}; };
        ops.forcing = [](double /*t*/) { return State{0.0}; };

        EnergyBookkeepingConfig e_cfg;
        e_cfg.u_weight = 1.0;
        e_cfg.chi_weight = 1.0;
        e_cfg.memory_decay_rates = mk.gamma;
        e_cfg.residual_tolerance = 100.0; // loose bound for Euler + simple model
        QuadraticEnergyMonitor monitor(e_cfg);

        AssumptionChecks checks;
        auto step = step_augmented(u0, chi0, 0.0, mk, cfg, ops, nullptr, &checks, &monitor);

        if (step.u_next.size() != 1 || step.chi_next.size() != 2) {
            std::cerr << "step_augmented shape failure in foundation test\n";
            return 1;
        }
        if (monitor.snapshots().empty() || monitor.residuals().empty()) {
            std::cerr << "energy monitor should record samples\n";
            return 1;
        }
        if (!checks.kernel_approximable || !checks.regularity_ok) {
            std::cerr << "assumption checks should report baseline pass for reference model\n";
            return 1;
        }

        // Tight residual enforcement should throw.
        EnergyBookkeepingConfig strict_cfg = e_cfg;
        strict_cfg.residual_tolerance = 1e-14;
        strict_cfg.enforce_residual_tolerance = true;
        QuadraticEnergyMonitor strict_monitor(strict_cfg);

        bool threw = false;
        try {
            (void)step_augmented(u0, chi0, 0.0, mk, cfg, ops, nullptr, nullptr, &strict_monitor);
        } catch (...) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "strict residual enforcement should throw\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "gfe_foundation_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "gfe_foundation_smoke_test passed\n";
    return 0;
}

