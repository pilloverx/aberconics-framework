#include "abersoe/abersoe_runtime.hpp"
#include "abersoe/abersoe_regression.hpp"
#include "abersoe/abersoe_scenarios.hpp"
#include "abersoe/abersoe_hooks.hpp"

#include <cmath>
#include <iostream>

int main() {
    try {
        abersoe::AberSOEModelSpec model;
        model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.0, 0.5}, gfe::Vec{0.4, 0.2}};
        model.dynamics.dt = 0.05;
        model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
        model.dynamics.coupling_index = 0;
        model.operators.apply_L = [](const gfe::State& u, double /*t*/) { return gfe::State{-u[0]}; };
        model.operators.apply_N = [](const gfe::State& /*u*/, double /*t*/) { return gfe::State{0.0}; };
        model.operators.forcing = [](double /*t*/) { return gfe::State{1.0}; };
        gfe::AssumptionChecks checks;
        abersoe::RecordingEnergyObserver obs;
        model.assumption_checks = &checks;
        model.energy_observer = &obs;

        abersoe::AberSOEState s0;
        s0.u = gfe::State{1.0};
        s0.chi = gfe::Vec{0.0, 0.0};
        s0.t = 0.0;

        abersoe::AberSOERuntimeConfig cfg;
        cfg.steps = 100;
        cfg.sample_stride = 10;
        cfg.strict_finite = true;

        const auto out = abersoe::run(model, s0, cfg);
        if (out.diagnostics.steps_executed != cfg.steps) {
            std::cerr << "steps_executed mismatch\n";
            return 1;
        }
        if (!out.diagnostics.all_finite) {
            std::cerr << "non-finite state detected\n";
            return 1;
        }
        if (out.samples.empty()) {
            std::cerr << "samples should not be empty\n";
            return 1;
        }
        if (!(out.diagnostics.mean_abs_u >= 0.0 && out.diagnostics.mean_abs_chi >= 0.0)) {
            std::cerr << "mean diagnostics should be non-negative\n";
            return 1;
        }
        if (!out.hooks.assumption_checks_attached || !out.hooks.energy_observer_attached) {
            std::cerr << "hook passthrough should be attached\n";
            return 1;
        }
        if (out.hooks.energy_events != cfg.steps || obs.event_count() != cfg.steps) {
            std::cerr << "energy observer event count mismatch\n";
            return 1;
        }
        if (out.hooks.assumption_note.empty()) {
            std::cerr << "assumption note should be propagated\n";
            return 1;
        }
        if (!out.final_kernel_w.empty()) {
            std::cerr << "final_kernel_w should be empty when Hebbian learning is disabled\n";
            return 1;
        }

        // Deterministic stochastic forcing with same seed should match exactly.
        abersoe::AberSOERuntimeConfig stochastic_cfg = cfg;
        stochastic_cfg.stochastic_forcing = true;
        stochastic_cfg.forcing_noise_std = 0.1;
        stochastic_cfg.seed = 42ULL;
        stochastic_cfg.stochastic_channel = 0;
        const auto run_a = abersoe::run(model, s0, stochastic_cfg);
        const auto run_b = abersoe::run(model, s0, stochastic_cfg);
        if (run_a.final_state.u.size() != run_b.final_state.u.size()) {
            std::cerr << "deterministic run shape mismatch\n";
            return 1;
        }
        for (std::size_t i = 0; i < run_a.final_state.u.size(); ++i) {
            if (std::fabs(run_a.final_state.u[i] - run_b.final_state.u[i]) > 1e-12) {
                std::cerr << "seeded deterministic forcing mismatch\n";
                return 1;
            }
        }

        // Backend-agnostic fit report checks (NNLS + Prony).
        gfe::Vec t_ref;
        gfe::Vec y_ref;
        for (int i = 0; i <= 200; ++i) {
            const double t = 0.05 * static_cast<double>(i);
            t_ref.push_back(t);
            y_ref.push_back(0.6 * std::exp(-0.8 * t) + 0.4 * std::exp(-0.15 * t));
        }

        model.fitter.backend = gfe::FitBackend::NnlsProjectedGradient;
        const auto fit_nnls = abersoe::build_fit_report(t_ref, y_ref, model);
        if (!fit_nnls.ok || fit_nnls.modes_kept == 0) {
            std::cerr << "NNLS fit report failed\n";
            return 1;
        }

        model.fitter.backend = gfe::FitBackend::Prony;
        model.fitter.prony.order = 4;
        model.fitter.prony.ridge = 1e-9;
        const auto fit_prony = abersoe::build_fit_report(t_ref, y_ref, model);
        if (!fit_prony.ok || fit_prony.modes_kept == 0) {
            std::cerr << "Prony fit report failed\n";
            return 1;
        }

        // Invalid runtime config should throw.
        bool threw = false;
        try {
            abersoe::AberSOERuntimeConfig bad = cfg;
            bad.sample_stride = 0;
            (void)abersoe::run(model, s0, bad);
        } catch (...) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "expected throw for sample_stride=0\n";
            return 1;
        }

        // Hebbian learning (default rule: OjaNormalized) check.
        {
            abersoe::AberSOEModelSpec learn_model = model;
            learn_model.learning.enabled = true;
            learn_model.learning.rule = abersoe::HebbianRule::OjaNormalized;
            learn_model.learning.learning_rate = 0.05;
            learn_model.learning.decay = 1e-3;
            learn_model.learning.renormalize = true;
            learn_model.learning.target_weight_sum = 1.0;
            learn_model.kernel.gamma = gfe::Vec{1.0, 0.4, 0.1};
            learn_model.kernel.w = gfe::Vec{0.5, 0.3, 0.2};

            abersoe::AberSOEState learn_s0;
            learn_s0.u = gfe::State{1.0};
            learn_s0.chi = gfe::Vec{0.0, 0.0, 0.0};
            learn_s0.t = 0.0;

            abersoe::AberSOERuntimeConfig learn_cfg;
            learn_cfg.steps = 100;
            learn_cfg.sample_stride = 10;
            learn_cfg.strict_finite = true;

            const auto learn_run = abersoe::run(learn_model, learn_s0, learn_cfg);
            if (learn_run.learning_updates != learn_cfg.steps) {
                std::cerr << "hebbian learning_updates mismatch\n";
                return 1;
            }
            if (learn_run.final_kernel_w.size() != learn_model.kernel.w.size()) {
                std::cerr << "hebbian final_kernel_w shape mismatch\n";
                return 1;
            }

            bool changed = false;
            double sum_w = 0.0;
            for (std::size_t i = 0; i < learn_run.final_kernel_w.size(); ++i) {
                changed = changed || (std::fabs(learn_run.final_kernel_w[i] - learn_model.kernel.w[i]) > 1e-9);
                sum_w += learn_run.final_kernel_w[i];
            }
            if (!changed) {
                std::cerr << "hebbian weights should update when enabled\n";
                return 1;
            }
            if (std::fabs(sum_w - 1.0) > 1e-6) {
                std::cerr << "hebbian renormalization failed\n";
                return 1;
            }
        }

        // Regression snapshot checks for default references.
        {
            const auto ref_linear = abersoe::default_regression_reference(abersoe::AberSOEScenario::LinearScalar);
            auto model_linear = abersoe::make_scenario_model(ref_linear.scenario, ref_linear.dt, ref_linear.form);
            auto s0_linear = abersoe::make_scenario_initial_state(ref_linear.scenario);
            abersoe::AberSOERuntimeConfig cfg_linear;
            cfg_linear.steps = ref_linear.steps;
            cfg_linear.sample_stride = 10;
            cfg_linear.strict_finite = true;
            const auto run_linear = abersoe::run(model_linear, s0_linear, cfg_linear);
            const auto snap_linear = abersoe::make_regression_snapshot(run_linear);
            const auto chk_linear =
                abersoe::check_regression_snapshot(snap_linear, ref_linear.baseline, ref_linear.tolerance);
            if (!chk_linear.pass) {
                std::cerr << "linear snapshot regression failed\n";
                return 1;
            }
        }

        {
            const auto ref_lorenz = abersoe::default_regression_reference(abersoe::AberSOEScenario::Lorenz63);
            auto model_lorenz = abersoe::make_scenario_model(ref_lorenz.scenario, ref_lorenz.dt, ref_lorenz.form);
            auto s0_lorenz = abersoe::make_scenario_initial_state(ref_lorenz.scenario);
            abersoe::AberSOERuntimeConfig cfg_lorenz;
            cfg_lorenz.steps = ref_lorenz.steps;
            cfg_lorenz.sample_stride = 20;
            cfg_lorenz.strict_finite = true;
            const auto run_lorenz = abersoe::run(model_lorenz, s0_lorenz, cfg_lorenz);
            const auto snap_lorenz = abersoe::make_regression_snapshot(run_lorenz);
            const auto chk_lorenz =
                abersoe::check_regression_snapshot(snap_lorenz, ref_lorenz.baseline, ref_lorenz.tolerance);
            if (!chk_lorenz.pass) {
                std::cerr << "lorenz snapshot regression failed\n";
                return 1;
            }
        }

        {
            const auto ref_resonant = abersoe::default_regression_reference(abersoe::AberSOEScenario::Resonant1D);
            auto model_resonant =
                abersoe::make_scenario_model(ref_resonant.scenario, ref_resonant.dt, ref_resonant.form);
            auto s0_resonant = abersoe::make_scenario_initial_state(ref_resonant.scenario);
            abersoe::AberSOERuntimeConfig cfg_resonant;
            cfg_resonant.steps = ref_resonant.steps;
            cfg_resonant.sample_stride = 20;
            cfg_resonant.strict_finite = true;
            const auto run_resonant = abersoe::run(model_resonant, s0_resonant, cfg_resonant);
            const auto snap_resonant = abersoe::make_regression_snapshot(run_resonant);
            const auto chk_resonant =
                abersoe::check_regression_snapshot(snap_resonant, ref_resonant.baseline, ref_resonant.tolerance);
            if (!chk_resonant.pass) {
                std::cerr << "resonant snapshot regression failed\n";
                return 1;
            }
        }

        // Nonlinear deterministic scenario sanity: Lorenz63 should run finite.
        {
            auto lorenz_model = abersoe::make_scenario_model(
                abersoe::AberSOEScenario::Lorenz63, 0.005, gfe::CouplingForm::NegativeFeedback);
            auto lorenz_state0 = abersoe::make_scenario_initial_state(abersoe::AberSOEScenario::Lorenz63);
            abersoe::AberSOERuntimeConfig lorenz_cfg;
            lorenz_cfg.steps = 200;
            lorenz_cfg.sample_stride = 20;
            lorenz_cfg.strict_finite = true;
            const auto lorenz_run = abersoe::run(lorenz_model, lorenz_state0, lorenz_cfg);
            if (!lorenz_run.diagnostics.all_finite) {
                std::cerr << "lorenz scenario should remain finite\n";
                return 1;
            }
            if (lorenz_run.final_state.u.size() != 3) {
                std::cerr << "lorenz scenario state dimension mismatch\n";
                return 1;
            }
        }

        // Resonant Formulation C scenario sanity.
        {
            auto r_model = abersoe::make_scenario_model(
                abersoe::AberSOEScenario::Resonant1D, 0.01, gfe::CouplingForm::ResonantSecondOrder);
            auto r_state0 = abersoe::make_scenario_initial_state(abersoe::AberSOEScenario::Resonant1D);
            abersoe::AberSOERuntimeConfig r_cfg;
            r_cfg.steps = 150;
            r_cfg.sample_stride = 15;
            r_cfg.strict_finite = true;
            const auto r_run = abersoe::run(r_model, r_state0, r_cfg);
            if (!r_run.diagnostics.all_finite) {
                std::cerr << "resonant scenario should remain finite\n";
                return 1;
            }
            if (r_run.final_state.u.size() != 2) {
                std::cerr << "resonant scenario state dimension mismatch\n";
                return 1;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "abersoe_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "abersoe_smoke_test passed\n";
    return 0;
}
