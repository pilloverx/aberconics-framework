#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <iostream>

int main() {
    try {
        {
            const gfe::MemoryKernelParams input{gfe::Vec{1.0, 0.25}, gfe::Vec{0.7, 0.3}};
            const gfe::MemoryKernelParams bath{gfe::Vec{0.6, 0.1}, gfe::Vec{0.8, 0.2}};
            const gfe::MemoryKernelParams direct{gfe::Vec{0.05}, gfe::Vec{0.15}};

            abersoe::HierarchicalRenormOptions opts;
            opts.n_samples = 161;
            opts.prune_weight_threshold = 1e-6;
            opts.fitter.n_basis = 24;
            opts.fitter.gamma_min = 1e-2;
            opts.fitter.gamma_max = 4.0;
            opts.fitter.threshold = 1e-6;
            opts.fitter.lr = 5e-3;
            opts.fitter.iters = 5000;
            opts.max_fit_l2_error = 0.15;
            opts.max_pruned_fit_l2_error = 0.2;

            const auto report = abersoe::build_renormalized_kernel(input, bath, &direct, opts);
            if (report.output_kernel.gamma.empty() || report.output_kernel.w.empty()) {
                std::cerr << "renormalized kernel should be non-empty\n";
                return 1;
            }
            if (report.output_kernel.gamma.size() != report.output_kernel.w.size()) {
                std::cerr << "renormalized kernel shape mismatch\n";
                return 1;
            }
            for (double gamma : report.output_kernel.gamma) {
                if (!(gamma > 0.0) || !std::isfinite(gamma)) {
                    std::cerr << "renormalized gamma should stay positive and finite\n";
                    return 1;
                }
            }
            for (double w : report.output_kernel.w) {
                if (!(w >= 0.0) || !std::isfinite(w)) {
                    std::cerr << "renormalized weights should stay nonnegative and finite\n";
                    return 1;
                }
            }
            if (!std::isfinite(report.fit_l2_error) || report.fit_l2_error > 0.15) {
                std::cerr << "renormalized fit error too large\n";
                return 1;
            }
            if (!std::isfinite(report.pruned_fit_l2_error) || report.pruned_fit_l2_error > 0.2) {
                std::cerr << "renormalized pruned fit error too large\n";
                return 1;
            }
            if (!report.output_sorted_descending) {
                std::cerr << "renormalized kernel should stay sorted\n";
                return 1;
            }
            if (!(report.retained_weight_fraction > 0.0 && report.retained_weight_fraction <= 1.0 + 1e-12)) {
                std::cerr << "retained weight fraction should be in (0, 1]\n";
                return 1;
            }
            if (!(report.output_spectral.Deff > 0.0) || !(report.output_spectral.Mcap > 0.0)) {
                std::cerr << "renormalized spectral units should be populated\n";
                return 1;
            }
            if (!std::isfinite(report.input_to_output_pruned_drift.abs_delta_mcap) ||
                !std::isfinite(report.raw_to_pruned_drift.abs_delta_deff)) {
                std::cerr << "spectral drift diagnostics should be finite\n";
                return 1;
            }
            if (!report.consistency.nonempty_output_pass ||
                !report.consistency.sorted_output_pass ||
                !report.consistency.fit_error_pass ||
                !report.consistency.retained_weight_pass ||
                !report.consistency.prune_drift_coherence_pass) {
                std::cerr << "default renorm consistency contract should pass structural checks\n";
                return 1;
            }

            abersoe::AberSOEModelSpec plug_model;
            plug_model.kernel = input;
            plug_model.dynamics.dt = 0.01;
            plug_model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
            plug_model.dynamics.coupling_index = 0;
            plug_model.operators.apply_L = [](const gfe::State& u, double) { return gfe::State{-0.18 * u[0]}; };
            plug_model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
            plug_model.operators.forcing = [](double) { return gfe::State{0.04}; };

            abersoe::AberSOEState state0;
            state0.u = gfe::State{0.5};

            abersoe::AberSOERuntimeConfig plug_cfg;
            plug_cfg.steps = 60;
            plug_cfg.sample_stride = 10;
            plug_cfg.strict_finite = true;
            const auto plugback =
                abersoe::validate_renormalized_kernel_runtime(plug_model, state0, report.output_kernel, plug_cfg);
            if (!plugback.checked || !plugback.all_finite || plugback.steps_executed != plug_cfg.steps) {
                std::cerr << "plug-back runtime validation should succeed and remain finite\n";
                return 1;
            }

            auto report_with_plugback = report;
            report_with_plugback.plugback = plugback;
            abersoe::HierarchicalRenormConsistencyOptions contract_opts;
            contract_opts.max_fit_l2_error = 0.15;
            contract_opts.max_pruned_fit_l2_error = 0.2;
            contract_opts.min_retained_weight_fraction = 0.7;
            contract_opts.max_fit_linf_error = 0.5;
            contract_opts.max_pruned_fit_linf_error = 0.5;
            contract_opts.require_mcap_nondecrease = false;
            contract_opts.require_mscale_nondecrease = false;
            contract_opts.require_deff_submultiplicative_bound = false;
            const auto contract = abersoe::evaluate_renormalization_consistency(report_with_plugback, contract_opts);
            if (!contract.pass || !contract.plugback_finite_pass) {
                std::cerr << "renorm contract with plug-back should pass\n";
                for (const auto& failure : contract.failures) {
                    std::cerr << "  " << failure << "\n";
                }
                return 1;
            }
        }

        {
            const auto hierarchy_model = abersoe::make_hierarchical_scenario_model(
                abersoe::HierarchicalScenario::ThreeLevelHomeostatic, true, true);
            const auto initial_states =
                abersoe::make_hierarchical_scenario_initial_states(abersoe::HierarchicalScenario::ThreeLevelHomeostatic);

            abersoe::HierarchicalRenormOptions opts;
            opts.n_samples = 181;
            opts.prune_weight_threshold = 1e-4;
            opts.fitter.n_basis = 24;
            opts.fitter.gamma_min = 1e-2;
            opts.fitter.gamma_max = 2.0;
            opts.fitter.threshold = 1e-6;
            opts.fitter.lr = 5e-3;
            opts.fitter.iters = 5000;
            opts.max_fit_l2_error = 0.2;
            opts.max_pruned_fit_l2_error = 0.25;

            const auto report = abersoe::build_renormalized_kernel(
                hierarchy_model.levels[0].model.kernel,
                hierarchy_model.levels[1].model.kernel,
                &hierarchy_model.levels[2].model.kernel,
                opts);

            abersoe::AberSOEModelSpec base_model = hierarchy_model.levels[0].model;
            abersoe::AberSOEState base_state = initial_states[0];
            base_state.chi.clear();
            abersoe::AberSOERuntimeConfig cfg;
            cfg.steps = 80;
            cfg.sample_stride = 10;
            cfg.strict_finite = true;

            const auto base_run = abersoe::run(base_model, base_state, cfg);
            const auto renorm_plugback =
                abersoe::validate_renormalized_kernel_runtime(base_model, base_state, report.output_kernel, cfg);
            if (!renorm_plugback.checked || !renorm_plugback.all_finite) {
                std::cerr << "hierarchy-like plug-back should stay finite\n";
                return 1;
            }

            const double base_deff = gfe::spectral_units(base_model.kernel.w, base_model.kernel.gamma).Deff;
            const double renorm_deff = report.output_spectral.Deff;
            if (!std::isfinite(base_deff) || !std::isfinite(renorm_deff)) {
                std::cerr << "pre/post Deff should be finite\n";
                return 1;
            }
            if (std::fabs(renorm_plugback.final_u_l2 - base_run.diagnostics.final_u_l2) > 2.5) {
                std::cerr << "renormalized plug-back drift in final_u_l2 is too large\n";
                return 1;
            }
            if (report.prune_changed_spectrum && report.raw_to_pruned_drift.abs_delta_mcap <= 0.0) {
                std::cerr << "prune_changed_spectrum should reflect measurable spectral drift\n";
                return 1;
            }
        }

        {
            const gfe::MemoryKernelParams input{gfe::Vec{0.5}, gfe::Vec{1.0}};
            const gfe::MemoryKernelParams bath{gfe::Vec{0.5}, gfe::Vec{1.0}};
            abersoe::HierarchicalRenormOptions opts;
            opts.n_samples = 81;
            opts.fitter.n_basis = 16;
            opts.fitter.gamma_min = 1e-2;
            opts.fitter.gamma_max = 2.0;
            opts.fitter.threshold = 1e-6;
            const auto report = abersoe::build_renormalized_kernel(input, bath, nullptr, opts);
            if (report.degenerate_pairs == 0) {
                std::cerr << "degenerate-rate path should be counted\n";
                return 1;
            }
        }

        {
            const gfe::MemoryKernelParams input{gfe::Vec{1.0}, gfe::Vec{1.0}};
            const gfe::MemoryKernelParams bath{gfe::Vec{0.4}, gfe::Vec{1.0}};
            abersoe::HierarchicalRenormOptions opts;
            opts.n_samples = 81;
            opts.fitter.n_basis = 12;
            opts.fitter.gamma_min = 1e-2;
            opts.fitter.gamma_max = 2.0;
            opts.max_fit_l2_error = 1e-12;
            bool threw = false;
            try {
                (void)abersoe::build_renormalized_kernel(input, bath, nullptr, opts);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            if (!threw) {
                std::cerr << "strict renorm fit threshold should throw on poor fit\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_renorm_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_renorm_test passed\n";
    return 0;
}
