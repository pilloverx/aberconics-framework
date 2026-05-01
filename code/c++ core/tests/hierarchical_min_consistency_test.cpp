#include "abersoe/hierarchical_min.hpp"

#include <iostream>

int main() {
    try {
        {
            const auto model = abersoe::make_hierarchical_scenario_model(
                abersoe::HierarchicalScenario::ThreeLevelHomeostatic, true, true);
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
                model.levels[0].model.kernel,
                model.levels[1].model.kernel,
                &model.levels[2].model.kernel,
                opts);
            if (!report.consistency.pass) {
                std::cerr << "realistic renorm consistency check should pass\n";
                return 1;
            }

            abersoe::HierarchicalRenormConsistencyOptions theory_options;
            theory_options.require_mscale_nondecrease = true;
            theory_options.mscale_nondecrease_tol = 1.0;
            const auto theory_check = abersoe::evaluate_renormalization_consistency(report, theory_options);
            if (!theory_check.pass ||
                !theory_check.mcap_nondecrease_pass ||
                !theory_check.mscale_nondecrease_pass ||
                !theory_check.deff_bound_pass) {
                std::cerr << "realistic renorm should satisfy Mcap/Mscale/Deff theory checks\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalRenormReport fake;
            fake.input_spectral = gfe::SpectralUnits{5.0, 0.4, 0.0, 0.0, 0.0, 2.0};
            fake.bath_spectral = gfe::SpectralUnits{3.0, 0.3, 0.0, 0.0, 0.0, 2.5};
            fake.output_spectral = gfe::SpectralUnits{4.0, 0.2, 0.0, 0.0, 0.0, 6.0};

            abersoe::HierarchicalRenormConsistencyOptions options;
            options.require_mscale_nondecrease = true;
            options.require_nonempty_output = false;
            options.require_sorted_output = false;
            options.require_plugback_finite_when_checked = false;
            options.require_prune_drift_coherence = false;
            const auto check = abersoe::evaluate_renormalization_consistency(fake, options);
            if (check.pass || check.mcap_nondecrease_pass || check.mscale_nondecrease_pass || check.deff_bound_pass) {
                std::cerr << "synthetic failing consistency case should fail all theory checks\n";
                return 1;
            }
            if (check.failures.size() < 3) {
                std::cerr << "failing consistency case should produce failure messages\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalRenormReport fake;
            fake.input_spectral = gfe::SpectralUnits{5.0, 0.4, 0.0, 0.0, 0.0, 3.0};
            fake.bath_spectral = gfe::SpectralUnits{3.0, 0.3, 0.0, 0.0, 0.0, 2.0};
            fake.output_spectral = gfe::SpectralUnits{4.9, 0.35, 0.0, 0.0, 0.0, 6.2};

            abersoe::HierarchicalRenormConsistencyOptions options;
            options.mcap_nondecrease_tol = 0.2;
            options.mscale_nondecrease_tol = 0.1;
            options.deff_bound_slack = 0.3;
            options.require_nonempty_output = false;
            options.require_sorted_output = false;
            options.require_plugback_finite_when_checked = false;
            options.require_prune_drift_coherence = false;
            const auto check = abersoe::evaluate_renormalization_consistency(fake, options);
            if (!check.pass) {
                std::cerr << "toleranced consistency case should pass\n";
                return 1;
            }
        }

        {
            const gfe::MemoryKernelParams input{gfe::Vec{1.0, 0.25}, gfe::Vec{0.7, 0.3}};
            const gfe::MemoryKernelParams bath{gfe::Vec{0.6, 0.1}, gfe::Vec{0.8, 0.2}};
            const gfe::MemoryKernelParams direct{gfe::Vec{0.05}, gfe::Vec{0.15}};

            abersoe::HierarchicalRenormOptions unstable;
            unstable.n_samples = 81;
            unstable.prune_weight_threshold = 1e-6;
            unstable.fitter.n_basis = 4;
            unstable.fitter.gamma_min = 1e-2;
            unstable.fitter.gamma_max = 1.0;
            unstable.fitter.threshold = 1e-6;
            unstable.fitter.lr = 1e-3;
            unstable.fitter.iters = 200;
            const auto unstable_report = abersoe::build_renormalized_kernel(input, bath, &direct, unstable);
            if (unstable_report.primary_diagnostic != abersoe::HierarchicalRenormDiagnosticCause::FitInstability) {
                std::cerr << "underspecified fit should classify as FitInstability\n";
                return 1;
            }

            abersoe::HierarchicalRenormOptions overpruned = unstable;
            overpruned.fitter.n_basis = 24;
            overpruned.fitter.gamma_max = 4.0;
            overpruned.fitter.lr = 5e-3;
            overpruned.fitter.iters = 5000;
            overpruned.prune_weight_threshold = 0.95;
            const auto overpruned_report = abersoe::build_renormalized_kernel(input, bath, &direct, overpruned);
            if (overpruned_report.primary_diagnostic != abersoe::HierarchicalRenormDiagnosticCause::OverPruning) {
                std::cerr << "aggressive prune should classify as OverPruning\n";
                return 1;
            }

            abersoe::HierarchicalRenormOptions merged = overpruned;
            merged.prune_weight_threshold = 0.3;
            merged.fitter.n_basis = 16;
            const auto merged_report = abersoe::build_renormalized_kernel(input, bath, &direct, merged);
            if (merged_report.primary_diagnostic != abersoe::HierarchicalRenormDiagnosticCause::PlausibleChannelMerging) {
                std::cerr << "moderate prune with retained weight should classify as PlausibleChannelMerging\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_consistency_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_consistency_test passed\n";
    return 0;
}
