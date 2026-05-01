#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <iostream>

namespace {

struct DriftBands {
    double mcap = 20.0;
    double mscale = 1.0;
    double deff = 4.0;
};

void require_within(const char* name, double observed, double limit) {
    if (std::fabs(observed) > limit) {
        throw std::runtime_error(
            std::string(name) + " drift exceeded limit: observed=" + std::to_string(observed) +
            " limit=" + std::to_string(limit));
    }
}

} // namespace

int main() {
    try {
        const auto base_model = abersoe::make_hierarchical_scenario_model(
            abersoe::HierarchicalScenario::ThreeLevelHomeostatic, true, false);
        const auto initial_states =
            abersoe::make_hierarchical_scenario_initial_states(abersoe::HierarchicalScenario::ThreeLevelHomeostatic);
        const auto base_state0 = abersoe::initialize_hierarchy_state(base_model, initial_states);

        abersoe::HierarchicalRuntimeConfig cfg;
        cfg.steps = 120;
        cfg.sample_stride = 12;
        cfg.strict_finite = true;

        const auto base_run = abersoe::run(base_model, base_state0, cfg);
        if (!base_run.diagnostics.all_finite) {
            std::cerr << "baseline hierarchy should remain finite\n";
            return 1;
        }

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
            base_model.levels[0].model.kernel,
            base_model.levels[1].model.kernel,
            &base_model.levels[2].model.kernel,
            opts);

        auto renorm_model = base_model;
        renorm_model.levels[0].model.kernel = report.output_kernel;
        auto renorm_initial_states = initial_states;
        renorm_initial_states[0].chi.clear();
        auto renorm_state0 = abersoe::initialize_hierarchy_state(renorm_model, renorm_initial_states);
        const auto renorm_run = abersoe::run(renorm_model, renorm_state0, cfg);
        if (!renorm_run.diagnostics.all_finite) {
            std::cerr << "renormalized hierarchy should remain finite\n";
            return 1;
        }

        const auto& base_fast = base_run.final_levels[0].spectral;
        const auto& renorm_fast = renorm_run.final_levels[0].spectral;
        const double delta_mcap = renorm_fast.Mcap - base_fast.Mcap;
        const double delta_mscale = renorm_fast.Mscale - base_fast.Mscale;
        const double delta_deff = renorm_fast.Deff - base_fast.Deff;

        const DriftBands bands;
        require_within("Mcap", delta_mcap, bands.mcap);
        require_within("Mscale", delta_mscale, bands.mscale);
        require_within("Deff", delta_deff, bands.deff);

        const auto expected_fast = gfe::spectral_units(
            renorm_model.levels[0].model.kernel.w,
            renorm_model.levels[0].model.kernel.gamma);
        if (std::fabs(renorm_fast.Mcap - expected_fast.Mcap) > 1e-9 ||
            std::fabs(renorm_fast.Mscale - expected_fast.Mscale) > 1e-9 ||
            std::fabs(renorm_fast.Deff - expected_fast.Deff) > 1e-9) {
            std::cerr << "renormalized fast-level summary should match the plugged-back kernel summary\n";
            return 1;
        }

        if (std::fabs(report.input_to_output_pruned_drift.delta_mcap - delta_mcap) > 1e-9 ||
            std::fabs(report.input_to_output_pruned_drift.delta_mscale - delta_mscale) > 1e-9 ||
            std::fabs(report.input_to_output_pruned_drift.delta_deff - delta_deff) > 1e-9) {
            std::cerr << "renorm report drift should match runtime summary drift for the plugged-back fast level\n";
            return 1;
        }

        const double adaptive_u_drift =
            std::fabs(renorm_run.final_levels[1].state.u[0] - base_run.final_levels[1].state.u[0]);
        const double slow_u_drift =
            std::fabs(renorm_run.final_levels[2].state.u[0] - base_run.final_levels[2].state.u[0]);
        if (adaptive_u_drift > 0.35 || slow_u_drift > 0.2) {
            std::cerr << "downstream hierarchy state drift is larger than expected after plug-back\n";
            return 1;
        }

        if (report.prune_changed_spectrum &&
            report.raw_to_pruned_drift.abs_delta_mcap == 0.0 &&
            report.raw_to_pruned_drift.abs_delta_mscale == 0.0 &&
            report.raw_to_pruned_drift.abs_delta_deff == 0.0) {
            std::cerr << "prune_changed_spectrum should agree with raw->pruned drift metrics\n";
            return 1;
        }

        if (!report.consistency.pass) {
            std::cerr << "renorm integration benchmark should satisfy built-in consistency checks\n";
            return 1;
        }

        abersoe::HierarchicalRenormConsistencyOptions theory_options;
        theory_options.require_mscale_nondecrease = true;
        theory_options.mscale_nondecrease_tol = bands.mscale;
        const auto theory_check = abersoe::evaluate_renormalization_consistency(report, theory_options);
        if (!theory_check.pass || !theory_check.mscale_nondecrease_pass) {
            std::cerr << "renorm integration benchmark should satisfy Mscale consistency check\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_renorm_integration_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_renorm_integration_test passed\n";
    return 0;
}
