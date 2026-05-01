#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <iostream>

int main() {
    try {
        {
            abersoe::HierarchicalChainSpec spec;
            spec.levels.push_back({"custom_fast",
                                   gfe::MemoryKernelParams{gfe::Vec{1.0, 0.4}, gfe::Vec{0.65, 0.35}},
                                   abersoe::AberSOEState{gfe::State{1.0}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01,
                                   gfe::State{0.25},
                                   gfe::State{0.0},
                                   gfe::CouplingForm::NegativeFeedback,
                                   0});
            spec.levels.push_back({"custom_slow",
                                   gfe::MemoryKernelParams{gfe::Vec{0.7, 0.15}, gfe::Vec{0.6, 0.4}},
                                   abersoe::AberSOEState{gfe::State{0.1}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01,
                                   gfe::State{0.08},
                                   gfe::State{0.01},
                                   gfe::CouplingForm::NegativeFeedback,
                                   0});
            spec.edges.push_back({0, 1, abersoe::HierarchicalChainRelationType::BottomUp, 0.35, true});
            spec.edges.push_back({1, 0, abersoe::HierarchicalChainRelationType::TopDown, 0.18, true});

            const auto custom_model = abersoe::make_hierarchical_model_from_spec(spec);
            const auto custom_initial_states = abersoe::make_hierarchical_initial_states_from_spec(spec);
            const auto custom_state0 = abersoe::initialize_hierarchy_state(custom_model, custom_initial_states);
            abersoe::HierarchicalRuntimeConfig custom_cfg;
            custom_cfg.steps = 18;
            custom_cfg.sample_stride = 6;
            custom_cfg.strict_finite = true;
            const auto custom_run = abersoe::run(custom_model, custom_state0, custom_cfg);
            if (!custom_run.diagnostics.all_finite || custom_run.final_levels.size() != spec.levels.size()) {
                std::cerr << "generic custom-chain model should run finitely\n";
                return 1;
            }

            abersoe::HierarchicalChainSpec uncoupled_spec = spec;
            uncoupled_spec.edges.clear();
            const auto uncoupled_model = abersoe::make_hierarchical_model_from_spec(uncoupled_spec);
            const auto uncoupled_state0 = abersoe::initialize_hierarchy_state(uncoupled_model, custom_initial_states);
            const auto uncoupled_run = abersoe::run(uncoupled_model, uncoupled_state0, custom_cfg);

            if (std::fabs(custom_run.final_levels[1].state.u[0] - uncoupled_run.final_levels[1].state.u[0]) < 1e-5) {
                std::cerr << "generic bottom-up custom-chain coupling should measurably affect the slow level\n";
                return 1;
            }
            const auto& coupled_fast_kernel = custom_run.final_levels[0].active_kernel;
            const auto& base_fast_kernel = custom_model.levels[0].model.kernel;
            const bool custom_kernel_changed =
                std::fabs(coupled_fast_kernel.w[0] - base_fast_kernel.w[0]) > 1e-6 ||
                std::fabs(coupled_fast_kernel.gamma[0] - base_fast_kernel.gamma[0]) > 1e-6;
            if (!custom_kernel_changed) {
                std::cerr << "generic top-down custom-chain coupling should measurably affect the fast kernel\n";
                return 1;
            }
        }

        const auto ref =
            abersoe::default_hierarchical_regression_reference(abersoe::HierarchicalScenario::TwoLevelBidirectional);
        const auto model = abersoe::make_hierarchical_scenario_model(
            ref.scenario, ref.enable_bottom_up, ref.enable_top_down);
        const auto initial_states = abersoe::make_hierarchical_scenario_initial_states(ref.scenario);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);

        abersoe::HierarchicalRuntimeConfig cfg;
        cfg.steps = ref.steps;
        cfg.sample_stride = ref.sample_stride;
        cfg.strict_finite = true;

        const auto run = abersoe::run(model, state0, cfg);
        const auto snapshot = abersoe::make_regression_snapshot(run);
        const auto check = abersoe::check_regression_snapshot(snapshot, ref.baseline, ref.tolerance);
        if (!check.pass) {
            std::cerr << "hierarchical snapshot regression failed\n";
            for (const auto& failure : check.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }

        const auto cross_level = abersoe::make_hierarchical_cross_level_report(model, run);
        const auto renorm = abersoe::make_hierarchical_adjacent_renorm_analyses(model);
        const auto cross_snapshot = abersoe::make_cross_level_regression_snapshot(cross_level);
        const auto cross_check = abersoe::check_cross_level_regression_snapshot(
            cross_snapshot,
            ref.cross_level_baseline,
            ref.cross_level_tolerance);
        if (!cross_check.pass) {
            std::cerr << "cross-level regression snapshot failed\n";
            for (const auto& failure : cross_check.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }
        const auto renorm_snapshot = abersoe::make_renorm_regression_snapshot(renorm);
        const auto renorm_check = abersoe::check_renorm_regression_snapshot(
            renorm_snapshot,
            ref.renorm_baseline,
            ref.renorm_tolerance);
        if (!renorm_check.pass) {
            std::cerr << "renorm regression snapshot failed\n";
            for (const auto& failure : renorm_check.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }
        if (!cross_level.contract.pass) {
            std::cerr << "cross-level contract should pass for the canonical 2-level hierarchy\n";
            for (const auto& failure : cross_level.contract.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }
        bool bad_actionable = false;
        for (const auto& warning : cross_level.warnings) {
            if (warning.actionable &&
                (warning.metric.empty() || warning.suggested_action.empty())) {
                bad_actionable = true;
            }
        }
        if (bad_actionable) {
            std::cerr << "actionable warnings should carry metric and suggested action\n";
            return 1;
        }

        const auto uncoupled_model =
            abersoe::make_hierarchical_scenario_model(ref.scenario, false, false);
        const auto uncoupled_state0 = abersoe::initialize_hierarchy_state(uncoupled_model, initial_states);
        const auto uncoupled_run = abersoe::run(uncoupled_model, uncoupled_state0, cfg);

        const double coupled_slow = run.final_levels[1].state.u[0];
        const double uncoupled_slow = uncoupled_run.final_levels[1].state.u[0];
        if (std::fabs(coupled_slow - uncoupled_slow) < 1e-4) {
            std::cerr << "coupled scenario should differ from uncoupled slow trajectory\n";
            return 1;
        }

        const auto& coupled_fast_kernel = run.final_levels[0].active_kernel;
        const auto& base_fast_kernel = model.levels[0].model.kernel;
        const bool kernel_changed =
            std::fabs(coupled_fast_kernel.w[0] - base_fast_kernel.w[0]) > 1e-6 ||
            std::fabs(coupled_fast_kernel.w[1] - base_fast_kernel.w[1]) > 1e-6 ||
            std::fabs(coupled_fast_kernel.gamma[0] - base_fast_kernel.gamma[0]) > 1e-6 ||
            std::fabs(coupled_fast_kernel.gamma[1] - base_fast_kernel.gamma[1]) > 1e-6;
        if (!kernel_changed) {
            std::cerr << "top-down modulation should alter the fast-level kernel\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_integration_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_integration_test passed\n";
    return 0;
}
