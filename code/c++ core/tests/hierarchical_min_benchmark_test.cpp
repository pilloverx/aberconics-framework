#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <iostream>

int main() {
    try {
        const auto ref =
            abersoe::default_hierarchical_regression_reference(abersoe::HierarchicalScenario::ThreeLevelHomeostatic);
        const auto model = abersoe::make_hierarchical_scenario_model(
            ref.scenario, ref.enable_bottom_up, ref.enable_top_down);
        const auto initial_states = abersoe::make_hierarchical_scenario_initial_states(ref.scenario);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);

        abersoe::HierarchicalRuntimeConfig cfg;
        cfg.steps = ref.steps;
        cfg.sample_stride = ref.sample_stride;
        cfg.strict_finite = true;

        const auto run = abersoe::run(model, state0, cfg);
        if (!run.diagnostics.all_finite || run.final_levels.size() != 3) {
            std::cerr << "3-level benchmark should run finite with 3 levels\n";
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
            std::cerr << "3-level benchmark cross-level regression failed\n";
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
            std::cerr << "3-level benchmark renorm regression failed\n";
            for (const auto& failure : renorm_check.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }
        if (!cross_level.contract.pass) {
            std::cerr << "3-level benchmark cross-level contract should pass\n";
            for (const auto& failure : cross_level.contract.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }
        if (cross_level.contract.actionable_warning_count == 0) {
            std::cerr << "3-level benchmark should expose actionable cross-level diagnostics\n";
            return 1;
        }

        const auto snapshot = abersoe::make_regression_snapshot(run);
        const auto check = abersoe::check_regression_snapshot(snapshot, ref.baseline, ref.tolerance);
        if (!check.pass) {
            std::cerr << "3-level benchmark regression failed\n";
            for (const auto& failure : check.failures) {
                std::cerr << "  " << failure << "\n";
            }
            return 1;
        }

        const auto no_coupling_model =
            abersoe::make_hierarchical_scenario_model(ref.scenario, false, false);
        const auto no_coupling_state0 = abersoe::initialize_hierarchy_state(no_coupling_model, initial_states);
        const auto no_coupling_run = abersoe::run(no_coupling_model, no_coupling_state0, cfg);

        const auto bu_only_model =
            abersoe::make_hierarchical_scenario_model(ref.scenario, true, false);
        const auto bu_only_state0 = abersoe::initialize_hierarchy_state(bu_only_model, initial_states);
        const auto bu_only_run = abersoe::run(bu_only_model, bu_only_state0, cfg);

        const auto td_only_model =
            abersoe::make_hierarchical_scenario_model(ref.scenario, false, true);
        const auto td_only_state0 = abersoe::initialize_hierarchy_state(td_only_model, initial_states);
        const auto td_only_run = abersoe::run(td_only_model, td_only_state0, cfg);

        const double slow_no = no_coupling_run.final_levels[2].state.u[0];
        const double slow_bu = bu_only_run.final_levels[2].state.u[0];
        const double slow_bidir = run.final_levels[2].state.u[0];
        if (std::fabs(slow_bu - slow_no) < 1e-4 || std::fabs(slow_bidir - slow_no) < 1e-4) {
            std::cerr << "bottom-up pathways should measurably affect the slowest level\n";
            return 1;
        }

        const auto& fast_kernel_td = td_only_run.final_levels[0].active_kernel;
        const auto& fast_kernel_base = td_only_model.levels[0].model.kernel;
        const bool fast_changed =
            std::fabs(fast_kernel_td.w[0] - fast_kernel_base.w[0]) > 1e-6 ||
            std::fabs(fast_kernel_td.gamma[0] - fast_kernel_base.gamma[0]) > 1e-6;
        const auto& mid_kernel_td = td_only_run.final_levels[1].active_kernel;
        const auto& mid_kernel_base = td_only_model.levels[1].model.kernel;
        const bool mid_changed =
            std::fabs(mid_kernel_td.w[0] - mid_kernel_base.w[0]) > 1e-6 ||
            std::fabs(mid_kernel_td.gamma[0] - mid_kernel_base.gamma[0]) > 1e-6;
        if (!fast_changed || !mid_changed) {
            std::cerr << "top-down pathways should modulate fast and intermediate kernels\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_benchmark_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_benchmark_test passed\n";
    return 0;
}
