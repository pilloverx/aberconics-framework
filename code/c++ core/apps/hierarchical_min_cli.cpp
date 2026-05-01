#include "abersoe/hierarchical_min.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " [--scenario two_level_bidirectional] [--list-scenarios]"
        << " [--steps N] [--sample-stride N] [--csv PATH] [--summary-csv PATH]"
        << " [--diagnostics-csv PATH] [--cross-level-csv PATH] [--renorm-csv PATH]"
        << " [--snapshot-csv PATH] [--snapshot-check] [--disable-bottom-up] [--disable-top-down]"
        << " [--no-strict]\n"
        << "\n"
        << "Runs hierarchical_min reference hierarchies and reports per-level, cross-level, and renorm summaries.\n";
}

void print_level_summary(std::size_t level_idx, const abersoe::HierarchicalLevelState& level, const std::string& name) {
    const double u0 = level.state.u.empty() ? 0.0 : level.state.u[0];
    const double chi0 = level.state.chi.empty() ? 0.0 : level.state.chi[0];
    const double w0 = level.active_kernel.w.empty() ? 0.0 : level.active_kernel.w[0];
    const double gamma0 = level.active_kernel.gamma.empty() ? 0.0 : level.active_kernel.gamma[0];

    std::cout << "level[" << level_idx << "].name=" << name << "\n";
    std::cout << "level[" << level_idx << "].t=" << level.state.t << "\n";
    std::cout << "level[" << level_idx << "].state_dim=" << level.state.u.size() << "\n";
    std::cout << "level[" << level_idx << "].memory_channels=" << level.state.chi.size() << "\n";
    std::cout << "level[" << level_idx << "].u0=" << u0 << "\n";
    std::cout << "level[" << level_idx << "].chi0=" << chi0 << "\n";
    std::cout << "level[" << level_idx << "].w0=" << w0 << "\n";
    std::cout << "level[" << level_idx << "].gamma0=" << gamma0 << "\n";
    std::cout << "level[" << level_idx << "].Mcap=" << level.spectral.Mcap << "\n";
    std::cout << "level[" << level_idx << "].Mscale=" << level.spectral.Mscale << "\n";
    std::cout << "level[" << level_idx << "].Hmem=" << level.spectral.Hmem << "\n";
    std::cout << "level[" << level_idx << "].Deff=" << level.spectral.Deff << "\n";
}

void print_cross_level_summary(std::size_t idx, const abersoe::HierarchicalCrossLevelSummary& relation) {
    std::cout << "cross_level[" << idx << "].relation=" << relation.relation << "\n";
    std::cout << "cross_level[" << idx << "].source=" << relation.source_name << "\n";
    std::cout << "cross_level[" << idx << "].target=" << relation.target_name << "\n";
    std::cout << "cross_level[" << idx << "].delta_Mcap=" << relation.delta_mcap << "\n";
    std::cout << "cross_level[" << idx << "].delta_Mscale=" << relation.delta_mscale << "\n";
    std::cout << "cross_level[" << idx << "].delta_Hmem=" << relation.delta_hmem << "\n";
    std::cout << "cross_level[" << idx << "].delta_Deff=" << relation.delta_deff << "\n";
    std::cout << "cross_level[" << idx << "].target_Mcap_not_lower=" << (relation.target_mcap_not_lower ? 1 : 0) << "\n";
    std::cout << "cross_level[" << idx << "].target_Mscale_not_lower=" << (relation.target_mscale_not_lower ? 1 : 0) << "\n";
}

void print_kernel_shift_summary(std::size_t idx, const abersoe::HierarchicalKernelShiftSummary& shift) {
    std::cout << "kernel_shift[" << idx << "].level=" << shift.level_name << "\n";
    std::cout << "kernel_shift[" << idx << "].delta_Mcap=" << shift.delta_mcap << "\n";
    std::cout << "kernel_shift[" << idx << "].delta_Mscale=" << shift.delta_mscale << "\n";
    std::cout << "kernel_shift[" << idx << "].delta_Hmem=" << shift.delta_hmem << "\n";
    std::cout << "kernel_shift[" << idx << "].delta_Deff=" << shift.delta_deff << "\n";
    std::cout << "kernel_shift[" << idx << "].changed=" << (shift.changed ? 1 : 0) << "\n";
}

void print_warning(std::size_t idx, const abersoe::HierarchicalWarning& warning) {
    std::cout << "warning[" << idx << "].severity=" << warning.severity << "\n";
    std::cout << "warning[" << idx << "].code=" << warning.code << "\n";
    std::cout << "warning[" << idx << "].message=" << warning.message << "\n";
}

void print_renorm_summary(std::size_t idx, const abersoe::HierarchicalAdjacentRenormAnalysis& analysis) {
    std::cout << "renorm[" << idx << "].input=" << analysis.input_name << "\n";
    std::cout << "renorm[" << idx << "].bath=" << analysis.bath_name << "\n";
    std::cout << "renorm[" << idx << "].direct=" << analysis.direct_name << "\n";
    std::cout << "renorm[" << idx << "].primary_diagnostic="
              << abersoe::hierarchical_renorm_diagnostic_cause_name(analysis.report.primary_diagnostic) << "\n";
    std::cout << "renorm[" << idx << "].raw_output_channels=" << analysis.report.raw_output_channels << "\n";
    std::cout << "renorm[" << idx << "].output_channels=" << analysis.report.output_channels << "\n";
    std::cout << "renorm[" << idx << "].pruned_channels=" << analysis.report.pruned_channels << "\n";
    std::cout << "renorm[" << idx << "].retained_weight_fraction=" << analysis.report.retained_weight_fraction << "\n";
    std::cout << "renorm[" << idx << "].fit_l2_error=" << analysis.report.fit_l2_error << "\n";
    std::cout << "renorm[" << idx << "].pruned_fit_l2_error=" << analysis.report.pruned_fit_l2_error << "\n";
    std::cout << "renorm[" << idx << "].delta_Mcap=" << analysis.report.input_to_output_pruned_drift.delta_mcap << "\n";
    std::cout << "renorm[" << idx << "].delta_Mscale=" << analysis.report.input_to_output_pruned_drift.delta_mscale << "\n";
    std::cout << "renorm[" << idx << "].delta_Deff=" << analysis.report.input_to_output_pruned_drift.delta_deff << "\n";
    std::cout << "renorm[" << idx << "].consistency_pass=" << (analysis.report.consistency.pass ? 1 : 0) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string scenario_name = "two_level_bidirectional";
    bool list_scenarios_only = false;
    std::size_t steps = 80;
    std::size_t sample_stride = 10;
    bool strict = true;
    bool enable_bottom_up = true;
    bool enable_top_down = true;
    bool snapshot_check = false;
    std::string csv_path;
    std::string summary_csv_path;
    std::string diagnostics_csv_path;
    std::string cross_level_csv_path;
    std::string renorm_csv_path;
    std::string snapshot_csv_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scenario") {
            if (i + 1 >= argc) {
                std::cerr << "--scenario requires a value\n";
                return 2;
            }
            scenario_name = argv[++i];
        } else if (arg == "--list-scenarios") {
            list_scenarios_only = true;
        } else if (arg == "--steps") {
            if (i + 1 >= argc) {
                std::cerr << "--steps requires a value\n";
                return 2;
            }
            steps = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--sample-stride") {
            if (i + 1 >= argc) {
                std::cerr << "--sample-stride requires a value\n";
                return 2;
            }
            sample_stride = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--csv") {
            if (i + 1 >= argc) {
                std::cerr << "--csv requires a path\n";
                return 2;
            }
            csv_path = argv[++i];
        } else if (arg == "--summary-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--summary-csv requires a path\n";
                return 2;
            }
            summary_csv_path = argv[++i];
        } else if (arg == "--diagnostics-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--diagnostics-csv requires a path\n";
                return 2;
            }
            diagnostics_csv_path = argv[++i];
        } else if (arg == "--cross-level-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--cross-level-csv requires a path\n";
                return 2;
            }
            cross_level_csv_path = argv[++i];
        } else if (arg == "--renorm-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--renorm-csv requires a path\n";
                return 2;
            }
            renorm_csv_path = argv[++i];
        } else if (arg == "--snapshot-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--snapshot-csv requires a path\n";
                return 2;
            }
            snapshot_csv_path = argv[++i];
        } else if (arg == "--snapshot-check") {
            snapshot_check = true;
        } else if (arg == "--disable-bottom-up") {
            enable_bottom_up = false;
        } else if (arg == "--disable-top-down") {
            enable_top_down = false;
        } else if (arg == "--no-strict") {
            strict = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list_scenarios_only) {
        for (const auto& name : abersoe::list_hierarchical_scenarios()) {
            std::cout << name << "\n";
        }
        return 0;
    }

    try {
        const auto scenario = abersoe::parse_hierarchical_scenario(scenario_name);
        const auto model = abersoe::make_hierarchical_scenario_model(scenario, enable_bottom_up, enable_top_down);
        const auto initial_states = abersoe::make_hierarchical_scenario_initial_states(scenario);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);

        abersoe::HierarchicalRuntimeConfig cfg;
        cfg.steps = steps;
        cfg.sample_stride = sample_stride;
        cfg.strict_finite = strict;

        const auto run = abersoe::run(model, state0, cfg);
        const auto snapshot = abersoe::make_regression_snapshot(run);
        const auto cross_level = abersoe::make_hierarchical_cross_level_report(model, run);
        const auto renorm = abersoe::make_hierarchical_adjacent_renorm_analyses(model);

        std::cout << "HIERARCHICAL_MIN run summary\n";
        std::cout << "scenario=" << abersoe::hierarchical_scenario_name(scenario) << "\n";
        std::cout << "levels=" << run.final_levels.size() << "\n";
        std::cout << "steps_executed=" << run.diagnostics.steps_executed << "\n";
        std::cout << "all_finite=" << (run.diagnostics.all_finite ? 1 : 0) << "\n";
        std::cout << "bottom_up_enabled=" << (enable_bottom_up ? 1 : 0) << "\n";
        std::cout << "top_down_enabled=" << (enable_top_down ? 1 : 0) << "\n";

        for (std::size_t level_idx = 0; level_idx < run.final_levels.size(); ++level_idx) {
            const auto& name = model.levels[level_idx].name.empty()
                ? std::string("level") + std::to_string(level_idx)
                : model.levels[level_idx].name;
            print_level_summary(level_idx, run.final_levels[level_idx], name);
        }
        for (std::size_t idx = 0; idx < cross_level.kernel_shifts.size(); ++idx) {
            print_kernel_shift_summary(idx, cross_level.kernel_shifts[idx]);
        }
        for (std::size_t idx = 0; idx < cross_level.relations.size(); ++idx) {
            print_cross_level_summary(idx, cross_level.relations[idx]);
        }
        std::cout << "warnings=" << cross_level.warnings.size() << "\n";
        for (std::size_t idx = 0; idx < cross_level.warnings.size(); ++idx) {
            print_warning(idx, cross_level.warnings[idx]);
        }
        std::cout << "renorm_analyses=" << renorm.size() << "\n";
        for (std::size_t idx = 0; idx < renorm.size(); ++idx) {
            print_renorm_summary(idx, renorm[idx]);
        }

        if (!csv_path.empty()) {
            abersoe::write_hierarchy_run_csv(csv_path, model, run);
            std::cout << "csv=" << csv_path << "\n";
        }
        if (!summary_csv_path.empty()) {
            abersoe::write_hierarchy_summary_csv(summary_csv_path, model, run);
            std::cout << "summary_csv=" << summary_csv_path << "\n";
        }
        if (!diagnostics_csv_path.empty()) {
            abersoe::write_hierarchy_diagnostics_csv(diagnostics_csv_path, model, run);
            std::cout << "diagnostics_csv=" << diagnostics_csv_path << "\n";
        }
        if (!cross_level_csv_path.empty()) {
            abersoe::write_hierarchy_cross_level_csv(cross_level_csv_path, cross_level);
            std::cout << "cross_level_csv=" << cross_level_csv_path << "\n";
        }
        if (!renorm_csv_path.empty()) {
            abersoe::write_hierarchy_renorm_csv(renorm_csv_path, renorm);
            std::cout << "renorm_csv=" << renorm_csv_path << "\n";
        }
        if (!snapshot_csv_path.empty()) {
            abersoe::write_hierarchical_regression_snapshot_csv(snapshot_csv_path, snapshot);
            std::cout << "snapshot_csv=" << snapshot_csv_path << "\n";
        }
        if (snapshot_check) {
            const auto ref = abersoe::default_hierarchical_regression_reference(scenario);
            const bool cfg_match =
                steps == ref.steps &&
                sample_stride == ref.sample_stride &&
                enable_bottom_up == ref.enable_bottom_up &&
                enable_top_down == ref.enable_top_down;
            if (!cfg_match) {
                std::cout << "snapshot_check_pass=0\n";
                std::cout << "snapshot_check_failure=config mismatch for reference baseline\n";
                std::cout << "snapshot_check_expected_steps=" << ref.steps << "\n";
                std::cout << "snapshot_check_expected_sample_stride=" << ref.sample_stride << "\n";
                std::cout << "snapshot_check_expected_bottom_up=" << (ref.enable_bottom_up ? 1 : 0) << "\n";
                std::cout << "snapshot_check_expected_top_down=" << (ref.enable_top_down ? 1 : 0) << "\n";
                if (strict) {
                    return 1;
                }
            } else {
                const auto check =
                    abersoe::check_regression_snapshot(snapshot, ref.baseline, ref.tolerance);
                std::cout << "snapshot_check_pass=" << (check.pass ? 1 : 0) << "\n";
                for (const auto& failure : check.failures) {
                    std::cout << "snapshot_check_failure=" << failure << "\n";
                }
                if (strict && !check.pass) {
                    return 1;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_cli failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
