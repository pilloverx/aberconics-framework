#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "abersoe/hierarchical_min_renorm.hpp"

namespace abersoe {

enum class HierarchicalScenario : int;
struct HierarchicalMINModel;
struct HierarchicalRunResult;

struct HierarchicalRegressionSnapshot {
    std::size_t levels = 0;
    std::size_t total_state_dim = 0;
    std::size_t total_memory_channels = 0;
    std::size_t steps_executed = 0;
    bool all_finite = true;
    std::vector<double> final_u_l2_per_level;
    std::vector<double> final_chi_l2_per_level;
    std::vector<double> first_weight_per_level;
    std::vector<double> first_rate_per_level;
    std::vector<double> deff_per_level;
    std::vector<double> mcap_per_level;
    std::vector<double> mscale_per_level;
};

struct HierarchicalRegressionTolerance {
    double final_u_l2 = 0.0;
    double final_chi_l2 = 0.0;
    double first_weight = 0.0;
    double first_rate = 0.0;
    double deff = 0.0;
    double mcap = 0.0;
    double mscale = 0.0;
};

struct HierarchicalRegressionCheckResult {
    bool pass = true;
    std::vector<std::string> failures;
};

struct HierarchicalCrossLevelRegressionSnapshot {
    std::size_t kernel_shift_count = 0;
    std::size_t relation_count = 0;
    std::size_t warning_count = 0;
    std::size_t actionable_warning_count = 0;
    bool contract_pass = true;
    bool all_finite_pass = true;
    bool bottom_up_mcap_pass = true;
    bool bottom_up_mscale_pass = true;
    std::vector<double> kernel_shift_delta_mcap;
    std::vector<double> kernel_shift_delta_mscale;
    std::vector<double> kernel_shift_delta_deff;
    std::vector<double> relation_delta_mcap;
    std::vector<double> relation_delta_mscale;
    std::vector<double> relation_delta_deff;
    std::vector<std::string> warning_codes;
};

struct HierarchicalCrossLevelRegressionTolerance {
    double kernel_shift_delta_mcap = 0.0;
    double kernel_shift_delta_mscale = 0.0;
    double kernel_shift_delta_deff = 0.0;
    double relation_delta_mcap = 0.0;
    double relation_delta_mscale = 0.0;
    double relation_delta_deff = 0.0;
};

struct HierarchicalRenormRegressionSnapshot {
    std::size_t analysis_count = 0;
    std::vector<std::string> diagnostic_names;
    std::vector<double> retained_weight_fraction;
    std::vector<double> fit_l2_error;
    std::vector<double> pruned_fit_l2_error;
    std::vector<double> drift_delta_mcap;
    std::vector<double> drift_delta_mscale;
    std::vector<double> drift_delta_deff;
    std::vector<std::size_t> raw_output_channels;
    std::vector<std::size_t> output_channels;
    std::vector<std::size_t> pruned_channels;
    std::vector<int> consistency_pass;
    std::vector<int> mcap_pass;
    std::vector<int> mscale_pass;
    std::vector<int> deff_pass;
};

struct HierarchicalRenormRegressionTolerance {
    double retained_weight_fraction = 0.0;
    double fit_l2_error = 0.0;
    double pruned_fit_l2_error = 0.0;
    double drift_delta_mcap = 0.0;
    double drift_delta_mscale = 0.0;
    double drift_delta_deff = 0.0;
};

struct HierarchicalScenarioRegressionReference {
    HierarchicalScenario scenario;
    std::size_t steps = 0;
    std::size_t sample_stride = 1;
    unsigned long long seed = 0ULL;
    bool enable_bottom_up = true;
    bool enable_top_down = true;
    HierarchicalRegressionSnapshot baseline;
    HierarchicalRegressionTolerance tolerance;
    HierarchicalCrossLevelRegressionSnapshot cross_level_baseline;
    HierarchicalCrossLevelRegressionTolerance cross_level_tolerance;
    HierarchicalRenormRegressionSnapshot renorm_baseline;
    HierarchicalRenormRegressionTolerance renorm_tolerance;
};

struct HierarchicalKernelShiftSummary {
    std::size_t level_idx = 0;
    std::string level_name;
    double delta_mcap = 0.0;
    double delta_mscale = 0.0;
    double delta_hmem = 0.0;
    double delta_deff = 0.0;
    bool changed = false;
};

struct HierarchicalCrossLevelSummary {
    std::string relation;
    std::size_t source_level = 0;
    std::size_t target_level = 0;
    std::string source_name;
    std::string target_name;
    double delta_mcap = 0.0;
    double delta_mscale = 0.0;
    double delta_hmem = 0.0;
    double delta_deff = 0.0;
    bool target_mcap_not_lower = true;
    bool target_mscale_not_lower = true;
};

struct HierarchicalWarning {
    std::string severity;
    std::string code;
    std::string message;
    std::string metric;
    double observed = 0.0;
    double threshold = 0.0;
    std::size_t source_level = static_cast<std::size_t>(-1);
    std::size_t target_level = static_cast<std::size_t>(-1);
    bool actionable = false;
    std::string suggested_action;
};

struct HierarchicalCrossLevelContractOptions {
    double kernel_shift_mscale_warn_threshold = 0.35;
    double kernel_shift_deff_warn_threshold = 1.5;
    double top_down_mscale_info_threshold = 0.25;
    double top_down_deff_warn_threshold = 2.0;
    bool require_bottom_up_mcap_nondecrease = true;
    bool require_bottom_up_mscale_nondecrease = false;
    bool require_all_finite = true;
};

struct HierarchicalCrossLevelContractReport {
    bool pass = true;
    bool all_finite_pass = true;
    bool bottom_up_mcap_pass = true;
    bool bottom_up_mscale_pass = true;
    std::size_t actionable_warning_count = 0;
    std::vector<std::string> failures;
};

struct HierarchicalCrossLevelReport {
    std::vector<HierarchicalKernelShiftSummary> kernel_shifts;
    std::vector<HierarchicalCrossLevelSummary> relations;
    std::vector<HierarchicalWarning> warnings;
    HierarchicalCrossLevelContractReport contract;
};

struct HierarchicalAdjacentRenormAnalysis {
    std::size_t input_level = 0;
    std::size_t bath_level = 0;
    std::size_t direct_level = static_cast<std::size_t>(-1);
    std::string input_name;
    std::string bath_name;
    std::string direct_name;
    HierarchicalRenormReport report;
};

HierarchicalRegressionSnapshot make_regression_snapshot(const HierarchicalRunResult& run);
HierarchicalRegressionCheckResult check_regression_snapshot(
    const HierarchicalRegressionSnapshot& observed,
    const HierarchicalRegressionSnapshot& baseline,
    const HierarchicalRegressionTolerance& tolerance);
HierarchicalCrossLevelRegressionSnapshot make_cross_level_regression_snapshot(
    const HierarchicalCrossLevelReport& report);
HierarchicalRegressionCheckResult check_cross_level_regression_snapshot(
    const HierarchicalCrossLevelRegressionSnapshot& observed,
    const HierarchicalCrossLevelRegressionSnapshot& baseline,
    const HierarchicalCrossLevelRegressionTolerance& tolerance);
HierarchicalRenormRegressionSnapshot make_renorm_regression_snapshot(
    const std::vector<HierarchicalAdjacentRenormAnalysis>& analyses);
HierarchicalRegressionCheckResult check_renorm_regression_snapshot(
    const HierarchicalRenormRegressionSnapshot& observed,
    const HierarchicalRenormRegressionSnapshot& baseline,
    const HierarchicalRenormRegressionTolerance& tolerance);
HierarchicalScenarioRegressionReference default_hierarchical_regression_reference(HierarchicalScenario scenario);
HierarchicalCrossLevelReport make_hierarchical_cross_level_report(
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run);
HierarchicalCrossLevelContractReport evaluate_hierarchical_cross_level_contract(
    const HierarchicalCrossLevelReport& report,
    const HierarchicalCrossLevelContractOptions& options = {});
std::vector<HierarchicalAdjacentRenormAnalysis> make_hierarchical_adjacent_renorm_analyses(
    const HierarchicalMINModel& model,
    const HierarchicalRenormOptions& options = {});

void write_hierarchy_run_csv(
    const std::string& path,
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run);
std::string hierarchy_run_csv_header();
void write_hierarchy_summary_csv(
    const std::string& path,
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run);
std::string hierarchy_summary_csv_header();
void write_hierarchy_diagnostics_csv(
    const std::string& path,
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run);
std::string hierarchy_diagnostics_csv_header();
void write_hierarchy_cross_level_csv(
    const std::string& path,
    const HierarchicalCrossLevelReport& report);
std::string hierarchy_cross_level_csv_header();
void write_hierarchy_renorm_csv(
    const std::string& path,
    const std::vector<HierarchicalAdjacentRenormAnalysis>& analyses);
std::string hierarchy_renorm_csv_header();
void write_hierarchical_regression_snapshot_csv(
    const std::string& path,
    const HierarchicalRegressionSnapshot& snapshot);
std::string hierarchy_regression_snapshot_csv_header();

} // namespace abersoe
