#include "abersoe/hierarchical_min_diagnostics.hpp"

#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace abersoe {
namespace {

double l2_norm(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) {
        s += v * v;
    }
    return std::sqrt(s);
}

void check_scalar(
    const char* name,
    double observed,
    double baseline,
    double tol,
    HierarchicalRegressionCheckResult& out) {
    if (std::fabs(observed - baseline) > tol) {
        out.pass = false;
        out.failures.push_back(
            std::string(name) + " drift: observed=" + std::to_string(observed) +
            " baseline=" + std::to_string(baseline) + " tol=" + std::to_string(tol));
    }
}

void check_vector(
    const std::vector<double>& observed,
    const std::vector<double>& baseline,
    double tol,
    const std::string& name,
    HierarchicalRegressionCheckResult& out) {
    if (observed.size() != baseline.size()) {
        out.pass = false;
        out.failures.push_back(name + " size mismatch");
        return;
    }
    for (std::size_t i = 0; i < observed.size(); ++i) {
        std::ostringstream label;
        label << name << "[" << i << "]";
        check_scalar(label.str().c_str(), observed[i], baseline[i], tol, out);
    }
}

void check_string_vector(
    const std::vector<std::string>& observed,
    const std::vector<std::string>& baseline,
    const std::string& name,
    HierarchicalRegressionCheckResult& out) {
    if (observed.size() != baseline.size()) {
        out.pass = false;
        out.failures.push_back(name + " size mismatch");
        return;
    }
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] != baseline[i]) {
            out.pass = false;
            out.failures.push_back(
                name + "[" + std::to_string(i) + "] mismatch: observed=" + observed[i] +
                " baseline=" + baseline[i]);
        }
    }
}

void check_size_vector(
    const std::vector<std::size_t>& observed,
    const std::vector<std::size_t>& baseline,
    const std::string& name,
    HierarchicalRegressionCheckResult& out) {
    if (observed.size() != baseline.size()) {
        out.pass = false;
        out.failures.push_back(name + " size mismatch");
        return;
    }
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] != baseline[i]) {
            out.pass = false;
            out.failures.push_back(
                name + "[" + std::to_string(i) + "] mismatch: observed=" + std::to_string(observed[i]) +
                " baseline=" + std::to_string(baseline[i]));
        }
    }
}

void check_int_vector(
    const std::vector<int>& observed,
    const std::vector<int>& baseline,
    const std::string& name,
    HierarchicalRegressionCheckResult& out) {
    if (observed.size() != baseline.size()) {
        out.pass = false;
        out.failures.push_back(name + " size mismatch");
        return;
    }
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] != baseline[i]) {
            out.pass = false;
            out.failures.push_back(
                name + "[" + std::to_string(i) + "] mismatch: observed=" + std::to_string(observed[i]) +
                " baseline=" + std::to_string(baseline[i]));
        }
    }
}

std::string level_name_or_fallback(const HierarchicalMINModel& model, std::size_t level_idx) {
    if (level_idx < model.levels.size() && !model.levels[level_idx].name.empty()) {
        return model.levels[level_idx].name;
    }
    return "level" + std::to_string(level_idx);
}

gfe::SpectralUnits spectral_or_zero(const gfe::MemoryKernelParams& kernel) {
    if (kernel.w.empty() || kernel.gamma.empty()) {
        return gfe::SpectralUnits{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    return gfe::spectral_units(kernel.w, kernel.gamma);
}

HierarchicalKernelShiftSummary make_kernel_shift_summary(
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run,
    std::size_t level_idx) {
    HierarchicalKernelShiftSummary summary;
    summary.level_idx = level_idx;
    summary.level_name = level_name_or_fallback(model, level_idx);
    const auto base = spectral_or_zero(model.levels[level_idx].model.kernel);
    const auto active = run.final_levels[level_idx].spectral;
    summary.delta_mcap = active.Mcap - base.Mcap;
    summary.delta_mscale = active.Mscale - base.Mscale;
    summary.delta_hmem = active.Hmem - base.Hmem;
    summary.delta_deff = active.Deff - base.Deff;
    summary.changed =
        std::fabs(summary.delta_mcap) > 1e-12 ||
        std::fabs(summary.delta_mscale) > 1e-12 ||
        std::fabs(summary.delta_hmem) > 1e-12 ||
        std::fabs(summary.delta_deff) > 1e-12;
    return summary;
}

HierarchicalCrossLevelSummary make_relation_summary(
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run,
    const char* relation,
    std::size_t source_level,
    std::size_t target_level) {
    const auto& source = run.final_levels[source_level].spectral;
    const auto& target = run.final_levels[target_level].spectral;

    HierarchicalCrossLevelSummary summary;
    summary.relation = relation;
    summary.source_level = source_level;
    summary.target_level = target_level;
    summary.source_name = level_name_or_fallback(model, source_level);
    summary.target_name = level_name_or_fallback(model, target_level);
    summary.delta_mcap = target.Mcap - source.Mcap;
    summary.delta_mscale = target.Mscale - source.Mscale;
    summary.delta_hmem = target.Hmem - source.Hmem;
    summary.delta_deff = target.Deff - source.Deff;
    summary.target_mcap_not_lower = target.Mcap + 1e-9 >= source.Mcap;
    summary.target_mscale_not_lower = target.Mscale + 1e-9 >= source.Mscale;
    return summary;
}

void push_warning(
    HierarchicalCrossLevelReport& report,
    const std::string& severity,
    const std::string& code,
    const std::string& message,
    const std::string& metric = "",
    double observed = 0.0,
    double threshold = 0.0,
    std::size_t source_level = static_cast<std::size_t>(-1),
    std::size_t target_level = static_cast<std::size_t>(-1),
    bool actionable = false,
    const std::string& suggested_action = "") {
    report.warnings.push_back(HierarchicalWarning{
        severity,
        code,
        message,
        metric,
        observed,
        threshold,
        source_level,
        target_level,
        actionable,
        suggested_action});
}

void push_contract_failure(
    HierarchicalCrossLevelContractReport& report,
    const std::string& failure) {
    report.pass = false;
    report.failures.push_back(failure);
}

} // namespace

HierarchicalRegressionSnapshot make_regression_snapshot(const HierarchicalRunResult& run) {
    HierarchicalRegressionSnapshot snapshot;
    snapshot.levels = run.final_levels.size();
    snapshot.steps_executed = run.diagnostics.steps_executed;
    snapshot.all_finite = run.diagnostics.all_finite;

    snapshot.final_u_l2_per_level.reserve(run.final_levels.size());
    snapshot.final_chi_l2_per_level.reserve(run.final_levels.size());
    snapshot.first_weight_per_level.reserve(run.final_levels.size());
    snapshot.first_rate_per_level.reserve(run.final_levels.size());
    snapshot.deff_per_level.reserve(run.final_levels.size());
    snapshot.mcap_per_level.reserve(run.final_levels.size());
    snapshot.mscale_per_level.reserve(run.final_levels.size());

    for (const auto& level : run.final_levels) {
        snapshot.total_state_dim += level.state.u.size();
        snapshot.total_memory_channels += level.state.chi.size();
        snapshot.final_u_l2_per_level.push_back(l2_norm(level.state.u));
        snapshot.final_chi_l2_per_level.push_back(l2_norm(level.state.chi));
        snapshot.first_weight_per_level.push_back(level.active_kernel.w.empty() ? 0.0 : level.active_kernel.w.front());
        snapshot.first_rate_per_level.push_back(level.active_kernel.gamma.empty() ? 0.0 : level.active_kernel.gamma.front());
        snapshot.deff_per_level.push_back(level.spectral.Deff);
        snapshot.mcap_per_level.push_back(level.spectral.Mcap);
        snapshot.mscale_per_level.push_back(level.spectral.Mscale);
    }

    return snapshot;
}

HierarchicalRegressionCheckResult check_regression_snapshot(
    const HierarchicalRegressionSnapshot& observed,
    const HierarchicalRegressionSnapshot& baseline,
    const HierarchicalRegressionTolerance& tolerance) {
    HierarchicalRegressionCheckResult out;

    if (observed.levels != baseline.levels) {
        out.pass = false;
        out.failures.push_back("levels mismatch");
    }
    if (observed.total_state_dim != baseline.total_state_dim) {
        out.pass = false;
        out.failures.push_back("total_state_dim mismatch");
    }
    if (observed.total_memory_channels != baseline.total_memory_channels) {
        out.pass = false;
        out.failures.push_back("total_memory_channels mismatch");
    }
    if (observed.steps_executed != baseline.steps_executed) {
        out.pass = false;
        out.failures.push_back("steps_executed mismatch");
    }
    if (!observed.all_finite) {
        out.pass = false;
        out.failures.push_back("observed run is non-finite");
    }

    check_vector(observed.final_u_l2_per_level, baseline.final_u_l2_per_level, tolerance.final_u_l2, "final_u_l2_per_level", out);
    check_vector(observed.final_chi_l2_per_level, baseline.final_chi_l2_per_level, tolerance.final_chi_l2, "final_chi_l2_per_level", out);
    check_vector(observed.first_weight_per_level, baseline.first_weight_per_level, tolerance.first_weight, "first_weight_per_level", out);
    check_vector(observed.first_rate_per_level, baseline.first_rate_per_level, tolerance.first_rate, "first_rate_per_level", out);
    check_vector(observed.deff_per_level, baseline.deff_per_level, tolerance.deff, "deff_per_level", out);
    check_vector(observed.mcap_per_level, baseline.mcap_per_level, tolerance.mcap, "mcap_per_level", out);
    check_vector(observed.mscale_per_level, baseline.mscale_per_level, tolerance.mscale, "mscale_per_level", out);

    return out;
}

HierarchicalCrossLevelRegressionSnapshot make_cross_level_regression_snapshot(
    const HierarchicalCrossLevelReport& report) {
    HierarchicalCrossLevelRegressionSnapshot snapshot;
    snapshot.kernel_shift_count = report.kernel_shifts.size();
    snapshot.relation_count = report.relations.size();
    snapshot.warning_count = report.warnings.size();
    snapshot.actionable_warning_count = report.contract.actionable_warning_count;
    snapshot.contract_pass = report.contract.pass;
    snapshot.all_finite_pass = report.contract.all_finite_pass;
    snapshot.bottom_up_mcap_pass = report.contract.bottom_up_mcap_pass;
    snapshot.bottom_up_mscale_pass = report.contract.bottom_up_mscale_pass;
    snapshot.kernel_shift_delta_mcap.reserve(report.kernel_shifts.size());
    snapshot.kernel_shift_delta_mscale.reserve(report.kernel_shifts.size());
    snapshot.kernel_shift_delta_deff.reserve(report.kernel_shifts.size());
    for (const auto& shift : report.kernel_shifts) {
        snapshot.kernel_shift_delta_mcap.push_back(shift.delta_mcap);
        snapshot.kernel_shift_delta_mscale.push_back(shift.delta_mscale);
        snapshot.kernel_shift_delta_deff.push_back(shift.delta_deff);
    }
    snapshot.relation_delta_mcap.reserve(report.relations.size());
    snapshot.relation_delta_mscale.reserve(report.relations.size());
    snapshot.relation_delta_deff.reserve(report.relations.size());
    for (const auto& relation : report.relations) {
        snapshot.relation_delta_mcap.push_back(relation.delta_mcap);
        snapshot.relation_delta_mscale.push_back(relation.delta_mscale);
        snapshot.relation_delta_deff.push_back(relation.delta_deff);
    }
    snapshot.warning_codes.reserve(report.warnings.size());
    for (const auto& warning : report.warnings) {
        snapshot.warning_codes.push_back(warning.code);
    }
    return snapshot;
}

HierarchicalRegressionCheckResult check_cross_level_regression_snapshot(
    const HierarchicalCrossLevelRegressionSnapshot& observed,
    const HierarchicalCrossLevelRegressionSnapshot& baseline,
    const HierarchicalCrossLevelRegressionTolerance& tolerance) {
    HierarchicalRegressionCheckResult out;
    if (observed.kernel_shift_count != baseline.kernel_shift_count) {
        out.pass = false;
        out.failures.push_back("kernel_shift_count mismatch");
    }
    if (observed.relation_count != baseline.relation_count) {
        out.pass = false;
        out.failures.push_back("relation_count mismatch");
    }
    if (observed.warning_count != baseline.warning_count) {
        out.pass = false;
        out.failures.push_back("warning_count mismatch");
    }
    if (observed.actionable_warning_count != baseline.actionable_warning_count) {
        out.pass = false;
        out.failures.push_back("actionable_warning_count mismatch");
    }
    if (observed.contract_pass != baseline.contract_pass) {
        out.pass = false;
        out.failures.push_back("contract_pass mismatch");
    }
    if (observed.all_finite_pass != baseline.all_finite_pass) {
        out.pass = false;
        out.failures.push_back("all_finite_pass mismatch");
    }
    if (observed.bottom_up_mcap_pass != baseline.bottom_up_mcap_pass) {
        out.pass = false;
        out.failures.push_back("bottom_up_mcap_pass mismatch");
    }
    if (observed.bottom_up_mscale_pass != baseline.bottom_up_mscale_pass) {
        out.pass = false;
        out.failures.push_back("bottom_up_mscale_pass mismatch");
    }
    check_vector(
        observed.kernel_shift_delta_mcap,
        baseline.kernel_shift_delta_mcap,
        tolerance.kernel_shift_delta_mcap,
        "kernel_shift_delta_mcap",
        out);
    check_vector(
        observed.kernel_shift_delta_mscale,
        baseline.kernel_shift_delta_mscale,
        tolerance.kernel_shift_delta_mscale,
        "kernel_shift_delta_mscale",
        out);
    check_vector(
        observed.kernel_shift_delta_deff,
        baseline.kernel_shift_delta_deff,
        tolerance.kernel_shift_delta_deff,
        "kernel_shift_delta_deff",
        out);
    check_vector(
        observed.relation_delta_mcap,
        baseline.relation_delta_mcap,
        tolerance.relation_delta_mcap,
        "relation_delta_mcap",
        out);
    check_vector(
        observed.relation_delta_mscale,
        baseline.relation_delta_mscale,
        tolerance.relation_delta_mscale,
        "relation_delta_mscale",
        out);
    check_vector(
        observed.relation_delta_deff,
        baseline.relation_delta_deff,
        tolerance.relation_delta_deff,
        "relation_delta_deff",
        out);
    check_string_vector(observed.warning_codes, baseline.warning_codes, "warning_codes", out);
    return out;
}

HierarchicalRenormRegressionSnapshot make_renorm_regression_snapshot(
    const std::vector<HierarchicalAdjacentRenormAnalysis>& analyses) {
    HierarchicalRenormRegressionSnapshot snapshot;
    snapshot.analysis_count = analyses.size();
    snapshot.diagnostic_names.reserve(analyses.size());
    snapshot.retained_weight_fraction.reserve(analyses.size());
    snapshot.fit_l2_error.reserve(analyses.size());
    snapshot.pruned_fit_l2_error.reserve(analyses.size());
    snapshot.drift_delta_mcap.reserve(analyses.size());
    snapshot.drift_delta_mscale.reserve(analyses.size());
    snapshot.drift_delta_deff.reserve(analyses.size());
    snapshot.raw_output_channels.reserve(analyses.size());
    snapshot.output_channels.reserve(analyses.size());
    snapshot.pruned_channels.reserve(analyses.size());
    snapshot.consistency_pass.reserve(analyses.size());
    snapshot.mcap_pass.reserve(analyses.size());
    snapshot.mscale_pass.reserve(analyses.size());
    snapshot.deff_pass.reserve(analyses.size());
    for (const auto& analysis : analyses) {
        snapshot.diagnostic_names.push_back(
            hierarchical_renorm_diagnostic_cause_name(analysis.report.primary_diagnostic));
        snapshot.retained_weight_fraction.push_back(analysis.report.retained_weight_fraction);
        snapshot.fit_l2_error.push_back(analysis.report.fit_l2_error);
        snapshot.pruned_fit_l2_error.push_back(analysis.report.pruned_fit_l2_error);
        snapshot.drift_delta_mcap.push_back(analysis.report.input_to_output_pruned_drift.delta_mcap);
        snapshot.drift_delta_mscale.push_back(analysis.report.input_to_output_pruned_drift.delta_mscale);
        snapshot.drift_delta_deff.push_back(analysis.report.input_to_output_pruned_drift.delta_deff);
        snapshot.raw_output_channels.push_back(analysis.report.raw_output_channels);
        snapshot.output_channels.push_back(analysis.report.output_channels);
        snapshot.pruned_channels.push_back(analysis.report.pruned_channels);
        snapshot.consistency_pass.push_back(analysis.report.consistency.pass ? 1 : 0);
        snapshot.mcap_pass.push_back(analysis.report.consistency.mcap_nondecrease_pass ? 1 : 0);
        snapshot.mscale_pass.push_back(analysis.report.consistency.mscale_nondecrease_pass ? 1 : 0);
        snapshot.deff_pass.push_back(analysis.report.consistency.deff_bound_pass ? 1 : 0);
    }
    return snapshot;
}

HierarchicalRegressionCheckResult check_renorm_regression_snapshot(
    const HierarchicalRenormRegressionSnapshot& observed,
    const HierarchicalRenormRegressionSnapshot& baseline,
    const HierarchicalRenormRegressionTolerance& tolerance) {
    HierarchicalRegressionCheckResult out;
    if (observed.analysis_count != baseline.analysis_count) {
        out.pass = false;
        out.failures.push_back("analysis_count mismatch");
    }
    check_string_vector(observed.diagnostic_names, baseline.diagnostic_names, "diagnostic_names", out);
    check_vector(
        observed.retained_weight_fraction,
        baseline.retained_weight_fraction,
        tolerance.retained_weight_fraction,
        "retained_weight_fraction",
        out);
    check_vector(observed.fit_l2_error, baseline.fit_l2_error, tolerance.fit_l2_error, "fit_l2_error", out);
    check_vector(
        observed.pruned_fit_l2_error,
        baseline.pruned_fit_l2_error,
        tolerance.pruned_fit_l2_error,
        "pruned_fit_l2_error",
        out);
    check_vector(
        observed.drift_delta_mcap,
        baseline.drift_delta_mcap,
        tolerance.drift_delta_mcap,
        "drift_delta_mcap",
        out);
    check_vector(
        observed.drift_delta_mscale,
        baseline.drift_delta_mscale,
        tolerance.drift_delta_mscale,
        "drift_delta_mscale",
        out);
    check_vector(
        observed.drift_delta_deff,
        baseline.drift_delta_deff,
        tolerance.drift_delta_deff,
        "drift_delta_deff",
        out);
    check_size_vector(observed.raw_output_channels, baseline.raw_output_channels, "raw_output_channels", out);
    check_size_vector(observed.output_channels, baseline.output_channels, "output_channels", out);
    check_size_vector(observed.pruned_channels, baseline.pruned_channels, "pruned_channels", out);
    check_int_vector(observed.consistency_pass, baseline.consistency_pass, "consistency_pass", out);
    check_int_vector(observed.mcap_pass, baseline.mcap_pass, "mcap_pass", out);
    check_int_vector(observed.mscale_pass, baseline.mscale_pass, "mscale_pass", out);
    check_int_vector(observed.deff_pass, baseline.deff_pass, "deff_pass", out);
    return out;
}

HierarchicalScenarioRegressionReference default_hierarchical_regression_reference(HierarchicalScenario scenario) {
    if (scenario == HierarchicalScenario::TwoLevelBidirectional) {
        HierarchicalScenarioRegressionReference ref{
            scenario,
            80,
            10,
            123456789ULL,
            true,
            true,
            HierarchicalRegressionSnapshot{
                2,
                2,
                4,
                80,
                true,
                {0.050917823030476969, 0.10952620084458638},
                {0.49745712040940815, 0.41934046300062194},
                {0.51148287902333556, 0.55000000000000004},
                {1.1505005914602175, 0.5},
                {3.9989451942703824, 3.9800166164438906},
                {3.5594081004875266, 8.4285714285714288},
                {0.67887477698154464, 0.69897000433601886}},
            HierarchicalRegressionTolerance{1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4},
            HierarchicalCrossLevelRegressionSnapshot{},
            HierarchicalCrossLevelRegressionTolerance{1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4},
            HierarchicalRenormRegressionSnapshot{},
            HierarchicalRenormRegressionTolerance{1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4}};
        ref.cross_level_baseline = HierarchicalCrossLevelRegressionSnapshot{
            2, 2, 1, 0, true, true, true, true,
            {0.37759017300439138, 0.0},
            {0.076814785653257163, 0.0},
            {0.078681807699261302, 0.0},
            {4.8691627634499022, -4.8691627634499022},
            {0.020095227354474216, -0.020095227354474216},
            {-0.018928577826491826, 0.018928577826491826},
            {"kernel_shift_detected"}};
        ref.renorm_baseline = HierarchicalRenormRegressionSnapshot{
            1,
            {"fit_instability"},
            {1.0},
            {0.10110237050965209},
            {0.10110237050965209},
            {11.716879142193794},
            {-0.38777364663092742},
            {-1.3273421820663497},
            {2},
            {2},
            {0},
            {1},
            {1},
            {1},
            {1}};
        return ref;
    }
    if (scenario == HierarchicalScenario::ThreeLevelHomeostatic) {
        HierarchicalScenarioRegressionReference ref{
            scenario,
            120,
            12,
            123456789ULL,
            true,
            true,
            HierarchicalRegressionSnapshot{
                3,
                3,
                6,
                120,
                true,
                {0.16905411258713479, 0.068717116543771911, 0.021442130907091475},
                {0.30075319952770985, 0.18600790485954644, 0.091987999989447469},
                {0.49535900495473412, 0.4989904020288789, 0.57999999999999996},
                {0.93745511151323979, 0.57485665918535722, 0.12},
                {3.9998276905570087, 3.9999918456982626, 3.9489083781324061},
                {4.0052468605549283, 7.02587478352853, 26.917404129793514},
                {0.64078775032060942, 0.67070105177735773, 0.6020599913279624}},
            HierarchicalRegressionTolerance{1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4},
            HierarchicalCrossLevelRegressionSnapshot{},
            HierarchicalCrossLevelRegressionTolerance{1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4},
            HierarchicalRenormRegressionSnapshot{},
            HierarchicalRenormRegressionTolerance{1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4}};
        ref.cross_level_baseline = HierarchicalCrossLevelRegressionSnapshot{
            3, 4, 3, 1, true, true, true, false,
            {0.82342867873674674, -3.5602765241643278, 0.0},
            {0.038727759329064776, 0.029722961614831202, 0.0},
            {0.07956428001029777, 0.0031914223860935488, 0.0},
            {3.0206279229736018, 19.891529346264984, -3.0206279229736018, -19.891529346264984},
            {0.029913301456748312, -0.068641060449395327, -0.029913301456748312, 0.068641060449395327},
            {0.00016415514125387791, -0.051083467565856457, -0.00016415514125387791, 0.051083467565856457},
            {"kernel_shift_detected", "kernel_shift_detected", "bottom_up_mscale_inversion"}};
        ref.renorm_baseline = HierarchicalRenormRegressionSnapshot{
            2,
            {"fit_instability", "fit_instability"},
            {1.0, 1.0},
            {0.074931685811017864, 0.37670852608973159},
            {0.074947753960315942, 0.37670852608973159},
            {16.477604170240225, 42.068501634315203},
            {-0.38777364663092742, -0.42669171499164084},
            {0.071667625490510506, -0.0029920729236360634},
            {2, 2},
            {2, 2},
            {0, 0},
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}};
        return ref;
    }

    throw std::invalid_argument("No default hierarchical regression reference for scenario");
}

HierarchicalCrossLevelReport make_hierarchical_cross_level_report(
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run) {
    if (model.levels.size() != run.final_levels.size()) {
        throw std::invalid_argument("model/run level count mismatch in make_hierarchical_cross_level_report");
    }

    HierarchicalCrossLevelReport report;
    for (std::size_t level_idx = 0; level_idx < run.final_levels.size(); ++level_idx) {
        report.kernel_shifts.push_back(make_kernel_shift_summary(model, run, level_idx));
    }

    for (const auto& edge : model.bottom_up_couplings) {
        report.relations.push_back(
            make_relation_summary(model, run, "bottom_up", edge.source_level, edge.target_level));
    }
    for (const auto& edge : model.top_down_couplings) {
        report.relations.push_back(
            make_relation_summary(model, run, "top_down", edge.source_level, edge.target_level));
    }

    if (!run.diagnostics.all_finite) {
        push_warning(
            report,
            "error",
            "non_finite_run",
            "run diagnostics reported non-finite values",
            "all_finite",
            0.0,
            1.0,
            static_cast<std::size_t>(-1),
            static_cast<std::size_t>(-1),
            true,
            "reduce step size, inspect forcing/modulation outputs, and rerun in strict finite mode");
    }

    for (const auto& shift : report.kernel_shifts) {
        if (shift.changed) {
            push_warning(
                report,
                "info",
                "kernel_shift_detected",
                "level " + shift.level_name + " active kernel differs from the base scenario kernel",
                "kernel_shift",
                1.0,
                0.0,
                shift.level_idx,
                shift.level_idx,
                false,
                "");
        }
        if (std::fabs(shift.delta_mscale) > 0.35 || std::fabs(shift.delta_deff) > 1.5) {
            push_warning(
                report,
                "warning",
                "large_kernel_shift",
                "level " + shift.level_name + " shows a large active-kernel spectral shift",
                std::fabs(shift.delta_mscale) > 0.35 ? "delta_mscale" : "delta_deff",
                std::fabs(shift.delta_mscale) > 0.35 ? std::fabs(shift.delta_mscale) : std::fabs(shift.delta_deff),
                std::fabs(shift.delta_mscale) > 0.35 ? 0.35 : 1.5,
                shift.level_idx,
                shift.level_idx,
                true,
                "inspect top-down modulation strength or prune/compress policy for this level");
        }
    }

    for (const auto& relation : report.relations) {
        if (relation.relation == "bottom_up" && !relation.target_mcap_not_lower) {
            push_warning(
                report,
                "warning",
                "bottom_up_mcap_inversion",
                relation.target_name + " has lower Mcap than bottom-up source " + relation.source_name,
                "delta_mcap",
                relation.delta_mcap,
                0.0,
                relation.source_level,
                relation.target_level,
                true,
                "check whether the slow level kernel, forcing map, or timescale assumptions are inverted");
        }
        if (relation.relation == "bottom_up" && !relation.target_mscale_not_lower) {
            push_warning(
                report,
                "warning",
                "bottom_up_mscale_inversion",
                relation.target_name + " has lower Mscale than bottom-up source " + relation.source_name,
                "delta_mscale",
                relation.delta_mscale,
                0.0,
                relation.source_level,
                relation.target_level,
                true,
                "check scale ordering, kernel bandwidths, and whether this hierarchy should really coarse-grain upward");
        }
        if (relation.relation == "top_down" && std::fabs(relation.delta_mscale) > 0.25) {
            push_warning(
                report,
                "info",
                "top_down_timescale_shift",
                relation.source_name + " induces a visible Mscale shift on " + relation.target_name,
                "delta_mscale",
                std::fabs(relation.delta_mscale),
                0.25,
                relation.source_level,
                relation.target_level,
                false,
                "");
        }
        if (relation.relation == "top_down" && std::fabs(relation.delta_deff) > 2.0) {
            push_warning(
                report,
                "warning",
                "top_down_deff_shift",
                relation.source_name + " induces a large Deff shift on " + relation.target_name,
                "delta_deff",
                std::fabs(relation.delta_deff),
                2.0,
                relation.source_level,
                relation.target_level,
                true,
                "regularize the modulation map or widen the target kernel basis before trusting this coupling");
        }
    }

    report.contract = evaluate_hierarchical_cross_level_contract(report);
    return report;
}

HierarchicalCrossLevelContractReport evaluate_hierarchical_cross_level_contract(
    const HierarchicalCrossLevelReport& report,
    const HierarchicalCrossLevelContractOptions& options) {
    HierarchicalCrossLevelContractReport out;
    for (const auto& warning : report.warnings) {
        if (warning.actionable) {
            out.actionable_warning_count += 1;
        }
        if (warning.code == "non_finite_run") {
            out.all_finite_pass = false;
            if (options.require_all_finite) {
                push_contract_failure(out, warning.message);
            }
        } else if (warning.code == "bottom_up_mcap_inversion") {
            out.bottom_up_mcap_pass = false;
            if (options.require_bottom_up_mcap_nondecrease) {
                push_contract_failure(out, warning.message);
            }
        } else if (warning.code == "bottom_up_mscale_inversion") {
            out.bottom_up_mscale_pass = false;
            if (options.require_bottom_up_mscale_nondecrease) {
                push_contract_failure(out, warning.message);
            }
        }
    }
    return out;
}

std::vector<HierarchicalAdjacentRenormAnalysis> make_hierarchical_adjacent_renorm_analyses(
    const HierarchicalMINModel& model,
    const HierarchicalRenormOptions& options) {
    std::vector<HierarchicalAdjacentRenormAnalysis> analyses;
    if (model.levels.size() < 2) {
        return analyses;
    }

    for (std::size_t level_idx = 0; level_idx + 1 < model.levels.size(); ++level_idx) {
        HierarchicalAdjacentRenormAnalysis analysis;
        analysis.input_level = level_idx;
        analysis.bath_level = level_idx + 1;
        analysis.input_name = level_name_or_fallback(model, level_idx);
        analysis.bath_name = level_name_or_fallback(model, level_idx + 1);

        const gfe::MemoryKernelParams* direct_kernel = nullptr;
        if (level_idx + 2 < model.levels.size()) {
            analysis.direct_level = level_idx + 2;
            analysis.direct_name = level_name_or_fallback(model, level_idx + 2);
            direct_kernel = &model.levels[level_idx + 2].model.kernel;
        }

        analysis.report = build_renormalized_kernel(
            model.levels[level_idx].model.kernel,
            model.levels[level_idx + 1].model.kernel,
            direct_kernel,
            options);
        analyses.push_back(analysis);
    }

    return analyses;
}

std::string hierarchy_run_csv_header() {
    return "sample_idx,level_idx,level_name,t,u0,chi0,u_l2,chi_l2,w0,gamma0,Mcap,Mscale,Hmem,Deff";
}

std::string hierarchy_summary_csv_header() {
    return "level_idx,level_name,t,u0,chi0,u_l2,chi_l2,w0,gamma0,Mcap,Mscale,Hmem,Deff,steps_executed,all_finite";
}

std::string hierarchy_diagnostics_csv_header() {
    return "level_idx,level_name,steps_executed,all_finite,max_abs_u,max_abs_chi,final_u_l2,final_chi_l2,mean_abs_u,mean_abs_chi,w0,gamma0,Mcap,Mscale,Hmem,Deff";
}

std::string hierarchy_cross_level_csv_header() {
    return "record_type,relation,source_level,target_level,source_name,target_name,delta_mcap,delta_mscale,delta_hmem,delta_deff,status_a,status_b,severity,code,message,metric,observed,threshold,actionable,suggested_action";
}

std::string hierarchy_renorm_csv_header() {
    return "input_level,bath_level,direct_level,input_name,bath_name,direct_name,primary_diagnostic,raw_output_channels,output_channels,pruned_channels,retained_weight_fraction,fit_l2_error,pruned_fit_l2_error,fit_linf_error,pruned_fit_linf_error,delta_mcap,delta_mscale,delta_deff,consistency_pass,mcap_pass,mscale_pass,deff_pass";
}

std::string hierarchy_regression_snapshot_csv_header() {
    return "level_idx,levels,total_state_dim,total_memory_channels,steps_executed,all_finite,final_u_l2,final_chi_l2,first_weight,first_rate,Deff,Mcap,Mscale";
}

void write_hierarchy_run_csv(
    const std::string& path,
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open hierarchy CSV path: " + path);
    }

    out << hierarchy_run_csv_header() << '\n';
    for (std::size_t sample_idx = 0; sample_idx < run.samples.size(); ++sample_idx) {
        const auto& sample = run.samples[sample_idx];
        for (std::size_t level_idx = 0; level_idx < sample.levels.size(); ++level_idx) {
            const auto& level = sample.levels[level_idx];
            const std::string name = level_name_or_fallback(model, level_idx);
            const double u0 = level.state.u.empty() ? 0.0 : level.state.u[0];
            const double chi0 = level.state.chi.empty() ? 0.0 : level.state.chi[0];
            const double w0 = level.active_kernel.w.empty() ? 0.0 : level.active_kernel.w[0];
            const double gamma0 = level.active_kernel.gamma.empty() ? 0.0 : level.active_kernel.gamma[0];
            out << sample_idx << ','
                << level_idx << ','
                << name << ','
                << std::setprecision(17) << level.state.t << ','
                << u0 << ','
                << chi0 << ','
                << l2_norm(level.state.u) << ','
                << l2_norm(level.state.chi) << ','
                << w0 << ','
                << gamma0 << ','
                << level.spectral.Mcap << ','
                << level.spectral.Mscale << ','
                << level.spectral.Hmem << ','
                << level.spectral.Deff << '\n';
        }
    }
}

void write_hierarchy_summary_csv(
    const std::string& path,
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open hierarchy summary CSV path: " + path);
    }

    out << hierarchy_summary_csv_header() << '\n';
    for (std::size_t level_idx = 0; level_idx < run.final_levels.size(); ++level_idx) {
        const auto& level = run.final_levels[level_idx];
        const std::string name = level_name_or_fallback(model, level_idx);
        const double u0 = level.state.u.empty() ? 0.0 : level.state.u[0];
        const double chi0 = level.state.chi.empty() ? 0.0 : level.state.chi[0];
        const double w0 = level.active_kernel.w.empty() ? 0.0 : level.active_kernel.w[0];
        const double gamma0 = level.active_kernel.gamma.empty() ? 0.0 : level.active_kernel.gamma[0];
        out << level_idx << ','
            << name << ','
            << std::setprecision(17) << level.state.t << ','
            << u0 << ','
            << chi0 << ','
            << l2_norm(level.state.u) << ','
            << l2_norm(level.state.chi) << ','
            << w0 << ','
            << gamma0 << ','
            << level.spectral.Mcap << ','
            << level.spectral.Mscale << ','
            << level.spectral.Hmem << ','
            << level.spectral.Deff << ','
            << run.diagnostics.steps_executed << ','
            << (run.diagnostics.all_finite ? 1 : 0) << '\n';
    }
}

void write_hierarchy_diagnostics_csv(
    const std::string& path,
    const HierarchicalMINModel& model,
    const HierarchicalRunResult& run) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open hierarchy diagnostics CSV path: " + path);
    }

    out << hierarchy_diagnostics_csv_header() << '\n';
    for (std::size_t level_idx = 0; level_idx < run.final_levels.size(); ++level_idx) {
        const auto& level = run.final_levels[level_idx];
        const auto& diag = run.diagnostics.per_level[level_idx];
        const std::string name = level_name_or_fallback(model, level_idx);
        const double w0 = level.active_kernel.w.empty() ? 0.0 : level.active_kernel.w[0];
        const double gamma0 = level.active_kernel.gamma.empty() ? 0.0 : level.active_kernel.gamma[0];
        out << level_idx << ','
            << name << ','
            << diag.steps_executed << ','
            << (run.diagnostics.all_finite ? 1 : 0) << ','
            << std::setprecision(17) << diag.max_abs_u << ','
            << diag.max_abs_chi << ','
            << diag.final_u_l2 << ','
            << diag.final_chi_l2 << ','
            << diag.mean_abs_u << ','
            << diag.mean_abs_chi << ','
            << w0 << ','
            << gamma0 << ','
            << level.spectral.Mcap << ','
            << level.spectral.Mscale << ','
            << level.spectral.Hmem << ','
            << level.spectral.Deff << '\n';
    }
}

void write_hierarchy_cross_level_csv(
    const std::string& path,
    const HierarchicalCrossLevelReport& report) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open hierarchy cross-level CSV path: " + path);
    }

    out << hierarchy_cross_level_csv_header() << '\n';
    for (const auto& shift : report.kernel_shifts) {
        out << "kernel_shift,,"
            << shift.level_idx << ','
            << shift.level_idx << ','
            << shift.level_name << ','
            << shift.level_name << ','
            << std::setprecision(17) << shift.delta_mcap << ','
            << shift.delta_mscale << ','
            << shift.delta_hmem << ','
            << shift.delta_deff << ','
            << (shift.changed ? 1 : 0) << ','
            << 0 << ','
            << "info" << ','
            << "kernel_shift_detected" << ','
            << "base_to_active_kernel_shift,"
            << "kernel_shift,"
            << 1.0 << ','
            << 0.0 << ','
            << 0 << ','
            << '\n';
    }
    for (const auto& relation : report.relations) {
        out << "relation,"
            << relation.relation << ','
            << relation.source_level << ','
            << relation.target_level << ','
            << relation.source_name << ','
            << relation.target_name << ','
            << std::setprecision(17) << relation.delta_mcap << ','
            << relation.delta_mscale << ','
            << relation.delta_hmem << ','
            << relation.delta_deff << ','
            << (relation.target_mcap_not_lower ? 1 : 0) << ','
            << (relation.target_mscale_not_lower ? 1 : 0) << ','
            << "info" << ','
            << "cross_level_relation" << ','
            << "cross_level_relation,"
            << ","
            << 0.0 << ','
            << 0.0 << ','
            << 0 << ','
            << '\n';
    }
    for (const auto& warning : report.warnings) {
        out << "warning,,"
            << std::numeric_limits<std::size_t>::max() << ','
            << std::numeric_limits<std::size_t>::max() << ','
            << ",,"
            << 0.0 << ','
            << 0.0 << ','
            << 0.0 << ','
            << 0.0 << ','
            << 0 << ','
            << 0 << ','
            << warning.severity << ','
            << warning.code << ','
            << warning.message << ','
            << warning.metric << ','
            << warning.observed << ','
            << warning.threshold << ','
            << (warning.actionable ? 1 : 0) << ','
            << warning.suggested_action << '\n';
    }
}

void write_hierarchy_renorm_csv(
    const std::string& path,
    const std::vector<HierarchicalAdjacentRenormAnalysis>& analyses) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open hierarchy renorm CSV path: " + path);
    }

    out << hierarchy_renorm_csv_header() << '\n';
    for (const auto& analysis : analyses) {
        out << analysis.input_level << ','
            << analysis.bath_level << ','
            << analysis.direct_level << ','
            << analysis.input_name << ','
            << analysis.bath_name << ','
            << analysis.direct_name << ','
            << hierarchical_renorm_diagnostic_cause_name(analysis.report.primary_diagnostic) << ','
            << analysis.report.raw_output_channels << ','
            << analysis.report.output_channels << ','
            << analysis.report.pruned_channels << ','
            << std::setprecision(17) << analysis.report.retained_weight_fraction << ','
            << analysis.report.fit_l2_error << ','
            << analysis.report.pruned_fit_l2_error << ','
            << analysis.report.fit_linf_error << ','
            << analysis.report.pruned_fit_linf_error << ','
            << analysis.report.input_to_output_pruned_drift.delta_mcap << ','
            << analysis.report.input_to_output_pruned_drift.delta_mscale << ','
            << analysis.report.input_to_output_pruned_drift.delta_deff << ','
            << (analysis.report.consistency.pass ? 1 : 0) << ','
            << (analysis.report.consistency.mcap_nondecrease_pass ? 1 : 0) << ','
            << (analysis.report.consistency.mscale_nondecrease_pass ? 1 : 0) << ','
            << (analysis.report.consistency.deff_bound_pass ? 1 : 0) << '\n';
    }
}

void write_hierarchical_regression_snapshot_csv(
    const std::string& path,
    const HierarchicalRegressionSnapshot& snapshot) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open hierarchical regression snapshot CSV path: " + path);
    }

    out << hierarchy_regression_snapshot_csv_header() << '\n';
    for (std::size_t level_idx = 0; level_idx < snapshot.levels; ++level_idx) {
        out << level_idx << ','
            << snapshot.levels << ','
            << snapshot.total_state_dim << ','
            << snapshot.total_memory_channels << ','
            << snapshot.steps_executed << ','
            << (snapshot.all_finite ? 1 : 0) << ','
            << std::setprecision(17) << snapshot.final_u_l2_per_level[level_idx] << ','
            << snapshot.final_chi_l2_per_level[level_idx] << ','
            << snapshot.first_weight_per_level[level_idx] << ','
            << snapshot.first_rate_per_level[level_idx] << ','
            << snapshot.deff_per_level[level_idx] << ','
            << snapshot.mcap_per_level[level_idx] << ','
            << snapshot.mscale_per_level[level_idx] << '\n';
    }
}

} // namespace abersoe
