#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "abersoe/abersoe_runtime.hpp"
#include "gfe/gfe_core.hpp"
#include "gfe/gfe_dynamics.hpp"

namespace abersoe {

struct HierarchicalRenormOptions {
    std::size_t n_samples = 201;
    double t_max = 0.0;
    double degenerate_rate_tol = 1e-10;
    double prune_weight_threshold = 0.0;
    double max_fit_l2_error = 0.0;
    double max_pruned_fit_l2_error = 0.0;
    gfe::SoeFitOptions fitter;
};

struct HierarchicalRenormPlugbackResult {
    bool checked = false;
    bool all_finite = true;
    std::size_t steps_executed = 0;
    double final_u_l2 = 0.0;
    double final_chi_l2 = 0.0;
};

struct HierarchicalSpectralDrift {
    double delta_mcap = 0.0;
    double delta_mscale = 0.0;
    double delta_hmem = 0.0;
    double delta_deff = 0.0;
    double abs_delta_mcap = 0.0;
    double abs_delta_mscale = 0.0;
    double abs_delta_hmem = 0.0;
    double abs_delta_deff = 0.0;
};

struct HierarchicalRenormConsistencyOptions {
    double mcap_nondecrease_tol = 0.0;
    double mscale_nondecrease_tol = 0.0;
    double deff_bound_slack = 0.0;
    double min_retained_weight_fraction = 0.0;
    double max_fit_l2_error = 0.0;
    double max_pruned_fit_l2_error = 0.0;
    double max_fit_linf_error = 0.0;
    double max_pruned_fit_linf_error = 0.0;
    bool require_mcap_nondecrease = true;
    bool require_mscale_nondecrease = false;
    bool require_deff_submultiplicative_bound = true;
    bool require_sorted_output = true;
    bool require_nonempty_output = true;
    bool require_plugback_finite_when_checked = true;
    bool require_prune_drift_coherence = true;
};

enum class HierarchicalRenormDiagnosticCause {
    None,
    FitInstability,
    OverPruning,
    PlausibleChannelMerging,
};

std::string hierarchical_renorm_diagnostic_cause_name(HierarchicalRenormDiagnosticCause cause);

struct HierarchicalRenormConsistencyReport {
    bool pass = true;
    bool mcap_nondecrease_pass = true;
    bool mscale_nondecrease_pass = true;
    bool deff_bound_pass = true;
    bool retained_weight_pass = true;
    bool fit_error_pass = true;
    bool sorted_output_pass = true;
    bool nonempty_output_pass = true;
    bool plugback_finite_pass = true;
    bool prune_drift_coherence_pass = true;
    double mcap_input = 0.0;
    double mcap_output = 0.0;
    double mscale_input = 0.0;
    double mscale_output = 0.0;
    double deff_input = 0.0;
    double deff_bath = 0.0;
    double deff_output = 0.0;
    double deff_bound_rhs = 0.0;
    double retained_weight_fraction = 1.0;
    double fit_l2_error = 0.0;
    double pruned_fit_l2_error = 0.0;
    double fit_linf_error = 0.0;
    double pruned_fit_linf_error = 0.0;
    std::vector<std::string> failures;
};

struct HierarchicalRenormReport {
    gfe::MemoryKernelParams input_kernel;
    gfe::MemoryKernelParams bath_kernel;
    gfe::MemoryKernelParams direct_kernel;
    gfe::MemoryKernelParams output_kernel_raw;
    gfe::MemoryKernelParams output_kernel;
    std::size_t input_channels = 0;
    std::size_t bath_channels = 0;
    std::size_t direct_channels = 0;
    std::size_t raw_output_channels = 0;
    std::size_t output_channels = 0;
    std::size_t pruned_channels = 0;
    std::size_t degenerate_pairs = 0;
    std::size_t samples_used = 0;
    double fit_l2_error = 0.0;
    double pruned_fit_l2_error = 0.0;
    double fit_linf_error = 0.0;
    double pruned_fit_linf_error = 0.0;
    double retained_weight_fraction = 1.0;
    bool output_sorted_descending = true;
    bool prune_changed_spectrum = false;
    HierarchicalRenormDiagnosticCause primary_diagnostic = HierarchicalRenormDiagnosticCause::None;
    gfe::SpectralUnits input_spectral{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    gfe::SpectralUnits bath_spectral{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    gfe::SpectralUnits direct_spectral{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    gfe::SpectralUnits output_spectral_raw{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    gfe::SpectralUnits output_spectral{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    HierarchicalSpectralDrift input_to_output_raw_drift;
    HierarchicalSpectralDrift input_to_output_pruned_drift;
    HierarchicalSpectralDrift raw_to_pruned_drift;
    HierarchicalRenormConsistencyReport consistency;
    HierarchicalRenormPlugbackResult plugback;
};

HierarchicalRenormReport build_renormalized_kernel(
    const gfe::MemoryKernelParams& input_kernel,
    const gfe::MemoryKernelParams& bath_kernel,
    const gfe::MemoryKernelParams* direct_kernel = nullptr,
    const HierarchicalRenormOptions& options = {});

HierarchicalRenormPlugbackResult validate_renormalized_kernel_runtime(
    const AberSOEModelSpec& model_template,
    const AberSOEState& initial_state,
    const gfe::MemoryKernelParams& kernel,
    const AberSOERuntimeConfig& cfg = {});

HierarchicalRenormConsistencyReport evaluate_renormalization_consistency(
    const HierarchicalRenormReport& report,
    const HierarchicalRenormConsistencyOptions& options = {});

} // namespace abersoe
