#include "abersoe/hierarchical_min_renorm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>

namespace abersoe {
namespace {

double kernel_eval(const gfe::MemoryKernelParams& kernel, double t) {
    double y = 0.0;
    for (std::size_t i = 0; i < kernel.gamma.size(); ++i) {
        y += kernel.w[i] * std::exp(-kernel.gamma[i] * t);
    }
    return y;
}

double convolved_eval(
    const gfe::MemoryKernelParams& a,
    const gfe::MemoryKernelParams& b,
    double t,
    double degenerate_rate_tol,
    std::size_t* degenerate_pairs) {
    double y = 0.0;
    for (std::size_t i = 0; i < a.gamma.size(); ++i) {
        for (std::size_t j = 0; j < b.gamma.size(); ++j) {
            const double ga = a.gamma[i];
            const double gb = b.gamma[j];
            const double coeff = a.w[i] * b.w[j];
            if (std::fabs(ga - gb) <= degenerate_rate_tol) {
                if (degenerate_pairs) {
                    *degenerate_pairs += 1;
                }
                y += coeff * t * std::exp(-ga * t);
            } else {
                y += coeff * (std::exp(-gb * t) - std::exp(-ga * t)) / (ga - gb);
            }
        }
    }
    return y;
}

gfe::MemoryKernelParams pruned_kernel(const gfe::MemoryKernelParams& kernel, double threshold, std::size_t* pruned) {
    if (threshold <= 0.0 || kernel.w.empty()) {
        if (pruned) {
            *pruned = 0;
        }
        return kernel;
    }

    const double max_w = *std::max_element(kernel.w.begin(), kernel.w.end());
    gfe::MemoryKernelParams out;
    for (std::size_t i = 0; i < kernel.w.size(); ++i) {
        if (kernel.w[i] >= threshold * max_w) {
            out.w.push_back(kernel.w[i]);
            out.gamma.push_back(kernel.gamma[i]);
        }
    }
    if (pruned) {
        *pruned = kernel.w.size() - out.w.size();
    }
    return out;
}

double infer_tmax(const gfe::MemoryKernelParams& a, const gfe::MemoryKernelParams& b, const gfe::MemoryKernelParams* d) {
    double min_rate = std::numeric_limits<double>::infinity();
    auto update_min = [&min_rate](const gfe::MemoryKernelParams& k) {
        for (double gamma : k.gamma) {
            if (gamma > 0.0) {
                min_rate = std::min(min_rate, gamma);
            }
        }
    };
    update_min(a);
    update_min(b);
    if (d) {
        update_min(*d);
    }
    if (!std::isfinite(min_rate) || min_rate <= 0.0) {
        return 10.0;
    }
    return 6.0 / min_rate;
}

gfe::SpectralUnits spectral_or_zero(const gfe::MemoryKernelParams& kernel) {
    if (kernel.w.empty() || kernel.gamma.empty()) {
        return gfe::SpectralUnits{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    return gfe::spectral_units(kernel.w, kernel.gamma);
}

HierarchicalSpectralDrift spectral_drift(const gfe::SpectralUnits& from, const gfe::SpectralUnits& to) {
    HierarchicalSpectralDrift drift;
    drift.delta_mcap = to.Mcap - from.Mcap;
    drift.delta_mscale = to.Mscale - from.Mscale;
    drift.delta_hmem = to.Hmem - from.Hmem;
    drift.delta_deff = to.Deff - from.Deff;
    drift.abs_delta_mcap = std::fabs(drift.delta_mcap);
    drift.abs_delta_mscale = std::fabs(drift.delta_mscale);
    drift.abs_delta_hmem = std::fabs(drift.delta_hmem);
    drift.abs_delta_deff = std::fabs(drift.delta_deff);
    return drift;
}

bool is_sorted_descending(const gfe::Vec& gamma) {
    for (std::size_t i = 1; i < gamma.size(); ++i) {
        if (gamma[i - 1] < gamma[i]) {
            return false;
        }
    }
    return true;
}

double l2_fit_error(const gfe::Vec& truth, const gfe::MemoryKernelParams& kernel, const gfe::Vec& t) {
    if (truth.size() != t.size()) {
        throw std::invalid_argument("truth/t size mismatch in l2_fit_error");
    }
    double s = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double e = kernel_eval(kernel, t[i]) - truth[i];
        s += e * e;
    }
    return std::sqrt(s / static_cast<double>(t.size()));
}

double linf_fit_error(const gfe::Vec& truth, const gfe::MemoryKernelParams& kernel, const gfe::Vec& t) {
    if (truth.size() != t.size()) {
        throw std::invalid_argument("truth/t size mismatch in linf_fit_error");
    }
    double m = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        m = std::max(m, std::fabs(kernel_eval(kernel, t[i]) - truth[i]));
    }
    return m;
}

HierarchicalRenormDiagnosticCause diagnose_report(const HierarchicalRenormReport& report) {
    const bool heavy_prune =
        report.pruned_channels > 0 &&
        report.retained_weight_fraction < 0.75 &&
        (report.raw_to_pruned_drift.abs_delta_mcap > 1.0 ||
         report.raw_to_pruned_drift.abs_delta_mscale > 0.1 ||
         report.raw_to_pruned_drift.abs_delta_deff > 0.5);
    if (heavy_prune) {
        return HierarchicalRenormDiagnosticCause::OverPruning;
    }

    const bool merged =
        report.output_channels < report.raw_output_channels &&
        report.retained_weight_fraction >= 0.75 &&
        report.pruned_fit_l2_error <= 0.25;
    if (merged) {
        return HierarchicalRenormDiagnosticCause::PlausibleChannelMerging;
    }

    const bool fit_unstable =
        report.fit_l2_error > 0.2 ||
        report.fit_linf_error > 0.3 ||
        report.pruned_fit_l2_error > 0.25 ||
        report.pruned_fit_linf_error > 0.35;
    if (fit_unstable) {
        return HierarchicalRenormDiagnosticCause::FitInstability;
    }

    return HierarchicalRenormDiagnosticCause::None;
}

} // namespace

std::string hierarchical_renorm_diagnostic_cause_name(HierarchicalRenormDiagnosticCause cause) {
    switch (cause) {
        case HierarchicalRenormDiagnosticCause::None:
            return "none";
        case HierarchicalRenormDiagnosticCause::FitInstability:
            return "fit_instability";
        case HierarchicalRenormDiagnosticCause::OverPruning:
            return "over_pruning";
        case HierarchicalRenormDiagnosticCause::PlausibleChannelMerging:
            return "plausible_channel_merging";
    }
    return "unknown";
}

HierarchicalRenormConsistencyReport evaluate_renormalization_consistency(
    const HierarchicalRenormReport& report,
    const HierarchicalRenormConsistencyOptions& options) {
    HierarchicalRenormConsistencyReport out;
    out.mcap_input = report.input_spectral.Mcap;
    out.mcap_output = report.output_spectral.Mcap;
    out.mscale_input = report.input_spectral.Mscale;
    out.mscale_output = report.output_spectral.Mscale;
    out.deff_input = report.input_spectral.Deff;
    out.deff_bath = report.bath_spectral.Deff;
    out.deff_output = report.output_spectral.Deff;
    out.deff_bound_rhs = out.deff_input * out.deff_bath;
    out.retained_weight_fraction = report.retained_weight_fraction;
    out.fit_l2_error = report.fit_l2_error;
    out.pruned_fit_l2_error = report.pruned_fit_l2_error;
    out.fit_linf_error = report.fit_linf_error;
    out.pruned_fit_linf_error = report.pruned_fit_linf_error;

    if (options.require_nonempty_output) {
        out.nonempty_output_pass =
            !report.output_kernel.gamma.empty() &&
            report.output_kernel.gamma.size() == report.output_kernel.w.size();
        if (!out.nonempty_output_pass) {
            out.pass = false;
            out.failures.push_back("output kernel is empty or shape-mismatched");
        }
    }

    if (options.require_sorted_output) {
        out.sorted_output_pass = report.output_sorted_descending;
        if (!out.sorted_output_pass) {
            out.pass = false;
            out.failures.push_back("output kernel lost descending decay ordering");
        }
    }

    out.retained_weight_pass = report.retained_weight_fraction + 1e-12 >= options.min_retained_weight_fraction;
    if (!out.retained_weight_pass) {
        out.pass = false;
        out.failures.push_back(
            "retained_weight_fraction below minimum: observed=" + std::to_string(report.retained_weight_fraction) +
            " min=" + std::to_string(options.min_retained_weight_fraction));
    }

    out.fit_error_pass =
        (options.max_fit_l2_error <= 0.0 || report.fit_l2_error <= options.max_fit_l2_error) &&
        (options.max_pruned_fit_l2_error <= 0.0 || report.pruned_fit_l2_error <= options.max_pruned_fit_l2_error) &&
        (options.max_fit_linf_error <= 0.0 || report.fit_linf_error <= options.max_fit_linf_error) &&
        (options.max_pruned_fit_linf_error <= 0.0 || report.pruned_fit_linf_error <= options.max_pruned_fit_linf_error);
    if (!out.fit_error_pass) {
        out.pass = false;
        out.failures.push_back("fit error thresholds violated");
    }

    if (options.require_plugback_finite_when_checked) {
        out.plugback_finite_pass = !report.plugback.checked || report.plugback.all_finite;
        if (!out.plugback_finite_pass) {
            out.pass = false;
            out.failures.push_back("plug-back runtime check was performed and returned non-finite output");
        }
    }

    if (options.require_prune_drift_coherence) {
        const bool measured_prune_drift =
            report.raw_to_pruned_drift.abs_delta_mcap > 1e-12 ||
            report.raw_to_pruned_drift.abs_delta_mscale > 1e-12 ||
            report.raw_to_pruned_drift.abs_delta_hmem > 1e-12 ||
            report.raw_to_pruned_drift.abs_delta_deff > 1e-12;
        out.prune_drift_coherence_pass = report.prune_changed_spectrum == measured_prune_drift;
        if (!out.prune_drift_coherence_pass) {
            out.pass = false;
            out.failures.push_back("prune_changed_spectrum flag disagrees with measured raw->pruned drift");
        }
    }

    if (options.require_mcap_nondecrease) {
        out.mcap_nondecrease_pass = out.mcap_output + options.mcap_nondecrease_tol >= out.mcap_input;
        if (!out.mcap_nondecrease_pass) {
            out.pass = false;
            out.failures.push_back(
                "Mcap nondecrease violated: output=" + std::to_string(out.mcap_output) +
                " input=" + std::to_string(out.mcap_input) +
                " tol=" + std::to_string(options.mcap_nondecrease_tol));
        }
    }

    if (options.require_mscale_nondecrease) {
        out.mscale_nondecrease_pass = out.mscale_output + options.mscale_nondecrease_tol >= out.mscale_input;
        if (!out.mscale_nondecrease_pass) {
            out.pass = false;
            out.failures.push_back(
                "Mscale nondecrease violated: output=" + std::to_string(out.mscale_output) +
                " input=" + std::to_string(out.mscale_input) +
                " tol=" + std::to_string(options.mscale_nondecrease_tol));
        }
    }

    if (options.require_deff_submultiplicative_bound) {
        out.deff_bound_pass = out.deff_output <= out.deff_bound_rhs + options.deff_bound_slack;
        if (!out.deff_bound_pass) {
            out.pass = false;
            out.failures.push_back(
                "Deff submultiplicative bound violated: output=" + std::to_string(out.deff_output) +
                " rhs=" + std::to_string(out.deff_bound_rhs) +
                " slack=" + std::to_string(options.deff_bound_slack));
        }
    }

    return out;
}

HierarchicalRenormReport build_renormalized_kernel(
    const gfe::MemoryKernelParams& input_kernel,
    const gfe::MemoryKernelParams& bath_kernel,
    const gfe::MemoryKernelParams* direct_kernel,
    const HierarchicalRenormOptions& options) {
    if (input_kernel.gamma.size() != input_kernel.w.size() || input_kernel.gamma.empty()) {
        throw std::invalid_argument("input_kernel must have matching non-empty gamma/w");
    }
    if (bath_kernel.gamma.size() != bath_kernel.w.size() || bath_kernel.gamma.empty()) {
        throw std::invalid_argument("bath_kernel must have matching non-empty gamma/w");
    }
    if (direct_kernel && direct_kernel->gamma.size() != direct_kernel->w.size()) {
        throw std::invalid_argument("direct_kernel must have matching gamma/w");
    }
    if (options.n_samples < 3) {
        throw std::invalid_argument("HierarchicalRenormOptions.n_samples must be >= 3");
    }

    HierarchicalRenormReport report;
    report.input_kernel = input_kernel;
    report.bath_kernel = bath_kernel;
    if (direct_kernel) {
        report.direct_kernel = *direct_kernel;
    }
    report.input_channels = input_kernel.gamma.size();
    report.bath_channels = bath_kernel.gamma.size();
    report.direct_channels = direct_kernel ? direct_kernel->gamma.size() : 0;
    report.input_spectral = spectral_or_zero(input_kernel);
    report.bath_spectral = spectral_or_zero(bath_kernel);
    report.direct_spectral = direct_kernel ? spectral_or_zero(*direct_kernel)
                                           : gfe::SpectralUnits{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const double t_max = options.t_max > 0.0 ? options.t_max : infer_tmax(input_kernel, bath_kernel, direct_kernel);
    gfe::Vec t(options.n_samples, 0.0);
    gfe::Vec y(options.n_samples, 0.0);
    for (std::size_t i = 0; i < options.n_samples; ++i) {
        const double ti = t_max * static_cast<double>(i) / static_cast<double>(options.n_samples - 1);
        t[i] = ti;
        y[i] = convolved_eval(input_kernel, bath_kernel, ti, options.degenerate_rate_tol, &report.degenerate_pairs);
        if (direct_kernel) {
            y[i] += kernel_eval(*direct_kernel, ti);
        }
    }

    auto fit = gfe::fit_soe_kernel(t, y, options.fitter);
    report.samples_used = options.n_samples;
    report.fit_l2_error = 0.0;
    report.fit_linf_error = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double e = fit.fit[i] - y[i];
        report.fit_l2_error += e * e;
        report.fit_linf_error = std::max(report.fit_linf_error, std::fabs(e));
    }
    report.fit_l2_error = std::sqrt(report.fit_l2_error / static_cast<double>(y.size()));

    report.output_kernel_raw = gfe::MemoryKernelParams{fit.gamma_fit, fit.w_fit};
    report.raw_output_channels = report.output_kernel_raw.gamma.size();
    report.output_kernel = pruned_kernel(report.output_kernel_raw, options.prune_weight_threshold, &report.pruned_channels);
    report.output_channels = report.output_kernel.gamma.size();
    report.output_spectral_raw = spectral_or_zero(report.output_kernel_raw);
    report.output_spectral = spectral_or_zero(report.output_kernel);
    report.output_sorted_descending = is_sorted_descending(report.output_kernel.gamma);

    double raw_weight_sum = 0.0;
    for (double w : report.output_kernel_raw.w) {
        raw_weight_sum += w;
    }
    double kept_weight_sum = 0.0;
    for (double w : report.output_kernel.w) {
        kept_weight_sum += w;
    }
    report.retained_weight_fraction = raw_weight_sum > 0.0 ? kept_weight_sum / raw_weight_sum : 1.0;
    report.pruned_fit_l2_error = l2_fit_error(y, report.output_kernel, t);
    report.pruned_fit_linf_error = linf_fit_error(y, report.output_kernel, t);
    report.input_to_output_raw_drift = spectral_drift(report.input_spectral, report.output_spectral_raw);
    report.input_to_output_pruned_drift = spectral_drift(report.input_spectral, report.output_spectral);
    report.raw_to_pruned_drift = spectral_drift(report.output_spectral_raw, report.output_spectral);
    report.prune_changed_spectrum =
        report.pruned_channels > 0 &&
        (report.raw_to_pruned_drift.abs_delta_mcap > 1e-12 ||
         report.raw_to_pruned_drift.abs_delta_mscale > 1e-12 ||
         report.raw_to_pruned_drift.abs_delta_hmem > 1e-12 ||
         report.raw_to_pruned_drift.abs_delta_deff > 1e-12);
    report.primary_diagnostic = diagnose_report(report);

    if (report.output_kernel.gamma.empty() || report.output_kernel.w.empty()) {
        throw std::runtime_error("renormalized kernel was empty after pruning");
    }
    if (!report.output_sorted_descending) {
        throw std::runtime_error("renormalized kernel lost descending decay ordering");
    }
    if (options.max_fit_l2_error > 0.0 && report.fit_l2_error > options.max_fit_l2_error) {
        throw std::runtime_error("renormalized raw fit exceeded max_fit_l2_error");
    }
    if (options.max_pruned_fit_l2_error > 0.0 && report.pruned_fit_l2_error > options.max_pruned_fit_l2_error) {
        throw std::runtime_error("renormalized pruned fit exceeded max_pruned_fit_l2_error");
    }
    report.consistency = evaluate_renormalization_consistency(report);
    return report;
}

HierarchicalRenormPlugbackResult validate_renormalized_kernel_runtime(
    const AberSOEModelSpec& model_template,
    const AberSOEState& initial_state,
    const gfe::MemoryKernelParams& kernel,
    const AberSOERuntimeConfig& cfg) {
    AberSOEModelSpec model = model_template;
    model.kernel = kernel;

    AberSOEState state0 = initial_state;
    if (!state0.chi.empty() && state0.chi.size() != kernel.gamma.size()) {
        throw std::invalid_argument("initial_state chi size must match plug-back kernel channel count");
    }
    if (state0.chi.empty()) {
        state0.chi.assign(kernel.gamma.size(), 0.0);
    }

    const auto run_result = abersoe::run(model, state0, cfg);
    HierarchicalRenormPlugbackResult out;
    out.checked = true;
    out.all_finite = run_result.diagnostics.all_finite;
    out.steps_executed = run_result.diagnostics.steps_executed;
    out.final_u_l2 = run_result.diagnostics.final_u_l2;
    out.final_chi_l2 = run_result.diagnostics.final_chi_l2;
    return out;
}

} // namespace abersoe
