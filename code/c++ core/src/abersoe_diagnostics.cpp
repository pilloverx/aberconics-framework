#include "abersoe/abersoe_diagnostics.hpp"
#include "abersoe/abersoe_learning.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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

bool finite_vec(const std::vector<double>& x) {
    for (double v : x) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

double max_abs(const std::vector<double>& x) {
    double m = 0.0;
    for (double v : x) {
        m = std::max(m, std::fabs(v));
    }
    return m;
}

double mean_abs(const std::vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }
    double s = 0.0;
    for (double v : x) {
        s += std::fabs(v);
    }
    return s / static_cast<double>(x.size());
}

const char* backend_name(gfe::FitBackend b) {
    switch (b) {
        case gfe::FitBackend::NnlsProjectedGradient:
            return "nnls_pg";
        case gfe::FitBackend::Prony:
            return "prony";
    }
    return "unknown";
}

} // namespace

void update_diagnostics(AberSOEDiagnostics& diag, const gfe::State& u, const gfe::Vec& chi) {
    diag.steps_executed += 1;
    diag.all_finite = diag.all_finite && finite_vec(u) && finite_vec(chi);
    diag.max_abs_u = std::max(diag.max_abs_u, max_abs(u));
    diag.max_abs_chi = std::max(diag.max_abs_chi, max_abs(chi));
    diag.final_u_l2 = l2_norm(u);
    diag.final_chi_l2 = l2_norm(chi);
    const double n = static_cast<double>(diag.steps_executed);
    diag.mean_abs_u += (mean_abs(u) - diag.mean_abs_u) / n;
    diag.mean_abs_chi += (mean_abs(chi) - diag.mean_abs_chi) / n;
}

void write_diagnostics_csv(const std::string& path, const AberSOEDiagnostics& diag) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open diagnostics CSV path: " + path);
    }
    out << "steps_executed,all_finite,max_abs_u,max_abs_chi,final_u_l2,final_chi_l2,mean_abs_u,mean_abs_chi\n";
    out << diag.steps_executed << ','
        << (diag.all_finite ? 1 : 0) << ','
        << std::setprecision(17) << diag.max_abs_u << ','
        << diag.max_abs_chi << ','
        << diag.final_u_l2 << ','
        << diag.final_chi_l2 << ','
        << diag.mean_abs_u << ','
        << diag.mean_abs_chi << '\n';
}

AberSOEConfigRecord make_config_record(const AberSOEModelSpec& model, const AberSOERuntimeConfig& cfg) {
    AberSOEConfigRecord out;
    out.memory_channels = model.kernel.gamma.size();
    out.dt = model.dynamics.dt;
    out.coupling_index = model.dynamics.coupling_index;
    out.steps = cfg.steps;
    out.sample_stride = cfg.sample_stride;
    out.strict_finite = cfg.strict_finite;
    out.stochastic_forcing = cfg.stochastic_forcing;
    out.forcing_noise_std = cfg.forcing_noise_std;
    out.stochastic_channel = cfg.stochastic_channel;
    out.seed = cfg.seed;
    out.fit_backend = backend_name(model.fitter.backend);
    out.hebbian_enabled = model.learning.enabled;
    out.hebbian_rule = hebbian_rule_name(model.learning.rule);
    out.hebbian_learning_rate = model.learning.learning_rate;
    out.hebbian_decay = model.learning.decay;
    return out;
}

void write_config_csv(const std::string& path, const AberSOEConfigRecord& cfg) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open config CSV path: " + path);
    }
    out << "memory_channels,dt,coupling_index,steps,sample_stride,strict_finite,stochastic_forcing,forcing_noise_std,stochastic_channel,seed,fit_backend,hebbian_enabled,hebbian_rule,hebbian_learning_rate,hebbian_decay\n";
    out << cfg.memory_channels << ','
        << std::setprecision(17) << cfg.dt << ','
        << cfg.coupling_index << ','
        << cfg.steps << ','
        << cfg.sample_stride << ','
        << (cfg.strict_finite ? 1 : 0) << ','
        << (cfg.stochastic_forcing ? 1 : 0) << ','
        << cfg.forcing_noise_std << ','
        << cfg.stochastic_channel << ','
        << cfg.seed << ','
        << cfg.fit_backend << ','
        << (cfg.hebbian_enabled ? 1 : 0) << ','
        << cfg.hebbian_rule << ','
        << cfg.hebbian_learning_rate << ','
        << cfg.hebbian_decay << '\n';
}

AberSOEFitReport build_fit_report(const gfe::Vec& t, const gfe::Vec& data, const AberSOEModelSpec& model) {
    AberSOEFitReport out;
    out.backend = backend_name(model.fitter.backend);
    try {
        out.fit = gfe::fit_soe_kernel(t, data, model.fitter);
        if (out.fit.fit.size() != data.size() || out.fit.gamma_fit.empty() || out.fit.w_fit.empty()) {
            out.ok = false;
            out.note = "fit output shape invalid";
            return out;
        }

        double l1 = 0.0;
        double l2 = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            const double e = out.fit.fit[i] - data[i];
            l1 += std::fabs(e);
            l2 += e * e;
        }
        out.l1_error = l1 / static_cast<double>(data.size());
        out.l2_error = std::sqrt(l2 / static_cast<double>(data.size()));
        out.modes_kept = out.fit.gamma_fit.size();
        out.spectral = gfe::spectral_units(out.fit.w_fit, out.fit.gamma_fit);
        out.ok = true;
        out.note = "ok";
        return out;
    } catch (const std::exception& e) {
        out.ok = false;
        out.note = e.what();
        out.spectral = gfe::SpectralUnits{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
        return out;
    }
}

void write_fit_report_csv(const std::string& path, const AberSOEFitReport& report) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open fit report CSV path: " + path);
    }
    out << "ok,backend,modes_kept,l1_error,l2_error,Mcap,Mscale,Mres,Hmem,Hnorm,Deff,note\n";
    out << (report.ok ? 1 : 0) << ','
        << report.backend << ','
        << report.modes_kept << ','
        << std::setprecision(17) << report.l1_error << ','
        << report.l2_error << ','
        << report.spectral.Mcap << ','
        << report.spectral.Mscale << ','
        << report.spectral.Mres << ','
        << report.spectral.Hmem << ','
        << report.spectral.Hnorm << ','
        << report.spectral.Deff << ','
        << '"' << report.note << '"' << '\n';
}

} // namespace abersoe
