#include "gfe/gfe_energy.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace gfe {
namespace {

double l2_sq(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) {
        s += v * v;
    }
    return s;
}

} // namespace

QuadraticEnergyMonitor::QuadraticEnergyMonitor(EnergyBookkeepingConfig cfg) : cfg_(std::move(cfg)) {}

void QuadraticEnergyMonitor::set_external_power_callback(std::function<double(double, const State&, const Vec&)> cb) {
    external_power_cb_ = std::move(cb);
}

double QuadraticEnergyMonitor::energy_u(const State& u) const {
    return 0.5 * cfg_.u_weight * l2_sq(u);
}

double QuadraticEnergyMonitor::energy_chi(const Vec& chi) const {
    if (!cfg_.chi_weights.empty() && cfg_.chi_weights.size() != chi.size()) {
        throw std::invalid_argument("chi_weights size must match chi size when provided");
    }

    double s = 0.0;
    for (std::size_t i = 0; i < chi.size(); ++i) {
        const double w = cfg_.chi_weights.empty() ? cfg_.chi_weight : cfg_.chi_weights[i];
        s += w * chi[i] * chi[i];
    }
    return 0.5 * s;
}

double QuadraticEnergyMonitor::dissipation(const Vec& chi) const {
    if (!cfg_.memory_decay_rates.empty() && cfg_.memory_decay_rates.size() != chi.size()) {
        throw std::invalid_argument("memory_decay_rates size must match chi size when provided");
    }

    double s = 0.0;
    for (std::size_t i = 0; i < chi.size(); ++i) {
        const double g = cfg_.memory_decay_rates.empty() ? 0.0 : cfg_.memory_decay_rates[i];
        s += g * chi[i] * chi[i];
    }
    return s;
}

void QuadraticEnergyMonitor::on_transition(double t_prev,
                                           double t_next,
                                           const State& u_prev,
                                           const Vec& chi_prev,
                                           const State& u_next,
                                           const Vec& chi_next) {
    const double E_prev_u = energy_u(u_prev);
    const double E_prev_chi = energy_chi(chi_prev);
    const double E_next_u = energy_u(u_next);
    const double E_next_chi = energy_chi(chi_next);

    const double E_prev = E_prev_u + E_prev_chi;
    const double E_next = E_next_u + E_next_chi;
    const double dt = t_next - t_prev;
    if (dt <= 0.0) {
        throw std::invalid_argument("Energy monitor requires positive time step");
    }

    const double dE_dt = (E_next - E_prev) / dt;
    const double external_power = external_power_cb_ ? external_power_cb_(t_prev, u_prev, chi_prev) : 0.0;
    const double diss = dissipation(chi_prev);
    const double rhs = external_power - diss;
    const double residual = dE_dt - rhs;
    const bool within = std::fabs(residual) <= cfg_.residual_tolerance;

    snapshots_.push_back(EnergySnapshot{t_next, E_next_u, E_next_chi, E_next});
    residuals_.push_back(EnergyResidualSample{t_prev, t_next, dE_dt, rhs, residual, within});
    all_within_tolerance_ = all_within_tolerance_ && within;

    if (cfg_.enforce_residual_tolerance && !within) {
        throw std::runtime_error("Energy residual tolerance violated");
    }
}

void write_energy_residuals_csv(const std::string& path, const std::vector<EnergyResidualSample>& residuals) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open energy residual CSV path: " + path);
    }

    out << "t_prev,t_next,dE_dt_observed,rhs_power_minus_dissipation,residual,within_tolerance\n";
    for (const auto& r : residuals) {
        out << std::setprecision(17)
            << r.t_prev << ','
            << r.t_next << ','
            << r.dE_dt_observed << ','
            << r.rhs_power_minus_dissipation << ','
            << r.residual << ','
            << (r.within_tolerance ? 1 : 0) << '\n';
    }
}

} // namespace gfe

