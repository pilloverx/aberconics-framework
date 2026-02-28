#pragma once

#include <functional>
#include <string>
#include <vector>

#include "gfe/gfe_dynamics.hpp"

namespace gfe {

struct EnergySnapshot {
    double t = 0.0;
    double kinetic = 0.0;
    double memory = 0.0;
    double total = 0.0;
};

struct EnergyResidualSample {
    double t_prev = 0.0;
    double t_next = 0.0;
    double dE_dt_observed = 0.0;
    double rhs_power_minus_dissipation = 0.0;
    double residual = 0.0;
    bool within_tolerance = true;
};

struct EnergyBookkeepingConfig {
    double u_weight = 1.0;
    double chi_weight = 1.0;
    Vec chi_weights;
    Vec memory_decay_rates;
    double residual_tolerance = 1e-6;
    bool enforce_residual_tolerance = false;
};

class QuadraticEnergyMonitor final : public EnergyObserver {
public:
    explicit QuadraticEnergyMonitor(EnergyBookkeepingConfig cfg = {});

    void set_external_power_callback(std::function<double(double, const State&, const Vec&)> cb);

    void on_transition(double t_prev,
                       double t_next,
                       const State& u_prev,
                       const Vec& chi_prev,
                       const State& u_next,
                       const Vec& chi_next) override;

    const std::vector<EnergySnapshot>& snapshots() const { return snapshots_; }
    const std::vector<EnergyResidualSample>& residuals() const { return residuals_; }
    bool all_within_tolerance() const { return all_within_tolerance_; }

private:
    double energy_u(const State& u) const;
    double energy_chi(const Vec& chi) const;
    double dissipation(const Vec& chi) const;

    EnergyBookkeepingConfig cfg_;
    std::function<double(double, const State&, const Vec&)> external_power_cb_;
    std::vector<EnergySnapshot> snapshots_;
    std::vector<EnergyResidualSample> residuals_;
    bool all_within_tolerance_ = true;
};

void write_energy_residuals_csv(const std::string& path, const std::vector<EnergyResidualSample>& residuals);

} // namespace gfe

