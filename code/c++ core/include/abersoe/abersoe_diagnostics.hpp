#pragma once

#include <cstddef>
#include <string>

#include "gfe/gfe_dynamics.hpp"
#include "abersoe/abersoe_model.hpp"
#include "abersoe/abersoe_config.hpp"

namespace abersoe {

struct AberSOEDiagnostics {
    std::size_t steps_executed = 0;
    bool all_finite = true;
    double max_abs_u = 0.0;
    double max_abs_chi = 0.0;
    double final_u_l2 = 0.0;
    double final_chi_l2 = 0.0;
    double mean_abs_u = 0.0;
    double mean_abs_chi = 0.0;
};

void update_diagnostics(AberSOEDiagnostics& diag, const gfe::State& u, const gfe::Vec& chi);
void write_diagnostics_csv(const std::string& path, const AberSOEDiagnostics& diag);

struct AberSOEConfigRecord {
    std::size_t memory_channels = 0;
    double dt = 0.0;
    std::size_t coupling_index = 0;
    std::size_t steps = 0;
    std::size_t sample_stride = 0;
    bool strict_finite = true;
    bool stochastic_forcing = false;
    double forcing_noise_std = 0.0;
    std::size_t stochastic_channel = 0;
    unsigned long long seed = 0ULL;
    std::string fit_backend;
    bool hebbian_enabled = false;
    std::string hebbian_rule;
    double hebbian_learning_rate = 0.0;
    double hebbian_decay = 0.0;
};

AberSOEConfigRecord make_config_record(const AberSOEModelSpec& model, const AberSOERuntimeConfig& cfg);
void write_config_csv(const std::string& path, const AberSOEConfigRecord& cfg);

struct AberSOEFitReport {
    bool ok = false;
    std::string backend;
    std::size_t modes_kept = 0;
    double l1_error = 0.0;
    double l2_error = 0.0;
    gfe::SpectralUnits spectral{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    gfe::FitResult fit;
    std::string note;
};

AberSOEFitReport build_fit_report(const gfe::Vec& t, const gfe::Vec& data, const AberSOEModelSpec& model);
void write_fit_report_csv(const std::string& path, const AberSOEFitReport& report);

} // namespace abersoe
