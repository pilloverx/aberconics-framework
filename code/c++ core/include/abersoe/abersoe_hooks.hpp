#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gfe/gfe_dynamics.hpp"

namespace abersoe {

struct EnergySample {
    double t = 0.0;
    double u_l2 = 0.0;
    double chi_l2 = 0.0;
};

class RecordingEnergyObserver final : public gfe::EnergyObserver {
public:
    void on_step(double t, const gfe::State& u, const gfe::Vec& chi) override;

    const std::vector<EnergySample>& samples() const { return samples_; }
    std::size_t event_count() const { return samples_.size(); }

private:
    std::vector<EnergySample> samples_;
};

void write_energy_samples_csv(const std::string& path, const std::vector<EnergySample>& samples);

struct HookPassThroughSummary {
    bool assumption_checks_attached = false;
    bool energy_observer_attached = false;
    bool assumption_kernel_approximable = true;
    bool assumption_regularity_ok = true;
    std::string assumption_note;
    std::size_t energy_events = 0;
};

} // namespace abersoe
