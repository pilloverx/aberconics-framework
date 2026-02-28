#include "abersoe/abersoe_hooks.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
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

} // namespace

void RecordingEnergyObserver::on_step(double t, const gfe::State& u, const gfe::Vec& chi) {
    EnergySample s;
    s.t = t;
    s.u_l2 = l2_norm(u);
    s.chi_l2 = l2_norm(chi);
    samples_.push_back(s);
}

void write_energy_samples_csv(const std::string& path, const std::vector<EnergySample>& samples) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open energy CSV path: " + path);
    }
    out << "t,u_l2,chi_l2\n";
    for (const auto& s : samples) {
        out << std::setprecision(17)
            << s.t << ','
            << s.u_l2 << ','
            << s.chi_l2 << '\n';
    }
}

} // namespace abersoe
