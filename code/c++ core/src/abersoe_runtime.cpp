#include "abersoe/abersoe_runtime.hpp"
#include "abersoe/abersoe_learning.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdexcept>

namespace abersoe {
namespace {

class CountingForwardEnergyObserver final : public gfe::EnergyObserver {
public:
    explicit CountingForwardEnergyObserver(gfe::EnergyObserver* inner) : inner_(inner) {}

    void on_step(double t, const gfe::State& u, const gfe::Vec& chi) override {
        ++count_;
        if (inner_) {
            inner_->on_step(t, u, chi);
        }
    }

    std::size_t count() const { return count_; }

private:
    gfe::EnergyObserver* inner_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace

AberSOEState step(const AberSOEModelSpec& model, const AberSOEState& state) {
    std::string err;
    if (!validate_model_spec(model, &err)) {
        throw std::invalid_argument("Invalid AberSOE model: " + err);
    }
    if (state.u.empty()) {
        throw std::invalid_argument("state.u must be non-empty");
    }
    if (state.chi.size() != model.kernel.gamma.size()) {
        throw std::invalid_argument("state.chi size must match number of memory channels");
    }

    auto step_result = gfe::step_augmented(
        state.u,
        state.chi,
        state.t,
        model.kernel,
        model.dynamics,
        model.operators,
        model.gst,
        model.assumption_checks,
        model.energy_observer);

    return AberSOEState{step_result.u_next, step_result.chi_next, state.t + model.dynamics.dt};
}

AberSOERunResult run(const AberSOEModelSpec& model, const AberSOEState& state0, const AberSOERuntimeConfig& cfg) {
    std::string err;
    if (!validate_model_spec(model, &err)) {
        throw std::invalid_argument("Invalid AberSOE model: " + err);
    }
    if (!validate_runtime_config(cfg, &err)) {
        throw std::invalid_argument("Invalid AberSOE runtime config: " + err);
    }

    AberSOERunResult out;
    out.config = make_config_record(model, cfg);
    out.seed_used = cfg.seed;
    out.hooks.assumption_checks_attached = (model.assumption_checks != nullptr);
    out.hooks.energy_observer_attached = (model.energy_observer != nullptr);
    out.samples.reserve((cfg.steps / cfg.sample_stride) + 1);

    AberSOEState state = state0;
    if (state.u.empty()) {
        throw std::invalid_argument("state0.u must be non-empty");
    }
    if (state.chi.empty()) {
        state.chi.assign(model.kernel.gamma.size(), 0.0);
    }
    if (cfg.stochastic_channel >= state.u.size()) {
        throw std::invalid_argument("stochastic_channel out of range for state dimension");
    }

    AberSOEModelSpec model_local = model;
    CountingForwardEnergyObserver energy_proxy(model_local.energy_observer);
    model_local.energy_observer = &energy_proxy;

    if (cfg.stochastic_forcing && cfg.forcing_noise_std > 0.0) {
        out.stochastic_forcing_applied = true;
        auto rng = std::make_shared<std::mt19937_64>(cfg.seed);
        auto normal = std::make_shared<std::normal_distribution<double>>(0.0, cfg.forcing_noise_std);
        const std::size_t dim = state.u.size();
        const std::size_t channel = cfg.stochastic_channel;
        const auto base_forcing = model_local.operators.forcing;
        model_local.operators.forcing = [rng, normal, base_forcing, dim, channel](double t) {
            gfe::State f = base_forcing ? base_forcing(t) : gfe::State(dim, 0.0);
            if (f.size() != dim) {
                throw std::invalid_argument("forcing(t) returned invalid dimension under stochastic wrapper");
            }
            f[channel] += (*normal)(*rng);
            return f;
        };
    }

    out.samples.push_back(state);
    for (std::size_t k = 0; k < cfg.steps; ++k) {
        state = step(model_local, state);
        update_diagnostics(out.diagnostics, state.u, state.chi);

        if (model_local.learning.enabled) {
            apply_hebbian_weight_update(
                model_local.kernel.w,
                state.chi,
                state.u,
                model_local.dynamics.coupling_index,
                model_local.learning,
                model_local.dynamics.dt);
            out.learning_updates += 1;
        }

        if (cfg.strict_finite && !out.diagnostics.all_finite) {
            throw std::runtime_error("AberSOE run became non-finite");
        }

        if ((k + 1) % cfg.sample_stride == 0) {
            out.samples.push_back(state);
        }
    }

    out.final_state = state;
    if (model_local.learning.enabled) {
        out.final_kernel_w = model_local.kernel.w;
    }
    out.hooks.energy_events = energy_proxy.count();
    if (model.assumption_checks) {
        out.hooks.assumption_kernel_approximable = model.assumption_checks->kernel_approximable;
        out.hooks.assumption_regularity_ok = model.assumption_checks->regularity_ok;
        out.hooks.assumption_note = model.assumption_checks->note;
    }
    return out;
}

void write_run_csv(const std::string& path, const AberSOERunResult& result) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open CSV path: " + path);
    }

    out << "t,u0,chi0,u_l2,chi_l2\n";
    for (const auto& s : result.samples) {
        const double u0 = s.u.empty() ? 0.0 : s.u[0];
        const double c0 = s.chi.empty() ? 0.0 : s.chi[0];

        double u_l2 = 0.0;
        for (double v : s.u) {
            u_l2 += v * v;
        }
        double chi_l2 = 0.0;
        for (double v : s.chi) {
            chi_l2 += v * v;
        }

        out << std::setprecision(17)
            << s.t << ','
            << u0 << ','
            << c0 << ','
            << std::sqrt(u_l2) << ','
            << std::sqrt(chi_l2) << '\n';
    }
}

} // namespace abersoe
