#include "abersoe/abersoe_learning.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace abersoe {

const char* hebbian_rule_name(HebbianRule rule) {
    switch (rule) {
        case HebbianRule::OjaNormalized:
            return "oja";
        case HebbianRule::PlainHebbian:
            return "plain";
    }
    return "unknown";
}

void apply_hebbian_weight_update(
    gfe::Vec& w,
    const gfe::Vec& chi,
    const gfe::State& u,
    std::size_t coupling_index,
    const HebbianLearningConfig& cfg,
    double dt) {
    if (!cfg.enabled) {
        return;
    }
    if (w.empty() || chi.empty() || w.size() != chi.size()) {
        throw std::invalid_argument("hebbian update requires non-empty w/chi with equal size");
    }
    if (u.empty() || coupling_index >= u.size()) {
        throw std::invalid_argument("hebbian update coupling_index out of range");
    }
    if (dt <= 0.0) {
        throw std::invalid_argument("hebbian update requires dt > 0");
    }

    const double pre = u[coupling_index];

    for (std::size_t i = 0; i < w.size(); ++i) {
        const double post = chi[i];
        double dw = 0.0;
        if (cfg.rule == HebbianRule::OjaNormalized) {
            dw = cfg.learning_rate * (pre * post - post * post * w[i]) - cfg.decay * w[i];
        } else {
            dw = cfg.learning_rate * (pre * post) - cfg.decay * w[i];
        }
        w[i] += dt * dw;
        w[i] = std::clamp(w[i], cfg.min_weight, cfg.max_weight);
    }

    if (cfg.renormalize) {
        const double sum_w = std::accumulate(w.begin(), w.end(), 0.0);
        if (sum_w > 0.0) {
            const double target = cfg.target_weight_sum > 0.0 ? cfg.target_weight_sum : 1.0;
            const double scale = target / sum_w;
            for (double& wi : w) {
                wi = std::clamp(wi * scale, cfg.min_weight, cfg.max_weight);
            }
        }
    }
}

} // namespace abersoe

