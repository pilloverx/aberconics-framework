#include "gfe/gfe_assumptions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gfe {
namespace {

double l2_norm(const AssumptionState& x) {
    double s = 0.0;
    for (double v : x) {
        s += v * v;
    }
    return std::sqrt(s);
}

AssumptionEvidence make_evidence(const std::string& name, bool pass, double margin, double value, double threshold, const std::string& note) {
    return AssumptionEvidence{name, pass, margin, value, threshold, note};
}

AssumptionEvaluation eval_internal(const Vec& gamma,
                                   const Vec& w,
                                   const AssumptionState& u,
                                   double t,
                                   const AssumptionOperatorCallbacks& ops,
                                   const AssumptionEvalConfig& cfg) {
    if (gamma.size() != w.size() || gamma.empty()) {
        throw std::invalid_argument("evaluate_assumptions requires non-empty gamma/w with equal size");
    }
    if (u.empty()) {
        throw std::invalid_argument("evaluate_assumptions requires non-empty sample_state");
    }

    AssumptionEvaluation out;

    const auto gmin_it = std::min_element(gamma.begin(), gamma.end());
    const auto gmax_it = std::max_element(gamma.begin(), gamma.end());
    const double gmin = *gmin_it;
    const double gmax = *gmax_it;
    const double span = std::log10(gmax / gmin);
    const double W = std::accumulate(w.begin(), w.end(), 0.0);

    const bool gamma_pos = gmin > cfg.min_gamma;
    const double gamma_margin = gmin - cfg.min_gamma;
    out.evidence.push_back(make_evidence("gamma_positive", gamma_pos, gamma_margin, gmin, cfg.min_gamma,
                                         "strict positivity of decay rates"));

    const bool weight_sum_ok = W > cfg.min_weight_sum;
    const double weight_margin = W - cfg.min_weight_sum;
    out.evidence.push_back(make_evidence("weight_sum", weight_sum_ok, weight_margin, W, cfg.min_weight_sum,
                                         "non-trivial kernel weight mass"));

    const bool span_min_ok = span > cfg.min_spectral_span;
    const bool span_max_ok = span < cfg.max_spectral_span;
    const bool span_ok = span_min_ok && span_max_ok;
    const double span_margin = std::min(span - cfg.min_spectral_span, cfg.max_spectral_span - span);
    out.evidence.push_back(make_evidence("spectral_span", span_ok, span_margin, span, cfg.max_spectral_span,
                                         "spectral span in practical approximation range"));

    const bool kernel_ok = gamma_pos && weight_sum_ok && span_ok;
    out.kernel_approximable = kernel_ok;
    out.kernel_margin = std::min({gamma_margin, weight_margin, span_margin});

    bool regularity_ok = true;
    double regularity_margin = cfg.max_lipschitz_proxy;
    if (ops.apply_L) {
        const auto Lu = ops.apply_L(u, t);
        if (Lu.size() != u.size()) {
            throw std::invalid_argument("apply_L returned dimension mismatch in assumption eval");
        }
        AssumptionState u2 = u;
        for (double& v : u2) {
            v *= cfg.boundedness_sample_scale;
        }
        const auto Lu2 = ops.apply_L(u2, t);
        if (Lu2.size() != u.size()) {
            throw std::invalid_argument("apply_L returned dimension mismatch in assumption eval");
        }
        AssumptionState dL(u.size(), 0.0);
        AssumptionState du(u.size(), 0.0);
        for (std::size_t i = 0; i < u.size(); ++i) {
            dL[i] = Lu2[i] - Lu[i];
            du[i] = u2[i] - u[i];
        }
        const double num = l2_norm(dL);
        const double den = std::max(l2_norm(du), 1e-12);
        const double lipschitz_proxy = num / den;
        regularity_ok = regularity_ok && (lipschitz_proxy <= cfg.max_lipschitz_proxy);
        regularity_margin = std::min(regularity_margin, cfg.max_lipschitz_proxy - lipschitz_proxy);
        out.evidence.push_back(make_evidence("lipschitz_proxy_L",
                                             lipschitz_proxy <= cfg.max_lipschitz_proxy,
                                             cfg.max_lipschitz_proxy - lipschitz_proxy,
                                             lipschitz_proxy,
                                             cfg.max_lipschitz_proxy,
                                             "bounded linear operator proxy"));
    }

    bool boundedness_ok = true;
    double boundedness_margin = cfg.max_lipschitz_proxy;
    if (ops.apply_N) {
        const auto Nu = ops.apply_N(u, t);
        if (Nu.size() != u.size()) {
            throw std::invalid_argument("apply_N returned dimension mismatch in assumption eval");
        }
        AssumptionState u2 = u;
        for (double& v : u2) {
            v *= cfg.boundedness_sample_scale;
        }
        const auto Nu2 = ops.apply_N(u2, t);
        if (Nu2.size() != u.size()) {
            throw std::invalid_argument("apply_N returned dimension mismatch in assumption eval");
        }
        AssumptionState dN(u.size(), 0.0);
        AssumptionState du(u.size(), 0.0);
        for (std::size_t i = 0; i < u.size(); ++i) {
            dN[i] = Nu2[i] - Nu[i];
            du[i] = u2[i] - u[i];
        }
        const double num = l2_norm(dN);
        const double den = std::max(l2_norm(du), 1e-12);
        const double bounded_proxy = num / den;
        boundedness_ok = bounded_proxy <= cfg.max_lipschitz_proxy;
        boundedness_margin = cfg.max_lipschitz_proxy - bounded_proxy;
        out.evidence.push_back(make_evidence("boundedness_proxy_N",
                                             boundedness_ok,
                                             boundedness_margin,
                                             bounded_proxy,
                                             cfg.max_lipschitz_proxy,
                                             "bounded nonlinear operator proxy"));
    }

    out.regularity_ok = regularity_ok;
    out.regularity_margin = regularity_margin;
    out.operator_boundedness_ok = boundedness_ok;
    out.operator_boundedness_margin = boundedness_margin;
    out.pass = out.kernel_approximable && out.regularity_ok && out.operator_boundedness_ok;
    out.note = out.pass ? "assumption checks passed" : "assumption checks failed";

    return out;
}

} // namespace

AssumptionEvaluation evaluate_assumptions(const Vec& gamma,
                                          const Vec& w,
                                          const AssumptionState& sample_state,
                                          double t,
                                          const AssumptionOperatorCallbacks& ops,
                                          const AssumptionEvalConfig& cfg) {
    return eval_internal(gamma, w, sample_state, t, ops, cfg);
}

std::vector<std::string> assumption_test_corpus_names() {
    return {
        "valid_reference",
        "invalid_gamma_negative",
        "invalid_weight_sum_zero",
        "invalid_operator_unbounded"
    };
}

AssumptionEvaluation run_assumption_test_case(const std::string& name) {
    const AssumptionState u{1.0, -0.5};
    const double t = 0.0;

    if (name == "valid_reference") {
        AssumptionOperatorCallbacks ops;
        ops.apply_L = [](const AssumptionState& x, double /*t*/) {
            return AssumptionState{-x[0], -0.5 * x[1]};
        };
        ops.apply_N = [](const AssumptionState& x, double /*t*/) {
            return AssumptionState{0.1 * x[0], 0.05 * x[1]};
        };
        return evaluate_assumptions(Vec{1.0, 0.1, 0.01}, Vec{0.5, 0.3, 0.2}, u, t, ops);
    }

    if (name == "invalid_gamma_negative") {
        AssumptionOperatorCallbacks ops;
        ops.apply_L = [](const AssumptionState& x, double /*t*/) {
            return AssumptionState{-x[0], -x[1]};
        };
        return evaluate_assumptions(Vec{1.0, -0.1, 0.01}, Vec{0.5, 0.3, 0.2}, u, t, ops);
    }

    if (name == "invalid_weight_sum_zero") {
        AssumptionOperatorCallbacks ops;
        ops.apply_L = [](const AssumptionState& x, double /*t*/) {
            return AssumptionState{-x[0], -x[1]};
        };
        return evaluate_assumptions(Vec{1.0, 0.1, 0.01}, Vec{0.0, 0.0, 0.0}, u, t, ops);
    }

    if (name == "invalid_operator_unbounded") {
        AssumptionEvalConfig cfg;
        cfg.max_lipschitz_proxy = 10.0;

        AssumptionOperatorCallbacks ops;
        ops.apply_L = [](const AssumptionState& x, double /*t*/) {
            return AssumptionState{-x[0], -x[1]};
        };
        ops.apply_N = [](const AssumptionState& x, double /*t*/) {
            return AssumptionState{1000.0 * x[0], -1000.0 * x[1]};
        };
        return evaluate_assumptions(Vec{1.0, 0.1, 0.01}, Vec{0.5, 0.3, 0.2}, u, t, ops, cfg);
    }

    throw std::invalid_argument("Unknown assumption test case: " + name);
}

} // namespace gfe

