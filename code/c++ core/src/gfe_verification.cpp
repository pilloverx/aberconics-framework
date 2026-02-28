#include "gfe/gfe_verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "gfe/gfe_core.hpp"
#include "gfe/gfe_dynamics.hpp"

namespace gfe {
namespace {

bool approx(double a, double b, double atol) {
    return std::fabs(a - b) <= atol;
}

VerificationResult op_basis() {
    const Vec gamma{1.0, 0.1, 0.01};
    Vec t;
    for (int i = 0; i <= 10; ++i) {
        t.push_back(static_cast<double>(i));
    }

    const auto phi = exponential_basis(gamma, t);
    const bool shape_ok = (phi.size() == 11 && phi.front().size() == 3);
    const bool t0_ok = shape_ok && approx(phi[0][0], 1.0, 1e-12) && approx(phi[0][1], 1.0, 1e-12) && approx(phi[0][2], 1.0, 1e-12);
    const bool decay_ok = shape_ok && (phi[1][0] < phi[0][0]) && (phi[1][1] < phi[0][1]) && (phi[1][2] < phi[0][2]);

    VerificationResult out;
    out.operation = "basis";
    out.pass = shape_ok && t0_ok && decay_ok;
    out.note = "Exponential basis shape and monotone decay check";
    out.metrics = {
        {"n_samples", static_cast<double>(phi.size())},
        {"n_basis", shape_ok ? static_cast<double>(phi.front().size()) : 0.0},
        {"phi_t0_col0", shape_ok ? phi[0][0] : std::numeric_limits<double>::quiet_NaN()}
    };
    return out;
}

VerificationResult op_soe_fit() {
    const Vec gamma_true{1.0, 0.1, 0.01};
    const Vec w_true{0.5, 0.3, 0.2};
    Vec t;
    for (int i = 0; i <= 400; ++i) {
        t.push_back(0.05 * static_cast<double>(i));
    }

    const auto A = design_matrix(gamma_true, t);
    Vec acf_true(t.size(), 0.0);
    for (std::size_t i = 0; i < t.size(); ++i) {
        acf_true[i] = A[i][0] * w_true[0] + A[i][1] * w_true[1] + A[i][2] * w_true[2];
    }

    // Keep this close to the Julia-style workflow: broader candidate basis then prune.
    const auto fit = fit_soe_kernel(t, acf_true, 12, 1e-2, 10.0, 0.01, 1e-3, 4000);

    double l1 = 0.0;
    for (std::size_t i = 0; i < acf_true.size(); ++i) {
        l1 += std::fabs(fit.fit[i] - acf_true[i]);
    }
    l1 /= static_cast<double>(acf_true.size());

    const bool size_ok = (fit.fit.size() == acf_true.size() && !fit.gamma_fit.empty() && !fit.w_fit.empty());
    const bool order_ok = std::is_sorted(fit.gamma_fit.begin(), fit.gamma_fit.end(), std::greater<double>());
    // The projected-gradient NNLS here is intentionally simple; use a conservative bound.
    const bool err_ok = l1 < 0.35;

    VerificationResult out;
    out.operation = "soe_fit";
    out.pass = size_ok && order_ok && err_ok;
    out.note = "Kernel fit quality and decay-order checks";
    out.metrics = {
        {"l1_error", l1},
        {"n_modes_kept", static_cast<double>(fit.gamma_fit.size())}
    };
    return out;
}

VerificationResult op_spectral_units() {
    const Vec gamma{1.0, 0.1, 0.01};
    const Vec w{0.33, 0.33, 0.34};

    const auto su_paper_default = spectral_units(w, gamma);
    SpectralUnitOptions legacy_opts;
    legacy_opts.deff_policy = DeffPolicy::EffectiveRankOnly;
    legacy_opts.mcap_policy = McapPolicy::LegacyWeightedTimescale;
    const auto su_legacy = spectral_units(w, gamma, legacy_opts);

    const bool range_ok = su_paper_default.Hnorm >= 0.0 && su_paper_default.Hnorm <= 1.0 &&
                          su_paper_default.Mcap > 0.0 && su_paper_default.Deff > 0.0;
    const bool deff_policy_ok = approx(su_paper_default.Deff, static_cast<double>(w.size()) * su_legacy.Deff, 1e-9);

    VerificationResult out;
    out.operation = "spectral_units";
    out.pass = range_ok && deff_policy_ok;
    out.note = "Spectral unit ranges and paper-default policy check";
    out.metrics = {
        {"Mcap_paper_default", su_paper_default.Mcap},
        {"Hnorm", su_paper_default.Hnorm},
        {"Deff_paper_default", su_paper_default.Deff},
        {"Deff_legacy", su_legacy.Deff}
    };
    return out;
}

VerificationResult op_pack_roundtrip() {
    const Vec gamma{1.0, 0.1, 0.01};
    const Vec w{0.5, 0.3, 0.2};

    const auto theta = pack_memory_params(gamma, w);
    const auto rec = unpack_memory_params(theta);

    const bool shape_ok = (rec.first.size() == gamma.size() && rec.second.size() == w.size());
    bool values_ok = shape_ok;
    for (std::size_t i = 0; i < gamma.size() && values_ok; ++i) {
        values_ok = approx(rec.first[i], gamma[i], 1e-10) && approx(rec.second[i], w[i], 1e-10);
    }

    VerificationResult out;
    out.operation = "pack_roundtrip";
    out.pass = shape_ok && values_ok;
    out.note = "Pack/unpack exactness check";
    out.metrics = {
        {"theta_size", static_cast<double>(theta.size())},
        {"gamma0_recovered", rec.first.empty() ? std::numeric_limits<double>::quiet_NaN() : rec.first[0]}
    };
    return out;
}

VerificationResult run_dynamics_case(CouplingForm form, const std::string& name) {
    const bool resonant = (form == CouplingForm::ResonantSecondOrder);
    const State u0 = resonant ? State{1.0, 0.0} : State{1.0};
    const Vec chi0{0.0, 0.0};
    const MemoryKernelParams mk{Vec{1.0, 0.5}, Vec{0.4, 0.2}};

    DynamicsConfig cfg;
    cfg.dt = 0.1;
    cfg.form = form;
    cfg.coupling_index = 0;
    cfg.resonant_stiffness = 1.0;
    cfg.resonant_damping = 0.2;
    cfg.resonant_feedback_gain = 1.0;

    OperatorCallbacks ops;
    if (resonant) {
        ops.apply_L = [](const State& /*u*/, double /*t*/) { return State{0.0, 0.0}; };
        ops.apply_N = [](const State& /*u*/, double /*t*/) { return State{0.0, 0.0}; };
        ops.forcing = [](double /*t*/) { return State{0.0, 0.0}; };
    } else {
        ops.apply_L = [](const State& u, double /*t*/) { return State{-u[0]}; };
        ops.apply_N = [](const State& /*u*/, double /*t*/) { return State{0.0}; };
        ops.forcing = [](double /*t*/) { return State{1.0}; };
    }

    AssumptionChecks checks;
    IdentityGSTAdapter gst;
    const auto step = step_augmented(u0, chi0, 0.0, mk, cfg, ops, &gst, &checks, nullptr);

    const bool shape_ok = (step.u_next.size() == u0.size() && step.chi_next.size() == 2);
    const bool chi_ok = shape_ok && (step.chi_next[0] >= 0.0 && step.chi_next[1] >= 0.0);

    VerificationResult out;
    out.operation = name;
    out.pass = shape_ok && chi_ok;
    out.note = checks.note;
    out.metrics = {
        {"u_next_0", shape_ok ? step.u_next[0] : std::numeric_limits<double>::quiet_NaN()},
        {"chi_next_0", shape_ok ? step.chi_next[0] : std::numeric_limits<double>::quiet_NaN()},
        {"chi_next_1", shape_ok ? step.chi_next[1] : std::numeric_limits<double>::quiet_NaN()}
    };
    return out;
}

} // namespace

std::vector<std::string> list_verification_operations() {
    return {
        "basis",
        "soe_fit",
        "spectral_units",
        "pack_roundtrip",
        "dynamics_a",
        "dynamics_b",
        "dynamics_c"
    };
}

std::vector<std::string> default_verification_operations() {
    // Important GFE operations by default.
    return list_verification_operations();
}

VerificationResult run_verification_operation(const std::string& operation) {
    if (operation == "basis") {
        return op_basis();
    }
    if (operation == "soe_fit") {
        return op_soe_fit();
    }
    if (operation == "spectral_units") {
        return op_spectral_units();
    }
    if (operation == "pack_roundtrip") {
        return op_pack_roundtrip();
    }
    if (operation == "dynamics_a") {
        return run_dynamics_case(CouplingForm::InputDriven, "dynamics_a");
    }
    if (operation == "dynamics_b") {
        return run_dynamics_case(CouplingForm::NegativeFeedback, "dynamics_b");
    }
    if (operation == "dynamics_c") {
        return run_dynamics_case(CouplingForm::ResonantSecondOrder, "dynamics_c");
    }

    throw std::invalid_argument("Unknown verification operation: " + operation);
}

std::vector<VerificationResult> run_verification_operations(const std::vector<std::string>& operations) {
    std::vector<VerificationResult> out;
    out.reserve(operations.size());
    for (const auto& op : operations) {
        out.push_back(run_verification_operation(op));
    }
    return out;
}

} // namespace gfe
