#include "gfe/gfe_core.hpp"
#include "gfe/gfe_backend.hpp"
#include "gfe/gfe_dynamics.hpp"
#include "gfe/gfe_verification.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace gfe;

    // Shared registry check: keeps smoke tests and future experiments linked.
    {
        const auto results = run_verification_operations(default_verification_operations());
        bool ok = true;
        for (const auto& r : results) {
            ok = ok && r.pass;
        }
        if (!ok) {
            std::cerr << "shared verification operations failed\n";
            return 1;
        }
    }

    const Vec gamma{1.0, 0.1, 0.01};
    Vec t;
    for (int i = 0; i <= 10; ++i) t.push_back(static_cast<double>(i));

    const auto phi = exponential_basis(gamma, t);
    if (phi.size() != 11 || phi.front().size() != 3) {
        std::cerr << "basis shape mismatch\n";
        return 1;
    }

    const Vec w{0.33, 0.33, 0.34};
    const auto su = spectral_units(w, gamma); // paper-default
    if (!(su.Mcap > 0.0 && su.Hnorm >= 0.0 && su.Hnorm <= 1.0 && su.Deff > 0.0)) {
        std::cerr << "spectral units check failed\n";
        return 1;
    }

    SpectralUnitOptions legacy_opts;
    legacy_opts.deff_policy = DeffPolicy::EffectiveRankOnly;
    legacy_opts.mcap_policy = McapPolicy::LegacyWeightedTimescale;
    const auto su_legacy = spectral_units(w, gamma, legacy_opts);
    if (std::fabs(su.Deff - static_cast<double>(w.size()) * su_legacy.Deff) > 1e-9) {
        std::cerr << "DeffPolicy check failed\n";
        return 1;
    }

    const Vec gamma_single{0.5};
    const Vec w_single{1.0};
    const auto su_single_default = spectral_units(w_single, gamma_single);
    if (!std::isnan(su_single_default.Mres)) {
        std::cerr << "Mscale==0 default handling should produce NaN Mres\n";
        return 1;
    }

    SpectralUnitOptions inf_opts;
    inf_opts.mres_policy = MresPolicy::ReturnInfinity;
    const auto su_single_inf = spectral_units(w_single, gamma_single, inf_opts);
    if (!std::isinf(su_single_inf.Mres)) {
        std::cerr << "Mscale==0 infinity policy failed\n";
        return 1;
    }

    const auto theta = pack_memory_params(gamma, Vec{0.5, 0.3, 0.2});
    const auto rec = unpack_memory_params(theta);
    if (rec.first.size() != 3 || rec.second.size() != 3) {
        std::cerr << "pack/unpack shape failed\n";
        return 1;
    }
    if (std::fabs(rec.first[0] - 1.0) > 1e-9) {
        std::cerr << "pack/unpack value mismatch\n";
        return 1;
    }

    // Prony side-line fitter check on clean synthetic signal.
    {
        Vec t_prony;
        for (int i = 0; i <= 200; ++i) {
            t_prony.push_back(0.05 * static_cast<double>(i));
        }
        const Vec gamma_true{0.9, 0.15};
        const Vec w_true{0.6, 0.4};
        const auto A_true = design_matrix(gamma_true, t_prony);
        Vec y(t_prony.size(), 0.0);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = A_true[i][0] * w_true[0] + A_true[i][1] * w_true[1];
        }

        SoeFitOptions fit_opts;
        fit_opts.backend = FitBackend::Prony;
        fit_opts.prony.order = 4;
        fit_opts.prony.ridge = 1e-9;
        const auto prony_fit = fit_soe_kernel(t_prony, y, fit_opts);
        if (prony_fit.gamma_fit.empty() || prony_fit.w_fit.empty() || prony_fit.fit.size() != y.size()) {
            std::cerr << "Prony fit shape check failed\n";
            return 1;
        }
        double l1 = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            l1 += std::fabs(prony_fit.fit[i] - y[i]);
        }
        l1 /= static_cast<double>(y.size());
        if (l1 > 0.05) {
            std::cerr << "Prony fit error too high\n";
            return 1;
        }

        BackendConfig backend_cfg;
        backend_cfg.preferred = ComputeBackend::GpuCuda;
        backend_cfg.allow_fallback = true;
        backend_cfg.fallback = ComputeBackend::CpuReference;
        const auto prony_fit_fallback = fit_soe_kernel(t_prony, y, fit_opts, backend_cfg);
        if (prony_fit_fallback.fit.size() != y.size()) {
            std::cerr << "Backend fallback dispatch fit shape failed\n";
            return 1;
        }
    }

    // Phase 1 dynamics checks (GST optional)
    State u0{1.0};
    Vec chi0{0.0, 0.0};
    MemoryKernelParams mk{Vec{1.0, 0.5}, Vec{0.4, 0.2}};
    DynamicsConfig cfg;
    cfg.dt = 0.1;
    cfg.coupling_index = 0;
    OperatorCallbacks ops;
    ops.apply_L = [](const State& u, double /*t*/) { return State{-u[0]}; };
    ops.apply_N = [](const State& /*u*/, double /*t*/) { return State{0.0}; };
    ops.forcing = [](double /*t*/) { return State{1.0}; };

    // Formulation A (input-driven): chi should increase due to forcing drive.
    cfg.form = CouplingForm::InputDriven;
    const auto step_a = step_augmented(u0, chi0, 0.0, mk, cfg, ops);
    if (!(step_a.u_next.size() == 1 && step_a.chi_next.size() == 2)) {
        std::cerr << "step_augmented A shape failed\n";
        return 1;
    }
    if (!(step_a.chi_next[0] > 0.0 && step_a.chi_next[1] > 0.0)) {
        std::cerr << "step_augmented A chi drive failed\n";
        return 1;
    }

    // Formulation B (negative feedback) with same initial chi should retain zero chi under u-driven update at chi=0 and u0=1.
    cfg.form = CouplingForm::NegativeFeedback;
    const auto step_b = step_augmented(u0, chi0, 0.0, mk, cfg, ops);
    if (!(step_b.u_next.size() == 1 && step_b.chi_next.size() == 2)) {
        std::cerr << "step_augmented B shape failed\n";
        return 1;
    }
    if (!(step_b.chi_next[0] > 0.0 && step_b.chi_next[1] > 0.0)) {
        std::cerr << "step_augmented B chi drive failed\n";
        return 1;
    }

    // GST optional: Identity adapter should preserve behavior/dimensions.
    IdentityGSTAdapter gst_identity;
    AssumptionChecks checks;
    const auto step_gst = step_augmented(u0, chi0, 0.0, mk, cfg, ops, &gst_identity, &checks, nullptr);
    if (step_gst.u_next.size() != 1 || step_gst.chi_next.size() != 2) {
        std::cerr << "step_augmented GST path failed\n";
        return 1;
    }

    // Formulation C (resonant second-order): state layout [q, v].
    {
        State uc0{1.0, 0.0};
        Vec chic0{0.0, 0.0};
        DynamicsConfig cfgc;
        cfgc.dt = 0.05;
        cfgc.form = CouplingForm::ResonantSecondOrder;
        cfgc.coupling_index = 0;
        cfgc.resonant_stiffness = 1.0;
        cfgc.resonant_damping = 0.1;
        cfgc.resonant_feedback_gain = 1.0;
        OperatorCallbacks opsc;
        opsc.apply_L = [](const State& /*u*/, double /*t*/) { return State{0.0, 0.0}; };
        opsc.apply_N = [](const State& /*u*/, double /*t*/) { return State{0.0, 0.0}; };
        opsc.forcing = [](double /*t*/) { return State{0.0, 0.0}; };
        const auto step_c = step_augmented(uc0, chic0, 0.0, mk, cfgc, opsc, nullptr, nullptr, nullptr);
        if (!(step_c.u_next.size() == 2 && step_c.chi_next.size() == 2)) {
            std::cerr << "step_augmented C shape failed\n";
            return 1;
        }
    }

    std::cout << "gfe_smoke_test passed\n";
    return 0;
}
