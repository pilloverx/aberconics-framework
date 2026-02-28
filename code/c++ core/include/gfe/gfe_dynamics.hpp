#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "gfe/gfe_core.hpp"

namespace gfe {

using State = std::vector<double>;

struct MemoryKernelParams {
    Vec gamma; // decay rates, size L
    Vec w;     // memory weights, size L
};

enum class CouplingForm {
    InputDriven,       // Formulation A
    NegativeFeedback,  // Formulation B
    ResonantSecondOrder // Formulation C
};

struct DynamicsConfig {
    double dt = 1e-3;
    CouplingForm form = CouplingForm::NegativeFeedback;
    std::size_t coupling_index = 0; // state channel coupled to chi
    // Formulation C controls (second-order resonant dynamics on paired q/v state).
    // State layout for form C: [q_0, ..., q_{n-1}, v_0, ..., v_{n-1}].
    double resonant_stiffness = 1.0;
    double resonant_damping = 0.1;
    double resonant_feedback_gain = 1.0;
};

struct OperatorCallbacks {
    // Required core operator term (L[u] in theorem notation)
    std::function<State(const State& u, double t)> apply_L;

    // Optional nonlinear term N[u]
    std::function<State(const State& u, double t)> apply_N;

    // Optional external forcing/input
    std::function<State(double t)> forcing;
};

// Placeholder hook: theorem-assumption checks (deferred, non-blocking)
struct AssumptionChecks {
    bool pass = true;
    bool kernel_approximable = true;
    bool regularity_ok = true;
    bool operator_boundedness_ok = true;
    double kernel_margin = 0.0;
    double regularity_margin = 0.0;
    double operator_boundedness_margin = 0.0;
    std::string note;
};

// Placeholder hook: energy/Hamiltonian observer (deferred, non-blocking)
struct EnergyObserver {
    virtual void on_step(double t, const State& u, const Vec& chi) { (void)t; (void)u; (void)chi; }
    virtual void on_transition(double t_prev,
                               double t_next,
                               const State& u_prev,
                               const Vec& chi_prev,
                               const State& u_next,
                               const Vec& chi_next) {
        (void)t_prev;
        (void)t_next;
        (void)u_prev;
        (void)chi_prev;
        (void)u_next;
        (void)chi_next;
    }
    virtual ~EnergyObserver() = default;
};

// GST is optional: pass nullptr to operate without GST.
struct GSTAdapter {
    virtual State project(const State& u, double t) const = 0;
    virtual State reconstruct(const State& coeffs, double t) const = 0;
    virtual ~GSTAdapter() = default;
};

struct IdentityGSTAdapter final : public GSTAdapter {
    State project(const State& u, double /*t*/) const override { return u; }
    State reconstruct(const State& coeffs, double /*t*/) const override { return coeffs; }
};

struct StepResult {
    State u_next;
    Vec chi_next;
};

StepResult step_augmented(
    const State& u,
    const Vec& chi,
    double t,
    const MemoryKernelParams& kernel,
    const DynamicsConfig& cfg,
    const OperatorCallbacks& ops,
    const GSTAdapter* gst = nullptr,
    AssumptionChecks* assumption_checks = nullptr,
    EnergyObserver* energy_observer = nullptr);

} // namespace gfe
